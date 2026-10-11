"""A common physical task for joint, Cartesian-delta, and torque interfaces."""

import numpy as np

from week10.pendulum import Pendulum, PHYSICS_DT
from week11.control import GOAL

INNER_DT = 0.01  # 100 Hz servo, independent of the policy's decision frequency.
ACTION_VALUES = {
    "joint": np.array([-.3, 0., .25, .5, .75, 1., 1.3]),
    "cartesian": np.array([-.08, -.04, -.015, 0., .015, .04, .08]),
    "torque": np.linspace(-3, 3, 7),
}
MODE_CODES = {"torque": 0, "joint": 1, "cartesian": 2, "pd": 3, "pd_gravity": 4}


def gravity_torque(q, xp=np):
    """Return g(q)=9.81*(1*0.3+0.2*0.6)*sin(q) in N m.

    q is a NumPy array or torch tensor (xp selects the corresponding module).
    The rod mass is 1 kg with COM 0.3 m below the joint, and bob mass is
    0.2 kg at 0.6 m. For q=0 hanging down, U=-m*g*r*cos(q), so dU/dq
    is the positive holding torque above (MR notes 8.1, 11.4.1). Armature
    affects inertia, not gravity. This exact geometry-specific model excludes
    applied disturbances; using it in a mismatched model would be feedforward
    error, not corrected gravity compensation.
    """
    return 9.81 * .42 * xp.sin(q)


def cartesian_target(q: np.ndarray, delta_x: np.ndarray) -> np.ndarray:
    """Map a world-x tip displacement to an approximate joint target, in rad.

    For p_x=-L*sin(q), the 1x1 point Jacobian is J_x=-L*cos(q), L=0.6 m.
    Linearize Delta x ~= J_x*Delta q. Minimizing
      (J_x*Delta q-Delta x)^2 + damping^2*Delta q^2
    gives Delta q=J_x*Delta x/(J_x^2+damping^2), damping=0.05 m/rad.
    This is damped least squares, not exact IK. Near q=pi/2 the x direction
    loses first-order authority; damping prevents division by zero but cannot
    create that authority (MR notes 5.1, 11.3.3; project-specific damping).
    Both inputs/output are (B,). Clip the correction to +/-0.2 rad, then the
    target to [-1.3,1.3] rad. Target is relative to CURRENT measured q, not
    accumulated previous commands. Only world x is commanded; this 1-DOF
    robot cannot independently specify x and z. Delta x is metres per
    decision, so changing decision frequency changes effective command speed.
    """
    jacobian = -.6 * np.cos(q)
    delta_q = jacobian * delta_x / (jacobian**2 + .05**2)
    return np.clip(q + np.clip(delta_q, -.2, .2), -1.3, 1.3)


def motor_torque(q, velocity, commands, modes, limit=3., xp=np):
    """Convert held commands to bounded torque at each 100 Hz servo tick.

    q,velocity,commands,modes are (B,) arrays/tensors. Modes 1/2 hold an
    absolute q_d in rad, and use tau=6*(q_d-q)-1.4*qdot+g(q). Baseline 3
    is pure PD and 4 adds g(q); both hold q_d=0.5. Mode 0 holds raw torque
    in N m, with no hidden feedback or gravity compensation. Return
    (clip(tau,-limit,limit), tau_raw) in N m. Common motor limits make
    saturation visible. Kp and Kd are torque gains, not acceleration gains.
    MR notes 11.4.1 explains PD and gravity compensation; discretization and
    clipping are implementation choices, not continuous-time guarantees.
    """
    servo = 6 * (commands - q) - 1.4 * velocity
    servo = servo + xp.where(modes == 3, 0., gravity_torque(q, xp))
    raw = xp.where(modes == 0, commands, servo)
    return raw.clip(-limit, limit), raw


class ControlPendulum(Pendulum):
    """Synchronized finite-horizon tasks, or a continuing disturbance task.

    State (q,qdot,h) and native/GPU reset storage reuse week 10. Low-level
    control is 100 Hz; physics is 500 Hz. Policy decisions default to 25 Hz.
    The motor limit is common to all interfaces. Disturbance is an external
    generalized torque, distinct from bounded motor torque. Contact is not
    modeled by this disturbance exercise.
    """

    def __init__(self, modes, duration=2., decision_dt=.04, backend="mjlab", limit=3., continuing=False):
        ticks = round(decision_dt / INNER_DT)
        horizon = round(duration / decision_dt)
        if (ticks < 1 or horizon < 1 or not np.isclose(ticks * INNER_DT, decision_dt)
                or not np.isclose(horizon * decision_dt, duration) or not 0 < limit <= 3):
            raise ValueError("Intervals must divide duration/servo period; limit in (0,3]")
        self.continuing = continuing
        super().__init__(len(modes), horizon, backend)
        self.modes = np.array([MODE_CODES[m] for m in modes])
        self.names = list(modes)
        self.decision_dt, self.ticks, self.limit = decision_dt, ticks, limit
        self.last_torque = np.zeros(len(modes))
        if backend == "mjlab":
            import torch
            self.xp = torch
            self.device_modes = torch.as_tensor(self.modes, device="cuda:0")
        else:
            self.xp = np

    def observation(self):
        """Read (B,3); continuing tasks use constant h=0.5 (no hidden horizon)."""
        state = super().observation()
        if self.continuing:
            state[:, 2] = .5
        return state

    def reset(self, initial):
        """Reset physical state, time, and diagnostic previous torque to zero."""
        self.last_torque = np.zeros(self.num_envs)
        return super().reset(initial)

    def kick(self, delta_velocity):
        """Apply a velocity impulse qdot <- qdot+Delta v (rad/s), shape (B,).

        This represents an instantaneous generalized momentum change M*Delta v,
        not a contact solver or a finite-duration external force. Called only
        for frozen-policy robustness evaluation, at the recorded mid-episode
        boundary. No reset of position, episode time, or learned state occurs.
        """
        if self.backend == "mjlab":
            self.sim.data.qvel[:, 0] += self.xp.as_tensor(delta_velocity, device="cuda:0",
                                                       dtype=self.sim.data.qvel.dtype)
            self.sim.forward()
        else:
            import mujoco
            for data, delta in zip(self.data, delta_velocity, strict=True):
                data.qvel[0] += delta
                mujoco.mj_forward(self.model, data)

    def step(self, actions, disturbance=None):
        """Execute discrete actions and integrate reward/diagnostics over one decision.

        actions is integer (B,), interpreted by each world's action interface.
        Convert joint/Cartesian commands once at S_t, then hold q_d while PD
        recomputes torque every 0.01 s. Direct torque is held for the entire
        decision. For each servo tick, using its PRE-integration q,v,tau:
          dR=-dt*((q-0.5)^2 + .05*v^2 + .005*tau^2),
          dW_abs=dt*|tau*v|, dE_effort=dt*tau^2.
        Weights use numerical rad, rad/s, N m with unit reference scales.
        Rectangular quadrature approximates integrals; W_abs is absolute
        mechanical work [J], NOT electrical energy. Effort integral is N^2 m^2 s.
        Torque variation sums |tau_k-tau_{k-1}| on the common servo grid,
        including initial switch from zero. Reward does not depend on previous
        torque, so this diagnostic memory is not hidden state for the learner.

        Return (next_state, reward, terminal, info), with rewards (B,) and info
        per-world integrated errors/work/effort/variation/saturation/instability.
        A true finite task ends at its known horizon; continuing collection
        never terminates. Physical threshold violations are measured, not
        silently reset. Nonfinite state aborts through observation().
        """
        if not self.continuing and self.elapsed >= self.horizon:
            raise RuntimeError("Reset after true task termination")
        actions = np.asarray(actions)
        if (actions.shape != (self.num_envs,) or not np.issubdtype(actions.dtype, np.integer)
                or np.any((actions < 0) | (actions >= 7))):
            raise ValueError("Need action indices (B,) in [0,6]")
        state = self.observation()
        commands = np.full(self.num_envs, GOAL)
        for name, values in ACTION_VALUES.items():
            mask = self.modes == MODE_CODES[name]
            selected = values[actions[mask]]
            commands[mask] = cartesian_target(state[mask, 0], selected) if name == "cartesian" else selected
        return self.step_commands(commands, disturbance)

    def step_commands(self, commands, disturbance=None):
        """Execute physical commands through the common control/physics loop.

        commands is finite (B,): torque worlds interpret N m; servo worlds
        interpret a held q_d in rad. Unlike step(), this entry accepts continuous
        commands (week 12). It shares identical motor limits, integration and
        diagnostics with week 11's discrete interfaces. At each servo tick:
          R += -dt*((q-GOAL)^2 + .05*qdot^2 + .005*tau^2),
          work += dt*abs(tau*qdot), effort += dt*tau^2,
        with dt=0.01 s and bounded motor torque from motor_torque(). These are
        left-rectangle numerical integrals, using pre-integration states; motor
        work excludes external disturbance work. Return (S', R, terminal, info)
        after a full policy interval. Inputs and output units, terminal semantics,
        and torque-variation conventions are detailed in step(). No derivative
        is propagated through clipping or physics by the RL score estimator.
        """
        if not self.continuing and self.elapsed >= self.horizon:
            raise RuntimeError("Reset after true task termination")
        commands = np.asarray(commands)
        if commands.shape != (self.num_envs,) or not np.isfinite(commands).all():
            raise ValueError("Commands must be finite (B,) physical values")
        disturbance = np.zeros(self.num_envs) if disturbance is None else np.asarray(disturbance)
        if disturbance.shape != (self.num_envs,) or not np.isfinite(disturbance).all():
            raise ValueError("External torque must be finite (B,)")
        if self.backend == "mjlab":
            xp = self.xp
            dtype = self.sim.data.qpos.dtype
            commands = xp.as_tensor(commands, device="cuda:0", dtype=dtype)
            modes = self.device_modes
            self.sim.data.qfrc_applied[:, 0] = xp.as_tensor(disturbance, device="cuda:0", dtype=dtype)
            previous = xp.as_tensor(self.last_torque, device="cuda:0", dtype=dtype)
            totals = xp.zeros((8, self.num_envs), device="cuda:0", dtype=dtype)
        else:
            xp, modes, previous = np, self.modes, self.last_torque.copy()
            totals = np.zeros((8, self.num_envs))
        for _ in range(self.ticks):
            if self.backend == "mjlab":
                q, v = self.sim.data.qpos[:, 0], self.sim.data.qvel[:, 0]
            else:
                current = self.observation()
                q, v = current[:, 0], current[:, 1]
            torque, raw = motor_torque(q, v, commands, modes, self.limit, xp)
            totals[0] -= INNER_DT * ((q - GOAL)**2 + .05 * v**2 + .005 * torque**2)
            totals[1] += INNER_DT * (q - GOAL)**2
            totals[2] += INNER_DT * xp.abs(torque * v)
            totals[3] += INNER_DT * torque**2
            totals[4] += xp.abs(torque - previous)
            totals[5] += INNER_DT * (xp.abs(raw) > self.limit)
            totals[6] += INNER_DT * ((xp.abs(q - GOAL) > 1.5) | (xp.abs(v) > 8))
            totals[7] += INNER_DT * torque
            previous = torque.copy() if xp is np else torque.clone()
            if self.backend == "mjlab":
                self.sim.data.ctrl[:, 0] = torque
                for _ in range(round(INNER_DT / PHYSICS_DT)):
                    self.sim.step()
            else:
                import mujoco
                for i, data in enumerate(self.data):
                    data.ctrl[0], data.qfrc_applied[0] = torque[i], disturbance[i]
                    mujoco.mj_step(self.model, data, nstep=round(INNER_DT / PHYSICS_DT))
        if self.backend == "mjlab":
            totals, previous = totals.cpu().numpy(), previous.cpu().numpy()
        self.last_torque = previous.copy()
        self.elapsed += 1
        info = dict(zip(("reward", "squared_error", "work_abs", "effort", "torque_variation",
                         "saturated_time", "unstable_time", "torque_integral"), totals, strict=True))
        return self.observation(), totals[0], not self.continuing and self.elapsed == self.horizon, info
