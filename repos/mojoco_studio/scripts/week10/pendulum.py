"""Fixed-policy pendulum rollouts using native MuJoCo or mjlab's simulation layer."""

from pathlib import Path

import mujoco
import numpy as np


MODEL_PATH = Path(__file__).resolve().parents[2] / "assets" / "pendulum.xml"
PHYSICS_DT = 0.002
SUBSTEPS = 10
CONTROL_DT = PHYSICS_DT * SUBSTEPS


def policy(observations: np.ndarray) -> np.ndarray:
    """Return the fixed downward PD torque for states (...,3), in N m.

    tau=clip(Kp*(q_des-q)+Kd*(v_des-v), -3, 3), with q_des=v_des=0,
    Kp=2 N m/rad and Kd=0.6 N m s/rad. These are torque gains, not computed
    torque acceleration gains. q is rotation about world +y; q=0 hangs down,
    unlike the horizontal-angle convention in MR notes 11.4.1. Here gravity
    already vanishes at the setpoint. Torque is held for 10 physics steps
    (20 ms), then recomputed; changing TD's n does not change this frequency.
    Saturation and sample-and-hold are implementation choices, so the book's
    ideal continuous linear PD analysis is not a global stability guarantee.
    """
    return np.clip(-2 * observations[..., 0] - 0.6 * observations[..., 1], -3, 3)


def reward(observations: np.ndarray, torques: np.ndarray) -> np.ndarray:
    """Define R_{t+1}=-dt*(q_t^2+0.1*v_t^2+0.01*tau_t^2).

    States have shape (...,3); torque/output shape (...,). The dimensionless
    cost uses numerical q in rad, v in rad/s and tau in N m (unit reference
    scales of 1 rad, 1 rad/s and 1 N m). dt=0.02 s makes this a left-rectangle
    approximation to a continuous cost integral, not an exact integral.
    This transition reward is computed from S_t,A_t before integration, then
    attached to the resulting S_{t+1}. There is no success bonus; horizon
    termination retains the last transition reward. This reward is a project
    design, not a formula attributed to Modern Robotics or the RL textbook.
    """
    return -CONTROL_DT * (observations[..., 0]**2 + 0.1 * observations[..., 1]**2
                          + 0.01 * torques**2)


def tip_kinematics(observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute world tip position and linear velocity (...,3) for book review.

    For rod length L=0.6 m, pivot (0,0,1) m and world +y hinge,
    p=(-L*sin(q),0,1-L*cos(q)) and J_p=dp/dq=(-L*cos(q),0,L*sin(q))^T.
    v_tip=J_p*qdot, in m/s. This is a 3x1 point Jacobian at the tip, NOT the
    6D space-twist Jacobian (MR notes 5.1-5.2). The derivative follows by
    rotating the downward rod about +y. It assumes the geometry of the shared
    pendulum XML; base and length are fixed. Used in the rollout CSV/viewer
    walkthrough to connect the RL state (q,qdot) with Cartesian motion.
    """
    q, velocity = observations[..., 0], observations[..., 1]
    zeros = np.zeros_like(q)
    position = np.stack((-0.6 * np.sin(q), zeros, 1 - 0.6 * np.cos(q)), axis=-1)
    jacobian = np.stack((-0.6 * np.cos(q), zeros, 0.6 * np.sin(q)), axis=-1)
    return position, jacobian * velocity[..., None]


class Pendulum:
    """B synchronized finite-horizon episodes; no implicit reset on step.

    The augmented Markov state is (q,qdot,h), where h=(T-t)/T is remaining
    time. At h=0 the task truly ends and V=0. This is part of the task, not
    an external time-limit truncation of a continuing problem. All rollouts
    include S_T and R_T. Nonfinite physics aborts rather than creating bogus
    terminal experience. mjlab provides batched physics via Simulation; this
    small exercise implements its own observation/reward/episode logic rather
    than requiring a manager environment or RSL-RL's PPO training loop.
    """

    def __init__(self, num_envs: int, horizon: int, backend: str = "mjlab"):
        if num_envs < 1 or horizon < 1 or backend not in ("mjlab", "native"):
            raise ValueError("Need positive num_envs/horizon and mjlab or native backend")
        self.num_envs, self.horizon, self.backend = num_envs, horizon, backend
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.model.opt.timestep = PHYSICS_DT
        self.elapsed = 0
        if backend == "mjlab":
            import torch
            import warp as wp
            from mjlab.sim import Simulation, SimulationCfg

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; select --backend native for CPU")
            wp.init()
            self.sim = Simulation(num_envs=num_envs, cfg=SimulationCfg(),
                                  model=self.model, device="cuda:0")
        else:
            self.data = [mujoco.MjData(self.model) for _ in range(num_envs)]

    def observation(self) -> np.ndarray:
        """Read (B,3) states in rad, rad/s and remaining fraction h=(T-t)/T.

        Native and Warp arrays are copied into NumPy float64 for transparent
        textbook updates. GPU physics still uses its own precision. The time
        coordinate prevents states with identical (q,qdot) but different
        remaining returns from being aliased. It is deterministic, not learned.
        """
        if self.backend == "mjlab":
            q = self.sim.data.qpos[:, 0].cpu().numpy().copy()
            v = self.sim.data.qvel[:, 0].cpu().numpy().copy()
        else:
            q = np.array([d.qpos[0] for d in self.data])
            v = np.array([d.qvel[0] for d in self.data])
        result = np.column_stack((q, v, np.full(self.num_envs, 1 - self.elapsed / self.horizon)))
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite pendulum state")
        return result

    def reset(self, initial: np.ndarray) -> np.ndarray:
        """Reset all worlds to explicit (B,2) initial q [rad], qdot [rad/s]."""
        initial = np.asarray(initial, dtype=np.float64)
        if initial.shape != (self.num_envs, 2) or not np.isfinite(initial).all():
            raise ValueError("Initial states must be finite (num_envs,2)")
        self.elapsed = 0
        if self.backend == "mjlab":
            import torch

            self.sim.reset()
            state = torch.as_tensor(initial, device="cuda:0", dtype=self.sim.data.qpos.dtype)
            self.sim.data.qpos[:, 0] = state[:, 0]
            self.sim.data.qvel[:, 0] = state[:, 1]
            self.sim.data.ctrl.zero_()
            self.sim.forward()
        else:
            for data, (q, v) in zip(self.data, initial, strict=True):
                mujoco.mj_resetData(self.model, data)
                data.qpos[0], data.qvel[0] = q, v
                mujoco.mj_forward(self.model, data)
        return self.observation()

    def step(self) -> tuple[np.ndarray, np.ndarray, bool]:
        """Apply A_t=pi(S_t), emit (S_{t+1}, R_{t+1}, terminated).

        MuJoCo numerically integrates M(q)qddot+c(q,qdot)+g(q)+b*qdot=tau
        with the XML's distributed rod/bob inertia and joint armature. This is
        the forward-dynamics viewpoint of MR notes 8.5, not a point-mass Euler
        implementation. implicitfast advances 10*0.002 s per decision. The
        horizon is the only terminal condition; no reward is generated after
        it. Reaching q=0 does not terminate. step after termination is an error.
        """
        if self.elapsed >= self.horizon:
            raise RuntimeError("Episode terminated; reset before stepping")
        state = self.observation()
        torques = policy(state)
        rewards = reward(state, torques)
        if self.backend == "mjlab":
            import torch

            self.sim.data.ctrl[:, 0] = torch.as_tensor(torques, device="cuda:0",
                                                      dtype=self.sim.data.ctrl.dtype)
            for _ in range(SUBSTEPS):
                self.sim.step()
        else:
            for data, torque in zip(self.data, torques, strict=True):
                data.ctrl[0] = torque
                mujoco.mj_step(self.model, data, nstep=SUBSTEPS)
        self.elapsed += 1
        return self.observation(), rewards, self.elapsed == self.horizon

    def rollout(self, initial: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Collect S_0..S_T and R_1..R_T, shaped (T+1,B,3) and (T,B)."""
        states = [self.reset(initial)]
        rewards = []
        for _ in range(self.horizon):
            state, r, _ = self.step()
            states.append(state)
            rewards.append(r)
        return np.stack(states), np.stack(rewards)


def sample_initial(rng: np.random.Generator, count: int) -> np.ndarray:
    """Sample independent q~U[-1.2,1.2] rad and qdot~U[-1,1] rad/s, (B,2).

    This reset distribution defines the on-policy visitation weighting together
    with the fixed PD controller. Evaluation uses a separate seeded generator.
    """
    return rng.uniform([-1.2, -1.0], [1.2, 1.0], size=(count, 2))
