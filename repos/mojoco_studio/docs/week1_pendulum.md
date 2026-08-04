# Week 1: Actuated Pendulum in MuJoCo

This exercise turns one MuJoCo simulation into a sequence of reinforcement
learning transitions:

$$
(s_t, a_t, r_t, s_{t+1}).
$$

The implementation uses the C++ API already configured in this repository.
The correspondence with the Python names in the weekly plan is direct:

| Concept | C++ API in this exercise | Python equivalent |
|---|---|---|
| Immutable model | `mjModel* model` | `mujoco.MjModel` |
| Mutable simulation state | `mjData* data` | `mujoco.MjData` |
| Joint position | `data->qpos[...]` | `data.qpos[...]` |
| Joint velocity | `data->qvel[...]` | `data.qvel[...]` |
| Actuator command | `data->ctrl[...]` | `data.ctrl[...]` |
| Advance physics | `mj_step(model, data)` | `mujoco.mj_step(model, data)` |

## Run it

Run these commands from the repository root:

```bash
./build.sh pendulum
./build/pendulum
```

The program creates:

- `build/pendulum_trajectory.csv`: every observed RL transition.
- `build/pendulum_plot.svg`: angle, angular velocity, and action plots.

Open the model in MuJoCo's interactive viewer with:

```bash
./simulate.sh pendulum
```

In the viewer, use the `hinge_motor` slider in the Control panel to apply a
manual torque. The standalone program instead applies a reproducible sine-wave
torque, which makes its recorded experiment easy to rerun and compare.

## Read the model first

The model is in `assets/pendulum.xml`. Its important choices are:

- One body with one hinge joint, so the configuration is one angle $q$.
- One motor attached to the hinge.
- `gear="1"`, so a control value of 1 produces a generalized torque of 1 N m.
- A 2 ms physics timestep, so MuJoCo integrates the dynamics at 500 Hz.
- A control range of $[-3, 3]$ N m.

The rod hangs downward at $q=0$. The simulation starts at $q=0.35$ rad, and
the red marker shows the end-point position for the target $q=0.75$ rad.

## Read one transition carefully

The core loop in `src/pendulum.cpp` does the following:

1. Copy the current state $s_t=[q_t,\dot q_t]$ out of `mjData`.
2. Compute the torque action $a_t$ and assign it to `data->ctrl`.
3. Call `mj_step` ten times. This holds the action for 20 ms: the policy acts
   at 50 Hz while MuJoCo integrates physics at 500 Hz.
4. Read the next state $s_{t+1}$.
5. Calculate the reward and append the complete transition to the CSV file.

It is important to copy the current state before calling `mj_step`: MuJoCo
updates the same `mjData` object in place.

The reward used here is:

$$
r_t =
-\operatorname{wrap}(q_{target}-q_{t+1})^2
-0.05\dot q_{t+1}^2
-0.001a_t^2.
$$

It rewards proximity to the target while mildly discouraging high speed and
large actions. This is only a reward definition; the sine-wave policy is not
yet learning to maximize it.

## Answers to the Week 1 questions

1. **What is the RL state?**
   $s_t=[q_t,\dot q_t]$. Time is logged for analysis but is not part of the
   physical state. If the policy is explicitly time-dependent, time or phase
   would also need to be included in its observation.

2. **What is the action?**
   One scalar hinge-torque command, $a_t=\tau_t$, clipped to $[-3,3]$ N m.

3. **What physical quantity does the actuator command represent?**
   The XML uses a MuJoCo motor with `gear=1`, so `ctrl` maps directly to
   generalized hinge torque in N m.

4. **What determines the next state?**
   The current state, held action, mass and inertia, gravity, damping, timestep,
   numerical integrator, contacts, and any other forces in the model.

5. **Is the state fully observable?**
   Yes for this idealized simulator: $q$ and $\dot q$ are sufficient for its
   Markov state. A real robot normally measures noisy or delayed observations.

6. **Is the task episodic or continuing?**
   Pendulum dynamics are naturally continuing. This program truncates one run
   after eight seconds, which can be treated as a finite training episode.

7. **What reward encourages a target angle?**
   Negative wrapped squared angle error, optionally with velocity and action
   penalties as shown above.

8. **How does action frequency affect behavior?**
   A lower frequency holds each action longer and gives the policy less ability
   to react. A higher frequency gives finer control but increases computation
   and can produce noisy actions. Physics integration should usually remain
   faster than policy inference.

9. **Where does the low-level controller fit?**
   This exercise sends torque directly, so it has no separate low-level
   controller. If a policy outputs a desired angle, a PD controller would sit
   between the policy and `data->ctrl` and convert that target into torque.

10. **How could a VLA policy replace this structure?**
    Replace `choose_action` with a policy that consumes images, language, and
    robot observations. It could output torque directly, but a desired joint or
    end-effector target executed by a fast low-level controller is generally a
    more structured action interface.

## Small experiments

After the baseline works, change one variable at a time and compare the CSV and
plot:

1. Set `kActionAmplitude` to zero and observe passive motion.
2. Change `kInitialAngle` and compare trajectories.
3. Change `kControlFrequency` to 10 Hz and 100 Hz.
4. Change joint `damping` in the XML.
5. Replace `choose_action` with a constant torque or a PD controller.

These comparisons make the transition function
$p(s_{t+1}\mid s_t,a_t)$ concrete before introducing a learned policy.
