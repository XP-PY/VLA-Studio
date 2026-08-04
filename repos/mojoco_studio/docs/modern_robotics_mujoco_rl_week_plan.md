# Integrated Learning Plan: Modern Robotics, MuJoCo, and Reinforcement Learning

## Intensive Version: 18 Hours per Week

## Goal

Build a strong foundation for Vision-Language-Action (VLA) research by learning three subjects together:

1. **Modern Robotics** — robot kinematics, dynamics, trajectories, planning, and control.
2. **MuJoCo** — physical simulation, robot models, actuators, sensors, contacts, and control loops.
3. **Reinforcement Learning** — sequential decision-making, value learning, policy gradients, and continuous control.

After completing these foundations, continue with **Underactuated Robotics** for nonlinear dynamics, optimal control, trajectory optimization, model predictive control, and advanced learning-based control.

The recommended sequence is:

$$
\text{Modern Robotics}
+
\text{MuJoCo}
+
\text{Reinforcement Learning}
\rightarrow
\text{Underactuated Robotics}
$$

---

# Core Resources

## Modern Robotics

**Book:** *Modern Robotics: Mechanics, Planning, and Control*  
**Authors:** Kevin M. Lynch and Frank C. Park

Official site:

<https://hades.mech.northwestern.edu/index.php/Modern_Robotics>

Main topics:

- Configuration space
- Rigid-body transformations
- Forward and inverse kinematics
- Jacobians
- Dynamics
- Trajectory generation
- Motion planning
- Feedback control
- Grasping and manipulation

My note book link:

<https://github.com/XP-PY/llm-prep-2026/blob/main/docs/Books/Robotics/Modern_Robotics.md>

---

## Reinforcement Learning

**Book:** *Reinforcement Learning: An Introduction, Second Edition*  
**Authors:** Richard S. Sutton and Andrew G. Barto

Official site:

<http://incompleteideas.net/book/the-book-2nd.html>

Main topics:

- Multi-armed bandits
- Markov decision processes
- Dynamic programming
- Monte Carlo learning
- Temporal-difference learning
- Function approximation
- Policy gradients
- Actor-critic methods

My note book link:

<https://github.com/XP-PY/llm-prep-2026/blob/main/docs/Books/Reinforcement_Learning/Reinforcement_Learning_An_Introduction.md>

---

## MuJoCo

Official documentation:

<https://mujoco.readthedocs.io/en/stable/>

Installation:

```bash
pip install mujoco
```

Main topics:

- MJCF models
- `MjModel` and `MjData`
- Joint positions and velocities
- Actuators
- Sensors
- Contacts
- Simulation stepping
- Position, velocity, and torque control
- Jacobians
- Inverse dynamics

---

# Overall Structure

This version assumes:

$$
18\text{ hours/week} \times 14\text{ weeks}
=
252\text{ total hours}
$$

The plan is intensive but still leaves enough time for implementation and review.

Use the following learning loop:

$$
\text{robotics theory}
\rightarrow
\text{MuJoCo implementation}
\rightarrow
\text{RL formulation}
\rightarrow
\text{VLA connection}
$$

Do not study the three tracks as unrelated courses.

---

# Weekly Time Allocation

Use this default weekly schedule:

| Activity | Hours per week |
|---|---:|
| Modern Robotics reading and videos | 4 |
| Robotics exercises and derivations | 3 |
| Sutton–Barto reading | 3 |
| RL implementation | 3 |
| MuJoCo implementation | 4 |
| VLA connection notes and review | 1 |
| **Total** | **18** |

A practical six-day schedule is:

| Day | Work |
|---|---|
| Monday | 2 h Modern Robotics + 1 h exercises |
| Tuesday | 2 h Sutton–Barto + 1 h RL code |
| Wednesday | 3 h MuJoCo implementation |
| Thursday | 2 h Modern Robotics + 1 h exercises |
| Friday | 1 h Sutton–Barto + 2 h RL code |
| Saturday | 1 h robotics exercises + 1 h MuJoCo + 1 h review/VLA notes |
| Sunday | Rest or optional catch-up |

Every week should produce at least one concrete output:

- A simulation
- A plot
- A controller
- An RL learning curve
- A derivation
- A comparison experiment
- A technical note

---

# 14-Week Integrated Curriculum

# Phase 1: Foundations

## Week 1: Configuration Space, RL Basics, and MuJoCo State

### Modern Robotics

Study:

- Chapter 1 overview
- Chapter 2: Configuration Space
- Degrees of freedom
- Joint coordinates
- Constraints
- Open-chain and closed-chain robots

### Reinforcement Learning

Study Chapter 1:

- Agent
- Environment
- State
- Action
- Reward
- Return
- Episode
- Policy
- Value function
- Episodic and continuing tasks

### MuJoCo

Learn:

- `MjModel`
- `MjData`
- `data.qpos`
- `data.qvel`
- `data.ctrl`
- `mujoco.mj_step`
- Passive viewer
- Simulation timestep

### Main Connection

A robot state may be:

$$
s_t =
\begin{bmatrix}
q_t \\
\dot q_t \\
x_{\text{object},t}
\end{bmatrix}
$$

An action may be:

$$
a_t = \tau_t
$$

MuJoCo implements the transition:

$$
s_{t+1}\sim p(s_{t+1}\mid s_t,a_t)
$$

### Deliverable

Create a pendulum simulation that:

1. Reads joint position and velocity.
2. Applies an actuator command.
3. Advances the simulation.
4. Records state, action, and time.
5. Plots angle, angular velocity, and control signal.

---

## Week 2: Rigid-Body Transformations and Bandits

### Modern Robotics

Study the first half of Chapter 3:

- Coordinate frames
- Rotation matrices
- Homogeneous transformations
- Transforming points between frames
- Composition and inverse transformations

### Reinforcement Learning

Study Chapter 2:

- Multi-armed bandits
- Exploration versus exploitation
- Sample-average estimates
- Incremental updates
- $\epsilon$-greedy
- Optimistic initialization
- UCB

### MuJoCo

Practice:

- Bodies
- Joints
- Geometries
- Sites
- Actuators
- Reading body and site poses
- Building a one-link MJCF model

### Deliverables

Implement:

- Ten-armed bandit
- $\epsilon$-greedy
- UCB
- Incremental action-value estimation

Create a simple MuJoCo model with:

- One link
- One hinge
- One actuator
- One site at the link end

---

# Phase 2: Kinematics and MDPs

## Week 3: Twists, Exponential Coordinates, and Finite MDPs

### Modern Robotics

Finish Chapter 3:

- Angular velocity
- Twists
- Screw axes
- Exponential coordinates
- Matrix logarithm and exponential
- Adjoint representation

### Reinforcement Learning

Study Chapter 3:

- Markov property
- Transition probabilities
- Rewards
- Discounting
- Policies
- State-value functions
- Action-value functions
- Bellman equations

### MuJoCo

Practice:

- Reading site rotation matrices
- Converting MuJoCo poses into homogeneous transforms
- Comparing world, base, and end-effector frames

### Deliverable

Write utilities for:

- Rotation matrix validation
- Homogeneous transform multiplication
- Transform inversion
- Point transformation
- Relative pose calculation

Connect these utilities to MuJoCo site poses.

---

## Week 4: Forward Kinematics

### Modern Robotics

Study Chapter 4:

- Forward kinematics
- Product of exponentials
- Space frame
- Body frame
- Screw axes
- End-effector pose calculation

### Reinforcement Learning

Review Chapter 3 and implement:

- A small MDP
- Bellman expectation equations
- State-value calculation under a fixed policy

### MuJoCo

Use a two-link arm:

- Set joint positions
- Read end-effector site pose
- Compare analytical and simulated forward kinematics

### Main Connection

$$
x=f(q)
$$

A reaching observation can be:

$$
s_t=
\begin{bmatrix}
q_t \\
\dot q_t \\
x_{\text{target}}-x_{\text{ee},t}
\end{bmatrix}
$$

A basic reward can be:

$$
r_t=-\|x_{\text{target}}-x_{\text{ee},t}\|_2
$$

### Deliverable

For at least 20 random joint configurations:

1. Compute forward kinematics yourself.
2. Set the same configuration in MuJoCo.
3. Read the end-effector pose.
4. Compare numerical errors.
5. Explain the largest error cases.

---

## Week 5: Jacobians and Dynamic Programming

### Modern Robotics

Study Chapter 5:

- Space Jacobian
- Body Jacobian
- Joint velocity
- End-effector twist
- Singularities
- Static force mapping

### Reinforcement Learning

Study Chapter 4:

- Policy evaluation
- Policy improvement
- Policy iteration
- Value iteration
- Generalized policy iteration

### MuJoCo

Practice:

- Computing Jacobians
- Using MuJoCo Jacobian functions
- Comparing analytical and simulator Jacobians
- Measuring end-effector velocity

### Deliverables

Implement:

- Policy evaluation
- Policy iteration
- Value iteration

Use Gridworld.

Also compare your own Jacobian against MuJoCo for random robot configurations.

---

## Week 6: Inverse Kinematics and Cartesian Control

### Modern Robotics

Study Chapter 6:

- Numerical inverse kinematics
- Newton-Raphson iteration
- Jacobian pseudoinverse
- Convergence
- Singularities
- Reachability

### Reinforcement Learning

Consolidate Chapters 3–4:

- MDP formulation
- Bellman equations
- Dynamic programming assumptions
- Model-based versus model-free learning

### MuJoCo

Implement:

$$
\dot q=J(q)^\dagger K(x_d-x)
$$

Practice:

- Cartesian reaching
- Jacobian pseudoinverse control
- Damped least squares
- Target reachability
- Joint-limit handling

### Milestone

Be able to explain the difference among:

- Robot model
- Planner
- Policy
- Controller
- Value function
- Environment dynamics

### Deliverable

Build a Cartesian reaching controller and compare:

- Standard pseudoinverse
- Damped pseudoinverse
- Different gain values
- Targets near singularities

---

# Phase 3: Dynamics and Model-Free RL

## Week 7: Robot Dynamics and Monte Carlo Learning

### Modern Robotics

Study the first half of Chapter 8:

- Mass matrix
- Coriolis and centrifugal effects
- Gravity terms
- Forward dynamics
- Inverse dynamics

Central equation:

$$
M(q)\ddot q+C(q,\dot q)\dot q+g(q)=\tau
$$

Focus first on physical meaning.

### Reinforcement Learning

Study Chapter 5:

- Monte Carlo prediction
- First-visit and every-visit methods
- Monte Carlo control
- Exploring starts
- On-policy learning
- Importance sampling

### MuJoCo

Run the same robot under:

1. No control
2. Joint damping
3. Constant torque
4. Gravity compensation
5. Position actuation
6. Torque actuation

### Deliverable

Implement Monte Carlo prediction and control in Blackjack.

Write a short comparison of:

- Uncontrolled dynamics
- Damping
- Gravity compensation
- Position actuators
- Torque actuators

---

## Week 8: Trajectory Generation and TD Learning

### Modern Robotics

Study Chapter 9:

- Joint-space trajectories
- Cartesian trajectories
- Cubic time scaling
- Quintic time scaling
- Velocity and acceleration continuity

### Reinforcement Learning

Study Chapter 6:

- TD prediction
- TD error
- SARSA
- Q-learning
- Expected SARSA
- On-policy versus off-policy learning

### MuJoCo

Implement:

- Cubic joint trajectories
- Quintic joint trajectories
- Trajectory playback
- Tracking-error logging
- Velocity and acceleration plots

### Deliverables

Implement:

- TD(0)
- SARSA
- Q-learning
- Expected SARSA

Use Cliff Walking.

Compare cubic and quintic trajectory smoothness in MuJoCo.

---

## Week 9: PD Control and the Complete Classical Pipeline

### Modern Robotics

Study selected parts of Chapter 11:

- Position control
- Velocity control
- Torque control
- Feedback control
- Feedforward control
- Error dynamics

A PD controller is:

$$
\tau_t=
K_p(q_d-q_t)+K_d(\dot q_d-\dot q_t)
$$

### Reinforcement Learning

Review Chapters 5–6:

- Bootstrapping
- Bias and variance
- On-policy versus off-policy
- Exploration during control

### MuJoCo

Experiment with:

- Small and large $K_p$
- Small and large $K_d$
- Target jumps
- Smooth targets
- Torque limits
- Different control frequencies
- Sensor noise

### Milestone Project

Build a complete classical reaching system:

1. Specify a Cartesian target.
2. Solve inverse kinematics.
3. Generate a smooth joint trajectory.
4. Track it with PD control.
5. Execute it in MuJoCo.
6. Plot tracking error, torque, and end-effector path.

### Important Connection

$$
\text{VLA/RL policy}
\rightarrow
\text{desired target}
\rightarrow
\text{trajectory generator}
\rightarrow
\text{PD controller}
\rightarrow
\text{robot torque}
$$

---

# Phase 4: Function Approximation and Continuous Control

## Week 10: Multi-Step Learning and Value Function Approximation

### Modern Robotics

Review:

- Jacobians
- Dynamics
- State representation
- Feedback control

### Reinforcement Learning

Study:

- Chapter 7 selectively
- Chapter 9

Topics:

- $n$-step returns
- Feature vectors
- Linear value functions
- Semi-gradient methods
- Generalization
- Tile coding

Replace:

$$
V(s)=\text{table}[s]
$$

with:

$$
\hat V(s,w)=w^\top x(s)
$$

### MuJoCo

Use:

- Pendulum
- Cart-pole
- A simple continuous-state arm

Practice:

- Observation normalization
- Episode termination
- Reward logging
- State-feature construction

### Deliverable

Implement semi-gradient TD using:

- Linear features
- Tile coding or radial-basis features
- Continuous position and velocity observations

---

## Week 11: Approximate Control and Action-Space Design

### Modern Robotics

Study more deeply:

- Joint-space control
- Task-space control
- Feedforward plus feedback
- Control limitations
- Contact and disturbance intuition

### Reinforcement Learning

Study:

- Chapter 10
- Chapter 12 selectively

Topics:

- Semi-gradient SARSA
- Average reward
- Differential SARSA
- Eligibility traces

### Action-Space Comparison

#### Policy outputs a joint target

$$
a_t=q_{d,t}
$$

Controller:

$$
\tau_t=
K_p(q_{d,t}-q_t)-K_d\dot q_t
$$

#### Policy outputs a Cartesian delta

$$
a_t=\Delta x_t
$$

#### Policy outputs torque

$$
a_t=\tau_t
$$

### MuJoCo Experiment

Compare:

1. Hand-designed PD control
2. RL policy outputting target position
3. RL policy outputting torque directly

Measure:

- Stability
- Smoothness
- Learning difficulty
- Reward
- Energy
- Robustness
- Sensitivity to action frequency

---

# Phase 5: Policy Gradients and Modern RL

## Week 12: Policy Gradients and Actor-Critic

### Modern Robotics

Review:

- Action representations
- Low-level execution
- Control frequency
- Trajectory tracking
- Safety limits

### Reinforcement Learning

Study Chapter 13:

- Policy-gradient theorem
- REINFORCE
- Baselines
- Actor-critic
- Advantage estimates
- Continuing tasks

Core update:

$$
\nabla_\theta J(\theta)
\approx
\mathbb E
\left[
\nabla_\theta\log\pi_\theta(a_t\mid s_t)\hat A_t
\right]
$$

For continuous actions:

$$
a_t\sim
\mathcal N
\left(
\mu_\theta(s_t),
\sigma_\theta(s_t)^2
\right)
$$

### MuJoCo

Use:

- Pendulum
- Simple reaching
- Gaussian action distributions
- Action clipping
- Observation normalization

### Deliverables

Implement:

1. REINFORCE
2. REINFORCE with a value baseline
3. One-step actor-critic

---

## Week 13: PPO and SAC

Sutton and Barto provides the foundations, but PPO and SAC require additional references.

Useful resources:

- OpenAI Spinning Up: <https://spinningup.openai.com/>
- PPO paper: <https://arxiv.org/abs/1707.06347>
- SAC paper: <https://arxiv.org/abs/1801.01290>
- Gymnasium MuJoCo environments: <https://gymnasium.farama.org/environments/mujoco/>

### PPO

Properties:

- On-policy
- Uses fresh rollouts
- Policy-gradient based
- Closely related to RLHF

Clipped objective:

$$
L^{\text{clip}}(\theta)
=
\mathbb E_t
\left[
\min
\left(
r_t(\theta)\hat A_t,
\operatorname{clip}
(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)
\right]
$$

### SAC

Properties:

- Off-policy
- Uses replay data
- Designed for continuous action spaces
- Encourages both return and entropy

### Deliverable

Train at least one baseline using PPO and one using SAC on:

- Pendulum, or
- A simple MuJoCo reaching task

Compare:

- Sample efficiency
- Stability
- Final return
- Action smoothness
- Training time
- Sensitivity to hyperparameters

---

# Phase 6: Integrated Capstone

## Week 14: Robot Learning and VLA-Oriented Capstone

Build a custom MuJoCo continuous-control reaching environment.

### Observation

$$
s_t=
[
q_t,
\dot q_t,
x_{\text{target}}-x_{\text{ee},t}
]
$$

Optional additions:

- Previous action
- Object pose
- End-effector velocity
- RGB observation
- Depth observation

### Action

Compare at least two:

$$
a_t=\Delta q_t
$$

$$
a_t=\Delta x_t
$$

$$
a_t=\tau_t
$$

### Reward

A possible reward:

$$
r_t=
-\lambda_d
\|x_{\text{target}}-x_{\text{ee},t}\|_2
-\lambda_a\|a_t\|_2^2
+b_{\text{success}}
$$

### Required Experiments

Evaluate:

- Success rate
- Episode return
- Final distance
- Action smoothness
- Energy use
- Generalization to new targets
- Changed link mass
- Observation noise
- Action delay
- Different action frequencies

### Classical Baseline

Compare the learned policy against:

- Inverse kinematics
- Trajectory generation
- PD control

### VLA Connection

Write a final report answering:

1. What observations would a VLA receive?
2. What action representation would it predict?
3. In which coordinate frame?
4. At what frequency?
5. Would it predict one action or an action chunk?
6. What low-level controller would execute the action?
7. Where could RL improve the policy?
8. What reward signal could be used?
9. What safety constraints are necessary?
10. Which parts should remain classical rather than learned?

---

# Recommended Modern Robotics Chapter Priority

## High Priority

1. Chapter 2: Configuration Space
2. Chapter 3: Rigid-Body Motions
3. Chapter 4: Forward Kinematics
4. Chapter 5: Velocity Kinematics and Statics
5. Chapter 6: Inverse Kinematics
6. Chapter 8: Dynamics
7. Chapter 9: Trajectory Generation
8. Chapter 11: Robot Control
9. Chapter 12: Grasping and Manipulation

## Study Later

- Chapter 10: Motion Planning
- Chapter 7: Closed-Chain Kinematics
- Chapter 13: Wheeled Mobile Robots

---

# Recommended Sutton–Barto Chapter Priority

## High Priority

- Chapter 1
- Chapter 2
- Chapter 3
- Chapter 4
- Chapter 5
- Chapter 6
- Chapter 9
- Chapter 10
- Chapter 13

## Medium Priority

- Chapter 7
- Chapter 8
- Chapter 11
- Chapter 12

## Lower Immediate Priority

- Chapter 14
- Chapter 15
- Chapter 16
- Chapter 17

---

# Suggested Repository Structure

```text
robotics-rl-learning/
├── modern_robotics/
│   ├── configuration_space/
│   ├── transformations/
│   ├── forward_kinematics/
│   ├── jacobians/
│   ├── inverse_kinematics/
│   ├── dynamics/
│   ├── trajectories/
│   └── control/
├── reinforcement_learning/
│   ├── bandits/
│   ├── mdps/
│   ├── dynamic_programming/
│   ├── monte_carlo/
│   ├── temporal_difference/
│   ├── function_approximation/
│   ├── policy_gradient/
│   ├── ppo/
│   └── sac/
├── mujoco/
│   ├── models/
│   ├── pendulum/
│   ├── two_link_arm/
│   ├── pd_control/
│   ├── jacobian_control/
│   ├── reaching/
│   └── contact_experiments/
├── integrated_projects/
│   ├── classical_reaching/
│   ├── actor_critic_pendulum/
│   ├── ppo_reaching/
│   ├── sac_reaching/
│   └── final_capstone/
└── notes/
    ├── action_representations.md
    ├── coordinate_frames.md
    ├── policy_vs_controller.md
    ├── reward_design.md
    ├── control_frequency.md
    └── vla_connections.md
```

---

# VLA Paper Reading Checklist

For every VLA paper, record:

## Observation

- Images
- Language
- Joint positions
- Joint velocities
- Previous actions
- Observation history
- Force or tactile information

## Action

- Joint position
- Joint-position delta
- Joint velocity
- Joint torque
- End-effector pose
- End-effector delta pose
- Gripper command
- Action chunk

## Coordinate Frame

- World
- Robot base
- Camera
- End effector
- Object

## Execution

- Policy frequency
- Controller frequency
- Interpolation
- Low-level controller
- Action clipping
- Joint limits
- Collision checking
- Safety constraints

## Reinforcement Learning

- On-policy or off-policy
- Online or offline
- Reward source
- Value model
- Critic
- Advantage estimation
- Replay buffer
- Policy regularization
- Reference policy
- Which module receives RL updates

---

# When to Begin Underactuated Robotics

Begin *Underactuated Robotics* after you can explain and implement:

1. Forward kinematics
2. Inverse kinematics
3. Jacobians
4. Robot dynamics
5. PD control
6. Smooth trajectory generation
7. Bellman equations
8. TD learning
9. Function approximation
10. Policy gradients
11. Actor-critic
12. PPO or SAC
13. A MuJoCo continuous-control task

Then study:

- Nonlinear systems
- Stability
- Lyapunov functions
- LQR
- Trajectory optimization
- Model predictive control
- System identification
- Robust control
- Learning-based control
- Underactuated manipulation
- Locomotion

Official notes:

<https://underactuated.mit.edu/>

---

# Expected Outcome

After 14 weeks at 18 hours per week, you should understand the complete robot-learning pipeline:

$$
\text{visual and language observations}
\rightarrow
\text{VLA or RL policy}
\rightarrow
\text{action representation}
\rightarrow
\text{trajectory or controller}
\rightarrow
\text{MuJoCo or real robot dynamics}
\rightarrow
\text{new observation and reward}
$$

This foundation will prepare you to study and reproduce:

- ACT
- RT-1
- RT-2
- OpenVLA
- Octo
- $\pi_0$
- $\pi_{0.5}$
- FAST-based VLA policies
- RL-fine-tuned VLA policies
