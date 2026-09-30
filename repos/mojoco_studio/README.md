# MoJoCo Studio

## Week 9 classical reaching and PD control

```bash
./build.sh classical_reaching
./build/classical_reaching
build/week5_plots/bin/python scripts/plot_week9.py
./build.sh classical_reaching_viewer
./build/classical_reaching_viewer
```

The [week 9 walkthrough](docs/week9_classical_reaching.md) connects Cartesian
targets, IK, quintic trajectories and sampled torque control. Fifteen cases
compare gains, target jumps, saturation, control frequency, noisy sensing and
feedforward laws. The viewer shows baseline versus a selected case; **N/B**
cycle cases, **Space** pauses, and **R** replays. Use `--help` for custom targets.

## Week 8 trajectories and TD learning

```bash
./build.sh trajectory_tracking
./build/trajectory_tracking
./build.sh cliff_walking
./build/cliff_walking
build/week5_plots/bin/python scripts/plot_week8.py
./build.sh trajectory_viewer
./build/trajectory_viewer  # optional live cubic/quintic comparison
```

Read [`docs/week8_trajectories_and_td.md`](docs/week8_trajectories_and_td.md)
for polynomial derivations, desired/actual velocity and acceleration comparisons,
and Cliff Walking TD(0), SARSA, Q-learning and Expected SARSA. The viewer supports
pause, replay, 1 ms stepping and speed controls. RL comparisons use 30 seeds and
report evaluation completion separately from completed-game returns.

## Week 7 dynamics and Monte Carlo learning

```bash
./build.sh robot_dynamics
./build/robot_dynamics
./build.sh blackjack_mc
./build/blackjack_mc
build/week5_plots/bin/python scripts/plot_week7.py  # optional plots
```

Start with [`docs/week7_dynamics_and_monte_carlo.md`](docs/week7_dynamics_and_monte_carlo.md).
It derives the two-link dynamics, compares six actuator/control modes plus two
additional gravity-compensation cases, and walks through Blackjack prediction,
exploring-starts control, epsilon-soft control and importance sampling.
Mathematical functions include review comments tied to the book notes.
Both programs check their results and write CSVs under `build/`.

View all eight dynamics experiments live in MuJoCo:

```bash
./build.sh robot_dynamics_viewer
./build/robot_dynamics_viewer
```

Press **1–8** to focus on a mode, **0** for all eight, **Space** to pause,
**R** to replay, and **[ / ]** to change playback speed. Green outlines show
the desired pose in servo modes. See the week 7 walkthrough for all controls.

## Week 6 inverse kinematics and Cartesian reaching

```bash
./build.sh inverse_kinematics
./build/inverse_kinematics
./build.sh cartesian_reaching
./build/cartesian_reaching
```

Read [`docs/week6_inverse_kinematics_and_cartesian_control.md`](docs/week6_inverse_kinematics_and_cartesian_control.md)
for the implementation walkthrough, standard/damped pseudoinverse comparison,
gain experiments, reachability and joint-limit handling, and the RL milestone.
The controller runs both ideal kinematics and MuJoCo velocity-servo dynamics.
Optional plots and a trajectory animation use `scripts/plot_week6.py --animate`.

## Week 5 Jacobians and dynamic programming

```bash
./build.sh gridworld_dp
./build/gridworld_dp
./build.sh jacobians
./build/jacobians
```

Start with [`docs/week5_jacobians_and_dp.md`](docs/week5_jacobians_and_dp.md).
It walks through Gridworld policy evaluation, policy iteration, value iteration,
and analytical/MuJoCo Jacobian comparisons, including tip velocity, singularities,
and force-to-torque mapping. Both programs check their results and save CSVs and
reports under `build/`. Optional plots use `scripts/plot_week5.py` with Matplotlib.

## Week 4 forward kinematics and finite MDP

Run fixed-policy Bellman evaluation and compare analytical two-link forward
kinematics against MuJoCo:

```bash
./build.sh finite_mdp
./build/finite_mdp
./build.sh forward_kinematics
./build/forward_kinematics
./simulate.sh two_link_arm  # optional interactive viewer
```

See
[`docs/week4_forward_kinematics_and_mdp.md`](docs/week4_forward_kinematics_and_mdp.md)
for the derivations, complete deliverable procedure, and error explanation.

## Week 3 coordinate frames

Build the frame-transform experiment and run all $SO(3)$/$SE(3)$ checks:

```bash
./build.sh frame_transforms
./build/frame_transforms
./simulate.sh frame_transforms  # optional interactive viewer
```

See [`docs/week3_frame_transforms.md`](docs/week3_frame_transforms.md) for the
frame convention, complete deliverable checklist, and MDP connection.

## Week 2 one-link arm

Build and inspect a one-link arm while reading body and site poses:

```bash
./build.sh one_link_arm
./build/one_link_arm
./simulate.sh one_link_arm  # optional interactive viewer
```

See [`docs/week2_one_link_arm.md`](docs/week2_one_link_arm.md) for the
rigid-body transformation derivation and experiments.

## Week 1 pendulum

Build and run the actuated pendulum experiment:

```bash
./build.sh pendulum
./build/pendulum
```

The program logs RL transitions to `build/pendulum_trajectory.csv` and plots
the state and action to `build/pendulum_plot.svg`. See
[`docs/week1_pendulum.md`](docs/week1_pendulum.md) for the walkthrough and the
answers to the Week 1 questions.

## Download

Download MoJoCo:
```
https://github.com/google-deepmind/mujoco/releases
```

Modify `MUJOCO_DIR` in `.env` file.
