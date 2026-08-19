# MoJoCo Studio

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
