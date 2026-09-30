# Week 8: Trajectory Generation and TD Learning

This implements the remaining week 8 exercises using your **Modern Robotics
Chapter 9** and **Reinforcement Learning Chapter 6** notes. Mathematical source
comments include the defining equations, derivations, units, timing conventions
and limitations. Section numbers in those comments refer to your notes.

There are two independent experiments: polynomial trajectories tracked by the
week 7 arm, and tabular TD prediction/control in Cliff Walking. No RL controller
is driving the arm in this week.

## Run the complete exercise

From the repository root:

```bash
./build.sh trajectory_tracking
./build/trajectory_tracking
./build.sh cliff_walking
./build/cliff_walking
build/week5_plots/bin/python scripts/plot_week8.py
```

The batch programs run without a display and perform their checks automatically.
The plotter needs NumPy and Matplotlib; on a fresh checkout, use an environment
with those packages instead of the existing week 5 environment. Cliff Walking
itself uses only the C++ standard library, although `build.sh` uses the project's
usual MuJoCo linker setup. Outputs are replaced on each run under ignored `build/`.

Live side-by-side MuJoCo playback:

```bash
./build.sh trajectory_viewer
./build/trajectory_viewer
```

| Control | Effect |
|---|---|
| Space | Pause/resume both arms |
| R | Reset both initial states and replay |
| Right arrow | Pause and advance **1 ms** |
| [ / ] | Halve/double speed, 0.125× to 2× |
| Left/right drag | Rotate/pan the shared camera |
| Wheel; C | Zoom; reset camera |
| Esc | Close |

Playback starts at half speed and stops at 3.5 simulation seconds. Green geometry
marks the moving desired pose; solid links show the actual simulated robot.
Because tracking errors are small, the poses nearly overlap. Pause and step near
0.5 s or 2.5 s to inspect the desired acceleration at the motion boundaries.
The overlays show desired velocity, desired/actual acceleration and joint error.
Speed changes only the scheduling of fixed 1 ms steps, not the physics timestep.

The viewer shares the batch's `Experiment`, reference generator and RK4 control
callback. It neither needs nor overwrites batch CSVs during normal viewing.
SDL2/OpenGL and a graphical display are needed, as for the week 7 viewer.

```bash
./build/trajectory_viewer --check       # compare with batch CSVs, no display
./build/trajectory_viewer --smoke-test  # render once, save build/week8_viewer.ppm, exit
```

## Source reading order

| File | Main functions to study |
|---|---|
| `src/trajectory_generation.h` | `scaling`, `joint_line`, `reference`, `minimum_duration`, `tip`, `cartesian_line` |
| `src/trajectory_experiment.h` | `control`, `Experiment::step`; shared batch/viewer execution |
| `src/trajectory_tracking.cpp` | Boundary/derivative tests and measured tracking comparisons |
| `src/trajectory_viewer.cpp` | Live playback, moving reference overlay, CSV equivalence check |
| `src/cliff_learning.h` | `step`, `epsilon_policy`, `td_update`, `control_update`, `train_episode` |
| `src/cliff_walking.cpp` | Fixed-policy prediction, target regression cases, seeded training/evaluation |
| `scripts/plot_week8.py` | Reference/actual plots and across-seed statistics |

For robotics, read **notes 9.1, 9.2.1 and 9.2.2** alongside the source. For RL,
read **6.1, 6.4, 6.5 and 6.6**. General screw/SE(3) interpolation, via-point
splines, torque-constrained time-optimal scaling, Double Q-learning and
afterstates are outside this week's implementation deliverables.

## Derive the trajectory before tracking it

A path supplies geometry; a time scaling supplies timing. Here

```text
q(s) = q_start + s * delta,     delta = q_goal - q_start
dq   = delta * ds
ddq  = delta * dds
```

The general chain rule is `dq=q'(s) ds` and
`ddq=q'(s) dds+q''(s) ds²`; the second term vanishes specifically for this
straight joint-space path. All joints share the same dimensionless progress s.
Coordinates are shoulder angle and relative elbow angle, in radians.

The experiment uses q_start=(0.6,−0.8), q_goal=(−0.3,0.6), a 0.5 s stationary
start, a **2 s movement**, then a 1 s stationary goal. Joint coordinates remain
on the chosen unwrapped branch; no shortest-angle wrapping is performed.

Set `u=t/T` for time local to the movement segment. Start with a cubic polynomial
and impose four conditions: s(0)=0, s(T)=1, ds(0)=ds(T)=0. Solving the linear
coefficient equations gives:

```text
s   = 3u² - 2u³
ds  = 6u(1-u) / T
dds = (6-12u) / T²
```

Adding zero endpoint accelerations creates six constraints, giving the quintic:

```text
s   = 10u³ - 15u⁴ + 6u⁵
ds  = 30u²(1-u)² / T
dds = 60u(1-u)(1-2u) / T²
```

Each differentiation in time introduces another factor 1/T. This is why merely
differentiating the polynomial in u and forgetting duration produces wrong
velocity/acceleration units.

| Property at stationary joins | Cubic | Quintic |
|---|---|---|
| Position | Continuous | Continuous |
| Velocity | Continuous, zero at endpoints | Continuous, zero at endpoints |
| Acceleration | Jumps between zero and ±6 delta/T² | Continuous, zero at endpoints |
| Jerk | Ideal impulses at acceleration jumps | Finite, but can jump at joins |

`scaling` returns polynomial-side derivatives at exact endpoints. `reference`
returns zero derivatives strictly outside the movement interval. This makes the
one-sided cubic acceleration mismatch explicit. `trajectory_boundaries.csv`
records both limits, so it does not depend on floating-point simulation time
landing exactly on a join. For joint 2, the start jump is **0 → 2.1 rad/s²**;
the final jump is **−2.1 → 0 rad/s²**. Quintic joins are zero on both sides.

### Smoothness is different from peak speed and feasibility

The exact scalar peaks are:

| Profile | max ds | max abs(dds) |
|---|---|---|
| Cubic | 3/(2T) | 6/T² |
| Quintic | 15/(8T) | 10/(sqrt(3) T²) |

At the same duration, quintic's peak speed is **25% higher**. In our joint-2
reference, cubic/quintic peak speeds are 1.05/1.3125 rad/s and acceleration
magnitudes are 2.10/2.0207 rad/s². Better boundary smoothness does not mean every
other metric becomes smaller.

Given positive velocity/acceleration bounds, `minimum_duration` derives

```text
T_min = max over joints i of:
        cv*abs(delta_i)/velocity_limit_i,
        sqrt(ca*abs(delta_i)/acceleration_limit_i)
```

with the peak constants from the table. Default reference bounds are 1.5 rad/s
and 3 rad/s² for each joint, satisfied analytically by T=2 s. The notes' separate
example `delta=(1,.5), vlim=(1,.75), alim=(2,1)` is checked against
T_cubic=sqrt(3) and T_quintic=1.875 s.

These are bounds on the **desired** trajectory. They do not certify actual
tracking, motor torque, collisions or jerk. All-stationary requests return zero
minimum duration and should be held rather than passed to a zero-duration scaler.

### What the MuJoCo controller actually does

Both profiles use the same vertical arm, timestep, initial state, gains and torque
bounds from week 7. The moving-reference baseline is

```text
tau = clip(Kp*(q_ref-q) + Kd*(dq_ref-dq) + g(q), -12, 12) N m
Kp = diag(25,15), Kd = diag(5,3)
```

The derivative term follows the desired velocity. Gravity compensation uses the
actual pose. There is no M*qdd_ref or Coriolis feedforward; the controller is a
project baseline for executing Chapter 9 trajectories, not the trajectory
generation formula itself. Week 9 can investigate controller design further.

The position-actuator group is disabled. Feedback is recomputed at each RK4
stage using that stage's simulation time, avoiding an artificial once-per-frame
control delay. Logged `qacc` is MuJoCo's instantaneous acceleration at the accepted
state, not a finite difference. No contacts or joint limits are introduced.

Measured results from this implementation:

| Metric | Cubic | Quintic |
|---|---:|---:|
| RMS joint-error norm, whole 3.5 s run (rad) | 0.003316 | 0.004175 |
| Maximum joint-error norm (rad) | 0.006856 | 0.007692 |
| Final joint-error norm (rad) | 0.00000621 | 0.00000377 |
| Peak actual component acceleration (rad/s²) | 2.3077 | 2.0165 |
| Peak absolute motor torque (N m) | 4.6622 | 4.7346 |
| Saturated logged samples | 0 | 0 |

Quintic is smoother at rest joins, yet its RMS tracking error is slightly larger
with this timing/controller. Compare desired and actual derivatives directly;
do not infer tracking accuracy solely from the polynomial degree. A reference
acceleration jump also does not automatically mean the actual acceleration
jumps: this controller consumes q_ref and dq_ref, not qdd_ref.

![Reference and actual derivatives](../build/week8_trajectories.png)

### Joint lines and Cartesian lines

`tip(q)` maps joint angles to world [x,z] using two-link forward kinematics.
Plotting the joint reference through FK produces a curved tip path. Cubic and
quintic traverse the **same geometry** with different timing.

`cartesian_line(s)` supplies the straight positional interpolation between the
endpoint tips for comparison. This line is **not executed**: no Cartesian IK or
independent orientation task is imposed. A planar two-joint arm cannot generally
specify arbitrary position and orientation simultaneously. The full Cartesian
target → IK → trajectory → tracking pipeline is the week 9 milestone.

![Tracking error, torque and tip geometry](../build/week8_tracking.png)

## Cliff Walking: get the transition semantics right

The grid has four rows and twelve columns, with rows increasing downward.
Start=(3,0), goal=(3,11), and bottom-row columns 1–10 are the cliff. Actions are
Up, Right, Down, Left. Moves at boundaries stay in the grid.

- Ordinary moves, including the move into goal, reward **−1**.
- A cliff fall rewards **−100** and moves immediately to start.
- **Falling does not terminate the episode.** Only reaching goal does.
- Gamma=1. Terminal continuation value is zero.

There are 37 occupiable nonterminal states. Cliff cells are transition events,
not states in which the agent waits for another action. The shortest nonfalling
route is Up, eleven Rights, Down: 13 steps, return −13.

### TD(0): predict a fixed policy

The predictor uses a fixed epsilon-soft policy: prefer Up until row ≤1, then
Right; in the last column prefer Down. Epsilon=.1 adds exploration, so even this
policy can fall. It is independent of V and is not being improved by TD(0).

After each transition:

```text
delta_t = R_(t+1) + gamma * V_old(S_(t+1)) - V_old(S_t)
V(S_t) <- V_old(S_t) + alpha * delta_t
```

Use zero continuation at goal; after a fall, use V(start). Even a self-loop must
read the old successor value before writing the updated state value. Unlike MC,
the update happens immediately and bootstraps from an estimate.

The experiment runs 200,000 episodes and uses per-state visit counts with
`alpha_s=(10+N(s))^-0.6`. The diminishing schedule meets the standard step-size
sum conditions if a state is visited infinitely often; it does not guarantee a
particular finite-run error. A separate synchronous Bellman evaluator provides
a reference **only for checking**. TD sees sampled transitions, never its values.

The resulting equal-state RMSE is **0.6634 reward units**, with at least 4679
visits to every active state. The reference is an undiscounted absorbing-policy
calculation; its validity relies on eventual goal reachability and finite mean
absorption time, not a gamma<1 contraction argument.

![TD prediction](../build/week8_td_prediction.png)

### Three control methods, three successor targets

Each method updates `Q(s,a) += alpha*(target-Q(s,a))` with alpha=.5, gamma=1.

| Method | Nonterminal target | Continuation policy |
|---|---|---|
| SARSA | r + Q(s',a') | Sampled epsilon-greedy behavior |
| Q-learning | r + max_a Q(s',a) | Greedy target despite exploratory behavior |
| Expected SARSA | r + sum_a pi(a\|s') Q(s',a) | Expected epsilon-greedy behavior |

For terminal transitions all three targets are just r. All backups read
pre-update estimates. SARSA samples a' **before updating**, carries it forward
and actually executes it. Resampling after the update would break the specified
SARSA sequence, especially for a boundary self-loop or cliff reset.

The shared epsilon-greedy helper handles ties explicitly:

```text
pi(a|s) = epsilon/4 + (1-epsilon)/number_of_greedy_actions, if a is greedy
          epsilon/4, otherwise
```

With a unique greedy action and epsilon=.1, probabilities are .925 and .025 for
each alternative. If all Q values tie, all four actions have probability .25.
Expected SARSA uses this exact distribution, rather than assuming one maximizer.
It averages action choices without needing transition probabilities.

The source tests distinct targets using q'=(4,4,1,−1), epsilon=.2, r=2, gamma=.9:
SARSA selecting the value-1 action targets 2.9; Q-learning targets 5.6; Expected
SARSA targets 5.24. Terminal tests deliberately put garbage in the terminal table
entry to ensure it cannot leak into a backup.

### Compare learning behavior and frozen evaluation separately

Each method trains for **1000 episodes on each of 30 independent seeds**.
The training curve is online return under epsilon=.1, while Q continues changing.
After training, freeze each Q table and evaluate 200 fresh games under both
epsilon=0 and epsilon=.1, using separate RNG streams.

| Method | Last-100 online return | Frozen greedy completed return | Greedy completion | Frozen exploratory completed return | Exploratory completion |
|---|---:|---:|---:|---:|---:|
| SARSA | −27.15 | −17.16 | 83.33% | −41.60 | 96.70% |
| Q-learning | −52.28 | −13.00 | 100% | −48.75 | 100% |
| Expected SARSA | −20.75 | −15.00 | 100% | −20.33 | 100% |

Q-learning learns the short greedy cliff-edge route. Executing epsilon-greedy
actions near that edge produces more falls. SARSA and on-policy Expected SARSA
include exploratory continuation in their targets; safer routes can therefore
have higher expected return under that policy. This is ordinary expected-return
optimization under different continuation policies, not a separate risk penalty.

**Read completion rates together with returns.** Fixed alpha=.5 and persistent
exploration do not guarantee a converged final policy. Five SARSA seeds produced
greedy loops in this run. Evaluation stops at 1000 steps and marks a game
truncated; its partial reward is excluded from completed-return averages. This
does not turn truncation into an environment terminal event. Training has a
separate 100000-step guard that fails explicitly instead of accepting incomplete
episodes. One regression test deliberately freezes a left-at-start looping policy
and checks that evaluation reports incompletion.

Completed-return summaries average each contributing seed's completed-game mean;
the `seeds` column records how many seeds contributed (25 for SARSA greedy
return, 30 for completion-rate statistics). This conditional statistic can hide
bad outcomes if read alone. Frozen exploratory SARSA also has high variability;
freezing removes the online updates that can correct loops during training.
It should not be presented as universally better or worse from one mean alone.

The plotter first averages 20-episode blocks **within each seed**, then computes
across-seed means and standard errors. Adjacent episodes are not treated as
independent observations. Shaded ±1.96 SE bands are approximate descriptive
intervals; cliff losses and only 30 seeds limit a normal approximation.

![Control learning and evaluation](../build/week8_cliff_learning.png)

The route plot selects the first successful deterministic greedy route for each
method and labels its seed. It is an illustration, not a claim that all seeds
learn that route. Display ties choose U,R,D,L order; statistical evaluation uses
uniform greedy tie-breaking. All seeds remain included in completion statistics.

![Illustrative greedy paths](../build/week8_cliff_paths.png)

## Outputs and checks

| Output | Contents |
|---|---|
| `build/trajectories/{cubic,quintic}.csv` | Every 1 ms: desired/actual q,dq,ddq, torque, error, tip paths |
| `build/trajectory_summary.csv` | Tracking errors, actual derivative peaks, torque and saturation |
| `build/trajectory_boundaries.csv` | Explicit one-sided acceleration limits at rest joins |
| `build/cliff_prediction{,_learning}.csv` | TD values/reference/visits and prediction-error checkpoints |
| `build/cliff_training.csv` | Every training episode's seed, return, steps and falls |
| `build/cliff_learning.csv` | Per-episode means/SE across seeds, without plot smoothing |
| `build/cliff_evaluation.csv` | Per-seed frozen-policy completed/truncated counts and conditional means |
| `build/cliff_summary.csv` | Across-seed metrics, SE and contributing seed counts |
| `build/cliff_{paths,policies}.csv` | Selected successful greedy illustrations, with seed and Q values |
| `build/week8_*.{png,svg}` | Five standalone figures for review/export |

Checks cover polynomial boundary conditions, central-difference derivatives,
analytic duration bounds, the notes' worked example, FK against MuJoCo, disabled
actuator forces, tracking/settling errors, all 3501 viewer samples per profile,
cliff/goal/wall transitions, terminal masking, epsilon/tie probabilities,
distinct TD targets, self-loop timing, evaluation truncation and prediction
agreement with a separate Bellman calculation. Seeds are explicit; exact sampled
numbers can vary across C++ standard-library implementations.

## Learn by changing one thing

1. Predict what halving T does to desired speed and acceleration (×2 and ×4),
   then compare that prediction with your plots and torque saturation. Update
   the experiment's duration/end-step constants consistently if changing timing.
2. Explain why the cubic reference has an acceleration jump but the actual
   acceleration need not have the same jump under this controller.
3. Compare the two Cartesian reference curves: same joint geometry, different
   timing. Explain why the straight Cartesian line is not the executed path.
4. In a copy of the cliff transition, make falling terminal. Identify which
   regression case fails and how the learning target incorrectly changes.
5. Work out the Expected SARSA target when two actions tie for maximum; compare
   it with sampling many SARSA next actions from the same fixed row.
6. Reduce alpha or use an appropriate diminishing schedule, rerun all seeds,
   and compare both policy completion and return. Do not discard failed seeds.
7. Compare epsilon=0 evaluation against exploratory evaluation and explain why
   Q-learning's greedy performance does not describe its training-time safety.
