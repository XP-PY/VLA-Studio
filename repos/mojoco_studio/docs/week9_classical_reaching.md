# Week 9: PD Control and the Complete Classical Pipeline

This milestone connects the components from weeks 6–8:

```text
Cartesian position in world coordinates
        → feasible IK branch
        → timed joint trajectory (q_d, dq_d, ddq_d)
        → sampled feedback / model-based effort
        → bounded motor torque
        → MuJoCo motion and sensor observations
```

The source comments refer to your **Modern Robotics notes 6.2, 9.2 and
11.1–11.4**, including error dynamics and the difference between feedforward and
computed torque. The RL part is the planned **Chapters 5–6 review**, connected to
the earlier Blackjack and Cliff Walking implementations; this week does not
train a robot policy.

## Run the milestone

From the repository root:

```bash
./build.sh classical_reaching
./build/classical_reaching
build/week5_plots/bin/python scripts/plot_week9.py

./build.sh classical_reaching_viewer
./build/classical_reaching_viewer
```

The batch runs **15 configurations / 42 simulations**: one run for each clean
case and ten seeds for each of three noisy cases. All results go under ignored
`build/`. The plotter requires NumPy and Matplotlib; use another Python
environment with those packages if the existing week 5 environment is absent.
The batch does not require a display. The viewer uses SDL2/OpenGL like week 7–8.

Run a selected experiment or supply your own target:

```bash
./build/classical_reaching --case baseline --target 0.6 0 1.15
./build/classical_reaching --case filtered_difference --seed 3
./build/classical_reaching --list
./build/classical_reaching_viewer --case rate_50hz
./build/classical_reaching_viewer --case filtered_difference --seed 3
./build/classical_reaching_viewer --target 0.6 0 1.15
```

Targets are **world x,y,z in metres**. Unreachable, out-of-plane, nonfinite and
planning-range-infeasible requests are rejected before creating result files.
Each batch invocation replaces the summary/plan and selected trajectory files;
other trajectory files remain. Rerun the full default batch before plotting the
complete comparison. The plotter checks the summary contains the full set.

### Live playback

The baseline is always on the left; the selected comparison is on the right.
The initial comparison is a target jump. **N/B** cycle next/previous cases and
restart both robots. **Space** pauses, **R** replays the same seeded observations,
**Right arrow** advances 1 ms, and **[ / ]** change speed. Left/right drag rotate
or pan, the wheel zooms, C restores the camera, and Esc closes the window.

Green geometry marks the current desired joint pose; the red sphere is the final
Cartesian goal. The overlay separates tracking error from distance to that final
goal. During a smooth trajectory these are different quantities. Saturation and
actual departure from the planning range are also visible. Simulation speed
does not change the physics timestep or controller period.

```bash
./build/classical_reaching_viewer --check
./build/classical_reaching_viewer --case filtered_difference --seed 3 --check
./build/classical_reaching_viewer --smoke-test
```

`--check` compares every sample against existing batch CSVs without opening a
window. With no case specified it checks all 15 cases at seed 0. The second
command checks the selected noisy seed; generate that matching batch CSV first.
Use the same target in both commands if testing a custom target. The render test
opens a window, renders each comparison at 1.2 s, saves PPM images under
`build/week9_viewer/`, and exits. Normal viewing does not write batch results.

## Read the source in this order

| Source | Purpose |
|---|---|
| `src/reaching_pipeline.h::make_plan` | World/base conversion, feasible IK branch selection, duration |
| `src/cartesian_control.h::geometric_ik` | Reused cosine-law derivation and range checks |
| `src/trajectory_generation.h::scaling`, `joint_line` | Reused quintic polynomial and analytical derivatives |
| `src/reaching_pipeline.h::reference` | Smooth reference versus setpoint-jump convention |
| `src/reaching_pipeline.h::command` | Four effort laws, gain units and error-dynamics derivation |
| `src/reaching_pipeline.h::SensorModel::sample` | Measurement noise, differencing and filtering |
| `src/reaching_pipeline.h::Experiment::step` | Integer controller clock and held torque |
| `src/classical_reaching.cpp` | Checks, measured metrics and batch orchestration |
| `src/classical_reaching_viewer.cpp` | Synchronized comparisons and seeded replay checks |

The existing mathematical helpers and week 8 exercises are reused without
changing their behavior. All new mathematical functions have review comments.

## From the Cartesian request to joint motion

The XML base sits at world (0,0,1), rotated by Rx(pi/2), so the world-to-base
position conversion is

```text
p_B = R_WBᵀ (p_W - p_WB) = [world_x, world_z-1, -world_y].
```

The two links move in base x-y, which is world x-z. Passing world z straight to
a horizontal-arm IK solver without subtracting the base height would be wrong.

The default goal is **(0.55,0,1.25) m**, starting at q=(0.6,−0.8) rad. Both
cosine-law elbow branches are considered and filtered by the week 6 planning
box: shoulder [−1.5,1.5], elbow [−1.7,1.7] rad. Among feasible solutions, choose
the nearest joint vector on the existing unwrapped branch. The chosen endpoint
is approximately **(1.038785,−1.443468) rad**. Only position is requested;
orientation follows from the two joint angles.

The joint line is `q(s)=q_start+s*(q_goal-q_start)`. It stays inside the convex
planning box when both endpoints do. It does not generally trace a Cartesian
straight line. Quintic timing gives zero endpoint velocity and acceleration:

```text
s=10u³−15u⁴+6u⁵, u=t/T
dq=(q_goal−q_start)*ds, ddq=(q_goal−q_start)*dds.
```

Duration is the larger of 1.5 s and the analytical minimum for reference limits
1.5 rad/s and 3 rad/s² per joint, rounded up to a 1 ms tick. There is a 0.5 s
initial hold and a 2 s final hold. The default experiment therefore lasts 4 s.
These kinematic bounds do not guarantee motor-torque feasibility.

The XML retains unlimited, contact-free hinges to isolate controller effects.
The planning box constrains the reference, not the actual dynamics: poor
controllers can leave it, and those departures are counted. This is a learning
experiment, not a collision-avoiding or limit-enforcing real-robot controller.

## Four control laws with different meanings

Let `e=q_d−q_hat` and `ed=dq_d−v_hat`, with sampled measured states denoted by hats.
The main comparison uses gravity-compensated PD. Three additional laws make the
book's distinctions concrete:

| Case | Requested torque before clipping |
|---|---|
| `pd_only` | Kp e + Kd ed |
| `baseline` | Kp e + Kd ed + g(q_hat) |
| `inverse_ff_pd` | Kp e + Kd ed + ID(q_d,dq_d,ddq_d) |
| `computed_torque` | M(q_hat)(ddq_d + Kp e + Kd ed) + c(q_hat,v_hat) + g(q_hat) |

For direct torque PD, Kp has units N m/rad and Kd has units N m s/rad. The default
values are Kp=diag(25,15), Kd=diag(5,3). By contrast, computed-torque gains act on
an **acceleration correction**: Kp has units s⁻² and Kd s⁻¹. This case uses 64 I
and 16 I, respectively, corresponding to ideal critically damped modes with
natural frequency 8 rad/s. Copying numerical torque gains into that acceleration
law would not preserve the same controller.

`ID(q_d,dq_d,ddq_d)` uses the **desired** state to predict the required motion
effort. It is feedforward; the PD addition corrects measured error. Computed
torque instead evaluates M,c,g at the **current measured** state. With an exact
model and ideal continuous, unsaturated commands,

```text
M qdd + c + g = M (qdd_d + Kp e + Kd ed) + c + g
         qdd = qdd_d + Kp e + Kd ed
         edd + Kd ed + Kp e = 0.
```

The code checks this instantaneous identity independently using analytical
forward dynamics. Sample-and-hold control, saturation and noisy state estimates
break the exact ideal cancellation. No model mismatch is added in this week's
feedforward comparison; very small errors should not be interpreted as a
robustness guarantee.

For plain PD at rest, a nonzero position error must generate the holding torque:
`Kp*e_ss=g(q_ss)`. Gravity compensation removes that requirement under the exact
model. It does not fully cancel inertia and Coriolis coupling. The scalar analogy
`omega_n=sqrt(Kp/M)`, `zeta=Kd/(2 sqrt(M Kp))` explains stiffness and damping,
but those are not independent global joint modes of this coupled arm. The notes'
energy argument for gravity PD gives regulation without asserting exact linear
error dynamics.

![Pipeline, control laws and paths](../build/week9_pipeline.png)

## Experiment map and measured results

All clean cases change the named setting relative to the baseline. The noisy
cases share 250 Hz control, so use `rate_250hz` as their clean reference.

| Case | Change |
|---|---|
| `small_kp`, `large_kp` | Kp=(3,2) or (100,60), Kd unchanged |
| `small_kd`, `large_kd` | Kd=(0.15,0.1) or (12,8), Kp unchanged |
| `target_jump` | Switch directly to q_goal at 0.5 s |
| `torque_limit_2` | Reduce each motor limit from ±12 to ±2 N m |
| `rate_250hz`, `rate_50hz` | Update every 4 or 20 physics ticks instead of every tick |
| `noisy_velocity` | Independent noisy position and velocity samples at 250 Hz |
| `noisy_difference` | Estimate velocity from noisy positions at 250 Hz |
| `filtered_difference` | Filter that differenced velocity with a 12 Hz low-pass |

The jump case specifies zero desired velocity on both sides. A literal position
step has derivative impulses; those are deliberately not sent to the motor.
This tests a changed setpoint rather than a physical instantaneous trajectory.

Selected default-target results:

| Case | Whole-run RMS joint error (rad) | Final tip error (m) | Saturated interval fraction |
|---|---:|---:|---:|
| Baseline | 0.002560 | 8.79e-9 | 0% |
| Small Kp | 0.005502 | 0.000940 | 0% |
| Large Kp | 0.000749 | 1.40e-15 | 0% |
| Small Kd | 0.004435 | 0.001730 | 0% |
| Large Kd | 0.001675 | 0.0000239 | 0% |
| Jump | 0.125470 | 3.39e-9 | 0.075% |
| 2 N m limit | 1.617557 | 0.096958 | 94.125% |
| 250 Hz | 0.002566 | 8.72e-9 | 0% |
| 50 Hz | 0.341238 | 0.119397 | 68% |
| Plain PD | 0.140616 | 0.083737 | 0% |
| Inverse feedforward + PD | 0.0000160 | 3.34e-11 | 0% |
| Computed torque | 0.0000687 | 1.97e-11 | 0% |

Tiny final errors are properties of this noiseless, exact-model simulation after
its final hold; they are not realistic sensor resolution. The jump has a large
initial reference error yet eventually reaches the goal, so final error alone
would hide its transient. In this smooth, noiseless test, larger gains improve
some metrics; the result is not a general recommendation to keep increasing them.

![Gain comparisons](../build/week9_gains.png)

### Torque limits and control frequency

The 2 N m shoulder limit is already insufficient for the initial pose's gravity
load. That arm can fall **before** the target moves. Requested torque is clipped
explicitly and the XML motor ctrlrange is set to the same bound; logs contain
both requested and actual torque. The poor outcome is measured, not hidden by
changing gains or forcing the simulated pose.

Physics remains RK4 at **1 ms in every case**. At each controller tick, sample
sensors, evaluate the reference, compute torque, and hold that torque until the
next tick. There is no feedback callback recomputing it at RK4 substeps. This is
an intentional difference from week 8's continuous stage-wise feedback and is
necessary to give “50 Hz control” a concrete sampled-data meaning.

The integer clock avoids drift. Rendering refreshes neither generate noise nor
update the command. Thus a slow viewer cannot silently change the controller
period. The unchanged gains oscillate badly at 50 Hz in this experiment; the
ideal continuous-time PD arguments do not certify the sampled loop.

![Limits and rates](../build/week9_limits_and_rates.png)

### Noise and velocity estimation

Position noise has sigma_q=0.003 rad; the direct velocity-sensor case has
sigma_v=0.05 rad/s. These are independent per-sample Gaussian errors, not
continuous-time noise densities. In the differencing cases the direct velocity
channel is unused:

```text
v_raw[k] = (q_hat[k] - q_hat[k-1]) / Tc
Var(velocity noise) = 2 sigma_q² / Tc²
```

At Tc=0.004 s, position differencing alone contributes velocity-noise standard
deviation about **1.061 rad/s**, far larger than the direct velocity channel.
Derivative feedback multiplies it by Kd and sends it toward the motor.

The filtered case uses

```text
beta = exp(-2*pi*12*Tc)
v_hat[k] = beta*v_hat[k-1] + (1-beta)*v_raw[k].
```

Filtering reduces noise while introducing lag. Differenced velocity starts at
zero, matching the initial-rest assumption. Equal seed and period preserve the
same position-noise sequence across variants; each run resets both RNG and
filter state. Ten seeds give these means ± **sample standard deviations**:

| Measurement mode | RMS true joint error (rad) | Torque variation (N m) | Mean saturation |
|---|---:|---:|---:|
| Direct noisy velocity | 0.002825 ± 0.000239 | 477.7 ± 11.3 | 0% |
| Position differences | 0.008377 ± 0.001432 | 10087.5 ± 198.9 | 7.0% |
| Filtered differences | 0.004249 ± 0.000315 | 1794.3 ± 36.7 | 0% |

Torque variation is `sum_k ||tau[k]-tau[k-1]||_2`, not mechanical energy or RMS
torque. It measures command changes over the same horizon. Noise statistics use
independent seeded runs, not thousands of correlated time samples as fake
replications. The plot's error bars show SD, not standard error or confidence
intervals. All tracking metrics use true simulator state; otherwise noisy
measurements could confuse sensor error with actual motion error.

![Sensor noise comparisons](../build/week9_sensor_noise.png)

## Metrics, outputs and verification

`build/classical_runs/<case>_seed<N>.csv` stores every 1 ms accepted state:
reference q,dq,ddq; true q,dq; latest measured q,dq; raw/applied torque; joint
error; actual/reference world tip; saturation; and planning-range status.
`control_time` identifies when the held measurement/command was last updated.
At controller boundaries, the logged applied torque is the new command for the
following interval; at the final endpoint no extra update occurs.

`build/classical_reaching_plan.csv` records the Cartesian goal, selected joint
branch and timing. `build/classical_reaching_summary.csv` stores per-run errors,
overshoot, saturation, torque variation, range violations and settling. The
plotter creates four PNG/SVG figures and `build/week9_noise_summary.csv`.

Settling means that, after reference arrival, **all remaining observed samples**
stay within 0.01 rad joint error and 0.03 rad/s speed, with at least 0.2 s of
observed dwell. NaN means that was not demonstrated. This is a finite-horizon
project metric, not the notes' asymptotic 2% definition. A noisy run's final position
can be accurate while its actual motion does not meet this speed band.

Checks cover IK/FK frame consistency and rejected targets, reference endpoints,
computed-torque error dynamics, clipping, empirical derivative-noise variance,
seeded resets, zero-order-hold timing, inactive actuator forces, actual applied
torque, and the default baseline's successful bounded reach. Poor-controller
cases are not required to succeed: their failures are the comparison results.
Viewer checks compare every trajectory sample, including noisy observations and
commands, even with extra render-style refreshes. All 15 cases also have an
automatic rendering smoke test.

Strict warning builds and the full batch under UndefinedBehaviorSanitizer pass.
An AddressSanitizer build was attempted, but this installation's MuJoCo 3.12.0
`mjsan.h` uses an attribute placement rejected by GCC when ASan is enabled; the
bundled SDK was not modified. Thus ASan coverage is not claimed.

## RL review and the policy-to-controller connection

Revisit `blackjack.h` and `cliff_learning.h` alongside RL notes 5.1, 5.5–5.6,
6.1–6.2 and 6.4–6.6. No additional RL algorithm is required by the week 9 plan.

| Review topic | What to explain using the existing implementations |
|---|---|
| Bootstrapping | MC averages complete sampled returns; TD(0) targets r+gamma V(s'). Computing returns backward does not itself bootstrap. |
| Bias/variance | A sampled MC return avoids successor-value approximation but may vary greatly. TD's target inherits current value error and uses a shorter sampled horizon; neither has universally smaller MSE. |
| On/off-policy | SARSA targets the sampled behavior action. Q-learning targets a greedy maximum while behavior can explore. Expected SARSA is on-policy here because its expectation uses the behavior probabilities. |
| Exploration during control | A good greedy Cliff Walking route can have poor exploratory return. Training performance, frozen-policy return and completion rate are distinct measurements. |

For example, with reward 2, gamma=.9, next values (4,1), probabilities (.9,.1),
and sampled next action of value 1: SARSA targets 2.9, Q-learning 5.6, Expected
SARSA 5.33. The expectation removes next-action sampling variance for that fixed
table, not uncertainty from every possible source. Ordinary versus weighted MC
importance sampling illustrates another finite-sample bias/variance tradeoff.
Sensor filtering in this robot exercise is a separate measurement-estimation
issue; it should not be confused with RL bootstrapping bias.

A future VLA/RL policy could supply the `world_target` input at a slower task
rate. The existing pipeline then performs feasibility checks, chooses a joint
trajectory and tracks it at a faster servo rate. Invalid policy outputs need
handling; a learned target does not make IK, timing or actuator bounds optional.
This implementation receives one target at episode start. It does not yet
implement online replanning: replacing a target mid-motion would require a new
trajectory consistent with the current position/velocity, rather than assuming
every new segment starts from rest.

## Review exercises

1. Derive the world-to-base conversion before trying a new target. Predict which
   elbow branch is nearest to the starting joints.
2. Compare `pd_only` and baseline at rest. Explain which term supplies gravity
   torque and why plain PD needs nonzero error.
3. Substitute each of the four laws into `M qdd+c+g=tau`. Identify which law
   really produces linear error dynamics under the ideal assumptions.
4. Compare small and large Kd at 1 kHz, then reduce the controller rate. Explain
   why continuous-time gain reasoning is insufficient for the held-input loop.
5. Derive the factor `2 sigma_q²/Tc²` for independent position samples. Explain
   why increasing sensor frequency alone can worsen a raw derivative estimate.
6. In the viewer, compare the red final-goal distance with tracking error before
   trajectory completion. Explain why a small tracking error need not mean the
   final goal has already been reached.
7. Describe where a future policy's exploration would enter this pipeline and
   why stable low-level PD would not guarantee safe high-level target choices.
