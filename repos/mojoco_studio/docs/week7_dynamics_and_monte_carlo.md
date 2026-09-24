# Week 7: Robot Dynamics and Monte Carlo Learning

This completes the implementation part of week 7: a two-link dynamics experiment
and Blackjack Monte Carlo prediction/control. The mathematical comments refer
to your **Modern Robotics notes, sections 8.1–8.5**, and your updated
**Reinforcement Learning notes, sections 5.1–5.6**. Section numbers in the source
refer to those notes. The advanced Chapter 5 extensions (off-policy control,
discount-aware and per-decision importance sampling) are not required here.

## Run everything

Run from this repository's root:

```bash
./build.sh robot_dynamics
./build/robot_dynamics
./build.sh blackjack_mc
./build/blackjack_mc
build/week5_plots/bin/python scripts/plot_week7.py
```

The C++ programs use the existing C++17/MuJoCo build setup and need no display.
Blackjack itself uses only the C++ standard library; it can also be compiled with
`g++ -std=c++17 -O2 src/blackjack_mc.cpp -o build/blackjack_mc` after creating
`build/`. On a fresh checkout, run the plotting script with a Python environment
containing NumPy and Matplotlib. The existing week 5 environment already has both.

Each executable runs checks, prints results, and overwrites its own CSV outputs
under the ignored `build/` directory. Fixed seeds make runs repeatable with the
same compiler/library; C++ random-distribution implementations can differ across
platforms. The default batch needs no arguments.

| Output | What to inspect |
|---|---|
| `build/dynamics_validation.csv` | Maximum analytical/MuJoCo and finite-difference errors |
| `build/dynamics_summary.csv` | Energy, final joint error and motion for all eight runs |
| `build/dynamics/*.csv` | Time, joint angles/rates, actual actuator torque, energy and error |
| `build/blackjack_values.csv` | All 200 states: predicted values, reference values, policies, Q and visit counts |
| `build/blackjack_learning.csv` | Prediction error and greedy-policy performance during learning |
| `build/blackjack_evaluation.csv` | Frozen-policy evaluation on fresh natural deals |
| `build/blackjack_importance.csv` | Ordinary/weighted IS error across 50 independent runs |
| `build/week7_dynamics.{png,svg}` | Controller and energy comparisons |
| `build/week7_blackjack_tables.{png,svg}` | Value tables and learned policies, with/without usable ace |
| `build/week7_mc_learning.{png,svg}` | Learning curves and importance-sampling MSE |

![Dynamics comparisons](../build/week7_dynamics.png)

## Suggested source reading order

1. `assets/dynamics_arm.xml`: match joint axes, masses, COMs and inertias to the formulas.
2. `src/two_link_dynamics.h`: read `mass`, `coriolis`, `potential`, `gravity_torque`,
   `solve`, `inverse`, `forward`, and `energy`. Their comments derive the calculations.
3. `src/dynamics_experiments.h`: inspect the shared `control` and `reset` functions,
   then `validate` and `run` in `src/robot_dynamics.cpp`.
4. `src/blackjack.h`: trace `add_card` and `generate`, then `returns`, `predict`,
   `improve`, and `importance_ratio`/`ImportanceEstimate::add`.
5. `src/blackjack_mc.cpp`: inspect `self_checks`, training loops and independent evaluation.
6. Read `Oracle` last: it is a separate model-based check, not part of MC learning.

## Watch all eight experiments live in MuJoCo

```bash
./build.sh robot_dynamics_viewer
./build/robot_dynamics_viewer
```

This opens a real MuJoCo-rendered, live simulation with **eight synchronized
panels**. It starts at half speed and stops after six simulation seconds, leaving
the final states visible for comparison. Press R to replay. CSVs are not needed
for normal viewing, and viewing does not overwrite batch results.

| Key or mouse action | Effect |
|---|---|
| 0 | Show all eight modes side by side |
| 1–8 | Focus on the corresponding numbered mode; all modes remain synchronized |
| Space | Pause or resume |
| R | Reset all initial conditions and replay from time zero |
| Right arrow | Pause and advance all experiments by 10 ms |
| [ / ] | Halve / double speed, between 0.125× and 2× |
| Left drag | Rotate the shared camera |
| Right drag | Pan the shared camera |
| Mouse wheel | Zoom |
| C | Restore the default front camera |
| Esc | Close the viewer |

You can start focused on one mode using its number or batch name:

```bash
./build/robot_dynamics_viewer --mode torque_pd_gravity
./build/robot_dynamics_viewer --mode 8
./build/robot_dynamics_viewer --help
```

The blue link connects the shoulder to the elbow; the orange link ends at the
tip. Light-colored spheres mark the joints. A thin **green target pose** appears
only in position-actuator, torque-PD and torque-PD-plus-gravity modes. It is a
rendering annotation with no mass or forces. The overlays show the control law,
time, mechanical energy E, change from the initial energy dE, unwrapped angles,
and speed `sqrt(dq1²+dq2²)`. Servo modes also show `||q_des−q||` in radians.

Try these comparisons:

1. Replay with **0** and compare modes **1 and 2**: free motion keeps its energy,
   while damping reduces it. The joints are unlimited, so free motion can rotate
   through multiple turns.
2. Compare **4 and 8**: the same gravity compensation holds the initially
   stationary arm but does not stop an initially moving arm.
3. Compare **5 and 6**: built-in position actuation and the matched motor PD law
   move together and settle away from the green target.
4. Watch **7**: adding gravity compensation closes that steady offset.
5. Focus on **3** and slow down with **[**: constant torque can accelerate the
   unbounded joints. At high speeds, sparse display frames can make rotation
   appear to reverse; the simulator still integrates at 0.001 s.

The viewer and batch program share `dynamics_experiments.h`, including all of
your controller comments, gains, actuator masks and reset states. Each mode has
its own MuJoCo model/data, because damping and actuator masks are model settings.
The single-threaded viewer selects the correct callback mode before each step,
including all RK4 stages. Changing playback speed changes only the scheduling
of fixed integration steps, not the timestep or controller equations.

`./simulate.sh dynamics_arm` opens the XML with MuJoCo's generic viewer, but that
program does not install our custom gravity/PD callbacks or mode resets. Use
`robot_dynamics_viewer` for these eight experiments.

The window uses SDL2 with MuJoCo's renderer. The bundled MuJoCo installation
provides SDL2 headers; `build.sh` uses `pkg-config sdl2` when available, or the
Linux SDL2 runtime library already installed on this machine. Normal viewing
requires a graphical desktop with OpenGL. The original batch executable still
has no window-library dependency.

Verification commands:

```bash
./build/robot_dynamics                          # regenerate batch references
./build/robot_dynamics_viewer --check           # no display needed
./build/robot_dynamics_viewer --smoke-test      # requires a display; exits automatically
```

`--check` compares all 601 logged samples in each of the eight trajectories,
including angles, velocities, torques and energies, then checks replay reset.
The measured maximum absolute discrepancy was `5.69e-14` across the checked
columns (each column retains its own units). `--smoke-test` advances to two
seconds, renders the grid and each focused view, and writes PPM screenshots to
`build/dynamics_viewer/` before exiting.

## Dynamics: derive before simulating

The arm has the previous exercises' link lengths, 0.45 m and 0.35 m, but now
moves in a **vertical plane** so gravity affects the joints. The base frame's
x-axis is world x, y-axis is world z, and z-axis is world −y. Both hinges rotate
about base +z. The shoulder sits 1 m above the world origin. Angles are radians;
q₂ is relative to link 1, so link 2's absolute angle is q₁+q₂.

This XML uses explicit distributed rigid-body inertias rather than inferring
them from visualization geometry. For rectangular links of thickness 0.03 m:

| Parameter | Link 1 | Link 2 |
|---|---:|---:|
| Mass m (kg) | 0.7 | 0.5 |
| Length L (m) | 0.45 | 0.35 |
| COM distance r=L/2 (m) | 0.225 | 0.175 |
| COM inertia about hinge direction I=m(L²+0.03²)/12 (kg m²) | 0.011865 | 0.0051416667 |

All eight modes use this same model. Joint limits, contacts, friction and armature
are absent to isolate the smooth dynamics. Only the damping experiment changes
the two passive damping coefficients. This is an intentionally simplified model;
the week 6 collision/limit considerations have not become unnecessary.

The COM positions, relative to the shoulder, are

```text
p1 = r1 [cos q1, sin q1]
p2 = L1 [cos q1, sin q1] + r2 [cos(q1+q2), sin(q1+q2)]
```

Differentiate these positions for COM velocities, then expand

```text
K = 1/2 m1 |v1|² + 1/2 I1 dq1²
  + 1/2 m2 |v2|² + 1/2 I2 (dq1+dq2)²
  = 1/2 dqᵀ M(q) dq.
```

Collecting the squared and cross terms yields

```text
M11 = I1 + I2 + m1 r1² + m2 (L1² + r2² + 2 L1 r2 cos q2)
M12 = M21 = I2 + m2 (r2² + L1 r2 cos q2)
M22 = I2 + m2 r2²
```

Unlike a point-mass-at-the-tip example, these expressions include **COM offsets
and rotational kinetic energy**. Substituting the wrong inertia model can give
plausible-looking motion while disagreeing with MuJoCo.

Euler–Lagrange then gives `M qdd + c + g = tau`, with

```text
h = m2 L1 r2 sin q2
c = [-h (2 dq1 dq2 + dq2²), h dq1²]
P = (m1 r1 + m2 L1) g0 sin q1 + m2 r2 g0 sin(q1+q2)
g = ∂P/∂q
```

Here `g0=9.81 m/s²` is gravitational acceleration; `g(q)` is a **joint torque
vector**. The physical gravitational torque is −g(q). Hence commanding **+g(q)**
balances it. `c` means only velocity-dependent terms in this document; MuJoCo's
`qfrc_bias` contains **c+g**. See MuJoCo's
[equations of motion and force conventions](https://mujoco.readthedocs.io/en/stable/computation/index.html).

With passive damping `−D dq`, the actuator torque required by inverse dynamics is

```text
tau = M qdd + c + g + D dq.
```

Forward dynamics instead solves `M qdd = tau − c − g − D dq` with Cholesky
factorization. Positive link masses and inertias make M symmetric positive
definite. The source shows the two triangular solves rather than explicitly
forming M⁻¹. Forward dynamics computes instantaneous acceleration; RK4 is the
separate numerical integration step that advances position and velocity.

### What is checked independently?

Across 200 seeded states, with and without passive damping, the executable compares:

- M against the dense matrix returned by `mj_fullM`.
- c+g against `qfrc_bias`.
- Analytical forward acceleration against `mj_forward` with applied torque.
- Analytical inverse torque against `mj_inverse` for a supplied acceleration.
- g against a central finite difference of potential energy.
- `dqᵀ c = 1/2 dqᵀ Mdot dq` against a finite difference of M along dq.

The last identity explains energy balance:
`d(K+P)/dt = dqᵀ tau − dqᵀ D dq`. The potential-energy zero is the shoulder's
height; negative energies are valid. An arbitrary constant in P cannot affect
g or the motion. The first run found maximum mass error `2.22e-16 kg m²`, bias
error `1.78e-15 N m`, and acceleration error `3.41e-13 rad/s²`; finite-difference
errors were below `1e-9` in their respective units. These are checks for this
specific model, not universal accuracy guarantees.

### Short controller comparison

All standard runs start at q=(0.6,−0.8), dq=(0,0) and last six seconds with
dt=0.001 s. Servo targets are q_des=(−0.3,0.6). Torque bounds are ±12 N m.

| Mode | Command or passive force | Observed behavior |
|---|---|---|
| No control | tau=0 | Falls and swings; energy stays constant up to integration error |
| Joint damping | tau=0, D=diag(0.2,0.1) | Dissipates energy and tends toward the hanging configuration |
| Constant torque | tau=(1,−0.4) N m | Drives ongoing motion; unlimited hinges can make many revolutions |
| Gravity compensation | tau=g(q) | Holds the initial stationary pose; does not move toward a target |
| Position actuator | Built-in PD servo | Reaches a gravity-loaded equilibrium with nonzero target error |
| Torque PD | tau=clip(Kp(q_des−q)−Kd dq) | Matches the position actuator when gains, limits and update timing match |
| Torque PD + gravity | tau=clip(Kp(q_des−q)−Kd dq+g(q)) | Removes the gravity-induced equilibrium offset in this experiment |
| Gravity compensation, moving start | tau=g(q), initial dq=(0.3,−0.2) | Keeps moving: gravity compensation supplies no velocity regulation |

Both PD forms use Kp=diag(25,15), Kd=diag(5,3). In the measured run, ordinary PD
and position actuation both ended with a **0.1776 rad** joint-error norm. Their
logged angles agreed within `1.12e-16 rad`. PD+gravity ended below `1e-3 rad`
(numerically about `1e-13` for this deterministic, undisturbed case). Damping
removed 6.485 J over the run; the uncontrolled final energy drift was about
`−2.89e-8 J`. Check your CSVs for your platform's values.

Two implementation details are essential for this comparison:

1. A position actuator with zero ctrl still pulls toward zero. We disable its
   entire actuator group during motor experiments, and disable the motors during
   the position experiment. The program checks disabled actuator forces are zero.
2. RK4 evaluates dynamics at intermediate states. Motor feedback uses
   `mjcb_control` so it is recalculated at these stages, matching the built-in
   servo's state dependence. Computing it only once per outer step would change
   the comparison. The callback is restored when the experiment exits.

Gravity compensation with a moving start can change total **K+P**: its actuator
does work `dqᵀg`. In the ideal compensated system it is **K** that stays constant
because `M qdd+c=0`. Neither observation means the arm is tracking a target.

![Blackjack tables](../build/week7_blackjack_tables.png)

## Blackjack: separate the environment, policy and estimator

The state is `(player sum 12..21, dealer upcard 1..10, usable ace)`, giving 200
decision states. Actions are **Stick=0, Hit=1**. Policy arrays store the probability
of Hit, whereas CSV action columns contain deterministic action indices.

Cards are sampled with replacement: ace through nine each have probability 1/13;
ten has probability 4/13. The dealer stands on soft 17. The player draws
automatically below 12. Intermediate rewards are zero, the terminal reward is
−1, 0 or +1, and gamma=1. There is no splitting, doubling, finite deck or extra
natural payout.

Your updated notes specify that a player's initial natural 21 wins immediately,
except against a dealer natural. `generate` resolves that event **before a
decision state**, so it contributes to full-game evaluation but has no state
visit to update. Soft 21 reached later is an ordinary decision state. A nonnatural
player 21 ties a dealer 21; there is no dealer peek rule. The exact reference and
sampled environment use the same conventions.

Exploring starts construct an **already active** state and first action
uniformly over 200×2 pairs. The dealer's hidden card is still sampled independently.
A synthetic starting soft 21 is therefore not treated as an initial natural.
This artificial reset capability is useful for the algorithm exercise; do not
assume a physical robot or a real card game permits arbitrary resets.

### Prediction and chronological first visits

Freeze the policy to hit below 20 and stick at 20/21. After a complete episode,

```text
G_T = 0
G_t = R_(t+1) + gamma G_(t+1)             # backward pass
N(s) <- N(s) + 1
V(s) <- V(s) + (G_t − V(s)) / N(s)       # eligible visits in forward order
```

The update reads sampled rewards, not estimated successor values. It therefore
does not bootstrap, even though the return is computed recursively.

First-visit MC selects the earliest occurrence of s in **forward time**.
The source tests your notes' A→B→A episode with rewards 1,2,3: returns are 6,5,3;
first-visit V(A)=6, every-visit V(A)=4.5. A backward `seen` set would incorrectly
select 3 as the first return. First-visit **action values** instead use earliest
occurrences of the pair (s,a); a second action at the same state is a different pair.

In this Blackjack state representation, complete decision states cannot recur:
the sum increases while ace status is unchanged, or a usable ace is lost and
cannot become usable again at sums ≥12. Thus the first/every implementations
produce identical results on the same games. That agreement alone would not
prove a generic first-visit implementation correct; the synthetic example does.

One million games produced equal-state RMSE **0.01981** against the fixed-policy
reference, with a minimum of 423 visits over the 200 states. Rare soft states
remain noisier. This is a single training seed, not a multi-seed confidence bound.

### Control: exploration is part of the experiment

Both algorithms estimate Q(s,a) with first-visit sample means after each episode:

```text
Q(s,a) <- Q(s,a) + (G_t − Q(s,a)) / N(s,a)
a* = argmax_a Q(s,a)                     # ties choose Stick
```

**Exploring starts** runs one million games. Each random initial state-action
pair supplies coverage; subsequent decisions use the current deterministic
greedy policy. Policy improvement takes place only after the complete game.

**On-policy epsilon-soft control** runs two million natural deals, starting
from a uniform policy and improving visited states to

```text
pi(a|s) = epsilon/2 + (1−epsilon) [a=a*], epsilon=0.1.
```

The greedy action gets probability **0.95**, the other gets **0.05**. Fixed epsilon
retains exploration and corresponds to improvement within epsilon-soft policies.
The executable evaluates both the learned stochastic behavior and its extracted
deterministic greedy policy; these are different policies. Sample-mean control
also mixes returns from policies that changed during learning, so stationary
prediction arguments do not by themselves prove finite-run control convergence.

In the recorded run, the minimum state-action visit count was 2354 for ES and 39
for epsilon-soft control. Their extracted greedy policies had uniform-state
value gaps to the optimal reference of **0.0000958** and **0.00154** respectively.
Rare states and nearly tied actions can still differ from the reference policy.

Frozen policies were then evaluated on 200,000 fresh games each, with independent
evaluation streams:

| Policy | Mean game return | Approximate 95% sampling half-width |
|---|---:|---:|
| Fixed stick-on-20 | −0.35176 | 0.00397 |
| MC exploring starts, greedy | −0.04448 | 0.00418 |
| MC epsilon-soft, greedy deployment | −0.04506 | 0.00418 |
| MC epsilon-soft behavior | −0.07328 | 0.00418 |
| Independent optimal reference | −0.03923 | 0.00417 |

The intervals describe held-out game noise, not uncertainty across training
seeds. Small apparent differences between good policies may be sampling noise.
Negative reward can persist even under the optimal policy; learning does not
remove the game's inherent disadvantage.

### Why include a Bellman reference in an MC exercise?

`Oracle` independently enumerates dealer outcomes and performs synchronous
Bellman backups. For a fixed policy,

```text
Q(s,Stick) = E[terminal reward | player sum, dealer upcard]
Q(s,Hit)   = sum_card p(card) * (−1 if bust, else V(next_state))
V_new(s)   = (1−pi(Hit|s))*Q(s,Stick) + pi(Hit|s)*Q(s,Hit)
```

Replacing the final expectation by `max(Q(s,Stick),Q(s,Hit))` gives the optimal
reference. gamma=1 does not supply a discounted contraction guarantee; this
particular game's decision graph is acyclic and all episodes terminate.
Each sweep reads the previous value vector and stops at residual <1e-13.

This oracle supplies error measurements and benchmark policies **only**. Neither
MC predictor nor MC controller receives its values or transition probabilities.
This makes the Chapter 4 versus Chapter 5 distinction visible in one program:
enumerate an expectation for validation, learn from sampled returns in training.

### Off-policy prediction with importance sampling

Start every game at **soft 13 against dealer 2**, generate actions with uniform
behavior b, and evaluate the fixed stick-on-20 target pi. The reference value is
approximately **−0.27720419**. For each completed trajectory,

```text
rho = product_t pi(A_t|S_t) / b(A_t|S_t)
ordinary V_n = sum_i rho_i G_i / n
weighted V_n = sum_i rho_i G_i / sum_i rho_i
```

Because this estimates a **state** value, the ratio includes the first action.
All action probabilities are recorded at sampling time. Here rho is zero if any
action disagrees with the deterministic target, and otherwise 2^T. The fixed
starting state requires no initial-state distribution correction.

Ordinary IS counts every episode, including those with rho=0. Weighted IS uses
`C += rho; V += rho/C * (G−V)` for positive weights and leaves the initial zero
unchanged until C>0. Ordinary IS is unbiased under the usual coverage and
integrability assumptions; weighted IS is generally biased at finite sample
sizes but often has lower variance. No assertion requires it to win at every
checkpoint. The code tests these distinct denominators explicitly.

Across 50 independent runs of 10,000 games, observed MSE was **0.0008025** for
ordinary and **0.0006439** for weighted IS. These are finite-run measurements;
the plot can be nonmonotonic. This is trajectory IS prediction, not off-policy
control and not the advanced per-decision variant.

![Learning and importance sampling](../build/week7_mc_learning.png)

## Review through small changes

1. In `dynamics_experiments.h::control`, temporarily compensate with `−g(q)`. Predict the acceleration
   direction before running. The stationary compensation check should fail.
2. Remove the rotational inertias from the analytical M only. Which checks fail,
   and why can the gravity check still pass?
3. Increase damping. Explain its effect using `dE/dt=−dqᵀD dq`, then inspect
   the energy curve rather than judging only the animation or final position.
4. Change the servo gains in **both** C++ and XML. Explain why ordinary PD needs
   nonzero equilibrium error to supply gravity torque.
5. Change epsilon and inspect both behavior reward and greedy deployment reward.
   Look at rare state-action visit counts before interpreting a policy heatmap.
6. In the revisiting synthetic episode, predict what a backward `seen` set would
   do. Keep the correct test to catch this deliberate bug.
7. Compare IS curves from more repetitions. Explain why one noisy learning curve
   is insufficient evidence about an estimator's variance.

The default executables fail with a nonzero exit code when their mathematical,
coverage, reference-error, energy or matched-controller checks fail. Generated
figures supplement these checks; they do not replace them.
