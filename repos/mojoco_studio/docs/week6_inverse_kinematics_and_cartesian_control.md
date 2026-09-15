# Week 6: Inverse Kinematics and Cartesian Control

The week 6 implementation builds on the verified week 5 Jacobian. Your reading
is already covered in **Modern Robotics, Chapter 6** and **Reinforcement Learning,
Chapters 3–4**. This exercise turns those notes into a numerical IK solver, a
Cartesian reaching controller, and measured comparisons.

The task is **tip position**, not independently specified position and
orientation. This planar 2R arm has two controllable task coordinates; its tip
orientation follows from the selected joint configuration.

## Run the complete experiment

From the project directory:

```bash
./build.sh inverse_kinematics
./build/inverse_kinematics

./build.sh cartesian_reaching
./build/cartesian_reaching

cat build/inverse_kinematics_report.txt
cat build/cartesian_reaching_report.txt
```

Both executables use the existing C++17 build workflow and run without a
display. The controller's default batch compares 11 scenarios, two inverse
methods, three gains, and two execution modes: 132 combinations.

Generate plots and an animation using the Python environment installed during
week 5:

```bash
build/week5_plots/bin/python scripts/plot_week6.py --animate
```

On a fresh checkout, use a Python environment with NumPy, Matplotlib and Pillow
(`python -m pip install numpy matplotlib pillow`), then run
`python scripts/plot_week6.py --animate`. Omit `--animate` for plots only.
The script first checks the C++ inverse results against NumPy's SVD/linear solver.
It creates `build/week6_overview.{svg,png}`, `build/week6_paths.{svg,png}` and,
optionally, `build/week6_reaching.gif`. The GIF reconstructs the arm from **logged
MuJoCo joint states**; it does not run a second simulation.

Run one experiment at a time:

```bash
./build/cartesian_reaching --case near_singular_start --mode dynamics --method pinv --gain 4
./build/cartesian_reaching --case near_singular_start --mode dynamics --method dls --gain 4
./build/cartesian_reaching --help
```

Each invocation replaces the summary/report and rewrites the selected
trajectories. Other trajectory files remain. Rerun the unfiltered default batch
before plotting the full comparison; the plotter selects runs from the summary.

## Read the implementation in this order

The mathematical functions now have source-level review comments in the style
of `two_link_kinematics.h`: role, equations, derivation, frames/units, parameters
and limitations. Start with the comments above `inverse_velocity`,
`geometric_ik`, `numerical_ik`, `limit_velocity` and `reach`, then trace their
code. Chapter 6 section references in those comments refer to your book notes;
damping, backtracking and limit guards are identified as implementation choices.

| File or function | What it teaches |
|---|---|
| `src/two_link_kinematics.h` | Week 5 FK, point Jacobian, frame utilities |
| `cartesian_control.h::geometric_ik` | Independent reachability and elbow-branch calculation |
| `cartesian_control.h::inverse_velocity` | Standard pseudoinverse and damped least squares |
| `cartesian_control.h::numerical_ik` | Bounded, backtracked local position corrections |
| `src/inverse_kinematics.cpp` | Numerical IK examples and checks against geometry |
| `cartesian_control.h::limit_velocity` | Speed scaling and slowing near joint limits |
| `cartesian_reaching.cpp::Robot` | Model loading, state, frames, MuJoCo Jacobian |
| `cartesian_reaching.cpp::reach` | Feedback loop, ideal integration, actual dynamics, logging |
| `assets/cartesian_reaching.xml` | Same arm kinematics with velocity actuators |
| `scripts/plot_week6.py` | Independent inverse check and visualization of logged results |

The C++ programs save:

| Output | Contents |
|---|---|
| `build/inverse_kinematics.csv` | Targets, seeds, feasible-branch counts, status, final joint configurations and errors |
| `build/inverse_velocity_audit.csv` | Jacobians, desired velocities, inverse results and singular values for NumPy comparison |
| `build/cartesian_reaching_summary.csv` | Status, settling time, final error, raw/commanded/measured peak rates, torque and limit metrics |
| `build/reaching/<case>_<mode>_<method>_k<gain>.csv` | One row per controller tick with measured state, error, singular values and commands |

## 1. Numerical IK is a sequence of configuration corrections

Your notes' local linearization is

$$
e^k=x_d-f(q^k),\qquad J(q^k)\Delta q^k\approx e^k,
\qquad q^{k+1}=q^k+J(q^k)^\dagger e^k.
$$

`numerical_ik` recomputes the position error and Jacobian at every iteration.
It limits the largest joint correction to 0.2 rad, then backtracks the step
until it reduces Cartesian error. Candidate positions are projected into the
physical joint limits. The maximum is 300 iterations; success requires a
position error below $10^{-6}$ m. A tiny correction with a large residual is
reported as `stalled`, not as success.

This is position-only Newton/pseudoinverse IK with practical step control.
It does not use the full-pose $\log(T^{-1}T_d)$ error from your §6.6 notes;
that is needed for a full-pose task with a compatible manipulator Jacobian.

The test program solves 100 deterministic random targets **from nearby seeds**
with each inverse method, checks both elbow branches, and includes difficult
named cases. These local tests do not claim convergence from arbitrary seeds.
For the regular case, standard IK takes 12 iterations and damped IK takes 13.
For the near-extension target, the counts are 9 and 146: damping can markedly
slow the final high-accuracy approach.

## 2. Reachability is a separate geometric question

The law-of-cosines oracle follows your planar 2R derivation:

$$
D=\frac{x^2+y^2-L_1^2-L_2^2}{2L_1L_2},\quad
q_2=\operatorname{atan2}(\pm\sqrt{1-D^2},D),
$$

$$
q_1=\operatorname{atan2}(y,x)
-\operatorname{atan2}(L_2\sin q_2,L_1+L_2\cos q_2).
$$

It checks that the target is in the base XY plane and in the annulus
$|L_1-L_2|\leq\sqrt{x^2+y^2}\leq L_1+L_2$, then checks **both** elbow branches
against joint limits. Here the annulus radii are 0.10 and 0.80 m. Joint limits
shrink that set: $(0.3,0)$ is inside the annulus but requires an elbow angle
outside $[-1.7,1.7]$ rad.

The oracle rejects geometrically infeasible targets before motion. It never
feeds a solved joint configuration into the numerical IK iterations or feedback
controller. This keeps its role as an independent feasibility check explicit.
The closed form is specialized to this ideal planar arm and to joint ranges
within $[-\pi,\pi]$; it is not a general robot reachability solver.

The controller also reserves 0.02 rad inside each physical joint limit. A target
reachable only within that reserved margin has status `rejected_guard_margin`.
The numerical IK executable uses physical limits, so the same exact-limit
target can be a valid IK solution while the controller declines to execute it.

## 3. From configuration corrections to a feedback controller

The week 6 law specifies a velocity:

$$
e=x_d-x,\qquad v_d=K e,\qquad \dot q_{raw}=J_p(q)^\dagger v_d.
$$

We use scalar $K\in\{1,4,12\}$ s$^{-1}$, equivalent to $K I$ in the task plane.
The simulation's base is translated and yawed. Scenario targets are written in
**base coordinates**, transformed into world coordinates, and compared with
the world-space site position. `mj_jacSite` supplies the world-oriented
point-position Jacobian. Thus target error and Jacobian share a frame and
reference point. The controller checks this Jacobian against the analytical
week 5 formula at every tick.

For ideal tracking at nonsingular configurations and an error in the task plane,
$\dot e=-K e$. This predicts exponential error decay. Under local forward-Euler
integration, $e_{k+1}\approx(1-K\Delta t)e_k$; the familiar scalar bound
$0<K\Delta t<2$ applies to that simplified local model. Saturation, nonlinear
kinematics, limits and actuator dynamics change this behavior, so the bound is
not a general robot stability guarantee.

An IK iteration has no physical duration. A feedback update does:
the controller runs every 0.01 s, while MuJoCo's physics timestep is 0.002 s.

## 4. Standard pseudoinverse versus damped least squares

For $J=U\Sigma V^T$, the standard pseudoinverse multiplies each retained task
direction by $1/\sigma_i$. The implementation discards singular values below
$\max(10^{-12},10^{-8}\sigma_{max})$ to determine numerical rank.

Damped least squares instead solves

$$
\min_{\dot q}\|J\dot q-v_d\|^2+\lambda^2\|\dot q\|^2,
\qquad
\dot q=J^T(JJ^T+\lambda^2I)^{-1}v_d,
$$

using the spectral gains

$$
\frac{\sigma_i}{\sigma_i^2+\lambda^2}
\quad\text{instead of}\quad \frac1{\sigma_i}.
$$

We use $\lambda=0.03$ in the position Jacobian's units (m when joint angles are
treated as dimensionless radians). The two-column implementation rotates the
Jacobian's columns into orthogonal directions and measures their lengths;
it avoids a singular matrix inverse and handles rank-zero and rank-one cases.
NumPy independently verifies its values, including near-singular and numerical
rank-cutoff cases. For larger robots, use a tested library SVD.

Damping suppresses joint-rate amplification but leaves more task-velocity
residual. Each attainable error direction decays at an idealized rate
$K\sigma_i^2/(\sigma_i^2+\lambda^2)$. Near a singularity, that can be much slower
than $K$. The CSV's `linear_residual_m_s` measures $\|v_d-J\dot q_{raw}\|$ before
speed or joint-limit modification.

At the nearly straight initial pose $q=(0,10^{-4})$, reaching toward $(0.65,0)$
with $K=4$ produces the following first commands:

| Method | Raw shoulder rate | Raw elbow rate | Sent shoulder/elbow rates |
|---|---:|---:|---:|
| Standard pseudoinverse | −13,333 rad/s | 30,476 rad/s | −0.656 / 1.500 rad/s |
| Damped, $\lambda=0.03$ | −0.00493 rad/s | 0.01097 rad/s | −0.00493 / 0.01097 rad/s |

The damped rates grow as the arm bends; damping does not replace the speed cap.
Logging raw and bounded commands separately prevents the cap from hiding the
inverse methods' very different behavior.

### Damping cannot create a missing motion direction

At the **exactly straight** pose $q=(0,0)$, the tip is at $(0.8,0)$ and the
position Jacobian only generates instantaneous Y motion. For target $(0.65,0)$,
the radial error is orthogonal to its column space, so both methods return zero
joint correction. The target has feasible IK solutions, yet the local method
is stationary with 0.15 m error.

The controller reports `local_stationary` before taking a step. It does not
inject a hidden perturbation or claim unreachability. Compare that run with
`near_singular_start`: a small, explicit change to the initial elbow angle
allows the controller to bend the arm and converge. Alternative seeds, restarts
or a planner are ways to address such cases in a more general system.

## 5. Joint limits and two execution modes

`limit_velocity` first scales the whole raw joint-rate vector until its largest
absolute component is at most 1.5 rad/s, preserving its direction. It then
bounds each component according to distance from the guarded limits:

$$
\dot q_i\in
[\max(-v_{max},k_l(q_{min,i}-q_i)),
 \min(v_{max},k_l(q_{max,i}-q_i))],\quad
k_l=\min(5,1/\Delta t).
$$

This slows outward motion near a bound. Independent component limiting can
change the Cartesian direction. It is a simple local rule, not a constrained
trajectory optimizer or a guarantee that every feasible target will be reached.

**Kinematic mode** integrates $q_{k+1}=q_k+\Delta t\dot q_{cmd}$ directly and
calls `mj_forward` to measure poses. This assumes perfect joint-velocity tracking.
The approach bound prevents an ideal Euler step from crossing the guarded
limits; assertions check this without clipping the integrated position.

**Dynamics mode** sends the bounded rate to two velocity actuators and calls
`mj_step` five times. After the initial reset, it never overwrites `qpos` or
`qvel`. The actuators generate approximately

$$
\tau_i=\operatorname{clip}(8(\dot q_{cmd,i}-\dot q_i),-12,12)\ \mathrm{N\,m}
$$

with unit gear ratios. Inertia, damping and coupling determine actual motion.
The MJCF uses `implicitfast` for the velocity servos; see the official
[MuJoCo velocity actuator reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-velocity).
After stepping, the program calls `mj_forward` before reading site poses for
the next feedback update; see [MuJoCo simulation stepping](https://mujoco.readthedocs.io/en/stable/programming/simulation.html).

The command cap is not a hard measured-speed bound: the largest measured rate
in this batch is about **1.539 rad/s**, despite the 1.5 rad/s command cap.
All physics substeps are checked for physical joint-limit violation, nonfinite
state and simulator warnings; the observed maximum limit violation is zero.
MuJoCo joint limits are soft constraints. The margin and approach rule worked
for these runs, but do not prove limit invariance for arbitrary inertia, gains
or disturbances. The default arm moves horizontally, so gravity creates no
joint-axis torque. Tilting its plane changes that property.

Contact is disabled in this teaching scene. Pose convergence here says nothing
about collision avoidance or validity of the path around obstacles.

## 6. Results and how to interpret them

Success requires error below 1 mm and joint-speed norm below 0.02 rad/s across
consecutive controller observations spanning 0.25 s. The time budget is 8 s.
`first_hit_s` records the first position-tolerance crossing; it is −1 if none.
`elapsed_s` is when the run finishes, including settling. `time_limit` means
the experiment stopped before meeting this criterion, not proven unreachability.

The default batch yielded:

| Scenario group | Combinations | Outcome |
|---|---:|---|
| Regular, near extension, nearly straight start, near joint limits | 48 | All converged |
| Exactly straight start with inward radial target | 12 | Local stationarity correctly detected |
| Outer radius, inner hole, elbow/shoulder limits, out of plane, guard margin | 72 | Rejected before execution |

Selected **MuJoCo dynamics**, $K=4$ results:

| Scenario | Standard settling time | Damped settling time | Standard final error | Damped final error |
|---|---:|---:|---:|---:|
| Regular | 2.33 s | 2.35 s | 0.258 mm | 0.280 mm |
| Near extension | 2.31 s | 3.16 s | 0.091 mm | 0.355 mm |
| Nearly straight start | 1.91 s | 1.99 s | 0.353 mm | 0.367 mm |
| Near joint limits | 1.94 s | 1.94 s | 0.315 mm | 0.321 mm |

For the regular damped dynamics run, gains 1, 4 and 12 settle in 6.99, 2.35
and 1.69 s respectively. Increasing gain speeds this example up, but speed and
torque saturation limit the improvement; faster decay does not scale directly
with gain. These measurements do not establish that a higher gain is always
stable or better.

The path plots show that saturation, joint limits and dynamics can bend the
Cartesian path. The summary's integrated error is a left-rectangle sum
$\sum_k\|e_k\|\Delta t$ **until that run ends**; it is not a learned value
function or an optimality certificate. Last digits and settling ticks can
differ across compilers and platforms.

## 7. RL milestone: name each part correctly

| Concept | Role in this project |
|---|---|
| Robot model | FK and Jacobian describe geometry; masses and inertias describe dynamics |
| Planner | Chooses goals or a path using a model and an objective; a global path planner is not implemented here |
| IK solver | Finds a candidate joint configuration for a desired tip position; it does not specify how to execute the motion |
| Policy | Maps observations to actions; this deterministic feedback rule can serve as a policy when actions are joint-velocity references |
| Controller | Repeatedly reduces tracking error; the Cartesian controller produces rates and the velocity servos produce torques |
| Value function | Expected discounted future return under a policy, as evaluated in week 5 Gridworld |
| Environment dynamics | MuJoCo advances physical state given actuator inputs, over the chosen action interval |

“Policy” and “controller” are roles that can overlap. Neither term implies
that the mapping was learned. The reaching controller is **model-based** because
it uses a Jacobian. It is not a dynamic-programming optimizer and does not
compute $V$ or perform policy improvement in the Bellman sense.

A possible reaching MDP at the outer controller rate is

$$
s_t=[q_t,\dot q_t,x_d-x_t],\quad a_t=\dot q_{cmd,t},\quad
s_{t+1}=F_{\Delta t}(s_t,a_t),\quad r_t=-\|x_d-x_{t+1}\|.
$$

The action is held over five physics steps. For a fixed arm model, the target
can be recovered from $x_d-x_t$ and FK. If success termination depends on the
settling counter used here, include that counter in the environment state;
the basic observation alone does not encode its history. Treat the 8 s budget
as an evaluation truncation, or include time remaining if it defines the task.
The current batch is a controller evaluation, not an RL environment API or
a training run.

Your Chapters 3–4 connection is that Bellman backups need the transition model
$p(s',r\mid s,a)$, whereas the Jacobian supplies only a **local velocity map**.
Having MuJoCo as a simulator does not make tabular DP tractable over continuous
joint positions, velocities and goals. Model-free learning would instead update
a value function or policy from sampled transitions/returns; this week's
controller performs neither update. The existing week 5 Gridworld remains the
executable DP reference.

For a future VLA system, a language-conditioned policy might choose a target
or Cartesian increment, then this controller could execute it. The target frame,
units, action rate and feasibility rules must agree across that interface.

## Suggested learning sequence

1. Trace `numerical_ik` for the regular case. Explain why it uses a joint
   correction while `reach` uses a rate with units of rad/s.
2. Inspect the two `two_branches` rows in the IK CSV. Compare elbow signs and
   endpoint positions; position alone does not determine endpoint orientation.
3. Predict the raw near-singular commands before inspecting the CSV. Compare
   them with the bounded commands and the measured joint rates.
4. Explain the exact-singularity failure using $J^Te=0$, then run the explicitly
   perturbed starting configuration. Reachability and local convergence differ.
5. Change `kDamping` in `cartesian_control.h` to 0.01 or 0.06 and rebuild both
   executables. Compare amplification, settling and numerical IK accuracy.
6. Try `--gain 0.1` for a regular reach. A `time_limit` result with a reachable
   target shows why a timeout must not be labeled “unreachable.” Rerun the
   default batch afterward to restore the complete plot inputs.
7. Change a scenario's target or initial configuration in `scenarios()`. Use
   the reachability status, residual and limits to explain the result.
8. Identify which quantities a learned policy would replace and which low-level
   checks you would retain. Explain why a negative distance reward is different
   from the long-term value function for that reward.
