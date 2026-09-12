# Week 5: Jacobians and Dynamic Programming

This completes the implementation and review portion of week 5, building on
the week 4 two-link arm and finite-MDP exercise. The reading is already covered
in your book notes: **Modern Robotics, Chapter 5, especially §§5.1–5.5 and
5.7–5.9**, and **Reinforcement Learning, Chapter 4, especially §§4.2–4.6**.
Section numbers here refer to your notes, whose grouping differs from the books.

## Run and inspect

Run from the project directory, using the existing MuJoCo installation and
`build.sh`. Neither executable needs Python or a display.

```bash
./build.sh gridworld_dp
./build/gridworld_dp
./build.sh jacobians
./build/jacobians

cat build/gridworld_report.txt
cat build/jacobian_report.txt
```

Each program exits with a nonzero status if a numerical check fails. Files in
`build/` are generated and can be recreated with these commands.

| File | What to inspect |
|---|---|
| `src/gridworld_dp.cpp` | Transition model, evaluation, improvement, policy iteration, value iteration |
| `src/two_link_kinematics.h` | Small analytical functions with explicit frame conventions |
| `src/jacobians.cpp` | MuJoCo API calls and independent numerical checks |
| `build/gridworld_values.csv` | Random and optimal values, all greedy actions for every state |
| `build/gridworld_convergence.csv` | Every sweep, current values, maximum update, Bellman residual |
| `build/jacobian_comparison.csv` | Configurations, joint rates, errors, velocities and torques |
| `build/jacobian_entries.csv` | Individual point/angular Jacobian entries and space/body linear entries |
| `build/jacobian_singularities.csv` | Elbow sweep, determinant, singular values and condition number |

Optional plots require Matplotlib in your chosen Python environment:

```bash
python -m pip install 'matplotlib>=3.7,<4'
python scripts/plot_week5.py
```

This produces `build/week5_overview.svg` and `build/week5_overview.png`: two
Gridworld value maps, convergence curves, and a singular-value plot. The plot
script reads the C++ CSVs; it does not implement another simulator or DP solver.

In the current workspace, an isolated plotting environment is already installed
under `build/week5_plots`. You can regenerate the plots directly with
`build/week5_plots/bin/python scripts/plot_week5.py`.

## 1. Gridworld: from evaluation to control

Use the 4×4 example from your Chapter 4 notes. State IDs increase row by row:

```text
 T   1   2   3
 4   5   6   7
 8   9  10  11
12  13  14   T
```

The two terminal cells are represented by IDs 0 and 15 with identical terminal
behavior. This is equivalent in value to the book's single terminal state
drawn in two places. Actions are up, right, down, left. A boundary action stays
in the same cell. Every transition **from a nonterminal cell** costs −1,
including hitting a wall and entering termination. After termination the
reward and value are zero. The default discount is $\gamma=1$.

Read the source in this order:

1. `transition` gives the known deterministic model.
2. `action_value` computes $Q_V(s,a)=r+\gamma V(s')$.
3. `expectation_backup` averages those action values under a fixed policy.
4. `evaluate_policy` repeats that backup with two value arrays.
5. `improve_policy` puts equal probability on every maximizing action.
6. `policy_iteration` alternates evaluation and improvement, reusing values.
7. `value_iteration` repeatedly applies the maximum backup and then extracts a policy.

### Policy evaluation

The random policy selects each action with probability $1/4$:

$$
V_{k+1}(s)=\sum_a\pi(a\mid s)[-1+\gamma V_k(s'_a)].
$$

Starting from zero, sweep 1 gives −1 in every nonterminal cell. At state 1,
sweep 2 gives $(-1-2-2-2)/4=-1.75$: one action terminates and the other three
continue. Find these entries in `gridworld_convergence.csv`.

The converged random-policy values reproduce your notes:

```text
  0  -14  -20  -22
-14  -18  -20  -20
-20  -20  -18  -14
-22  -20  -14    0
```

Here $V_\pi(s)$ is minus the expected number of steps until termination.
Evaluation holds the policy fixed throughout; displaying a greedy policy does
not change the policy being evaluated.

### Policy iteration and ties

Policy iteration evaluates the current policy to tolerance, then changes it
to a greedy policy. We distribute probability equally across all actions
within $10^{-9}$ of the maximum, instead of choosing a random tie winner.
This produces a reproducible policy table. Stop when that table is unchanged,
then independently check the Bellman optimality residual and evaluate the
returned policy again from zero.

The evaluation tolerance is $10^{-12}$; the larger tie tolerance prevents
roundoff from arbitrarily removing mathematically tied actions in this example.
For a different reward scale, revisit both tolerances. Policy stability alone
with approximate values is not a proof of exact optimality.

Because $\gamma=1$, a policy such as always-UP can get trapped at the top edge
forever, accumulating return $-\infty$. `validate_policy` checks normalization
and that every state has a positive-probability path to termination. In this
finite stationary Markov chain, this excludes closed nonterminal recurrent
classes. Starting with the uniform policy is valid; blindly initializing an
arbitrary deterministic policy need not be. An explicit check confirms that
always-UP is rejected. These are episodic assumptions, not a claim that the
discounted contraction proof applies at $\gamma=1$.

### Value iteration

Replace the action average by a maximum:

$$
V_{k+1}(s)=\max_a[-1+\gamma V_k(s'_a)].
$$

The optimal values are minus the shortest distance to either terminal:

```text
 0  -1  -2  -3
-1  -2  -3  -2
-2  -3  -2  -1
-3  -2  -1   0
```

Both control algorithms are checked against this independent distance formula,
not just against each other. The returned policies are also evaluated to verify
they actually achieve the reported values. In the report, `UL` means choose up
or left with probability $1/2$ each; `URDL` means all four actions are optimal
at that state. Multiple optimal policies can share one optimal value function.

Observed default run:

| Method | Work | Final relevant Bellman residual |
|---|---|---|
| Random-policy evaluation | 510 sweeps | $9.38\times10^{-13}$ |
| Policy iteration | 3 evaluation/improvement rounds, 515 total evaluation sweeps | 0 |
| Value iteration | 4 sweeps | 0 |

Value iteration establishes the correct values after three sweeps here; the
fourth detects no further change. The maximum shortest path is three steps.
This is a property of this tiny problem and initialization, not a general speed
guarantee. These counts are planning sweeps, not episodes or environment samples.
The PI convergence CSV records the expectation residual for the policy currently
being evaluated; that residual can jump when improvement changes the policy.

This illustrates **generalized policy iteration (GPI)**: value estimates and
policy improvement interact. Policy iteration separates them into converged
evaluation phases and improvement steps; value iteration interleaves them at
each optimality backup. GPI describes that interaction, rather than requiring
a fourth executable algorithm for this exercise.

## 2. Three Jacobians and their reference frames

Reuse `assets/two_link_arm.xml`: $L_1=0.45$ m, $L_2=0.35$ m, two $+z$ hinge
joints, and the same translated/yawed base. Let $T_{WB}$ be the base pose and
$T_{BE}(q)$ the tip pose relative to the base. Our twist ordering is $[\omega;v]$.

### Point-position Jacobian

Differentiate the week 4 planar position equations:

$$
J_p^B=\begin{bmatrix}
-L_1\sin q_1-L_2\sin(q_1+q_2)&-L_2\sin(q_1+q_2)\\
 L_1\cos q_1+L_2\cos(q_1+q_2)& L_2\cos(q_1+q_2)\\
0&0
\end{bmatrix}.
$$

`point_jacobian` implements this expression. Since the base is fixed,
$J_p^W=R_{WB}J_p^B$ and $\dot p_{WE}=J_p^W\dot q$. Translation does not affect
this conversion of a point velocity between fixed coordinate axes.

### Space and body Jacobians

The home screws from week 4 yield

$$
J_s^B=[S_1,\operatorname{Ad}_{e^{[S_1]q_1}}S_2]
=\begin{bmatrix}
0&0\\0&0\\1&1\\0&L_1\sin q_1\\0&-L_1\cos q_1\\0&0
\end{bmatrix}.
$$

`space_jacobian` writes this specialized PoE result explicitly. It does not
depend on $q_2$: only upstream joints affect a space-Jacobian column.
`body_jacobian` uses

$$
J_b=\operatorname{Ad}_{T_{BE}^{-1}}J_s^B.
$$

Its angular columns are both $[0,0,1]^T$ and its linear block simplifies to

$$
J_{b,v}=\begin{bmatrix}
L_1\sin q_2&0\\L_1\cos q_2+L_2&L_2\\0&0
\end{bmatrix}.
$$

The executable checks this independent closed form as well. Transforming the
**space twist** to the world requires the full adjoint,
$J_s^W=\operatorname{Ad}_{T_{WB}}J_s^B$, including base translation.

### Match MuJoCo's conventions before comparing

`mj_jacSite` fills separate row-major arrays of shape `3 × model->nv`:
`jacp` maps joint rates to site-origin linear velocity in world axes, and
`jacr` maps them to angular velocity in world axes. Columns use
`jnt_dofadr`; setting positions uses `jnt_qposadr`. They coincide for this
hinge-only model but need not for other joint types. See the official
[MuJoCo Jacobian API](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-jac).

For each column, with $p=p_{WE}$:

$$
J_{s,v}^W=J_p^W-\omega^W\times p,
\qquad J_{b,v}=R_{WE}^T J_p^W,
\qquad J_{b,\omega}=R_{WE}^T J_\omega^W.
$$

The lower block of $J_s$ is **not** generally $\partial p/\partial q$.
At home, for example, column 1 of $J_s^B$ has zero linear component, but
column 1 of $J_p^B$ is $[0,0.8,0]^T$. A shoulder rotation moves the tip even
though the rotation axis passes through the space-frame origin.

## 3. Velocity and finite-difference experiment

For 100 seeded random configurations plus six explicit edge cases:

1. Set joint positions and random joint rates in $[-1,1]$ rad/s.
2. Call `mj_forward` to refresh position and velocity calculations.
3. Compare the analytical point, angular, space and body Jacobians with the
   corresponding MuJoCo quantities, after frame/reference-point conversion.
4. Perturb each joint by $\pm h$, $h=10^{-6}$ rad, and differentiate site position.
   Differentiate rotation matrices using $[\omega]=\dot R R^T$ as well.
5. Restore the unperturbed state and compare $J\dot q$ with `mj_objectVelocity`
   in world and local axes.
6. Independently sample $q(t)=q+t\dot q$ at $t=\pm10^{-6}$ seconds and measure
   the central-difference site velocity.

`mj_objectVelocity` returns angular components first, linear components second,
centered at the requested object. With local axes it gives the body twist;
with world axes its linear part is still the site-origin velocity. See the
[official velocity API](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-objectvelocity).

These are prescribed kinematic snapshots. `mj_step` would evolve joint rates
under dynamics and the existing position servos; it would measure a different
experiment. The finite-difference edge probes may briefly cross a joint limit:
they evaluate smooth FK, not feasible control trajectories.

## 4. Singularities and static force mapping

For planar XY position,

$$
\det J_{xy}=L_1L_2\sin q_2.
$$

The singularity CSV sweeps $q_2$ from −1.6 to 1.6 rad at $q_1=0.4$ rad. At
$q_2=0$, the straight arm cannot instantaneously move its tip along the arm:
the two position-Jacobian columns are collinear. The smallest singular value
vanishes and the condition number is reported as `inf` at numerical rank loss.
The folded case $q_2=\pi$ is also singular mathematically but outside this MJCF's
joint limits, so it is not included in the simulation sweep.

For this 2R arm the full **6×2 twist Jacobian remains rank 2**: orientation
distinguishes motions that have the same tip-position velocity. Its space-frame
Gram determinant is $L_1^2>0$. Thus specify the task when saying “singular”;
space and body Jacobians have the same rank, but dropping orientation rows
changes the task. This matters for the position-reaching controller in week 6.

For a world-frame force $f=[1.2,-0.7,0.4]$ N and a moment about the tip
$n=[0.1,0.2,0.3]$ N m, compute

$$
\tau=(J_p^W)^Tf+(J_\omega^W)^Tn.
$$

Compare it with `mj_applyFT` using a zero-initialized generalized-force buffer
(the API adds to the buffer). Also verify the body and space wrench versions:

$$
F_b=[R_{WE}^Tn;R_{WE}^Tf],\qquad F_s^W=[n+p\times f;f],
\qquad \tau=J_b^TF_b=(J_s^W)^TF_s^W.
$$

Finally check power balance $\tau^T\dot q=f^T\dot p+n^T\omega$.
These are generalized torques induced by an external wrench; holding against
that wrench requires opposite actuator torques plus compensation for other
loads. This experiment checks the static mapping without running an equilibrium
controller. See [MuJoCo force application](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-applyft).

## 5. Observed numerical results

The default run with the bundled MuJoCo produced these maximum absolute
component errors across all 106 configurations:

| Check | Maximum error | Pass threshold |
|---|---|---|
| Analytical vs MuJoCo Jacobians | $7.77\times10^{-16}$ | $10^{-12}$ |
| Central-difference Jacobians | $3.49\times10^{-10}$ | $10^{-8}$ |
| World/body site velocities | $8.88\times10^{-16}$ | $10^{-12}$ |
| Central-difference velocities | $3.34\times10^{-10}$ | $10^{-8}$ |
| Wrench-to-torque mapping | $6.11\times10^{-16}$ | $10^{-12}$ |
| Power balance | $3.33\times10^{-16}$ | $10^{-12}$ |

The Jacobian/velocity summary combines angular and linear components, so it is
a componentwise validation statistic, not a physical norm with a single unit.
The entry CSV lets you inspect individual quantities in their appropriate units.
Analytical differences are floating-point roundoff. Central differences also
have $O(h^2)$ truncation error and roughly $O(\epsilon_{machine}/h)$ cancellation
error. Making $h$ smaller indefinitely does not improve the estimate. A
singularity does not invalidate computing $J$; inversion is what becomes
ill-conditioned. Exact last digits can differ across compilers/platforms.

## 6. Suggested study session and VLA connection

1. Predict the first two random-policy sweeps before opening the CSV.
2. Trace one policy-improvement decision at state 5, then examine ties at state 6.
3. Change `kGamma` to 0.9. The distance oracle uses discounted step costs; the
   undiscounted textbook-table check is deliberately enabled only for gamma 1.
4. At $q=(0,0)$ and $\dot q=(1,0)$, predict base-frame tip velocity
   $[0,0.8,0]$ m/s. Then rotate it by the base yaw to predict world velocity.
5. Inspect `jacobian_entries.csv` at sample 0 to explain why point and space
   linear columns differ. Set `dq={1,0}` temporarily to check your velocity answer.
6. Change `kH` to $10^{-4}$ or $10^{-8}$ and observe the finite-difference errors.
   Threshold failures are useful evidence; do not simply loosen them to hide errors.
7. Compare the straight and bent arm in the optional viewer
   (`./simulate.sh two_link_arm`) and in the singular-value plot. Predict which
   tip velocity direction a future reaching controller will find difficult.

DP and Jacobians both use known models for different jobs. Gridworld's model
supports planning over long-term returns. The Jacobian predicts the local
motion caused by joint rates:

$$
\Delta p\approx J_p(q)\Delta q.
$$

A future RL/VLA policy might output a Cartesian displacement, joint target,
or torque. Its action frame, reference point, units and execution interval
must agree with the controller. The checks here make that interface explicit.
MuJoCo provides continuous dynamics, but that alone does not make exhaustive
tabular DP practical for continuous robot states. Week 6 can now build on the
verified Jacobian to implement inverse kinematics and Cartesian control.
