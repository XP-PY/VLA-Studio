# Week 10: continuous-state value prediction with MuJoCo and mjlab

This exercise evaluates a **fixed PD policy** on the existing pendulum. It learns
how much discounted future cost remains from a continuous state. The learned
value does not select actions yet; approximate control is a later week's task.

The four experiments combine two feature families (affine and radial basis)
with two backup horizons (one and eight decisions). All four see the same
training trajectories for a given seed. mjlab supplies GPU physics through its
`Simulation` class; our observation, reward, episode, and learning code stays
small enough to read alongside the books. RSL-RL/PPO is not used in this week.

## Run and inspect

From the repository root:

```bash
uv sync --locked
uv run --locked python scripts/week10_value_prediction.py
uv run --locked python scripts/plot_week10.py
uv run --locked python scripts/view_week10.py
```

The default run uses mjlab on `cuda:0`, 32 simultaneous pendulums, 100 decisions
per episode, 1,024 training episodes **per seed**, and seeds 0, 1, 2. First-time
Warp compilation may take longer. A short CPU example uses the same equations:

```bash
uv run --locked python scripts/week10_value_prediction.py \
  --backend native --episodes 128 --num-envs 16 --seeds 0 \
  --output build/week10_quick
uv run --locked python scripts/plot_week10.py --input build/week10_quick
```

Use a distinct output directory for experiments you want to retain. Reusing a
directory replaces files with the same names; `config.json` and the current
CSVs describe the latest run. `--episodes` must be divisible by `--num-envs`.

The MuJoCo viewer replays recorded held-out states with a value/return overlay:

- **N / P**: next / previous held-out episode.
- **M**: change the displayed predictor.
- **Space**: pause; **R**: replay.
- Close the window or use Escape to exit. `--speed 0.5` slows the replay.

Changing the predictor does not change the motion: all four evaluate the same
policy. At the true terminal state the overlay explicitly displays zero value.
The viewer uses a desktop OpenGL display; training and plots work without one.
The old orange reaching marker is hidden because the PD target here is downward.

## Read the source together with book notes

The references below were checked against your local files:

- `llm-prep-2026/docs/Books/Reinforcement_Learning/Reinforcement_Learning_An_Introduction.md`
- `llm-prep-2026/docs/Books/Robotics/Modern_Robotics.md`

Section labels in this table refer to **book notes**; especially the Chapter 5
robotics headings are not a claim about the textbook's subsection numbering.

| Source to read | Relevant notes | What to reconstruct |
| --- | --- | --- |
| `scripts/week10/pendulum.py`: `tip_kinematics` | MR 5.1–5.2 | Differentiate tip position; multiply the point Jacobian by joint velocity |
| `Pendulum.step` | MR 8.5, numerical integration | Forward dynamics and the role of state `(q,qdot)` |
| `policy` | MR 11.4.1 | Torque PD feedback, saturation, and sample-and-hold |
| `prediction.py`: `normalize`, `Features` | RL 9.4, 9.5.5, 9.6 | Fixed nonlinear features can still give a linear predictor |
| `n_step_target`, `Predictor.learn` | RL 7.1 and 9.4 | Delayed targets, current weights, and terminal flushing |
| `semi_gradient_update` | RL 9.3–9.4 | Differentiate the current-state prediction only |
| `complete_returns`, `value_rmse` | RL 9.2–9.3 | Compare value predictions with held-out complete returns |
| `generalization_response` | RL 9.4 | Explain how one update changes predictions at neighboring states |

Each mathematical function has its own formula, conventions, and assumptions
in its docstring. This guide explains how those functions fit together.

## 1. Connect robot state to the prediction task

The XML's hinge rotates about world **+y**, at `(0,0,1)` metres. Zero angle hangs
down, so the 0.6 m tip position and its point Jacobian are

$$
p(q)=(-0.6\sin q,\;0,\;1-0.6\cos q),\qquad
J_p(q)=(-0.6\cos q,\;0,\;0.6\sin q)^T.
$$

Thus `v_tip = J_p * qdot`. This 3×1 Jacobian maps to **tip linear velocity**,
not the six-dimensional spatial twist. A test compares both the position and
velocity against MuJoCo's site position and site Jacobian.

MuJoCo advances the dynamics using the XML's rod and bob inertias, damping, and
joint armature. We do not replace these with a point-mass pendulum equation.
Both Python backends select `implicitfast` in memory and integrate at 2 ms;
the existing XML and C++ exercises retain their original settings.

Every 20 ms, the fixed policy chooses

$$
\tau_t=\operatorname{clip}(-2q_t-0.6\dot q_t,-3,3)\quad\text{N m}.
$$

The torque is held for ten physics steps. Unlike the horizontal-angle example
in MR notes 11.4.1, gravity torque is zero at our downward target. The gains are
torque gains, not the acceleration gains of computed-torque control. This is
a small regulation task, not upright balancing or swing-up.

The reset distribution is `q ~ Uniform[-1.2,1.2]` rad and
`qdot ~ Uniform[-1,1]` rad/s. The transition reward is defined as

$$
R_{t+1}=-0.02\left(q_t^2+0.1\dot q_t^2+0.01\tau_t^2\right).
$$

It uses the state/action **before** integration. Unit reference scales are
1 rad, 1 rad/s and 1 N m; the coefficients define a project-specific cost.
Multiplying by the control interval approximates an integrated cost. More
negative return means more remaining cost. A reward curve here does not show
an improving controller: only the value estimator learns.

## 2. Make termination observable

Each task episode lasts `T=100` decisions (2 seconds by default). Its state is

$$
S_t=(q_t,\dot q_t,h_t),\qquad h_t=(T-t)/T.
$$

Two equal `(q,qdot)` pairs with different remaining time can have different
returns. Including `h` avoids hiding that distinction from the predictor.
At `h=0`, the **finite-horizon task truly terminates**, its continuation value
is zero, and the final reward is retained. No automatic reset overwrites the
terminal observation. Reaching the downward pose does not end an episode.

This differs from externally truncating a continuing task: such a cutoff
usually still needs a bootstrap from the final observation. This implementation
does not provide that continuing-task variant. Nonfinite simulation state raises
an error; it is never silently converted into a terminal transition.

## 3. Replace the value table with shared features

Fixed normalization gives

$$
z=(q/1.5,\dot q/4,2h-1).
$$

No observation clipping or running statistics are used. The two predictors are
both `Vhat(s,w) = x(s)^T w`:

- **Affine**, four weights: `x = [1,z_q,z_v,z_h]/2`.
- **RBF**, 246 weights: a constant plus 245 Gaussian activations on a 7×7×5
  grid in normalized position, velocity, and time. Gaussian widths equal grid
  spacings. The resulting vector is divided by its Euclidean norm.

For RBF center `c_i` and coordinate widths `sigma_j`,

$$
b_i(z)=\exp\left[-\frac12\sum_j((z_j-c_{ij})/\sigma_j)^2\right].
$$

Normalization of the final RBF vector is a project choice to control step-size
scale. For one frozen-target update,

$$
\Delta\hat V(s)=\alpha\delta\|x(s)\|^2,\qquad
\Delta\hat V(s')=\alpha\delta x(s')^Tx(s).
$$

The second identity explains `generalization.png`. RBF overlap concentrates
the change near the updated state, while the constant basis adds a global
component. Affine features share changes broadly. RBF activations are nonlinear
in state, but their centers and widths are fixed: learning is still linear in
the output weights. The feature families need not have independent columns,
and the code does not invert a possibly singular feature covariance matrix.

The radial-basis option fulfills the plan's “tile coding **or** radial-basis
features” deliverable. Tile coding remains a useful comparison exercise.

## 4. Derive the delayed semi-gradient backup

Unroll `G_t = R_{t+1} + gamma*G_{t+1}` for `n` transitions. At update time, with
`j=min(t+n,T)`, substitute the current predictor only if the endpoint continues:

$$
U_t=\sum_{k=t}^{j-1}\gamma^{k-t}R_{k+1}
 +\mathbf1\{j<T\}\gamma^{j-t}x(S_j)^Tw.
$$

Then use `grad_w Vhat = x(S_t)`:

$$
w\leftarrow w+\alpha[U_t-x(S_t)^Tw]x(S_t).
$$

The target is treated as constant during this update. Differentiating the
bootstrap too would add a successor-feature term and implement another
algorithm. With one transition this is semi-gradient TD(0); a window reaching
termination uses a full remaining return.

The notes' numerical example is tested directly: rewards `(1,2,3)`, gamma 0.9,
and endpoint value 4 give `8.146`; a terminal endpoint gives `5.23` instead.

`Predictor.learn` implements the delayed schedule: after transition `k`, update
`tau=k-n+1` if it exists. After the final transition, flush the remaining origin
states without inventing new steps. The endpoint value is recomputed using the
current weights at each backup. Tests cover `n=1`, `n=2`, and `n>T`.

At the default `n=8`, normal backups wait for 160 ms of experience, but torque
commands are still recomputed every 20 ms. Backup horizon is not action duration.

For B simultaneous worlds the implementation uses

$$
w\leftarrow w+\frac{\alpha}{B}X^T(U-Xw).
$$

All worlds read the same pre-update weights. This mean-batch extension is not
B sequential updates. The default step size is 0.1 and gamma is 0.98. Constant
step sizes, correlated features, and different feature norms mean that there
is no promised exact finite-time convergence or universal best backup horizon.

The simulator collects a full batch of episodes first; all predictors then
consume it once in the online delayed order. Because the policy never reads
the learned weights, this gives the same updates as interleaving collection
and learning. Future rewards are not read before their scheduled backup.
This arrangement makes comparisons fair without collecting four separate
datasets. Learning uses NumPy on CPU; mjlab performs batched physics on GPU.
The small task prioritizes readability over maximum GPU throughput.

## 5. Read the results correctly

An independent reset stream produces 32 held-out episodes. A backwards pass
computes their complete returns with `G_T=0`. The policy and physics are
deterministic, so these are numerical value references at the observed states;
they are not analytical solutions. A stochastic policy or noisy dynamics would
require repeated rollouts to estimate conditional expected values.

RMSE weights all 3,200 nonterminal held-out states equally. That measures the
sampled visitation distribution, not uniform error over the whole continuous
state space. No held-out data enters a learning update. The three training seeds
vary reset samples; they share this same held-out set.

The default GPU run on 2026-10-08 produced:

| Predictor | Initial RMSE | Final RMSE, mean ± population SD across 3 seeds |
| --- | --- | --- |
| Affine, TD(0) | 0.11494 | 0.08997 ± 0.00145 |
| Affine, 8-step TD | 0.11494 | 0.08887 ± 0.00209 |
| RBF, TD(0) | 0.11494 | 0.06537 ± 0.00034 |
| RBF, 8-step TD | 0.11494 | 0.02961 ± 0.00117 |

Here the richer features and longer backup help. This does not establish that
eight-step TD always beats one-step TD. The affine model cannot represent the
curvature of a quadratic regulation cost. In particular, the task is symmetric
under `(q,qdot) -> (-q,-qdot)`, whereas nonzero affine position/velocity terms
change sign. More data or a longer backup cannot remove that representation
limit. Value estimates can also be positive even though true returns are
nonpositive: we intentionally do not clip predictions to hide approximation
error.

Changing `--num-envs` changes the number of weight updates for a fixed episode
count. `learning.csv` records both counts. For a fair comparison of batch sizes,
consider update count as well as samples and wall time.

Generated files in ignored `build/week10/`:

| File | Meaning |
| --- | --- |
| `config.json` | Arguments, backend, control interval, and installed versions |
| `learning.csv` | Held-out value RMSE and training target residual by seed/method |
| `episodes.csv` | Initial states, rewards/returns, lengths, and termination reasons |
| `summary.csv` | Final errors relative to the zero predictor |
| `*_seed*.npz` | Weights, feature kind, horizon and learning parameters |
| `evaluation.npz` | Recorded held-out states, rewards, complete returns, and predictions |
| `trajectory.csv` | One held-out episode with joint and Cartesian tip signals |
| `learning.png/.svg` | Learning curves and predicted-versus-reference values |
| `trajectory.png/.svg` | Motion, effort, returns and predictions along one episode |
| `generalization.png/.svg` | How one update affects neighboring states |

Checkpoints describe this feature implementation and task configuration; do
not reuse them for a different horizon or normalization without retraining.

## Validation and review exercises

```bash
uv run --locked python -m unittest discover -s tests -p test_week10.py -v
WEEK10_GPU_TESTS=1 uv run --locked python -m unittest discover -s tests -p test_week10.py -v
```

All eight tests passed with GPU checks enabled: return arithmetic, terminal
reward handling, delayed/tail updates, current-weight bootstrapping, mean-batch
updates, normalization/generalization, reward/reset timing, analytical site
kinematics, and CPU/GPU rollout/reset agreement are covered by those tests.
The full default three-seed run and figure generation were also executed.

Review in this order:

1. Work the three-reward target example by hand, including the terminal case.
2. Trace `Predictor.learn` with an episode shorter than `n`; count updates.
3. Explain why elapsed time must enter this finite-horizon state.
4. Inspect the generalization heatmap and predict what a wider RBF would do.
5. Compare value RMSE with training TD residual; explain why they differ.
6. Repeat with a smaller alpha or a longer training run in a separate output
   directory. Identify approximation limits versus incomplete learning.

These complete week 10's implementation deliverable using the pendulum option.
Cart-pole, continuous-state arm prediction, tile coding, and learned action
selection are natural extensions rather than additional requirements for this
first value-approximation exercise.
