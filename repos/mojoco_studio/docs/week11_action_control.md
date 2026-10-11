# Week 11: approximate control and action-space design with mjlab

Week 10 evaluated a fixed controller. Here the learned action values choose
the commands sent to the robot. We implement **semi-gradient SARSA** and
**accumulating SARSA(lambda)**, compare three action interfaces, and run a
separate **differential SARSA** experiment without episode resets.

The state is continuous, but each learned interface has **seven discrete
commands**. This lets us enumerate action values directly as in Chapter 10.
Continuous Gaussian policies and actor-critic methods belong to the next stage.

## Run the experiments

```bash
uv sync --locked
uv run --locked python scripts/week11_action_control.py
uv run --locked python scripts/plot_week11.py
uv run --locked python scripts/view_week11.py
```

The default run uses mjlab GPU simulation, 16 worlds per learned method,
1,024 episodes per method per seed, and three seeds. Six learners train in
96 parallel worlds. A separate continuing experiment runs 4,000 decisions
per seed, with 16 worlds and no resets after initialization. The complete
run took about 136 seconds on the local RTX 4070 SUPER with cached kernels;
compilation, hardware, and system load affect that time.

A short CPU run checks the program flow without waiting for full learning:

```bash
uv run --locked python scripts/week11_action_control.py \
  --backend native --episodes 32 --num-envs 4 --eval-envs 4 --seeds 0 \
  --continuing-steps 40 --output build/week11_quick
uv run --locked python scripts/plot_week11.py --input build/week11_quick
```

Use a separate `--output` directory when retaining multiple runs. Reusing a
directory replaces matching output files. Episode count must be divisible
by `--num-envs`; duration must be a positive multiple of 0.2 seconds so all
tested decision frequencies have exact episode boundaries.

The viewer replays recorded trajectories from the frozen-policy evaluations:

- **M**: cycle the eight controllers.
- **C**: cycle nominal, impulse, weak motor, 50 Hz, and 10 Hz cases.
- **N / P**: select a paired initial condition.
- **Space**: pause; **R**: restart; close the window to exit.
- `--seed 1` selects another trained seed; `--speed 0.5` slows playback.

Unlike week 10, switching the policy now changes the robot's recorded motion.
The orange sphere is the actual target. The torque overlay shows the mean
motor torque over a decision interval, not every 100 Hz servo command.

## Source map and verified local notes

References below use the headings in your local notes:

- `/home/yuel/workspace/llm-prep-2026/docs/Books/Reinforcement_Learning/Reinforcement_Learning_An_Introduction.md`
- `/home/yuel/workspace/llm-prep-2026/docs/Books/Robotics/Modern_Robotics.md`

The Modern Robotics Chapter 5 references are **notes section numbers**, not
invented textbook subsection or equation numbers. Mathematical functions
also explain their formulas and assumptions directly in their docstrings.

| Implementation | Notes to review | Main idea |
| --- | --- | --- |
| `scripts/week11/control.py`: `state_features`, `Sarsa.values` | RL 10.1; 9.4–9.5 | Separate linear feature block for each action |
| `epsilon_greedy` | RL 10.1 | Exploration includes greedy actions; ties are uniform |
| `Sarsa.update` | RL 10.1, 12.7 | Sampled successor action and accumulating eligibility traces |
| `experiment.py`: `train_episode` | RL 10.1, 12.7 interaction order | Select the next action before updating and execute that same action |
| `continuing_experiment` | RL 10.3 | Differential values and a learned average reward rate |
| `robot.py`: `gravity_torque`, `motor_torque` | MR 8.1, 11.4.1 | Gravity compensation, PD feedback, saturation |
| `cartesian_target` | MR notes 5.1, 11.3.3 | Scalar task Jacobian and damped least squares |
| `ControlPendulum.step`, `evaluate` | MR 8.5, 11.4 | Numerical integration, action frequency, effort and disturbances |

The robot uses mjlab's `Simulation` layer and the shared MuJoCo pendulum XML.
The educational environment and NumPy learner explicitly implement actions,
rewards, state, and resets. RSL-RL's PPO runner is not needed for these textbook
algorithms. GPU physics is batched; feature construction and learning run on CPU.

## 1. A common task and three command interfaces

The pendulum rotates about world +y. Zero angle hangs down; the fixed target is
$q_*=0.5$ rad. Initial states are independently sampled as
$q_0\sim U[-0.6,1.0]$ rad and $\dot q_0\sim U[-0.8,0.8]$ rad/s. The task lasts
three seconds. As in week 10, $h=(T-t)/T$ is part of state $(q,\dot q,h)$, so
remaining time is observable. Reaching the target does not terminate early.

| Interface | Seven actions | Execution |
| --- | --- | --- |
| Joint target | `[-0.3, 0, 0.25, 0.5, 0.75, 1.0, 1.3]` rad | Hold desired joint angle; PD plus gravity compensation |
| Cartesian delta | `[-0.08, -0.04, -0.015, 0, 0.015, 0.04, 0.08]` m | Convert world-x tip displacement to a local joint target, then use the same servo |
| Torque | `[-3, -2, -1, 0, 1, 2, 3]` N m | Hold motor torque directly |

The command hierarchy has three separate rates:

1. Physics: 500 Hz (2 ms, `implicitfast`).
2. PD servo: 100 Hz (10 ms), independent of policy frequency.
3. Policy decisions: 25 Hz during training; 50/25/10 Hz during evaluation.

For target commands, the desired angle stays fixed while the servo continues
to read the current position and velocity. Direct torque commands have no
hidden PD feedback or gravity compensation. All motors are capped at 3 N m.
The comparison deliberately includes the extra model/control structure of a
target interface; it is not an experiment where only the action's unit changes.

### Joint feedback and gravity

The XML has a 1 kg rod with center of mass 0.3 m from the hinge and a 0.2 kg bob
at 0.6 m. Since angle is measured from downward,

$$
g(q)=9.81(1\cdot0.3+0.2\cdot0.6)\sin q.
$$

This follows from differentiating gravitational potential
$U(q)=-9.81(1\cdot0.3+0.2\cdot0.6)\cos q$. Armature affects inertia, not gravity.
The holding-torque formula is checked against MuJoCo's bias force at zero speed.

The target servo computes

$$
\tau=\operatorname{clip}\left(6(q_d-q)-1.4\dot q+g(q),-3,3\right).
$$

These are torque gains. This is PD with gravity compensation, not full computed
torque: it does not multiply a desired acceleration by the inertia or cancel
every velocity-dependent term. Saturation and sampling also limit ideal
continuous-time conclusions.

The two hand-designed baselines hold $q_d=q_*$: `pd` omits $g(q)$;
`pd_gravity` includes it. The pure PD baseline needs a nonzero position error
to produce holding torque, illustrating the gravity-induced offset in MR 11.4.1.

### Cartesian delta and a damped Jacobian

World tip x-position is $p_x=-0.6\sin q$, giving $J_x=-0.6\cos q$. Linearize
$\Delta x\approx J_x\Delta q$ and minimize

$$
\left(J_x\Delta q-\Delta x\right)^2+\eta^2\Delta q^2.
$$

Setting the derivative to zero gives

$$
\Delta q=\frac{J_x}{J_x^2+\eta^2}\Delta x,\qquad \eta=0.05\;\text{m/rad}.
$$

The implementation bounds this correction to 0.2 rad and the joint target to
`[-1.3,1.3]` rad. It adds the correction to the current measured angle at the
decision boundary. No previous target is accumulated as hidden state.

This is local damped least squares, not exact inverse kinematics. Near
$q=\pi/2$, x-motion loses first-order controllability; damping avoids division
by zero but cannot recover an unavailable direction. A one-joint pendulum
cannot independently control tip x and z. This action commands only world x.

## 2. Reward, termination, and measurements

At each 100 Hz servo sample, before integration, accumulate

$$
R_{t+1}=-\sum_{k\text{ in decision }t}\Delta t_s
\left[(q_k-q_*)^2+0.05\dot q_k^2+0.005\tau_k^2\right],
\qquad \Delta t_s=0.01\;\text{s}.
$$

Weights use numerical radians, radians/second, and N m relative to unit scales.
This is a project reward design and rectangle-rule approximation to an
integrated cost. Accumulating on the same servo grid makes physical cost
comparisons meaningful when policy frequency changes.

Every finite episode retains its last reward and then has zero continuation
value. There are no time-limit truncations disguised as terminals and no
failure-triggered resets that erase difficult trajectories. Nonfinite physics
raises an error. Threshold failures are recorded even when the episode continues.

| Quantity | Definition and interpretation |
| --- | --- |
| Reward | Undiscounted sum of the integrated negative costs; higher is better |
| Tracking RMSE | $\sqrt{\int(q-q_*)^2dt/\text{duration}}$ in rad |
| Final success | Final $|q-q_*|<0.1$ rad **and** $|\dot q|<0.25$ rad/s |
| Instability flag | Any sampled $|q-q_*|>1.5$ rad or $|\dot q|>8$ rad/s |
| Absolute mechanical work | $\int|\tau\dot q|dt$ in joules |
| Effort integral | $\int\tau^2dt$ in N² m² s |
| Torque slew | $\sum|\tau_k-\tau_{k-1}|/\text{duration}$ in N m/s, at 100 Hz |
| Saturation fraction | Time requesting torque beyond the motor cap / duration |

Mechanical work is not electrical energy: holding still against gravity can
have zero mechanical work while consuming electrical power. Initial torque is
zero when measuring the first torque change. This previous-torque memory is
used only for diagnostics, not reward, and therefore does not add hidden state
to the learning task. A final-state success threshold is not proof of stability.

## 3. From a value estimator to SARSA control

Reuse week 10's 246 fixed RBF features, shifting joint position by the goal:
$z_q=(q-0.5)/1.5$, $z_v=\dot q/4$, and $z_h=2h-1$. For each of seven actions,
store a separate weight row:

$$
\hat Q(s,a)=W_a^Tx(s),\qquad
\nabla_W\hat Q(s,a)=\mathbf e_a x(s)^T.
$$

Features are nonlinear in state but the approximator is linear in $W$.
No neural network, analytic reward oracle, or hand-designed action label is used
to initialize the learner. Weights start at zero. As returns are negative, this
is optimistic initialization and can produce poor early greedy policies.

Actions follow epsilon-greedy with $\epsilon=0.15$ during training. All maximizing
actions share the greedy probability uniformly, including at initialization.
Evaluation uses epsilon zero with uniform ties and frozen weights.

The semi-gradient error is

$$
\delta_t=R_{t+1}+\gamma\hat Q(S_{t+1},A_{t+1})-\hat Q(S_t,A_t),
\qquad \gamma=0.98.
$$

The interaction order matters:

1. Execute the stored $A_t$ and observe the successor and reward.
2. If nonterminal, select $A_{t+1}$ using the **pre-update** weights.
3. Compute the error and update the weights.
4. Execute that same stored $A_{t+1}$ next, even if the update changed which
   action now looks best.

Resampling after the update breaks this correspondence. Replacing the sampled
successor with a maximum gives Q-learning instead. The code test deliberately
changes the greedy action after an update to check that the runner carries the
old sampled action forward.

At termination, the successor term is omitted without selecting another action.
The target is frozen during differentiation, hence “semi-gradient.”

## 4. Eligibility traces

The baseline uses lambda zero. The trace variant uses lambda 0.8:

$$
Z_t=\gamma\lambda Z_{t-1}+\mathbf e_{A_t}x(S_t)^T,
\qquad W\leftarrow W+\alpha\delta_t Z_t.
$$

Each parallel world has its own trace. With a shared weight matrix and B worlds,
the implementation takes the simultaneous mean:

$$
W\leftarrow W+\frac{\alpha}{B}\sum_{b=1}^B\delta_bZ_b.
$$

All errors use the same pre-update weights. This becomes textbook SARSA at B=1;
it is not B sequential updates. Traces reset at new episodes, after the final
reward has used them. Exploratory actions do not clear ordinary on-policy traces.

The traces **accumulate** real-valued RBF gradients. They are neither replacing
traces for binary features nor true-online SARSA's Dutch traces. Ordinary online
accumulating traces are not claimed to exactly match the forward lambda-return
algorithm under changing weights.

A unit test gives reward only on the third transition. With gamma 1, lambda 0.8,
alpha 0.1, and three distinct state-action features, the corrections are
`[0.064, 0.08, 0.1]`. The terminal reward credits all three eligible features.

The default alpha is 0.2 for both variants. Equal alpha does not imply equal
effective update size: accumulating traces amplify repeated features. Constant
step sizes and changing approximate control policies provide no general optimality
guarantee. Chapter 9's fixed-policy linear prediction argument does not establish
convergence of this control experiment.

## 5. Differential SARSA is a different objective

The continuing task uses the joint-target interface, constant $h=0.5$, and
independent external torque $N(0,0.4^2)$ N m at each 25 Hz decision, held for
that interval. It has no periodic resets, success termination, or hidden
remaining-time state. A collection/logging boundary is not a terminal transition.

The objective is the long-run mean reward per decision, rather than a discounted
finite return. Maintain a scalar rate estimate $\bar R$:

$$
\delta_t=R_{t+1}-\bar R_t+\hat Q(S_{t+1},A_{t+1})-\hat Q(S_t,A_t).
$$

Using this **same pre-update error**,

$$
\bar R\leftarrow\bar R+\beta\operatorname{mean}_b(\delta_b),\qquad
W\leftarrow W+\frac{\alpha}{B}\sum_b\delta_b\mathbf e_{A_b}x(S_b)^T,
\qquad \beta=0.01.
$$

There is no discount and this demonstration uses lambda zero. The next action
is still selected before updating and then carried forward. Even the last
collected transition bootstraps because the physical task continues conceptually.

The notes' numerical example is checked: reward 4, old rate 2, current Q 3,
next Q 2 give delta 1. Alpha 0.1 and beta 0.05 produce Q 3.1 and rate 2.05.
Recomputing delta after changing the rate would be a different algorithm.

The plots divide the learned rate and observed rewards by 0.04 seconds to show
reward per simulated second. The internal estimate remains reward per decision.
It tracks the exploratory behavior policy, not a separately evaluated greedy
policy. Differential values are defined up to an additive offset. Neither a
finite run nor noisy exploration proves ergodicity, unique recurrent behavior,
or convergence to an optimal average-reward policy.

## 6. Frozen-policy experiments and results

Each seed uses the same 16 independent held-out initial states for all controllers.
Training policies have different trajectories, so each learner interacts with
its own environment block; they do not learn off-policy from another controller.
No evaluation data enters an update. The baseline is evaluated, not trained.

Five conditions are evaluated without retraining:

| Scenario | Change |
| --- | --- |
| `nominal` | 25 Hz decisions, 3 N m motor cap |
| `kick` | Alternating paired worlds receive +1.5 or -1.5 rad/s velocity impulse near the midpoint (1.48 s by default) |
| `weak_motor` | Motor cap reduced to 1.5 N m |
| `50hz` | Same trained policy, 50 Hz decisions |
| `10hz` | Same trained policy, 10 Hz decisions |

The impulse represents a generalized momentum change, not contact geometry,
friction, or an impact solver. It develops disturbance intuition without claiming
to implement contact control. Gravity at the goal requires about 1.98 N m, so
the weak motor cannot sustain that exact pose; occasional finite-time threshold
success does not remove this physical limit.

The PD servo remains at 100 Hz in frequency comparisons. Cartesian deltas are
metres **per decision**, so rate changes also change their effective velocity
scale. The learned discounted horizon was defined at 25 Hz. These tests measure
transfer of a frozen policy; they do not claim equal action authority or equal
discounted objectives at every frequency. Evaluation reports the same integrated
undiscounted physical cost.

The full default run on 2026-10-09 produced these nominal averages across
three seeds and 16 paired initial states per seed:

| Controller | Reward | Tracking RMSE (rad) | Final success | Absolute work (J) | Torque slew (N m/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pure PD | -0.261 | 0.246 | 0% | 0.806 | 2.48 |
| PD + gravity | -0.187 | 0.144 | 100% | 0.627 | 2.11 |
| Joint SARSA | -0.499 | 0.278 | 33.3% | 4.680 | 12.88 |
| Joint SARSA(0.8) | -0.209 | 0.160 | 66.7% | 1.343 | 15.43 |
| Cartesian SARSA | -1.532 | 0.581 | 12.5% | 2.109 | 3.27 |
| Cartesian SARSA(0.8) | -0.780 | 0.383 | 29.2% | 1.199 | 6.69 |
| Torque SARSA | -0.326 | 0.222 | 4.2% | 2.283 | 21.22 |
| Torque SARSA(0.8) | -0.289 | 0.206 | 6.3% | 2.332 | 29.96 |

These are experimental outcomes, not a promised ranking of action spaces.
The gravity-compensated PD baseline remains strongest on this simple known
regulation task. Joint-target traces approach its cost but switch targets more,
and direct torque policies are much less smooth. The Cartesian basis/action
combination is harder to learn reliably with this budget. Some training-seed
curves regress; a low final reward should not be silently discarded.

One-step torque control has a particularly poor early greedy evaluation in
one seed. The learning plot uses a symmetric-log reward axis so this failure
remains visible alongside later progress. Bands show population SD across
training seeds, not confidence intervals or independent replication of every
held-out state.

The final continuing rate estimates were approximately `-0.0662`, `-0.0654`,
and `-0.0299` reward/second across the three seeds, close to their respective
last-window observed rates. Different seeds can settle into different
approximate policies; rate tracking alone does not demonstrate optimal control.

## Outputs, checks, and review

Everything generated stays in ignored `build/week11/`:

- `config.json`: versions, action grids, seeds, parameters and runtime.
- `learning.csv`: frozen greedy reward/success versus episode and update count.
- `episodes.csv`: exploratory training rewards and true termination flags.
- `evaluation.csv`: all controller/scenario/seed/world physical measurements.
- `continuing.csv`: estimated and measured reward rates, with no terminal flags set.
- `*_seed*.npz`: learned weight matrices and learning/task parameters.
- `replay_<scenario>_seed<seed>.npz`: states, selected actions and decision-mean
  torques, used by the viewer.
- `learning`, `robustness`, `trajectories`, `continuing` in PNG and SVG formats.

Weights are for this feature construction and action ordering. Do not reuse
them with a different target, normalization, or command grid without retraining.
Changing parallel-world count changes the number of mean-batch updates for a
fixed episode budget; `learning.csv` records both counts.

```bash
uv run --locked python -m unittest discover -s tests -p test_week11.py -v
WEEK11_GPU_TESTS=1 uv run --locked python -m unittest discover -s tests -p test_week11.py -v
```

All 12 tests passed with CUDA enabled. They cover sampled-action SARSA, terminal
masking, trace credit/reset, mean-batch updates, differential update timing,
epsilon-greedy ties, carrying the selected successor action, analytical gravity,
damped Cartesian conversion, torque-interface behavior, reward units, continuing
cutoff semantics, and CPU/GPU agreement across interfaces and an impulse. The
full three-seed GPU run, native smoke run, plot generation, and timed MuJoCo
viewer startup/replay/shutdown were also exercised.

Suggested review order:

1. Read `Sarsa.update` with RL notes 10.1 and 12.7. Set lambda to zero on paper.
2. Trace `train_episode`: identify exactly when the successor action is selected.
3. Follow one Cartesian action through `cartesian_target`, the held joint target,
   the 100 Hz servo, the torque cap, and MuJoCo integration.
4. Use the viewer to compare a paired joint-target and direct-torque rollout.
5. Inspect weak-motor results before blaming a learning algorithm for failure.
6. Explain why a continuing collection cutoff must still bootstrap and why the
   reward-rate estimate is not the same quantity as episodic discounted return.
7. Repeat with another step size or budget in a new directory; retain failures
   and compare multiple seeds rather than selecting the best checkpoint.
