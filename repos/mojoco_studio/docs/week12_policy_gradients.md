# Week 12: Gaussian policy gradients with MuJoCo and mjlab

This week implements the three Chapter 13 deliverables:

1. REINFORCE using complete sampled returns.
2. REINFORCE with a learned state-value baseline.
3. One-step actor-critic using a bootstrapped TD error.

All learn a **continuous Gaussian torque policy** for the same one-joint
reaching task as week 11. A separate continuing actor-critic experiment reviews
the average-reward setting. Physics uses mjlab's MuJoCo Warp simulation layer;
the small NumPy algorithms expose the book's updates without a PPO training
framework or automatic differentiation hiding the score calculation.

## Run and view

```bash
uv sync --locked
uv run --locked python scripts/week12_policy_gradient.py
uv run --locked python scripts/plot_week12.py
uv run --locked python scripts/view_week12.py
```

Defaults are 4,096 episodes **per method per seed**, 32 parallel worlds per
method, and three seeds. The three algorithms occupy 96 training worlds.
Each episode lasts 3 seconds, with 75 policy decisions. Continuing learning
uses another 4,000 decisions per world per seed, with no episode resets.
The verified local GPU run took about 304 seconds with cached Warp kernels;
runtime depends on hardware, compilation, and system load.

A short CPU run is useful for stepping through the source:

```bash
uv run --locked python scripts/week12_policy_gradient.py \
  --backend native --episodes 64 --num-envs 8 --eval-envs 8 --seeds 0 \
  --continuing-steps 100 --output build/week12_quick
uv run --locked python scripts/plot_week12.py --input build/week12_quick
```

Use separate output directories to retain experiments. Episode count must be
divisible by `--num-envs`. Duration must be a positive multiple of 0.2 seconds
so the training and frequency-transfer evaluations have exact boundaries.

The MuJoCo viewer replays held-out robot motion:

- **M**: PD reference, REINFORCE, baseline REINFORCE, or actor-critic.
- **D**: sampled Gaussian policy versus deterministic mean-action controller.
- **C**: nominal, velocity impulse, or 10 Hz transfer scenario.
- **N/P**: paired initial condition; **Space**: pause; **R**: replay.
- `--seed 1` selects another run; `--speed 0.5` slows playback.

The overlay shows raw Gaussian sample, mean, standard deviation, and mean
motor torque over the decision interval. These are different quantities.
The viewer needs a desktop display; training and figure generation are headless.

## Read the source with your local notes

The references below were checked against:

- `/home/yuel/workspace/llm-prep-2026/docs/Books/Reinforcement_Learning/Reinforcement_Learning_An_Introduction.md`
- `/home/yuel/workspace/llm-prep-2026/docs/Books/Robotics/Modern_Robotics.md`

Section numbers here identify headings in **your notes**. The source docstrings
include formulas, shapes, update timing, and assumptions; this guide connects
the functions into experiments.

| Source | Notes | Review question |
| --- | --- | --- |
| `scripts/week12/policy_gradient.py`: `actor_features`, `GaussianActor` | RL 13.1, 13.7 | What does the policy parameterize directly? |
| `gaussian_log_probability`, `gaussian_score` | RL 13.7 | Why hold the sampled action fixed when differentiating? |
| `episodic_gradient` | RL 13.2–13.3 | Why are both relative-return discounting and outer discounting present? |
| `Learner.monte_carlo` | RL 13.3–13.4 | Why freeze the actor and baseline before using this batch? |
| `Learner.transition` | RL 13.5 | Why can a successor value error bias the actor? |
| `experiment.py`: `continuing_experiment` | RL 13.6 | What changes when optimizing average reward? |
| `ContinuousPendulum.step` and week 11's `step_commands` | MR 8.5, 11.4.1 | Where do motor limits and sample-and-hold enter? |
| `evaluate` | RL 13.7; MR control review | Is executing the mean the same policy as sampling? |

The critic reuses week 11's fixed RBF features. The physics command executor
was factored out of week 11 so discrete and continuous policies use identical
integration, limits, rewards, and measurement conventions.

## 1. Physical task, coordinates, and timing

The pendulum's angle is measured about world +y, with zero hanging down. The
goal is $q_*=0.5$ rad. This also defines a reachable Cartesian tip goal:

$$
p_* = (-0.6\sin q_*,\;0,\;1-0.6\cos q_*)\quad\text{metres}.
$$

It is a one-joint reaching example, not a new multi-joint arm or upright
swing-up benchmark. MR's trajectory/control review helps distinguish this
fixed-target regulation task from week 9's generated trajectory tracking.

Initial states use $q\sim U[-0.6,1]$ rad and $\dot q\sim U[-0.8,0.8]$ rad/s.
The state is $(q,\dot q,h)$, with $h=(T-t)/T$ the remaining fraction of the
finite task. Two identical physical states with different time remaining
need not have equal value. Reaching the target does not terminate early.

Physics runs at 500 Hz, reward diagnostics at 100 Hz, and policy decisions at
25 Hz. The actor holds torque over each 40 ms interval. The PD-plus-gravity
reference instead recomputes its feedback at 100 Hz, exactly as in week 11.
That extra control/model structure is part of the reference, not hidden inside
the learned torque policies.

Each transition reward integrates the same project cost on the 10 ms grid:

$$
R_{t+1}=-\sum_{k\text{ in decision }t}0.01
\left[(q_k-q_*)^2+0.05\dot q_k^2+0.005\tau_k^2\right].
$$

The units use numerical radians, radians/second and N m relative to unit
reference scales. This is rectangle-rule quadrature, not an exact integral.
More negative reward means greater cost. Motor work is measured separately
as $\int|\tau\dot q|dt$; it is not electrical energy consumption.

At the true task horizon, the final reward is retained and continuation value
is zero. There is no automatic reset overwriting the final observation.
The continuing variant instead fixes $h=0.5$ and never terminates at its
collection cutoff.

## 2. Gaussian policy and the score derivation

The actor has four mean weights and one log-standard-deviation parameter:

$$
x_\mu(s)=\begin{bmatrix}1&(q-0.5)/1.5&\dot q/4&2h-1\end{bmatrix}^T,
\qquad \mu_\theta(s)=\theta_\mu^Tx_\mu(s),\qquad \sigma=\exp\eta.
$$

Normalization is fixed and does not clip observations. The mean starts at
zero, with no PD gains or gravity model embedded in it. Its velocity and
position terms can learn feedback; the intercept can learn holding effort.
The shared scalar $\eta$ is the notes' $x_\sigma(s)=1$ special case, rather
than a state-dependent variance network. Initial $\sigma=0.5$.

Sample a dimensionless latent command

$$
U=\mu_\theta(s)+\sigma Z,\qquad Z\sim\mathcal N(0,1).
$$

Holding the realized $u$ fixed, differentiate the log density:

$$
\log\pi_\theta(u\mid s)=-\eta-\tfrac12\log(2\pi)
-\frac{(u-\mu_\theta(s))^2}{2\exp(2\eta)}.
$$

The two parameter blocks give

$$
\nabla_{\theta_\mu}\log\pi=\frac{u-\mu}{\sigma^2}x_\mu(s),\qquad
\frac{\partial\log\pi}{\partial\eta}=\frac{(u-\mu)^2}{\sigma^2}-1.
$$

The first term moves the mean toward or away from the sampled action; the
second adjusts spread depending on whether the sample lies more or less than
one standard deviation from the mean. These are density derivatives, not
probabilities of an exact continuous action. The notes' example $(\mu,\sigma,u)
=(0,2,3)$ yields scalar scores $(0.75,1.25)$, checked by a unit test.

### Score the raw sample, execute the bounded torque

The actuator applies

$$
\tau=3\operatorname{clip}(u,-1,1)\quad\text{N m}.
$$

We retain and score **raw $u$**. The clipping map is fixed, so it can be treated
as part of the environment transition/reward function for an unbounded latent
Gaussian action. The likelihood-ratio identity applies to that augmented MDP.
The 3 N m scale enters the physical consequence and hence the return, not an
extra factor pasted into the log-density gradient.

The distribution of executed torque has probability masses at the two limits.
It is not an ordinary Gaussian density. Scoring `clip(u)` as though it were
the Gaussian sample, discarding saturated samples, or zeroing their scores
would change the estimator. This is a project extension of the book's unbounded
Gaussian example, not a claim that the notes already define a clipped density.
We use neither a tanh-transformed distribution nor its change-of-variables term.

A numerical test makes the distinction concrete. For the bandit objective
$J(\mu)=E[\operatorname{clip}(U,-1,1)]$ at fixed sigma,

$$
\frac{dJ}{d\mu}=P(-1<U<1)
=E\left[\operatorname{clip}(U,-1,1)\frac{U-\mu}{\sigma^2}\right].
$$

Integrating the raw-score expression agrees with the analytical Gaussian CDF
difference. Substituting the clipped sample into the score fails the same check.

## 3. REINFORCE: a frozen-policy episode gradient

Our discounted start-distribution objective is

$$
J(\theta)=E_{\pi_\theta}\left[\sum_{t=0}^{T-1}\gamma^tR_{t+1}\right],
\qquad \gamma=0.98.
$$

Differentiating the Bellman relation produces a local policy derivative plus
the effect of later decisions:

$$
\nabla V_\pi(s)=\int Q_\pi(s,u)\nabla\pi(u\mid s)du
+\gamma E_\pi[\nabla V_\pi(S')\mid s].
$$

Repeated substitution expands the second term into discounted future visits.
Then use $\nabla\pi=\pi\nabla\log\pi$ and a complete sampled return for Q.
This accounts for how actions change later state visits without differentiating
the unknown state distribution or MuJoCo's integration code. The theorem does
not say that future values or state occupancy are independent of the policy.

Compute complete returns backward, $G_T=0$ and
$G_t=R_{t+1}+\gamma G_{t+1}$. For B episodes, take

$$
\widehat{\nabla J}=\frac1B\sum_{b=1}^B\sum_{t=0}^{T-1}
\gamma^tG_{t,b}\nabla_\theta\log\pi_\theta(U_{t,b}\mid S_{t,b}).
$$

There are two distinct discounts: $G_t$ discounts future rewards relative
to time t; the outer $\gamma^t$ weights decisions relative to episode start.
Omitting the outer factor changes this estimator when gamma is below one.

The actor is frozen throughout collection. Scores are computed when actions
are sampled, accumulated using those same parameters, and applied in one
gradient-ascent update after the batch. This follows the frozen-parameter
convention in your notes. Re-evaluating scores after sequential weight changes
would not be this exact batch estimator at a finite learning rate.

We average **episodes**, not all episode timesteps. With a fixed horizon an
additional division by T could be absorbed into learning rate, but the code
keeps the objective's sum visible. No replay, multiple actor epochs, return
normalization, entropy bonus, or PPO surrogate is introduced.

## 4. A baseline that is independent of the current sampled action

The baseline uses $\hat V(s,w)=x_V(s)^Tw$ with the 246 fixed RBF features from
weeks 10–11. It has parameters separate from the actor. Replace $G_t$ by

$$
A_t=G_t-\hat V(S_t,w_{\mathrm{old}}).
$$

The expected subtraction vanishes because

$$
E[\hat V(s)\nabla_\theta\log\pi(U\mid s)\mid s]
=\hat V(s)\nabla_\theta\int\pi(u\mid s)du=0.
$$

The baseline is held fixed for the whole sampled batch. We calculate all
advantages first, update the actor using those residuals, then fit the baseline:

$$
w\leftarrow w+\frac{\alpha_w}{TB}\sum_{t,b}A_{t,b}x_V(S_{t,b}).
$$

Fitting the current returns before computing their actor advantages could
couple the baseline to those sampled actions, invalidating the simple
action-independence argument. A test uses a large critic step to distinguish
the correct pre-fit baseline from that incorrect ordering.

This method still waits for complete returns. Its value function centers the
signal; it does not bootstrap future consequences. That is why the notes
distinguish it from actor-critic despite having both actor and value parameters.

For the baseline learner, `variance.csv` computes **paired** episode-gradient
covariance traces with and without the old baseline on the exact same data.
This is sample gradient dispersion, not a confidence interval for performance.
A poor baseline can increase it. With fewer than two training worlds, this
diagnostic is undefined and written as NaN, not falsely reported as zero.

## 5. One-step actor-critic: update after every transition

Compute the pre-update TD error

$$
\delta_t=R_{t+1}+\gamma\hat V(S_{t+1},w_t)-\hat V(S_t,w_t).
$$

At a true terminal successor, omit its value but keep the reward and both
updates. Use the same error to update critic and actor:

$$
w\leftarrow w+\frac{\alpha_w}{B}\sum_b\delta_bx_V(S_b),\qquad
\theta\leftarrow\theta+\frac{\alpha_\theta}{B}
\sum_b\gamma^t\delta_b\nabla_\theta\log\pi_\theta(U_b\mid S_b).
$$

Both value predictions and the score use pre-update parameters. The target
and advantage are treated as constants for their respective updates. The next
action is sampled **after** updating; a state-value critic has no SARSA
successor-action sample that must be carried forward.

With an exact value function, conditional expected TD error is the advantage.
With value error $e(s)=\hat V(s)-V_\pi(s)$,

$$
E[\delta_t\mid s,u]=A_\pi(s,u)
+\gamma E[e(S_{t+1})\mid s,u]-e(s).
$$

The current-state error is a baseline and cancels against the score in
expectation. The successor error can depend on which action was taken and
need not cancel. Online updates therefore gain immediacy but can introduce
bias from an imperfect critic. This is different from merely centering a
complete return with an inaccurate action-independent baseline.

MC methods make 128 actor updates per seed by default; actor-critic makes
9,600. They see the same number of episodes but have intentionally different
update schedules. `learning.csv` records actor update counts as well as
episodes. All errors within one parallel batch read the same old parameters;
mean-batch updates are not B sequential single-world updates.

## 6. Numerical choices and continuing tasks

Actor step size is 0.02; critic step size is 0.2. The critic uses a mean
over T×B states for a Monte Carlo batch and over B states for each online TD
update. These are explicit optimization choices, not a claim of equal optimizer
work between algorithms.

Before applying an actor update, its Euclidean gradient norm is capped at 10.
Log standard deviation is then projected to $[\log0.1,\log2]$. The Gaussian
mean is not clipped as a policy parameter. These numerical safeguards are
logged and separate from motor clipping. The raw REINFORCE estimator has the
score identity above; a safeguarded parameter increment is not an unbiased
unconstrained gradient step. Constant step sizes, limited features, and policy
changes do not guarantee monotonic improvement or globally optimal control.

The continuing experiment starts a fresh actor and differential critic. An
independent external torque $N(0,0.4^2)$ N m is held over each decision, while
the state uses constant $h=0.5$. There are no episode resets or discount factors.
Use

$$
\delta=R-\bar R+\hat V(S')-\hat V(S),\qquad
\bar R\leftarrow\bar R+0.01\operatorname{mean}_b\delta_b.
$$

This same **pre-update** error drives critic and actor corrections with no
outer $\gamma^t$. This is the zero-trace special case of notes 13.6. The final
collected transition still bootstraps because collection stopping does not
terminate the conceptual task.

The learned rate is reward per decision. Logs divide it and measured reward
by 0.04 seconds to report reward per second. They describe the stochastic
behavior policy, not its deterministic mean-action controller. Differential
values have an arbitrary additive offset; a finite run does not prove the
stationary-distribution or ergodicity assumptions behind the theorem.

## 7. Evaluation and output files

Every checkpoint uses 32 held-out initial states independent of training resets.
All methods share those starts, but each learns from its own on-policy trajectories.
No evaluation updates occur. We evaluate both:

- **Stochastic policy**: sample from the learned Gaussian, matching the training
  policy and its expected-return objective.
- **Mean-action controller**: execute $3\operatorname{clip}(\mu(s),-1,1)$.
  This is a different deterministic controller and generally is not even the
  expected executed torque of the clipped Gaussian.

One stochastic trajectory per initial state is a sample, not an exact conditional
value. Repeated checkpoints reuse evaluation RNG seeds for comparability. The
three training seeds share this evaluation protocol; plot bands show population
standard deviation across seeds, not confidence intervals.

Frozen final policies are also evaluated with a paired +/-1.5 rad/s velocity
impulse near the midpoint, and at 10 Hz without retraining. The impulse represents
a generalized momentum change, not contact simulation. The frequency test changes
the sample-and-hold dynamics and per-second discount horizon. Compare integrated
undiscounted reward across rates; discounted return is directly comparable to
the training objective at the nominal 25 Hz rate.

Generated artifacts stay in ignored `build/week12/`:

| File | Purpose |
| --- | --- |
| `learning.csv` | Stochastic and mean-action evaluations versus samples/update count |
| `episodes.csv` | Exploratory episode rewards and true termination flags |
| `diagnostics.csv` | Gradient norms, norm-clipping use, learned sigma, motor-clipped sample fraction |
| `variance.csv` | Paired MC gradient dispersion with/without the frozen baseline |
| `evaluation.csv` | Final controller, scenario, seed, world and policy-mode measurements |
| `continuing.csv` | Differential rate estimate and observed rates, without episode boundaries |
| `<method>_seed*.npz` | Actor/critic weights and task/learning parameters |
| `replay_<scenario>_<policy>_seed*.npz` | State, raw action, mean, sigma and actual mean torque for replay |
| `config.json` | Versions, arguments, limits and wall time |
| `learning`, `diagnostics`, `trajectories`, `continuing` PNG/SVG | Figures generated by `plot_week12.py` |

The Gaussian and RBF feature definitions, normalization, target and motor scaling
are part of the checkpoint's meaning. Do not reuse weights after changing these
conventions without retraining.

### Observed default-run results

The full 2026-10-09 run produced the following nominal averages across three
training seeds and 32 held-out initial conditions per seed. Higher reward is
better; success requires final $|q-0.5|<0.1$ rad and $|\dot q|<0.25$ rad/s.

| Controller | Sampled reward | Sampled discounted return | Sampled success | Mean-action reward | Mean-action success |
| --- | ---: | ---: | ---: | ---: | ---: |
| PD + gravity reference | -0.167 | -0.131 | 100% | -0.167 | 100% |
| REINFORCE | -0.276 | -0.192 | 22.9% | -0.205 | 100% |
| REINFORCE + baseline | -0.291 | -0.201 | 22.9% | -0.205 | 100% |
| One-step actor-critic | -0.336 | -0.223 | 15.6% | -0.239 | 9.4% |

The reference has no sampling, so its two columns are the same controller.
All three learned stochastic policies improve substantially over initialization,
but none beats the known-model PD reference in this budget. REINFORCE's
deterministic mean controller succeeds more reliably than its noisy policy;
that does not turn the mean-action result into the trained stochastic objective.
Conversely, the imperfect actor-critic mean misses the narrow final threshold
more often than some sampled rollouts. Threshold success and expected return
rank different aspects of behavior.

Final normalized standard deviations were about 0.33–0.39, corresponding to
roughly 1.0–1.2 N m of raw torque spread before saturation. Sampled torque slew
averaged about 29–31 N m/s, versus about 1.3 N m/s for learned mean-action
controllers. Gradient norm clipping was not activated in this run. These
diagnostics explain the visible difference when toggling **D** in the viewer.

On the baseline learner's last 32 batches, mean paired gradient-covariance
trace fell from about 5.95 without centering to 1.15 with the old baseline,
about an 81% reduction. This is a same-data diagnostic within that learner;
its policy and trajectories differ from the separate plain REINFORCE run.
Lower variance therefore does not establish a superior final reward, especially
with fixed step sizes and limited training. Actor-critic also introduces an
approximate successor value and a different update schedule, so it is not
guaranteed to outperform either Monte Carlo method.

The continuing rate estimates finished near `-0.0845`, `-0.0841`, and `-0.0752`
reward/second; the corresponding last-window observed rates were approximately
`-0.0799`, `-0.0831`, and `-0.0762`. Their agreement is a useful rate-tracking
check, not evidence that the policies have reached the optimal average reward.

## Validation and review order

```bash
WEEK12_GPU_TESTS=1 uv run --locked python -m unittest discover -s tests -p test_week12.py -v
```

The tests check the notes' Gaussian example, finite-difference scores, the clipped
action objective, outer discounting, frozen-baseline timing, pre-update actor/critic
errors, terminal masking, continuing rate updates, explicit parameter safeguards,
MC versus online update schedules, motor clipping, and native/mjlab agreement.
The shared physics extraction also runs the earlier week 10/11 regression tests.
All **32 tests** across weeks 10–12 passed with their GPU checks enabled,
including 12 week 12 tests. The full three-seed GPU experiment, a native pilot,
plot generation, and timed MuJoCo viewer startup/replay/shutdown were exercised.

Read and experiment in this order:

1. Derive both Gaussian score components, holding the sampled action fixed.
2. Follow raw action through clipping and identify which value enters the score.
3. Trace `train_batch` to see when each actor and critic changes.
4. Work a two-step discounted episode by hand; keep both discount factors.
5. Explain why baseline fitting happens after constructing its actor advantages.
6. Compare the paired variance diagnostic with performance curves; variance
   reduction alone need not produce a better controller with a fixed budget.
7. Toggle **D** in the viewer and explain why removing sampling changes the
   policy being evaluated.
8. Inspect the continuing update and locate every episodic term that disappears.
