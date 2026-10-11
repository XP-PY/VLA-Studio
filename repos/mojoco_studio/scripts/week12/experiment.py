"""On-policy Gaussian rollouts, held-out evaluation, and continuing learning."""

import numpy as np

from week11.control import GOAL, state_features
from week11.robot import ControlPendulum
from week12.policy_gradient import METHODS, Learner, actor_features

EVAL_METHODS = ("pd_gravity", *METHODS)
SCENARIOS = {"nominal": .04, "kick": .04, "10hz": .1}


class ContinuousPendulum(ControlPendulum):
    """Week 11's same physics and reward, with continuous normalized torque."""

    def step(self, raw, disturbance=None):
        """Execute normalized latent commands (B,) as tau=3*clip(u,-1,1) N m.

        The inherited command executor handles the fixed motor saturation and
        integrates reward at 100 Hz, physics at 500 Hz. Every torque command
        stays fixed for a policy interval (25 Hz by default). Worlds labeled
        pd_gravity instead hold q_d=0.5 rad for the 100 Hz reference controller.
        Return the same (state,reward,terminal,info) contract as week 11. Raw u
        remains available to the caller for Gaussian scoring; never replace
        it by its clipped image when computing log likelihoods (RL notes 13.7).
        """
        raw = np.asarray(raw, dtype=float)
        if raw.shape != (self.num_envs,) or not np.isfinite(raw).all():
            raise ValueError("Need finite latent actions shaped (B,)")
        commands = np.where(self.modes == 4, GOAL, 3*raw)
        return self.step_commands(commands, disturbance)


def train_batch(env, learners, initial, rngs):
    """Collect one synchronized batch per method, each under its own policy.

    Groups have B paired initial states but diverge through their own actions.
    REINFORCE policies and baselines stay frozen for the entire T transitions;
    score vectors are saved at sampling time, then a single mean-episode update
    is applied after termination. Actor-critic updates every transition, and
    samples its NEXT action only after that update (a state-value critic needs
    no SARSA next-action sample). No replay or cross-policy experience is used.
    Returns per-episode rewards (3,B), diagnostics, and paired baseline gradient
    variance. Each critic reads pre-update weights; terminal rewards are kept.
    """
    count = len(initial)
    states = env.reset(np.tile(initial, (len(METHODS), 1)))
    scores, features, rewards = {m: [] for m in METHODS}, {m: [] for m in METHODS}, {m: [] for m in METHODS}
    stats = {m: [] for m in METHODS}
    saturated = np.zeros(len(METHODS))
    for t in range(env.horizon):
        sampled, current_x = [], []
        for i, name in enumerate(METHODS):
            state = states[i*count:(i+1)*count]
            raw, score, _ = learners[name].actor.sample(actor_features(state), rngs[name])
            sampled.append(raw)
            current_x.append(state_features(state))
            scores[name].append(score)
            features[name].append(current_x[-1])
            saturated[i] += np.mean(np.abs(raw) > 1)/env.horizon
        successor, r, terminal, _ = env.step(np.concatenate(sampled))
        for i, name in enumerate(METHODS):
            sl = slice(i*count, (i+1)*count)
            rewards[name].append(r[sl].copy())
            if name == "actor_critic":
                stats[name].append(learners[name].transition(scores[name][-1], current_x[i], r[sl],
                                                            state_features(successor[sl]), t, terminal))
        states = successor
    result = []
    diagnostics, variance = {}, None
    for i, name in enumerate(METHODS):
        r = np.stack(rewards[name])
        result.append(r.sum(axis=0))
        if name != "actor_critic":
            stats[name].append(learners[name].monte_carlo(np.stack(scores[name]), np.stack(features[name]), r))
        diagnostics[name] = dict(gradient_norm=np.mean([s["gradient_norm"] for s in stats[name]]),
                                 gradient_clipped=np.mean([s["gradient_clipped"] for s in stats[name]]),
                                 sigma=np.exp(learners[name].actor.theta[-1]),
                                 sample_clip_fraction=saturated[i])
        if name == "reinforce_baseline":
            variance = {k: stats[name][0][k] for k in ("gradient_variance_plain", "gradient_variance_baseline")}
    return np.stack(result), diagnostics, variance


def make_eval_env(count, backend, duration, scenario):
    return ContinuousPendulum(["pd_gravity"]*count + ["torque"]*(len(METHODS)*count),
                              duration=duration, decision_dt=SCENARIOS[scenario], backend=backend)


def evaluate(env, learners, initial, seed, scenario="nominal", stochastic=True, record=False):
    """Evaluate frozen Gaussian policies or a separate deterministic mean-action rule.

    Stochastic evaluation samples u~N(mu,sigma^2), matching the learned policy.
    Mean-action evaluation executes 3*clip(mu,-1,1), which is generally NOT
    E[3*clip(u,-1,1)] and is a different controller. Both use paired held-out
    initial states. Discounted return measures J=E[G_0] at the training rate;
    undiscounted reward reports integrated physical cost. A 10 Hz transfer run
    has the same physical duration but a different per-second discount horizon.
    No update of actor, critic, baseline or rate occurs here. Success/error/work
    definitions follow week 11; kick occurs at the middle decision boundary.
    """
    count = len(initial)
    states = env.reset(np.tile(initial, (len(EVAL_METHODS), 1)))
    rngs = {name: np.random.default_rng(np.random.SeedSequence([seed, i, 1212]))
            for i, name in enumerate(METHODS)}
    totals, discounted = {}, np.zeros(env.num_envs)
    history, raw_history, means, sigmas, torque_history = [states.copy()], [], [], [], []
    for t in range(env.horizon):
        if scenario == "kick" and t == env.horizon // 2:
            env.kick(np.tile(np.where(np.arange(count) % 2 == 0, 1.5, -1.5), len(EVAL_METHODS)))
            states = env.observation()
            if record:
                history[-1] = states.copy()
        raw, mean, sigma = [np.zeros(count)], [np.zeros(count)], [np.zeros(count)]
        for i, name in enumerate(METHODS, start=1):
            phi = actor_features(states[i*count:(i+1)*count])
            actor = learners[name].actor
            mu = phi @ actor.theta[:-1]
            action = actor.sample(phi, rngs[name])[0] if stochastic else mu
            raw.append(action)
            mean.append(mu)
            sigma.append(np.full(count, np.exp(actor.theta[-1])))
        states, rewards, _, info = env.step(np.concatenate(raw))
        discounted += next(iter(learners.values())).gamma**t * rewards
        for key, values in info.items():
            totals[key] = totals.get(key, np.zeros_like(values)) + values
        if record:
            history.append(states.copy())
            raw_history.append(np.concatenate(raw))
            means.append(np.concatenate(mean))
            sigmas.append(np.concatenate(sigma))
            torque_history.append(info["torque_integral"]/env.decision_dt)
    duration = env.horizon*env.decision_dt
    rows = []
    for i, name in enumerate(EVAL_METHODS):
        for world in range(count):
            j = i*count+world
            rows.append(dict(seed=seed, method=name, scenario=scenario, world=world,
                             policy="stochastic" if stochastic else "mean_action",
                             reward=float(totals["reward"][j]), discounted_return=float(discounted[j]),
                             tracking_rmse=float(np.sqrt(totals["squared_error"][j]/duration)),
                             final_error=float(abs(states[j, 0]-GOAL)),
                             success=int(abs(states[j, 0]-GOAL)<.1 and abs(states[j, 1])<.25),
                             work_j=float(totals["work_abs"][j]),
                             torque_slew=float(totals["torque_variation"][j]/duration),
                             saturation_fraction=float(totals["saturated_time"][j]/duration),
                             unstable=int(totals["unstable_time"][j]>0)))
    replay = None if not record else dict(states=np.stack(history), raw=np.stack(raw_history),
                                         mean=np.stack(means), sigma=np.stack(sigmas),
                                         torques=np.stack(torque_history), methods=np.array(EVAL_METHODS),
                                         num_envs=count, decision_dt=env.decision_dt, goal=GOAL,
                                         policy="stochastic" if stochastic else "mean_action",
                                         scenario=scenario, seed=seed)
    return rows, replay


def continuing_experiment(backend, count, steps, seed, actor_alpha, critic_alpha, initial):
    """Run zero-trace differential actor-critic on an unbroken noisy task.

    R-rate+V(S')-V(S) drives actor, critic and reward-rate updates (RL 13.6).
    A fresh actor learns continuous torque under independent N(0,0.4^2) N m
    external disturbances held for each 0.04 s decision. h=0.5 forever.
    No terminal mask, outer gamma^t, episode reset or logging-boundary reset
    is applied; the last collected transition still bootstraps. Logs divide
    rate-per-decision by dt to report reward/second. This finite sample does
    not establish stationary-distribution assumptions or global convergence.
    """
    env = ContinuousPendulum(["torque"]*count, backend=backend, continuing=True)
    learner = Learner("actor_critic", actor_alpha, critic_alpha, gamma=1.)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1312]))
    noise = np.random.default_rng(np.random.SeedSequence([seed, 1313]))
    state = env.reset(initial)
    rows, window, total = [], [], 0.
    for t in range(steps):
        raw, score, _ = learner.actor.sample(actor_features(state), rng)
        successor, rewards, terminal, _ = env.step(raw, noise.normal(0, .4, count))
        learner.transition(score, state_features(state), rewards, state_features(successor),
                           t, terminal, continuing=True)
        state = successor
        window.append(float(rewards.mean()))
        total += float(rewards.mean())
        if (t+1) % 200 == 0 or t+1 == steps:
            rows.append(dict(seed=seed, decisions=t+1, seconds=(t+1)*env.decision_dt,
                             rate_estimate=learner.rate/env.decision_dt,
                             window_reward_rate=np.mean(window)/env.decision_dt,
                             cumulative_reward_rate=total/(t+1)/env.decision_dt,
                             sigma=np.exp(learner.actor.theta[-1]), terminal=False))
            window.clear()
    return learner, rows
