"""On-policy interaction, frozen evaluation, and robot-performance measurements."""

import numpy as np

from week11.control import GOAL, Sarsa, epsilon_greedy, state_features
from week11.robot import ControlPendulum

METHODS = {f"{mode}_{label}": (mode, lam)
           for mode in ("joint", "cartesian", "torque")
           for label, lam in (("sarsa", 0.), ("trace", .8))}
ALL_METHODS = ["pd", "pd_gravity", *METHODS]
SCENARIOS = {"nominal": (.04, 3.), "kick": (.04, 3.),
             "weak_motor": (.04, 1.5), "50hz": (.02, 3.), "10hz": (.1, 3.)}


def initial_states(rng, count):
    """Sample q~U[-0.6,1] rad and qdot~U[-0.8,0.8] rad/s, output (B,2).

    Common starts across action interfaces reduce comparison noise. Training
    and evaluation generators are separate; resets do not depend on returns.
    """
    return rng.uniform([-.6, -.8], [1., .8], size=(count, 2))


def train_episode(env, learners, initial, rngs, epsilon):
    """Interleave simulation and sampled-action SARSA without replay.

    Worlds are grouped into len(learners) blocks of B matching initial states.
    At S_t execute stored A_t. At nonterminal S_{t+1}, compute x', select A'
    with PRE-update weights, then update and carry precisely A' forward. This
    ordering is essential to on-policy SARSA (RL notes 10.1 and 12.7). At true
    termination retain reward, omit bootstrap, and do not select another action.
    Reset traces only at the start of the next episode. Returns undiscounted
    episode rewards, shape (methods,B), for the exploratory behavior policy.
    """
    count = len(initial)
    names = list(learners)
    states = env.reset(np.tile(initial, (len(names), 1)))
    x = [state_features(states[i*count:(i+1)*count]) for i in range(len(names))]
    actions = []
    for name, features in zip(names, x, strict=True):
        learner = learners[name]
        learner.reset_traces(count)
        actions.append(epsilon_greedy(learner.values(features), epsilon, rngs[name]))
    returns = np.zeros((len(names), count))
    for _ in range(env.horizon):
        successor, rewards, terminal, _ = env.step(np.concatenate(actions))
        for i, name in enumerate(names):
            learner, sl = learners[name], slice(i*count, (i+1)*count)
            nx = state_features(successor[sl])
            na = np.zeros(count, dtype=int) if terminal else epsilon_greedy(learner.values(nx), epsilon, rngs[name])
            learner.update(x[i], actions[i], rewards[sl], nx, na, terminal)
            returns[i] += rewards[sl]
            x[i], actions[i] = nx, na
    return returns


def evaluate(env, learners, initial, seed, scenario, record=False):
    """Measure frozen greedy policies and baselines on common reset states.

    No weights, traces or reward-rate estimates change. Greedy ties are sampled
    with a fresh reproducible RNG. Return per-world rows of integrated physical
    metrics and optional decision-boundary replay arrays. Tracking RMSE is
    sqrt(integral(e^2 dt)/duration); work is integral(|tau*qdot| dt) [J].
    Torque variation is sum(|Delta tau|)/duration on the fixed 100 Hz servo
    grid, an average absolute torque slew [N m/s], including the first command.
    Success requires |q-goal|<0.1 rad AND |qdot|<0.25 rad/s at the final state;
    this finite-time threshold is not a Lyapunov stability certificate.
    Instability means any observed 100 Hz sample has |error|>1.5 rad or
    |qdot|>8 rad/s. A kick changes velocity by +/-1.5 rad/s halfway through
    the episode; all methods receive the same impulse per paired world.
    """
    count = len(initial)
    names = ALL_METHODS
    states = env.reset(np.tile(initial, (len(names), 1)))
    totals = {}
    history, commands, torque_history = [states.copy()], [], []
    rngs = [np.random.default_rng(np.random.SeedSequence([seed, i, 777])) for i in range(len(names))]
    for t in range(env.horizon):
        if scenario == "kick" and t == env.horizon // 2:
            env.kick(np.tile(np.where(np.arange(count) % 2 == 0, 1.5, -1.5), len(names)))
            states = env.observation()
            # Replace the boundary sample so replay includes the actual impulse.
            if record:
                history[-1] = states.copy()
        actions = []
        for i, name in enumerate(names):
            if name in learners:
                features = state_features(states[i*count:(i+1)*count])
                actions.append(epsilon_greedy(learners[name].values(features), 0., rngs[i]))
            else:
                actions.append(np.zeros(count, dtype=int))
        selected = np.concatenate(actions)
        states, _, _, info = env.step(selected)
        for key, values in info.items():
            totals[key] = totals.get(key, np.zeros_like(values)) + values
        if record:
            history.append(states.copy())
            commands.append(selected)
            torque_history.append(info["torque_integral"] / env.decision_dt)
    duration = env.horizon * env.decision_dt
    rows = []
    for i, name in enumerate(names):
        for world in range(count):
            j = i * count + world
            rows.append(dict(seed=seed, scenario=scenario, method=name, world=world,
                             reward=float(totals["reward"][j]),
                             tracking_rmse=float(np.sqrt(totals["squared_error"][j] / duration)),
                             final_error=float(abs(states[j, 0] - GOAL)),
                             success=int(abs(states[j, 0] - GOAL) < .1 and abs(states[j, 1]) < .25),
                             unstable=int(totals["unstable_time"][j] > 0),
                             work_j=float(totals["work_abs"][j]),
                             effort=float(totals["effort"][j]),
                             torque_slew=float(totals["torque_variation"][j] / duration),
                             saturation_fraction=float(totals["saturated_time"][j] / duration),
                             decision_hz=1 / env.decision_dt))
    replay = None if not record else dict(states=np.stack(history), actions=np.stack(commands),
                                         torques=np.stack(torque_history), methods=np.array(names),
                                         num_envs=count, decision_dt=env.decision_dt,
                                         goal=GOAL, scenario=scenario, seed=seed)
    return rows, replay


def make_evaluation_env(count, scenario, backend, duration):
    dt, limit = SCENARIOS[scenario]
    modes = [name if name not in METHODS else METHODS[name][0] for name in ALL_METHODS for _ in range(count)]
    return ControlPendulum(modes, duration, dt, backend, limit)


def continuing_experiment(backend, count, steps, seed, alpha, initial):
    """Learn differential SARSA in one unbroken noisy joint-target task.

    h=0.5 is constant. Each decision receives an independent external torque
    N(0,0.4^2) N m, held for 0.04 s. This stationary disturbance, together
    with epsilon=0.15 exploration, prevents a trivial motionless rollout. No
    reset occurs at logging intervals or at the collection cutoff; the final
    transition still bootstraps. This demonstrates notes 10.3 rather than
    pretending a finite run proves ergodicity or average-reward optimality.
    Log rate/0.04 and measured mean reward/0.04 as reward per simulated second;
    the learner's internal baseline remains reward per decision.
    """
    env = ControlPendulum(["joint"] * count, backend=backend, continuing=True)
    learner = Sarsa(alpha=alpha, gamma=1., differential=True)
    learner.reset_traces(count)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1011]))
    disturbance_rng = np.random.default_rng(np.random.SeedSequence([seed, 1012]))
    state = env.reset(initial)
    x = state_features(state)
    actions = epsilon_greedy(learner.values(x), .15, rng)
    rows, window = [], []
    total_reward = 0.
    for step in range(1, steps + 1):
        successor, reward, terminal, _ = env.step(actions, disturbance_rng.normal(0, .4, count))
        if terminal:
            raise RuntimeError("Continuing task unexpectedly terminated")
        nx = state_features(successor)
        na = epsilon_greedy(learner.values(nx), .15, rng)
        learner.update(x, actions, reward, nx, na)
        x, actions = nx, na
        window.append(float(reward.mean()))
        total_reward += float(reward.mean())
        if step % 200 == 0 or step == steps:
            rows.append(dict(seed=seed, decisions=step, seconds=step * env.decision_dt,
                             rate_estimate=learner.rate / env.decision_dt,
                             window_reward_rate=np.mean(window) / env.decision_dt,
                             cumulative_reward_rate=total_reward / step / env.decision_dt,
                             terminal=False))
            window.clear()
    return learner, rows
