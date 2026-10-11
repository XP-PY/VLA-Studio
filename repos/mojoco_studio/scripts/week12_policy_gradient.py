"""Learn continuous Gaussian torque with REINFORCE, a baseline, and actor-critic."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np

from week10_value_prediction import write_csv
from week11.experiment import initial_states
from week12.experiment import (EVAL_METHODS, SCENARIOS, ContinuousPendulum, continuing_experiment,
                               evaluate, make_eval_env, train_batch)
from week12.policy_gradient import METHODS, Learner


def save_learner(path, learner, duration):
    np.savez(path, actor=learner.actor.theta, critic=learner.weights, method=learner.method,
             gamma=learner.gamma, actor_alpha=learner.actor.alpha, critic_alpha=learner.critic_alpha,
             duration=duration, rate_per_decision=learner.rate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mjlab", "native"), default="mjlab")
    parser.add_argument("--episodes", type=int, default=4096, help="Episodes per algorithm per seed")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--eval-envs", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--duration", type=float, default=3.)
    parser.add_argument("--actor-alpha", type=float, default=.02)
    parser.add_argument("--critic-alpha", type=float, default=.2)
    parser.add_argument("--gamma", type=float, default=.98)
    parser.add_argument("--continuing-steps", type=int, default=4000)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "build/week12")
    args = parser.parse_args()
    if (min(args.episodes, args.num_envs, args.eval_envs, args.continuing_steps) < 1
            or args.episodes % args.num_envs or min(args.seeds)<0 or len(set(args.seeds))!=len(args.seeds)
            or not 0 <= args.gamma <= 1 or not np.isfinite(args.duration) or args.duration<=0
            or not np.isclose(args.duration/.2, round(args.duration/.2))
            or any(not np.isfinite(a) or a<=0 for a in (args.actor_alpha, args.critic_alpha))):
        parser.error("Use positive sizes/rates, episodes divisible by num-envs, distinct nonnegative seeds, "
                     "gamma in [0,1], duration a positive multiple of 0.2 s")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    train_env = ContinuousPendulum(["torque"]*(len(METHODS)*args.num_envs),
                                   duration=args.duration, backend=args.backend)
    eval_env = make_eval_env(args.eval_envs, args.backend, args.duration, "nominal")
    heldout = initial_states(np.random.default_rng(120012), args.eval_envs)
    learning, episodes, diagnostics, variances, evaluations, continuing = [], [], [], [], [], []
    for seed in args.seeds:
        learners = {name: Learner(name, args.actor_alpha, args.critic_alpha, args.gamma) for name in METHODS}
        starts = np.random.default_rng(seed)
        rngs = {name: np.random.default_rng(np.random.SeedSequence([seed, i, 1211]))
                for i, name in enumerate(METHODS)}
        for batch in range(args.episodes//args.num_envs + 1):
            count = batch*args.num_envs
            if batch % 8 == 0 or count == args.episodes:
                for stochastic in (True, False):
                    rows, _ = evaluate(eval_env, learners, heldout, seed, stochastic=stochastic)
                    for name in EVAL_METHODS:
                        rr = [r for r in rows if r["method"]==name]
                        learning.append(dict(seed=seed, method=name, episodes=count,
                                             policy="stochastic" if stochastic else "mean_action",
                                             actor_updates=batch*train_env.horizon if name=="actor_critic" else batch if name in METHODS else 0,
                                             reward=np.mean([r["reward"] for r in rr]),
                                             discounted_return=np.mean([r["discounted_return"] for r in rr]),
                                             success=np.mean([r["success"] for r in rr])))
                latest = [r for r in learning[-2*len(EVAL_METHODS):] if r["policy"]=="stochastic"]
                print(f"seed={seed} episodes={count}: stochastic reward " + ", ".join(
                    f"{r['method']}={r['reward']:.3f}" for r in latest), flush=True)
            if count==args.episodes:
                break
            returns, stats, variance = train_batch(train_env, learners, initial_states(starts, args.num_envs), rngs)
            for i, name in enumerate(METHODS):
                diagnostics.append(dict(seed=seed, method=name, episodes=count+args.num_envs, **stats[name]))
                for world in range(args.num_envs):
                    episodes.append(dict(seed=seed, method=name, episode=count+world+1,
                                         reward=returns[i, world], terminal=True, truncated=False))
            variances.append(dict(seed=seed, episodes=count+args.num_envs, **variance))
        for name, learner in learners.items():
            save_learner(args.output/f"{name}_seed{seed}.npz", learner, args.duration)
        for scenario in SCENARIOS:
            env = eval_env if scenario=="nominal" else make_eval_env(args.eval_envs, args.backend, args.duration, scenario)
            for stochastic in (True, False):
                rows, replay = evaluate(env, learners, heldout, seed, scenario, stochastic, record=True)
                evaluations.extend(rows)
                mode = "stochastic" if stochastic else "mean_action"
                np.savez_compressed(args.output/f"replay_{scenario}_{mode}_seed{seed}.npz", **replay)
            if scenario!="nominal":
                del env
        diff, records = continuing_experiment(args.backend, args.num_envs, args.continuing_steps, seed,
                                              args.actor_alpha, args.critic_alpha,
                                              initial_states(np.random.default_rng(seed+9120), args.num_envs))
        continuing.extend(records)
        save_learner(args.output/f"continuing_seed{seed}.npz", diff, 0.)
        for filename, records in (("learning", learning), ("episodes", episodes), ("diagnostics", diagnostics),
                                   ("variance", variances), ("evaluation", evaluations), ("continuing", continuing)):
            write_csv(args.output/f"{filename}.csv", records)
    config = vars(args).copy()
    config.update(output=str(args.output.resolve()), wall_seconds=time.monotonic()-started,
                  versions={p: version(p) for p in ("numpy", "mujoco", "mjlab", "mujoco-warp", "torch")},
                  policy_dt=.04, motor_limit=3., std_bounds=[.1, 2.], gradient_norm_limit=10.)
    (args.output/"config.json").write_text(json.dumps(config, indent=2)+"\n")
    print(f"Saved week 12 results in {args.output}; elapsed {config['wall_seconds']:.1f}s", flush=True)


if __name__=="__main__":
    main()
