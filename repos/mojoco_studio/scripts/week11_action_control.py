"""Train SARSA action interfaces, compare robot controllers, and study differential SARSA."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np

from week10_value_prediction import write_csv
from week11.control import Sarsa
from week11.experiment import (ALL_METHODS, METHODS, SCENARIOS, continuing_experiment,
                               evaluate, initial_states, make_evaluation_env, train_episode)
from week11.robot import ACTION_VALUES, ControlPendulum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mjlab", "native"), default="mjlab")
    parser.add_argument("--episodes", type=int, default=1024, help="Episodes per learned method per seed")
    parser.add_argument("--num-envs", type=int, default=16, help="Parallel worlds per learned method")
    parser.add_argument("--eval-envs", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--duration", type=float, default=3.)
    parser.add_argument("--alpha", type=float, default=.2)
    parser.add_argument("--epsilon", type=float, default=.15)
    parser.add_argument("--gamma", type=float, default=.98)
    parser.add_argument("--continuing-steps", type=int, default=4000)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "build/week11")
    args = parser.parse_args()
    if (min(args.episodes, args.num_envs, args.eval_envs) < 1 or args.episodes % args.num_envs
            or args.continuing_steps < 1 or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0
            or not np.isfinite(args.alpha) or args.alpha <= 0
            or not 0 <= args.epsilon <= 1 or not 0 <= args.gamma <= 1
            or not np.isfinite(args.duration) or args.duration <= 0
            or not np.isclose(args.duration / .2, round(args.duration / .2))):
        parser.error("Use positive sizes/alpha, episodes divisible by num-envs, distinct nonnegative seeds, "
                     "epsilon/gamma in [0,1], and duration a positive multiple of 0.2 s")
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    learning, episodes, evaluations, continuing = [], [], [], []
    heldout = initial_states(np.random.default_rng(110011), args.eval_envs)
    training_modes = [mode for mode, _ in METHODS.values() for _ in range(args.num_envs)]
    train_env = ControlPendulum(training_modes, duration=args.duration, backend=args.backend)
    eval_env = make_evaluation_env(args.eval_envs, "nominal", args.backend, args.duration)
    for seed in args.seeds:
        learners = {name: Sarsa(alpha=args.alpha, gamma=args.gamma, lam=lam)
                    for name, (_, lam) in METHODS.items()}
        rng = np.random.default_rng(seed)
        rngs = {name: np.random.default_rng(np.random.SeedSequence([seed, i, 11]))
                for i, name in enumerate(METHODS)}
        for batch in range(args.episodes // args.num_envs + 1):
            count = batch * args.num_envs
            if batch % 8 == 0 or count == args.episodes:
                rows, _ = evaluate(eval_env, learners, heldout, seed, "nominal")
                for name in ALL_METHODS:
                    selected = [r for r in rows if r["method"] == name]
                    learning.append(dict(seed=seed, method=name, episodes=count,
                                         updates=batch * train_env.horizon,
                                         reward=np.mean([r["reward"] for r in selected]),
                                         success=np.mean([r["success"] for r in selected])))
                print(f"seed={seed} episodes={count}: " + ", ".join(
                    f"{r['method']}={r['reward']:.3f}" for r in learning[-len(ALL_METHODS):]), flush=True)
            if count == args.episodes:
                break
            returns = train_episode(train_env, learners, initial_states(rng, args.num_envs), rngs, args.epsilon)
            for i, name in enumerate(METHODS):
                for world in range(args.num_envs):
                    episodes.append(dict(seed=seed, method=name, episode=count + world + 1,
                                         reward=returns[i, world], terminal=True, truncated=False))
        for name, learner in learners.items():
            np.savez(args.output / f"{name}_seed{seed}.npz", weights=learner.weights,
                     mode=METHODS[name][0], lam=learner.lam, gamma=args.gamma,
                     alpha=args.alpha, duration=args.duration, actions=ACTION_VALUES[METHODS[name][0]])
        for scenario in SCENARIOS:
            env = eval_env if scenario == "nominal" else make_evaluation_env(
                args.eval_envs, scenario, args.backend, args.duration)
            rows, replay = evaluate(env, learners, heldout, seed, scenario, record=True)
            evaluations.extend(rows)
            np.savez_compressed(args.output / f"replay_{scenario}_seed{seed}.npz", **replay)
            if scenario != "nominal":
                del env
        diff, records = continuing_experiment(args.backend, args.num_envs, args.continuing_steps,
                                              seed, args.alpha, initial_states(np.random.default_rng(seed + 9000), args.num_envs))
        continuing.extend(records)
        np.savez(args.output / f"differential_seed{seed}.npz", weights=diff.weights,
                 rate_per_decision=diff.rate, alpha=diff.alpha, beta=diff.beta, gamma=1., lam=0.)
        # Save progress after each complete seed, including a potentially slow continuation run.
        write_csv(args.output / "learning.csv", learning)
        write_csv(args.output / "episodes.csv", episodes)
        write_csv(args.output / "evaluation.csv", evaluations)
        write_csv(args.output / "continuing.csv", continuing)
    config = vars(args).copy()
    config["output"] = str(args.output.resolve())
    config["wall_seconds"] = time.monotonic() - start
    config["versions"] = {p: version(p) for p in ("numpy", "mujoco", "mjlab", "mujoco-warp", "torch")}
    config["interfaces"] = {key: value.tolist() for key, value in ACTION_VALUES.items()}
    config["rates_hz"] = {"physics": 500, "servo": 100, "training_policy": 25}
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Saved week 11 experiments in {args.output}; elapsed {config['wall_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
