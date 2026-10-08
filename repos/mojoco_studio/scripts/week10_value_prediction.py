"""Compare affine/RBF semi-gradient TD(0) and eight-step TD on a fixed PD policy."""

import argparse
import csv
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from week10.pendulum import CONTROL_DT, Pendulum, policy, sample_initial, tip_kinematics
from week10.prediction import Features, Predictor, complete_returns, value_rmse


ROOT = Path(__file__).resolve().parents[1]


def write_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mjlab", "native"), default="mjlab")
    parser.add_argument("--episodes", type=int, default=1024, help="Training episodes PER seed")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=100, help="Decisions per finite-horizon task")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "week10")
    args = parser.parse_args()
    if (args.num_envs < 1 or args.episodes < 1 or args.episodes % args.num_envs
            or args.horizon < 1 or not 0 <= args.gamma <= 1
            or not np.isfinite(args.alpha) or args.alpha <= 0
            or min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds)):
        parser.error("Use positive sizes, episodes divisible by num-envs, distinct nonnegative seeds, "
                     "gamma in [0,1] and finite alpha>0")
    args.output.mkdir(parents=True, exist_ok=True)
    # Independent RNG stream: evaluation states are fixed across all training seeds.
    evaluation_rng = np.random.default_rng(np.random.SeedSequence([2026, 10, 999]))
    env = Pendulum(args.num_envs, args.horizon, args.backend)
    eval_states, eval_rewards = env.rollout(sample_initial(evaluation_rng, args.num_envs))
    reference = complete_returns(eval_rewards, args.gamma)
    bases = {name: Features(name) for name in ("affine", "rbf")}
    eval_x = {name: basis(eval_states[:-1]) for name, basis in bases.items()}
    zero_rmse = value_rmse(np.zeros_like(reference), reference)
    metrics, episodes, summaries = [], [], []
    evaluation = {"observations": eval_states, "rewards": eval_rewards, "returns": reference}
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        predictors = {f"{kind}_n{n}": Predictor(basis, n, args.alpha, args.gamma)
                      for kind, basis in bases.items() for n in (1, 8)}
        for name in predictors:
            metrics.append(dict(seed=seed, method=name, episodes=0, updates=0,
                                value_rmse=zero_rmse, td_mse=0.0))
        for batch in range(args.episodes // args.num_envs):
            states, rewards = env.rollout(sample_initial(rng, args.num_envs))
            returns = complete_returns(rewards, args.gamma)
            count = (batch + 1) * args.num_envs
            for world in range(args.num_envs):
                episodes.append(dict(seed=seed, episode=batch * args.num_envs + world + 1,
                                     q0=states[0, world, 0], v0=states[0, world, 1],
                                     discounted_return=returns[0, world],
                                     reward_sum=float(rewards[:, world].sum()),
                                     length=args.horizon, terminated=True, truncated=False,
                                     reason="task_horizon"))
            for name, learner in predictors.items():
                td_mse = learner.learn(states, rewards)
                prediction = np.einsum("tbd,d->tb", eval_x[learner.features.kind], learner.weights)
                rmse = value_rmse(prediction, reference)
                metrics.append(dict(seed=seed, method=name, episodes=count,
                                    updates=(batch + 1) * args.horizon,
                                    value_rmse=rmse, td_mse=td_mse))
            if (batch + 1) % 4 == 0 or count == args.episodes:
                scores = ", ".join(f"{r['method']}={r['value_rmse']:.4f}" for r in metrics[-4:])
                print(f"seed={seed} episodes={count}: held-out RMSE {scores}", flush=True)
        for name, learner in predictors.items():
            prediction = np.einsum("tbd,d->tb", eval_x[learner.features.kind], learner.weights)
            evaluation[f"{name}_seed{seed}"] = prediction
            final_rmse = value_rmse(prediction, reference)
            summaries.append(dict(seed=seed, method=name, initial_rmse=zero_rmse,
                                  final_rmse=final_rmse, ratio_to_zero=final_rmse / max(zero_rmse, 1e-12)))
            np.savez(args.output / f"{name}_seed{seed}.npz", weights=learner.weights,
                     kind=learner.features.kind, n=learner.n, alpha=args.alpha,
                     gamma=args.gamma, horizon=args.horizon)
    write_csv(args.output / "learning.csv", metrics)
    write_csv(args.output / "episodes.csv", episodes)
    write_csv(args.output / "summary.csv", summaries)
    np.savez_compressed(args.output / "evaluation.npz", **evaluation)
    tip, tip_v = tip_kinematics(eval_states[:, 0])
    trajectory = [dict(step=t, time=t * CONTROL_DT, q=s[0], qdot=s[1], remaining=s[2],
                       torque=float(policy(s)) if t < args.horizon else 0.0,
                       reward=eval_rewards[t, 0] if t < args.horizon else 0.0,
                       return_to_go=reference[t, 0] if t < args.horizon else 0.0,
                       tip_x=tip[t, 0], tip_z=tip[t, 2], tip_vx=tip_v[t, 0], tip_vz=tip_v[t, 2])
                  for t, s in enumerate(eval_states[:, 0])]
    write_csv(args.output / "trajectory.csv", trajectory)
    config = vars(args).copy()
    config["output"] = str(args.output.resolve())
    config["control_dt"] = CONTROL_DT
    config["versions"] = {p: version(p) for p in ("numpy", "mujoco", "mjlab", "mujoco-warp", "torch")}
    config["evaluation"] = "Independent fixed reset stream; all nonterminal visited states equally weighted"
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Saved CSVs, checkpoints and evaluation trajectories in {args.output}", flush=True)


if __name__ == "__main__":
    main()
