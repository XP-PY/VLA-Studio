"""Plot week 11 learning, robustness, effort, motion, and average-reward results."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from week11.experiment import ALL_METHODS, SCENARIOS


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def save(fig, root, name):
    for ext in ("png", "svg"):
        fig.savefig(root / f"{name}.{ext}", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parents[1] / "build/week11")
    args = parser.parse_args()
    root = args.input
    learning = read_rows(root / "learning.csv")
    evaluation = read_rows(root / "evaluation.csv")
    continuing = read_rows(root / "continuing.csv")
    seeds = sorted({int(r["seed"]) for r in learning})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for name in ALL_METHODS:
        runs = [[r for r in learning if r["method"] == name and int(r["seed"]) == seed] for seed in seeds]
        x = [int(r["episodes"]) for r in runs[0]]
        for ax, metric in zip(axes, ("reward", "success"), strict=True):
            values = np.array([[float(r[metric]) for r in run] for run in runs])
            mean, sd = values.mean(axis=0), values.std(axis=0)
            ax.plot(x, mean, label=name, linestyle="--" if name.startswith("pd") else "-")
            ax.fill_between(x, mean - sd, mean + sd, alpha=.1)
            ax.set(xlabel="Training episodes per method per seed", ylabel=metric)
    axes[0].set_title("Frozen greedy evaluation: mean ± seed SD")
    # Keep poor early torque policies visible without flattening later progress.
    axes[0].set_yscale("symlog", linthresh=.5)
    axes[1].set_title("Final pose/velocity threshold success")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    save(fig, root, "learning")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    metrics = [("reward", "Integrated reward (higher is better)"),
               ("tracking_rmse", "Tracking RMSE (rad)"), ("work_j", "Absolute mechanical work (J)"),
               ("torque_slew", "Average absolute torque slew (N m/s)"),
               ("success", "Final-state success fraction"), ("saturation_fraction", "Saturated time fraction")]
    for ax, (metric, title) in zip(axes.flat, metrics, strict=True):
        table = np.array([[np.mean([float(r[metric]) for r in evaluation
                                   if r["method"] == method and r["scenario"] == scenario])
                           for scenario in SCENARIOS] for method in ALL_METHODS])
        im = ax.imshow(table, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(SCENARIOS)), SCENARIOS, rotation=35, ha="right")
        ax.set_yticks(range(len(ALL_METHODS)), ALL_METHODS)
        ax.set_title(title)
        for row in range(len(ALL_METHODS)):
            for col in range(len(SCENARIOS)):
                ax.text(col, row, f"{table[row, col]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if table[row, col] < (table.min()+table.max())/2 else "black")
        fig.colorbar(im, ax=ax, shrink=.7)
    save(fig, root, "robustness")

    seed = seeds[0]
    replay = np.load(root / f"replay_nominal_seed{seed}.npz")
    states, torques = replay["states"], replay["torques"]
    batch, dt = int(replay["num_envs"]), float(replay["decision_dt"])
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    for i, method in enumerate(replay["methods"]):
        axes[0].plot(np.arange(len(states))*dt, states[:, i*batch, 0], label=str(method))
        axes[1].step(np.arange(len(torques))*dt, torques[:, i*batch], where="post", label=str(method))
    axes[0].axhline(float(replay["goal"]), color="black", linestyle="--", label="goal")
    axes[0].set(ylabel="q (rad)", title="Paired initial state: frozen policies")
    axes[1].set(xlabel="Time (s)", ylabel="Decision-mean motor torque (N m)")
    for ax in axes:
        ax.legend(ncol=3, fontsize=8)
        ax.grid(alpha=.2)
    save(fig, root, "trajectories")

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for seed in seeds:
        rows = [r for r in continuing if int(r["seed"]) == seed]
        seconds = [float(r["seconds"]) for r in rows]
        ax.plot(seconds, [float(r["rate_estimate"]) for r in rows], label=f"Estimated rate, seed {seed}")
        ax.plot(seconds, [float(r["window_reward_rate"]) for r in rows], "--", alpha=.6,
                label=f"Observed window, seed {seed}")
    ax.set(xlabel="Simulated time per world (s)", ylabel="Reward per second",
           title="Differential SARSA: continuing exploratory policy, no episode resets")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, ncol=2)
    save(fig, root, "continuing")
    print(f"Saved learning, robustness, trajectories, continuing PNG/SVG figures in {root}")


if __name__ == "__main__":
    main()
