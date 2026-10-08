"""Plot held-out value errors and pendulum motion from week10_value_prediction.py."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from week10.prediction import Features, generalization_response


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parents[1] / "build/week10")
    args = parser.parse_args()
    with (args.input / "learning.csv").open() as stream:
        records = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for method in sorted({r["method"] for r in records}):
        runs = [[r for r in records if r["method"] == method and r["seed"] == seed]
                for seed in sorted({r["seed"] for r in records})]
        x = [int(r["episodes"]) for r in runs[0]]
        y = np.array([[float(r["value_rmse"]) for r in run] for run in runs])
        mean, spread = y.mean(axis=0), y.std(axis=0)
        axes[0].plot(x, mean, label=method)
        axes[0].fill_between(x, mean - spread, mean + spread, alpha=0.15)
    axes[0].set(xlabel="Training episodes per seed", ylabel="Held-out value RMSE",
                title="Mean ± population SD across seeds (not a confidence interval)")
    evaluation = np.load(args.input / "evaluation.npz")
    seed = min(int(r["seed"]) for r in records)
    reference = evaluation["returns"].reshape(-1)
    for method in ("affine_n8", "rbf_n8"):
        predicted = evaluation[f"{method}_seed{seed}"].reshape(-1)
        axes[1].scatter(reference[::5], predicted[::5], s=6, alpha=0.35, label=method)
    axes[1].plot([reference.min(), reference.max()], [reference.min(), reference.max()], "k--")
    axes[1].set(xlabel="Complete-return reference", ylabel="Predicted value",
                title=f"Held-out states; seed {seed}")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend()
    for suffix in ("png", "svg"):
        fig.savefig(args.input / f"learning.{suffix}", dpi=150)
    plt.close(fig)
    data = np.genfromtxt(args.input / "trajectory.csv", delimiter=",", names=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes[0, 0].plot(data["time"], data["q"], label="q (rad)")
    axes[0, 0].plot(data["time"], data["qdot"], label="qdot (rad/s)")
    axes[0, 1].step(data["time"][:-1], data["torque"][:-1], where="post", label="Torque (N m)")
    axes[1, 0].plot(data["time"][:-1], data["return_to_go"][:-1], label="Complete return")
    for method in ("affine_n1", "affine_n8", "rbf_n1", "rbf_n8"):
        axes[1, 0].plot(data["time"][:-1], evaluation[f"{method}_seed{seed}"][:, 0], label=method)
    axes[1, 1].plot(data["tip_x"], data["tip_z"], label="World tip path (m)")
    axes[1, 1].set(xlabel="World x (m)", ylabel="World z (m)", aspect="equal")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xlabel("Time (s)")
    for suffix in ("png", "svg"):
        fig.savefig(args.input / f"trajectory.{suffix}", dpi=150)
    plt.close(fig)
    q, v = np.meshgrid(np.linspace(-1.5, 1.5, 81), np.linspace(-4, 4, 81))
    queries = np.stack((q, v, np.full_like(q, .5)), axis=-1)
    anchor = np.array([.6, .4, .5])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, kind in zip(axes, ("affine", "rbf"), strict=True):
        response = generalization_response(Features(kind), anchor, queries)
        heatmap = ax.pcolormesh(q, v, response, shading="auto", vmin=0, vmax=1)
        ax.plot(anchor[0], anchor[1], "rx", label="Updated state")
        ax.set(xlabel="q (rad)", ylabel="qdot (rad/s)", title=f"{kind}: feature overlap at h=0.5")
        ax.legend()
        fig.colorbar(heatmap, ax=ax, label="Value change for alpha × TD error = 1")
    for suffix in ("png", "svg"):
        fig.savefig(args.input / f"generalization.{suffix}", dpi=150)
    plt.close(fig)
    print(f"Saved learning/trajectory/generalization PNG and SVG figures in {args.input}")


if __name__ == "__main__":
    main()
