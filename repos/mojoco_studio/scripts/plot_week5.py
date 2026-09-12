"""Optional plots of the C++ experiments; requires matplotlib, no simulator binding.

Run both executables first, then: python scripts/plot_week5.py
Paths are resolved relative to the repository, regardless of the current directory.
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # also works on a machine without a display
import matplotlib.pyplot as plt


BUILD = Path(__file__).resolve().parents[1] / "build"


def read_csv(name):
    with (BUILD / name).open(newline="") as source:
        return list(csv.DictReader(source))


def main():
    values = read_csv("gridworld_values.csv")
    history = read_csv("gridworld_convergence.csv")
    singular = read_csv("jacobian_singularities.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for ax, key, title in zip(
        axes[0], ("v_random", "v_vi"), ("Uniform random policy", "Optimal values and actions")
    ):
        grid = [[float(values[4 * r + c][key]) for c in range(4)] for r in range(4)]
        ax.imshow(grid, cmap="Blues_r", vmin=-22, vmax=0)
        for s, row in enumerate(values):
            label = f"{float(row[key]):.0f}"
            if key == "v_vi":
                label += "\n" + row["vi_actions"]
            ax.text(s % 4, s // 4, label, ha="center", va="center",
                    color="white" if float(row[key]) < -12 else "black", fontsize=13)
        ax.set(title=title, xticks=range(4), yticks=range(4), xlabel="Column", ylabel="Row")

    ax = axes[1, 0]
    for algorithm, label in (("evaluation", "Random-policy evaluation"),
                             ("policy_iteration", "Policy iteration"),
                             ("value_iteration", "Value iteration")):
        rows = [r for r in history if r["algorithm"] == algorithm]
        # For PI the residual is relative to the policy being evaluated in that
        # round. It can jump when the policy changes; this is not divergence.
        residuals = [max(float(r["bellman_residual"]), 1e-16) for r in rows]
        ax.semilogy(range(1, len(rows) + 1), residuals, label=label)
    ax.set(title="Bellman residual after each sweep", xlabel="Total sweeps",
           ylabel="Residual (zeros shown at 1e-16)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    q2 = [float(r["q2_rad"]) for r in singular]
    for key, label in (("sigma_max_m", "Largest singular value"),
                       ("sigma_min_m", "Smallest singular value")):
        ax.plot(q2, [float(r[key]) for r in singular], label=label)
    ax.axvline(0, color="gray", linestyle=":")
    ax.set(title="Planar tip-position Jacobian, q1 = 0.4 rad", xlabel="Elbow q2 (rad)",
           ylabel="Singular value (m; angles in rad)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.suptitle("Week 5: dynamic programming and velocity kinematics", fontsize=16)
    for suffix in ("svg", "png"):
        target = BUILD / f"week5_overview.{suffix}"
        fig.savefig(target, dpi=160)
        print(target)
    plt.close(fig)


if __name__ == "__main__":
    main()
