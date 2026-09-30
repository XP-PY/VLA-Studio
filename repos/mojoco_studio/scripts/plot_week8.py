"""Plot week 8 reference/actual trajectories and seeded Cliff Walking results.

Run trajectory_tracking and cliff_walking first. NumPy and Matplotlib suffice;
the figures are standalone PNG/SVG artifacts and do not run new experiments.
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

BUILD = Path(__file__).resolve().parents[1] / "build"
COLORS = {"cubic": "#2677b6", "quintic": "#d17a20", "sarsa": "#2677b6",
          "q_learning": "#d95336", "expected_sarsa": "#278750"}
METHODS = ("sarsa", "q_learning", "expected_sarsa")


def rows(filename):
    with (BUILD / filename).open(newline="") as source:
        return list(csv.DictReader(source))


def numeric(filename):
    data = rows(filename)
    return {key: np.array([float(row[key]) for row in data]) for key in data[0]}


def save(fig, name):
    for ext in ("png", "svg"):
        path = BUILD / f"{name}.{ext}"
        fig.savefig(path, dpi=150)
        print(path)
    plt.close(fig)


def trajectories():
    data = {p: numeric(f"trajectories/{p}.csv") for p in ("cubic", "quintic")}
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True, constrained_layout=True)
    for row, (prefix, label) in enumerate((("q", "Position (rad)"), ("dq", "Velocity (rad/s)"),
                                           ("ddq", "Acceleration (rad/s²)"))):
        for col in range(2):
            ax = axes[row, col]
            for profile, d in data.items():
                ax.plot(d["time"], d[f"{prefix}{col+1}_ref"], color=COLORS[profile], label=f"{profile} reference")
                ax.plot(d["time"], d[f"{prefix}{col+1}"], "--", color=COLORS[profile], label=f"{profile} actual")
            ax.axvspan(0, .5, color="gray", alpha=.1)
            ax.axvspan(2.5, 3.5, color="gray", alpha=.1)
            ax.set(ylabel=label, title=f"Joint {col+1}")
            ax.grid(alpha=.2)
            if row == 0:
                ax.legend(fontsize=8)
            if row == 2:
                ax.set_xlabel("Time (s); shaded = stationary reference")
    save(fig, "week8_trajectories")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for profile, d in data.items():
        axes[0].plot(d["time"], d["error_norm"], color=COLORS[profile], label=profile)
        for joint, style in ((1, "-"), (2, "--")):
            axes[1].plot(d["time"], d[f"tau{joint}"], style, color=COLORS[profile], label=f"{profile}, joint {joint}")
        axes[2].plot(d["tip_x"], d["tip_z"], color=COLORS[profile], label=f"{profile} actual")
    d = data["cubic"]
    axes[2].plot(d["ref_tip_x"], d["ref_tip_z"], "k--", label="FK of joint reference")
    axes[2].plot(d["line_x"], d["line_z"], ":", color="gray", label="Cartesian line (not executed)")
    axes[0].set(title="Tracking error, same controller", xlabel="Time (s)", ylabel="Joint error norm (rad)")
    axes[1].set(title="Actual motor torque", xlabel="Time (s)", ylabel="Torque (N m)")
    axes[2].set(title="Geometry differs from time scaling", xlabel="World x (m)", ylabel="World z (m)")
    axes[2].set_aspect("equal", adjustable="datalim")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    save(fig, "week8_tracking")


def block_statistics(records, method, field, block=20):
    """Average consecutive episode blocks WITHIN seeds before across-seed SE.

    X[k,b] is seed k's mean metric in block b (20 episodes). Return block-end
    indices, mean_k X, and std_k(X, ddof=1)/sqrt(number of seeds). This preserves
    within-run dependence instead of treating adjacent episodes as independent
    confidence samples. Inputs must contain equally sized complete runs.
    """
    selected = [row for row in records if row["method"] == method]
    seeds = sorted({int(row["seed"]) for row in selected})
    per_seed = []
    for seed in seeds:
        run = sorted((row for row in selected if int(row["seed"]) == seed), key=lambda r: int(r["episode"]))
        values = np.array([float(row[field]) for row in run])
        if len(values) % block:
            raise ValueError("Episode count must be divisible by block size")
        per_seed.append(values.reshape(-1, block).mean(axis=1))
    values = np.array(per_seed)
    return block * np.arange(1, values.shape[1] + 1), values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(len(seeds))


def cliff_plots():
    training = rows("cliff_training.csv")
    summary = {(r["method"], r["metric"]): r for r in rows("cliff_summary.csv")}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for method in METHODS:
        x, mean, se = block_statistics(training, method, "return")
        axes[0, 0].plot(x, mean, color=COLORS[method], label=method)
        axes[0, 0].fill_between(x, mean-1.96*se, mean+1.96*se, color=COLORS[method], alpha=.15)
        x, mean, se = block_statistics(training, method, "falls")
        axes[0, 1].plot(x, mean, color=COLORS[method], label=method)
    axes[0, 0].set(title="Online learning: 30 seeds, 20-episode blocks", xlabel="Episode", ylabel="Mean episode return (shading ≈95% interval)")
    axes[0, 1].set(title="Cliff falls during learning", xlabel="Episode", ylabel="Mean falls per episode")
    x = np.arange(3)
    for offset, metric, label in ((-.2, "frozen_greedy_completed_return", "Greedy: completed games only"),
                                  (.2, "frozen_epsilon_completed_return", "Exploratory: completed games only")):
        values = [float(summary[(m, metric)]["mean"]) for m in METHODS]
        errors = [1.96*float(summary[(m, metric)]["standard_error"]) for m in METHODS]
        axes[1, 0].bar(x+offset, values, width=.38, yerr=errors, capsize=3, label=label)
    axes[1, 0].set(title="Frozen policies on fresh episodes (conditional returns)", ylabel="Mean completed return", xticks=x, xticklabels=METHODS)
    for offset, metric, label in ((-.2, "frozen_greedy_completion_rate", "Greedy"),
                                  (.2, "frozen_epsilon_completion_rate", "Exploratory")):
        values = [100*float(summary[(m, metric)]["mean"]) for m in METHODS]
        axes[1, 1].bar(x+offset, values, width=.38, label=label)
    axes[1, 1].set(title="Completion within 1000 evaluation steps", ylabel="Completed (%)", ylim=(0, 110), xticks=x, xticklabels=METHODS)
    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2, axis="y")
    save(fig, "week8_cliff_learning")

    paths = rows("cliff_paths.csv")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    for ax, method in zip(axes, METHODS):
        for row in range(4):
            for col in range(12):
                color = "#efb5a8" if row == 3 and 0 < col < 11 else "#f4f5f6"
                ax.add_patch(Rectangle((col-.5, row-.5), 1, 1, facecolor=color, edgecolor="#b2b7ba"))
        path = [r for r in paths if r["method"] == method]
        ax.plot([int(r["column"]) for r in path], [int(r["row"]) for r in path], "o-", color=COLORS[method])
        ax.text(0, 3, "S", ha="center", va="center", weight="bold")
        ax.text(11, 3, "G", ha="center", va="center", weight="bold")
        ax.text(5.5, 3, "CLIFF: -100 and return to S; episode continues", ha="center", va="center", fontsize=9)
        ax.set(xlim=(-.5, 11.5), ylim=(3.5, -.5), xticks=range(12), yticks=range(4),
               title=f"{method}: one successful greedy route (seed {path[0]['seed']}, {len(path)-1} steps)")
        ax.set_aspect("equal")
    save(fig, "week8_cliff_paths")

    curve = numeric("cliff_prediction_learning.csv")
    prediction = rows("cliff_prediction.csv")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    axes[0].loglog(curve["episodes"], curve["rmse"], "o-")
    axes[0].set(title="TD(0) under fixed policy", xlabel="Episodes", ylabel="Equal-state value RMSE")
    values = {}
    for field in ("td_value", "reference_value"):
        grid = np.full((4, 12), np.nan)
        for r in prediction:
            if r["active"] == "1":
                grid[int(r["row"]), int(r["column"])] = float(r[field])
        values[field] = grid
    low = min(np.nanmin(v) for v in values.values())
    for ax, field in zip(axes[1:], values):
        picture = ax.imshow(values[field], vmin=low, vmax=0, cmap="viridis")
        ax.set(title=field.replace("_", " "), xlabel="Column", ylabel="Row")
        fig.colorbar(picture, ax=ax, shrink=.65, label="Expected return")
    save(fig, "week8_td_prediction")


if __name__ == "__main__":
    try:
        trajectories()
        cliff_plots()
    except FileNotFoundError as error:
        raise SystemExit("Run ./build/trajectory_tracking and ./build/cliff_walking before plotting.") from error
