"""Plot the checked week 7 dynamics and Monte Carlo CSVs without a display.

Run robot_dynamics and blackjack_mc first. Requires NumPy and Matplotlib.
All figures are saved as PNG and SVG under build/; no simulation or learning
is performed by this script.
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

BUILD = Path(__file__).resolve().parents[1] / "build"


def rows(name):
    with (BUILD / name).open(newline="") as source:
        return list(csv.DictReader(source))


def numbers(name):
    records = rows(name)
    return {key: np.array([float(row[key]) for row in records]) for key in records[0]}


def save(fig, name):
    for extension in ("png", "svg"):
        path = BUILD / f"{name}.{extension}"
        fig.savefig(path, dpi=150)
        print(path)
    plt.close(fig)


def dynamics_plots():
    modes = ["no_control", "joint_damping", "constant_torque", "gravity_compensation",
             "position_actuator", "torque_pd", "torque_pd_gravity", "gravity_moving"]
    runs = {mode: numbers(f"dynamics/{mode}.csv") for mode in modes}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for mode in ("no_control", "joint_damping"):
        run = runs[mode]
        axes[0, 0].plot(run["time"], run["energy"], label=mode)
    axes[0, 0].set(title="Damping dissipates mechanical energy", ylabel="K + P (J)")
    for mode in ("gravity_compensation", "gravity_moving"):
        run = runs[mode]
        axes[0, 1].plot(run["time"], run["q1"], label=f"{mode}: shoulder")
    axes[0, 1].set(title="Gravity balance does not stop motion", ylabel="Shoulder angle (rad)")
    for mode, style in (("position_actuator", "-"), ("torque_pd", "--"), ("torque_pd_gravity", "-")):
        run = runs[mode]
        axes[0, 2].semilogy(run["time"], np.maximum(run["target_error"], 1e-12), style, label=mode)
    axes[0, 2].set(title="Matched PD laws; adding gravity removes offset", ylabel="Joint error norm (rad)")
    for mode, style in (("position_actuator", "-"), ("torque_pd", "--"), ("torque_pd_gravity", "-")):
        run = runs[mode]
        axes[1, 0].plot(run["time"], run["tau1"], style, label=mode)
    axes[1, 0].set(title="Actual shoulder actuator torque", ylabel="Torque (N m)")
    run = runs["constant_torque"]
    for joint in (1, 2):
        axes[1, 1].plot(run["time"], run[f"q{joint}"], label=f"joint {joint}")
    axes[1, 1].set(title="Constant torque, unlimited hinges", ylabel="Unwrapped angle (rad)")
    run = runs["no_control"]
    axes[1, 2].plot(run["time"], run["energy"] - run["energy"][0], label="RK4 energy error")
    axes[1, 2].set(title="Conservation is numerical, not exact", ylabel="E(t) - E(0) (J)")
    for ax in axes.flat:
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    save(fig, "week7_dynamics")


def blackjack_tables():
    records = rows("blackjack_values.csv")
    fig, axes = plt.subplots(2, 5, figsize=(18, 8), constrained_layout=True)
    fields = [("first_value", "MC stick-on-20 value"),
              ("reference_value", "Reference stick-on-20 value"),
              ("optimal_action", "Reference optimal policy"),
              ("es_action", "MC exploring-starts policy"),
              ("soft_greedy_action", "MC epsilon-soft: greedy policy")]
    for usable in (0, 1):
        for column, (key, title) in enumerate(fields):
            grid = np.full((10, 10), np.nan)
            for row in records:
                if int(row["usable_ace"]) == usable:
                    grid[int(row["sum"]) - 12, int(row["dealer"]) - 1] = float(row[key])
            ax = axes[usable, column]
            value_plot = column < 2
            picture = ax.imshow(grid, origin="lower", extent=(.5, 10.5, 11.5, 21.5),
                                aspect="auto", vmin=-1 if value_plot else 0, vmax=1,
                                cmap="coolwarm" if value_plot else ListedColormap(["#dfb66c", "#558bb7"]))
            ax.set(title=title, xlabel="Dealer upcard (1 = ace)",
                   ylabel=("Usable ace" if usable else "No usable ace") + ": player sum",
                   xticks=[1, 2, 4, 6, 8, 10], yticks=range(12, 22))
            if column in (1, 4):
                bar = fig.colorbar(picture, ax=ax, shrink=.7)
                if not value_plot:
                    bar.set_ticks([0, 1], labels=["Stick", "Hit"])
    save(fig, "week7_blackjack_tables")


def learning_plots():
    records = rows("blackjack_learning.csv")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for method in ("prediction", "exploring_starts", "epsilon_soft"):
        selected = [r for r in records if r["method"] == method]
        field = "prediction_rmse" if method == "prediction" else "greedy_policy_gap"
        ax = axes[0] if method == "prediction" else axes[1]
        ax.loglog([float(r["episodes"]) for r in selected],
                  [float(r[field]) for r in selected], "o-", label=method)
    axes[0].set(title="Prediction vs independent reference", ylabel="Equal-state RMSE")
    axes[1].set(title="Extracted greedy policy quality", ylabel="Mean state-value gap to optimal")
    importance = numbers("blackjack_importance.csv")
    for method in ("ordinary", "weighted"):
        axes[2].loglog(importance["episodes"], importance[f"{method}_mse"], "o-", label=method)
    axes[2].set(title="Importance sampling: 50 independent runs", ylabel="MSE at soft 13 vs dealer 2")
    for ax in axes:
        ax.set_xlabel("Episodes")
        ax.grid(alpha=.25)
        ax.legend()
    save(fig, "week7_mc_learning")


if __name__ == "__main__":
    try:
        dynamics_plots()
        blackjack_tables()
        learning_plots()
    except FileNotFoundError as error:
        raise SystemExit("Missing week 7 CSVs. Run ./build/robot_dynamics and ./build/blackjack_mc first.") from error
