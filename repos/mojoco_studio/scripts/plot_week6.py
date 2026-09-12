"""Verify the custom inverse against NumPy, then plot the default week 6 batch.

Run inverse_kinematics and cartesian_reaching first. Uses the same Matplotlib
environment as week 5. --animate also renders logged joint states as a GIF;
the animation is a planar reconstruction, not an additional dynamics simulation.
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

BUILD = Path(__file__).resolve().parents[1] / "build"
COLORS = {"pinv": "#2372b0", "dls": "#db7d20"}


def read_csv(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def verify_inverse():
    """A library SVD provides an independent check of the hand-coded inverse."""
    maximum = 0.0
    for row in read_csv(BUILD / "inverse_velocity_audit.csv"):
        j = np.array([[float(row[f"j{r}{c}"]) for c in range(2)] for r in range(3)])
        desired = np.array([float(row[f"desired_{axis}"]) for axis in "xyz"])
        damping = float(row["lambda"])
        if damping == 0:
            expected = np.linalg.pinv(j, rcond=1e-8) @ desired
        else:
            expected = np.linalg.solve(j.T @ j + damping**2 * np.eye(2), j.T @ desired)
        actual = np.array([float(row["dq1"]), float(row["dq2"])])
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(
            [float(row["sigma_max"]), float(row["sigma_min"])],
            np.linalg.svd(j, compute_uv=False), rtol=1e-8, atol=1e-12,
        )
        maximum = max(maximum, float(np.max(np.abs(actual - expected))))
    print(f"NumPy inverse/SVD cross-check passed; max absolute dq difference {maximum:.3e}")


class Runs:
    def __init__(self):
        self.rows = read_csv(BUILD / "cartesian_reaching_summary.csv")
        self.index = {(r["case"], r["mode"], r["method"], float(r["gain_s_inv"])): r for r in self.rows}
        self.cache = {}

    def get(self, case, mode="dynamics", method="dls", gain=4):
        key = (case, mode, method, gain)
        if key not in self.index:
            raise RuntimeError("Missing default comparison runs. Rerun ./build/cartesian_reaching without filters.")
        if key not in self.cache:
            path = BUILD / "reaching" / f"{self.index[key]['run']}.csv"
            rows = read_csv(path)
            self.cache[key] = {name: np.array([float(r[name]) for r in rows]) for name in rows[0]}
        return self.cache[key]


def save(fig, name):
    for extension in ("svg", "png"):
        path = BUILD / f"{name}.{extension}"
        fig.savefig(path, dpi=150)
        print(path)
    plt.close(fig)


def overview(runs):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    ax = axes[0, 0]
    for gain in (1, 4, 12):
        run = runs.get("regular", gain=gain)
        ax.semilogy(run["time_s"], run["error_m"], label=f"K={gain}/s")
    ax.axhline(0.001, color="gray", linestyle=":", label="1 mm tolerance")
    ax.set(title="Regular reach: DLS gain comparison", xlabel="Time (s)", ylabel="Position error (m)")

    ax = axes[0, 1]
    for mode, style in (("kinematic", "--"), ("dynamics", "-")):
        for method in COLORS:
            run = runs.get("near_extension", mode=mode, method=method)
            ax.semilogy(run["time_s"], run["error_m"], style, color=COLORS[method], label=f"{method}, {mode}")
    ax.axhline(0.001, color="gray", linestyle=":")
    ax.set(title="Near extension: damping slows the final approach", xlabel="Time (s)", ylabel="Position error (m)")

    ax = axes[0, 2]
    for method in COLORS:
        run = runs.get("near_singular_start", method=method)
        for prefix, style in (("raw", "-"), ("command", "--")):
            peak = np.maximum(np.abs(run[f"{prefix}_dq1"]), np.abs(run[f"{prefix}_dq2"]))
            ax.semilogy(run["time_s"], np.maximum(peak, 1e-12), style, color=COLORS[method], label=f"{method}, {prefix}")
    ax.set(title="Nearly straight start: raw vs bounded rates", xlabel="Time (s)", ylabel="Max joint rate magnitude (rad/s)")

    ax = axes[1, 0]
    run = runs.get("near_joint_limit")
    for i, safe in ((1, 1.48), (2, 1.68)):
        line, = ax.plot(run["time_s"], run[f"q{i}"], label=f"q{i}")
        ax.axhline(safe, color=line.get_color(), linestyle=":", label=f"q{i} guarded upper bound")
    ax.set(title="Joint-limit approach: measured positions", xlabel="Time (s)", ylabel="Joint angle (rad)")

    ax = axes[1, 1]
    run = runs.get("regular")
    for i in (1, 2):
        line, = ax.plot(run["time_s"], run[f"dq{i}_actual"], label=f"q{i} measured")
        ax.plot(run["time_s"], run[f"command_dq{i}"], "--", color=line.get_color(), label=f"q{i} command")
    ax.set(title="Velocity servos have tracking error", xlabel="Time (s)", ylabel="Joint rate (rad/s)")

    ax = axes[1, 2]
    for method in COLORS:
        run = runs.get("near_singular_start", method=method)
        ax.plot(run["tip_base_x"], run["tip_base_y"], color=COLORS[method], label=f"{method}, slightly bent start")
    ax.scatter([0.8], [0], marker="x", color="black", s=65, label="exact straight start: stationary")
    ax.scatter([0.65], [0], marker="*", color="crimson", s=110, label="reachable target")
    ax.set(title="The starting configuration matters", xlabel="Base x (m)", ylabel="Base y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Week 6: Cartesian reaching — K=4/s unless shown otherwise", fontsize=16)
    save(fig, "week6_overview")


def paths(runs):
    cases = ("regular", "near_extension", "near_singular_start", "near_joint_limit")
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    for ax, case in zip(axes.flat, cases):
        for mode, style in (("kinematic", "--"), ("dynamics", "-")):
            for method in COLORS:
                run = runs.get(case, mode=mode, method=method)
                ax.plot(run["tip_base_x"], run["tip_base_y"], style, color=COLORS[method], label=f"{method}, {mode}")
        ax.scatter(run["tip_base_x"][0], run["tip_base_y"][0], color="black", marker="o", label="start")
        ax.scatter(run["target_base_x"][0], run["target_base_y"][0], color="crimson", marker="*", s=100, label="target")
        ax.set(title=case.replace("_", " "), xlabel="Base x (m)", ylabel="Base y (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Tip paths, K=4/s: feedback does not guarantee a straight Cartesian path", fontsize=14)
    save(fig, "week6_paths")


def animate(runs):
    cases = ("regular", "near_extension", "near_singular_start", "exact_singular_start")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    drawings = []
    end = max(runs.get(case, method=method)["time_s"][-1] for case in cases for method in COLORS)
    for ax, case in zip(axes.flat, cases):
        ax.set(title=case.replace("_", " "), xlim=(-0.1, 0.85), ylim=(-0.5, 0.85),
               xlabel="Base x (m)", ylabel="Base y (m)", aspect="equal")
        ax.grid(alpha=0.2)
        for method, color in COLORS.items():
            run = runs.get(case, method=method)
            arm, = ax.plot([], [], "o-", lw=3, color=color, label=method, alpha=0.8)
            trail, = ax.plot([], [], color=color, alpha=0.4)
            drawings.append((run, arm, trail))
        ax.scatter(run["target_base_x"][0], run["target_base_y"][0], marker="*", s=110, color="crimson")
        ax.legend(loc="upper left")
    title = fig.suptitle("Logged MuJoCo dynamics: K=4/s; two methods overlaid")

    def update(time):
        for run, arm, trail in drawings:
            index = min(int(np.searchsorted(run["time_s"], time)), len(run["time_s"])-1)
            q1, q2 = run["q1"][index], run["q2"][index]
            x1, y1 = 0.45*np.cos(q1), 0.45*np.sin(q1)
            arm.set_data([0, x1, x1+0.35*np.cos(q1+q2)], [0, y1, y1+0.35*np.sin(q1+q2)])
            trail.set_data(run["tip_base_x"][:index+1], run["tip_base_y"][:index+1])
        title.set_text(f"Logged MuJoCo dynamics: K=4/s, t={min(time, end):.2f} s (finished runs held)")

    times = np.arange(0, end+0.55, 1/25)
    animation = FuncAnimation(fig, update, frames=times, interval=40)
    target = BUILD / "week6_reaching.gif"
    animation.save(target, writer=PillowWriter(fps=25), dpi=90)
    plt.close(fig)
    print(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--animate", action="store_true")
    args = parser.parse_args()
    verify_inverse()
    runs = Runs()
    overview(runs)
    paths(runs)
    if args.animate:
        animate(runs)


if __name__ == "__main__":
    main()
