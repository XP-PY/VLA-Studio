"""Plot the full week 9 batch, including failed-controller and noisy runs.

Requires NumPy and Matplotlib. Run classical_reaching without filters first.
Produces PNG/SVG figures and a ten-seed noise summary under build/.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BUILD = Path(__file__).resolve().parents[1] / "build"
NOISY = ("noisy_velocity", "noisy_difference", "filtered_difference")


def rows(name):
    with (BUILD / name).open(newline="") as source:
        return list(csv.DictReader(source))


def numeric(name):
    records = rows(name)
    return {key: np.array([float(r[key]) for r in records]) for key in records[0]}


def run(name, seed=0):
    return numeric(f"classical_runs/{name}_seed{seed}.csv")


def save(fig, name):
    for ext in ("png", "svg"):
        path = BUILD / f"{name}.{ext}"
        fig.savefig(path, dpi=150)
        print(path)
    plt.close(fig)


def decorate(axes):
    for ax in np.asarray(axes).flat:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)


def pipeline():
    base = run("baseline")
    plan = rows("classical_reaching_plan.csv")[0]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for j in (1, 2):
        axes[0, 0].plot(base["time"], base[f"q{j}_ref"], label=f"joint {j} desired")
        axes[0, 0].plot(base["time"], base[f"q{j}"], "--", label=f"joint {j} actual")
    axes[0, 0].set(title="Cartesian goal → IK → quintic joint reference", xlabel="Time (s)", ylabel="Joint angle (rad)")
    for name in ("pd_only", "baseline", "inverse_ff_pd", "computed_torque"):
        d = run(name)
        axes[0, 1].semilogy(d["time"], np.maximum(d["error_norm"], 1e-10), label=name)
    axes[0, 1].set(title="Four effort laws; error floor shown at 1e-10", xlabel="Time (s)", ylabel="True joint tracking error (rad)")
    for name in ("baseline", "target_jump"):
        d = run(name)
        axes[1, 0].plot(d["time"], d["tau1"], label=name)
    axes[1, 0].set(title="Smooth target vs setpoint jump", xlabel="Time (s)", ylabel="Shoulder torque (N m)")
    axes[1, 1].plot(base["ref_tip_x"], base["ref_tip_z"], "k--", label="Reference via FK")
    for name in ("baseline", "pd_only", "target_jump"):
        d = run(name)
        axes[1, 1].plot(d["tip_x"], d["tip_z"], label=name)
    axes[1, 1].plot(float(plan["target_x"]), float(plan["target_z"]), "r*", ms=13, label="Requested Cartesian goal")
    axes[1, 1].set(title="Actual end-effector paths", xlabel="World x (m)", ylabel="World z (m)")
    axes[1, 1].set_aspect("equal", adjustable="datalim")
    decorate(axes)
    save(fig, "week9_pipeline")


def gains():
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for col, (parameter, names) in enumerate((("Kp", ("small_kp", "baseline", "large_kp")),
                                              ("Kd", ("small_kd", "baseline", "large_kd")))):
        for name in names:
            d = run(name)
            axes[0, col].plot(d["time"], d["error_norm"], label=name)
            axes[1, col].plot(d["time"], d["tau1"], label=name)
        axes[0, col].set(title=f"Vary {parameter}; other baseline settings fixed", xlabel="Time (s)", ylabel="True joint error (rad)")
        axes[1, col].set(xlabel="Time (s)", ylabel="Shoulder torque (N m)")
    decorate(axes)
    save(fig, "week9_gains")


def limits_and_rates():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for name in ("baseline", "torque_limit_2"):
        d = run(name)
        axes[0, 0].plot(d["time"], d["q1"], label=name)
    axes[0, 0].plot(d["time"], d["q1_ref"], "k--", label="Reference")
    axes[0, 0].set(title="Insufficient torque can lose the initial hold", xlabel="Time (s)", ylabel="Shoulder angle (rad)")
    axes[1, 0].plot(d["time"], d["tau1_raw"], label="Requested (before clipping)")
    axes[1, 0].plot(d["time"], d["tau1"], label="Applied, ±2 N m")
    axes[1, 0].set(title="Requested and applied torque differ", xlabel="Time (s)", ylabel="Shoulder torque (N m)")
    for name in ("baseline", "rate_250hz", "rate_50hz"):
        d = run(name)
        axes[0, 1].semilogy(d["time"], np.maximum(d["error_norm"], 1e-10), label=name)
        mask = (d["time"] >= .8) & (d["time"] <= 1)
        axes[1, 1].step(d["time"][mask], d["tau1"][mask], where="post", label=name)
    axes[0, 1].set(title="Same 1 ms physics; different controller periods", xlabel="Time (s)", ylabel="True joint error (rad)")
    axes[1, 1].set(title="Held torque between controller updates", xlabel="Time (s)", ylabel="Shoulder torque (N m)")
    decorate(axes)
    save(fig, "week9_limits_and_rates")


def noise_statistics(summary):
    """Return mean and sample SD across ten independently seeded sensor runs.

    Each X_i is a whole-run scalar, so mean=sum_i X_i/n and
    SD=sqrt(sum_i (X_i-mean)^2/(n-1)). Error bars show SD, NOT standard error
    or a confidence interval. Same seed/period pairs the three measurement
    variants' position-noise streams; within-run time samples are correlated
    and are not treated as independent replications.
    """
    results = {}
    records = []
    for name in NOISY:
        selected = [r for r in summary if r["case"] == name]
        if len(selected) != 10:
            raise ValueError("Need the full ten-seed noise batch; rerun ./build/classical_reaching without filters")
        for metric in ("rms_error_rad", "tail_rms_rad", "saturation_fraction", "torque_variation_Nm"):
            values = np.array([float(r[metric]) for r in selected])
            mean, sd = values.mean(), values.std(ddof=1)
            results[(name, metric)] = (mean, sd)
            records.append({"case": name, "metric": metric, "mean": mean, "sample_sd": sd, "seeds": len(values)})
    with (BUILD / "week9_noise_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)
    return results


def noise(summary):
    stats = noise_statistics(summary)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for name in NOISY:
        d = run(name)
        mask = (d["time"] >= d["time"][-1]-.5) & (d["time"] <= d["time"][-1]-.3)
        axes[0, 0].plot(d["time"][mask], d["dq1_measured"][mask], label=name)
        axes[0, 1].step(d["time"][mask], d["tau1"][mask], where="post", label=name)
    axes[0, 0].plot(d["time"][mask], d["dq1"][mask], "k--", label="True velocity (filtered run)")
    axes[0, 0].set(title="Seed 0: velocity measurements near final hold", xlabel="Time (s)", ylabel="Shoulder velocity (rad/s)")
    axes[0, 1].set(title="Seed 0: noise propagated into motor torque", xlabel="Time (s)", ylabel="Shoulder torque (N m)")
    for ax, metric, title, unit in ((axes[1, 0], "rms_error_rad", "True tracking error, mean ± SD", "RMS joint error (rad)"),
                                   (axes[1, 1], "torque_variation_Nm", "Torque variation, mean ± SD", "Sum of torque-vector changes (N m)")):
        ax.bar(range(3), [stats[(name, metric)][0] for name in NOISY],
               yerr=[stats[(name, metric)][1] for name in NOISY], capsize=4)
        ax.set(title=title, ylabel=unit, xticks=range(3), xticklabels=[name.replace("_", "\n") for name in NOISY])
        ax.grid(alpha=.2, axis="y")
    decorate(axes[0])
    save(fig, "week9_sensor_noise")


if __name__ == "__main__":
    try:
        summary = rows("classical_reaching_summary.csv")
        if len({r["case"] for r in summary}) != 15:
            raise ValueError("Need all 15 cases; rerun ./build/classical_reaching without filters")
        pipeline()
        gains()
        limits_and_rates()
        noise(summary)
    except FileNotFoundError as error:
        raise SystemExit("Missing CSVs: run ./build/classical_reaching before plotting") from error
