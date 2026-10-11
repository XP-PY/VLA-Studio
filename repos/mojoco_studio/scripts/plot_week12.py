"""Plot Gaussian-policy learning, paired baseline variance, and robot rollouts."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from week12.policy_gradient import METHODS
from week12.experiment import EVAL_METHODS


def read(path):
    with path.open() as source:
        return list(csv.DictReader(source))


def save(fig, root, name):
    for suffix in ("png", "svg"):
        fig.savefig(root/f"{name}.{suffix}", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parents[1]/"build/week12")
    args = parser.parse_args()
    root = args.input
    learning, diagnostics = read(root/"learning.csv"), read(root/"diagnostics.csv")
    variance, continuing = read(root/"variance.csv"), read(root/"continuing.csv")
    seeds = sorted({int(r["seed"]) for r in learning})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for ax, policy in zip(axes, ("stochastic", "mean_action"), strict=True):
        for name in EVAL_METHODS:
            runs = [[r for r in learning if r["method"]==name and r["policy"]==policy and int(r["seed"])==seed]
                    for seed in seeds]
            x = [int(r["episodes"]) for r in runs[0]]
            values = np.array([[float(r["discounted_return"]) for r in run] for run in runs])
            ax.plot(x, values.mean(axis=0), label=name, linestyle="--" if name=="pd_gravity" else "-")
            ax.fill_between(x, values.mean(axis=0)-values.std(axis=0), values.mean(axis=0)+values.std(axis=0), alpha=.15)
        ax.set(xlabel="Training episodes per method per seed", ylabel="Held-out discounted return",
               title="Sampled Gaussian policy" if policy=="stochastic" else "Deterministic mean action (different controller)")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Mean ± population SD across seeds; higher return is better")
    save(fig, root, "learning")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for name in METHODS:
        runs = [[r for r in diagnostics if r["method"]==name and int(r["seed"])==seed] for seed in seeds]
        x = [int(r["episodes"]) for r in runs[0]]
        for ax, metric in zip(axes[:2], ("sigma", "sample_clip_fraction"), strict=True):
            values = np.array([[float(r[metric]) for r in run] for run in runs])
            ax.plot(x, values.mean(axis=0), label=name)
            ax.set(xlabel="Training episodes", ylabel=metric)
    axes[0].set_title("Learned Gaussian spread (normalized torque)")
    axes[1].set_title("Fraction of raw samples beyond motor limits")
    runs = [[r for r in variance if int(r["seed"])==seed] for seed in seeds]
    for key, label in (("gradient_variance_plain", "G * score"),
                       ("gradient_variance_baseline", "(G - frozen V) * score")):
        y = np.array([[float(r[key]) for r in run] for run in runs])
        axes[2].semilogy([int(r["episodes"]) for r in runs[0]], np.maximum(y.mean(axis=0), 1e-12), label=label)
    axes[2].set(xlabel="Baseline learner's training episodes", ylabel="Trace of sample gradient covariance",
                title="Paired estimates on the SAME sampled episodes")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    save(fig, root, "diagnostics")

    seed = seeds[0]
    replay = np.load(root/f"replay_nominal_stochastic_seed{seed}.npz")
    states, raw = replay["states"], replay["raw"]
    batch, dt = int(replay["num_envs"]), float(replay["decision_dt"])
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for i, name in enumerate(replay["methods"]):
        index = i*batch
        axes[0,0].plot(np.arange(len(states))*dt, states[:,index,0], label=str(name))
        axes[0,1].step(np.arange(len(raw))*dt, replay["torques"][:,index], where="post", label=str(name))
    axes[0,0].axhline(float(replay["goal"]), color="black", linestyle="--", label="target")
    axes[0,0].set(ylabel="q (rad)", title="Paired initial state, stochastic policies")
    axes[0,1].set(ylabel="Decision-mean torque (N m)", title="Motor commands, including reference PD")
    index = 3*batch  # actor-critic's first held-out world
    times = np.arange(len(raw))*dt
    axes[1,0].plot(times, raw[:,index], ".", label="Raw Gaussian u")
    axes[1,0].plot(times, replay["mean"][:,index], label="mu")
    axes[1,0].fill_between(times, replay["mean"][:,index]-replay["sigma"][:,index],
                          replay["mean"][:,index]+replay["sigma"][:,index], alpha=.2, label="mu ± sigma")
    axes[1,0].axhline(1, color="black", linestyle="--")
    axes[1,0].axhline(-1, color="black", linestyle="--")
    axes[1,0].set(ylabel="Normalized torque action", title="Actor-critic: sample before motor clipping")
    rows = read(root/"evaluation.csv")
    pos = np.arange(len(EVAL_METHODS))
    for offset, policy in ((-.18, "stochastic"), (.18, "mean_action")):
        values = [np.mean([float(r["success"]) for r in rows if r["method"]==name
                           and r["scenario"]=="nominal" and r["policy"]==policy]) for name in EVAL_METHODS]
        axes[1,1].bar(pos+offset, values, width=.36, label=policy)
    axes[1,1].set_xticks(pos, EVAL_METHODS, rotation=20, ha="right")
    axes[1,1].set(ylabel="Final pose/velocity success fraction", ylim=(0,1.05), title="Nominal evaluation, all seeds")
    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    for ax in (axes[0,0], axes[0,1], axes[1,0]):
        ax.set_xlabel("Time (s)")
    save(fig, root, "trajectories")

    fig, ax = plt.subplots(figsize=(11,4), constrained_layout=True)
    for seed in seeds:
        rows = [r for r in continuing if int(r["seed"])==seed]
        seconds = [float(r["seconds"]) for r in rows]
        ax.plot(seconds, [float(r["rate_estimate"]) for r in rows], label=f"Estimated rate, seed {seed}")
        ax.plot(seconds, [float(r["window_reward_rate"]) for r in rows], "--", alpha=.6,
                label=f"Observed window, seed {seed}")
    ax.set(xlabel="Simulated time per world (s)", ylabel="Reward per second",
           title="Continuing actor-critic: no discount factor or episode resets")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=.2)
    save(fig, root, "continuing")
    print(f"Saved learning, diagnostics, trajectories and continuing PNG/SVG figures in {root}")


if __name__=="__main__":
    main()
