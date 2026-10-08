"""Replay held-out MuJoCo trajectories with learned values and complete returns.

All predictors evaluate the SAME fixed PD policy; switching a predictor changes
the displayed value, not the robot's motion. N/P: episode; M: predictor;
Space: pause; R: replay. Close the window or press Escape to exit.
"""

import argparse
import json
from pathlib import Path
from queue import Empty, SimpleQueue
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np

from week10.pendulum import MODEL_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parents[1] / "build/week10")
    parser.add_argument("--seed", type=int, default=None, help="Training seed; defaults to first saved seed")
    parser.add_argument("--speed", type=float, default=1.)
    parser.add_argument("--seconds", type=float, default=None, help="Optionally close after this wall time")
    args = parser.parse_args()
    if args.speed <= 0 or not np.isfinite(args.speed) or (args.seconds is not None and args.seconds <= 0):
        parser.error("speed and seconds must be positive")
    config = json.loads((args.input / "config.json").read_text())
    seed = config["seeds"][0] if args.seed is None else args.seed
    if seed not in config["seeds"]:
        parser.error(f"Available saved seeds: {config['seeds']}")
    evaluation = np.load(args.input / "evaluation.npz")
    states = evaluation["observations"]
    methods = ["affine_n1", "affine_n8", "rbf_n1", "rbf_n8"]
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    # The old orange reaching target is unrelated to this downward PD task.
    model.site_rgba[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target"), 3] = 0
    data = mujoco.MjData(model)
    keys = SimpleQueue()
    world = step = method = 0
    paused = False
    started = time.monotonic()
    threads_before = set(threading.enumerate())
    with mujoco.viewer.launch_passive(model, data, key_callback=keys.put,
                                      show_left_ui=False, show_right_ui=False) as viewer:
        # MuJoCo 3.11 launches a daemon renderer. Closing requests its exit but
        # does not join it; finish cleanup before Python's GLFW atexit handler.
        render_threads = [t for t in threading.enumerate()
                          if t not in threads_before and t.daemon]
        viewer.cam.lookat[:] = [0, 0, .65]
        viewer.cam.distance = 2.6
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -15
        while viewer.is_running():
            tick = time.monotonic()
            if args.seconds is not None and tick - started >= args.seconds:
                break
            try:
                while True:
                    key = keys.get_nowait()
                    if key == ord(" "):
                        paused = not paused
                    elif key in (ord("N"), ord("P")):
                        world = (world + (1 if key == ord("N") else -1)) % states.shape[1]
                        step = 0
                    elif key == ord("M"):
                        method = (method + 1) % len(methods)
                    elif key == ord("R"):
                        step = 0
            except Empty:
                pass
            terminal = step == len(states) - 1
            reference = 0.0 if terminal else evaluation["returns"][step, world]
            prediction = 0.0 if terminal else evaluation[f"{methods[method]}_seed{seed}"][step, world]
            with viewer.lock():
                data.qpos[0], data.qvel[0] = states[step, world, :2]
                data.time = step * config["control_dt"]
                mujoco.mj_forward(model, data)
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150,
                              mujoco.mjtGridPos.mjGRID_TOPLEFT,
                              "Fixed PD policy\nEpisode / state\nValue predictor\nPredicted V\nComplete return\nControls",
                              f"Recorded {config['backend']} rollout\n{world + 1} / {step}"
                              f"{' (terminal)' if terminal else ''}\n{methods[method]} / seed {seed}"
                              f"\n{prediction:.5f}\n{reference:.5f}\nN/P episode, M method, Space pause, R replay"))
            viewer.sync()
            if not paused:
                step = (step + 1) % len(states)
            time.sleep(max(0, config["control_dt"] / args.speed - (time.monotonic() - tick)))
    for thread in render_threads:
        thread.join()


if __name__ == "__main__":
    main()
