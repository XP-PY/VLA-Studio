"""Replay learned robot controllers in MuJoCo: M method, C scenario, N/P world, Space pause, R restart."""

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
from week11.experiment import SCENARIOS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parents[1] / "build/week11")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--speed", type=float, default=1.)
    parser.add_argument("--seconds", type=float, help="Close automatically after this wall time")
    args = parser.parse_args()
    if (not np.isfinite(args.speed) or args.speed <= 0
            or args.seconds is not None and (not np.isfinite(args.seconds) or args.seconds <= 0)):
        parser.error("speed and seconds must be finite and positive")
    config = json.loads((args.input / "config.json").read_text())
    seed = config["seeds"][0] if args.seed is None else args.seed
    if seed not in config["seeds"]:
        parser.error(f"Saved seeds are {config['seeds']}")
    replays = {name: np.load(args.input / f"replay_{name}_seed{seed}.npz") for name in SCENARIOS}
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    goal = float(replays["nominal"]["goal"])
    target = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target")
    model.site_pos[target] = [-.6*np.sin(goal), 0., 1-.6*np.cos(goal)]
    keys, old_threads = SimpleQueue(), set(threading.enumerate())
    method = scenario = world = step = 0
    paused = False
    render_threads = []
    try:
        with mujoco.viewer.launch_passive(model, data, key_callback=keys.put,
                                          show_left_ui=False, show_right_ui=False) as viewer:
            # Join the pinned MuJoCo version's daemon renderer before GLFW atexit.
            render_threads = [t for t in threading.enumerate() if t not in old_threads and t.daemon]
            viewer.cam.lookat[:] = [0, 0, .65]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.6, 90, -15
            started = time.monotonic()
            while viewer.is_running():
                tick = time.monotonic()
                if args.seconds is not None and tick - started >= args.seconds:
                    break
                replay = replays[list(SCENARIOS)[scenario]]
                try:
                    while True:
                        key = keys.get_nowait()
                        if key == ord(" "):
                            paused = not paused
                        elif key == ord("M"):
                            method = (method + 1) % len(replay["methods"])
                            step = 0
                        elif key == ord("C"):
                            scenario = (scenario + 1) % len(SCENARIOS)
                            step = 0
                        elif key in (ord("N"), ord("P")):
                            world = (world + (1 if key == ord("N") else -1)) % int(replay["num_envs"])
                            step = 0
                        elif key == ord("R"):
                            step = 0
                except Empty:
                    pass
                replay = replays[list(SCENARIOS)[scenario]]
                index = method * int(replay["num_envs"]) + world
                state = replay["states"][step, index]
                dt = float(replay["decision_dt"])
                terminal = step == len(replay["states"]) - 1
                torque = 0. if terminal else float(replay["torques"][step, index])
                with viewer.lock():
                    data.qpos[0], data.qvel[0] = state[:2]
                    data.ctrl[0], data.time = torque, step * dt
                    mujoco.mj_forward(model, data)
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                  "Week 11 controller\nScenario / rate\nPaired world / seed\nAngle / target\nMean motor torque\nControls",
                                  f"{replay['methods'][method]}\n{list(SCENARIOS)[scenario]} / {1/dt:g} Hz"
                                  f"\n{world+1} / {seed}\n{state[0]:.3f} / {goal:.3f} rad\n{torque:.3f} N m"
                                  "\nM method, C scenario, N/P world, Space pause, R restart"))
                viewer.sync()
                if not paused:
                    step = (step + 1) % len(replay["states"])
                time.sleep(max(0, dt / args.speed - (time.monotonic() - tick)))
    finally:
        for thread in render_threads:
            thread.join()


if __name__ == "__main__":
    main()
