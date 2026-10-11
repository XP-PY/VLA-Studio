"""MuJoCo Gaussian-policy replay: M method, D sampled/mean, C scenario, N/P world, Space pause, R replay."""

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
from week12.experiment import SCENARIOS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parents[1]/"build/week12")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--speed", type=float, default=1.)
    parser.add_argument("--seconds", type=float, help="Optional automatic close time")
    args = parser.parse_args()
    if (not np.isfinite(args.speed) or args.speed<=0
            or args.seconds is not None and (not np.isfinite(args.seconds) or args.seconds<=0)):
        parser.error("speed and seconds must be finite and positive")
    config = json.loads((args.input/"config.json").read_text())
    seed = config["seeds"][0] if args.seed is None else args.seed
    if seed not in config["seeds"]:
        parser.error(f"Saved seeds: {config['seeds']}")
    policies = ("stochastic", "mean_action")
    replays = {(s,p): np.load(args.input/f"replay_{s}_{p}_seed{seed}.npz") for s in SCENARIOS for p in policies}
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    goal = float(replays[("nominal", "stochastic")]["goal"])
    target = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target")
    model.site_pos[target] = [-.6*np.sin(goal),0,1-.6*np.cos(goal)]
    keys, before = SimpleQueue(), set(threading.enumerate())
    method = scenario = policy = world = step = 0
    paused, render_threads = False, []
    try:
        with mujoco.viewer.launch_passive(model, data, key_callback=keys.put,
                                          show_left_ui=False, show_right_ui=False) as viewer:
            render_threads = [t for t in threading.enumerate() if t not in before and t.daemon]
            viewer.cam.lookat[:] = [0,0,.65]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.6,90,-15
            started = time.monotonic()
            while viewer.is_running():
                tick = time.monotonic()
                if args.seconds is not None and tick-started>=args.seconds:
                    break
                replay = replays[(list(SCENARIOS)[scenario],policies[policy])]
                try:
                    while True:
                        key = keys.get_nowait()
                        if key==ord(" "):
                            paused = not paused
                        elif key==ord("M"):
                            method=(method+1)%len(replay["methods"]); step=0
                        elif key==ord("C"):
                            scenario=(scenario+1)%len(SCENARIOS); step=0
                        elif key==ord("D"):
                            policy=1-policy; step=0
                        elif key in (ord("N"),ord("P")):
                            world=(world+(1 if key==ord("N") else -1))%int(replay["num_envs"]); step=0
                        elif key==ord("R"):
                            step=0
                except Empty:
                    pass
                replay = replays[(list(SCENARIOS)[scenario],policies[policy])]
                index=method*int(replay["num_envs"])+world
                state=replay["states"][step,index]
                dt=float(replay["decision_dt"])
                terminal=step==len(replay["states"])-1
                torque=0. if terminal else float(replay["torques"][step,index])
                distribution="Reference controller" if method==0 else "Terminal"
                if method>0 and not terminal:
                    distribution=f"u={replay['raw'][step,index]:.3f}, mu={replay['mean'][step,index]:.3f}, sigma={replay['sigma'][step,index]:.3f}"
                with viewer.lock():
                    data.qpos[0],data.qvel[0]=state[:2]
                    data.ctrl[0],data.time=torque,step*dt
                    mujoco.mj_forward(model,data)
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150,mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                  "Algorithm\nPolicy / scenario\nWorld / seed\nGaussian (normalized)\nMean motor torque\nAngle / goal\nControls",
                                  f"{replay['methods'][method]}\n{policies[policy]} / {list(SCENARIOS)[scenario]}"
                                  f"\n{world+1} / {seed}\n{distribution}\n{torque:.3f} N m\n{state[0]:.3f} / {goal:.3f} rad"
                                  "\nM method, D sampled/mean, C scenario, N/P world, Space pause, R replay"))
                viewer.sync()
                if not paused:
                    step=(step+1)%len(replay["states"])
                time.sleep(max(0,dt/args.speed-(time.monotonic()-tick)))
    finally:
        for thread in render_threads:
            thread.join()


if __name__=="__main__":
    main()
