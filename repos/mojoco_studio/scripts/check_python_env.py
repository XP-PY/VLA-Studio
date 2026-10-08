"""Check the local Python simulation stack without opening a window or training."""

import argparse
from importlib.metadata import version
from pathlib import Path

import mujoco
import numpy as np


MODEL_PATH = Path(__file__).resolve().parents[1] / "assets" / "pendulum.xml"
STEPS = 100
INITIAL_ANGLE = 0.3


def check_native() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Step the week 1 pendulum from q=0.3 rad, qdot=0 with zero torque.

    q and qdot are one-element arrays in radians and radians/second; q=0 is
    hanging down. Use implicitfast with dt=0.002 s for the GPU comparison,
    overriding RK4 only in memory. After N steps, check t=N*dt and finite,
    nonstationary motion. This checks execution, not an analytical solution.
    """
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    data = mujoco.MjData(model)
    data.qpos[0] = INITIAL_ANGLE
    mujoco.mj_forward(model, data)
    for _ in range(STEPS):
        mujoco.mj_step(model, data)
    np.testing.assert_allclose(data.time, STEPS * model.opt.timestep)
    if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
        raise RuntimeError("Native MuJoCo produced nonfinite state")
    if abs(data.qpos[0] - INITIAL_ANGLE) < 1e-5:
        raise RuntimeError("Native MuJoCo pendulum did not move")
    print(f"PASS: native MuJoCo, {STEPS} steps, t={data.time:.3f} s", flush=True)
    return model, data


def check_mjlab(model: mujoco.MjModel, reference: mujoco.MjData) -> None:
    """Step two GPU pendulums and compare their final state with native MuJoCo.

    Batched qpos/qvel have shape (2, 1), with the same initial state, zero
    torque, timestep and integrator as check_native. Each world is checked
    against the CPU reference using |x_gpu-x_cpu| <= 1e-4 + 1e-4*|x_cpu|.
    This numerical smoke-test tolerance allows float32/float64 differences;
    it is not a guarantee of equivalence for other models or long rollouts.
    """
    import torch
    import warp as wp
    from mjlab.sim import Simulation, SimulationCfg

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA; use --cpu-only for a native check")
    print(f"GPU: {torch.cuda.get_device_name(0)}; PyTorch CUDA {torch.version.cuda}", flush=True)
    wp.init()
    sim = Simulation(num_envs=2, cfg=SimulationCfg(), model=model, device="cuda:0")
    # Graph capture may execute initialization kernels. Reset before measuring.
    sim.reset()
    sim.data.qpos[:, 0] = INITIAL_ANGLE
    sim.data.qvel.zero_()
    sim.data.ctrl.zero_()
    sim.forward()
    for _ in range(STEPS):
        sim.step()
    wp.synchronize()
    torch.cuda.synchronize()
    if not sim.data.qpos.is_cuda:
        raise RuntimeError("mjlab state is not on the GPU")
    for actual, expected in ((sim.data.qpos, reference.qpos), (sim.data.qvel, reference.qvel)):
        values = actual.cpu().numpy()
        if not np.isfinite(values).all():
            raise RuntimeError("mjlab produced nonfinite state")
        np.testing.assert_allclose(values, np.broadcast_to(expected, values.shape), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(sim.data.time.cpu().numpy(), reference.time, rtol=1e-5, atol=1e-6)
    print(f"PASS: mjlab / MuJoCo Warp, 2 CUDA environments, {STEPS} steps each", flush=True)
    print("PASS: GPU positions and velocities agree with the CPU smoke-test reference", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true", help="Check native MuJoCo only; skip CUDA")
    args = parser.parse_args()
    for package in ("mujoco", "mjlab", "mujoco-warp", "warp-lang", "torch"):
        print(f"{package}: {version(package)}", flush=True)
    model, reference = check_native()
    if not args.cpu_only:
        check_mjlab(model, reference)


if __name__ == "__main__":
    main()
