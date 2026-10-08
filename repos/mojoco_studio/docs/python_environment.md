# Python environment for week 10 and later

Run the following from the repository root:

```bash
uv sync --locked
uv run --locked python scripts/check_python_env.py
```

The first command creates or synchronizes `.venv/` from `uv.lock`. The second
prints installed versions, steps the existing pendulum in native MuJoCo, then
steps two copies through mjlab on `cuda:0`. It checks simulation time, finite
states, and agreement of the final CPU/GPU positions and velocities. Initial
Warp kernel compilation can take several minutes; subsequent runs use its cache.
No window, training run, or experiment-tracking login is needed.

For a native MuJoCo check without initializing CUDA:

```bash
uv run --locked python scripts/check_python_env.py --cpu-only
```

This smaller check does not validate GPU support. To explore installed mjlab
tasks or launch its interactive demo:

```bash
uv run --locked python -m mjlab.scripts.list_envs
uv run --locked demo
```

The demo is optional and may download robot assets or checkpoints. The local
pendulum check above uses only this repository's XML asset.
The module form of the task listing exits successfully; mjlab 1.6.0's
`list-envs` console command instead returns the number of tasks as its exit code.

## Files and dependency choices

- `pyproject.toml` declares the Python project and dependencies. It is a script
  project (`package = false`), so the existing C++ `src/` is not packaged.
- `.python-version` selects Python 3.12; `requires-python` keeps resolution in
  that minor version. The Python patch version can differ between machines.
- `uv.lock` records the exact resolved packages and should be committed along
  with the configuration. Use `uv sync --locked` to reproduce them.
- `.venv/` contains local installed packages and is ignored by Git.

We start with mjlab 1.6.0 and compatible MuJoCo 3.11.x. PyTorch is selected from
the official CUDA 12.8 wheel index, and Warp from NVIDIA's index. Torchvision
uses the same CUDA index as PyTorch to keep its compiled operators compatible.
These source
settings live in this project's configuration because a dependency's `tool.uv`
settings are not inherited. This setup targets the current Linux/NVIDIA machine;
CPU-only or macOS installations need a different PyTorch source configuration.

The source requirements are documented in the
[mjlab installation guide](https://mujocolab.github.io/mjlab/v1.6.0/source/installation.html)
and [mjlab 1.6.0 project configuration](https://github.com/mujocolab/mjlab/blob/v1.6.0/pyproject.toml).
Update dependencies deliberately and rerun the smoke test after changing them.
An NVIDIA driver is a system prerequisite, outside the virtual environment;
the installed wheels provide their runtime libraries.

Validated locally on 2026-10-08 with Python 3.12.3, an RTX 4070 SUPER (12 GB),
and NVIDIA driver 580.178.04:

| Package | Locked version |
| --- | --- |
| mjlab | 1.6.0 |
| MuJoCo / MuJoCo Warp | 3.11.0 / 3.11.0 |
| PyTorch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 |
| Warp | 1.18.0 |

The native and two-world GPU pendulum checks passed, as did `uv pip check`
and importing RSL-RL's training runner. Warp reports its own CUDA 13.4 toolkit;
that is independent of PyTorch's CUDA 12.8 runtime. Both executed successfully
with this driver. Interactive rendering and RL training were not tested here.

## Relationship to the C++ exercises

The C++ programs continue to use `build.sh`, `.env`, and the local MuJoCo SDK.
Python imports the MuJoCo wheel inside `.venv/`; it does not use `MUJOCO_DIR`.
You do not need to activate `.venv/` when using `uv run`. For an editor or an
interactive shell, select `.venv/bin/python` or run `source .venv/bin/activate`.

The existing C++ SDK is 3.12.0, while Python follows mjlab's 3.11.x requirement.
Record both versions when comparing experiments. Shared XML models remain
useful, but version-specific features and numerical differences need checking.
The smoke test compares CPU and GPU within the Python stack, not against C++.
It selects the `implicitfast` integrator in memory for both backends; the
original pendulum XML keeps its RK4 setting.

The [week 10 walkthrough](week10_value_approximation.md) now covers value-function
features, semi-gradient TD, training experiments, and their mathematical
docstrings. Its scripts use this same environment, with Matplotlib declared
as a direct dependency for the generated learning figures.
