# Example and Inspection Tools

The files in this directory are reusable programs, not generated experiment output.

## Main Groups

| Files | Role |
|---|---|
| `camera_connect.py`, `test_rm65_config.py` | Camera and RM65 hardware smoke tests. |
| `inspect_train_batch.py` | Inspect the batch produced by the LeRobot training data path. |
| `inspect_smolvla_*`, `inspect_smolvlm_*` | Inspect SmolVLA preprocessing and model tensors. |
| `inspect_pi0_fast_*` | Inspect $\pi_0$-FAST preprocessing, tokens, and tensor shapes. |
| `evaluate_pi0_prompts.py` | Interactive multi-prompt rollout evaluation. |
| `action_flow_v2/` | Action Flow annotation, dataset construction, review, and evaluation tools. |

Generated logs belong under `logs`; generated images belong under `runs/inspection`. Root-level
`*.log`, `*.png`, and `*.jpg` files in `examples` are ignored to prevent inspection output from
being committed accidentally. Reference assets required by Action Flow remain under
`examples/action_flow_v2/ref`.
