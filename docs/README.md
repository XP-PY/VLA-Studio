# VLA Training Logs

This directory collects training notes, evaluation results, and lessons learned while building VLA policies for the RM65 robotic arm.

## Logs

- [ACT Training Log](act.md): dataset recording, hyperparameter experiments, rollout evaluation, failure analysis, and the final conclusion for training `ACT` on the RM65 tasks.

- [SmolVLA Training Log](smolvla.md): dataset recording, hyperparameter experiments, rollout evaluation, failure analysis, and the final conclusion for training `SmolVLA` on the RM65 tasks.

- [Action Flow Training Log](action_flow.md): Policy structure, dataset recording, hyperparameter experiments, rollout evaluation, failure analysis, and the final conclusion for training `Action Flow` on the RM65 tasks.

- [$\pi_0$ and $\pi_0$-FAST Training Log](pi_0.md): dataset design, fine-tuning configuration,
  rollout evaluation, and failure analysis for the RM65 table tasks.

- [Policy Evaluation](evaluation.md): paired comparison of the SmolVLA baseline, Action Flow V2.3,
  and Action Flow V2.4.

- [Project Structure](project_structure.md): source layout, generated-data locations, checkpoint
  retention, and maintenance commands.

## Notes

The logs are written as experiment records. They include both successful and failed runs, because the failure categories are useful for deciding the next data collection or training change.
