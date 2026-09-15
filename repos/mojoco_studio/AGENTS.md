# Learning-oriented implementation requirements

This project implements the weekly Modern Robotics, MuJoCo, and reinforcement
learning plan in `docs/modern_robotics_mujoco_rl_week_plan.md`.

## Mathematical function documentation

The user requires mathematical documentation in week 6 and **all subsequent
weekly implementations**. Apply this to robotics and RL functions when adding
or materially changing them. Use the explanatory comments in
`src/two_link_kinematics.h` and `src/cartesian_control.h` as style references.

- Put a Doxygen-style `/** ... */` comment immediately above a mathematical C++
  function, or a proper docstring inside a mathematical Python function.
- Explain the function's role: what problem it solves and where it is used in
  the learning exercise.
- State its defining equations in readable mathematical notation and connect
  symbols to code parameters. For nontrivial calculations, explain the main
  derivation or update steps so the source can be used for book review.
- Describe input/output meaning, vector and matrix shapes/storage, coordinate
  frames, reference points, units and conventions where applicable. For RL,
  clarify state/action/reward timing, policy/expectation conventions and which
  estimates an update reads when relevant.
- Document assumptions, rank/convergence conditions and relevant failure cases.
  Distinguish exact identities from numerical approximations and implementation
  choices such as damping, clipping, backtracking or stopping thresholds.
- Reference the relevant book chapter or the user's notes section when verified.
  Identify notes-specific section numbering explicitly; do not invent textbook
  equation numbers or attribute project-specific extensions to the books.
- Scale detail to the mathematics: a norm helper needs a short definition and
  its role; an IK solver, Bellman backup, dynamics calculation or learning update
  needs formulas and enough explanation to reconstruct its reasoning. Avoid
  adding long mathematical boilerplate to ordinary I/O and CLI functions.
- Keep comments synchronized with the actual implementation. A separate weekly
  walkthrough supplements source documentation rather than replacing it.

The user's book notes are available locally at:

- `/home/yuel/workspace/llm-prep-2026/docs/Books/Robotics/Modern_Robotics.md`
- `/home/yuel/workspace/llm-prep-2026/docs/Books/Reinforcement_Learning/Reinforcement_Learning_An_Introduction.md`
