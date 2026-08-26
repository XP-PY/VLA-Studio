# Week 4: Forward Kinematics and a Finite MDP

This exercise combines all Week 4 implementation requirements:

1. Define a small MDP and evaluate a fixed policy with Bellman expectation
   backups.
2. Build a two-link MuJoCo arm.
3. Compute its forward kinematics independently with the space-form Product of
   Exponentials formula.
4. Compare analytical and MuJoCo poses for 100 random configurations.
5. Rank and explain the largest numerical errors.

## Run both implementations

```bash
./build.sh finite_mdp
./build/finite_mdp

./build.sh forward_kinematics
./build/forward_kinematics
```

Optionally open the arm in the interactive viewer:

```bash
./simulate.sh two_link_arm
```

## Part 1: Fixed-policy evaluation in a finite MDP

The five-state line world has terminal states 0 and 4:

```text
terminal 0 ← 1 ↔ 2 ↔ 3 → terminal 4
```

In states 1–3:

- Actions are `LEFT` and `RIGHT`.
- The chosen action succeeds with probability $0.8$.
- With probability $0.2$, the agent stays in the same state.
- Every transition gives reward $-1$.
- The fixed policy selects each action with probability $0.5$.
- The discount is $\gamma=0.9$.

The program implements the Bellman expectation equation directly:

$$
V_\pi(s)=
\sum_a\pi(a\mid s)
\sum_{s',r}p(s',r\mid s,a)
\left[r+\gamma V_\pi(s')\right].
$$

It performs synchronous iterative updates until the maximum value change is
less than $10^{-12}$, then independently evaluates the Bellman residual.

Outputs:

- `build/finite_mdp_convergence.csv`
- `build/finite_mdp_report.txt`

Read `src/finite_mdp.cpp` in this order:

1. `transition_outcomes` defines $p(s',r\mid s,a)$.
2. `make_fixed_policy` defines $\pi(a\mid s)$.
3. `bellman_expectation_backup` implements the equation above.
4. `evaluate_policy` repeatedly applies the Bellman operator.

## Part 2: Two-link MuJoCo model

The model is `assets/two_link_arm.xml` and uses:

- Link lengths $L_1=0.45$ m and $L_2=0.35$ m.
- Two revolute joints about $+z$.
- One end-effector site at the end of link 2.
- A translated and yawed base, so base-frame and world-frame poses differ.

The executable sets `data->qpos` directly and calls `mj_forward`. This is
intentional: the experiment compares kinematics, so it does not need to wait
for an actuator and physical dynamics to reach each configuration.

## Analytical Product of Exponentials

At the home configuration, the end effector is at

$$
M=
\begin{bmatrix}
I & [L_1+L_2,0,0]^T\\
0 & 1
\end{bmatrix}.
$$

The space-frame screw axes are

$$
S_1=
\begin{bmatrix}
0&0&1&0&0&0
\end{bmatrix}^T,
$$

$$
S_2=
\begin{bmatrix}
0&0&1&0&-L_1&0
\end{bmatrix}^T.
$$

For joint 2, $v_2=-\omega_2\times q_2$ with
$q_2=[L_1,0,0]^T$, giving $v_2=[0,-L_1,0]^T$.

The program computes

$$
T_{BE}(q)=e^{[S_1]q_1}e^{[S_2]q_2}M.
$$

It implements each revolute twist exponential using Rodrigues' formula rather
than asking MuJoCo for forward kinematics. A second independent check uses the
planar equations

$$
x=L_1\cos q_1+L_2\cos(q_1+q_2),
$$

$$
y=L_1\sin q_1+L_2\sin(q_1+q_2),
\qquad
\phi=q_1+q_2.
$$

## Comparison procedure

For 100 configurations generated with a fixed random seed:

1. Sample $q_1$ and $q_2$ inside their joint limits.
2. Calculate $T_{BE}$ using PoE.
3. Write the same values to MuJoCo `qpos`.
4. Call `mj_forward`.
5. Read the end-effector pose from `site_xpos` and `site_xmat`.
6. Calculate $T_{BE}^{\text{MuJoCo}}=T_{WB}^{-1}T_{WE}$.
7. Compare position, orientation, every transform entry, and PoE versus the
   elementary trigonometric formula.

Outputs:

- `build/forward_kinematics_comparison.csv`
- `build/forward_kinematics_report.txt`

The report lists the five configurations with the largest transform-entry
errors. Errors near $10^{-16}$ are normal floating-point roundoff caused by
equivalent expressions being evaluated in different orders. Large errors would
usually indicate:

- An incorrect screw axis or home configuration $M$.
- Mixing world and base frames.
- Using the wrong `qpos` address.
- Forgetting `mj_forward` after modifying `qpos`.
- A mismatch between link/site offsets in the XML and analytical model.

Small numerical differences are not caused by kinematic singularities; forward
kinematics remains well-defined at singular configurations.

## Main RL–robotics connection

Forward kinematics provides the task-space feature used in a reaching state:

$$
s_t=
\begin{bmatrix}
q_t\\
\dot q_t\\
x_{target}-x_{ee,t}
\end{bmatrix},
\qquad
x_{ee,t}=f(q_t).
$$

A simple distance reward is

$$
r_t=-\|x_{target}-x_{ee,t}\|_2.
$$

Thus the MDP state representation can contain a feature calculated from the
same forward-kinematics function verified in the MuJoCo experiment.

## Suggested experiments

1. Change either link length in only the XML and observe the resulting error.
2. Then update $L_1$, $L_2$, the screw axes, and $M$ to restore agreement.
3. Increase the random configuration count.
4. Set $q_2=0$ and derive the resulting single straight-link formula.
5. Change the fixed policy or discount in `finite_mdp.cpp` and predict how the
   values change before rerunning it.
