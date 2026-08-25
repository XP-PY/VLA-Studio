# Week 3: MuJoCo Frames and Homogeneous Transformations

This exercise combines every item from the Week 3 **MuJoCo** and
**Deliverable** sections. It reads body and site poses from MuJoCo, converts
them to homogeneous transformations, and applies reusable $SO(3)$ and $SE(3)$
utilities.

The notation follows *Modern Robotics* Chapter 3:

$$
T_{AB}=
\begin{bmatrix}
R_{AB} & p_{AB}\\
0 & 1
\end{bmatrix}
$$

represents frame $\{B\}$ relative to frame $\{A\}$ and maps point coordinates
from $\{B\}$ to $\{A\}$.

## Run the exercise

From the repository root:

```bash
./build.sh frame_transforms
./build/frame_transforms
```

Open the model interactively with:

```bash
./simulate.sh frame_transforms
```

Generated artifacts:

- `build/frame_transforms.csv`: all poses and numerical errors at 50 Hz.
- `build/frame_transforms_report.txt`: final transform matrices and maximum
  validation errors.

## Files

| File | Purpose |
|---|---|
| `assets/frame_transforms.xml` | Base, moving link, oriented end-effector site, and actuator |
| `src/transform_utils.h` | Reusable Week 3 transformation utilities |
| `src/frame_transforms.cpp` | MuJoCo connection and numerical validation |

## Frames in the model

The model intentionally avoids coincident frames:

- $\{W\}$: MuJoCo world frame.
- $\{B\}$: fixed base translated by $(0.25,-0.15,0.75)$ m and yawed by
  $0.4$ rad relative to $\{W\}$.
- $\{L\}$: moving link frame driven by the shoulder hinge.
- $\{E\}$: end-effector site translated and rotated relative to $\{L\}$.

The complete chain is

$$
T_{WE}=T_{WB}T_{BL}T_{LE}.
$$

This is the subscript-cancellation rule: the adjacent $B$ and $L$ frame labels
cancel, leaving a transform from end-effector coordinates to world coordinates.

## Reading a MuJoCo site pose

MuJoCo stores the site pose as separate arrays:

```cpp
const mjtNum* position = data->site_xpos + 3 * site_id;
const mjtNum* rotation = data->site_xmat + 9 * site_id;
```

Both describe the site in world coordinates. `site_xmat` contains a row-major
$3\times3$ rotation matrix. The program copies them into

$$
T_{WE}=
\begin{bmatrix}
R_{WE} & p_{WE}\\
0 & 1
\end{bmatrix}.
$$

Body poses use the analogous `data->xpos` and `data->xmat` arrays.
`mj_forward` is called before reading them so all derived Cartesian quantities
match the current `qpos`.

## Required utilities

All required functions are implemented in `src/transform_utils.h`.

### 1. Rotation-matrix validation

A proper rotation belongs to $SO(3)$ and satisfies

$$
R^TR=I,\qquad \det(R)=+1.
$$

`validate_rotation` reports the maximum orthogonality error, determinant, and a
boolean result. Checking only $R^TR=I$ is insufficient because a reflection has
determinant $-1$.

### 2. Homogeneous-transform multiplication

`multiply_transform(T_AB, T_BC)` calculates

$$
T_{AC}=T_{AB}T_{BC}.
$$

The experiment checks both

$$
T_{WE}=T_{WB}T_{BE}
$$

and the full chain $T_{WE}=T_{WB}T_{BL}T_{LE}$ against the end-effector site
pose read directly from MuJoCo.

### 3. Transform inversion

For $T=(R,p)$, `inverse_transform` implements

$$
T^{-1}=
\begin{bmatrix}
R^T & -R^Tp\\
0 & 1
\end{bmatrix}.
$$

The program verifies $T_{BW}T_{WB}=I$ at every sample.

### 4. Point transformation

`transform_point` calculates

$$
p_A=R_{AB}p_B+p_{AB}.
$$

A test point in $\{E\}$ is transformed directly to $\{W\}$ and indirectly
through $\{B\}$. Both results must agree.

### 5. Relative pose

Given world poses $T_{WA}$ and $T_{WB}$, `relative_pose` returns

$$
T_{AB}=T_{WA}^{-1}T_{WB}.
$$

It is used to calculate $T_{BL}$, $T_{LE}$, and $T_{BE}$. Because the site is
rigidly attached to the link, $T_{LE}$ must remain constant while the joint
moves; the report measures its numerical drift.

## Connection to the Week 3 MDP chapter

At each control step, a simple MDP interpretation is:

$$
S_t=[q_t,\dot q_t],\qquad A_t=q_{d,t}.
$$

The position command is held while MuJoCo advances the robot dynamics, yielding
$S_{t+1}$. The frame transforms are derived observations computed from the
state. An end-effector pose alone is generally not a complete Markov state for
this dynamical task because it omits velocity. If the base could move, its pose
and velocity—or equivalent sufficient information—would also be needed.

This matches Sutton and Barto Chapter 3's agent-environment interface: the
actuator command is the agent action, while the actuator, robot mechanics, and
MuJoCo integration belong to the environment transition.

## Suggested experiments

1. Set the base yaw to zero and predict how $T_{WB}$ and the world trajectory
   change while $T_{BE}$ remains unaffected.
2. Change the end-effector site's `euler` values. Its position can remain the
   same while `site_xmat` and $T_{LE}$ change.
3. Change the base translation. Confirm that $p_{WE}$ changes but $T_{BE}$ does
   not.
4. Replace `relative_pose(world_base, world_end)` with reversed arguments and
   explain why the result becomes $T_{EB}$ rather than $T_{BE}$.
5. Deliberately change one rotation-matrix entry before validation and observe
   which $SO(3)$ condition fails.
