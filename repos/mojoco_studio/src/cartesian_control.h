#ifndef MOJOCO_STUDIO_CARTESIAN_CONTROL_H_
#define MOJOCO_STUDIO_CARTESIAN_CONTROL_H_

// Week 6: small, readable position-IK and resolved-rate control primitives.
// Task vectors and Jacobian columns must use the SAME coordinate frame.
// Review references below identify sections in the user's Modern_Robotics.md
// notes; their numbering is not necessarily the textbook's section numbering.
#include "two_link_kinematics.h"
#include <algorithm>
#include <limits>
#include <string>
#include <vector>

namespace cartesian {
using namespace two_link;
using Joints = std::array<double, 2>;
constexpr double kPi = 3.14159265358979323846;
constexpr double kRelativeCutoff = 1e-8;
constexpr double kDamping = 0.03;  // same units as the position-Jacobian singular values
struct Limits { Joints lower; Joints upper; };
constexpr Limits kPhysicalLimits{{-1.5,-1.7}, {1.5,1.7}};
constexpr double kLimitMargin = 0.02;  // rad, reserve braking room in dynamics
/**
 * @brief Reserve an interior interval for joint-limit approach control.
 *
 * For each joint i, with margin m = kLimitMargin:
 *
 *      q_min_guarded[i] = q_min[i] + m
 *      q_max_guarded[i] = q_max[i] - m.
 *
 * Inputs and outputs are hinge angles in radians. Each original interval must
 * be wider than 2m to leave a nonempty interior. The physical joint limits in
 * MuJoCo are not changed; these bounds are used by the controller and its
 * reachability check. Reserving braking room is a project implementation
 * choice, not a guarantee that arbitrary dynamics will respect the bounds.
 */
inline Limits guarded(const Limits& limits) {
  return {{limits.lower[0]+kLimitMargin,limits.lower[1]+kLimitMargin},
          {limits.upper[0]-kLimitMargin,limits.upper[1]-kLimitMargin}};
}
/**
 * @brief Measure a Cartesian vector's Euclidean magnitude: ||v||_2 = sqrt(v^T v).
 * Used for position-error and task-velocity residuals. The result retains the
 * input units (e.g. m or m/s); all components must have compatible units.
 */
inline double norm(const Vector3& v) { return std::sqrt(dot(v,v)); }
/**
 * @brief Measure joint-vector magnitude: ||q||_2 = sqrt(q1^2 + q2^2).
 * Both joints are revolute, so components have the same units. Used for IK
 * corrections (rad) and the controller's measured settling speed (rad/s).
 */
inline double joint_norm(const Joints& q) { return std::hypot(q[0],q[1]); }
/**
 * @brief Return the infinity norm: ||q||_inf = max(|q1|, |q2|).
 * This measures the largest individual joint correction/rate. Uniformly
 * scaling by bound/||q||_inf enforces a component bound without changing the
 * vector's direction; it differs from the Euclidean settling-speed criterion.
 */
inline double max_joint(const Joints& q) { return std::max(std::abs(q[0]),std::abs(q[1])); }
/**
 * @brief Evaluate the position task f(q) used by numerical IK.
 *
 *      p_BE = f(q) = [L1 cos(q1) + L2 cos(q1+q2),
 *                     L1 sin(q1) + L2 sin(q1+q2), 0]^T.
 *
 * q contains shoulder/elbow angles in radians; the result is the tip position
 * in the fixed base frame, in metres. This extracts the translation of T_BE;
 * endpoint orientation is not constrained by this task.
 * Review: Modern Robotics notes, Chapter 6, "Analytic Example: Planar 2R Arm".
 */
inline Vector3 position(const Joints& q) {
  return transform_utils::translation_of(forward_kinematics(q[0],q[1]));
}
/**
 * @brief Form the signed residual e = target - current.
 *
 * For IK, e = x_d - f(q), so the local correction has the PLUS sign:
 *      J(q) Delta_q approximately equals e; q_next = q + Delta_q.
 * Also used to compare velocities or Jacobian columns. Both arguments must
 * have the same units, frame and reference point. Position subtraction here
 * is not the full-pose SE(3) error vee(log(T_current^{-1} T_desired)).
 * Review: Modern Robotics notes, sections 6.4 and 6.6.
 */
inline Vector3 error(const Vector3& target, const Vector3& current) {
  return add(target,scale(current,-1));
}
/**
 * @brief Apply a point-position Jacobian: J dq = J_1 dq1 + J_2 dq2.
 *
 * j stores a 3x2 matrix as TWO COLUMNS. With dq = q_dot (rad/s), the result
 * is tip linear velocity (m/s), expressed in j's frame. With dq = Delta_q
 * (rad), it predicts a local displacement (m). inverse_velocity also uses
 * this matrix-vector product with dimensionless singular-vector directions.
 * This is the point-velocity map from Chapter 5, not a six-component twist.
 */
inline Vector3 apply(const PointJacobian& j, const Joints& dq) {
  return add(scale(j[0],dq[0]),scale(j[1],dq[1]));
}
/**
 * @brief Check finite joint coordinates against a box constraint.
 *
 *      q_min[i] - tolerance <= q[i] <= q_max[i] + tolerance, for all i.
 *
 * Coordinates, bounds and tolerance are in radians. The tolerance handles
 * numerical comparisons; it is not an extra commanded motion allowance.
 * Role: reject invalid IK seeds/branches and monitor executed configurations.
 */
inline bool inside(const Joints& q, const Limits& limits, double tolerance = 1e-12) {
  for (int i = 0; i < 2; ++i)
    if (!std::isfinite(q[i]) || q[i] < limits.lower[i]-tolerance ||
        q[i] > limits.upper[i]+tolerance) return false;
  return true;
}

struct InverseResult {
  Joints dq{};
  std::array<double,2> sigma{};  // descending singular values
};

/**
 * @brief Solve the local map J dq approximately equals desired, with optional damping.
 *
 * Role:
 *   In resolved-rate control, desired is Cartesian velocity and dq is a joint
 *   velocity. In numerical IK, the same linear solver maps a position error
 *   to a joint correction. Neither use solves nonlinear IK in a single step.
 *   Review: Modern Robotics notes 6.5 (pseudoinverse) and 6.8 (inverse velocity).
 *   Damped least squares is the regularized comparison added for week 6.
 *
 * 1. What the SVD means for this velocity map
 *
 *   Write d = desired and take the full singular value decomposition:
 *
 *      J = U Sigma V^T,       J: 3x2, U: 3x3, Sigma: 3x2, V: 2x2
 *      U^T U = I_3,          V^T V = I_2
 *      Sigma = [sigma_1    0   ]       sigma_1 >= sigma_2 >= 0.
 *              [   0    sigma_2]
 *              [   0       0   ]
 *
 *   Columns v_i of V are orthonormal JOINT-space directions. Columns u_i of U
 *   are orthonormal TASK-space directions. Multiplying J by v_i gives
 *
 *      J v_i = sigma_i u_i.
 *
 *   Thus a unit joint motion along v_i creates task motion along u_i with
 *   magnitude sigma_i. To generate one unit along an attainable u_i, the
 *   inverse needs 1/sigma_i units along v_i. This explains amplification when
 *   a retained singular value is small.
 *
 * 2. Derive J^dagger from a minimum-norm least-squares problem
 *
 *   We want to minimize ||J dq - d||_2, then choose the smallest ||dq||_2
 *   among all minimizers. Change coordinates into the singular-vector bases:
 *
 *      z = V^T dq,      y = U^T d,      dq = V z.
 *
 *   Orthogonal matrices preserve Euclidean norms, so
 *
 *      ||J dq - d||_2^2
 *        = ||U^T (J dq - d)||_2^2
 *        = ||Sigma z - y||_2^2
 *        = (sigma_1 z_1 - y_1)^2 + (sigma_2 z_2 - y_2)^2 + y_3^2.
 *
 *   The coupled matrix problem has become independent scalar problems:
 *      sigma_i > 0: minimize by choosing z_i = y_i / sigma_i;
 *      sigma_i = 0: the residual is independent of z_i, so choose z_i = 0.
 *
 *   Why zero in the second case? Since ||dq||_2 = ||z||_2, a nonzero component
 *   in J's null space increases joint motion without reducing task error.
 *   Setting it to zero selects the minimum-norm least-squares solution.
 *   The y_3 component, and any y_i with sigma_i=0, remain unattainable.
 *
 *   Define Sigma^dagger (2x3) by transposing the rectangular diagonal shape
 *   and replacing positive singular values by their reciprocals:
 *
 *      Sigma^dagger = [sigma_1^dagger         0          0]
 *                    [       0        sigma_2^dagger   0],
 *      sigma_i^dagger = 1/sigma_i if sigma_i>0, otherwise 0.
 *
 *   Transform the solution back to joint coordinates:
 *
 *      dq = V Sigma^dagger y = V Sigma^dagger U^T d
 *      J^dagger = V Sigma^dagger U^T                 (a 2x3 matrix)
 *               = sum_{sigma_i>0} (1/sigma_i) v_i u_i^T
 *      dq = sum_{sigma_i>0} v_i (u_i^T d) / sigma_i.
 *
 *   Here u_i^T d projects the request onto a task direction, division by
 *   sigma_i compensates for its motion gain, and v_i maps it to joint motion.
 *   If U_r contains the attainable directions, J J^dagger d = U_r U_r^T d:
 *   this is the orthogonal projection of d onto J's column space, not always d.
 *
 *   Exact zero singular values are replaced by a numerical rank decision in
 *   code: discard sigma_i <= max(1e-12, kRelativeCutoff * sigma_max). Discarding
 *   a small POSITIVE singular value is a truncated-SVD approximation to the
 *   mathematical pseudoinverse; it deliberately leaves that direction unused.
 *
 * 3. Derive the right singular vectors from the 2x2 Gram matrix
 *
 *   The implementation does not construct full U or Sigma. Starting from SVD:
 *
 *      G = J^T J = V Sigma^T U^T U Sigma V^T
 *                = V diag(sigma_1^2, sigma_2^2) V^T.
 *
 *   Therefore G's eigenvectors are J's right singular vectors, and its
 *   eigenvalues are the squared singular values. For columns J_1 and J_2:
 *
 *      G = [a b; b c],    a=J_1^T J_1, b=J_1^T J_2, c=J_2^T J_2.
 *
 *   Choose V as a planar rotation with columns
 *      v_1 = [cos(theta), sin(theta)]^T,
 *      v_2 = [-sin(theta), cos(theta)]^T.
 *
 *   To diagonalize G, require the off-diagonal entry of V^T G V to vanish:
 *
 *      v_1^T G v_2
 *        = b cos(2 theta) + (c-a)/2 sin(2 theta) = 0.
 *
 *   A quadrant-aware solution is
 *      theta = 0.5 atan2(2b, a-c).
 *
 *   This places the larger eigenvalue first. If G is a scalar multiple of I,
 *   both eigenvalues coincide and any orthonormal basis is valid. Eigenvector
 *   signs can also differ from a library SVD without changing J^dagger.
 *
 *   Now form B_i = J v_i, stored as columns[i] in code. Since V diagonalizes G,
 *      B_i^T B_j = v_i^T G v_j = 0 for i != j;
 *      sigma_i = ||B_i||_2;
 *      u_i = B_i/sigma_i when sigma_i > 0.
 *
 *   The code computes sigma_i from ||B_i|| rather than from the smaller
 *   eigenvalue (a+c-sqrt((a-c)^2+4b^2))/2, which can lose precision by
 *   subtracting nearly equal numbers. This still uses a Gram matrix and is
 *   specialized to 3x2; a general manipulator should use a tested library SVD.
 *
 * 4. Derive the damped inverse in the same coordinates
 *
 *   For lambda = damping > 0, add a joint-motion penalty:
 *
 *      minimize_dq ||J dq - d||_2^2 + lambda^2 ||dq||_2^2
 *        = minimize_z sum_{i=1,2} [(sigma_i z_i-y_i)^2 + lambda^2 z_i^2]
 *                     + y_3^2.
 *
 *   Differentiating with respect to each z_i and setting it to zero gives
 *      2 sigma_i (sigma_i z_i-y_i) + 2 lambda^2 z_i = 0
 *      z_i = [sigma_i/(sigma_i^2+lambda^2)] y_i.
 *
 *   Thus the damped solution is
 *      dq = sum_i v_i [sigma_i/(sigma_i^2+lambda^2)] (u_i^T d)
 *         = (J^T J + lambda^2 I)^(-1) J^T d.
 *
 *   The latter identity also follows from the normal equations; the code
 *   evaluates the spectral sum rather than explicitly inverting a matrix.
 *   Each scalar gain is at most 1/(2 lambda), reached at sigma_i=lambda.
 *   Damping trades task residual for smaller corrections, but cannot create a
 *   missing direction: J^T d = 0 still gives dq = 0. Unlike the undamped branch,
 *   the damped branch does not apply the hard singular-value cutoff.
 *
 * 5. Why the code divides by sigma_i^2, not sigma_i
 *
 *   Substitute B_i = sigma_i u_i into the damped spectral expression:
 *
 *      [sigma_i/(sigma_i^2+lambda^2)] (u_i^T d)
 *        = (B_i^T d)/(sigma_i^2+lambda^2)
 *      dq = sum_i v_i (B_i^T d)/(sigma_i^2+lambda^2).
 *
 *   The numerator already contains one factor sigma_i. For lambda=0, this
 *   reduces to (u_i^T d)/sigma_i, exactly the standard pseudoinverse gain.
 *   There is no need to normalize B_i into u_i, which also avoids division by
 *   zero when a column vanishes. The code-to-equation correspondence is:
 *
 *      v[i]           -> right singular vector v_i (2 components)
 *      columns[i]     -> B_i = J v_i (3 components)
 *      result.sigma[i]-> ||B_i||_2
 *      denominator    -> sigma_i^2 + lambda^2
 *      coefficient    -> (B_i^T desired)/denominator
 *      result.dq      -> sum_i v_i * coefficient.
 *
 *   The undamped branch skips discarded directions BEFORE division. In the
 *   damped branch lambda^2 keeps the denominator positive, even for sigma_i=0.
 *   No full pseudoinverse matrix is needed when only its product with d is used.
 *
 * Example: the straight 2R arm at q=(0,0)
 *
 *      J = [0    0  ],    d = [0,1,0]^T m/s.
 *          [0.8  0.35]
 *          [0    0  ]
 *
 *   This matrix has rank one. Its nonzero singular value is sqrt(0.8^2+0.35^2),
 *   with v_1=[0.8,0.35]^T/sigma_1 and u_1=[0,1,0]^T. Therefore
 *      J^dagger d = [0.8,0.35]^T/(0.8^2+0.35^2) rad/s.
 *   Both joints contribute to the minimum-norm tangential motion. Replacing d
 *   by [1,0,0]^T gives zero: a radial request cannot be met instantaneously.
 *
 * @param j Position Jacobian, 3x2 stored as columns, in m/rad.
 * @param desired Three-vector in j's frame: m/s for control or m for IK.
 * @param damping Nonnegative lambda, in the same units as singular values.
 * @return Joint rates (rad/s) or corrections (rad), plus descending singular
 *         values. No joint limits or speed caps are imposed by this function.
 */
inline InverseResult inverse_velocity(const PointJacobian& j, const Vector3& desired,
                                      double damping) {
  // Construct Gram matrix J^T J
  // The eigenvectors of J^T J are the right singular vectors of J.
  const double a = dot(j[0],j[0]), b = dot(j[0],j[1]), c = dot(j[1],j[1]);

  // Find the singular directions
  // Compute the rotation that diagonalizes the 2x2 matrix J^T J
  const double angle = 0.5*std::atan2(2*b,a-c);
  const double cs = std::cos(angle), sn = std::sin(angle);
  // Two orthogonal joint-space directions: v_1, v_2. These are the right singular vectors.
  const std::array<Joints,2> v = {{{cs,sn}, {-sn,cs}}};
  // B_i = J v_i (i=1,2)
  std::array<Vector3,2> columns = {apply(j,v[0]), apply(j,v[1])};
  // Length of B_i is the singular value.
  InverseResult result;
  for (int i = 0; i < 2; ++i) result.sigma[i] = norm(columns[i]);
  // Set a truncation threshold (to prevent division by zero)
  const double cutoff = std::max(1e-12,kRelativeCutoff*result.sigma[0]);

  // 
  for (int i = 0; i < 2; ++i) {
    // J^dagger: retain 1/sigma above the numerical rank threshold.
    // DLS: replace 1/sigma by sigma/(sigma^2+lambda^2), smoothly.

    // Truncated SVD
    if (damping == 0 && result.sigma[i] <= cutoff) continue;

    // Standard pseudoinverse: denominator = sigma_i^2
    // DLS: denominator = sigma_i^2 + lambda^2
    const double denominator = result.sigma[i]*result.sigma[i]+damping*damping;
    const double coefficient = dot(columns[i],desired)/denominator;
    for (int k = 0; k < 2; ++k) result.dq[k] += v[i][k]*coefficient;
  }
  return result;
}

struct Reachability { std::string status; std::vector<Joints> solutions; };

/**
 * @brief Find feasible elbow branches by planar geometry, independently of local IK.
 *
 * Review: Modern Robotics notes 6.2, "Analytic Example: Planar 2R Arm".
 * Squaring and adding the forward-position equations gives the cosine law:
 *
 *      r^2 = x^2 + y^2 = L1^2 + L2^2 + 2 L1 L2 cos(q2)
 *      D = (r^2 - L1^2 - L2^2) / (2 L1 L2).
 *
 * An in-plane target has an unlimited-joint solution iff |D| <= 1, equivalently
 *      |L1-L2| <= r <= L1+L2.
 * Both elbow signs must then be checked:
 *      q2 = atan2(+/-sqrt(1-D^2), D)
 *      q1 = atan2(y,x) - atan2(L2 sin(q2), L1 + L2 cos(q2)).
 *
 * The first q1 term points toward the target; the second corrects for link 2's
 * contribution. atan2 preserves quadrants. Each q1 is normalized modulo 2pi
 * and both branches are filtered through the supplied joint limits. At a
 * workspace boundary the branches coincide and only one copy is returned.
 *
 * @param target Tip position [x,y,z] in the BASE frame, metres; z must be zero
 *               within the plane-check tolerance. Nonfinite input is rejected.
 * @param limits Hinge limits in radians, within [-pi,pi]. This implementation
 *               assumes the fixed positive unequal lengths L1=0.45, L2=0.35 m.
 * @return A status and all feasible branches (0, 1 or 2). outside_annulus is
 *         distinct from joint_limits: annulus membership alone is insufficient.
 *
 * Role in this project:
 *   Reject infeasible targets and independently validate numerical solutions.
 *   No returned branch is used to steer numerical_ik or reach. A reachable
 *   status does not guarantee that a local solver converges from its seed.
 */
inline Reachability geometric_ik(const Vector3& target, const Limits& limits) {
  if (!std::isfinite(target[0]) || !std::isfinite(target[1]) || !std::isfinite(target[2]))
    return {"invalid_target",{}};
  if (std::abs(target[2]) > 1e-10) return {"outside_plane",{}};
  double cosine = (dot(target,target)-kL1*kL1-kL2*kL2)/(2*kL1*kL2);
  if (cosine < -1-1e-12 || cosine > 1+1e-12) return {"outside_annulus",{}};
  cosine = std::clamp(cosine,-1.0,1.0);
  Reachability result{"joint_limits",{}};
  for (double sign : {1.0,-1.0}) {
    const double q2 = std::atan2(sign*std::sqrt(std::max(0.0,1-cosine*cosine)),cosine);
    const double q1 = std::remainder(std::atan2(target[1],target[0])-
        std::atan2(kL2*std::sin(q2),kL1+kL2*std::cos(q2)),2*kPi);
    const Joints q = {q1,q2};
    if (inside(q,limits)) {
      if (result.solutions.empty() || joint_norm({q1-result.solutions[0][0],
                                                 q2-result.solutions[0][1]}) > 1e-10)
        result.solutions.push_back(q);
    }
  }
  if (!result.solutions.empty()) result.status = "reachable";
  return result;
}

struct LimitedVelocity { Joints dq; bool speed_limited; bool limit_limited; };
/**
 * @brief Bound a joint-rate request and slow motion toward joint limits.
 *
 * This is a practical controller guard, not a textbook pseudoinverse identity.
 * First preserve the direction of the raw vector while enforcing a speed cap:
 *
 *      s = min(1, v_max / ||dq_raw||_inf); dq_scaled = s dq_raw
 *      (take s=1 when dq_raw is zero).
 *
 * Then bound each component with a distance-to-limit approach law:
 *      k_l = min(5 s^(-1), 1/dt)
 *      lower_i = max(-v_max, k_l (q_min[i] - q[i]))
 *      upper_i = min(+v_max, k_l (q_max[i] - q[i]))
 *      dq_cmd[i] = clip(dq_scaled[i], lower_i, upper_i).
 *
 * For q initially inside the box and ideal Euler integration, dt*k_l <= 1
 * implies q + dt*dq_cmd stays inside it. Component clipping can change the
 * Cartesian direction. This is not a constrained IK optimizer, and the ideal
 * bound does not guarantee limit invariance under inertia or servo lag.
 *
 * @param raw Requested joint rates, rad/s.
 * @param q Measured joint angles, rad, normally inside the guarded bounds.
 * @param limits Guarded bounds in radians. Clipping requires lower_i <= upper_i;
 *               the monitored physical state and margin ensure this here.
 * @param maximum_speed Positive per-joint reference-speed cap, rad/s.
 * @param control_dt Positive controller period, s.
 * @return Bounded reference rates and flags indicating which guard changed them.
 */
inline LimitedVelocity limit_velocity(Joints raw, const Joints& q, const Limits& limits,
                                      double maximum_speed, double control_dt) {
  LimitedVelocity result{raw,false,false};
  if (max_joint(raw) > maximum_speed) {
    const double factor = maximum_speed/max_joint(raw);
    for (double& v : result.dq) v *= factor;  // preserve joint-velocity direction
    result.speed_limited = true;
  }
  // Slow down toward each limit. dt*k <= 1 also prevents overshoot under ideal
  // Euler integration. Real dynamics still require monitoring and braking room.
  const double approach_gain = std::min(5.0,1.0/control_dt);
  for (int i = 0; i < 2; ++i) {
    const double bounded = std::clamp(result.dq[i],
        std::max(-maximum_speed,approach_gain*(limits.lower[i]-q[i])),
        std::min(maximum_speed,approach_gain*(limits.upper[i]-q[i])));
    result.limit_limited |= std::abs(bounded-result.dq[i]) > 1e-12;
    result.dq[i] = bounded;
  }
  return result;
}

struct IKResult { Joints q; std::string status; int iterations; double final_error; };
/**
 * @brief Find a joint configuration for a tip position by repeated local corrections.
 *
 * Review: Modern Robotics notes 6.4 (Newton-Raphson), 6.5 (pseudoinverse),
 * and 6.9 (convergence and the initial guess).
 *
 * Linearize at the current iterate q^k:
 *      f(q^k + Delta_q) approximately equals f(q^k) + J(q^k) Delta_q
 *      e^k = target - f(q^k)
 *      Delta_q = J(q^k)^dagger e^k.
 *
 * With positive damping, use inverse_velocity's regularized correction instead.
 * The PLUS update sign follows from defining error as desired minus current.
 * Error and Jacobian must be recomputed after every accepted update.
 *
 * Project-specific safeguards added to that local iteration:
 *   1. Reject an infeasible target or invalid seed before iterating.
 *   2. Scale Delta_q so ||Delta_q||_inf <= 0.2 rad.
 *   3. Try alpha = 1, 1/2, ..., 1/32768 and form
 *          q_candidate = clip(q^k + alpha Delta_q, q_min, q_max).
 *   4. Accept only if ||target - f(q_candidate)||_2 < ||e^k||_2 - 1e-14 m.
 *
 * Stop successfully only when ||e^k||_2 < 1e-6 m. A nonzero residual with no
 * accepted step is stalled; reaching the iteration budget is iteration_limit.
 * A reachable target can still stall at a singular seed (J^T e = 0), at a
 * bound, or because this is a local search. Minimum-norm corrections do not
 * imply a globally nearest solution. The geometric oracle does not initialize
 * or replace the numerical iterates.
 *
 * @param target Desired position in the fixed BASE frame, m; orientation is free.
 * @param q Initial shoulder/elbow guess, rad (copied and updated locally).
 * @param limits Joint-coordinate box in radians, checked by geometric_ik.
 * @param damping Nonnegative regularization parameter; zero selects pseudoinverse.
 * @param maximum_iterations Nonnegative maximum number of correction attempts.
 * @return Final q, termination status, iteration index and residual norm (m).
 *
 * No physical time passes in an IK iteration: here inverse_velocity receives
 * metres and returns a correction in radians, not a velocity in rad/s. This
 * position-only solver does not implement the full-pose SE(3) log-error method.
 */
inline IKResult numerical_ik(const Vector3& target, Joints q, const Limits& limits,
                             double damping, int maximum_iterations = 300) {
  const auto reachable = geometric_ik(target,limits);
  if (reachable.status != "reachable") return {q,reachable.status,0,norm(error(target,position(q)))};
  if (!inside(q,limits)) return {q,"invalid_seed",0,norm(error(target,position(q)))};
  for (int iteration = 0; iteration <= maximum_iterations; ++iteration) {
    const auto e = error(target,position(q));   // e^k = target - f(q^k)
    const double distance = norm(e);
    if (distance < 1e-6) return {q,"converged",iteration,distance};
    if (iteration == maximum_iterations) return {q,"iteration_limit",iteration,distance};
    auto delta = inverse_velocity(point_jacobian(q[0],q[1]),e,damping).dq;  // Delta_q = J(q^k)^dagger e^k
    if (max_joint(delta) > 0.2) {
      // Limit the single-step correction amount
      const double factor = 0.2/max_joint(delta);
      for (double& component : delta) component *= factor;
    }
    // Backtrack the LOCAL correction and project candidates into joint limits.
    bool accepted = false;
    for (double alpha = 1.0; alpha >= 1.0/32768; alpha *= 0.5) {
      Joints candidate{};
      for (int i = 0; i < 2; ++i)
        candidate[i] = std::clamp(q[i]+alpha*delta[i],limits.lower[i],limits.upper[i]);
      if (norm(error(target,position(candidate))) < distance-1e-14) {
        q = candidate;
        accepted = true;
        break;
      }
    }
    if (!accepted) return {q,"stalled",iteration,distance};
  }
  return {q,"iteration_limit",maximum_iterations,norm(error(target,position(q)))};
}

struct Scenario { std::string name; Vector3 target; Joints initial; };
/**
 * @brief Define repeatable experiments linking reachability and local convergence.
 * Targets use base-frame metres and seeds use radians. position(q_goal) creates
 * reachable position targets from known configurations. The straight-arm pair
 * contrasts q2=0 (J^T e=0 for inward radial error) with q2=1e-4 (a small bend).
 * The other cases isolate annulus, plane, physical-limit and guard-margin checks.
 */
inline std::vector<Scenario> scenarios() {
  return {
    {"regular",position({0.4,0.9}),{-0.7,0.7}},
    {"near_extension",position({0.3,0.03}),{-0.4,0.9}},
    {"near_singular_start",{0.65,0,0},{0,1e-4}},
    {"exact_singular_start",{0.65,0,0},{0,0}},
    {"exact_singular_start_with_column_component", {0.65, 0.1, 0}, {0, 0}},
    {"near_joint_limit",position({1.45,1.6}),{0.8,1.0}},
    {"outside_outer_radius",{0.95,0.05,0},{0,0.8}},
    {"inside_inner_hole",{0.05,0,0},{0,0.8}},
    {"elbow_limit_unreachable",{0.3,0,0},{0,0.8}},
    {"shoulder_limit_unreachable",position({2.1,0.4}),{0,0.8}},
    {"out_of_plane",{0.5,0.15,0.1},{0,0.8}},
    {"inside_guard_margin",position({1.5,1.7}),{0.8,1.0}}
  };
}
}  // namespace cartesian
#endif
