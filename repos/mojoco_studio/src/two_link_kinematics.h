#ifndef MOJOCO_STUDIO_TWO_LINK_KINEMATICS_H_
#define MOJOCO_STUDIO_TWO_LINK_KINEMATICS_H_

// Week 5 analytical model of assets/two_link_arm.xml, independent of MuJoCo.
// T_AB maps B coordinates into A. Twists are ordered [omega; v].
#include "transform_utils.h"
#include <array>
#include <cmath>

namespace two_link {
using transform_utils::Matrix3;
using transform_utils::Matrix4;
using transform_utils::Vector3;
constexpr double kL1 = 0.45, kL2 = 0.35;
struct Twist { Vector3 omega; Vector3 v; };
using Jacobian = std::array<Twist, 2>;  // stored as two columns, not row-major
using PointJacobian = std::array<Vector3, 2>;
inline Vector3 add(const Vector3& a, const Vector3& b) {
  return {a[0]+b[0], a[1]+b[1], a[2]+b[2]};
}
inline Vector3 scale(const Vector3& a, double s) {
  return {s*a[0], s*a[1], s*a[2]};
}
inline Vector3 cross(const Vector3& a, const Vector3& b) {
  return {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};
}
inline double dot(const Vector3& a, const Vector3& b) {
  return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}
inline Vector3 rotate(const Matrix3& r, const Vector3& v) {
  return {r[0]*v[0]+r[1]*v[1]+r[2]*v[2],
          r[3]*v[0]+r[4]*v[1]+r[5]*v[2],
          r[6]*v[0]+r[7]*v[1]+r[8]*v[2]};
}
inline Matrix4 forward_kinematics(double q1, double q2) {
  const double c = std::cos(q1+q2), s = std::sin(q1+q2);
  return transform_utils::make_transform({c,-s,0, s,c,0, 0,0,1},
      {kL1*std::cos(q1)+kL2*c, kL1*std::sin(q1)+kL2*s, 0});
}

/**
 * @brief Compute the space Jacobian J_s(q) of the 2R arm.
 *
 * The space Jacobian maps joint velocity to the end-effector spatial twist:
 *
 *      V_s = J_s(q) q_dot.
 *
 * Each column is a 6D twist:
 *
 *      J_si(q) =
 *      [ omega_i ]
 *      [    v_i  ]
 *
 * and represents the CURRENT screw axis of joint i, expressed in the
 * space/base frame.
 *
 * For a revolute joint whose axis has unit direction omega and passes
 * through a point q_axis,
 *
 *      S = [ omega ]
 *          [   v   ],
 *
 *      v = -omega × q_axis.
 *
 * For this robot:
 *
 * Joint 1:
 *   Its axis is the z-axis through the base origin:
 *
 *      omega_1 = [0, 0, 1]^T
 *      q_axis1 = [0, 0, 0]^T
 *
 *   therefore
 *
 *      J_s1 =
 *      [ 0 ]
 *      [ 0 ]
 *      [ 1 ]
 *      [ 0 ]
 *      [ 0 ]
 *      [ 0 ].
 *
 * Joint 2:
 *   At the current posture, joint 2 is located at
 *
 *      q_axis2 =
 *      [ L1 cos(q1) ]
 *      [ L1 sin(q1) ]
 *      [      0      ].
 *
 *   Since omega_2 = [0,0,1]^T,
 *
 *      v_2 = -omega_2 × q_axis2
 *
 *          = [  L1 sin(q1) ]
 *            [ -L1 cos(q1) ]
 *            [       0      ].
 *
 * Thus
 *
 *      J_s = [ J_s1  J_s2 ].
 *
 * Relation to the Product of Exponentials formulation:
 *
 *      J_s1 = S1
 *
 *      J_s2 = Ad_{exp([S1] q1)} S2.
 *
 * This explains why J_s2 depends on q1 but not q2:
 * joint 1 moves joint 2's axis, while joint 2 does not move its own axis.
 *
 * General rule:
 *
 *      J_si depends only on joints 1, ..., i-1.
 *
 * Role:
 *   J_s describes the full rigid-body instantaneous motion of the
 *   end effector in space coordinates:
 *
 *      V_s = [ omega_s ; v_s ].
 *
 *   It is used for velocity kinematics, statics, singularity analysis,
 *   manipulability, and later inverse velocity kinematics.
 */
inline Jacobian space_jacobian(double q1) {
  return {{{{0,0,1}, {0,0,0}},
           {{0,0,1}, {kL1*std::sin(q1), -kL1*std::cos(q1), 0}}}};
}

/**
 * @brief Transform a twist by the Adjoint of a rigid transformation T.
 *
 * A twist is stored as
 *
 *      V = [ omega ]
 *          [   v   ]
 *
 * where:
 *
 *      omega = angular velocity / screw-axis angular component
 *      v     = linear part of the twist
 *
 * Suppose the rigid transformation is
 *
 *      T =
 *      [ R  p ]
 *      [ 0  1 ].
 *
 * The Adjoint transformation of T is
 *
 *      Ad_T =
 *      [   R      0 ]
 *      [ [p]R    R ].
 *
 * Therefore, if
 *
 *      V' = Ad_T V,
 *
 * then
 *
 *      omega' = R omega
 *
 *      v' = p × (R omega) + R v.
 *
 * This function implements exactly those two equations:
 *
 *      omega' = rotate(R, omega)
 *
 *      v' = cross(p, omega') + rotate(R, v).
 *
 * Meaning:
 *   The physical rigid-body motion does not change.
 *   Only the coordinate frame used to EXPRESS the twist changes.
 *
 * Why translation p appears:
 *   A twist is not just an ordinary direction vector. Its linear component
 *   also depends on the choice of reference origin.
 *
 *   If the reference origin changes by p, a rotational motion omega creates
 *   an additional linear contribution
 *
 *      p × omega.
 *
 * This is why a twist must be transformed with Ad_T rather than with R alone.
 *
 * Typical robotics relations:
 *
 *      V_s = Ad_{T_sb} V_b
 *
 * converts a body twist into a spatial twist, while
 *
 *      V_b = Ad_{T_sb^{-1}} V_s.
 *
 * The same operation also applies to screw axes and Jacobian columns, since
 * each Jacobian column is itself a twist.
 *
 * Role in this project:
 *   - convert analytical screw axes/Jacobian columns between frames;
 *   - convert the base-frame space Jacobian to the MuJoCo world frame;
 *   - construct the body Jacobian from the space Jacobian.
 */
inline Twist adjoint(const Matrix4& t, const Twist& twist) {
  const auto r = transform_utils::rotation_of(t);

  const auto omega = rotate(r, twist.omega);

  return {
      omega,
      add(
          cross(transform_utils::translation_of(t), omega),
          rotate(r, twist.v)
      )
  };
}

/**
 * @brief Transform every column of a Jacobian by the Adjoint of T.
 *
 * Suppose
 *
 *      T =
 *      [ R  p ]
 *      [ 0  1 ].
 *
 * A twist
 *
 *      V = [ omega ]
 *          [   v   ]
 *
 * transforms according to
 *
 *      V' = Ad_T V,
 *
 * where
 *
 *      Ad_T =
 *      [   R      0 ]
 *      [ [p]R    R ].
 *
 * Therefore
 *
 *      omega' = R omega
 *
 *      v' = p × (R omega) + R v.
 *
 * Since a Jacobian is a collection of twist columns,
 *
 *      J = [ J_1  J_2 ... J_n ],
 *
 * changing the coordinate frame of every twist column gives
 *
 *      J' = Ad_T J.
 *
 * This function performs exactly that operation:
 *
 *      J'_i = Ad_T J_i
 *
 * for every Jacobian column.
 *
 * Important:
 *   A twist cannot in general be transformed using rotation R alone.
 *   The translation p also matters because the linear part of a twist
 *   depends on the reference origin:
 *
 *      v' = p × (R omega) + R v.
 *
 * Role:
 *   Used whenever the same physical Jacobian must be expressed in a
 *   different coordinate frame.
 *
 * Example:
 *
 *      J_s^world = Ad_{T_WB} J_s^base
 *
 * converts a space Jacobian expressed in the robot base frame into the
 * MuJoCo world frame.
 *
 * It is also used to convert a space Jacobian into a body Jacobian by
 * using T^{-1}.
 */
inline Jacobian transform_jacobian(const Matrix4& t, const Jacobian& j) {
  return {adjoint(t, j[0]), adjoint(t, j[1])};
}

/**
 * @brief Compute the body Jacobian J_b(q).
 *
 * The body Jacobian maps joint velocity to the end-effector body twist:
 *
 *      V_b = J_b(q) q_dot.
 *
 * Here V_b is the same physical end-effector motion as V_s, but expressed
 * in the current end-effector/body frame instead of the space/base frame.
 *
 * Space and body twists are related by
 *
 *      V_s = Ad_T V_b,
 *
 * therefore
 *
 *      V_b = Ad_{T^{-1}} V_s.
 *
 * Since
 *
 *      V_s = J_s(q) q_dot,
 *
 * we obtain
 *
 *      V_b
 *        = Ad_{T^{-1}} J_s(q) q_dot
 *
 * and hence
 *
 *      J_b(q) = Ad_{T^{-1}(q)} J_s(q).
 *
 * In this implementation:
 *
 *      T_BE(q) = forward_kinematics(q1, q2)
 *
 * is the current end-effector pose relative to the base frame.
 *
 * Therefore
 *
 *      T_EB = T_BE^{-1}
 *
 * converts quantities from the base/space frame into the current
 * end-effector frame, and
 *
 *      J_b = Ad_{T_EB} J_s.
 *
 * Each column J_bi is the current screw axis of joint i expressed in the
 * end-effector/body frame.
 *
 * For this 2R arm, the body Jacobian linear parts are
 *
 *      J_b1.v =
 *      [ L1 sin(q2)          ]
 *      [ L1 cos(q2) + L2     ]
 *      [ 0                   ]
 *
 *      J_b2.v =
 *      [ 0  ]
 *      [ L2 ]
 *      [ 0  ].
 *
 * Notice the body-Jacobian dependency rule:
 *
 *      J_bi depends only on joints i+1, ..., n.
 *
 * Therefore:
 *   - J_b1 depends on q2;
 *   - J_b2 is constant because no joints follow joint 2.
 *
 * Role:
 *   Used whenever end-effector motion is most naturally expressed in the
 *   end-effector frame. In particular, Modern Robotics numerical inverse
 *   kinematics commonly uses
 *
 *      V_b = J_b(q) q_dot
 *
 * and
 *
 *      Delta q = J_b^dagger V_b,error.
 */
inline Jacobian body_jacobian(double q1, double q2) {
  return transform_jacobian(
      transform_utils::inverse_transform(forward_kinematics(q1,q2)),
      space_jacobian(q1));
}

/**
 * @brief Compute the end-effector point-position Jacobian J_p(q).
 *
 * Mathematical definition:
 *
 *      p_BE(q) =
 *      [ L1 cos(q1) + L2 cos(q1+q2) ]
 *      [ L1 sin(q1) + L2 sin(q1+q2) ]
 *      [              0              ]
 *
 *      J_p(q) = ∂p_BE / ∂q
 *
 *             = [ ∂p/∂q1   ∂p/∂q2 ]
 *
 * and therefore
 *
 *      p_dot_BE = J_p(q) q_dot.
 *
 * Each column has the usual Jacobian interpretation:
 *
 *      J_p[:, i]
 *
 * is the linear velocity of the end-effector point when joint i moves
 * with unit velocity and all other joints are fixed.
 *
 * For this planar 2R arm:
 *
 *      J_p1 =
 *      [ -L1 sin(q1) - L2 sin(q1+q2) ]
 *      [  L1 cos(q1) + L2 cos(q1+q2) ]
 *      [                  0           ]
 *
 *      J_p2 =
 *      [ -L2 sin(q1+q2) ]
 *      [  L2 cos(q1+q2) ]
 *      [         0       ]
 *
 * Important:
 *   This is a POINT Jacobian. Its columns contain the actual linear
 *   velocity p_dot of the end-effector origin.
 *
 *   It is NOT simply the lower three rows of the space Jacobian J_s.
 *   For a spatial twist V_s = [omega_s; v_s],
 *
 *      p_dot = omega_s × p + v_s,
 *
 *   so in general
 *
 *      p_dot != v_s.
 *
 * Role:
 *   Used when we care specifically about the Cartesian velocity of the
 *   end-effector point, e.g. position kinematics, finite-difference
 *   checks, point manipulability, or force mapping at the tip.
 */
inline PointJacobian point_jacobian(double q1, double q2) {
  const double s = std::sin(q1+q2), c = std::cos(q1+q2);
  return {{{-kL1*std::sin(q1)-kL2*s, kL1*std::cos(q1)+kL2*c, 0},
           {-kL2*s, kL2*c, 0}}};
}

/**
 * @brief Compute the rigid-body twist produced by joint velocities q_dot.
 *
 * The manipulator Jacobian is stored as two twist columns:
 *
 *      J = [ J_1  J_2 ],
 *
 * where
 *
 *      J_i =
 *      [ omega_i ]
 *      [   v_i   ].
 *
 * The basic Jacobian velocity relation is
 *
 *      V = J(q) q_dot.
 *
 * For this 2R robot,
 *
 *      q_dot =
 *      [ dq1 ]
 *      [ dq2 ],
 *
 * so
 *
 *      V = J_1 dq1 + J_2 dq2.
 *
 * Separating angular and linear parts gives
 *
 *      omega =
 *          omega_1 dq1
 *        + omega_2 dq2
 *
 *      v =
 *          v_1 dq1
 *        + v_2 dq2.
 *
 * This function implements exactly those two weighted sums.
 *
 * Physical meaning:
 *   Each Jacobian column J_i describes the twist generated when joint i
 *   moves with unit speed and all other joints are fixed.
 *
 *   Multiplying by dq_i scales that unit-speed twist to the actual
 *   contribution of joint i:
 *
 *      actual contribution of joint i = J_i dq_i.
 *
 *   The total end-effector twist is the sum of all joint contributions.
 *
 * Important:
 *   The meaning of the returned Twist depends on which Jacobian is passed:
 *
 *      velocity(space_jacobian(...), dq)
 *
 * gives a spatial twist
 *
 *      V_s = J_s q_dot,
 *
 * while
 *
 *      velocity(body_jacobian(...), dq)
 *
 * gives a body twist
 *
 *      V_b = J_b q_dot.
 *
 * If the supplied Jacobian is a spatial Jacobian, the lower component v
 * is the spatial-twist linear component, not necessarily the actual
 * end-effector-origin velocity p_dot.
 *
 * They are related by
 *
 *      p_dot = omega × p + v.
 *
 * Role:
 *   Used to verify the fundamental Chapter 5 relation
 *
 *      V = J(q) q_dot
 *
 * against MuJoCo-computed site velocities and finite-difference velocities.
 */
inline Twist velocity(const Jacobian& j,
                      const std::array<double,2>& dq) {
  return {
      add(
          scale(j[0].omega, dq[0]),
          scale(j[1].omega, dq[1])
      ),
      add(
          scale(j[0].v, dq[0]),
          scale(j[1].v, dq[1])
      )
  };
}

}  // namespace two_link
#endif
