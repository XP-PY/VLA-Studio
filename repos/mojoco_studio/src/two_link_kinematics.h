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

// Js=[S1, Ad_exp([S1]q1) S2]. Joint 1 is at the base origin;
// joint 2 is at (L1 cos q1, L1 sin q1, 0). Each v = -omega x axis_point.
inline Jacobian space_jacobian(double q1) {
  return {{{{0,0,1}, {0,0,0}},
           {{0,0,1}, {kL1*std::sin(q1), -kL1*std::cos(q1), 0}}}};
}
inline Twist adjoint(const Matrix4& t, const Twist& twist) {
  const auto r = transform_utils::rotation_of(t);
  const auto omega = rotate(r, twist.omega);
  return {omega, add(cross(transform_utils::translation_of(t), omega),
                     rotate(r, twist.v))};
}
inline Jacobian transform_jacobian(const Matrix4& t, const Jacobian& j) {
  return {adjoint(t, j[0]), adjoint(t, j[1])};
}
inline Jacobian body_jacobian(double q1, double q2) {
  return transform_jacobian(transform_utils::inverse_transform(forward_kinematics(q1,q2)),
                            space_jacobian(q1));
}

// This is d p_BE / d q, not the lower three rows of Js.
inline PointJacobian point_jacobian(double q1, double q2) {
  const double s = std::sin(q1+q2), c = std::cos(q1+q2);
  return {{{-kL1*std::sin(q1)-kL2*s, kL1*std::cos(q1)+kL2*c, 0},
           {-kL2*s, kL2*c, 0}}};
}
inline Twist velocity(const Jacobian& j, const std::array<double,2>& dq) {
  return {add(scale(j[0].omega,dq[0]), scale(j[1].omega,dq[1])),
          add(scale(j[0].v,dq[0]), scale(j[1].v,dq[1]))};
}
}  // namespace two_link
#endif
