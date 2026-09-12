#ifndef MOJOCO_STUDIO_CARTESIAN_CONTROL_H_
#define MOJOCO_STUDIO_CARTESIAN_CONTROL_H_

// Week 6: small, readable position-IK and resolved-rate control primitives.
// Task vectors and Jacobian columns must use the SAME coordinate frame.
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
inline Limits guarded(const Limits& limits) {
  return {{limits.lower[0]+kLimitMargin,limits.lower[1]+kLimitMargin},
          {limits.upper[0]-kLimitMargin,limits.upper[1]-kLimitMargin}};
}
inline double norm(const Vector3& v) { return std::sqrt(dot(v,v)); }
inline double joint_norm(const Joints& q) { return std::hypot(q[0],q[1]); }
inline double max_joint(const Joints& q) { return std::max(std::abs(q[0]),std::abs(q[1])); }
inline Vector3 position(const Joints& q) {
  return transform_utils::translation_of(forward_kinematics(q[0],q[1]));
}
inline Vector3 error(const Vector3& target, const Vector3& current) {
  return add(target,scale(current,-1));
}
inline Vector3 apply(const PointJacobian& j, const Joints& dq) {
  return add(scale(j[0],dq[0]),scale(j[1],dq[1]));
}
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

// A two-column SVD: rotate J's columns to orthogonal directions B=J*V.
// The symmetric 2x2 Gram matrix supplies V; compute sigma=||B_i|| directly
// to avoid subtracting nearly equal eigenvalues near a singularity.
// This specialized routine handles a 3x2 position Jacobian, including rank 0/1.
// For a general robot, use a tested linear algebra library's SVD.
inline InverseResult inverse_velocity(const PointJacobian& j, const Vector3& desired,
                                      double damping) {
  const double a = dot(j[0],j[0]), b = dot(j[0],j[1]), c = dot(j[1],j[1]);
  const double angle = 0.5*std::atan2(2*b,a-c);
  const double cs = std::cos(angle), sn = std::sin(angle);
  const std::array<Joints,2> v = {{{cs,sn}, {-sn,cs}}};
  std::array<Vector3,2> columns = {apply(j,v[0]),apply(j,v[1])};
  InverseResult result;
  for (int i = 0; i < 2; ++i) result.sigma[i] = norm(columns[i]);
  const double cutoff = std::max(1e-12,kRelativeCutoff*result.sigma[0]);
  for (int i = 0; i < 2; ++i) {
    // J^dagger: retain 1/sigma above the numerical rank threshold.
    // DLS: replace 1/sigma by sigma/(sigma^2+lambda^2), smoothly.
    if (damping == 0 && result.sigma[i] <= cutoff) continue;
    const double denominator = result.sigma[i]*result.sigma[i]+damping*damping;
    const double coefficient = dot(columns[i],desired)/denominator;
    for (int k = 0; k < 2; ++k) result.dq[k] += v[i][k]*coefficient;
  }
  return result;
}

struct Reachability { std::string status; std::vector<Joints> solutions; };

// Geometric IK is an independent reachability oracle, NOT a controller command.
// This model has a planar task, unequal positive lengths and limits within [-pi,pi].
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
inline IKResult numerical_ik(const Vector3& target, Joints q, const Limits& limits,
                             double damping, int maximum_iterations = 300) {
  const auto reachable = geometric_ik(target,limits);
  if (reachable.status != "reachable") return {q,reachable.status,0,norm(error(target,position(q)))};
  if (!inside(q,limits)) return {q,"invalid_seed",0,norm(error(target,position(q)))};
  for (int iteration = 0; iteration <= maximum_iterations; ++iteration) {
    const auto e = error(target,position(q));
    const double distance = norm(e);
    if (distance < 1e-6) return {q,"converged",iteration,distance};
    if (iteration == maximum_iterations) return {q,"iteration_limit",iteration,distance};
    auto delta = inverse_velocity(point_jacobian(q[0],q[1]),e,damping).dq;
    if (max_joint(delta) > 0.2) {
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
inline std::vector<Scenario> scenarios() {
  return {
    {"regular",position({0.4,0.9}),{-0.7,0.7}},
    {"near_extension",position({0.3,0.03}),{-0.4,0.9}},
    {"near_singular_start",{0.65,0,0},{0,1e-4}},
    {"exact_singular_start",{0.65,0,0},{0,0}},
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
