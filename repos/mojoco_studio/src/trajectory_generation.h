#ifndef MOJOCO_STUDIO_TRAJECTORY_GENERATION_H_
#define MOJOCO_STUDIO_TRAJECTORY_GENERATION_H_

#include "two_link_dynamics.h"
#include <algorithm>

namespace trajectory {
using dynamics::Vec;
enum class Profile { Cubic, Quintic };
struct Scaling { double s,velocity,acceleration,jerk; };
struct Sample { Vec q,dq,ddq; };
constexpr Vec start={.6,-.8},goal={-.3,.6};
constexpr double hold_before=.5,duration=2,hold_after=1,total_time=3.5;
inline const char* name(Profile profile) { return profile==Profile::Cubic?"cubic":"quintic"; }

/**
 * @brief Rest-to-rest polynomial time scaling, including analytical derivatives.
 * Review: Modern Robotics notes 9.2.2 (not an invented textbook equation number).
 * Write u=t/T. Cubic s=a0+a1*u+a2*u^2+a3*u^3 has four constraints:
 * s(0)=0, s(1)=1, s'(0)=s'(1)=0. They give a0=a1=0,a2=3,a3=-2.
 * Quintic adds s''(0)=s''(1)=0; solving six linear constraints gives
 * a0=a1=a2=0, a3=10,a4=-15,a5=6. Thus
 * cubic: s=3u^2-2u^3, ds=6u(1-u)/T, dds=(6-12u)/T^2;
 * quintic: s=10u^3-15u^4+6u^5, ds=30u^2(1-u)^2/T,
 * dds=60u(1-u)(1-2u)/T^2. Each time differentiation adds a factor 1/T.
 * jerk is the ordinary third derivative INSIDE the polynomial segment;
 * cubic jerk=-12/T^3, quintic jerk=(60-360u+360u^2)/T^3.
 * t,T are seconds; s dimensionless, derivatives in 1/s,1/s^2,1/s^3.
 * Requires finite T>0 and 0<=t<=T; rejects invalid inputs. At endpoints returns
 * polynomial-side derivatives. Cubic acceleration jumps when joined to rest,
 * so its join jerk includes ideal impulses NOT represented by this finite jerk.
 * Quintic acceleration is continuous at rest joins, but jerk may still jump.
 */
inline Scaling scaling(double t,double T,Profile profile) {
  if (!std::isfinite(t) || !std::isfinite(T) || T<=0 || t<0 || t>T)
    throw std::invalid_argument("Scaling needs finite T>0 and 0<=t<=T");
  const double u=t/T;
  if (profile==Profile::Cubic)
    return {u*u*(3-2*u),6*u*(1-u)/T,(6-12*u)/(T*T),-12/(T*T*T)};
  return {u*u*u*(10+u*(-15+6*u)),30*u*u*(1-u)*(1-u)/T,
          60*u*(1-u)*(1-2*u)/(T*T),(60-360*u+360*u*u)/(T*T*T)};
}

/**
 * @brief Synchronize two joints along the straight joint path q(s)=q0+s(q1-q0).
 * Chain rule gives dq=(q1-q0)*ds and ddq=(q1-q0)*dds because q''(s)=0.
 * Inputs q0,q1 are shoulder/relative-elbow angles in rad on ONE chosen unwrapped
 * branch; outputs q,dq,ddq have units rad,rad/s,rad/s^2. No shortest-angle wrap,
 * IK, collision checking or torque feasibility is implied. Scaling contains
 * time derivatives, not derivatives with respect to normalized u.
 * Review: Modern Robotics notes 9.1 and 9.2.1.
 */
inline Sample joint_line(const Vec& q0,const Vec& q1,const Scaling& time) {
  Sample result{};
  for (int i=0;i<2;++i) {
    const double delta=q1[i]-q0[i];
    result.q[i]=q0[i]+delta*time.s;
    result.dq[i]=delta*time.velocity;
    result.ddq[i]=delta*time.acceleration;
  }
  return result;
}

/**
 * @brief Compose a 0.5 s initial hold, 2 s motion, and final stationary hold.
 * Outside [hold_before,hold_before+duration], dq=ddq=0. At the exact joins,
 * choose the polynomial-side derivative to make the cubic acceleration jump
 * explicit. This endpoint convention does not remove its discontinuity.
 * Joint position and velocity are continuous for both profiles; acceleration
 * is continuous only for quintic. time is elapsed simulation time in seconds.
 */
inline Sample reference(double time,Profile profile) {
  if (!std::isfinite(time)) throw std::invalid_argument("Nonfinite reference time");
  if (time<hold_before) return {start,{0,0},{0,0}};
  if (time>hold_before+duration) return {goal,{0,0},{0,0}};
  return joint_line(start,goal,scaling(time-hold_before,duration,profile));
}

/**
 * @brief Minimum duration within the chosen polynomial family under joint bounds.
 * Analytical scalar peaks are cv/T and ca/T^2, with (cv,ca)=(1.5,6) for cubic
 * or (15/8,10/sqrt(3)) for quintic. Enforcing |delta_i|*cv/T<=vlim_i and
 * |delta_i|*ca/T^2<=alim_i gives T_min=max_i(cv|delta_i|/vlim_i,
 * sqrt(ca|delta_i|/alim_i)). These exact extrema are stronger than sampled checks.
 * delta is rad, vlim rad/s, alim rad/s^2; bounds must be finite and positive.
 * Zero displacement imposes no bound; all-stationary input returns 0 (hold,
 * do not call scaling with T=0). This does NOT check torque, collision or jerk
 * limits and is not globally time-optimal. Review: Modern Robotics notes 9.2.2.
 */
inline double minimum_duration(const Vec& delta,const Vec& vlim,const Vec& alim,Profile profile) {
  const double cv=profile==Profile::Cubic?1.5:15.0/8;
  const double ca=profile==Profile::Cubic?6:10/std::sqrt(3.0);
  double T=0;
  for (int i=0;i<2;++i) {
    if (!std::isfinite(delta[i]) || !std::isfinite(vlim[i]) || !std::isfinite(alim[i]) || vlim[i]<=0 || alim[i]<=0)
      throw std::invalid_argument("Invalid joint bounds");
    T=std::max({T,cv*std::abs(delta[i])/vlim[i],std::sqrt(ca*std::abs(delta[i])/alim[i])});
  }
  return T;
}

/**
 * @brief Two-link tip position in the vertical WORLD x-z plane, in metres.
 * p=[L1 cos(q1)+L2 cos(q1+q2), 1+L1 sin(q1)+L2 sin(q1+q2)].
 * Output stores [world x,world z], NOT two joint angles; world y=0. The 1 m
 * shoulder height and link lengths match dynamics_arm.xml. Used to show that
 * a joint-space line generally maps to a curved Cartesian path (notes 9.2.1).
 */
inline Vec tip(const Vec& q) {
  return {.45*std::cos(q[0])+.35*std::cos(q[0]+q[1]),
          1+.45*std::sin(q[0])+.35*std::sin(q[0]+q[1])};
}

/**
 * @brief Straight Cartesian POSITION reference between the endpoint tip positions.
 * p(t)=p0+s(t)(p1-p0), in world [x,z] metres. This is the translational part of
 * the decoupled Cartesian trajectory in Modern Robotics notes 9.2.1; it does
 * not interpolate orientation or solve IK. We plot it for geometric comparison
 * only. A 2R arm cannot independently impose arbitrary planar position AND
 * orientation; a Cartesian line is not automatically a feasible joint motion.
 */
inline Vec cartesian_line(double s) {
  const Vec p0=tip(start),p1=tip(goal);
  return {p0[0]+s*(p1[0]-p0[0]),p0[1]+s*(p1[1]-p0[1])};
}
}  // namespace trajectory
#endif
