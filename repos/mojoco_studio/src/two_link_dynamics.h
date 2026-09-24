#ifndef MOJOCO_STUDIO_TWO_LINK_DYNAMICS_H_
#define MOJOCO_STUDIO_TWO_LINK_DYNAMICS_H_

#include <array>
#include <cmath>
#include <stdexcept>

namespace dynamics {
using Vec = std::array<double, 2>;
using Mat = std::array<double, 4>;  // 2 x 2, row-major
constexpr double l1 = .45, r1 = .225, r2 = .175;
constexpr double m1 = .7, m2 = .5, i1 = .011865, i2 = .005141666666666667;
constexpr double gravity = 9.81;

/** @brief Inner product a^T b; used for mechanical power and kinetic energy.
 * Both vectors have two joint components; the product carries their combined
 * units (e.g. torque times angular velocity gives W).
 */
inline double dot(const Vec& a, const Vec& b) { return a[0]*b[0]+a[1]*b[1]; }

/** @brief Matrix-vector product y_i = sum_j A_ij x_j.
 * A is a row-major 2x2 matrix; x and y are joint vectors. In M*qdd, units are
 * kg m^2 times rad/s^2, giving generalized torques in N m (radians dimensionless).
 */
inline Vec apply(const Mat& a, const Vec& x) {
  return {a[0]*x[0]+a[1]*x[1], a[2]*x[0]+a[3]*x[1]};
}

/**
 * @brief Joint-space inertia of the distributed-mass vertical 2R arm.
 * Review: Modern Robotics notes 8.1 (Lagrangian), 8.4 (mass matrix).
 *
 * q=[shoulder, elbow] is in radians; q2 is relative to link 1. 
 * Both axes are  base +z; base +y points vertically up. 
 * COM positions in that base frame are
 *  p1=r1[cos q1,sin q1], p2=l1[cos q1,sin q1]+r2[cos(q1+q2),sin(q1+q2)].
 * Differentiating q gives 
 *  ||v1||^2=r1^2*dq1^2 and 
 *  ||v2||^2=l1^2*dq1^2+r2^2*(dq1+dq2)^2+2*l1*r2*cos(q2)*dq1*(dq1+dq2).
 * Insert these in 
 *  K=1/2 sum_i(m_i ||v_i||^2+I_i omega_i^2), 
 * with
 *  omega1=dq1, omega2=dq1+dq2, and collect K=1/2 dq^T M dq:
 *    M11=I1+I2+m1*r1^2+m2*(l1^2+r2^2+2*l1*r2*cos q2),
 *    M12=M21=I2+m2*(r2^2+l1*r2*cos q2), M22=I2+m2*r2^2.
 * Returns row-major 2x2 M in kg m^2, symmetric positive definite for these
 * positive inertias. I_i is each link's COM inertia about its hinge direction,
 * NOT about the shoulder. These rectangular rods match dynamics_arm.xml;
 * this extends the notes' point-mass example to rigid links. No armature.
 */
inline Mat mass(const Vec& q) {
  const double b=m2*l1*r2*std::cos(q[1]);
  const double d=i2+m2*r2*r2;
  return {i1+m1*r1*r1+m2*l1*l1+d+2*b, d+b, d+b, d};
}

/**
 * @brief Coriolis/centrifugal vector c(q,dq), NOT a uniquely chosen C matrix.
 * Review: Modern Robotics notes 8.1 and 8.4.
 * Euler-Lagrange gives c_i=sum_jk Gamma_ijk dq_j dq_k, where
 * Gamma_ijk=1/2(partial_k M_ij+partial_j M_ik-partial_i M_jk).
 * Only q2 changes M: partial_2 M11=-2h, partial_2 M12=-h,
 * h=m2*l1*r2*sin(q2). Substitution yields
 * c=[-h*(2*dq1*dq2+dq2^2), h*dq1^2].
 * q is rad, dq rad/s, output joint torque N m. This exact rigid-body term
 * contains no gravity or friction; MuJoCo qfrc_bias instead contains c+g.
 */
inline Vec coriolis(const Vec& q, const Vec& dq) {
  const double h=m2*l1*r2*std::sin(q[1]);
  return {-h*(2*dq[0]*dq[1]+dq[1]*dq[1]), h*dq[0]*dq[0]};
}

/**
 * @brief Gravitational potential P=sum m_i*g*height_i, in joules.
 * P=(m1*r1+m2*l1)*g*sin(q1)+m2*r2*g*sin(q1+q2).
 * Heights are relative to the shoulder, so P may be negative. The world-base
 * height contributes only an omitted constant and cannot affect dynamics.
 * q uses radians and the mass() frame. Review: Modern Robotics notes 8.1.
 */
inline double potential(const Vec& q) {
  return gravity*((m1*r1+m2*l1)*std::sin(q[0])+m2*r2*std::sin(q[0]+q[1]));
}

/**
 * @brief Torque g(q)=partial P/partial q needed to balance gravity at rest.
 * Differentiating potential() gives  
 *  g1=(m1*r1+m2*l1)*g*cos(q1)+m2*r2*g*cos(q1+q2), 
 *  g2=m2*r2*g*cos(q1+q2), 
 * in N m.
 * Gravity itself exerts -g(q); therefore compensation commands PLUS g(q).
 * This alone does not regulate velocity or position after a disturbance.
 * Review: Modern Robotics notes 8.1, 8.3 (inverse dynamics special cases).
 */
inline Vec gravity_torque(const Vec& q) {
  const double elbow=m2*r2*gravity*std::cos(q[0]+q[1]);
  return {(m1*r1+m2*l1)*gravity*std::cos(q[0])+elbow, elbow};
}

/**
 * @brief Solve symmetric positive definite A*x=b by Cholesky, without A^-1.
 * For row-major 2x2 A, set L11=sqrt(A11), L21=A21/L11,
 * L22=sqrt(A22-L21^2). Forward substitution solves L*y=b, then backward
 * substitution solves L^T*x=y. This is the numerical step in forward dynamics
 * (Modern Robotics notes 8.5). Returns acceleration when A=M and b is torque.
 * Assumes finite inputs; rejects asymmetry and nonpositive pivots. Positive
 * definiteness guarantees a solution; near-zero pivots would amplify roundoff.
 */
inline Vec solve(const Mat& a, const Vec& b) {
  if (!(a[0]>0) || std::abs(a[1]-a[2])>1e-12) throw std::runtime_error("M not SPD");
  const double l11=std::sqrt(a[0]), l21=a[2]/l11;
  const double pivot=a[3]-l21*l21;
  if (!(pivot>0)) throw std::runtime_error("M not SPD");
  const double l22=std::sqrt(pivot), y1=b[0]/l11, y2=(b[1]-l21*y1)/l22;
  const double x2=y2/l22;
  return {(y1-l21*x2)/l11,x2};
}

/**
 * @brief Required actuator torque tau=M*qdd+c+g+D*dq (inverse dynamics).
 * q,dq,qdd are two joint angles/rates/accelerations in rad, rad/s, rad/s^2.
 * damping holds diagonal viscous coefficients D>=0 in N m s/rad. The passive
 * torque is -D*dq, so it must be overcome with PLUS D*dq. Returns N m.
 * Assumes no contacts, joint limits or external wrenches. Modern Robotics
 * notes 8.3-8.4 supply the rigid-body identity; viscous damping is this project's
 * additional comparison. This is required force, without actuator clipping.
 */
inline Vec inverse(const Vec& q, const Vec& dq, const Vec& qdd, const Vec& damping={0,0}) {
  Vec tau=dynamics::apply(mass(q),qdd);
  const Vec c=coriolis(q,dq), g=gravity_torque(q);
  for (int i=0;i<2;++i) tau[i]+=c[i]+g[i]+damping[i]*dq[i];
  return tau;
}

/**
 * @brief Forward dynamics: solve M*qdd=tau-c-g-D*dq.
 * Same units, frames and force assumptions as inverse(); tau is actual joint
 * actuator torque after saturation. The zero-acceleration inverse calculation
 * isolates c+g+D*dq; Cholesky then solves for qdd. This is an instantaneous
 * identity, separate from RK4's approximate integration in time.
 * Review: Modern Robotics notes 8.5.
 */
inline Vec forward(const Vec& q, const Vec& dq, const Vec& tau, const Vec& damping={0,0}) {
  Vec rhs=inverse(q,dq,{0,0},damping);
  for (int i=0;i<2;++i) rhs[i]=tau[i]-rhs[i];
  return solve(mass(q),rhs);
}

/** @brief Mechanical energy E=K+P=1/2 dq^T M(q) dq+P(q), in joules.
 * Under our no-contact assumptions dE/dt=dq^T(tau-D*dq). Numerical RK4
 * trajectories only approximately conserve E; absolute differences avoid
 * misleading relative errors when the arbitrary energy zero gives E near 0.
 */
inline double energy(const Vec& q, const Vec& dq) {
  return .5*dot(dq,dynamics::apply(mass(q),dq))+potential(q);
}
}  // namespace dynamics
#endif
