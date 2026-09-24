#ifndef MOJOCO_STUDIO_DYNAMICS_EXPERIMENTS_H_
#define MOJOCO_STUDIO_DYNAMICS_EXPERIMENTS_H_

#include "two_link_dynamics.h"
#include <mujoco/mujoco.h>
#include <algorithm>
#include <array>

// Shared by the CSV batch and the interactive MuJoCo viewer. The callback is
// process-global: activate the appropriate mode before stepping each model.
// All simulations are sequential on one thread.
namespace experiments {
using dynamics::Vec;
enum class Mode { Free, Damping, Constant, Gravity, Position, PD, PDGravity, MovingGravity };
inline Mode mode=Mode::Free;
constexpr Vec target={-.3,.6}, kp={25,15}, kd={5,3};
/**
 * @brief Feedback laws used at EVERY RK4 stage through mjcb_control.
 * Gravity modes command tau=g(q); torque PD commands
 * tau_i=clip(Kp_i*(q_des_i-q_i)-Kd_i*dq_i [+g_i(q)], -12,12).
 * Gains have units N m/rad and N m s/rad; q_des is fixed in radians. Motors
 * have unit gear so ctrl is joint torque (N m). Position actuators implement
 * the identical PD law internally with their own force bounds; their ctrl is
 * q_des, NOT torque. Inactive actuator groups are disabled in reset().
 * These are project controller comparisons, not full computed-torque control:
 * neither PD+g nor gravity compensation cancels c or multiplies by M.
 * The global mode is fixed throughout each mj_forward/mj_step call, including
 * RK4 stages. The viewer switches it between experiments on the same thread.
 */
inline void control(const mjModel*, mjData* d) {
  const Vec q={d->qpos[0],d->qpos[1]}, g=dynamics::gravity_torque(q);
  for (int i=0;i<2;++i) {
    // Default active control torques 0 for `Free` and `Damping` modes
    double tau=0;
    // Constant control torques [1.0; -0.4] in `Constant` mode
    if (mode==Mode::Constant) tau=i==0?1.0:-.4;
    // Add gravity compensation
    //  In `Gravity` mode,
    //    if robotic arm is stationary (Velocity and acceleration are 0),
    //    set torque = gravity, robotic arm should maintain its current pose.
    //  In `MovingGravity` mode,
    //    If the robotic arm itself has velocity,
    //    Gravity compensation cancels only `g`; it does not cancel `c`
    //    Nor does it generate damping, so the robot will not come to a stop on its own.
    if (mode==Mode::Gravity || mode==Mode::MovingGravity) tau=g[i];
    if (mode==Mode::PD || mode==Mode::PDGravity)
      tau=kp[i]*(target[i]-q[i])-kd[i]*d->qvel[i]+(mode==Mode::PDGravity?g[i]:0);

    d->ctrl[i]=std::clamp(tau,-12.0,12.0);  // Torque motor
    d->ctrl[2+i]=target[i];                 // Position servo
  }
}

struct Description { Mode mode; const char* name; const char* law; const char* observation; };
inline constexpr std::array<Description,8> descriptions={{
  {Mode::Free,"no_control","tau = 0","Energy conserved; arm swings freely"},
  {Mode::Damping,"joint_damping","tau = 0; passive torque = -D dq","Damping dissipates energy"},
  {Mode::Constant,"constant_torque","tau = [1, -0.4] N m","Unlimited joints can keep rotating"},
  {Mode::Gravity,"gravity_compensation","tau = g(q); initial dq = 0","Balances gravity at the initial pose"},
  {Mode::Position,"position_actuator","Built-in position PD servo","Gravity causes a steady target error"},
  {Mode::PD,"torque_pd","tau = clip(Kp e - Kd dq)","Same PD law as the position actuator"},
  {Mode::PDGravity,"torque_pd_gravity","tau = clip(Kp e - Kd dq + g)","Gravity compensation removes offset"},
  {Mode::MovingGravity,"gravity_moving","tau = g(q); initial dq = [0.3,-0.2]","Gravity balance does not stop motion"}
}};
inline constexpr int steps=6000;

inline bool is_servo(Mode selected) {
  return selected==Mode::Position || selected==Mode::PD || selected==Mode::PDGravity;
}

/**
 * @brief Restore the identical initial-value problem for batch and live viewing.
 * q(0)=[0.6,-0.8] rad; dq(0)=0 except MovingGravity=[0.3,-0.2] rad/s.
 * D=diag(0.2,0.1) N m s/rad only in Damping; other modes have D=0.
 * Position mode disables motor group 0, all other modes disable servo group 1:
 * zero position ctrl would still exert force, so group masking is essential.
 * mj_resetData clears time and applied forces; this function does NOT call
 * mj_forward, allowing the caller to install the shared RK4 callback first.
 */
inline void reset(mjModel* m, mjData* d, Mode selected) {
  mode=selected;
  m->opt.disableactuator=selected==Mode::Position?1:2;
  m->dof_damping[0]=selected==Mode::Damping?.2:0;
  m->dof_damping[1]=selected==Mode::Damping?.1:0;
  mj_resetData(m,d);
  d->qpos[0]=.6; d->qpos[1]=-.8;
  if (selected==Mode::MovingGravity) { d->qvel[0]=.3; d->qvel[1]=-.2; }
}

struct CallbackGuard {
  mjfGeneric old=mjcb_control;
  CallbackGuard() { mjcb_control=control; }
  ~CallbackGuard() { mjcb_control=old; }
  CallbackGuard(const CallbackGuard&)=delete;
  CallbackGuard& operator=(const CallbackGuard&)=delete;
};
}  // namespace experiments
#endif
