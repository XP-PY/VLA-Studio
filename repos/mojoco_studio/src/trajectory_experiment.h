#ifndef MOJOCO_STUDIO_TRAJECTORY_EXPERIMENT_H_
#define MOJOCO_STUDIO_TRAJECTORY_EXPERIMENT_H_

#include "trajectory_generation.h"
#include <mujoco/mujoco.h>
#include <memory>

namespace trajectory {
inline Profile active_profile=Profile::Cubic;
constexpr Vec kp={25,15},kd={5,3};

/**
 * @brief Track a moving joint reference using the week 7 PD+gravity baseline.
 * tau_i=clip(Kp_i*(q_ref_i-q_i)+Kd_i*(dq_ref_i-dq_i)+g_i(q), -12,12) N m.
 * The derivative term tracks desired velocity, unlike damping toward zero.
 * Gains are N m/rad and N m s/rad. Gravity uses CURRENT q. No M*qdd_ref or c
 * feedforward is added: this is a project tracking baseline, not Chapter 9's
 * trajectory generator or a computed-torque law. Actual q, dq, qdd can differ
 * from their references, especially during acceleration or saturation.
 * mjcb_control evaluates the reference at d->time for every RK4 substage;
 * active_profile must remain fixed throughout each call to mj_step/mj_forward.
 */
inline void control(const mjModel*,mjData* d) {
  const auto desired=reference(d->time,active_profile);
  const Vec g=dynamics::gravity_torque({d->qpos[0],d->qpos[1]});
  for (int i=0;i<2;++i)
    d->ctrl[i]=std::clamp(kp[i]*(desired.q[i]-d->qpos[i])+kd[i]*(desired.dq[i]-d->qvel[i])+g[i],-12.0,12.0);
}

struct CallbackGuard {
  mjfGeneric previous=mjcb_control;
  CallbackGuard() { mjcb_control=control; }
  ~CallbackGuard() { mjcb_control=previous; }
  CallbackGuard(const CallbackGuard&)=delete;
  CallbackGuard& operator=(const CallbackGuard&)=delete;
};

class Experiment {
 public:
  std::unique_ptr<mjModel,decltype(&mj_deleteModel)> model{nullptr,mj_deleteModel};
  std::unique_ptr<mjData,decltype(&mj_deleteData)> data{nullptr,mj_deleteData};
  Profile profile;
  int ticks=0;
  static constexpr int total_steps=3500;
  explicit Experiment(Profile selected):profile(selected) {
    char error[1024]{};
    model.reset(mj_loadXML("assets/dynamics_arm.xml",nullptr,error,sizeof(error)));
    if (!model) throw std::runtime_error(error);
    if (model->nq!=2 || model->nv!=2 || model->nu!=4 || std::abs(model->opt.timestep-.001)>1e-12)
      throw std::runtime_error("Expected week 7 arm with two joints and dt=0.001 s");
    model->opt.disableactuator=2;  // position group off; use unit-gear motors
    model->dof_damping[0]=model->dof_damping[1]=0;
    data.reset(mj_makeData(model.get()));
    if (!data) throw std::runtime_error("MuJoCo data allocation failed");
    reset();
  }
  void refresh() {
    active_profile=profile;
    mj_forward(model.get(),data.get());
  }
  void reset() {
    ticks=0;
    mj_resetData(model.get(),data.get());
    for (int i=0;i<2;++i) data->qpos[i]=start[i];
    refresh();
  }
  /** @brief One fixed 0.001 s RK4 integration step with stage-wise feedback.
   * Integer ticks determine the end time; rendering and playback speed never
   * change the physics timestep. refresh() afterward makes qacc and actuator
   * force correspond to the accepted q,dq,time for logging/visualization.
   */
  void step() {
    if (ticks==total_steps) return;
    active_profile=profile;
    mj_step(model.get(),data.get());
    ++ticks;
    refresh();
  }
};
}  // namespace trajectory
#endif
