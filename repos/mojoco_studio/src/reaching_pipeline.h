#ifndef MOJOCO_STUDIO_REACHING_PIPELINE_H_
#define MOJOCO_STUDIO_REACHING_PIPELINE_H_

#include "cartesian_control.h"
#include "trajectory_generation.h"
#include <mujoco/mujoco.h>
#include <memory>
#include <random>
#include <vector>

namespace reaching {
using Vec=dynamics::Vec;
using Point=transform_utils::Vector3;
constexpr double dt=.001,delay=.5,final_hold=2;
constexpr Vec initial={.6,-.8};
constexpr Point default_target={.55,0,1.25};
enum class Law { PD,GravityPD,InverseFeedforwardPD,ComputedTorque };
enum class Sensor { Velocity,PositionDifference,FilteredDifference };
struct Config {
  std::string name="baseline";
  Vec kp={25,15},kd={5,3};
  Law law=Law::GravityPD;
  bool jump=false;
  int period_steps=1;
  double torque_limit=12,q_noise=0,v_noise=0,cutoff_hz=12;
  Sensor sensor=Sensor::Velocity;
};

inline std::vector<Config> configurations() {
  std::vector<Config> cases;
  Config c; cases.push_back(c);
  c=Config{}; c.name="small_kp"; c.kp={3,2}; cases.push_back(c);
  c=Config{}; c.name="large_kp"; c.kp={100,60}; cases.push_back(c);
  c=Config{}; c.name="small_kd"; c.kd={.15,.1}; cases.push_back(c);
  c=Config{}; c.name="large_kd"; c.kd={12,8}; cases.push_back(c);
  c=Config{}; c.name="target_jump"; c.jump=true; cases.push_back(c);
  c=Config{}; c.name="torque_limit_2"; c.torque_limit=2; cases.push_back(c);
  c=Config{}; c.name="rate_250hz"; c.period_steps=4; cases.push_back(c);
  c=Config{}; c.name="rate_50hz"; c.period_steps=20; cases.push_back(c);
  c=Config{}; c.name="noisy_velocity"; c.period_steps=4; c.q_noise=.003; c.v_noise=.05; cases.push_back(c);
  c.name="noisy_difference"; c.sensor=Sensor::PositionDifference; c.v_noise=0; cases.push_back(c);
  c.name="filtered_difference"; c.sensor=Sensor::FilteredDifference; cases.push_back(c);
  c=Config{}; c.name="pd_only"; c.law=Law::PD; cases.push_back(c);
  c=Config{}; c.name="inverse_ff_pd"; c.law=Law::InverseFeedforwardPD; cases.push_back(c);
  c=Config{}; c.name="computed_torque"; c.law=Law::ComputedTorque;
  c.kp={64,64}; c.kd={16,16}; cases.push_back(c);
  return cases;
}

inline Config configuration(const std::string& name) {
  for (const auto& c:configurations()) if (c.name==name) return c;
  throw std::invalid_argument("Unknown case: "+name+" (use --help)");
}

struct Plan { Point world_target; Vec start,goal; double duration; int total_steps; };

/**
 * @brief Convert a world POSITION request into a feasible joint endpoint and timing.
 * The XML shoulder is p_WB=(0,0,1), R_WB=Rx(pi/2). Therefore
 * p_B=R_WB^T(p_W-p_WB)=[world_x,world_z-1,-world_y] metres.
 * Reuse week 6 geometric_ik: cos(q2)=(||p_B||^2-L1^2-L2^2)/(2 L1 L2),
 * with both elbow signs and q1=atan2(y,x)-atan2(L2 sin q2,L1+L2 cos q2).
 * Reject nonfinite/out-of-plane/unreachable/out-of-planning-range requests.
 * Select the feasible branch minimizing ||q_goal-q_start||_2 on the existing
 * unwrapped coordinate branch. This is an endpoint position task, not full pose.
 *
 * Choose quintic duration T=max(1.5 s,T_min), where T_min enforces reference
 * joint speed <=1.5 rad/s and acceleration <=3 rad/s^2; round UP to a physics
 * step. Zero motion still uses a harmless 1.5 s hold profile. Joint interpolation
 * stays in the convex planning box, but no collision avoidance or torque
 * feasibility follows. Physics uses unlimited hinges; actual range departures
 * in poor-controller experiments are logged, not prevented by hidden hard stops.
 * Review: Modern Robotics notes 6.2, 9.2.1-9.2.2, 11.4.3 (joint-reference route).
 */
inline Plan make_plan(const Point& world_target,const Vec& start=initial) {
  if (!cartesian::inside(start,cartesian::kPhysicalLimits))
    throw std::invalid_argument("Initial joints outside planning bounds");
  const Point base={world_target[0],world_target[2]-1,-world_target[1]};
  const auto ik=cartesian::geometric_ik(base,cartesian::kPhysicalLimits);
  if (ik.solutions.empty()) throw std::invalid_argument("Cartesian target rejected: "+ik.status);
  const auto distance=[&](const Vec& q) { return std::hypot(q[0]-start[0],q[1]-start[1]); };
  const Vec goal=*std::min_element(ik.solutions.begin(),ik.solutions.end(),
    [&](const Vec& a,const Vec& b){ return distance(a)<distance(b); });
  const double minimum=trajectory::minimum_duration({goal[0]-start[0],goal[1]-start[1]},
    {1.5,1.5},{3,3},trajectory::Profile::Quintic);
  const double duration=std::ceil(std::max(1.5,minimum)/dt)*dt;
  return {world_target,start,goal,duration,int(std::lround((delay+duration+final_hold)/dt))};
}

/**
 * @brief Desired joint q,dq,ddq, with a stationary start and final hold.
 * Smooth mode composes q=q_start+s(q_goal-q_start) with week 8 quintic scaling;
 * its dq,ddq follow by chain rule. time is elapsed seconds, coordinates rad.
 * Jump mode switches q at t=0.5 s and sets desired dq=ddq=0 on both sides.
 * A setpoint jump is not a physically differentiable trajectory: its ideal
 * derivative impulses are intentionally NOT sent to the motor (no derivative
 * kick). The comparison isolates proportional response to a new setpoint.
 * Review: Modern Robotics notes 9.2.2 and 11.1, project jump-test convention.
 */
inline trajectory::Sample reference(double time,const Plan& plan,const Config& config) {
  if (time<delay) return {plan.start,{0,0},{0,0}};
  if (config.jump || time>=delay+plan.duration) return {plan.goal,{0,0},{0,0}};
  return trajectory::joint_line(plan.start,plan.goal,
    trajectory::scaling(time-delay,plan.duration,trajectory::Profile::Quintic));
}

/** @brief True Euclidean joint error ||q_ref-q||_2 in rad, used for tracking metrics.
 * Both vectors are on the same unwrapped two-revolute-joint branch. This does
 * not wrap modulo 2*pi, which would conceal extra turns in unstable experiments.
 */
inline double error_norm(const Vec& a,const Vec& b) { return std::hypot(a[0]-b[0],a[1]-b[1]); }

struct Command { Vec raw{},applied{},feedback{},feedforward{}; };

/**
 * @brief Form effort from a desired trajectory and sampled estimated joint state.
 * e=q_d-q_hat, ed=dq_d-v_hat. Four deliberately distinct laws:
 *  PD: tau=Kp*e+Kd*ed.
 *  Gravity PD: tau=Kp*e+Kd*ed+g(q_hat).
 *  Inverse-feedforward PD: tau=Kp*e+Kd*ed+ID(q_d,dq_d,ddq_d).
 *  Computed torque: tau=M(q_hat)*(ddq_d+Kp*e+Kd*ed)+c(q_hat,v_hat)+g(q_hat).
 * Desired-state inverse dynamics is OPEN-LOOP feedforward, while computed
 * torque uses CURRENT estimated state for cancellation. The first three gain
 * units are N m/rad and N m s/rad. Computed-torque gains instead have units
 * s^-2 and s^-1; its stored feedback is M times acceleration correction (N m).
 * raw=feedback+feedforward; applied_i=clip(raw_i,+/-torque_limit), in N m.
 *
 * With exact unsaturated continuous-time computed torque, substituting
 * M*qdd+c+g=tau yields edd+Kd*ed+Kp*e=0. No such exact decoupling holds for
 * ordinary gravity PD; for a fixed setpoint its virtual energy satisfies
 * dE/dt=-dq^T Kd dq under ideal assumptions. For a scalar uncoupled inertia,
 * omega_n=sqrt(Kp/M), zeta=Kd/(2 sqrt(M Kp)); these are NOT global modes of
 * this coupled nonlinear arm. Sampling, noise, clipping and model errors break
 * ideal cancellation/stability arguments. No contacts, friction or armature.
 * Review: Modern Robotics notes 11.2 and 11.4.1-11.4.2.
 */
inline Command command(const trajectory::Sample& desired,const Vec& measured_q,
                       const Vec& measured_v,const Config& c) {
  Command result;
  for (int i=0;i<2;++i)
    result.feedback[i]=c.kp[i]*(desired.q[i]-measured_q[i])+c.kd[i]*(desired.dq[i]-measured_v[i]);
  if (c.law==Law::GravityPD) result.feedforward=dynamics::gravity_torque(measured_q);
  if (c.law==Law::InverseFeedforwardPD)
    result.feedforward=dynamics::inverse(desired.q,desired.dq,desired.ddq);
  if (c.law==Law::ComputedTorque) {
    result.feedback=dynamics::apply(dynamics::mass(measured_q),result.feedback);
    result.feedforward=dynamics::inverse(measured_q,measured_v,desired.ddq);
  }
  for (int i=0;i<2;++i) {
    result.raw[i]=result.feedback[i]+result.feedforward[i];
    result.applied[i]=std::clamp(result.raw[i],-c.torque_limit,c.torque_limit);
  }
  return result;
}

struct Observation { Vec q{},v{}; };
class SensorModel {
  std::mt19937 rng_;
  std::normal_distribution<double> normal_{0,1};
  bool first_=true;
  Vec previous_q_{},filtered_v_{};
 public:
  explicit SensorModel(unsigned seed):rng_(seed) {}
  void reset(unsigned seed) { rng_.seed(seed); normal_.reset(); first_=true; previous_q_={}; filtered_v_={}; }

  /**
   * @brief Sample sensors ONCE per controller tick, never per render/RK4 stage.
   * q_hat=q+sigma_q*N(0,1). Direct velocity uses v_hat=v+sigma_v*N(0,1).
   * Difference mode uses v_raw=(q_hat[k]-q_hat[k-1])/Tc; if position noise is
   * independent with variance sigma_q^2, its velocity-noise variance is
   * 2*sigma_q^2/Tc^2. Smaller sampling periods can therefore amplify derivative
   * noise. First differenced velocity is initialized to zero (initial rest).
   * Filtered mode: v_hat=beta*v_hat_old+(1-beta)*v_raw,
   * beta=exp(-2*pi*cutoff_hz*Tc). This discrete first-order low-pass approximation
   * reduces noise at the cost of lag; it is not an unbiased noiseless derivative.
   * Four independent normal draws occur in every mode, so equal seeds and
   * periods give identical position-noise sequences for paired comparisons.
   * Inputs/outputs: q rad, velocity rad/s, Tc seconds, cutoff Hz. Noise levels
   * are per-sample standard deviations, NOT continuous-time noise densities.
   * Review: Modern Robotics notes 11.1 sampling/differentiation discussion;
   * the Gaussian sensor and low-pass discretization are project choices.
   */
  Observation sample(const Vec& q,const Vec& velocity,const Config& config) {
    Observation observation;
    const double period=config.period_steps*dt;
    for (int i=0;i<2;++i) {
      observation.q[i]=q[i]+config.q_noise*normal_(rng_);
      const double direct=velocity[i]+config.v_noise*normal_(rng_);
      const double difference=first_?0:(observation.q[i]-previous_q_[i])/period;
      const double beta=std::exp(-2*cartesian::kPi*config.cutoff_hz*period);
      filtered_v_[i]=beta*filtered_v_[i]+(1-beta)*difference;
      observation.v[i]=config.sensor==Sensor::Velocity?direct:
        (config.sensor==Sensor::PositionDifference?difference:filtered_v_[i]);
    }
    previous_q_=observation.q; first_=false;
    return observation;
  }
};

// The week 9 experiment intentionally holds torque through ALL RK4 stages.
// Remove any preexisting callback while running, then restore it on exit.
struct HeldControlGuard {
  mjfGeneric old=mjcb_control;
  HeldControlGuard() { mjcb_control=nullptr; }
  ~HeldControlGuard() { mjcb_control=old; }
  HeldControlGuard(const HeldControlGuard&)=delete;
  HeldControlGuard& operator=(const HeldControlGuard&)=delete;
};

class Experiment {
  SensorModel sensor_;
 public:
  std::unique_ptr<mjModel,decltype(&mj_deleteModel)> model{nullptr,mj_deleteModel};
  std::unique_ptr<mjData,decltype(&mj_deleteData)> data{nullptr,mj_deleteData};
  Config config;
  Plan plan;
  unsigned seed;
  int ticks=0,updates=0;
  double control_time=0;
  Observation observation;
  Command effort;
  Experiment(Config c,Plan p,unsigned random_seed=0):sensor_(random_seed),config(c),plan(p),seed(random_seed) {
    if (c.period_steps<=0 || c.torque_limit<=0 || !std::isfinite(c.torque_limit))
      throw std::invalid_argument("Invalid controller period or torque limit");
    char error[1024]{};
    model.reset(mj_loadXML("assets/dynamics_arm.xml",nullptr,error,sizeof(error)));
    if (!model) throw std::runtime_error(error);
    if (model->nq!=2 || model->nv!=2 || model->nu!=4 || std::abs(model->opt.timestep-dt)>1e-12)
      throw std::runtime_error("Expected two-link arm, four actuators, 1 ms physics timestep");
    model->opt.disableactuator=2;
    for (int i=0;i<2;++i) {
      model->dof_damping[i]=0;
      model->actuator_ctrlrange[2*i]=-c.torque_limit;
      model->actuator_ctrlrange[2*i+1]=c.torque_limit;
    }
    data.reset(mj_makeData(model.get()));
    if (!data) throw std::runtime_error("MuJoCo allocation failed");
    reset();
  }
  void refresh() { mj_forward(model.get(),data.get()); }
  void update_control() {
    control_time=ticks*dt;
    observation=sensor_.sample({data->qpos[0],data->qpos[1]},{data->qvel[0],data->qvel[1]},config);
    effort=command(reference(control_time,plan,config),observation.q,observation.v,config);
    for (int i=0;i<2;++i) data->ctrl[i]=effort.applied[i];
    ++updates;
  }
  void reset() {
    ticks=updates=0; sensor_.reset(seed); mj_resetData(model.get(),data.get());
    for (int i=0;i<2;++i) data->qpos[i]=plan.start[i];
    update_control(); refresh();
  }
  /**
   * @brief Advance one physics tick with a sampled, zero-order-held torque.
   * At controller times k*Tc, sample sensors, evaluate the reference, and set
   * ctrl; for the following interval tau(t)=tau[k]. RK4 integrates the plant
   * with this held input. Controller timing uses integer ticks, avoiding drift
   * or dependence on rendering. At an accepted controller boundary, update the
   * next interval's torque before logging; qacc then reflects that new torque.
   * No extra update is performed at the experiment's final endpoint.
   * Requires HeldControlGuard for the experiment's lifetime. A frame refresh
   * performs no sensor draws, no feedback update, and no advancement of time.
   */
  void step() {
    if (ticks>=plan.total_steps) return;
    mj_step(model.get(),data.get()); ++ticks;
    if (ticks<plan.total_steps && ticks%config.period_steps==0) update_control();
    refresh();
  }
};
}  // namespace reaching
#endif
