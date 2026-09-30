#include "trajectory_experiment.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace {
using namespace trajectory;
void require(bool condition,const char* message) {
  if (!condition) throw std::runtime_error(message);
}

/**
 * @brief Check endpoint constraints, derivatives, analytic extrema and rest joins.
 * Central differences (h=1e-5 s) independently approximate ds/dt and d(ds)/dt
 * at interior times, avoiding discontinuities. Boundary tests use one-sided
 * polynomial accelerations versus the exact stationary value zero. The notes'
 * example delta=[1,.5], vlim=[1,.75], alim=[2,1] must yield T3=sqrt(3),T5=1.875.
 * These tests distinguish endpoint smoothness from merely sampling a nice plot.
 */
void validate() {
  for (Profile profile:{Profile::Cubic,Profile::Quintic}) {
    const auto a=scaling(0,2,profile),b=scaling(2,2,profile);
    require(a.s==0 && b.s==1 && a.velocity==0 && b.velocity==0,"Endpoint constraints failed");
    require(profile==Profile::Cubic?(a.acceleration==1.5 && b.acceleration==-1.5):
      (a.acceleration==0 && b.acceleration==0),"Acceleration boundary constraints failed");
    for (int i=1;i<100;++i) {
      const double t=.02*i,h=1e-5;
      const auto s=scaling(t,2,profile),plus=scaling(t+h,2,profile),minus=scaling(t-h,2,profile);
      require(std::abs((plus.s-minus.s)/(2*h)-s.velocity)<1e-8,"Velocity derivative failed");
      require(std::abs((plus.velocity-minus.velocity)/(2*h)-s.acceleration)<1e-8,"Acceleration derivative failed");
      require(std::abs((plus.acceleration-minus.acceleration)/(2*h)-s.jerk)<1e-8,"Jerk derivative failed");
      require(s.s>=0 && s.s<=1 && s.velocity>=0,"Progress not monotonic");
    }
    const Vec delta={goal[0]-start[0],goal[1]-start[1]};
    const double min_time=minimum_duration(delta,{1.5,1.5},{3,3},profile);
    require(duration>=min_time,"Default trajectory violates kinematic bounds");
    require(minimum_duration({0,0},{1,1},{1,1},profile)==0,"Stationary duration failed");
    const double example=minimum_duration({1,.5},{1,.75},{2,1},profile);
    require(std::abs(example-(profile==Profile::Cubic?std::sqrt(3.0):1.875))<1e-12,"Notes example failed");
  }
  bool rejected=false;
  try { scaling(0,0,Profile::Cubic); } catch (const std::invalid_argument&) { rejected=true; }
  require(rejected,"Zero duration was not rejected");
  std::cout << "Trajectory endpoints, derivatives, continuity and duration checks passed.\n";
}

/**
 * @brief Execute one trajectory and log reference versus actual robot state.
 * At accepted 1 ms states, e=q_ref-q, RMS=sqrt(sum_t ||e||^2/N), maximum error
 * is max_t ||e||; these are two-joint Euclidean errors in rad over the whole
 * hold/move/hold experiment. peak speed/acceleration are sampled componentwise
 * maxima, distinct from exact reference extrema checked by minimum_duration().
 * qacc is MuJoCo's instantaneous dynamics acceleration, not a finite difference.
 * The Cartesian line is comparison geometry only; all motors track joint_line.
 */
void run(Profile profile,std::ofstream& summary) {
  Experiment sim(profile);
  const int tip_id=mj_name2id(sim.model.get(),mjOBJ_SITE,"tip");
  require(tip_id>=0,"Missing tip site");
  std::ofstream out(std::string("build/trajectories/")+name(profile)+".csv");
  out << "time,s,q1_ref,q2_ref,dq1_ref,dq2_ref,ddq1_ref,ddq2_ref,q1,q2,dq1,dq2,ddq1,ddq2,tau1,tau2,error_norm,tip_x,tip_z,ref_tip_x,ref_tip_z,line_x,line_z\n"
      << std::setprecision(16);
  double squared_error=0,max_error=0,peak_velocity=0,peak_acceleration=0,peak_torque=0;
  int saturated=0;
  for (int tick=0;tick<=Experiment::total_steps;++tick) {
    const auto* d=sim.data.get();
    const auto ref=reference(d->time,profile);
    const Vec p=tip({d->qpos[0],d->qpos[1]}),pr=tip(ref.q);
    const double s=(ref.q[0]-start[0])/(goal[0]-start[0]);
    const Vec pc=cartesian_line(s);
    const double error=std::hypot(ref.q[0]-d->qpos[0],ref.q[1]-d->qpos[1]);
    squared_error+=error*error; max_error=std::max(max_error,error);
    require(std::isfinite(error),"Nonfinite trajectory state");
    require(std::abs(p[0]-d->site_xpos[3*tip_id])<1e-10 &&
            std::abs(p[1]-d->site_xpos[3*tip_id+2])<1e-10,"World FK mismatch");
    for (int j=0;j<2;++j) {
      peak_velocity=std::max(peak_velocity,std::abs(d->qvel[j]));
      peak_acceleration=std::max(peak_acceleration,std::abs(d->qacc[j]));
      peak_torque=std::max(peak_torque,std::abs(d->qfrc_actuator[j]));
      require(std::abs(d->actuator_force[2+j])<1e-12,"Position actuator was not disabled");
    }
    saturated+=std::abs(d->ctrl[0])>=12-1e-10 || std::abs(d->ctrl[1])>=12-1e-10;
    out << d->time << ',' << s;
    for (const Vec& v:{ref.q,ref.dq,ref.ddq}) for (double x:v) out << ',' << x;
    for (const auto* array:{d->qpos,d->qvel,d->qacc,d->qfrc_actuator})
      for (int j=0;j<2;++j) out << ',' << array[j];
    out << ',' << error << ',' << p[0] << ',' << p[1] << ',' << pr[0] << ',' << pr[1]
        << ',' << pc[0] << ',' << pc[1] << '\n';
    if (tick<Experiment::total_steps) sim.step();
  }
  const auto* d=sim.data.get();
  const double final_error=std::hypot(goal[0]-d->qpos[0],goal[1]-d->qpos[1]);
  const double rms=std::sqrt(squared_error/(Experiment::total_steps+1));
  require(final_error<.005,"Trajectory failed to settle within 5 mrad");
  require(max_error<.15,"Tracking error unexpectedly large");
  summary << name(profile) << ',' << rms << ',' << max_error << ',' << final_error << ','
          << peak_velocity << ',' << peak_acceleration << ',' << peak_torque << ',' << saturated << '\n';
  std::cout << name(profile) << ": RMS error=" << rms << " rad, max=" << max_error
            << ", final=" << final_error << ", saturated samples=" << saturated << '\n';
}
}  // namespace

int main() {
  try {
    std::filesystem::create_directories("build/trajectories");
    validate();
    trajectory::CallbackGuard callback;
    std::ofstream summary("build/trajectory_summary.csv");
    summary << "profile,rms_error_rad,max_error_rad,final_error_rad,peak_actual_velocity_rad_s,peak_actual_acceleration_rad_s2,peak_torque_Nm,saturated_samples\n"
            << std::setprecision(12);
    std::ofstream joins("build/trajectory_boundaries.csv");
    joins << "profile,boundary,time,ddq1_left,ddq1_right,ddq2_left,ddq2_right\n";
    for (auto profile:{trajectory::Profile::Cubic,trajectory::Profile::Quintic}) {
      run(profile,summary);
      const auto a=trajectory::reference(trajectory::hold_before,profile);
      const auto b=trajectory::reference(trajectory::hold_before+trajectory::duration,profile);
      joins << trajectory::name(profile) << ",start,0.5,0," << a.ddq[0] << ",0," << a.ddq[1] << '\n';
      joins << trajectory::name(profile) << ",end,2.5," << b.ddq[0] << ",0," << b.ddq[1] << ",0\n";
    }
    std::cout << "All MuJoCo tracking checks passed. See build/trajectories/ and trajectory_*.csv.\n";
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
