#include "reaching_pipeline.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>

namespace {
using namespace reaching;
void require(bool okay,const std::string& message) { if (!okay) throw std::runtime_error(message); }

/**
 * @brief Check the mathematical pipeline independently of plotted behavior.
 * Validate world/base IK mapping, infeasibility rejection, nearest branch,
 * quintic endpoints, scalar clipping and exact unsaturated computed-torque
 * error acceleration: ddq_d-FD(q,dq,tau)=-Kp*e-Kd*ed.
 * Sensor tests compare the empirical derivative-noise variance to 2*sigma_q^2/Tc^2
 * over 20000 rest measurements (5% finite-sample tolerance). Reset tests verify
 * seeded observation reproduction. Timing tests check update counts and that
 * repeated rendering-style refreshes consume no samples or change held torque.
 */
void validate() {
  const auto p=make_plan(default_target);
  const auto endpoint=trajectory::tip(p.goal);
  require(std::hypot(endpoint[0]-default_target[0],endpoint[1]-default_target[2])<1e-12,"World IK/FK mismatch");
  require(p.goal[1]<0,"Nearest elbow branch changed unexpectedly");
  for (const Point target:std::vector<Point>{{2,0,1},{.55,.1,1.25},{0,0,1}}) {
    bool rejected=false;
    try { make_plan(target); } catch (const std::invalid_argument&) { rejected=true; }
    require(rejected,"Infeasible target accepted");
  }
  Config c;
  const auto a=reference(delay,p,c),b=reference(delay+p.duration,p,c);
  require(error_norm(a.q,p.start)<1e-12 && error_norm(b.q,p.goal)<1e-12 &&
    error_norm(a.dq,{0,0})==0 && error_norm(b.ddq,{0,0})==0,"Reference endpoint failure");
  c=configuration("computed_torque"); c.torque_limit=10000;
  const Vec q={.4,-.7},v={.2,-.3};
  const auto desired=reference(delay+.4,p,c);
  const auto effort=command(desired,q,v,c);
  const auto acceleration=dynamics::forward(q,v,effort.raw);
  for (int i=0;i<2;++i)
    require(std::abs(desired.ddq[i]-acceleration[i]+c.kp[i]*(desired.q[i]-q[i])+
      c.kd[i]*(desired.dq[i]-v[i]))<1e-10,"Computed-torque error dynamics failed");
  c.torque_limit=.1;
  const auto limited=command(desired,q,v,c);
  require(std::abs(limited.applied[0])<=.1 && std::abs(limited.applied[1])<=.1,"Torque clipping failed");
  c=configuration("noisy_difference"); SensorModel sensor(9);
  const auto first=sensor.sample({0,0},{0,0},c);
  sensor.reset(9);
  const auto repeated=sensor.sample({0,0},{0,0},c);
  require(first.q==repeated.q && first.v==repeated.v,"Sensor reset not reproducible");
  double squared=0;
  for (int i=0;i<20000;++i) {
    const auto observation=sensor.sample({0,0},{0,0},c);
    squared+=observation.v[0]*observation.v[0];
  }
  const double predicted=2*c.q_noise*c.q_noise/std::pow(c.period_steps*dt,2);
  require(std::abs(squared/20000/predicted-1)<.05,"Position-difference noise variance mismatch");
  Experiment clock_test(configuration("rate_250hz"),p);
  const auto saved=clock_test.observation.q;
  clock_test.refresh(); clock_test.refresh();
  require(clock_test.updates==1 && saved==clock_test.observation.q,"Refresh resampled sensors");
  for (int tick=1;tick<=20;++tick) {
    const auto held=clock_test.effort.applied;
    clock_test.step();
    require(clock_test.updates==1+tick/4,"Controller timing drift");
    if (tick%4) require(clock_test.effort.applied==held,"Torque changed between updates");
  }
  std::cout << "IK, reference, inverse dynamics, sensor noise and held-control checks passed.\n";
}

struct Metrics {
  double rms=0,maximum=0,final_tip=0,tail_rms=0,overshoot=0,saturation=0,variation=0;
  double settled=std::numeric_limits<double>::quiet_NaN();
  int outside=0;
};

/**
 * @brief Measure tracking, final reach, saturation and noisy control effort.
 * Joint RMS=sqrt(mean_t ||q_ref-q_true||^2), never measured/noisy error. Final
 * tip error is ||p_true-p_target|| in world x-z metres. Tail RMS uses the last
 * 0.5 s. Overshoot=max_i,t max(0,sign(delta_i)*(q_i-q_goal_i)/|delta_i|)*100%,
 * skipping stationary joints: this is goal overshoot, not an exact linear-unit
 * response fit. Torque variation=sum_control_ticks ||tau[k]-tau[k-1]||_2 (N m).
 * Saturation fraction counts 1 ms intervals with any clipped component.
 * Settling is horizon-limited: earliest time after reference arrival with all
 * remaining samples inside 0.01 rad joint error AND 0.03 rad/s speed, requiring
 * at least 0.2 s observed dwell. NaN means not demonstrated; this is not the
 * book's asymptotic 2% settling definition. Actual planning-box departures are
 * counted, even though the MuJoCo hinges deliberately have no hard stops.
 */
Metrics run(const Config& config,const Plan& plan,unsigned seed,std::ofstream& summary) {
  Experiment sim(config,plan,seed);
  const std::string run_name=config.name+"_seed"+std::to_string(seed);
  std::ofstream out("build/classical_runs/"+run_name+".csv");
  out << "time,control_time,q1_ref,q2_ref,dq1_ref,dq2_ref,ddq1_ref,ddq2_ref,q1,q2,dq1,dq2,q1_measured,q2_measured,dq1_measured,dq2_measured,tau1_raw,tau2_raw,tau1,tau2,error_norm,tip_x,tip_z,ref_tip_x,ref_tip_z,saturated,in_planning_box\n"
      << std::setprecision(16);
  Metrics metrics;
  double squared=0,tail_squared=0,last_bad=-dt;
  int tail_count=0,saturated_intervals=0;
  const double arrival=config.jump?delay:delay+plan.duration;
  int old_updates=sim.updates;
  Vec old_torque=sim.effort.applied;
  const int tip_id=mj_name2id(sim.model.get(),mjOBJ_SITE,"tip");
  require(tip_id>=0,"Missing tip site");
  for (int tick=0;tick<=plan.total_steps;++tick) {
    auto* d=sim.data.get();
    const double time=tick*dt;
    const auto desired=reference(time,plan,config);
    const Vec q={d->qpos[0],d->qpos[1]},v={d->qvel[0],d->qvel[1]};
    const double error=error_norm(desired.q,q);
    require(std::isfinite(error) && std::isfinite(v[0]) && std::isfinite(v[1]),"Nonfinite simulation: "+run_name);
    const Vec point=trajectory::tip(q),desired_point=trajectory::tip(desired.q);
    require(std::abs(point[0]-d->site_xpos[3*tip_id])<1e-9 &&
      std::abs(point[1]-d->site_xpos[3*tip_id+2])<1e-9,"MuJoCo/FK frame mismatch");
    const bool clipped=sim.effort.raw!=sim.effort.applied;
    const bool inside=cartesian::inside(q,cartesian::kPhysicalLimits);
    for (int i=0;i<2;++i) {
      require(std::abs(d->actuator_force[2+i])<1e-12,"Inactive servo force");
      require(std::abs(d->qfrc_actuator[i]-sim.effort.applied[i])<1e-12,"Applied torque mismatch");
      const double delta=plan.goal[i]-plan.start[i];
      if (std::abs(delta)>1e-12)
        metrics.overshoot=std::max(metrics.overshoot,100*(q[i]-plan.goal[i])/delta);
    }
    squared+=error*error; metrics.maximum=std::max(metrics.maximum,error);
    metrics.outside+=!inside;
    if (tick<plan.total_steps) saturated_intervals+=clipped;
    if (time>=plan.total_steps*dt-.5) { tail_squared+=error*error; ++tail_count; }
    if (time>=arrival && (error_norm(q,plan.goal)>.01 || std::hypot(v[0],v[1])>.03)) last_bad=time;
    if (old_updates!=sim.updates) {
      metrics.variation+=error_norm(sim.effort.applied,old_torque);
      old_torque=sim.effort.applied; old_updates=sim.updates;
    }
    out << time << ',' << sim.control_time;
    for (const Vec& values:{desired.q,desired.dq,desired.ddq,q,v,sim.observation.q,sim.observation.v,
                           sim.effort.raw,sim.effort.applied})
      for (double value:values) out << ',' << value;
    out << ',' << error << ',' << point[0] << ',' << point[1] << ',' << desired_point[0]
        << ',' << desired_point[1] << ',' << clipped << ',' << inside << '\n';
    if (tick<plan.total_steps) sim.step();
  }
  metrics.rms=std::sqrt(squared/(plan.total_steps+1));
  metrics.tail_rms=std::sqrt(tail_squared/tail_count);
  metrics.saturation=double(saturated_intervals)/plan.total_steps;
  const Vec actual=trajectory::tip({sim.data->qpos[0],sim.data->qpos[1]});
  metrics.final_tip=std::hypot(actual[0]-plan.world_target[0],actual[1]-plan.world_target[2]);
  const double band_start=std::max(arrival,last_bad+dt);
  if (plan.total_steps*dt-band_start>=.2) metrics.settled=band_start-arrival;
  summary << config.name << ',' << seed << ',' << metrics.rms << ',' << metrics.maximum << ','
    << metrics.final_tip << ',' << metrics.tail_rms << ',' << metrics.overshoot << ',' << metrics.saturation
    << ',' << metrics.variation << ',' << metrics.settled << ',' << metrics.outside << ','
    << 1/(config.period_steps*dt) << ',' << config.torque_limit << ',' << config.q_noise << ',' << config.v_noise << '\n';
  std::cout << run_name << ": RMS=" << metrics.rms << " rad, final tip=" << metrics.final_tip
            << " m, saturation=" << 100*metrics.saturation << "%\n";
  return metrics;
}

void help() {
  std::cout << "Usage: ./build/classical_reaching [--case NAME] [--seed N] [--target X Y Z]\n"
    "Default: all cases; noisy cases use seeds 0..9. A selected case uses seed 0 unless supplied.\n"
    "Target is a world position in metres (default 0.55 0 1.25). Use --list to see cases.\n";
  for (const auto& c:configurations()) std::cout << "  " << c.name << '\n';
}
}  // namespace

int main(int argc,char** argv) {
  try {
    Point target=default_target; std::string selected; unsigned seed=0; bool supplied_seed=false;
    for (int i=1;i<argc;++i) {
      const std::string arg=argv[i];
      if (arg=="--help" || arg=="--list") { help(); return 0; }
      if (arg=="--case" && i+1<argc) selected=argv[++i];
      else if (arg=="--seed" && i+1<argc) {
        const std::string value=argv[++i];
        if (value.empty() || value.find_first_not_of("0123456789")!=std::string::npos)
          throw std::invalid_argument("Seed must be a nonnegative integer");
        const auto n=std::stoul(value);
        if (n>std::numeric_limits<unsigned>::max()) throw std::invalid_argument("Seed too large");
        seed=static_cast<unsigned>(n); supplied_seed=true;
      } else if (arg=="--target" && i+3<argc) {
        for (double& x:target) { std::size_t used; const std::string value=argv[++i]; x=std::stod(value,&used);
          if (used!=value.size()) throw std::invalid_argument("Invalid target coordinate"); }
      } else throw std::invalid_argument("Unknown/incomplete argument: "+arg);
    }
    if (supplied_seed && selected.empty()) throw std::invalid_argument("Use --seed with --case");
    const Plan plan=make_plan(target);
    const auto cases=selected.empty()?configurations():std::vector<Config>{configuration(selected)};
    std::filesystem::create_directories("build/classical_runs");
    HeldControlGuard guard;
    validate();
    std::ofstream summary("build/classical_reaching_summary.csv");
    summary << "case,seed,rms_error_rad,max_error_rad,final_tip_error_m,tail_rms_rad,goal_overshoot_percent,saturation_fraction,torque_variation_Nm,settling_after_arrival_s,outside_planning_samples,control_hz,torque_limit_Nm,q_noise_rad,v_noise_rad_s\n"
            << std::setprecision(12);
    std::ofstream planning("build/classical_reaching_plan.csv");
    planning << "target_x,target_y,target_z,q1_start,q2_start,q1_goal,q2_goal,duration_s,end_time_s\n"
      << std::setprecision(16) << target[0] << ',' << target[1] << ',' << target[2] << ','
      << plan.start[0] << ',' << plan.start[1] << ',' << plan.goal[0] << ',' << plan.goal[1] << ','
      << plan.duration << ',' << plan.total_steps*dt << '\n';
    for (const auto& config:cases) {
      const int repetitions=selected.empty() && config.q_noise>0?10:1;
      for (int n=0;n<repetitions;++n) {
        const auto metrics=run(config,plan,selected.empty()?unsigned(n):seed,summary);
        if (config.name=="baseline" && target==default_target)
          require(metrics.final_tip<.002 && metrics.rms<.05 && metrics.outside==0,"Default milestone reach failed");
      }
    }
    std::cout << "Complete. World target -> IK -> quintic reference -> sampled torque control -> MuJoCo.\n"
      << "CSV files: build/classical_runs/ and build/classical_reaching_{summary,plan}.csv\n";
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
