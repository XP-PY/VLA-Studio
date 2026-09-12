/* Week 6: qdot_cmd = inverse(Jp_world) * gain * (target_world - tip_world).
   Compare ideal velocity integration and actual MuJoCo velocity-servo dynamics.
   The geometric IK oracle only rejects infeasible targets; it never steers the arm. */
#include "mujoco.h"
#include "cartesian_control.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace {
using namespace cartesian;
using namespace transform_utils;
constexpr int kSubsteps = 5;
constexpr double kDuration = 8.0;
constexpr double kSuccessDistance = 0.001;
constexpr double kMaximumSpeed = 1.5;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
Matrix4 pose(const mjtNum* p, const mjtNum* r) {
  return make_transform({r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7],r[8]}, {p[0],p[1],p[2]});
}
struct Robot {
  std::unique_ptr<mjModel,decltype(&mj_deleteModel)> model{nullptr,mj_deleteModel};
  std::unique_ptr<mjData,decltype(&mj_deleteData)> data{nullptr,mj_deleteData};
  int base, site, mocap;
  std::array<int,2> qa{}, va{}, actuator{};
  Limits physical{}, safe{};
  Matrix4 world_base{};
  double dt;

  Robot() {
    char message[1024]{};
    model.reset(mj_loadXML("assets/cartesian_reaching.xml",nullptr,message,sizeof(message)));
    require(model != nullptr,std::string("Cannot load model: ")+message);
    data.reset(mj_makeData(model.get()));
    require(data != nullptr,"Cannot allocate mjData");
    auto* m = model.get(); auto* d = data.get();
    require(m->nq == 2 && m->nv == 2 && m->nu == 2,"Expected fixed-base two-hinge arm");
    base = mj_name2id(m,mjOBJ_BODY,"base");
    site = mj_name2id(m,mjOBJ_SITE,"end_effector");
    const int target = mj_name2id(m,mjOBJ_BODY,"target");
    require(base >= 0 && site >= 0 && target >= 0,"Missing model body/site");
    mocap = m->body_mocapid[target];
    require(mocap >= 0,"Target must be a mocap body");
    const char* joint_names[] = {"shoulder","elbow"};
    const char* actuator_names[] = {"shoulder_velocity","elbow_velocity"};
    for (int i = 0; i < 2; ++i) {
      const int joint = mj_name2id(m,mjOBJ_JOINT,joint_names[i]);
      actuator[i] = mj_name2id(m,mjOBJ_ACTUATOR,actuator_names[i]);
      require(joint >= 0 && actuator[i] >= 0,"Missing joint/actuator");
      require(m->jnt_type[joint] == mjJNT_HINGE && m->jnt_limited[joint],"Expected limited hinge");
      qa[i] = m->jnt_qposadr[joint]; va[i] = m->jnt_dofadr[joint];
      physical.lower[i] = m->jnt_range[2*joint]; physical.upper[i] = m->jnt_range[2*joint+1];
      require(physical.lower[i] >= -kPi && physical.upper[i] <= kPi,
              "Geometric oracle assumes joint ranges inside [-pi,pi]");
      const int a = actuator[i];
      require(m->actuator_trnid[2*a] == joint && m->actuator_gear[6*a] == 1 &&
              m->actuator_biasprm[10*a+1] == 0 && m->actuator_gainprm[10*a] > 0 &&
              m->actuator_biasprm[10*a+2] == -m->actuator_gainprm[10*a] &&
              m->actuator_ctrlrange[2*a] <= -kMaximumSpeed &&
              m->actuator_ctrlrange[2*a+1] >= kMaximumSpeed,
              "Expected unit-gear velocity servos with sufficient control range");
    }
    safe = guarded(physical);
    dt = kSubsteps*m->opt.timestep;
    require(dt > 0 && dt <= 0.02,"Expected a short positive control period");
    mj_forward(m,d);
    world_base = pose(d->xpos+3*base,d->xmat+9*base);
  }
  Joints q() const { return {data->qpos[qa[0]],data->qpos[qa[1]]}; }
  Joints dq() const { return {data->qvel[va[0]],data->qvel[va[1]]}; }
  Vector3 tip_world() const {
    const auto* p = data->site_xpos+3*site;
    return {p[0],p[1],p[2]};
  }
  Vector3 tip_base() const { return transform_point(inverse_transform(world_base),tip_world()); }
  void reset(const Scenario& scenario) {
    require(inside(scenario.initial,safe),"Initial configuration is outside guarded joint limits");
    mj_resetData(model.get(),data.get());
    for (int i = 0; i < 2; ++i) data->qpos[qa[i]] = scenario.initial[i];
    const auto target = transform_point(world_base,scenario.target);
    for (int i = 0; i < 3; ++i) data->mocap_pos[3*mocap+i] = target[i];
    mj_forward(model.get(),data.get());
    require(norm(error(tip_base(),position(q()))) < 1e-12,"MJCF and analytical geometry disagree");
  }
  PointJacobian jacobian() const {
    std::array<mjtNum,6> jacp{};
    mj_jacSite(model.get(),data.get(),jacp.data(),nullptr,site);
    PointJacobian result{};
    const auto analytical = point_jacobian(q()[0],q()[1]);
    for (int col = 0; col < 2; ++col) {
      for (int row = 0; row < 3; ++row) result[col][row] = jacp[row*model->nv+va[col]];
      require(norm(error(result[col],rotate(rotation_of(world_base),analytical[col]))) < 1e-12,
              "MuJoCo/analytical world Jacobian mismatch");
    }
    return result;
  }
};

struct Metrics {
  std::string status;
  double elapsed = 0, first_hit = -1, final_error = 0, integrated_error = 0;
  double peak_raw = 0, peak_command = 0, peak_actual = 0, peak_torque = 0, limit_violation = 0;
  int speed_limited = 0, limit_limited = 0;
};

Metrics reach(Robot& robot, const Scenario& scenario, const std::string& mode,
              double damping, double gain, const std::string& id) {
  robot.reset(scenario);
  Metrics stats;
  stats.final_error = norm(error(scenario.target,robot.tip_base()));
  const auto physical = geometric_ik(scenario.target,robot.physical);
  if (physical.status != "reachable") { stats.status = "rejected_"+physical.status; return stats; }
  if (geometric_ik(scenario.target,robot.safe).status != "reachable") {
    stats.status = "rejected_guard_margin"; return stats;
  }
  std::ofstream log("build/reaching/"+id+".csv");
  require(bool(log),"Cannot create reaching trajectory");
  log << std::setprecision(17)
      << "tick,time_s,q1,q2,dq1_actual,dq2_actual,tip_base_x,tip_base_y,tip_base_z,"
         "target_base_x,target_base_y,target_base_z,error_m,raw_dq1,raw_dq2,command_dq1,command_dq2,"
         "sigma_max,sigma_min,linear_residual_m_s,speed_limited,limit_limited\n";
  const auto target_world = transform_point(robot.world_base,scenario.target);
  auto* m = robot.model.get(); auto* d = robot.data.get();
  int settled_ticks = 0;
  const int maximum_ticks = static_cast<int>(std::round(kDuration/robot.dt));
  const auto monitor = [&]() {
    const auto q = robot.q(), dq = robot.dq();
    require(inside(q,robot.physical,1e-6),"Physical joint limit crossed");
    require(std::isfinite(dq[0]) && std::isfinite(dq[1]),"Nonfinite measured velocity");
    stats.peak_actual = std::max(stats.peak_actual,max_joint(dq));
    for (int i = 0; i < 2; ++i) {
      stats.limit_violation = std::max({stats.limit_violation,
          robot.physical.lower[i]-q[i],q[i]-robot.physical.upper[i]});
      if (mode == "dynamics")
        stats.peak_torque = std::max(stats.peak_torque,std::abs(d->actuator_force[robot.actuator[i]]));
    }
  };
  for (int tick = 0; tick <= maximum_ticks; ++tick) {
    // mj_step's derived poses can belong to its pre-integration state.
    // Refresh before measuring the new state and calculating feedback.
    mj_forward(m,d);
    monitor();
    const auto q = robot.q(), dq = robot.dq();
    const auto current = robot.tip_base();
    const auto e_world = error(target_world,robot.tip_world());
    const auto j = robot.jacobian();
    const auto desired_velocity = scale(e_world,gain);
    const auto inverse = inverse_velocity(j,desired_velocity,damping);
    require(std::isfinite(inverse.dq[0]) && std::isfinite(inverse.dq[1]),"Nonfinite inverse output");
    const auto limited = limit_velocity(inverse.dq,q,robot.safe,kMaximumSpeed,robot.dt);
    stats.elapsed = tick*robot.dt;
    stats.final_error = norm(e_world);
    if (stats.final_error < kSuccessDistance && stats.first_hit < 0) stats.first_hit = stats.elapsed;
    settled_ticks = stats.final_error < kSuccessDistance && joint_norm(dq) < 0.02 ? settled_ticks+1 : 0;
    log << tick << ',' << stats.elapsed << ',' << q[0] << ',' << q[1] << ',' << dq[0] << ',' << dq[1];
    for (double x : current) log << ',' << x;
    for (double x : scenario.target) log << ',' << x;
    log << ',' << stats.final_error << ',' << inverse.dq[0] << ',' << inverse.dq[1]
        << ',' << limited.dq[0] << ',' << limited.dq[1] << ',' << inverse.sigma[0] << ',' << inverse.sigma[1]
        << ',' << norm(error(desired_velocity,apply(j,inverse.dq))) << ',' << limited.speed_limited
        << ',' << limited.limit_limited << '\n';
    // N consecutive observations span N-1 controller intervals.
    if ((settled_ticks-1)*robot.dt >= 0.25-1e-12) { stats.status = "converged"; break; }
    if (stats.final_error > kSuccessDistance && joint_norm(inverse.dq) < 1e-10 && joint_norm(dq) < 1e-8) {
      stats.status = "local_stationary"; break;
    }
    if (tick == maximum_ticks) { stats.status = "time_limit"; break; }
    stats.peak_raw = std::max(stats.peak_raw,max_joint(inverse.dq));
    stats.peak_command = std::max(stats.peak_command,max_joint(limited.dq));
    stats.speed_limited += limited.speed_limited;
    stats.limit_limited += limited.limit_limited;
    stats.integrated_error += stats.final_error*robot.dt;
    require(max_joint(limited.dq) <= kMaximumSpeed+1e-12,"Velocity command exceeds bound");

    if (mode == "kinematic") {
      // Ideal velocity tracking: discretized differential IK, not dynamics.
      // No qpos clipping is needed: the approach limiter bounds this step.
      for (int i = 0; i < 2; ++i) {
        d->qpos[robot.qa[i]] += robot.dt*limited.dq[i];
        d->qvel[robot.va[i]] = limited.dq[i];
      }
      require(inside(robot.q(),robot.safe),"Ideal integration crossed guarded limit");
      d->time += robot.dt;
    } else {
      // ctrl is a velocity reference; the actuators generate torque. Actual
      // positions and velocities here change ONLY through mj_step.
      for (int i = 0; i < 2; ++i) d->ctrl[robot.actuator[i]] = limited.dq[i];
      const double start = d->time;
      for (int step = 0; step < kSubsteps; ++step) { mj_step(m,d); monitor(); }
      require(std::abs(d->time-start-robot.dt) < 1e-10,"Unexpected simulation reset/time increment");
    }
  }
  for (int warning = 0; warning < mjNWARNING; ++warning)
    require(d->warning[warning].number == 0,"MuJoCo issued a simulation warning");
  log.flush(); require(bool(log),"Failed writing reaching trajectory");
  return stats;
}

struct Options { std::string scenario, mode, method; double gain = 0; };
Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    require(i+1 < argc,"Missing value for "+key);
    const std::string value = argv[++i];
    if (key == "--case") options.scenario = value;
    else if (key == "--mode") options.mode = value;
    else if (key == "--method") options.method = value;
    else if (key == "--gain") {
      std::size_t count = 0;
      options.gain = std::stod(value,&count);
      require(count == value.size() && std::isfinite(options.gain) && options.gain > 0,"Gain must be positive and finite");
    } else throw std::runtime_error("Unknown option: "+key);
  }
  require(options.mode.empty() || options.mode == "kinematic" || options.mode == "dynamics","Invalid mode");
  require(options.method.empty() || options.method == "pinv" || options.method == "dls","Invalid method");
  bool found = options.scenario.empty();
  for (const auto& scenario : scenarios()) found |= scenario.name == options.scenario;
  require(found,"Unknown scenario (use --help)");
  return options;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--help") {
      std::cout << "Usage: ./build/cartesian_reaching [--case NAME] [--mode kinematic|dynamics]\n"
                   "       [--method pinv|dls] [--gain POSITIVE_NUMBER]\n"
                   "Default: all cases, both modes/methods, gains 1/4/12 per second.\n"
                   "Each invocation replaces the summary; only selected trajectories are rewritten.\nCases:\n";
      for (const auto& scenario : scenarios()) std::cout << "  " << scenario.name << '\n';
      return 0;
    }
    const auto options = parse_options(argc,argv);
    Robot robot;
    std::filesystem::create_directories("build/reaching");
    std::ofstream summary("build/cartesian_reaching_summary.csv"), report("build/cartesian_reaching_report.txt");
    require(summary && report,"Cannot create reaching reports");
    summary << std::setprecision(17) << "run,case,mode,method,gain_s_inv,lambda,status,target_base_x,target_base_y,"
               "target_base_z,elapsed_s,first_hit_s,final_error_m,integrated_error_m_s,peak_raw_rad_s,"
               "peak_command_rad_s,peak_actual_rad_s,peak_torque_nm,speed_limited_ticks,"
               "limit_limited_ticks,max_limit_violation_rad\n";
    report << "Week 6 Cartesian reaching: 1 mm position tolerance; speed below 0.02 rad/s\n"
           << "for 0.25 s to settle; timeout " << kDuration << " s; control dt " << robot.dt
           << " s; physics dt " << robot.model->opt.timestep << " s.\n"
           << "Pseudoinverse relative cutoff " << kRelativeCutoff << "; DLS lambda " << kDamping
           << "; max command " << kMaximumSpeed << " rad/s; joint guard margin " << kLimitMargin << " rad.\n\n";
    int total = 0, converged = 0, rejected = 0, stationary = 0, timeouts = 0;
    for (const auto& scenario : scenarios()) {
      if (!options.scenario.empty() && scenario.name != options.scenario) continue;
      for (const std::string mode : {"kinematic","dynamics"}) {
        if (!options.mode.empty() && mode != options.mode) continue;
        for (const std::string method : {"pinv","dls"}) {
          if (!options.method.empty() && method != options.method) continue;
          const std::vector<double> gains = options.gain > 0 ? std::vector<double>{options.gain}
                                                             : std::vector<double>{1,4,12};
          for (double gain : gains) {
            const double damping = method == "pinv" ? 0 : kDamping;
            std::ostringstream identifier;
            identifier << scenario.name << '_' << mode << '_' << method << "_k" << std::setprecision(17) << gain;
            const std::string id = identifier.str();
            const auto stats = reach(robot,scenario,mode,damping,gain,id);
            ++total;
            converged += stats.status == "converged";
            rejected += stats.status.rfind("rejected_",0) == 0;
            stationary += stats.status == "local_stationary";
            timeouts += stats.status == "time_limit";
            if (scenario.name == "exact_singular_start")
              require(stats.status == "local_stationary","Exact singularity must report local stationarity");
            // Regression expectation for the default, well-conditioned reaches.
            if (options.gain == 0 && scenario.name == "regular")
              require(stats.status == "converged","Default regular reach failed to settle");
            summary << id << ',' << scenario.name << ',' << mode << ',' << method << ',' << gain << ',' << damping
                    << ',' << stats.status;
            for (double x : scenario.target) summary << ',' << x;
            summary << ',' << stats.elapsed << ',' << stats.first_hit << ',' << stats.final_error
                    << ',' << stats.integrated_error << ',' << stats.peak_raw << ',' << stats.peak_command
                    << ',' << stats.peak_actual << ',' << stats.peak_torque << ',' << stats.speed_limited
                    << ',' << stats.limit_limited << ',' << stats.limit_violation << '\n';
            report << id << ": " << stats.status << ", error=" << stats.final_error
                   << " m, duration=" << stats.elapsed << " s, peak raw/command=" << stats.peak_raw
                   << '/' << stats.peak_command << " rad/s\n";
          }
        }
      }
    }
    report << "\nRuns=" << total << ", converged=" << converged << ", rejected=" << rejected
           << ", local_stationary=" << stationary << ", time_limit=" << timeouts
           << "\nAll executed trajectories passed finite-state, command, joint-limit and simulator-warning checks.\n"
           << "A local_stationary/time_limit result does not establish geometric unreachability.\n";
    summary.flush(); report.flush(); require(summary && report,"Failed writing reaching reports");
    std::cout << "Reaching experiment checks passed: " << total << " runs, " << converged << " converged, "
              << rejected << " rejected, " << stationary << " stationary, " << timeouts << " timed out.\n"
              << "Read build/cartesian_reaching_report.txt, build/cartesian_reaching_summary.csv and build/reaching/*.csv\n";
  } catch (const std::exception& exception) {
    std::cerr << "Error: " << exception.what() << '\n';
    return 1;
  }
}
