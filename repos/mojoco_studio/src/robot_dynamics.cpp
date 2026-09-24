#include "dynamics_experiments.h"
#include <mujoco/mujoco.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

using dynamics::Vec;
namespace {
using namespace experiments;
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

/**
 * @brief Cross-check independent rigid-body formulas and energy identities.
 * For seeded samples, compare M, c+g, forward qdd, and inverse tau against
 * MuJoCo. All actuators/constraints are disabled; applied torque is explicit.
 * Also check g_i approximately [P(q+h e_i)-P(q-h e_i)]/(2h) and
 * dq^T c approximately 1/2 dq^T [M(q+h*dq)-M(q-h*dq)]/(2h) dq.
 * The latter follows dK/dt=dq^T M qdd+1/2 dq^T Mdot dq and energy balance.
 * h=1e-6 is a central-difference choice, not an exact derivative. Testing with
 * and without viscous damping verifies MuJoCo inverse force includes the
 * torque required to overcome passive damping. Errors carry each column's
 * physical units; thresholds are numerical checks for this small model.
 */
void validate(mjModel* m, mjData* d) {
  // Disable all actuators. 
  // `opt.disableactuator` is the bit mask of actuator group
  // Binary: 3 = 11_2. Therefore, both group 0 and group 1 are disabled.
  m->opt.disableactuator=3;

  // Create a random test
  std::mt19937 rng(7001);
  std::uniform_real_distribution<double> sample(-2.5,2.5);

  // Record error
  // worst[0]: M
  // worst[1]: c+g
  // worst[2]: forward dynamics
  // worst[3]: inverse dynamics
  // worst[4]: gravity = ∂P/∂q
  // worst[5]: energy identity
  std::array<double,6> worst{};

  for (int n=0;n<200;++n) {
    // Reset mojoco
    mj_resetData(m,d);

    // Randomly generate joint angles, velocities, torques, and accelerations.
    Vec q={sample(rng),sample(rng)}, dq={sample(rng),sample(rng)};
    Vec tau={sample(rng),sample(rng)}, acc={sample(rng),sample(rng)};

    // Random viscous damping test
    Vec damping=n%2?Vec{.2,.1}:Vec{0,0};

    for (int i=0;i<2;++i) {
      d->qpos[i]=q[i]; 
      d->qvel[i]=dq[i]; 
      // Note that the actuator does not generate torque here.
      // Instead, external generalized forces are applied directly to 
      // the joints using `qfrc_applied`.
      d->qfrc_applied[i]=tau[i];
      m->dof_damping[i]=damping[i];
    }
    mj_forward(m,d);

    // MuJoCo uses a compressed structure when storing `M` internally.
    // `mj_fullM()` expands it.
    mjtNum full[4]; mj_fullM(m,d,full);

    const auto M=dynamics::mass(q);                   // M: inertia/mass matrix
    const Vec bias=dynamics::inverse(q,dq,{0,0});     // c + g
    const Vec fd=dynamics::forward(q,dq,tau,damping); // acc: forward dynamics
    const Vec id=dynamics::inverse(q,dq,acc,damping); // tau: inverse dynamics

    // Compare the self-calculated inertia matrix `dynamics::mass` with that of MuJoCo.
    for (int i=0;i<4;++i) worst[0]=std::max(worst[0],std::abs(M[i]-full[i]));
    for (int i=0;i<2;++i) {
      // Compare the self-calculated c + g with that of MuJoCo.
      worst[1]=std::max(worst[1],std::abs(bias[i]-d->qfrc_bias[i]));
      // Compare the self-calculated acceleration with that of MuJoCo.
      worst[2]=std::max(worst[2],std::abs(fd[i]-d->qacc[i]));
      // Vertify gravity = ∂P/∂q using Central difference
      d->qacc[i]=acc[i];
      Vec plus=q,minus=q; plus[i]+=1e-6; minus[i]-=1e-6;
      const double grad=(dynamics::potential(plus)-dynamics::potential(minus))/2e-6;
      worst[4]=std::max(worst[4],std::abs(grad-dynamics::gravity_torque(q)[i]));
    }
    // Compare the self-calculated torque with that of MuJoCo.
    mj_inverse(m,d);
    for (int i=0;i<2;++i) worst[3]=std::max(worst[3],std::abs(id[i]-d->qfrc_inverse[i]));
    // Verify the energy properties of the velocity term.
    Vec plus=q,minus=q;
    for (int i=0;i<2;++i) { plus[i]+=1e-6*dq[i]; minus[i]-=1e-6*dq[i]; }
    auto derivative=dynamics::mass(plus); const auto lower=dynamics::mass(minus);
    for (int i=0;i<4;++i) derivative[i]=(derivative[i]-lower[i])/2e-6;
    worst[5]=std::max(worst[5],std::abs(dynamics::dot(dq,dynamics::coriolis(q,dq))-
                                 .5*dynamics::dot(dq,dynamics::apply(derivative,dq))));
  }
  std::ofstream out("build/dynamics_validation.csv");
  out << "quantity,max_absolute_error\n" << std::setprecision(16);
  const char* names[]={"mass_kg_m2","bias_Nm","forward_rad_s2","inverse_Nm","potential_gradient_Nm","power_identity_W"};
  for (int i=0;i<6;++i) {
    out << names[i] << ',' << worst[i] << '\n';
    require(std::isfinite(worst[i]) && worst[i]<1e-7,std::string("Dynamics validation failed: ")+names[i]);
  }
  std::cout << "200 states: mass/bias/forward/inverse and energy identities passed.\n";
}

struct Result {
  double initial_energy=0,final_energy=0,max_energy_change=0,final_error=0,max_motion=0;
  std::vector<Vec> positions;
};

/**
 * @brief Integrate one six-second experiment and measure its physical outcome.
 * Samples are stored every 0.01 s; the integrator step is 0.001 s. Mechanical
 * energy uses the shoulder as height zero; final_error=||q_des-q(6)||_2 in rad
 * is meaningful as a regulation metric only for the three servo modes.
 * max_motion=max_t ||q(t)-q(0)||_2 distinguishes static gravity balance from
 * velocity regulation. Joint damping contributes passive torque -D*dq.
 * No collision, limit, friction or armature forces obscure this comparison.
 */
Result run(mjModel* m, mjData* d, Mode selected, const std::string& name) {
  experiments::reset(m,d,selected);
  std::ofstream out("build/dynamics/"+name+".csv");
  out << "time,q1,q2,dq1,dq2,tau1,tau2,energy,target_error\n" << std::setprecision(16);
  Result result;
  for (int step=0;step<=experiments::steps;++step) {
    // Refresh derived quantities at the accepted RK4 state before logging.
    mj_forward(m,d);
    Vec q={d->qpos[0],d->qpos[1]}, dq={d->qvel[0],d->qvel[1]};
    const double e=dynamics::energy(q,dq);
    require(std::isfinite(e),"Non-finite simulation state");
    if (step==0) result.initial_energy=e;
    result.final_energy=e;
    result.max_energy_change=std::max(result.max_energy_change,std::abs(e-result.initial_energy));
    result.max_motion=std::max(result.max_motion,std::hypot(q[0]-.6,q[1]+.8));
    result.final_error=std::hypot(target[0]-q[0],target[1]-q[1]);
    const int disabled_start=selected==Mode::Position?0:2;
    for (int i=disabled_start;i<disabled_start+2;++i)
      require(std::abs(d->actuator_force[i])<1e-12,"Disabled actuator produced force");
    if (step%10==0) {
      out << d->time << ',' << q[0] << ',' << q[1] << ',' << dq[0] << ',' << dq[1]
          << ',' << d->qfrc_actuator[0] << ',' << d->qfrc_actuator[1] << ',' << e
          << ',' << result.final_error << '\n';
      result.positions.push_back(q);
    }
    if (step<experiments::steps) mj_step(m,d);
  }
  return result;
}
}  // namespace

int main() {
  try {
    std::filesystem::create_directories("build/dynamics");
    char error[1024];
    std::unique_ptr<mjModel,decltype(&mj_deleteModel)> m(
        mj_loadXML("assets/dynamics_arm.xml",nullptr,error,sizeof(error)),mj_deleteModel);
    require(m!=nullptr,error);
    // Two joint positions, two joint velocities and four actuators
    require(m->nq==2 && m->nv==2 && m->nu==4,"Unexpected model dimensions");
    std::unique_ptr<mjData,decltype(&mj_deleteData)> d(mj_makeData(m.get()),mj_deleteData);

    // Before the actual simulation, it verifies:
    //  Do the M, c, g terms, forward dynamics, and inverse dynamics 
    //  that we implemented ourselves match MuJoCo's exactly?
    validate(m.get(),d.get());

    // Register control callback
    experiments::CallbackGuard guard;

    std::vector<Result> results;
    std::ofstream summary("build/dynamics_summary.csv");
    summary << "mode,initial_energy_J,final_energy_J,max_energy_change_J,final_target_error_rad,max_motion_rad\n"
            << std::setprecision(12);
    for (const auto& experiment:experiments::descriptions) {
      const auto r=run(m.get(),d.get(),experiment.mode,experiment.name);
      results.push_back(r);
      summary << experiment.name << ',' << r.initial_energy << ',' << r.final_energy << ','
              << r.max_energy_change << ',' << r.final_error << ',' << r.max_motion << '\n';
      std::cout << experiment.name << ": final target error=" << r.final_error
                << " rad; energy change=" << r.final_energy-r.initial_energy << " J\n";
    }
    require(results[0].max_energy_change<1e-4,"Uncontrolled energy drift too large");
    require(results[1].final_energy<results[1].initial_energy-1,"Damping did not dissipate energy");
    require(results[3].max_motion<1e-9,"Static gravity compensation drift");
    double difference=0;
    for (std::size_t n=0;n<results[4].positions.size();++n)
      for (int j=0;j<2;++j) difference=std::max(difference,
          std::abs(results[4].positions[n][j]-results[5].positions[n][j]));
    require(difference<1e-9,"Position actuator and matched torque PD differ");
    require(results[6].final_error<1e-3,"PD + gravity did not settle");
    require(results[7].max_motion>.1,"Gravity compensation unexpectedly stopped motion");
    std::cout << "All controller checks passed; matched servo difference=" << difference
              << " rad. CSVs: build/dynamics/ and build/dynamics_{summary,validation}.csv\n";
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
