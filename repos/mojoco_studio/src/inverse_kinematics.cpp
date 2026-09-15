// Week 6: position-only numerical IK and independent geometric/SVD checks.
#include "cartesian_control.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>

namespace {
using namespace cartesian;
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
/**
 * @brief Check the inverse solver against independently known linear identities.
 *
 * The Chapter 6 minimum-norm rule has a simple rank-one example at q=(0,0):
 *      J = [0 0; 0.8 0.35; 0 0],
 *      J^dagger [0,1,0]^T = [0.8,0.35]^T / (0.8^2 + 0.35^2).
 * A purely radial desired velocity is orthogonal to the columns and must yield
 * zero. The zero-Jacobian and numerical-rank-cutoff checks cover related cases.
 *
 * The regularized least-squares objective adds two independent checks:
 *      J^T (desired - J dq) = lambda^2 dq       (normal equations)
 *      ||dq||_2 <= ||desired||_2 / (2 lambda)   (lambda > 0).
 * The bound follows from max_sigma sigma/(sigma^2+lambda^2) = 1/(2 lambda).
 * These validate damping's mathematical role, not just agreement between two
 * controller runs. Also verify a known distance-to-limit rate bound.
 *
 * Task velocities are m/s, joint rates rad/s, and J is m/rad. Failed identities
 * throw so the executable cannot silently report an invalid inverse as a pass.
 */
void check_inverse_solver() {
  const auto straight = point_jacobian(0,0);
  const auto tangent = inverse_velocity(straight,{0,1,0},0);
  const double squared_length = 0.8*0.8+0.35*0.35;
  require(std::abs(tangent.dq[0]-0.8/squared_length) < 1e-12 &&
          std::abs(tangent.dq[1]-0.35/squared_length) < 1e-12,
          "Rank-one minimum-norm inverse failed");
  require(joint_norm(inverse_velocity(straight,{1,0,0},0).dq) < 1e-12,
          "Unachievable radial motion should project to zero");
  require(joint_norm(inverse_velocity(PointJacobian{}, {1,2,3},0).dq) == 0,
          "Zero Jacobian pseudoinverse failed");
  require(joint_norm(inverse_velocity(PointJacobian{}, {1,2,3},kDamping).dq) == 0,
          "Zero Jacobian damped inverse failed");
  const auto almost_straight = point_jacobian(0,1e-4);
  const double raw = joint_norm(inverse_velocity(almost_straight,{-0.15,0,0},0).dq);
  const double damped = joint_norm(inverse_velocity(almost_straight,{-0.15,0,0},kDamping).dq);
  require(raw > 1000 && damped < raw*1e-3 && damped <= 0.15/(2*kDamping)+1e-12,
          "Near-singular damping comparison or DLS norm bound failed");
  const auto cutoff = inverse_velocity(point_jacobian(0,1e-12),{1,0,0},0);
  require(joint_norm(cutoff.dq) < 1e-10, "Numerical rank cutoff failed");

  // A fabricated Jacobian tests an out-of-plane task direction as well.
  const PointJacobian j = {{{1,2,3},{-2,1,1}}};
  const Vector3 desired = {0.3,-0.5,0.9};
  for (double lambda : {0.0,kDamping}) {
    const auto solution = inverse_velocity(j,desired,lambda);
    const auto remaining = error(desired,apply(j,solution.dq));
    for (int col = 0; col < 2; ++col)
      require(std::abs(dot(j[col],remaining)-lambda*lambda*solution.dq[col]) < 1e-12,
              "Least-squares normal equation failed");
  }
  const auto bounded = limit_velocity({100,-50},{1.47,-1.67},guarded(kPhysicalLimits),1.5,0.01);
  require(bounded.speed_limited && bounded.limit_limited &&
          std::abs(bounded.dq[0]-0.05) < 1e-12 && std::abs(bounded.dq[1]+0.05) < 1e-12,
          "Joint-limit approach limiter failed");
}
}  // namespace

int main() {
  try {
    check_inverse_solver();
    std::filesystem::create_directories("build");
    std::ofstream csv("build/inverse_kinematics.csv"), report("build/inverse_kinematics_report.txt"),
                  audit("build/inverse_velocity_audit.csv");
    require(csv && report && audit,"Cannot create IK outputs");
    csv << std::setprecision(17) << "case,method,target_x,target_y,target_z,seed_q1,seed_q2,"
           "reachability,branch_count,status,iterations,final_q1,final_q2,error_m\n";
    audit << std::setprecision(17) << "q1,q2,lambda,j00,j01,j10,j11,j20,j21,"
             "desired_x,desired_y,desired_z,dq1,dq2,sigma_max,sigma_min\n";

    // Export J, desired and dq = inverse(J)*desired for an independent NumPy
    // SVD/normal-equation comparison; exporting alone does not validate them.
    const auto write_audit = [&](const Joints& q, double damping) {
      const auto j = point_jacobian(q[0],q[1]);
      const Vector3 desired = {0.3,-0.2,0.1};
      const auto inv = inverse_velocity(j,desired,damping);
      audit << q[0] << ',' << q[1] << ',' << damping;
      for (int row = 0; row < 3; ++row) audit << ',' << j[0][row] << ',' << j[1][row];
      for (double x : desired) audit << ',' << x;
      audit << ',' << inv.dq[0] << ',' << inv.dq[1] << ',' << inv.sigma[0] << ',' << inv.sigma[1] << '\n';
    };

    // Validate IK via the original nonlinear task: ||target - f(q_solution)||.
    // The geometric branches check feasibility; the numerical solver still
    // starts at seed, so this test can reveal branch dependence and local stalls.
    const auto solve = [&](const std::string& name, const Vector3& target, const Joints& seed,
                           double damping, bool expect_success) {
      const auto oracle = geometric_ik(target,kPhysicalLimits);
      for (const auto& q : oracle.solutions)
        require(norm(error(target,position(q))) < 1e-10,"Geometric IK branch has wrong FK");
      const auto result = numerical_ik(target,seed,kPhysicalLimits,damping);
      require(inside(result.q,kPhysicalLimits) && std::isfinite(result.final_error),
              "IK returned invalid joints or error");
      if (expect_success) require(result.status == "converged",name+": expected IK convergence");
      if (name == "exact_singular_start")
        require(result.status == "stalled" && result.final_error > 0.1,
                "Singular start must not falsely report success");
      if (oracle.status != "reachable")
        require(result.status == oracle.status && result.iterations == 0,
                "Unreachable target should be rejected before iteration");
      csv << name << ',' << (damping == 0 ? "pinv" : "dls") << ','
          << target[0] << ',' << target[1] << ',' << target[2] << ',' << seed[0] << ',' << seed[1]
          << ',' << oracle.status << ',' << oracle.solutions.size() << ',' << result.status << ','
          << result.iterations << ',' << result.q[0] << ',' << result.q[1] << ',' << result.final_error << '\n';
      return result;
    };
    
    report << "Week 6: numerical position IK (not full-pose IK)\n"
           << "Tolerance 1e-6 m; at most 300 iterations; max joint correction 0.2 rad; backtracking.\n"
           << "Geometric IK validates reachability but never supplies numerical iterates.\n\n";
    for (const auto& scenario : scenarios()) {
      for (double damping : {0.0,kDamping}) {
        const auto result = solve(scenario.name,scenario.target,scenario.initial,damping,
                                  scenario.name == "regular" || scenario.name == "near_joint_limit");
        report << scenario.name << " / " << (damping == 0 ? "pinv" : "dls")
               << ": " << result.status << ", " << result.iterations
               << " iterations, error=" << result.final_error << " m\n";
      }
    }
    // Both elbow branches are real solutions. Different seeds preserve each
    // branch locally; minimum-norm steps do not promise the globally nearest IK.
    const auto target = position({0.2,0.9});
    const auto branches = geometric_ik(target,kPhysicalLimits);
    require(branches.solutions.size() == 2,"Expected two feasible elbow branches");
    for (const auto& branch : branches.solutions) {
      const Joints seed = {branch[0]+0.08,branch[1]-0.05};
      const auto result = solve("two_branches",target,seed,0,true);
      require(result.q[1]*branch[1] > 0,"Local branch example changed elbow sign");
    }

    std::mt19937 rng(20260912);
    std::uniform_real_distribution<double> shoulder(-1.0,1.0), bend(0.3,1.3);
    for (int sample = 0; sample < 100; ++sample) {
      const Joints goal = {shoulder(rng),(sample%2 ? -1 : 1)*bend(rng)};
      const Joints seed = {goal[0]+0.08,goal[1]-0.05};
      for (double damping : {0.0,kDamping}) {
        solve("random_"+std::to_string(sample),position(goal),seed,damping,true);
        write_audit(goal,damping);
      }
    }
    for (const Joints q : {Joints{0,0},Joints{0,1e-12},Joints{0,1e-4},Joints{0.5,-1e-4}})
      for (double damping : {0.0,kDamping}) write_audit(q,damping);
    report << "\nPASS: rank-zero/rank-one inverse, rank cutoff, DLS normal equations,\n"
           << "near-singular amplification, limits, both elbow branches, and\n"
           << "100 seeded random local IK problems per method.\n"
           << "Named difficult cases retain their actual statuses; reachability does not guarantee local convergence.\n";
    csv.flush(); report.flush(); audit.flush();
    require(csv && report && audit,"Failed writing IK outputs");
    std::cout << "IK and inverse-velocity checks passed; 100 random targets per method.\n"
              << "Read build/inverse_kinematics_report.txt and build/inverse_kinematics.csv\n";
  } catch (const std::exception& exception) {
    std::cerr << "Error: " << exception.what() << '\n';
    return 1;
  }
}
