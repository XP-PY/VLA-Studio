/* Week 5: compare analytical space/body/point Jacobians with MuJoCo,
   central differences, site velocities, and force-to-torque mapping.
   This is a kinematics experiment: qpos and qvel are prescribed snapshots. */
#include "mujoco.h"
#include "two_link_kinematics.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace two_link;
using namespace transform_utils;
constexpr unsigned kSeed = 20260911;
constexpr int kRandomCount = 100;
constexpr double kH = 1e-6;

void require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}
void check(double error, double tolerance, const char* label) {
  require(std::isfinite(error) && error <= tolerance,
          std::string(label) + " failed: error=" + std::to_string(error));
}
double difference(const Vector3& a, const Vector3& b) {
  double error = 0;
  for (int i = 0; i < 3; ++i) {
    if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
      return std::numeric_limits<double>::infinity();
    error = std::max(error, std::abs(a[i]-b[i]));
  }
  return error;
}
Matrix4 pose(const mjtNum* p, const mjtNum* r) {
  return make_transform({r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7],r[8]},
                        {p[0],p[1],p[2]});
}
Matrix4 site_pose(const mjData* d, int site) {
  return pose(d->site_xpos+3*site, d->site_xmat+9*site);
}
// Central-difference R, then omega_hat = Rdot R^T in world coordinates.
Vector3 angular_difference(const Matrix4& plus, const Matrix4& minus,
                           const Matrix4& center) {
  Matrix3 derivative{};
  const auto rp = rotation_of(plus), rm = rotation_of(minus);
  for (int i = 0; i < 9; ++i) derivative[i] = (rp[i]-rm[i])/(2*kH);
  const auto skew = multiply(derivative, transpose(rotation_of(center)));
  return {(skew[7]-skew[5])/2, (skew[2]-skew[6])/2, (skew[3]-skew[1])/2};
}
struct SingularValues { double determinant, maximum, minimum; };
SingularValues singular_values(double q1, double q2) {
  const auto j = point_jacobian(q1,q2);
  const double a = dot(j[0],j[0]), b = dot(j[0],j[1]), c = dot(j[1],j[1]);
  const double det = j[0][0]*j[1][1]-j[1][0]*j[0][1];
  const double maximum = std::sqrt((a+c+std::hypot(a-c,2*b))/2);
  // sigma_min=|det|/sigma_max avoids subtracting nearly equal eigenvalues.
  return {det, maximum, std::abs(det)/maximum};
}
}  // namespace

int main() {
  try {
    // Load the MuJoCo robot
    char error[1024]{};
    std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model(
        mj_loadXML("assets/two_link_arm.xml", nullptr, error, sizeof(error)), mj_deleteModel);
    require(model != nullptr, std::string("Cannot load arm: ")+error);
    std::unique_ptr<mjData, decltype(&mj_deleteData)> data(mj_makeData(model.get()), mj_deleteData);
    require(data != nullptr, "Cannot allocate mjData");

    // Find the robot parts by name
    auto* m = model.get(); auto* d = data.get();
    const int base = mj_name2id(m, mjOBJ_BODY, "base");
    const int site = mj_name2id(m, mjOBJ_SITE, "end_effector");
    const std::array<int,2> joints = {mj_name2id(m,mjOBJ_JOINT,"shoulder"),
                                     mj_name2id(m,mjOBJ_JOINT,"elbow")};
    require(base >= 0 && site >= 0 && joints[0] >= 0 && joints[1] >= 0,
            "Missing named model object");
    require(m->nq == 2 && m->nv == 2 && m->jnt_type[joints[0]] == mjJNT_HINGE &&
            m->jnt_type[joints[1]] == mjJNT_HINGE, "Expected the fixed-base 2R arm");
    const std::array<int,2> qa = {m->jnt_qposadr[joints[0]], m->jnt_qposadr[joints[1]]};
    const std::array<int,2> va = {m->jnt_dofadr[joints[0]], m->jnt_dofadr[joints[1]]};
    const auto set_state = [&](const std::array<double,2>& q, const std::array<double,2>& dq) {
      for (int i = 0; i < 2; ++i) { d->qpos[qa[i]] = q[i]; d->qvel[va[i]] = dq[i]; }
      mj_forward(m,d);  // refresh both position AND velocity-dependent quantities
    };

    // Generate test configurations
    std::filesystem::create_directories("build");
    std::ofstream csv("build/jacobian_comparison.csv"), entries("build/jacobian_entries.csv"),
                  singular("build/jacobian_singularities.csv"), report("build/jacobian_report.txt");
    require(csv && entries && singular && report, "Cannot create Jacobian outputs");
    csv << std::setprecision(17)
        << "sample,kind,q1_rad,q2_rad,dq1_rad_s,dq2_rad_s,jacobian_error,fd_jacobian_error,"
           "velocity_error,fd_velocity_error,torque_error,power_error,"
           "vx_analytical,vy_analytical,vz_analytical,vx_mujoco,vy_mujoco,vz_mujoco,"
           "tau1_analytical,tau2_analytical,tau1_mujoco,tau2_mujoco\n";
    entries << std::setprecision(17) << "sample,joint,component,point_world_analytical,"
               "point_world_mujoco,point_world_fd,angular_world_analytical,"
               "angular_world_mujoco,angular_world_fd,space_world_analytic_v,"
               "space_world_mujoco_v,body_analytic_v,body_mujoco_v\n";
    std::mt19937 rng(kSeed);
    std::uniform_real_distribution<double> q1dist(-1.4,1.4), q2dist(-1.6,1.6), vdist(-1,1);
    std::vector<std::array<double,2>> configurations = {
        {0,0}, {0.4,0}, {0.4,1e-6}, {0.4,-1e-6}, {1.5,-1.7}, {-1.5,1.7}};
    for (int i = 0; i < kRandomCount; ++i) configurations.push_back({q1dist(rng),q2dist(rng)});

    // Test loop
    std::array<double,6> maxima{};
    for (std::size_t sample = 0; sample < configurations.size(); ++sample) {
      // Sample posture and velocity
      const auto q = configurations[sample];
      const std::array<double,2> dq = {vdist(rng),vdist(rng)};
      set_state(q,dq);

      // Extract matrixs
      const Matrix4 twb = pose(d->xpos+3*base,d->xmat+9*base);              // T_{wb}: base pose in world frame
      const Matrix4 twe = site_pose(d,site);                                // T_{we}: end-effector pose in world frame
      const auto rwb = rotation_of(twb), rew = transpose(rotation_of(twe)); // R_{wb}, R_{ew}: rotation matrixs
      const auto p = translation_of(twe);                                   // p_{we}: end-effector world position
      
      // Calculate the analytical Jacobians
      const auto jp_base = point_jacobian(q[0],q[1]);
      const auto js_base = space_jacobian(q[0]);
      const auto js_world = transform_jacobian(twb,js_base);
      const auto jb = body_jacobian(q[0],q[1]);
      check(max_abs_difference(multiply_transform(twb,forward_kinematics(q[0],q[1])), twe),
            1e-12, "Model geometry/FK");

      // Ask MuJoCo for its Jacobian
      // jacp: Jacobian of the site point position, 
      //      Partial derivative of end-effector position with respect to joint velocity
      // jacr: maps joint velocity to the site's angular velocity, 
      //      Partial derivative of the end-effector angular velocity with respect to the joint velocities
      std::vector<mjtNum> jacp(3*m->nv), jacr(3*m->nv); 
      mj_jacSite(m,d,jacp.data(),jacr.data(),site);

      // Loop through each Jacobian column
      PointJacobian jp_world{}, jp_mj{};
      Jacobian geometric_world{};
      double jac_error = 0, fd_error = 0;
      for (int col = 0; col < 2; ++col) {
        // Convert analytical point Jacobian into world coordinates
        jp_world[col] = rotate(rwb, jp_base[col]);
        const auto omega = rotate(rwb, js_base[col].omega);
        geometric_world[col] = {omega, jp_world[col]};  // geometric Jacobian at the end-effector point

        // Read MuJoCo's corresponding column
        Vector3 omega_mj{};
        for (int row = 0; row < 3; ++row) {
          jp_mj[col][row] = jacp[row*m->nv+va[col]];
          omega_mj[row] = jacr[row*m->nv+va[col]];
        }

        // Convert MuJoCo's point velocity to a space twist about WORLD origin.
        const auto vs_mj = add(jp_mj[col],scale(cross(omega_mj,p),-1));
        const auto vb_mj = rotate(rew,jp_mj[col]);
        jac_error = std::max({jac_error, 
            difference(jp_world[col],jp_mj[col]),
            difference(omega,omega_mj), 
            difference(js_world[col].v,vs_mj),
            difference(js_world[col].omega,omega_mj), 
            difference(jb[col].v,vb_mj),
            difference(jb[col].omega,rotate(rew,omega_mj))
        });

        // Independent closed-form body Jacobian, derived from body screw axes.
        const Vector3 closed_body = col == 0
            ? Vector3{kL1*std::sin(q[1]),kL1*std::cos(q[1])+kL2,0}
            : Vector3{0,kL2,0};
        jac_error = std::max(jac_error, difference(jb[col].v,closed_body));

        // Finite-difference test of one Jacobian column
        auto perturbed = q;
        perturbed[col] += kH; set_state(perturbed,dq); const auto plus = site_pose(d,site);
        perturbed[col] -= 2*kH; set_state(perturbed,dq); const auto minus = site_pose(d,site);
        const auto fd = scale(add(translation_of(plus),scale(translation_of(minus),-1)),1/(2*kH));
        const auto fd_omega = angular_difference(plus,minus,twe);
        fd_error = std::max({fd_error,difference(fd,jp_world[col]),difference(fd_omega,omega)});
        for (int row = 0; row < 3; ++row)
          entries << sample << ',' << col << ',' << "xyz"[row] << ',' << jp_world[col][row]
                  << ',' << jp_mj[col][row] << ',' << fd[row] << ',' << omega[row]
                  << ',' << omega_mj[row] << ',' << fd_omega[row] << ',' << js_world[col].v[row]
                  << ',' << vs_mj[row] << ',' << jb[col].v[row] << ',' << vb_mj[row] << '\n';
      }

      // restore after perturbations before querying velocity/force
      set_state(q,dq);

      // Test V=J*q_dot
      mjtNum world_velocity[6]{}, body_velocity[6]{};
      mj_objectVelocity(m,d,mjOBJ_SITE,site,world_velocity,0);  // Ask MuJoCo for the end-effector site's velocity expressed in the world frame
      mj_objectVelocity(m,d,mjOBJ_SITE,site,body_velocity,1);   // Ask for the same physical velocity expressed in the site/body frame
      const auto predicted = velocity(geometric_world,dq), predicted_body = velocity(jb,dq);
      double velocity_error = 0;
      for (int row = 0; row < 3; ++row)
        velocity_error = std::max({velocity_error,
            std::abs(predicted.omega[row]-world_velocity[row]),
            std::abs(predicted.v[row]-world_velocity[row+3]),
            std::abs(predicted_body.omega[row]-body_velocity[row]),
            std::abs(predicted_body.v[row]-body_velocity[row+3])});

      // Sample FK along q(t)=q+t*dq: an independent measurement of tip velocity.
      auto moved = q;
      for (int i = 0; i < 2; ++i) moved[i] = q[i]+kH*dq[i];
      set_state(moved,dq); const auto plus = site_pose(d,site);
      for (int i = 0; i < 2; ++i) moved[i] = q[i]-kH*dq[i];
      set_state(moved,dq); const auto minus = site_pose(d,site);
      const auto fd_v = scale(add(translation_of(plus),scale(translation_of(minus),-1)),1/(2*kH));
      const double fd_velocity_error = std::max(difference(fd_v,predicted.v),
                                               difference(angular_difference(plus,minus,twe),predicted.omega));

      // Apply a wrench at the end effector
      set_state(q,dq);
      const Vector3 force = {1.2,-0.7,0.4}, moment = {0.1,0.2,0.3};  // world, at tip
      const Vector3 world_origin_moment = add(moment,cross(p,force));
      const auto body_force = rotate(rew,force), body_moment = rotate(rew,moment);
      std::vector<mjtNum> applied(m->nv,0);  // mj_applyFT ADDS into this buffer
      mj_applyFT(m,d,force.data(),moment.data(),p.data(),m->site_bodyid[site],applied.data());

      // Calculate torque = J^T F in three analytical ways
      std::array<double,2> tau{};
      double torque_error = 0;
      for (int col = 0; col < 2; ++col) {
        // Use the geometric point Jacobian
        tau[col] = dot(jp_world[col],force)+dot(geometric_world[col].omega,moment);
        // Use the Modern Robotics space Jacobian
        const double tau_space = dot(js_world[col].omega,world_origin_moment)+dot(js_world[col].v,force);
        // Use the Modern Robotics body Jacobian
        const double tau_body = dot(jb[col].omega,body_moment)+dot(jb[col].v,body_force);

        torque_error = std::max({torque_error,
                              std::abs(tau[col]-applied[va[col]]),
                              std::abs(tau[col]-tau_space),
                              std::abs(tau[col]-tau_body)
                          });
      }

      // Check conservation of power
      const double power_error = std::abs(tau[0]*dq[0]+tau[1]*dq[1]
                                          -dot(force,predicted.v)-dot(moment,predicted.omega));
      const std::array<double,6> errors = {jac_error,fd_error,velocity_error,fd_velocity_error,torque_error,power_error};
      constexpr std::array<double,6> tolerances = {1e-12,1e-8,1e-12,1e-8,1e-12,1e-12};
      for (int i = 0; i < 6; ++i) { check(errors[i],tolerances[i],"Jacobian/velocity/statics"); maxima[i] = std::max(maxima[i],errors[i]); }
      csv << sample << ',' << (sample < 6 ? "edge" : "random") << ',' << q[0] << ',' << q[1]
          << ',' << dq[0] << ',' << dq[1];
      for (double e : errors) csv << ',' << e;
      for (double v : predicted.v) csv << ',' << v;
      for (int i = 3; i < 6; ++i) csv << ',' << world_velocity[i];
      csv << ',' << tau[0] << ',' << tau[1] << ',' << applied[va[0]] << ',' << applied[va[1]] << '\n';
    }

    singular << std::setprecision(17) << "q2_rad,det_Jxy,sigma_max_m,sigma_min_m,condition_number\n";
    for (int i = -160; i <= 160; ++i) {
      const double q2 = i*0.01;
      const auto s = singular_values(0.4,q2);
      check(std::abs(s.determinant-kL1*kL2*std::sin(q2)),1e-14,"Planar determinant");
      singular << q2 << ',' << s.determinant << ',' << s.maximum << ',' << s.minimum << ','
               << (s.minimum < 1e-14 ? std::numeric_limits<double>::infinity() : s.maximum/s.minimum) << '\n';
    }
    check(singular_values(0.4,0).minimum,1e-14,"Straight-arm position singularity");
    require(singular_values(0.4,0.8).minimum > 0.1, "Bent arm should have rank 2 for position");
    // Full Js has Gram determinant L1^2 > 0, including at q2=0.
    const auto js = space_jacobian(0.4);
    const double a = dot(js[0].omega,js[0].omega)+dot(js[0].v,js[0].v);
    const double b = dot(js[0].omega,js[1].omega)+dot(js[0].v,js[1].v);
    const double c = dot(js[1].omega,js[1].omega)+dot(js[1].v,js[1].v);
    check(std::abs(a*c-b*b-kL1*kL1),1e-14,"Full twist rank");
    report << "Week 5: Jacobians of the week 4 two-link arm\n"
           << kRandomCount << " random + 6 edge configurations; seed " << kSeed
           << "; finite difference h=" << kH << "\n"
           << "Rows below are maximum absolute component errors across all samples:\n"
           << std::scientific << std::setprecision(4);
    const char* labels[] = {"Analytical/MuJoCo Jacobians", "Central-difference Jacobians",
        "Site velocity (world and body)", "Central-difference velocity", "Wrench-to-torque mapping", "Power balance"};
    for (int i = 0; i < 6; ++i) report << labels[i] << ": " << maxima[i] << '\n';
    report << "\nAll checks passed. No dynamics integration was used.\n"
           << "MuJoCo jacp maps to point velocity; Js linear rows map to v= pdot - omega x p.\n"
           << "Point Jacobian world conversion uses rotation only; space twists use the full adjoint.\n"
           << "At q2=0 the XY position Jacobian has rank 1, but the full 6x2 twist Jacobian has rank 2.\n"
           << "Roundoff explains analytical errors; finite differences add cancellation and truncation.\n";
    csv.flush(); entries.flush(); singular.flush(); report.flush();
    require(csv && entries && singular && report,"Failed writing Jacobian outputs");
    std::cout << "Jacobian, velocity, statics and singularity checks passed for " << configurations.size()
              << " configurations.\n" << std::scientific << std::setprecision(3)
              << "Max analytical/MuJoCo Jacobian error: " << maxima[0]
              << "; finite-difference error: " << maxima[1]
              << "\nRead build/jacobian_report.txt and build/jacobian_*.csv\n";
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
}
