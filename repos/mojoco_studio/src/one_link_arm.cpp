/*
  Week 2 MuJoCo experiment: model and inspect a one-link robot arm.

  The program reads the world poses of a body and a site, then checks the
  rigid-body transformation

      p_W = p_WL + R_WL p_L.
*/

#include "mujoco.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDuration = 8.0;           // total experiment time in seconds
constexpr double kControlFrequency = 50.0;  // desired-angle update rate in Hz
constexpr double kMotionFrequency = 0.25;   // sine trajectory frequency in Hz
constexpr double kTargetAmplitude = 1.0;    // sine trajectory amplitude in rad
constexpr double kLinkLength = 0.6;         // hinge-to-tip distance in meters

// Small fixed-size types make the frame calculations explicit. Matrix3 uses
// the same row-major element order as MuJoCo's xmat arrays.
using Vector3 = std::array<double, 3>;
using Matrix3 = std::array<double, 9>;

// One row of the output CSV. It records the joint/actuator state, poses read
// from MuJoCo, and errors from two independent transformation checks.
struct Sample {
  double time;                    // simulation time after this control interval
  double target_q;                // desired shoulder angle sent to ctrl
  double q;                       // measured shoulder angle
  double qvel;                    // measured shoulder angular velocity
  double actuator_force;          // force produced by the position actuator
  Vector3 body_position;          // p_WL: link-frame origin in world coordinates
  Matrix3 body_rotation;          // R_WL: link-frame orientation in world
  Vector3 site_position;          // p_WS: tip-site origin in world coordinates
  Matrix3 site_rotation;          // R_WS: tip-site orientation in world
  Vector3 transformed_tip;        // p_WL + R_WL p_L, computed by this program
  double tip_position_error;      // distance between transformed tip and site
  double site_rotation_error;     // Frobenius distance between R_WL and R_WS
  double analytic_pose_error;     // error against [L cos(q), L sin(q), 0.8]
};

// Generate a smooth desired joint trajectory. The position actuator will try
// to track this command; this is prescribed motion rather than an RL policy.
double desired_angle(double time) {
  return kTargetAmplitude *
         std::sin(2.0 * kPi * kMotionFrequency * time);
}

// Copy three adjacent MuJoCo scalar values into an owning C++ vector.
Vector3 read_vector3(const mjtNum* source) {
  return {source[0], source[1], source[2]};
}

// Copy nine adjacent MuJoCo scalar values into a row-major 3x3 matrix.
Matrix3 read_matrix3(const mjtNum* source) {
  Matrix3 result{};
  std::copy_n(source, result.size(), result.begin());
  return result;
}

// MuJoCo stores xmat and site_xmat as row-major 3x3 rotation matrices.
Vector3 rotate(const Matrix3& rotation, const Vector3& vector) {
  return {
      rotation[0] * vector[0] + rotation[1] * vector[1] +
          rotation[2] * vector[2],
      rotation[3] * vector[0] + rotation[4] * vector[1] +
          rotation[5] * vector[2],
      rotation[6] * vector[0] + rotation[7] * vector[1] +
          rotation[8] * vector[2],
  };
}

// Add two vectors component by component.
Vector3 add(const Vector3& left, const Vector3& right) {
  return {left[0] + right[0], left[1] + right[1], left[2] + right[2]};
}

// Compute Euclidean distance ||left - right||_2.
double vector_distance(const Vector3& left, const Vector3& right) {
  const double dx = left[0] - right[0];
  const double dy = left[1] - right[1];
  const double dz = left[2] - right[2];
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// Compute the Frobenius distance between two 3x3 matrices. It is zero when
// corresponding rotation matrices are identical.
double matrix_distance(const Matrix3& left, const Matrix3& right) {
  double squared_error = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    const double error = left[index] - right[index];
    squared_error += error * error;
  }
  return std::sqrt(squared_error);
}

// Closed-form forward kinematics for a length-L link rotating about world z.
// This is independent of MuJoCo and provides a second correctness check.
Vector3 analytic_tip_position(double q) {
  return {kLinkLength * std::cos(q),
          kLinkLength * std::sin(q),
          0.8};
}

// Store all joint values, world poses, and transformation errors in a CSV so
// individual time steps can be inspected outside the simulator.
void write_csv(const std::vector<Sample>& samples,
               const std::filesystem::path& output_path) {
  std::ofstream output(output_path);
  if (!output) {
    throw std::runtime_error("Could not open " + output_path.string());
  }

  output << "time_s,target_q_rad,q_rad,qvel_rad_s,actuator_force_nm,"
            "body_x_m,body_y_m,body_z_m,"
            "body_r00,body_r01,body_r02,body_r10,body_r11,body_r12,"
            "body_r20,body_r21,body_r22,"
            "site_x_m,site_y_m,site_z_m,"
            "site_r00,site_r01,site_r02,site_r10,site_r11,site_r12,"
            "site_r20,site_r21,site_r22,"
            "transformed_tip_x_m,transformed_tip_y_m,transformed_tip_z_m,"
            "tip_position_error_m,site_rotation_error,analytic_pose_error_m\n";
  output << std::setprecision(12);

  for (const Sample& sample : samples) {
    output << sample.time << ',' << sample.target_q << ',' << sample.q << ','
           << sample.qvel << ',' << sample.actuator_force;
    for (double value : sample.body_position) {
      output << ',' << value;
    }
    for (double value : sample.body_rotation) {
      output << ',' << value;
    }
    for (double value : sample.site_position) {
      output << ',' << value;
    }
    for (double value : sample.site_rotation) {
      output << ',' << value;
    }
    for (double value : sample.transformed_tip) {
      output << ',' << value;
    }
    output << ',' << sample.tip_position_error << ','
           << sample.site_rotation_error << ','
           << sample.analytic_pose_error << '\n';
  }
}

// Create a dependency-free SVG visualization of the end-site path in the
// world x-y plane. The final orange segment shows the last link pose.
void write_path_svg(const std::vector<Sample>& samples,
                    const std::filesystem::path& output_path) {
  constexpr int kCanvasSize = 760;                 // SVG width and height
  constexpr double kCenterX = 380.0;               // world origin in SVG x
  constexpr double kCenterY = 400.0;               // world origin in SVG y
  constexpr double kPixelsPerMeter = 480.0;         // visualization scale
  constexpr double kRadius = kLinkLength * kPixelsPerMeter;

  std::ofstream svg(output_path);
  if (!svg) {
    throw std::runtime_error("Could not open " + output_path.string());
  }

  // SVG y points downward, so negate world y when converting coordinates.
  const auto svg_x = [](double x) { return kCenterX + kPixelsPerMeter * x; };
  const auto svg_y = [](double y) { return kCenterY - kPixelsPerMeter * y; };

  svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\""
      << kCanvasSize << "\" height=\"" << kCanvasSize
      << "\" viewBox=\"0 0 " << kCanvasSize << ' ' << kCanvasSize << "\">\n"
      << "<rect width=\"100%\" height=\"100%\" fill=\"#0b0f16\"/>\n"
      << "<g font-family=\"sans-serif\">\n"
      << "<text x=\"380\" y=\"36\" text-anchor=\"middle\" "
         "fill=\"#f8fafc\" font-size=\"24\" font-weight=\"bold\">"
         "Week 2 one-link arm: tip path in the world x-y plane</text>\n"
      << "<line x1=\"60\" y1=\"" << kCenterY << "\" x2=\"700\" y2=\""
      << kCenterY << "\" stroke=\"#475569\"/>\n"
      << "<line x1=\"" << kCenterX << "\" y1=\"80\" x2=\"" << kCenterX
      << "\" y2=\"720\" stroke=\"#475569\"/>\n"
      << "<circle cx=\"" << kCenterX << "\" cy=\"" << kCenterY
      << "\" r=\"" << kRadius
      << "\" fill=\"none\" stroke=\"#334155\" stroke-dasharray=\"7 7\"/>\n"
      << "<polyline fill=\"none\" stroke=\"#38bdf8\" stroke-width=\"4\" "
         "points=\"";

  for (const Sample& sample : samples) {
    svg << svg_x(sample.site_position[0]) << ','
        << svg_y(sample.site_position[1]) << ' ';
  }
  svg << "\"/>\n";

  const Sample& last = samples.back();
  svg << "<line x1=\"" << kCenterX << "\" y1=\"" << kCenterY
      << "\" x2=\"" << svg_x(last.site_position[0]) << "\" y2=\""
      << svg_y(last.site_position[1])
      << "\" stroke=\"#f59e0b\" stroke-width=\"10\" stroke-linecap=\"round\"/>\n"
      << "<circle cx=\"" << kCenterX << "\" cy=\"" << kCenterY
      << "\" r=\"10\" fill=\"#f8fafc\"/>\n"
      << "<circle cx=\"" << svg_x(last.site_position[0]) << "\" cy=\""
      << svg_y(last.site_position[1]) << "\" r=\"9\" fill=\"#fb7185\"/>\n"
      << "<text x=\"690\" y=\"425\" fill=\"#cbd5e1\" font-size=\"17\">"
         "+x_W</text>\n"
      << "<text x=\"390\" y=\"90\" fill=\"#cbd5e1\" font-size=\"17\">"
         "+y_W</text>\n"
      << "<text x=\"380\" y=\"744\" text-anchor=\"middle\" "
         "fill=\"#94a3b8\" font-size=\"15\">"
         "blue: MuJoCo site position; orange: final link pose</text>\n"
      << "</g>\n</svg>\n";
}

// Print one human-readable pose in addition to the machine-readable CSV.
void print_pose_snapshot(const Sample& sample) {
  std::cout << std::fixed << std::setprecision(6)
            << "\nFinal pose snapshot\n"
            << "q = " << sample.q << " rad\n"
            << "World position of link body frame: ["
            << sample.body_position[0] << ", " << sample.body_position[1]
            << ", " << sample.body_position[2] << "] m\n"
            << "World rotation of link body frame (row-major):\n";
  for (int row = 0; row < 3; ++row) {
    std::cout << "  [";
    for (int column = 0; column < 3; ++column) {
      if (column > 0) {
        std::cout << ", ";
      }
      std::cout << sample.body_rotation[3 * row + column];
    }
    std::cout << "]\n";
  }
  std::cout << "World position of tip site: [" << sample.site_position[0]
            << ", " << sample.site_position[1] << ", "
            << sample.site_position[2] << "] m\n";
}

}  // namespace

int main() {
  // Input MJCF and generated output paths are relative to the repository root.
  constexpr char kModelPath[] = "assets/one_link_arm.xml";
  constexpr char kCsvPath[] = "build/one_link_arm_poses.csv";
  constexpr char kPlotPath[] = "build/one_link_arm_path.svg";

  char error[1024] = {};

  // Parse the MJCF file and compile it into MuJoCo's immutable physical model.
  mjModel* model = mj_loadXML(kModelPath, nullptr, error, sizeof(error));
  if (model == nullptr) {
    std::cerr << "Failed to load " << kModelPath << ": " << error << '\n';
    return 1;
  }

  // Allocate mutable simulation state and workspace for this specific model.
  mjData* data = mj_makeData(model);
  if (data == nullptr) {
    std::cerr << "Failed to allocate mjData\n";
    mj_deleteModel(model);
    return 1;
  }

  try {
    // Resolve readable MJCF names to integer indices used by MuJoCo's arrays.
    const int body_id = mj_name2id(model, mjOBJ_BODY, "link");
    const int joint_id = mj_name2id(model, mjOBJ_JOINT, "shoulder");
    const int site_id = mj_name2id(model, mjOBJ_SITE, "tip");
    const int actuator_id =
        mj_name2id(model, mjOBJ_ACTUATOR, "shoulder_position");
    if (body_id < 0 || joint_id < 0 || site_id < 0 || actuator_id < 0) {
      throw std::runtime_error("A named body, joint, site, or actuator is missing");
    }

    // qpos and qvel are flat arrays. These address tables give the first
    // position coordinate and velocity degree of freedom for this joint.
    const int qpos_index = model->jnt_qposadr[joint_id];
    const int qvel_index = model->jnt_dofadr[joint_id];

    // The site's coordinates expressed in the link body frame {L}.
    const Vector3 local_tip = {kLinkLength, 0.0, 0.0};

    // Establish the initial generalized state and synchronize all Cartesian
    // quantities derived from it before beginning the simulation.
    data->qpos[qpos_index] = 0.0;
    data->qvel[qvel_index] = 0.0;
    data->ctrl[actuator_id] = 0.0;
    mj_forward(model, data);

    // The controller runs more slowly than physics. At 50 Hz control and a
    // 0.002 s timestep, one desired angle is held for ten physics steps.
    const int physics_steps_per_control = std::max(
        1, static_cast<int>(std::lround(
               (1.0 / kControlFrequency) / model->opt.timestep)));
    const double control_period =
        physics_steps_per_control * model->opt.timestep;
    const int sample_count =
        static_cast<int>(std::ceil(kDuration / control_period));

    std::vector<Sample> samples;
    samples.reserve(sample_count);

    for (int sample_index = 0; sample_index < sample_count; ++sample_index) {
      // A position actuator interprets ctrl as a desired joint coordinate.
      data->ctrl[actuator_id] = desired_angle(data->time);

      // Hold this desired angle while MuJoCo advances continuous dynamics.
      for (int step = 0; step < physics_steps_per_control; ++step) {
        mj_step(model, data);
      }

      // qpos has just been integrated. Recompute Cartesian quantities so
      // xpos/xmat and site_xpos/site_xmat exactly match that new qpos.
      mj_forward(model, data);

      // Cartesian pose arrays are flat: positions occupy 3 values per object
      // and rotation matrices occupy 9 values per object.
      const Vector3 body_position =
          read_vector3(data->xpos + 3 * body_id);
      const Matrix3 body_rotation =
          read_matrix3(data->xmat + 9 * body_id);
      const Vector3 site_position =
          read_vector3(data->site_xpos + 3 * site_id);
      const Matrix3 site_rotation =
          read_matrix3(data->site_xmat + 9 * site_id);

      // Apply the homogeneous transform T_WL to the local site position.
      const Vector3 transformed_tip =
          add(body_position, rotate(body_rotation, local_tip));
      const Vector3 analytic_tip =
          analytic_tip_position(data->qpos[qpos_index]);

      // Capture values now because the next mj_step will mutate mjData.
      samples.push_back({
          data->time,
          data->ctrl[actuator_id],
          data->qpos[qpos_index],
          data->qvel[qvel_index],
          data->actuator_force[actuator_id],
          body_position,
          body_rotation,
          site_position,
          site_rotation,
          transformed_tip,
          vector_distance(transformed_tip, site_position),
          matrix_distance(body_rotation, site_rotation),
          vector_distance(analytic_tip, site_position),
      });
    }

    // Write persistent artifacts only after the complete simulation succeeds.
    std::filesystem::create_directories("build");
    write_csv(samples, kCsvPath);
    write_path_svg(samples, kPlotPath);

    double maximum_tip_error = 0.0;
    double maximum_rotation_error = 0.0;
    double maximum_analytic_error = 0.0;

    // Reduce per-sample errors into concise numerical correctness checks.
    for (const Sample& sample : samples) {
      maximum_tip_error =
          std::max(maximum_tip_error, sample.tip_position_error);
      maximum_rotation_error =
          std::max(maximum_rotation_error, sample.site_rotation_error);
      maximum_analytic_error =
          std::max(maximum_analytic_error, sample.analytic_pose_error);
    }

    std::cout << "MuJoCo " << mj_versionString() << '\n'
              << "Model objects (including world body): bodies=" << model->nbody
              << ", joints=" << model->njnt << ", geoms=" << model->ngeom
              << ", sites=" << model->nsite
              << ", actuators=" << model->nu << '\n'
              << "Physics: " << 1.0 / model->opt.timestep
              << " Hz; position commands: " << 1.0 / control_period << " Hz\n"
              << "Logged pose samples: " << samples.size() << '\n'
              << std::scientific << std::setprecision(3)
              << "Max |T_WL p_L - site_xpos|: " << maximum_tip_error
              << " m\n"
              << "Max |body R - site R|_F: " << maximum_rotation_error << '\n'
              << "Max analytic planar-pose error: " << maximum_analytic_error
              << " m\n";
    print_pose_snapshot(samples.back());
    std::cout << "\nCSV: " << kCsvPath << '\n'
              << "Path plot: " << kPlotPath << '\n';
  } catch (const std::exception& exception) {
    // Release MuJoCo allocations on every handled failure path.
    std::cerr << "Error: " << exception.what() << '\n';
    mj_deleteData(data);
    mj_deleteModel(model);
    return 1;
  }

  // mj_makeData and mj_loadXML transfer ownership to the caller.
  mj_deleteData(data);
  mj_deleteModel(model);
  return 0;
}
