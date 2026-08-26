/*
  Week 3 MuJoCo experiment: world, base, link, and end-effector frames.

  Every control sample reads MuJoCo's body/site poses, converts them to
  homogeneous transformations, and validates the frame chain

      T_WE = T_WB T_BL T_LE.

  See transform_utils.h for the reusable Week 3 deliverable functions.
*/

#include "mujoco.h"
#include "transform_utils.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using transform_utils::Matrix3;
using transform_utils::Matrix4;
using transform_utils::RotationValidation;
using transform_utils::Vector3;

constexpr double kPi = 3.14159265358979323846;
constexpr double kDuration = 8.0;           // total experiment time in seconds
constexpr double kControlFrequency = 50.0;  // desired-angle update rate in Hz
constexpr double kMotionFrequency = 0.20;   // sinusoidal motion frequency in Hz
constexpr double kTargetAmplitude = 1.0;    // desired-angle amplitude in rad
constexpr double kValidationTolerance = 1e-9;

struct Sample {
  double time;                         // simulation time after stepping
  double target_q;                     // desired shoulder angle (action)
  double q;                            // measured shoulder angle
  double qvel;                         // measured angular velocity
  Vector3 world_base_position;         // translation part of T_WB
  Vector3 world_end_position;          // translation part of T_WE
  Vector3 base_end_position;           // translation part of T_BE
  double base_rotation_error;          // max entry of R_WB^T R_WB - I
  double end_rotation_error;           // max entry of R_WE^T R_WE - I
  double base_rotation_determinant;    // det(R_WB)
  double end_rotation_determinant;     // det(R_WE)
  double composition_error;            // |T_WB T_BE - T_WE|_max
  double inverse_error;                // |T_BW T_WB - I|_max
  double chain_error;                  // |T_WB T_BL T_LE - T_WE|_max
  double point_transform_error;        // direct versus chained point mapping
  double fixed_tool_pose_drift;        // variation of T_LE during motion
};

double desired_angle(double time) {
  return kTargetAmplitude *
         std::sin(2.0 * kPi * kMotionFrequency * time);
}

// Convert MuJoCo's separate world position and row-major rotation arrays into
// the homogeneous matrix T=(R,p) used by the Week 3 utilities.
Matrix4 transform_from_mujoco_pose(const mjtNum* position,
                                   const mjtNum* rotation) {
  const Vector3 translation = {position[0], position[1], position[2]};
  Matrix3 rotation_matrix{};
  std::copy_n(rotation, rotation_matrix.size(), rotation_matrix.begin());
  return transform_utils::make_transform(rotation_matrix, translation);
}

// Read a named body pose T_WB or T_WL from xpos/xmat.
Matrix4 body_world_transform(const mjData* data, int body_id) {
  return transform_from_mujoco_pose(data->xpos + 3 * body_id,
                                    data->xmat + 9 * body_id);
}

// Read a named site pose T_WE from site_xpos/site_xmat.
Matrix4 site_world_transform(const mjData* data, int site_id) {
  return transform_from_mujoco_pose(data->site_xpos + 3 * site_id,
                                    data->site_xmat + 9 * site_id);
}

void write_csv(const std::vector<Sample>& samples,
               const std::filesystem::path& output_path) {
  std::ofstream output(output_path);
  if (!output) {
    throw std::runtime_error("Could not open " + output_path.string());
  }

  output << "time_s,target_q_rad,q_rad,qvel_rad_s,"
            "world_base_x_m,world_base_y_m,world_base_z_m,"
            "world_end_x_m,world_end_y_m,world_end_z_m,"
            "base_end_x_m,base_end_y_m,base_end_z_m,"
            "base_rotation_error,end_rotation_error,"
            "base_rotation_determinant,end_rotation_determinant,"
            "composition_error,inverse_error,chain_error,"
            "point_transform_error,fixed_tool_pose_drift\n";
  output << std::setprecision(12);

  for (const Sample& sample : samples) {
    output << sample.time << ',' << sample.target_q << ',' << sample.q << ','
           << sample.qvel;
    for (double value : sample.world_base_position) {
      output << ',' << value;
    }
    for (double value : sample.world_end_position) {
      output << ',' << value;
    }
    for (double value : sample.base_end_position) {
      output << ',' << value;
    }
    output << ',' << sample.base_rotation_error
           << ',' << sample.end_rotation_error
           << ',' << sample.base_rotation_determinant
           << ',' << sample.end_rotation_determinant
           << ',' << sample.composition_error
           << ',' << sample.inverse_error
           << ',' << sample.chain_error
           << ',' << sample.point_transform_error
           << ',' << sample.fixed_tool_pose_drift << '\n';
  }
}

void print_transform(std::ostream& output,
                     const std::string& name,
                     const Matrix4& transform) {
  output << name << " =\n" << std::fixed << std::setprecision(6);
  for (int row = 0; row < 4; ++row) {
    output << "  [";
    for (int column = 0; column < 4; ++column) {
      if (column > 0) {
        output << ", ";
      }
      output << std::setw(9) << transform[4 * row + column];
    }
    output << "]\n";
  }
}

void write_report(const std::filesystem::path& output_path,
                  const Matrix4& world_base,
                  const Matrix4& base_link,
                  const Matrix4& link_end,
                  const Matrix4& world_end,
                  const std::vector<Sample>& samples) {
  std::ofstream output(output_path);
  if (!output) {
    throw std::runtime_error("Could not open " + output_path.string());
  }

  double maximum_composition_error = 0.0;
  double maximum_inverse_error = 0.0;
  double maximum_chain_error = 0.0;
  double maximum_point_error = 0.0;
  double maximum_tool_drift = 0.0;
  double maximum_rotation_error = 0.0;
  double maximum_determinant_error = 0.0;
  for (const Sample& sample : samples) {
    maximum_composition_error =
        std::max(maximum_composition_error, sample.composition_error);
    maximum_inverse_error =
        std::max(maximum_inverse_error, sample.inverse_error);
    maximum_chain_error = std::max(maximum_chain_error, sample.chain_error);
    maximum_point_error =
        std::max(maximum_point_error, sample.point_transform_error);
    maximum_tool_drift =
        std::max(maximum_tool_drift, sample.fixed_tool_pose_drift);
    maximum_rotation_error = std::max(
        maximum_rotation_error,
        std::max(sample.base_rotation_error, sample.end_rotation_error));
    maximum_determinant_error = std::max(
        maximum_determinant_error,
        std::max(std::abs(sample.base_rotation_determinant - 1.0),
                 std::abs(sample.end_rotation_determinant - 1.0)));
  }

  output << "Week 3 MuJoCo frame-transform report\n"
         << "Convention: T_AB maps coordinates from frame {B} to frame {A}.\n"
         << "Identity checked: T_WE = T_WB T_BL T_LE.\n\n";
  print_transform(output, "T_WB", world_base);
  output << '\n';
  print_transform(output, "T_BL", base_link);
  output << '\n';
  print_transform(output, "T_LE", link_end);
  output << '\n';
  print_transform(output, "T_WE (read directly from site pose)", world_end);

  output << "\nMaximum errors over " << samples.size() << " samples:\n"
         << std::scientific << std::setprecision(3)
         << "  rotation orthogonality: " << maximum_rotation_error << '\n'
         << "  |det(R)-1|:             " << maximum_determinant_error << '\n'
         << "  relative composition:   " << maximum_composition_error << '\n'
         << "  transform inverse:       " << maximum_inverse_error << '\n'
         << "  full frame chain:        " << maximum_chain_error << '\n'
         << "  point transformation:    " << maximum_point_error << '\n'
         << "  fixed T_LE drift:         " << maximum_tool_drift << '\n';
}

}  // namespace

int main() {
  constexpr char kModelPath[] = "assets/frame_transforms.xml";
  constexpr char kCsvPath[] = "build/frame_transforms.csv";
  constexpr char kReportPath[] = "build/frame_transforms_report.txt";

  char error[1024] = {};
  mjModel* model = mj_loadXML(kModelPath, nullptr, error, sizeof(error));
  if (model == nullptr) {
    std::cerr << "Failed to load " << kModelPath << ": " << error << '\n';
    return 1;
  }

  mjData* data = mj_makeData(model);
  if (data == nullptr) {
    std::cerr << "Failed to allocate mjData\n";
    mj_deleteModel(model);
    return 1;
  }

  try {
    // Convert readable MJCF names to indices for MuJoCo's flat arrays.
    const int base_id = mj_name2id(model, mjOBJ_BODY, "base");
    const int link_id = mj_name2id(model, mjOBJ_BODY, "link");
    const int joint_id = mj_name2id(model, mjOBJ_JOINT, "shoulder");
    const int site_id = mj_name2id(model, mjOBJ_SITE, "end_effector");
    const int actuator_id =
        mj_name2id(model, mjOBJ_ACTUATOR, "shoulder_position");
    if (base_id < 0 || link_id < 0 || joint_id < 0 || site_id < 0 ||
        actuator_id < 0) {
      throw std::runtime_error("A named model object is missing");
    }

    const int qpos_index = model->jnt_qposadr[joint_id];
    const int qvel_index = model->jnt_dofadr[joint_id];
    data->qpos[qpos_index] = 0.0;
    data->qvel[qvel_index] = 0.0;
    data->ctrl[actuator_id] = 0.0;
    mj_forward(model, data);

    // Record the rigid attachment T_LE once; it must remain constant while the
    // joint moves because the site is fixed to the link body.
    const Matrix4 initial_world_link = body_world_transform(data, link_id);     // T_WL
    const Matrix4 initial_world_end = site_world_transform(data, site_id);      // T_WE
    const Matrix4 initial_link_end =
        transform_utils::relative_pose(initial_world_link, initial_world_end);  // T_LE

    const int physics_steps_per_control = std::max(
        1, static_cast<int>(std::lround(
               (1.0 / kControlFrequency) / model->opt.timestep)));
    const double control_period =
        physics_steps_per_control * model->opt.timestep;
    const int sample_count =
        static_cast<int>(std::ceil(kDuration / control_period));

    std::vector<Sample> samples;
    samples.reserve(sample_count);

    Matrix4 final_world_base{};
    Matrix4 final_base_link{};
    Matrix4 final_link_end{};
    Matrix4 final_world_end{};

    for (int sample_index = 0; sample_index < sample_count; ++sample_index) {
      // In the MDP view this desired angle is A_t. MuJoCo and the actuator
      // together form the environment transition to the next robot state.
      data->ctrl[actuator_id] = desired_angle(data->time);
      for (int step = 0; step < physics_steps_per_control; ++step) {
        mj_step(model, data);
      }

      // Synchronize Cartesian poses with the just-integrated qpos.
      mj_forward(model, data);

      const Matrix4 world_base = body_world_transform(data, base_id);  // T_WB
      const Matrix4 world_link = body_world_transform(data, link_id);  // T_WL
      const Matrix4 world_end = site_world_transform(data, site_id);   // T_WE

      if (!transform_utils::is_homogeneous_transform(
              world_base, kValidationTolerance) ||
          !transform_utils::is_homogeneous_transform(
              world_link, kValidationTolerance) ||
          !transform_utils::is_homogeneous_transform(
              world_end, kValidationTolerance)) {
        throw std::runtime_error("MuJoCo returned an invalid rigid transform");
      }

      // Calculate poses between frames using T_AB = inverse(T_WA) T_WB.
      const Matrix4 base_link =
          transform_utils::relative_pose(world_base, world_link);      // T_BL
      const Matrix4 link_end =
          transform_utils::relative_pose(world_link, world_end);       // T_LE
      const Matrix4 base_end =
          transform_utils::relative_pose(world_base, world_end);       // T_BE

      // Test composition two ways: T_WB T_BE and T_WB T_BL T_LE.
      const Matrix4 composed_world_end =
          transform_utils::multiply_transform(world_base, base_end);
      const Matrix4 chained_world_end = transform_utils::multiply_transform(
          transform_utils::multiply_transform(world_base, base_link), link_end);

      // Transform a nonzero point in {E} directly to {W}, then through {B}.
      // Equal results demonstrate point transformation and frame composition.
      const Vector3 point_end = {0.08, -0.03, 0.02};
      const Vector3 point_world_direct =
          transform_utils::transform_point(world_end, point_end);
      const Vector3 point_base =
          transform_utils::transform_point(base_end, point_end);
      const Vector3 point_world_via_base =
          transform_utils::transform_point(world_base, point_base);

      const RotationValidation base_rotation =
          transform_utils::validate_rotation(
              transform_utils::rotation_of(world_base));
      const RotationValidation end_rotation =
          transform_utils::validate_rotation(
              transform_utils::rotation_of(world_end));

      const Matrix4 inverse_identity = transform_utils::multiply_transform(
          transform_utils::inverse_transform(world_base), world_base);

      samples.push_back({
          data->time,
          data->ctrl[actuator_id],
          data->qpos[qpos_index],
          data->qvel[qvel_index],
          transform_utils::translation_of(world_base),
          transform_utils::translation_of(world_end),
          transform_utils::translation_of(base_end),
          base_rotation.orthogonality_error,
          end_rotation.orthogonality_error,
          base_rotation.determinant,
          end_rotation.determinant,
          transform_utils::max_abs_difference(
              composed_world_end, world_end),
          transform_utils::max_abs_difference(
              inverse_identity, transform_utils::identity_transform()),
          transform_utils::max_abs_difference(chained_world_end, world_end),
          transform_utils::point_distance(
              point_world_direct, point_world_via_base),
          transform_utils::max_abs_difference(link_end, initial_link_end),
      });

      final_world_base = world_base;
      final_base_link = base_link;
      final_link_end = link_end;
      final_world_end = world_end;
    }

    std::filesystem::create_directories("build");
    write_csv(samples, kCsvPath);
    write_report(kReportPath,
                 final_world_base,
                 final_base_link,
                 final_link_end,
                 final_world_end,
                 samples);

    double maximum_error = 0.0;
    for (const Sample& sample : samples) {
      maximum_error = std::max(
          maximum_error,
          std::max({sample.base_rotation_error,
                    sample.end_rotation_error,
                    std::abs(sample.base_rotation_determinant - 1.0),
                    std::abs(sample.end_rotation_determinant - 1.0),
                    sample.composition_error,
                    sample.inverse_error,
                    sample.chain_error,
                    sample.point_transform_error,
                    sample.fixed_tool_pose_drift}));
    }
    if (maximum_error > kValidationTolerance) {
      throw std::runtime_error("A transform check exceeded its tolerance");
    }

    std::cout << "MuJoCo " << mj_versionString() << '\n'
              << "Frames checked: world {W}, base {B}, link {L}, "
                 "end-effector {E}\n"
              << "Physics: " << 1.0 / model->opt.timestep
              << " Hz; control: " << 1.0 / control_period << " Hz\n"
              << "Logged samples: " << samples.size() << '\n'
              << std::scientific << std::setprecision(3)
              << "Maximum transform validation error: " << maximum_error
              << '\n'
              << "CSV: " << kCsvPath << '\n'
              << "Report: " << kReportPath << '\n';
  } catch (const std::exception& exception) {
    std::cerr << "Error: " << exception.what() << '\n';
    mj_deleteData(data);
    mj_deleteModel(model);
    return 1;
  }

  mj_deleteData(data);
  mj_deleteModel(model);
  return 0;
}
