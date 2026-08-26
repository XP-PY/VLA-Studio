/*
  Week 4 MuJoCo experiment: two-link forward kinematics.

  For each deterministic random configuration, this program:

    1. Computes T_BE with the space-form Product of Exponentials formula.
    2. Writes the same joint coordinates into MuJoCo qpos.
    3. Reads the end-effector site's world pose.
    4. Converts it to T_BE and compares position and orientation errors.

  The analytical calculation is independent of MuJoCo's kinematics engine.
*/

#include "mujoco.h"
#include "transform_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using transform_utils::Matrix3;
using transform_utils::Matrix4;
using transform_utils::Vector3;

constexpr double kLink1Length = 0.45;
constexpr double kLink2Length = 0.35;
constexpr int kConfigurationCount = 100;
constexpr unsigned int kRandomSeed = 20260826;
constexpr double kValidationTolerance = 1e-12;

struct ScrewAxis {
  Vector3 omega;  // unit rotation axis expressed in the fixed base frame
  Vector3 v;      // linear component v=-omega x q for a revolute joint
};

struct Comparison {
  int sample;
  double q1;
  double q2;
  Vector3 analytical_base_position;
  Vector3 mujoco_base_position;
  Vector3 analytical_world_position;
  Vector3 mujoco_world_position;
  double position_error;
  double orientation_error;
  double transform_error;
  double poe_closed_form_error;
};

// Construct Skew-symmetric matrix
Matrix3 skew(const Vector3& vector) {
  return {0.0,        -vector[2], vector[1],
          vector[2], 0.0,        -vector[0],
          -vector[1], vector[0],  0.0};
}

// Weighted sum of matrices
Matrix3 add_scaled(const Matrix3& first,
                   double first_scale,
                   const Matrix3& second,
                   double second_scale,
                   const Matrix3& third,
                   double third_scale) {
  Matrix3 result{};
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index] = first_scale * first[index] +
                    second_scale * second[index] +
                    third_scale * third[index];
  }
  return result;
}

// Matrix multiplication of vectors
Vector3 matrix_vector_product(const Matrix3& matrix, const Vector3& vector) {
  return {
      matrix[0] * vector[0] + matrix[1] * vector[1] +
          matrix[2] * vector[2],
      matrix[3] * vector[0] + matrix[4] * vector[1] +
          matrix[5] * vector[2],
      matrix[6] * vector[0] + matrix[7] * vector[1] +
          matrix[8] * vector[2],
  };
}

// Closed-form matrix exponential e^[S]theta for a unit revolute screw axis:
// R=I+sin(theta)[omega]+(1-cos(theta))[omega]^2 and
// p=(I theta+(1-cos(theta))[omega]+(theta-sin(theta))[omega]^2)v.
Matrix4 revolute_twist_exponential(const ScrewAxis& screw, double theta) {
  const double omega_norm = std::sqrt(
      screw.omega[0] * screw.omega[0] +
      screw.omega[1] * screw.omega[1] +
      screw.omega[2] * screw.omega[2]);
  if (std::abs(omega_norm - 1.0) > 1e-12) {
    throw std::runtime_error("Revolute screw axis omega must have unit length");
  }

  const Matrix3 identity = transform_utils::identity_rotation();
  const Matrix3 omega_hat = skew(screw.omega);
  const Matrix3 omega_hat_squared =
      transform_utils::multiply(omega_hat, omega_hat);

  const Matrix3 rotation = add_scaled(
      identity, 1.0,
      omega_hat, std::sin(theta),
      omega_hat_squared, 1.0 - std::cos(theta));
  const Matrix3 translation_operator = add_scaled(
      identity, theta,
      omega_hat, 1.0 - std::cos(theta),
      omega_hat_squared, theta - std::sin(theta));
  const Vector3 translation =
      matrix_vector_product(translation_operator, screw.v);
  return transform_utils::make_transform(rotation, translation);
}

// Space-form PoE: T_BE(theta)=e^[S1]theta1 e^[S2]theta2 M.
Matrix4 analytical_forward_kinematics(double q1, double q2) {
  const ScrewAxis shoulder = {{0.0, 0.0, 1.0}, {0.0, 0.0, 0.0}};
  const ScrewAxis elbow = {
      {0.0, 0.0, 1.0},
      {0.0, -kLink1Length, 0.0},
  };
  const Matrix4 home = transform_utils::make_transform(
      transform_utils::identity_rotation(),
      {kLink1Length + kLink2Length, 0.0, 0.0});

  return transform_utils::multiply_transform(
      transform_utils::multiply_transform(
          revolute_twist_exponential(shoulder, q1),
          revolute_twist_exponential(elbow, q2)),
          home);
}

// Independent planar trigonometric formula used to validate the PoE code.
Matrix4 closed_form_forward_kinematics(double q1, double q2) {
  const double total_angle = q1 + q2;
  const Matrix3 rotation = {
      std::cos(total_angle), -std::sin(total_angle), 0.0,
      std::sin(total_angle), std::cos(total_angle),  0.0,
      0.0,                   0.0,                    1.0,
  };
  const Vector3 position = {
      kLink1Length * std::cos(q1) + kLink2Length * std::cos(total_angle),
      kLink1Length * std::sin(q1) + kLink2Length * std::sin(total_angle),
      0.0,
  };
  return transform_utils::make_transform(rotation, position);
}

Matrix4 transform_from_mujoco_pose(const mjtNum* position,
                                   const mjtNum* rotation) {
  const Vector3 translation = {position[0], position[1], position[2]};
  Matrix3 rotation_matrix{};
  std::copy_n(rotation, rotation_matrix.size(), rotation_matrix.begin());
  return transform_utils::make_transform(rotation_matrix, translation);
}

Matrix4 body_world_transform(const mjData* data, int body_id) {
  return transform_from_mujoco_pose(data->xpos + 3 * body_id,
                                    data->xmat + 9 * body_id);
}

Matrix4 site_world_transform(const mjData* data, int site_id) {
  return transform_from_mujoco_pose(data->site_xpos + 3 * site_id,
                                    data->site_xmat + 9 * site_id);
}

// Geodesic angle of R_a^T R_b, expressed in radians.
double rotation_angle_error(const Matrix4& first, const Matrix4& second) {
  const Matrix3 relative_rotation = transform_utils::multiply(
      transform_utils::transpose(transform_utils::rotation_of(first)),
      transform_utils::rotation_of(second));
  const double trace =
      relative_rotation[0] + relative_rotation[4] + relative_rotation[8];
  const double skew_x = relative_rotation[7] - relative_rotation[5];
  const double skew_y = relative_rotation[2] - relative_rotation[6];
  const double skew_z = relative_rotation[3] - relative_rotation[1];
  const double sine = 0.5 *
      std::sqrt(skew_x * skew_x + skew_y * skew_y + skew_z * skew_z);
  const double cosine = std::clamp(0.5 * (trace - 1.0), -1.0, 1.0);
  return std::atan2(sine, cosine);
}

void write_csv(const std::vector<Comparison>& comparisons,
               const std::filesystem::path& output_path) {
  std::ofstream output(output_path);
  if (!output) {
    throw std::runtime_error("Could not open " + output_path.string());
  }

  output << "sample,q1_rad,q2_rad,"
            "analytical_base_x_m,analytical_base_y_m,analytical_base_z_m,"
            "mujoco_base_x_m,mujoco_base_y_m,mujoco_base_z_m,"
            "analytical_world_x_m,analytical_world_y_m,analytical_world_z_m,"
            "mujoco_world_x_m,mujoco_world_y_m,mujoco_world_z_m,"
            "position_error_m,orientation_error_rad,max_transform_error,"
            "poe_closed_form_error\n";
  output << std::setprecision(14);

  for (const Comparison& item : comparisons) {
    output << item.sample << ',' << item.q1 << ',' << item.q2;
    for (double value : item.analytical_base_position) {
      output << ',' << value;
    }
    for (double value : item.mujoco_base_position) {
      output << ',' << value;
    }
    for (double value : item.analytical_world_position) {
      output << ',' << value;
    }
    for (double value : item.mujoco_world_position) {
      output << ',' << value;
    }
    output << ',' << item.position_error
           << ',' << item.orientation_error
           << ',' << item.transform_error
           << ',' << item.poe_closed_form_error << '\n';
  }
}

void write_report(const std::vector<Comparison>& comparisons,
                  const std::filesystem::path& output_path) {
  std::ofstream output(output_path);
  if (!output) {
    throw std::runtime_error("Could not open " + output_path.string());
  }

  std::vector<Comparison> ranked = comparisons;
  std::sort(ranked.begin(), ranked.end(),
            [](const Comparison& left, const Comparison& right) {
              return left.transform_error > right.transform_error;
            });

  double maximum_position_error = 0.0;
  double maximum_orientation_error = 0.0;
  double maximum_transform_error = 0.0;
  double maximum_closed_form_error = 0.0;
  for (const Comparison& item : comparisons) {
    maximum_position_error =
        std::max(maximum_position_error, item.position_error);
    maximum_orientation_error =
        std::max(maximum_orientation_error, item.orientation_error);
    maximum_transform_error =
        std::max(maximum_transform_error, item.transform_error);
    maximum_closed_form_error =
        std::max(maximum_closed_form_error, item.poe_closed_form_error);
  }

  output << "Week 4 two-link forward-kinematics comparison\n\n"
         << "Space-form model:\n"
         << "  S1 = [0, 0, 1, 0, 0, 0]^T\n"
         << "  S2 = [0, 0, 1, 0, -0.45, 0]^T\n"
         << "  M translation = [0.80, 0, 0] m\n"
         << "  T_BE = exp([S1]q1) exp([S2]q2) M\n\n"
         << "Configurations: " << comparisons.size()
         << " (deterministic seed " << kRandomSeed << ")\n"
         << std::scientific << std::setprecision(3)
         << "Maximum position error:       " << maximum_position_error
         << " m\n"
         << "Maximum orientation error:    " << maximum_orientation_error
         << " rad\n"
         << "Maximum transform-entry error: " << maximum_transform_error
         << "\n"
         << "Maximum PoE/trigonometric error: " << maximum_closed_form_error
         << "\n\n"
         << "Five largest transform-entry errors:\n";

  const std::size_t rank_count = std::min<std::size_t>(5, ranked.size());
  for (std::size_t rank = 0; rank < rank_count; ++rank) {
    const Comparison& item = ranked[rank];
    output << "  " << rank + 1 << ". sample " << item.sample
           << ", q=[" << item.q1 << ", " << item.q2 << "]"
           << ", position=" << item.position_error << " m"
           << ", rotation=" << item.orientation_error << " rad"
           << ", max-entry=" << item.transform_error << '\n';
  }

  output << "\nExplanation of the largest cases:\n"
         << "  All errors are at floating-point roundoff scale. The PoE code, "
            "closed-form planar equations,\n"
         << "  and MuJoCo use mathematically equivalent trigonometric operations "
            "but evaluate them in different\n"
         << "  orders. Consequently, the last few binary digits can differ. "
            "There is no configuration-dependent\n"
         << "  modeling failure, and these cases are not kinematic singularities. "
            "Errors substantially above this\n"
         << "  scale would instead suggest an incorrect screw axis or home pose, "
            "a world/base frame mix-up, a\n"
         << "  wrong joint address, or failure to call mj_forward after setting qpos.\n";
}

}  // namespace

int main() {
  constexpr char kModelPath[] = "assets/two_link_arm.xml";
  constexpr char kCsvPath[] = "build/forward_kinematics_comparison.csv";
  constexpr char kReportPath[] = "build/forward_kinematics_report.txt";

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
    const int base_id = mj_name2id(model, mjOBJ_BODY, "base");
    const int shoulder_id = mj_name2id(model, mjOBJ_JOINT, "shoulder");
    const int elbow_id = mj_name2id(model, mjOBJ_JOINT, "elbow");
    const int site_id = mj_name2id(model, mjOBJ_SITE, "end_effector");
    if (base_id < 0 || shoulder_id < 0 || elbow_id < 0 || site_id < 0) {
      throw std::runtime_error("A named model object is missing");
    }

    const int shoulder_qpos = model->jnt_qposadr[shoulder_id];
    const int elbow_qpos = model->jnt_qposadr[elbow_id];

    std::mt19937 random_engine(kRandomSeed);
    std::uniform_real_distribution<double> shoulder_distribution(-1.4, 1.4);
    std::uniform_real_distribution<double> elbow_distribution(-1.6, 1.6);

    std::vector<Comparison> comparisons;
    comparisons.reserve(kConfigurationCount);

    for (int sample = 0; sample < kConfigurationCount; ++sample) {
      const double q1 = shoulder_distribution(random_engine);
      const double q2 = elbow_distribution(random_engine);

      // Directly set joint coordinates for a kinematic comparison, then ask
      // MuJoCo to recompute every derived body and site pose.
      data->qpos[shoulder_qpos] = q1;
      data->qpos[elbow_qpos] = q2;
      mj_forward(model, data);

      const Matrix4 world_base = body_world_transform(data, base_id);
      const Matrix4 world_end_mujoco = site_world_transform(data, site_id);
      const Matrix4 base_end_mujoco =
          transform_utils::relative_pose(world_base, world_end_mujoco);

      const Matrix4 base_end_analytical =
          analytical_forward_kinematics(q1, q2);
      const Matrix4 base_end_closed_form =
          closed_form_forward_kinematics(q1, q2);
      const Matrix4 world_end_analytical =
          transform_utils::multiply_transform(
              world_base, base_end_analytical);

      const Vector3 analytical_base_position =
          transform_utils::translation_of(base_end_analytical);
      const Vector3 mujoco_base_position =
          transform_utils::translation_of(base_end_mujoco);
      const Vector3 analytical_world_position =
          transform_utils::translation_of(world_end_analytical);
      const Vector3 mujoco_world_position =
          transform_utils::translation_of(world_end_mujoco);

      comparisons.push_back({
          sample,
          q1,
          q2,
          analytical_base_position,
          mujoco_base_position,
          analytical_world_position,
          mujoco_world_position,
          transform_utils::point_distance(
              analytical_world_position, mujoco_world_position),
          rotation_angle_error(world_end_analytical, world_end_mujoco),
          transform_utils::max_abs_difference(
              world_end_analytical, world_end_mujoco),
          transform_utils::max_abs_difference(
              base_end_analytical, base_end_closed_form),
      });
    }

    double maximum_error = 0.0;
    for (const Comparison& item : comparisons) {
      maximum_error = std::max(
          maximum_error,
          std::max({item.position_error,
                    item.orientation_error,
                    item.transform_error,
                    item.poe_closed_form_error}));
    }
    if (maximum_error > kValidationTolerance) {
      throw std::runtime_error(
          "Analytical and MuJoCo forward kinematics disagree");
    }

    std::filesystem::create_directories("build");
    write_csv(comparisons, kCsvPath);
    write_report(comparisons, kReportPath);

    std::cout << "Two-link Product-of-Exponentials comparison\n"
              << "Random configurations: " << comparisons.size()
              << " (seed " << kRandomSeed << ")\n"
              << std::scientific << std::setprecision(3)
              << "Maximum combined error: " << maximum_error << '\n'
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
