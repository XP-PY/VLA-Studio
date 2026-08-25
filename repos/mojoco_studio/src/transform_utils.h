#ifndef MOJOCO_STUDIO_TRANSFORM_UTILS_H_
#define MOJOCO_STUDIO_TRANSFORM_UTILS_H_

/*
  Week 3 reusable rigid-body transformation utilities.

  Frame convention:

      T_AB maps coordinates expressed in frame {B} into frame {A}.

  Therefore T_AB * T_BC = T_AC, inverse(T_AB) = T_BA, and
  relative_pose(T_WA, T_WB) returns T_AB.

  Matrix3 and Matrix4 are row-major, matching MuJoCo's xmat/site_xmat layout.
*/

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace transform_utils {

using Vector3 = std::array<double, 3>;
using Matrix3 = std::array<double, 9>;
using Matrix4 = std::array<double, 16>;

struct RotationValidation {
  double orthogonality_error;  // max absolute entry of R^T R - I
  double determinant;          // must be close to +1 for a proper rotation
  bool valid;
};

// Return the 3x3 identity rotation.
inline Matrix3 identity_rotation() {
  return {1.0, 0.0, 0.0,
          0.0, 1.0, 0.0,
          0.0, 0.0, 1.0};
}

// Return the 4x4 identity homogeneous transformation.
inline Matrix4 identity_transform() {
  return {1.0, 0.0, 0.0, 0.0,
          0.0, 1.0, 0.0, 0.0,
          0.0, 0.0, 1.0, 0.0,
          0.0, 0.0, 0.0, 1.0};
}

// Transpose a row-major 3x3 matrix.
inline Matrix3 transpose(const Matrix3& matrix) {
  Matrix3 result{};
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t column = 0; column < 3; ++column) {
      result[3 * row + column] = matrix[3 * column + row];
    }
  }
  return result;
}

// Multiply two row-major 3x3 matrices.
inline Matrix3 multiply(const Matrix3& left, const Matrix3& right) {
  Matrix3 result{};
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t column = 0; column < 3; ++column) {
      for (std::size_t inner = 0; inner < 3; ++inner) {
        result[3 * row + column] +=
            left[3 * row + inner] * right[3 * inner + column];
      }
    }
  }
  return result;
}

// Calculate det(R). A proper rotation matrix must have determinant +1.
inline double determinant(const Matrix3& matrix) {
  return matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]) -
         matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6]) +
         matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
}

// Validate the two defining numerical properties of SO(3):
// R^T R = I and det(R) = +1.
inline RotationValidation validate_rotation(
    const Matrix3& rotation, double tolerance = 1e-9) {
  const Matrix3 gram = multiply(transpose(rotation), rotation);
  const Matrix3 identity = identity_rotation();

  double orthogonality_error = 0.0;
  for (std::size_t index = 0; index < gram.size(); ++index) {
    orthogonality_error = std::max(
        orthogonality_error, std::abs(gram[index] - identity[index]));
  }

  const double det = determinant(rotation);
  return {orthogonality_error,
          det,
          orthogonality_error <= tolerance &&
              std::abs(det - 1.0) <= tolerance};
}

// Assemble T=(R,p) in the standard homogeneous form [R p; 0 1].
inline Matrix4 make_transform(const Matrix3& rotation,
                              const Vector3& translation) {
  return {rotation[0], rotation[1], rotation[2], translation[0],
          rotation[3], rotation[4], rotation[5], translation[1],
          rotation[6], rotation[7], rotation[8], translation[2],
          0.0,         0.0,         0.0,         1.0};
}

// Extract R from T=(R,p).
inline Matrix3 rotation_of(const Matrix4& transform) {
  return {transform[0], transform[1], transform[2],
          transform[4], transform[5], transform[6],
          transform[8], transform[9], transform[10]};
}

// Extract p from T=(R,p).
inline Vector3 translation_of(const Matrix4& transform) {
  return {transform[3], transform[7], transform[11]};
}

// Verify that the upper-left block is a rotation and the last row is
// [0, 0, 0, 1].
inline bool is_homogeneous_transform(const Matrix4& transform,
                                     double tolerance = 1e-9) {
  const bool valid_last_row =
      std::abs(transform[12]) <= tolerance &&
      std::abs(transform[13]) <= tolerance &&
      std::abs(transform[14]) <= tolerance &&
      std::abs(transform[15] - 1.0) <= tolerance;
  return valid_last_row &&
         validate_rotation(rotation_of(transform), tolerance).valid;
}

// Homogeneous-transform composition. With the stated frame convention,
// multiply_transform(T_AB, T_BC) returns T_AC.
inline Matrix4 multiply_transform(const Matrix4& left,
                                  const Matrix4& right) {
  Matrix4 result{};
  for (std::size_t row = 0; row < 4; ++row) {
    for (std::size_t column = 0; column < 4; ++column) {
      for (std::size_t inner = 0; inner < 4; ++inner) {
        result[4 * row + column] +=
            left[4 * row + inner] * right[4 * inner + column];
      }
    }
  }
  return result;
}

// Rigid-transform inverse using T^-1=[R^T, -R^T p; 0, 1].
inline Matrix4 inverse_transform(const Matrix4& transform) {
  const Matrix3 inverse_rotation = transpose(rotation_of(transform));
  const Vector3 translation = translation_of(transform);
  const Vector3 inverse_translation = {
      -(inverse_rotation[0] * translation[0] +
        inverse_rotation[1] * translation[1] +
        inverse_rotation[2] * translation[2]),
      -(inverse_rotation[3] * translation[0] +
        inverse_rotation[4] * translation[1] +
        inverse_rotation[5] * translation[2]),
      -(inverse_rotation[6] * translation[0] +
        inverse_rotation[7] * translation[1] +
        inverse_rotation[8] * translation[2]),
  };
  return make_transform(inverse_rotation, inverse_translation);
}

// Transform a point: p_A = T_AB p_B = R_AB p_B + p_AB.
inline Vector3 transform_point(const Matrix4& transform,
                               const Vector3& point) {
  return {
      transform[0] * point[0] + transform[1] * point[1] +
          transform[2] * point[2] + transform[3],
      transform[4] * point[0] + transform[5] * point[1] +
          transform[6] * point[2] + transform[7],
      transform[8] * point[0] + transform[9] * point[1] +
          transform[10] * point[2] + transform[11],
  };
}

// Given poses T_WA and T_WB, calculate frame {B} relative to frame {A}:
// T_AB = T_AW T_WB = inverse(T_WA) T_WB.
inline Matrix4 relative_pose(const Matrix4& world_reference,
                             const Matrix4& world_target) {
  return multiply_transform(inverse_transform(world_reference), world_target);
}

// Maximum componentwise difference, useful for numerical transform checks.
inline double max_abs_difference(const Matrix4& left, const Matrix4& right) {
  double maximum = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    maximum = std::max(maximum, std::abs(left[index] - right[index]));
  }
  return maximum;
}

// Euclidean distance between two 3D points.
inline double point_distance(const Vector3& left, const Vector3& right) {
  const double dx = left[0] - right[0];
  const double dy = left[1] - right[1];
  const double dz = left[2] - right[2];
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

}  // namespace transform_utils

#endif  // MOJOCO_STUDIO_TRANSFORM_UTILS_H_
