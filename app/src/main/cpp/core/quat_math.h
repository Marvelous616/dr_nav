#pragma once
#include "types.h"
#include <cmath>

// ─── Quaternion maths (Hamilton convention: q = [w, x, y, z]) ────────────────
// Reference: Solà "Quaternion kinematics for the error-state KF" (arXiv
// 1711.02508)

namespace quat {

// q_out = q_a ⊗ q_b  (right-to-left composition: apply b first, then a)
inline Quat mul(const Quat &a, const Quat &b) {
  return {a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
          a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
          a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
          a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w};
}

inline Quat conjugate(const Quat &q) { return {q.w, -q.x, -q.y, -q.z}; }

// Rotate vector v by quaternion q  (v' = q ⊗ [0,v] ⊗ q*)
inline Vec3 rotate(const Quat &q, const Vec3 &v) {
  // Optimised form
  float t2 = q.w * q.x, t3 = q.w * q.y, t4 = q.w * q.z;
  float t5 = -q.x * q.x, t6 = q.x * q.y, t7 = q.x * q.z;
  float t8 = -q.y * q.y, t9 = q.y * q.z, t10 = -q.z * q.z;
  return {2 * ((t8 + t10) * v.x + (t6 - t4) * v.y + (t3 + t7) * v.z) + v.x,
          2 * ((t4 + t6) * v.x + (t5 + t10) * v.y + (t9 - t2) * v.z) + v.y,
          2 * ((t7 - t3) * v.x + (t2 + t9) * v.y + (t5 + t8) * v.z) + v.z};
}

// Small-angle rotation vector → quaternion  (first-order approx)
inline Quat from_rot_vec(const Vec3 &rv) {
  float angle2 = rv.x * rv.x + rv.y * rv.y + rv.z * rv.z;
  float s = 0.5f;
  // Use sin(|θ|/2)/|θ| ≈ 0.5 for small angles
  if (angle2 > 1e-8f) {
    float angle = sqrtf(angle2);
    s = sinf(angle * 0.5f) / angle;
  }
  Quat q{cosf(sqrtf(angle2) * 0.5f), rv.x * s, rv.y * s, rv.z * s};
  q.normalise();
  return q;
}

// Quaternion → rotation vector (logarithmic map)
inline Vec3 to_rot_vec(const Quat &q) {
  float xyz_norm = sqrtf(q.x * q.x + q.y * q.y + q.z * q.z);
  if (xyz_norm < 1e-8f)
    return {0, 0, 0};
  float angle = 2.0f * atan2f(xyz_norm, q.w);
  float s = angle / xyz_norm;
  return {q.x * s, q.y * s, q.z * s};
}

// Error quaternion δq = q_true ⊗ q_est*   (body-frame error)
inline Quat body_error(const Quat &q_true, const Quat &q_est) {
  return mul(q_true, conjugate(q_est));
}

// Error quaternion δq = q_est* ⊗ q_true   (navigation-frame error)
// Reference: Li & Chang, IEEE Sensors 2020
inline Quat nav_error(const Quat &q_true, const Quat &q_est) {
  return mul(conjugate(q_est), q_true);
}

// Convert quaternion → roll/pitch/yaw (ZYX Euler, radians)
inline void to_euler(const Quat &q, float &roll, float &pitch, float &yaw) {
  roll = atan2f(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y));
  pitch = asinf(2 * (q.w * q.y - q.z * q.x));
  yaw = atan2f(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z));
}

// Identity quaternion
inline Quat identity() { return {1, 0, 0, 0}; }

} // namespace quat
