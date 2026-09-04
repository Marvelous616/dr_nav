#pragma once
#include "quat_math.h"
#include "types.h"
#include <cmath>

// ─── IMU Mechanization
// ──────────────────────────────────────────────────────── Integrates
// calibrated specific force (accel) and angular rate (gyro) to propagate
// position, velocity and attitude in the local-level NED frame. Reference:
// Groves "Principles of GNSS, Inertial, and Multisensor Navigation"

namespace mech {

// WGS-84 constants
static constexpr double EARTH_RADIUS = 6378137.0; // metres (equatorial)
static constexpr float g0 = 9.80665f;             // m/s² standard gravity
static constexpr float G_NAV[3] = {0, 0, g0};     // gravity in NED frame

// Convert NED position offset (metres) to lat/lon delta (degrees)
inline void ned_to_lld(double lat_rad, float dN, float dE, double &dlat,
                       double &dlon) {
  dlat = (dN / EARTH_RADIUS) * (180.0 / M_PI);
  dlon = (dE / (EARTH_RADIUS * cos(lat_rad))) * (180.0 / M_PI);
}

// Propagate NavState forward by one IMU sample (Euler integration)
// acc_b : specific force in body frame (m/s²), already bias-subtracted
// gyr_b : angular rate in body frame (rad/s), already bias-subtracted
// dt    : time step (seconds)
inline void propagate(NavState &s, const Vec3 &acc_b, const Vec3 &gyr_b,
                      float dt) {

  // 1. Attitude update — integrate angular rate via rotation vector
  Vec3 rot_vec = {gyr_b.x * dt, gyr_b.y * dt, gyr_b.z * dt};
  Quat dq = quat::from_rot_vec(rot_vec);
  s.q = quat::mul(s.q, dq);
  s.q.normalise();
  quat::to_euler(s.q, s.roll, s.pitch, s.yaw);

  // 2. Specific force in navigation frame
  Vec3 acc_n = quat::rotate(s.q, acc_b);

  // 3. Velocity update (subtract gravity)
  float vN_prev = s.vN, vE_prev = s.vE, vD_prev = s.vD;
  s.vN += (acc_n.x - G_NAV[0]) * dt;
  s.vE += (acc_n.y - G_NAV[1]) * dt;
  s.vD += (acc_n.z - G_NAV[2]) * dt; // gravity is +g in D direction

  // 4. Position update (use average velocity for trapezoidal integration)
  float avg_vN = (vN_prev + s.vN) * 0.5f;
  float avg_vE = (vE_prev + s.vE) * 0.5f;
  float avg_vD = (vD_prev + s.vD) * 0.5f;

  double lat_rad = s.lat * M_PI / 180.0;
  double dlat, dlon;
  ned_to_lld(lat_rad, avg_vN * dt, avg_vE * dt, dlat, dlon);
  s.lat += dlat;
  s.lon += dlon;
  s.alt -= avg_vD * dt; // D is positive down, alt is positive up
}

// Non-Holonomic Constraints (NHC) pseudo-measurement
// A ground vehicle cannot slide sideways (v_lateral ≈ 0) or fly (v_up ≈ 0)
// Returns: lateral and vertical velocity residuals in body frame
inline void nhc_residuals(const NavState &s, float &v_lat, float &v_up) {
  // Rotate nav-frame velocity into body frame
  Quat q_inv = quat::conjugate(s.q);
  Vec3 v_nav = {s.vN, s.vE, s.vD};
  Vec3 v_body = quat::rotate(q_inv, v_nav);
  v_lat = v_body.y; // lateral (Y body axis)
  v_up = v_body.z;  // vertical (Z body axis, positive down in NED)
}

} // namespace mech
