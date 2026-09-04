#pragma once
#include <cmath>
#include <cstdint>
#include <string>

// ─── Raw sensor frames
// ────────────────────────────────────────────────────────

struct ImuFrame {
  int64_t timestamp_ns;
  float ax, ay, az; // m/s²  (uncalibrated — bias subtracted by calibrator)
  float gx, gy, gz; // rad/s
};

struct GnssFrame {
  int64_t timestamp_ns;
  double lat, lon, alt; // degrees, degrees, metres (WGS-84)
  float vN, vE, vD;     // velocity NED m/s  (NaN if unavailable)
  float hacc;           // horizontal accuracy 1-sigma (m)
  float vacc;           // vertical  accuracy 1-sigma (m)
  int satellites;
  float hdop;
  bool valid;
};

struct AuxFrame {
  int64_t timestamp_ns;
  float mx, my, mz; // magnetometer µT
  float baro_pa;    // barometer Pa (0 if absent)
};

// ─── Navigation state
// ─────────────────────────────────────────────────────────

enum class GnssMode : uint8_t {
  GNSS_GOOD = 0,
  GNSS_DEGRADED = 1,
  DR_ACTIVE = 2,
  MAP_MATCHED = 3,
};

struct Quat {
  float w, x, y, z;
  Quat() : w(1), x(0), y(0), z(0) {}
  Quat(float w, float x, float y, float z) : w(w), x(x), y(y), z(z) {}
  float norm() const { return sqrtf(w * w + x * x + y * y + z * z); }
  void normalise() {
    float n = norm();
    w /= n;
    x /= n;
    y /= n;
    z /= n;
  }
};

struct Vec3 {
  float x, y, z;
  Vec3() : x(0), y(0), z(0) {}
  Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
};

struct NavState {
  int64_t timestamp_ns = 0;
  // Position
  double lat = 0, lon = 0, alt = 0; // WGS-84 degrees / metres
  // Velocity NED (m/s)
  float vN = 0, vE = 0, vD = 0;
  // Attitude
  Quat q;                             // body-to-nav quaternion
  float roll = 0, pitch = 0, yaw = 0; // derived Euler (rad)
  // Sensor biases
  Vec3 bias_acc, bias_gyro;
  // Uncertainty (1-sigma, metres)
  float pos_std_N = 999, pos_std_E = 999;
  // Mode
  GnssMode mode = GnssMode::GNSS_GOOD;
  // Outage duration (s)
  float outage_seconds = 0;
};

// ─── Pseudo-GNSS measurement (from ML model during outage) ───────────────────

struct PseudoMeas {
  int64_t timestamp_ns;
  float dN, dE;       // position increment in nav frame (metres)
  float std_N, std_E; // uncertainty (m)
};
