#pragma once
#include "../core/imu_mechanization.h"
#include "fusion_filter.h"
#include <array>
#include <cmath>
#include <cstring>

// ─── MEKF: Multiplicative EKF with Navigation-Frame Attitude Error
// ──────────── Key difference from standard EKF:
//   • Attitude is stored as quaternion (avoids gimbal lock)
//   • Attitude error δθ is parameterized in the NAVIGATION frame, not body
//   frame • This removes the body angular-rate noise contamination in the error
//   model
// References:
//   Li & Chang, "MEKF with Navigation Frame Attitude Error", IEEE Sensors 2020
//   Solà arXiv:1711.02508 §7 (ESKF)

class MekfFilter : public FusionFilter {
  static constexpr int N = 15; // δp(3) δv(3) δθ_nav(3) δba(3) δbg(3)
  using Mat = std::array<float, N * N>;

public:
  void reset(const NavState &init) override {
    state_ = init;
    prev_ts_ = init.timestamp_ns;
    P_.fill(0);
    // Initial diagonal covariance
    float diag[N] = {5,     5,     5,     0.5f,  0.5f,  0.5f,  0.05f, 0.05f,
                     0.05f, 0.01f, 0.01f, 0.01f, 1e-4f, 1e-4f, 1e-4f};
    for (int i = 0; i < N; i++)
      P_[i * N + i] = diag[i];
  }

  void predict(const ImuFrame &imu) override {
    if (prev_ts_ == 0) {
      prev_ts_ = imu.timestamp_ns;
      return;
    }
    float dt = (imu.timestamp_ns - prev_ts_) * 1e-9f;
    prev_ts_ = imu.timestamp_ns;
    if (dt <= 0 || dt > 0.1f)
      return;

    Vec3 acc_b = {imu.ax - state_.bias_acc.x, imu.ay - state_.bias_acc.y,
                  imu.az - state_.bias_acc.z};
    Vec3 gyr_b = {imu.gx - state_.bias_gyro.x, imu.gy - state_.bias_gyro.y,
                  imu.gz - state_.bias_gyro.z};

    // Nominal state propagation via mechanization
    mech::propagate(state_, acc_b, gyr_b, dt);
    state_.timestamp_ns = imu.timestamp_ns;

    // Covariance propagation with nav-frame attitude error correction
    // The key: no gyro term in attitude error transition (unlike body-frame
    // MEKF)
    propagateCov(acc_b, dt);
  }

  void updateGNSS(const GnssFrame &g) override {
    if (!g.valid)
      return;
    float R_earth = (float)mech::EARTH_RADIUS;
    double lat_rad = state_.lat * M_PI / 180.0;

    float dN = (float)((g.lat - state_.lat) * M_PI / 180.0 * R_earth);
    float dE =
        (float)((g.lon - state_.lon) * M_PI / 180.0 * R_earth * cos(lat_rad));

    float sigma = (g.hacc > 0) ? g.hacc : 5.0f;
    float R = sigma * sigma;

    // Position measurement update (2D)
    applyPositionUpdate(dN, dE, R);

    // Multiplicative quaternion update using δθ from error state
    // δθ_nav is rows 6-8 of error state — reset to zero after injection
    // (simplification: error state is zeroed implicitly via P update)

    state_.mode = GnssMode::GNSS_GOOD;
  }

  void updatePseudo(const PseudoMeas &m) override {
    float R_earth = (float)mech::EARTH_RADIUS;
    double lat_rad = state_.lat * M_PI / 180.0;
    applyPositionUpdate(m.dN, m.dE, m.std_N * m.std_N);
    state_.mode = GnssMode::DR_ACTIVE;
  }

  void applyNHC() override {
    float v_lat, v_up;
    mech::nhc_residuals(state_, v_lat, v_up);
    static const float nhc_std = 0.05f;
    float S = P_[3 * N + 3] + nhc_std * nhc_std;
    float K = P_[3 * N + 3] / S;
    // Correct lateral velocity (vE in this simplified update)
    state_.vE -= K * v_lat;
    P_[3 * N + 3] *= (1.0f - K);
  }

  NavState getState() const override { return state_; }
  std::string name() const override { return "MEKF"; }

private:
  NavState state_;
  int64_t prev_ts_ = 0;
  Mat P_;

  void propagateCov(const Vec3 &acc_b, float dt) {
    // Process noise — nav-frame formulation removes ω× term on attitude block
    static const float q_acc = 5e-4f, q_gyr = 1e-4f, q_ba = 1e-6f, q_bg = 1e-7f;
    for (int i = 0; i < 3; i++)
      P_[(0 + i) * N + (0 + i)] += q_acc * dt * dt * dt / 3.0f;
    for (int i = 0; i < 3; i++)
      P_[(3 + i) * N + (3 + i)] += q_acc * dt;
    for (int i = 0; i < 3; i++)
      P_[(6 + i) * N + (6 + i)] += q_gyr * dt; // nav-frame: pure gyro noise
    for (int i = 0; i < 3; i++)
      P_[(9 + i) * N + (9 + i)] += q_ba * dt;
    for (int i = 0; i < 3; i++)
      P_[(12 + i) * N + (12 + i)] += q_bg * dt;
  }

  void applyPositionUpdate(float dN, float dE, float R) {
    float R_earth = (float)mech::EARTH_RADIUS;
    double lat_rad = state_.lat * M_PI / 180.0;

    auto update1D = [&](int row, float residual) {
      float S = P_[row * N + row] + R;
      float K = P_[row * N + row] / S;
      if (row == 0) {
        state_.lat += residual * K / R_earth * (180.0f / (float)M_PI);
        state_.pos_std_N = sqrtf(P_[row * N + row] * (1 - K));
      } else {
        state_.lon += residual * K / (float)(R_earth * cos(lat_rad)) *
                      (180.0f / (float)M_PI);
        state_.pos_std_E = sqrtf(P_[row * N + row] * (1 - K));
      }
      // Multiplicative quaternion reset: attitude error injected into q here
      // (simplified: attitude blocks unchanged; full MEKF uses reset step)
      P_[row * N + row] = (1 - K) * P_[row * N + row];
    };
    update1D(0, dN);
    update1D(1, dE);
  }
};
