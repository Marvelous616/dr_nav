#pragma once
#include "../core/imu_mechanization.h"
#include "fusion_filter.h"
#include <array>
#include <cmath>
#include <cstring>

// ─── EKF: standard Extended Kalman Filter ────────────────────────────────────
// Error state: δx = [δp(3), δv(3), δθ(3), δba(3), δbg(3)]  = 15-dim
// Attitude error expressed in BODY frame (traditional formulation).
// Reference: Groves ch.14, Solà arXiv:1711.02508

class EkfFilter : public FusionFilter {
  static constexpr int N = 15;
  using Mat = std::array<float, N * N>;

public:
  void reset(const NavState &init) override {
    state_ = init;
    prev_ts_ = init.timestamp_ns;
    // Initial covariance: diagonal
    P_.fill(0);
    setDiag(P_, {1, 1, 1, 0.1f, 0.1f, 0.1f, 0.01f, 0.01f, 0.01f, 0.01f, 0.01f,
                 0.01f, 1e-4f, 1e-4f, 1e-4f});
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

    // Propagate nominal state
    mech::propagate(state_, acc_b, gyr_b, dt);
    state_.timestamp_ns = imu.timestamp_ns;

    // Propagate covariance: P = F*P*F' + Q
    // Simplified F (identity + kinematics coupling)
    buildAndApplyF(acc_b, gyr_b, dt);
  }

  void updateGNSS(const GnssFrame &g) override {
    if (!g.valid)
      return;
    // Measurement residual: z = [lat_err_m, lon_err_m]
    double lat_rad = state_.lat * M_PI / 180.0;
    float R_earth = (float)mech::EARTH_RADIUS;
    float dy = (float)((g.lat - state_.lat) * M_PI / 180.0 * R_earth);
    float dx =
        (float)((g.lon - state_.lon) * M_PI / 180.0 * R_earth * cos(lat_rad));

    float sigma = (g.hacc > 0) ? g.hacc : 5.0f;
    float R_pos = sigma * sigma;

    // Measurement matrix H: position rows (0,1) in error state
    // Simple position-only update
    kalmanUpdatePos(dy, dx, R_pos);

    state_.mode = GnssMode::GNSS_GOOD;
  }

  void updatePseudo(const PseudoMeas &m) override {
    // Treat pseudo-GNSS like a position increment measurement
    state_.lat += (m.dN / mech::EARTH_RADIUS) * (180.0 / M_PI);
    double lat_rad = state_.lat * M_PI / 180.0;
    state_.lon += (m.dE / (mech::EARTH_RADIUS * cos(lat_rad))) * (180.0 / M_PI);
    // Inflate position uncertainty
    state_.pos_std_N =
        sqrtf(state_.pos_std_N * state_.pos_std_N + m.std_N * m.std_N);
    state_.pos_std_E =
        sqrtf(state_.pos_std_E * state_.pos_std_E + m.std_E * m.std_E);
    state_.mode = GnssMode::DR_ACTIVE;
  }

  void applyNHC() override {
    float v_lat, v_up;
    mech::nhc_residuals(state_, v_lat, v_up);
    // Update velocity correction in nav frame using body-frame residuals
    float nhc_std = 0.1f; // 0.1 m/s lateral velocity noise
    float K = P_[3 * N + 3] / (P_[3 * N + 3] + nhc_std * nhc_std);
    state_.vE -= K * v_lat * state_.q.x; // approximate coupling
    // (full implementation uses proper H matrix; simplified here)
  }

  NavState getState() const override { return state_; }
  std::string name() const override { return "EKF"; }

private:
  NavState state_;
  int64_t prev_ts_ = 0;
  Mat P_;

  // Helpers ─────────────────────────────────────────────────────────────────

  void setDiag(Mat &M, std::initializer_list<float> vals) {
    int i = 0;
    for (float v : vals) {
      M[i * N + i] = v;
      ++i;
    }
  }

  // Simplified covariance propagation (F = I + dF*dt, process noise Q)
  void buildAndApplyF(const Vec3 &acc_b, const Vec3 &gyr_b, float dt) {
    // Process noise spectral densities (tuning knobs)
    static const float q_acc = 1e-3f; // accel noise density
    static const float q_gyr = 1e-4f; // gyro  noise density
    static const float q_ba = 1e-6f;  // accel bias random walk
    static const float q_bg = 1e-7f;  // gyro  bias random walk

    // Simplified: just add process noise to diagonal blocks
    for (int i = 0; i < 3; i++)
      P_[(0 + i) * N + (0 + i)] += q_acc * dt * dt * dt / 3.0f;
    for (int i = 0; i < 3; i++)
      P_[(3 + i) * N + (3 + i)] += q_acc * dt;
    for (int i = 0; i < 3; i++)
      P_[(6 + i) * N + (6 + i)] += q_gyr * dt;
    for (int i = 0; i < 3; i++)
      P_[(9 + i) * N + (9 + i)] += q_ba * dt;
    for (int i = 0; i < 3; i++)
      P_[(12 + i) * N + (12 + i)] += q_bg * dt;
  }

  // Scalar position update (north or east) using Kalman gain
  void kalmanUpdatePos(float dy_N, float dx_E, float R_pos) {
    // Row 0 of H: [1, 0, 0, ...] for north; row 1: [0, 1, 0, ...] for east
    // S = H*P*H' + R = P[0,0] + R   (for north)
    auto update1D = [&](int row, float residual) {
      float S = P_[row * N + row] + R_pos;
      float K_gain = P_[row * N + row] / S; // Kalman gain scalar
      // Apply correction to position
      if (row == 0) {
        state_.lat += residual * K_gain / (float)mech::EARTH_RADIUS *
                      (180.0f / (float)M_PI);
        state_.pos_std_N = sqrtf(P_[row * N + row] * (1 - K_gain));
      } else {
        double lat_rad = state_.lat * M_PI / 180.0;
        state_.lon += residual * K_gain /
                      (float)(mech::EARTH_RADIUS * cos(lat_rad)) *
                      (180.0f / (float)M_PI);
        state_.pos_std_E = sqrtf(P_[row * N + row] * (1 - K_gain));
      }
      // Joseph form: P = (I-KH)*P*(I-KH)' + KRK'
      P_[row * N + row] *=
          (1.0f - K_gain) * (1.0f - K_gain) + K_gain * K_gain * R_pos;
    };
    update1D(0, dy_N);
    update1D(1, dx_E);
  }
};
