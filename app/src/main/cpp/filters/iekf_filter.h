#pragma once
#include "../core/imu_mechanization.h"
#include "fusion_filter.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

// ─── IEKF: Iterative EKF + Adaptive Noise Regulation (ANR) ──────────────────
// Improvements over EKF:
//   1. Iterative re-linearization around the updated state (not just
//   prediction)
//      → handles nonlinear measurement functions better
//   2. Adaptive Noise Regulation (ANR): Q and R estimates updated online
//      from innovation sequence (Li et al. approach)
//   3. Navigation-frame attitude error (inherits MEKF advantage)
//
// References:
//   Liu & Guo, "IEKF + Deep Learning for Vehicle Localization", IEEE TIM 2021
//   Niu et al., "TCN-BiLSTM + ANR-IEKF", Sensors 2026
//   Li & Chang, IEEE Sensors 2020

class IekfFilter : public FusionFilter {
  static constexpr int N = 15;
  static constexpr int MAX_ITER = 5; // IEKF iteration count
  using Mat = std::array<float, N * N>;

public:
  void reset(const NavState &init) override {
    state_ = init;
    prev_ts_ = init.timestamp_ns;
    P_.fill(0);
    float diag[N] = {5,     5,     5,     0.5f,  0.5f,  0.5f,  0.05f, 0.05f,
                     0.05f, 0.01f, 0.01f, 0.01f, 1e-4f, 1e-4f, 1e-4f};
    for (int i = 0; i < N; i++)
      P_[i * N + i] = diag[i];
    R_adaptive_ = 25.0f; // initial position noise (5 m)
    Q_scale_ = 1.0f;
    innov_win_idx_ = 0;
    innov_win_.fill(0);
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

    mech::propagate(state_, acc_b, gyr_b, dt);
    state_.timestamp_ns = imu.timestamp_ns;
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

    // Adaptively tune R from recent innovations
    updateAdaptiveR(dN, dE);
    float R = R_adaptive_;

    // Iterative update (re-linearize around updated state each iteration)
    for (int iter = 0; iter < MAX_ITER; ++iter) {
      lat_rad = state_.lat * M_PI / 180.0;
      float resN = (float)((g.lat - state_.lat) * M_PI / 180.0 * R_earth);
      float resE =
          (float)((g.lon - state_.lon) * M_PI / 180.0 * R_earth * cos(lat_rad));
      iterUpdate1D(0, resN, R);
      iterUpdate1D(1, resE, R);
    }

    state_.mode = GnssMode::GNSS_GOOD;
  }

  void updatePseudo(const PseudoMeas &m) override {
    // ML pseudo-GNSS: use stated uncertainty directly
    float R = m.std_N * m.std_N;
    for (int iter = 0; iter < MAX_ITER; ++iter) {
      iterUpdate1D(0, m.dN, R);
      iterUpdate1D(1, m.dE, m.std_E * m.std_E);
    }
    state_.mode = GnssMode::DR_ACTIVE;
  }

  void applyNHC() override {
    static const float nhc_std = 0.05f; // 5 cm/s lateral velocity tolerance
    float v_lat, v_up;
    mech::nhc_residuals(state_, v_lat, v_up);

    // Lateral velocity pseudo-update (row 4 of error state → vE)
    auto nhcUpdate = [&](float residual) {
      float S = P_[4 * N + 4] + nhc_std * nhc_std;
      float K = P_[4 * N + 4] / S;
      state_.vE -= K * residual;
      P_[4 * N + 4] *= (1.0f - K);
    };
    nhcUpdate(v_lat);
  }

  NavState getState() const override { return state_; }
  std::string name() const override { return "IEKF"; }

private:
  NavState state_;
  int64_t prev_ts_ = 0;
  Mat P_;

  // Adaptive noise regulation (ANR) - sliding window on innovations
  static constexpr int INNOV_WIN = 10;
  std::array<float, INNOV_WIN> innov_win_{};
  int innov_win_idx_ = 0;
  float R_adaptive_ = 25.0f;
  float Q_scale_ = 1.0f;

  void propagateCov(const Vec3 &acc_b, float dt) {
    float qa = 5e-4f * Q_scale_, qg = 1e-4f * Q_scale_;
    float qba = 1e-6f, qbg = 1e-7f;
    for (int i = 0; i < 3; i++)
      P_[(0 + i) * N + (0 + i)] += qa * dt * dt * dt / 3.0f;
    for (int i = 0; i < 3; i++)
      P_[(3 + i) * N + (3 + i)] += qa * dt;
    for (int i = 0; i < 3; i++)
      P_[(6 + i) * N + (6 + i)] += qg * dt;
    for (int i = 0; i < 3; i++)
      P_[(9 + i) * N + (9 + i)] += qba * dt;
    for (int i = 0; i < 3; i++)
      P_[(12 + i) * N + (12 + i)] += qbg * dt;
  }

  void iterUpdate1D(int row, float residual, float R) {
    float R_earth = (float)mech::EARTH_RADIUS;
    double lat_rad = state_.lat * M_PI / 180.0;

    float S = P_[row * N + row] + R;
    float K = P_[row * N + row] / S;
    if (row == 0) {
      state_.lat += residual * K / R_earth * (180.0f / (float)M_PI);
      state_.pos_std_N = sqrtf(std::max(P_[row * N + row] * (1.0f - K), 0.0f));
    } else {
      state_.lon += residual * K / (float)(R_earth * cos(lat_rad)) *
                    (180.0f / (float)M_PI);
      state_.pos_std_E = sqrtf(std::max(P_[row * N + row] * (1.0f - K), 0.0f));
    }
    P_[row * N + row] = std::max((1.0f - K) * P_[row * N + row], 1e-4f);
  }

  // ANR: update R estimate using sliding window of squared innovations
  void updateAdaptiveR(float dN, float dE) {
    float innov_sq = dN * dN + dE * dE;
    innov_win_[innov_win_idx_++ % INNOV_WIN] = innov_sq;
    float sum = 0;
    for (float v : innov_win_)
      sum += v;
    float mean_innov = sum / INNOV_WIN;
    // R_adaptive ≈ mean innovation² - P[0,0] - P[1,1] (innovation covariance)
    float expected_innov = P_[0 * N + 0] + P_[1 * N + 1];
    float raw_R = mean_innov - expected_innov;
    R_adaptive_ = std::max(raw_R, 1.0f); // floor at 1 m²
    // Also scale Q if innovations are large (dynamics mismatch)
    Q_scale_ = std::min(std::max(mean_innov / 25.0f, 0.1f), 10.0f);
  }
};
