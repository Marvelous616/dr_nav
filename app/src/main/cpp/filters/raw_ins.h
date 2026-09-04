#pragma once
#include "../core/imu_mechanization.h"
#include "fusion_filter.h"

// ─── RawINS: pure mechanization, no Kalman filter ────────────────────────────
// Used as the drift-baseline reference.  No update step; applyNHC is a no-op.

class RawINS : public FusionFilter {
public:
  void reset(const NavState &init) override {
    state_ = init;
    prev_ts_ = init.timestamp_ns;
  }

  void predict(const ImuFrame &imu) override {
    if (prev_ts_ == 0) {
      prev_ts_ = imu.timestamp_ns;
      return;
    }
    float dt = (imu.timestamp_ns - prev_ts_) * 1e-9f;
    prev_ts_ = imu.timestamp_ns;
    if (dt <= 0 || dt > 0.1f)
      return; // bad sample guard

    // Subtract stored bias
    Vec3 acc_b = {imu.ax - state_.bias_acc.x, imu.ay - state_.bias_acc.y,
                  imu.az - state_.bias_acc.z};
    Vec3 gyr_b = {imu.gx - state_.bias_gyro.x, imu.gy - state_.bias_gyro.y,
                  imu.gz - state_.bias_gyro.z};

    mech::propagate(state_, acc_b, gyr_b, dt);
    state_.timestamp_ns = imu.timestamp_ns;
  }

  void updateGNSS(const GnssFrame &) override {}    // intentionally empty
  void updatePseudo(const PseudoMeas &) override {} // intentionally empty
  void applyNHC() override {}                       // intentionally empty

  NavState getState() const override { return state_; }
  std::string name() const override { return "RawINS"; }

private:
  NavState state_;
  int64_t prev_ts_ = 0;
};
