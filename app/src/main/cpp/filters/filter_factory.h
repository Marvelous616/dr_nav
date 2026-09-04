#pragma once
#include "ekf_filter.h"
#include "fusion_filter.h"
#include "iekf_filter.h"
#include "mekf_filter.h"
#include "raw_ins.h"
#include <memory>
#include <stdexcept>
#include <string>

// ─── Factory: instantiate any registered filter by name string ───────────────
// Usage (C++):
//   auto f = FilterFactory::create("IEKF");
//   f->reset(init); f->predict(imu); f->updateGNSS(gnss);
//
// Usage (from Kotlin/JNI):
//   nativeCreate("IEKF")

class FilterFactory {
public:
  static std::unique_ptr<FusionFilter> create(const std::string &name) {
    if (name == "RawINS")
      return std::make_unique<RawINS>();
    if (name == "EKF")
      return std::make_unique<EkfFilter>();
    if (name == "MEKF")
      return std::make_unique<MekfFilter>();
    if (name == "IEKF")
      return std::make_unique<IekfFilter>();
    throw std::invalid_argument("FilterFactory: unknown filter name: " + name);
  }
};
