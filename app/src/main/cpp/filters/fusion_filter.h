#pragma once
#include "core/types.h"
#include <string>

// ─── Abstract plug-in interface that every filter must implement ─────────────
// Add a new filter: subclass FusionFilter, implement these methods,
// register it in FilterFactory::create().

class FusionFilter {
public:
  virtual ~FusionFilter() = default;

  // Reset filter to a known starting state
  virtual void reset(const NavState &init) = 0;

  // Propagate state forward with one IMU sample (called @ 100–200 Hz)
  virtual void predict(const ImuFrame &imu) = 0;

  // Update with a real GNSS fix
  virtual void updateGNSS(const GnssFrame &gnss) = 0;

  // Update with a pseudo-GNSS measurement from the ML outage model
  virtual void updatePseudo(const PseudoMeas &meas) = 0;

  // Apply Non-Holonomic Constraint (lateral & vertical velocity = 0)
  virtual void applyNHC() = 0;

  // Retrieve the current navigation state
  virtual NavState getState() const = 0;

  // Human-readable identifier
  virtual std::string name() const = 0;
};
