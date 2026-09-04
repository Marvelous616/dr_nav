package com.dr.dr_nav.model

import com.dr.dr_nav.engine.NavState
import com.dr.dr_nav.sensor.RawImuFrame
import kotlin.math.atan2

/**
 * Non-Holonomic Constraint (NHC) outage model.
 *
 * No ML required. The constraint is that a ground vehicle cannot slide
 * sideways (lateral velocity ≈ 0). This lightweight model:
 *  - Returns a zero position increment (NHC is enforced inside the C++ filter
 *    via applyNhc() — this Kotlin class simply provides a minimal Kotlin-side
 *    counterpart for the model registry / benchmark harness).
 *  - Sets a generous uncertainty (stdN=stdE=2 m) so the filter treats it as a
 *    soft constraint rather than an exact measurement.
 */
class NHCModel : OutageModel {
    override val name = "NHC"

    override fun predict(
        window      : ArrayDeque<RawImuFrame>,
        state       : NavState,
        elapsedSecs : Float,
    ): PseudoMeasurement = PseudoMeasurement(
        dN   = 0f,
        dE   = 0f,
        stdN = 2f,
        stdE = 2f,
    )
}
