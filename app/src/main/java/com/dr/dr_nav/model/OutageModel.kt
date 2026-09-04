package com.dr.dr_nav.model

import com.dr.dr_nav.engine.NavState
import com.dr.dr_nav.sensor.RawImuFrame

/**
 * Common pseudo-GNSS measurement produced by an outage model.
 * Injected into the C++ filter via NativeFilterEngine.updatePseudo().
 */
data class PseudoMeasurement(
    val dN   : Float,    // position increment North (metres)
    val dE   : Float,    // position increment East  (metres)
    val stdN : Float,    // 1-sigma uncertainty North (metres)
    val stdE : Float,    // 1-sigma uncertainty East  (metres)
) {
    companion object {
        /** No correction — used when the model has nothing to say. */
        val NONE = PseudoMeasurement(0f, 0f, 999f, 999f)
    }
}

/**
 * Base interface for all GNSS-outage position predictors.
 *
 * Implementations:
 *  - NHCModel         — zero lateral velocity constraint, no ML
 *  - LSTMModel        — TFLite BiLSTM
 *  - TCNBiLSTMModel   — TFLite TCN-BiLSTM (SOTA)
 *
 * [predict] is called every 100 ms during a GNSS outage.
 * The implementation must be thread-safe (called from Dispatchers.Default).
 */
interface OutageModel {
    val name: String

    /**
     * Predict a position increment for the next 0.1 s step.
     *
     * @param window  Last [N_WINDOW] raw IMU frames (newest last).
     * @param state   Current navigation state from the filter.
     * @param elapsedSecs  Seconds elapsed since GNSS was lost.
     * @return Pseudo-measurement to inject into the filter.
     */
    fun predict(
        window      : ArrayDeque<RawImuFrame>,
        state       : NavState,
        elapsedSecs : Float,
    ): PseudoMeasurement

    companion object {
        /** Sliding window length (frames @ 100 Hz = 1 second). */
        const val N_WINDOW = 100
        const val N_FEATURES = 12
    }
}
