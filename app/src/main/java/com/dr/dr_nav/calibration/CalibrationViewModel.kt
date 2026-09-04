package com.dr.dr_nav.calibration

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * ViewModel for the calibration UI screen.
 *
 * Exposes:
 *  - [calibState] — current [CalibState] (idle / capturing / done / error)
 *  - [startCapture] / [cancelCapture] — control data collection
 *  - [saveCalibration] — persist computed calibration to internal storage
 */
class CalibrationViewModel(app: Application) : AndroidViewModel(app) {

    private val _state = MutableStateFlow<CalibState>(CalibState.Idle(
        CalibrationManager.load(app)
    ))
    val calibState: StateFlow<CalibState> = _state

    private var captureJob: kotlinx.coroutines.Job? = null

    // ── Actions ───────────────────────────────────────────────────

    fun startCapture(durationSec: Int = 60) {
        captureJob?.cancel()
        captureJob = viewModelScope.launch(Dispatchers.IO) {
            CalibrationManager.collectStaticCapture(getApplication(), durationSec)
                .collect { progress ->
                    when (progress) {
                        is CalibProgress.Collecting ->
                            _state.value = CalibState.Capturing(
                                framesCollected = progress.frames,
                                target          = progress.target,
                                pct             = progress.pct,
                            )

                        is CalibProgress.Done -> {
                            // Compute static calibration from collected frames
                            val result = withContext(Dispatchers.Default) {
                                computeStaticCalib(progress.rawFrames)
                            }
                            _state.value = CalibState.Done(result)
                        }

                        is CalibProgress.Error ->
                            _state.value = CalibState.Error(progress.message)
                    }
                }
        }
    }

    fun cancelCapture() {
        captureJob?.cancel()
        _state.value = CalibState.Idle(CalibrationManager.current)
    }

    fun saveCalibration(calib: ImuCalibration) {
        CalibrationManager.save(getApplication(), calib)
        CalibrationManager.applyToFilter()
        _state.value = CalibState.Idle(calib)
    }

    // ── Static calibration computation ────────────────────────────

    /**
     * Compute accelerometer and gyroscope bias from a static capture.
     * (Simplified: just averages. For full 6-position tumble, use the Python script.)
     */
    private fun computeStaticCalib(frames: List<FloatArray>): ImuCalibration {
        if (frames.isEmpty()) return ImuCalibration.DEFAULT
        val n = frames.size.toFloat()

        var sumAx = 0f; var sumAy = 0f; var sumAz = 0f
        var sumGx = 0f; var sumGy = 0f; var sumGz = 0f
        for (f in frames) {
            sumAx += f[0]; sumAy += f[1]; sumAz += f[2]
            sumGx += f[3]; sumGy += f[4]; sumGz += f[5]
        }

        // Gyro: average is the bias when stationary
        val biasGyro = floatArrayOf(sumGx / n, sumGy / n, sumGz / n)

        // Accel: bias is the horizontal components; Z should equal g0
        val meanAz   = sumAz / n
        val biasAccZ = meanAz - 9.80665f    // deviation from expected gravity
        val biasAcc  = floatArrayOf(sumAx / n, sumAy / n, biasAccZ)

        return ImuCalibration(
            biasAcc  = biasAcc,
            biasGyro = biasGyro,
            mode     = "static_runtime",
        )
    }
}

// ─── UI state sealed class ────────────────────────────────────────────────────

sealed class CalibState {
    data class Idle(val calibration: ImuCalibration) : CalibState()
    data class Capturing(val framesCollected: Int, val target: Int, val pct: Int) : CalibState()
    data class Done(val calibration: ImuCalibration) : CalibState()
    data class Error(val message: String) : CalibState()
}
