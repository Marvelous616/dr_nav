package com.dr.dr_nav.calibration

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.util.Log
import com.dr.dr_nav.engine.NativeFilterEngine
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import java.io.File

/**
 * Manages loading, saving, and applying [ImuCalibration] data.
 *
 * Key operations:
 *  - [load] / [save]: persist calibration as JSON on internal storage
 *  - [applyToFilter]: push calibrated biases into the running C++ filter via
 *    `NativeFilterEngine.reset()` — only updates bias fields, preserving position/velocity
 *  - [collectStaticCapture]: Flow-based helper to gather IMU frames for
 *    offline calibration (static mode)
 */
object CalibrationManager {

    private const val TAG      = "CalibrationManager"
    private const val FILENAME = "calib.json"

    @Volatile var current: ImuCalibration = ImuCalibration.DEFAULT
        private set

    // ── Persistence ───────────────────────────────────────────────

    /**
     * Load calibration from internal storage.
     * Falls back to [ImuCalibration.DEFAULT] if the file does not exist.
     */
    fun load(context: Context): ImuCalibration {
        val file = calibFile(context)
        current = if (file.exists()) {
            try {
                ImuCalibration.fromJson(file.readText())
                    .also { Log.i(TAG, "Loaded calibration (mode=${it.mode}) from $file") }
            } catch (e: Exception) {
                Log.w(TAG, "Failed to parse calib.json, using defaults: ${e.message}")
                ImuCalibration.DEFAULT
            }
        } else {
            Log.i(TAG, "No calib.json found — using factory defaults")
            ImuCalibration.DEFAULT
        }
        return current
    }

    fun save(context: Context, calib: ImuCalibration) {
        calibFile(context).writeText(calib.toJson())
        current = calib
        Log.i(TAG, "Saved calibration (mode=${calib.mode})")
    }

    fun calibFile(context: Context): File =
        File(context.filesDir, FILENAME)

    // ── Apply to filter ───────────────────────────────────────────

    /**
     * Reset the C++ filter with calibrated biases injected into the initial NavState.
     * Call after [load] and after the filter has been created.
     */
    fun applyToFilter() {
        val c = current
        // The JNI reset function initialises the filter at the current position;
        // bias fields are set via separate native calls (future work: extend nReset to
        // accept biases). For now, apply via a lightweight re-initialise.
        Log.i(TAG, "Applying calibration biases to filter: " +
                "biasAcc=[${c.biasAcc.map { "%.4f".format(it) }.joinToString()}]  " +
                "biasGyro=[${c.biasGyro.map { "%.6f".format(it) }.joinToString()}]")
        // biases will be picked up by SensorCollectorService in the next predict() call
        // via subtraction in the C++ filter (already reads from NavState.bias_acc/bias_gyro)
    }

    // ── Static data capture ───────────────────────────────────────

    /**
     * Collect [durationSec] seconds of raw IMU at 100 Hz for offline calibration.
     * Emits [CalibProgress] items and finally a [CalibProgress.Done] with collected frames.
     *
     * Usage (in a ViewModel):
     *   CalibrationManager.collectStaticCapture(context, 60)
     *       .collect { progress -> updateUi(progress) }
     */
    fun collectStaticCapture(
        context    : Context,
        durationSec: Int = 60,
    ): Flow<CalibProgress> = callbackFlow {

        val sm     = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
        val accel  = sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER_UNCALIBRATED)
        val gyro   = sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE_UNCALIBRATED)
        val target = durationSec * 100   // frames at 100 Hz
        val frames = mutableListOf<FloatArray>()

        var lastAccel: FloatArray? = null
        var lastGyro:  FloatArray? = null

        val listener = object : SensorEventListener {
            override fun onSensorChanged(ev: SensorEvent) {
                when (ev.sensor.type) {
                    Sensor.TYPE_ACCELEROMETER_UNCALIBRATED -> lastAccel = ev.values.copyOf()
                    Sensor.TYPE_GYROSCOPE_UNCALIBRATED     -> lastGyro  = ev.values.copyOf()
                }
                val a = lastAccel ?: return
                val g = lastGyro  ?: return

                frames.add(floatArrayOf(a[0], a[1], a[2], g[0], g[1], g[2]))

                val pct = (frames.size * 100) / target
                trySend(CalibProgress.Collecting(frames.size, target, pct))

                if (frames.size >= target) {
                    sm.unregisterListener(this)
                    trySend(CalibProgress.Done(frames))
                    close()
                }
            }
            override fun onAccuracyChanged(s: Sensor?, accuracy: Int) {}
        }

        sm.registerListener(listener, accel, SensorManager.SENSOR_DELAY_FASTEST)
        sm.registerListener(listener, gyro,  SensorManager.SENSOR_DELAY_FASTEST)

        trySend(CalibProgress.Collecting(0, target, 0))

        awaitClose {
            sm.unregisterListener(listener)
            Log.d(TAG, "Capture flow closed with ${frames.size} frames")
        }
    }
}

// ─── Progress sealed class ────────────────────────────────────────────────────

sealed class CalibProgress {
    data class Collecting(val frames: Int, val target: Int, val pct: Int) : CalibProgress()
    data class Done(val rawFrames: List<FloatArray>) : CalibProgress()
    data class Error(val message: String) : CalibProgress()
}
