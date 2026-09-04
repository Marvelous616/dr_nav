package com.dr.dr_nav.calibration

import org.json.JSONArray
import org.json.JSONObject

/**
 * Immutable snapshot of all IMU calibration parameters.
 *
 * Produced by `pipeline/imu_calibrate.py` and persisted as `calib.json`.
 * Applied to the C++ filter via [CalibrationManager.applyToFilter].
 */
data class ImuCalibration(
    /** Accelerometer bias (m/s²) — 3 components [X, Y, Z]. */
    val biasAcc    : FloatArray = FloatArray(3),
    /** Accelerometer scale factors — 3 components, ideally near 1.0. */
    val scaleAcc   : FloatArray = floatArrayOf(1f, 1f, 1f),
    /** Gyroscope bias (rad/s) — 3 components. */
    val biasGyro   : FloatArray = FloatArray(3),
    /** Gyroscope scale factors — 3 components. */
    val scaleGyro  : FloatArray = floatArrayOf(1f, 1f, 1f),
    /** Angle Random Walk (rad/s/√Hz) — from Allan Deviation, used for Q matrix. */
    val noiseGyroArw: FloatArray = floatArrayOf(1e-3f, 1e-3f, 1e-3f),
    /** Bias Instability (rad/s) — from Allan Deviation. */
    val noiseGyroBi : FloatArray = floatArrayOf(5e-5f, 5e-5f, 5e-5f),
    /** Velocity Random Walk (m/s²/√Hz) — accelerometer noise from Allan. */
    val noiseAccVrw : FloatArray = floatArrayOf(1e-2f, 1e-2f, 1e-2f),
    /** Human-readable label of calibration mode used. */
    val mode: String = "default",
) {
    // ── Serialisation ──────────────────────────────────────────────

    fun toJson(): String = JSONObject().apply {
        put("mode",           mode)
        put("bias_acc",       biasAcc.toJsonArray())
        put("scale_acc",      scaleAcc.toJsonArray())
        put("bias_gyro",      biasGyro.toJsonArray())
        put("scale_gyro",     scaleGyro.toJsonArray())
        put("noise_gyro_arw", noiseGyroArw.toJsonArray())
        put("noise_gyro_bi",  noiseGyroBi.toJsonArray())
        put("noise_acc_vrw",  noiseAccVrw.toJsonArray())
    }.toString(2)

    companion object {
        /** Default (uncalibrated) calibration — zeroed biases, unity scales. */
        val DEFAULT = ImuCalibration()

        fun fromJson(json: String): ImuCalibration {
            val obj = JSONObject(json)
            fun floatsOf(key: String, default: FloatArray): FloatArray {
                if (!obj.has(key)) return default
                val arr = obj.getJSONArray(key)
                return FloatArray(arr.length()) { arr.getDouble(it).toFloat() }
            }
            return ImuCalibration(
                biasAcc      = floatsOf("bias_acc",       FloatArray(3)),
                scaleAcc     = floatsOf("scale_acc",      floatArrayOf(1f, 1f, 1f)),
                biasGyro     = floatsOf("bias_gyro",      FloatArray(3)),
                scaleGyro    = floatsOf("scale_gyro",     floatArrayOf(1f, 1f, 1f)),
                noiseGyroArw = floatsOf("noise_gyro_arw", floatArrayOf(1e-3f, 1e-3f, 1e-3f)),
                noiseGyroBi  = floatsOf("noise_gyro_bi",  floatArrayOf(5e-5f, 5e-5f, 5e-5f)),
                noiseAccVrw  = floatsOf("noise_acc_vrw",  floatArrayOf(1e-2f, 1e-2f, 1e-2f)),
                mode         = obj.optString("mode", "unknown"),
            )
        }

        private fun FloatArray.toJsonArray() =
            JSONArray().also { a -> forEach { a.put(it.toDouble()) } }
    }

    // FloatArray doesn't have equals/hashCode by default — override for data class
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is ImuCalibration) return false
        return biasAcc.contentEquals(other.biasAcc)
            && scaleAcc.contentEquals(other.scaleAcc)
            && biasGyro.contentEquals(other.biasGyro)
            && scaleGyro.contentEquals(other.scaleGyro)
            && mode == other.mode
    }

    override fun hashCode(): Int {
        var result = biasAcc.contentHashCode()
        result = 31 * result + scaleAcc.contentHashCode()
        result = 31 * result + biasGyro.contentHashCode()
        result = 31 * result + mode.hashCode()
        return result
    }
}
