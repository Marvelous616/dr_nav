package com.dr.dr_nav.engine

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Kotlin JNI wrapper around the C++ filter engine.
 *
 * Usage:
 *   NativeFilterEngine.create("IEKF")   // "RawINS" | "EKF" | "MEKF" | "IEKF"
 *   NativeFilterEngine.reset(lat, lon, alt, ...)
 *   NativeFilterEngine.predict(ts, ax, ay, az, gx, gy, gz)
 *   NativeFilterEngine.updateGnss(...)
 *   val state = NativeFilterEngine.getNavState()
 */
object NativeFilterEngine {

    init { System.loadLibrary("dr_nav") }

    // ─── External (JNI) declarations ─────────────────────────────────────────

    @JvmStatic private external fun nCreate(filterName: String): Boolean
    @JvmStatic private external fun nReset(lat: Double, lon: Double, alt: Double,
                                            vN: Float, vE: Float, vD: Float,
                                            timestampNs: Long)
    @JvmStatic private external fun nPredict(timestampNs: Long,
                                              ax: Float, ay: Float, az: Float,
                                              gx: Float, gy: Float, gz: Float)
    @JvmStatic private external fun nUpdateGnss(timestampNs: Long,
                                                 lat: Double, lon: Double, alt: Double,
                                                 vN: Float, vE: Float, vD: Float,
                                                 hacc: Float, vacc: Float,
                                                 satellites: Int, hdop: Float,
                                                 valid: Boolean)
    @JvmStatic private external fun nUpdatePseudo(timestampNs: Long,
                                                   dN: Float, dE: Float,
                                                   stdN: Float, stdE: Float)
    @JvmStatic private external fun nApplyNhc()
    @JvmStatic private external fun nGetState(): FloatArray    // see native-lib.cpp for layout
    @JvmStatic private external fun nGetLatLon(): DoubleArray  // [lat, lon] double precision

    // ─── Public API ───────────────────────────────────────────────────────────

    @Volatile var activeFilterName: String = "IEKF"
        private set

    /** Switch to a different filter. Thread-safe: call from any coroutine context. */
    fun create(filterName: String): Boolean {
        val ok = nCreate(filterName)
        if (ok) activeFilterName = filterName
        return ok
    }

    fun reset(lat: Double, lon: Double, alt: Double,
              vN: Float = 0f, vE: Float = 0f, vD: Float = 0f,
              timestampNs: Long = System.nanoTime()) =
        nReset(lat, lon, alt, vN, vE, vD, timestampNs)

    fun predict(timestampNs: Long,
                ax: Float, ay: Float, az: Float,
                gx: Float, gy: Float, gz: Float) =
        nPredict(timestampNs, ax, ay, az, gx, gy, gz)

    fun updateGnss(timestampNs: Long,
                   lat: Double, lon: Double, alt: Double,
                   vN: Float = Float.NaN, vE: Float = Float.NaN, vD: Float = Float.NaN,
                   hacc: Float = -1f, vacc: Float = -1f,
                   satellites: Int = 0, hdop: Float = 99f,
                   valid: Boolean = true) =
        nUpdateGnss(timestampNs, lat, lon, alt, vN, vE, vD, hacc, vacc, satellites, hdop, valid)

    fun updatePseudo(timestampNs: Long, dN: Float, dE: Float, stdN: Float, stdE: Float) =
        nUpdatePseudo(timestampNs, dN, dE, stdN, stdE)

    fun applyNhc() = nApplyNhc()

    /** Get the latest navigation state. Call from the sensor/update thread. */
    fun getNavState(): NavState {
        val raw    = nGetState()
        val latlon = nGetLatLon()
        return NavState(
            timestampNs  = System.nanoTime(),
            lat          = latlon[0],
            lon          = latlon[1],
            alt          = raw[2].toDouble(),
            vN           = raw[3],  vE = raw[4],   vD = raw[5],
            roll         = raw[6],  pitch = raw[7], yaw = raw[8],
            posStdN      = raw[9],  posStdE = raw[10],
            mode         = GnssMode.fromInt(raw[11].toInt()),
            outageSecs   = raw[12],
            biasAcc      = Triple(raw[13], raw[14], raw[15]),
            biasGyro     = Triple(raw[16], raw[17], 0f)
        )
    }
}

// ─── Data classes ─────────────────────────────────────────────────────────────

enum class GnssMode(val code: Int) {
    GNSS_GOOD(0), GNSS_DEGRADED(1), DR_ACTIVE(2), MAP_MATCHED(3);
    companion object { fun fromInt(v: Int) = entries.firstOrNull { it.code == v } ?: GNSS_GOOD }
}

data class NavState(
    val timestampNs : Long,
    val lat         : Double,
    val lon         : Double,
    val alt         : Double,
    val vN          : Float,
    val vE          : Float,
    val vD          : Float,
    val roll        : Float,
    val pitch       : Float,
    val yaw         : Float,
    val posStdN     : Float,
    val posStdE     : Float,
    val mode        : GnssMode,
    val outageSecs  : Float,
    val biasAcc     : Triple<Float, Float, Float>,
    val biasGyro    : Triple<Float, Float, Float>,
)
