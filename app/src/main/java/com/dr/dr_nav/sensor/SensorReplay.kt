package com.dr.dr_nav.sensor

import com.dr.dr_nav.engine.GnssMode
import com.dr.dr_nav.engine.NativeFilterEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.delay
import java.io.BufferedReader
import java.io.File
import java.io.FileReader

/**
 * Replays a previously recorded IMU + GNSS log (CSV) through any registered filter.
 * Useful for offline benchmarking and model evaluation.
 *
 * Usage:
 *   val replay = SensorReplay(imuFile, gnssFile)
 *   NativeFilterEngine.create("IEKF")
 *   replay.replay(speedMultiplier = 1f).collect { state -> ... }
 */
class SensorReplay(
    private val imuFile  : File,
    private val gnssFile : File,
) {
    data class ReplayFrame(
        val timestampNs : Long,
        val isImu       : Boolean,
        // IMU fields
        val ax: Float = 0f, val ay: Float = 0f, val az: Float = 0f,
        val gx: Float = 0f, val gy: Float = 0f, val gz: Float = 0f,
        // GNSS fields
        val lat: Double = 0.0, val lon: Double = 0.0, val alt: Double = 0.0,
        val hacc: Float = -1f,
    )

    /**
     * Returns a Flow of NavState at each IMU step.
     * [speedMultiplier] = 1.0 → real-time, 10.0 → 10× faster.
     */
    fun replay(speedMultiplier: Float = 1f,
               filterName: String = NativeFilterEngine.activeFilterName)
    : Flow<com.dr.dr_nav.engine.NavState> = flow {

        NativeFilterEngine.create(filterName)

        val frames = loadAndMerge()
        if (frames.isEmpty()) return@flow

        var initialized = false
        var prevTs = frames.first().timestampNs

        frames.forEach { f ->
            // Real-time pacing
            val dtNs  = f.timestampNs - prevTs
            val delayMs = (dtNs / 1_000_000L / speedMultiplier).toLong()
            if (delayMs in 1..500) delay(delayMs)
            prevTs = f.timestampNs

            if (f.isImu) {
                if (!initialized) return@forEach   // wait for first GNSS
                NativeFilterEngine.predict(f.timestampNs, f.ax, f.ay, f.az, f.gx, f.gy, f.gz)
                NativeFilterEngine.applyNhc()
                emit(NativeFilterEngine.getNavState())
            } else {
                if (!initialized) {
                    NativeFilterEngine.reset(f.lat, f.lon, f.alt, timestampNs = f.timestampNs)
                    initialized = true
                } else {
                    NativeFilterEngine.updateGnss(f.timestampNs, f.lat, f.lon, f.alt, hacc = f.hacc)
                }
            }
        }
    }.flowOn(Dispatchers.IO)

    // ─── CSV loading ───────────────────────────────────────────────────────────

    private fun loadAndMerge(): List<ReplayFrame> {
        val imuFrames  = parseCsv(imuFile,  isImu = true)
        val gnssFrames = parseCsv(gnssFile, isImu = false)
        return (imuFrames + gnssFrames).sortedBy { it.timestampNs }
    }

    private fun parseCsv(file: File, isImu: Boolean): List<ReplayFrame> {
        val frames = mutableListOf<ReplayFrame>()
        BufferedReader(FileReader(file)).use { reader ->
            reader.readLine() // skip header
            reader.forEachLine { line ->
                val cols = line.split(",")
                try {
                    if (isImu && cols.size >= 7) {
                        frames += ReplayFrame(
                            timestampNs = cols[0].trim().toLong(),
                            isImu = true,
                            ax = cols[1].trim().toFloat(), ay = cols[2].trim().toFloat(),
                            az = cols[3].trim().toFloat(), gx = cols[4].trim().toFloat(),
                            gy = cols[5].trim().toFloat(), gz = cols[6].trim().toFloat(),
                        )
                    } else if (!isImu && cols.size >= 5) {
                        frames += ReplayFrame(
                            timestampNs = cols[0].trim().toLong(),
                            isImu = false,
                            lat  = cols[1].trim().toDouble(),
                            lon  = cols[2].trim().toDouble(),
                            alt  = cols[3].trim().toDouble(),
                            hacc = cols[4].trim().toFloat(),
                        )
                    }
                } catch (_: NumberFormatException) { /* skip bad lines */ }
            }
        }
        return frames
    }
}
