package com.dr.dr_nav.map

/**
 * Result returned by any [MapMatcher] implementation.
 *
 * @param lat         Snapped (or original) latitude in WGS-84 degrees
 * @param lon         Snapped (or original) longitude in WGS-84 degrees
 * @param snapped     true  → position was moved to the nearest road segment
 *                    false → returned as-is (no confident road found)
 * @param roadBearing Forward bearing of the matched road segment (radians, NED)
 * @param confidence  Viterbi / emission probability [0, 1]
 * @param mode        Which algorithm produced this result
 */
data class MatchResult(
    val lat         : Double,
    val lon         : Double,
    val snapped     : Boolean,
    val roadBearing : Float,
    val confidence  : Float,
    val mode        : MatchMode,
)

enum class MatchMode { PASSTHROUGH, HMM_MATCHED }

/**
 * Pluggable map-matching interface.
 *
 * Implementations must be thread-safe; [match] is called from [Dispatchers.Default].
 */
interface MapMatcher {
    val name: String

    /**
     * Snap a position to the road network.
     *
     * @param lat      Estimated latitude (degrees)
     * @param lon      Estimated longitude (degrees)
     * @param headingRad  Current heading from the filter (radians, NED yaw)
     * @param speedMs  Current speed (m/s) — used to skip matching when stationary
     * @param posStdM  Position uncertainty 1-sigma (m) — feeds emission σ
     */
    fun match(
        lat       : Double,
        lon       : Double,
        headingRad: Float,
        speedMs   : Float,
        posStdM   : Float = 5f,
    ): MatchResult

    /**
     * Load the SQLite road graph produced by `osm_extract.py`.
     * Call once after Context is available (before the first [match] call).
     * No-op for [PassthroughMatcher].
     */
    fun loadGraph(dbPath: String) {}
}
