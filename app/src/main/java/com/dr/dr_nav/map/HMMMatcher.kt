package com.dr.dr_nav.map

import android.util.Log
import kotlin.math.*

/**
 * Viterbi HMM Map Matcher — Newson & Krumm (2009), Microsoft Research.
 *
 * At each position fix, the top-K nearest road segments are candidate states.
 * The Viterbi algorithm decodes the most probable road sequence using:
 *
 *   Emission probability:
 *     P(z|s) ∝ exp(−d² / 2σ²)
 *     d = perpendicular distance (metres) from fix to road segment
 *     σ = position uncertainty from filter (posStdM)
 *
 *   Transition probability (consecutive candidates s₁→s₂):
 *     P(s₁→s₂) ∝ exp(−|‖z₁z₂‖ − routeDist(s₁,s₂)| / β)
 *     β = 10 m, routeDist approximated by great-circle distance between segment midpoints
 *
 * A sliding window of [WINDOW] fixes is maintained. The best candidate for the
 * oldest fix in the window is committed as the final snapped position.
 *
 * If the snapped road bearing is within 20° of the filter heading, a heading
 * correction hint is stored in [lastBearingHint] for the caller to inject.
 */
class HMMMatcher : MapMatcher {

    override val name = "HMM"

    private val TAG = "HMMMatcher"

    // ── Tuning parameters ─────────────────────────────────────────
    private val K          = 5       // top-K candidate segments per fix
    private val WINDOW     = 5       // Viterbi sliding window length
    private val BETA       = 10.0    // transition sharpness (metres)
    private val RADIUS_M   = 50.0    // candidate search radius (metres)
    private val HEADING_THRESHOLD = Math.toRadians(20.0)  // 20° heading gate

    // ── State ─────────────────────────────────────────────────────
    private val graph = RoadGraph()

    // Sliding window: list of (candidates, observation_lat, observation_lon, posStd)
    private data class Obs(
        val candidates: List<RoadSegment>,
        val lat: Double,
        val lon: Double,
        val sigma: Double,
    )
    private val window = ArrayDeque<Obs>(WINDOW + 1)

    /** Last committed snapped position (for UI) */
    var lastResult: MatchResult = PassthroughMatcher().match(0.0, 0.0, 0f, 0f)
        private set

    /** Last road bearing hint to feed back into the filter (NaN = none) */
    var lastBearingHint: Float = Float.NaN
        private set

    // ── MapMatcher implementation ─────────────────────────────────

    override fun loadGraph(dbPath: String) {
        graph.open(dbPath)
        if (!graph.isLoaded) {
            Log.w(TAG, "Road graph failed to load — HMM will behave as Passthrough")
        }
    }

    override fun match(
        lat       : Double,
        lon       : Double,
        headingRad: Float,
        speedMs   : Float,
        posStdM   : Float,
    ): MatchResult {
        if (!graph.isLoaded) {
            return MatchResult(lat, lon, false, headingRad, 0f, MatchMode.PASSTHROUGH)
        }

        // Skip matching when nearly stationary (avoid snapping while parked)
        if (speedMs < 0.5f) {
            return MatchResult(lat, lon, false, headingRad, 0f, MatchMode.PASSTHROUGH)
        }

        val candidates = graph.candidatesNear(lat, lon, RADIUS_M, K)
        if (candidates.isEmpty()) {
            return MatchResult(lat, lon, false, headingRad, 0f, MatchMode.PASSTHROUGH)
        }

        val sigma = posStdM.toDouble().coerceAtLeast(1.0)
        window.addLast(Obs(candidates, lat, lon, sigma))
        if (window.size > WINDOW) window.removeFirst()

        // Run Viterbi on the current window
        val best = viterbi()
        val bestSeg = best ?: candidates.minByOrNull { it.perpendicularDistM(lat, lon) }!!

        val (snapLat, snapLon, dist) = bestSeg.snapTo(lat, lon)
        val emission = emissionProb(dist, sigma)

        // Heading hint if road bearing is close to filter heading
        val angleDiff = angularDiff(bestSeg.bearingRad, headingRad)
        lastBearingHint = if (abs(angleDiff) < HEADING_THRESHOLD)
            bestSeg.bearingRad else Float.NaN

        lastResult = MatchResult(
            lat         = snapLat,
            lon         = snapLon,
            snapped     = dist < RADIUS_M,
            roadBearing = bestSeg.bearingRad,
            confidence  = emission.toFloat().coerceIn(0f, 1f),
            mode        = MatchMode.HMM_MATCHED,
        )
        return lastResult
    }

    // ── Viterbi decoder ───────────────────────────────────────────

    /**
     * Runs Viterbi over the current [window] and returns the best candidate
     * for the most recent observation.
     */
    private fun viterbi(): RoadSegment? {
        if (window.isEmpty()) return null
        val w = window.toList()

        // delta[t][k] = best log-prob to reach candidate k at step t
        val delta = Array(w.size) { DoubleArray(K) { Double.NEGATIVE_INFINITY } }
        val psi   = Array(w.size) { IntArray(K) { -1 } }

        // Initialise at t=0
        val obs0 = w[0]
        obs0.candidates.forEachIndexed { k, seg ->
            val d = seg.perpendicularDistM(obs0.lat, obs0.lon)
            delta[0][k] = ln(emissionProb(d, obs0.sigma).coerceAtLeast(1e-300))
        }

        // Forward pass
        for (t in 1 until w.size) {
            val obs  = w[t]
            val prev = w[t - 1]
            val gcDist = haversine(prev.lat, prev.lon, obs.lat, obs.lon)

            obs.candidates.forEachIndexed { j, segJ ->
                val dJ    = segJ.perpendicularDistM(obs.lat, obs.lon)
                val emitJ = ln(emissionProb(dJ, obs.sigma).coerceAtLeast(1e-300))

                var bestScore = Double.NEGATIVE_INFINITY
                var bestK     = 0
                prev.candidates.forEachIndexed { i, segI ->
                    if (delta[t-1][i] == Double.NEGATIVE_INFINITY) return@forEachIndexed
                    val routeApprox = haversine(
                        (segI.lat1 + segI.lat2) / 2, (segI.lon1 + segI.lon2) / 2,
                        (segJ.lat1 + segJ.lat2) / 2, (segJ.lon1 + segJ.lon2) / 2,
                    )
                    val transProb = transitionProb(gcDist, routeApprox)
                    val score = delta[t-1][i] + ln(transProb.coerceAtLeast(1e-300))
                    if (score > bestScore) { bestScore = score; bestK = i }
                }
                delta[t][j] = bestScore + emitJ
                psi[t][j]   = bestK
            }
        }

        // Back-track to get best candidate at last step
        val lastObs = w.last()
        val lastIdx = w.size - 1
        var bestJ = delta[lastIdx].indices.maxByOrNull { delta[lastIdx][it] } ?: 0
        return lastObs.candidates.getOrNull(bestJ)
    }

    // ── Probability functions ─────────────────────────────────────

    /** Gaussian emission probability: exp(−d²/2σ²) / (σ√2π) — unnormalised. */
    private fun emissionProb(distM: Double, sigma: Double): Double {
        val z = distM / sigma
        return exp(-0.5 * z * z)
    }

    /** Transition probability: exp(−|gcDist − routeDist| / β). */
    private fun transitionProb(gcDist: Double, routeDist: Double): Double =
        exp(-abs(gcDist - routeDist) / BETA)

    // ── Geometry ──────────────────────────────────────────────────

    private fun haversine(la1: Double, lo1: Double, la2: Double, lo2: Double): Double {
        val R = RoadGraph.EARTH_R
        val dlat = Math.toRadians(la2 - la1)
        val dlon = Math.toRadians(lo2 - lo1)
        val a = sin(dlat / 2).pow(2) +
                cos(Math.toRadians(la1)) * cos(Math.toRadians(la2)) * sin(dlon / 2).pow(2)
        return R * 2 * atan2(sqrt(a), sqrt(1 - a))
    }

    /** Signed angular difference in radians, wrapped to [−π, π]. */
    private fun angularDiff(a: Float, b: Float): Float {
        var d = (a - b).toDouble()
        while (d >  Math.PI) d -= 2 * Math.PI
        while (d < -Math.PI) d += 2 * Math.PI
        return d.toFloat()
    }
}
