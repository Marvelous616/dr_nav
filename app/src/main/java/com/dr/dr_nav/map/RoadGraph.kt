package com.dr.dr_nav.map

import android.database.sqlite.SQLiteDatabase
import android.util.Log
import kotlin.math.*

/**
 * Read-only accessor for the SQLite road graph built by `pipeline/osm_extract.py`.
 *
 * Schema (created by osm_extract.py):
 *   nodes(id, lat, lon)
 *   edges(id, from_id, to_id, lat1, lon1, lat2, lon2, bearing_rad, length_m, highway)
 *
 * Thread-safe: [SQLiteDatabase] read operations are safe to call from multiple threads.
 */
class RoadGraph {

    private val TAG = "RoadGraph"
    private var db: SQLiteDatabase? = null

    // ── Lifecycle ─────────────────────────────────────────────────

    /** Open the database. Must be called before [candidatesNear]. */
    fun open(dbPath: String) {
        db?.close()
        db = try {
            SQLiteDatabase.openDatabase(dbPath, null, SQLiteDatabase.OPEN_READONLY)
                .also { Log.i(TAG, "Opened road graph: $dbPath") }
        } catch (e: Exception) {
            Log.w(TAG, "Could not open road graph at $dbPath: ${e.message}")
            null
        }
    }

    fun close() { db?.close(); db = null }

    val isLoaded: Boolean get() = db != null

    // ── Spatial query ─────────────────────────────────────────────

    /**
     * Return road segments whose bounding box overlaps a circle of [radiusM] metres
     * around ([lat], [lon]).  Uses the edge lat/lon bounding-box index.
     *
     * Returns at most [limit] candidates, ordered by distance to the query point.
     */
    fun candidatesNear(
        lat    : Double,
        lon    : Double,
        radiusM: Double = 50.0,
        limit  : Int    = 10,
    ): List<RoadSegment> {
        val db = db ?: return emptyList()

        // Convert radius to degree deltas (approximate, sufficient for < 1 km)
        val dLat = Math.toDegrees(radiusM / EARTH_R)
        val dLon = Math.toDegrees(radiusM / (EARTH_R * cos(Math.toRadians(lat))))

        val minLat = lat - dLat;  val maxLat = lat + dLat
        val minLon = lon - dLon;  val maxLon = lon + dLon

        val cursor = db.rawQuery("""
            SELECT from_id, to_id, lat1, lon1, lat2, lon2, bearing_rad, length_m, highway
            FROM   edges
            WHERE  lat1 BETWEEN ? AND ?
              AND  lon1 BETWEEN ? AND ?
            LIMIT  ?
        """.trimIndent(),
            arrayOf(minLat.toString(), maxLat.toString(),
                    minLon.toString(), maxLon.toString(),
                    (limit * 3).toString())   // over-fetch, sort below
        )

        val results = mutableListOf<RoadSegment>()
        cursor.use { c ->
            while (c.moveToNext()) {
                val seg = RoadSegment(
                    fromId      = c.getLong(0),
                    toId        = c.getLong(1),
                    lat1        = c.getDouble(2),
                    lon1        = c.getDouble(3),
                    lat2        = c.getDouble(4),
                    lon2        = c.getDouble(5),
                    bearingRad  = c.getFloat(6),
                    lengthM     = c.getFloat(7),
                    highway     = c.getString(8) ?: "road",
                )
                results += seg
            }
        }

        // Sort by perpendicular distance to query point, return top-limit
        return results
            .sortedBy { it.perpendicularDistM(lat, lon) }
            .take(limit)
    }

    // ── Constants ─────────────────────────────────────────────────

    companion object {
        const val EARTH_R = 6_378_137.0
    }
}

// ─── Road segment data class ──────────────────────────────────────────────────

data class RoadSegment(
    val fromId    : Long,
    val toId      : Long,
    val lat1      : Double,
    val lon1      : Double,
    val lat2      : Double,
    val lon2      : Double,
    val bearingRad: Float,
    val lengthM   : Float,
    val highway   : String,
) {
    /** Snap [lat],[lon] onto this segment; return snapped position + perp distance. */
    fun snapTo(lat: Double, lon: Double): Triple<Double, Double, Double> {
        val R = RoadGraph.EARTH_R
        // Work in flat-earth local metres
        val latRef = (lat1 + lat2) / 2
        val cosLat = cos(Math.toRadians(latRef))

        val px = (lon  - lon1) * R * cosLat * (Math.PI / 180.0)
        val py = (lat  - lat1) * R           * (Math.PI / 180.0)
        val ex = (lon2 - lon1) * R * cosLat * (Math.PI / 180.0)
        val ey = (lat2 - lat1) * R           * (Math.PI / 180.0)
        val segLen = sqrt(ex * ex + ey * ey)

        val t = if (segLen < 1e-6) 0.0
                else    ((px * ex + py * ey) / (segLen * segLen)).coerceIn(0.0, 1.0)

        val snapLon = lon1 + t * (lon2 - lon1)
        val snapLat = lat1 + t * (lat2 - lat1)
        val diffX   = px - t * ex
        val diffY   = py - t * ey
        val dist    = sqrt(diffX * diffX + diffY * diffY)
        return Triple(snapLat, snapLon, dist)
    }

    fun perpendicularDistM(lat: Double, lon: Double) = snapTo(lat, lon).third

    /** Great-circle distance from (lat1,lon1) to (lat2,lon2) in metres. */
    fun haversine(la1: Double, lo1: Double, la2: Double, lo2: Double): Double {
        val R = RoadGraph.EARTH_R
        val dlat = Math.toRadians(la2 - la1)
        val dlon = Math.toRadians(lo2 - lo1)
        val a = sin(dlat / 2).pow(2) +
                cos(Math.toRadians(la1)) * cos(Math.toRadians(la2)) * sin(dlon / 2).pow(2)
        return R * 2 * atan2(sqrt(a), sqrt(1 - a))
    }
}
