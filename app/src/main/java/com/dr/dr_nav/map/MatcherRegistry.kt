package com.dr.dr_nav.map

import android.util.Log

/**
 * Runtime registry of all [MapMatcher] implementations.
 *
 * Pre-registers [PassthroughMatcher] and [HMMMatcher].
 * The [HMMMatcher] loads its road graph from the path set via [setGraphPath].
 */
object MatcherRegistry {

    private const val TAG = "MatcherRegistry"

    private val passthrough = PassthroughMatcher()
    private val hmm         = HMMMatcher()

    private val registry = linkedMapOf<String, MapMatcher>(
        passthrough.name to passthrough,
        hmm.name         to hmm,
    )

    @Volatile private var _active: MapMatcher = passthrough

    // ── Graph loading ─────────────────────────────────────────────

    /**
     * Load the SQLite road graph for [HMMMatcher].
     * Call once after a Context is available, before selecting "HMM".
     */
    fun setGraphPath(dbPath: String) {
        hmm.loadGraph(dbPath)
        Log.i(TAG, "Road graph loaded for HMMMatcher: $dbPath")
    }

    // ── Active model management ───────────────────────────────────

    val active: MapMatcher get() = _active

    val names: List<String> get() = synchronized(registry) { registry.keys.toList() }

    fun select(name: String) {
        val m = synchronized(registry) { registry[name] }
        if (m == null) {
            Log.w(TAG, "Unknown matcher '$name' — keeping ${_active.name}")
            return
        }
        _active = m
        Log.i(TAG, "Active matcher → ${m.name}")
    }

    fun register(matcher: MapMatcher) {
        synchronized(registry) { registry[matcher.name] = matcher }
        Log.d(TAG, "Registered matcher: ${matcher.name}")
    }
}
