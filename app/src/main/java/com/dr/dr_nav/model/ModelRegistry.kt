package com.dr.dr_nav.model

import android.util.Log

/**
 * Runtime registry of all available [OutageModel] implementations.
 *
 * Default models are registered at first access. The active model can be
 * swapped at any time from any thread — the switch is atomic.
 */
object ModelRegistry {

    private const val TAG = "ModelRegistry"

    private val registry = linkedMapOf<String, OutageModel>()

    @Volatile private var _active: OutageModel = NHCModel()

    init {
        register(NHCModel())
        // LSTM and TCNBiLSTM are registered lazily on first access
        // to avoid loading TFLite assets until we know they exist.
    }

    // ── Registration ──────────────────────────────────────────────

    fun register(model: OutageModel) {
        synchronized(registry) { registry[model.name] = model }
        Log.d(TAG, "Registered model: ${model.name}")
    }

    /** Register TFLite-backed models. Call after Context is available. */
    fun registerTFLite(lstmModel: OutageModel, tcnModel: OutageModel) {
        register(lstmModel)
        register(tcnModel)
    }

    // ── Active model management ───────────────────────────────────

    /** Thread-safe read of the currently active model. */
    val active: OutageModel get() = _active

    /** Names of all registered models (for UI selector). */
    val names: List<String> get() = synchronized(registry) { registry.keys.toList() }

    /**
     * Switch to the model identified by [name].
     * No-op (with a warning) if [name] is not registered.
     */
    fun select(name: String) {
        val m = synchronized(registry) { registry[name] }
        if (m == null) {
            Log.w(TAG, "Unknown model '$name' — keeping ${_active.name}")
            return
        }
        _active = m
        Log.i(TAG, "Active model → ${m.name}")
    }
}
