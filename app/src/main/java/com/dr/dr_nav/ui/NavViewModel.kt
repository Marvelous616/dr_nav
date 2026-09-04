package com.dr.dr_nav.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.dr.dr_nav.engine.NavState
import com.dr.dr_nav.sensor.SensorCollectorService
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch

/**
 * ViewModel bridging [SensorCollectorService.navState] SharedFlow to the UI layer.
 *
 * Also exposes the filter/model/matcher selector state so the control panel
 * can change them reactively without going through the Activity.
 */
class NavViewModel : ViewModel() {

    // ── Live navigation state ─────────────────────────────────────

    /** Latest NavState from the running filter. Null before first fix. */
    val navState: StateFlow<NavState?> = SensorCollectorService.navState
        .stateIn(viewModelScope, SharingStarted.Eagerly, null)

    /** True when the sensor collection service is actively running. */
    val isRunning: StateFlow<Boolean> = SensorCollectorService.isRunning
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)

    // ── Selector state ────────────────────────────────────────────

    private val _selectedFilter  = MutableStateFlow("IEKF")
    private val _selectedModel   = MutableStateFlow("NHC")
    private val _selectedMatcher = MutableStateFlow("Passthrough")

    val selectedFilter : StateFlow<String> = _selectedFilter.asStateFlow()
    val selectedModel  : StateFlow<String> = _selectedModel.asStateFlow()
    val selectedMatcher: StateFlow<String> = _selectedMatcher.asStateFlow()

    fun selectFilter (name: String) { _selectedFilter.value  = name }
    fun selectModel  (name: String) { _selectedModel.value   = name }
    fun selectMatcher(name: String) { _selectedMatcher.value = name }

    // ── Trajectory history ────────────────────────────────────────

    /** Rolling buffer of last 200 (lat, lon) pairs for polyline rendering. */
    val trajectory: StateFlow<List<Pair<Double, Double>>> =
        navState
            .filterNotNull()
            .scan(emptyList<Pair<Double, Double>>()) { acc, state ->
                val next = acc + Pair(state.lat, state.lon)
                if (next.size > 200) next.drop(next.size - 200) else next
            }
            .stateIn(viewModelScope, SharingStarted.Eagerly, emptyList())
}
