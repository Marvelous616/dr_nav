package com.dr.dr_nav.ui

import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Spinner
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.dr.dr_nav.R
import com.dr.dr_nav.engine.FilterFactory
import com.dr.dr_nav.map.MatcherRegistry
import com.dr.dr_nav.model.ModelRegistry
import com.dr.dr_nav.sensor.SensorCollectorService
import com.google.android.material.floatingactionbutton.ExtendedFloatingActionButton
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch

class ControlPanelFragment : Fragment() {

    private val viewModel: NavViewModel by activityViewModels()

    private lateinit var tvLatLon: TextView
    private lateinit var tvSpeed: TextView
    private lateinit var tvHeading: TextView
    private lateinit var tvAlt: TextView
    private lateinit var fabStartStop: ExtendedFloatingActionButton

    private lateinit var spFilter: Spinner
    private lateinit var spModel: Spinner
    private lateinit var spMatcher: Spinner

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedState: Bundle?): View? {
        val v = inflater.inflate(R.layout.fragment_control_panel, container, false)
        tvLatLon = v.findViewById(R.id.tvLatLon)
        tvSpeed = v.findViewById(R.id.tvSpeed)
        tvHeading = v.findViewById(R.id.tvHeading)
        tvAlt = v.findViewById(R.id.tvAlt)
        fabStartStop = v.findViewById(R.id.fabStartStop)
        spFilter = v.findViewById(R.id.spinnerFilter)
        spModel = v.findViewById(R.id.spinnerModel)
        spMatcher = v.findViewById(R.id.spinnerMatcher)
        return v
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        setupSpinners()

        fabStartStop.setOnClickListener {
            if (viewModel.isRunning.value) {
                stopService()
            } else {
                startService()
            }
        }

        viewLifecycleOwner.lifecycleScope.launch {
            viewLifecycleOwner.repeatOnLifecycle(Lifecycle.State.STARTED) {
                launch {
                    viewModel.navState.collectLatest { state ->
                        if (state != null) {
                            tvLatLon.text = String.format("%.6f°, %.6f°", state.lat, state.lon)
                            val speed = Math.hypot(state.vE.toDouble(), state.vN.toDouble())
                            tvSpeed.text = String.format("%.1f m/s", speed)
                            tvHeading.text = String.format("HDG: %.0f°", Math.toDegrees(state.yaw.toDouble()))
                            tvAlt.text = String.format("ALT: %.0f m", state.alt)
                        } else {
                            tvLatLon.text = "---.------°, ---.------°"
                            tvSpeed.text = "0.0 m/s"
                            tvHeading.text = "HDG: ---°"
                            tvAlt.text = "ALT: --- m"
                        }
                    }
                }

                launch {
                    viewModel.isRunning.collectLatest { running ->
                        if (running) {
                            fabStartStop.text = "Stop"
                            fabStartStop.setIconResource(android.R.drawable.ic_media_pause)
                            spFilter.isEnabled = false
                            spModel.isEnabled = false
                            spMatcher.isEnabled = false
                        } else {
                            fabStartStop.text = "Start"
                            fabStartStop.setIconResource(android.R.drawable.ic_media_play)
                            spFilter.isEnabled = true
                            spModel.isEnabled = true
                            spMatcher.isEnabled = true
                        }
                    }
                }
            }
        }
    }

    private fun setupSpinners() {
        // We'll hardcode the filter names for now (from FilterFactory)
        val filters = listOf("IEKF", "EKF", "MEKF", "RawINS")
        val models = ModelRegistry.names
        val matchers = MatcherRegistry.names

        spFilter.adapter = ArrayAdapter(requireContext(), android.R.layout.simple_spinner_dropdown_item, filters)
        spModel.adapter = ArrayAdapter(requireContext(), android.R.layout.simple_spinner_dropdown_item, models)
        spMatcher.adapter = ArrayAdapter(requireContext(), android.R.layout.simple_spinner_dropdown_item, matchers)

        spFilter.setSelection(filters.indexOf(viewModel.selectedFilter.value).coerceAtLeast(0))
        spModel.setSelection(models.indexOf(viewModel.selectedModel.value).coerceAtLeast(0))
        spMatcher.setSelection(matchers.indexOf(viewModel.selectedMatcher.value).coerceAtLeast(0))

        spFilter.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, pos: Int, id: Long) {
                viewModel.selectFilter(filters[pos])
            }
            override fun onNothingSelected(p0: AdapterView<*>?) {}
        }

        spModel.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, pos: Int, id: Long) {
                viewModel.selectModel(models[pos])
            }
            override fun onNothingSelected(p0: AdapterView<*>?) {}
        }

        spMatcher.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, pos: Int, id: Long) {
                viewModel.selectMatcher(matchers[pos])
            }
            override fun onNothingSelected(p0: AdapterView<*>?) {}
        }
    }

    private fun startService() {
        val intent = Intent(requireContext(), SensorCollectorService::class.java).apply {
            action = SensorCollectorService.ACTION_START
            putExtra(SensorCollectorService.EXTRA_FILTER_NAME, viewModel.selectedFilter.value)
            putExtra(SensorCollectorService.EXTRA_MODEL_NAME, viewModel.selectedModel.value)
            putExtra(SensorCollectorService.EXTRA_MATCHER_NAME, viewModel.selectedMatcher.value)
            putExtra(SensorCollectorService.EXTRA_GRAPH_PATH, "") // Hardcode empty for now
        }
        ContextCompat.startForegroundService(requireContext(), intent)
    }

    private fun stopService() {
        val intent = Intent(requireContext(), SensorCollectorService::class.java).apply {
            action = SensorCollectorService.ACTION_STOP
        }
        requireContext().startService(intent)
    }
}
