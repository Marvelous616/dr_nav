package com.dr.dr_nav.ui

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.dr.dr_nav.R
import com.dr.dr_nav.engine.GnssMode
import com.google.android.material.chip.Chip
import kotlinx.coroutines.launch
import org.maplibre.android.MapLibre
import org.maplibre.android.geometry.LatLng
import org.maplibre.android.maps.MapView
import org.maplibre.android.maps.MapLibreMap
import org.maplibre.android.maps.Style
import org.maplibre.android.style.layers.LineLayer
import org.maplibre.android.style.layers.PropertyFactory.*
import org.maplibre.android.style.layers.SymbolLayer
import org.maplibre.android.style.sources.GeoJsonSource
import org.maplibre.geojson.Feature
import org.maplibre.geojson.LineString
import org.maplibre.geojson.Point

class MapFragment : Fragment() {

    private val viewModel: NavViewModel by activityViewModels()

    private lateinit var mapView: MapView
    private lateinit var chipMode: Chip
    private lateinit var tvOutageTimer: TextView

    private var map: MapLibreMap? = null
    private var isFirstFix = true

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        MapLibre.getInstance(requireContext())
    }

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedState: Bundle?): View? {
        val v = inflater.inflate(R.layout.fragment_map, container, false)
        mapView       = v.findViewById(R.id.mapView)
        chipMode      = v.findViewById(R.id.chipMode)
        tvOutageTimer = v.findViewById(R.id.tvOutageTimer)
        chipMode.visibility = View.GONE
        return v
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        mapView.onCreate(savedInstanceState)
        mapView.getMapAsync { mapLibreMap ->
            map = mapLibreMap
            // Use a free OSM-based vector style (or fallback to raster)
            mapLibreMap.setStyle(Style.Builder().fromUri("https://demotiles.maplibre.org/style.json")) { style ->
                initMapLayers(style)
                observeViewModel()
            }
        }
    }

    private fun initMapLayers(style: Style) {
        // 1. Trajectory line layer
        style.addSource(GeoJsonSource("traj-source"))
        style.addLayer(LineLayer("traj-layer", "traj-source").withProperties(
            lineColor("#3F51B5"), // Indigo
            lineWidth(4f),
            lineCap("round"),
            lineJoin("round")
        ))

        // 2. Current position marker layer (simple circle via symbol fallback or icon)
        style.addSource(GeoJsonSource("pos-source"))
        style.addLayer(SymbolLayer("pos-layer", "pos-source").withProperties(
            iconImage("marker-icon"), // Assuming style has this, or we rely on fallback
            iconSize(1.5f),
            iconAllowOverlap(true),
            textColor("#FFFFFF"),
            textField("{mode}") // Show first letter of mode
        ))
    }

    private fun observeViewModel() {
        viewLifecycleOwner.lifecycleScope.launch {
            viewLifecycleOwner.repeatOnLifecycle(Lifecycle.State.STARTED) {
                // Observe trajectory list
                launch {
                    viewModel.trajectory.collect { traj ->
                        if (traj.isEmpty()) return@collect
                        val style = map?.style ?: return@collect
                        val src = style.getSourceAs<GeoJsonSource>("traj-source")
                        val pts = traj.map { Point.fromLngLat(it.second, it.first) }
                        src?.setGeoJson(LineString.fromLngLats(pts))
                    }
                }

                // Observe current NavState for marker and badge update
                launch {
                    viewModel.navState.collect { state ->
                        if (state == null) return@collect

                        val pt = Point.fromLngLat(state.lon, state.lat)
                        val style = map?.style ?: return@collect
                        val src = style.getSourceAs<GeoJsonSource>("pos-source")
                        src?.setGeoJson(Feature.fromGeometry(pt))

                        // Centre map on first fix
                        if (isFirstFix) {
                            map?.cameraPosition = org.maplibre.android.camera.CameraPosition.Builder()
                                .target(LatLng(state.lat, state.lon))
                                .zoom(16.0)
                                .build()
                            isFirstFix = false
                        }

                        // Update badges
                        chipMode.visibility = View.VISIBLE
                        when (state.mode) {
                            GnssMode.GNSS_GOOD -> {
                                chipMode.text = "GNSS"
                                chipMode.setChipBackgroundColorResource(android.R.color.holo_green_dark)
                                tvOutageTimer.visibility = View.GONE
                            }
                            GnssMode.GNSS_DEGRADED -> {
                                chipMode.text = "WEAK GNSS"
                                chipMode.setChipBackgroundColorResource(android.R.color.holo_orange_dark)
                                tvOutageTimer.visibility = View.GONE
                            }
                            GnssMode.DR_ACTIVE -> {
                                chipMode.text = "DR_ACTIVE"
                                chipMode.setChipBackgroundColorResource(android.R.color.holo_red_dark)
                                tvOutageTimer.visibility = View.VISIBLE
                                tvOutageTimer.text = "Outage: ${"%.1f".format(state.outageSecs)} s"
                            }
                            GnssMode.MAP_MATCHED -> {
                                chipMode.text = "SNAPPED"
                                chipMode.setChipBackgroundColorResource(android.R.color.holo_purple)
                                tvOutageTimer.visibility = View.GONE
                            }
                        }
                    }
                }
            }
        }
    }

    // Lifecycle forwarding
    override fun onStart() { super.onStart(); mapView.onStart() }
    override fun onResume() { super.onResume(); mapView.onResume() }
    override fun onPause() { super.onPause(); mapView.onPause() }
    override fun onStop() { super.onStop(); mapView.onStop() }
    override fun onLowMemory() { super.onLowMemory(); mapView.onLowMemory() }
    override fun onDestroyView() { super.onDestroyView(); mapView.onDestroy() }
}
