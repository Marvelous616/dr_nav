package com.dr.dr_nav.sensor

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.GnssStatus
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.IBinder
import android.os.SystemClock
import androidx.core.app.NotificationCompat
import com.dr.dr_nav.engine.GnssMode
import com.dr.dr_nav.engine.NativeFilterEngine
import com.dr.dr_nav.map.MatcherRegistry
import com.dr.dr_nav.map.MatchMode
import com.dr.dr_nav.model.LSTMModel
import com.dr.dr_nav.model.ModelRegistry
import com.dr.dr_nav.model.TCNBiLSTMModel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter

/**
 * Foreground service that:
 *  1. Reads raw (uncalibrated) IMU at full rate → feeds to NativeFilterEngine
 *  2. Reads raw GNSS fixes → feeds to NativeFilterEngine
 *  3. Logs all raw frames to disk (CSV) for offline replay / training
 *  4. Publishes NavState updates via [navStateFlow]
 *
 * Bind to this service from MainActivity to read [navStateFlow].
 * Start with ACTION_START, stop with ACTION_STOP.
 */
class SensorCollectorService : Service() {

    companion object {
        const val ACTION_START       = "com.dr.dr_nav.START_COLLECTION"
        const val ACTION_STOP        = "com.dr.dr_nav.STOP_COLLECTION"
        const val EXTRA_FILTER_NAME  = "filter_name"
        const val EXTRA_MODEL_NAME   = "model_name"
        const val EXTRA_MATCHER_NAME = "matcher_name"
        const val EXTRA_GRAPH_PATH   = "graph_path"
        const val NOTIF_CHANNEL_ID   = "dr_nav_sensor"
        const val NOTIF_ID           = 1001
        const val IMU_RATE_US        = 10_000
        const val GNSS_MIN_MS        = 100L
        const val PSEUDO_INTERVAL_MS = 100L

        // Shared state — accessible from ViewModels
        private val _navState = MutableStateFlow<com.dr.dr_nav.engine.NavState?>(null)
        val navStateFlow: StateFlow<com.dr.dr_nav.engine.NavState?> = _navState.asStateFlow()

        private val _rawImuFlow = MutableSharedFlow<RawImuFrame>(extraBufferCapacity = 200)
        val rawImuFlow: SharedFlow<RawImuFrame> = _rawImuFlow.asSharedFlow()
    }

    private lateinit var sensorManager : SensorManager
    private lateinit var locationManager: LocationManager
    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())

    private var imuLogWriter  : BufferedWriter? = null
    private var gnssLogWriter : BufferedWriter? = null
    private var initialized   = false
    private var outageSecs    = 0f
    private var lastGnssTs    = 0L

    // ── Outage-model state ────────────────────────────────────────
    /** Sliding window of the last [OutageModel.N_WINDOW] IMU frames. */
    private val imuWindow = ArrayDeque<RawImuFrame>(110)
    /** Wall-clock ms at which the current GNSS outage started (-1 = no outage). */
    private var outageStartMs  = -1L
    /** Last time we fired a pseudo-update (ms). */
    private var lastPseudoMs   = 0L

    // ─── Lifecycle ─────────────────────────────────────────────────────────────

    override fun onCreate() {
        super.onCreate()
        sensorManager  = getSystemService(SENSOR_SERVICE) as SensorManager
        locationManager= getSystemService(LOCATION_SERVICE) as LocationManager
        createNotifChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val filterName  = intent?.getStringExtra(EXTRA_FILTER_NAME)  ?: "IEKF"
        val modelName   = intent?.getStringExtra(EXTRA_MODEL_NAME)   ?: "NHC"
        val matcherName = intent?.getStringExtra(EXTRA_MATCHER_NAME) ?: "Passthrough"
        val graphPath   = intent?.getStringExtra(EXTRA_GRAPH_PATH)   ?: ""
        startForeground(NOTIF_ID, buildNotification(
            "Collection active — filter: $filterName  model: $modelName  matcher: $matcherName"))

        NativeFilterEngine.create(filterName)

        // Register TFLite models lazily (need Context)
        ModelRegistry.registerTFLite(
            lstmModel = LSTMModel(applicationContext),
            tcnModel  = TCNBiLSTMModel(applicationContext),
        )
        ModelRegistry.select(modelName)

        // Load road graph + select matcher
        if (graphPath.isNotEmpty()) MatcherRegistry.setGraphPath(graphPath)
        MatcherRegistry.select(matcherName)

        openLogFiles()
        registerSensors()
        registerGnss()
        return START_STICKY
    }

    override fun onDestroy() {
        sensorManager.unregisterListener(imuListener)
        stopGnss()
        imuLogWriter?.close()
        gnssLogWriter?.close()
        scope.cancel()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    // ─── IMU listener ──────────────────────────────────────────────────────────

    private val imuListener = object : SensorEventListener {
        var lastAccel = FloatArray(3)
        var lastGyro  = FloatArray(3)
        var hasAccel  = false
        var hasGyro   = false

        override fun onSensorChanged(event: SensorEvent) {
            val ts = event.timestamp   // nanoseconds (monotonic)
            when (event.sensor.type) {
                Sensor.TYPE_ACCELEROMETER_UNCALIBRATED -> {
                    lastAccel[0] = event.values[0]
                    lastAccel[1] = event.values[1]
                    lastAccel[2] = event.values[2]
                    hasAccel = true
                }
                Sensor.TYPE_GYROSCOPE_UNCALIBRATED -> {
                    lastGyro[0] = event.values[0]
                    lastGyro[1] = event.values[1]
                    lastGyro[2] = event.values[2]
                    hasGyro = true
                }
            }
            if (!hasAccel || !hasGyro) return

            // Maintain sliding window for outage model
            val frame = RawImuFrame(ts, lastAccel[0], lastAccel[1], lastAccel[2],
                                       lastGyro[0],  lastGyro[1],  lastGyro[2])
            imuWindow.addLast(frame)
            while (imuWindow.size > com.dr.dr_nav.model.OutageModel.N_WINDOW + 10)
                imuWindow.removeFirst()

            // Feed engine
            NativeFilterEngine.predict(ts,
                lastAccel[0], lastAccel[1], lastAccel[2],
                lastGyro[0],  lastGyro[1],  lastGyro[2])
            NativeFilterEngine.applyNhc()

            // ── Outage model pseudo-update ─────────────────────────────────
            val nowMs = System.currentTimeMillis()
            val state = NativeFilterEngine.getNavState()
            if (state.mode == GnssMode.DR_ACTIVE) {
                if (outageStartMs < 0) {
                    outageStartMs = nowMs
                    lastPseudoMs  = nowMs
                }
                if (nowMs - lastPseudoMs >= PSEUDO_INTERVAL_MS) {
                    lastPseudoMs = nowMs
                    val elapsedSecs = (nowMs - outageStartMs) / 1000f
                    val meas = ModelRegistry.active.predict(imuWindow, state, elapsedSecs)
                    if (meas.stdN < 500f) {  // ignore NONE measurements
                        NativeFilterEngine.updatePseudo(
                            timestampNs = ts,
                            dN = meas.dN, dE = meas.dE,
                            stdN = meas.stdN, stdE = meas.stdE,
                        )
                    }
                }
            } else {
                outageStartMs = -1L
            }

            // Emit raw frame
            scope.launch { _rawImuFlow.emit(frame) }

            // Log to CSV
            imuLogWriter?.write("$ts,${lastAccel[0]},${lastAccel[1]},${lastAccel[2]},${lastGyro[0]},${lastGyro[1]},${lastGyro[2]}\n")

            // Publish nav state every ~10 Hz (every 10 IMU samples @ 100 Hz)
            if (ts % 10_000_000L < 200_000L) {  // crude 10 Hz gate
                val state = NativeFilterEngine.getNavState()
                _navState.value = state
            }
        }
        override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {}
    }

    private fun registerSensors() {
        val accel  = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER_UNCALIBRATED)
        val gyro   = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE_UNCALIBRATED)
        accel?.let { sensorManager.registerListener(imuListener, it, IMU_RATE_US) }
        gyro?.let  { sensorManager.registerListener(imuListener, it, IMU_RATE_US) }
    }

    // ─── GNSS listener ─────────────────────────────────────────────────────────

    private val gnssListener = LocationListener { loc ->
        val ts = loc.elapsedRealtimeNanos
        lastGnssTs = ts
        if (!initialized) {
            NativeFilterEngine.reset(loc.latitude, loc.longitude, loc.altitude,
                                     timestampNs = ts)
            initialized = true
        }
        NativeFilterEngine.updateGnss(
            timestampNs = ts,
            lat         = loc.latitude,
            lon         = loc.longitude,
            alt         = loc.altitude,
            hacc        = if (loc.hasAccuracy()) loc.accuracy else -1f,
            satellites  = 0,
            valid       = true
        )

        // ── Map matching ───────────────────────────────────────
        val filterState = NativeFilterEngine.getNavState()
        val speed   = Math.hypot(filterState.vN.toDouble(), filterState.vE.toDouble()).toFloat()
        val matchResult = MatcherRegistry.active.match(
            lat        = loc.latitude,
            lon        = loc.longitude,
            headingRad = filterState.yaw,
            speedMs    = speed,
            posStdM    = filterState.posStdN,
        )
        if (matchResult.snapped) {
            // Push snapped position back into the filter as a tight pseudo-GNSS fix
            val snappedHacc = if (loc.hasAccuracy()) loc.accuracy * 0.5f else 2f
            NativeFilterEngine.updateGnss(
                timestampNs = ts,
                lat         = matchResult.lat,
                lon         = matchResult.lon,
                alt         = loc.altitude,
                hacc        = snappedHacc,
                satellites  = 0,
                valid       = true,
            )
            // Road bearing hint: inject as soft heading pseudo-update if within 20°
            val hint = (MatcherRegistry.active as? com.dr.dr_nav.map.HMMMatcher)?.lastBearingHint
            if (hint != null && !hint.isNaN()) {
                // Encode bearing as a very small (0 m) pseudo-measurement with tight std
                // so the filter only updates heading, not position
                NativeFilterEngine.updatePseudo(ts, 0f, 0f, 0.1f, 0.1f)
            }
        }

        gnssLogWriter?.write("$ts,${loc.latitude},${loc.longitude},${loc.altitude},${loc.accuracy}\n")
    }

    @Suppress("MissingPermission")
    private fun registerGnss() {
        try {
            locationManager.requestLocationUpdates(
                LocationManager.GPS_PROVIDER, GNSS_MIN_MS, 0f, gnssListener)
        } catch (e: SecurityException) { /* permission not granted yet */ }
    }

    private fun stopGnss() {
        try { locationManager.removeUpdates(gnssListener) } catch (_: Exception) {}
    }

    // ─── Disk logging ──────────────────────────────────────────────────────────

    private fun openLogFiles() {
        val dir = File(getExternalFilesDir(null), "dr_logs")
        dir.mkdirs()
        val ts = System.currentTimeMillis()
        imuLogWriter  = BufferedWriter(FileWriter(File(dir, "imu_$ts.csv")))
        gnssLogWriter = BufferedWriter(FileWriter(File(dir, "gnss_$ts.csv")))
        imuLogWriter?.write("timestamp_ns,ax,ay,az,gx,gy,gz\n")
        gnssLogWriter?.write("timestamp_ns,lat,lon,alt,hacc\n")
    }

    // ─── Notification ──────────────────────────────────────────────────────────

    private fun createNotifChannel() {
        val ch = NotificationChannel(NOTIF_CHANNEL_ID, "Dead Reckoning", NotificationManager.IMPORTANCE_LOW)
        getSystemService(NotificationManager::class.java).createNotificationChannel(ch)
    }

    private fun buildNotification(content: String): Notification =
        NotificationCompat.Builder(this, NOTIF_CHANNEL_ID)
            .setContentTitle("DR Navigation")
            .setContentText(content)
            .setSmallIcon(android.R.drawable.ic_menu_mylocation)
            .build()
}

// ─── Data class for raw IMU (emitted to UI for debug overlay) ─────────────────
data class RawImuFrame(
    val timestampNs : Long,
    val ax: Float, val ay: Float, val az: Float,
    val gx: Float, val gy: Float, val gz: Float,
)
