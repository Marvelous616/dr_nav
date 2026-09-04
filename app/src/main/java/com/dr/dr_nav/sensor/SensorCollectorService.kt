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
        const val ACTION_START      = "com.dr.dr_nav.START_COLLECTION"
        const val ACTION_STOP       = "com.dr.dr_nav.STOP_COLLECTION"
        const val EXTRA_FILTER_NAME = "filter_name"
        const val NOTIF_CHANNEL_ID  = "dr_nav_sensor"
        const val NOTIF_ID          = 1001
        const val IMU_RATE_US       = 10_000   // 100 Hz  (SensorManager.SENSOR_DELAY_FASTEST ≈ this)
        const val GNSS_MIN_MS       = 100L      // 10 Hz GNSS request

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

    // ─── Lifecycle ─────────────────────────────────────────────────────────────

    override fun onCreate() {
        super.onCreate()
        sensorManager  = getSystemService(SENSOR_SERVICE) as SensorManager
        locationManager= getSystemService(LOCATION_SERVICE) as LocationManager
        createNotifChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val filterName = intent?.getStringExtra(EXTRA_FILTER_NAME) ?: "IEKF"
        startForeground(NOTIF_ID, buildNotification("Sensor collection active — filter: $filterName"))

        NativeFilterEngine.create(filterName)
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

            // Feed engine
            NativeFilterEngine.predict(ts,
                lastAccel[0], lastAccel[1], lastAccel[2],
                lastGyro[0],  lastGyro[1],  lastGyro[2])
            NativeFilterEngine.applyNhc()

            // Emit raw frame
            val frame = RawImuFrame(ts, lastAccel[0], lastAccel[1], lastAccel[2],
                                        lastGyro[0], lastGyro[1], lastGyro[2])
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
