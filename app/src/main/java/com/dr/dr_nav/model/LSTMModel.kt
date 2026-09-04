package com.dr.dr_nav.model

import android.content.Context
import android.util.Log
import com.dr.dr_nav.engine.NavState
import com.dr.dr_nav.sensor.RawImuFrame
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.atan2

/**
 * BiLSTM TFLite outage model.
 *
 * Loads `assets/lstm.tflite`. If the asset is absent (e.g., before training),
 * the model gracefully falls back to NHC (returns PseudoMeasurement.NONE).
 *
 * Input tensor  : [1, N_WINDOW=100, N_FEATURES=12]  float32
 * Output tensor : [1, 2]  float32  — [dN, dE] metres
 *
 * Feature layout (per frame):
 *   0  ax,  1  ay,  2  az   — accelerometer (m/s²)
 *   3  gx,  4  gy,  5  gz   — gyroscope (rad/s)
 *   6  ins_N,  7  ins_E     — dead-reckoning position relative to outage start (m)
 *   8  vN,  9  vE           — filter velocity NED (m/s)
 *   10 heading              — yaw (rad)
 *   11 elapsed_s            — seconds since GNSS was lost
 */
class LSTMModel(context: Context) : TFLiteOutageModel(context, ASSET_NAME) {
    override val name = "LSTM"

    companion object {
        const val ASSET_NAME = "lstm.tflite"
    }
}

/**
 * TCN-BiLSTM TFLite outage model.
 * Same interface as [LSTMModel]; loads `assets/tcn_bilstm.tflite`.
 */
class TCNBiLSTMModel(context: Context) : TFLiteOutageModel(context, ASSET_NAME) {
    override val name = "TCNBiLSTM"

    companion object {
        const val ASSET_NAME = "tcn_bilstm.tflite"
    }
}

/**
 * Shared TFLite inference base for all sliding-window position predictors.
 */
abstract class TFLiteOutageModel(
    context   : Context,
    assetName : String,
) : OutageModel {

    private val TAG = "TFLiteOutageModel"

    // ── TFLite interpreter (null if asset not found) ──────────────
    private val interpreter: Interpreter? = try {
        val model = loadModelFile(context, assetName)
        val opts  = Interpreter.Options().apply { setNumThreads(2) }
        Interpreter(model, opts).also {
            Log.i(TAG, "Loaded $assetName  " +
                    "in=${it.getInputTensor(0).shape().contentToString()} " +
                    "out=${it.getOutputTensor(0).shape().contentToString()}")
        }
    } catch (e: Exception) {
        Log.w(TAG, "Asset '$assetName' not found — using NHC fallback (${e.message})")
        null
    }

    // ── Sliding window ────────────────────────────────────────────
    private val window = ArrayDeque<FloatArray>(OutageModel.N_WINDOW + 1)

    // Dead-reckoning accumulators for ins_N/ins_E features
    private var insN = 0f
    private var insE = 0f

    // ── TFLite buffers (pre-allocated) ────────────────────────────
    private val inputBuf  = ByteBuffer.allocateDirect(
        1 * OutageModel.N_WINDOW * OutageModel.N_FEATURES * Float.SIZE_BYTES
    ).apply { order(ByteOrder.nativeOrder()) }

    private val outputBuf = ByteBuffer.allocateDirect(
        1 * 2 * Float.SIZE_BYTES
    ).apply { order(ByteOrder.nativeOrder()) }

    // ── OutageModel implementation ────────────────────────────────

    override fun predict(
        window      : ArrayDeque<RawImuFrame>,
        state       : NavState,
        elapsedSecs : Float,
    ): PseudoMeasurement {
        if (interpreter == null) return PseudoMeasurement.NONE

        val frame = window.lastOrNull() ?: return PseudoMeasurement.NONE

        // Update dead-reckoning position
        val dt = 0.01f  // 100 Hz → 10 ms steps
        insN += state.vN * dt
        insE += state.vE * dt

        val heading = state.yaw

        val row = floatArrayOf(
            frame.ax, frame.ay, frame.az,
            frame.gx, frame.gy, frame.gz,
            insN, insE,
            state.vN, state.vE,
            heading, elapsedSecs,
        )

        this.window.addLast(row)
        while (this.window.size > OutageModel.N_WINDOW) this.window.removeFirst()

        if (this.window.size < OutageModel.N_WINDOW) return PseudoMeasurement.NONE

        // Fill input buffer
        inputBuf.rewind()
        for (r in this.window) for (v in r) inputBuf.putFloat(v)
        outputBuf.rewind()

        try {
            interpreter.run(inputBuf, outputBuf)
        } catch (e: Exception) {
            Log.e(TAG, "Inference error: ${e.message}")
            return PseudoMeasurement.NONE
        }

        outputBuf.rewind()
        val dN = outputBuf.float
        val dE = outputBuf.float
        return PseudoMeasurement(dN, dE, stdN = 1.5f, stdE = 1.5f)
    }

    /** Reset internal accumulators when a new outage starts. */
    fun reset() {
        window.clear()
        insN = 0f
        insE = 0f
    }

    // ── Asset loader ──────────────────────────────────────────────

    private fun loadModelFile(context: Context, assetName: String): MappedByteBuffer {
        val fd  = context.assets.openFd(assetName)
        val fis = FileInputStream(fd.fileDescriptor)
        val ch  = fis.channel
        return ch.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }
}
