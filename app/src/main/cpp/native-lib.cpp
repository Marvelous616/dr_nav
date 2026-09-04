#include "filters/filter_factory.h"
#include <android/log.h>
#include <jni.h>
#include <memory>
#include <string>

#define LOG_TAG "DR_Native"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// ─── One global filter instance per JNI session
// ─────────────────────────────── Thread safety: caller (Kotlin) must
// synchronize around these calls.
static std::unique_ptr<FusionFilter> g_filter;

extern "C" {

// Create / switch filter by name ("RawINS", "EKF", "MEKF", "IEKF")
JNIEXPORT jboolean JNICALL
Java_com_dr_dr_1nav_engine_NativeFilterEngine_nCreate(JNIEnv *env, jobject,
                                                      jstring jname) {
  const char *name_c = env->GetStringUTFChars(jname, nullptr);
  std::string name(name_c);
  env->ReleaseStringUTFChars(jname, name_c);
  try {
    g_filter = FilterFactory::create(name);
    LOGI("Filter created: %s", name.c_str());
    return JNI_TRUE;
  } catch (std::exception &e) {
    LOGE("FilterFactory error: %s", e.what());
    return JNI_FALSE;
  }
}

// Reset with initial NavState from GNSS fix
JNIEXPORT void JNICALL Java_com_dr_dr_1nav_engine_NativeFilterEngine_nReset(
    JNIEnv *, jobject, jdouble lat, jdouble lon, jdouble alt, jfloat vN,
    jfloat vE, jfloat vD, jlong timestamp_ns) {
  if (!g_filter)
    return;
  NavState init;
  init.lat = lat;
  init.lon = lon;
  init.alt = alt;
  init.vN = vN;
  init.vE = vE;
  init.vD = vD;
  init.timestamp_ns = timestamp_ns;
  g_filter->reset(init);
}

// Feed one IMU sample
JNIEXPORT void JNICALL Java_com_dr_dr_1nav_engine_NativeFilterEngine_nPredict(
    JNIEnv *, jobject, jlong timestamp_ns, jfloat ax, jfloat ay, jfloat az,
    jfloat gx, jfloat gy, jfloat gz) {
  if (!g_filter)
    return;
  ImuFrame f{timestamp_ns, ax, ay, az, gx, gy, gz};
  g_filter->predict(f);
}

// Feed GNSS fix
JNIEXPORT void JNICALL
Java_com_dr_dr_1nav_engine_NativeFilterEngine_nUpdateGnss(
    JNIEnv *, jobject, jlong timestamp_ns, jdouble lat, jdouble lon,
    jdouble alt, jfloat vN, jfloat vE, jfloat vD, jfloat hacc, jfloat vacc,
    jint satellites, jfloat hdop, jboolean valid) {
  if (!g_filter)
    return;
  GnssFrame f;
  f.timestamp_ns = timestamp_ns;
  f.lat = lat;
  f.lon = lon;
  f.alt = alt;
  f.vN = vN;
  f.vE = vE;
  f.vD = vD;
  f.hacc = hacc;
  f.vacc = vacc;
  f.satellites = satellites;
  f.hdop = hdop;
  f.valid = (valid == JNI_TRUE);
  g_filter->updateGNSS(f);
}

// Feed pseudo-GNSS from ML model
JNIEXPORT void JNICALL
Java_com_dr_dr_1nav_engine_NativeFilterEngine_nUpdatePseudo(
    JNIEnv *, jobject, jlong timestamp_ns, jfloat dN, jfloat dE, jfloat stdN,
    jfloat stdE) {
  if (!g_filter)
    return;
  PseudoMeas m{timestamp_ns, dN, dE, stdN, stdE};
  g_filter->updatePseudo(m);
}

// Apply NHC
JNIEXPORT void JNICALL
Java_com_dr_dr_1nav_engine_NativeFilterEngine_nApplyNhc(JNIEnv *, jobject) {
  if (!g_filter)
    return;
  g_filter->applyNHC();
}

// Get state — returns a float[] of length 18:
// [lat(1), lon(1), alt(1), vN, vE, vD, roll, pitch, yaw, posStdN, posStdE,
// mode(1), outage_s, biasAX,AY,AZ, BGX,BGY,BGZ] Note: lat/lon are packed as
// float but we double-encode via separate call for precision
JNIEXPORT jfloatArray JNICALL
Java_com_dr_dr_1nav_engine_NativeFilterEngine_nGetState(JNIEnv *env, jobject) {
  jfloatArray arr = env->NewFloatArray(18);
  if (!g_filter)
    return arr;
  NavState s = g_filter->getState();
  float buf[18] = {(float)s.lat,
                   (float)s.lon,
                   s.alt,
                   s.vN,
                   s.vE,
                   s.vD,
                   s.roll,
                   s.pitch,
                   s.yaw,
                   s.pos_std_N,
                   s.pos_std_E,
                   (float)s.mode,
                   s.outage_seconds,
                   s.bias_acc.x,
                   s.bias_acc.y,
                   s.bias_acc.z,
                   s.bias_gyro.x,
                   s.bias_gyro.y};
  env->SetFloatArrayRegion(arr, 0, 18, buf);
  return arr;
}

// Precision lat/lon getter (double precision)
JNIEXPORT jdoubleArray JNICALL
Java_com_dr_dr_1nav_engine_NativeFilterEngine_nGetLatLon(JNIEnv *env, jobject) {
  jdoubleArray arr = env->NewDoubleArray(2);
  if (!g_filter)
    return arr;
  NavState s = g_filter->getState();
  double buf[2] = {s.lat, s.lon};
  env->SetDoubleArrayRegion(arr, 0, 2, buf);
  return arr;
}

// Legacy: original JNI function expected by MainActivity skeleton
JNIEXPORT jstring JNICALL
Java_com_dr_dr_1nav_MainActivity_stringFromJNI(JNIEnv *env, jobject) {
  return env->NewStringUTF(
      "DR Engine ready — call nCreate() to select a filter.");
}

} // extern "C"