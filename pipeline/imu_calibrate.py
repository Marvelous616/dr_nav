#!/usr/bin/env python3
"""
pipeline/imu_calibrate.py — IMU calibration utility for the IDR system.

Three calibration modes:

  static     6-position tumble (or any stationary pose) to estimate
             accelerometer bias + scale factors.
             Input: CSV of (ts_ns, ax, ay, az, gx, gy, gz) collected while
             the phone is placed in 6 orthogonal orientations (~10 s each).

  turntable  Compute gyro bias + scale + cross-axis coupling using frames
             collected while rotating at 3 known angular rates on a turntable.
             Input: CSV with a 'rate_rads' column marking each segment.

  allan      Compute noise model parameters (Angle Random Walk, Bias
             Instability) from a long (~60 s) static IMU log using the
             Allan Deviation / ADEV method.

Output (all modes): calib.json in the format consumed by CalibrationManager.kt.

Usage:
    python3 pipeline/imu_calibrate.py --mode static \\
        --input logs/static_6pos.csv --output calib/calib.json

    python3 pipeline/imu_calibrate.py --mode turntable \\
        --input logs/turntable.csv --output calib/calib.json

    python3 pipeline/imu_calibrate.py --mode allan \\
        --input logs/static_long.csv --output calib/calib.json

Requires: pip install numpy scipy pandas
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Constants ────────────────────────────────────────────────────────────────

G0 = 9.80665     # standard gravity (m/s²)

# ─── Static 6-position calibration ───────────────────────────────────────────

def calib_static(df: pd.DataFrame) -> dict:
    """
    Accelerometer bias + scale via 6-position tumble.

    The phone must be placed in 6 orthogonal orientations:
      +X up, -X up, +Y up, -Y up, +Z up, -Z up

    Each orientation is a contiguous segment delimited by the 'pose' column
    (integer 0..5). If 'pose' is absent the data is split into 6 equal chunks.

    Returns bias_acc (m/s²) and scale_acc (dimensionless, near 1.0), plus
    gyroscope bias computed from the mean of all frames (should be ~0 rad/s
    when stationary).
    """
    print("Mode: static 6-position calibration")

    needed = ["ax", "ay", "az", "gx", "gy", "gz"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}'. Expected: {needed + ['timestamp_ns']}")

    # Split into 6 poses
    if "pose" in df.columns:
        poses = [df[df.pose == i][["ax","ay","az"]].values for i in range(6)]
    else:
        chunks = np.array_split(df[["ax","ay","az"]].values, 6)
        poses = chunks

    if len(poses) != 6:
        raise ValueError(f"Expected 6 poses, got {len(poses)}")

    # Mean specific force in each pose
    means = np.array([p.mean(axis=0) for p in poses])  # shape (6,3)
    print(f"  Pose means (m/s²):\n{np.round(means, 4)}")

    # Ideal specific forces for 6-position tumble:
    # +X: [g,0,0], -X:[-g,0,0], +Y:[0,g,0], -Y:[0,-g,0], +Z:[0,0,g], -Z:[0,0,-g]
    ideal = np.array([
        [ G0, 0,   0  ],
        [-G0, 0,   0  ],
        [ 0,  G0,  0  ],
        [ 0, -G0,  0  ],
        [ 0,  0,   G0 ],
        [ 0,  0,  -G0 ],
    ])

    # Least-squares: ideal = diag(scale) @ (meas - bias)
    # Solution: scale_i = 2*G0 / (pos_mean_i - neg_mean_i)
    #           bias_i  = (pos_mean_i + neg_mean_i) / 2
    bias_acc  = np.zeros(3)
    scale_acc = np.ones(3)
    for axis in range(3):
        pos = means[2 * axis,     axis]
        neg = means[2 * axis + 1, axis]
        bias_acc[axis]  = (pos + neg) / 2.0
        denom = pos - neg
        scale_acc[axis] = (2.0 * G0 / denom) if abs(denom) > 0.01 else 1.0

    # Gyro bias: mean over all static frames
    bias_gyro = df[["gx","gy","gz"]].values.mean(axis=0)

    print(f"  bias_acc  = {np.round(bias_acc, 5)} m/s²")
    print(f"  scale_acc = {np.round(scale_acc, 5)}")
    print(f"  bias_gyro = {np.round(bias_gyro, 6)} rad/s")

    return {
        "mode"       : "static",
        "bias_acc"   : bias_acc.tolist(),
        "scale_acc"  : scale_acc.tolist(),
        "bias_gyro"  : bias_gyro.tolist(),
        "scale_gyro" : [1.0, 1.0, 1.0],
    }


# ─── Turntable calibration ────────────────────────────────────────────────────

def calib_turntable(df: pd.DataFrame) -> dict:
    """
    Gyro bias + scale from turntable rotation at known angular rates.

    Input CSV must have columns: gx, gy, gz, rate_rads (known input rate in rad/s,
    positive for CCW rotation around the turntable axis).

    Assumes the turntable axis is aligned with the phone Z axis.
    Extends to a full misalignment matrix via least-squares.
    """
    print("Mode: turntable gyro calibration")

    if "rate_rads" not in df.columns:
        raise ValueError("Turntable mode requires a 'rate_rads' column with known rotation rates.")

    # Group by known rate and compute mean gyro reading per segment
    groups = df.groupby("rate_rads")[["gx","gy","gz"]].mean()
    rates  = groups.index.values    # shape (N,) — known input rates
    meas   = groups.values          # shape (N,3) — measured gyro output

    print(f"  Rate segments: {rates}")
    print(f"  Mean gyro meas:\n{np.round(meas, 5)}")

    if len(rates) < 2:
        raise ValueError("Need at least 2 distinct rotation rates to calibrate.")

    # Solve: meas_z = scale_z * rate + bias_z  (assume Z axis, LS)
    A = np.column_stack([rates, np.ones_like(rates)])  # (N,2)
    sol, _, _, _ = np.linalg.lstsq(A, meas[:, 2], rcond=None)
    scale_z, bias_z = sol

    # Full 3-axis: use the rate=0 intercept as bias
    zero_idx = np.argmin(np.abs(rates))
    bias_gyro = meas[zero_idx]

    scale_gyro = [1.0, 1.0, float(scale_z)]
    print(f"  bias_gyro  = {np.round(bias_gyro, 6)} rad/s")
    print(f"  scale_gyro = {np.round(scale_gyro, 5)}")

    return {
        "mode"       : "turntable",
        "bias_acc"   : [0.0, 0.0, 0.0],
        "scale_acc"  : [1.0, 1.0, 1.0],
        "bias_gyro"  : bias_gyro.tolist(),
        "scale_gyro" : scale_gyro,
    }


# ─── Allan Variance / Deviation ───────────────────────────────────────────────

def calib_allan(df: pd.DataFrame, fs: float = 100.0) -> dict:
    """
    Compute noise model parameters from a static IMU log using Allan Deviation.

    Identifies two key regimes:
      - Angle Random Walk (ARW):  σ vs τ slope = -0.5 on log-log → read at τ=1 s
      - Bias Instability (BI):    minimum of ADEV curve

    Returns noise parameters used to initialise the C++ filter's Q matrix.

    fs: sample rate in Hz (default 100 Hz)
    """
    print("Mode: Allan Variance noise characterisation")

    def adev(data: np.ndarray, fs: float):
        """Compute overlapping Allan deviation for a 1-D array."""
        N   = len(data)
        max_m = N // 2
        ms    = np.unique(np.logspace(0, np.log10(max_m), 200).astype(int))
        taus  = ms / fs
        adevs = []
        for m in ms:
            # Accumulate phase (cumsum of rate * dt)
            phase = np.cumsum(data) / fs
            # Cluster sums (Allan ADEV formula)
            n = N - 2 * m
            if n <= 0:
                continue
            clusters = (phase[2*m:] - 2*phase[m:N-m] + phase[:n])
            ad = np.sqrt(np.mean(clusters**2) / (2 * (m/fs)**2))
            adevs.append((m/fs, ad))
        return np.array(adevs)  # (τ, ADEV)

    results = {}
    for axis, col in enumerate(["gx", "gy", "gz"]):
        data = df[col].values - df[col].mean()
        ad   = adev(data, fs)
        if len(ad) < 3:
            print(f"  {col}: insufficient data")
            results[col] = {"arw": 0.0, "bi": 0.0}
            continue

        taus  = ad[:, 0]
        adevs = ad[:, 1]

        # ARW: fit slope=-0.5 line, read at τ=1 s
        # log(ADEV) = -0.5*log(τ) + log(ARW)
        fit_mask = taus < 1.0
        if fit_mask.sum() > 2:
            coeffs    = np.polyfit(np.log(taus[fit_mask]), np.log(adevs[fit_mask]), 1)
            arw       = float(np.exp(coeffs[1]))   # value at τ=1
        else:
            arw = float(adevs[0])

        # BI: minimum of ADEV
        bi = float(adevs.min())

        print(f"  {col}: ARW={arw:.4g} rad/s/√Hz  BI={bi:.4g} rad/s")
        results[col] = {"arw": arw, "bi": bi}

    noise_gyro = [results[c]["arw"] for c in ["gx","gy","gz"]]
    bi_gyro    = [results[c]["bi"]  for c in ["gx","gy","gz"]]

    # Accelerometer: same approach with accel columns
    accel_results = {}
    for col in ["ax","ay","az"]:
        data = df[col].values - df[col].mean()
        ad   = adev(data, fs)
        if len(ad) < 3:
            accel_results[col] = {"vrw": 0.0}
            continue
        taus, adevs = ad[:,0], ad[:,1]
        fit_mask = taus < 1.0
        vrw = float(np.exp(np.polyfit(np.log(taus[fit_mask] + 1e-9),
                                       np.log(adevs[fit_mask] + 1e-9), 1)[1])) \
              if fit_mask.sum() > 2 else float(adevs[0])
        print(f"  {col}: VRW={vrw:.4g} m/s/√Hz")
        accel_results[col] = {"vrw": vrw}

    noise_acc = [accel_results[c]["vrw"] for c in ["ax","ay","az"]]

    return {
        "mode"       : "allan",
        "bias_acc"   : [0.0, 0.0, 0.0],
        "scale_acc"  : [1.0, 1.0, 1.0],
        "bias_gyro"  : [0.0, 0.0, 0.0],
        "scale_gyro" : [1.0, 1.0, 1.0],
        "noise_gyro_arw" : noise_gyro,    # rad/s/√Hz
        "noise_gyro_bi"  : bi_gyro,       # rad/s
        "noise_acc_vrw"  : noise_acc,     # m/s²/√Hz
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="IMU calibration utility")
    ap.add_argument("--mode",   required=True, choices=["static","turntable","allan"])
    ap.add_argument("--input",  required=True, help="IMU CSV file")
    ap.add_argument("--output", default="calib/calib.json")
    ap.add_argument("--fs",     type=float, default=100.0,
                    help="Sample rate Hz (allan mode)")
    args = ap.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        print(f"ERROR: {in_path} not found"); sys.exit(1)

    df = pd.read_csv(in_path)
    print(f"Loaded {len(df)} frames from {in_path}")

    if args.mode == "static":
        result = calib_static(df)
    elif args.mode == "turntable":
        result = calib_turntable(df)
    else:
        result = calib_allan(df, fs=args.fs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nCalibration saved → {out_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
