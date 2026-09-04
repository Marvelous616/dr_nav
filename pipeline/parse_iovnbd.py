#!/usr/bin/env python3
"""
parse_iovnbd.py — Parse the IO-VNBD dataset into standardized CSV files.

IO-VNBD repo: https://github.com/onyekpeu/IO-VNBD

Usage:
    python3 pipeline/parse_iovnbd.py --input datasets/IO-VNBD --output datasets/parsed

Output per sequence:
    <output>/<seq>/imu.csv          — timestamp_ns, ax, ay, az, gx, gy, gz
    <output>/<seq>/gnss.csv         — timestamp_ns, lat, lon, alt, hacc
    <output>/<seq>/groundtruth.csv  — timestamp_ns, lat, lon, alt, roll, pitch, yaw
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path


# ─── Field mappings for IO-VNBD (update if dataset schema differs) ────────────
# These are the expected column names in the raw dataset files.
IMU_COLS  = {
    "time"  : "timestamp_s",    # seconds since epoch
    "accX"  : "ax", "accY": "ay", "accZ": "az",   # m/s²
    "gyrX"  : "gx", "gyrY": "gy", "gyrZ": "gz",   # rad/s
}
GNSS_COLS = {
    "time"  : "timestamp_s",
    "lat"   : "lat", "lon": "lon", "alt": "alt",   # deg, deg, m
    "hAcc"  : "hacc",                               # m  (1-sigma)
}
GT_COLS = {
    "time"  : "timestamp_s",
    "lat"   : "lat", "lon": "lon", "alt": "alt",
    "roll"  : "roll", "pitch": "pitch", "yaw": "yaw",   # radians
}


def find_sequences(root: Path):
    """Yield (seq_name, imu_file, gnss_file, gt_file) tuples from dataset root."""
    # Look for subdirectories or grouped CSV files
    candidates = sorted(root.glob("**/imu*.csv"))
    if not candidates:
        # Flat layout: root/imu.csv, root/gps.csv, root/gt.csv
        candidates = [root / "imu.csv"]

    for imu_path in candidates:
        seq_dir  = imu_path.parent
        seq_name = seq_dir.name if seq_dir != root else "seq01"
        # Try common naming conventions for GNSS and ground truth
        gnss_path = next(
            (p for p in [seq_dir/"gnss.csv", seq_dir/"gps.csv", seq_dir/"GPS.csv",
                          seq_dir/"GNSS.csv", seq_dir/"reference.csv"]
             if p.exists()), None)
        gt_path = next(
            (p for p in [seq_dir/"groundtruth.csv", seq_dir/"gt.csv",
                          seq_dir/"reference.csv", seq_dir/"RTK.csv"]
             if p.exists()), None)
        yield seq_name, imu_path, gnss_path, gt_path


def parse_imu(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Try to map known column names
    rename = {}
    for raw, std in IMU_COLS.items():
        for col in df.columns:
            if raw.lower() in col.lower():
                rename[col] = std
                break
    df = df.rename(columns=rename)
    # Ensure timestamp in nanoseconds
    if "timestamp_s" in df.columns and "timestamp_ns" not in df.columns:
        df["timestamp_ns"] = (df["timestamp_s"] * 1e9).astype(np.int64)
    # Fill any missing sensor columns with zeros
    for c in ["ax","ay","az","gx","gy","gz"]:
        if c not in df.columns:
            df[c] = 0.0
    return df[["timestamp_ns","ax","ay","az","gx","gy","gz"]].dropna()


def parse_gnss(path: Path) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["timestamp_ns","lat","lon","alt","hacc"])
    df = pd.read_csv(path)
    rename = {}
    for raw, std in GNSS_COLS.items():
        for col in df.columns:
            if raw.lower() in col.lower():
                rename[col] = std
                break
    df = df.rename(columns=rename)
    if "timestamp_s" in df.columns and "timestamp_ns" not in df.columns:
        df["timestamp_ns"] = (df["timestamp_s"] * 1e9).astype(np.int64)
    if "hacc" not in df.columns:
        df["hacc"] = 5.0
    return df[["timestamp_ns","lat","lon","alt","hacc"]].dropna()


def parse_groundtruth(path: Path) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["timestamp_ns","lat","lon","alt","roll","pitch","yaw"])
    df = pd.read_csv(path)
    rename = {}
    for raw, std in GT_COLS.items():
        for col in df.columns:
            if raw.lower() in col.lower():
                rename[col] = std
                break
    df = df.rename(columns=rename)
    if "timestamp_s" in df.columns and "timestamp_ns" not in df.columns:
        df["timestamp_ns"] = (df["timestamp_s"] * 1e9).astype(np.int64)
    for c in ["roll","pitch","yaw"]:
        if c not in df.columns:
            df[c] = 0.0
    return df[["timestamp_ns","lat","lon","alt","roll","pitch","yaw"]].dropna()


def compute_stats(gt: pd.DataFrame) -> dict:
    """Compute basic dataset statistics."""
    if len(gt) < 2:
        return {}
    from math import radians, cos, sin, sqrt, atan2
    def haversine(lat1, lon1, lat2, lon2):
        R = 6378137.0
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
        return R * 2 * atan2(sqrt(a), sqrt(1-a))
    dists = [haversine(gt.lat.iloc[i], gt.lon.iloc[i],
                       gt.lat.iloc[i+1], gt.lon.iloc[i+1])
             for i in range(len(gt)-1)]
    return {
        "total_distance_m": sum(dists),
        "duration_s": (gt.timestamp_ns.iloc[-1] - gt.timestamp_ns.iloc[0]) / 1e9,
        "num_imu_samples": len(gt),
    }


def main():
    ap = argparse.ArgumentParser(description="Parse IO-VNBD dataset to standard CSV")
    ap.add_argument("--input",  default="datasets/IO-VNBD", help="Dataset root directory")
    ap.add_argument("--output", default="datasets/parsed",  help="Output directory")
    args = ap.parse_args()

    root   = Path(args.input)
    outdir = Path(args.output)

    if not root.exists():
        print(f"ERROR: Dataset root not found: {root}")
        print("Download IO-VNBD from: https://github.com/onyekpeu/IO-VNBD")
        return

    for seq_name, imu_f, gnss_f, gt_f in find_sequences(root):
        print(f"\n── Sequence: {seq_name} ──────────────────────────────────")
        seq_out = outdir / seq_name
        seq_out.mkdir(parents=True, exist_ok=True)

        imu = parse_imu(imu_f)
        print(f"  IMU:         {len(imu):,} samples  ({imu_f.name})")
        imu.to_csv(seq_out / "imu.csv", index=False)

        gnss = parse_gnss(gnss_f)
        print(f"  GNSS:        {len(gnss):,} fixes   ({gnss_f.name if gnss_f else 'not found'})")
        gnss.to_csv(seq_out / "gnss.csv", index=False)

        gt = parse_groundtruth(gt_f)
        print(f"  GroundTruth: {len(gt):,} samples  ({gt_f.name if gt_f else 'not found'})")
        gt.to_csv(seq_out / "groundtruth.csv", index=False)

        stats = compute_stats(gt if len(gt) > 1 else imu)
        print(f"  Route:       {stats.get('total_distance_m',0):.0f} m, "
              f"{stats.get('duration_s',0):.0f} s")

    print(f"\nDone. Parsed data written to: {outdir}")


if __name__ == "__main__":
    main()
