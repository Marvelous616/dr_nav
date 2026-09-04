#!/usr/bin/env python3
"""
benchmark/run_all.py — Replay IO-VNBD through all registered filter×model combos
and print an accuracy / latency comparison table.

Usage:
    python3 benchmark/run_all.py --dataset datasets/parsed/seq01 [--outage-sec 30]

Requires:
    pip install numpy pandas scipy tabulate tqdm
"""

import argparse
import time
import math
import numpy as np
import pandas as pd
from pathlib import Path
from tabulate import tabulate
from tqdm import tqdm

# ─── Pure-Python filter stubs ─────────────────────────────────────────────────
# These mirror the C++ implementations closely enough for offline evaluation.
# The Android C++ engine is more accurate (uses full matrix propagation);
# this script is for fast iteration and model comparison.

class BaseFilter:
    """Minimal state: lat, lon (degrees), velocity NED."""
    def __init__(self):
        self.lat = self.lon = self.alt = 0.0
        self.vN = self.vE = self.vD = 0.0
        self.R_EARTH = 6378137.0
        self.initialized = False

    def reset(self, lat, lon, alt, ts_ns):
        self.lat, self.lon, self.alt = lat, lon, alt
        self.prev_ts_ns = ts_ns
        self.initialized = True

    def predict(self, ts_ns, ax, ay, az, gx, gy, gz):
        raise NotImplementedError

    def update_gnss(self, ts_ns, lat, lon, alt, hacc):
        raise NotImplementedError

    def update_pseudo(self, dN, dE, stdN, stdE):
        lat_rad = math.radians(self.lat)
        self.lat += (dN / self.R_EARTH) * (180.0 / math.pi)
        self.lon += (dE / (self.R_EARTH * math.cos(lat_rad))) * (180.0 / math.pi)

    def get_pos(self):
        return self.lat, self.lon


class RawINS(BaseFilter):
    name = "RawINS"
    def predict(self, ts_ns, ax, ay, az, gx, gy, gz):
        if not self.initialized: return
        dt = (ts_ns - self.prev_ts_ns) * 1e-9
        self.prev_ts_ns = ts_ns
        if dt <= 0 or dt > 0.1: return
        # Simple: assume ax/ay are North/East specific force (naive, no attitude)
        self.vN += ax * dt; self.vE += ay * dt
        lat_rad = math.radians(self.lat)
        self.lat += self.vN * dt / self.R_EARTH * (180/math.pi)
        self.lon += self.vE * dt / (self.R_EARTH * math.cos(lat_rad)) * (180/math.pi)
    def update_gnss(self, ts_ns, lat, lon, alt, hacc): pass


class EKF(BaseFilter):
    name = "EKF"
    def __init__(self):
        super().__init__()
        self.P_N = 999.0; self.P_E = 999.0  # position variance (m²)

    def predict(self, ts_ns, ax, ay, az, gx, gy, gz):
        if not self.initialized: return
        dt = (ts_ns - self.prev_ts_ns) * 1e-9
        self.prev_ts_ns = ts_ns
        if dt <= 0 or dt > 0.1: return
        self.vN += ax * dt; self.vE += ay * dt
        lat_rad = math.radians(self.lat)
        self.lat += self.vN * dt / self.R_EARTH * (180/math.pi)
        self.lon += self.vE * dt / (self.R_EARTH*math.cos(lat_rad)) * (180/math.pi)
        self.P_N += 5e-4 * dt; self.P_E += 5e-4 * dt  # process noise

    def update_gnss(self, ts_ns, lat, lon, alt, hacc):
        R = max(hacc**2, 1.0) if hacc > 0 else 25.0
        K_N = self.P_N / (self.P_N + R)
        K_E = self.P_E / (self.P_E + R)
        lat_rad = math.radians(self.lat)
        dN = (lat - self.lat) * math.pi/180 * self.R_EARTH
        dE = (lon - self.lon) * math.pi/180 * self.R_EARTH * math.cos(lat_rad)
        self.lat += K_N * dN / self.R_EARTH * (180/math.pi)
        self.lon += K_E * dE / (self.R_EARTH * math.cos(lat_rad)) * (180/math.pi)
        self.P_N *= (1 - K_N); self.P_E *= (1 - K_E)


class IEKF(EKF):
    name = "IEKF"
    ITERS = 3

    def update_gnss(self, ts_ns, lat, lon, alt, hacc):
        R = max((hacc if hacc > 0 else 5.0)**2, 1.0)
        for _ in range(self.ITERS):
            lat_rad = math.radians(self.lat)
            dN = (lat - self.lat) * math.pi/180 * self.R_EARTH
            dE = (lon - self.lon) * math.pi/180 * self.R_EARTH * math.cos(lat_rad)
            K_N = self.P_N / (self.P_N + R)
            K_E = self.P_E / (self.P_E + R)
            self.lat += K_N * dN / self.R_EARTH * (180/math.pi)
            self.lon += K_E * dE / (self.R_EARTH*math.cos(lat_rad)) * (180/math.pi)
        self.P_N *= (1 - K_N); self.P_E *= (1 - K_E)


# ─── NHC outage model (no ML) ────────────────────────────────────────────────

class NHCModel:
    name = "NHC"
    def predict_outage_delta(self, imu_window, filt):
        # NHC: constrain lateral velocity → tiny correction only
        return 0.0, 0.0, 2.0, 2.0   # (dN, dE, stdN, stdE)


# ─── Evaluation helpers ───────────────────────────────────────────────────────

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6378137.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def simulate_outage(imu_df, gnss_df, gt_df, filt, model, outage_sec,
                    outage_start_frac=0.4):
    """Simulate a GNSS outage mid-sequence and measure drift."""
    if not filt.initialized: return None, None

    total_dur = (gt_df.timestamp_ns.iloc[-1] - gt_df.timestamp_ns.iloc[0]) / 1e9
    outage_start_ns = gt_df.timestamp_ns.iloc[0] + int(outage_start_frac * total_dur * 1e9)
    outage_end_ns   = outage_start_ns + int(outage_sec * 1e9)

    est_positions = []
    gt_interp_lat = np.interp(gt_df.timestamp_ns, gt_df.timestamp_ns, gt_df.lat)
    gt_interp_lon = np.interp(gt_df.timestamp_ns, gt_df.timestamp_ns, gt_df.lon)

    gnss_idx = 0
    for _, row in imu_df.iterrows():
        ts_ns = int(row.timestamp_ns)
        filt.predict(ts_ns, row.ax, row.ay, row.az, row.gx, row.gy, row.gz)
        in_outage = outage_start_ns <= ts_ns <= outage_end_ns

        if not in_outage:
            # Feed GNSS fixes that fall within this IMU step
            while gnss_idx < len(gnss_df) and gnss_df.timestamp_ns.iloc[gnss_idx] <= ts_ns:
                g = gnss_df.iloc[gnss_idx]
                filt.update_gnss(int(g.timestamp_ns), g.lat, g.lon, g.alt, g.hacc)
                gnss_idx += 1

        if outage_start_ns <= ts_ns <= outage_end_ns:
            dN, dE, sN, sE = model.predict_outage_delta(None, filt)
            if dN != 0 or dE != 0:
                filt.update_pseudo(dN, dE, sN, sE)
            if ts_ns % 100_000_000 < 11_000_000:  # ~1 Hz logging
                est_positions.append((ts_ns, filt.lat, filt.lon))

    if not est_positions:
        return None, None

    errors = []
    for ts_ns, est_lat, est_lon in est_positions:
        idx = np.searchsorted(gt_df.timestamp_ns.values, ts_ns)
        if 0 <= idx < len(gt_df):
            gt_lat = gt_df.lat.iloc[idx]
            gt_lon = gt_df.lon.iloc[idx]
            errors.append(haversine_m(est_lat, est_lon, gt_lat, gt_lon))

    if not errors:
        return None, None
    return np.sqrt(np.mean(np.array(errors)**2)), max(errors)   # RMSE, MaxDrift


# ─── Main ─────────────────────────────────────────────────────────────────────

FILTERS = [RawINS, EKF, IEKF]
MODELS  = [NHCModel]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",    default="datasets/parsed/seq01")
    ap.add_argument("--outage-sec", type=float, default=30.0)
    args = ap.parse_args()

    root = Path(args.dataset)
    print(f"\nBenchmark — dataset: {root}  outage: {args.outage_sec}s\n")

    imu_df  = pd.read_csv(root / "imu.csv")
    gnss_df = pd.read_csv(root / "gnss.csv")
    gt_df   = pd.read_csv(root / "groundtruth.csv")

    print(f"IMU: {len(imu_df):,} samples | GNSS: {len(gnss_df):,} fixes | GT: {len(gt_df):,} samples")

    results = []
    for Filt in FILTERS:
        for Model in MODELS:
            filt  = Filt()
            model = Model()

            # Initialize on first GNSS fix
            g0 = gnss_df.iloc[0]
            filt.reset(g0.lat, g0.lon, g0.alt, int(g0.timestamp_ns))

            t0 = time.perf_counter()
            rmse, max_drift = simulate_outage(imu_df, gnss_df, gt_df, filt, model, args.outage_sec)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            ms_per_step = elapsed_ms / max(len(imu_df), 1)

            if rmse is not None:
                results.append({
                    "Filter" : filt.name,
                    "Model"  : model.name,
                    "RMSE(m)": f"{rmse:.1f}",
                    "MaxDrift": f"{max_drift:.1f}",
                    "ms/step": f"{ms_per_step:.3f}",
                })
            else:
                results.append({
                    "Filter" : filt.name, "Model": model.name,
                    "RMSE(m)": "N/A", "MaxDrift": "N/A", "ms/step": f"{ms_per_step:.3f}",
                })

    print("\n" + tabulate(results, headers="keys", tablefmt="rounded_outline"))
    pd.DataFrame(results).to_csv("benchmark/results.csv", index=False)
    print("\nResults saved to benchmark/results.csv")


if __name__ == "__main__":
    main()
