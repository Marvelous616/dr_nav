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


# ─── OutageModel base class ──────────────────────────────────────────────────

class OutageModel:
    """Base class for all outage-duration pseudo-GNSS predictors."""
    name = "Base"

    def predict_outage_delta(self, imu_window, filt):
        """
        Given the recent IMU window and filter state, return a pseudo-GNSS
        position increment (dN, dE, stdN, stdE) in metres.
        imu_window: list of (ax,ay,az,gx,gy,gz) tuples, last N_WINDOW steps.
        filt: BaseFilter with .lat, .lon, .vN, .vE attributes.
        """
        raise NotImplementedError


# ─── NHC (Non-Holonomic Constraint) — no ML ──────────────────────────────────

class NHCModel(OutageModel):
    """Zero lateral velocity constraint. Minimal correction, no ML."""
    name = "NHC"

    def predict_outage_delta(self, imu_window, filt):
        # NHC: lateral vel ≈ 0 → push position estimate along heading only.
        # Return zero delta (heading-aligned motion handled inside filter).
        return 0.0, 0.0, 2.0, 2.0   # (dN, dE, stdN, stdE)


# ─── LSTM outage model ────────────────────────────────────────────────────────

class LSTMModel(OutageModel):
    """Loads models/lstm.pth if present; falls back to NHC silently."""
    name = "LSTM"
    _N_WINDOW   = 100
    _N_FEATURES = 12

    def __init__(self):
        self._model = None
        self._device = None
        self._window = []   # rolling deque of feature rows
        self._load()

    def _load(self):
        ckpt_path = Path("models/lstm.pth")
        if not ckpt_path.exists():
            return
        try:
            import torch
            from pipeline.train_lstm import LSTMPredictor, N_WINDOW, N_FEATURES
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
            m = LSTMPredictor(hidden=ckpt.get("hidden", 128),
                              layers=ckpt.get("layers", 2))
            m.load_state_dict(ckpt["model_state"])
            m.eval()
            self._model = m
            self._device = torch.device("cpu")
        except Exception as e:
            print(f"[LSTMModel] Could not load checkpoint ({e}). Using NHC fallback.")

    def _make_row(self, imu_window, filt, elapsed):
        """Build one feature row from the latest IMU reading."""
        if imu_window and len(imu_window[-1]) >= 6:
            ax, ay, az, gx, gy, gz = imu_window[-1][:6]
        else:
            ax = ay = az = gx = gy = gz = 0.0
        ins_N = ins_E = 0.0  # relative N/E from filter origin (approx 0 in Python sim)
        vN = getattr(filt, "vN", 0.0)
        vE = getattr(filt, "vE", 0.0)
        heading = math.atan2(vE, vN) if (vN != 0 or vE != 0) else 0.0
        return [ax, ay, az, gx, gy, gz, ins_N, ins_E, vN, vE, heading, float(elapsed)]

    def predict_outage_delta(self, imu_window, filt, elapsed=0.0):
        if self._model is None:
            return 0.0, 0.0, 2.0, 2.0

        import torch, numpy as np
        row = self._make_row(imu_window, filt, elapsed)
        self._window.append(row)
        if len(self._window) > self._N_WINDOW:
            self._window = self._window[-self._N_WINDOW:]

        if len(self._window) < self._N_WINDOW:
            return 0.0, 0.0, 2.0, 2.0

        x = torch.tensor(self._window, dtype=torch.float32).unsqueeze(0)  # (1,T,F)
        with torch.no_grad():
            dN_dE = self._model(x).squeeze(0).numpy()
        return float(dN_dE[0]), float(dN_dE[1]), 1.5, 1.5


# ─── TCN-BiLSTM outage model ──────────────────────────────────────────────────

class TCNBiLSTMModel(LSTMModel):
    """Loads models/tcn_bilstm.pth if present; falls back to NHC silently."""
    name = "TCNBiLSTM"

    def _load(self):
        ckpt_path = Path("models/tcn_bilstm.pth")
        if not ckpt_path.exists():
            return
        try:
            import torch
            from pipeline.train_tcn_bilstm import TCNBiLSTMPredictor
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
            m = TCNBiLSTMPredictor()
            m.load_state_dict(ckpt["model_state"])
            m.eval()
            self._model = m
            self._device = torch.device("cpu")
        except Exception as e:
            print(f"[TCNBiLSTMModel] Could not load checkpoint ({e}). Using NHC fallback.")


# ─── GBDT outage model ────────────────────────────────────────────────────────

class GBDTModel(OutageModel):
    """Loads models/gbdt_dN.json + gbdt_dE.json + gbdt_scaler.npz if present."""
    name = "GBDT"
    _N_WINDOW   = 100
    _N_FEATURES = 12

    def __init__(self):
        self._models = []
        self._scaler_mean = None
        self._scaler_scale = None
        self._window = []
        self._load()

    def _load(self):
        try:
            import xgboost as xgb, numpy as np
            mN = xgb.XGBRegressor(); mN.load_model("models/gbdt/gbdt_dN.json")
            mE = xgb.XGBRegressor(); mE.load_model("models/gbdt/gbdt_dE.json")
            sc = np.load("models/gbdt/gbdt_scaler.npz")
            self._models = [mN, mE]
            self._scaler_mean  = sc["mean"]
            self._scaler_scale = sc["scale"]
        except Exception as e:
            pass  # Graceful fallback to NHC

    def predict_outage_delta(self, imu_window, filt, elapsed=0.0):
        if not self._models:
            return 0.0, 0.0, 2.0, 2.0

        import numpy as np
        if imu_window and len(imu_window[-1]) >= 6:
            ax, ay, az, gx, gy, gz = imu_window[-1][:6]
        else:
            ax = ay = az = gx = gy = gz = 0.0
        vN = getattr(filt, "vN", 0.0)
        vE = getattr(filt, "vE", 0.0)
        heading = math.atan2(vE, vN) if (vN != 0 or vE != 0) else 0.0

        row = [ax, ay, az, gx, gy, gz, 0.0, 0.0, vN, vE, heading, float(elapsed)]
        self._window.append(row)
        if len(self._window) > self._N_WINDOW:
            self._window = self._window[-self._N_WINDOW:]
        if len(self._window) < self._N_WINDOW:
            return 0.0, 0.0, 2.0, 2.0

        W = np.array(self._window, dtype=np.float32)
        feat = np.concatenate([W.mean(0), W.std(0), W[-1]]).reshape(1, -1)
        feat = (feat - self._scaler_mean) / self._scaler_scale
        dN = float(self._models[0].predict(feat)[0])
        dE = float(self._models[1].predict(feat)[0])
        return dN, dE, 1.2, 1.2


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
MODELS  = [NHCModel, LSTMModel, TCNBiLSTMModel, GBDTModel]


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
