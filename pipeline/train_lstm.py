#!/usr/bin/env python3
"""
pipeline/train_lstm.py — Train a simple BiLSTM to predict position increments
during GNSS outages, using IO-VNBD parsed data.

Usage:
    python3 pipeline/train_lstm.py \
        --dataset datasets/parsed/seq01 \
        --epochs 50 \
        --output models/lstm.pth

Model I/O
---------
Input  : (batch, N_WINDOW=100, 12)
         [ax, ay, az, gx, gy, gz, ins_N, ins_E, vN, vE, heading_rad, elapsed_s]
Output : (batch, 2)   [dN, dE]  — position increment in nav frame (metres)
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────
# Lazy-import torch so the file is importable even without PyTorch
# (benchmark stubs use the class interface without training)
# ──────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, random_split
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

N_WINDOW   = 100   # 1 second @ 100 Hz
N_FEATURES = 12
STEP_HZ    = 100   # IMU rate


# ─── Dataset ──────────────────────────────────────────────────────

class OutageDataset:
    """
    Sliding-window supervised dataset. Each sample is a 100-frame IMU window
    labelled with the ground-truth position change at the end of the window.
    Outages are simulated by masking GNSS over random intervals.
    """

    def __init__(self, imu_df: pd.DataFrame, gnss_df: pd.DataFrame,
                 gt_df: pd.DataFrame, outage_sec: float = 30.0,
                 n_outages: int = 5, stride: int = 10):
        self.X, self.Y = self._build(imu_df, gnss_df, gt_df,
                                     outage_sec, n_outages, stride)

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2):
        R = 6_378_137.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2
             + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
             * math.sin(dlon / 2) ** 2)
        dist = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        bearing = math.atan2(
            math.sin(dlon) * math.cos(math.radians(lat2)),
            math.cos(math.radians(lat1)) * math.sin(math.radians(lat2))
            - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2))
            * math.cos(dlon),
        )
        return dist * math.cos(bearing), dist * math.sin(bearing)  # dN, dE

    def _build(self, imu_df, gnss_df, gt_df, outage_sec, n_outages, stride):
        imu = imu_df.reset_index(drop=True)
        gt  = gt_df.reset_index(drop=True)

        total_ns = imu.timestamp_ns.iloc[-1] - imu.timestamp_ns.iloc[0]
        outage_ns = int(outage_sec * 1e9)

        # Simulate multiple non-overlapping outage windows
        rng = np.random.default_rng(42)
        starts = []
        for _ in range(n_outages):
            s = rng.integers(int(0.2 * total_ns),
                             int(0.8 * total_ns) - outage_ns)
            starts.append(imu.timestamp_ns.iloc[0] + int(s))
        outage_intervals = [(s, s + outage_ns) for s in starts]

        # Interpolate ground-truth lat/lon onto IMU timestamps
        gt_lat_itp = np.interp(imu.timestamp_ns.values,
                               gt.timestamp_ns.values, gt.lat.values)
        gt_lon_itp = np.interp(imu.timestamp_ns.values,
                               gt.timestamp_ns.values, gt.lon.values)

        # Build feature array
        N = len(imu)
        feat = np.zeros((N, N_FEATURES), dtype=np.float32)
        feat[:, 0] = imu.ax.values
        feat[:, 1] = imu.ay.values
        feat[:, 2] = imu.az.values
        feat[:, 3] = imu.gx.values
        feat[:, 4] = imu.gy.values
        feat[:, 5] = imu.gz.values
        # INS position (relative to start) — filled below
        feat[:, 8] = 0.0   # vN (placeholder — not available in parsed data)
        feat[:, 9] = 0.0   # vE
        feat[:, 10] = 0.0  # heading (rad)

        # Simple dead-reckoning to populate ins_N, ins_E, heading, elapsed
        lat0, lon0 = gt_lat_itp[0], gt_lon_itp[0]
        ins_N, ins_E = 0.0, 0.0
        vN, vE = 0.0, 0.0
        heading = 0.0
        outage_start_idx = -1

        for i in range(N):
            ts_ns = imu.timestamp_ns.iloc[i]

            # Check outage
            in_outage = any(s <= ts_ns <= e for s, e in outage_intervals)
            if in_outage and outage_start_idx < 0:
                outage_start_idx = i
            elif not in_outage:
                outage_start_idx = -1

            elapsed = 0.0 if outage_start_idx < 0 else (
                (ts_ns - imu.timestamp_ns.iloc[outage_start_idx]) * 1e-9)

            if i > 0:
                dt = (imu.timestamp_ns.iloc[i] - imu.timestamp_ns.iloc[i - 1]) * 1e-9
                dt = max(0.0, min(dt, 0.05))
                # Heading from gyro z
                heading += imu.gz.iloc[i] * dt
                # Accel → velocity (body x → N, body y → E via heading)
                acc_N = imu.ax.iloc[i] * math.cos(heading) - imu.ay.iloc[i] * math.sin(heading)
                acc_E = imu.ax.iloc[i] * math.sin(heading) + imu.ay.iloc[i] * math.cos(heading)
                vN += acc_N * dt
                vE += acc_E * dt
                ins_N += vN * dt
                ins_E += vE * dt

            feat[i, 6]  = ins_N
            feat[i, 7]  = ins_E
            feat[i, 8]  = vN
            feat[i, 9]  = vE
            feat[i, 10] = heading
            feat[i, 11] = elapsed

        # Build windows and labels
        X_list, Y_list = [], []
        for i in range(0, N - N_WINDOW - 1, stride):
            end = i + N_WINDOW
            ts_start = imu.timestamp_ns.iloc[i]
            ts_end   = imu.timestamp_ns.iloc[end]

            # Only include windows that are fully inside an outage
            if not any(s <= ts_start and ts_end <= e for s, e in outage_intervals):
                continue

            window = feat[i:end]  # (N_WINDOW, N_FEATURES)

            # Label: ground-truth dN, dE from start of window to end
            dN, dE = self._haversine(
                gt_lat_itp[i], gt_lon_itp[i],
                gt_lat_itp[end], gt_lon_itp[end])

            X_list.append(window)
            Y_list.append([dN, dE])

        if not X_list:
            print("WARNING: No outage windows found — check dataset size vs outage_sec")
            return np.zeros((1, N_WINDOW, N_FEATURES), np.float32), np.zeros((1, 2), np.float32)

        return np.stack(X_list).astype(np.float32), np.array(Y_list, dtype=np.float32)

    def __len__(self):  return len(self.X)
    def __getitem__(self, i):
        if not TORCH_AVAILABLE:
            return self.X[i], self.Y[i]
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i])


# ─── Model ────────────────────────────────────────────────────────

class LSTMPredictor(nn.Module if TORCH_AVAILABLE else object):
    """Bidirectional LSTM for GNSS-denial position increment prediction."""

    def __init__(self, input_size=N_FEATURES, hidden=128, layers=2, dropout=0.2):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required. pip install torch")
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers=layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.fc   = nn.Sequential(
            nn.Linear(hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):           # x: (B, T, F)
        out, _ = self.lstm(x)       # (B, T, H*2)
        return self.fc(out[:, -1])  # take last timestep → (B, 2)


# ─── Training ─────────────────────────────────────────────────────

def train(args):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required. pip install torch")

    root = Path(args.dataset)
    imu_df  = pd.read_csv(root / "imu.csv")
    gnss_df = pd.read_csv(root / "gnss.csv")
    gt_df   = pd.read_csv(root / "groundtruth.csv")

    print(f"Building dataset from {root} ...")
    ds = OutageDataset(imu_df, gnss_df, gt_df,
                       outage_sec=args.outage_sec, n_outages=args.n_outages)
    print(f"  Windows: {len(ds)}  shape: {ds.X.shape}")

    n_val   = max(1, int(0.1 * len(ds)))
    n_test  = max(1, int(0.1 * len(ds)))
    n_train = len(ds) - n_val - n_test
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test],
                                             generator=torch.Generator().manual_seed(42))

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = LSTMPredictor().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=5, factor=0.5)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            train_loss += loss.item() * len(xb)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += loss_fn(model(xb), yb).item() * len(xb)
        val_loss /= n_val
        sched.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs} | train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(),
                        "n_features": N_FEATURES, "n_window": N_WINDOW,
                        "hidden": 128, "layers": 2}, str(out_path))

    # ── Test RMSE ─────────────────────────────────────────────────
    ckpt = torch.load(str(out_path), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    errors = []
    with torch.no_grad():
        for xb, yb in test_dl:
            pred = model(xb.to(device)).cpu().numpy()
            gt   = yb.numpy()
            errors.extend(np.linalg.norm(pred - gt, axis=1).tolist())
    rmse = math.sqrt(np.mean(np.array(errors) ** 2))
    print(f"\nTest RMSE: {rmse:.3f} m   (best val MSE={best_val:.4f})")
    print(f"Saved: {out_path}")


# ─── CLI ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Train BiLSTM outage predictor")
    ap.add_argument("--dataset",    default="datasets/parsed/seq01")
    ap.add_argument("--outage-sec", type=float, default=30.0)
    ap.add_argument("--n-outages",  type=int,   default=8)
    ap.add_argument("--epochs",     type=int,   default=50)
    ap.add_argument("--batch",      type=int,   default=64)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--output",     default="models/lstm.pth")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
