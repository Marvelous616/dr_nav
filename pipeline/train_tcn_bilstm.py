#!/usr/bin/env python3
"""
pipeline/train_tcn_bilstm.py — Train a TCN-BiLSTM to predict position
increments during GNSS outages. Matches the architecture in:
  Niu et al. "TCN-BiLSTM + ANR-IEKF" — Sensors 2026.

Usage:
    python3 pipeline/train_tcn_bilstm.py \
        --dataset datasets/parsed/seq01 \
        --epochs 80 \
        --output models/tcn_bilstm.pth

Same I/O contract as train_lstm.py:
  Input  : (batch, 100, 12)
  Output : (batch, 2)  [dN, dE] metres
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, random_split
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Re-use the Dataset and constants from train_lstm
from pipeline.train_lstm import OutageDataset, N_WINDOW, N_FEATURES


# ─── TCN residual block ───────────────────────────────────────────

class _TCNBlock(nn.Module if TORCH_AVAILABLE else object):
    """Dilated causal 1-D convolution + residual connection."""

    def __init__(self, in_ch, out_ch, kernel=3, dilation=1, dropout=0.2):
        super().__init__()
        pad = (kernel - 1) * dilation   # causal padding
        self.conv = nn.Sequential(
            nn.ConstantPad1d((pad, 0), 0),
            nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):   # x: (B, C, T)
        return self.conv(x) + self.down(x)


# ─── Full TCN-BiLSTM model ────────────────────────────────────────

class TCNBiLSTMPredictor(nn.Module if TORCH_AVAILABLE else object):
    """
    TCN encoder (3 residual blocks, dilations 1/2/4) +
    BiLSTM decoder (2 layers × 128 hidden) +
    MLP head → [dN, dE].
    """

    def __init__(self, input_size=N_FEATURES, tcn_channels=64,
                 lstm_hidden=128, lstm_layers=2, dropout=0.2):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required.")
        super().__init__()
        self.tcn = nn.Sequential(
            _TCNBlock(input_size,   tcn_channels, dilation=1, dropout=dropout),
            _TCNBlock(tcn_channels, tcn_channels, dilation=2, dropout=dropout),
            _TCNBlock(tcn_channels, tcn_channels, dilation=4, dropout=dropout),
        )
        self.lstm = nn.LSTM(tcn_channels, lstm_hidden, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):               # x: (B, T, F)
        x = x.permute(0, 2, 1)         # → (B, F, T)  for Conv1d
        x = self.tcn(x)                 # (B, C, T)
        x = x.permute(0, 2, 1)         # → (B, T, C)  for LSTM
        out, _ = self.lstm(x)
        return self.head(out[:, -1])    # last step → (B, 2)


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
    train_ds, val_ds, test_ds = random_split(
        ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(0))

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model   = TCNBiLSTMPredictor().to(device)
    optim   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    loss_fn = nn.HuberLoss(delta=5.0)   # more robust to outlier GNSS labels

    best_val = float("inf")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            train_loss += loss.item() * len(xb)
        train_loss /= n_train
        sched.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += loss_fn(model(xb), yb).item() * len(xb)
        val_loss /= n_val

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs} | train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "arch": "TCNBiLSTM",
                "n_features": N_FEATURES, "n_window": N_WINDOW,
                "tcn_channels": 64, "lstm_hidden": 128, "lstm_layers": 2,
            }, str(out_path))

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
    print(f"\nTest RMSE: {rmse:.3f} m   (best val loss={best_val:.4f})")
    print(f"Saved: {out_path}")


# ─── CLI ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Train TCN-BiLSTM outage predictor")
    ap.add_argument("--dataset",    default="datasets/parsed/seq01")
    ap.add_argument("--outage-sec", type=float, default=30.0)
    ap.add_argument("--n-outages",  type=int,   default=8)
    ap.add_argument("--epochs",     type=int,   default=80)
    ap.add_argument("--batch",      type=int,   default=64)
    ap.add_argument("--lr",         type=float, default=5e-4)
    ap.add_argument("--output",     default="models/tcn_bilstm.pth")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
