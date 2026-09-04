#!/usr/bin/env python3
"""
pipeline/train_gbdt.py — Train an XGBoost GBDT regressor as a fast,
non-sequential fallback model for GNSS denial.

Based on Gao & Feng (GPS Solutions 2025): GBDT is more robust than LSTM
under short outages and sensor noise spikes.

Usage:
    python3 pipeline/train_gbdt.py \
        --dataset datasets/parsed/seq01 \
        --output  models/gbdt.json

Requires:
    pip install xgboost numpy pandas scikit-learn onnxmltools skl2onnx

Output:
    models/gbdt_N.json   (XGBoost checkpoint, one per output dimension)
    models/gbdt.onnx     (ONNX export for Android ONNX Runtime)
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Reuse dataset builder from LSTM script
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pipeline.train_lstm import OutageDataset, N_WINDOW, N_FEATURES

# For GBDT, we use per-step (non-sequential) features: collapse the window
# to summary statistics (mean, std, last) across time for each feature.
GBDT_FEATURES_PER_COL = 3   # mean, std, last_value


def window_to_flat(X: np.ndarray) -> np.ndarray:
    """(N, T, F) → (N, F*3)  — mean, std, last across time axis."""
    mean = X.mean(axis=1)
    std  = X.std(axis=1)
    last = X[:, -1, :]
    return np.concatenate([mean, std, last], axis=1)


def train(args):
    if not XGB_AVAILABLE:
        print("ERROR: xgboost not installed. pip install xgboost")
        sys.exit(1)

    root   = Path(args.dataset)
    imu_df = pd.read_csv(root / "imu.csv")
    gnss_df= pd.read_csv(root / "gnss.csv")
    gt_df  = pd.read_csv(root / "groundtruth.csv")

    print(f"Building dataset from {root} ...")
    ds = OutageDataset(imu_df, gnss_df, gt_df,
                       outage_sec=args.outage_sec, n_outages=args.n_outages)
    X_raw = ds.X   # (N, N_WINDOW, N_FEATURES)
    Y     = ds.Y   # (N, 2)
    print(f"  Windows: {len(X_raw)}")

    # Flatten windows to per-sample feature vector
    X_flat = window_to_flat(X_raw)

    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X_flat, Y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    models = []
    for dim, label in enumerate(["dN", "dE"]):
        print(f"\nTraining XGBoost for output={label} ...")
        model = xgb.XGBRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.lr,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_tr, Y_tr[:, dim],
                  eval_set=[(X_te, Y_te[:, dim])],
                  verbose=args.n_estimators // 5)
        pred = model.predict(X_te)
        rmse = math.sqrt(mean_squared_error(Y_te[:, dim], pred))
        print(f"  Test RMSE ({label}): {rmse:.4f} m")

        ckpt = Path(args.output).with_suffix("") / f"gbdt_{label}.json"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(ckpt))
        print(f"  Saved: {ckpt}")
        models.append(model)

    # ── Final 2D RMSE ─────────────────────────────────────────────
    pred_2d = np.column_stack([m.predict(X_te) for m in models])
    rmse_2d = math.sqrt(np.mean(np.linalg.norm(pred_2d - Y_te, axis=1) ** 2))
    print(f"\nFinal 2D RMSE: {rmse_2d:.4f} m")

    # ── ONNX export (best-effort) ──────────────────────────────────
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        # Export each dim separately; Android will run both
        for dim, (m, label) in enumerate(zip(models, ["dN", "dE"])):
            # XGBoost → sklearn-api → onnx via skl2onnx
            init_types = [("input", FloatTensorType([None, X_tr.shape[1]]))]
            onnx_model = convert_sklearn(m, initial_types=init_types)
            onnx_path  = Path(args.output).with_suffix("") / f"gbdt_{label}.onnx"
            with open(str(onnx_path), "wb") as f:
                f.write(onnx_model.SerializeToString())
            print(f"ONNX exported: {onnx_path}")
    except Exception as e:
        print(f"ONNX export skipped ({e}) — install skl2onnx for ONNX export")

    # Save scaler params for inference
    scaler_path = out_dir / "gbdt_scaler.npz"
    np.savez(str(scaler_path), mean=scaler.mean_, scale=scaler.scale_)
    print(f"Scaler saved: {scaler_path}")


# ─── CLI ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Train XGBoost GBDT outage predictor")
    ap.add_argument("--dataset",       default="datasets/parsed/seq01")
    ap.add_argument("--outage-sec",    type=float, default=30.0)
    ap.add_argument("--n-outages",     type=int,   default=8)
    ap.add_argument("--n-estimators",  type=int,   default=500)
    ap.add_argument("--max-depth",     type=int,   default=6)
    ap.add_argument("--lr",            type=float, default=0.05)
    ap.add_argument("--output",        default="models/gbdt")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
