#!/usr/bin/env python3
"""
pipeline/export_model.py — Export trained PyTorch models to ONNX and TFLite INT8.

Usage:
    python3 pipeline/export_model.py --model lstm
    python3 pipeline/export_model.py --model tcn_bilstm

Requires:
    pip install torch onnx
    pip install onnx2tf tensorflow  (for TFLite conversion)
    -- OR --
    pip install ai-edge-torch   (Google's newer approach, preferred)

Outputs (models/ dir):
    models/lstm.onnx
    models/lstm.tflite        ← INT8 quantized, copy to app/src/main/assets/
    models/tcn_bilstm.onnx
    models/tcn_bilstm.tflite
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from pipeline.train_lstm import LSTMPredictor, N_WINDOW, N_FEATURES
from pipeline.train_tcn_bilstm import TCNBiLSTMPredictor


MODEL_CFGS = {
    "lstm": {
        "cls"   : LSTMPredictor,
        "ckpt"  : "models/lstm.pth",
        "onnx"  : "models/lstm.onnx",
        "tflite": "models/lstm.tflite",
    },
    "tcn_bilstm": {
        "cls"   : TCNBiLSTMPredictor,
        "ckpt"  : "models/tcn_bilstm.pth",
        "onnx"  : "models/tcn_bilstm.onnx",
        "tflite": "models/tcn_bilstm.tflite",
    },
}


# ─── ONNX export ──────────────────────────────────────────────────

def export_onnx(model, onnx_path: Path):
    """Export PyTorch model to ONNX opset 17."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required")
    model.eval()
    dummy = torch.zeros(1, N_WINDOW, N_FEATURES)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        opset_version=17,
        input_names=["imu_window"],
        output_names=["position_delta"],
        dynamic_axes={"imu_window": {0: "batch"},
                      "position_delta": {0: "batch"}},
        do_constant_folding=True,
    )
    print(f"  ONNX  → {onnx_path}  ({onnx_path.stat().st_size / 1024:.0f} KB)")


# ─── TFLite export ────────────────────────────────────────────────

def _representative_dataset():
    """Generates ~100 random calibration samples for INT8 quantization."""
    for _ in range(100):
        yield [np.random.randn(1, N_WINDOW, N_FEATURES).astype(np.float32)]


def export_tflite_via_onnx2tf(onnx_path: Path, tflite_path: Path):
    """Convert ONNX → TF saved model → TFLite INT8 using onnx2tf."""
    try:
        import onnx2tf
        import tensorflow as tf
    except ImportError:
        print("  WARNING: onnx2tf or tensorflow not installed. "
              "pip install onnx2tf tensorflow")
        return False

    tf_dir = tflite_path.with_suffix("_tf_savedmodel")
    print(f"  Converting ONNX → TF SavedModel: {tf_dir}")
    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(tf_dir),
        non_verbose=True,
    )
    print(f"  Converting TF SavedModel → TFLite INT8: {tflite_path}")
    converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.float32   # keep float I/O for ease
    converter.inference_output_type = tf.float32
    tflite_model = converter.convert()
    tflite_path.write_bytes(tflite_model)
    print(f"  TFLite → {tflite_path}  ({len(tflite_model) / 1024:.0f} KB)")
    return True


def export_tflite_via_aiedge(model, tflite_path: Path):
    """Convert PyTorch model → TFLite using ai-edge-torch (Google, preferred)."""
    try:
        import ai_edge_torch
        import torch
    except ImportError:
        print("  WARNING: ai_edge_torch not installed. pip install ai-edge-torch")
        return False

    model.eval()
    sample = torch.zeros(1, N_WINDOW, N_FEATURES)
    edge_model = ai_edge_torch.convert(model.eval(), (sample,))
    edge_model.export(str(tflite_path))
    print(f"  TFLite (ai-edge-torch) → {tflite_path}  "
          f"({tflite_path.stat().st_size / 1024:.0f} KB)")
    return True


def export_tflite(model, onnx_path: Path, tflite_path: Path):
    """Try ai-edge-torch first, fall back to onnx2tf, warn if neither works."""
    if export_tflite_via_aiedge(model, tflite_path):
        return
    if export_tflite_via_onnx2tf(onnx_path, tflite_path):
        return
    print("  SKIPPED TFLite export — install either ai-edge-torch or onnx2tf+tensorflow")
    print(f"  Manually copy {onnx_path} to Android and use ONNX Runtime Mobile instead.")


# ─── Main ─────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Export IDR models to ONNX + TFLite")
    ap.add_argument("--model", choices=MODEL_CFGS.keys(), default="tcn_bilstm")
    args = ap.parse_args()

    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch required. pip install torch")
        sys.exit(1)

    cfg = MODEL_CFGS[args.model]
    ckpt_path   = Path(cfg["ckpt"])
    onnx_path   = Path(cfg["onnx"])
    tflite_path = Path(cfg["tflite"])

    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        print(f"  Run: python3 pipeline/train_{args.model}.py first")
        sys.exit(1)

    print(f"\nExporting  model={args.model}  from {ckpt_path}")

    device = torch.device("cpu")
    ckpt   = torch.load(str(ckpt_path), map_location=device)
    model  = cfg["cls"]()
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    export_onnx(model, onnx_path)
    export_tflite(model, onnx_path, tflite_path)

    # ── Copy hint for Android ──────────────────────────────────────
    asset_dir = Path("app/src/main/assets")
    if asset_dir.exists() and tflite_path.exists():
        import shutil
        dest = asset_dir / tflite_path.name
        shutil.copy2(str(tflite_path), str(dest))
        print(f"\n  Auto-copied → {dest}")
    else:
        print(f"\n  Next step: copy {tflite_path} → app/src/main/assets/")

    print("\nDone.")


if __name__ == "__main__":
    main()
