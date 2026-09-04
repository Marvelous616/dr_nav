#!/usr/bin/env python3
"""
benchmark/plot_trajectory.py — Replay filter×model combos and overlay
all resulting trajectories on a single map-style plot.

Usage:
    python3 benchmark/plot_trajectory.py \
        --dataset datasets/parsed/seq01 \
        --outage-sec 30 \
        --filters RawINS EKF IEKF \
        --models NHC TCNBiLSTM

Requires:
    pip install numpy pandas matplotlib tabulate tqdm
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Import all filter + model classes from run_all
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from benchmark.run_all import (
    RawINS, EKF, IEKF,
    NHCModel, LSTMModel, TCNBiLSTMModel, GBDTModel,
    haversine_m,
)


FILTER_CLASSES = {c.name: c for c in [RawINS, EKF, IEKF]}
MODEL_CLASSES  = {c.name: c for c in [NHCModel, LSTMModel, TCNBiLSTMModel, GBDTModel]}

# Colour palette for (filter, model) combos
_PALETTE = [
    "#e63946", "#457b9d", "#2a9d8f", "#e9c46a",
    "#f4a261", "#264653", "#a8dadc", "#6a4c93",
    "#b5838d", "#6c757d",
]


def replay_trace(imu_df, gnss_df, gt_df, filt, model, outage_sec,
                 outage_start_frac=0.4):
    """
    Replay the sequence with the given filter × model and return lists of
    (timestamp_ns, lat, lon) sampled at ~1 Hz during the outage window.
    Also returns (outage_start_ns, outage_end_ns) for plotting the marker.
    """
    total_dur     = (gt_df.timestamp_ns.iloc[-1] - gt_df.timestamp_ns.iloc[0]) / 1e9
    outage_start_ns = gt_df.timestamp_ns.iloc[0] + int(outage_start_frac * total_dur * 1e9)
    outage_end_ns   = outage_start_ns + int(outage_sec * 1e9)

    # Init filter on first GNSS fix
    g0 = gnss_df.iloc[0]
    filt.reset(g0.lat, g0.lon, g0.alt, int(g0.timestamp_ns))

    outage_ts, outage_lat, outage_lon = [], [], []
    full_ts,   full_lat,   full_lon   = [], [], []
    gnss_idx   = 0
    outage_start_actual = -1

    for _, row in imu_df.iterrows():
        ts_ns = int(row.timestamp_ns)
        filt.predict(ts_ns, row.ax, row.ay, row.az, row.gx, row.gy, row.gz)
        in_outage = outage_start_ns <= ts_ns <= outage_end_ns

        if in_outage:
            if outage_start_actual < 0:
                outage_start_actual = ts_ns
            elapsed = (ts_ns - outage_start_actual) * 1e-9
            dN, dE, sN, sE = model.predict_outage_delta(
                [(row.ax, row.ay, row.az, row.gx, row.gy, row.gz)], filt, elapsed)
            if dN != 0 or dE != 0:
                filt.update_pseudo(dN, dE, sN, sE)
        else:
            while gnss_idx < len(gnss_df) and gnss_df.timestamp_ns.iloc[gnss_idx] <= ts_ns:
                g = gnss_df.iloc[gnss_idx]
                filt.update_gnss(int(g.timestamp_ns), g.lat, g.lon, g.alt, g.hacc)
                gnss_idx += 1

        # Sample at ~1 Hz
        if ts_ns % 1_000_000_000 < 11_000_000:
            full_ts.append(ts_ns)
            full_lat.append(filt.lat)
            full_lon.append(filt.lon)
            if in_outage:
                outage_ts.append(ts_ns)
                outage_lat.append(filt.lat)
                outage_lon.append(filt.lon)

    return (full_ts, full_lat, full_lon,
            outage_ts, outage_lat, outage_lon,
            outage_start_ns, outage_end_ns)


def meters_to_deg(m, lat_ref):
    """Convert metre offsets to degrees (for axis labels)."""
    R = 6_378_137.0
    dlat = m / R * (180 / math.pi)
    dlon = m / (R * math.cos(math.radians(lat_ref))) * (180 / math.pi)
    return dlat, dlon


def plot(args):
    root   = Path(args.dataset)
    seq    = root.name

    imu_df  = pd.read_csv(root / "imu.csv")
    gnss_df = pd.read_csv(root / "gnss.csv")
    gt_df   = pd.read_csv(root / "groundtruth.csv")

    print(f"Dataset: {root}  |  outage: {args.outage_sec}s")

    # Select requested combos
    filter_names = args.filters or list(FILTER_CLASSES.keys())
    model_names  = args.models  or ["NHC"]
    combos = [(fn, mn) for fn in filter_names for mn in model_names
              if fn in FILTER_CLASSES and mn in MODEL_CLASSES]

    if not combos:
        print("ERROR: No valid filter×model combos. "
              f"Filters: {list(FILTER_CLASSES)}, Models: {list(MODEL_CLASSES)}")
        return

    # ── Create figure ─────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8), facecolor="#0f0f1a")
    gs  = GridSpec(1, 2, figure=fig, width_ratios=[2, 1], wspace=0.3)
    ax_map  = fig.add_subplot(gs[0])
    ax_err  = fig.add_subplot(gs[1])

    for ax in [ax_map, ax_err]:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#adb5bd")
        for spine in ax.spines.values():
            spine.set_color("#495057")

    # ── Ground truth ──────────────────────────────────────────────
    lat0 = gt_df.lat.mean()
    lon0 = gt_df.lon.mean()

    def to_local(lat_arr, lon_arr):
        """Convert lat/lon arrays to local East-North metres."""
        R = 6_378_137.0
        dN = (np.array(lat_arr) - lat0) * np.pi / 180 * R
        dE = (np.array(lon_arr) - lon0) * np.pi / 180 * R * math.cos(math.radians(lat0))
        return dE, dN   # x=East, y=North

    gt_E, gt_N = to_local(gt_df.lat.values, gt_df.lon.values)
    ax_map.plot(gt_E, gt_N, color="#a8dadc", linewidth=2.5,
                label="Ground Truth", zorder=10)

    # ── Replay each combo ─────────────────────────────────────────
    error_data = {}   # {label: [errors over time in outage]}

    for i, (fn, mn) in enumerate(combos):
        colour = _PALETTE[i % len(_PALETTE)]
        label  = f"{fn}+{mn}"

        filt  = FILTER_CLASSES[fn]()
        model = MODEL_CLASSES[mn]()
        t0 = time.perf_counter()
        (full_ts, full_lat, full_lon,
         out_ts, out_lat, out_lon,
         outage_start_ns, outage_end_ns) = replay_trace(
            imu_df, gnss_df, gt_df, filt, model, args.outage_sec)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if not full_lat:
            print(f"  {label}: no trace — skipping")
            continue
        print(f"  {label}: {len(full_lat)} pts | {elapsed_ms:.0f} ms")

        # Full trace (faded)
        E, N = to_local(full_lat, full_lon)
        ax_map.plot(E, N, color=colour, linewidth=1.0, alpha=0.4)

        # Outage segment (bright)
        if out_lat:
            Eo, No = to_local(out_lat, out_lon)
            ax_map.plot(Eo, No, color=colour, linewidth=2.0, alpha=0.9, label=label)

        # ── Error vs time during outage ────────────────────────────
        if out_ts and len(gt_df) > 0:
            errs = []
            for ts_ns, elat, elon in zip(out_ts, out_lat, out_lon):
                idx = np.searchsorted(gt_df.timestamp_ns.values, ts_ns)
                if 0 <= idx < len(gt_df):
                    errs.append(haversine_m(elat, elon,
                                            gt_df.lat.iloc[idx],
                                            gt_df.lon.iloc[idx]))
            if errs:
                t_rel = [(ts - out_ts[0]) * 1e-9 for ts in out_ts[:len(errs)]]
                ax_err.plot(t_rel, errs, color=colour, linewidth=1.8, label=label)
                error_data[label] = errs

    # ── Outage zone marker on map ─────────────────────────────────
    if full_ts:
        out_mask = [outage_start_ns <= t <= outage_end_ns for t in full_ts]
        if any(out_mask):
            ax_map.axvspan(
                min(to_local([full_lat[i] for i, m in enumerate(out_mask) if m],
                              [full_lon[i] for i, m in enumerate(out_mask) if m])[0],
                    default=0),
                max(to_local([full_lat[i] for i, m in enumerate(out_mask) if m],
                              [full_lon[i] for i, m in enumerate(out_mask) if m])[0],
                    default=1),
                color="#ff006e", alpha=0.08, label="Outage zone")

    # ── Map styling ────────────────────────────────────────────────
    ax_map.set_title("Trajectory Comparison", color="white", fontsize=14, pad=10)
    ax_map.set_xlabel("East (m)", color="#adb5bd")
    ax_map.set_ylabel("North (m)", color="#adb5bd")
    ax_map.legend(loc="upper left", fontsize=8,
                  facecolor="#1d3557", edgecolor="#457b9d", labelcolor="white")
    ax_map.grid(color="#2d3561", linewidth=0.5, alpha=0.5)
    ax_map.set_aspect("equal")

    # ── Error plot styling ─────────────────────────────────────────
    ax_err.set_title(f"Position Error During Outage\n({args.outage_sec}s window)",
                     color="white", fontsize=13, pad=10)
    ax_err.set_xlabel("Time in outage (s)", color="#adb5bd")
    ax_err.set_ylabel("Horizontal error (m)", color="#adb5bd")
    ax_err.legend(loc="upper left", fontsize=8,
                  facecolor="#1d3557", edgecolor="#457b9d", labelcolor="white")
    ax_err.grid(color="#2d3561", linewidth=0.5, alpha=0.5)

    fig.suptitle(f"IDR Benchmark — {seq}  |  outage={args.outage_sec}s",
                 color="white", fontsize=15, y=1.01)

    out_img = Path(f"benchmark/trajectory_{seq}.png")
    out_img.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_img), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nSaved: {out_img}")
    plt.show()


# ─── CLI ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Overlay trajectory plots for IDR benchmark")
    ap.add_argument("--dataset",    default="datasets/parsed/seq01")
    ap.add_argument("--outage-sec", type=float, default=30.0)
    ap.add_argument("--filters",    nargs="+", default=None,
                    help="Subset of filters to plot (default: all)")
    ap.add_argument("--models",     nargs="+", default=["NHC"],
                    help="Subset of outage models to plot (default: NHC)")
    plot(ap.parse_args())


if __name__ == "__main__":
    main()
