#!/usr/bin/env python3
"""Overlay two (or more) scored real-robot runs: 2-D paths against the tracker truth
and planar error over time.

    python3 script/plot_run_compare.py --out mycuriosity/compare.png \\
        "stereo-only=mycuriosity/score_camonly/dataset_realv7_002" \\
        "stereo-inertial=mycuriosity/score_s/dataset_realv7_002"

Each argument is LABEL=<scoring dir of eval_real_gt.py> (the directory holding
aligned_orb.csv, ground_truth_robotclock.csv and metrics.json).  The paths are the
scorer's SE(2)-aligned estimates, so both runs sit in the tracker frame.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLS = ["#2a78d6", "#cc8319", "#a347b2", "#458554", "#b00020"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="LABEL=scoring_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--system", default="orb")
    a = ap.parse_args()
    runs = []
    for spec in a.runs:
        label, d = spec.split("=", 1); d = Path(d)
        al = np.loadtxt(d / f"aligned_{a.system}.csv", delimiter=",")
        m = json.load(open(d / "metrics.json")); m = m[0] if isinstance(m, list) else list(m.values())[0]
        runs.append(dict(label=label, t=al[:, 0] * 1e-9, gt=al[:, 1:3], est=al[:, 3:5], err=al[:, 5], m=m, dir=d))
    gt = np.loadtxt(runs[0]["dir"] / "ground_truth_robotclock.csv", delimiter=",")
    t0 = min(r["t"][0] for r in runs)

    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[1, 1])
    ax = fig.add_subplot(gs[:, 0])
    ax.plot(gt[:, 1], gt[:, 2], color="#2a2a2a", lw=2.2, label="tracker truth")
    for r, c in zip(runs, COLS):
        ax.plot(r["est"][:, 0], r["est"][:, 1], color=c, lw=1.3,
                label=f"{r['label']}: ATE {r['m']['ate_rmse']:.3f} m, RPE1 {r['m'].get('rpe', {}).get('1.0', {}).get('trans_rmse_clean', float('nan')):.3f} m")
        ax.plot(r["est"][0, 0], r["est"][0, 1], "o", color=c, ms=6); ax.plot(r["est"][-1, 0], r["est"][-1, 1], "s", color=c, ms=6)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3); ax.set_xlabel("x [m] (tracker frame)"); ax.set_ylabel("y [m]")
    ax.legend(fontsize=8, loc="lower right"); ax.set_title("paths after rigid SE(2) alignment (circle = start, square = end)", fontsize=10, loc="left")
    ax1 = fig.add_subplot(gs[0, 1])
    for r, c in zip(runs, COLS):
        ax1.plot(r["t"] - t0, r["err"], color=c, lw=1.2, label=r["label"])
    ax1.set_ylabel("planar error [m]"); ax1.grid(True, alpha=0.3); ax1.legend(fontsize=8); ax1.set_title("error against the tracker over time", fontsize=10, loc="left")
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
    # difference between the two estimates themselves, on the first run's timestamps
    if len(runs) >= 2:
        r0, r1 = runs[0], runs[1]
        x1 = np.interp(r0["t"], r1["t"], r1["est"][:, 0]); y1 = np.interp(r0["t"], r1["t"], r1["est"][:, 1])
        dd = np.hypot(r0["est"][:, 0] - x1, r0["est"][:, 1] - y1)
        ax2.plot(r0["t"] - t0, dd, color="#555", lw=1.2, label=f"|{r0['label']} - {r1['label']}|: rms {np.sqrt((dd**2).mean()):.3f} m, max {dd.max():.3f} m")
        ax2.legend(fontsize=8)
    ax2.set_ylabel("distance between the two estimates [m]"); ax2.set_xlabel("time from first matched sample [s]"); ax2.grid(True, alpha=0.3)
    fig.suptitle(f"{runs[0]['dir'].name}: " + " vs ".join(r["label"] for r in runs), fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(a.out, dpi=160); plt.close(fig)
    for r in runs:
        m = r["m"]
        print(f"{r['label']:18s} ATE {m['ate_rmse']:.3f} median {m['ate_median']:.3f} max {m['ate_max']:.3f} path ratio {m['path_ratio']:.2f} matched {m['matched']}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
