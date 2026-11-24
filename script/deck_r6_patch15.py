#!/usr/bin/env python3
"""Figure for slide R6 of the update_181126 deck: a robot standing still at the
start resets ORB-SLAM3's map, and fork patch 15 removes it.

    python3 script/deck_r6_patch15.py

Two panels, every mark a measured value:

  (a) per realv7 recording, on the estimator's clock: how long the robot stood
      still (wheel odometry, first 5 cm of travel), when stock ORB-SLAM3 reset
      its map (3 Nov runs), and when the IMU initialised with patch 15
      (11 Nov runs, local_mapping.csv imu_initialized).
  (b) per run, the tilt of the plane fitted to the estimated path, stock vs
      patch 15.  A floor robot's path should be level.

PROVENANCE
  The 3 Nov stock runs were overwritten by the 11 Nov re-runs.  Their tilt and
  reset times survive only in the FROZEN table
  output/compare/real/realv7-insights/tilt_table_prepatch15.tex and are copied
  below by hand from it.  Everything else is recomputed from disk.
  The plane fit is the table's own definition: SVD of the 3-D positions in
  vio.csv, angle between the plane normal and vertical.

Output: docs/report/insights/update_181126/fig/r6_patch15.png
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs/report/insights/update_181126/fig/r6_patch15.png")

RECS = ["dataset_realv7_000", "dataset_realv7_001", "dataset_realv7_002"]
LABEL = {"dataset_realv7_000": "realv7_000", "dataset_realv7_001": "realv7_001",
         "dataset_realv7_002": "realv7_002"}
CFGS = ["loopclosing_rate1", "noloopclosing_rate1", "loopclosing_rate1_cpu", "noloopclosing_rate1_cpu"]

# FROZEN stock values, tilt_table_prepatch15.tex, same CFGS order
# (GPU LC on, GPU LC off, CPU LC on, CPU LC off).
STOCK_TILT = {"dataset_realv7_000": [89.1, 89.5, 88.8, 87.9],
              "dataset_realv7_001": [88.8, 6.1, 1.5, 13.0],
              "dataset_realv7_002": [37.7, 59.7, 5.2, 21.7]}
STOCK_RESETS = {"dataset_realv7_000": [[3], [3], [5], [4]],
                "dataset_realv7_001": [[4], [3, 5], [4], [5]],
                "dataset_realv7_002": [[3, 49, 59], [49], [], []]}

TH = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#dedcd4",
          still="#d9d6cc", bad="#c0392b", good="#1e8449")
COL = {"dataset_realv7_000": "#c77c1e", "dataset_realv7_001": "#8e44ad",
       "dataset_realv7_002": "#1f7a8c"}


def plane_tilt(P):
    c = P - P.mean(0)
    n = np.linalg.svd(c, full_matrices=False)[2][2]
    return float(np.degrees(np.arccos(abs(n[2]) / np.linalg.norm(n))))


def run_dir(ds, cfg):
    return os.path.join(ROOT, "output/output_orb/real", ds, cfg)


def t0_of(ds):
    V = np.loadtxt(os.path.join(run_dir(ds, "loopclosing_rate1"), "vio.csv"), delimiter=",")
    return V[0, 0]


def wheel_start(ds):
    W = np.loadtxt(os.path.join(ROOT, "output/output_wheel/real", ds, "gyro_wheel/wheel.csv"),
                   delimiter=",", comments="#")
    t = (W[:, 0] - t0_of(ds)) / 1e9
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(W[:, 1:3], axis=0), axis=1))]
    return float(t[np.argmax(s > 0.05)])


def imu_init(ds, cfg):
    import csv
    rows = list(csv.DictReader(open(os.path.join(run_dir(ds, cfg), "local_mapping.csv"))))
    tf = open(os.path.join(run_dir(ds, cfg), "tracking_frontend.csv")).readlines()[1]
    t0 = min(int(rows[0]["timestamp_ns"]), int(tf.split(",")[0]))
    for r in rows:
        if r["imu_initialized"] == "1":
            return (int(r["timestamp_ns"]) - t0) / 1e9
    return None


def main():
    still = {ds: wheel_start(ds) for ds in RECS}
    init = {ds: [imu_init(ds, c) for c in CFGS] for ds in RECS}
    after = {ds: [plane_tilt(np.loadtxt(os.path.join(run_dir(ds, c), "vio.csv"),
                                        delimiter=",")[:, 1:4]) for c in CFGS] for ds in RECS}
    for ds in RECS:
        print(LABEL[ds], f"still until {still[ds]:.2f} s",
              "init", [round(x, 2) for x in init[ds]],
              "tilt after", [round(x, 2) for x in after[ds]])

    # Drawn at the size it is shown on the slide (left column, ~6.3 x 5.3 cm),
    # so 6 pt here is 6 pt on the slide.
    plt.rcParams.update({"font.size": 6.0, "font.family": "DejaVu Sans"})
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(2.55, 2.25),
                                 gridspec_kw=dict(height_ratios=[1.25, 1.0], hspace=0.62))
    fig.patch.set_facecolor(TH["surface"])
    for a in (ax, bx):
        a.set_facecolor(TH["surface"])
        for s in a.spines.values():
            s.set_color(TH["grid"])
        a.tick_params(colors=TH["ink2"], length=2, pad=1.5)

    # ---- (a) timeline
    XMAX = 13.5
    jit = [-0.24, -0.08, 0.08, 0.24]
    for row, ds in enumerate(RECS[::-1]):
        y = row
        ax.barh(y, still[ds], height=0.66, color=TH["still"], zorder=1)
        for k, ts in enumerate(STOCK_RESETS[ds]):
            for t in ts:
                if t <= XMAX:
                    ax.plot(t, y + jit[k], marker="x", ms=3.4, mew=1.0,
                            color=TH["bad"], zorder=3)
        late = sorted({t for ts in STOCK_RESETS[ds] for t in ts if t > XMAX})
        if late:
            ax.text(XMAX - 0.15, y, "\u2192 more resets\nat " + ", ".join(f"{t:g}" for t in late) + " s",
                    ha="right", va="center", color=TH["bad"], fontsize=5.0, linespacing=1.0)
        for k, t in enumerate(init[ds]):
            ax.plot(t, y + jit[k], marker="o", ms=2.3, color=TH["good"], zorder=4, mec="none")
    ax.set_yticks(range(3))
    ax.set_yticklabels([LABEL[d] for d in RECS[::-1]])
    ax.set_xlim(0, XMAX)
    ax.set_ylim(-0.55, 2.55)
    ax.set_xlabel("seconds since the recording starts", color=TH["ink2"], labelpad=1.0)
    ax.grid(axis="x", color=TH["grid"], lw=0.5, zorder=0)
    from matplotlib.lines import Line2D
    h2 = Line2D([], [], ls="", marker="x", color=TH["bad"], ms=3.4, mew=1.0, label="stock: map reset")
    h3 = Line2D([], [], ls="", marker="o", color=TH["good"], ms=2.3, mec="none", label="patch 15: IMU ready")
    from matplotlib.patches import Patch
    h1 = Patch(color=TH["still"], label="robot standing still")
    ax.legend(handles=[h1, h2, h3], loc="upper right", frameon=True, fancybox=False, framealpha=1.0,
              edgecolor=TH["grid"], fontsize=5.0, handlelength=0.8, borderpad=0.35,
              labelspacing=0.3, labelcolor=TH["ink2"], bbox_to_anchor=(1.0, 1.0))
    ax.set_title("(a) when the map was reset (one mark per run)", loc="left",
                 fontsize=6.0, color=TH["ink"], pad=3)

    # ---- (b) tilt before / after, as two columns of dots joined per run
    for ds in RECS:
        for b, a_ in zip(STOCK_TILT[ds], after[ds]):
            bx.plot([b, a_], [1, 0], color=COL[ds], lw=0.7, alpha=0.8)
            bx.plot(b, 1, "o", ms=2.4, color=COL[ds], mec="none")
            bx.plot(a_, 0, "o", ms=2.4, color=COL[ds], mec="none")
    bx.axvspan(0, 3, color=TH["good"], alpha=0.15, lw=0)
    bx.set_ylim(-0.45, 1.45)
    bx.set_xlim(-2, 93)
    bx.set_yticks([1, 0])
    bx.set_yticklabels(["stock", "patch 15"])
    bx.set_xticks([0, 30, 60, 90])
    bx.set_xlabel("tilt of the estimated floor [deg]   (a real floor is 0)",
                  color=TH["ink2"], labelpad=1.0)
    bx.grid(axis="x", color=TH["grid"], lw=0.5)
    hs = [Line2D([], [], color=COL[ds], lw=1.0, label=LABEL[ds]) for ds in RECS]
    bx.legend(handles=hs, loc="center right", frameon=False, fontsize=4.8,
              handlelength=0.9, labelspacing=0.2, labelcolor=TH["ink2"], bbox_to_anchor=(1.0, 0.42))
    bx.set_title("(b) how tilted the result is (one line per run)", loc="left",
                 fontsize=6.0, color=TH["ink"], pad=3)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor=TH["surface"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
