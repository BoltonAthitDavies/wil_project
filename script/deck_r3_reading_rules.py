#!/usr/bin/env python3
"""Deck slide R3: three rules for reading an ORB-SLAM3 number, one panel each.

    python3 script/deck_r3_reading_rules.py

  (a) Rule 1, repeatability.  The only repeated simulation runs: three
      identical repeats of each configuration on the two mapping recordings
      (U, I), closer disabled.  output/compare/simulation/exp3-repeats/
      mapping_repeats.csv, ate_rms.
  (b) Rule 2, rotation.  Orientation error against time for the 16 ORB-SLAM3
      runs on realistic_000-003 (4 configurations each).  Each estimate is
      rigidly aligned on its positions after 6 s; the error is the geodesic
      angle between estimated and true orientation.  x = seconds since the
      estimator's first pose.  Before inertial initialisation the node publishes
      in the camera optical frame, so the error sits at ~90 deg.
  (c) Rule 3, path length.  The same 16 runs, path length over the truth's,
      three ways, as in script/exp1_path_scale.py: as the scorer reports it;
      after dropping the pre-initialisation spike (up to the published frame
      switch + 1 s) and jump steps; and with both series sampled once a
      second.  The two loop-closing runs on realistic_001 (the false closure,
      R7) are left out: their 23 % is that closure.
"""
import os
import sys

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import deck_style as S

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vio_metrics as vm      # noqa: E402
import exp1_path_scale as PS  # noqa: E402

BAGS = ["dataset_allsensor_realistic_%03d" % i for i in range(4)]
RUNS = ["loopclosing_cpu", "loopclosing_gpu", "noloopclosing_cpu", "noloopclosing_gpu"]


def rotation_series(ds, run, g):
    t, p, q, _ = vm.load(os.path.join(S.ROOT, "output/output_orb/simulation", ds, run, "vio.csv"))
    m = (t >= g[0][0]) & (t <= g[0][-1])
    t, p, q = t[m], p[m], q[m]
    gp, gq = vm.resample(g[0], g[1], t), vm.resample_quat(g[0], g[2], t)
    x = t - t[0]
    R, _ = vm.umeyama(p[x > 6], gp[x > 6])
    e = vm.geodesic_deg(np.einsum("ij,njk->nik", R, vm.quat_to_R(q)), vm.quat_to_R(gq))
    return x, e


def path_excess(ds, run, g):
    res = vm.analyse("simulation/" + ds, systems=[("ORB-SLAM3", "output_orb", run)])
    row = res[0][0]
    reported = 100 * (row["path"] / row["ref"] - 1)
    d = os.path.join(S.ROOT, "output/output_orb/simulation", ds, run)
    t, p = vm.load(os.path.join(d, "vio.csv"))[:2]
    m = (t >= max(t[0], g[0][0])) & (t <= min(t[-1], g[0][-1]))
    t, p = t[m], p[m]
    pre = t < vm.world_frame_time(d) + 1.0
    spike = np.linalg.norm(np.diff(p[pre], axis=0), axis=1).sum()
    t, p = t[~pre], p[~pre]
    gq = vm.resample(g[0], g[1], t)
    se, sg = np.linalg.norm(np.diff(p, axis=0), axis=1), np.linalg.norm(np.diff(gq, axis=0), axis=1)
    jumps = vm.find_jumps(t, p)[0]
    keep = np.ones(len(se), bool)
    keep[list(jumps)] = False
    after = 100 * (se[keep].sum() / sg[keep].sum() - 1)
    s1e, s1g, idx = PS.decimated(t, p, gq, 1.0)
    k1 = np.array([not any((j >= idx[n]) & (j < idx[n + 1]) for j in jumps) for n in range(len(idx) - 1)])
    at1 = 100 * (s1e[k1].sum() / s1g[k1].sum() - 1)
    local = float(np.median((s1e[k1] / np.maximum(s1g[k1], 1e-9))[s1g[k1] > 0.3]))
    return reported, after, at1, spike, local


def card(h=0.98):
    import matplotlib.pyplot as plt
    S.setup()
    plt.rcParams.update({"font.size": 5.6})
    fig, ax = plt.subplots(figsize=(1.62, h))
    fig.patch.set_facecolor(S.TH["surface"])
    return fig, ax


def main():
    # ---- rule 1: repeats
    fig, ax = card()
    S.style(ax, grid="y")
    d = pd.read_csv(os.path.join(S.ROOT, "output/compare/simulation/exp3-repeats/mapping_repeats.csv"))
    groups = [("U", "ORB-SLAM3"), ("U", "ORB-SLAM3 GPU"), ("U", "VINS-Fusion"),
              ("I", "ORB-SLAM3"), ("I", "ORB-SLAM3 GPU"), ("I", "VINS-Fusion")]
    for i, (lay, lab) in enumerate(groups):
        v = d[(d.layout == lay) & (d.label == lab)].ate_rms.values
        col = S.VINS if "VINS" in lab else S.ORB
        x = i + (0.5 if i >= 3 else 0)
        ax.plot([x, x], [v.min(), v.max()], color=col, lw=2.0, alpha=0.35, solid_capstyle="round")
        ax.scatter([x] * 3, v, s=5, color=col, zorder=3, lw=0)
        print("%s %-14s repeats %s  spread %.3f m = %.0f %% of mean" % (lay, lab, np.round(v, 3), np.ptp(v), 100 * np.ptp(v) / v.mean()))
    ax.set_yscale("log")
    ax.set_ylim(0.1, 3)
    ax.set_yticks([0.1, 0.3, 1, 3])
    ax.set_yticklabels(["0.1", "0.3", "1", "3"])
    ax.set_xlim(-0.6, 6.1)
    ax.set_xticks([0, 1, 2, 3.5, 4.5, 5.5])
    ax.set_xticklabels(["C", "G", "V", "C", "G", "V"])
    tr = ax.get_xaxis_transform()
    ax.text(1, -0.30, "U recording", ha="center", va="top", fontsize=5.0, color=S.TH["ink"], transform=tr)
    ax.text(4.5, -0.30, "I recording", ha="center", va="top", fontsize=5.0, color=S.TH["ink"], transform=tr)
    ax.text(2.75, -0.50, "C, G = ORB-SLAM3 CPU, GPU;  V = VINS", ha="center", va="top", fontsize=4.6,
            color=S.TH["ink2"], transform=tr)
    S.label(ax, y="ATE [m]")
    S.save(fig, "r3a_repeats.png")

    # ---- rule 2: rotation, and the numbers for rule 3
    fig, bx = card()
    S.style(bx)
    rows = []
    for ds in BAGS:
        g = vm.load_gt([c for c in vm.gt_candidates(ds) if os.path.exists(c)][0])
        for run in RUNS:
            x, e = rotation_series(ds, run, g)
            bx.plot(x, e, color=S.ORB, lw=0.6, alpha=0.5)
            rows.append((ds[-3:], run, float(np.sqrt(np.mean(e ** 2))), float(np.median(e)), float(np.mean(e > 45)),
                         *path_excess(ds, run, g)))
    bx.set_xlim(0, 10)
    bx.set_ylim(-5, 100)
    bx.set_yticks([0, 45, 90])
    S.label(bx, x="seconds since the estimate starts", y="rotation error [deg]")
    S.save(fig, "r3b_rotation.png")
    R = pd.DataFrame(rows, columns=["bag", "run", "rms", "med", "pre", "reported", "after", "at1", "spike", "local"])
    print(R.round(3).to_string())
    ok = ~((R.bag == "001") & R.run.str.startswith("loop"))
    print("rotation RMS %.1f-%.1f deg; share of samples > 45 deg %.1f-%.1f %%"
          % (R.rms.min(), R.rms.max(), 100 * R.pre.min(), 100 * R.pre.max()))
    print("path: reported %.1f-%.1f %% (excl.), spike %.1f-%.1f m, every second %.2f-%.2f %%, local scale %.4f-%.4f"
          % (R.reported[ok].min(), R.reported[ok].max(), R.spike.min(), R.spike.max(),
             R.at1.min(), R.at1.max(), R.local.min(), R.local.max()))

    # ---- rule 3: path excess, three ways
    fig, cx = card()
    S.style(cx, grid="y")
    for _, r in R[ok].iterrows():
        cx.plot([0, 1, 2], [r.reported, r.after, r.at1], "-o", color=S.ORB, lw=0.6, ms=1.6, alpha=0.6, mec="none")
    cx.axhline(0, color=S.TH["ink2"], lw=0.6)
    cx.set_xticks([0, 1, 2])
    cx.set_xticklabels(["as\nscored", "spike\nremoved", "1 sample\nper second"])
    cx.set_xlim(-0.3, 2.3)
    cx.set_ylim(-1, 6)
    S.label(cx, y="path too long [%]")
    S.save(fig, "r3c_path.png")


if __name__ == "__main__":
    main()
