#!/usr/bin/env python3
"""Deck slide R7: one false loop closure on realistic_001 (static, I layout).

    python3 script/deck_r7_false_closure.py

  (a) the two keyframes ORB-7 joined (GPU front end), as exported by
      script/exp1_false_closure_frames.py from loop_closing.csv + the bag:
      the same yellow box-stack, seen from opposite sides of the warehouse.
  (b) position error against time for the four ORB-SLAM3 runs on that
      recording: CPU/GPU extractor x loop closer on/off.  Each estimate is
      aligned to ground truth (rigid fit, Umeyama) on 5-70 s, BEFORE the
      closure, so the plot shows what the closure did rather than a fit that
      spreads it over the whole run.  The first 4 s (pre-initialisation frame)
      are dropped.  NOTE this differs from the ATE in the tables, which fits
      the whole run; the end-of-run figures here are therefore larger.
"""
import os
import sys

import numpy as np
from matplotlib.lines import Line2D
from PIL import Image

import deck_style as S

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vio_metrics as vm  # noqa: E402

DS = "dataset_allsensor_realistic_001"
RUNS = os.path.join(S.ROOT, "output/output_orb/simulation", DS)
IMG = os.path.join(S.ROOT, "output/compare/simulation/exp1-false-closure")
STY = {"loopclosing_cpu": (S.TH["bad"], "-", "closer on, CPU"),
       "loopclosing_gpu": (S.TH["bad"], (0, (2.5, 1.2)), "closer on, GPU"),
       "noloopclosing_cpu": (S.TH["good"], "-", "closer off, CPU"),
       "noloopclosing_gpu": (S.TH["good"], (0, (2.5, 1.2)), "closer off, GPU")}


def error_series(run, g):
    t, p, _, _ = vm.load(os.path.join(RUNS, run, "vio.csv"))
    m = (t > t[0] + 4) & (t >= g[0][0]) & (t <= g[0][-1])
    t, p = t[m], p[m]
    r = vm.resample(g[0], g[1], t)
    x = t - g[0][0]
    w = (x > 5) & (x < 70)
    R, tt = vm.umeyama(p[w], r[w])
    return x, np.linalg.norm(p @ R.T + tt - r, axis=1)


def main():
    import matplotlib.pyplot as plt
    S.setup()
    fig = plt.figure(figsize=(S.W, S.H))
    fig.patch.set_facecolor(S.TH["surface"])
    gs = fig.add_gridspec(2, 2, height_ratios=[0.62, 1.0], hspace=0.55, wspace=0.06)
    for k, (name, cap) in enumerate((("closure_loopclosing_gpu_current_kf272.jpg", "keyframe at 73 s"),
                                     ("closure_loopclosing_gpu_matched_kf104.jpg", "matched to: 17.7 m away"))):
        a = fig.add_subplot(gs[0, k])
        a.imshow(Image.open(os.path.join(IMG, name)))
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_color(S.TH["grid"])
        a.set_xlabel(cap, color=S.TH["ink2"], labelpad=1.5, fontsize=5.2)
        if k == 0:
            S.title(a, "(a) the two views ORB-7 called the same place")

    bx = fig.add_subplot(gs[1, :])
    S.style(bx)
    g = vm.load_gt([c for c in vm.gt_candidates(DS) if os.path.exists(c)][0])
    for run, (col, ls, _) in STY.items():
        x, e = error_series(run, g)
        bx.plot(x, e, color=col, ls=ls, lw=0.9)
        print(run, "error at 70 s %.2f m, at 75 s %.2f m, end %.2f m, end of run at %.0f s"
              % (np.interp(70, x, e), np.interp(75, x, e), e[-1], x[-1]))
    bx.axvline(73.2, color=S.TH["ink2"], lw=0.6, ls=":")
    bx.text(72.0, 16.5, "closure\n73.2 s", ha="right", va="top", fontsize=4.9, color=S.TH["ink2"],
            linespacing=1.0)
    bx.set_xlim(0, 100)
    bx.set_ylim(0, 18)
    S.label(bx, x="seconds since the recording starts", y="position error [m]")
    S.title(bx, "(b) error of the four runs on this recording")
    S.legend(bx, [Line2D([], [], color=c, ls=l, lw=1.0, label=n) for c, l, n in STY.values()],
             loc="upper left", ncol=1)
    S.save(fig, "r7_false_closure.png")


if __name__ == "__main__":
    main()
