#!/usr/bin/env python3
"""Deck slide R4: the one moving object that hurts -- a pallet jack close
ahead of the robot, driving the same way.

    python3 script/deck_r4_jack_ahead.py

realistic_003 (I layout, traffic).  Mover tracks and the camera-view test are
those of script/exp1_movers_in_view.py (movers_003.csv, the simulator's
ground-truth poses of the models that move).
  (a) distance from the camera to the moving pallet jack, at each half second
      it is inside the left camera's view (camera depth z of its centre).
  (b) per-second translational error (RPE over 1 s, rolling median over
      0.5 s, log axis) of every run on the recording: four ORB-SLAM3, one
      VINS-Fusion.  Episode band = 31.5-35.6 s as in the Experiment 1 note.
"""
import os
import sys

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import deck_style as S

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp1_local_error_drivers as L  # noqa: E402
import exp1_movers_in_view as MV      # noqa: E402

DS = MV.DS["003"]
EP = MV.EPISODE


def main():
    gt = L.gt_for(DS)
    jacks = {k: v for k, v in MV.movers("003").items() if "PalletJack" in k}
    ts = np.arange(20, 46, 0.5)
    dist = []
    for tg in ts:
        c = MV.covariates(jacks, gt, gt["t0"] + tg)
        dist.append(c[2] if c[0] and c[3] else np.nan)
    dist = np.array(dist)
    seen = ts[np.isfinite(dist)]
    pre = (ts < EP[0]) & np.isfinite(dist)
    print("jack moving and in view from %.1f s; before the band at %.1f-%.1f m; in the band min %.1f m"
          % (seen.min(), np.nanmin(dist[pre]), np.nanmax(dist[pre]),
             np.nanmin(dist[(ts >= EP[0]) & (ts <= EP[1])])))

    fig, (ax, bx) = S.figure(2, 1, gridspec_kw=dict(height_ratios=[0.8, 1.2], hspace=0.62))
    for a in (ax, bx):
        S.style(a)
        a.axvspan(*EP, color=S.TH["ink2"], alpha=0.12, lw=0, zorder=0)
        a.set_xlim(22, 44)
    ax.plot(ts, dist, "o-", color=S.TH["ink"], ms=1.8, lw=0.8)
    ax.set_ylim(0, 22)
    ax.set_yticks([0, 10, 20])
    S.label(ax, y="distance to\nthe jack [m]")
    S.title(ax, "(a) the moving pallet jack, while the camera sees it")

    runs = [(lbl, L.windows_orb(DS, run, gt), S.ORB) for lbl, run in MV.ORB]
    runs.append(("VINS", L.windows_vins(DS, "cpu", gt), S.VINS))
    for lbl, h, col in runs:
        tg = h.t.values - gt["t0"]
        m = (tg >= 20) & (tg <= 46)
        s = pd.Series(h.rpe.values[m], index=tg[m]).rolling(15, center=True, min_periods=5).median()
        bx.plot(s.index, s.values, lw=0.8, color=col, alpha=0.85)
        base = h.rpe[(tg >= 15) & (tg <= 28)].median()
        ep = h.rpe[(tg >= EP[0]) & (tg <= EP[1])].median()
        print("%-13s per-second error %.4f -> %.4f m (x%.1f)" % (lbl, base, ep, ep / base))
    bx.set_yscale("log")
    bx.set_ylim(0.004, 0.3)
    bx.set_yticks([0.01, 0.03, 0.1, 0.3])
    bx.set_yticklabels(["0.01", "0.03", "0.1", "0.3"])
    S.label(bx, x="seconds since the recording starts", y="error made in\neach second [m]")
    S.title(bx, "(b) the error each run makes per second")
    S.legend(bx, [Line2D([], [], color=S.ORB, lw=1.0, label="ORB-SLAM3 (4 runs)"),
                  Line2D([], [], color=S.VINS, lw=1.0, label="VINS-Fusion")], loc="upper left")
    S.save(fig, "r4_jack_ahead.png")


if __name__ == "__main__":
    main()
