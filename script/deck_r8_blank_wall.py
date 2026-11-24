#!/usr/bin/env python3
"""Deck slide R8: at a blank wall VINS-Fusion runs out of features and drifts;
ORB-SLAM3 passes the same wall on the same recording.

    python3 script/deck_r8_blank_wall.py

realistic_002 (U layout, traffic).  Series from script/exp1_local_error_drivers.py:
  VINS-Fusion, two runs of the same recording (cpu, cpu_001): tracked features
  (frontend.csv feature_count, ceiling 150) and start-anchored planar error.
  ORB-SLAM3, closer off, GPU extractor: tracked map points
  (tracking_frontend.csv map_matches_inliers) and the same error measure.
(a) shows each front end's tracking as a share of its own normal level: VINS of
its 150-feature ceiling, ORB of its median over 5-60 s; the two counts are not
the same quantity, so only the shape is compared.
"""
import os
import sys

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import deck_style as S

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp1_local_error_drivers as L  # noqa: E402

DS = "dataset_allsensor_realistic_002"


def main():
    gt = L.gt_for(DS)
    v1, v2 = L.vins_growth(DS, "cpu", gt), L.vins_growth(DS, "cpu_001", gt)
    a, b, mn, step = v1["episodes"][0]
    print("VINS dip %.1f-%.1f s, min %.0f features, error step %+.3f m; repeat step %+.3f m"
          % (a, b, mn, step, v2["episodes"][0][3]))
    d = pd.read_csv(os.path.join(S.ROOT, "output/output_orb/simulation", DS,
                                 "noloopclosing_gpu/tracking_frontend.csv"))
    to = d.timestamp_ns.values / 1e9 - gt["t0"]
    po = d.map_matches_inliers.values.astype(float)
    med = np.median(po[(to > 5) & (to < 60)])
    te_o, e_o = L.planar(DS, "output_orb", "noloopclosing_gpu", gt)
    w = (to > a - 2) & (to < b + 2)
    print("ORB median points %.0f, min near the wall %.0f (%.0f %%); error %.3f -> %.3f m"
          % (med, po[w].min(), 100 * po[w].min() / med, np.interp(a, te_o, e_o), np.interp(b, te_o, e_o)))

    fig, (ax, bx) = S.figure(2, 1, gridspec_kw=dict(hspace=0.62))
    for x in (ax, bx):
        S.style(x)
        x.axvspan(a, b, color=S.TH["ink2"], alpha=0.12, lw=0, zorder=0)
        x.set_xlim(10, 40)
    for r, ls in ((v1, "-"), (v2, (0, (2.5, 1.2)))):
        ax.plot(r["tf"], 100 * r["fc"] / 150.0, color=S.VINS, lw=0.9, ls=ls)
        bx.plot(r["te"], r["e"], color=S.VINS, lw=0.9, ls=ls)
    k = np.ones(5) / 5
    ax.plot(to[2:-2], 100 * np.convolve(po, k, "valid") / med, color=S.ORB, lw=0.9)
    bx.plot(te_o, e_o, color=S.ORB, lw=0.9)
    ax.set_ylim(0, 150)
    ax.set_yticks([0, 50, 100, 150])
    ax.text((a + b) / 2, 6, "camera faces\na blank wall", ha="center", va="bottom", fontsize=4.9,
            color=S.TH["ink2"], linespacing=1.0)
    S.label(ax, y="still tracked [% of normal]")
    S.title(ax, "(a) how much each front end can still follow")
    bx.set_ylim(0, 0.7)
    S.label(bx, x="seconds since the recording starts", y="position error [m]")
    S.title(bx, "(b) position error, same runs")
    S.legend(bx, [Line2D([], [], color=S.VINS, lw=1.0, label="VINS-Fusion, run 1"),
                  Line2D([], [], color=S.VINS, lw=1.0, ls=(0, (2.5, 1.2)), label="VINS-Fusion, run 2"),
                  Line2D([], [], color=S.ORB, lw=1.0, label="ORB-SLAM3")], loc="upper right")
    S.save(fig, "r8_blank_wall.png")


if __name__ == "__main__":
    main()
