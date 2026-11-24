#!/usr/bin/env python3
"""Deck slide R5: ORB-SLAM3 finds gravity only while the robot turns; a route
that drives straight through the estimation window keeps a tilted 'down', and
the tilt turns distance into height.

    python3 script/deck_r5_gravity.py

  (a) cumulative yaw the robot sweeps (simulator ground truth) against time, one
      line per recording, with the window ORB-6 estimates gravity in: IMU
      initialisation (median over runs) to the second inertial refinement (latest
      over runs), milestones from local_mapping.csv as in exp1_imu_init_figure.py.
  (b) the HEIGHT of ORB-SLAM3's estimate against time, loop closer disabled, both
      front ends (CPU and GPU extractor), so nothing pulls a tilt back.  The floor
      is flat: the true height is constant.  The first 4 s are cut: they hold the
      pre-initialisation frame (see R3).  realistic_003 with the closer active is
      drawn dashed.
"""
import os
import sys

import numpy as np
from matplotlib.lines import Line2D

import deck_style as S

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vio_metrics as vm                 # noqa: E402
import exp1_paper_assumptions as PA      # noqa: E402

RUNS = os.path.join(S.ROOT, "output/output_orb/simulation")
BAGS = ["dataset_allsensor_realistic_%03d" % i for i in range(4)]
LAY = {"000": "U", "001": "I", "002": "U", "003": "I"}
COL = {"000": "#1f7a8c", "002": "#7fb7c4", "001": "#c0392b", "003": "#e67e22"}


def vio(ds, run):
    V = np.loadtxt(os.path.join(RUNS, ds, run, "vio.csv"), delimiter=",")
    return (V[:, 0] - V[0, 0]) / 1e9, V[:, 3]


def main():
    fig, (ax, bx) = S.figure(2, 1, gridspec_kw=dict(height_ratios=[1.0, 1.0], hspace=0.72))
    S.style(ax)
    S.style(bx)

    inits, ba2 = [], []
    for ds in BAGS:
        g = vm.load_gt([c for c in vm.gt_candidates(ds) if os.path.exists(c)][0])
        t = g[0] - g[0][0]
        y = np.degrees(np.unwrap(np.radians(vm.euler_zyx(vm.quat_to_R(g[2]))[:, 0])))
        cum = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(y)))])
        m = t <= 32
        ax.plot(t[m], cum[m], lw=1.0, color=COL[ds[-3:]])
        for run in ("loopclosing_cpu", "loopclosing_gpu", "noloopclosing_cpu", "noloopclosing_gpu"):
            ms = PA.milestones(os.path.join(RUNS, ds, run))
            if ms:
                inits.append(ms["imu_initialized"][0])
                ba2.append(ms["inertial_ba2"][0])
        lo_hi = None
    lo, hi = float(np.median(inits)), float(np.max(ba2))
    for ds in BAGS:
        g = vm.load_gt([c for c in vm.gt_candidates(ds) if os.path.exists(c)][0])
        t = g[0] - g[0][0]
        y = np.degrees(np.unwrap(np.radians(vm.euler_zyx(vm.quat_to_R(g[2]))[:, 0])))
        cum = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(y)))])
        print(ds[-3:], LAY[ds[-3:]], "yaw swept inside window: %.0f deg" % (np.interp(hi, t, cum) - np.interp(lo, t, cum)))
    ax.axvspan(lo, hi, color=S.TH["ink2"], alpha=0.12, lw=0, zorder=0)
    ax.text((lo + hi) / 2, 205, "gravity is\nestimated here", ha="center", va="top", fontsize=4.9,
            color=S.TH["ink2"], linespacing=1.0)
    ax.set_xlim(0, 32)
    ax.set_ylim(-8, 215)
    ax.set_yticks([0, 100, 200])
    S.label(ax, x="seconds since the recording starts", y="turned so far [deg]")
    S.title(ax, "(a) how much the robot has turned")
    print("window %.2f-%.1f s" % (lo, hi))

    for ds in BAGS:
        for run in ("noloopclosing_cpu", "noloopclosing_gpu"):
            t, z = vio(ds, run)
            m = t > 4
            bx.plot(t[m], z[m], lw=0.9, color=COL[ds[-3:]])
            print(ds[-3:], run, "height at end %.2f m" % z[-1])
    for run in ("loopclosing_cpu", "loopclosing_gpu"):
        t, z = vio(BAGS[3], run)
        m = t > 4
        bx.plot(t[m], z[m], lw=0.8, ls=(0, (2.5, 1.5)), color=COL["003"])
        print("003", run, "height at end %.2f m, max %.2f" % (z[-1], z[m].max()))
    bx.axhline(0, color=S.TH["ink2"], lw=0.5, ls=":")
    bx.set_xlim(0, 125)
    bx.set_ylim(-2.3, 9.0)
    bx.set_yticks([0, 2.5, 5])
    S.label(bx, x="seconds since the recording starts", y="estimated height [m]")
    S.title(bx, "(b) where the estimate thinks the floor is")
    S.legend(bx, [Line2D([], [], color=COL[k], lw=1.2, label=f"{k} ({LAY[k]})") for k in ("000", "002", "001", "003")]
             + [Line2D([], [], color=COL["003"], lw=0.9, ls=(0, (2.5, 1.5)), label="003, closer on")],
             loc="upper left", ncol=2, columnspacing=0.8)
    S.save(fig, "r5_gravity.png")


if __name__ == "__main__":
    main()
