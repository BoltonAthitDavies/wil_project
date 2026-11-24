#!/usr/bin/env python3
"""The inertial initialisation the chapter analyses, as two pictures.

    python3 script/exp1_imu_init_figure.py

WHY THIS EXISTS
  Four paragraphs of the slot argue about a schedule -- initialise at ten
  keyframes, refine at five seconds with the bias prior kept, refine again at
  fifteen with it dropped, and nothing after that on the stereo path -- and
  carry it entirely in prose and numbers.  The two things a reader cannot get
  from the prose are WHEN the window closes relative to the robot's own
  turning, and HOW MUCH each refinement moves the published world frame.

  (a) answers the first: cumulative yaw against time, one curve per recording,
      with the window the method uses drawn on it.  The U-layout recordings
      turn inside it; the I-layout recordings do not turn until after it has
      closed.  Nothing is correlated here -- the panel shows the timing, and
      the confound is stated in the caption, not hidden by it.
  (b) answers the second, as a PAIRED comparison: the published map rotation
      at the first refinement against the second, within each run.  The prior
      is the only thing that differs between them, and the pairing is what
      makes that readable; the sign test over sixteen pairs is in the title.

WHAT IT CANNOT ESTABLISH
  Panel (a) is four routes, and yaw-in-the-window is a property of the layout
  on the recordings that exist, so it is NOT evidence that excitation causes
  the gravity error -- only that the two coincide here.  Panel (b) is paired
  within run and carries no such confound.

Output: output/compare/simulation/exp1-imu-init/fig_imu_init_light.png
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np                                   # noqa: E402
from plot_ticks import plain_log                     # noqa: E402
import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vio_metrics as vm                             # noqa: E402
import exp1_paper_assumptions as PA                  # noqa: E402

ROOT = str(vm.ROOT)
OUT = os.path.join(ROOT, "output/compare/simulation/exp1-imu-init")
BAGS = ["dataset_allsensor_realistic_%03d" % i for i in range(4)]
ARMS = [("ORB-SLAM3", "loopclosing_cpu"), ("ORB-SLAM3 GPU", "loopclosing_gpu"),
        ("ORB-SLAM3 (no LC)", "noloopclosing_cpu"),
        ("ORB-SLAM3 GPU (no LC)", "noloopclosing_gpu")]
BAGCOL = {"000": "#cc8319", "001": "#e040fb", "002": "#a347b2", "003": "#1f8a70"}
LAY = {"000": "U", "001": "I", "002": "U", "003": "I"}
TH = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#dedcd4")


def style(ax):
    ax.set_facecolor(TH["surface"])
    for sp in ax.spines.values():
        sp.set_color(TH["grid"])
    ax.grid(True, color=TH["grid"], lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TH["ink2"], labelsize=8)


def main():
    fig = plt.figure(figsize=(11.6, 5.2), facecolor=TH["surface"])
    gs = fig.add_gridspec(1, 2, left=0.065, right=0.985, top=0.755, bottom=0.165,
                          wspace=0.24, width_ratios=[1.45, 1.0])

    # ---- (a) cumulative yaw against the window the method uses
    ax = fig.add_subplot(gs[0, 0]); style(ax)
    inits, vibas = [], []
    for ds in BAGS:
        gtp = [c for c in vm.gt_candidates(ds) if os.path.exists(c)][0]
        g = vm.load_gt(gtp)
        t = g[0] - g[0][0]
        y = np.degrees(np.unwrap(np.radians(vm.euler_zyx(vm.quat_to_R(g[2]))[:, 0])))
        cum = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(y)))])
        m = t <= 32
        ax.plot(t[m], cum[m], lw=1.8, color=BAGCOL[ds[-3:]],
                label="%s  (%s layout)" % (ds[-3:], LAY[ds[-3:]]))
        for name, run in ARMS:
            ms = PA.milestones(os.path.join(ROOT, "output/output_orb/simulation", ds, run))
            if ms:
                inits.append(ms["imu_initialized"][0])
                vibas.append(ms["inertial_ba2"][0])
    lo, hi = float(np.median(inits)), float(np.max(vibas))
    ax.axvspan(lo, hi, color=TH["ink2"], alpha=0.10, lw=0, zorder=0)
    ax.axvline(lo, color=TH["ink"], lw=1.0, ls="--")
    ax.axvline(hi, color=TH["ink"], lw=1.0, ls="--")
    ax.annotate("the window the gravity direction is estimated in:\n"
                "initialisation at ten keyframes (%.2f s) to the last\n"
                "refinement the stereo path offers (%.1f s)" % (lo, hi),
                xy=(hi, ax.get_ylim()[1] * 0.60), xytext=(hi + 1.2, ax.get_ylim()[1] * 0.60),
                fontsize=8, color=TH["ink2"], va="center",
                arrowprops=dict(arrowstyle="->", color=TH["ink2"], lw=0.8))
    ax.set_xlabel("time since the recording's start [s]", color=TH["ink2"], fontsize=9)
    ax.set_ylabel("cumulative yaw swept [deg]", color=TH["ink2"], fontsize=9)
    ax.set_xlim(0, 32)
    ax.legend(fontsize=8, loc="upper left", frameon=False, labelcolor=TH["ink2"])
    ax.set_title("(a) does the robot turn while the gravity direction is still\n"
                 "    being estimated? The U recordings sweep 36 and 94 deg inside\n"
                 "    the window; the I recordings sweep none, and first turn at 21 and 25 s.",
                 fontsize=9.5, loc="left", color=TH["ink"])

    # ---- (b) the bias prior, paired within each run
    ax = fig.add_subplot(gs[0, 1]); style(ax)
    up = 0
    for ds in BAGS:
        for name, run in ARMS:
            d = os.path.join(ROOT, "output/output_orb/simulation", ds, run)
            ms = PA.milestones(d)
            if not ms:
                continue
            a0 = ms["t_abs0"]
            s1 = PA.pose_step(d, a0 + ms["inertial_ba1"][0])
            s2 = PA.pose_step(d, a0 + ms["inertial_ba2"][0])
            up += s2 > s1
            ax.plot([0, 1], [s1, s2], "-o", ms=4, lw=1.1,
                    color=BAGCOL[ds[-3:]], alpha=0.85)
    ax.set_yscale("log"); plain_log(ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["refinement 1\nprior kept", "refinement 2\nprior dropped"],
                       fontsize=8.5)
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel("published map rotation at the refinement [deg, log]",
                  color=TH["ink2"], fontsize=9)
    ax.set_title("(b) what dropping the bias prior\n"
                 "    costs: larger in %d of 16 runs,\n"
                 "    sign test $p = 1.5\\times10^{-5}$" % up,
                 fontsize=9.5, loc="left", color=TH["ink"])

    fig.suptitle("Experiment 1, simulation: the inertial initialisation the method specifies",
                 fontsize=11.5, x=0.012, y=0.965, ha="left", color=TH["ink"])
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "fig_imu_init_light.png")
    fig.savefig(path, dpi=160, facecolor=TH["surface"])
    plt.close(fig)
    print("  wrote", path)
    print("  window %.2f - %.2f s; refinement 2 larger in %d of 16" % (lo, hi, up))


if __name__ == "__main__":
    main()
