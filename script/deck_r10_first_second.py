#!/usr/bin/env python3
"""Figure for slide R10 of the update_181126 deck: the front end's first
second damages both RTAB-Map layers.

    python3 script/deck_r10_first_second.py

  (a) localise.  The rotation part of the map->odom correction RTAB-Map
      publishes, over simulation time, for the aisle-sweep variants J and M
      (map/dataset_allsensor_realistic_map/localize_sweep_{J,M}N.csv, kind=tf).
      J and M differ in ONE setting: M holds RTAB-Map back 5 s
      (localize_delay_s 5.0).  The right correction is ~0 deg on this prior.
  (b) map.  For each of the four closure-free realistic maps, the share of
      phantom cells (occupied in the map, > 0.50 m from any real structure)
      first observed by graph nodes 1-5: over all phantoms, and over the
      phantoms that block a probe route.  From
      output/compare/simulation/exp4-insights/insights_realistic.json
      (script/exp4_insights.py --realistic).

Output: docs/report/insights/update_181126/fig/r10_first_second.png
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs/report/insights/update_181126/fig/r10_first_second.png")
SWEEP = os.path.join(ROOT, "map/dataset_allsensor_realistic_map")
INS = os.path.join(ROOT, "output/compare/simulation/exp4-insights/insights_realistic.json")

TH = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#dedcd4",
          bad="#c0392b", good="#1e8449", all="#b9b5a8")
MAPS = [("dataset_allsensor_realistic_map", "U prior"), ("map_realistic", "U alt"),
        ("dataset_allsensor_realistic_01_map", "I prior"), ("map_realistic_01", "I alt")]
# "prior" = the map Experiment 2B relocalises on (script/exp4_protocol.yaml):
#           1 Hz nodes, Grid/RangeMax 10 m, Grid/MaxGroundHeight 0.10 m.
# "alt"   = the other build of the same mapping recording: 4 Hz, 8 m, 0.29 m.
# Named by role, never by build date.


def series(run):
    rows = [r for r in csv.DictReader(open(os.path.join(SWEEP, f"localize_sweep_{run}.csv")))
            if r["kind"] == "tf" and r["corr_yaw_deg"] != ""]
    return [float(r["stamp"]) for r in rows], [float(r["corr_yaw_deg"]) for r in rows]


def main():
    plt.rcParams.update({"font.size": 6.0, "font.family": "DejaVu Sans"})
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(2.55, 2.3),
                                 gridspec_kw=dict(height_ratios=[1.2, 1.0], hspace=0.72))
    fig.patch.set_facecolor(TH["surface"])
    for a in (ax, bx):
        a.set_facecolor(TH["surface"])
        for s in a.spines.values():
            s.set_color(TH["grid"])
        a.tick_params(colors=TH["ink2"], length=2, pad=1.5)

    # ---- (a) correction yaw over time
    for run in ("J1", "J2", "J3"):
        t, y = series(run)
        ax.step(t, y, where="post", color=TH["bad"], lw=0.9, alpha=0.85)
        print(run, "starts", t[0], "max |yaw|", max(abs(v) for v in y))
    for run in ("M1", "M2", "M3"):
        t, y = series(run)
        ax.step(t, y, where="post", color=TH["good"], lw=0.9, alpha=0.9)
        print(run, "starts", t[0], "max |yaw|", max(abs(v) for v in y))
    ax.axhline(0, color=TH["ink2"], lw=0.5, ls=":")
    ax.set_xlim(14, 46)
    ax.set_ylim(-180, 180)
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.set_xlabel("simulation time [s]", color=TH["ink2"], labelpad=1.0)
    ax.set_ylabel("correction\nrotation [deg]", color=TH["ink2"], labelpad=1.0)
    ax.grid(color=TH["grid"], lw=0.4)
    ax.annotate("74° wrong, held 16 s", xy=(25, 74), xytext=(27.5, 128), color=TH["bad"],
                fontsize=5.2, arrowprops=dict(arrowstyle="-", color=TH["bad"], lw=0.6))
    hs = [Line2D([], [], color=TH["bad"], lw=1.0, label="localising from the start (J, 3 runs)"),
          Line2D([], [], color=TH["good"], lw=1.0, label="same, held back 5 s (M, 3 runs)")]
    ax.legend(handles=hs, loc="lower right", frameon=True, fancybox=False, framealpha=1.0,
              edgecolor=TH["grid"], fontsize=4.8, handlelength=1.2, borderpad=0.3, labelspacing=0.25,
              labelcolor=TH["ink2"])
    ax.set_title("(a) localise: the correction RTAB-Map publishes", loc="left",
                 fontsize=6.0, color=TH["ink"], pad=3)

    # ---- (b) where the phantom cells come from
    d = json.load(open(INS))["blocked_on"]
    xs = range(len(MAPS))
    allv = [100 * d[k]["all_phantom_first_le5"] for k, _ in MAPS]
    blk = [100 * d[k]["blocking_first_le5"] for k, _ in MAPS]
    w = 0.36
    bx.bar([x - w / 2 for x in xs], allv, w, color=TH["all"])
    bx.bar([x + w / 2 for x in xs], blk, w, color=TH["bad"])
    for x, a_, b in zip(xs, allv, blk):
        bx.text(x - w / 2, a_ + 3, f"{a_:.0f}", ha="center", va="bottom", fontsize=4.8, color=TH["ink2"])
        bx.text(x + w / 2, b + 3, f"{b:.0f}", ha="center", va="bottom", fontsize=4.8, color=TH["bad"])
    print("all", [round(v) for v in allv], "blocking", [round(v) for v in blk])
    bx.set_xticks(list(xs))
    bx.set_xticklabels([n for _, n in MAPS])
    bx.set_ylim(0, 150)
    bx.set_yticks([0, 50, 100])
    bx.set_ylabel("% laid by\nnodes 1–5", color=TH["ink2"], labelpad=1.0)
    bx.grid(axis="y", color=TH["grid"], lw=0.4)
    bx.legend(handles=[Patch(color=TH["all"], label="all phantom cells"),
                       Patch(color=TH["bad"], label="phantoms blocking the route")],
              loc="upper center", ncol=2, frameon=False, fontsize=4.8, handlelength=1.0,
              columnspacing=0.8, labelcolor=TH["ink2"], bbox_to_anchor=(0.5, 1.02))
    bx.set_title("(b) map: who laid the false obstacles (4 maps)", loc="left",
                 fontsize=6.0, color=TH["ink"], pad=3)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor=TH["surface"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
