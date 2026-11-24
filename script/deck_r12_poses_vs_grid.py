#!/usr/bin/env python3
"""Deck slide R12: the four realistic map graphs put their poses in the right
place; the occupancy grids built on them fail both map targets.

    python3 script/deck_r12_poses_vs_grid.py

From output/compare/simulation/exp4-realistic-maps/:
  graph_vs_gt.csv          (script/eval_map_graph.py) every graph node, placed
                           with the map's fitted pose, against the ground truth
                           of the recording that built it: placed_rms, path_gt
  realistic_map_metrics.csv (script/eval_realistic_maps.py) the exported grid
                           against the world, probed with a recording that did
                           not build it: precision, false_obstacle
Targets are the report's: precision >= 90 %, false obstacles <= 1 %.
"""
import csv
import os

import numpy as np

import deck_style as S

D = os.path.join(S.ROOT, "output/compare/simulation/exp4-realistic-maps")
MAPS = [("dataset_allsensor_realistic_map", "U prior"), ("map_realistic", "U alt"),
        ("dataset_allsensor_realistic_01_map", "I prior"), ("map_realistic_01", "I alt")]
# "prior" = the map Experiment 2B relocalises on (script/exp4_protocol.yaml):
#           1 Hz nodes, Grid/RangeMax 10 m, Grid/MaxGroundHeight 0.10 m.
# "alt"   = the other build of the same mapping recording: 4 Hz, 8 m, 0.29 m.
# Named by role, never by build date.


def main():
    g = {r["key"]: r for r in csv.DictReader(open(os.path.join(D, "graph_vs_gt.csv")))}
    m = {r["key"]: r for r in csv.DictReader(open(os.path.join(D, "realistic_map_metrics.csv")))}
    rms = [float(g[k]["placed_rms"]) for k, _ in MAPS]
    route = [float(g[k]["path_gt"]) for k, _ in MAPS]
    prec = [100 * float(m[k]["precision"]) for k, _ in MAPS]
    fo = [100 * float(m[k]["false_obstacle"]) for k, _ in MAPS]
    for (k, n), a, b, c, d in zip(MAPS, rms, route, prec, fo):
        print("%s: node error %.3f m over %.0f m | grid precision %.1f %% | false obstacles %.2f %%" % (n, a, b, c, d))

    # the same maps re-assembled with the cells under the mapping path cleared
    # (GridGlobal/FootprintRadius 0.5 m; script/eval_footprint_clearing.py)
    fp = {(r["map"], float(r["radius"])): r for r in csv.DictReader(open(os.path.join(D, "footprint_clearing.csv")))}
    prec_fp = [100 * float(fp[(n, 0.5)]["precision"]) for _, n in MAPS]
    fo_fp = [100 * float(fp[(n, 0.5)]["false_obstacle"]) for _, n in MAPS]
    for (k, n), c, d in zip(MAPS, prec_fp, fo_fp):
        print("%s with footprint clearing 0.5 m: precision %.1f %% | route blocked %.2f %%" % (n, c, d))

    fig, axes = S.figure(3, 1, gridspec_kw=dict(hspace=0.95))
    names = [n for _, n in MAPS]
    x = np.arange(4)
    a = axes[0]
    S.style(a, grid="y")
    a.bar(x, rms, 0.55, color=S.TH["good"])
    for i, y in zip(x, rms):
        a.text(i, y + 0.015, "%.2f" % y, ha="center", va="bottom", fontsize=4.9, color=S.TH["ink"])
    a.set_ylim(0, 0.56)
    a.set_xticks(x); a.set_xticklabels([])
    S.label(a, y="error [m]")
    S.title(a, "(a) graph poses vs truth (routes %.0f\u2013%.0f m)" % (min(route), max(route)))
    w = 0.38
    for a, (v0, v1, yl, tgt, top, ttl, fmt) in zip(axes[1:], [
            (prec, prec_fp, "right [%]", 90, 104, "(b) occupied grid cells that are real (target \u2265 90 %)", "%.0f"),
            (fo, fo_fp, "blocked [%]", 1, 21, "(c) poses of a real route the map blocks (target \u2264 1 %)", "%.1f")]):
        S.style(a, grid="y")
        a.bar(x - w / 2, v0, w, color=S.TH["bad"])
        a.bar(x + w / 2, v1, w, color=S.FP)
        for i in x:
            for dx, y in ((-w / 2, v0[i]), (w / 2, v1[i])):
                if top > 100:
                    a.text(i + dx, y - 2, fmt % y, ha="center", va="top", fontsize=4.4, color="white")
                else:
                    a.text(i + dx, y + 0.3, fmt % y, ha="center", va="bottom", fontsize=4.4, color=S.TH["ink"])
        a.axhline(tgt, color=S.TH["ink"], lw=0.8, ls="--")
        a.set_ylim(0, top)
        a.set_xticks(x)
        a.set_xticklabels(names if a is axes[-1] else [])
        S.label(a, y=yl)
        S.title(a, ttl)
    from matplotlib.patches import Patch
    S.legend(axes[2], [Patch(color=S.TH["bad"], label="as built"),
                       Patch(color=S.FP, label="cells under the mapping path cleared")],
             loc="upper left", ncol=2, frameon=False, columnspacing=0.8)
    S.save(fig, "r12_poses_vs_grid.png")


if __name__ == "__main__":
    main()
