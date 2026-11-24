#!/usr/bin/env python3
"""Deck slide R1: swapping ORB-SLAM3's feature extractor (CPU -> GPU) changes
how many frames are processed, not how accurate the result is.

    python3 script/deck_r1_frontend.py

Pairs hold the recording and the loop-closer state fixed and swap only the
extractor.
  simulation: realistic_000-003 x closer on/off = 8 pairs,
              output/compare/simulation/exp-realistic-ab/runs_metrics.csv
              (of_bag = share of the bag's frames processed, ate_rms)
  robot:      realv7_000-002 x closer on/off = 6 pairs, patch-15 runs scored
              raw with the bags mounted,
              output/compare/real/realv7-insights/patch15_runs.csv
              (kept = share of frames processed, ate_new)
realv6_000 is left out: its tracker export fails the validity rules.
"""
import csv
import os

from matplotlib.lines import Line2D

import deck_style as S

SIM_CSV = os.path.join(S.ROOT, "output/compare/simulation/exp-realistic-ab/runs_metrics.csv")
REAL_CSV = os.path.join(S.ROOT, "output/compare/real/realv7-insights/patch15_runs.csv")


def sim_pairs():
    rows = [r for r in csv.DictReader(open(SIM_CSV))
            if "realistic_00" in r["dataset"] and r["tree"] == "output_orb"]
    out = []
    for ds in sorted({r["dataset"] for r in rows}):
        for lc in ("loopclosing", "noloopclosing"):
            c = [r for r in rows if r["dataset"] == ds and r["run"] == lc + "_cpu"]
            g = [r for r in rows if r["dataset"] == ds and r["run"] == lc + "_gpu"]
            if c and g:
                out.append(((100 * float(c[0]["of_bag"]), 100 * float(g[0]["of_bag"])),
                            (float(c[0]["ate_rms"]), float(g[0]["ate_rms"]))))
    return out


def robot_pairs():
    rows = [r for r in csv.DictReader(open(REAL_CSV)) if "realv7" in r["ds"]]
    out = []
    for ds in sorted({r["ds"] for r in rows}):
        for lc in ("lc", "nolc"):
            c = [r for r in rows if r["ds"] == ds and r["key"] == lc + "_cpu"][0]
            g = [r for r in rows if r["ds"] == ds and r["key"] == lc + "_gpu"][0]
            out.append(((100 * float(c["kept"]), 100 * float(g["kept"])),
                        (float(c["ate_new"]), float(g["ate_new"]))))
    return out


def main():
    sp, rp = sim_pairs(), robot_pairs()
    print("sim", len(sp), "pairs:", [(round(a[0]), round(a[1]), round(b[0], 3), round(b[1], 3)) for a, b in sp])
    print("robot", len(rp), "pairs:", [(round(a[0]), round(a[1]), round(b[0], 3), round(b[1], 3)) for a, b in rp])
    print("sim ATE up/down:", sum(b[1] > b[0] for _, b in sp), sum(b[1] < b[0] for _, b in sp))
    print("robot max |dATE|:", max(abs(b[1] - b[0]) for _, b in rp))

    fig, (ax, bx) = S.figure(1, 2, gridspec_kw=dict(wspace=0.55))
    for a, (k, lab) in zip((ax, bx), ((0, "frames processed [%]"), (1, "position error, ATE [m]"))):
        S.style(a, grid="y")
        for pairs, col in ((sp, S.SIM), (rp, S.ROBOT)):
            for p in pairs:
                a.plot([0, 1], p[k], "-o", color=col, lw=0.8, ms=2.2, alpha=0.85, mec="none")
        a.set_xticks([0, 1])
        a.set_xticklabels(["CPU", "GPU"])
        a.set_xlim(-0.3, 1.3)
        S.label(a, y=lab)
    ax.set_ylim(0, 105)
    bx.set_yscale("log")
    bx.set_ylim(0.1, 10)
    bx.set_yticks([0.1, 0.3, 1, 3, 10])
    bx.set_yticklabels(["0.1", "0.3", "1", "3", "10"])
    S.title(ax, "(a) frames")
    S.title(bx, "(b) accuracy")
    S.legend(ax, [Line2D([], [], color=S.SIM, lw=1.0, marker="o", ms=2.2, label="sim, 8 pairs"),
                  Line2D([], [], color=S.ROBOT, lw=1.0, marker="o", ms=2.2, label="robot, 6 pairs")],
             loc="lower right")
    S.save(fig, "r1_frontend.png")


if __name__ == "__main__":
    main()
