#!/usr/bin/env python3
"""The CPU/GPU front-end comparison as the paired experiment it actually is.

    python3 script/exp1_frontend_pairs.py

WHY THIS EXISTS
  The slot claims the front end changes throughput and not accuracy.  That
  claim rests on eight PAIRS -- the same recording, the same loop-closer
  state, the extractor swapped -- and a table of sixteen numbers does not
  show a reader that the throughput shift is consistent while the accuracy
  shift is not.  A paired slope chart does: eight lines that all run the same
  way in one panel, and eight that run both ways in the other.

  This is deliberately NOT a scatter of accuracy against cost pooled over
  twenty runs.  That figure existed, showed no relationship because the
  accuracy axis was dominated by two identified failure modes, and was
  removed.  Pairing is what makes the comparison legitimate: each line holds
  the recording and the closer state fixed and varies one thing.

WHAT IT CANNOT ESTABLISH
  One run per cell.  ORB-SLAM3's spread over identical repeats reaches
  0.277 m on these recordings, wider than five of the eight pairs' gaps, so
  an individual line is not evidence.  What the panel supports is the
  CONTRAST between the two panels: a shift that every pair agrees on against
  one that they do not.

Output: output/compare/simulation/exp1-frontend-pairs/fig_frontend_pairs_light.png
"""

import csv
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np                                   # noqa: E402
import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "output/compare/simulation/exp-realistic-ab/runs_metrics.csv")
OUT = os.path.join(ROOT, "output/compare/simulation/exp1-frontend-pairs")
TH = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#dedcd4")
LC = {"loopclosing": "closer active", "noloopclosing": "closer disabled"}
COL = {"closer active": "#e040fb", "closer disabled": "#a347b2"}


def style(ax):
    ax.set_facecolor(TH["surface"])
    for sp in ax.spines.values():
        sp.set_color(TH["grid"])
    ax.grid(True, axis="y", color=TH["grid"], lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TH["ink2"], labelsize=8)


def main():
    rows = [r for r in csv.DictReader(open(SRC))
            if "realistic_00" in r["dataset"] and r["tree"] == "output_orb"]
    pairs = []
    for ds in sorted({r["dataset"] for r in rows}):
        for lc in ("loopclosing", "noloopclosing"):
            c = [r for r in rows if r["dataset"] == ds and r["run"] == lc + "_cpu"]
            g = [r for r in rows if r["dataset"] == ds and r["run"] == lc + "_gpu"]
            if c and g:
                pairs.append((ds[-3:], LC[lc], c[0], g[0]))
    if not pairs:
        print("no pairs found"); return

    fig = plt.figure(figsize=(10.0, 4.6), facecolor=TH["surface"])
    gs = fig.add_gridspec(1, 2, left=0.085, right=0.985, top=0.80, bottom=0.17, wspace=0.26)

    for k, (key, lbl, pct) in enumerate(
            [("of_bag", "frames of the bag processed [%]", True),
             ("ate_rms", "aligned ATE [m]", False)]):
        ax = fig.add_subplot(gs[0, k]); style(ax)
        up = dn = 0
        ends = []
        for ds, lc, c, g in pairs:
            a, b = float(c[key]), float(g[key])
            if pct:
                a, b = 100 * a, 100 * b
            up += b > a
            dn += b < a
            ax.plot([0, 1], [a, b], "-o", ms=5, lw=1.4, color=COL[lc], alpha=0.85)
            ends.append((b, ds))
        # The per-point recording labels that used to sit here were dropped:
        # several pairs land within a hair of each other on the right, so they
        # collided or escaped the axis, and the figure's point is the PATTERN
        # -- eight lines agreeing against eight disagreeing -- which the panel
        # titles already count.  Per-run numbers belong in the table.
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["CPU extractor", "GPU extractor"], fontsize=9)
        ax.set_xlim(-0.25, 1.35)
        ax.set_ylabel(lbl, color=TH["ink2"], fontsize=9)
        if not pct:
            ax.set_yscale("log")
        ax.set_title("(%s) %s\n    GPU higher in %d of %d pairs, lower in %d"
                     % ("ab"[k],
                        "throughput: every pair moves the same way" if pct
                        else "accuracy: the pairs disagree",
                        up, len(pairs), dn),
                     fontsize=9.5, loc="left", color=TH["ink"])

    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=COL[v], lw=3, label=v) for v in COL],
               loc="lower center", ncol=2, frameon=False, fontsize=8.5,
               labelcolor=TH["ink2"], bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Experiment 1, simulation: the front end paired within recording "
                 "and loop-closer state", fontsize=11.5, x=0.012, y=0.955,
                 ha="left", color=TH["ink"])
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "fig_frontend_pairs_light.png")
    fig.savefig(path, dpi=160, facecolor=TH["surface"])
    plt.close(fig)
    print("  wrote", path)
    for ds, lc, c, g in pairs:
        print("    %s %-16s of_bag %5.1f -> %5.1f %%   ATE %6.3f -> %6.3f m"
              % (ds, lc, 100 * float(c["of_bag"]), 100 * float(g["of_bag"]),
                 float(c["ate_rms"]), float(g["ate_rms"])))


if __name__ == "__main__":
    main()
