#!/usr/bin/env python3
"""Deck slide R11: RTAB-Map's words propose a place, its geometry disposes.

    python3 script/deck_r11_words_geometry.py

Data from script/exp4_rtabmap_gating.py collect(): the six reported
Experiment 2B relocalisation runs.  Each rejection is logged by RTAB-Map as
"Not enough inliers i/N (matches=m)": m = visual words the proposal shared,
i = matched features that agreed in metres (Vis/EstimationType 0).  Accepted
fixes are counted from the localisation CSV; RTAB-Map logs no inlier count for
them, so (a) holds rejections only.
"""
import os
import sys

import numpy as np
from matplotlib.patches import Patch

import deck_style as S

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp4_rtabmap_gating as G  # noqa: E402


def main():
    rows = G.collect()
    inl = np.array([v for r in rows for v in r["inliers"]])
    mat = np.array([v for r in rows for v in r["matches"]])
    thr = rows[0]["thr"]
    acc = sum(r["accepted"] for r in rows)
    print("%d proposals, %d accepted, %d rejected; zero-inlier rejections %d, their median words %.0f, max %d"
          % (acc + len(inl), acc, len(inl), (inl == 0).sum(), np.median(mat[inl == 0]), mat[inl == 0].max()))

    fig, (ax, bx) = S.figure(2, 1, gridspec_kw=dict(height_ratios=[1.0, 1.0], hspace=0.75))
    S.style(ax)
    jit = (np.random.default_rng(0).random(len(inl)) - 0.5) * 0.8
    ax.scatter(mat, inl + jit, s=4, color=S.TH["bad"], alpha=0.75, lw=0)
    ax.axhline(thr, color=S.TH["ink2"], lw=0.7, ls="--")
    ax.text(198, thr + 0.8, "needed to accept: %d" % thr, ha="right", va="bottom", fontsize=4.9, color=S.TH["ink2"])
    ax.set_xlim(0, 200)
    ax.set_ylim(-2, 25)
    S.label(ax, x="words the two places share", y="features that agree\nin metres")
    S.title(ax, "(a) the %d proposals RTAB-6 threw out" % len(inl))

    S.style(bx, grid="y")
    x = np.arange(len(rows))
    a = np.array([r["accepted"] for r in rows], float)
    some = np.array([sum(v > 0 for v in r["inliers"]) for r in rows], float)
    zero = np.array([sum(v == 0 for v in r["inliers"]) for r in rows], float)
    bx.bar(x, a, 0.62, color=S.TH["good"])
    bx.bar(x, some, 0.62, bottom=a, color="#f0a868")
    bx.bar(x, zero, 0.62, bottom=a + some, color=S.TH["bad"])
    bx.set_xticks(x)
    bx.set_xticklabels([r["run"].replace(" ", "\n").replace("figure-8", "fig-8") for r in rows], fontsize=5.2)
    S.label(bx, y="proposals")
    S.title(bx, "(b) every proposal, per run")
    S.legend(bx, [Patch(color=S.TH["good"], label="accepted"),
                  Patch(color="#f0a868", label="rejected, some agreement"),
                  Patch(color=S.TH["bad"], label="rejected, none agrees")], loc="upper left")
    bx.set_ylim(0, 135)
    S.save(fig, "r11_words_geometry.png")


if __name__ == "__main__":
    main()
