#!/usr/bin/env python3
"""Deck slide R2: the real rig's images are better than the simulator's; the
robot is slower, and that alone makes drift look ten times worse per metre.

    python3 script/deck_r2_real_vs_sim.py

  (a) real rig divided by simulation, from output/compare/datasets/
      measurements.json (script/exp1_dataset_compare.py): mean image gradient
      and corners per frame (FAST corners per megapixel x megapixels) on sampled
      left frames, and the ground-truth speed while moving (median).
      Recordings: realistic_000-003 and realv7_000-002.
  (b) the same drift measured two ways, every run of both front ends, from
      output/compare/datasets/drift_rate.csv (script/exp1_drift_rate.py): the
      1-s relative pose error (error made per second) and that error divided by
      the distance driven in the second (per metre).  Bars = median per domain.
"""
import json
import os

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import deck_style as S


def main():
    M = json.load(open(os.path.join(S.ROOT, "output/compare/datasets/measurements.json")))

    def pooled(dom, f):
        return float(np.median([f(M[dom][d]) for d in M[dom]]))
    grad = pooled("real", lambda m: m["grad"]) / pooled("sim", lambda m: m["grad"])
    corners = pooled("real", lambda m: m["fast_per_mpx"] * m["mpx"]) / pooled("sim", lambda m: m["fast_per_mpx"] * m["mpx"])
    vr = np.median(np.concatenate([M["real"][d]["motion"]["v"] for d in M["real"]]))
    vs = np.median(np.concatenate([M["sim"][d]["motion"]["v"] for d in M["sim"]]))
    print("real/sim: gradient %.2f, corners per frame %.2f, speed %.2f/%.2f = %.2f" % (grad, corners, vr, vs, vr / vs))

    d = pd.read_csv(os.path.join(S.ROOT, "output/compare/datasets/drift_rate.csv"))
    med = d.groupby("domain")[["rpe", "rpe_per_m"]].median()
    rs = med.loc["real rig", "rpe"] / med.loc["simulation", "rpe"]
    rm = med.loc["real rig", "rpe_per_m"] / med.loc["simulation", "rpe_per_m"]
    print("drift real/sim: per second x%.1f, per metre x%.1f" % (rs, rm))

    fig, (ax, bx) = S.figure(2, 1, gridspec_kw=dict(height_ratios=[0.9, 1.1], hspace=0.62))
    S.style(ax, grid="x")
    names = ["image gradient", "corners per frame", "robot speed"]
    vals = [grad, corners, vr / vs]
    cols = [S.TH["good"], S.TH["good"], S.TH["grey"]]
    ax.barh(range(3)[::-1], vals, color=cols, height=0.6)
    for y, v in zip(range(3)[::-1], vals):
        ax.text(v * 1.08, y, "×%.2f" % v, va="center", fontsize=5.2, color=S.TH["ink"])
    ax.axvline(1, color=S.TH["ink2"], lw=0.7)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 10)
    ax.set_xticks([0.1, 0.3, 1, 3, 10])
    ax.set_xticklabels(["×0.1", "×0.3", "×1", "×3", "×10"])
    ax.set_yticks(range(3)[::-1])
    ax.set_yticklabels(names)
    S.label(ax, x="real rig ÷ simulation")
    S.title(ax, "(a) what the real rig gives the front end")

    S.style(bx, grid="y")
    rng = np.random.default_rng(1)
    for k, col_ in enumerate(("rpe", "rpe_per_m")):
        for j, (dom, c) in enumerate((("simulation", S.SIM), ("real rig", S.ROBOT))):
            v = d[d.domain == dom][col_].values
            x = k * 1.0 + (j - 0.5) * 0.36
            bx.scatter(x + rng.uniform(-0.07, 0.07, len(v)), v, s=4, color=c, alpha=0.75, lw=0)
            bx.plot([x - 0.13, x + 0.13], [np.median(v)] * 2, color=c, lw=1.6)
        ratio = rs if k == 0 else rm
        bx.text(k, 1.1, "real ×%.1f" % ratio, ha="center", fontsize=5.4, color=S.TH["bad"])
    bx.set_yscale("log")
    bx.set_ylim(0.005, 2.5)
    bx.set_yticks([0.01, 0.1, 1])
    bx.set_yticklabels(["0.01", "0.1", "1"])
    bx.set_xticks([0, 1])
    bx.set_xticklabels(["error per second\n[m]", "error per metre driven\n[m/m]"])
    bx.set_xlim(-0.5, 1.5)
    S.title(bx, "(b) the same drift, measured two ways")
    S.legend(bx, [Line2D([], [], ls="", marker="o", ms=2.5, color=S.SIM, label="simulation runs"),
                  Line2D([], [], ls="", marker="o", ms=2.5, color=S.ROBOT, label="real-rig runs")],
             loc="center", bbox_to_anchor=(0.5, 0.62), frameon=False)
    S.save(fig, "r2_real_vs_sim.png")


if __name__ == "__main__":
    main()
