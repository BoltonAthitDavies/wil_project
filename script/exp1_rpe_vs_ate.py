#!/usr/bin/env python3
"""Experiment 1: local error against accumulated error, one point per visual run.

x = jump-free translational RPE at 1 s, y = aligned 3-D ATE (log), with the
project's two targets drawn (RPE <= 0.1 m, ATE <= 0.05 m). The point of the
figure is the shape: every run passes the RPE target and fails the ATE one,
and ATE spreads over a factor of ~40 while RPE spans a factor of ~4 with no
ordering between them -- the accumulated error is not the local error added up.

Reads output/compare/simulation/exp-realistic-ab/runs_metrics.csv (the same
file the baseline and RPE tables come from) and writes
docs/report/final_report/figures/exp1_rpe_vs_ate.png plus a copy beside the
csv. Prints the rank correlation quoted in the caption.
"""
import csv
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "output/compare/simulation/exp-realistic-ab/runs_metrics.csv")
OUT = os.path.join(ROOT, "docs/report/final_report/figures/exp1_rpe_vs_ate.png")
COPY = os.path.join(os.path.dirname(SRC), "rpe_vs_ate_light.png")

RPE_TARGET, ATE_TARGET = 0.10, 0.05
DIVERGED_M = 1.0
ARM = {  # colour per arm, as analyse_realistic_runs.COLORS
    "VINS-Fusion": ("VINS-Fusion", "#cc8319"),
    "active": ("ORB-SLAM3, loop closing on", "#e040fb"),
    "disabled": ("ORB-SLAM3, loop closing off", "#7b2d8e"),
}
LAYOUT = {"000": "U", "001": "I", "002": "U", "003": "I"}
# Why each diverged run is off, from the chapter's analysis.
CAUSE = {
    ("001", "active"): "false closure",
    ("003", "disabled"): "vertical drift",
    ("002", "active"): "initialisation offset",
}


def main():
    rows = [r for r in csv.DictReader(open(SRC))
            if r["filtered"] == "False" and r["valid"] == "True"]
    assert len(rows) == 20, len(rows)
    x = np.array([float(r["rpe1_trans_clean"]) for r in rows])
    y = np.array([float(r["ate_rms"]) for r in rows])
    rho_all = spearmanr(x, y)[0]
    held = y < DIVERGED_M
    rho_held = spearmanr(x[held], y[held])[0]
    print(f"RPE {x.min():.3f}-{x.max():.3f} m (x{x.max()/x.min():.1f}); "
          f"ATE {y.min():.3f}-{y.max():.3f} m (x{y.max()/y.min():.1f})")
    print(f"Spearman rho(RPE, ATE): all 20 {rho_all:+.2f}; 15 held {rho_held:+.2f}")

    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    ax.set_yscale("log")
    ax.set_xlim(0, 0.12)
    ax.set_ylim(0.03, 10)

    # target region: meets both targets (empty)
    ax.axvspan(0, RPE_TARGET, ymin=0, ymax=(np.log10(ATE_TARGET) - np.log10(0.03))
               / (np.log10(10) - np.log10(0.03)), color="#1f8a4c", alpha=0.10, lw=0)
    ax.axvline(RPE_TARGET, color="#c62828", ls="--", lw=1)
    ax.axhline(ATE_TARGET, color="#c62828", ls="--", lw=1)
    ax.text(RPE_TARGET + 0.0015, 0.2, "RPE target 0.1 m", color="#c62828",
            ha="left", va="center", fontsize=9, rotation=90)
    ax.text(0.118, ATE_TARGET * 1.12, "ATE target 0.05 m", color="#c62828",
            ha="right", va="bottom", fontsize=9)
    ax.text(0.003, 0.036, "meets both targets: no run", color="#1f8a4c", fontsize=9,
            va="center")

    seen = set()
    groups = {}
    for r, xi, yi in zip(rows, x, y):
        key = "VINS-Fusion" if r["estimator"] == "VINS-Fusion" else r["lc_state"]
        name, col = ARM[key]
        gpu = "gpu" in r["run"]
        ds = r["dataset"][-3:]
        marker = "o" if LAYOUT[ds] == "U" else "^"
        lab = None
        if (key, gpu, marker) not in seen:
            seen.add((key, gpu, marker))
        ax.scatter(xi, yi, s=60, marker=marker, zorder=3,
                   facecolor="white" if gpu else col, edgecolor=col, linewidth=1.6)
        cause = CAUSE.get((ds, r["lc_state"]))
        if cause and yi >= DIVERGED_M:
            groups.setdefault((cause, ds), []).append((xi, yi))
    # one label per cause: right of the pair, at its geometric-mean height
    for (cause, ds), pts in groups.items():
        px = max(p[0] for p in pts); py = float(np.exp(np.mean([np.log(p[1]) for p in pts])))
        n = len(pts)
        ax.annotate(f"{cause}\nrealistic_{ds}" + (f", {n} runs" if n > 1 else ""),
                    (px, py), xytext=(10, -16 if cause.startswith("init") else 0),
                    textcoords="offset points",
                    fontsize=8.5, va="center", color="#333333")

    # legends: colour = arm, fill = front end, shape = layout
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker="s", ms=8, color=c, label=n) for n, c in ARM.values()]
    h += [Line2D([], [], ls="", marker="s", ms=8, color="#777", label="CPU front end"),
          Line2D([], [], ls="", marker="s", ms=8, mfc="white", mec="#777", mew=1.5,
                 label="GPU front end"),
          Line2D([], [], ls="", marker="o", ms=7, color="#777", label="U layout"),
          Line2D([], [], ls="", marker="^", ms=7, color="#777", label="I layout")]
    ax.legend(handles=h, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
              fontsize=9)
    ax.set_xlabel("jump-free translational RPE at 1 s [m]  (local error)")
    ax.set_ylabel("aligned ATE [m], log  (accumulated error)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.grid(True, which="major", color="#e3e3e3", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200, facecolor="white")
    shutil.copyfile(OUT, COPY)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
