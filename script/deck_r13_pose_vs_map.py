#!/usr/bin/env python3
"""Deck slide R13: localised against the prior, the pose is as good as the map
behind it -- and over these distances the prior gives a fixed reference, not a
large accuracy gain.

    python3 script/deck_r13_pose_vs_map.py

Experiment 2B, the six reported runs (gate eps0 1.5 m), protocol in
script/exp4_protocol.yaml.
  (a) planar error of the published pose (map -> base_link from the
      localisation CSV, from the first fix) against ground truth, over the
      distance driven -- the computation of exp4_plot.summary(), one line per
      run.  Band: the prior map graphs' own node error, 0.10-0.40 m (R12).
  (b) per run, the median error localised against the prior and the median
      drift of unaided ORB-SLAM3 rigidly aligned onto ground truth
      (exp4_nav_insights.prior_value()), runs ordered by route length.
"""
import csv
import os
import sys

import numpy as np
import yaml
from matplotlib.lines import Line2D

import deck_style as S

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp4_goals as eg          # noqa: E402
import exp4_score as es          # noqa: E402
import exp4_nav_insights as ni   # noqa: E402

SHAPE = {"P1_transit": "straight", "P2_return": "square", "P3_circuit": "figure-8"}
COL = {"U": "#1f7a8c", "I": "#e67e22"}


def tracks(spec):
    out = []
    for world in spec["worlds"]:
        lay = spec["worlds"][world]["layout"]
        prior = spec["priors"][lay]["database"]
        al = eg._align(prior)
        for path in spec["priors"][lay]["paths"]:
            for rep in es.reps_present(prior, world, path):
                rund = os.path.join(S.ROOT, "map", prior, "exp4", f"{world}_{path}_{rep}")
                if es.is_gated(rund) is not True:
                    continue
                rob = os.path.join(rund, "robot_gt.csv")
                est = os.path.join(S.ROOT, "map", prior, f"localize_exp4_{world}_{path}_{rep}.csv")
                if not (os.path.exists(rob) and os.path.exists(est)):
                    continue
                t, gt = es.load_robot_gt(rob)
                gm = es.world_to_map(al, gt)
                rec = list(csv.DictReader(open(est)))
                ff = next((float(r["stamp"]) for r in rec if r["kind"] == "pose" and r.get("cov_trace")
                           and float(r["cov_trace"]) < 1e4), None)
                E = [r for r in rec if r["kind"] == "tf" and r.get("x") and (ff is None or float(r["stamp"]) >= ff)]
                et = np.array([float(r["stamp"]) for r in E])
                ex = np.array([float(r["x"]) for r in E])
                ey = np.array([float(r["y"]) for r in E])
                m = (et >= t[0]) & (et <= t[-1])
                err = np.hypot(ex[m] - np.interp(et[m], t, gm[:, 0]), ey[m] - np.interp(et[m], t, gm[:, 1]))
                step = np.hypot(*np.diff(gm, axis=0).T)
                step[step > 1.0] = 0.0
                dist = np.interp(et[m], t, np.r_[0, np.cumsum(step)])
                out.append((lay, SHAPE.get(path, path), dist, err, et[m]))
    return out


def main():
    spec = yaml.safe_load(open(os.path.join(S.ROOT, "script", "exp4_protocol.yaml")))
    T = tracks(spec)
    fig, (ax, bx) = S.figure(2, 1, gridspec_kw=dict(height_ratios=[1.1, 0.9], hspace=0.7))

    S.style(ax)
    ax.axhspan(0.10, 0.40, color=S.TH["good"], alpha=0.12, lw=0)
    for lay, shape, dist, err, tt in T:
        up = err > 1.0
        dur = float(np.sum(np.diff(tt)[up[:-1]])) if up.any() else 0.0
        print("%s %-8s %5.0f m  median %.3f m  max %.2f m  time over 1 m %.1f s"
              % (lay, shape, dist[-1], np.median(err), err.max(), dur))
        ax.plot(dist, err, lw=0.7, color=COL[lay], alpha=0.9)
    # the same U-square drive WITHOUT the gate (RTAB-9): RTAB-Map's own
    # corrections, from script/exp4_gate_counterfactual.py
    cf = np.load(os.path.join(S.ROOT, "output/compare/simulation/exp4-nav-insights/gate_counterfactual_series.npy"),
                 allow_pickle=True).item()
    d_u, e_w, e_u, _ = cf["U square"]
    ax.plot(d_u, e_u, color=S.TH["bad"], lw=0.8, ls=(0, (2.5, 1.2)))
    print("U square without the gate: max %.2f m, median %.3f m" % (e_u.max(), np.median(e_u)))
    ax.axhline(1.0, color=S.TH["bad"], lw=0.6, ls=":")
    ax.set_yscale("log")
    ax.set_ylim(0.01, 80)
    ax.set_yticks([0.01, 0.1, 1, 10])
    ax.set_yticklabels(["0.01", "0.1", "1", "10"])
    ax.set_xlim(0, 150)
    S.label(ax, x="distance driven since the first fix [m]", y="position error [m]")
    S.title(ax, "(a) the localised pose, 6 runs")
    S.legend(ax, [Line2D([], [], color=COL["U"], lw=1.0, label="U layout"),
                  Line2D([], [], color=COL["I"], lw=1.0, label="I layout"),
                  Line2D([], [], color=S.TH["bad"], lw=1.0, ls=(0, (2.5, 1.2)), label="U square, gate off")],
             loc="upper center", ncol=3, handlelength=1.4, columnspacing=0.7, frameon=False)

    S.style(bx, grid="y")
    pv = sorted(ni.prior_value(spec), key=lambda r: r["driven"])
    x = np.arange(len(pv))
    w = 0.36
    bx.bar(x - w / 2, [r["loc_med"] for r in pv], w, color=S.TH["good"])
    bx.bar(x + w / 2, [r["vio_med"] for r in pv], w, color=S.ORB)
    for r in pv:
        print("%-22s %5.1f m: localised median %.3f, unaided ORB median %.3f"
              % (r["run"], r["driven"], r["loc_med"], r["vio_med"]))
    bx.set_xticks(x)
    bx.set_xticklabels(["%.0f m" % r["driven"] for r in pv])
    S.label(bx, x="route length", y="median error [m]")
    S.title(bx, "(b) with the prior vs ORB-SLAM3 alone")
    from matplotlib.patches import Patch
    S.legend(bx, [Patch(color=S.TH["good"], label="localised on the prior"),
                  Patch(color=S.ORB, label="ORB-SLAM3 alone")], loc="upper center", ncol=2, frameon=False)
    bx.set_ylim(0, 0.46)
    S.save(fig, "r13_pose_vs_map.png")


if __name__ == "__main__":
    main()
