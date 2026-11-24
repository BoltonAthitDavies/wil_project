#!/usr/bin/env python3
"""The correction gate (RTAB-9) on real data: every correction RTAB-Map
proposed in the six reported Experiment 2B runs, against the gate's bound,
coloured by the position error that correction would have caused.

    python3 script/deck_rtab9_bound.py

For each fresh candidate in gate_verdicts.csv (warm-up rows left out: the
gate accepts everything then, by design):
    x  d_odom, metres the front end reports since the last accepted fix
    y  dp, the step from the correction in force to the candidate (and dyaw)
    line  the bound: 1.5 m + 0.02 m per m, 8 deg + 0.1 deg per m
    colour  the error of the robot's pose had this candidate been applied at
            that instant: candidate map->odom composed with the front end's
            odom->base (from the localisation CSV), against ground truth placed
            in the map frame as in exp4_gate_counterfactual.py
    marker  dot = accepted, cross = refused

Output: docs/report/insights/update_181126/fig/rtab9_bound.png
"""
import csv
import math
import os
import sys

import numpy as np
import yaml
from matplotlib.lines import Line2D

import deck_style as S

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp4_goals as eg                       # noqa: E402
import exp4_score as es                       # noqa: E402
import exp4_gate_counterfactual as CF         # noqa: E402

EPS, EPS0, EPS_YAW, YAW0 = 0.02, 1.5, 0.10, 8.0


def main():
    spec = yaml.safe_load(open(os.path.join(S.ROOT, "script", "exp4_protocol.yaml")))
    P = []      # (d_odom, dp, dyaw_deg, verdict, error_m, run)
    for lay, prior, world, path, rep, rund in CF.runs(spec):
        al = eg._align(prior)
        t_gt, gt = es.load_robot_gt(os.path.join(rund, "robot_gt.csv"))
        gm = es.world_to_map(al, gt)
        rec = list(csv.DictReader(open(os.path.join(S.ROOT, "map", prior, f"localize_exp4_{world}_{path}_{rep}.csv"))))
        tf = [r for r in rec if r["kind"] == "tf" and r.get("x")]
        tt = np.array([float(r["stamp"]) for r in tf])
        for r in csv.DictReader(open(os.path.join(rund, "gate_verdicts.csv"))):
            if r["verdict"] == "warmup":
                continue
            ts = float(r["stamp"])
            k = int(np.clip(np.searchsorted(tt, ts), 0, len(tf) - 1))
            q = tf[k]
            mb = (float(q["x"]), float(q["y"]), math.radians(float(q["yaw_deg"])))
            mo = (float(q["corr_x"]), float(q["corr_y"]), math.radians(float(q["corr_yaw_deg"])))
            ob = CF.compose(CF.inverse(mo), mb)
            c = (float(r["cand_x"]), float(r["cand_y"]), math.radians(float(r["cand_yaw_deg"])))
            p = CF.compose(c, ob)
            g = (np.interp(tt[k], t_gt, gm[:, 0]), np.interp(tt[k], t_gt, gm[:, 1]))
            err = math.hypot(p[0] - g[0], p[1] - g[1])
            # d_odom is recovered from the logged allowance: correction_gate.py
            # writes accepted rows AFTER accept() has reset d_odom to 0
            # (lines 254-255), but allow_p was computed before the reset.
            d_odom = (float(r["allow_p"]) - EPS0) / EPS
            P.append((d_odom, float(r["dp"]), float(r["dyaw_deg"]), r["verdict"], err,
                      f"{lay} {CF.SHAPE.get(path, path)}"))
    P.sort(key=lambda x: x[4])
    acc = [x for x in P if x[3] == "accept"]
    rej = [x for x in P if x[3] == "reject"]
    # 'concede' rows (logged since 20 Nov 2026) repeat the reject just before
    # them; they are counted, not drawn, so a refusal is never drawn twice.
    con = [x for x in P if x[3] == "concede"]
    if con:
        print("%d concessions (refused 8 times, then accepted)" % len(con))
    print("%d candidates: %d accepted, %d refused" % (len(P), len(acc), len(rej)))
    print("accepted: dp %.2f-%.2f m, dyaw %.1f-%.1f deg, would-be error %.2f-%.2f m"
          % (min(x[1] for x in acc), max(x[1] for x in acc), min(x[2] for x in acc), max(x[2] for x in acc),
             min(x[4] for x in acc), max(x[4] for x in acc)))
    for x in rej:
        print("refused: %s d_odom %.1f m, dp %.2f m, dyaw %.1f deg -> would-be error %.2f m" % (x[5], x[0], x[1], x[2], x[4]))

    # ---- one panel: step against turn; the gate's bound is a box
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    S.setup()
    fig, ax = plt.subplots(figsize=(3.0, 1.55))
    fig.patch.set_facecolor(S.TH["surface"])
    S.style(ax)
    dmax = max(x[0] for x in P)
    for d, ls, lab in ((0.0, "--", "bound right after a fix"),
                       (dmax, ":", "bound after %.0f m without one" % dmax)):
        bx_, by_ = EPS0 + EPS * d, YAW0 + EPS_YAW * d
        ax.plot([1e-3, bx_, bx_], [by_, by_, 1e-3], color=S.TH["ink"], lw=0.8, ls=ls, label=lab)
    ax.fill_between([1e-3, EPS0], 1e-3, YAW0, color=S.TH["good"], alpha=0.10, lw=0)
    norm = mpl.colors.LogNorm(0.05, 10)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("e", [S.TH["good"], "#f1c40f", S.TH["bad"]])
    for grp, mk, sz in ((acc, "o", 8), (rej, "X", 30)):
        ax.scatter([max(x[1], 0.003) for x in grp], [max(x[2], 0.015) for x in grp], c=[x[4] for x in grp],
                   cmap=cmap, norm=norm, marker=mk, s=sz, lw=0.3, edgecolors=S.TH["ink"], zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.002, 20); ax.set_ylim(0.01, 60)
    ax.set_xticks([0.01, 0.1, 1, 10]); ax.set_xticklabels(["0.01", "0.1", "1", "10"])
    ax.set_yticks([0.1, 1, 10]); ax.set_yticklabels(["0.1", "1", "10"])
    S.label(ax, x="correction step [m]", y="correction turn [deg]")
    ax.text(0.0028, 5.2, "accepted: inside the box", fontsize=5.0, color=S.TH["good"], va="top")
    ax.text(14, 0.016, "refused", fontsize=5.0, color=S.TH["bad"], ha="right", va="bottom")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker="o", ms=3, mfc="white", mec=S.TH["ink"], label="accepted (%d)" % len(acc)),
         Line2D([], [], ls="", marker="X", ms=4, mfc="white", mec=S.TH["ink"], label="refused (%d)" % len(rej))]
    h += ax.get_legend_handles_labels()[0]
    S.legend(ax, h, loc="upper left", fontsize=4.6, ncol=2, columnspacing=0.8, frameon=False)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.02, aspect=18)
    cb.set_label("pose error if applied [m]", color=S.TH["ink2"], fontsize=5.0)
    cb.ax.tick_params(labelsize=4.8, colors=S.TH["ink2"], length=2)
    cb.set_ticks([0.1, 0.3, 1, 3, 10]); cb.set_ticklabels(["0.1", "0.3", "1", "3", "10"])
    S.save(fig, "rtab9_bound.png")


if __name__ == "__main__":
    main()
