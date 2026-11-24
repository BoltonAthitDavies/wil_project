#!/usr/bin/env python3
"""Experiment 2B with and without the correction gate (RTAB-9), on the SAME
drives: a paired counterfactual from the reported runs' own logs.

    python3 script/exp4_gate_counterfactual.py

WHY THIS IS A FAIR COMPARISON
    RTAB-Map never sees the gate: correction_gate.py reads /localization_pose,
    decides, and republishes map -> odom, and nothing flows back.  So every
    correction RTAB-Map proposed during a gated run is exactly what it would
    have published with the gate switched off.  The gate logs each fresh
    candidate correction to gate_verdicts.csv (warm-up, accept or reject), and
    the localisation CSV logs the published map -> base_link and map -> odom at
    ~2 Hz.  So for the same route, the same moment and the same map:

        WITH     the published pose (localisation CSV, kind=tf), as scored in
                 the experiment
        WITHOUT  the latest candidate correction, of any verdict, composed with
                 the front end's odom -> base at that instant, where
                 odom -> base = inv(published map -> odom) * published map -> base

    A run in which the gate refused nothing gives WITHOUT == WITH; that is the
    check that the reconstruction is right.

    Scoring is exp4_plot.summary()'s: ground truth from robot_gt.csv placed in
    the map frame with the prior's fitted alignment, samples before the first
    fix excluded.

LIMIT
    Files written before 20 Nov 2026 carry no row for a concession (8 refusals
    in a row, then accept); later ones log it as 'concede', a repeat of the
    reject row before it, which this script treats as one more candidate at the
    same instant -- harmless.  WITH uses the published pose either way.  One
    behaviour cannot be replayed: a robot driven by Nav2 on a wrong pose would
    have steered differently.  These routes were hand-driven, so the path is the
    same either way.

Writes output/compare/simulation/exp4-nav-insights/gate_counterfactual.csv
"""
import csv
import math
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp4_goals as eg   # noqa: E402
import exp4_score as es   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output/compare/simulation/exp4-nav-insights")
SHAPE = {"P1_transit": "straight", "P2_return": "square", "P3_circuit": "figure-8"}


def compose(a, b):
    """SE(2) a*b, each (x, y, yaw_rad)."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2])


def inverse(a):
    c, s = math.cos(a[2]), math.sin(a[2])
    return (-c * a[0] - s * a[1], s * a[0] - c * a[1], -a[2])


def runs(spec):
    for world in spec["worlds"]:
        lay = spec["worlds"][world]["layout"]
        prior = spec["priors"][lay]["database"]
        for path in spec["priors"][lay]["paths"]:
            for rep in es.reps_present(prior, world, path):
                rund = os.path.join(ROOT, "map", prior, "exp4", f"{world}_{path}_{rep}")
                if es.is_gated(rund) is True and es.gate_eps0(rund) == 1.5:
                    yield lay, prior, world, path, rep, rund


def stats(t, e):
    over = e > 1.0
    dt = np.diff(t, append=t[-1])
    return dict(median=float(np.median(e)), p95=float(np.percentile(e, 95)), max=float(e.max()),
                over1_s=float(dt[over].sum()))


def main():
    spec = yaml.safe_load(open(os.path.join(ROOT, "script", "exp4_protocol.yaml")))
    rows, series = [], {}
    for lay, prior, world, path, rep, rund in runs(spec):
        al = eg._align(prior)
        t_gt, gt = es.load_robot_gt(os.path.join(rund, "robot_gt.csv"))
        gm = es.world_to_map(al, gt)
        rec = list(csv.DictReader(open(os.path.join(ROOT, "map", prior, f"localize_exp4_{world}_{path}_{rep}.csv"))))
        ff = next(float(r["stamp"]) for r in rec if r["kind"] == "pose" and r.get("cov_trace")
                  and float(r["cov_trace"]) < 1e4)
        tf = [r for r in rec if r["kind"] == "tf" and r.get("x") and float(r["stamp"]) >= ff]
        cand = [(float(r["stamp"]), (float(r["cand_x"]), float(r["cand_y"]), math.radians(float(r["cand_yaw_deg"]))), r["verdict"])
                for r in csv.DictReader(open(os.path.join(rund, "gate_verdicts.csv")))]
        t, with_xy, without_xy = [], [], []
        k = -1
        for r in tf:
            ts = float(r["stamp"])
            mb = (float(r["x"]), float(r["y"]), math.radians(float(r["yaw_deg"])))
            mo = (float(r["corr_x"]), float(r["corr_y"]), math.radians(float(r["corr_yaw_deg"])))
            ob = compose(inverse(mo), mb)
            while k + 1 < len(cand) and cand[k + 1][0] <= ts:
                k += 1
            wo = compose(cand[k][1], ob) if k >= 0 else mb
            t.append(ts); with_xy.append(mb[:2]); without_xy.append(wo[:2])
        t = np.array(t); W = np.array(with_xy); U = np.array(without_xy)
        m = (t >= t_gt[0]) & (t <= t_gt[-1])
        g = np.stack([np.interp(t[m], t_gt, gm[:, 0]), np.interp(t[m], t_gt, gm[:, 1])], 1)
        ew = np.hypot(*(W[m] - g).T)
        eu = np.hypot(*(U[m] - g).T)
        step = np.hypot(*np.diff(gm, axis=0).T)
        step[step > 1.0] = 0.0
        dist = np.interp(t[m], t_gt, np.r_[0, np.cumsum(step)])
        name = f"{lay} {SHAPE.get(path, path)}"
        rej = sum(v == "reject" for _, _, v in cand)
        sw, su = stats(t[m], ew), stats(t[m], eu)
        rows.append(dict(run=name, route_m=round(float(dist[-1]), 1), rejected=rej,
                         **{f"with_{k}": round(v, 3) for k, v in sw.items()},
                         **{f"without_{k}": round(v, 3) for k, v in su.items()},
                         max_gap=round(float(np.abs(eu - ew).max()), 3)))
        series[name] = (dist, ew, eu, t[m])
        print("%-12s %5.0f m  refused %d | WITH median %.3f p95 %.2f max %.2f over1m %4.1f s"
              " | WITHOUT median %.3f p95 %.2f max %.2f over1m %4.1f s | largest gap %.3f m"
              % (name, dist[-1], rej, sw["median"], sw["p95"], sw["max"], sw["over1_s"],
                 su["median"], su["p95"], su["max"], su["over1_s"], np.abs(eu - ew).max()))
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, "gate_counterfactual.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    np.save(os.path.join(OUT, "gate_counterfactual_series.npy"),
            {k: tuple(np.asarray(x) for x in v) for k, v in series.items()}, allow_pickle=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
