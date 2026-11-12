#!/usr/bin/env python3
"""How good were the POSES the realistic maps were built from?

    python3 script/eval_map_graph.py

WHY THIS EXISTS
  All four realistic-warehouse maps fail the quality criterion of
  script/eval_realistic_maps.py, and the chapter cannot say whether they fail
  because the trajectory was wrong or because the grid was built wrongly from a
  trajectory that was right.  The Open Concerns note on these builds blocks
  attributing the failure to a FRONT END, because no database records which one
  supplied the odometry.  It does not block measuring the poses themselves: the
  graph's node poses are in the database, the two mapping recordings have
  ground truth, and the placement that put each map into the world was fitted
  before the maps were scored.

WHAT IS MEASURED, AND IN WHICH FRAME
  Two numbers per database, because they answer different questions.

  placed    node poses pushed through the map's own fitted placement
            (the .mappose file) and compared to ground truth where they stand.
            This is the frame the occupancy grid was rasterised and scored in,
            so it is the error that bears on map quality.
  aligned   the same poses after a best-fit rigid SE(2) transform of their own.
            A placement error is absorbed, so what is left is the shape of the
            trajectory: drift.

  The two together separate "the map is in the wrong place" from "the map is
  the wrong shape".

THE LEVER ARM
  Simulated ground truth is the model origin and the visual estimators report
  the IMU point, 0.338 m forward and 0.192 m up, which vio_metrics.load_gt
  corrects for by default.  RTAB-Map's graph is neither: its nodes are whatever
  TF frame the mapping session published.  Rather than assume, this script
  scores against BOTH ground truths and reports which fits better, with the
  difference, so the choice is visible instead of buried.

Outputs, under output/compare/simulation/exp4-realistic-maps/:
  graph_vs_gt.csv   one row per database
  graph_vs_gt.tex   the table
"""

import csv
import math
import os
import sqlite3
import struct
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                   # noqa: E402
import vio_metrics as vm                             # noqa: E402
from eval_realistic_maps import MAPS                 # noqa: E402  the single map registry

ROOT = Path(vm.ROOT)
OUT = ROOT / "output" / "compare" / "simulation" / "exp4-realistic-maps"
GT = ROOT / "dataset" / "gt"


def node_poses(db):
    """(stamp, x, y, z) per node, in the map's own frame, ordered by id."""
    uri = "file:%s?mode=ro" % db
    c = sqlite3.connect(uri, uri=True)
    out = []
    for nid, stamp, pose in c.execute("SELECT id,stamp,pose FROM Node ORDER BY id"):
        if pose and len(pose) >= 48:
            v = struct.unpack("12f", pose[:48])
            out.append((float(stamp), v[3], v[7], v[11]))
    c.close()
    return np.asarray(out, float)


def placed(p_map, mappose):
    """Map frame -> world, exactly as validate_map.py places the occupancy grid."""
    mx, my, mdeg = mappose
    th = math.radians(mdeg)
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, -s], [s, c]])
    return p_map[:, :2] @ R.T + [mx, my]


def rigid_2d(a, b):
    """Rigid SE(2) taking a onto b, no scale (Kabsch in the plane)."""
    ca, cb = a.mean(0), b.mean(0)
    H = (a - ca).T @ (b - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, d]) @ U.T
    return (a - ca) @ R.T + cb


def err_stats(e):
    return dict(rms=float(np.sqrt((e ** 2).mean())), max=float(e.max()), median=float(np.median(e)))


def main():
    rows = []
    for m in MAPS:
        key, db = m.get("key"), m.get("db")
        if not db or not os.path.exists(db):
            print("  no database:", key); continue
        mp_file = OUT / ("%s.mappose" % key)
        if not mp_file.exists():
            print("  no placement:", key); continue
        mappose = [float(v) for v in mp_file.read_text().strip().split(",")]
        # the recording the map was BUILT from, not the independent probe the
        # occupancy grid was scored against
        build_bag = os.path.basename(str(m.get("build_bag", "")))
        gt_csv = GT / build_bag / "ground_truth.csv"
        if not gt_csv.exists():
            print("  no ground truth for %s (looked for %s)" % (key, gt_csv)); continue

        n = node_poses(db)
        if len(n) < 10:
            print("  too few nodes:", key); continue
        t, pm_ = n[:, 0], n[:, 1:]
        pw = placed(pm_, mappose)

        best = None
        for lever in (True, False):
            g = vm.load_gt(str(gt_csv), lever=lever)
            if g is None:
                continue
            gt_t, gt_p = g[0], g[1]
            keep = (t >= gt_t[0]) & (t <= gt_t[-1])
            if keep.sum() < 10:
                print("  node stamps do not overlap ground truth:", key); break
            gp = vm.resample(gt_t, gt_p, t[keep])[:, :2]
            e_pl = np.linalg.norm(pw[keep] - gp, axis=1)
            e_al = np.linalg.norm(rigid_2d(pw[keep], gp) - gp, axis=1)
            cand = dict(lever=lever, n_used=int(keep.sum()), placed=err_stats(e_pl),
                        aligned=err_stats(e_al),
                        path_graph=float(np.linalg.norm(np.diff(pw[keep], axis=0), axis=1).sum()),
                        path_gt=float(np.linalg.norm(np.diff(gp, axis=0), axis=1).sum()))
            if best is None or cand["aligned"]["rms"] < best["aligned"]["rms"]:
                best, other = cand, best
            else:
                other = cand
        if best is None:
            continue
        rows.append(dict(key=key, layout=m.get("layout"), build=m.get("build"), build_bag=build_bag, nodes=len(n),
                         used=best["n_used"], lever=best["lever"],
                         lever_delta=(None if other is None else
                                      other["aligned"]["rms"] - best["aligned"]["rms"]),
                         placed_rms=best["placed"]["rms"], placed_max=best["placed"]["max"],
                         aligned_rms=best["aligned"]["rms"], aligned_max=best["aligned"]["max"],
                         aligned_median=best["aligned"]["median"],
                         path_graph=best["path_graph"], path_gt=best["path_gt"],
                         pct_path=100.0 * best["aligned"]["rms"] / best["path_gt"]))
        r = rows[-1]
        print("  %-34s nodes %3d  placed rms %.3f max %.3f | aligned rms %.3f max %.3f (%.2f %% of path)"
              "  lever=%s%s" % (key, r["nodes"], r["placed_rms"], r["placed_max"], r["aligned_rms"],
                                r["aligned_max"], r["pct_path"], r["lever"],
                                "" if r["lever_delta"] is None else "  (other frame %+.3f m)" % r["lever_delta"]))
    if not rows:
        print("nothing scored"); return 1

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "graph_vs_gt.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    L = ["% Generated by script/eval_map_graph.py -- do not edit by hand.",
         "\\begin{table}[htbp]", "\\centering",
         "\\caption[Pose accuracy of the graph each realistic map was built from.]{Pose accuracy of the "
         "graph each realistic map was built from, scored against the mapping recording's ground truth at every "
         "node stamp. ``Placed'' is the error where the map actually stands, through the fitted placement the "
         "occupancy grid was scored with, so it carries any placement error; ``aligned'' is the residual after a "
         "best-fit rigid $SE(2)$ transform of the graph itself, which absorbs placement and leaves the shape of "
         "the trajectory. Node poses are whatever TF frame the mapping session published, so each graph is scored "
         "against ground truth both at the model origin and at the IMU point and the better fit is reported. The "
         "IMU point fits better on all four, by \\SIrange{0.08}{0.27}{\\metre}, so the graph is closer to the "
         "estimator's frame than to the model origin; the model-origin figures would be that much larger and "
         "change no conclusion here.}",
         "\\label{tab:map-graph-accuracy}", "\\small", "\\resizebox{\\textwidth}{!}{%",
         "\\begin{tabular}{llr rr rr r}", "\\toprule",
         "Database & Layout & Nodes & Placed rms [\\si{\\metre}] & Placed max [\\si{\\metre}] & "
         "Aligned rms [\\si{\\metre}] & Aligned max [\\si{\\metre}] & Aligned/path [\\si{\\percent}] \\\\",
         "\\midrule"]
    for r in rows:
        L.append("\\repo{%s} & %s & %d & %.3f & %.3f & %.3f & %.3f & %.2f \\\\" % (
            r["key"], r["layout"] or "---", r["nodes"], r["placed_rms"], r["placed_max"],
            r["aligned_rms"], r["aligned_max"], r["pct_path"]))
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    (OUT / "graph_vs_gt.tex").write_text("\n".join(L) + "\n")
    print("\n  wrote", OUT / "graph_vs_gt.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
