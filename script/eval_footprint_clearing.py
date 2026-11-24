#!/usr/bin/env python3
"""Does RTAB-Map's own GridGlobal/FootprintRadius remove the phantom cells that
block the robot's route?  An offline test on the four closure-free realistic
maps -- no new recording.

    python3 script/eval_footprint_clearing.py [--radii 0,0.30,0.50,0.68]

WHAT THE SETTING DOES
    GridGlobal/FootprintRadius: "Footprint radius (m) used to clear all
    obstacles under the graph."  When the global grid is assembled, every cell
    within this radius of a graph node is cleared.  A node is where the MAPPING
    robot itself stood, so an obstacle there is false by definition.  It was
    0 (off) in all four databases.

THE RADIUS, FROM THE SIMULATOR
    ackermann_robot/model.sdf: chassis box 0.6 x 0.463 m centred on
    base_footprint; wheels at (+-0.21, +-0.21), radius 0.0585, width 0.065, so
    y reaches +-0.2425; the camera mast reaches x = +0.506.  Body extent about
    the chassis centre: x -0.30 .. +0.506, y +-0.2425.
    The graph nodes are NOT at the chassis centre: RTAB-Map's body frame in
    simulation is at the IMU, base_footprint + (0.338, 0, 0.192).  About that
    point the body spans x -0.638 .. +0.168, y +-0.2425, so
        0.30 m  ~ the body's half-width plus margin, around the node only
        0.50 m  most of the body
        0.68 m  the whole body (rear corners at hypot(0.638, 0.2425) = 0.683 m),
                but the circle then also reaches 0.68 m AHEAD of the IMU,
                0.51 m past the robot's nose -- where a real obstacle could be.

HOW IT IS SCORED
    rtabmap-export --map --opt 3 re-assembles the grid from the stored local
    grids along the odometry poses (= the optimised poses: these graphs have no
    closures) with the parameter overridden on the command line.  --opt 2 would
    return the map already saved in the database and ignore the override, so
    radius 0 is re-assembled too and is the baseline the others are compared
    with.  Each grid is scored by script/validate_map.py exactly as
    eval_realistic_maps.py does: same world, same independent probe route, same
    fitted placement.

Writes output/compare/simulation/exp4-realistic-maps/footprint_clearing.csv
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_realistic_maps as E  # noqa: E402

SHORT = {"map_realistic": "U alt", "dataset_allsensor_realistic_map": "U prior",
         "map_realistic_01": "I alt", "dataset_allsensor_realistic_01_map": "I prior"}


def export(db, radius, out_dir, name):
    subprocess.run(["rtabmap-export", "--map", "--opt", "3", "--output_dir", out_dir, "--output", name,
                    "--GridGlobal/FootprintRadius", "%.3f" % radius, db],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pgm = os.path.join(out_dir, name + ".pgm")
    if not os.path.exists(pgm):
        raise SystemExit("rtabmap-export wrote no %s" % pgm)
    return pgm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radii", default="0,0.30,0.50,0.68")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    radii = [float(r) for r in a.radii.split(",")]
    tmp = tempfile.mkdtemp(prefix="fpclear.")
    rows = []
    try:
        for m in E.MAPS:
            if a.only and m["key"] != a.only:
                continue
            pose = open(E.placement_path(m["key"])).read().strip()
            for r in radii:
                name = "%s_fp%03d" % (m["key"], round(100 * r))
                pgm = export(m["db"], r, tmp, name)
                js = os.path.join(tmp, name + ".json")
                subprocess.run([sys.executable, os.path.join(E.ROOT, "script/validate_map.py"), pgm,
                                "--world", m["world"], "--gt", m["probe"], "--map-pose", pose,
                                "--out", os.path.join(tmp, name + ".png"), "--json", js],
                               check=True, stdout=subprocess.DEVNULL, cwd=E.ROOT)
                d = json.load(open(js))
                row = dict(map=SHORT[m["key"]], radius=r, occupied=d["occupied_cells"],
                           precision=d["precision"], recall_seen=d["recall_seen"],
                           false_obstacle=d["false_obstacle"], unmapped=d.get("unmapped", float("nan")))
                rows.append(row)
                print("%-6s r=%.2f  occupied %6d  precision %.1f %%  recall %.1f %%  route blocked %.2f %%"
                      % (row["map"], r, row["occupied"], 100 * row["precision"],
                         100 * row["recall_seen"], 100 * row["false_obstacle"]), flush=True)
                shutil.copy(os.path.join(tmp, name + ".png"),
                            os.path.join(E.OUT, "footprint_%s.png" % name))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out = os.path.join(E.OUT, "footprint_clearing.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
