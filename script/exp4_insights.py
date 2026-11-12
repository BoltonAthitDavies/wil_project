#!/usr/bin/env python3
"""Experiment 4 insight analysis: WHERE the dense map is wrong, and why.

The raw write-up scores map/map_tiny.db once (precision 96.6 %, recall 89.5 % of
the observed surfaces, drivability).  This script asks what the 3.4 % of wrong
occupied cells and the 10.5 % of missed surfaces are, and whether the map's
quality follows anything a reader would propose: range from the route, how many
graph nodes observed a cell, the loop-closure correction, the structure type.
It also opens the four unregistered databases beside map_tiny.db and scores the
one that turns out to be a second map of the same world.

Method (same raster conventions as script/validate_map.py): the exported
occupancy grid is placed in the world by the fitted map pose, true structure is
the world file's obstacle footprints filled plus the wall ring, true SURFACES
are the one-cell boundary of that, and a cell is correct within TOL metres of
the truth.  Per-node observation counts come from the local occupancy cells
RTAB-Map stores per node (Data.obstacle_cells / empty_cells / ground_cells,
node frame), moved through the optimised poses that rtabmap-export writes.

Outputs (output/compare/simulation/exp4-insights/):
  cell_classes.tex        occupied cells by distance to truth, per map
  missed_by_obstacle.tex  recall per world object (observed surfaces), map_tiny
  bins_distance.tex       precision / recall by distance from the route
  bins_views.tex          precision / recall by number of observing nodes
  drivability.tex         footprint drivability of both maps on the recorded route
  databases.tex           the five databases: contents, fit, identification
  closures.tex            loop closures and the odometry-to-optimised correction
  fig_cells_<map>_light.png   map overview: correct / thickened / phantom / missed
  fig_bins_light.png          precision and recall against range and view count
  fig_graph_light.png         odometry vs optimised node poses, closures marked
  insights.json               every number quoted in the document
"""
import json
import math
import os
import shutil
import sqlite3
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, label as cc_label
from matplotlib.path import Path as MplPath

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import viewer                                   # noqa: E402
from fit_map_pose import export_db, read_pgm, read_yaml, edge_points   # noqa: E402
from validate_map import FOOTPRINT, load_gt     # noqa: E402

import matplotlib                               # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # noqa: E402
from matplotlib.lines import Line2D             # noqa: E402

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = ROOT / "output" / "compare" / "simulation" / "exp4-insights"
WORLD_TINY = ROOT / "aws-robomaker-small-warehouse-world/worlds/small_warehouse/small_warehouse_static_tinymap.world"
GT_TINY = ROOT / "dataset/gt/dataset_map_tiny/ground_truth.csv"   # canonical since 6 Nov 2026; was output_vins/simulation/dataset_map_tiny/
TOL = 0.25            # metres, as validate_map.py
NEAR = 0.50           # beyond TOL but within NEAR: boundary thickening; beyond: phantom
LEVER = 0.338         # IMU ahead of base_footprint; the map frame is the IMU body frame
MAPS = {              # map pose from fit_map_pose.py (22 October 2026 re-fit)
    "map_tiny": dict(db=ROOT / "map/map_tiny/map_tiny.db", pose=(1.85, 8.60, -89.9), closures=None),  # moved into map_tiny/ on 6 Nov 2026
    "map_02":   dict(db=ROOT / "map/map_02.db",   pose=(1.85, 8.50, -179.8), closures=None),
    # Realistic-warehouse maps (26 October 2026), fitted by eval_realistic_maps.py
    # step 1 with the realistic spawn; see exp4-realistic-maps/*.mappose.
    "map_realistic":    dict(db=ROOT / "map/map_realistic/map_realistic.db",       pose=(5.30, 19.50, -179.9), closures=None),
    "map_realistic_01": dict(db=ROOT / "map/map_realistic_01/map_realistic_01.db", pose=(5.15, 20.45, 179.4),  closures=None),
    # 6 Nov 2026 rebuilds of the same two mapping bags under the current pipeline configuration
    # (DetectionRate 1 Hz, Grid/RangeMax 10 m, Grid/MaxGroundHeight 0.10 m; October: 4 Hz, 8 m, 0.29 m).
    # Placements from exp4-realistic-maps/*.mappose; the U one fitted with --yaw-near -180.
    "dataset_allsensor_realistic_map":    dict(db=ROOT / "map/dataset_allsensor_realistic_map/dataset_allsensor_realistic_map.db",       pose=(5.20, 19.80, -180.1), closures=None),
    "dataset_allsensor_realistic_01_map": dict(db=ROOT / "map/dataset_allsensor_realistic_01_map/dataset_allsensor_realistic_01_map.db", pose=(4.95, 20.40, -179.9), closures=None),
}
# --realistic: each map against its own world, probed by the static twin of its
# layout (the recording that did NOT build it), as the raw write-up scores them.
WORLDS_STATIC = ROOT / "aws-robomaker-small-warehouse-world/worlds/small_warehouse"
REALISTIC = [
    dict(name="map_realistic",    world=WORLDS_STATIC / "small_warehouse_static_realistic_nofloortexture.world",
         gt=ROOT / "dataset/gt/dataset_allsensor_realistic_000/ground_truth.csv", probe="realistic_000", layout="U"),
    dict(name="map_realistic_01", world=WORLDS_STATIC / "small_warehouse_static_realistic_nofloortexture_01.world",
         gt=ROOT / "dataset/gt/dataset_allsensor_realistic_001/ground_truth.csv", probe="realistic_001", layout="I"),
    dict(name="dataset_allsensor_realistic_map", world=WORLDS_STATIC / "small_warehouse_static_realistic_nofloortexture.world",
         gt=ROOT / "dataset/gt/dataset_allsensor_realistic_000/ground_truth.csv", probe="realistic_000", layout="U"),
    dict(name="dataset_allsensor_realistic_01_map", world=WORLDS_STATIC / "small_warehouse_static_realistic_nofloortexture_01.world",
         gt=ROOT / "dataset/gt/dataset_allsensor_realistic_001/ground_truth.csv", probe="realistic_001", layout="I"),
]
# Databases of the tiny-map insight. map_01/map_02/map_02_vins/map_03 were deleted
# with the 6 Nov 2026 dataset decision (only map_tiny of the old maps remains), so the
# tiny path scores whatever of (map_tiny, map_02) exists and summarises only the
# databases on disk; the 22 Oct outputs that carried map_02 and map_03 are kept in git
# and under trash/output/exp4-insights-tiny-with-map_02_20261106/.
TINY_SCORED = ["map_tiny", "map_02"]
ALL_DBS = ["map_tiny", "map_01", "map_02", "map_02_vins", "map_03"]
# Display names for table first columns and headers. Empty on the tiny path, so its
# output is unchanged; main_realistic() fills it because the rebuilds' database names
# are 36 characters and overflow every table.
SHORT = {}
def nm(name):
    return SHORT.get(name, name)

DIST_BINS = [0, 1, 2, 3, 4, 6, 8, 99]
VIEW_BINS = [1, 2, 3, 5, 8, 99]


# ------------------------------------------------------------------ rtabmap io

def decode_mat(blob):
    """rtabmap compressData2: zlib payload followed by (rows, cols, type) ints."""
    rows, cols, typ = struct.unpack("<iii", blob[-12:])
    raw = zlib.decompress(blob[:-12])
    ch = (typ >> 3) + 1
    depth = typ & 7
    dt = {0: np.uint8, 4: np.int32, 5: np.float32, 6: np.float64}[depth]
    a = np.frombuffer(raw, dtype=dt)
    return a.reshape(rows * cols, ch) if ch > 1 else a.reshape(rows, cols)


def db_summary(name):
    db = ROOT / "map" / name / f"{name}.db"   # map/<name>/<name>.db since 6 Nov 2026
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    nodes = con.execute("select id, stamp, pose from Node order by id").fetchall()
    P = np.array([struct.unpack("<12f", b) for _, _, b in nodes]).reshape(-1, 3, 4)
    links = dict(con.execute("select type, count(*) from Link group by type").fetchall())
    closures = con.execute("select from_id, to_id from Link where type in (1,2,3) order by from_id").fetchall()
    created = con.execute("select time_enter from Admin").fetchone()[0]
    con.close()
    t = P[:, :, 3]
    return dict(name=name, size_mb=db.stat().st_size / 1e6, nodes=len(nodes),
                stamps=(nodes[0][1], nodes[-1][1]), created=created,
                path_m=float(np.linalg.norm(np.diff(t[:, :2], axis=0), axis=1).sum()),
                extent=(float(np.ptp(t[:, 0])), float(np.ptp(t[:, 1]))), z_range=(float(t[:, 2].min()), float(t[:, 2].max())),
                neighbour_links=links.get(0, 0), closures=closures, n_closures=len(closures), odom=P,
                stamps_all=np.array([s for _, s, _ in nodes]), ids=[i for i, _, _ in nodes])


def export_poses(db, tmp):
    """rtabmap-export --poses writes <stem>_poses.txt beside the database."""
    stem = Path(db).stem
    beside = Path(db).parent / f"{stem}_poses.txt"
    if beside.exists():
        raise SystemExit(f"{beside} exists; move it aside")
    subprocess.run(["rtabmap-export", "--poses", "--opt", "2", "--poses_format", "0", str(db)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    dst = Path(tmp) / beside.name
    shutil.move(beside, dst)
    return np.loadtxt(dst).reshape(-1, 3, 4)


def node_cells(db):
    """Per node: (id, local cells (N,3) of obstacle+empty+ground in the node frame)."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out = []
    for i, ob, em, gr in con.execute("select id, obstacle_cells, empty_cells, ground_cells from Data order by id"):
        parts = [decode_mat(b)[:, :3] for b in (ob, em, gr) if b is not None]
        out.append((i, np.vstack(parts) if parts else np.zeros((0, 3))))
    con.close()
    return out


# ------------------------------------------------------------------- raster

class Scene:
    """Common world raster: truth, surfaces, per-object labels, route distance."""

    def __init__(self, world, res, gt_xy, pad=1.0):
        obst, wall = viewer.load_obstacles(str(world))
        self.polys = [(lab, np.asarray(p, float)) for lab, p, _ in obst]
        x0, y0, x1, y1 = wall
        self.wall = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        self.res = res
        self.lo = np.array([x0 - pad, y0 - pad])
        self.hi = np.array([x1 + pad, y1 + pad])
        self.gw, self.gh = (np.ceil((self.hi - self.lo) / res).astype(int) + 1)
        yy, xx = np.mgrid[0:self.gh, 0:self.gw]
        self.cell_xy = np.stack([self.lo[0] + xx.ravel() * res, self.lo[1] + yy.ravel() * res], 1)
        T = np.zeros(self.gh * self.gw, bool)
        self.obj = np.full(self.gh * self.gw, -1, int)          # object index per true cell
        for k, (lab, q) in enumerate(self.polys):
            m = MplPath(q).contains_points(self.cell_xy)
            T |= m; self.obj[m] = k
        T = T.reshape(self.gh, self.gw)
        ring = self.rast(edge_points(self.wall, res * 0.5))
        self.obj = self.obj.reshape(self.gh, self.gw)
        self.obj[ring & ~T] = len(self.polys)                    # wall gets its own index
        self.T_occ = T | ring
        self.T_edge = self.T_occ & binary_dilation(~self.T_occ, np.ones((3, 3)))
        self.d_true = distance_transform_edt(~self.T_occ, sampling=res)
        self.route = self.rast(gt_xy)
        self.d_route = distance_transform_edt(~self.route, sampling=res)
        # outward normal of each true-surface cell: direction to its nearest free cell
        _, idx = distance_transform_edt(self.T_occ, return_indices=True)
        yy, xx = np.mgrid[0:self.gh, 0:self.gw]
        self.normal = np.stack([idx[1] - xx, idx[0] - yy], -1).astype(float)
        nrm = np.linalg.norm(self.normal, axis=-1, keepdims=True)
        self.normal = np.where(nrm > 0, self.normal / np.maximum(nrm, 1e-9), 0.0)

    def rast(self, pts):
        idx = np.round((pts - self.lo) / self.res).astype(int)
        ok = (idx[:, 0] >= 0) & (idx[:, 0] < self.gw) & (idx[:, 1] >= 0) & (idx[:, 1] < self.gh)
        g = np.zeros((self.gh, self.gw), bool)
        g[idx[ok, 1], idx[ok, 0]] = True
        return g

    def world_of(self, mask):
        r, c = np.nonzero(mask)
        return np.stack([self.lo[0] + c * self.res, self.lo[1] + r * self.res], 1)


def map_to_world(pts, pose):
    th = math.radians(pose[2]); c, s = math.cos(th), math.sin(th)
    return pts @ np.array([[c, -s], [s, c]]).T + pose[:2]


def score_map(name, scene, gt, tmp):
    spec = MAPS[name]
    pgm = export_db(str(spec["db"]), tmp)
    res, (ox, oy) = read_yaml(pgm.replace(".pgm", ".yaml"))
    img = read_pgm(pgm)
    h = img.shape[0]

    def to_world(mask):
        r, cc = np.nonzero(mask)
        p = np.stack([ox + (cc + 0.5) * res, oy + (h - 1 - r + 0.5) * res], 1)
        return map_to_world(p, spec["pose"])

    M_occ = scene.rast(to_world(img == 0))
    M_known = scene.rast(to_world(img != 205))
    d_map = distance_transform_edt(~M_occ, sampling=res)
    seen = scene.T_edge & binary_dilation(M_known, np.ones((3, 3)))
    missed = seen & (d_map > TOL)
    dt = scene.d_true
    correct = M_occ & (dt <= TOL)
    thick = M_occ & (dt > TOL) & (dt <= NEAR)
    phantom = M_occ & (dt > NEAR)

    # per-node observation count and first-observing node, from the stored local cells
    poses = export_poses(spec["db"], tmp)
    cells = node_cells(spec["db"])
    views = np.zeros((scene.gh, scene.gw), int)
    first = np.full((scene.gh, scene.gw), 10**6, int)
    for k, (nid, loc) in enumerate(cells):
        if k >= len(poses) or len(loc) == 0:
            continue
        R, t = poses[k][:, :3], poses[k][:, 3]
        w = (R @ loc.T).T + t
        g = scene.rast(map_to_world(w[:, :2], spec["pose"]))
        views += g
        first[g] = np.minimum(first[g], nid)
    first[views == 0] = -1

    # phantom components
    lab, n_comp = cc_label(phantom)
    comps = []
    for j in range(1, n_comp + 1):
        m = lab == j
        xy = scene.world_of(m)
        comps.append(dict(cells=int(m.sum()), x=float(xy[:, 0].mean()), y=float(xy[:, 1].mean()),
                          d_route=float(scene.d_route[m].mean()), d_true=float(dt[m].mean()),
                          views=float(np.median(views[m]))))
    comps.sort(key=lambda c: -c["cells"])

    # drivability on the recorded route (validate_map logic)
    fx = np.arange(FOOTPRINT[:, 0].min(), FOOTPRINT[:, 0].max() + res, res)
    fy = np.arange(FOOTPRINT[:, 1].min(), FOOTPRINT[:, 1].max() + res, res)
    fgx, fgy = np.meshgrid(fx, fy)
    foot = np.stack([fgx.ravel(), fgy.ravel()], 1)
    foot = foot[MplPath(FOOTPRINT).contains_points(foot)]
    cy, sy = np.cos(gt[:, 2]), np.sin(gt[:, 2])
    wxy = np.empty((len(gt), len(foot), 2))
    wxy[:, :, 0] = gt[:, None, 0] + np.outer(cy, foot[:, 0]) - np.outer(sy, foot[:, 1])
    wxy[:, :, 1] = gt[:, None, 1] + np.outer(sy, foot[:, 0]) + np.outer(cy, foot[:, 1])
    ij = np.round((wxy - scene.lo) / res).astype(int)
    np.clip(ij[:, :, 0], 0, scene.gw - 1, out=ij[:, :, 0]); np.clip(ij[:, :, 1], 0, scene.gh - 1, out=ij[:, :, 1])
    hit = M_occ[ij[:, :, 1], ij[:, :, 0]].any(1)
    frac_known = M_known[ij[:, :, 1], ij[:, :, 0]].mean(1)
    gi = np.round((gt[:, :2] - scene.lo) / res).astype(int)
    clear = d_map[gi[:, 1].clip(0, scene.gh - 1), gi[:, 0].clip(0, scene.gw - 1)]

    # incidence angle of the nearest graph node's line of sight on each true surface cell
    P = map_to_world(poses[:, :2, 3], spec["pose"])
    r, c = np.nonzero(seen)
    cxy = np.stack([scene.lo[0] + c * res, scene.lo[1] + r * res], 1)
    dd = cxy[:, None, :] - P[None, :, :]
    jn = np.argmin(np.linalg.norm(dd, axis=2), axis=1)
    v = -dd[np.arange(len(r)), jn]                       # from the cell towards the sensor
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    nrm = scene.normal[r, c]
    cosang = np.clip(np.abs((v * nrm).sum(1)), 0, 1)
    angle = np.full((scene.gh, scene.gw), np.nan)
    angle[r, c] = np.degrees(np.arccos(cosang))          # 0 = frontal, 90 = grazing
    # contiguous runs of blocked poses on the route
    segs = []
    i = 0
    while i < len(hit):
        if hit[i]:
            j = i
            while j + 1 < len(hit) and hit[j + 1]:
                j += 1
            segs.append(dict(i0=i, i1=j, n=j - i + 1, x=float(gt[i:j + 1, 0].mean()), y=float(gt[i:j + 1, 1].mean()),
                             clear_min=float(clear[i:j + 1].min())))
            i = j + 1
        else:
            i += 1
    return dict(name=name, res=res, M_occ=M_occ, M_known=M_known, d_map=d_map, seen=seen, missed=missed, angle=angle, segs=segs,
                correct=correct, thick=thick, phantom=phantom, views=views, first=first, comps=comps,
                poses=poses, n_occ=int(M_occ.sum()),
                precision=float((dt[M_occ] <= TOL).mean()),
                recall_seen=float((d_map[seen] <= TOL).mean()),
                recall_all=float((d_map[scene.T_edge] <= TOL).mean()),
                driv=dict(poses=len(gt), blocked=float(hit.mean()), unmapped=float((frac_known < 0.5).mean()),
                          clear_med=float(np.median(clear)), clear_p05=float(np.percentile(clear, 5)),
                          hit=hit, clear=clear))


# -------------------------------------------------------------------- tables

def f(v, p=1):
    return "---" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{p}f}"


def tex_cell_classes(scores, path, label="tab:insight4-classes", route="the recorded route"):
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption[Occupied map cells by distance to the true structure.]{Occupied map cells by their distance to the "
         r"nearest true structure (filled shelf footprints and the wall ring of the world file). ``Correct'' is within the "
         r"\SI{0.25}{\metre} tolerance of the chapter; ``thickened'' is a cell beyond the tolerance but within "
         r"\SI{0.50}{\metre} of a true surface, i.e.\ a surface drawn too thick; ``phantom'' is further than that from any "
         r"structure. The last columns describe the phantom cells: how many connected clusters they form, the size of the "
         r"largest, and the median distance of a phantom cell from " + route + ".}",
         r"\label{" + label + "}", r"\begin{tabular}{lrrrrrrr}", r"\toprule",
         r"Map & occupied & correct & thickened & phantom & clusters & largest & route dist. \\",
         r" & cells & [\si{\percent}] & [\si{\percent}] & [\si{\percent}] & & [cells] & med.\ [m] \\", r"\midrule"]
    for s in scores:
        n = s["n_occ"]; ph = s["phantom"]
        L.append(f"\\repo{{{nm(s['name'])}}} & {n} & {f(100*s['correct'].sum()/n)} & {f(100*s['thick'].sum()/n)} & "
                 f"{f(100*ph.sum()/n)} & {len(s['comps'])} & {s['comps'][0]['cells'] if s['comps'] else 0} & "
                 f"{f(float(np.median(s['scene'].d_route[ph])) if ph.any() else float('nan'), 2)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(L) + "\n")


def tex_missed_by_obstacle(s, scene, path):
    rows = []
    for k, (lab, q) in enumerate(list(scene.polys) + [("wall ring", None)]):
        m = scene.T_edge & (scene.obj == k)
        sn = m & s["seen"]
        if sn.sum() == 0:
            continue
        rec = float((s["d_map"][sn] <= TOL).mean())
        dr = float(np.median(scene.d_route[sn]))
        vw = float(np.median(s["views"][sn]))
        rows.append((lab.replace("aws_robomaker_warehouse_", ""), int(sn.sum()), int(m.sum()), rec, dr, vw))
    rows.sort(key=lambda r: r[3])
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption[Recall of the observed surfaces, per world object.]{Recall of \repo{map_tiny.db} on the observed "
         r"surfaces of each world object, worst first: surface cells the map observed, their share of the object's "
         r"surface, the share of them that the map reproduces within tolerance, the median distance of those cells from "
         r"the recorded route, and the median number of graph nodes that observed them.}",
         r"\label{tab:insight4-objects}", r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Object & observed & of surface & recall & route dist. & nodes \\",
         r" & [cells] & [\si{\percent}] & [\si{\percent}] & med.\ [m] & med. \\", r"\midrule"]
    for lab, n, tot, rec, dr, vw in rows:
        L.append(f"\\repo{{{lab}}} & {n} & {f(100*n/tot,0)} & {f(100*rec)} & {f(dr,1)} & {f(vw,0)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(L) + "\n")
    return rows


def binned(s, scene, key_raster, bins, metric):
    """precision or recall per bin of a per-cell quantity."""
    out = []
    if metric == "precision":
        sel, good = s["M_occ"], scene.d_true <= TOL
    else:
        sel, good = s["seen"], s["d_map"] <= TOL
    q = key_raster
    for a, b in zip(bins[:-1], bins[1:]):
        m = sel & (q >= a) & (q < b)
        out.append((a, b, int(m.sum()), float(good[m].mean()) if m.sum() else float("nan")))
    return out


def tex_bins(scores, scene, path, key, bins, label, unit, caption, tag, integer=False, short=None):
    # `short` maps a database name to the header text; the caption names the maps in full.
    short = short or {}
    L = [r"\begin{table}[htbp]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2pt}", r"\caption[" + caption.split(". ")[0] + ".]{" + caption + "}",
         r"\label{tab:insight4-" + tag + "}", (r"\resizebox{\textwidth}{!}{%" if len(scores) > 2 else ""), r"\begin{tabular}{l" + "rr" * (2 * len(scores)) + "}", r"\toprule",
         label + " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{\\repo{{{short.get(s['name'], s['name'])}}} precision}} & \\multicolumn{{2}}{{c}}{{\\repo{{{short.get(s['name'], s['name'])}}} recall}}" for s in scores) + r" \\",
         "{" + unit + "} & " + " & ".join(r"cells & [\si{\percent}]" for _ in range(2 * len(scores))) + r" \\", r"\midrule"]
    allb = []
    for s in scores:
        sc = s.get("scene", scene)                 # maps of different worlds carry their own scene
        allb.append((binned(s, sc, s[key] if key in s else getattr(sc, key), bins, "precision"),
                     binned(s, sc, s[key] if key in s else getattr(sc, key), bins, "recall")))
    for i, (a, b) in enumerate(zip(bins[:-1], bins[1:])):
        if b >= 99:
            lab = f"$\\geq {a}$"
        elif integer:
            lab = f"{a}" if b - a == 1 else f"{a}--{b - 1}"
        else:
            lab = f"{a}--{b}"
        cells = []
        for pb, rb in allb:
            cells += [str(pb[i][2]), f(100 * pb[i][3]), str(rb[i][2]), f(100 * rb[i][3])]
        L.append(lab + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}" + ("}" if len(scores) > 2 else ""), r"\end{table}"]
    path.write_text("\n".join(L) + "\n")
    return allb


ANGLE_BINS = [0, 30, 60, 91]


def tex_bins_angle(scores, path, label="tab:insight4-angle"):
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption[Recall of the observed surfaces by incidence angle.]{Recall of the observed surfaces by the "
         r"incidence angle of the nearest graph node's line of sight on the surface: \SI{0}{\degree} is a surface faced "
         r"head-on, \SI{90}{\degree} one seen edge-on along its own plane (the faces of neighbouring shelves across a "
         r"narrow gap, looked into from the aisle). Normals come from the world footprints; the viewing node is the "
         r"nearest optimised graph node.}",
         r"\label{" + label + "}", (r"\resizebox{\textwidth}{!}{%" if len(scores) > 2 else ""), r"\begin{tabular}{l" + "rr" * len(scores) + "}", r"\toprule",
         r"Incidence angle & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{\\repo{{{nm(s['name'])}}}}}" for s in scores) + r" \\",
         r"{[\si{\degree}]} & " + " & ".join(r"surface cells & recall [\si{\percent}]" for _ in scores) + r" \\", r"\midrule"]
    out = {}
    for a, b in zip(ANGLE_BINS[:-1], ANGLE_BINS[1:]):
        cells = []
        for s in scores:
            m = s["seen"] & (s["angle"] >= a) & (s["angle"] < b)
            rec = float((s["d_map"][m] <= TOL).mean()) if m.sum() else float("nan")
            out.setdefault(s["name"], []).append((a, b, int(m.sum()), rec))
            cells += [str(int(m.sum())), f(100 * rec)]
        L.append(f"{a}--{min(b, 90)} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}" + ("}" if len(scores) > 2 else ""), r"\end{table}"]
    path.write_text("\n".join(L) + "\n")
    return out


def tex_blocked(scores, path, label="tab:insight4-blocked", route=r"the recorded \repo{dataset_map_tiny} route"):
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption[Where the probe route collides with the maps.]{Contiguous stretches of " + route +
         r" on which the true footprint overlaps an occupied cell of the map: pose range, "
         r"number of poses, location, and the smallest clearance from the route centre to a mapped obstacle inside the stretch.}",
         r"\label{" + label + "}", r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Map & poses & count & $x$ [m] & $y$ [m] & min.\ clearance [m] \\", r"\midrule"]
    for s in scores:
        if not s["segs"]:
            L.append(f"\\repo{{{nm(s['name'])}}} & --- & 0 & --- & --- & --- \\\\")
        for g in sorted(s["segs"], key=lambda g: -g["n"])[:8]:
            L.append(f"\\repo{{{nm(s['name'])}}} & {g['i0']}--{g['i1']} & {g['n']} & {f(g['x'],1)} & {f(g['y'],1)} & {f(g['clear_min'],2)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(L) + "\n")


def tex_drivability(scores, path, label="tab:insight4-driv", caption=None, route_of=None):
    if caption is None:
        caption = (r"\caption[Drivability of both maps under the true footprint along the recorded tiny route.]{Drivability of "
                   r"both tiny-warehouse maps under the nav2 footprint stamped along the recorded route of "
                   r"\repo{dataset_map_tiny} (3032 poses). For \repo{map_tiny.db} this is the route that built the map; for "
                   r"\repo{map_02.db} it is an independent route through the same world, which the lost "
                   r"\repo{dataset_static_tiny_repeat} reference was meant to provide.}")
    if route_of is None:
        route_of = lambda name: 'built the map' if name == 'map_tiny' else 'independent'
    L = [r"\begin{table}[htbp]", r"\centering", r"\small", caption,
         r"\label{" + label + "}", r"\begin{tabular}{llrrrr}", r"\toprule",
         r"Map & route & false obstacle & unmapped & clearance med. & clearance p05 \\",
         r" & & [\si{\percent} of poses] & [\si{\percent}] & [m] & [m] \\", r"\midrule"]
    for s in scores:
        d = s["driv"]
        L.append(f"\\repo{{{nm(s['name'])}}} & {route_of(s['name'])} & "
                 f"{f(100*d['blocked'])} & {f(100*d['unmapped'])} & {f(d['clear_med'],2)} & {f(d['clear_p05'],2)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(L) + "\n")


def tex_databases(summ, fits, ident, path):
    L = [r"\begin{table}[htbp]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2pt}",
         r"\caption[The five RTAB-Map databases under \repo{map/}.]{The five RTAB-Map databases under \repo{map/}: "
         r"size, graph nodes, simulation-time span of the nodes, odometry path length and planar extent, vertical range "
         r"of the node poses, loop closures, the placement fit against the tiny world (agreement score and margin over "
         r"the best rival yaw, from \path{script/fit_map_pose.py}), and what the database is identified as. Only "
         r"\repo{map_tiny.db} is registered in the project records.}",
         r"\label{tab:insight4-dbs}", r"\begin{tabular}{lrrrrrrrlp{2.4cm}}", r"\toprule",
         r"Database & MB & nodes & stamps [s] & path [m] & extent [m] & $z$ [m] & closures & fit (agr., margin) & identified as \\", r"\midrule"]
    for s in summ:
        fit = fits.get(s["name"], "---")
        L.append(f"\\repo{{{s['name']}.db}} & {f(s['size_mb'])} & {s['nodes']} & {f(s['stamps'][0],0)}--{f(s['stamps'][1],0)} & "
                 f"{f(s['path_m'])} & {f(s['extent'][0],0)}$\\times${f(s['extent'][1],0)} & {f(s['z_range'][0])}..{f(s['z_range'][1])} & "
                 f"{s['n_closures']} & {fit} & {ident[s['name']]} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(L) + "\n")


def tex_closures(summ_tiny, s, scene, path):
    odom, opt = summ_tiny["odom"], s["poses"]
    n = min(len(odom), len(opt))
    d = np.linalg.norm(odom[:n, :2, 3] - opt[:n, :2, 3], axis=1)
    dz = odom[:n, 2, 3] - opt[:n, 2, 3]
    # precision of occupied cells by the first node that observed them
    first = s["first"]
    pre = s["M_occ"] & (first >= 1) & (first <= 18)
    post = s["M_occ"] & (first >= 19)
    prec_pre = float((scene.d_true[pre] <= TOL).mean()) if pre.any() else float("nan")
    prec_post = float((scene.d_true[post] <= TOL).mean()) if post.any() else float("nan")
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption[Loop closures and the correction they applied, \repo{map_tiny.db}.]{Loop closures of "
         r"\repo{map_tiny.db} and the correction the graph optimisation applied: planar and vertical distance between "
         r"each node's odometry pose and its optimised pose, by node range, and the precision of the occupied cells "
         r"first observed before and after the revisit that closed the loop (nodes 1--18 against 19--29).}",
         r"\label{tab:insight4-closures}", r"\begin{tabular}{lrrr}", r"\toprule",
         r"Quantity & nodes 1--18 & nodes 19--29 & all \\", r"\midrule",
         f"planar correction, median [m] & {f(float(np.median(d[:18])),2)} & {f(float(np.median(d[18:n])),2)} & {f(float(np.median(d)),2)} \\\\",
         f"planar correction, max [m] & {f(float(d[:18].max()),2)} & {f(float(d[18:n].max()),2)} & {f(float(d.max()),2)} \\\\",
         f"vertical correction, max [m] & {f(float(np.abs(dz[:18]).max()),2)} & {f(float(np.abs(dz[18:n]).max()),2)} & {f(float(np.abs(dz).max()),2)} \\\\",
         f"occupied cells first seen & {int(pre.sum())} & {int(post.sum())} & {int(s['M_occ'].sum())} \\\\",
         f"precision of those cells [\\si{{\\percent}}] & {f(100*prec_pre)} & {f(100*prec_post)} & {f(100*s['precision'])} \\\\",
         r"\midrule",
         r"\multicolumn{4}{l}{closure links (from $\rightarrow$ to): " + ", ".join(f"{a}$\\rightarrow${b}" for a, b in summ_tiny["closures"]) + r"} \\",
         r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(L) + "\n")
    return dict(d=d, dz=dz, prec_pre=prec_pre, prec_post=prec_post, n_pre=int(pre.sum()), n_post=int(post.sum()))


# -------------------------------------------------------------------- figures

def fig_cells(s, scene, gt, path, title):
    fig, ax = plt.subplots(figsize=(11, 7.2))
    for lab, q in scene.polys:
        ax.fill(q[:, 0], q[:, 1], color="#e4e6ea", ec="#9aa0a6", lw=0.6, zorder=1)
    w = scene.wall
    ax.plot(w[[0, 1, 2, 3, 0], 0], w[[0, 1, 2, 3, 0], 1], color="#555", lw=1.4, zorder=1)
    for mask, col, lab, sz in ((s["correct"], "#458554", "occupied, correct", 1.2),
                               (s["thick"], "#cc8319", "occupied, thickened (0.25--0.50 m)", 4),
                               (s["phantom"], "#b00020", "occupied, phantom (>0.50 m)", 6),
                               (s["missed"], "#2a78d6", "true surface observed but missed", 3)):
        xy = scene.world_of(mask)
        ax.scatter(xy[:, 0], xy[:, 1], s=sz, color=col, label=f"{lab} ({mask.sum()})", zorder=3 if sz < 4 else 4, lw=0)
    ax.plot(gt[:, 0], gt[:, 1], color="#888", lw=0.8, label="recorded route (base)", zorder=2)
    P = map_to_world(s["poses"][:, :2, 3], MAPS[s["name"]]["pose"])
    ax.plot(P[:, 0], P[:, 1], "o-", color="#a347b2", ms=3, lw=0.6, label="graph nodes (optimised)", zorder=5)
    for c in s["comps"][:5]:
        ax.annotate(f"{c['cells']}", (c["x"], c["y"]), fontsize=7, color="#b00020", xytext=(4, 4), textcoords="offset points")
    ax.set_aspect("equal"); ax.set_xlabel("world x [m]"); ax.set_ylabel("world y [m]")
    ax.set_title(title, fontsize=10, loc="left"); ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="lower left", markerscale=3, framealpha=0.9)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def fig_bins(bins_d, bins_v, names, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    cols = {"map_tiny": "#2a78d6", "map_02": "#cc8319", "map_realistic": "#458554", "map_realistic_01": "#a347b2",
            "dataset_allsensor_realistic_map": "#8fd19e", "dataset_allsensor_realistic_01_map": "#d49be0"}
    for ax, (bins, allb, xl) in zip(axes, ((DIST_BINS, bins_d, "distance from the recorded route [m]"),
                                            (VIEW_BINS, bins_v, "graph nodes that observed the cell"))):
        x = np.arange(len(bins) - 1)
        labs = [(f"$\\geq${a}" if b >= 99 else (f"{a}" if (bins is VIEW_BINS and b - a == 1) else (f"{a}--{b - 1}" if bins is VIEW_BINS else f"{a}--{b}"))) for a, b in zip(bins[:-1], bins[1:])]
        for k, (name, (pb, rb)) in enumerate(zip(names, allb)):
            ax.plot(x, [100 * p[3] for p in pb], "o-", color=cols[name], label=f"{name} precision")
            ax.plot(x, [100 * r[3] for r in rb], "s--", color=cols[name], label=f"{name} recall (observed surfaces)")
        ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8); ax.set_xlabel(xl); ax.set_ylabel("[%]")
        ax.set_ylim(70, 101); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Does map quality follow range from the route or the number of observing nodes?", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(path, dpi=160); plt.close(fig)


def fig_graph(summ, s, path):
    odom = map_to_world(summ["odom"][:, :2, 3], MAPS["map_tiny"]["pose"])
    opt = map_to_world(s["poses"][:, :2, 3], MAPS["map_tiny"]["pose"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax = axes[0]
    ax.plot(odom[:, 0], odom[:, 1], "o-", color="#888", ms=3, lw=0.8, label="odometry poses (VINS-Fusion)")
    ax.plot(opt[:, 0], opt[:, 1], "o-", color="#a347b2", ms=3, lw=0.8, label="optimised poses")
    for a, b in summ["closures"]:
        if a - 1 < len(opt) and b - 1 < len(opt):
            ax.plot([opt[a - 1, 0], opt[b - 1, 0]], [opt[a - 1, 1], opt[b - 1, 1]], color="#b00020", lw=1.2)
    ax.plot([], [], color="#b00020", lw=1.2, label="loop-closure link")
    for i in range(0, len(opt), 3):
        ax.annotate(str(i + 1), opt[i], fontsize=6, color="#a347b2", xytext=(3, 3), textcoords="offset points")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.08), ncol=3); ax.set_xlabel("world x [m]"); ax.set_ylabel("world y [m]")
    ax.set_title("map_tiny.db: pose graph before and after optimisation", fontsize=9, loc="left")
    n = min(len(odom), len(opt))
    d = np.linalg.norm(odom[:n] - opt[:n], axis=1); dz = summ["odom"][:n, 2, 3] - s["poses"][:n, 2, 3]
    axes[1].plot(np.arange(1, n + 1), d, "o-", color="#2a78d6", ms=3, label="planar")
    axes[1].plot(np.arange(1, n + 1), np.abs(dz), "s--", color="#cc8319", ms=3, label="vertical")
    axes[1].set_xlabel("node"); axes[1].set_ylabel("|odometry - optimised| [m]"); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)
    axes[1].set_title("correction applied per node", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


# ----------------------------------------------------------------------- main

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="exp4ins.")
    try:
        gt = load_gt(str(GT_TINY))
        res = 0.05
        scene = Scene(WORLD_TINY, res, gt[:, :2])
        scores = []
        tiny_names = [n for n in TINY_SCORED if Path(MAPS[n]["db"]).exists()]
        all_dbs = [n for n in ALL_DBS if Path(ROOT / "map" / n / (n + ".db")).exists() or Path(ROOT / "map" / (n + ".db")).exists()]
        print("tiny path: scoring", tiny_names, "| summarising", all_dbs)
        for name in tiny_names:
            s = score_map(name, scene, gt, tmp); s["scene"] = scene; scores.append(s)
            print(f"{name}: occ {s['n_occ']} precision {100*s['precision']:.1f} recall(seen) {100*s['recall_seen']:.1f} "
                  f"recall(all) {100*s['recall_all']:.1f} | correct {s['correct'].sum()} thick {s['thick'].sum()} "
                  f"phantom {s['phantom'].sum()} in {len(s['comps'])} clusters | missed {s['missed'].sum()} | "
                  f"driv blocked {100*s['driv']['blocked']:.1f}% unmapped {100*s['driv']['unmapped']:.1f}% clear med {s['driv']['clear_med']:.2f}")
        summ = [db_summary(n) for n in all_dbs]
        fits = {"map_tiny": "0.387, +61\\,\\%", "map_02": "0.385, +54\\,\\%",
                "map_03": "0.19, +7\\,\\% (large worlds; weak)", "map_01": "---", "map_02_vins": "---"}
        ident = {"map_tiny": "the scored tiny map (registered); built on the \\repo{dataset_map_tiny} route",
                 "map_02": "a second tiny-warehouse map on a different route (21 Sep)",
                 "map_03": "a 37 m straight run that left the floor: the lost large-warehouse attempt, unplaceable",
                 "map_01": "aborted start (2 nodes)", "map_02_vins": "aborted start (2 nodes)"}
        tex_cell_classes(scores, OUT / "cell_classes.tex")
        obj_rows = tex_missed_by_obstacle(scores[0], scene, OUT / "missed_by_obstacle.tex")
        bins_d = tex_bins(scores, scene, OUT / "bins_distance.tex", "d_route", DIST_BINS, "Distance from route", "[m]",
                          "Precision and recall of the observed surfaces by distance from the recorded route "
                          "(\\repo{dataset_map_tiny}, the route that built \\repo{map_tiny.db}; for \\repo{map_02.db} the same "
                          "route is only a proxy for its own, unknown route). Stereo depth error grows with range, so a "
                          "range effect would show as falling precision and recall with distance.", "dist")
        bins_v = tex_bins(scores, scene, OUT / "bins_views.tex", "views", VIEW_BINS, "Observing nodes", "",
                          "Precision and recall of the observed surfaces by the number of graph nodes whose stored local "
                          "grid covers the cell. A map that improves with repeated observation would show both rising "
                          "with the count.", "views", integer=True)
        tex_drivability(scores, OUT / "drivability.tex")
        bins_a = tex_bins_angle(scores, OUT / "bins_angle.tex")
        tex_blocked(scores, OUT / "blocked_segments.tex")
        tex_databases(summ, fits, ident, OUT / "databases.tex")
        clos = tex_closures(summ[0], scores[0], scene, OUT / "closures.tex")
        fig_cells(scores[0], scene, gt, OUT / "fig_cells_map_tiny_light.png",
                  "map_tiny.db against the tiny world: what the wrong cells are and where")
        if len(scores) > 1:
            fig_cells(scores[1], scene, gt, OUT / "fig_cells_map_02_light.png",
                      "map_02.db (unregistered) against the tiny world, placed by its own fit")
        fig_bins(bins_d, bins_v, tiny_names, OUT / "fig_bins_light.png")
        fig_graph(summ[0], scores[0], OUT / "fig_graph_light.png")
        # node-vs-ground-truth identity test (which bag built the map)
        ident_test = {}
        summ_by = {sm["name"]: sm for sm in summ} if summ and "name" in summ[0] else dict(zip(all_dbs, summ))
        for s in scores:
            sm = summ_by[s["name"]]
            g = np.loadtxt(GT_TINY, delimiter=",")
            gt_t = g[:, 0] / 1e9; yaw = gt[:, 2]
            gix, giy = gt[:, 0] + LEVER * np.cos(yaw), gt[:, 1] + LEVER * np.sin(yaw)
            P = map_to_world(s["poses"][:, :2, 3], MAPS[s["name"]]["pose"])
            errs = []
            for k in range(len(P)):
                j = int(np.argmin(np.abs(gt_t - sm["stamps_all"][k])))
                errs.append(math.hypot(P[k, 0] - gix[j], P[k, 1] - giy[j]))
            ident_test[s["name"]] = dict(median_m=float(np.median(errs)), max_m=float(np.max(errs)))
        out = dict(maps={s["name"]: dict(n_occ=s["n_occ"], precision=s["precision"], recall_seen=s["recall_seen"],
                                         recall_all=s["recall_all"], correct=int(s["correct"].sum()), thick=int(s["thick"].sum()),
                                         phantom=int(s["phantom"].sum()), clusters=len(s["comps"]), comps=s["comps"][:8],
                                         missed=int(s["missed"].sum()), driv={k: v for k, v in s["driv"].items() if k not in ("hit", "clear")})
                         for s in scores},
                   objects=obj_rows, closures=dict(prec_pre=clos["prec_pre"], prec_post=clos["prec_post"], n_pre=clos["n_pre"], n_post=clos["n_post"],
                                                   planar_med=float(np.median(clos["d"])), planar_max=float(clos["d"].max()), vert_max=float(np.abs(clos["dz"]).max())),
                   bins_distance={n: [(p[0], p[1], p[2], p[3], r[2], r[3]) for p, r in zip(*b)] for n, b in zip(tiny_names, bins_d)},
                   bins_views={n: [(p[0], p[1], p[2], p[3], r[2], r[3]) for p, r in zip(*b)] for n, b in zip(tiny_names, bins_v)},
                   identity_test=ident_test, bins_angle=bins_a,
                   blocked={s["name"]: s["segs"] for s in scores},
                   databases=[{k: v for k, v in sm.items() if k not in ("odom", "stamps_all", "ids")} for sm in summ])
        (OUT / "insights.json").write_text(json.dumps(out, indent=1, default=float))
        print("identity test:", ident_test)
        print("closures:", {k: v for k, v in out["closures"].items()})
        print("phantom clusters map_tiny:", out["maps"]["map_tiny"]["comps"][:5])
        print("objects worst:", obj_rows[:4])
        print("angle bins:", {k: [(a, b, n, round(100*r, 1)) for a, b, n, r in v] for k, v in bins_a.items()})
        for s in scores:
            print(s["name"], "blocked segments:", [(g["i0"], g["i1"], g["n"], round(g["x"], 1), round(g["y"], 1), round(g["clear_min"], 2)) for g in sorted(s["segs"], key=lambda g: -g["n"])[:6]])
        for nm in tiny_names:
            print(nm, "bins distance:", [(a, b, n, round(100*p, 1), m, round(100*r, 1)) for a, b, n, p, m, r in out["bins_distance"][nm]])
            print(nm, "bins views:", [(a, b, n, round(100*p, 1), m, round(100*r, 1)) for a, b, n, p, m, r in out["bins_views"][nm]])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main_realistic():
    """Insight 6: do the realistic-warehouse maps fail by the tiny map's mechanism?
    Writes *_realistic.* beside the tiny outputs; never rewrites them."""
    SHORT.update({"map_realistic": "U-A", "map_realistic_01": "I-A",
                  "dataset_allsensor_realistic_map": "U-B", "dataset_allsensor_realistic_01_map": "I-B"})
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="exp4ins.")
    try:
        res = 0.05
        scores = []
        for spec in REALISTIC:
            gt = load_gt(str(spec["gt"]))
            scene = Scene(spec["world"], res, gt[:, :2])
            s = score_map(spec["name"], scene, gt, tmp); s["scene"] = scene; s["gt"] = gt; s["probe"] = spec["probe"]; s["layout"] = spec["layout"]
            scores.append(s)
            print(f"{spec['name']} ({spec['layout']}, probe {spec['probe']}): occ {s['n_occ']} precision {100*s['precision']:.1f} "
                  f"recall(seen) {100*s['recall_seen']:.1f} recall(all) {100*s['recall_all']:.1f} | correct {s['correct'].sum()} "
                  f"thick {s['thick'].sum()} phantom {s['phantom'].sum()} in {len(s['comps'])} clusters | missed {s['missed'].sum()} | "
                  f"driv blocked {100*s['driv']['blocked']:.1f}% unmapped {100*s['driv']['unmapped']:.1f}% clear med {s['driv']['clear_med']:.2f} p05 {s['driv']['clear_p05']:.2f}")
        tex_cell_classes(scores, OUT / "cell_classes_realistic.tex", label="tab:insight4-classes-realistic",
                         route="the independent probe route")
        bins_v = tex_bins(scores, scores[0]["scene"], OUT / "bins_views_realistic.tex", "views", VIEW_BINS, "Observing nodes", "",
                          "Precision and recall of the observed surfaces by the number of graph nodes whose stored local "
                          "grid covers the cell, for the two realistic-warehouse maps (U is \\repo{map_realistic}, I is \\repo{map_realistic_01}). The tiny-map contrast was 91 to 99 "
                          "percent precision and 78 to 97 percent recall from one node to three or more.", "views-realistic", integer=True,
                          short={"map_realistic": "U-A", "map_realistic_01": "I-A", "dataset_allsensor_realistic_map": "U-B", "dataset_allsensor_realistic_01_map": "I-B"})
        bins_d = tex_bins(scores, scores[0]["scene"], OUT / "bins_distance_realistic.tex", "d_route", DIST_BINS, "Distance from route", "[m]",
                          "Precision and recall of the observed surfaces by distance from the independent probe route "
                          "(\\repo{realistic_000} for U, \\repo{map_realistic}; \\repo{realistic_001} for I, \\repo{map_realistic_01}). The probe is not the route "
                          "that built either map, so this is only a proxy for range from the mapping camera.", "dist-realistic",
                          short={"map_realistic": "U-A", "map_realistic_01": "I-A", "dataset_allsensor_realistic_map": "U-B", "dataset_allsensor_realistic_01_map": "I-B"})
        bins_a = tex_bins_angle(scores, OUT / "bins_angle_realistic.tex", label="tab:insight4-angle-realistic")
        tex_drivability(scores, OUT / "drivability_realistic.tex", label="tab:insight4-driv-realistic",
                        caption=(r"\caption[Drivability of the realistic-warehouse maps under the true footprint along their "
                                 r"independent probe routes.]{Drivability of the two realistic-warehouse maps under the nav2 "
                                 r"footprint stamped along the static-twin recording of each layout, which did not build the map "
                                 r"(\repo{realistic_000}, 6278 poses, for the U map; \repo{realistic_001}, 4847 poses, for the I map). "
                                 r"Same method as the raw write-up, which it reproduces.}"),
                        route_of=lambda name: "independent")
        tex_blocked(scores, OUT / "blocked_realistic.tex", label="tab:insight4-blocked-realistic",
                    route="the independent probe route of each layout")
        for s in scores:
            fig_cells(s, s["scene"], s["gt"], OUT / f"fig_cells_{s['name']}_light.png",
                      f"{s['name']}.db against the {s['layout']}-layout realistic world: what the wrong cells are and where "
                      f"(route drawn: probe {s['probe']}, which did not build the map)")
        fig_bins(bins_d, bins_v, [s["name"] for s in scores], OUT / "fig_bins_realistic_light.png")
        # how many blocked poses sit on phantom clusters vs thickened surface (the Insight-4 question)
        on_phantom = {}
        for s in scores:
            ph_d = distance_transform_edt(~s["phantom"], sampling=res)
            th_d = distance_transform_edt(~s["thick"], sampling=res)
            gi = np.round((s["gt"][:, :2] - s["scene"].lo) / res).astype(int)
            gi[:, 0] = gi[:, 0].clip(0, s["scene"].gw - 1); gi[:, 1] = gi[:, 1].clip(0, s["scene"].gh - 1)
            hit = s["driv"]["hit"]
            near_ph = ph_d[gi[hit, 1], gi[hit, 0]] <= 0.60
            near_th = th_d[gi[hit, 1], gi[hit, 0]] <= 0.60
            # WHEN were the blocking phantoms laid down? If their first-observing node
            # is among the earliest, they date from the inertial-initialisation
            # transient (node 1 of these databases is not at identity; z settles over
            # the first nodes), which would make them removable by construction.
            blk = np.zeros((s["scene"].gh, s["scene"].gw), bool)
            blk[gi[hit, 1], gi[hit, 0]] = True
            near_blk = binary_dilation(blk, iterations=int(round(0.60 / res)))
            ph_blk = s["phantom"] & near_blk
            fb = s["first"][ph_blk]; fb = fb[fb > 0]
            fa = s["first"][s["phantom"]]; fa = fa[fa > 0]
            on_phantom[s["name"]] = dict(blocked=int(hit.sum()), near_phantom=int(near_ph.sum()),
                                         near_thick_only=int((near_th & ~near_ph).sum()),
                                         neither=int((~near_th & ~near_ph).sum()),
                                         blocking_phantom_cells=int(ph_blk.sum()),
                                         blocking_first_node_median=float(np.median(fb)) if len(fb) else float("nan"),
                                         blocking_first_le5=float((fb <= 5).mean()) if len(fb) else float("nan"),
                                         blocking_first_le10=float((fb <= 10).mean()) if len(fb) else float("nan"),
                                         all_phantom_first_le5=float((fa <= 5).mean()) if len(fa) else float("nan"),
                                         all_phantom_first_median=float(np.median(fa)) if len(fa) else float("nan"))
        out = dict(maps={s["name"]: dict(layout=s["layout"], probe=s["probe"], n_occ=s["n_occ"], precision=s["precision"],
                                         recall_seen=s["recall_seen"], recall_all=s["recall_all"], correct=int(s["correct"].sum()),
                                         thick=int(s["thick"].sum()), phantom=int(s["phantom"].sum()), clusters=len(s["comps"]),
                                         comps=s["comps"][:8], missed=int(s["missed"].sum()),
                                         driv={k: v for k, v in s["driv"].items() if k not in ("hit", "clear")},
                                         max_views=int(s["views"].max()))
                         for s in scores},
                   bins_views={s["name"]: [(p[0], p[1], p[2], p[3], r[2], r[3]) for p, r in zip(*b)] for s, b in zip(scores, bins_v)},
                   bins_distance={s["name"]: [(p[0], p[1], p[2], p[3], r[2], r[3]) for p, r in zip(*b)] for s, b in zip(scores, bins_d)},
                   bins_angle=bins_a, blocked={s["name"]: s["segs"] for s in scores}, blocked_on=on_phantom)
        (OUT / "insights_realistic.json").write_text(json.dumps(out, indent=1, default=float))
        for k, v in on_phantom.items():
            print(f"{k}: blocked {v['blocked']} | near phantom {v['near_phantom']} thick-only {v['near_thick_only']} neither {v['neither']} | "
                  f"blocking phantom cells {v['blocking_phantom_cells']}: first node median {v['blocking_first_node_median']:.0f}, "
                  f"share first-seen by node<=5 {100*v['blocking_first_le5']:.0f}% (<=10: {100*v['blocking_first_le10']:.0f}%) | "
                  f"ALL phantoms: first<=5 {100*v['all_phantom_first_le5']:.0f}%, median first node {v['all_phantom_first_median']:.0f}")
        print("angle bins:", {k: [(a, b, n, round(100*r, 1)) for a, b, n, r in v] for k, v in bins_a.items()})
        for s in scores:
            print(s["name"], "views bins:", [(a, b, n, round(100*p, 1), m, round(100*r, 1)) for a, b, n, p, m, r in out["bins_views"][s["name"]]])
            print(s["name"], "largest phantom clusters:", [(c["cells"], round(c["x"], 1), round(c["y"], 1), round(c["d_route"], 2), c["views"]) for c in s["comps"][:5]])
            print(s["name"], "blocked segments:", [(g["i0"], g["i1"], g["n"], round(g["x"], 1), round(g["y"], 1), round(g["clear_min"], 2)) for g in sorted(s["segs"], key=lambda g: -g["n"])[:6]])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main_realistic() if "--realistic" in sys.argv else main()
