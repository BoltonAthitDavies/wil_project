#!/usr/bin/env python3
"""Experiment 3 insight analysis: WHAT the pose error rises with.

Tests, on every Experiment 3 run of both testbeds, the mechanisms a reader would
propose for the error growth, each as a within-run contrast so that the run is
its own control:

  turning      does the planar error (and the heading error) grow faster while the
               ground-truth yaw rate exceeds TURN_DEG_S than on straights?
  object       in the dynamic recordings, does it grow faster while a movable object
  in view      is in the camera's view (detector boxes present) than when none is?
  features     does the tracked-feature count fall before the error onset, i.e. is
               the onset preceded by feature starvation?
  onset        what is the robot doing at the onset: turning or straight, object in
  context      view or not, feature level relative to the run's own median?

All error series come from analyse_exp3_timelines (start-anchored SE(3) alignment,
planar error, onset = 1 s sustained excess over 3x the static median of the same
estimator and testbed).  Growth rates are computed on the "tracking regime" only:
samples after the anchor window, before the first impossible-speed jump, and while
the error is below REGIME_MAX_M, because once the estimate has failed the error
series says nothing about the mechanism.  The object-in-view indicator for an
UNFILTERED run is borrowed from the class-filtered run on the same bag with the
longest detector log (the detector ran live on the same timestamps), which is a
proxy, not a measurement for that run.

Outputs (output/compare/simulation/exp3-insights/):
  insights.csv                     one row per run, every contrast
  contrast_<testbed>.tex           pooled contrasts per estimator x condition
  onset_context_<testbed>.tex      what the robot was doing at onset
  fig_turn_contrast_light.png      drift rate turning vs straight, per run
  fig_object_contrast_light.png    drift rate object-in-view vs not, per run
  fig_onset_epoch_light.png        features %, object area %, yaw rate around onset
"""
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vio_metrics as vm                      # noqa: E402
import analyse_exp3_timelines as tl           # noqa: E402
import map_underlay as mu                     # noqa: E402
from shapely.geometry import LineString, Polygon as SPolygon   # noqa: E402
from shapely.strtree import STRtree                             # noqa: E402

import matplotlib                             # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt               # noqa: E402

ROOT = Path(vm.ROOT)
OUT = ROOT / "output" / "compare" / "simulation" / "exp3-insights"
TURN_DEG_S = 10.0          # same as the timeline analysis
REGIME_MAX_M = 3.0         # growth is analysed only while the estimate is still tracking
MIN_SEG_S = 3.0            # a contrast needs at least this much time in each state
EPOCH = (-10.0, 5.0)       # window around the onset for the epoch figure
IN_VIEW_MIN_BOXES = 1
LEVER_SWEEP_M = (0.0, 0.17, 0.34, 0.50)   # forward offset of the IMU from the model origin: 0.338 m in the SDF
LEVER_FINE_M = tuple(np.round(np.arange(0.0, 0.601, 0.04), 2))
LEVER_SDF_M = 0.338
SUFFIX = ""                                 # set by --lever: outputs get this suffix
HFOV_HALF_DEG = 45.0       # simulated camera: fx = 480 px, width 960 px
TRAFFIC_RANGE_M = 20.0     # beyond this a carrier is a few pixels wide
TRAFFIC_WIDTH = {"Pallet": 1.2, "default": 0.6}   # nominal widths for the angular-size proxy

COL = {"VINS-Fusion": "#cc8319", "ORB-SLAM3": "#2a78d6",
       "VINS-Fusion+YOLO": "#e040fb", "ORB-SLAM3+YOLO": "#b00020"}


# ------------------------------------------------------------------ helpers

def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def heading_error(run, es, gt):
    """|yaw_est (aligned) - yaw_gt| in degrees on es['t'].  The alignment rotation
    of the planar error is reused, so the heading error is anchored the same way."""
    e = vm.load(os.path.join(run["dir"], "vio.csv"))
    t_all, q_all = e[0], e[2]
    sel = (t_all >= es["t"][0]) & (t_all <= es["t"][-1])
    t, q = t_all[sel], q_all[sel]
    yaw_est = np.degrees(np.unwrap(np.radians(vm.euler_zyx(vm.quat_to_R(q))[:, 0])))
    # rotation of the SE(3) fit: recover from pa vs raw p over the anchor window
    p_raw = e[1][sel]
    R, _ = vm.umeyama(p_raw[:10], es["pa"][:10]) if len(p_raw) >= 10 else (np.eye(3), 0)
    yaw_off = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    yaw_gt = np.degrees(np.interp(es["t"], gt["t"], gt["yaw_unwrapped"]))
    y_est = np.interp(es["t"], t, yaw_est) + yaw_off
    herr = wrap_deg(y_est - yaw_gt)
    herr = herr - herr[: max(5, min(50, len(herr) // 10))].mean()   # anchor at start
    return np.abs(wrap_deg(herr))


def regime_mask(es):
    """Samples in the tracking regime: after the anchor window, before the first
    jump, error below REGIME_MAX_M (first crossing ends the regime)."""
    t, err = es["t"], es["err"]
    m = t >= es["anchor"][1]
    if es["jump_t"] is not None and es["jump_t"] > es["anchor"][1]:
        m &= t < es["jump_t"]
    over = np.where(m & (err > REGIME_MAX_M))[0]
    if len(over):
        m &= t < t[over[0]]
    return m


def rate_contrast(t, y, state, m):
    """Growth rate of y (per s) while state is True vs False, over samples m.
    Returns (rate_true, rate_false, time_true, time_false) or None."""
    idx = np.where(m)[0]
    if len(idx) < 20:
        return None
    tt, yy, ss = t[idx], y[idx], state[idx]
    dt, dy = np.diff(tt), np.diff(yy)
    ok = (dt > 0) & (dt < 1.0)
    s_mid = ss[:-1] & ss[1:]
    n_mid = ~ss[:-1] & ~ss[1:]
    T1, T0 = dt[ok & s_mid].sum(), dt[ok & n_mid].sum()
    if T1 < MIN_SEG_S or T0 < MIN_SEG_S:
        return None
    return dy[ok & s_mid].sum() / T1, dy[ok & n_mid].sum() / T0, T1, T0


def donor_mask(runs, dataset):
    """Longest class-mode detector log on this bag: object-in-view proxy."""
    best, best_n = None, 0
    for r in runs:
        if r["dataset"] != dataset or not r["filtered"] or r["mode"] != "class":
            continue
        p = os.path.join(r["dir"], "yolo_mask.csv")
        if not os.path.exists(p):
            continue
        n = sum(int(float(row["matched"])) for row in csv.DictReader(open(p)))
        if n > best_n:
            best, best_n = r, n
    if best is None:
        return None
    rows = list(csv.DictReader(open(os.path.join(best["dir"], "yolo_mask.csv"))))
    t = np.array([int(r["timestamp_ns"]) for r in rows]) * 1e-9
    nb = np.array([int(float(r["n_boxes"])) for r in rows])
    cov = np.array([float(r["mask_coverage"]) for r in rows])
    return dict(t=t, n_boxes=nb, cov=cov, run=best["run"], system=best["system"])


def at(t_src, y_src, t_dst, hold=0.3, fill=0.0):
    """Sample-and-hold y_src onto t_dst; fill where no source sample within hold."""
    if t_src is None or len(t_src) == 0:
        return np.full(len(t_dst), fill)
    i = np.clip(np.searchsorted(t_src, t_dst) - 1, 0, len(t_src) - 1)
    out = y_src[i].astype(float)
    out[np.abs(t_src[i] - t_dst) > hold] = fill
    return out


def gt_boxes(dataset):
    """Per-frame ground-truth moving-object visibility for an extreme bag, from the
    Experiment 2 evaluation (projected simulator truth): time, number of moving
    objects in view, image-area fraction they cover."""
    if "nofloortexture_" not in dataset or "dynamic" not in dataset:
        return None
    p = ROOT / "output" / "yolo_eval" / ("dyn_" + dataset.split("nofloortexture_")[1]) / "yolo_eval.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p)))
    return dict(t=np.array([int(r["timestamp_ns"]) for r in rows]) * 1e-9,
                n=np.array([int(r["n_gt_moving"]) for r in rows]),
                area=np.array([float(r["gt_area_frac_moving"]) for r in rows]))


def frontend(run):
    """Per-frame front-end log: ORB-SLAM3 tracking state and map-match inliers,
    VINS-Fusion tracked-feature count (state is None for VINS)."""
    orb = run["system"].startswith("ORB")
    p = os.path.join(run["dir"], "tracking_frontend.csv" if orb else "frontend.csv")
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    if not rows:
        return None
    t = np.array([int(r["timestamp_ns"]) for r in rows]) * 1e-9
    if orb:
        return dict(t=t, state=np.array([int(float(r["tracking_state"] or 0)) for r in rows]),
                    n=np.array([float(r["map_matches_inliers"] or "nan") for r in rows]))
    return dict(t=t, state=None, n=np.array([float(r["feature_count"] or "nan") for r in rows]))


def mechanism_extreme(run, es, gb):
    """Within-run contrasts against the TRUE moving-object area in view (extreme
    bags only): tracking-loss share and jump rate in the top vs bottom tercile of
    moving area, and the rank correlation of matched features with moving area."""
    out = dict(mv_area_med=np.nan, lost_hi=np.nan, lost_lo=np.nan, rho_feat_area=np.nan,
               jump_rate_hi=np.nan, jump_rate_lo=np.nan, mv_n=0)
    fr = frontend(run)
    if gb is None or fr is None:
        return out
    sel = (fr["t"] >= es["t"][0]) & (fr["t"] <= es["t"][-1])
    if sel.sum() < 60:
        return out
    t, n = fr["t"][sel], fr["n"][sel]
    area = at(gb["t"], gb["area"], t, hold=0.1, fill=np.nan)
    ok = np.isfinite(area)
    if ok.sum() < 60:
        return out
    q1, q2 = np.nanpercentile(area, [33.3, 66.7])
    lo, hi = ok & (area <= q1), ok & (area >= q2)
    out["mv_area_med"] = 100 * float(np.nanmedian(area)); out["mv_n"] = int(ok.sum())
    if fr["state"] is not None:
        st = fr["state"][sel]
        lost = st != 2
        out["lost_hi"], out["lost_lo"] = float(lost[hi].mean()), float(lost[lo].mean())
        good = ok & (st == 2) & np.isfinite(n)
    else:
        good = ok & np.isfinite(n)
    if good.sum() >= 30 and np.nanstd(n[good]) > 0:
        out["rho_feat_area"] = float(stats.spearmanr(area[good], n[good])[0])
    # jumps of the estimate, assigned to the moving-area tercile at their time
    e = vm.load(os.path.join(run["dir"], "vio.csv"))
    te, pe = e[0], e[1]
    m = (te >= es["t"][0]) & (te <= es["t"][-1])
    ji = vm.find_jumps(te[m], pe[m])[0]
    tj = te[m][ji + 1] if len(ji) else np.array([])
    aj = at(gb["t"], gb["area"], tj, hold=0.1, fill=np.nan) if len(tj) else np.array([])
    dt = float(np.median(np.diff(t)))
    T_hi, T_lo = hi.sum() * dt, lo.sum() * dt
    if T_hi > MIN_SEG_S and T_lo > MIN_SEG_S:
        out["jump_rate_hi"] = float((aj >= q2).sum() / T_hi)
        out["jump_rate_lo"] = float((aj <= q1).sum() / T_lo)
    return out


_TRAFFIC, _OCC = {}, {}


def traffic(dataset):
    """Moving models of a realistic bag (extract_traffic.py): model -> (t, x, y)."""
    if dataset in _TRAFFIC:
        return _TRAFFIC[dataset]
    p = OUT / f"traffic_{tl.short(dataset)}.csv"
    out = None
    if p.exists():
        per = defaultdict(list)
        for r in csv.DictReader(open(p)):
            per[r["model"]].append((int(r["t_ns"]) * 1e-9, float(r["x"]), float(r["y"])))
        out = {k: np.array(v) for k, v in per.items()} or None
    _TRAFFIC[dataset] = out
    return out


def occluders(dataset):
    if dataset not in _OCC:
        fp = mu.footprints(dataset)
        polys = []
        if fp is not None:
            for _l, poly, _p in fp[0]:
                poly = np.asarray(poly, float)
                if len(poly) >= 3:
                    polys.append(SPolygon(poly).buffer(0))
        _OCC[dataset] = (polys, STRtree(polys) if polys else None)
    return _OCC[dataset]


def traffic_view(dataset, gt, tq):
    """Occlusion-aware visibility of the traffic from the robot at times tq:
    a mover is in view when it is within TRAFFIC_RANGE_M, within the horizontal
    field of view about the ground-truth heading, and the straight line from the
    robot to it crosses no obstacle footprint.  Returns n_in_view, nearest range,
    and an angular-size proxy (sum of nominal width / range) per time."""
    tr = traffic(dataset)
    if tr is None:
        return None
    polys, tree = occluders(dataset)
    rx = np.interp(tq, gt["t"], gt["p"][:, 0]); ry = np.interp(tq, gt["t"], gt["p"][:, 1])
    ryaw = np.interp(tq, gt["t"], gt["yaw_unwrapped"])
    n_in = np.zeros(len(tq), int); near = np.full(len(tq), np.inf); ang = np.zeros(len(tq))
    for name, arr in tr.items():
        w = TRAFFIC_WIDTH["Pallet"] if "Pallet" in name or "pallet" in name else TRAFFIC_WIDTH["default"]
        inside = (tq >= arr[0, 0]) & (tq <= arr[-1, 0])
        mx = np.interp(tq, arr[:, 0], arr[:, 1]); my = np.interp(tq, arr[:, 0], arr[:, 2])
        dx, dy = mx - rx, my - ry
        rng = np.hypot(dx, dy)
        bearing = np.degrees(wrap_deg_rad(np.arctan2(dy, dx) - ryaw))
        cand = inside & (rng <= TRAFFIC_RANGE_M) & (np.abs(bearing) <= HFOV_HALF_DEG) & (rng > 0.3)
        for i in np.where(cand)[0]:
            if tree is not None:
                seg = LineString([(rx[i], ry[i]), (mx[i], my[i])])
                hit = any(polys[j].intersects(seg) for j in tree.query(seg))
                if hit:
                    continue
            n_in[i] += 1; near[i] = min(near[i], rng[i]); ang[i] += w / rng[i]
    near[~np.isfinite(near)] = np.nan
    return dict(t=tq, n=n_in, near=near, ang=ang)


def wrap_deg_rad(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def traffic_contrasts(run, es, gt):
    out = dict(tr_share=np.nan, tr_rate_in=np.nan, tr_rate_out=np.nan, tr_T_in=np.nan, tr_T_out=np.nan,
               tr_lost_in=np.nan, tr_lost_out=np.nan, tr_feat_in=np.nan, tr_feat_out=np.nan,
               tr_jumps_in=np.nan, tr_jumps_out=np.nan, tr_n_models=0)
    tr = traffic(run["dataset"])
    if tr is None or tl.is_static(run):
        return out
    out["tr_n_models"] = len(tr)
    t, err = es["t"], es["err"]
    tq = np.arange(t[0], t[-1], 0.2)
    tv = traffic_view(run["dataset"], gt, tq)
    in_view_t = at(tv["t"], (tv["n"] > 0).astype(int), t, hold=0.25) > 0.5
    m = regime_mask(es)
    out["tr_share"] = float(in_view_t[m].mean()) if m.sum() else np.nan
    c = rate_contrast(t, err, in_view_t, m)
    if c:
        out["tr_rate_in"], out["tr_rate_out"], out["tr_T_in"], out["tr_T_out"] = c
    fr = frontend(run)
    if fr is not None:
        sel = (fr["t"] >= t[0]) & (fr["t"] <= t[-1])
        ft, fn = fr["t"][sel], fr["n"][sel]
        iv = at(tv["t"], (tv["n"] > 0).astype(int), ft, hold=0.25) > 0.5
        if iv.sum() >= 30 and (~iv).sum() >= 30:
            if fr["state"] is not None:
                lost = fr["state"][sel] != 2
                out["tr_lost_in"], out["tr_lost_out"] = float(lost[iv].mean()), float(lost[~iv].mean())
                okf = np.isfinite(fn) & (fr["state"][sel] == 2)
            else:
                okf = np.isfinite(fn)
            if (okf & iv).sum() >= 10 and (okf & ~iv).sum() >= 10:
                out["tr_feat_in"], out["tr_feat_out"] = float(np.nanmedian(fn[okf & iv])), float(np.nanmedian(fn[okf & ~iv]))
    e = vm.load(os.path.join(run["dir"], "vio.csv"))
    te, pe = e[0], e[1]
    mm = (te >= t[0]) & (te <= t[-1])
    ji = vm.find_jumps(te[mm], pe[mm])[0]
    tj = te[mm][ji + 1] if len(ji) else np.array([])
    jv = at(tv["t"], (tv["n"] > 0).astype(int), tj, hold=0.25) > 0.5 if len(tj) else np.array([], bool)
    dtq = 0.2
    T_in, T_out = (tv["n"] > 0).sum() * dtq, (tv["n"] == 0).sum() * dtq
    if T_in > MIN_SEG_S and T_out > MIN_SEG_S:
        out["tr_jumps_in"], out["tr_jumps_out"] = float(jv.sum() / T_in), float((~jv).sum() / T_out)
    return out


# ------------------------------------------------------------------- per run

def analyse_run(run, gt, es, thr, runs, donors):
    t, err = es["t"], es["err"]
    herr = heading_error(run, es, gt)
    yaw_rate = np.interp(t, gt["t"], gt["yaw_rate"])
    turning = yaw_rate > TURN_DEG_S
    m = regime_mask(es)
    row = dict(testbed=run["testbed"], dataset=tl.short(run["dataset"]), system=run["system"],
               run=run["run"], static=tl.is_static(run), filtered=run["filtered"], mode=run["mode"],
               lc=run["lc"], regime_s=float(np.ptp(t[m])) if m.sum() > 1 else 0.0)
    # turning contrast
    for key, y in (("pos", err), ("yaw", herr)):
        c = rate_contrast(t, y, turning, m)
        row[f"turn_{key}_rate_turn"], row[f"turn_{key}_rate_straight"] = (c[0], c[1]) if c else (np.nan, np.nan)
        row[f"turn_{key}_T_turn"], row[f"turn_{key}_T_straight"] = (c[2], c[3]) if c else (np.nan, np.nan)
    # object-in-view contrast (dynamic bags only)
    d = donors.get(run["dataset"])
    if d is not None and not tl.is_static(run):
        own = tl.filter_series(run) if run["filtered"] else None
        if own is not None and len(own["t"]) > 0.5 * len(d["t"]):
            # a filtered run carries its own log; prefer it
            nb_t, nb = own["t"], (own["cov"] > 0).astype(int)
            row["view_src"] = "own"
        else:
            nb_t, nb = d["t"], (d["n_boxes"] >= IN_VIEW_MIN_BOXES).astype(int)
            row["view_src"] = f"donor:{d['system']}/{d['run']}"
        in_view = at(nb_t, nb, t) > 0.5
        c = rate_contrast(t, err, in_view, m)
        row["view_rate_in"], row["view_rate_out"] = (c[0], c[1]) if c else (np.nan, np.nan)
        row["view_T_in"], row["view_T_out"] = (c[2], c[3]) if c else (np.nan, np.nan)
        row["view_share_regime"] = float(in_view[m].mean()) if m.sum() else np.nan
    else:
        row.update(view_src="", view_rate_in=np.nan, view_rate_out=np.nan, view_T_in=np.nan,
                   view_T_out=np.nan, view_share_regime=np.nan)
    # onset context
    on = tl.onset(t, err, thr)
    row["onset_s"] = on - t[0] if on is not None else np.nan
    fs = tl.feature_series(run)
    if on is not None:
        row["onset_yaw_rate"] = float(np.interp(on, gt["t"], gt["yaw_rate"]))
        row["onset_turning"] = bool(row["onset_yaw_rate"] > TURN_DEG_S)
        if d is not None and not tl.is_static(run):
            cov_at = at(d["t"], d["cov"], np.array([on]), fill=np.nan)[0]
            nb_at = at(d["t"], d["n_boxes"], np.array([on]), fill=np.nan)[0]
            row["onset_in_view"] = bool(nb_at >= IN_VIEW_MIN_BOXES) if np.isfinite(nb_at) else None
            row["onset_cov_pct"] = 100 * cov_at if np.isfinite(cov_at) else np.nan
        else:
            row["onset_in_view"], row["onset_cov_pct"] = None, np.nan
        if fs is not None and len(fs["t"]) > 10:
            pre = (fs["t"] >= on - 5.0) & (fs["t"] < on)
            base = (fs["t"] >= t[0]) & (fs["t"] < on - 5.0)
            row["onset_feat_pre5_pct"] = float(np.nanmedian(fs["pct"][pre])) if pre.sum() >= 3 else np.nan
            row["onset_feat_base_pct"] = float(np.nanmedian(fs["pct"][base])) if base.sum() >= 3 else np.nan
        else:
            row["onset_feat_pre5_pct"] = row["onset_feat_base_pct"] = np.nan
    else:
        row.update(onset_yaw_rate=np.nan, onset_turning=None, onset_in_view=None, onset_cov_pct=np.nan,
                   onset_feat_pre5_pct=np.nan, onset_feat_base_pct=np.nan)
    # feature-vs-growth correlation in the tracking regime (2 s bins)
    row["feat_growth_rho"], row["feat_growth_n"] = np.nan, 0
    if fs is not None and m.sum() > 40:
        tm, em = t[m], err[m]
        edges = np.arange(tm[0], tm[-1], 2.0)
        g, f = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            s = (tm >= a) & (tm < b)
            q = (fs["t"] >= a) & (fs["t"] < b)
            if s.sum() >= 3 and q.sum() >= 3:
                g.append((em[s][-1] - em[s][0]) / (tm[s][-1] - tm[s][0] + 1e-9))
                f.append(np.nanmedian(fs["pct"][q]))
        if len(g) >= 8:
            rho, _ = stats.spearmanr(f, g)
            row["feat_growth_rho"], row["feat_growth_n"] = float(rho), len(g)
    # realistic traffic in the camera's view (occlusion-aware, from the bag's model poses)
    row.update(traffic_contrasts(run, es, gt))
    tv_on = None
    if on is not None and traffic(run["dataset"]) is not None and not tl.is_static(run):
        tv_on = traffic_view(run["dataset"], gt, np.array([on]))
    row["onset_traffic_in_view"] = (bool(tv_on["n"][0] > 0) if tv_on is not None else None)
    row["onset_traffic_range_m"] = (float(tv_on["near"][0]) if tv_on is not None else np.nan)
    # true moving-object area (extreme bags) and the mechanism contrasts
    gb = gt_boxes(run["dataset"])
    row.update(mechanism_extreme(run, es, gb) if not tl.is_static(run) else
               dict(mv_area_med=np.nan, lost_hi=np.nan, lost_lo=np.nan, rho_feat_area=np.nan,
                    jump_rate_hi=np.nan, jump_rate_lo=np.nan, mv_n=0))
    # epoch traces around onset (for the figure)
    ep = None
    if on is not None:
        tt = np.arange(EPOCH[0], EPOCH[1] + 0.25, 0.25)
        fr = frontend(run)
        matched = None
        if fr is not None and len(fr["t"]) > 10:
            pre = (fr["t"] >= t[0]) & (fr["t"] < on - 5.0) & np.isfinite(fr["n"])
            ref = np.nanmedian(fr["n"][pre]) if pre.sum() >= 5 else np.nan
            if np.isfinite(ref) and ref > 0:
                matched = 100 * np.interp(on + tt, fr["t"], np.nan_to_num(fr["n"], nan=0.0)) / ref
        ep = dict(
            feat=matched,
            cov=100 * at(d["t"], d["cov"], on + tt, fill=np.nan) if d is not None and not tl.is_static(run) else None,
            mv=100 * at(gb["t"], gb["area"], on + tt, hold=0.1, fill=np.nan) if gb is not None else None,
            tr=(traffic_view(run["dataset"], gt, on + tt)["n"].astype(float)
                if traffic(run["dataset"]) is not None and not tl.is_static(run) else None),
            yaw=np.interp(on + tt, gt["t"], gt["yaw_rate"]),
            err=np.interp(on + tt, t, err) - np.interp(on, t, err))
    return row, ep


# ------------------------------------------------------------- lever-arm sweep

def shifted_gt(gt, L):
    """Ground truth moved L metres forward along its own heading: where the IMU
    (the frame both estimators report) is, if the model origin is the GT point."""
    g = dict(gt)
    yaw = gt["yaw_unwrapped"]
    g["p"] = gt["p"] + np.stack([L * np.cos(yaw), L * np.sin(yaw), np.zeros_like(yaw)], axis=1)
    return g


def whole_run_ate(es):
    R, tr = vm.umeyama(es["pa"], es["gp"])
    q = (R @ es["pa"].T).T + tr
    return float(np.sqrt(np.mean(np.sum((q[:, :2] - es["gp"][:, :2]) ** 2, axis=1))))


def lever_sweep(run, gt):
    """ATE and the turning/straight drift rates of a run when the GT is shifted by
    each lever arm in LEVER_SWEEP_M.  Returns {L: (ate, rate_turn, rate_straight)}."""
    out = {}
    for L in LEVER_SWEEP_M:
        g = shifted_gt(gt, L)
        es = tl.error_series(run, g)
        if es is None:
            out[L] = (np.nan, np.nan, np.nan); continue
        t, err = es["t"], es["err"]
        turning = np.interp(t, g["t"], g["yaw_rate"]) > TURN_DEG_S
        c = rate_contrast(t, err, turning, regime_mask(es))
        out[L] = (whole_run_ate(es), c[0] if c else np.nan, c[1] if c else np.nan)
    return out


def lever_fine(run, gt):
    """ATE against a fine grid of forward lever arms; returns (grid, ates)."""
    ates = []
    for L in LEVER_FINE_M:
        es = tl.error_series(run, shifted_gt(gt, L))
        ates.append(whole_run_ate(es) if es is not None else np.nan)
    return np.array(LEVER_FINE_M), np.array(ates)


def tex_lever_argmin(rows, path):
    sel = [r for r in rows if "fine" in r]
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption[Lever arm that minimises each run's ATE.]{The forward lever arm that minimises each tracked "
         r"unfiltered run's whole-run planar ATE, from a sweep of $L$ in \SIrange{0}{0.60}{\metre} in "
         r"\SI{0.04}{\metre} steps, with the ATE at $L=0$ (as reported in the chapter) and at the SDF value "
         r"$L=\SI{0.338}{\metre}$. The IMU sits \SI{0.338}{\metre} ahead of the model origin in "
         r"\path{models/ackermann_robot/model.sdf}.}",
         r"\label{tab:insight-lever-argmin}",
         r"\begin{tabular}{lllrrrr}", r"\toprule",
         r"Testbed & Recording & Estimator & $L^{*}$ [m] & ATE($0$) [m] & ATE($L^{*}$) [m] & ATE($0.338$) [m] \\", r"\midrule"]
    for r in sorted(sel, key=lambda r: (r["testbed"], r["dataset"], r["system"], r["run"])):
        g, a = r["fine"]
        if not np.isfinite(a).any():
            continue
        i = int(np.nanargmin(a)); a338 = float(np.interp(LEVER_SDF_M, g, a))
        L.append(f"{r['testbed']} & \\repo{{{r['dataset']}}} & {r['system']} & {g[i]:.2f} & {a[0]:.2f} & {a[i]:.2f} & {a338:.2f} \\\\")
    ls = [r["fine"][0][int(np.nanargmin(r["fine"][1]))] for r in sel if np.isfinite(r["fine"][1]).any()]
    L += [r"\midrule", f"\\multicolumn{{7}}{{l}}{{median $L^{{*}}$ = {np.median(ls):.2f}\\,m, IQR {np.percentile(ls,25):.2f}--{np.percentile(ls,75):.2f}\\,m over {len(ls)} runs}} \\\\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(L) + "\n")


def fig_lever(rows, path):
    sel = [r for r in rows if "fine" in r]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, tb in zip(axes, ("extreme", "realistic")):
        for r in sel:
            if r["testbed"] != tb:
                continue
            g, a = r["fine"]
            ax.plot(g, a, color=COL[r["system"]], alpha=0.6, lw=1.2,
                    ls="-" if r["static"] else "--", label=f"{r['system']} ({'static' if r['static'] else 'dynamic'})")
        ax.axvline(LEVER_SDF_M, color="#444", lw=0.9, ls=":")
        ax.text(LEVER_SDF_M + 0.01, ax.get_ylim()[1] * 0.95, "SDF: 0.338 m", fontsize=8, va="top")
        ax.set_xlabel("assumed forward lever arm L [m]"); ax.set_title(f"{tb} world", fontsize=10, loc="left")
        ax.grid(True, alpha=0.3)
        h, l = ax.get_legend_handles_labels(); seen = dict(zip(l, h))
        ax.legend(seen.values(), seen.keys(), fontsize=7)
    axes[0].set_ylabel("whole-run planar ATE [m]")
    fig.suptitle("ATE of every tracked unfiltered run against the assumed IMU lever arm", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=160); plt.close(fig)


def tex_lever(rows, path):
    """Pooled lever-arm sweep over the runs that track (static recordings of both
    testbeds and the realistic dynamic recordings), unfiltered estimators."""
    sel = [r for r in rows if "lever" in r and not r["filtered"] and (r["static"] or r["testbed"] == "realistic")]
    L = [r"\begin{table}[htbp]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\caption[Lever-arm sweep: ATE and turning drift against the assumed IMU offset.]{Lever-arm sweep. Both "
         r"estimators report the IMU frame, which sits \SI{0.338}{\metre} ahead of the model origin that the ground "
         r"truth reports (\path{models/ackermann_robot/model.sdf}); no scoring step compensates this. The table "
         r"re-scores the tracked unfiltered runs with the ground truth moved forward by $L$ along its heading: whole-run "
         r"aligned planar ATE (median over runs) and the drift rate while turning and on straights (median). If the "
         r"error that appears at turns were the lever arm, ATE and the turning rate would reach their minimum near "
         r"$L = \SI{0.34}{\metre}$.}",
         r"\label{tab:insight-lever}",
         r"\begin{tabular}{llr" + "rrr" * len(LEVER_SWEEP_M) + "}", r"\toprule",
         r"Testbed & Estimator & $n$ & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{$L=\\SI{{{l:.2f}}}{{\\metre}}$}}" for l in LEVER_SWEEP_M) + r" \\",
         r" & & & " + " & ".join(r"ATE & turn & str." for _ in LEVER_SWEEP_M) + r" \\",
         r" & & & " + " & ".join(r"[m] & \multicolumn{2}{c}{[mm/s]}" for _ in LEVER_SWEEP_M) + r" \\", r"\midrule"]
    for tb in ("extreme", "realistic"):
        for sysn in ("VINS-Fusion", "ORB-SLAM3"):
            g = [r for r in sel if r["testbed"] == tb and r["system"] == sysn]
            if not g:
                continue
            cells = []
            for l in LEVER_SWEEP_M:
                a = np.array([r["lever"][l][0] for r in g], float); tr_ = np.array([r["lever"][l][1] for r in g], float)
                st = np.array([r["lever"][l][2] for r in g], float)
                cells += [fmt(float(np.nanmedian(a)), 2), fmt(1000 * float(np.nanmedian(tr_)), 1), fmt(1000 * float(np.nanmedian(st)), 1)]
            L.append(f"{tb} & {sysn} & {len(g)} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(L) + "\n")


# --------------------------------------------------------------------- pooled

def pooled(rows, a_key, b_key, label):
    """Paired contrast over runs: median of the two rates, median ratio, sign count,
    Wilcoxon p on the paired difference."""
    a = np.array([r[a_key] for r in rows], float)
    b = np.array([r[b_key] for r in rows], float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) == 0:
        return dict(label=label, n=0)
    d = a - b
    p = stats.wilcoxon(d).pvalue if len(d) >= 5 and np.any(d != 0) else np.nan
    return dict(label=label, n=int(len(a)), med_a=float(np.median(a)), med_b=float(np.median(b)),
                n_pos=int((d > 0).sum()), p=float(p))


def fmt(v, p=3):
    return "---" if v is None or not np.isfinite(v) else f"{v:.{p}f}"


def tex_contrast(rows, testbed, path):
    groups = defaultdict(list)
    for r in rows:
        if r["testbed"] == testbed:
            groups[(r["static"], r["system"])].append(r)
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption[Within-run drift contrasts, " + testbed + r" world.]{Within-run drift contrasts, "
         + testbed + r" world. Each run is its own control: the growth rate of the start-anchored error "
         r"is computed separately in the two states and the two rates are compared across runs "
         r"(median of each, number of runs where the first exceeds the second, Wilcoxon signed-rank $p$). "
         r"``Turning'': ground-truth yaw rate above \SI{10}{\degree\per\second}. ``Object in view'': "
         r"detector boxes present on the frame (borrowed from the class-filtered run on the same bag for "
         r"unfiltered runs). Rates are taken only while the estimate is still tracking (before the first "
         r"jump and below \SI{3}{\metre} error). Source: \path{output/compare/simulation/exp3-insights/insights.csv}.}",
         r"\label{tab:insight-contrast-" + testbed + "}",
         r"\begin{tabular}{llrrrrr}", r"\toprule",
         r"Condition & Estimator & $n$ & turning & straight & $n_{>}$ & $p$ \\",
         r" & & & \multicolumn{2}{c}{[\si{\milli\metre\per\second}]} & & \\", r"\midrule"]
    for static in (True, False):
        for system in ("VINS-Fusion", "ORB-SLAM3", "VINS-Fusion+YOLO", "ORB-SLAM3+YOLO"):
            g = groups.get((static, system), [])
            if not g:
                continue
            c = pooled(g, "turn_pos_rate_turn", "turn_pos_rate_straight", "turn")
            if c["n"] == 0:
                continue
            L.append(f"{'static' if static else 'dynamic'} & {system} & {c['n']} & {fmt(1000*c['med_a'],1)} & "
                     f"{fmt(1000*c['med_b'],1)} & {c['n_pos']} & {fmt(c['p'],2)} \\\\")
    L += [r"\midrule", r"\multicolumn{7}{l}{\emph{Heading error, same contrast} [\si{\milli\degree\per\second}]} \\"]
    for static in (True, False):
        for system in ("VINS-Fusion", "ORB-SLAM3", "VINS-Fusion+YOLO", "ORB-SLAM3+YOLO"):
            g = groups.get((static, system), [])
            c = pooled(g, "turn_yaw_rate_turn", "turn_yaw_rate_straight", "yaw") if g else dict(n=0)
            if c["n"] == 0:
                continue
            L.append(f"{'static' if static else 'dynamic'} & {system} & {c['n']} & {fmt(1000*c['med_a'],1)} & "
                     f"{fmt(1000*c['med_b'],1)} & {c['n_pos']} & {fmt(c['p'],2)} \\\\")
    L += [r"\midrule", r"\multicolumn{7}{l}{\emph{Object in view vs none, dynamic recordings} [\si{\milli\metre\per\second}]} \\",
          r" & & & in view & none & & \\"]
    for system in ("VINS-Fusion", "ORB-SLAM3", "VINS-Fusion+YOLO", "ORB-SLAM3+YOLO"):
        g = groups.get((False, system), [])
        c = pooled(g, "view_rate_in", "view_rate_out", "view") if g else dict(n=0)
        if c["n"] == 0:
            continue
        L.append(f"dynamic & {system} & {c['n']} & {fmt(1000*c['med_a'],1)} & {fmt(1000*c['med_b'],1)} & "
                 f"{c['n_pos']} & {fmt(c['p'],2)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(L) + "\n")


def tex_onset(rows, testbed, path):
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption[What the robot was doing at the onset, " + testbed + r" world.]{What the robot was doing "
         r"at the error onset, " + testbed + r" world, per estimator and condition. ``Turning'' and ``object in "
         r"view'' as in the contrast table; ``features'' is the median tracked-feature level in the \SI{5}{\second} "
         r"before onset as a percentage of the run's own level before that. Runs without an onset are counted "
         r"separately. Source: \path{output/compare/simulation/exp3-insights/insights.csv}.}",
         r"\label{tab:insight-onset-" + testbed + "}",
         r"\begin{tabular}{llrrrrr}", r"\toprule",
         r"Condition & Estimator & runs & no onset & turning & object in view & features \\",
         r" & & & & [of onsets] & [of onsets] & [\si{\percent} of own] \\", r"\midrule"]
    groups = defaultdict(list)
    for r in rows:
        if r["testbed"] == testbed:
            groups[(r["static"], r["system"])].append(r)
    for static in (True, False):
        for system in ("VINS-Fusion", "ORB-SLAM3", "VINS-Fusion+YOLO", "ORB-SLAM3+YOLO"):
            g = groups.get((static, system), [])
            if not g:
                continue
            on = [r for r in g if np.isfinite(r["onset_s"])]
            turn = sum(1 for r in on if r["onset_turning"])
            iv = [r for r in on if r["onset_in_view"] is not None]
            iv_n = sum(1 for r in iv if r["onset_in_view"])
            ratio = [100 * r["onset_feat_pre5_pct"] / r["onset_feat_base_pct"] for r in on
                     if np.isfinite(r["onset_feat_pre5_pct"]) and np.isfinite(r["onset_feat_base_pct"]) and r["onset_feat_base_pct"] > 0]
            L.append(f"{'static' if static else 'dynamic'} & {system} & {len(g)} & {len(g)-len(on)} & "
                     f"{turn}/{len(on)} & {(str(iv_n)+'/'+str(len(iv))) if iv else '---'} & "
                     f"{fmt(float(np.median(ratio)),0) if ratio else '---'} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(L) + "\n")


def tex_mechanism(rows, path):
    g = [r for r in rows if r["testbed"] == "extreme" and not r["static"]]
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption[Extreme world: tracking events against the true moving-object area in view.]{Extreme world, "
         r"dynamic recordings: tracking events against the \emph{true} moving-object area in view, from the "
         r"projected simulator truth of Experiment~2 (\path{output/yolo_eval/dyn_*/yolo_eval.csv}). Within each run "
         r"the frames are split into terciles of moving-object area; the table gives the median over runs of the "
         r"share of frames in which ORB-SLAM3 was not in the OK state, the rate of impossible-speed jumps of the "
         r"estimate, and the Spearman correlation between matched features (ORB-SLAM3 map inliers, VINS-Fusion "
         r"tracked features) and moving-object area. $n_{>}$ counts runs where the top tercile exceeds the bottom; "
         r"$p$ is the Wilcoxon signed-rank test over runs.}",
         r"\label{tab:insight-mechanism-extreme}",
         r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"Estimator & $n$ & area med. [\si{\percent}] & top tercile & bottom tercile & $n_{>}$ & $p$ \\", r"\midrule"]
    for sysn in ("ORB-SLAM3", "ORB-SLAM3+YOLO"):
        gg = [r for r in g if r["system"] == sysn]
        c = pooled(gg, "lost_hi", "lost_lo", "lost")
        if c["n"]:
            L.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{Share of frames not tracking OK}}}} \\\\")
            L.append(f"{sysn} & {c['n']} & {fmt(float(np.median([r['mv_area_med'] for r in gg if np.isfinite(r['mv_area_med'])])),1)} & "
                     f"{fmt(c['med_a'],3)} & {fmt(c['med_b'],3)} & {c['n_pos']} & {fmt(c['p'],2)} \\\\")
    L.append(r"\multicolumn{7}{l}{\emph{Jumps of the estimate per second}} \\")
    for sysn in ("VINS-Fusion", "ORB-SLAM3", "VINS-Fusion+YOLO", "ORB-SLAM3+YOLO"):
        gg = [r for r in g if r["system"] == sysn]
        c = pooled(gg, "jump_rate_hi", "jump_rate_lo", "jump")
        if c["n"]:
            L.append(f"{sysn} & {c['n']} & {fmt(float(np.median([r['mv_area_med'] for r in gg if np.isfinite(r['mv_area_med'])])),1)} & "
                     f"{fmt(c['med_a'],2)} & {fmt(c['med_b'],2)} & {c['n_pos']} & {fmt(c['p'],2)} \\\\")
    L.append(r"\multicolumn{7}{l}{\emph{Spearman $\rho$, matched features against moving-object area (median over runs; runs with $\rho<0$)}} \\")
    for sysn in ("VINS-Fusion", "ORB-SLAM3", "VINS-Fusion+YOLO", "ORB-SLAM3+YOLO"):
        rho = [r["rho_feat_area"] for r in g if r["system"] == sysn and np.isfinite(r["rho_feat_area"])]
        if rho:
            L.append(f"{sysn} & {len(rho)} & & \\multicolumn{{2}}{{c}}{{$\\rho$ = {fmt(float(np.median(rho)),2)}}} & "
                     f"{sum(1 for x in rho if x < 0)} & \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(L) + "\n")


def tex_traffic(rows, path):
    g = [r for r in rows if r["testbed"] == "realistic" and not r["static"]]
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption[Realistic world: tracking against traffic in the camera's view.]{Realistic world, dynamic "
         r"recordings: within-run contrasts between the moments a traffic agent (carrier or person) is in the "
         r"camera's view and the moments none is. Visibility is computed from the bag's own model poses "
         r"(\path{/world/default/pose/info}): within \SI{20}{\metre}, within $\pm$\SI{45}{\degree} of the "
         r"ground-truth heading, and with no shelf footprint on the line of sight. Drift rate while tracking "
         r"(before the first jump, below \SI{3}{\metre}); share of frames ORB-SLAM3 was not tracking OK; "
         r"median matched features (ORB-SLAM3 map inliers, VINS-Fusion tracked features); jumps per second. "
         r"$n_{>}$ counts runs where the in-view value exceeds the other; $p$ is the Wilcoxon signed-rank test.}",
         r"\label{tab:insight-traffic-realistic}",
         r"\begin{tabular}{llrrrrr}", r"\toprule",
         r"Quantity & Estimator & $n$ & in view & none & $n_{>}$ & $p$ \\", r"\midrule"]
    specs = (("drift rate [\si{\milli\metre\per\second}]", "tr_rate_in", "tr_rate_out", 1000, 1),
             ("not tracking OK [share]", "tr_lost_in", "tr_lost_out", 1, 3),
             ("matched features", "tr_feat_in", "tr_feat_out", 1, 0),
             ("jumps [\si{\per\second}]", "tr_jumps_in", "tr_jumps_out", 1, 2))
    for lab, a, b, k, pdig in specs:
        first = True
        for sysn in ("VINS-Fusion", "ORB-SLAM3", "VINS-Fusion+YOLO", "ORB-SLAM3+YOLO"):
            gg = [r for r in g if r["system"] == sysn]
            c = pooled(gg, a, b, lab) if gg else dict(n=0)
            if c["n"] == 0:
                continue
            L.append(f"{lab if first else ''} & {sysn} & {c['n']} & {fmt(k*c['med_a'],pdig)} & {fmt(k*c['med_b'],pdig)} & "
                     f"{c['n_pos']} & {fmt(c['p'],2)} \\\\")
            first = False
        L.append(r"\addlinespace")
    share = [r["tr_share"] for r in g if np.isfinite(r["tr_share"])]
    L += [r"\bottomrule", r"\multicolumn{7}{l}{\footnotesize Traffic was in view for a median "
          + fmt(100 * float(np.median(share)) if share else np.nan, 0) + r"\,\si{\percent} of the tracked time.} \\",
          r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(L) + "\n")


def tex_realistic_runs(rows, path):
    g = [r for r in rows if r["testbed"] == "realistic"]
    L = [r"\begin{table}[htbp]", r"\centering", r"\scriptsize",
         r"\caption[Realistic world, every run: drift while turning, drift on straights, drift with traffic in view.]"
         r"{Realistic world, every run: growth rate of the start-anchored planar error while the robot turns and on "
         r"straights, and with traffic in the camera's view and without, over the tracked time ($T$). Blank: fewer "
         r"than \SI{3}{\second} in one of the two states. Onset as in the timeline analysis; ``at onset'' gives "
         r"whether the robot was turning and whether traffic was in view at that moment.}",
         r"\label{tab:insight-realistic-runs}",
         r"\begin{tabular}{lllrrrrrrl}", r"\toprule",
         r"Recording & Estimator & Run & $T$ & turning & straight & traffic & none & onset & at onset \\",
         r" & & & [s] & \multicolumn{4}{c}{[\si{\milli\metre\per\second}]} & [s] & turn / traffic \\", r"\midrule"]
    for r in sorted(g, key=lambda r: (r["dataset"], r["system"], r["run"])):
        def yn(v):
            return "---" if v is None else ("yes" if v else "no")
        L.append(f"\\repo{{{r['dataset']}}} & {r['system']} & \\repo{{{r['run']}}} & {fmt(r['regime_s'],0)} & "
                 f"{fmt(1000*r['turn_pos_rate_turn'],1)} & {fmt(1000*r['turn_pos_rate_straight'],1)} & "
                 f"{fmt(1000*r['tr_rate_in'],1)} & {fmt(1000*r['tr_rate_out'],1)} & {fmt(r['onset_s'],0)} & "
                 f"{yn(r['onset_turning'])} / {yn(r['onset_traffic_in_view'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(L) + "\n")


# -------------------------------------------------------------------- figures

def fig_contrast(rows, a_key, b_key, xlabel, ylabel, path, title, note):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharex=False)
    for ax, testbed in zip(axes, ("extreme", "realistic")):
        lim = 1e-4
        for r in rows:
            if r["testbed"] != testbed or not (np.isfinite(r[a_key]) and np.isfinite(r[b_key])):
                continue
            x, y = max(r[b_key], lim), max(r[a_key], lim)
            ax.scatter(x, y, s=28 if r["static"] else 36, marker="o" if r["static"] else "^",
                       color=COL[r["system"]], alpha=0.8, edgecolor="white", linewidth=0.5,
                       label=f"{r['system']} ({'static' if r['static'] else 'dynamic'})")
        ax.plot([lim, 1], [lim, 1], color="#888", lw=0.8, ls="--")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(f"{testbed} world", fontsize=10, loc="left")
        ax.grid(True, which="major", alpha=0.3)
        h, l = ax.get_legend_handles_labels()
        seen = dict(zip(l, h))
        ax.legend(seen.values(), seen.keys(), fontsize=7, loc="lower right", frameon=True)
    fig.suptitle(title, fontsize=12, x=0.01, ha="left")
    fig.text(0.01, 0.92, note, fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=160); plt.close(fig)


def fig_epoch(epochs, path):
    """Superposed-epoch averages around the onset, per testbed x estimator family."""
    tt = np.arange(EPOCH[0], EPOCH[1] + 0.25, 0.25)
    fig, axes = plt.subplots(4, 2, figsize=(10, 10.5), sharex=True)
    for j, testbed in enumerate(("extreme", "realistic")):
        for sysn, col in COL.items():
            eps = [e for (tb, s, e) in epochs if tb == testbed and s == sysn and e is not None]
            if not eps:
                continue
            for i, (key, lab) in enumerate((("feat", "matched features [% of own pre-onset median]"),
                                            ("mv", "true moving-object area in view [%]"),
                                            ("cov", "image area masked by the filter [%]"),
                                            ("yaw", "ground-truth yaw rate [deg/s]"))):
                k2 = "tr" if (key == "mv" and testbed == "realistic") else key
                tr = np.array([e[k2] for e in eps if e.get(k2) is not None], float)
                if len(tr) == 0:
                    continue
                if k2 == "tr":
                    lab = "traffic agents in the camera's view [count]"
                med = np.nanmedian(tr, axis=0)
                axes[i, j].plot(tt, med, color=col, lw=1.6, label=f"{sysn} (n={len(tr)})")
                lo, hi = np.nanpercentile(tr, 25, axis=0), np.nanpercentile(tr, 75, axis=0)
                axes[i, j].fill_between(tt, lo, hi, color=col, alpha=0.12, lw=0)
                axes[i, j].set_ylabel(lab, fontsize=8)
        for i in range(4):
            axes[i, j].axvline(0, color="#444", lw=0.8, ls=":")
            axes[i, j].grid(True, alpha=0.3)
            if axes[i, j].get_legend_handles_labels()[0]:
                axes[i, j].legend(fontsize=7, loc="upper left")
        axes[0, j].set_title(f"{testbed} world: median (IQR) across runs, time relative to onset", fontsize=9, loc="left")
        axes[3, j].set_xlabel("time from error onset [s]")
    fig.suptitle("What precedes the error onset", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=160); plt.close(fig)


def fig_traffic_timeline(runs, gts, series, path):
    """Realistic dynamic recordings, unfiltered runs: anchored error over time with
    the moments traffic is in the camera's view and the moments the robot turns."""
    sel = [r for r in runs if r["testbed"] == "realistic" and not tl.is_static(r) and not r["filtered"] and id(r) in series]
    sel = sorted(sel, key=lambda r: (r["dataset"], r["system"], r["run"]))
    if not sel:
        return
    fig, axes = plt.subplots(len(sel), 1, figsize=(10, 1.9 * len(sel) + 1), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, sel):
        gt, es = gts[r["gt"]], series[id(r)]
        t, err = es["t"], es["err"]
        t0 = t[0]
        tq = np.arange(t[0], t[-1], 0.2)
        tv = traffic_view(r["dataset"], gt, tq)
        turning = np.interp(tq, gt["t"], gt["yaw_rate"]) > TURN_DEG_S
        ymax = max(1.0, float(np.nanpercentile(err, 99)) * 1.1)
        if tv is not None:
            ax.fill_between(tq - t0, 0, ymax, where=tv["n"] > 0, color="#e040fb", alpha=0.18, lw=0, label="traffic in view")
        ax.fill_between(tq - t0, 0, ymax, where=turning, color="#888", alpha=0.18, lw=0, label="turning")
        ax.plot(t - t0, err, color=COL[r["system"]], lw=1.3, label="anchored planar error")
        ax.axvspan(es["anchor"][0] - t0, es["anchor"][1] - t0, color="#458554", alpha=0.10, lw=0, label="anchor window")
        ax.set_ylim(0, ymax); ax.set_xlim(0, t[-1] - t0)
        ax.set_ylabel("error [m]", fontsize=8)
        ax.set_title(f"{tl.short(r['dataset'])}  {r['system']}  {r['run']}", fontsize=9, loc="left")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time from first estimate [s]")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Realistic world, dynamic recordings: when the traffic is in view and when the robot turns", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=160); plt.close(fig)


# ----------------------------------------------------------------------- main

def main():
    global SUFFIX
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lever", type=float, default=0.0,
                    help="forward lever arm applied to the ground truth before scoring (m); 0 = as the chapter scores")
    args = ap.parse_args()
    SUFFIX = "" if args.lever == 0.0 else f"_lever{args.lever:.3f}".replace(".", "p")
    OUT.mkdir(parents=True, exist_ok=True)
    runs = tl.registry()
    gts, series = {}, {}
    for r in runs:
        if r["gt"] not in gts:
            g0 = tl.gt_series(r["gt"], lever=False)   # raw model origin; the sweep adds L itself
            gts[r["gt"]] = shifted_gt(g0, args.lever) if args.lever else g0
        es = tl.error_series(r, gts[r["gt"]])
        if es is not None:
            series[id(r)] = es
    # onset thresholds: the timeline rule (3 x median static anchored error), recomputed
    # for the lever variant because the static errors shrink when the arm is applied
    thr = {}
    for tb in ("extreme", "realistic"):
        for base in ("VINS-Fusion", "ORB-SLAM3"):
            errs = [series[id(r)]["err"] for r in runs if id(r) in series and r["testbed"] == tb and not r["filtered"]
                    and tl.base_of(r["system"]) == base and tl.is_static(r)]
            thr[(tb, base)] = tl.THR_FACTOR * float(np.median(np.concatenate(errs))) if errs else np.nan
    print("lever %.3f m; onset thresholds:" % args.lever, {k: round(v, 3) for k, v in thr.items()})
    donors = {ds: donor_mask(runs, ds) for ds in {r["dataset"] for r in runs}}
    rows, epochs = [], []
    for r in runs:
        if id(r) not in series:
            continue
        row, ep = analyse_run(r, gts[r["gt"]], series[id(r)], thr[(r["testbed"], tl.base_of(r["system"]))], runs, donors)
        if not r["filtered"] and (tl.is_static(r) or r["testbed"] == "realistic") and args.lever == 0.0:
            row["lever"] = lever_sweep(r, gts[r["gt"]])
            row["fine"] = lever_fine(r, gts[r["gt"]])
        rows.append(row); epochs.append((r["testbed"], r["system"], ep))
    keys = [k for k in rows[0].keys() if k not in ("lever", "fine")]
    for r in rows:
        if "lever" in r:
            for l, (a, b, c) in r["lever"].items():
                r[f"lever_{l:.2f}_ate"], r[f"lever_{l:.2f}_turn"], r[f"lever_{l:.2f}_straight"] = a, b, c
    keys += [k for k in rows[0].keys() if k.startswith("lever_")] if "lever" in rows[0] else []
    keys = list(dict.fromkeys(keys + [k for r in rows for k in r if k.startswith("lever_")]))
    with open(OUT / f"insights{SUFFIX}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore"); w.writeheader()
        for row in rows:
            w.writerow({k: ("" if (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in row.items() if k not in ("lever", "fine")})
    for tb in ("extreme", "realistic"):
        tex_contrast(rows, tb, OUT / f"contrast_{tb}{SUFFIX}.tex")
        tex_onset(rows, tb, OUT / f"onset_context_{tb}{SUFFIX}.tex")
    tex_mechanism(rows, OUT / f"mechanism_extreme{SUFFIX}.tex")
    tex_traffic(rows, OUT / f"traffic_realistic{SUFFIX}.tex")
    if args.lever == 0.0:
        tex_lever(rows, OUT / "lever_arm_sweep.tex")
        tex_lever_argmin(rows, OUT / "lever_arm_argmin.tex")
        fig_lever(rows, OUT / "fig_lever_sweep_light.png")
    tex_realistic_runs(rows, OUT / f"realistic_runs{SUFFIX}.tex")
    fig_contrast(rows, "turn_pos_rate_turn", "turn_pos_rate_straight",
                 "drift rate on straights [m/s]", "drift rate while turning [m/s]",
                 OUT / f"fig_turn_contrast{SUFFIX}_light.png", "Does the error grow faster while the robot turns?",
                 "one marker per run; above the diagonal = faster while turning; tracking regime only")
    fig_contrast([r for r in rows if not r["static"]], "view_rate_in", "view_rate_out",
                 "drift rate with no object in view [m/s]", "drift rate with an object in view [m/s]",
                 OUT / f"fig_object_contrast{SUFFIX}_light.png", "Does the error grow faster while a movable object is in view?",
                 "dynamic recordings only; in-view from the detector log of the same bag; tracking regime only")
    fig_epoch(epochs, OUT / f"fig_onset_epoch{SUFFIX}_light.png")
    fig_traffic_timeline(runs, gts, series, OUT / f"fig_traffic_timeline{SUFFIX}_light.png")
    # console digest
    print(f"{len(rows)} runs analysed")
    print("\n== lever-arm sweep (median over tracked unfiltered runs): L -> ATE m / turn mm/s / straight mm/s")
    for tb in ("extreme", "realistic"):
        for sysn in ("VINS-Fusion", "ORB-SLAM3"):
            g = [r for r in rows if "lever" in r and r["testbed"] == tb and r["system"] == sysn]
            if g:
                print(f"  {tb:9s} {sysn:12s} n={len(g):2d} " + " | ".join(
                    f"L={l:.2f}: {fmt(float(np.nanmedian([r['lever'][l][0] for r in g])),2)} / {fmt(1000*float(np.nanmedian([r['lever'][l][1] for r in g])),1)} / {fmt(1000*float(np.nanmedian([r['lever'][l][2] for r in g])),1)}"
                    for l in LEVER_SWEEP_M))
    for tb in ("extreme", "realistic"):
        print(f"\n== {tb}")
        for static in (True, False):
            for sysn in COL:
                g = [r for r in rows if r["testbed"] == tb and r["static"] == static and r["system"] == sysn]
                if not g:
                    continue
                c = pooled(g, "turn_pos_rate_turn", "turn_pos_rate_straight", "turn")
                y = pooled(g, "turn_yaw_rate_turn", "turn_yaw_rate_straight", "yaw")
                v = pooled(g, "view_rate_in", "view_rate_out", "view")
                rho = [r["feat_growth_rho"] for r in g if np.isfinite(r["feat_growth_rho"])]
                if tb == "realistic" and not static:
                    for lab, a, b, k in (("drift", "tr_rate_in", "tr_rate_out", 1000), ("lost", "tr_lost_in", "tr_lost_out", 1),
                                         ("feat", "tr_feat_in", "tr_feat_out", 1), ("jumps", "tr_jumps_in", "tr_jumps_out", 1)):
                        c = pooled(g, a, b, lab)
                        if c["n"]:
                            print(f"  {sysn:18s} TRAFFIC {lab:5s} in/out {fmt(k*c['med_a'],2)}/{fmt(k*c['med_b'],2)} n>={c['n_pos']}/{c['n']} p={fmt(c['p'],2)}")
                if tb == "extreme" and not static:
                    lo_ = pooled(g, "lost_hi", "lost_lo", "lost"); jp = pooled(g, "jump_rate_hi", "jump_rate_lo", "jump")
                    ra = [r["rho_feat_area"] for r in g if np.isfinite(r["rho_feat_area"])]
                    print(f"  {sysn:18s} MOVING-AREA: lost hi/lo {fmt(lo_.get('med_a',np.nan),3)}/{fmt(lo_.get('med_b',np.nan),3)} n>={lo_.get('n_pos','-')}/{lo_['n']} p={fmt(lo_.get('p',np.nan),2)} | "
                          f"jumps/s hi/lo {fmt(jp.get('med_a',np.nan),2)}/{fmt(jp.get('med_b',np.nan),2)} n>={jp.get('n_pos','-')}/{jp['n']} p={fmt(jp.get('p',np.nan),2)} | "
                          f"rho(feat,area) med {fmt(float(np.median(ra)) if ra else np.nan,2)} neg {sum(1 for x in ra if x<0)}/{len(ra)}")
                print(f"  {'static ' if static else 'dynamic'} {sysn:18s} n={len(g):2d} "
                      f"turn/straight pos {fmt(1000*c.get('med_a',np.nan),1)}/{fmt(1000*c.get('med_b',np.nan),1)} mm/s "
                      f"n>={c.get('n_pos','-')}/{c['n']} p={fmt(c.get('p',np.nan),2)} | "
                      f"yaw {fmt(1000*y.get('med_a',np.nan),1)}/{fmt(1000*y.get('med_b',np.nan),1)} mdeg/s n>={y.get('n_pos','-')}/{y['n']} p={fmt(y.get('p',np.nan),2)} | "
                      f"view in/out {fmt(1000*v.get('med_a',np.nan),1)}/{fmt(1000*v.get('med_b',np.nan),1)} n>={v.get('n_pos','-')}/{v['n']} p={fmt(v.get('p',np.nan),2)} | "
                      f"feat-growth rho med {fmt(float(np.median(rho)) if rho else np.nan,2)} (n={len(rho)})")


if __name__ == "__main__":
    main()
