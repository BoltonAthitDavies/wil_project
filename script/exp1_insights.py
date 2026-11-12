#!/usr/bin/env python3
"""Experiment 1 insight analysis: WHY each estimator errs the way it does.

Tests, on the four `dataset_allsensor_realistic_*` recordings, the mechanisms a
reader would propose for the Experiment 1 numbers, each within a recording so
that the recording is its own control:

  decomposition  the anchored error of each proprioceptive baseline split into
                 along-track (distance/scale) and cross-track (heading) parts in
                 the ground-truth path frame, with the heading error itself
  turning        growth rate of position and heading error while the ground-truth
                 yaw rate exceeds TURN_DEG_S against the rate on straights
  IMU law        the exponent of the IMU dead-reckoning error against time, and
                 how much of that error the heading error alone reproduces
  NEES timing    when the EKF's 3-DOF NEES leaves its 95 % band, against the
                 first heading change and the distance travelled; measured
                 noise model against the transferred (VSLAM-file) one
  visual         for VINS-Fusion and ORB-SLAM3 (scored at the IMU point): drift
                 while turning against straights in the tracking regime, matched
                 features against drift, and the anatomy of the one failure
                 (ORB-SLAM3 with loop closure on the I-layout recording)

Ground truth: the baselines are referenced to the model origin and use the raw
ground truth; the visual estimators report the IMU and are scored against the
ground truth moved to the IMU point (vio_metrics.load_gt).

Outputs (output/compare/simulation/exp1-insights/):
  decomposition.csv / decomposition.tex      per bag x baseline
  turning.csv / turning.tex                  per estimator, pooled over bags
  imu_law.tex                                exponent and heading-explained share
  nees_timing.tex                            crossing time vs first turn, both EKFs
  visual.tex                                 visual contrasts and the ORB failure
  fig_decomposition_light.png                along/cross/heading error, wheel+EKF
  fig_nees_light.png                         NEES over time, both EKFs, first turn
  fig_imu_law_light.png                      IMU error vs time, log-log, and heading-only prediction
  fig_orb_failure_light.png                  ORB-SLAM3 error, jumps and route on the
                                             one recording where it diverges
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vio_metrics as vm                      # noqa: E402
import analyse_exp3_timelines as tl           # noqa: E402
import exp3_insights as x3                    # noqa: E402
from map_underlay import draw_map             # noqa: E402

import matplotlib                             # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt               # noqa: E402

ROOT = Path(vm.ROOT)
OUT = ROOT / "output" / "compare" / "simulation" / "exp1-insights"
# The dataset set fixed on 6 Nov 2026. The dataset_allsensor_00x recordings
# this used to name were retired that day (001 and 004 are lost outright), so
# every artefact of this script was retired with them rather than regenerated.
BAGS = ["dataset_allsensor_realistic_%03d" % i for i in range(4)]
# The one recording where ORB-SLAM3 with its closer active diverges (4.78 m, the
# false 14 m closure between the I layout's repeated aisles). It was
# dataset_allsensor_003 in the retired campaign.
ORB_FAILURE_BAG = "dataset_allsensor_realistic_001"
TURN_DEG_S = 10.0
NEES_BAND = stats.chi2.ppf(0.975, 3)          # 9.35 for 3 DOF
NEES_HZ = 1.0
SUSTAIN_S = 5.0
BASELINES = [("Wheel odometry", "output_wheel", "#887b2c"),
             ("IMU dead reckoning", "output_imu", "#b00020"),
             # 6 Nov 2026: output_ekf IS the VSLAM-noise baseline and
             # output_ekf_measured the measured-noise diagnostic. Before that
             # date the two names meant the opposite; do not swap them back.
             # The labels must match the reported table of
             # output/compare/simulation/exp1-realistic/proprio_baseline_table.tex,
             # which calls output_ekf "EKF wheel+IMU".  Using that label for
             # output_ekf_measured here put the same name on opposite arms in
             # two documents (12 Nov 2026), so both are named by their noise
             # model and neither carries the bare name.
             ("EKF @ measured noise", "output_ekf_measured", "#458554"),
             ("EKF @ VSLAM noise", "output_ekf", "#2a9d8f")]
# Run names since 6 Nov 2026; the logging_20260916_* campaign these used to
# name was retired that day and no longer exists on disk.
VISUAL = [("VINS-Fusion", "output_vins", "cpu", "#cc8319"),
          ("ORB-SLAM3", "output_orb", "loopclosing_gpu", "#2a78d6"),
          ("ORB-SLAM3 (no LC)", "output_orb", "noloopclosing_gpu", "#a347b2")]
COL = {n: c for n, _, c in BASELINES}
COL.update({n: c for n, _, _, c in VISUAL})


# ------------------------------------------------------------------ helpers

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def gt_path(ds):
    """Canonical ground truth for a simulation bag, following the 6 Nov 2026 move."""
    for c in vm.gt_candidates("simulation/" + ds):
        if os.path.exists(c):
            return c
    raise SystemExit("no ground truth for %s: looked in\n  %s"
                     % (ds, "\n  ".join(vm.gt_candidates("simulation/" + ds))))


def gt_raw(ds):
    # Canonical truth since 6 Nov 2026 is dataset/gt/<bag>/ground_truth.csv; the
    # flat per-tree copies this used to read went to trash/ that day, which is
    # why every artefact of this script was retired rather than regenerated.
    # vio_metrics.gt_candidates() returns the canonical path first and keeps the
    # tree copies as fallbacks, so this follows the move automatically.
    p = gt_path(ds)
    g = tl.gt_series(p, lever=False)
    g["yaw"] = g["yaw_unwrapped"]
    g["speed"] = np.gradient(g["p"][:, 0], g["t"]), np.gradient(g["p"][:, 1], g["t"])
    return g, str(p)


def load_est(d):
    r = vm.load(os.path.join(d, "vio.csv"))
    if r is None:
        return None
    t, p, q, v = r
    yaw = np.unwrap(np.radians(vm.euler_zyx(vm.quat_to_R(q))[:, 0]))
    return dict(t=t, p=p, q=q, yaw=yaw)


def first_turn(g):
    i = np.argmax(g["yaw_rate"] > TURN_DEG_S)
    return g["t"][i] if (g["yaw_rate"] > TURN_DEG_S).any() else None


def dist_at(g, t):
    seg = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(g["p"][:, :2], axis=0), axis=1))])
    return float(np.interp(t, g["t"], seg))


# ---------------------------------------------------------- decomposition

def decompose(ds, g, name, tree):
    e = load_est(str(ROOT / "output" / tree / "simulation" / ds))
    if e is None:
        return None
    t = e["t"]
    m = (t >= g["t"][0]) & (t <= g["t"][-1])
    t, p, yaw = t[m], e["p"][m], e["yaw"][m]
    gp = vm.resample(g["t"], g["p"], t)
    gyaw = np.interp(t, g["t"], g["yaw"])
    err = p[:, :2] - gp[:, :2]
    c, s = np.cos(gyaw), np.sin(gyaw)
    along = err[:, 0] * c + err[:, 1] * s
    cross = -err[:, 0] * s + err[:, 1] * c
    herr = wrap(yaw - gyaw)
    herr = herr - herr[0]                                # both start aligned
    dist = np.linalg.norm(err, axis=1)
    # path-length (scale) error of the estimate against the truth
    L_est = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum()
    L_gt = np.linalg.norm(np.diff(gp[:, :2], axis=0), axis=1).sum()
    # position error that the heading error ALONE would produce: integrate the
    # true velocity rotated by the heading error
    vg = np.gradient(gp[:, :2], t, axis=0)
    ch, sh = np.cos(herr), np.sin(herr)
    v_rot = np.stack([vg[:, 0] * ch - vg[:, 1] * sh, vg[:, 0] * sh + vg[:, 1] * ch], axis=1)
    pred = np.cumsum((v_rot - vg) * np.gradient(t)[:, None], axis=0)
    pred_mag = np.linalg.norm(pred, axis=1)
    turning = np.interp(t, g["t"], g["yaw_rate"]) > TURN_DEG_S
    cpos = x3.rate_contrast(t, dist, turning, np.ones(len(t), bool))
    chead = x3.rate_contrast(t, np.abs(np.degrees(herr)), turning, np.ones(len(t), bool))
    return dict(dataset=ds, name=name, secs=float(t[-1] - t[0]),
                rms=float(np.sqrt((dist ** 2).mean())), final=float(dist[-1]),
                along_rms=float(np.sqrt((along ** 2).mean())), cross_rms=float(np.sqrt((cross ** 2).mean())),
                along_final=float(along[-1]), cross_final=float(cross[-1]),
                head_rms_deg=float(np.degrees(np.sqrt((herr ** 2).mean()))), head_final_deg=float(np.degrees(herr[-1])),
                scale_pct=100.0 * (L_est / L_gt - 1.0),
                heading_explained_final=float(pred_mag[-1] / max(dist[-1], 1e-9)),
                heading_explained_rms=float(np.sqrt((pred_mag ** 2).mean()) / max(np.sqrt((dist ** 2).mean()), 1e-9)),
                turn_pos=cpos[0] if cpos else np.nan, straight_pos=cpos[1] if cpos else np.nan,
                turn_head=chead[0] if chead else np.nan, straight_head=chead[1] if chead else np.nan,
                series=dict(t=t - t[0], along=along, cross=cross, herr=np.degrees(herr), dist=dist, pred=pred_mag, turning=turning))


def imu_law(rec):
    """Exponent of |error| against time (log-log fit after the first 5 s)."""
    s = rec["series"]
    m = (s["t"] > 5.0) & (s["dist"] > 1e-3)
    if m.sum() < 20:
        return np.nan
    slope, _, r, _, _ = stats.linregress(np.log(s["t"][m]), np.log(s["dist"][m]))
    return float(slope), float(r ** 2)


# -------------------------------------------------------------------- NEES

def nees_series(ds, g, tree):
    d = ROOT / "output" / tree / "simulation" / ds
    e = load_est(str(d))
    if e is None or not (d / "covariance.csv").exists():
        return None
    rows = list(csv.DictReader(open(d / "covariance.csv")))
    tc = np.array([int(r["t_ns"]) for r in rows]) * 1e-9
    P = np.array([[[float(r["var_x"]), float(r["cov_xy"]), float(r["cov_xyaw"])],
                   [float(r["cov_xy"]), float(r["var_y"]), float(r["cov_yyaw"])],
                   [float(r["cov_xyaw"]), float(r["cov_yyaw"]), float(r["var_yaw"])]] for r in rows])
    # subsample at NEES_HZ within the overlap
    t0, t1 = max(tc[0], g["t"][0], e["t"][0]), min(tc[-1], g["t"][-1], e["t"][-1])
    ts = np.arange(t0 + 1.0, t1, 1.0 / NEES_HZ)
    ic = np.clip(np.searchsorted(tc, ts), 0, len(tc) - 1)
    ie = np.clip(np.searchsorted(e["t"], ts), 0, len(e["t"]) - 1)
    gp = vm.resample(g["t"], g["p"], ts); gyaw = np.interp(ts, g["t"], g["yaw"])
    ex = e["p"][ie, 0] - gp[:, 0]; ey = e["p"][ie, 1] - gp[:, 1]
    eyaw = wrap(e["yaw"][ie] - gyaw)
    nees = np.full(len(ts), np.nan)
    for k in range(len(ts)):
        Pk = P[ic[k]]
        if np.all(np.isfinite(Pk)) and np.linalg.det(Pk) > 1e-30:
            v = np.array([ex[k], ey[k], eyaw[k]])
            nees[k] = v @ np.linalg.solve(Pk, v)
    return dict(t=ts, nees=nees)


def nees_crossing(ns):
    """First time NEES stays above the 97.5 % band for SUSTAIN_S."""
    t, n = ns["t"], ns["nees"]
    above = n > NEES_BAND
    k = int(SUSTAIN_S * NEES_HZ)
    for i in range(len(t) - k):
        if above[i:i + k].all():
            return t[i]
    return None


# ------------------------------------------------------------------- visual

def visual_run(ds, name, tree, run):
    d = ROOT / "output" / tree / "simulation" / ds / run
    if not (d / "vio.csv").exists():
        return None
    return dict(testbed="exp1", dataset=ds, tree=tree, system=name.split(" (")[0], run=run, dir=str(d),
                filtered=False, mode="off", lc="n/a", atlas=False, september=True,
                gt=gt_path(ds), label=name)


def visual_contrasts(ds, g_imu, rec):
    es = tl.error_series(rec, g_imu)
    if es is None:
        return None
    t, err = es["t"], es["err"]
    turning = np.interp(t, g_imu["t"], g_imu["yaw_rate"]) > TURN_DEG_S
    m = x3.regime_mask(es)
    c = x3.rate_contrast(t, err, turning, m)
    fr = x3.frontend(rec)
    rho, nb = np.nan, 0
    if fr is not None and m.sum() > 40:
        tm, em = t[m], err[m]
        edges = np.arange(tm[0], tm[-1], 2.0)
        gr, fe = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            s = (tm >= a) & (tm < b); q = (fr["t"] >= a) & (fr["t"] < b) & np.isfinite(fr["n"])
            if s.sum() >= 3 and q.sum() >= 3:
                gr.append((em[s][-1] - em[s][0]) / (tm[s][-1] - tm[s][0] + 1e-9)); fe.append(np.nanmedian(fr["n"][q]))
        if len(gr) >= 8:
            rho = float(stats.spearmanr(fe, gr)[0]); nb = len(gr)
    # whole-run aligned ATE at the IMU point, and jump anatomy
    R, tr = vm.umeyama(es["pa"], es["gp"]); q = (R @ es["pa"].T).T + tr
    ate = float(np.sqrt(np.mean(np.sum((q[:, :2] - es["gp"][:, :2]) ** 2, axis=1))))
    ate3 = float(np.sqrt(np.mean(np.sum((q - es["gp"]) ** 2, axis=1))))
    zrms = float(np.sqrt(np.mean((q[:, 2] - es["gp"][:, 2]) ** 2)))
    return dict(dataset=ds, name=rec["label"], ate=ate, ate3=ate3, zrms=zrms, regime_s=float(np.ptp(t[m])) if m.sum() > 1 else 0.0,
                turn=c[0] if c else np.nan, straight=c[1] if c else np.nan, T_turn=c[2] if c else np.nan, T_straight=c[3] if c else np.nan,
                rho=rho, rho_n=nb, n_jumps=es["n_jumps"], jump_t=(es["jump_t"] - t[0]) if es["jump_t"] else np.nan,
                final=float(err[-1]), maxerr=float(err.max()), es=es, rec=rec)


def orb_failure(ds, g_imu, rec):
    """Anatomy of the ORB_FAILURE_BAG ORB-SLAM3 failure: where the error is accrued."""
    e = vm.load(os.path.join(rec["dir"], "vio.csv"))
    t, p = e[0], e[1]
    m = (t >= g_imu["t"][0]) & (t <= g_imu["t"][-1])
    t, p = t[m], p[m]
    gp = vm.resample(g_imu["t"], g_imu["p"], t)
    R, tr = vm.umeyama(p, gp); pa = (R @ p.T).T + tr
    err = np.linalg.norm(pa[:, :2] - gp[:, :2], axis=1)
    jumps, steps = vm.find_jumps(t, p)
    de = np.diff(err)
    in_jump = np.zeros(len(de), bool); in_jump[jumps] = True
    acc_jump = float(de[in_jump].sum()); acc_drift = float(de[~in_jump].sum())
    fr = x3.frontend(rec)
    lost = None
    if fr is not None:
        sel = (fr["t"] >= t[0]) & (fr["t"] <= t[-1])
        st = fr["state"][sel]; ft = fr["t"][sel]
        lost = ft[st != 2]
    prog = np.interp(t, g_imu["t"], g_imu["progress"])
    return dict(t=t - t[0], err=err, pa=pa, gp=gp, jumps=jumps, jump_t=t[jumps + 1] - t[0] if len(jumps) else np.array([]),
                jump_size=steps[jumps] if len(jumps) else np.array([]), acc_jump=acc_jump, acc_drift=acc_drift,
                lost_t=(lost - t[0]) if lost is not None else None, prog=prog, ate=float(np.sqrt((err ** 2).mean())))


# -------------------------------------------------------------------- tables

def fmt(v, p=2):
    return "---" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{p}f}"


def tex_decomposition(recs, path):
    L = [r"\begin{table}[htbp]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3.5pt}",
         r"\caption[Anchored error of the proprioceptive baselines split into along-track, cross-track and heading parts.]"
         r"{Anchored error of the proprioceptive baselines in the ground-truth path frame: along-track (distance) and "
         r"cross-track (heading) RMS and final values, the heading error itself, the path-length (scale) error of the "
         r"estimate, and the share of the final position error that the heading error alone reproduces when the true "
         r"velocity is integrated through it. Model-origin reference. Source: \path{output/compare/simulation/exp1-insights/decomposition.csv}.}",
         r"\label{tab:e1i-decomp}",
         r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
         r"Recording & Estimator & RMS & along & cross & final & heading RMS & scale & heading-only \\",
         r" & & [m] & RMS [m] & RMS [m] & [m] & [\si{\degree}] & [\si{\percent}] & share of final \\", r"\midrule"]
    for ds in BAGS:
        first = True
        for name, _, _ in BASELINES:
            r = next((x for x in recs if x["dataset"] == ds and x["name"] == name), None)
            if r is None:
                continue
            lead = ("\\repo{" + ds.replace("dataset_", "") + "}") if first else ""
            L.append(f"{lead} & {name} & {fmt(r['rms'])} & {fmt(r['along_rms'])} & "
                     f"{fmt(r['cross_rms'])} & {fmt(r['final'])} & {fmt(r['head_rms_deg'])} & {fmt(r['scale_pct'],1)} & "
                     f"{fmt(r['heading_explained_final'])} \\\\")
            first = False
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    path.write_text("\n".join(L) + "\n")


def tex_turning(recs, vis, path):
    L = [r"\begin{table}[htbp]", r"\centering", r"\footnotesize",
         r"\caption[Growth of position and heading error while turning against straights, every estimator.]"
         r"{Growth rate of the position error and of the heading error while the ground-truth yaw rate exceeds "
         r"\SI{10}{\degree\per\second} and on straights, median over the four recordings, with the number of recordings "
         r"in which the turning rate is the larger ($n_{>}$). Baselines: anchored error over the whole run. Visual "
         r"estimators: start-anchored error at the IMU point, tracking regime only (before the first jump, below "
         r"\SI{3}{\metre}); heading contrast not computed for them.}",
         r"\label{tab:e1i-turning}",
         r"\begin{tabular}{lrrrrrrr}", r"\toprule",
         r"Estimator & $n$ & \multicolumn{3}{c}{position [\si{\milli\metre\per\second}]} & \multicolumn{3}{c}{heading [\si{\milli\degree\per\second}]} \\",
         r" & & turning & straight & $n_{>}$ & turning & straight & $n_{>}$ \\", r"\midrule"]
    for name, _, _ in BASELINES:
        g = [r for r in recs if r["name"] == name and np.isfinite(r["turn_pos"])]
        if not g:
            continue
        tp = np.median([r["turn_pos"] for r in g]); sp = np.median([r["straight_pos"] for r in g])
        th = np.median([r["turn_head"] for r in g]); sh = np.median([r["straight_head"] for r in g])
        L.append(f"{name} & {len(g)} & {fmt(1000*tp,1)} & {fmt(1000*sp,1)} & {sum(r['turn_pos']>r['straight_pos'] for r in g)} & "
                 f"{fmt(1000*th,1)} & {fmt(1000*sh,1)} & {sum(r['turn_head']>r['straight_head'] for r in g)} \\\\")
    L.append(r"\midrule")
    for name, _, _, _ in VISUAL:
        g = [r for r in vis if r["name"] == name and np.isfinite(r["turn"])]
        if not g:
            continue
        tp = np.median([r["turn"] for r in g]); sp = np.median([r["straight"] for r in g])
        L.append(f"{name} & {len(g)} & {fmt(1000*tp,1)} & {fmt(1000*sp,1)} & {sum(r['turn']>r['straight'] for r in g)} & --- & --- & --- \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(L) + "\n")


def tex_imu(recs, path):
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption[IMU dead reckoning: growth law of the error and the share the heading error explains.]"
         r"{IMU dead reckoning: exponent $k$ of a power-law fit $|e| \propto t^{k}$ to the anchored position error "
         r"after the first \SI{5}{\second} (with its $R^2$), the final heading error, and the share of the final "
         r"position error that integrating the true velocity through the heading error alone reproduces. A share near "
         r"one means the position error is heading error integrated along the path; a share well below one means the "
         r"velocity (accelerometer) integration itself is wrong.}",
         r"\label{tab:e1i-imu}",
         r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Recording & $k$ & $R^2$ & final error [m] & final heading [\si{\degree}] & heading-only share \\", r"\midrule"]
    for ds in BAGS:
        r = next((x for x in recs if x["dataset"] == ds and x["name"] == "IMU dead reckoning"), None)
        if r is None:
            continue
        k, r2 = r["law"] if isinstance(r["law"], tuple) else (np.nan, np.nan)
        L.append(f"\\repo{{{ds.replace('dataset_', '')}}} & {fmt(k)} & {fmt(r2)} & {fmt(r['final'],1)} & {fmt(r['head_final_deg'],1)} & {fmt(r['heading_explained_final'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(L) + "\n")


def tex_nees(rows, path):
    L = [r"\begin{table}[htbp]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\caption[When the EKF's NEES leaves its band, against the first turn.]"
         r"{When the EKF's 3-DOF NEES first stays above the \SI{97.5}{\percent} band (\num{9.35}) for \SI{5}{\second}, "
         r"as time since the start and as distance travelled, against the time of the first ground-truth heading change "
         r"(yaw rate above \SI{10}{\degree\per\second}); for the EKF with its measured per-recording noise model and for "
         r"the same filter on the VSLAM noise file. NEES sampled at \SI{1}{\hertz} from the logged covariance. "
         r"``never'': the band is not left for \SI{5}{\second} in the run.}",
         r"\label{tab:e1i-nees}",
         r"\begin{tabular}{lrrrrrrr}", r"\toprule",
         r"Recording & first turn & \multicolumn{3}{c}{EKF @ measured noise} & \multicolumn{3}{c}{EKF @ VSLAM noise} \\",
         r" & [s] & leaves at [s] & at [m] & median NEES & leaves at [s] & at [m] & median NEES \\", r"\midrule"]
    for r in rows:
        L.append(f"\\repo{{{r['dataset'].replace('dataset_', '')}}} & {fmt(r['first_turn'],1)} & "
                 f"{fmt(r['cross_m'],1) if r['cross_m'] is not None else 'never'} & {fmt(r['dist_m'],1) if r['dist_m'] is not None else '---'} & {fmt(r['med_m'],1)} & "
                 f"{fmt(r['cross_v'],1) if r['cross_v'] is not None else 'never'} & {fmt(r['dist_v'],1) if r['dist_v'] is not None else '---'} & {fmt(r['med_v'],1)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(L) + "\n")


def tex_visual(vis, fail, path):
    L = [r"\begin{table}[htbp]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
         r"\caption[Visual estimators at the IMU point: drift while turning, features against drift, jumps.]"
         r"{Visual estimators scored at the IMU point after the initialisation transient: whole-run aligned ATE in 3-D "
         r"(as the chapter reports it) and in the plane, the vertical RMS error that separates them, length of the tracking regime, "
         r"drift rate while turning and on straights in that regime, Spearman $\rho$ between the matched-feature "
         r"count (ORB-SLAM3 map inliers, VINS-Fusion tracked features) and the drift rate in \SI{2}{\second} bins, "
         r"and the number of impossible-speed jumps of the estimate.}",
         r"\label{tab:e1i-visual}",
         r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
         r"Recording & Estimator & ATE 3-D & ATE planar & vertical & regime & turning & straight & $\rho$ (feat., drift) & jumps \\",
         r" & & [m] & [m] & RMS [m] & [s] & \multicolumn{2}{c}{[\si{\milli\metre\per\second}]} & (bins) & \\", r"\midrule"]
    for ds in BAGS:
        first = True
        for name, _, _, _ in VISUAL:
            r = next((x for x in vis if x["dataset"] == ds and x["name"] == name), None)
            if r is None:
                continue
            lead = ("\\repo{" + ds.replace("dataset_", "") + "}") if first else ""
            L.append(f"{lead} & {name} & {fmt(r['ate3'])} & {fmt(r['ate'])} & {fmt(r['zrms'])} & {fmt(r['regime_s'],0)} & "
                     f"{fmt(1000*r['turn'],1)} & {fmt(1000*r['straight'],1)} & {fmt(r['rho'])} ({r['rho_n']}) & {r['n_jumps']} \\\\")
            first = False
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    if fail is not None:
        L += ["", r"\begin{table}[htbp]", r"\centering", r"\small",
              r"\caption[Anatomy of the ORB-SLAM3 failure on the I layout.]{Anatomy of the ORB-SLAM3 (loop closer on) run "
              "on \\repo{" + ORB_FAILURE_BAG.replace("dataset_", "") + "}: whole-run aligned ATE, the number and sizes "
              r"of impossible-speed jumps, when they "
              r"occur, how much of the net error change is taken in jump steps against ordinary drift, and the "
              r"tracking-state losses the front end logged.}",
              r"\label{tab:e1i-orbfail}",
              r"\begin{tabular}{lr}", r"\toprule",
              f"Aligned ATE [m] & {fmt(fail['ate'])} \\\\",
              f"Jumps & {len(fail['jump_t'])} \\\\",
              f"Jump times [s] & {', '.join(fmt(x,1) for x in fail['jump_t'][:8])}{' ...' if len(fail['jump_t']) > 8 else ''} \\\\",
              f"Jump sizes [m] & {', '.join(fmt(x,2) for x in fail['jump_size'][:8])}{' ...' if len(fail['jump_size']) > 8 else ''} \\\\",
              f"Net error change in jump steps [m] & {fmt(fail['acc_jump'])} \\\\",
              f"Net error change in ordinary steps [m] & {fmt(fail['acc_drift'])} \\\\",
              f"Frames not tracking OK & {len(fail['lost_t']) if fail['lost_t'] is not None else '---'} \\\\",
              f"Route progress at the largest jump & {fmt(float(np.interp(fail['jump_t'][int(np.argmax(fail['jump_size']))], fail['t'], fail['prog'])),2) if len(fail['jump_t']) else '---'} \\\\",
              f"Time of the largest jump [s] & {fmt(float(fail['jump_t'][int(np.argmax(fail['jump_size']))]),1) if len(fail['jump_t']) else '---'} \\\\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(L) + "\n")


# -------------------------------------------------------------------- figures

def fig_decomposition(recs, path):
    fig, axes = plt.subplots(3, 5, figsize=(15, 8), sharex="col")
    for j, ds in enumerate(BAGS):
        for name in ("Wheel odometry", "EKF @ measured noise"):
            r = next((x for x in recs if x["dataset"] == ds and x["name"] == name), None)
            if r is None:
                continue
            s = r["series"]
            axes[0, j].plot(s["t"], s["along"], color=COL[name], lw=1.1, label=name)
            axes[1, j].plot(s["t"], s["cross"], color=COL[name], lw=1.1)
            axes[2, j].plot(s["t"], s["herr"], color=COL[name], lw=1.1)
        r = next((x for x in recs if x["dataset"] == ds and x["name"] == "Wheel odometry"), None)
        if r is not None:
            s = r["series"]
            for ax in axes[:, j]:
                ax.fill_between(s["t"], *ax.get_ylim(), where=s["turning"], color="#888", alpha=0.15, lw=0)
                ax.grid(True, alpha=0.3)
        axes[0, j].set_title(ds.replace("dataset_", ""), fontsize=9, loc="left")
        axes[2, j].set_xlabel("time [s]")
    axes[0, 0].set_ylabel("along-track error [m]"); axes[1, 0].set_ylabel("cross-track error [m]"); axes[2, 0].set_ylabel("heading error [deg]")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Anchored error of wheel odometry and the EKF in the path frame (grey: turning)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(path, dpi=150); plt.close(fig)


def fig_nees(series, rows, path):
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.6), sharey=True)
    for j, ds in enumerate(BAGS):
        ax = axes[j]
        for tree, lab, col in (("output_ekf_measured", "measured noise", COL["EKF @ measured noise"]), ("output_ekf", "VSLAM noise", COL["EKF @ VSLAM noise"])):
            ns = series.get((ds, tree))
            if ns is None:
                continue
            ax.plot(ns["t"] - ns["t"][0], ns["nees"], color=col, lw=1.0, label=lab)
        r = next((x for x in rows if x["dataset"] == ds), None)
        if r and r["first_turn"] is not None:
            ax.axvline(r["first_turn"], color="#444", lw=0.9, ls=":", label="first turn")
        ax.axhline(NEES_BAND, color="#b00020", lw=0.8, ls="--", label="97.5 % band")
        ax.axhline(3.0, color="#888", lw=0.8, ls="-")
        ax.set_yscale("log"); ax.set_title(ds.replace("dataset_", ""), fontsize=9, loc="left"); ax.set_xlabel("time [s]"); ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("NEES (3 DOF)"); axes[0].legend(fontsize=7)
    fig.suptitle("EKF consistency over time: NEES of the measured and the transferred noise model", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(path, dpi=150); plt.close(fig)


def fig_imu(recs, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ds in BAGS:
        r = next((x for x in recs if x["dataset"] == ds and x["name"] == "IMU dead reckoning"), None)
        if r is None:
            continue
        s = r["series"]; m = s["t"] > 1
        axes[0].loglog(s["t"][m], s["dist"][m], lw=1.1, label=ds.replace("dataset_", ""))
        axes[1].plot(s["t"], s["dist"], lw=1.1, label=f"{ds.replace('dataset_', '')} error")
        axes[1].plot(s["t"], s["pred"], lw=1.0, ls="--", color=axes[1].lines[-1].get_color())
    tt = np.array([5, 100]); axes[0].loglog(tt, 0.002 * tt ** 2, color="#444", ls=":", lw=0.9, label="$t^2$ guide")
    axes[0].loglog(tt, 0.1 * tt, color="#888", ls=":", lw=0.9, label="$t$ guide")
    axes[0].set_xlabel("time [s]"); axes[0].set_ylabel("anchored error [m]"); axes[0].legend(fontsize=7); axes[0].grid(True, which="both", alpha=0.3)
    axes[1].set_xlabel("time [s]"); axes[1].set_ylabel("error [m]  (dashed: heading-only prediction)"); axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)
    fig.suptitle("IMU dead reckoning: growth law, and the error the heading error alone would produce", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(path, dpi=150); plt.close(fig)


def fig_orb(fail, path):
    fig = plt.figure(figsize=(10, 4.2)); gs = fig.add_gridspec(1, 2, width_ratios=(2, 1))
    ax = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1])
    ax.plot(fail["t"], fail["err"], color=COL["ORB-SLAM3"], lw=1.2, label="aligned error")
    for jt in fail["jump_t"]:
        ax.axvline(jt, color="#b00020", lw=0.8, alpha=0.6)
    if fail["lost_t"] is not None and len(fail["lost_t"]):
        ax.scatter(fail["lost_t"], np.zeros(len(fail["lost_t"])), s=6, color="#e040fb", label="not tracking OK")
    ax.set_xlabel("time [s]"); ax.set_ylabel("error [m]"); ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
    ax.set_title(ORB_FAILURE_BAG.replace("dataset_", "") + ", ORB-SLAM3 (loop closer on): error and jumps (red)",
                 fontsize=9, loc="left")
    draw_map(ax2, "simulation/" + ORB_FAILURE_BAG)
    ax2.plot(fail["gp"][:, 0], fail["gp"][:, 1], color="#8a8983", lw=2, label="ground truth (IMU point)")
    ax2.plot(fail["pa"][:, 0], fail["pa"][:, 1], color=COL["ORB-SLAM3"], lw=0.9, label="ORB-SLAM3 aligned")
    if len(fail["jumps"]):
        ax2.scatter(fail["gp"][fail["jumps"] + 1, 0], fail["gp"][fail["jumps"] + 1, 1], s=30, color="#b00020", zorder=5, label="jump")
    ax2.set_aspect("equal"); ax2.legend(fontsize=7); ax2.set_xlabel("x [m]"); ax2.set_ylabel("y [m]")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ----------------------------------------------------------------------- main

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    recs, nees_rows, nees_ser, vis, fail = [], [], {}, [], None
    for ds in BAGS:
        g, gpath = gt_raw(ds)
        g_imu = tl.gt_series(gpath, lever=True)
        for name, tree, _ in BASELINES:
            r = decompose(ds, g, name, tree)
            if r is None:
                continue
            if name == "IMU dead reckoning":
                r["law"] = imu_law(r)
            recs.append(r)
        row = dict(dataset=ds, first_turn=None, cross_m=None, dist_m=None, med_m=np.nan, cross_v=None, dist_v=None, med_v=np.nan)
        ft = first_turn(g)
        row["first_turn"] = (ft - g["t"][0]) if ft is not None else None
        for tree, k in (("output_ekf_measured", "m"), ("output_ekf", "v")):
            ns = nees_series(ds, g, tree)
            if ns is None:
                continue
            nees_ser[(ds, tree)] = ns
            c = nees_crossing(ns)
            row[f"cross_{k}"] = (c - ns["t"][0]) if c is not None else None
            row[f"dist_{k}"] = (dist_at(g, c) - dist_at(g, ns["t"][0])) if c is not None else None
            row[f"med_{k}"] = float(np.nanmedian(ns["nees"]))
        nees_rows.append(row)
        for name, tree, run, _ in VISUAL:
            rec = visual_run(ds, name, tree, run)
            if rec is None:
                continue
            v = visual_contrasts(ds, g_imu, rec)
            if v is not None:
                vis.append(v)
            if ds == ORB_FAILURE_BAG and name == "ORB-SLAM3":
                fail = orb_failure(ds, g_imu, rec)
    # csv
    keys = [k for k in recs[0] if k not in ("series", "law")] + ["law_k", "law_r2"]
    with open(OUT / "decomposition.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore"); w.writeheader()
        for r in recs:
            law = r.get("law") if isinstance(r.get("law"), tuple) else (np.nan, np.nan)
            w.writerow({**{k: r.get(k) for k in keys}, "law_k": law[0], "law_r2": law[1]})
    with open(OUT / "visual.csv", "w", newline="") as fh:
        ks = [k for k in vis[0] if k not in ("es", "rec")]
        w = csv.DictWriter(fh, fieldnames=ks, extrasaction="ignore"); w.writeheader()
        for r in vis:
            w.writerow({k: r[k] for k in ks})
    tex_decomposition(recs, OUT / "decomposition.tex")
    tex_turning(recs, vis, OUT / "turning.tex")
    tex_imu(recs, OUT / "imu_law.tex")
    tex_nees(nees_rows, OUT / "nees_timing.tex")
    tex_visual(vis, fail, OUT / "visual.tex")
    fig_decomposition(recs, OUT / "fig_decomposition_light.png")
    fig_nees(nees_ser, nees_rows, OUT / "fig_nees_light.png")
    fig_imu(recs, OUT / "fig_imu_law_light.png")
    if fail is not None:
        fig_orb(fail, OUT / "fig_orb_failure_light.png")
    # digest
    print("== decomposition (per bag): along / cross RMS [m], heading RMS [deg], scale %, heading-only share")
    for r in recs:
        print(f"  {r['dataset'][-3:]} {r['name']:20s} rms {r['rms']:6.2f} along {r['along_rms']:6.2f} cross {r['cross_rms']:6.2f} head {r['head_rms_deg']:6.2f} scale {r['scale_pct']:6.2f} hshare {r['heading_explained_final']:5.2f} "
              f"turn/str pos {1000*r['turn_pos']:7.1f}/{1000*r['straight_pos']:7.1f} mm/s head {1000*r['turn_head']:7.1f}/{1000*r['straight_head']:7.1f} mdeg/s"
              + (f" law k={r['law'][0]:.2f} r2={r['law'][1]:.2f}" if isinstance(r.get('law'), tuple) else ""))
    print("== NEES timing")
    for r in nees_rows:
        print(f"  {r['dataset'][-3:]} first turn {fmt(r['first_turn'],1)} s | measured leaves {fmt(r['cross_m'],1) if r['cross_m'] is not None else 'never'} s ({fmt(r['dist_m'],1) if r['dist_m'] is not None else '-'} m) med {fmt(r['med_m'],1)} | vslam leaves {fmt(r['cross_v'],1) if r['cross_v'] is not None else 'never'} s ({fmt(r['dist_v'],1) if r['dist_v'] is not None else '-'} m) med {fmt(r['med_v'],1)}")
    print("== visual")
    for r in vis:
        print(f"  {r['dataset'][-3:]} {r['name']:18s} ate3d {r['ate3']:.3f} planar {r['ate']:.3f} z {r['zrms']:.3f} regime {r['regime_s']:5.0f}s turn/str {1000*r['turn']:6.1f}/{1000*r['straight']:6.1f} rho {fmt(r['rho'])} (n={r['rho_n']}) jumps {r['n_jumps']} first jump {fmt(r['jump_t'],1)}")
    if fail:
        print(f"== ORB 003: ate {fail['ate']:.2f} jumps {len(fail['jump_t'])} at {[round(x,1) for x in fail['jump_t']]} sizes {[round(x,2) for x in fail['jump_size']]} acc_jump {fail['acc_jump']:.2f} acc_drift {fail['acc_drift']:.2f} lost frames {len(fail['lost_t']) if fail['lost_t'] is not None else '-'}")


if __name__ == "__main__":
    main()
