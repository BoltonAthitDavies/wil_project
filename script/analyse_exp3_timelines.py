#!/usr/bin/env python3
"""Experiment 3 timelines: when the error rises, where on the route, how long the
filter ran, and how much image was left after masking.

    python3 script/analyse_exp3_timelines.py

Asked for on 21 October 2026, modelled on the progress-update slides that showed a
"straight path" and a "heading-change path" failure with the ground-truth route
marked. Those slides were viewer screenshots; this is the same question answered
from the logs, for every Experiment 3 run in both testbeds.

DEFINITIONS (fixed before the numbers were read)
  error series     start-anchored aligned position error: the estimate is aligned
                   to ground truth by a rigid SE(3) fit over an ANCHOR WINDOW near
                   the start (scale fixed), then that one transform is applied to
                   the whole run. A global fit would spread a late divergence back
                   over the early, good part of the run and hide where it began.
                   The window begins 0.5 s after the last discontinuity within the
                   first 15 s (ORB-SLAM3 re-seats its world frame to gravity about
                   2.6 s in; anchoring across that step mis-aligns the whole run) and
                   ends when the ground truth has turned 60 deg since then, with a
                   floor of 10 s and a ceiling of 40 s: a fit on a straight leaves yaw
                   unobserved and turns into a spurious, growing error. Samples before
                   the window start are not scored.
  threshold        3 x the pooled median of that error over the estimator's STATIC
                   runs in the same testbed (printed to thresholds.csv), so the
                   rule is anchored in the estimator's own static behaviour.
  onset            first time the error exceeds the threshold and stays above it
                   for at least 1 s. "never" if it does not.
  where            ground-truth position at onset; the segment is "heading change"
                   when the 1 s-smoothed ground-truth yaw rate there exceeds 10 deg/s
                   and "straight" otherwise; route progress is the fraction of the
                   ground-truth path length driven by then.
  first jump       first step implying more than 10 m/s (vio_metrics.find_jumps),
                   reported beside the onset for comparison.
  filter working   from yolo_mask.csv: a 5 s window is "working" when its matched
                   fraction is >= 0.98 (the admissibility gate). "Working until" is
                   the end of the initial stretch of working windows: "never" if the
                   first window fails, "whole run" if none fails.
  unmasked area    100 x (1 - mask_coverage) per matched frame; median and minimum.

Outputs, under output/compare/simulation/exp3-timelines/:
  thresholds.csv                 the static medians and thresholds used
  timelines.csv                  one row per run, every quantity above
  extreme_filtered_timeline.tex  realistic_filtered_timeline.tex   (one row per filtered run)
  extreme_baseline_timeline.tex  realistic_baseline_timeline.tex   (onset per repeat)
  timelines/<testbed>__<recording>__<system>__<run>.png   per-run figure
"""
import csv
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vio_metrics as vm                 # noqa: E402
import analyse_realistic_runs as ar      # noqa: E402
from map_underlay import draw_map        # noqa: E402

import matplotlib                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

ROOT = Path(vm.ROOT)
OUT = ROOT / "output" / "compare" / "simulation" / "exp3-timelines"
ANCHOR_MIN_S, ANCHOR_MAX_S, ANCHOR_TURN_DEG, INIT_SEARCH_S = 10.0, 40.0, 60.0, 15.0
SUSTAIN_S = 1.0
THR_FACTOR = 3.0
TURN_DEG_S = 10.0
WIN_S = 5.0
MATCH_OK = 0.98
SEPT = re.compile(r"logging_20260916_\d+")
TREES = {"output_vins": "VINS-Fusion", "output_orb": "ORB-SLAM3",
         "output_vins_yolo": "VINS-Fusion+YOLO", "output_orb_yolo": "ORB-SLAM3+YOLO"}
COLORS = dict(ar.COLORS)
TH = ar.THEMES["light"]


def base_of(s): return s.replace("+YOLO", "")
def short(ds): return ds.replace("dataset_", "").replace("_nofloortexture", "").replace("allsensor_", "")


# -------------------------------------------------------------------- registry

def registry():
    runs = []
    for tree, sysn in TREES.items():
        for dsdir in sorted((ROOT / "output" / tree / "simulation").glob("dataset_*_nofloortexture_*")):
            for d in sorted(dsdir.iterdir()):
                if not (d / "vio.csv").exists():
                    continue
                meta = ar.kv(str(d / "run_metadata.csv"))
                m = re.match(r"(class|motion)", d.name)
                runs.append(dict(testbed="extreme", dataset=dsdir.name, tree=tree, system=sysn, run=d.name, dir=str(d),
                                 filtered=sysn.endswith("+YOLO"),
                                 mode=(meta.get("filter_mode") or (m.group(1) if m else "class")) if sysn.endswith("+YOLO") else "off",
                                 lc=(ar.loop_closer_state(str(d)) if sysn.startswith("ORB") and not SEPT.fullmatch(d.name) else ("active" if sysn.startswith("ORB") else "n/a")),
                                 atlas=False, september=bool(SEPT.fullmatch(d.name)),
                                 gt=str(ROOT / "output" / "compare" / "relogged_20260916" / "ground_truth" / (dsdir.name + ".csv"))))
    for r in ar.discover():
        sysn = r["estimator"] + ("+YOLO" if r["filtered"] else "")
        runs.append(dict(testbed="realistic", dataset=r["dataset"], tree=r["tree"], system=sysn, run=r["run"], dir=r["dir"],
                         filtered=r["filtered"], mode=r["mode"],
                         lc=ar.loop_closer_state(r["dir"]) if sysn.startswith("ORB") else "n/a",
                         atlas=r["atlas"], september=False,
                         gt=str(ROOT / "output" / "output_vins" / "simulation" / r["dataset"] / "ground_truth.csv")))
    return runs


def is_static(run):
    return ("static_" in run["dataset"]) or run["dataset"].endswith(("realistic_000", "realistic_001"))


# --------------------------------------------------------------------- series

def gt_series(path, lever=True):
    """Ground-truth series; by default moved to the IMU point both estimators
    report (vio_metrics.IMU_LEVER_M). lever=False keeps the model origin."""
    g = vm.load_gt(path, lever=lever)
    t, p = g[0], g[1]
    yaw = np.unwrap(np.radians(vm.euler_zyx(vm.quat_to_R(g[2]))[:, 0]))
    w = np.degrees(np.gradient(yaw, t))
    yaw_unwrapped = yaw
    k = max(1, int(round(1.0 / max(np.median(np.diff(t)), 1e-3))))
    w_s = np.convolve(np.abs(w), np.ones(k) / k, mode="same")
    seg = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1))])
    return dict(t=t, p=p, yaw_rate=w_s, yaw_unwrapped=yaw_unwrapped, progress=seg / max(seg[-1], 1e-9), path_m=seg[-1])


def error_series(run, gt):
    e = vm.load(os.path.join(run["dir"], "vio.csv"))
    if e is None:
        return None
    t, p = e[0], e[1]
    lo, hi = max(t[0], gt["t"][0]), min(t[-1], gt["t"][-1])
    m = (t >= lo) & (t <= hi)
    if m.sum() < 30:
        return None
    t, p = t[m], p[m]
    jumps_all = vm.find_jumps(t, p)[0]
    first_jump_t = t[jumps_all[0] + 1] if len(jumps_all) else None
    # anchor start: after the initialisation transient
    early = [j for j in jumps_all if t[j + 1] <= t[0] + INIT_SEARCH_S]
    t_a = (t[early[-1] + 1] + 0.5) if early else t[0]
    keep = t >= t_a
    if keep.sum() < 30:
        return None
    t, p = t[keep], p[keep]
    gp = vm.resample(gt["t"], gt["p"], t)
    # anchor end: when the ground truth has turned ANCHOR_TURN_DEG since t_a
    gyaw = np.interp(t, gt["t"], gt["yaw_unwrapped"])
    turned = np.degrees(np.abs(gyaw - gyaw[0]))
    t_turn = t[np.argmax(turned >= ANCHOR_TURN_DEG)] if (turned >= ANCHOR_TURN_DEG).any() else t[-1]
    t_end = min(max(t_turn, t[0] + ANCHOR_MIN_S), t[0] + ANCHOR_MAX_S)
    a = t <= t_end
    if a.sum() < 10:
        a[:10] = True
    R, tr = vm.umeyama(p[a], gp[a])
    pa = (R @ p.T).T + tr
    err = np.linalg.norm(pa[:, :2] - gp[:, :2], axis=1)   # planar: all these worlds are flat
    jumps = vm.find_jumps(t, p)[0]
    return dict(t=t, err=err, pa=pa, gp=gp, jump_t=first_jump_t, n_jumps=int(len(jumps_all)),
                anchor=(t_a, t_end))


def onset(t, err, thr):
    above = err > thr
    i = 0
    while i < len(t):
        if above[i]:
            j = np.searchsorted(t, t[i] + SUSTAIN_S)
            if j <= len(t) and above[i:j].all() and (j == len(t) or t[j - 1] - t[i] >= SUSTAIN_S * 0.8):
                return t[i]
            i += 1
        else:
            i += 1
    return None


def filter_series(run):
    p = os.path.join(run["dir"], "yolo_mask.csv")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        rd = csv.DictReader(fh)
        rows = list(rd)
    if not rows:
        return None
    t = np.array([int(r["timestamp_ns"]) for r in rows]) * 1e-9
    matched = np.array([int(float(r["matched"])) for r in rows])
    cov = np.array([float(r["mask_coverage"]) for r in rows])
    return dict(t=t, matched=matched, cov=cov)


def feature_budget(run):
    """The estimator's per-frame feature budget, read from the run's own config:
    ORB-SLAM3 `ORBextractor.nFeatures` (the extractor fills this quota every
    frame, so it is the 100 % mark), VINS-Fusion `max_cnt` (the tracker's cap)."""
    meta = ar.kv(os.path.join(run["dir"], "run_metadata.csv"))
    cfg = meta.get("config_file", "")
    key = "ORBextractor.nFeatures" if run["system"].startswith("ORB") else "max_cnt"
    fallback = 1500.0 if run["system"].startswith("ORB") else 150.0
    try:
        for line in open(cfg):
            if line.strip().startswith(key):
                return float(re.split(r"[:\s]+", line.strip())[1]), cfg
    except OSError:
        pass
    return fallback, "fallback"


def feature_series(run):
    """Per-frame feature count: ORB-SLAM3 `features` from tracking_frontend.csv
    (keypoints that survived extraction, hence after the mask when the filter is
    on); VINS-Fusion `feature_count` from frontend.csv (tracked features)."""
    orb = run["system"].startswith("ORB")
    p = os.path.join(run["dir"], "tracking_frontend.csv" if orb else "frontend.csv")
    col = "features" if orb else "feature_count"
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        rows = list(csv.DictReader(fh))
    if not rows or col not in rows[0]:
        return None
    t = np.array([int(r["timestamp_ns"]) for r in rows]) * 1e-9
    n = np.array([float(r[col] or "nan") for r in rows])
    budget, src = feature_budget(run)
    return dict(t=t, n=n, pct=100.0 * n / budget, budget=budget, budget_src=src)


def filter_stats(fs, t0, t_end):
    if fs is None or len(fs["t"]) == 0:
        return dict(match_rate=float("nan"), working_until=None, working_label="no mask log",
                    unmasked_median=float("nan"), unmasked_min=float("nan"), windows_ok=None)
    rel = fs["t"] - t0
    edges = np.arange(rel[0], rel[-1] + WIN_S, WIN_S)
    ok = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (rel >= a) & (rel < b)
        if sel.sum() >= 5:
            ok.append((a, b, fs["matched"][sel].mean() >= MATCH_OK))
    if not ok:
        label, until = "no mask log", None
    elif not ok[0][2]:
        label, until = "never", 0.0
    else:
        until = None
        for a, b, good in ok:
            if not good:
                until = a; break
        label = "whole run" if until is None else f"{until:.0f} s"
        if until is None:
            until = ok[-1][1]
    mt = fs["matched"] == 1
    unm = 100 * (1 - fs["cov"][mt]) if mt.any() else np.array([float("nan")])
    return dict(match_rate=float(fs["matched"].mean()), working_until=until, working_label=label,
                unmasked_median=float(np.nanmedian(unm)), unmasked_min=float(np.nanmin(unm)),
                windows_ok=ok)


# --------------------------------------------------------------------- figure

def figure(run, gt, es, fs, thr, res, path):
    fig = plt.figure(figsize=(12, 5.2), facecolor=TH["surface"])
    gs = fig.add_gridspec(2, 2, width_ratios=[1.75, 1], height_ratios=[2.2, 1], hspace=0.08, wspace=0.18)
    ax = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[1, 0], sharex=ax); ax3 = fig.add_subplot(gs[:, 1])
    for a in (ax, ax2, ax3):
        ar.style(a, TH)
    t0 = gt["t"][0]
    tr = es["t"] - t0
    col = COLORS.get(run["system"], "#999")
    # filter-active shading on both time panels
    if fs is not None:
        m = fs["matched"] == 1
        if m.any():
            rt = fs["t"] - t0
            edges = np.flatnonzero(np.diff(np.concatenate([[0], m.astype(int), [0]])))
            for a, b in zip(edges[::2], edges[1::2]):
                for axx in (ax, ax2):
                    axx.axvspan(rt[a], rt[min(b, len(rt) - 1)], color="#2e7d5b", alpha=0.10, lw=0)
    ax.axvspan(es["anchor"][0] - t0, es["anchor"][1] - t0, color=TH["ink2"], alpha=0.08, lw=0, label="anchor window")
    ax.plot(tr, es["err"], color=col, lw=1.4, label="start-anchored position error")
    ax.axhline(thr, color=TH["ink2"], lw=0.9, ls="--", label="threshold = 3 x static median (%.2f m)" % thr)
    if res["onset_s"] is not None:
        ax.axvline(res["onset_s"], color="#b00020", lw=1.2,
                   label="onset %.1f s, %s (%.0f %% of route)" % (res["onset_s"], res["segment"], 100 * res["progress"]))
    if es["jump_t"] is not None:
        ax.axvline(es["jump_t"] - t0, color=TH["ink2"], lw=0.8, ls=":", label="first jump %.1f s" % (es["jump_t"] - t0))
    ymax = np.nanpercentile(es["err"], 99)
    ax.set_ylim(0, max(1.5 * thr, min(ymax * 1.15, 50.0)))
    ax.set_ylabel("error [m]"); ax.legend(fontsize=8, loc="upper right", facecolor=TH["surface"], edgecolor=TH["grid"], framealpha=0.92)
    plt.setp(ax.get_xticklabels(), visible=False)
    if fs is not None:
        ax2.plot(fs["t"] - t0, 100 * (1 - fs["cov"]), color="#52514e", lw=1.0, label="unmasked image area [%]")
        ax2.step(fs["t"] - t0, 100 * fs["matched"], color="#2e7d5b", lw=0.9, where="post", alpha=0.8, label="mask present (0/100)")
        ax2.set_ylim(-3, 103)
        ax2.legend(fontsize=7.5, loc="lower left", ncol=2, facecolor=TH["surface"], edgecolor=TH["grid"])
    else:
        ax2.text(0.5, 0.5, "no per-frame mask log (filter off)", transform=ax2.transAxes, ha="center", color=TH["ink2"], fontsize=9)
    ax2.set_xlabel("time since ground-truth start [s]"); ax2.set_ylabel("%")
    # route inset
    turn = gt["yaw_rate"] > TURN_DEG_S
    draw_map(ax3, run["dataset"])
    ax3.plot(gt["p"][:, 0], gt["p"][:, 1], color=TH["ink"], lw=1.2, label="ground truth")
    ax3.scatter(gt["p"][turn, 0], gt["p"][turn, 1], s=4, color="#cc8319", label="heading change (>%g deg/s)" % TURN_DEG_S, zorder=3)
    ax3.plot(es["pa"][:, 0], es["pa"][:, 1], color=col, lw=0.9, alpha=0.85, label="estimate, start-anchored")
    if res["onset_s"] is not None:
        gx = np.interp(res["onset_s"] + t0, gt["t"], gt["p"][:, 0]); gy = np.interp(res["onset_s"] + t0, gt["t"], gt["p"][:, 1])
        ax3.plot(gx, gy, "o", ms=11, mfc="none", mec="#b00020", mew=2.2, label="onset", zorder=5)
    lo, hi = gt["p"][:, :2].min(0), gt["p"][:, :2].max(0); pad = 0.12 * (hi - lo).max()
    ax3.set_xlim(lo[0] - pad, hi[0] + pad); ax3.set_ylim(lo[1] - pad, hi[1] + pad); ax3.set_aspect("equal")
    ax3.set_xlabel("x [m]"); ax3.set_ylabel("y [m]")
    ax3.legend(fontsize=7.5, loc="best", facecolor=TH["surface"], edgecolor=TH["grid"])
    fig.suptitle("%s / %s / %s / %s%s" % (run["testbed"], short(run["dataset"]), run["system"], run["run"],
                                          "  [prior atlas]" if run["atlas"] else ""), color=TH["ink"], fontsize=11, x=0.02, ha="left")
    fig.savefig(path, dpi=140, facecolor=TH["surface"], bbox_inches="tight"); plt.close(fig)


# -------------------------------------------------------------- dataset figure

def figure_dataset(testbed, ds, items, gt, thr_by_est, path):
    """All runs of one recording in the single-run layout: error traces (one per
    run, colour by system, dashed when filtered), a filter-activity row per
    filtered run, and the route with every onset marked."""
    filt = [it for it in items if it["run"]["filtered"]]
    n_f = len(filt)
    fig = plt.figure(figsize=(12, 4.6 + 0.26 * n_f), facecolor=TH["surface"])
    gs = fig.add_gridspec(2, 2, width_ratios=[1.75, 1], height_ratios=[2.4, max(0.35, 0.22 * n_f)], hspace=0.08, wspace=0.18)
    ax = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[1, 0], sharex=ax); ax3 = fig.add_subplot(gs[:, 1])
    for a in (ax, ax2, ax3):
        ar.style(a, TH)
    t0 = gt["t"][0]
    seen = set()
    for it in items:
        r, es, res = it["run"], it["es"], it["res"]
        col = COLORS.get(r["system"], "#999"); ls = "--" if r["filtered"] else "-"
        lab = r["system"] + (" (filtered, dashed)" if r["filtered"] else "")
        ax.plot(es["t"] - t0, np.maximum(es["err"], 0.02), color=col, lw=1.0, ls=ls, alpha=0.9,
                label=None if lab in seen else lab); seen.add(lab)
        if res["onset_s"] is not None:
            ax.plot(res["onset_s"], max(np.interp(res["onset_s"] + t0, es["t"], es["err"]), 0.02), "o", ms=6,
                    color=col, mec=TH["surface"], mew=1.0, zorder=5)
    for est, th in thr_by_est.items():
        ax.axhline(th, color=COLORS.get(est, TH["ink2"]), lw=0.9, ls=":", label="threshold %s (%.2f m)" % (est, th))
    ax.set_yscale("log"); ax.set_ylim(0.02, max(60, 1.2 * max((np.nanmax(it["es"]["err"]) for it in items), default=1)))
    ax.set_ylabel("start-anchored position error [m]")
    ax.plot([], [], "o", color=TH["ink2"], ms=6, label="onset (sustained excess)")
    ax.legend(fontsize=7.5, loc="upper left", ncol=2, facecolor=TH["surface"], edgecolor=TH["grid"], framealpha=0.92)
    plt.setp(ax.get_xticklabels(), visible=False)
    if n_f:
        for i, it in enumerate(sorted(filt, key=lambda x: (x["run"]["system"], x["run"]["run"]))):
            fs, r, res = it["fs"], it["run"], it["res"]
            y = n_f - 1 - i
            if fs is not None:
                rt = fs["t"] - t0; m = fs["matched"] == 1
                edges = np.flatnonzero(np.diff(np.concatenate([[0], m.astype(int), [0]])))
                for a, b in zip(edges[::2], edges[1::2]):
                    ax2.axvspan(rt[a], rt[min(b, len(rt) - 1)], ymin=(y + 0.1) / n_f, ymax=(y + 0.9) / n_f, color="#2e7d5b", alpha=0.55, lw=0)
            name = r["run"].replace("logging_20260916_", "rep ").replace("_noatlas", "").replace("noloopclosing", "no-LC").replace("loopclosing", "LC")
            ax2.text(-0.01, (y + 0.5) / n_f, "%s %s" % (r["system"].replace("-Fusion", "").replace("-SLAM3", ""), name),
                     transform=ax2.transAxes, ha="right", va="center", fontsize=6.5, color=TH["ink2"])
            if np.isfinite(res["unmasked_median"]):
                ax2.text(0.995, (y + 0.5) / n_f, "%.0f %% of image left" % res["unmasked_median"], transform=ax2.transAxes,
                         ha="right", va="center", fontsize=6.5, color=TH["ink2"],
                         bbox=dict(facecolor=TH["surface"], edgecolor="none", alpha=0.85, pad=1.0))
        ax2.set_ylim(0, 1); ax2.set_yticks([])
        ax2.set_title("mask present (green), one row per filtered run", fontsize=7.5, color=TH["ink2"], loc="left", pad=2)
    else:
        ax2.text(0.5, 0.5, "no filtered runs on this recording", transform=ax2.transAxes, ha="center", color=TH["ink2"], fontsize=9)
        ax2.set_yticks([])
    ax2.set_xlabel("time since ground-truth start [s]")
    turn = gt["yaw_rate"] > TURN_DEG_S
    draw_map(ax3, ds)
    ax3.plot(gt["p"][:, 0], gt["p"][:, 1], color=TH["ink"], lw=1.2, label="ground truth")
    ax3.scatter(gt["p"][turn, 0], gt["p"][turn, 1], s=4, color="#cc8319", label="heading change (>%g deg/s)" % TURN_DEG_S, zorder=3)
    for it in items:
        res, r = it["res"], it["run"]
        if res["onset_s"] is not None:
            gx = np.interp(res["onset_s"] + t0, gt["t"], gt["p"][:, 0]); gy = np.interp(res["onset_s"] + t0, gt["t"], gt["p"][:, 1])
            ax3.plot(gx, gy, "o" if not r["filtered"] else "s", ms=7, mfc=COLORS.get(r["system"], "#999"), mec=TH["surface"], mew=0.8, alpha=0.85, zorder=5)
    ax3.plot([], [], "o", color=TH["ink2"], ms=7, label="onset, unfiltered run"); ax3.plot([], [], "s", color=TH["ink2"], ms=7, label="onset, filtered run")
    lo, hi = gt["p"][:, :2].min(0), gt["p"][:, :2].max(0); pad = 0.12 * (hi - lo).max()
    ax3.set_xlim(lo[0] - pad, hi[0] + pad); ax3.set_ylim(lo[1] - pad, hi[1] + pad); ax3.set_aspect("equal")
    ax3.set_xlabel("x [m]"); ax3.set_ylabel("y [m]")
    ax3.legend(fontsize=7, loc="best", facecolor=TH["surface"], edgecolor=TH["grid"])
    fig.suptitle("%s / %s: %d runs (%d filtered)" % (testbed, short(ds), len(items), n_f), color=TH["ink"], fontsize=11, x=0.02, ha="left")
    fig.savefig(path, dpi=140, facecolor=TH["surface"], bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------- dataset figure, 2 x 2 grid

# One hue per run inside a cell (a cell holds one system and one output tree, so
# the runs are the identities). Checked with the ported palette validator on
# 22 Oct 2026: lightness band, chroma, >= 3:1 on the light surface, adjacent
# normal-vision dE >= 15 and CVD dE >= 8 on adjacent legend entries; runs five to
# eight are dashed as well, so no pair relies on hue alone.
RUN_PALETTE = ["#cc8319", "#2a78d6", "#e040fb", "#458554", "#b00020", "#0097a7", "#a347b2", "#887b2c"]
TREE_LABEL = {"output_orb": "output_orb (unfiltered)", "output_vins": "output_vins (unfiltered)",
              "output_orb_yolo": "output_orb_yolo (filtered)", "output_vins_yolo": "output_vins_yolo (filtered)"}


def run_label(r):
    n = r["run"].replace("logging_20260916_", "rep ").replace("_noatlas", "").replace("noloopclosing", "no-LC").replace("loopclosing", "LC")
    if r["testbed"] == "realistic" and not r["filtered"] and r["system"].startswith("ORB"):
        n = {"loopclosing": "LC", "noloopclosing": "no-LC"}.get(r["run"], n)
    return n.replace("_", " ")


def figure_dataset_grid(testbed, ds, items, gt, thr_by_est, path):
    """Rows: ORB-SLAM3, VINS-Fusion. Columns: unfiltered tree, filtered tree.
    Each cell draws its own runs with one hue per run, the filtered cells carry
    their mask rows beneath, and one route inset at the bottom collects every
    onset (marker shape by tree, hue by estimator)."""
    cells = {}
    for it in items:
        r = it["run"]
        cells.setdefault((base_of(r["system"]), r["filtered"]), []).append(it)
    n_mask = {est: len(cells.get((est, True), [])) for est in ("ORB-SLAM3", "VINS-Fusion")}
    h_mask = [1.0 if n_mask["ORB-SLAM3"] else 0.4, 1.0 if n_mask["VINS-Fusion"] else 0.4]
    fig = plt.figure(figsize=(11.5, 17.5), facecolor=TH["surface"])
    gs = fig.add_gridspec(7, 2, height_ratios=[2.1, 1.0, h_mask[0], 2.1, 1.0, h_mask[1], 2.8], hspace=0.14, wspace=0.14,
                          top=0.955, bottom=0.035, left=0.07, right=0.98)
    budgets = {est: sorted({int(it["ff"]["budget"]) for it in items if it["ff"] is not None and base_of(it["run"]["system"]) == est})
               for est in ("ORB-SLAM3", "VINS-Fusion")}
    ymax = max(60, 1.2 * max((np.nanmax(it["es"]["err"]) for it in items), default=1))
    t0 = gt["t"][0]
    xlim = (-1.0, gt["t"][-1] - t0 + 1.0)   # one time axis for all four cells
    for row, est in enumerate(("ORB-SLAM3", "VINS-Fusion")):
        for col, filtered in enumerate((False, True)):
            ax = fig.add_subplot(gs[3 * row, col]); axf = fig.add_subplot(gs[3 * row + 1, col], sharex=ax)
            axm = fig.add_subplot(gs[3 * row + 2, col], sharex=ax)
            ar.style(ax, TH); ar.style(axf, TH); ar.style(axm, TH)
            runs = sorted(cells.get((est, filtered), []), key=lambda it: (it["run"]["mode"], it["run"]["lc"], it["run"]["atlas"], it["run"]["run"]))
            tree = runs[0]["run"]["tree"] if runs else ("output_%s%s" % ("orb" if est.startswith("ORB") else "vins", "_yolo" if filtered else ""))
            ax.set_title("%s%s -- %s, %d run%s" % (est, "+YOLO" if filtered else "", TREE_LABEL.get(tree, tree), len(runs), "" if len(runs) == 1 else "s"),
                         fontsize=9, loc="left", color=TH["ink"])
            for k, it in enumerate(runs):
                r, es, res = it["run"], it["es"], it["res"]
                col_k = RUN_PALETTE[k % len(RUN_PALETTE)]
                ax.plot(es["t"] - t0, np.maximum(es["err"], 0.02), color=col_k, lw=1.0, ls="-" if k < 4 else "--",
                        label=run_label(r) + (" (atlas)" if r["atlas"] else ""))
                if res["onset_s"] is not None:
                    ax.plot(res["onset_s"], max(np.interp(res["onset_s"] + t0, es["t"], es["err"]), 0.02), "o", ms=6, color=col_k, mec=TH["surface"], mew=1.0, zorder=5)
            ax.axhline(thr_by_est[est], color=TH["ink2"], lw=0.9, ls=":", label="threshold %.2f m" % thr_by_est[est])
            ax.set_yscale("log"); ax.set_ylim(0.02, ymax); ax.set_xlim(*xlim)
            if col == 0:
                ax.set_ylabel("error [m]")
            if runs:
                ax.legend(fontsize=6.5, loc="upper left", ncol=2, facecolor=TH["surface"], edgecolor=TH["grid"], framealpha=0.9)
            else:
                ax.text(0.5, 0.5, "no runs in this tree", transform=ax.transAxes, ha="center", color=TH["ink2"], fontsize=9)
            plt.setp(ax.get_xticklabels(), visible=False)
            # features per frame, same y-scale across the row so masking's cost is read directly
            for k, it in enumerate(runs):
                if it["ff"] is not None:
                    axf.plot(it["ff"]["t"] - t0, it["ff"]["pct"], color=RUN_PALETTE[k % len(RUN_PALETTE)], lw=0.8,
                             ls="-" if k < 4 else "--", alpha=0.9)
            axf.set_ylim(0, 108); axf.set_yticks([0, 50, 100]); axf.set_xlim(*xlim)
            axf.set_ylabel("features [%]", fontsize=8)
            axf.set_title("%s per frame, %% of budget (%s)" % (
                "ORB keypoints after extraction" if est.startswith("ORB") else "tracked features",
                "/".join(str(b) for b in budgets[est]) or "?"), fontsize=7, loc="left", color=TH["ink2"], pad=2)
            plt.setp(axf.get_xticklabels(), visible=False)
            # mask rows (filtered column only)
            if filtered and runs:
                # Image area left after masking, per frame: 100 x (1 - mask_coverage).
                # Drawn only where a mask was present; a gap means no detection
                # reached the tracker for that frame, so nothing was masked and the
                # filter was not acting -- which must not read as "100 % left".
                for i, it in enumerate(runs):
                    fs, r = it["fs"], it["run"]
                    col_k = RUN_PALETTE[i % len(RUN_PALETTE)]
                    if fs is not None:
                        left = 100.0 * (1.0 - fs["cov"])
                        left = np.where(fs["matched"] == 1, left, np.nan)
                        axm.plot(fs["t"] - t0, left, color=col_k, lw=0.9, ls="-" if i < 4 else "--", alpha=0.95)
                axm.set_ylim(-3, 103); axm.set_yticks([0, 50, 100]); axm.set_ylabel("image left [%]", fontsize=8)
                axm.set_title("image area left after masking, per frame; gap = no mask on that frame", fontsize=7, loc="left", color=TH["ink2"], pad=2)
            else:
                axm.set_yticks([])
                axm.text(0.5, 0.5, "no mask log (filter off)" if not filtered else "", transform=axm.transAxes, ha="center", color=TH["ink2"], fontsize=8)
            if row == 1:
                axm.set_xlabel("time since ground-truth start [s]")
            else:
                plt.setp(axm.get_xticklabels(), visible=False)
    # route inset across the bottom
    axr = fig.add_subplot(gs[6, :]); ar.style(axr, TH)
    turn = gt["yaw_rate"] > TURN_DEG_S
    draw_map(axr, ds)
    axr.plot(gt["p"][:, 0], gt["p"][:, 1], color=TH["ink"], lw=1.2, label="ground truth")
    axr.scatter(gt["p"][turn, 0], gt["p"][turn, 1], s=4, color="#cc8319", label="heading change (>%g deg/s)" % TURN_DEG_S, zorder=3)
    for it in items:
        res, r = it["res"], it["run"]
        if res["onset_s"] is not None:
            gx = np.interp(res["onset_s"] + t0, gt["t"], gt["p"][:, 0]); gy = np.interp(res["onset_s"] + t0, gt["t"], gt["p"][:, 1])
            axr.plot(gx, gy, "s" if r["filtered"] else "o", ms=7, mfc=COLORS.get(base_of(r["system"]), "#999"), mec=TH["surface"], mew=0.8, alpha=0.85, zorder=5)
    for est in ("ORB-SLAM3", "VINS-Fusion"):
        axr.plot([], [], "o", color=COLORS[est], ms=7, label="onset, %s unfiltered" % est)
        axr.plot([], [], "s", color=COLORS[est], ms=7, label="onset, %s filtered" % est)
    lo, hi = gt["p"][:, :2].min(0), gt["p"][:, :2].max(0); pad = 0.12 * (hi - lo).max()
    axr.set_xlim(lo[0] - pad, hi[0] + pad); axr.set_ylim(lo[1] - pad, hi[1] + pad); axr.set_aspect("equal")
    axr.set_xlabel("x [m]"); axr.set_ylabel("y [m]")
    axr.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), facecolor=TH["surface"], edgecolor=TH["grid"])
    fig.suptitle("%s / %s: %d runs" % (testbed, short(ds), len(items)), color=TH["ink"], fontsize=12, x=0.02, ha="left", y=0.992)
    fig.savefig(path, dpi=140, facecolor=TH["surface"]); plt.close(fig)


# ---------------------------------------------------------------------- tables

def f(v, p=1):
    return "---" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{p}f}"


def tex_filtered(rows, path, testbed):
    rows = [r for r in rows if r["testbed"] == testbed and r["filtered"]]
    with open(path, "w") as fh:
        w = fh.write
        w("% Generated by script/analyse_exp3_timelines.py -- do not edit by hand.\n\\begin{table}[htbp]\n\\centering\n")
        w("\\caption{%s world, filtered runs: when the error rises, where, and how long the filter ran. ``Onset'' is the "
          "first time the start-anchored position error stays above $3\\times$ the estimator's static median for "
          "\\SI{1}{\\second} (threshold in the text); ``where'' gives the route progress at that moment and whether the "
          "robot was on a straight or in a heading change (ground-truth yaw rate above \\SI{10}{\\degree\\per\\second}). "
          "``Filter until'' is the end of the initial stretch of \\SI{5}{\\second} windows with a mask on at least "
          "\\SI{98}{\\percent} of frames; ``unmasked'' is the image area left after masking on matched frames, median "
          "(minimum). Times are seconds from the ground-truth start.}\n" % testbed.capitalize())
        w("\\label{tab:%s-timelines-filtered}\n\\scriptsize\n\\resizebox{\\textwidth}{!}{%%\n" % testbed)
        w("\\begin{tabular}{lllll r l r r r r}\n\\toprule\n")
        w("Recording & Estimator & Mask & LC & Run & Onset [\\si{\\second}] & Where & First jump [\\si{\\second}] & "
          "Filter until & Match & Unmasked [\\si{\\percent}] \\\\\n\\midrule\n")
        last = None
        for r in rows:
            first = r["dataset"] != last; last = r["dataset"]
            where = "---" if r["onset_s"] is None else "%s, %.0f\\,\\%%" % (r["segment"], 100 * r["progress"])
            w("%s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s \\\\\n" % (
                "\\repo{%s}" % short(r["dataset"]) if first else "", r["system"].replace("+YOLO", ""), r["mode"],
                r["lc"] if r["lc"] != "n/a" else "---", r["run"].replace("logging_20260916_", "rep ").replace("_", "\\_") + (" (atlas)" if r["atlas"] else ""),
                "never" if r["onset_s"] is None else f(r["onset_s"]), where,
                "---" if r["jump_s"] is None else f(r["jump_s"]), r["working_label"], f(r["match_rate"], 2),
                "---" if not np.isfinite(r["unmasked_median"]) else "%.0f (%.0f)" % (r["unmasked_median"], r["unmasked_min"])))
        w("\\bottomrule\n\\end{tabular}}\n\\end{table}\n")


def tex_baseline(rows, path, testbed):
    rows = [r for r in rows if r["testbed"] == testbed and not r["filtered"]]
    groups = {}
    for r in rows:
        groups.setdefault((r["dataset"], r["system"], r["lc"]), []).append(r)
    with open(path, "w") as fh:
        w = fh.write
        w("% Generated by script/analyse_exp3_timelines.py -- do not edit by hand.\n\\begin{table}[htbp]\n\\centering\n")
        w("\\caption{%s world, unfiltered runs: onset of sustained error and its place on the route, per run, under the "
          "same definitions as \\cref{tab:%s-timelines-filtered}. One entry per run, in repeat order; ``never'' means "
          "the error stayed below the threshold for the whole run.}\n" % (testbed.capitalize(), testbed))
        w("\\label{tab:%s-timelines-baseline}\n\\scriptsize\n\\resizebox{\\textwidth}{!}{%%\n" % testbed)
        w("\\begin{tabular}{lll l l}\n\\toprule\nRecording & Estimator & LC & Onset [\\si{\\second}] (where) & First jump [\\si{\\second}] \\\\\n\\midrule\n")
        last = None
        for (ds, sysn, lc), rs in sorted(groups.items()):
            first = ds != last; last = ds
            on = "; ".join("never" if r["onset_s"] is None else "%.0f (%s %.0f\\,\\%%)" % (r["onset_s"], "turn" if r["segment"] == "heading change" else "straight", 100 * r["progress"]) for r in rs)
            jp = "; ".join("---" if r["jump_s"] is None else "%.0f" % r["jump_s"] for r in rs)
            w("%s & %s & %s & %s & %s \\\\\n" % ("\\repo{%s}" % short(ds) if first else "", sysn, lc if lc != "n/a" else "---", on, jp))
        w("\\bottomrule\n\\end{tabular}}\n\\end{table}\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "timelines").mkdir(exist_ok=True)
    runs = registry()
    gts = {}
    series = {}
    for r in runs:
        if r["gt"] not in gts:
            gts[r["gt"]] = gt_series(r["gt"])
        es = error_series(r, gts[r["gt"]])
        if es is None:
            print("  [skip] no overlap:", r["dir"]); continue
        series[id(r)] = es
    # thresholds from static runs
    thr = {}
    with open(OUT / "thresholds.csv", "w") as fh:
        fh.write("testbed,estimator,static_runs,static_median_m,threshold_m\n")
        for tb in ("extreme", "realistic"):
            for base in ("VINS-Fusion", "ORB-SLAM3"):
                errs = [series[id(r)]["err"] for r in runs if id(r) in series and r["testbed"] == tb and not r["filtered"]
                        and base_of(r["system"]) == base and is_static(r)]
                med = float(np.median(np.concatenate(errs))) if errs else float("nan")
                thr[(tb, base)] = THR_FACTOR * med
                fh.write("%s,%s,%d,%.4f,%.4f\n" % (tb, base, len(errs), med, THR_FACTOR * med))
                print("threshold %-9s %-12s static runs %2d  median %.3f m  -> %.3f m" % (tb, base, len(errs), med, THR_FACTOR * med))
    rows = []
    items = []
    for r in runs:
        if id(r) not in series:
            continue
        gt, es = gts[r["gt"]], series[id(r)]
        t0 = gt["t"][0]
        th = thr[(r["testbed"], base_of(r["system"]))]
        on_abs = onset(es["t"], es["err"], th)
        res = dict(onset_s=None if on_abs is None else on_abs - t0, segment=None, progress=None,
                   jump_s=None if es["jump_t"] is None else es["jump_t"] - t0, n_jumps=es["n_jumps"])
        if on_abs is not None:
            wr = np.interp(on_abs, gt["t"], gt["yaw_rate"])
            res["segment"] = "heading change" if wr > TURN_DEG_S else "straight"
            res["progress"] = float(np.interp(on_abs, gt["t"], gt["progress"]))
        fs = filter_series(r) if r["filtered"] else None
        res.update(filter_stats(fs, t0, es["t"][-1]) if r["filtered"] else
                   dict(match_rate=float("nan"), working_until=None, working_label="---", unmasked_median=float("nan"), unmasked_min=float("nan")))
        res["threshold_m"] = th; res["final_err_m"] = float(es["err"][-1]); res["max_err_m"] = float(np.nanmax(es["err"]))
        res["anchor_start_s"] = es["anchor"][0] - t0; res["anchor_end_s"] = es["anchor"][1] - t0
        res["gt_duration_s"] = float(gt["t"][-1] - t0)
        row = {k: r[k] for k in ("testbed", "dataset", "system", "run", "filtered", "mode", "lc", "atlas", "september")}
        row.update(res); rows.append(row)
        items.append(dict(run=r, es=es, fs=fs, res=res, ff=feature_series(r)))
        figure(r, gt, es, fs, th, res, OUT / "timelines" / ("%s__%s__%s__%s.png" % (r["testbed"], short(r["dataset"]), r["system"].replace("+", "p"), r["run"])))
    (OUT / "datasets").mkdir(exist_ok=True)
    for (tb, ds) in sorted({(it["run"]["testbed"], it["run"]["dataset"]) for it in items}):
        group = [it for it in items if it["run"]["testbed"] == tb and it["run"]["dataset"] == ds]
        figure_dataset(tb, ds, group, gts[group[0]["run"]["gt"]],
                       {b: thr[(tb, b)] for b in ("VINS-Fusion", "ORB-SLAM3")},
                       OUT / "datasets" / ("%s__%s.png" % (tb, short(ds))))
        figure_dataset_grid(tb, ds, group, gts[group[0]["run"]["gt"]],
                            {b: thr[(tb, b)] for b in ("VINS-Fusion", "ORB-SLAM3")},
                            OUT / "datasets" / ("%s__%s__grid.png" % (tb, short(ds))))
    rows.sort(key=lambda x: (x["testbed"], x["dataset"], x["filtered"], x["system"], x["run"]))
    with open(OUT / "timelines.csv", "w", newline="") as fh:
        keys = [k for k in rows[0] if k != "windows_ok"]
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    for tb in ("extreme", "realistic"):
        tex_filtered(rows, OUT / f"{tb}_filtered_timeline.tex", tb)
        tex_baseline(rows, OUT / f"{tb}_baseline_timeline.tex", tb)
    print("\n%-9s %-16s %-18s %-26s %7s %-18s %7s %-10s %5s %s" % ("testbed", "recording", "system", "run", "onset", "where", "jump", "filter", "match", "unmasked med(min)"))
    for x in rows:
        print("%-9s %-16s %-18s %-26s %7s %-18s %7s %-10s %5s %s" % (
            x["testbed"], short(x["dataset"]), x["system"], x["run"][:26], f(x["onset_s"]) if x["onset_s"] is not None else "never",
            "" if x["onset_s"] is None else "%s %.0f%%" % (x["segment"], 100 * x["progress"]),
            f(x["jump_s"]) if x["jump_s"] is not None else "-", x["working_label"], f(x["match_rate"], 2) if np.isfinite(x["match_rate"]) else "",
            "" if not np.isfinite(x["unmasked_median"]) else "%.0f (%.0f)" % (x["unmasked_median"], x["unmasked_min"])))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
