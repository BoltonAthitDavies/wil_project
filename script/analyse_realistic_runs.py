#!/usr/bin/env python3
"""Score the October 2026 runs on the realistic-warehouse recordings.

    python3 script/analyse_realistic_runs.py            # all four bags, all trees

Experiment 3 in the REALISTIC worlds (chapter 4.3.1, added after the cutoff):
dataset_allsensor_realistic_000/001 are the static twins (U and I layout),
002/003 the dynamic worlds with pallet jacks on logistics jobs and walking
people. Three arms per estimator -- filter off, class mask, motion gate -- plus
ORB-SLAM3 with and without its loop closer, and an exploratory "atlas" factor
(ORB-SLAM3 loading a prior map) that is outside the A/B and is only listed.

WHAT IS COMPUTED, AND WHY HERE RATHER THAN IN analyse_relogged_runs.py
    That script discovers one experimental unit per ``logging_*`` directory. The
    October runs are named by condition instead (``loopclosing``, ``nofilter``,
    ``class_loopclosing_noatlas`` ...), so this module carries an explicit
    registry parsed from the directory names and pins every run by name. The
    numbers come from the same code: vio_metrics.analyse() for aligned ATE and
    rotation, vio_metrics.rpe() on the UNALIGNED poses for RPE, find_jumps() for
    discontinuities. Nothing is re-derived.

ADMISSIBILITY OF A FILTERED RUN (section 3.6, fixed before these numbers were read)
    A filtered run is compared with its baseline -- same estimator, same loop
    closer state, no prior atlas -- only when all three hold:
      filter active   detection_match_rate >= 0.98 (the detector mask was present
                      on essentially every frame; a lower rate means the filter
                      did not RUN, not that it did not help)
      input matched   processed frames and overlap duration within 5 % of the
                      baseline's
      same software   both runs made with the node binary built 2026-10-16 10:48
                      (+07); the ORB-SLAM3 source changed that morning, between
                      the 14-15 October filtered runs and the 21 October baselines
    Every run is scored and listed regardless; the gate decides which pairs the
    report may read as a treatment effect.

LOOP-CLOSER STATE
    Not in the manifest. Read from loop_closing.csv: since fork patch 7 a disabled
    closer logs one row per keyframe in well under a millisecond (median about
    0.4 ms, the early return), an active one spends tens of milliseconds on place
    recognition. Before the patch a disabled closer busy-spun and logged tens of
    thousands of rows with keyframe id -1; both signatures are recognised.

Outputs under output/compare/simulation/exp-realistic-ab/:
    runs_metrics.csv          one row per run, every number and gate field
    baseline_table.tex        the unfiltered arms, four bags x three estimators
    filtered_table.tex        every filtered run with its gates and verdict
    trajectories_<theme>.png  aligned overlays, one panel per bag
    metrics_<theme>.png       ATE and jump-free RPE per bag and arm
"""
import csv
import datetime as dt
import os
import re
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vio_metrics as vm  # noqa: E402
from map_underlay import draw_map  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = vm.ROOT
OUT = os.path.join(ROOT, "output", "compare", "simulation", "exp-realistic-ab")
DATASETS = ["dataset_allsensor_realistic_%03d" % i for i in range(4)]
SCENE = {"dataset_allsensor_realistic_000": ("U", "static"),
         "dataset_allsensor_realistic_001": ("I", "static"),
         "dataset_allsensor_realistic_002": ("U", "dynamic"),
         "dataset_allsensor_realistic_003": ("I", "dynamic")}
# build/orbslam3_ros2/orbslam3_node mtime; runs started before it ran older code.
BINARY_BUILT = dt.datetime(2026, 10, 16, 10, 48, tzinfo=dt.timezone(dt.timedelta(hours=7)))
MATCH_ACTIVE = 0.98
INPUT_TOL = 0.05
# /cam0/image_raw/compressed message counts from each bag's metadata.yaml (flash
# drive copies, inspected 14 and 21 October 2026). An estimator can lose frames
# before its own queue -- VINS reports every frame it RECEIVED as processed --
# so retention is stated against the bag, not against the node's own count.
BAG_FRAMES = {"dataset_allsensor_realistic_000": 3805, "dataset_allsensor_realistic_001": 2937,
              "dataset_allsensor_realistic_002": 2409, "dataset_allsensor_realistic_003": 3547}

# One hue family per estimator, as in eval_real_gt.py / analyse_proprio.py, so a
# reader never re-learns a legend -- but two of the project's exact steps fail the
# palette checks (OKLab, Machado CVD; the dataviz skill's validator, ported because
# node is absent here): ORB magenta vs the no-LC purple sat at normal-vision
# Delta E 13.2 (floor 15) and the amber was out of the lightness band at 1.7:1 on
# the light surface. The amber and the purple were snapped to the nearest passing
# step of their own hue (search in script history, 21 Oct 2026); magenta is kept.
# Secondary encoding remains: no-LC dashed, filtered arms dotted / open markers.
COLORS = {"VINS-Fusion": "#cc8319", "ORB-SLAM3": "#e040fb", "ORB-SLAM3 (no LC)": "#a347b2",
          "VINS-Fusion+YOLO": "#f57c00", "ORB-SLAM3+YOLO": "#8e24aa",
          # GPU front end (6 Nov 2026 runs): same hues, lighter, so CPU/GPU pairs read as pairs
          "ORB-SLAM3 GPU": "#f48fff", "ORB-SLAM3 GPU (no LC)": "#c98fd9", "ORB-SLAM3 GPU+YOLO": "#c060e0"}
THEMES = {"light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#dedcd4"),
          "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", grid="#3a3a37")}


# ------------------------------------------------------------------ registry

def kv(path):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            if "," in line:
                k, _, v = line.rstrip("\n").partition(",")
                out[k] = v
    return out


def parse_run(tree, name):
    """Directory name -> condition fields, or None for a run outside the design."""
    est = {"output_vins": "VINS-Fusion", "output_orb": "ORB-SLAM3",
           "output_vins_yolo": "VINS-Fusion", "output_orb_yolo": "ORB-SLAM3"}[tree]
    filtered = tree.endswith("_yolo")
    # Since the 6 November 2026 restructuring the unfiltered runs carry a
    # front-end suffix (loopclosing_gpu, noloopclosing_cpu; VINS-Fusion's single
    # run is "cpu"). The October names (loopclosing, nofilter) are kept for the
    # retired runs under trash/. `frontend` is a factor of Experiment 3, so it is
    # part of the label and never pooled over.
    if tree == "output_orb":
        m = re.fullmatch(r"(no)?loopclosing(?:_(cpu|gpu))?", name)
        if not m:
            return None
        return dict(estimator=est, filtered=False, mode="off", lc_named=m.group(1) is None, atlas=False,
                    frontend=m.group(2) or "cpu")
    if tree == "output_vins":
        return (dict(estimator=est, filtered=False, mode="off", lc_named=None, atlas=False, frontend="cpu")
                if name in ("nofilter", "cpu") else None)
    if tree == "output_vins_yolo":
        return (dict(estimator=est, filtered=True, mode=name, lc_named=None, atlas=False, frontend="cpu")
                if name in ("class", "motion") else None)
    m = re.fullmatch(r"(class|motion)_(no)?loopclosing_(no)?atlas(?:_(cpu|gpu))?", name)
    if not m:
        return None
    return dict(estimator=est, filtered=True, mode=m.group(1), lc_named=m.group(2) is None, atlas=m.group(3) is None,
                frontend=m.group(4) or "cpu")


def discover():
    runs = []
    for tree in ("output_vins", "output_orb", "output_vins_yolo", "output_orb_yolo"):
        for ds in DATASETS:
            base = os.path.join(ROOT, "output", tree, "simulation", ds)
            if not os.path.isdir(base):
                continue
            for name in sorted(os.listdir(base)):
                d = os.path.join(base, name)
                if not os.path.exists(os.path.join(d, "vio.csv")):
                    continue
                cond = parse_run(tree, name)
                if cond is None:
                    print(f"  [outside design] {tree}/{ds}/{name}")
                    continue
                runs.append(dict(dataset=ds, tree=tree, run=name, dir=d, **cond))
    return runs


def loop_closer_state(d):
    p = os.path.join(d, "loop_closing.csv")
    if not os.path.exists(p):
        return "unknown"
    rows = list(csv.DictReader(open(p)))
    if not rows:
        return "unknown"
    ids = [r.get("current_keyframe_id") for r in rows]
    if all(i in (None, "", "-1") for i in ids):
        return "disabled"                      # pre-patch-7 busy-spin signature
    durs = [float(r["duration_ms"]) for r in rows if r.get("duration_ms")]
    return "disabled" if durs and statistics.median(durs) < 1.0 else "active"


def started(meta):
    ns = int(meta.get("start_wall_time_unix_ns", "0") or 0)
    return dt.datetime.fromtimestamp(ns / 1e9, tz=dt.timezone(dt.timedelta(hours=7))) if ns else None


# ------------------------------------------------------------------ scoring

def score(run):
    ds = run["dataset"]
    label = (run["estimator"] + (" GPU" if run.get("frontend") == "gpu" else "")
             + (" (no LC)" if run["lc_named"] is False else "") + ("+YOLO" if run["filtered"] else ""))
    # The second entry never loads a run (output_vins/<ds>/vio.csv does not exist, the
    # runs are in named subdirectories) but puts output_vins on the ground-truth
    # candidate list, which is where extract_gt.py writes it. Without it an ORB or
    # YOLO run would be scored against itself and report a zero error.
    res = vm.analyse("simulation/" + ds, systems=[(label, run["tree"], run["run"]), ("VINS-Fusion", "output_vins")])
    out = dict(run)
    out["label"] = label
    meta, summ = kv(os.path.join(run["dir"], "run_metadata.csv")), kv(os.path.join(run["dir"], "run_summary.csv"))
    fsum = kv(os.path.join(run["dir"], "filter_summary.csv"))
    out.update(
        started=started(meta),
        binary="post-2026-10-16" if (started(meta) and started(meta) > BINARY_BUILT) else "pre-2026-10-16",
        replay_rate=meta.get("replay_rate"),
        received=int(summ.get("stereo_pairs_received", summ.get("images_received", 0)) or 0),
        processed=int(summ.get("stereo_pairs_processed", summ.get("images_processed", 0)) or 0),
        dropped=int(summ.get("queue_dropped_frames", 0) or 0),
        resets=int(summ.get("resets", summ.get("active_map_reset_requests", 0)) or 0),
        shutdown=summ.get("shutdown_status", ""),
        match_rate=float(summ.get("detection_match_rate", fsum.get("detection_match_rate", "nan")) or "nan"),
        filter_status=summ.get("filter_status", fsum.get("filter_status", "")),
        lc_state=loop_closer_state(run["dir"]) if run["estimator"] == "ORB-SLAM3" else "n/a",
    )
    out["retention"] = out["processed"] / out["received"] if out["received"] else float("nan")
    out["of_bag"] = out["processed"] / BAG_FRAMES[ds]
    if res is None or not res[0]:
        out["valid"] = False
        return out, None
    row, aligned = res[0][0], res[1][0]
    out.update(valid=True, ate_rms=row["rms"], ate_max=row["max"], rot_rms=row["rot_rms"],
               jumps=int(len(row["jumps"])), biggest_jump=row["biggest"], secs=row["secs"], poses=row["poses"],
               path_est=row["path"], path_ref=row["ref"])
    # RPE on the raw poses against ground truth, same call as the other chapters.
    e = vm.load(os.path.join(run["dir"], "vio.csv"))
    g = None
    for c in vm.gt_candidates("simulation/" + ds, [(label, run["tree"], run["run"])] + vm.SYSTEMS):
        g = vm.load(c)
        if g is not None:
            break
    if e is not None and g is not None:
        t, p, q = e[0], e[1], e[2]
        lo, hi = max(t[0], g[0][0]), min(t[-1], g[0][-1])
        m = (t >= lo) & (t <= hi)
        if m.sum() >= 10:
            tt, pp, qq = t[m], p[m], q[m]
            jid = vm.find_jumps(tt, pp)[0]
            r = vm.rpe(tt, pp, qq, vm.resample(g[0], g[1], tt), vm.resample_quat(g[0], g[2], tt),
                       delta_s=1.0, jump_idx=jid)
            if r:
                out.update(rpe1_trans=r["trans_rmse"], rpe1_rot=r["rot_rmse"], rpe1_pairs=r["pairs"],
                           rpe1_trans_clean=r.get("trans_rmse_clean"), rpe1_rot_clean=r.get("rot_rmse_clean"))
    return out, aligned


def baseline_for(run, rows):
    """The unfiltered run this filtered run is paired with, or None."""
    if not run["filtered"] or run["atlas"]:
        return None
    for b in rows:
        if b["dataset"] == run["dataset"] and b["estimator"] == run["estimator"] and not b["filtered"]:
            if run["estimator"] == "VINS-Fusion" or b["lc_state"] == run["lc_state"]:
                return b
    return None


def gate(run, base):
    g = dict(filter_active=run["match_rate"] >= MATCH_ACTIVE,
             input_matched=None, same_binary=None, admissible=False, baseline=None)
    if base is not None:
        dp = abs(run["processed"] - base["processed"]) / max(base["processed"], 1)  # magnitude for the gate
        dsec = abs(run.get("secs", 0) - base.get("secs", 0)) / max(base.get("secs", 1), 1e-9)
        g.update(input_matched=(dp <= INPUT_TOL and dsec <= INPUT_TOL),
                 input_delta=(run["processed"] - base["processed"]) / max(base["processed"], 1),
                 same_binary=(run["binary"] == base["binary"] == "post-2026-10-16"), baseline=base["run"])
        g["admissible"] = bool(g["filter_active"] and g["input_matched"] and g["same_binary"] and run.get("valid"))
    return g


# ------------------------------------------------------------------ outputs

def fmt(v, p=3):
    return "---" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{p}f}"


def best_of(rows, key, lower_is_better=True):
    """The best finite value of `key` over `rows`, or None.

    Used to set the rows a table prints in bold. Bold marks the best value in a
    comparison the table actually makes, not a tested difference: these cells
    hold one run each, so a bold number says "this run was lowest here", never
    "this configuration is better".
    """
    vals = [r.get(key) for r in rows]
    vals = [v for v in vals if v is not None and isinstance(v, (int, float)) and np.isfinite(v)]
    if not vals:
        return None
    return min(vals) if lower_is_better else max(vals)


def fmt_b(v, best, p=3):
    """fmt(), emboldened when v is the group's best AT THE PRINTED PRECISION.

    Comparing the rounded strings, not the floats: two runs whose RPE differs in
    the fourth decimal both print 0.078, and bolding only one of them would claim
    a distinction the table does not show. A tie at the printed precision gets
    bold on every row that ties.
    """
    t = fmt(v, p)
    if best is not None and v is not None and isinstance(v, (int, float)) and np.isfinite(v) and t == fmt(best, p):
        return "\\textbf{%s}" % t
    return t


def write_csv(rows):
    keys = ["dataset", "layout", "scene", "tree", "run", "label", "estimator", "filtered", "mode", "lc_state", "atlas",
            "started", "binary", "replay_rate", "received", "processed", "dropped", "retention", "of_bag", "resets", "shutdown",
            "match_rate", "filter_status", "valid", "poses", "secs", "ate_rms", "ate_max", "rot_rms", "jumps",
            "biggest_jump", "path_est", "path_ref", "rpe1_trans", "rpe1_trans_clean", "rpe1_rot", "rpe1_rot_clean",
            "rpe1_pairs", "baseline", "filter_active", "input_matched", "input_delta", "same_binary", "admissible"]
    with open(os.path.join(OUT, "runs_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (v.isoformat() if isinstance(v, dt.datetime) else v) for k, v in r.items()})


def write_baseline_table(rows):
    with open(os.path.join(OUT, "baseline_table.tex"), "w") as f:
        w = f.write
        w("% Generated by script/analyse_realistic_runs.py -- do not edit by hand.\n")
        w("\\begin{table}[htbp]\n\\centering\n")
        w("\\caption[Unfiltered arms on the four realistic-warehouse recordings.]{Unfiltered arms on the four realistic-warehouse recordings, 6 November 2026, replay rate 1.0. "
          "Aligned ATE is RMSE after a rigid $SE(3)$ fit with scale fixed; RPE is translational, at "
          "$\\Delta=\\SI{1}{\\second}$ over pairs that do not span a jump; ``Jumps'' counts steps above "
          "\\SI{10}{\\metre\\per\\second}; ``Kept'' is the fraction of the bag's stereo pairs the estimator processed (frames lost before the node's own queue count as lost). "
          "Loop-closer state is read from \\repo{loop_closing.csv}, not from the run name. Static and dynamic rows of "
          "one layout share a world geometry and differ in traffic only. "
          "\\textbf{Bold} marks the lowest ATE and the lowest jump-free RPE among the five arms of a recording; "
          "with one run per arm it records which run came lowest, not that the arm is better.}\n")
        w("\\label{tab:realistic-baseline}\n\\small\n\\resizebox{\\textwidth}{!}{%\n")
        w("\\begin{tabular}{llll rrrrrr}\n\\toprule\n")
        w("Recording & Layout & Scene & Estimator & ATE [\\si{\\metre}] & Rot.\\ [\\si{\\degree}] & "
          "RPE$_{1\\,\\mathrm{s}}$ [\\si{\\metre}] & Jumps & Kept & Resets \\\\\n\\midrule\n")
        unfiltered = [r for r in rows if not r["filtered"]]
        best = {ds: (best_of([r for r in unfiltered if r["dataset"] == ds], "ate_rms"),
                     best_of([r for r in unfiltered if r["dataset"] == ds], "rpe1_trans_clean"))
                for ds in {r["dataset"] for r in unfiltered}}
        last = None
        for r in rows:
            if r["filtered"]:
                continue
            lay, sc = SCENE[r["dataset"]]
            b_ate, b_rpe = best[r["dataset"]]
            first = r["dataset"] != last
            if not first:
                pass
            elif last is not None:
                w("\\addlinespace\n")
            last = r["dataset"]
            w("%s & %s & %s & %s & %s & %s & %s & %s & %s & %s \\\\\n" % (
                "\\repo{%s}" % r["dataset"].replace("dataset_allsensor_", "") if first else "",
                lay if first else "", sc if first else "", r["label"],
                fmt_b(r.get("ate_rms"), b_ate), fmt(r.get("rot_rms"), 2),
                fmt_b(r.get("rpe1_trans_clean"), b_rpe),
                r.get("jumps", "---"), fmt(r["of_bag"], 2), r["resets"]))
        w("\\bottomrule\n\\end{tabular}}\n\\end{table}\n")


def write_filtered_table(rows):
    with open(os.path.join(OUT, "filtered_table.tex"), "w") as f:
        w = f.write
        w("% Generated by script/analyse_realistic_runs.py -- do not edit by hand.\n")
        w("\\begin{table}[htbp]\n\\centering\n")
        w("\\caption{Every filtered run on the dynamic realistic recordings, with the three admissibility gates of "
          "\\cref{sec:vio-filter-method}. ``Match'' is \\repo{detection_match_rate}, the fraction of frames that "
          "carried a current detector mask; below %.2f the filter did not run. ``Input'' is the processed-frame count "
          "relative to the paired baseline. ``Binary'' says whether the run and its baseline share the node build of "
          "16 October 2026. A run passes only when all three hold; ATE and RPE are listed for every run but may be "
          "read as a treatment effect only on a passing row. Runs that loaded a prior atlas are outside the A/B and "
          "carry no baseline.}\n" % MATCH_ACTIVE)
        w("\\label{tab:realistic-filtered}\n\\scriptsize\n\\resizebox{\\textwidth}{!}{%\n")
        w("\\begin{tabular}{lllll rrll rrr l}\n\\toprule\n")
        w("Recording & Estimator & Mask & LC & Atlas & Match & Status & Input & Binary & ATE [\\si{\\metre}] & "
          "RPE$_{1\\,\\mathrm{s}}$ [\\si{\\metre}] & Jumps & Verdict \\\\\n\\midrule\n")
        last = None
        for r in rows:
            if not r["filtered"]:
                continue
            first = r["dataset"] != last
            if first and last is not None:
                w("\\addlinespace\n")
            last = r["dataset"]
            inp = "---" if r.get("input_delta") is None else "%+.0f\\,\\%%" % (100 * r["input_delta"])
            verdict = ("\\textbf{admissible}" if r["admissible"] else
                       "atlas, not paired" if r["atlas"] else
                       ", ".join(x for x, ok in (("filter inactive", r["filter_active"]),
                                                  ("input", r.get("input_matched")),
                                                  ("binary", r.get("same_binary"))) if ok is False) or "no baseline")
            w("%s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s \\\\\n" % (
                "\\repo{%s}" % r["dataset"].replace("dataset_allsensor_", "") if first else "",
                r["estimator"], r["mode"], r["lc_state"] if r["estimator"] == "ORB-SLAM3" else "---",
                "yes" if r["atlas"] else "no", fmt(r["match_rate"], 3), r["filter_status"] or "---", inp,
                "same" if r.get("same_binary") else ("older" if r["binary"] == "pre-2026-10-16" else "---"),
                fmt(r.get("ate_rms")), fmt(r.get("rpe1_trans_clean")), r.get("jumps", "---"), verdict))
        w("\\bottomrule\n\\end{tabular}}\n\\end{table}\n")


def style(ax, th):
    ax.set_facecolor(th["surface"])
    for s in ax.spines.values():
        s.set_color(th["grid"])
    ax.tick_params(colors=th["ink2"], labelsize=8)
    ax.grid(True, color=th["grid"], lw=0.6)
    ax.xaxis.label.set_color(th["ink2"]); ax.yaxis.label.set_color(th["ink2"]); ax.title.set_color(th["ink"])


def fig_trajectories(rows, aligned, theme):
    th = THEMES[theme]
    fig, axes = plt.subplots(2, 2, figsize=(11, 10.5), facecolor=th["surface"])
    for ax, ds in zip(axes.ravel(), DATASETS):
        style(ax, th)
        draw_map(ax, ds)
        g = vm.load_gt(os.path.join(ROOT, "output", "output_vins", "simulation", ds, "ground_truth.csv"))
        if g is not None:
            ax.plot(g[1][:, 0], g[1][:, 1], color=th["ink"], lw=1.3, label="ground truth", zorder=5)
        for r in rows:
            if r["dataset"] != ds or not r.get("valid") or r["atlas"]:
                continue
            if r["filtered"] and not r["admissible"]:
                continue
            a = aligned[id(r)]
            ls = ":" if r["filtered"] else ("--" if r["lc_state"] == "disabled" else "-")
            ax.plot(a["p"][:, 0], a["p"][:, 1], color=COLORS[r["label"].replace("+YOLO", "") if r["label"] not in COLORS else r["label"]],
                    lw=1.0, ls=ls, label="%s  ATE %.2f m" % (r["label"] + (" [%s]" % r["mode"] if r["filtered"] else ""), r["ate_rms"]))
        lay, sc = SCENE[ds]
        ax.set_title("%s  (%s layout, %s)" % (ds.replace("dataset_allsensor_", ""), lay, sc), fontsize=10)
        ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.legend(fontsize=7.5, loc="best", facecolor=th["surface"], edgecolor=th["grid"], labelcolor=th["ink"])
    fig.suptitle("Realistic worlds: estimates after rigid SE(3) alignment (scale fixed)", color=th["ink"], fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"trajectories_{theme}.png"), dpi=160, facecolor=th["surface"]); plt.close(fig)


def fig_metrics(rows, theme):
    th = THEMES[theme]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=th["surface"])
    # CPU arms filled, GPU arms open (same marker, same hue): the front end is the
    # one factor added on 6 Nov 2026 and the pairs must read as pairs.
    arms = [("VINS-Fusion", "o"), ("ORB-SLAM3", "s"), ("ORB-SLAM3 GPU", "s"), ("ORB-SLAM3 (no LC)", "D"), ("ORB-SLAM3 GPU (no LC)", "D")]
    x = np.arange(len(DATASETS))
    for ax, key, name in zip(axes, ("ate_rms", "rpe1_trans_clean"), ("aligned ATE RMSE [m]", "jump-free RPE at 1 s [m]")):
        style(ax, th)
        for k, (lab, mk) in enumerate(arms):
            ys = []
            for ds in DATASETS:
                r = next((r for r in rows if r["dataset"] == ds and r["label"] == lab and not r["filtered"] and r.get("valid")), None)
                ys.append(r.get(key, np.nan) if r else np.nan)
            gpu = "GPU" in lab
            ax.plot(x + (k - 2) * 0.11, ys, mk, ms=8, mfc="none" if gpu else COLORS[lab], mec=COLORS[lab] if gpu else th["surface"],
                    mew=1.8 if gpu else 1.2, label=lab, zorder=4)
        # admissible filtered arms as open markers next to their baseline
        for r in rows:
            if r["filtered"] and r["admissible"] and r.get("valid"):
                i = DATASETS.index(r["dataset"]); k = [a[0] for a in arms].index(r["label"].replace("+YOLO", ""))
                ax.plot(i + (k - 2) * 0.11 + 0.05, r.get(key, np.nan), "^", ms=9, mfc="none", mec=COLORS[r["label"]], mew=1.8,
                        label="%s, %s mask (admissible)" % (r["estimator"], r["mode"]), zorder=5)
        ax.set_xticks(x); ax.set_xticklabels(["%s\n%s %s" % (d.replace("dataset_allsensor_", ""), *SCENE[d]) for d in DATASETS], fontsize=8)
        ax.set_ylabel(name); ax.set_ylim(bottom=0)
    h, l = axes[0].get_legend_handles_labels()
    seen = {}
    for hh, ll in zip(h, l):
        seen.setdefault(ll, hh)
    fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=5, fontsize=8, frameon=False, labelcolor=th["ink"])
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(OUT, f"metrics_{theme}.png"), dpi=160, facecolor=th["surface"]); plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = discover()
    rows, aligned = [], {}
    for run in runs:
        r, a = score(run)
        rows.append(r)
        if a is not None:
            aligned[id(r)] = a
    for r in rows:
        b = baseline_for(r, rows)
        r.update(gate(r, b))
    rows.sort(key=lambda r: (r["dataset"], r["filtered"], r["estimator"], r.get("lc_state", ""), r["mode"], r["atlas"]))
    write_csv(rows); write_baseline_table(rows); write_filtered_table(rows)
    for th in THEMES:
        fig_trajectories(rows, aligned, th); fig_metrics(rows, th)
    print("\n%-28s %-22s %-6s %-8s %-6s %7s %7s %6s %6s %5s %5s %-6s %s" % (
        "dataset", "label", "mode", "LC", "atlas", "ATE", "RPE1c", "kept", "ofbag", "jumps", "match", "binary", "verdict"))
    for r in rows:
        print("%-28s %-22s %-6s %-8s %-6s %7s %7s %6.2f %6.2f %5s %5s %-6s %s" % (
            r["dataset"].replace("dataset_allsensor_", ""), r["label"], r["mode"], r["lc_state"], "yes" if r["atlas"] else "",
            fmt(r.get("ate_rms")), fmt(r.get("rpe1_trans_clean")), r["retention"], r["of_bag"], r.get("jumps", "-"),
            fmt(r["match_rate"], 2) if r["filtered"] else "", r["binary"][:3] if r["filtered"] else "",
            "ADMISSIBLE" if r["admissible"] else ("" if not r["filtered"] else "x")))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
