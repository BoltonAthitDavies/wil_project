#!/usr/bin/env python3
"""Experiment 3 tables under the reporting rules fixed on 21 October 2026.

    python3 script/analyse_exp3_rules.py

THE RULES (user decision, 21 Oct 2026; chapter 4.3.1 of the final report)
  Extreme world (dataset_{static,dynamic}_nofloortexture_*):
    accuracy, ORB-SLAM3 / VINS-Fusion   best of the three logging_20260916_* repeats
                                         per recording, "best" = lowest aligned ATE;
                                         the best run supplies the whole row, and the
                                         median and worst ATE of the three are kept
                                         beside it so the selection is visible
    accuracy, +YOLO arms                 every run that is NOT a logging_20260916_*
                                         directory (the October class/motion runs),
                                         each on its own
    latency / CPU / memory, all arms     mean +- SD over the three logging_20260916_*
                                         repeats, per recording, plus a pooled row
                                         per scene
  Realistic world (dataset_allsensor_realistic_*):
    every run on its own: estimator x filter arm, accuracy and resources, no pooling.

  The two logging_20260916_01 directories of dynamic_00_000 that hold 15 October
  re-runs (VINS-Fusion, VINS-Fusion+YOLO) are used as they stand: the user chose to
  re-analyse the campaign from the raw tree and drop the September-only record.

WHERE THE NUMBERS COME FROM -- nothing is re-derived here
  accuracy        analyse_relogged_runs.trajectory_group (campaign code; the same
                  umeyama / jump / rpe calls), read back from its trajectory_metrics.csv
                  for the September repeats and called directly for the October runs
  resources       analyse_relogged_runs.summarize_run and module_values: throughput,
                  processed fraction, CPU as process-ms over wall-ms, peak RSS, and the
                  per-run MEDIAN module latency; the mean +- SD is over those per-run
                  medians
  realistic       analyse_realistic_runs.score for accuracy (vio_metrics), the same
                  summarize_run / module_values for resources

Outputs, under output/compare/simulation/exp3-rules/:
  extreme_accuracy_best.{csv,tex}   T1  best-of-three baselines
  extreme_filtered_october.{csv,tex} T2  October filtered runs, one row each
  extreme_resources.{csv,tex}       T3  mean +- SD of the September repeats
  realistic_resources.{csv,tex}     T4  realistic world, one row per run
"""
import csv
import os
import re
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyse_relogged_runs as rl      # noqa: E402  campaign code: accuracy + resources
import analyse_realistic_runs as ar     # noqa: E402  realistic-world registry + scoring
import vio_metrics as vm                # noqa: E402

ROOT = Path(vm.ROOT)
OUT = ROOT / "output" / "compare" / "simulation" / "exp3-rules"
SEPT = re.compile(r"logging_20260916_\d+")
BASE_SYS = ["VINS-Fusion", "ORB-SLAM3"]
ALL_SYS = ["VINS-Fusion", "ORB-SLAM3", "VINS-Fusion+YOLO", "ORB-SLAM3+YOLO"]
FRONT = {"VINS-Fusion": "feature_tracking_ms", "ORB-SLAM3": "tracking_total_ms"}
BACK = {"VINS-Fusion": "backend_total_ms", "ORB-SLAM3": "local_mapping_total_ms"}
def base_of(system): return system.replace("+YOLO", "")
def scene_of(ds): return "static" if "static_" in ds else "dynamic"
def short(ds): return ds.replace("dataset_", "").replace("_nofloortexture", "").replace("allsensor_", "")


def rows_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def f(v, p=3):
    return "---" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{p}f}"


def pm(vals, p=1):
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return "---"
    if len(vals) == 1:
        return f"{vals[0]:.{p}f}"
    return f"{statistics.mean(vals):.{p}f} $\\pm$ {statistics.stdev(vals):.{p}f}"


def resources_for(run):
    """summarize_run + per-run median module latency, as the campaign defines them."""
    s = rl.summarize_run(run)
    med = {m: float(np.median(v)) for m, v in rl.module_values(run)}
    sysb = base_of(run["system"])
    return dict(front_ms=med.get(FRONT[sysb], float("nan")), back_ms=med.get(BACK[sysb], float("nan")),
                throughput_hz=s["throughput_hz"], kept_pct=s["processed_fraction_pct"],
                cpu_pct=s["cpu_avg_pct_one_core"], rss_peak_mb=s["memory_peak_mb"],
                processed=s["processed_frames"], received=s["input_frames"])


# ----------------------------------------------------------------- extreme world

def extreme_best():
    tm = rows_csv(rl.OUT / "trajectory_metrics.csv")
    out = []
    for ds in sorted({r["dataset"] for r in tm}):
        for sysn in BASE_SYS:
            reps = [r for r in tm if r["dataset"] == ds and r["system"] == sysn and SEPT.fullmatch(r["repeat"])]
            if not reps:
                continue
            ates = sorted(fnum(r["trans_rmse_m"]) for r in reps)
            best = min(reps, key=lambda r: fnum(r["trans_rmse_m"]))
            out.append(dict(dataset=ds, scene=scene_of(ds), system=sysn, n=len(reps), best_repeat=best["repeat"],
                            ate_best=ates[0], ate_median=float(np.median(ates)), ate_worst=ates[-1],
                            rot_rmse=fnum(best["rot_rmse_deg"]), rpe1_clean=fnum(best["rpe1_trans_rmse_clean_m"]),
                            rpe1_rot_median=fnum(best["rpe1_rot_median_deg"]), jumps=int(fnum(best["jumps"])),
                            path_m=fnum(best["path_m"]), overlap_s=fnum(best["overlap_s"])))
    return out


def extreme_october():
    out = []
    for tree, sysn in (("output_vins_yolo", "VINS-Fusion+YOLO"), ("output_orb_yolo", "ORB-SLAM3+YOLO")):
        for dsdir in sorted((ROOT / "output" / tree / "simulation").glob("dataset_*_nofloortexture_*")):
            for d in sorted(dsdir.iterdir()):
                if not (d / "vio.csv").exists() or SEPT.fullmatch(d.name):
                    continue
                run = dict(path=d, tree=tree, system=sysn, dataset=dsdir.name, repeat=d.name)
                meta, summ = rl.read_kv(d / "run_metadata.csv"), rl.read_kv(d / "run_summary.csv")
                fsum = rl.read_kv(d / "filter_summary.csv")
                g = rl.trajectory_group([run], dsdir.name, d.name, {(sysn, dsdir.name, d.name)})
                acc = g[2][0] if g and g[2] else {}
                res = resources_for(run)
                m = re.match(r"(class|motion)", d.name)
                out.append(dict(dataset=dsdir.name, scene=scene_of(dsdir.name), system=sysn, run=d.name,
                                mode=meta.get("filter_mode") or (m.group(1) if m else "?"),
                                lc_state=ar.loop_closer_state(str(d)) if sysn.startswith("ORB") else "n/a",
                                match_rate=fnum(summ.get("detection_match_rate", fsum.get("detection_match_rate"))),
                                filter_status=summ.get("filter_status", fsum.get("filter_status", "")),
                                ate=acc.get("trans_rmse_m"), rot_rmse=acc.get("rot_rmse_deg"),
                                rpe1_clean=acc.get("rpe1_trans_rmse_clean_m"), jumps=acc.get("jumps"),
                                started=meta.get("start_wall_time_unix_ns"), **res))
    return out


def extreme_resources():
    rm = rows_csv(rl.OUT / "run_metrics.csv")
    ml = rows_csv(rl.OUT / "module_latency_summary.csv")
    med = {(r["system"], r["dataset"], r["repeat"], r["module"]): fnum(r["median_ms"]) for r in ml}
    per_run = []
    for r in rm:
        if not SEPT.fullmatch(r["repeat"]):
            continue
        sysb = base_of(r["system"])
        per_run.append(dict(dataset=r["dataset"], scene=scene_of(r["dataset"]), system=r["system"], repeat=r["repeat"],
                            front_ms=med.get((r["system"], r["dataset"], r["repeat"], FRONT[sysb]), float("nan")),
                            back_ms=med.get((r["system"], r["dataset"], r["repeat"], BACK[sysb]), float("nan")),
                            throughput_hz=fnum(r["throughput_hz"]), kept_pct=fnum(r["processed_fraction_pct"]),
                            cpu_pct=fnum(r["cpu_avg_pct_one_core"]), rss_peak_mb=fnum(r["memory_peak_mb"]),
                            treatment=r["treatment_status"]))
    return per_run


# ---------------------------------------------------------------- realistic world

def realistic_rows():
    out = []
    for run in ar.discover():
        r, _ = ar.score(run)
        sysn = run["estimator"] + ("+YOLO" if run["filtered"] else "")
        res = resources_for(dict(path=Path(run["dir"]), tree=run["tree"], system=sysn,
                                 dataset=run["dataset"], repeat=run["run"]))
        out.append(dict(dataset=run["dataset"], layout=ar.SCENE[run["dataset"]][0], scene=ar.SCENE[run["dataset"]][1],
                        system=sysn, run=run["run"], mode=run["mode"], lc_state=r.get("lc_state", "n/a"),
                        atlas=run["atlas"], match_rate=r.get("match_rate"), filter_status=r.get("filter_status", ""),
                        ate=r.get("ate_rms"), rot_rmse=r.get("rot_rms"), rpe1_clean=r.get("rpe1_trans_clean"),
                        jumps=r.get("jumps"), of_bag=r.get("of_bag"), **res))
    return out


# ------------------------------------------------------------------------ tables

def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)


def tex_best(rows, path):
    with open(path, "w") as fh:
        w = fh.write
        w("% Generated by script/analyse_exp3_rules.py -- do not edit by hand.\n\\begin{table}[htbp]\n\\centering\n")
        w("\\caption{Extreme world, unfiltered arms: the best of the three September repeats per recording, "
          "``best'' being the lowest aligned ATE. Rotation, jump-free RPE and jump count are those of the selected "
          "run; the median and worst ATE of the three are kept beside the best so the selection is visible. "
          "Jump-free RPE excludes pairs spanning a step above \\SI{10}{\\metre\\per\\second}; the rotational figure "
          "is the median over pairs, because a single orientation step inflates the RMSE "
          "(\\cref{sec:realistic-ab}).}\n\\label{tab:extreme-accuracy-best}\n\\small\n\\resizebox{\\textwidth}{!}{%\n")
        w("\\begin{tabular}{lll rrr rrrr}\n\\toprule\n")
        w("Recording & Estimator & Best run & \\multicolumn{3}{c}{Aligned ATE [\\si{\\metre}]} & Rot.\\ RMSE & "
          "RPE$_{1\\,\\mathrm{s}}$ & RPE$_{1\\,\\mathrm{s}}$ rot. & Jumps \\\\\n")
        w("\\cmidrule(lr){4-6}\n & & & best & median & worst & [\\si{\\degree}] & [\\si{\\metre}] & med.\\ [\\si{\\degree}] & \\\\\n\\midrule\n")
        last_scene, last_ds = None, None
        for r in rows:
            if r["scene"] != last_scene and last_scene is not None:
                w("\\midrule\n")
            last_scene = r["scene"]
            first = r["dataset"] != last_ds; last_ds = r["dataset"]
            w("%s & %s & %s & %s & %s & %s & %s & %s & %s & %s \\\\\n" % (
                "\\repo{%s}" % short(r["dataset"]) if first else "", r["system"], r["best_repeat"][-2:],
                f(r["ate_best"]), f(r["ate_median"]), f(r["ate_worst"]), f(r["rot_rmse"], 2),
                f(r["rpe1_clean"]), f(r["rpe1_rot_median"], 2), r["jumps"]))
        w("\\bottomrule\n\\end{tabular}}\n\\end{table}\n")


def tex_october(rows, path):
    with open(path, "w") as fh:
        w = fh.write
        w("% Generated by script/analyse_exp3_rules.py -- do not edit by hand.\n\\begin{table}[htbp]\n\\centering\n")
        w("\\caption{Extreme world, filtered arms: every run outside the September repeat set, one row each "
          "(October 2026, replay rate 1.0, live detector). ``Match'' is \\repo{detection_match_rate}; a run below "
          "0.98 did not filter on every frame. ``Kept'' is the processed fraction of received frames. Loop-closer "
          "state is read from the log. These runs are reported on their own, not as pairs: their baselines are the "
          "September repeats of \\cref{tab:extreme-accuracy-best}, a different session and node build.}\n"
          "\\label{tab:extreme-filtered}\n\\small\n\\resizebox{\\textwidth}{!}{%\n")
        w("\\begin{tabular}{llll rr rrrr}\n\\toprule\n")
        w("Recording & Estimator & Mask & LC & Match & Kept & ATE [\\si{\\metre}] & Rot.\\ [\\si{\\degree}] & "
          "RPE$_{1\\,\\mathrm{s}}$ [\\si{\\metre}] & Jumps \\\\\n\\midrule\n")
        last = None
        for r in sorted(rows, key=lambda r: (r["dataset"], r["system"], r["mode"], r["lc_state"])):
            first = r["dataset"] != last; last = r["dataset"]
            w("%s & %s & %s & %s & %s & %s & %s & %s & %s & %s \\\\\n" % (
                "\\repo{%s}" % short(r["dataset"]) if first else "", r["system"], r["mode"],
                r["lc_state"] if r["lc_state"] != "n/a" else "---", f(r["match_rate"], 3), f(r["kept_pct"], 0) + "\\,\\%",
                f(r["ate"]), f(r["rot_rmse"], 2), f(r["rpe1_clean"]), r["jumps"] if r["jumps"] is not None else "---"))
        w("\\bottomrule\n\\end{tabular}}\n\\end{table}\n")


def tex_resources(per_run, path, caption, label, group_keys=("dataset", "system"), scene_rows=True, realistic=False):
    with open(path, "w") as fh:
        w = fh.write
        w("% Generated by script/analyse_exp3_rules.py -- do not edit by hand.\n\\begin{table}[htbp]\n\\centering\n")
        w("\\caption{%s}\n\\label{%s}\n\\scriptsize\n\\resizebox{\\textwidth}{!}{%%\n" % (caption, label))
        if realistic:
            w("\\begin{tabular}{llllc rr rrrr}\n\\toprule\n")
            w("Recording & Estimator & Mask & LC & Atlas & Front-end [\\si{\\milli\\second}] & Back-end [\\si{\\milli\\second}] & "
              "Throughput [\\si{\\hertz}] & Kept & CPU [\\si{\\percent}] & Peak RSS [\\si{\\mebi\\byte}] \\\\\n\\midrule\n")
            last = None
            for r in per_run:
                first = r["dataset"] != last; last = r["dataset"]
                w("%s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s \\\\\n" % (
                    "\\repo{%s}" % short(r["dataset"]) if first else "", r["system"], r["mode"],
                    r["lc_state"] if r["lc_state"] != "n/a" else "---", "yes" if r["atlas"] else "no",
                    f(r["front_ms"], 1), f(r["back_ms"], 1), f(r["throughput_hz"], 1),
                    f(100 * r["of_bag"], 0) + "\\,\\%" if r.get("of_bag") is not None else "---",
                    f(r["cpu_pct"], 0), f(r["rss_peak_mb"], 0)))
        else:
            w("\\begin{tabular}{ll r rr rrrr}\n\\toprule\n")
            w("Recording & Estimator & $n$ & Front-end [\\si{\\milli\\second}] & Back-end [\\si{\\milli\\second}] & "
              "Throughput [\\si{\\hertz}] & Kept [\\si{\\percent}] & CPU [\\si{\\percent}] & Peak RSS [\\si{\\mebi\\byte}] \\\\\n\\midrule\n")
            groups = {}
            for r in per_run:
                groups.setdefault((r["scene"], r["dataset"], r["system"]), []).append(r)
            def line(label_ds, sysn, rs, first):
                w("%s & %s & %d & %s & %s & %s & %s & %s & %s \\\\\n" % (
                    label_ds if first else "", sysn, len(rs),
                    pm([x["front_ms"] for x in rs]), pm([x["back_ms"] for x in rs]), pm([x["throughput_hz"] for x in rs]),
                    pm([x["kept_pct"] for x in rs], 0), pm([x["cpu_pct"] for x in rs], 0), pm([x["rss_peak_mb"] for x in rs], 0)))
            for scene in ("static", "dynamic"):
                dss = sorted({k[1] for k in groups if k[0] == scene})
                for ds in dss:
                    first = True
                    for sysn in ALL_SYS:
                        rs = groups.get((scene, ds, sysn))
                        if rs:
                            line("\\repo{%s}" % short(ds), sysn, rs, first); first = False
                if scene_rows:
                    w("\\addlinespace\n")
                    first = True
                    for sysn in ALL_SYS:
                        rs = [x for k, v in groups.items() if k[0] == scene and k[2] == sysn for x in v]
                        if rs:
                            line("\\emph{all %s}" % scene, sysn, rs, first); first = False
                if scene == "static":
                    w("\\midrule\n")
        w("\\bottomrule\n\\end{tabular}}\n\\end{table}\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    best = extreme_best(); octo = extreme_october(); res = extreme_resources(); real = realistic_rows()
    write_csv(OUT / "extreme_accuracy_best.csv", best); tex_best(best, OUT / "extreme_accuracy_best.tex")
    write_csv(OUT / "extreme_filtered_october.csv", octo); tex_october(octo, OUT / "extreme_filtered_october.tex")
    write_csv(OUT / "extreme_resources.csv", res)
    tex_resources(res, OUT / "extreme_resources.tex",
                  "Extreme world, real-time and resource measurements: mean $\\pm$ SD over the three September repeats "
                  "per recording ($n$), with a pooled row per scene. Front-end and back-end are each run's median "
                  "module latency (ORB-SLAM3: tracking and local mapping; VINS-Fusion: feature tracking and back-end "
                  "optimisation); throughput is processed frames per second of sensor time; ``Kept'' the processed "
                  "fraction of received frames; CPU is process time over wall time, in percent of one core; RSS the "
                  "peak resident set. Replay rate $0.5\\times$. The VINS-Fusion+YOLO rows are the September runs, in "
                  "which the filter did not activate except on \\repo{dynamic_00_000} repeat~1 (re-run 15 October).",
                  "tab:extreme-resources")
    real.sort(key=lambda r: (r["dataset"], r["system"].endswith("+YOLO"), r["system"], r["lc_state"], r["mode"], r["atlas"]))
    write_csv(OUT / "realistic_resources.csv", real)
    tex_resources(real, OUT / "realistic_resources.tex",
                  "Realistic world, real-time and resource measurements, one row per run (replay rate 1.0, live "
                  "detector). Columns as in \\cref{tab:extreme-resources}; ``Kept'' here is the processed fraction of "
                  "the bag's frames, so frames lost before the node's own queue count as lost. Accuracy for the same "
                  "runs is in \\cref{tab:realistic-baseline,tab:realistic-filtered}.", "tab:realistic-resources",
                  realistic=True)
    # console
    print("\n== T1 best of three (ATE best/median/worst, rot, rpe1c, jumps)")
    for r in best:
        print("  %-24s %-12s %s  %.3f / %.3f / %.3f  rot %.2f  rpe %.3f  jumps %d" % (
            short(r["dataset"]), r["system"], r["best_repeat"][-2:], r["ate_best"], r["ate_median"], r["ate_worst"],
            r["rot_rmse"], r["rpe1_clean"] or float("nan"), r["jumps"]))
    print("\n== T2 October filtered runs")
    for r in sorted(octo, key=lambda r: (r["dataset"], r["system"], r["mode"])):
        print("  %-24s %-18s %-7s %-9s match %.3f kept %3.0f%%  ATE %s rot %s rpe %s jumps %s" % (
            short(r["dataset"]), r["system"], r["mode"], r["lc_state"], r["match_rate"], r["kept_pct"],
            f(r["ate"]), f(r["rot_rmse"], 1), f(r["rpe1_clean"]), r["jumps"]))
    print("\n== T3 resources, pooled per scene (mean of per-run medians)")
    for scene in ("static", "dynamic"):
        for sysn in ALL_SYS:
            rs = [x for x in res if x["scene"] == scene and x["system"] == sysn]
            if rs:
                print("  %-8s %-18s n=%2d front %s  back %s  thr %s  kept %s  cpu %s  rss %s" % (
                    scene, sysn, len(rs), pm([x["front_ms"] for x in rs]), pm([x["back_ms"] for x in rs]),
                    pm([x["throughput_hz"] for x in rs]), pm([x["kept_pct"] for x in rs], 0),
                    pm([x["cpu_pct"] for x in rs], 0), pm([x["rss_peak_mb"] for x in rs], 0)))
    print("\n== T4 realistic resources (per run)")
    for r in real:
        print("  %-14s %-18s %-7s %-9s atlas=%-3s front %6s back %7s thr %5s kept %3.0f%% cpu %4s rss %5s" % (
            short(r["dataset"]), r["system"], r["mode"], r["lc_state"], "yes" if r["atlas"] else "no",
            f(r["front_ms"], 1), f(r["back_ms"], 1), f(r["throughput_hz"], 1), 100 * (r["of_bag"] or 0),
            f(r["cpu_pct"], 0), f(r["rss_peak_mb"], 0)))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
