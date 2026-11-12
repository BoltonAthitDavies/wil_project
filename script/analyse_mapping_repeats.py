#!/usr/bin/env python3
"""Within-configuration repeatability from the eighteen mapping-bag repeats.

    python3 script/analyse_mapping_repeats.py

WHY THIS EXISTS
  Every accuracy cell of Experiment 3 is one run, so chapter 4 can say that a
  difference between two cells is "a candidate, not an effect" but cannot say
  how large a difference has to be before it is one.  The eighteen runs behind
  the latency and resource table (six configurations, three repeats each, two
  mapping recordings) are the only repeated runs in the simulation campaign.
  They were scored for milliseconds and never for accuracy.  Their trajectories
  and the two recordings' ground truth are both on disk, so the noise floor can
  be measured without recording anything.

WHAT IT IS NOT
  These are closure-free runs on the two LONG mapping routes (377 m and 400 m),
  not the four accuracy recordings (103-172 m).  The spread measured here bounds
  run-to-run noise for this configuration on these routes.  It does not license
  a difference between two cells of table 4.12 on its own; it says what size of
  difference is worth taking seriously at all.

  The three repeats of a configuration were also run back to back rather than
  interleaved (chapter 4.3.1, "Confound: run order"), so a drift across the
  three is confounded with the laptop's thermal state.  That matters less for
  accuracy than for latency -- a thermally throttled run drops frames, and the
  Kept column shows whether it did.

Outputs, under output/compare/simulation/exp3-repeats/:
  mapping_repeats.csv        one row per run
  mapping_repeats.tex        one row per configuration: mean +- SD and spread
"""

import csv
import os
import statistics
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                              # noqa: E402
import vio_metrics as vm                        # noqa: E402
import analyse_realistic_runs as ar             # noqa: E402

ROOT = Path(vm.ROOT)
OUT = ROOT / "output" / "compare" / "simulation" / "exp3-repeats"
DSS = ("dataset_allsensor_realistic_map", "dataset_allsensor_realistic_01_map")
LAYOUT = {"dataset_allsensor_realistic_map": "U", "dataset_allsensor_realistic_01_map": "I"}
# (tree, run prefix) -> label.  The loop closer is disabled in every one of these
# runs; loop_closer_state() re-reads that from loop_closing.csv rather than the name.
CONF = (("output_orb", "noloopclosing_cpu", "ORB-SLAM3"),
        ("output_orb", "noloopclosing_gpu", "ORB-SLAM3 GPU"),
        ("output_vins", "cpu", "VINS-Fusion"))


def runs_for(ds):
    out = []
    for tree, prefix, label in CONF:
        d = ROOT / "output" / tree / "simulation" / ds
        if not d.is_dir():
            continue
        for sub in sorted(p.name for p in d.iterdir() if p.is_dir() and p.name.startswith(prefix)):
            # "cpu_001" must not also match a hypothetical "cpu_gpu_..."; the
            # repeats are always <prefix>_NNN.
            tail = sub[len(prefix):]
            if not (tail.startswith("_") and tail[1:].isdigit()):
                continue
            out.append(dict(dataset=ds, tree=tree, run=sub, label=label, repeat=int(tail[1:])))
    return out


def score(r):
    res = vm.analyse("simulation/" + r["dataset"],
                     systems=[(r["label"], r["tree"], r["run"]), ("VINS-Fusion", "output_vins")])
    if res is None or not res[0]:
        return None
    row, aligned = res[0][0], res[1][0]
    d = ROOT / "output" / r["tree"] / "simulation" / r["dataset"] / r["run"]
    summ = ar.kv(str(d / "run_summary.csv"))
    rec = int(summ.get("stereo_pairs_received", summ.get("images_received", 0)) or 0)
    pro = int(summ.get("stereo_pairs_processed", summ.get("images_processed", 0)) or 0)
    out = dict(r)
    out.update(ate_rms=row["rms"], ate_max=row["max"], ate_final=row["final"],
               ate_max_post=ar.post_init_max(aligned), path_est=row["path"], path_ref=row["ref"],
               secs=row["secs"], poses=row["poses"], jumps=int(len(row["jumps"])),
               rot_rms=row["rot_rms"], received=rec, processed=pro,
               kept=(pro / rec if rec else float("nan")),
               resets=int(summ.get("resets", summ.get("active_map_reset_requests", 0)) or 0),
               lc_state=ar.loop_closer_state(str(d)) if r["tree"] == "output_orb" else "n/a")
    out["ate_pct_path"] = 100.0 * row["rms"] / row["ref"] if row["ref"] else float("nan")
    return out


def pm(vals, p=3):
    """mean +- SD; SD of three repeats, stated as such in the caption."""
    if not vals:
        return "---"
    if len(vals) == 1:
        return "%.*f" % (p, vals[0])
    return "%.*f $\\pm$ %.*f" % (p, statistics.mean(vals), p, statistics.stdev(vals))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in DSS:
        for r in runs_for(ds):
            s = score(r)
            if s is None:
                print("  no score:", ds, r["run"])
                continue
            rows.append(s)
            print("  %-34s %-14s rep %d  ATE %.3f  max* %.3f  %.2f %%  kept %.2f  jumps %d  LC %s" % (
                ds.replace("dataset_allsensor_", ""), s["label"], s["repeat"], s["ate_rms"],
                s["ate_max_post"] or float("nan"), s["ate_pct_path"], s["kept"], s["jumps"], s["lc_state"]))
    if not rows:
        print("no runs found"); return 1

    keys = ["dataset", "layout", "tree", "run", "label", "repeat", "lc_state", "ate_rms", "ate_max",
            "ate_max_post", "ate_final", "ate_pct_path", "rot_rms", "jumps", "path_est", "path_ref",
            "secs", "poses", "received", "processed", "kept", "resets"]
    for r in rows:
        r["layout"] = LAYOUT[r["dataset"]]
    with open(OUT / "mapping_repeats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ---- per-configuration summary
    groups = {}
    for r in rows:
        groups.setdefault((r["dataset"], r["label"]), []).append(r)
    L = ["% Generated by script/analyse_mapping_repeats.py -- do not edit by hand.",
         "\\begin{table}[htbp]", "\\centering",
         "\\caption[Run-to-run spread over the three repeats of each mapping-recording "
         "configuration.]{Run-to-run spread over the three repeats of each configuration on the two mapping "
         "recordings, the only repeated runs in the simulation campaign; the same eighteen runs as "
         "\\cref{tab:mapping-resources}, scored for accuracy. Every run has its loop closer disabled, read from "
         "\\repo{loop_closing.csv}. ``ATE'' is mean $\\pm$ SD over the three repeats and ``spread'' the largest "
         "difference between two of them; ``Max'' is the largest error after the initialisation transient, on the "
         "convention of the accuracy table of \\cref{sec:exp1-sim}. The three repeats of a configuration were run back to back "
         "rather than interleaved, so a drift across them is confounded with the laptop's thermal state "
         "(\\cref{sec:exp1-sim}); the Kept column is where that would show.}",
         "\\label{tab:mapping-repeats}", "\\small", "\\resizebox{\\textwidth}{!}{%",
         "\\begin{tabular}{llr rrrrr}", "\\toprule",
         "Recording & Estimator & $n$ & ATE [\\si{\\metre}] & spread [\\si{\\metre}] & "
         "Max [\\si{\\metre}] & ATE/path [\\si{\\percent}] & Kept \\\\", "\\midrule"]
    last = None
    summary = []
    for (ds, label) in sorted(groups, key=lambda k: (DSS.index(k[0]), [c[2] for c in CONF].index(k[1]))):
        g = sorted(groups[(ds, label)], key=lambda r: r["repeat"])
        ates = [r["ate_rms"] for r in g]
        spread = max(ates) - min(ates)
        first = ds != last
        if not first:
            pass
        elif last is not None:
            L.append("\\addlinespace")
        last = ds
        L.append("%s & %s & %d & %s & %.3f & %s & %s & %s \\\\" % (
            "\\repo{%s}" % ds.replace("dataset_allsensor_", "") if first else "", label, len(g),
            pm(ates), spread, pm([r["ate_max_post"] for r in g if r["ate_max_post"] is not None]),
            pm([r["ate_pct_path"] for r in g], 2), pm([r["kept"] for r in g], 2)))
        summary.append((ds, label, len(g), statistics.mean(ates), spread))
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    (OUT / "mapping_repeats.tex").write_text("\n".join(L) + "\n")

    print("\n== within-configuration spread (the noise floor this campaign can measure)")
    for ds, label, n, mean, spread in summary:
        print("  %-22s %-14s n=%d  mean %.3f m  spread %.3f m  (%.1f %% of the mean)" % (
            ds.replace("dataset_allsensor_", ""), label, n, mean, spread, 100 * spread / mean))
    allspread = [s for *_, s in summary]
    print("\n  largest spread over any configuration: %.3f m" % max(allspread))
    print("  wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
