#!/usr/bin/env python3
"""The realv7_002 re-runs with fork patch 15, scored without the frame repair.

    python3 script/realv7_patch15.py

WHAT THIS IS
  The twelve realv7 runs of 3 November 2026 were made with stock ORB-SLAM3
  initialisation behaviour: the IMU initialised at 3-4 s, upstream then reset
  the map for "insufficient motion" because the robot was still at rest, and
  the re-initialisation was starved behind a full keyframe queue.  The chapter
  scores those runs only after a frame repair applied in the scorer.

  Fork patch 15 adds two switches that turn both upstream behaviours off.  On
  11 November 2026 the four realv7_002 configurations were re-run with them on.
  This script reports those runs scored RAW -- no --join-imu-init, no
  --level-plane -- which is the point: if the switches work, no repair is
  needed.

PROVENANCE WARNING
  The re-runs overwrote output/output_orb/real/dataset_realv7_002/*/ in place,
  so the 3 November estimates for THIS RECORDING no longer exist and its "as
  written in the live log" table and figure cannot be regenerated.  The frozen
  copies tilt_table_prepatch15.{tex,png} are the only record.  realv7_000 and
  realv7_001 are untouched.

WHAT IS AND IS NOT ATTRIBUTABLE
  The map-discard rule tests keyframe spacing under 2 cm within 10 s and is
  independent of IMU noise, so the loss of the tilt is patch 15's doing.  The
  ATE improvement is NOT cleanly attributable: IMU.NoiseGyro/NoiseAcc were also
  inflated tenfold between the two sets of runs, and run_metadata.csv records
  the config PATH but not its contents, so no artefact says which values the
  3 November runs used.  A control run with the switches off at today's noise
  would separate them; it has not been made.

Outputs, under output/compare/real/realv7-insights/:
  patch15_table.tex
"""

import csv
import glob
import json
import math
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                # noqa: E402
import vio_metrics as vm                          # noqa: E402

ROOT = Path(vm.ROOT)
R = ROOT / "output" / "compare" / "real"
OUT = R / "realv7-insights"
DSS = ("dataset_realv6_000", "dataset_realv7_000", "dataset_realv7_001", "dataset_realv7_002")
# The "before" scoring differs by recording: realv7 was scored after the frame
# repair (experiment-id *_lvl), realv6 never needed it and was scored directly
# against the cleaned tracker export (*_1103pm_*_clean).
BEFORE = {"dataset_realv6_000": "realv6_1103pm_%s_clean",
          "dataset_realv7_000": "realv7_000_%s_lvl",
          "dataset_realv7_001": "realv7_001_%s_lvl",
          "dataset_realv7_002": "realv7_002_%s_lvl"}
# key -> (label, run directory).  Front-end identity is confirmed by the median
# tracking latency, not by the directory name: GPU 43-50 ms, CPU 111-113 ms.
CONF = {"lc_gpu": ("GPU, LC on", "loopclosing_rate1"),
        "nolc_gpu": ("GPU, LC off", "noloopclosing_rate1"),
        "lc_cpu": ("CPU, LC on", "loopclosing_rate1_cpu"),
        "nolc_cpu": ("CPU, LC off", "noloopclosing_rate1_cpu")}


def summary(d):
    p = d / "run_summary.csv"
    return dict(x.split(",", 1) for x in p.read_text().splitlines() if "," in x) if p.exists() else {}


def tilt(vio):
    a = np.loadtxt(vio, delimiter=",")
    p = a[:, 1:4]
    c = p - p.mean(0)
    n = np.linalg.svd(c, full_matrices=False)[2][2]
    return math.degrees(math.acos(min(1.0, abs(n[2])))), float(np.ptp(p[:, 2]))


def metrics_dir(name, ds):
    fs = glob.glob(str(R / name / ds / "metrics.json"))
    if not fs:
        return {}
    m = json.load(open(fs[0]))
    e = m if isinstance(m, list) else list(m.values())
    d = [x for x in e if isinstance(x, dict) and "ORB" in str(x.get("system", ""))]
    return d[0] if d else {}


COL = {"lc_gpu": "#2a78d6", "nolc_gpu": "#a347b2", "lc_cpu": "#0097a7", "nolc_cpu": "#e040fb"}


def paths_figure(rows):
    """The twelve patch-15 estimates against the tracker truth, same layout as the
    repaired figure of the stock runs so the two can be set side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(DSS), 4, figsize=(14, 3.5 * len(DSS)))
    for i, ds in enumerate(DSS):
        for j, key in enumerate(CONF):
            ax = axes[i, j]
            r = next((x for x in rows if x["ds"] == ds and x["key"] == key), None)
            d = R / ("%s_%s_p15" % (ds[8:], key)) / ds / "aligned_orb.csv"
            if r is None or not d.exists():
                ax.axis("off"); continue
            a = np.loadtxt(d, delimiter=",")
            ax.plot(a[:, 1], a[:, 2], color="#2a2a2a", lw=1.8, label="tracker truth")
            ax.plot(a[:, 3], a[:, 4], color=COL[key], lw=1.1, label="estimate, SE(2) aligned")
            ax.plot(a[0, 3], a[0, 4], "o", color=COL[key], ms=5)
            ax.plot(a[-1, 3], a[-1, 4], "s", color=COL[key], ms=5)
            ax.set_title("%s: ATE %.2f m, tilt %.1f$^\\circ$, %s"
                         % (r["label"], r["ate_new"], r["tilt_deg"], r["status"]),
                         fontsize=8.5, loc="left")
            ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
            if i == len(DSS) - 1:
                ax.set_xlabel("x [m]")
            if j == 0:
                # the recording labels the row, which keeps the per-panel titles
                # short enough not to overrun an equal-aspect panel
                ax.set_ylabel("%s\ny [m]" % ds[8:], fontsize=9)
    axes[0, 0].legend(fontsize=7, loc="lower right")
    fig.suptitle("The twelve realv7 runs re-made with fork patch 15, scored raw: no frame repair applied "
                 "(circle = start, square = end)", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "fig_paths_patch15_light.png", dpi=150)
    plt.close(fig)
    print("  wrote", OUT / "fig_paths_patch15_light.png")


def main():
    rows = []
    for ds in DSS:
      for key, (label, run) in CONF.items():
        d = ROOT / "output" / "output_orb" / "real" / ds / run
        if not (d / "vio.csv").exists():
            print("  missing:", d); continue
        s = summary(d)
        ang, zr = tilt(d / "vio.csv")
        new = metrics_dir("%s_%s_p15" % (ds[8:], key), ds)
        old = metrics_dir(BEFORE[ds] % key, ds)
        rec, pro = int(s.get("stereo_pairs_received", 0) or 0), int(s.get("stereo_pairs_processed", 0) or 0)
        rows.append(dict(
            ds=ds, key=key, label=label, run=run, status=(new.get("status") or "").split(" (")[0],
            imu_init_s=float(s.get("imu_initialization_time_s", "nan") or "nan"),
            ba2=s.get("inertial_ba2"), resets=int(s.get("active_map_reset_requests", 0) or 0),
            reset_reason=s.get("last_reset_reason", "none"),
            loss=int(s.get("tracking_loss_events", 0) or 0),
            tilt_deg=ang, z_range_m=zr, kept=(pro / rec if rec else float("nan")),
            ate_new=new.get("ate_rmse"), max_new=new.get("ate_max"),
            path_new=new.get("gt_path_m"), ate_old=old.get("ate_rmse")))
        r = rows[-1]
        print("  %-4s %-12s imu %4.1f s  resets %d  tilt %4.1f deg  z %4.2f m  "
              "ATE %.3f (repaired stock %.3f)  kept %.2f  %s" % (
                  ds[-3:], label, r["imu_init_s"], r["resets"],
                  r["tilt_deg"], r["z_range_m"], r["ate_new"] or float("nan"),
                  r["ate_old"] or float("nan"), r["kept"], r["status"]))
    if not rows:
        return 1

    L = ["% Generated by script/realv7_patch15.py -- do not edit by hand.",
         "\\begin{table}[htbp]", "\\centering",
         "\\caption[Every real-robot run, with the upstream initialisation defect patched out.]{"
         "Every real-robot ORB-SLAM3 run on the four recordings that carry an independent planar "
         "reference, with fork patch 15 active "
         "(\\repo{LocalMapping.resetOnLowMotion} 0, \\repo{LocalMapping.initWithBacklog} 1) and scored "
         "\\emph{raw}: neither \\texttt{--join-imu-init} nor \\texttt{--level-plane}, the repairs every other "
         "\\repo{realv7} figure in this chapter depends on. ``Tilt'' and ``z range'' are measured the same way as "
         "in \\cref{tab:realv7-tilt}, so the two are comparable. ``ATE before'' is the same configuration on the "
         "same recording on 3 November, scored raw. Front-end identity was confirmed from the median tracking "
         "latency (GPU \\SIrange{41}{50}{\\milli\\second}, CPU \\SIrange{104}{113}{\\milli\\second}), not from the "
         "run name. ``Stock + repair'' is the same configuration on the same recording on 3 November, under "
         "upstream initialisation and, for \\repo{realv7}, only after the scorer-side frame repair that "
         "has since been retired: "
         "it is the fair comparison, each estimator at the best obtainable from it. For \\repo{realv6_000}, which "
         "never showed the tilt and needed no repair, ``before'' is the same run scored against the same cleaned "
         "tracker export without the switches. Status is the scorer's: it rejects a run whose estimate spans more "
         "than \\SI{1}{\\metre} vertically, and it rejects every \\repo{realv6} run because that recording's "
         "tracker export is \\SI{1.7}{\\hertz} and fails the validity rules written for the \\SI{7}{\\hertz} one, "
         "so its rows are diagnostics. Lever arm still unmeasured, so every value is provisional.}",
         "\\label{tab:realv7-patch15}", "\\small", "\\resizebox{\\textwidth}{!}{%",
         "\\begin{tabular}{llrrr rr rrl}", "\\toprule",
         "Recording & Run & IMU init [\\si{\\second}] & resets & tilt [\\si{\\degree}] & z range [\\si{\\metre}] & "
         "Kept & ATE [\\si{\\metre}] & Before [\\si{\\metre}] & status \\\\", "\\midrule"]
    last = None
    for r in rows:
        first = r["ds"] != last
        if first and last is not None:
            L.append("\\addlinespace")
        last = r["ds"]
        L.append("%s & %s & %.1f & %d & %.1f & %.2f & %.2f & %.3f & %.3f & %s \\\\" % (
            "\\repo{%s}" % r["ds"].replace("dataset_", "") if first else "",
            r["label"], r["imu_init_s"], r["resets"], r["tilt_deg"], r["z_range_m"], r["kept"],
            r["ate_new"], r["ate_old"], r["status"]))
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    (OUT / "patch15_table.tex").write_text("\n".join(L) + "\n")

    paths_figure(rows)

    with open(OUT / "patch15_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\n  wrote", OUT / "patch15_table.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
