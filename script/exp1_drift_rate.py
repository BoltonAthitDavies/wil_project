#!/usr/bin/env python3
"""Is the real rig worse, or only slower? Drift per metre against drift per second.

    python3 script/exp1_drift_rate.py

WHY THIS EXISTS
  The simulation recordings run 103-172 m in 80-125 s; the real ones run 21.6-31.7 m
  in 97-134 s.  Per metre of route the real estimator therefore integrates 5.8 times
  more inertial time and 5.5 times more frames (script/exp1_dataset_compare.py).  Any
  comparison normalised by distance charges the real rig for that, and the "12-18x
  worse per metre" reading of the local-error analysis is partly an artefact of it.

  The same runs are therefore scored three ways:
    per metre    aligned ATE / route length, and the 1-s RPE per metre travelled
    per second   the 1-s RPE itself, which is already a per-second quantity, and
                 ATE / duration
  A defect that lives in the estimator shows up in both.  A difference that is only
  the pace of the robot shows up in the first and not the second.

WHAT IT CANNOT ESTABLISH
  The real rig's lever arm is unmeasured, so its ATE carries a 0.2-0.4 m along-track
  offset that no normalisation removes.  The RPE columns are unaffected: a constant
  offset cancels in a relative displacement.  Read the RPE rows, not the ATE rows,
  when the two disagree.

Writes output/compare/datasets/drift_rate.csv and fig_drift_rate.png.
"""
import csv
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np                                   # noqa: E402
import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "docs/report/final_report/figures")
sys.path.insert(0, os.path.join(ROOT, "script"))

OUT = os.path.join(ROOT, "output/compare/datasets")
TH = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#dedcd4", sim="#4c72b0", real="#c44e52")
SIMSRC = os.path.join(ROOT, "output/compare/simulation/exp-realistic-ab/runs_metrics.csv")
REAL_ORB = [("GPU, LC on", "lc_gpu"), ("GPU, LC off", "nolc_gpu"), ("CPU, LC on", "lc_cpu"), ("CPU, LC off", "nolc_cpu")]
REALDS = ["realv7_000", "realv7_001", "realv7_002"]
VINS_RUN = "rate1_002"


def sim_rows():
    """the twenty reported realistic-world runs. No run is excluded: every statistic below is a
    median, which the five identified divergences move by less than 0.05 m (0.441 against 0.400)."""
    rows = []
    for r in csv.DictReader(open(SIMSRC)):
        if "realistic_00" not in r["dataset"] or r.get("filtered", "").lower() in ("true", "yes", "1"):
            continue
        try:
            ate, rpe = float(r["ate_rms"]), float(r["rpe1_trans_clean"])
            route, span = float(r["path_ref"]), float(r["secs"])
        except (KeyError, ValueError, TypeError):
            continue
        if not np.isfinite(rpe):
            continue
        rows.append(dict(domain="simulation", recording=r["dataset"][-3:],
                         system="VINS-Fusion" if r["tree"] == "output_vins" else "ORB-SLAM3",
                         ate=ate, rpe=rpe, route=route, span=span))
    return rows


def real_rows():
    rows = []
    for ds in REALDS:
        for lbl, key in REAL_ORB + [("fov 70", "vins")]:
            sub = f"{ds}_{key}_p15" if key != "vins" else f"{ds}_vins_{VINS_RUN}"
            f = os.path.join(ROOT, "output/compare/real", sub, "real_gt_metrics.csv")
            if not os.path.exists(f):
                continue
            m = list(csv.DictReader(open(f)))[0]
            rows.append(dict(domain="real rig", recording=ds[-3:],
                             system="VINS-Fusion" if key == "vins" else "ORB-SLAM3",
                             ate=float(m["ate_rmse"]), rpe=float(m["rpe1_trans_rmse"]),
                             route=float(m["gt_path_m"]), span=float(m["matched_duration_s"])))
    return rows


def main():
    rows = sim_rows() + real_rows()
    for r in rows:
        r["ate_per_m"] = r["ate"] / r["route"]
        r["ate_per_s"] = r["ate"] / r["span"]
        r["rpe_per_m"] = r["rpe"] / (r["route"] / r["span"])     # 1-s RPE divided by mean speed
        r["speed"] = r["route"] / r["span"]
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "drift_rate.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    def med(dom, sysn, key):
        v = [r[key] for r in rows if r["domain"] == dom and (sysn is None or r["system"] == sysn)]
        return float(np.median(v)) if v else float("nan")

    print("%-12s %-12s %3s | %8s %8s | %9s %9s %9s" % ("domain", "system", "n", "ATE m", "RPE m", "ATE/m", "RPE/m", "ATE/s"))
    for dom in ("simulation", "real rig"):
        for sysn in ("ORB-SLAM3", "VINS-Fusion"):
            n = len([r for r in rows if r["domain"] == dom and r["system"] == sysn])
            print("%-12s %-12s %3d | %8.3f %8.3f | %9.4f %9.4f %9.4f" % (
                dom, sysn, n, med(dom, sysn, "ate"), med(dom, sysn, "rpe"),
                med(dom, sysn, "ate_per_m"), med(dom, sysn, "rpe_per_m"), med(dom, sysn, "ate_per_s")))
    print("\nreal / simulation, median over the runs of each domain:")
    for key, name in (("rpe", "1-s RPE (per second, as measured)"), ("rpe_per_m", "1-s RPE per metre travelled"),
                      ("ate_per_s", "ATE per second of run"), ("ate_per_m", "ATE per metre of route")):
        print("  %-34s %5.1fx" % (name, med("real rig", None, key) / med("simulation", None, key)))
    print("  %-34s %5.1fx" % ("(for reference) seconds per metre", (1 / med("real rig", None, "speed")) / (1 / med("simulation", None, "speed"))))

    # ---- figure
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.5)); fig.patch.set_facecolor("white")
    for ax, (key, lbl, unit) in zip(axs, [("rpe", "1-s relative pose error", "m per second of travel"),
                                          ("rpe_per_m", "the same error per metre travelled", "m per metre of route")]):
        ax.set_facecolor(TH["surface"])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(True, axis="y", color=TH["grid"], lw=0.6); ax.set_axisbelow(True)
        ax.tick_params(labelsize=8, colors=TH["ink2"])
        pos = {"simulation": 0, "real rig": 1}
        for dom, col in (("simulation", TH["sim"]), ("real rig", TH["real"])):
            for i, sysn in enumerate(("ORB-SLAM3", "VINS-Fusion")):
                v = [r[key] for r in rows if r["domain"] == dom and r["system"] == sysn]
                xx = pos[dom] + (i - 0.5) * 0.42
                ax.scatter(np.full(len(v), xx) + np.random.RandomState(0).uniform(-0.055, 0.055, len(v)), v,
                           s=26, color=col, alpha=0.55, marker="o" if i == 0 else "^", zorder=3)
                ax.plot([xx - 0.14, xx + 0.14], [np.median(v)] * 2, color=col, lw=2.4, zorder=4)
                ax.text(xx + 0.17, np.median(v), "%.3f" % np.median(v), ha="left", va="center", fontsize=8, color=col)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["simulation\n(20 runs)", "real rig\n(15 runs)"], fontsize=9)
        ax.set_xlim(-0.55, 1.75); ax.set_yscale("log"); ax.set_ylabel(unit, fontsize=8.5)
        r = med("real rig", None, key) / med("simulation", None, key)
        ax.set_title("%s\n    real rig is %.1fx the simulation" % (lbl, r), loc="left", fontsize=9.5, color=TH["ink"])
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=TH["ink2"], marker=m, ls="", label=s) for s, m in (("ORB-SLAM3", "o"), ("VINS-Fusion", "^"))],
               loc="lower center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Experiment 1: the real rig's drift, charged by time and by distance", fontsize=11.5, x=0.02, ha="left", color=TH["ink"])
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    for d in (OUT, FIGS):
        fig.savefig(os.path.join(d, "exp1_drift_rate.png" if d == FIGS else "fig_drift_rate.png"), dpi=165); plt.close(fig)
    print("\nwrote", os.path.join(OUT, "fig_drift_rate.png"))


if __name__ == "__main__":
    main()
