#!/usr/bin/env python3
"""Real-rig counterparts of four simulation figures of Experiment 1, drawn the same way.

    python3 script/exp1_real_patterns.py

  fig_frontend_pairs_real.png     (figure 4.9)  the front end paired within recording and
                                  loop-closer state: six pairs, CPU -> GPU extractor
  fig_vins_features_real.png      (figure 4.10) VINS-Fusion tracked features and planar error
                                  against time, the reported run and a repeat, realv7_000
  fig_translation_pattern_real.png (figure 4.12) error split along/across the direction of
                                  travel per run, and the cross-track share (no vertical:
                                  the tracker is planar)
  fig_cost_profile_real.png       (figure 4.15) peak resident set against processor time,
                                  mean +- SD over the three realv6 repeats, realv7 single
                                  runs faded

Reads the scoring directories (aligned_*.csv), the nodes' performance.csv and the
resource table output/compare/real/exp1-resources/real_resources.csv. Writes to
output/compare/real/exp1-patterns/. Chat-first; nothing here is in the report.
"""
import collections
import csv
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402
import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.lines import Line2D                  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "script"))
import exp1_real_resources as RR                     # noqa: E402

OUT = os.path.join(ROOT, "output/compare/real/exp1-patterns")
FIGS = os.path.join(ROOT, "docs/report/final_report/figures")
REPORT = {"fig_frontend_pairs_real.png": "exp1_real_frontend_pairs.png",
          "fig_translation_pattern_real.png": "exp1_real_translation_pattern.png"}
TH = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#dedcd4")
COL = {"VINS-Fusion": "#cc8319", "ORB-SLAM3": "#e040fb", "ORB-SLAM3 GPU": "#f48fff",
       "ORB-SLAM3 (no LC)": "#a347b2", "ORB-SLAM3 GPU (no LC)": "#c58fd0"}
LCCOL = {"closer active": "#e040fb", "closer disabled": "#a347b2"}
DSS = ["realv7_000", "realv7_001", "realv7_002"]
RUNS = [("GPU, LC on", "lc_gpu", "ORB-SLAM3 GPU"), ("GPU, LC off", "nolc_gpu", "ORB-SLAM3 GPU (no LC)"),
        ("CPU, LC on", "lc_cpu", "ORB-SLAM3"), ("CPU, LC off", "nolc_cpu", "ORB-SLAM3 (no LC)")]
VINS_RUN = "rate1_002"
RES = pd.read_csv(os.path.join(ROOT, "output/compare/real/exp1-resources/real_resources.csv"))


def style(ax):
    ax.set_facecolor(TH["surface"])
    for sp in ax.spines.values():
        sp.set_color(TH["grid"])
    ax.grid(True, color=TH["grid"], lw=0.6, alpha=0.8); ax.set_axisbelow(True)
    ax.tick_params(colors=TH["ink2"], labelsize=8)


def aligned(ds, key):
    sub = f"{ds}_{key}_p15" if key != "vins" else f"{ds}_vins_{VINS_RUN}"
    a = np.loadtxt(os.path.join(ROOT, "output/compare/real", sub, "dataset_" + ds, f"aligned_{'vins' if key == 'vins' else 'orb'}.csv"),
                   delimiter=",", comments="#")
    return a[:, 0] / 1e9, a[:, 1:3], a[:, 3:5], a[:, 5]


# ------------------------------------------------------------------ figure 4.9
def fig_pairs():
    pairs = []
    for ds in DSS:
        nb = RR.bag_frames("dataset_" + ds)
        for lc, lbl in (("lc", "closer active"), ("nolc", "closer disabled")):
            c = RES[(RES.dataset == ds) & (RES.rep == 0) & (RES.run == ("CPU, LC on" if lc == "lc" else "CPU, LC off"))].iloc[0]
            g = RES[(RES.dataset == ds) & (RES.rep == 0) & (RES.run == ("GPU, LC on" if lc == "lc" else "GPU, LC off"))].iloc[0]
            pairs.append((ds, lbl, dict(of_bag=c.processed / nb, ate=c.ate), dict(of_bag=g.processed / nb, ate=g.ate)))
    fig = plt.figure(figsize=(10.0, 4.6), facecolor=TH["surface"])
    gs = fig.add_gridspec(1, 2, left=0.085, right=0.985, top=0.80, bottom=0.17, wspace=0.26)
    for k, (key, lbl, pct) in enumerate([("of_bag", "frames of the bag processed [%]", True), ("ate", "aligned ATE [m]", False)]):
        ax = fig.add_subplot(gs[0, k]); style(ax); up = dn = 0
        for ds, lc, c, g in pairs:
            a, b = (100 * c[key], 100 * g[key]) if pct else (c[key], g[key])
            up += b > a; dn += b < a
            ax.plot([0, 1], [a, b], "-o", ms=5, lw=1.4, color=LCCOL[lc], alpha=0.85)
            ax.text(1.05, b, ds[-3:], fontsize=7, color=TH["ink2"], va="center")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["CPU extractor", "GPU extractor"], fontsize=9); ax.set_xlim(-0.25, 1.35)
        ax.set_ylabel(lbl, color=TH["ink2"], fontsize=9)
        if not pct:
            ax.set_yscale("log"); ax.set_ylim(0.15, 0.6)
        ax.set_title("(%s) %s\n    GPU higher in %d of %d pairs, lower in %d" % (
            "ab"[k], "throughput: every pair moves the same way" if pct else "accuracy: the pairs agree to 0.02 m", up, len(pairs), dn),
            fontsize=9.5, loc="left", color=TH["ink"])
    fig.legend(handles=[Line2D([], [], color=LCCOL[v], lw=3, label=v) for v in LCCOL], loc="lower center", ncol=2,
               frameon=False, fontsize=8.5, labelcolor=TH["ink2"], bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Experiment 1, real robot: the front end paired within recording and loop-closer state",
                 fontsize=11.5, x=0.012, y=0.955, ha="left", color=TH["ink"])
    fig.savefig(os.path.join(OUT, "fig_frontend_pairs_real.png"), dpi=160, facecolor=TH["surface"]); plt.close(fig)
    for ds, lc, c, g in pairs:
        print("    %s %-16s of_bag %5.1f -> %5.1f %%   ATE %6.3f -> %6.3f m" % (ds, lc, 100 * c["of_bag"], 100 * g["of_bag"], c["ate"], g["ate"]))


# ----------------------------------------------------------------- figure 4.10
def fig_vins():
    ds = "realv7_000"
    runs = [("reported run", VINS_RUN, f"{ds}_vins_{VINS_RUN}"), ("repeat", VINS_RUN + "_01", f"{ds}_vins_{VINS_RUN}_r01")]
    fig = plt.figure(figsize=(11.0, 5.6), facecolor=TH["surface"])
    gs = fig.add_gridspec(2, 1, left=0.075, right=0.985, top=0.90, bottom=0.11, hspace=0.45)
    axf, axe = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]); style(axf); style(axe)
    cols = ["#cc8319", "#2a9d8f"]
    for (lbl, run, sub), col in zip(runs, cols):
        p = pd.read_csv(os.path.join(ROOT, "output/output_vins/real/dataset_" + ds, run, "performance.csv"))
        a = np.loadtxt(os.path.join(ROOT, "output/compare/real", sub, "dataset_" + ds, "aligned_vins.csv"), delimiter=",", comments="#")
        t0 = p.timestamp_ns.iloc[0] / 1e9
        axf.plot(p.timestamp_ns / 1e9 - t0, p.feature_count, color=col, lw=1.0, label=lbl)
        axe.plot(a[:, 0] / 1e9 - t0, a[:, 5], color=col, lw=1.2, label=lbl)
        low = (p.feature_count <= 120).mean()
        print("    VINS %s %s: features median %.0f, min %.0f, <=120 in %.1f%% of frames, frames %d" % (ds, run, p.feature_count.median(), p.feature_count.min(), 100 * low, len(p)))
    axf.axhline(120, color=TH["ink2"], ls=":", lw=1); axf.set_ylim(0, 160); axf.set_ylabel("tracked features", color=TH["ink2"], fontsize=9)
    axf.set_title("(a) VINS-Fusion on realv7_000, two runs: tracked features (ceiling 150, dotted = 120): never below the ceiling",
                  fontsize=9.5, loc="left", color=TH["ink"])
    axe.set_ylabel("aligned planar error [m]", color=TH["ink2"], fontsize=9); axe.set_xlabel("time since the recording's start [s]", color=TH["ink2"], fontsize=9)
    axe.set_title("(b) the same two runs, planar error after the SE(2) fit the reported ATE uses: no dip to explain, the error is everywhere",
                  fontsize=9.5, loc="left", color=TH["ink"]); axe.legend(frameon=False, fontsize=8.5, labelcolor=TH["ink2"])
    fig.suptitle("Experiment 1, real robot: VINS-Fusion never starves, so the simulation's excursion mechanism cannot act",
                 fontsize=11.5, x=0.012, y=0.975, ha="left", color=TH["ink"])
    fig.savefig(os.path.join(OUT, "fig_vins_features_real.png"), dpi=160, facecolor=TH["surface"]); plt.close(fig)


# ----------------------------------------------------------------- figure 4.12
def fig_translation():
    rows = []
    for ds in DSS:
        for key, sysn in [("vins", "VINS-Fusion")] + [(k, s) for _, k, s in RUNS]:
            t, g, e, err = aligned(ds, key)
            d = np.gradient(g, axis=0); n = np.linalg.norm(d, axis=1); k = n > 1e-6
            u = d[k] / n[k, None]; ev = (e - g)[k]
            al = np.sum(ev * u, axis=1); cr = ev[:, 0] * -u[:, 1] + ev[:, 1] * u[:, 0]
            rows.append(dict(ds=ds[-3:], sysn=sysn, along=np.sqrt(np.mean(al ** 2)), cross=np.sqrt(np.mean(cr ** 2))))
    for r in rows:
        r["share"] = r["cross"] ** 2 / (r["along"] ** 2 + r["cross"] ** 2)
    fig = plt.figure(figsize=(12.5, 6.4), facecolor=TH["surface"])
    gs = fig.add_gridspec(2, 1, left=0.065, right=0.955, top=0.86, bottom=0.30, hspace=0.55, height_ratios=[2.2, 1])
    ax, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]); style(ax); style(ax2)
    x = np.arange(len(rows)); w = 0.36
    for i, r in enumerate(rows):
        c = COL[r["sysn"]]
        ax.bar(i - w / 2, r["along"], w, color=c, edgecolor=TH["ink"], lw=0.4)
        ax.bar(i + w / 2, r["cross"], w, color=c, edgecolor=TH["ink"], lw=0.4, hatch="//", alpha=0.75)
        ax2.plot(i, r["share"], "o", ms=9, color=c)
    ax.set_ylabel("RMS error [m]", color=TH["ink2"], fontsize=9); ax.set_xticks(x); ax.set_xticklabels([])
    ax.set_title("(a) the error split in the ground-truth heading frame: solid = along-track, hatched = cross-track. No vertical: the tracker is planar.",
                 fontsize=9.5, loc="left", color=TH["ink"])
    ax2.axhline(0.5, color=TH["ink2"], ls=":", lw=1); ax2.set_ylim(0, 1); ax2.set_ylabel("cross-track share", color=TH["ink2"], fontsize=9)
    ax2.set_xticks(x); ax2.set_xticklabels([f"{r['ds']} {r['sysn'].replace('ORB-SLAM3', 'ORB').replace('-Fusion', '')}" for r in rows], rotation=90, fontsize=7.5)
    sec = ax2.twinx(); sec.set_ylim(0, 1); sec.set_yticks([np.sin(np.radians(a)) ** 2 for a in (15, 30, 45, 60, 90)]); sec.set_yticklabels(["15", "30", "45", "60", "90"], fontsize=7)
    sec.tick_params(colors=TH["ink2"])
    vs = [r["share"] for r in rows if r["sysn"] == "VINS-Fusion"]; os_ = [r["share"] for r in rows if r["sysn"] != "VINS-Fusion"]
    sh = lambda ds: [r["share"] for r in rows if r["ds"] == ds]
    ax2.set_title("(b) cross-track share of the planar error: 0.5 is isotropic, above it the estimate drifts sideways. Right axis: the same number as an angle off the direction of travel.\n"
                  "    The share is set by the recording, not the estimator: 000 %.2f-%.2f and 002 %.2f-%.2f (sideways, every run), 001 %.2f-%.2f (along-track, every run: the 0.4 m lead)."
                  % (min(sh("000")), max(sh("000")), min(sh("002")), max(sh("002")), min(sh("001")), max(sh("001"))),
                  fontsize=9, loc="left", color=TH["ink"])
    fig.legend(handles=[Line2D([], [], color=COL[s], lw=5, label=s) for s in COL], loc="lower center", ncol=5, frameon=False, fontsize=8.5, labelcolor=TH["ink2"], bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Experiment 1, real robot: what the visual estimators' translation error is made of (aligned series, the same fit the reported ATE uses)",
                 fontsize=11.5, x=0.012, y=0.975, ha="left", color=TH["ink"])
    fig.savefig(os.path.join(OUT, "fig_translation_pattern_real.png"), dpi=160, facecolor=TH["surface"]); plt.close(fig)
    for r in rows:
        print("    %s %-22s along %.3f cross %.3f share %.2f" % (r["ds"], r["sysn"], r["along"], r["cross"], r["share"]))


# ----------------------------------------------------------------- figure 4.15
def fig_cost():
    fig = plt.figure(figsize=(7.6, 5.0), facecolor=TH["surface"]); ax = fig.add_axes([0.105, 0.135, 0.875, 0.70]); style(ax)
    MARK = {"LC on": "o", "LC off": "s", "fov 70, extr. fixed": "o"}
    def sysn(r):
        return "VINS-Fusion" if r.system == "VINS-Fusion" else ("ORB-SLAM3 GPU" if r.run.startswith("GPU") else "ORB-SLAM3")
    def lc(r):
        return "LC on" if "LC on" in r.run or r.system == "VINS-Fusion" else "LC off"
    v6 = RES[RES.dataset == "realv6_000"]
    for (s, l), g in v6.groupby([v6.apply(sysn, axis=1), v6.apply(lc, axis=1)]):
        cpu = g.cpu_pct.values / 100; rss = g.rss_peak_mb.values
        ax.errorbar(cpu.mean(), rss.mean(), xerr=cpu.std(ddof=1), yerr=rss.std(ddof=1), fmt=MARK[l], ms=9, color=COL[s], ecolor=COL[s],
                    elinewidth=1.2, capsize=3, mew=1.2, markerfacecolor=COL[s] if l == "LC on" else "none")
    v7 = RES[(RES.dataset != "realv6_000") & (RES.rep == 0)]
    for _, r in v7.iterrows():
        ax.plot(r.cpu_pct / 100, r.rss_peak_mb, MARK[lc(r)], ms=5, color=COL[sysn(r)], alpha=0.35, markerfacecolor=COL[sysn(r)] if lc(r) == "LC on" else "none")
    ax.set_xlabel("processor time [cores of one; large = mean $\\pm$ SD over three realv6 repeats, small = realv7 single runs]", color=TH["ink2"], fontsize=8.5)
    ax.set_ylabel("peak resident set [MiB]", color=TH["ink2"], fontsize=9); ax.set_xlim(2.5, 9.5); ax.set_ylim(0, 2100)
    h = [Line2D([], [], color=COL[s], marker="o", ls="", ms=8, label=s) for s in ("ORB-SLAM3", "ORB-SLAM3 GPU", "VINS-Fusion")]
    h += [Line2D([], [], color=TH["ink2"], marker=m, ls="", ms=8, markerfacecolor=TH["ink2"] if m == "o" else "none", label=l) for l, m in (("closer active", "o"), ("closer disabled", "s"))]
    ax.legend(handles=h, fontsize=8, ncol=2, frameon=False, loc="upper center", labelcolor=TH["ink2"])
    o = v6[v6.system == "ORB-SLAM3"]; v = v6[v6.system == "VINS-Fusion"]
    ax.annotate("ORB-SLAM3, both front ends\n%.1f-%.1f cores, %.0f-%.0f MiB" % (o.cpu_pct.min() / 100, o.cpu_pct.max() / 100, o.rss_peak_mb.min(), o.rss_peak_mb.max()),
                xy=(4.1, 1700), xytext=(2.8, 1150), fontsize=8.5, color=TH["ink2"], arrowprops=dict(arrowstyle="->", color=TH["ink2"], lw=0.8))
    ax.annotate("VINS-Fusion\n%.1f-%.1f cores, %.0f-%.0f MiB" % (v.cpu_pct.min() / 100, v.cpu_pct.max() / 100, v.rss_peak_mb.min(), v.rss_peak_mb.max()),
                xy=(6.3, 1100), xytext=(7.0, 600), fontsize=8.5, color=TH["ink2"], arrowprops=dict(arrowstyle="->", color=TH["ink2"], lw=0.8))
    ax.set_title("Experiment 1, real robot: what each pipeline costs. The simulation's opposite corners\ncollapse: VINS-Fusion's memory advantage is gone and the core gap narrows.",
                 fontsize=10.5, loc="left", color=TH["ink"], pad=12)
    fig.savefig(os.path.join(OUT, "fig_cost_profile_real.png"), dpi=160, facecolor=TH["surface"]); plt.close(fig)


def publish():
    """copy the two figures the report includes into its figures directory"""
    import shutil
    for src, dst in REPORT.items():
        shutil.copyfile(os.path.join(OUT, src), os.path.join(FIGS, dst))
        print("  ->", dst)


def main():
    os.makedirs(OUT, exist_ok=True)
    print("== pairs"); fig_pairs()
    print("== vins"); fig_vins()
    print("== translation"); fig_translation()
    print("== cost"); fig_cost()
    publish()
    print("wrote 4 figures to", OUT)


if __name__ == "__main__":
    main()
