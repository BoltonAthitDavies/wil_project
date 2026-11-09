#!/usr/bin/env python3
"""Real-robot figures and tables for dataset_realv6_000 (Experiment 3, slot 4.3.2).

Only runs whose run directories still exist are included (3 November 2026
afternoon series, IMU noise x10, repaired library, replay rate 1.0): ORB-SLAM3
stereo-inertial with GPU or CPU front end and loop closing on or off, and one
VINS-Fusion run.  The 2 November and the 3 November morning runs were overwritten
and are deliberately absent.

Reads the scorer's outputs (script/eval_real_gt.py; cleaned truth = the tracker
export with the one spurious sample at 184.9 s removed, offset pinned to
-4.141 s; "raw" = export as delivered, offset pinned to +4.377 s):
  output/compare/real/realv6_1103pm_lc_gpu_clean/     ORB GPU, LC on
  output/compare/real/realv6_1103pm_lc_gpu_raw/       the same run on the raw export
  output/compare/real/realv6_1103pm_lc_cpu_clean/     ORB CPU, LC on
  output/compare/real/realv6_1103pm_nolc_gpu_clean/   ORB GPU, LC off
  output/compare/real/realv6_1103pm_nolc_cpu_clean/   ORB CPU, LC off
  output/compare/real/realv6_1103pm_vins_clean/       VINS-Fusion
and writes to output/compare/real/realv6-insights/:
  final_table.tex           every surviving run: ATE, median, max, RPE1, path ratio, resets, dropped
  gt_cleaning.tex           raw vs cleaned truth on the GPU/LC-on run
  turning.tex               drift while turning vs on straights, per ORB run
  vins_table.tex            the VINS run: frames, IMU use, resets, tracked span
  fig_paths_light.png       2-D path of EVERY run against the tracker truth (one panel each)
  fig_gt_cleaning_light.png the tracker export raw vs cleaned, and the spike in the speed profile
  fig_orb_timeline_light.png ORB error over time per run, turning bands, resets
  fig_vins_light.png        VINS position, IMU message use and resets over time

Turning is taken from the aligned estimate's planar heading (the tracker gives no
heading and is 1.7 Hz); threshold 10 deg/s as in the simulation analysis. Nothing
here changes the scorer's validity rules: the cleaned truth still fails its gap
rule and every number stays provisional.
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vio_metrics as vm                      # noqa: E402

import matplotlib                             # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt               # noqa: E402

ROOT = Path(vm.ROOT)
R = ROOT / "output" / "compare" / "real"
OUT = R / "realv6-insights"
DS = "dataset_realv6_000"
ORB_DIR = ROOT / "output" / "output_orb" / "real" / DS
VINS_DIR = ROOT / "output" / "output_vins" / "real" / DS
RUNS = {  # label -> (clean scoring dir, run dir)
    "ORB-SLAM3, GPU, LC on":  (R / "realv6_1103pm_lc_gpu_clean" / DS,   ORB_DIR / "loopclosing_rate1"),
    "ORB-SLAM3, CPU, LC on":  (R / "realv6_1103pm_lc_cpu_clean" / DS,   ORB_DIR / "loopclosing_rate1_cpu"),
    "ORB-SLAM3, GPU, LC off": (R / "realv6_1103pm_nolc_gpu_clean" / DS, ORB_DIR / "noloopclosing_rate1"),
    "ORB-SLAM3, CPU, LC off": (R / "realv6_1103pm_nolc_cpu_clean" / DS, ORB_DIR / "noloopclosing_rate1_cpu"),
}
RAW_RUN = ("ORB-SLAM3, GPU, LC on", R / "realv6_1103pm_lc_gpu_raw" / DS)
VINS = ("VINS-Fusion", R / "realv6_1103pm_vins_clean" / DS, VINS_DIR / "rate1")
GT_RAW = ROOT / "dataset" / "gt" / DS
TURN_DEG_S = 10.0
COL = {"ORB-SLAM3, GPU, LC on": "#2a78d6", "ORB-SLAM3, CPU, LC on": "#0097a7",
       "ORB-SLAM3, GPU, LC off": "#a347b2", "ORB-SLAM3, CPU, LC off": "#e040fb", "VINS-Fusion": "#cc8319"}


# ---------------------------------------------------------------- loaders

def load_gt_export(path):
    rows = list(csv.DictReader(open(path)))
    t = np.array([float(r["Recorded Time (sec.)"]) for r in rows])
    return t, np.array([float(r["X (cm.)"]) for r in rows]) / 100, np.array([float(r["Y (cm.)"]) for r in rows]) / 100


def load_aligned(d, system="orb"):
    a = np.loadtxt(d / f"aligned_{system}.csv", delimiter=",")
    return dict(t=a[:, 0] * 1e-9, gt=a[:, 1:3], est=a[:, 3:5], err=a[:, 5])


def metrics(d, system="ORB"):
    m = json.load(open(d / "metrics.json"))
    entries = m if isinstance(m, list) else list(m.values())
    for e in entries:
        if isinstance(e, dict) and system in str(e.get("system", "")):
            return e
    return entries[0] if entries else {}


def offset_info(d):
    ta = json.load(open(d / "manifest.json")).get("time_alignment", {})
    return dict(value=ta.get("offset_s", float("nan")), source=ta.get("source", ""))


def summary(rdir):
    return dict(x.split(",", 1) for x in open(rdir / "run_summary.csv").read().splitlines() if "," in x)


def heading_rate(t, xy, win_s=1.0):
    d = np.diff(xy, axis=0); dt = np.diff(t)
    ok = np.hypot(d[:, 0], d[:, 1]) > 0.01
    hd = np.full(len(t), np.nan); hd[1:][ok] = np.arctan2(d[ok, 1], d[ok, 0])
    for i in range(1, len(hd)):
        if np.isnan(hd[i]):
            hd[i] = hd[i - 1]
    hd[0] = hd[1] if len(hd) > 1 else 0.0
    w = np.degrees(np.gradient(np.unwrap(np.nan_to_num(hd)), t))
    k = max(1, int(round(win_s / max(np.median(dt), 1e-3))))
    return np.convolve(np.abs(w), np.ones(k) / k, mode="same")


def rate_contrast(t, err, turning):
    dt, de = np.diff(t), np.diff(err)
    ok = (dt > 0) & (dt < 3.0)
    s1 = turning[:-1] & turning[1:]; s0 = ~turning[:-1] & ~turning[1:]
    T1, T0 = dt[ok & s1].sum(), dt[ok & s0].sum()
    if T1 < 3 or T0 < 3:
        return None
    return de[ok & s1].sum() / T1, de[ok & s0].sum() / T0, T1, T0


def fmt(v, p=2):
    return "---" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{p}f}"


# ---------------------------------------------------------------- outputs

def final_table():
    L = [r"\begin{table}[htbp]", r"\centering", r"\scriptsize\setlength{\tabcolsep}{3pt}",
         r"\caption[Realv6: every surviving run against the cleaned tracker truth.]{ORB-SLAM3 on "
         r"\repo{dataset_realv6_000} against the cleaned tracker truth, 3 November 2026 afternoon series: repaired "
         r"library, replay rate 1.0, 3 November stereo extrinsic, \texttt{IMU.NoiseGyro} 0.1 and "
         r"\texttt{IMU.NoiseAcc} 1.0, GPU or CPU feature extractor, loop closing (LC) on or off. Aligned planar ATE "
         r"after a rigid SE(2) fit with scale fixed; RPE at \SI{1}{\second} translational; path ratio = estimate / "
         r"truth; resets = \repo{local_mapping_bad_imu} map resets; dropped = camera frames the node's queue discarded "
         r"of about 6\,480 received. Lever arm tracker-marker to IMU unmeasured (zero applied); time offset pinned to "
         r"\SI{-4.14}{\second}; all values provisional under the scorer's validity rules. \\textbf{Bold} marks the "
         r"lowest ATE of the five runs; with one run per configuration that records which run came lowest, not "
         r"that the configuration is better.}",
         r"\label{tab:realv6-final}",
         r"\begin{tabular}{lrrrrrrr}", r"\toprule",
         r"Run & ATE [m] & median [m] & max [m] & RPE$_{1\,\mathrm{s}}$ [m] & path ratio & resets & dropped \\", r"\midrule"]
    mall = {label: metrics(dclean) for label, (dclean, rdir) in RUNS.items()}
    finite = [m["ate_rmse"] for m in mall.values() if m.get("ate_rmse") is not None and np.isfinite(m["ate_rmse"])]
    best = min(finite) if finite else None
    for label, (dclean, rdir) in RUNS.items():
        m = mall[label]; rpe1 = m.get("rpe", {}).get("1.0", {}).get("trans_rmse_clean", float("nan"))
        ate = f"{m['ate_rmse']:.2f}"
        if best is not None and abs(m["ate_rmse"] - best) < 1e-9:
            ate = r"\textbf{" + ate + "}"
        L.append(f"{label} & {ate} & {m['ate_median']:.2f} & {m['ate_max']:.2f} & {rpe1:.3f} & "
                 f"{m['path_ratio']:.2f} & {m.get('resets', '---')} & {m.get('frames_dropped', '---')} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "final_table.tex").write_text("\n".join(L) + "\n")


def gt_cleaning():
    t, x, y = load_gt_export(GT_RAW)
    spike = (t > 184.5) & (t < 185.2)
    step = np.hypot(np.diff(x), np.diff(y)); dt = np.diff(t)
    tc, xc, yc = t[~spike], x[~spike], y[~spike]
    stepc = np.hypot(np.diff(xc), np.diff(yc))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    ax.plot(x, y, color="#b00020", lw=0.8, alpha=0.6, label="tracker export, raw")
    ax.plot(xc, yc, color="#2a2a2a", lw=1.2, label="one sample removed (184.9 s)")
    ax.scatter(x[spike], y[spike], color="#b00020", s=40, zorder=5, label="the sample")
    ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right"); ax.set_title("Tracker path", fontsize=10, loc="left")
    ax = axes[1]
    ax.plot(t[1:], step / dt, color="#b00020", lw=0.8, label="speed from raw samples")
    ax.plot(tc[1:], stepc / np.diff(tc), color="#2a2a2a", lw=1.0, label="speed after removal")
    ax.set_xlabel("tracker time [s]"); ax.set_ylabel("speed [m/s]"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8); ax.set_title("Speed used for the time alignment", fontsize=10, loc="left")
    fig.suptitle("dataset_realv6_000: what one tracker sample did", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(OUT / "fig_gt_cleaning_light.png", dpi=160); plt.close(fig)

    label, draw = RAW_RUN; dclean = RUNS[label][0]
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption[Realv6: raw versus cleaned tracker export.]{The same ORB-SLAM3 run (GPU, LC on) scored against the "
         r"tracker export as delivered and with one sample removed (the \SI{5.2}{\metre} excursion at "
         r"\SI{184.9}{\second}). The time offsets are the values the wheel-speed cross-correlation found on each "
         r"export on 2 November (peak correlation 0.10 raw, 0.61 cleaned), applied here as pinned values. The cleaned "
         r"truth still fails the gap rule (66 gaps above \SI{1}{\second}, tracker at \SI{1.7}{\hertz}).}",
         r"\label{tab:realv6-cleaning}",
         r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Truth & GT path [m] & offset [s] & ATE [m] & median [m] & path ratio \\", r"\midrule"]
    for tag, d in (("as delivered", draw), ("one sample removed", dclean)):
        m = metrics(d); o = offset_info(d)
        L.append(f"{tag} & {m['gt_path_m']:.1f} & {o['value']:.2f} & {m['ate_rmse']:.3f} & {m['ate_median']:.3f} & {m['path_ratio']:.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "gt_cleaning.tex").write_text("\n".join(L) + "\n")
    return step.sum(), stepc.sum()


def paths_figure():
    """Every surviving run, one panel each, against the cleaned tracker truth."""
    gtr = np.loadtxt(RUNS["ORB-SLAM3, GPU, LC on"][0] / "ground_truth_robotclock.csv", delimiter=",")
    items = [(label, dclean, "orb") for label, (dclean, _) in RUNS.items()] + [(VINS[0], VINS[1], "vins")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.6))
    for ax, (label, d, sysk) in zip(axes.ravel(), items):
        a = load_aligned(d, sysk); m = metrics(d, "ORB" if sysk == "orb" else "VINS")
        ax.plot(gtr[:, 1], gtr[:, 2], color="#2a2a2a", lw=2.0, label="tracker truth (cleaned)")
        ax.plot(a["est"][:, 0], a["est"][:, 1], color=COL[label], lw=1.3, label="estimate, SE(2) aligned")
        ax.plot(a["est"][0, 0], a["est"][0, 1], "o", color=COL[label], ms=6); ax.plot(a["est"][-1, 0], a["est"][-1, 1], "s", color=COL[label], ms=6)
        span = a["t"][-1] - a["t"][0]
        ax.set_title(f"{label}: ATE {m['ate_rmse']:.2f} m over {span:.0f} s, {m.get('matched', '?')} samples", fontsize=9, loc="left")
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.set_xlim(gtr[:, 1].min() - 0.8, gtr[:, 1].max() + 0.8); ax.set_ylim(gtr[:, 2].min() - 0.8, gtr[:, 2].max() + 0.8)
        ax.legend(fontsize=7, loc="lower right")
    axes.ravel()[-1].axis("off")
    fig.suptitle("dataset_realv6_000: every surviving run against the cleaned tracker truth (circle = start, square = end)", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(OUT / "fig_paths_light.png", dpi=160); plt.close(fig)


def orb_timeline():
    fig, axes = plt.subplots(len(RUNS), 1, figsize=(10, 2.4 * len(RUNS) + 0.6), sharex=True)
    rows = []
    for ax, (label, (dclean, rdir)) in zip(axes, RUNS.items()):
        a = load_aligned(dclean); t0 = a["t"][0]
        turning = heading_rate(a["t"], a["est"]) > TURN_DEG_S
        ymax = max(1.0, float(a["err"].max()) * 1.15)
        ax.fill_between(a["t"] - t0, 0, ymax, where=turning, color="#888", alpha=0.2, lw=0, label="turning (estimate heading > 10 deg/s)")
        ax.plot(a["t"] - t0, a["err"], color=COL[label], lw=1.3, label="planar error vs tracker")
        fr = list(csv.DictReader(open(rdir / "tracking_frontend.csv")))
        ft = np.array([int(r["timestamp_ns"]) for r in fr]) * 1e-9; st = np.array([int(r["tracking_state"]) for r in fr])
        ch = np.where((st[:-1] == 2) & (st[1:] != 2))[0]
        for i in ch:
            ax.axvline(ft[i + 1] - t0, color="#b00020", lw=1.0, ls=":", label="tracking left OK" if i == ch[0] else None)
        ax.set_ylim(0, ymax); ax.set_ylabel("error [m]"); ax.set_title(label, fontsize=10, loc="left"); ax.grid(True, alpha=0.3)
        h, l = ax.get_legend_handles_labels(); ax.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(), fontsize=7, loc="upper right")
        c = rate_contrast(a["t"], a["err"], turning); rows.append((label, c))
    axes[-1].set_xlabel("time from first matched sample [s]")
    fig.suptitle("ORB-SLAM3 on dataset_realv6_000 (cleaned truth): error over time", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(OUT / "fig_orb_timeline_light.png", dpi=160); plt.close(fig)
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption[Realv6: ORB-SLAM3 drift while turning versus on straights.]{ORB-SLAM3 on "
         r"\repo{dataset_realv6_000}, cleaned truth: growth rate of the planar error while the aligned estimate's "
         r"heading changes faster than \SI{10}{\degree\per\second} and on straights, with the time in each state. "
         r"Positive = error growing. One run per configuration, so no test is possible.}",
         r"\label{tab:realv6-turning}",
         r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Run & turning [mm/s] & straight [mm/s] & time turning [s] & time straight [s] \\", r"\midrule"]
    for label, c in rows:
        L.append(f"{label} & {fmt(1000*c[0],1)} & {fmt(1000*c[1],1)} & {fmt(c[2],0)} & {fmt(c[3],0)} \\\\" if c else f"{label} & --- & --- & --- & --- \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (OUT / "turning.tex").write_text("\n".join(L) + "\n")
    return rows


def vins_outputs():
    label, dclean, rdir = VINS
    a = np.loadtxt(rdir / "vio.csv", delimiter=","); t = (a[:, 0] - a[0, 0]) * 1e-9; r = np.hypot(a[:, 1], a[:, 2])
    ev = list(csv.DictReader(open(rdir / "events.csv")))
    resets = [(int(e["timestamp_ns"]) - a[0, 0]) * 1e-9 for e in ev if e["event"] == "reset"]
    perf = list(csv.DictReader(open(rdir / "performance.csv")))
    pt = (np.array([int(x["timestamp_ns"]) for x in perf]) - a[0, 0]) * 1e-9
    imu_used = np.array([float(x["imu_used"]) for x in perf]); imu_rx = np.array([float(x["imu_received"]) for x in perf])
    s = summary(rdir); m = metrics(dclean, "VINS")
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
    axes[0].plot(t, r, color=COL[label], lw=1.3, label="|planar position| of the estimate")
    for i, x in enumerate(resets):
        axes[0].axvline(x, color="#b00020", lw=0.5, alpha=0.5, label="imu_gap reset" if i == 0 else None)
    axes[0].set_ylabel("|position| [m]"); axes[0].legend(fontsize=8, loc="upper left"); axes[0].grid(True, alpha=0.3)
    axes[1].plot(pt, imu_rx, color="#555", lw=1.0, label="IMU messages received (cumulative)")
    axes[1].plot(pt, imu_used, color=COL[label], lw=1.3, label="IMU samples used (cumulative)")
    axes[1].set_ylabel("count"); axes[1].set_xlabel("time from first pose [s]"); axes[1].legend(fontsize=8, loc="upper left"); axes[1].grid(True, alpha=0.3)
    fig.suptitle("VINS-Fusion on dataset_realv6_000, 3 November: %d imu_gap resets, poses stop at %.0f s" % (len(resets), t[-1]), fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(OUT / "fig_vins_light.png", dpi=160); plt.close(fig)
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption[Realv6: the VINS-Fusion run.]{VINS-Fusion on \repo{dataset_realv6_000}, 3 November 2026 "
         r"(\repo{config/wil/stereo_imu.yaml}, IMU noise $\times 10$, gravity norm still \num{9.2642} against the "
         r"\SI{9.878}{\metre\per\second\squared} this IMU reads at rest). The scored span is the \SI{18}{\second} it "
         r"tracked before its last reset and is not a result.}",
         r"\label{tab:realv6-vins}",
         r"\begin{tabular}{lr}", r"\toprule", r"Quantity & Value \\", r"\midrule",
         f"camera frames received (bag: 6\\,601) & {s.get('images_received', '?')} \\\\",
         f"IMU messages received / samples used & {s.get('imu_messages_received', '?')} / {s.get('imu_samples_used', '?')} \\\\",
         f"\\repo{{imu_gap}} resets & {len(resets)} \\\\",
         f"poses written / span of poses & {s.get('poses_written', '?')} / \\SI{{{t[-1]:.0f}}}{{\\second}} \\\\",
         f"tracker samples matched & {m.get('matched', '?')} of {m.get('gt_in_span', '?')} in span \\\\",
         f"ATE over the tracked span & \\SI{{{m['ate_rmse']:.2f}}}{{\\metre}} (not a result) \\\\",
         r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "vins_table.tex").write_text("\n".join(L) + "\n")
    return len(resets), t[-1]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in ("orb_summary.tex", "vins_summary.tex", "fig_all_runs_light.png", "fig_vins_divergence_light.png"):
        p = OUT / stale
        if p.exists():
            p.unlink()
    final_table()
    raw_path, clean_path = gt_cleaning()
    paths_figure()
    rows = orb_timeline()
    n_resets, span = vins_outputs()
    print("GT path raw %.1f m, cleaned %.1f m" % (raw_path, clean_path))
    for label, c in rows:
        print(label, "turn/straight mm/s:", None if c is None else (round(1000 * c[0], 1), round(1000 * c[1], 1)))
    print("VINS: %d resets, poses stop at %.0f s" % (n_resets, span))


if __name__ == "__main__":
    main()
