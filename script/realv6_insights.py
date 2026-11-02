#!/usr/bin/env python3
"""Real-robot insight figures and tables for dataset_realv6_000 (Experiment 3, 4.3.2).

Reads the outputs of script/eval_real_gt.py for four scorings of the same bag:
  output/compare/real/realv6/              raw tracker export, ORB rate 1 + VINS
  output/compare/real/realv6_rate05/       raw tracker export, ORB rate 0.5
  output/compare/real/realv6_clean/        one spurious tracker sample removed, ORB rate 1
  output/compare/real/realv6_clean_rate05/ same, ORB rate 0.5
and the run directories themselves, and writes to output/compare/real/realv6-insights/:
  gt_cleaning.tex          raw vs cleaned ground truth: what one sample changed
  orb_summary.tex          ORB-SLAM3 on the cleaned truth, both replay rates
  turning.tex              drift rate while turning vs on straights (ORB, cleaned truth)
  fig_gt_cleaning_light.png   tracker path raw vs cleaned, and the spike in time
  fig_orb_timeline_light.png  ORB error over time with turning bands and the init resets
  fig_vins_divergence_light.png  VINS |position| and frame interval over time

Turning is taken from the ALIGNED ESTIMATE's planar heading (the tracker gives no
heading and is too sparse, 1.7 Hz, for a yaw rate); threshold 10 deg/s as in the
simulation analysis. Nothing here changes the scoring rules of eval_real_gt.py:
the cleaned truth still fails its gap rule and every number stays provisional.
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
RUNS = {  # label -> (eval dir raw, eval dir clean, run dir)
    "ORB-SLAM3, replay 1.0": (R / "realv6" / DS, R / "realv6_clean" / DS,
                              ROOT / "output" / "output_orb" / "real" / DS / "loopclosing_rate1"),
    "ORB-SLAM3, replay 0.5": (R / "realv6_rate05" / DS, R / "realv6_clean_rate05" / DS,
                              ROOT / "output" / "output_orb" / "real" / DS / "loopclosing_rate05"),
}
VINS_RUNS = {"VINS-Fusion, replay 1.0": ROOT / "output" / "output_vins" / "real" / DS / "rate1",
             "VINS-Fusion, replay 0.5": ROOT / "output" / "output_vins" / "real" / DS / "rate05"}
VINS_EVAL = {"VINS-Fusion, replay 1.0": (R / "realv6" / DS, R / "realv6_clean" / DS),
             "VINS-Fusion, replay 0.5": (R / "realv6_rate05" / DS, R / "realv6_clean_rate05" / DS)}
VCOL = {"VINS-Fusion, replay 1.0": "#cc8319", "VINS-Fusion, replay 0.5": "#887b2c"}
GT_RAW = ROOT / "dataset" / "gt" / DS
TURN_DEG_S = 10.0
COL = {"ORB-SLAM3, replay 1.0": "#2a78d6", "ORB-SLAM3, replay 0.5": "#a347b2", "VINS-Fusion": "#cc8319"}


def load_gt_export(path):
    rows = list(csv.DictReader(open(path)))
    t = np.array([float(r["Recorded Time (sec.)"]) for r in rows])
    x = np.array([float(r["X (cm.)"]) for r in rows]) / 100
    y = np.array([float(r["Y (cm.)"]) for r in rows]) / 100
    return t, x, y


def load_aligned(d):
    a = np.loadtxt(d / "aligned_orb.csv", delimiter=",")
    return dict(t=a[:, 0] * 1e-9, gt=a[:, 1:3], est=a[:, 3:5], err=a[:, 5])


def metrics(d):
    m = json.load(open(d / "metrics.json"))
    entries = m if isinstance(m, list) else list(m.values())
    for e in entries:
        if isinstance(e, dict) and "ORB" in str(e.get("system", "")):
            return e
    return entries[0] if entries else {}


def offset_info(d):
    ta = json.load(open(d / "manifest.json")).get("time_alignment", {})
    wheel = ta.get("curves", {}).get("wheel /feedback/velocity", {})
    return dict(value=ta.get("offset_s", float("nan")), r=wheel.get("peak_r", float("nan")), ok=ta.get("ok"))


def heading_rate(t, xy, win_s=1.0):
    """Yaw rate of a planar path from its direction of travel, smoothed over win_s."""
    d = np.diff(xy, axis=0)
    dt = np.diff(t)
    ok = np.hypot(d[:, 0], d[:, 1]) > 0.01
    hd = np.full(len(t), np.nan)
    hd[1:][ok] = np.arctan2(d[ok, 1], d[ok, 0])
    # forward-fill heading where the robot did not move
    for i in range(1, len(hd)):
        if np.isnan(hd[i]):
            hd[i] = hd[i - 1]
    hd[0] = hd[1] if len(hd) > 1 else 0.0
    hd = np.unwrap(np.nan_to_num(hd))
    w = np.degrees(np.gradient(hd, t))
    k = max(1, int(round(win_s / max(np.median(dt), 1e-3))))
    return np.convolve(np.abs(w), np.ones(k) / k, mode="same")


def rate_contrast(t, err, turning):
    dt, de = np.diff(t), np.diff(err)
    ok = (dt > 0) & (dt < 3.0)
    s1 = turning[:-1] & turning[1:]
    s0 = ~turning[:-1] & ~turning[1:]
    T1, T0 = dt[ok & s1].sum(), dt[ok & s0].sum()
    if T1 < 3 or T0 < 3:
        return None
    return de[ok & s1].sum() / T1, de[ok & s0].sum() / T0, T1, T0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # ---------------------------------------------------------------- GT cleaning
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

    # table: raw vs clean, both runs
    rows = []
    for label, (draw, dclean, _) in RUNS.items():
        for tag, d in (("raw", draw), ("cleaned", dclean)):
            m = metrics(d); o = offset_info(d)
            rows.append((label, tag, m, o))
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption[Realv6: raw versus cleaned tracker export.]{\repo{dataset_realv6_000}: the same ORB-SLAM3 runs "
         r"scored against the tracker export as delivered and with one sample removed (the \SI{5.2}{\metre} "
         r"excursion at \SI{184.9}{\second}). Time offset by wheel-speed cross-correlation with its peak "
         r"correlation $r$; aligned planar ATE; path ratio = estimate / truth. The cleaned truth still fails the "
         r"gap rule (66 gaps above \SI{1}{\second}, tracker at \SI{1.7}{\hertz}), so every value is provisional.}",
         r"\label{tab:realv6-cleaning}",
         r"\begin{tabular}{llrrrrrr}", r"\toprule",
         r"Run & truth & GT path [m] & offset [s] & $r$ & ATE [m] & median [m] & path ratio \\", r"\midrule"]
    for label, tag, m, o in rows:
        L.append(f"{label} & {tag} & {m.get('gt_path_m', float('nan')):.1f} & {o['value']:.2f} & {o['r']:.2f} & "
                 f"{m.get('ate_rmse', float('nan')):.3f} & {m.get('ate_median', float('nan')):.3f} & "
                 f"{m.get('path_ratio', float('nan')):.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "gt_cleaning.tex").write_text("\n".join(L) + "\n")

    # ---------------------------------------------------------------- ORB timeline
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    turn_rows = []
    for ax, (label, (_, dclean, rdir)) in zip(axes, RUNS.items()):
        a = load_aligned(dclean)
        t0 = a["t"][0]
        w = heading_rate(a["t"], a["est"])
        turning = w > TURN_DEG_S
        ymax = max(1.0, float(a["err"].max()) * 1.15)
        ax.fill_between(a["t"] - t0, 0, ymax, where=turning, color="#888", alpha=0.2, lw=0, label="turning (estimate heading > 10 deg/s)")
        ax.plot(a["t"] - t0, a["err"], color=COL[label], lw=1.3, label="planar error vs tracker")
        # resets from the tracking state log
        fr = list(csv.DictReader(open(rdir / "tracking_frontend.csv")))
        ft = np.array([int(r["timestamp_ns"]) for r in fr]) * 1e-9
        st = np.array([int(r["tracking_state"]) for r in fr])
        ch = np.where((st[:-1] == 2) & (st[1:] != 2))[0]
        for i in ch:
            ax.axvline(ft[i + 1] - t0, color="#b00020", lw=1.0, ls=":", label="tracking left OK (init resets)" if i == ch[0] else None)
        ax.set_ylim(0, ymax); ax.set_ylabel("error [m]")
        ax.set_title(label, fontsize=10, loc="left"); ax.grid(True, alpha=0.3)
        h, l = ax.get_legend_handles_labels(); ax.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(), fontsize=8, loc="upper right")
        c = rate_contrast(a["t"], a["err"], turning)
        turn_rows.append((label, c, float(turning.mean()), float(np.ptp(a["t"]))))
    axes[-1].set_xlabel("time from first matched sample [s]")
    fig.suptitle("ORB-SLAM3 on dataset_realv6_000 (cleaned truth): error over time", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(OUT / "fig_orb_timeline_light.png", dpi=160); plt.close(fig)

    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption[Realv6: ORB-SLAM3 drift while turning versus on straights.]{ORB-SLAM3 on "
         r"\repo{dataset_realv6_000}, cleaned truth: growth rate of the planar error while the aligned estimate's "
         r"heading changes faster than \SI{10}{\degree\per\second} and on straights, with the time spent in each state. "
         r"Positive = error growing. One run per rate, so no test is possible.}",
         r"\label{tab:realv6-turning}",
         r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Run & turning [mm/s] & straight [mm/s] & time turning [s] & time straight [s] \\", r"\midrule"]
    for label, c, share, span in turn_rows:
        if c:
            L.append(f"{label} & {1000*c[0]:.1f} & {1000*c[1]:.1f} & {c[2]:.0f} & {c[3]:.0f} \\\\")
        else:
            L.append(f"{label} & --- & --- & --- & --- \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "turning.tex").write_text("\n".join(L) + "\n")

    # ---------------------------------------------------------------- ORB summary table
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption[Realv6: ORB-SLAM3 against the cleaned tracker truth.]{ORB-SLAM3 on \repo{dataset_realv6_000} "
         r"against the cleaned tracker truth, both replay rates. Aligned planar ATE after a rigid SE(2) fit with scale "
         r"fixed; RPE at \SI{1}{\second} translational only; resets are ORB-SLAM3's \repo{local_mapping_bad_imu} map "
         r"resets, all inside the first \SI{13}{\second}; dropped frames are camera frames the node's queue discarded. "
         r"Lever arm tracker-marker to IMU unmeasured (zero applied); the least-squares fit is a diagnostic only.}",
         r"\label{tab:realv6-orb}",
         r"\scriptsize\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
         r"Run & matched & ATE [m] & median [m] & max [m] & RPE$_{1\,\mathrm{s}}$ [m] & path ratio & resets & dropped frames \\", r"\midrule"]
    for label, (_, dclean, rdir) in RUNS.items():
        m = metrics(dclean)
        summ = dict(r.split(",", 1) for r in open(rdir / "run_summary.csv").read().splitlines() if "," in r)
        rpe1 = m.get("rpe", {}).get("1.0", {}).get("trans_rmse_clean", float("nan"))
        L.append(f"{label} & {m.get('matched', '---')}/{m.get('gt_in_span', '---')} & {m.get('ate_rmse', float('nan')):.3f} & "
                 f"{m.get('ate_median', float('nan')):.3f} & {m.get('ate_max', float('nan')):.3f} & "
                 f"{rpe1:.3f} & {m.get('path_ratio', float('nan')):.2f} & "
                 f"{m.get('resets', '---')} & {m.get('frames_dropped', '---')} of {m.get('frames_received', '---')} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "orb_summary.tex").write_text("\n".join(L) + "\n")

    # ---------------------------------------------------------------- VINS divergence, both rates
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    vins_rows = []
    for label, rdir in VINS_RUNS.items():
        a = np.loadtxt(rdir / "vio.csv", delimiter=",")
        t = (a[:, 0] - a[0, 0]) * 1e-9; r = np.hypot(a[:, 1], a[:, 2])
        es = list(csv.DictReader(open(rdir / "estimator_state.csv")))
        te = (np.array([int(x["timestamp_ns"]) for x in es]) - a[0, 0]) * 1e-9
        v = np.sqrt(sum(np.array([float(x[c] or "nan") for x in es]) ** 2 for c in ("vx", "vy", "vz")))
        ba = np.sqrt(sum(np.array([float(x[c] or "nan") for x in es]) ** 2 for c in ("accel_bias_x", "accel_bias_y", "accel_bias_z")))
        fr = list(csv.DictReader(open(rdir / "frontend.csv")))
        ft = (np.array([int(x["timestamp_ns"]) for x in fr]) - a[0, 0]) * 1e-9
        summ = dict(x.split(",", 1) for x in open(rdir / "run_summary.csv").read().splitlines() if "," in x)
        axes[0].semilogy(t, np.maximum(r, 1e-2), color=VCOL[label], lw=1.4, label=f"{label}: {summ.get('images_received','?')} of 6601 frames received")
        axes[1].plot(te, v, color=VCOL[label], lw=1.2, label=label)
        axes[2].plot(te, ba, color=VCOL[label], lw=1.2, label=label)
        # divergence onset: first time |v| exceeds 1 m/s for good
        over = np.where(v > 1.0)[0]
        t_div = float(te[over[0]]) if len(over) else float("nan")
        i60, i120 = min(np.searchsorted(t, 60), len(t) - 1), min(np.searchsorted(t, 120), len(t) - 1)
        vins_rows.append((label, summ.get("images_received", "?"), float(np.median(np.diff(ft))), t_div,
                          r[i60], r[i120], r[-1], float(np.nanmedian(ba[(te > 30) & (te < 50)]))))
    axes[0].set_ylabel("|planar position| [m], log"); axes[1].set_ylabel("|velocity| [m/s]"); axes[2].set_ylabel("|accel. bias| [m/s$^2$]")
    axes[2].axhline(9.886 - 9.2642, color="#b00020", lw=0.9, ls="--", label="IMU |a| at rest minus configured g_norm (9.886 - 9.264)")
    for ax in axes:
        ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=8, loc="upper left")
    axes[2].set_xlabel("time from first pose [s]")
    fig.suptitle("VINS-Fusion on dataset_realv6_000, both replay rates: position, velocity and accelerometer bias", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(OUT / "fig_vins_divergence_light.png", dpi=160); plt.close(fig)

    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption[Realv6: the two VINS-Fusion runs.]{VINS-Fusion on \repo{dataset_realv6_000} at both replay rates: "
         r"camera frames the node received (bag: 6\,601), median interval between processed frames, the time at which "
         r"the estimated speed first exceeds \SI{1}{\metre\per\second} (the robot never exceeds \SI{0.9}{\metre\per\second}), "
         r"the distance of the estimate from its origin at three times, and the accelerometer-bias magnitude the "
         r"estimator settled on between 30 and \SI{50}{\second}. The configured gravity norm is \num{9.2642}; the IMU "
         r"in this bag reads \num{9.886} at rest.}",
         r"\label{tab:realv6-vins}",
         r"\scriptsize\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{lrrrrrrr}", r"\toprule",
         r"Run & frames & interval [s] & speed $>$1 at [s] & $|p|$ 60 s [m] & $|p|$ 120 s [m] & $|p|$ end [m] & accel.\ bias [m/s$^2$] \\", r"\midrule"]
    for row in vins_rows:
        L.append(f"{row[0]} & {row[1]} & {row[2]:.2f} & {row[3]:.0f} & {row[4]:.0f} & {row[5]:.0f} & {row[6]:.0f} & {row[7]:.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "vins_summary.tex").write_text("\n".join(L) + "\n")

    # ---------------------------------------------------------------- all four, aligned, one figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    gtr = np.loadtxt(R / "realv6_clean" / DS / "ground_truth_robotclock.csv", delimiter=",")
    for ax in axes:
        ax.plot(gtr[:, 1], gtr[:, 2], color="#2a2a2a", lw=2.2, label="tracker truth (cleaned)")
    for label, (_, dclean, _) in RUNS.items():
        a = load_aligned(dclean); axes[0].plot(a["est"][:, 0], a["est"][:, 1], color=COL[label], lw=1.3, label=label)
    for label, (_, dclean) in VINS_EVAL.items():
        a = np.loadtxt(dclean / "aligned_vins.csv", delimiter=",")
        axes[1].plot(a[:, 3], a[:, 4], color=VCOL[label], lw=1.0, label=label)
    axes[0].set_title("ORB-SLAM3, both replay rates (SE(2) aligned)", fontsize=10, loc="left")
    axes[1].set_title("VINS-Fusion, both replay rates (same alignment; note the axes)", fontsize=10, loc="left")
    for ax in axes:
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.legend(fontsize=8)
    fig.suptitle("dataset_realv6_000: every run against the cleaned tracker truth", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(OUT / "fig_all_runs_light.png", dpi=160); plt.close(fig)

    # console digest
    print("GT raw path %.1f m, cleaned %.1f m; spike samples removed %d" % (step.sum(), stepc.sum(), int(spike.sum())))
    for label, c, share, span in turn_rows:
        print(label, "turning share %.2f" % share, "contrast", None if c is None else tuple(round(v, 4) for v in c))
    for row in vins_rows:
        print(row[0], "frames", row[1], "interval %.2f  speed>1 at %.0f s  |p| 60/120/end %.0f/%.0f/%.0f  bias %.2f" % row[2:])


if __name__ == "__main__":
    main()
