#!/usr/bin/env python3
"""What the visual estimators' error is MADE OF: direction and rotation.

    python3 script/exp1_error_pattern.py

WHY THIS EXISTS
  Experiment 1 reports a scalar ATE per run and then asks which switch changes
  it -- loop closer, front end, traffic.  A scalar cannot distinguish a robot
  that drifts along the aisle from one that drifts into the racking, and for a
  0.83 x 0.50 m robot in a 3.0 m aisle those are not the same failure.  The
  proprioceptive family has had this decomposition since exp1_insights.py was
  written (along/cross in the ground-truth heading frame); the visual family
  never got it, so today the two families are compared on different quantities.

  This script adds, for the visual arms only:

  A  TRANSLATION PATTERN.  The error vector rotated into the ground-truth
     heading frame at each sample: along-track, cross-track, vertical.  Same
     convention as exp1_insights.decomposition(), so the two families line up.
  B  ROTATION PATTERN.  The orientation error split into yaw, pitch and roll,
     and binned against the ground truth's own angular rate, to ask whether
     heading error grows while the robot turns.

WHICH ALIGNMENT, AND WHY IT MATTERS
  The decomposition is computed on the SAME aligned series the reported ATE
  comes from (vio_metrics.analyse -> rigid SE(3) fit, scale fixed), so the
  parts sum to the number chapter 4 quotes.  exp1_insights.py decomposes the
  proprioceptive arms ANCHORED instead, because dead reckoning has no
  alignment freedom; the anchored variant is reported here too so the families
  can be set side by side.  The two answer different questions and are never
  mixed in one figure.

WHAT THIS CANNOT ESTABLISH
  One run per cell, and ORB-SLAM3's run-to-run spread over identical settings
  reaches 0.277 m -- wider than most differences between settings.  Every
  figure here is therefore DESCRIPTIVE: a pattern that holds across the twenty
  runs and both estimators is reportable; a difference between two runs is
  not.  The script prints the pooled statistic beside every per-run one for
  exactly this reason, and refuses to rank runs.

Outputs (output/compare/simulation/exp1-error-pattern/):
  translation.csv / translation.tex     along/cross/vertical RMS and bias per run
  rotation.csv / rotation.tex           yaw/pitch/roll RMS, and error by turn rate
  fig_translation_light.png             bars + body-frame scatter + growth
  fig_rotation_light.png                components, error vs |omega|, time series
"""

import csv
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                               # noqa: E402
import matplotlib                                # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                  # noqa: E402
import vio_metrics as vm                         # noqa: E402

ROOT = Path(vm.ROOT)
OUT = ROOT / "output" / "compare" / "simulation" / "exp1-error-pattern"
BAGS = ["dataset_allsensor_realistic_%03d" % i for i in range(4)]
# The twenty accuracy runs of Experiment 1's simulation slot.
ARMS = [("VINS-Fusion", "output_vins", "cpu", "#cc8319"),
        ("ORB-SLAM3", "output_orb", "loopclosing_cpu", "#e040fb"),
        ("ORB-SLAM3 GPU", "output_orb", "loopclosing_gpu", "#f48fff"),
        ("ORB-SLAM3 (no LC)", "output_orb", "noloopclosing_cpu", "#a347b2"),
        ("ORB-SLAM3 GPU (no LC)", "output_orb", "noloopclosing_gpu", "#c98fd9")]
COL = {n: c for n, _, _, c in ARMS}
TURN_DEG_S = 10.0          # same threshold the timeline analysis uses
TH = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#dedcd4")


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def series(ds, name, tree, run):
    """The aligned series the reported ATE is computed from, plus the parts."""
    res = vm.analyse("simulation/" + ds,
                     systems=[(name, tree, run), ("VINS-Fusion", "output_vins")])
    if res is None or not res[0]:
        return None
    row, al = res[0][0], res[1][0]
    if al.get("refeul") is None:
        return None
    t, p, ref = al["t"], al["p"], al["ref"]
    err = p - ref                                   # (N,3) vector, not the norm
    gyaw = np.radians(al["refeul"][:, 0])
    c, s = np.cos(gyaw), np.sin(gyaw)
    along = err[:, 0] * c + err[:, 1] * s           # + = ahead of the truth
    cross = -err[:, 0] * s + err[:, 1] * c          # + = left of the truth
    vert = err[:, 2]
    # rotation: component-wise, wrapped; the scalar geodesic is al["roterr"]
    de = wrap180(al["eul"] - al["refeul"])
    # the ground truth's own turn rate, from its unwrapped heading
    gyaw_u = np.unwrap(gyaw)
    omega = np.degrees(np.gradient(gyaw_u, t))
    dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ref[:, :2], axis=0), axis=1))])
    return dict(t=t, along=along, cross=cross, vert=vert, dyaw=de[:, 0], dpitch=de[:, 1],
                droll=de[:, 2], roterr=al["roterr"], omega=omega, dist=dist,
                ate=row["rms"], name=name, dataset=ds)


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a, float) ** 2)))


def collect():
    rows, ser = [], {}
    for ds in BAGS:
        for name, tree, run, _ in ARMS:
            d = ROOT / "output" / tree / "simulation" / ds / run
            if not (d / "vio.csv").exists():
                continue
            e = series(ds, name, tree, run)
            if e is None:
                print("  no series:", ds, name); continue
            ser[(ds, name)] = e
            turning = np.abs(e["omega"]) > TURN_DEG_S
            rows.append(dict(
                dataset=ds, name=name, n=len(e["t"]), secs=float(e["t"][-1] - e["t"][0]),
                ate=e["ate"],
                along_rms=rms(e["along"]), cross_rms=rms(e["cross"]), vert_rms=rms(e["vert"]),
                along_bias=float(np.mean(e["along"])), cross_bias=float(np.mean(e["cross"])),
                vert_bias=float(np.mean(e["vert"])),
                cross_share=float(rms(e["cross"]) ** 2 /
                                  max(rms(e["along"]) ** 2 + rms(e["cross"]) ** 2, 1e-12)),
                yaw_rms=rms(e["dyaw"]), pitch_rms=rms(e["dpitch"]), roll_rms=rms(e["droll"]),
                yaw_med=float(np.median(np.abs(e["dyaw"]))),
                rot_med=float(np.median(e["roterr"])), rot_rms=rms(e["roterr"]),
                turn_frac=float(turning.mean()),
                rot_turn=float(np.median(e["roterr"][turning])) if turning.sum() > 20 else float("nan"),
                rot_straight=float(np.median(e["roterr"][~turning])) if (~turning).sum() > 20 else float("nan"),
                pos_turn=float(np.median(np.abs(e["cross"][turning]))) if turning.sum() > 20 else float("nan"),
                pos_straight=float(np.median(np.abs(e["cross"][~turning]))) if (~turning).sum() > 20 else float("nan"),
            ))
            r = rows[-1]
            print("  %-26s %-22s ATE %5.3f | along %5.3f cross %5.3f vert %5.3f | "
                  "yaw %5.2f pitch %5.2f roll %5.2f | rot med turn/straight %5.2f/%5.2f"
                  % (ds.replace("dataset_allsensor_", ""), name, r["ate"], r["along_rms"],
                     r["cross_rms"], r["vert_rms"], r["yaw_rms"], r["pitch_rms"], r["roll_rms"],
                     r["rot_turn"], r["rot_straight"]))
    return rows, ser


# ----------------------------------------------------------------- figures

def style(ax):
    ax.set_facecolor(TH["surface"])
    for sp in ax.spines.values():
        sp.set_color(TH["grid"])
    ax.grid(True, color=TH["grid"], lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TH["ink2"], labelsize=8)


def fig_translation(rows, ser, path):
    fig = plt.figure(figsize=(13.5, 9.6), facecolor=TH["surface"])
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.62, wspace=0.26)

    # (a) along / cross / vertical RMS, grouped per run
    ax = fig.add_subplot(gs[0, :]); style(ax)
    labels, x = [], 0
    for ds in BAGS:
        for name, _, _, _ in ARMS:
            r = next((q for q in rows if q["dataset"] == ds and q["name"] == name), None)
            if r is None:
                continue
            for k, (key, hatch) in enumerate((("along_rms", ""), ("cross_rms", "///"), ("vert_rms", ".."))):
                ax.bar(x + k * 0.27, r[key], 0.26, color=COL[name], hatch=hatch,
                       edgecolor=TH["ink2"], lw=0.4)
            labels.append((x + 0.27, ds.replace("dataset_allsensor_realistic_", "") + " " + name))
            x += 1.2
        x += 0.5
    ax.set_xticks([p for p, _ in labels])
    ax.set_xticklabels([l for _, l in labels], rotation=90, ha="center", va="top", fontsize=6.2)
    ax.set_ylabel("RMS error [m]", color=TH["ink2"], fontsize=9)
    ax.set_title("(a) the error split in the ground-truth heading frame: "
                 "solid = along-track, hatched = cross-track, dotted = vertical",
                 fontsize=9.5, loc="left", color=TH["ink"])

    # (b) body-frame scatter: anisotropy at a glance
    ax = fig.add_subplot(gs[1, 0]); style(ax)
    allv = np.concatenate([np.abs(np.r_[e["along"], e["cross"]]) for e in ser.values()])
    lim = float(np.percentile(allv, 99))            # the five diverged runs reach metres
    for (ds, name), e in ser.items():
        ax.plot(e["along"], e["cross"], ".", ms=1.0, alpha=0.25, color=COL[name])
    ax.axhline(0, color=TH["ink2"], lw=0.6); ax.axvline(0, color=TH["ink2"], lw=0.6)
    ax.set_aspect("equal"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("along-track [m]", color=TH["ink2"], fontsize=9)
    ax.set_ylabel("cross-track [m]", color=TH["ink2"], fontsize=9)
    ax.set_title("(b) every sample, body frame (clipped to the 99th\npercentile; the diverged runs run off it)",
                 fontsize=9.5, loc="left", color=TH["ink"])

    # (c) cross-track share of the planar error, per run
    ax = fig.add_subplot(gs[1, 1]); style(ax)
    for i, name in enumerate([n for n, _, _, _ in ARMS]):
        v = [r["cross_share"] for r in rows if r["name"] == name]
        if v:
            ax.plot([i] * len(v), v, "o", ms=7, color=COL[name], alpha=0.85)
            ax.plot(i, np.median(v), "_", ms=22, color=TH["ink"], mew=2)
    ax.axhline(0.5, color=TH["ink2"], ls=":", lw=1.0)
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([n for n, _, _, _ in ARMS], rotation=30, ha="right", fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("cross-track share of planar error", color=TH["ink2"], fontsize=9)
    ax.set_title("(c) 0.5 = isotropic; above = drifts sideways", fontsize=9.5,
                 loc="left", color=TH["ink"])

    # (d) does it accumulate? error against distance travelled
    ax = fig.add_subplot(gs[1, 2]); style(ax)
    for (ds, name), e in ser.items():
        ax.plot(e["dist"], np.hypot(e["along"], e["cross"]), lw=0.7, alpha=0.7, color=COL[name])
    ax.set_yscale("symlog", linthresh=0.5)
    ax.set_xlabel("distance travelled [m]", color=TH["ink2"], fontsize=9)
    ax.set_ylabel("planar error [m, symlog]", color=TH["ink2"], fontsize=9)
    ax.set_title("(d) accumulating or bounded?\n(log above 0.5 m so the healthy runs stay visible)",
                 fontsize=9.5, loc="left", color=TH["ink"])

    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=COL[n], lw=3, label=n) for n, _, _, _ in ARMS],
               loc="lower center", ncol=5, frameon=False, fontsize=8.5,
               labelcolor=TH["ink2"], bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Experiment 1, simulation: what the visual estimators' translation error is made of "
                 "(aligned series, the same fit the reported ATE uses)",
                 fontsize=11.5, x=0.012, ha="left", color=TH["ink"])
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    fig.savefig(path, dpi=160, facecolor=TH["surface"])
    plt.close(fig)
    print("  wrote", path)


def fig_rotation(rows, ser, path):
    fig = plt.figure(figsize=(13.5, 8.0), facecolor=TH["surface"])
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.26)

    # (a) yaw / pitch / roll RMS per run
    ax = fig.add_subplot(gs[0, :]); style(ax)
    labels, x = [], 0
    for ds in BAGS:
        for name, _, _, _ in ARMS:
            r = next((q for q in rows if q["dataset"] == ds and q["name"] == name), None)
            if r is None:
                continue
            for k, (key, hatch) in enumerate((("yaw_rms", ""), ("pitch_rms", "///"), ("roll_rms", ".."))):
                ax.bar(x + k * 0.27, r[key], 0.26, color=COL[name], hatch=hatch,
                       edgecolor=TH["ink2"], lw=0.4)
            labels.append((x + 0.27, ds.replace("dataset_allsensor_realistic_", "") + " " + name))
            x += 1.2
        x += 0.5
    ax.set_yscale("log")
    ax.set_xticks([p for p, _ in labels])
    ax.set_xticklabels([l for _, l in labels], rotation=60, ha="right", fontsize=6.5)
    ax.set_ylabel("RMS [deg, log]", color=TH["ink2"], fontsize=9)
    ax.set_title("(a) orientation error by axis: solid = yaw, hatched = pitch, dotted = roll. "
                 "A floor robot's pitch and roll error is gravity alignment, not drift.",
                 fontsize=9.5, loc="left", color=TH["ink"])

    # (b) rotation error against the truth's own turn rate
    ax = fig.add_subplot(gs[1, 0]); style(ax)
    edges = np.array([0, 2, 5, 10, 20, 40, 80])
    for name in [n for n, _, _, _ in ARMS]:
        es = [e for (d, n2), e in ser.items() if n2 == name]
        if not es:
            continue
        w = np.concatenate([np.abs(e["omega"]) for e in es])
        r = np.concatenate([e["roterr"] for e in es])
        med, ctr = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (w >= a) & (w < b)
            if m.sum() > 40:
                med.append(np.median(r[m])); ctr.append(0.5 * (a + b))
        if med:
            ax.plot(ctr, med, "o-", ms=4, lw=1.4, color=COL[name], label=name)
    ax.set_xlabel("ground-truth turn rate |$\\omega$| [deg/s]", color=TH["ink2"], fontsize=9)
    ax.set_ylabel("median rotation error [deg]", color=TH["ink2"], fontsize=9)
    ax.set_title("(b) does turning raise it?", fontsize=9.5, loc="left", color=TH["ink"])

    # (c) turning vs straight, paired per run
    ax = fig.add_subplot(gs[1, 1]); style(ax)
    for r in rows:
        if np.isfinite(r["rot_turn"]) and np.isfinite(r["rot_straight"]):
            ax.plot([0, 1], [r["rot_straight"], r["rot_turn"]], "-o", ms=4, lw=1.0,
                    color=COL[r["name"]], alpha=0.8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["straight", "turning"], fontsize=9)
    ax.set_ylabel("median rotation error [deg]", color=TH["ink2"], fontsize=9)
    ax.set_title("(c) each run, paired", fontsize=9.5, loc="left", color=TH["ink"])

    # (d) one time series, with turns shaded: is the RMS one event?
    ax = fig.add_subplot(gs[1, 2]); style(ax)
    key = next((k for k in ser if k[1] == "ORB-SLAM3 GPU"), None) or (list(ser) or [None])[0]
    if key is not None:
        e = ser[key]
        turning = np.abs(e["omega"]) > TURN_DEG_S
        ax.fill_between(e["t"], 0, 1, where=turning, transform=ax.get_xaxis_transform(),
                        color=TH["grid"], alpha=0.7, lw=0)
        ax.plot(e["t"], e["roterr"], lw=0.9, color=COL[key[1]])
        ax.set_title("(d) %s, %s: shaded = turning" % (key[0].replace("dataset_allsensor_", ""), key[1]),
                     fontsize=9.5, loc="left", color=TH["ink"])
    ax.set_xlabel("time [s]", color=TH["ink2"], fontsize=9)
    ax.set_ylabel("rotation error [deg]", color=TH["ink2"], fontsize=9)

    fig.legend(*fig.axes[1].get_legend_handles_labels(), loc="lower center", ncol=5,
               frameon=False, fontsize=8.5, labelcolor=TH["ink2"], bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Experiment 1, simulation: the orientation error, by axis and against turn rate",
                 fontsize=11.5, x=0.012, ha="left", color=TH["ink"])
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    fig.savefig(path, dpi=160, facecolor=TH["surface"])
    plt.close(fig)
    print("  wrote", path)


# ----------------------------------------------------------------- tables

def tex_translation(rows, path):
    L = ["% Generated by script/exp1_error_pattern.py -- do not edit by hand.",
         "\\begin{table}[htbp]", "\\centering",
         "\\caption[Translation error of the visual arms, split by direction.]{Translation error of "
         "the visual arms split in the ground-truth heading frame, on the same aligned series the "
         "reported ATE is computed from, so that the parts account for the whole. ``Along'' is the "
         "component along the truth's heading and ``cross'' the one perpendicular to it; ``bias'' is "
         "the mean of the signed component, so a bias near the RMS means a constant offset rather "
         "than scatter. ``Cross share'' is the cross-track fraction of the planar squared error: "
         "\\num{0.5} is isotropic and above it the estimate drifts sideways, which is the direction "
         "that matters in a \\SI{3.0}{\\metre} aisle. One run per cell; ORB-SLAM3's run-to-run spread "
         "reaches \\SI{0.277}{\\metre}, so differences smaller than that between single runs are not "
         "evidence.}",
         "\\label{tab:exp1-translation-pattern}", "\\small", "\\resizebox{\\textwidth}{!}{%",
         "\\begin{tabular}{llrrrrrrr}", "\\toprule",
         "Recording & Estimator & ATE [\\si{\\metre}] & Along [\\si{\\metre}] & Cross [\\si{\\metre}] & "
         "Vert.\\ [\\si{\\metre}] & Along bias & Cross bias & Cross share \\\\", "\\midrule"]
    last = None
    for r in rows:
        first = r["dataset"] != last
        if first and last is not None:
            L.append("\\addlinespace")
        last = r["dataset"]
        L.append("%s & %s & %.3f & %.3f & %.3f & %.3f & %+.3f & %+.3f & %.2f \\\\" % (
            "\\repo{%s}" % r["dataset"].replace("dataset_allsensor_", "") if first else "",
            r["name"], r["ate"], r["along_rms"], r["cross_rms"], r["vert_rms"],
            r["along_bias"], r["cross_bias"], r["cross_share"]))
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    open(path, "w").write("\n".join(L) + "\n")


def tex_rotation(rows, path):
    L = ["% Generated by script/exp1_error_pattern.py -- do not edit by hand.",
         "\\begin{table}[htbp]", "\\centering",
         "\\caption[Orientation error of the visual arms, by axis and by turn rate.]{Orientation error "
         "of the visual arms. Yaw, pitch and roll are the wrapped per-axis differences between the "
         "aligned estimate and the truth; for a robot on a flat floor a sustained pitch or roll error "
         "is a gravity-alignment defect rather than drift. The median is given beside the RMS because "
         "the RMS of an ORB-SLAM3 run is dominated by the single re-alignment at its inertial "
         "initialisation. ``Turning'' is a ground-truth turn rate above "
         "\\SI{10}{\\degree\\per\\second}, the threshold the onset analysis uses; the two medians are "
         "paired within a run. One run per cell.}",
         "\\label{tab:exp1-rotation-pattern}", "\\small", "\\resizebox{\\textwidth}{!}{%",
         "\\begin{tabular}{llrrrrrrr}", "\\toprule",
         "Recording & Estimator & Yaw [\\si{\\degree}] & Pitch [\\si{\\degree}] & Roll [\\si{\\degree}] & "
         "Rot.\\ RMS & Rot.\\ med. & Med.\\ turning & Med.\\ straight \\\\", "\\midrule"]
    last = None
    for r in rows:
        first = r["dataset"] != last
        if first and last is not None:
            L.append("\\addlinespace")
        last = r["dataset"]
        f = lambda v: "---" if not np.isfinite(v) else "%.2f" % v
        L.append("%s & %s & %.2f & %.2f & %.2f & %.2f & %.2f & %s & %s \\\\" % (
            "\\repo{%s}" % r["dataset"].replace("dataset_allsensor_", "") if first else "",
            r["name"], r["yaw_rms"], r["pitch_rms"], r["roll_rms"], r["rot_rms"],
            r["rot_med"], f(r["rot_turn"]), f(r["rot_straight"])))
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    open(path, "w").write("\n".join(L) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("== per-run decomposition")
    rows, ser = collect()
    if not rows:
        print("no runs scored"); return 1
    # one row per run carries both decompositions; written under both names so
    # each table's provenance line points at a file of its own
    keys = list(rows[0])
    for nm in ("translation.csv", "rotation.csv"):
        with open(OUT / nm, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
            for r in rows:
                w.writerow(r)
    tex_translation(rows, OUT / "translation.tex")
    tex_rotation(rows, OUT / "rotation.tex")
    fig_translation(rows, ser, OUT / "fig_translation_light.png")
    fig_rotation(rows, ser, OUT / "fig_rotation_light.png")

    # ---- pooled digest: this is what may be reported, not the per-run rows
    print("\n== pooled over the %d runs (what is reportable; single runs are not)" % len(rows))
    cs = [r["cross_share"] for r in rows]
    print("  cross-track share of planar error: median %.2f, range %.2f-%.2f"
          % (np.median(cs), min(cs), max(cs)))
    print("  (0.5 = isotropic; consistently above 0.5 means the estimates drift sideways)")
    for name in [n for n, _, _, _ in ARMS]:
        g = [r for r in rows if r["name"] == name]
        if not g:
            continue
        print("  %-22s cross share %.2f | yaw RMS %.2f deg | rot med %.2f deg"
              % (name, np.median([r["cross_share"] for r in g]),
                 np.median([r["yaw_rms"] for r in g]), np.median([r["rot_med"] for r in g])))
    tt = [(r["rot_turn"], r["rot_straight"]) for r in rows
          if np.isfinite(r["rot_turn"]) and np.isfinite(r["rot_straight"])]
    if tt:
        d = [a - b for a, b in tt]
        print("  rotation error turning minus straight: median %+.2f deg over %d runs, %d of them positive"
              % (np.median(d), len(d), sum(1 for x in d if x > 0)))
    print("\n  wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
