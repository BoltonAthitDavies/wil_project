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
from matplotlib.ticker import FuncFormatter            # noqa: E402
from plot_ticks import plain_log                     # noqa: E402
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
                ate=row["rms"], name=name, dataset=ds, t0=al["t0"])


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a, float) ** 2)))


def anchored(e, run_dir):
    """Rotation error measured only after the published world frame exists.

    ORB-SLAM3 publishes in the camera OPTICAL frame until inertial
    initialization gives it a gravity-aligned world frame, and the switch
    between the two is a ~90 deg step in the published orientation. Those
    earlier poses are accurate but expressed in a different frame, so scoring
    them against a world-frame reference measures the frame, not the estimate:
    on these runs 2.0-3.6 %% of the samples at ~90 deg produce the whole
    12.8-19.0 deg RMS.

    vio_metrics.world_frame_time() finds the switch in the POSE STREAM ALONE,
    so no part of the reference enters the anchor it is then scored against.
    An arm that publishes in one frame throughout -- VINS-Fusion here -- has no
    switch, and its anchored columns are its ordinary ones rather than blanks,
    so a reader comparing the two families compares like with like.
    """
    t_abs = vm.world_frame_time(str(run_dir))
    keep = (np.ones_like(e["t"], bool) if t_abs is None
            else e["t"] >= t_abs - e["t0"])
    if keep.sum() < 30:
        keep = np.ones_like(e["t"], bool)
        t_abs = None
    return dict(t_switch=float("nan") if t_abs is None else t_abs - e["t0"],
                rot_rms_anch=rms(e["roterr"][keep]),
                rot_med_anch=float(np.median(e["roterr"][keep])),
                yaw_rms_anch=rms(e["dyaw"][keep]),
                pitch_rms_anch=rms(e["dpitch"][keep]),
                roll_rms_anch=rms(e["droll"][keep]))


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
                **anchored(e, d),
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


# A run whose aligned ATE passes this is one of the five divergences reported
# in the slot (two false closures on realistic_001, two vertical-drift runs on
# realistic_003, one marginal run on realistic_002).  They are MARKED in every
# panel rather than pooled silently with the fifteen that held: the spread they
# create is a failure mode, and a reader who cannot see which points they are
# will read that spread as ordinary variation.
DIVERGED_ATE_M = 1.0


def fig_translation(rows, ser, path):
    """Two panels, sharing one run order: the split, and the direction.

    The earlier version carried two further panels -- a body-frame scatter of
    every sample and planar error against distance -- which no sentence of the
    slot cited.  The scatter was dominated by the diverged runs' loops with
    the fifteen healthy runs an unreadable blob at the origin, and the
    distance panel asked `accumulating or bounded?` over twenty unlabelled
    overlapping traces, which cannot be read off it.  Both are removed.  What
    is left is the decomposition itself and the one quantity the slot quotes
    from it.
    """
    fig = plt.figure(figsize=(13.5, 7.3), facecolor=TH["surface"])
    L, R = 0.068, 0.986
    gs_a = fig.add_gridspec(1, 1, left=L, right=R, top=0.895, bottom=0.560)
    gs_b = fig.add_gridspec(1, 1, left=L, right=R, top=0.505, bottom=0.392)

    # one run order, used by both panels so a run sits at the same x in each
    order = [(ds, name) for ds in BAGS for name, _, _, _ in ARMS
             if any(q["dataset"] == ds and q["name"] == name for q in rows)]
    pos, labels, gap = {}, [], 0.0
    x = 0.0
    for i, (ds, name) in enumerate(order):
        if i and ds != order[i - 1][0]:
            x += 0.5
        pos[(ds, name)] = x
        r = next(q for q in rows if q["dataset"] == ds and q["name"] == name)
        mark = "" if r["ate"] < DIVERGED_ATE_M else " $\\dagger$"
        labels.append((x + 0.27, short_arm(ds, name) + mark))
        x += 1.2

    # (a) along / cross / vertical RMS, grouped per run
    ax = fig.add_subplot(gs_a[0, 0]); style(ax)
    for (ds, name), xp in pos.items():
        r = next(q for q in rows if q["dataset"] == ds and q["name"] == name)
        for k, (key, hatch) in enumerate((("along_rms", ""), ("cross_rms", "///"),
                                          ("vert_rms", ".."))):
            ax.bar(xp + k * 0.27, r[key], 0.26, color=COL[name], hatch=hatch,
                   edgecolor=TH["ink2"], lw=0.4)
        if r["ate"] >= DIVERGED_ATE_M:
            ax.axvspan(xp - 0.17, xp + 0.71, color=TH["ink2"], alpha=0.07, lw=0, zorder=0)
    ax.set_xticks([])
    ax.set_xlim(-0.6, x - 0.5)
    ax.set_ylabel("RMS error [m]", color=TH["ink2"], fontsize=9)
    ax.set_title("(a) the error split in the ground-truth heading frame: "
                 "solid = along-track, hatched = cross-track, dotted = vertical. "
                 "Shaded columns, $\\dagger$: the five diverged runs.",
                 fontsize=9.5, loc="left", color=TH["ink"])

    # (b) cross-track share of the planar error, same run order as (a)
    ax = fig.add_subplot(gs_b[0, 0]); style(ax)
    for (ds, name), xp in pos.items():
        r = next(q for q in rows if q["dataset"] == ds and q["name"] == name)
        div = r["ate"] >= DIVERGED_ATE_M
        if div:
            ax.axvspan(xp - 0.17, xp + 0.71, color=TH["ink2"], alpha=0.07, lw=0, zorder=0)
        ax.plot(xp + 0.27, r["cross_share"], "o", ms=7, zorder=3,
                color="none" if div else COL[name],
                markerfacecolor="none" if div else COL[name],
                markeredgecolor=COL[name], mew=1.6)
    ax.axhline(0.5, color=TH["ink2"], ls=":", lw=1.0)
    ax.set_xticks([p for p, _ in labels])
    ax.set_xticklabels([l for _, l in labels], rotation=90, ha="center",
                       va="top", fontsize=6.6)
    ax.set_xlim(-0.6, x - 0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("cross-track share", color=TH["ink2"], fontsize=9)
    # The share is sin^2(theta), where theta is the angle between the error's
    # principal direction and the robot's heading, so the same axis can be
    # read as an angle.  That is what "perpendicular" in the text means, and
    # without this axis the reader has to take the word on trust.
    ang = ax.secondary_yaxis(
        "right", functions=(lambda v: np.degrees(np.arcsin(np.sqrt(np.clip(v, 0, 1)))),
                            lambda a: np.sin(np.radians(np.clip(a, 0, 90))) ** 2))
    ang.set_yticks([0, 15, 30, 45, 60, 75, 90])
    ang.set_ylabel("error direction off heading [deg]", color=TH["ink2"], fontsize=8)
    ang.tick_params(colors=TH["ink2"], labelsize=7)
    ax.set_title("(b) the same runs, cross-track share of the planar error: "
                 "0.5 is isotropic, above it the estimate drifts sideways. "
                 "Open markers, $\\dagger$: the diverged runs.\n"
                 "    The right axis reads the same number as an angle off the "
                 "direction of travel, since the share is $\\sin^2\\theta$: "
                 "VINS-Fusion near $56^\\circ$, ORB-SLAM3 near $25^\\circ$.",
                 fontsize=9.5, loc="left", color=TH["ink"])

    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=COL[n], lw=3, label=n)
                        for n, _, _, _ in ARMS],
               loc="lower center", ncol=5, frameon=False, fontsize=8.5,
               labelcolor=TH["ink2"], bbox_to_anchor=(0.5, 0.004))
    fig.suptitle("Experiment 1, simulation: what the visual estimators' translation "
                 "error is made of (aligned series, the same fit the reported ATE uses)",
                 fontsize=11.5, x=0.012, y=0.972, ha="left", color=TH["ink"])
    fig.savefig(path, dpi=160, facecolor=TH["surface"])
    plt.close(fig)
    print("  wrote", path)


def short_arm(ds, name):
    """`000 ORB GPU (no LC)` rather than `000 ORB-SLAM3 GPU (no LC)`.

    Twenty tick labels stand vertically under panel (a) of the rotation
    figure; at full length they reach into the title of the panel below it.
    The legend carries the full arm names, so the axis does not have to.
    """
    return (ds.replace("dataset_allsensor_realistic_", "") + " " +
            name.replace("ORB-SLAM3", "ORB").replace("VINS-Fusion", "VINS"))


def fig_rotation(rows, ser, path):
    """The orientation error, with the frame artefact separated from the estimate.

    Panel (a) used to plot the unanchored per-axis RMS alone, which put every
    ORB-SLAM3 bar at 13-19 deg and so read against the text beside it: the
    chapter reports 0.27-3.14 deg once the published world frame exists.  Both
    are now drawn -- the unanchored bar ghosted behind, the anchored bar solid
    in front -- so the tall part is visibly the pre-initialization frame and
    the solid part is the tracking.  VINS-Fusion publishes in one frame
    throughout, so its two coincide and only the solid bar shows, which is the
    comparison the reader needs.

    Panel (b) is the evidence for that split and is given the full width: one
    step at the switch, then nothing, on every arm at once.

    Panel (a) carries twenty run labels up to 28 characters long.  At any
    rotation but 90 deg they reach sideways as well as down and run into the
    titles below; the translation figure sets 90 deg for the same reason.
    """
    # The three bands are placed explicitly rather than by one hspace.  A
    # single hspace has to be large enough for the twenty vertical labels
    # under (a), which then leaves the same gap under (b), where nothing but
    # an axis label sits -- a band of dead paper in the middle of the figure.
    fig = plt.figure(figsize=(13.5, 10.6), facecolor=TH["surface"])
    L, R = 0.068, 0.986
    gs_a = fig.add_gridspec(1, 1, left=L, right=R, top=0.902, bottom=0.612)
    gs_b = fig.add_gridspec(1, 1, left=L, right=R, top=0.468, bottom=0.330)
    gs_c = fig.add_gridspec(1, 1, left=0.30, right=0.74, top=0.250, bottom=0.085)

    # (a) yaw / pitch / roll RMS per run: unanchored ghosted, anchored solid
    ax = fig.add_subplot(gs_a[0, 0]); style(ax)
    labels, x = [], 0
    # Markers, not bars: on a log axis a bar's length has no meaning (it depends
    # on where the axis happens to start), so each value is a point. The open
    # marker is the whole-run RMS, the filled one the RMS after the published
    # world frame exists, and the thin line between them is the excess the
    # pre-initialisation frame adds.
    AXES = (("yaw_rms", "yaw_rms_anch", "o"),
            ("pitch_rms", "pitch_rms_anch", "^"),
            ("roll_rms", "roll_rms_anch", "s"))
    for ds in BAGS:
        for name, _, _, _ in ARMS:
            r = next((q for q in rows if q["dataset"] == ds and q["name"] == name), None)
            if r is None:
                continue
            for k, (key, akey, mk) in enumerate(AXES):
                xx = x + k * 0.27
                ax.plot([xx, xx], [r[akey], r[key]], color=COL[name], lw=0.9, alpha=0.55, zorder=2)
                ax.plot(xx, r[key], mk, ms=5.5, mfc=TH["surface"], mec=COL[name], mew=1.1, zorder=3)
                ax.plot(xx, r[akey], mk, ms=5.5, mfc=COL[name], mec=TH["ink2"], mew=0.4, zorder=4)
            if r["ate"] >= DIVERGED_ATE_M:
                ax.axvspan(x - 0.17, x + 0.71, color=TH["ink2"], alpha=0.07,
                           lw=0, zorder=0)
            labels.append((x + 0.27, short_arm(ds, name) +
                           ("" if r["ate"] < DIVERGED_ATE_M else " $\\dagger$")))
            x += 1.2
        x += 0.5
    ax.set_yscale("log"); plain_log(ax)
    ax.set_xticks([p for p, _ in labels])
    ax.set_xticklabels([l for _, l in labels], rotation=90, ha="center",
                       va="top", fontsize=6.6)
    ax.set_ylabel("RMS [deg, log]", color=TH["ink2"], fontsize=9)
    ax.set_title("(a) orientation error by axis: circle = yaw, triangle = pitch, square = roll; "
                 "open = scored over the whole run, filled = after the published world frame exists.\n"
                 "    The gap between them is the pre-initialization frame, not drift: all three axes move together "
                 "and VINS-Fusion has none of it. Shaded, $\\dagger$: the five diverged runs.",
                 fontsize=9.5, loc="left", color=TH["ink"])

    # (b) the frame switch is one event -- the evidence panel (a) is split on
    ax = fig.add_subplot(gs_b[0, 0]); style(ax)
    ds0 = BAGS[0]
    sw = [r["t_switch"] for r in rows
          if r["dataset"] == ds0 and np.isfinite(r["t_switch"])]
    for name, _, _, _ in ARMS:
        e = ser.get((ds0, name))
        if e is None:
            continue
        ax.plot(e["t"], np.maximum(e["roterr"], 1e-3), lw=0.9, color=COL[name],
                label=name, zorder=3)
    if sw:
        ax.axvline(float(np.median(sw)), color=TH["ink"], lw=1.0, ls="--", zorder=4)
        ax.annotate("published world frame\nbegins (%.2f s), found in the\npose stream alone"
                    % float(np.median(sw)),
                    xy=(float(np.median(sw)), 60), xytext=(14, 60),
                    fontsize=8, color=TH["ink2"], va="center",
                    arrowprops=dict(arrowstyle="->", color=TH["ink2"], lw=0.8))
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_ylim(0, 200)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: "%g" % v))
    ax.set_xlabel("time [s]", color=TH["ink2"], fontsize=9)
    ax.set_ylabel("rotation error [deg, symlog]", color=TH["ink2"], fontsize=9)
    ax.set_title("(b) the excess is one event, not drift: %s, every arm. One step of ~90 deg at "
                 "initialization,\n    then sub-degree for the remaining two minutes; "
                 "VINS-Fusion publishes in one frame throughout and has no such step."
                 % ds0.replace("dataset_allsensor_", ""),
                 fontsize=9.5, loc="left", color=TH["ink"])

    # (c) turning vs straight, paired WITHIN each run -- the test the text quotes
    #     The binned-by-turn-rate panel that used to sit beside this one pooled
    #     twenty runs across four routes and three estimators into one curve,
    #     and read against the sentence it illustrated: the pooled top bin rose
    #     while the paired test below is flat.  A median over a mixture is not
    #     evidence about any member of it, so only the paired test remains.
    ax = fig.add_subplot(gs_c[0, 0]); style(ax)
    for r in rows:
        if np.isfinite(r["rot_turn"]) and np.isfinite(r["rot_straight"]):
            div = r["ate"] >= DIVERGED_ATE_M
            ax.plot([0, 1], [r["rot_straight"], r["rot_turn"]],
                    "--" if div else "-", marker="o", ms=4, lw=1.0,
                    color=COL[r["name"]], alpha=0.55 if div else 0.9,
                    markerfacecolor="none" if div else COL[r["name"]])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["straight", "turning"], fontsize=9)
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylabel("median rotation error [deg]", color=TH["ink2"], fontsize=9)
    ax.set_title("(c) each run, paired: straight against turning\n"
                 "    (dashed, open = the diverged runs)",
                 fontsize=9.5, loc="left", color=TH["ink"])

    handles = dict()
    for a in fig.axes:
        for h, l in zip(*a.get_legend_handles_labels()):
            handles.setdefault(l, h)
    order = [n for n, _, _, _ in ARMS if n in handles]
    fig.legend([handles[n] for n in order], order, loc="lower center", ncol=5,
               frameon=False, fontsize=8.5, labelcolor=TH["ink2"],
               bbox_to_anchor=(0.5, 0.004))
    fig.suptitle("Experiment 1, simulation: the orientation error, with the published "
                 "frame separated from the estimate",
                 fontsize=11.5, x=0.012, y=0.978, ha="left", color=TH["ink"])
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
         "is a gravity-alignment defect rather than drift. ``Anch.'' is the same RMS measured only "
         "after the instant each run\'s published poses enter the gravity-aligned world frame: until "
         "inertial initialization ORB-SLAM3 publishes in the camera optical frame, and the "
         "\\SIrange{89.8}{90.2}{\\degree} step between the two is located in the pose stream alone, "
         "without reference to the truth. The unanchored RMS is therefore a measure of that frame "
         "switch rather than of tracking, and the anchored column is the one to read; the median "
         "beside it is less sensitive to single spikes. ``Turning'' is a "
         "ground-truth turn rate above \\SI{10}{\\degree\\per\\second}, the threshold the onset "
         "analysis uses; the two medians are paired within a run. One run per cell.}",
         "\\label{tab:exp1-rotation-pattern}", "\\small", "\\resizebox{\\textwidth}{!}{%",
         "\\begin{tabular}{llrrrrrrrrr}", "\\toprule",
         "Recording & Estimator & Yaw [\\si{\\degree}] & Pitch [\\si{\\degree}] & Roll [\\si{\\degree}] & "
         "Rot.\\ RMS & Anch.\\ [\\si{\\second}] & Rot.\\ RMS anch. & Rot.\\ med. & "
         "Med.\\ turning & Med.\\ straight \\\\", "\\midrule"]
    last = None
    for r in rows:
        first = r["dataset"] != last
        if first and last is not None:
            L.append("\\addlinespace")
        last = r["dataset"]
        f = lambda v: "---" if not np.isfinite(v) else "%.2f" % v
        L.append("%s & %s & %.2f & %.2f & %.2f & %.2f & %s & %.2f & %.2f & %s & %s \\\\" % (
            "\\repo{%s}" % r["dataset"].replace("dataset_allsensor_", "") if first else "",
            r["name"], r["yaw_rms"], r["pitch_rms"], r["roll_rms"], r["rot_rms"],
            "---" if not np.isfinite(r["t_switch"]) else "%.2f" % r["t_switch"],
            r["rot_rms_anch"], r["rot_med"], f(r["rot_turn"]), f(r["rot_straight"])))
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
