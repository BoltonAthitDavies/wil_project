#!/usr/bin/env python3
"""Covariance-consistency and accuracy analysis for the non-visual baselines.

    python3 script/analyse_proprio.py --dataset simulation/dataset_allsensor_000 \\
        --experiment-id exp-estimator-baseline

    # both bags at once, into one comparison directory
    python3 script/analyse_proprio.py \\
        --dataset simulation/dataset_allsensor_000 \\
        --dataset simulation/dataset_allsensor_001 \\
        --experiment-id exp-estimator-baseline

WHAT THIS ANSWERS
    Accuracy is the easy half: how far did wheel odometry, IMU dead reckoning and
    the EKF end up from ground truth. The interesting half is whether each
    estimator's REPORTED uncertainty describes its ACTUAL error. A filter whose
    3-sigma envelope excludes the truth 40% of the time is overconfident, and
    downstream consumers -- a costmap, a relocaliser, a pose-graph edge weight --
    will trust it more than they should.

    That is measured here with NEES, the normalised estimation error squared:

        NEES(t) = e(t)^T P(t)^-1 e(t),    e = [dx, dy, dyaw] against ground truth

    If P is right, NEES averages the number of degrees of freedom -- 3 for the pose,
    2 for position alone. Much larger means overconfident, much smaller means
    needlessly conservative. The chi-square interval turns "much" into a number.

WHY THE NEES TEST SUBSAMPLES, AND WHY THAT MATTERS
    The chi-square bounds on an N-sample average assume the N samples are
    independent. At 200 Hz they are emphatically not: consecutive poses of a
    dead-reckoning estimator differ by one integration step and their errors are
    almost perfectly correlated. Using all 17416 samples would shrink the interval
    by a factor of sqrt(N) and declare every estimator inconsistent, which would be
    an artefact of the sampling rate rather than a property of the filter.

    So the consistency test is run on a subsample (--nees-hz, default 1 Hz) and the
    effective sample count is reported next to the verdict. This is a stated
    approximation, not a fix: the residuals at 1 Hz are still correlated, just far
    less so. Read the verdict as indicative and lean on the sigma-coverage figures,
    which make no independence assumption at all.

TWO ERROR CONVENTIONS, BOTH REPORTED
    anchored   estimate minus ground truth, no alignment. These estimators start
               from the ground-truth pose by construction, so this is the error a
               robot would actually have experienced. It is the honest number for a
               dead-reckoning baseline, and it is what NEES must use -- P describes
               uncertainty about the true pose, not about a post-hoc best fit.

    aligned    SE(3) Umeyama fit over the overlap, scale fixed, via vio_metrics.
               This is the convention the rest of Chapter 4 already uses for
               ORB-SLAM3 and VINS, so it is the only one that can be put in a table
               beside them.

    They are not interchangeable and the aligned figure is always the smaller of the
    two. Quoting one where the other belongs is the easiest way to make this
    baseline look better or worse than it is.
"""

import argparse
import csv
import math
import os
import sys

import numpy as np

try:
    from scipy.stats import chi2
except ImportError:
    chi2 = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vio_metrics as vm

ROOT = vm.ROOT

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#dedcd4"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", grid="#3a3a37"),
}
# One colour per estimator, held fixed across every figure so a reader who has
# learned the legend once does not have to re-learn it per plot.
COLOUR = {"Wheel odometry": "#4fc3f7",
          "IMU dead reckoning": "#e57373",
          "EKF wheel+IMU": "#81c784",
          # Same hues viewer.py and plot_compare.py already use for these two, so a
          # reader moving between the overlay, the comparison figures and this one
          # does not have to re-learn which trail is which.
          "VINS-Fusion": "#ffb74d",
          "ORB-SLAM3": "#e040fb",
          "ORB-SLAM3 (no LC)": "#ab47bc"}


# Trees whose estimator starts FROM the ground-truth pose, which is the only case
# where an unaligned difference against ground truth means anything. VINS-Fusion and
# ORB-SLAM3 each build their own world frame -- ORB's is its first keyframe -- so the
# same subtraction would measure the arbitrary offset between two frames, produce a
# number in the tens of metres, and say nothing about either estimator. Their anchored
# columns are left empty rather than filled with that.
ANCHORED_TREES = {"output_wheel", "output_imu", "output_ekf"}


def load_cov(path):
    if not os.path.exists(path):
        return None
    t, blocks, var_v, var_b = [], [], [], []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            t.append(int(row["t_ns"]) * 1e-9)
            vx, vy, vyaw = (float(row["var_x"]), float(row["var_y"]),
                            float(row["var_yaw"]))
            cxy, cxt, cyt = (float(row["cov_xy"]), float(row["cov_xyaw"]),
                             float(row["cov_yyaw"]))
            blocks.append([[vx, cxy, cxt], [cxy, vy, cyt], [cxt, cyt, vyaw]])
            var_v.append(float(row["var_v"]))
            var_b.append(float(row["var_bgyro"]))
    return dict(t=np.array(t), P=np.array(blocks),
                var_v=np.array(var_v), var_b=np.array(var_b))


def load_traj(path):
    """vio.csv -> t, x, y, yaw. Yaw is unwrapped so differences stay continuous."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    a = np.loadtxt(path, delimiter=",", ndmin=2)
    if a.shape[0] < 10:
        return None
    t = a[:, 0] * 1e-9
    qw, qz = a[:, 4], a[:, 7]
    yaw = np.unwrap(2.0 * np.arctan2(qz, qw))
    return dict(t=t, x=a[:, 1], y=a[:, 2], yaw=yaw)


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def loop_closer_state(run_dir):
    """'active', 'disabled', or None when there is no loop_closing.csv to read.

    Read from the artifact, never from the directory name. The name cannot be
    trusted across campaigns: logging_20260916_02 is loop-closure-OFF for the
    allsensor datasets but simply REPEAT 2 for the relogged ones, where all three
    runs have an active loop closer. Pointing this script at a relogged dataset
    with the default run names would otherwise label a repeat as "no LC" and
    produce a comparison of one configuration against itself.

    A disabled run is unmistakable in the log: the loop-closing thread spins with
    nothing queued to it, so every row carries current_keyframe_id = -1. Note that
    a summary line reading loop_closures,0 proves nothing -- on allsensor_000 the
    enabled run found 90 candidates and rejected all 90.
    """
    path = os.path.join(run_dir, "loop_closing.csv")
    if not os.path.exists(path):
        return None
    n = nulls = 0
    with open(path) as f:
        for row in csv.DictReader(f):
            n += 1
            if row.get("current_keyframe_id") == "-1":
                nulls += 1
    if n == 0:
        return None
    return "disabled" if nulls == n else "active"


def analyse_one(dataset, args):
    """Per-estimator anchored error, NEES and sigma coverage for one dataset."""
    out = []
    gt = None
    for entry in args.systems:
        tree = vm.split(entry)[1]
        c = os.path.join(ROOT, "output", tree, dataset, "ground_truth.csv")
        if os.path.exists(c):
            a = np.loadtxt(c, delimiter=",", ndmin=2)
            gt = dict(t=a[:, 0] * 1e-9, x=a[:, 1], y=a[:, 2],
                      yaw=np.unwrap(2.0 * np.arctan2(a[:, 7], a[:, 4])))
            break
    if gt is None:
        print("  [skip] %s: no ground_truth.csv under any baseline tree" % dataset)
        return out

    for entry in args.systems:
        label, tree, run = vm.split(entry)
        d = vm.run_dir(tree, dataset, run or "auto")
        # Check the pinned run really is the configuration its label claims.
        want = ("disabled" if "no LC" in label else
                "active" if label == "ORB-SLAM3" else None)
        if want is not None:
            got = loop_closer_state(d)
            if got is not None and got != want:
                print("  [WARN] %s: %s has a %s loop closer, not %s. The run name "
                      "means something different for this dataset -- pass "
                      "--orb-run/--orb-nolc-run explicitly."
                      % (label, os.path.basename(d), got, want))
        est = load_traj(os.path.join(d, "vio.csv"))
        cov = load_cov(os.path.join(d, "covariance.csv"))
        if est is None:
            print("  [skip] %s: no usable vio.csv in %s" % (label, d))
            continue

        lo = max(est["t"][0], gt["t"][0])
        hi = min(est["t"][-1], gt["t"][-1])
        m = (est["t"] >= lo) & (est["t"] <= hi)
        if m.sum() < 10:
            print("  [skip] %s: %d overlapping samples" % (label, m.sum()))
            continue
        t = est["t"][m]
        ex = est["x"][m] - np.interp(t, gt["t"], gt["x"])
        ey = est["y"][m] - np.interp(t, gt["t"], gt["y"])
        eyaw = wrap(est["yaw"][m] - np.interp(t, gt["t"], gt["yaw"]))
        epos = np.hypot(ex, ey)

        rec = dict(dataset=dataset, name=label, t=t - t[0], t_abs=t,
                   ex=ex, ey=ey, eyaw=eyaw, epos=epos,
                   anchored=tree in ANCHORED_TREES,
                   n=len(t), secs=t[-1] - t[0],
                   anchored_rms=float(np.sqrt((epos ** 2).mean())),
                   anchored_final=float(epos[-1]),
                   anchored_max=float(epos.max()),
                   yaw_rms_deg=float(np.degrees(np.sqrt((eyaw ** 2).mean()))),
                   yaw_final_deg=float(np.degrees(abs(eyaw[-1]))))

        if cov is not None:
            # Interpolate the nine block entries once each, then reshape. Doing the
            # interpolation inside a per-sample loop is O(n^2) and takes minutes on
            # a 17k-sample run for a result identical to this.
            P = np.empty((len(t), 3, 3))
            for i in range(3):
                for j in range(3):
                    P[:, i, j] = np.interp(t, cov["t"], cov["P"][:, i, j])
            rec["sx"] = np.sqrt(np.clip(P[:, 0, 0], 0, None))
            rec["sy"] = np.sqrt(np.clip(P[:, 1, 1], 0, None))
            rec["syaw"] = np.sqrt(np.clip(P[:, 2, 2], 0, None))

            # P is exactly zero at the anchored first pose, so it is singular there.
            # Skip until it is positive definite rather than regularising it: a
            # pseudo-inverse would invent an uncertainty the filter never claimed.
            e3 = np.stack([ex, ey, eyaw], axis=1)
            nees = np.full(len(t), np.nan)
            nees2 = np.full(len(t), np.nan)
            for k in range(len(t)):
                try:
                    w = np.linalg.eigvalsh(P[k])
                    if w.min() <= 1e-15:
                        continue
                    nees[k] = float(e3[k] @ np.linalg.solve(P[k], e3[k]))
                    p2 = P[k][:2, :2]
                    if np.linalg.eigvalsh(p2).min() > 1e-15:
                        nees2[k] = float(e3[k, :2] @ np.linalg.solve(p2, e3[k, :2]))
                except np.linalg.LinAlgError:
                    continue
            rec["nees"] = nees
            rec["nees_pos"] = nees2

            # Subsample before averaging -- see the module docstring.
            step = max(1, int(round(len(t) / max(1e-9, t[-1] - t[0]) / args.nees_hz)))
            sel = np.arange(0, len(t), step)
            v3 = nees[sel][np.isfinite(nees[sel])]
            v2 = nees2[sel][np.isfinite(nees2[sel])]
            rec["anees"] = float(v3.mean()) if len(v3) else float("nan")
            rec["anees_pos"] = float(v2.mean()) if len(v2) else float("nan")
            rec["nees_samples"] = int(len(v3))
            rec["verdict"] = verdict(rec["anees"], len(v3), 3)

            # Sigma coverage makes no independence assumption, so it is the more
            # trustworthy of the two consistency statements.
            with np.errstate(invalid="ignore", divide="ignore"):
                zx = np.abs(ex) / np.where(rec["sx"] > 0, rec["sx"], np.nan)
                zy = np.abs(ey) / np.where(rec["sy"] > 0, rec["sy"], np.nan)
            ok = np.isfinite(zx) & np.isfinite(zy)
            z = np.maximum(zx[ok], zy[ok])
            for s in (1, 2, 3):
                rec["cov%d" % s] = float((z <= s).mean()) if ok.sum() else float("nan")
            rec["sx_final"] = float(rec["sx"][-1])
            rec["sy_final"] = float(rec["sy"][-1])
            rec["syaw_final_deg"] = float(np.degrees(rec["syaw"][-1]))
        out.append(rec)
    return out


def verdict(anees, n, dof):
    """Chi-square consistency verdict for an average NEES over n samples."""
    if not np.isfinite(anees) or n < 5:
        return "insufficient"
    if chi2 is None:
        return "optimistic" if anees > dof * 1.5 else (
            "conservative" if anees < dof * 0.5 else "consistent")
    lo = chi2.ppf(0.025, dof * n) / n
    hi = chi2.ppf(0.975, dof * n) / n
    if anees > hi:
        return "overconfident"
    if anees < lo:
        return "conservative"
    return "consistent"


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def figure(recs, dataset, outdir, theme):
    th = THEMES[theme]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor=th["surface"])

    def style(ax, xl, yl, title):
        ax.set_facecolor(th["surface"])
        ax.set_xlabel(xl, color=th["ink2"])
        ax.set_ylabel(yl, color=th["ink2"])
        ax.set_title(title, color=th["ink"], fontsize=10.5, loc="left")
        ax.tick_params(colors=th["ink2"], labelsize=8)
        ax.grid(True, color=th["grid"], lw=0.5)
        for s in ax.spines.values():
            s.set_color(th["grid"])

    ax = axes[0][0]
    for r in recs:
        if r["anchored"]:
            tt, ee, tag = r["t"], r["epos"], "anchored"
        elif "al_err" in r:
            tt, ee, tag = r["al_t"], r["al_err"], "SE(3)-aligned"
        else:
            continue
        ax.plot(tt, ee, lw=1.2, color=COLOUR.get(r["name"], "#999"),
                label="%s, %s (final %.2f m)" % (r["name"], tag, ee[-1]))
    style(ax, "t [s]", "position error [m]",
          "Position error vs ground truth (baselines anchored, VIO SE(3)-aligned)")
    ax.set_yscale("log")
    ax.legend(fontsize=7.5, facecolor=th["surface"], labelcolor=th["ink2"],
              edgecolor=th["grid"])

    ax = axes[0][1]
    for r in recs:
        if "sx" not in r or not r["anchored"]:
            continue
        c = COLOUR.get(r["name"], "#999")
        ax.plot(r["t"], np.abs(r["ex"]), lw=1.0, color=c, label="%s |dx|" % r["name"])
        ax.plot(r["t"], 3.0 * r["sx"], lw=1.0, ls="--", color=c, alpha=0.65,
                label="%s 3-sigma" % r["name"])
    style(ax, "t [s]", "x error and 3-sigma bound [m]",
          "Claimed uncertainty vs actual error (x axis)")
    ax.set_yscale("log")
    ax.legend(fontsize=7, facecolor=th["surface"], labelcolor=th["ink2"],
              edgecolor=th["grid"], ncol=2)

    ax = axes[1][0]
    for r in recs:
        if "nees" not in r:
            continue
        ok = np.isfinite(r["nees"])
        ax.plot(r["t"][ok], r["nees"][ok], lw=0.8,
                color=COLOUR.get(r["name"], "#999"),
                label="%s (ANEES %.1f, %s)" % (r["name"], r["anees"], r["verdict"]))
    ax.axhline(3.0, color=th["ink2"], lw=1.0, ls=":")
    ax.text(0.01, 0.95, "consistent = 3 (3 DOF)", transform=ax.transAxes,
            color=th["ink2"], fontsize=7.5, va="top")
    style(ax, "t [s]", "NEES [-]", "Covariance consistency: NEES over time")
    ax.set_yscale("log")
    ax.legend(fontsize=7.5, facecolor=th["surface"], labelcolor=th["ink2"],
              edgecolor=th["grid"])

    ax = axes[1][1]
    for r in recs:
        if "syaw" not in r or not r["anchored"]:
            continue
        c = COLOUR.get(r["name"], "#999")
        ax.plot(r["t"], np.degrees(np.abs(r["eyaw"])), lw=1.0, color=c,
                label="%s |dyaw|" % r["name"])
        ax.plot(r["t"], np.degrees(3.0 * r["syaw"]), lw=1.0, ls="--", color=c,
                alpha=0.65, label="%s 3-sigma" % r["name"])
    style(ax, "t [s]", "heading error and bound [deg]",
          "Heading error vs claimed uncertainty")
    ax.set_yscale("log")
    ax.legend(fontsize=7, facecolor=th["surface"], labelcolor=th["ink2"],
              edgecolor=th["grid"], ncol=2)

    n = recs[0]["n"] if recs else 0
    fig.suptitle("%s   n=%d poses, %.1f s   (anchored error, no SE(3) alignment)"
                 % (dataset, n, recs[0]["secs"] if recs else 0),
                 color=th["ink"], fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    # The dataset belongs in the filename: without it the second dataset silently
    # overwrites the first one's figures, and the surviving PNG looks like a
    # complete result rather than half of one.
    p = os.path.join(outdir, "proprio_covariance_%s_%s.png"
                     % (dataset.split("/")[-1], theme))
    fig.savefig(p, dpi=160, facecolor=th["surface"])
    plt.close(fig)
    return p


def trajectory_figure(recs, dataset, outdir, theme, systems):
    th = THEMES[theme]
    fig, ax = plt.subplots(figsize=(8, 8), facecolor=th["surface"])
    ax.set_facecolor(th["surface"])
    gt = None
    for entry in systems:
        tree = vm.split(entry)[1]
        c = os.path.join(ROOT, "output", tree, dataset, "ground_truth.csv")
        if os.path.exists(c):
            a = np.loadtxt(c, delimiter=",", ndmin=2)
            gt = (a[:, 1], a[:, 2])
            break
    if gt is not None:
        ax.plot(gt[0], gt[1], lw=2.0, color="#8a8983", label="ground truth")
    for r in recs:
        if r["anchored"]:
            byname = {vm.split(x)[0]: vm.split(x) for x in systems}
            _, tr, rn = byname[r["name"]]
            e = load_traj(os.path.join(vm.run_dir(tr, dataset, rn or "auto"),
                                       "vio.csv"))
            if e is None:
                continue
            xs, ys = e["x"], e["y"]
            lbl = "%s (final err %.2f m)" % (r["name"], r["anchored_final"])
        elif "al_p" in r:
            xs, ys = r["al_p"][:, 0], r["al_p"][:, 1]
            lbl = "%s, SE(3)-aligned (ATE %.2f m)" % (
                r["name"], r.get("aligned_ate_rms", float("nan")))
        else:
            continue
        ax.plot(xs, ys, lw=1.2, color=COLOUR.get(r["name"], "#999"), label=lbl)
    ax.set_aspect("equal", "datalim")
    ax.set_xlabel("x [m]", color=th["ink2"])
    ax.set_ylabel("y [m]", color=th["ink2"])
    ax.set_title("%s -- estimator comparison, world frame" % dataset,
                 color=th["ink"], fontsize=11, loc="left")
    ax.tick_params(colors=th["ink2"], labelsize=8)
    ax.grid(True, color=th["grid"], lw=0.5)
    for s in ax.spines.values():
        s.set_color(th["grid"])
    ax.legend(fontsize=8, facecolor=th["surface"], labelcolor=th["ink2"],
              edgecolor=th["grid"])
    fig.tight_layout()
    p = os.path.join(outdir, "proprio_trajectories_%s_%s.png"
                     % (dataset.split("/")[-1], theme))
    fig.savefig(p, dpi=160, facecolor=th["surface"])
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------

def write_csv(path, recs):
    cols = ["dataset", "name", "n", "secs", "anchored_rms", "anchored_max",
            "anchored_final", "yaw_rms_deg", "yaw_final_deg", "aligned_ate_rms",
            "aligned_ate_max", "anees", "anees_pos", "nees_samples", "verdict",
            "cov1", "cov2", "cov3", "sx_final", "sy_final", "syaw_final_deg"]
    anch = {"anchored_rms", "anchored_max", "anchored_final", "yaw_rms_deg",
            "yaw_final_deg"}
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in recs:
            cells = []
            for c in cols:
                if c in anch and not r.get("anchored", True):
                    cells.append("")
                    continue
                v = r.get(c, "")
                cells.append(("%.6g" % v) if isinstance(v, float) else str(v))
            f.write(",".join(cells) + "\n")


def write_tex(path, recs):
    """Chapter 4 table. booktabs, matching docs/report/minor_report conventions."""
    with open(path, "w") as f:
        f.write("% Generated by script/analyse_proprio.py -- do not edit by hand.\n")
        f.write("\\begin{table}[htbp]\n  \\centering\n")
        f.write("  \\caption{Non-visual baseline estimators on the "
                "\\texttt{dataset\\_allsensor} pair. Anchored error is the direct "
                "difference from ground truth with no alignment; aligned ATE uses the "
                "SE(3) Umeyama convention applied to the visual estimators elsewhere "
                "in this chapter. ANEES is the average normalised estimation error "
                "squared over $n$ decorrelated samples; a consistent filter yields "
                "3.0.}\n")
        f.write("  \\label{tab:proprio-baseline}\n")
        f.write("  \\begin{tabular}{llrrrrrl}\n    \\toprule\n")
        f.write("    Dataset & Estimator & \\multicolumn{1}{c}{Anch.\\ RMS} & "
                "\\multicolumn{1}{c}{Anch.\\ final} & \\multicolumn{1}{c}{Aligned ATE} "
                "& \\multicolumn{1}{c}{Yaw RMS} & \\multicolumn{1}{c}{ANEES} & "
                "Covariance \\\\\n")
        f.write("     & & \\multicolumn{1}{c}{[\\si{\\metre}]} & "
                "\\multicolumn{1}{c}{[\\si{\\metre}]} & "
                "\\multicolumn{1}{c}{[\\si{\\metre}]} & "
                "\\multicolumn{1}{c}{[\\si{\\degree}]} & \\multicolumn{1}{c}{[-]} & "
                "verdict \\\\\n    \\midrule\n")
        last = None
        for r in recs:
            ds = r["dataset"].split("/")[-1].replace("_", "\\_")
            show = "" if ds == last else "\\texttt{%s}" % ds
            last = ds
            ate = r.get("aligned_ate_rms")
            a = r.get("anchored", True)
            f.write("    %s & %s & %s & %s & %s & %s & %s & %s \\\\\n"
                    % (show, r["name"],
                       ("%.2f" % r["anchored_rms"]) if a else "--",
                       ("%.2f" % r["anchored_final"]) if a else "--",
                       ("%.2f" % ate) if isinstance(ate, float) else "--",
                       ("%.2f" % r["yaw_rms_deg"]) if a else "--",
                       ("%.1f" % r["anees"]) if np.isfinite(r.get("anees", np.nan))
                       else "--",
                       r.get("verdict", "--")))
        f.write("    \\bottomrule\n  \\end{tabular}\n\\end{table}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", action="append", required=True,
                    help="'group/name', e.g. simulation/dataset_allsensor_000. "
                         "Repeatable.")
    ap.add_argument("--experiment-id", default="exp-estimator-baseline")
    ap.add_argument("--domain", default="simulation")
    ap.add_argument("--no-vio", action="store_true",
                    help="baselines only; omit VINS-Fusion and ORB-SLAM3 even when "
                         "their runs exist for this dataset")
    ap.add_argument("--vins-run", default="logging_20260916_01",
                    help="VINS run directory to read. Pinned by name, not 'newest', "
                         "so adding a run never silently restates published numbers.")
    ap.add_argument("--orb-run", default="logging_20260916_01",
                    help="ORB-SLAM3 run directory with loop closure enabled.")
    ap.add_argument("--orb-nolc-run", default="logging_20260916_02",
                    help="ORB-SLAM3 run directory with loop closure disabled. Set to "
                         "'' to leave that variant out.")
    ap.add_argument("--nees-hz", type=float, default=1.0,
                    help="subsample rate for the chi-square consistency test. "
                         "Default 1 Hz -- see the module docstring on correlation.")
    args = ap.parse_args()
    # The baselines come first so they set the reading order in every table and
    # legend: the question this experiment asks is what the visual systems add ON TOP
    # of proprioception, which reads backwards if VIO is listed first.
    # Runs are pinned by name. ORB-SLAM3 appears twice when a loop-closure-free run
    # exists, because with loop closure its behaviour varies per dataset (0, 2 and 1
    # closures across the first three bags) and the two regimes are not one system.
    args.systems = list(vm.SYSTEMS_PROPRIO)
    if not args.no_vio:
        args.systems.append(("VINS-Fusion", "output_vins", args.vins_run))
        args.systems.append(("ORB-SLAM3", "output_orb", args.orb_run))
        if args.orb_nolc_run:
            args.systems.append(("ORB-SLAM3 (no LC)", "output_orb",
                                 args.orb_nolc_run))

    outdir = os.path.join(ROOT, "output", "compare", args.domain, args.experiment_id)
    os.makedirs(outdir, exist_ok=True)

    allrecs = []
    for ds in args.dataset:
        print("== %s" % ds)
        recs = analyse_one(ds, args)
        if not recs:
            continue
        # Aligned ATE from the shared implementation, so the baselines and the
        # visual estimators are measured by exactly the same code.
        res = vm.analyse(ds, systems=args.systems, logging="auto")
        byname, alby = {}, {}
        if res:
            byname = {row["name"]: row for row in res[0]}
            alby = {a["name"]: a for a in res[1]}
        for r in recs:
            row = byname.get(r["name"])
            if row:
                r["aligned_ate_rms"] = float(row["rms"])
                r["aligned_ate_max"] = float(row["max"])
            al = alby.get(r["name"])
            if al is not None:
                # Kept so the figures can draw the visual systems in the reference
                # frame. Without this they would be plotted in their own world frame,
                # which for ORB-SLAM3 is its first keyframe -- a trajectory that looks
                # wildly wrong for reasons that have nothing to do with its accuracy.
                r["al_t"], r["al_p"], r["al_err"] = al["t"], al["p"], al["err"]
            if r["anchored"]:
                print("   %-20s anchored rms %8.3f m  final %8.3f m  yaw rms %6.2f deg"
                      % (r["name"], r["anchored_rms"], r["anchored_final"],
                         r["yaw_rms_deg"]), end="")
            else:
                print("   %-20s anchored    --  (own world frame)    aligned ATE "
                      "%8.3f m" % (r["name"], r.get("aligned_ate_rms", float("nan"))),
                      end="")
            if "anees" in r:
                print("  ANEES %8.2f (%s, n=%d)"
                      % (r["anees"], r["verdict"], r["nees_samples"]), end="")
            print()
        for theme in ("light", "dark"):
            figure(recs, ds, outdir, theme)
            trajectory_figure(recs, ds, outdir, theme, args.systems)
        allrecs += recs

    if not allrecs:
        sys.exit("no baseline runs found -- has proprio_estimator.py been run?")
    write_csv(os.path.join(outdir, "proprio_metrics.csv"), allrecs)
    write_tex(os.path.join(outdir, "proprio_baseline_table.tex"), allrecs)
    print("\nwrote %s" % outdir)
    for f in sorted(os.listdir(outdir)):
        print("   %s" % f)


if __name__ == "__main__":
    main()
