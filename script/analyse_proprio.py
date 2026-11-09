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

A THIRD CONVENTION: RPE, WHICH NEEDS NEITHER
    Both conventions above are whole-trajectory statements, and both are therefore
    dominated by whatever the estimator did worst over the run. For a dead-reckoning
    baseline that is drift, by construction: wheel odometry can be locally excellent
    and still finish three metres out, and ATE cannot tell the two apart.

    RPE asks the local question instead -- over the next delta_s seconds, did the
    estimate move the way the truth moved -- and is immune to the global frame and to
    accumulated drift alike. It is what separates "this estimator is noisy" from
    "this estimator is precise but drifts", which is exactly the distinction the
    complementary-failure finding of section 4.2 rests on. Section 4.5.1 already
    reports it for the visual campaign; this module adds it here so the two sections
    use one definition.

    The implementation is vio_metrics.rpe(), unchanged, on the FULL SE(3) poses as
    written to vio.csv -- not on a planar lift. That choice is forced: section 4.2.7
    already reports RPE for the EKF noise-transfer ablation via
    analyse_noise_transfer.score_one(), which uses exactly this path, and a second
    convention here would make the same estimator carry two different RPE values in
    one section of the report. The proprioceptive estimators are planar by
    construction so the distinction costs them nothing; for VINS-Fusion and
    ORB-SLAM3 it correctly retains out-of-plane error, which is also what
    section 4.5.1 reports.
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
from map_underlay import draw_map

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
          # Deliberately a desaturated sibling of the EKF green: it is the SAME
          # filter, so it must not read as a different estimator.
          "EKF @ measured noise": "#2e7d5b",
          # Same hues viewer.py and plot_compare.py already use for these two, so a
          # reader moving between the overlay, the comparison figures and this one
          # does not have to re-learn which trail is which.
          # Two steps re-set on 21 Oct 2026 after the palette checks (OKLab /
          # Machado CVD, the dataviz validator ported to Python): the amber was
          # out of the lightness band at 1.7:1 on the light surface, and the two
          # purples sat at normal-vision Delta E 13.2 (floor 15). Same hue
          # families, nearest passing steps; identical to analyse_realistic_runs.py.
          "VINS-Fusion": "#cc8319",
          "ORB-SLAM3": "#e040fb",
          "ORB-SLAM3 (no LC)": "#a347b2"}


# Trees whose estimator starts FROM the ground-truth pose, which is the only case
# where an unaligned difference against ground truth means anything. VINS-Fusion and
# ORB-SLAM3 each build their own world frame -- ORB's is its first keyframe -- so the
# same subtraction would measure the arbitrary offset between two frames, produce a
# number in the tens of metres, and say nothing about either estimator. Their anchored
# columns are left empty rather than filled with that.
ANCHORED_TREES = {"output_wheel", "output_imu", "output_ekf",
                  "output_ekf_vslamcfg", "output_ekf_measured"}


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
    allsensor datasets but simply REPEAT 2 for the relogged ones, and on the
    realistic recordings the run named loopclosing_gpu on realistic_002 (6 Nov
    2026) has its closer disabled. Two signatures, both from the fork:
      before patch 7   the thread spun with nothing queued, so every row carried
                       current_keyframe_id = -1;
      patch 7 onward   rows are written per keyframe with duration_ms; a
                       disabled closer returns in well under a millisecond
                       (~0.4 ms), an active one takes several.
    The same rule as analyse_realistic_runs.loop_closer_state. A summary line
    reading loop_closures,0 proves nothing -- on allsensor_000 the enabled run
    found 90 candidates and rejected all 90.
    """
    import statistics
    path = os.path.join(run_dir, "loop_closing.csv")
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    if all(row.get("current_keyframe_id") == "-1" for row in rows):
        return "disabled"
    durs = []
    for row in rows:
        try:
            durs.append(float(row["duration_ms"]))
        except (KeyError, ValueError, TypeError):
            pass
    if not durs:
        return "active"
    return "disabled" if statistics.median(durs) < 1.0 else "active"

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
                      p=a[:, 1:4], q=a[:, 4:8],
                      yaw=np.unwrap(2.0 * np.arctan2(a[:, 7], a[:, 4])))
            break
    if gt is None:
        print("  [skip] %s: no ground_truth.csv under any baseline tree" % dataset)
        return out
    # The visual estimators report the IMU, 0.338 m ahead of the model origin the
    # ground truth reports (vio_metrics.IMU_LEVER_M); score them against the IMU
    # point. The baselines are referenced to the origin and keep the raw truth.
    p_imu = vm.gt_at_imu(gt["p"], gt["q"])
    gt_imu = dict(gt, p=p_imu, x=p_imu[:, 0], y=p_imu[:, 1])
    gt_origin = gt

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
        gt = gt_imu if vm.is_visual_tree(tree) else gt_origin

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

        # ---- RPE, no alignment of any kind ------------------------------------
        # Deliberately computed on the RAW estimate against the RAW truth. Aligning
        # first would be harmless (RPE is invariant to it) but would suggest the
        # metric depends on the fit, which is the misreading this column exists to
        # prevent.
        #
        # vm.load() rather than the local load_traj(): it returns the same (t, p, q)
        # that analyse_noise_transfer.score_one() feeds to vm.rpe for the ablation in
        # section 4.2.7. Same loader, same resampler, same rpe() call, so the EKF row
        # of this table and the "measured throughout" row of that one are the same
        # number computed the same way.
        # NOTE the separate t_r/m_r names. `t` and `m` above are consumed by the
        # covariance and NEES block below; rebinding either here would silently
        # rescore consistency against the wrong sample set.
        rec["rpe"] = {}
        raw = vm.load(os.path.join(d, "vio.csv"))
        if raw is not None:
            t_raw, p_raw, q_raw = raw[0], raw[1], raw[2]
            m_r = (t_raw >= lo) & (t_raw <= hi)
            if m_r.sum() >= 10:
                t_r = t_raw[m_r]
                p_est, q_est = p_raw[m_r], q_raw[m_r]
                p_ref = vm.resample(gt["t"], gt["p"], t_r)
                q_ref = vm.resample_quat(gt["t"], gt["q"], t_r)
                jump_idx = vm.find_jumps(t_r, p_est)[0]
                rec["jumps"] = int(len(jump_idx))
                for dl in vm.RPE_DELTAS:
                    r = vm.rpe(t_r, p_est, q_est, p_ref, q_ref, delta_s=dl,
                               jump_idx=jump_idx)
                    if r is None:
                        # No pair satisfied the interval -- a short or gappy log,
                        # not a zero. Left absent so the table prints a dash.
                        continue
                    rec["rpe"][dl] = r
        one = rec["rpe"].get(1.0)
        if one is not None:
            # Flat columns for the delta the report quotes in prose. The full sweep
            # stays in rpe_metrics.csv; this keeps proprio_metrics.csv readable as
            # one row per estimator.
            rec["rpe1_trans_rmse"] = float(one["trans_rmse"])
            rec["rpe1_rot_rmse"] = float(one["rot_rmse"])
            rec["rpe1_pairs"] = int(one["pairs"])
            for k_src, k_dst in (("trans_rmse_clean", "rpe1_trans_rmse_clean"),
                                 ("rot_rmse_clean", "rpe1_rot_rmse_clean")):
                if one.get(k_src) is not None:
                    rec[k_dst] = float(one[k_src])

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


def _draw_trajectory(ax, recs, dataset, theme, systems, scale=1.0, title=None,
                     legend=True):
    """Draw one dataset's estimator comparison into `ax`.

    Shared by the single-dataset figure and the all-datasets grid so the two can
    never disagree about what is drawn. `scale` enlarges fonts for the grid, whose
    panels are reproduced at about half this size on the page.
    """
    th = THEMES[theme]
    ax.set_facecolor(th["surface"])
    draw_map(ax, dataset)   # warehouse footprints, 50 % transparent, world frame
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
    ax.set_xlabel("x [m]", color=th["ink2"], fontsize=10 * scale)
    ax.set_ylabel("y [m]", color=th["ink2"], fontsize=10 * scale)
    ax.set_title(title if title is not None else
                 "%s -- estimator comparison, world frame" % dataset,
                 color=th["ink"], fontsize=11 * scale, loc="left")
    ax.tick_params(colors=th["ink2"], labelsize=8 * scale)
    ax.grid(True, color=th["grid"], lw=0.5)
    for s_ in ax.spines.values():
        s_.set_color(th["grid"])
    if legend:
        ax.legend(fontsize=8 * scale, facecolor=th["surface"], labelcolor=th["ink2"],
                  edgecolor=th["grid"])


def trajectory_figure(recs, dataset, outdir, theme, systems):
    th = THEMES[theme]
    fig, ax = plt.subplots(figsize=(8, 8), facecolor=th["surface"])
    _draw_trajectory(ax, recs, dataset, theme, systems)
    fig.tight_layout()
    p = os.path.join(outdir, "proprio_trajectories_%s_%s.png"
                     % (dataset.split("/")[-1], theme))
    fig.savefig(p, dpi=160, facecolor=th["surface"])
    plt.close(fig)
    return p


def r_anchored(by_dataset, name):
    """Whether estimator `name` is drawn as estimated (anchored) or aligned."""
    for recs in by_dataset.values():
        for r in recs:
            if r["name"] == name:
                return bool(r["anchored"])
    return False


def trajectory_grid(by_dataset, outdir, theme, systems):
    """Every dataset's comparison on one sheet, two columns, for the report.

    The single-dataset figures stay as they were; this only places the same
    drawings side by side so the report can show every recording rather than
    one example. Fonts are scaled up because the sheet is printed at about half
    the per-panel size of the single figure.
    """
    th = THEMES[theme]
    names = list(by_dataset)
    cols = 2
    rows_n = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(5.6 * cols, 5.8 * rows_n),
                             facecolor=th["surface"], squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for k, ds in enumerate(names):
        ax = axes[k // cols, k % cols]
        ax.set_visible(True)
        # No per-panel legend: seven entries at this font size cover the paths.
        # Identities go in one shared legend in the spare cell; the per-run
        # numbers the single-figure legends carry are in the tables.
        _draw_trajectory(ax, by_dataset[ds], ds, theme, systems, scale=1.35,
                         title=ds.split("/")[-1].replace("dataset_", ""), legend=False)
    from matplotlib.lines import Line2D
    order = ["ground truth"] + [vm.split(e)[0] for e in systems]
    handles = [Line2D([], [], color="#8a8983" if n == "ground truth" else COLOUR.get(n, "#999"),
                      lw=2.4 if n == "ground truth" else 1.8) for n in order]
    labels = [n if n == "ground truth" else
              (n + ("" if r_anchored(by_dataset, n) else ", SE(3)-aligned")) for n in order]
    spare = axes.ravel()[len(names)] if len(names) < rows_n * cols else None
    if spare is not None:
        spare.set_visible(True); spare.axis("off")
        spare.legend(handles, labels, loc="center", fontsize=13, frameon=True,
                     facecolor=th["surface"], edgecolor=th["grid"], labelcolor=th["ink2"],
                     title="estimators", title_fontsize=13)
    else:
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=12, frameon=False,
                   labelcolor=th["ink2"])
    # leave the bottom strip to the shared legend when every cell holds a panel
    fig.tight_layout(rect=(0, 0 if spare is not None else 0.07, 1, 1))
    p = os.path.join(outdir, "proprio_trajectories_grid_%s.png" % theme)
    fig.savefig(p, dpi=160, facecolor=th["surface"])
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------

def write_csv(path, recs):
    cols = ["dataset", "name", "n", "secs", "anchored_rms", "anchored_max",
            "anchored_final", "yaw_rms_deg", "yaw_final_deg", "aligned_rot_rms",
            "aligned_ate_rms",
            "aligned_ate_max", "anees", "anees_pos", "nees_samples", "verdict",
            "cov1", "cov2", "cov3", "sx_final", "sy_final", "syaw_final_deg",
            "jumps", "rpe1_pairs", "rpe1_trans_rmse", "rpe1_trans_rmse_clean",
            "rpe1_rot_rmse", "rpe1_rot_rmse_clean"]
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


def write_rpe_csv(path, recs):
    """The full interval sweep, one row per (dataset, estimator, delta).

    Kept separate from proprio_metrics.csv because it is the only product of this
    module that is not one row per estimator. Section 9 of the evaluation protocol
    requires the numbers behind every figure and table to be saved; this is that
    file for the RPE table.
    """
    cols = ["dataset", "name", "delta_s", "pairs", "jumps",
            "trans_rmse", "trans_median", "rot_rmse", "rot_median",
            "trans_rmse_clean", "rot_rmse_clean"]
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in recs:
            for dl in sorted(r.get("rpe", {})):
                d = r["rpe"][dl]
                row = dict(dataset=r["dataset"], name=r["name"], delta_s=dl,
                           jumps=r.get("jumps", ""), **{k: d.get(k) for k in
                           ("pairs", "trans_rmse", "trans_median", "rot_rmse",
                            "rot_median", "trans_rmse_clean", "rot_rmse_clean")})
                cells = []
                for c in cols:
                    v = row.get(c)
                    cells.append("" if v is None else
                                 ("%.6g" % v) if isinstance(v, float) else str(v))
                f.write(",".join(cells) + "\n")


def write_rpe_tex(path, recs, delta=1.0):
    """Chapter 4 RPE table at one interval, all estimators, all datasets."""
    order, seen = [], set()
    for r in recs:
        if r["name"] not in seen:
            seen.add(r["name"]); order.append(r["name"])
    datasets = []
    for r in recs:
        if r["dataset"] not in datasets:
            datasets.append(r["dataset"])
    by = {(r["dataset"], r["name"]): r for r in recs}

    def cell(r, key):
        if r is None:
            return "---"
        d = r.get("rpe", {}).get(delta)
        if d is None or d.get(key) is None:
            return "---"
        return "%.3f" % d[key]

    with open(path, "w") as f:
        w = f.write
        w("% Generated by script/analyse_proprio.py -- do not edit by hand.\n")
        w("\\begin{table}[htbp]\n\\centering\n")
        w("\\caption{Relative pose error at $\\Delta=\\SI{%g}{\\second}$ on the five "
          "\\repo{allsensor} bags. RPE uses \\emph{no} alignment, so unlike aligned "
          "ATE it is unaffected by the rigid fit and by accumulated drift; it "
          "measures only whether the estimate moved as the truth moved over the "
          "next \\SI{%g}{\\second}. ``Clean'' excludes pairs that span a pose "
          "discontinuity (a step implying more than \\SI{10}{\\metre\\per\\second}), "
          "so the difference between the two columns isolates the jump "
          "contribution. Poses are full $SE(3)$, scored by the same "
          "\\repo{vio_metrics.rpe} call used for the visual campaign "
          "(\\cref{sec:rpe}) and for the noise-transfer ablation of "
          "\\cref{sec:proprio-tuning}, so the three sets of figures are directly "
          "comparable. Lower is better throughout.}\n" % (delta, delta))
        w("\\label{tab:proprio-rpe}\n\\small\n")
        w("\\resizebox{\\textwidth}{!}{%\n")
        w("\\begin{tabular}{ll rrrrr}\n\\toprule\n")
        w("Dataset & Estimator & Jumps & \\multicolumn{2}{c}{Translation [\\si{\\metre}]}"
          " & \\multicolumn{2}{c}{Rotation [\\si{\\degree}]} \\\\\n")
        w("\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n")
        w(" & & & all pairs & clean & all pairs & clean \\\\\n\\midrule\n")
        for i, ds in enumerate(datasets):
            if i:
                w("\\midrule\n")
            short = ds.split("/")[-1].replace("dataset_", "")
            for j, nm in enumerate(order):
                r = by.get((ds, nm))
                w("%s & %s & %s & %s & %s & %s & %s \\\\\n" % (
                    ("\\repo{%s}" % short) if j == 0 else "",
                    nm.replace("&", "\\&"),
                    "---" if r is None else str(r.get("jumps", "---")),
                    cell(r, "trans_rmse"), cell(r, "trans_rmse_clean"),
                    cell(r, "rot_rmse"), cell(r, "rot_rmse_clean")))
        w("\\bottomrule\n\\end{tabular}}\n\\end{table}\n")


def write_tex(path, recs):
    """Chapter 4 table. booktabs, matching docs/report/minor_report conventions."""
    with open(path, "w") as f:
        f.write("% Generated by script/analyse_proprio.py -- do not edit by hand.\n")
        f.write("\\begin{table}[htbp]\n  \\centering\n")
        f.write("  \\caption{Estimators on the datasets listed. "
                "Anchored error is the direct difference from ground truth "
                "with no alignment; aligned ATE uses the SE(3) Umeyama convention "
                "applied to the visual estimators elsewhere in this chapter. ANEES "
                "is the average normalised estimation error squared over $n$ "
                "decorrelated samples; a consistent filter yields 3.0. "
                "\\emph{EKF @ VSLAM noise} is the same filter, on the same bags, "
                "with only its four IMU noise terms replaced by the values "
                "\\repo{config/wil_sim/stereo_imu.yaml} declares --- the "
                "config every VINS-Fusion and ORB-SLAM3 run in this table actually "
                "used. It is included because the EKF's noise model is measured "
                "per bag from the data while the visual systems read a hand-set "
                "file, so without it tuning is an uncontrolled variable between the "
                "two families and the comparison is not like for like.}\n")
        f.write("  \\label{tab:proprio-baseline}\n")
        f.write("  \\begin{tabular}{llrrrrrrl}\n    \\toprule\n")
        f.write("    Dataset & Estimator & \\multicolumn{1}{c}{Anch.\\ RMS} & "
                "\\multicolumn{1}{c}{Anch.\\ final} & \\multicolumn{1}{c}{Aligned ATE} "
                "& \\multicolumn{1}{c}{Aligned rot.} "
                "& \\multicolumn{1}{c}{Anch.\\ yaw} & \\multicolumn{1}{c}{ANEES} & "
                "Covariance \\\\\n")
        f.write("     & & \\multicolumn{1}{c}{[\\si{\\metre}]} & "
                "\\multicolumn{1}{c}{[\\si{\\metre}]} & "
                "\\multicolumn{1}{c}{[\\si{\\metre}]} & "
                "\\multicolumn{1}{c}{[\\si{\\degree}]} & "
                "\\multicolumn{1}{c}{[\\si{\\degree}]} & \\multicolumn{1}{c}{[-]} & "
                "verdict \\\\\n    \\midrule\n")
        last = None
        for r in recs:
            ds = r["dataset"].split("/")[-1].replace("_", "\\_")
            show = "" if ds == last else "\\texttt{%s}" % ds
            last = ds
            ate = r.get("aligned_ate_rms")
            a = r.get("anchored", True)
            rot = r.get("aligned_rot_rms")
            f.write("    %s & %s & %s & %s & %s & %s & %s & %s & %s \\\\\n"
                    % (show, r["name"],
                       ("%.2f" % r["anchored_rms"]) if a else "--",
                       ("%.2f" % r["anchored_final"]) if a else "--",
                       ("%.2f" % ate) if isinstance(ate, float) else "--",
                       ("%.2f" % rot) if isinstance(rot, float) else "--",
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
    ap.add_argument("--ekf-diagnostic", default="",
                    help="tree of a second EKF to list right after the primary one, "
                         "e.g. output_ekf_measured (since 6 Nov 2026 the primary EKF in "
                         "output_ekf runs on the VSLAM configs' IMU noise, so the "
                         "measured-noise filter is the diagnostic, not the other way round).")
    ap.add_argument("--no-ekf-vslamcfg", action="store_true",
                    help="omit the EKF re-run at the VSLAM systems' own IMU noise "
                         "config. That row exists because the EKF's noise model is "
                         "MEASURED per bag while VINS and ORB read a hand-set "
                         "config, which makes tuning an uncontrolled variable "
                         "between the two families; re-running the filter on the "
                         "config the visual systems actually used is the half of "
                         "the control that needs no new logging.")
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
    # Immediately after the EKF, so the two tunings of the same filter are adjacent
    # and the comparison is read before the visual systems are introduced.
    if args.ekf_diagnostic:
        args.systems.append(("EKF @ measured noise", args.ekf_diagnostic))
    elif not args.no_ekf_vslamcfg and os.path.isdir(os.path.join(ROOT, "output", "output_ekf_vslamcfg")):
        args.systems.append(("EKF @ VSLAM noise", "output_ekf_vslamcfg"))
    if not args.no_vio:
        args.systems.append(("VINS-Fusion", "output_vins", args.vins_run))
        args.systems.append(("ORB-SLAM3", "output_orb", args.orb_run))
        if args.orb_nolc_run:
            args.systems.append(("ORB-SLAM3 (no LC)", "output_orb",
                                 args.orb_nolc_run))

    outdir = os.path.join(ROOT, "output", "compare", args.domain, args.experiment_id)
    os.makedirs(outdir, exist_ok=True)

    allrecs = []
    by_dataset = {}
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
                # Aligned rotational error. Unlike the anchored yaw column this is
                # defined for EVERY estimator: the same rigid SE(3) fit that makes
                # aligned ATE comparable also resolves the visual estimators' own
                # world-frame orientation, so their orientations become comparable
                # too. The anchored yaw column stays baseline-only.
                r["aligned_rot_rms"] = float(row["rot_rms"])
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
            if "rpe1_trans_rmse" in r:
                print("  RPE@1s %6.3f m / %5.3f deg (%d pairs, %d jumps)"
                      % (r["rpe1_trans_rmse"], r["rpe1_rot_rmse"],
                         r["rpe1_pairs"], r.get("jumps", 0)), end="")
            print()
        for theme in ("light", "dark"):
            figure(recs, ds, outdir, theme)
            trajectory_figure(recs, ds, outdir, theme, args.systems)
        allrecs += recs
        by_dataset[ds] = recs

    if not allrecs:
        sys.exit("no baseline runs found -- has proprio_estimator.py been run?")
    if len(by_dataset) > 1:
        for theme in ("light", "dark"):
            trajectory_grid(by_dataset, outdir, theme, args.systems)
    write_csv(os.path.join(outdir, "proprio_metrics.csv"), allrecs)
    write_tex(os.path.join(outdir, "proprio_baseline_table.tex"), allrecs)
    write_rpe_csv(os.path.join(outdir, "proprio_rpe_metrics.csv"), allrecs)
    write_rpe_tex(os.path.join(outdir, "proprio_rpe_table.tex"), allrecs)
    print("\nwrote %s" % outdir)
    for f in sorted(os.listdir(outdir)):
        print("   %s" % f)


if __name__ == "__main__":
    main()
