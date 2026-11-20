#!/usr/bin/env python3
"""Check the ORB-SLAM3 logs against the published method's own assumptions.

    python3 script/exp1_paper_assumptions.py

WHY THIS EXISTS
  Experiment 1 reports that ORB-SLAM3's 13-19 deg rotation column is a frame
  convention and that realistic_003 drifts vertically.  Both are descriptions.
  This script asks whether they follow from what the method SAYS it does, so
  that the chapter can name a mechanism instead of a correlation.

  The reference is Campos et al., 2021, "ORB-SLAM3: An Accurate Open-Source
  Library for Visual, Visual-Inertial and Multi-Map SLAM", IEEE Transactions
  on Robotics, Vol. 37, No. 6, pp. 1874-1890 -- section V.B, IMU
  Initialization, equations 5, 8 and 9 -- read against
  thirdparty/ORB_SLAM3/src/LocalMapping.cc as built here.

WHAT IS CHECKED, AND WHAT EACH CHECK CAN ESTABLISH

  1 THE RECIPE.  Eq. 5 estimates Yk = {s, Rwg, b, v_0:k} from k = 10 keyframes
    after about 2 s of visual-only SLAM.  LocalMapping.cc sets nMinKF = 10 for
    both modes and minTime = 1.0 s for stereo.  The logs carry the keyframe
    count at the instant imu_initialized flips, so the recipe is checkable
    directly.  A match establishes that this build runs the published
    procedure; it establishes nothing about accuracy.

  2 THE MAP ROTATION.  On success InitializeIMU calls ApplyScaledRotation with
    Twg = Rwg^T, rotating the whole map into the estimated gravity frame.  In
    the published stream that is a step.  Measuring it identifies the reported
    frame switch with a specific line of the method, and its SIZE is the angle
    between the camera optical frame and gravity -- a fact about the mounting.

  3 THE BIAS PRIOR.  Eq. 8 minimises the preintegration residuals together
    with ||b||^2 under Sigma_b, a prior pulling the biases to zero.  VIBA 1
    (mTinit > 5 s) keeps it, priorA = 1e5; VIBA 2 (mTinit > 15 s) sets both
    priors to zero and so removes the term.  Each re-estimate rotates the map
    again, so the step at each is a direct measurement of what dropping the
    prior does.  This is a within-run paired comparison and is NOT confounded
    by route.

  4 OBSERVABILITY.  Eq. 9 updates Rwg with two angles about the horizontal
    axes; accelerometer bias and gravity direction separate only under
    rotation, and a floor robot rotates about one axis.  The method states the
    failure: slow motion may leave the inertial parameters poorly observable
    and initialisation unconverged within fifteen seconds.  The test is the
    yaw the ground truth sweeps between initialisation and VIBA 2.
    *** This check is CONFOUNDED: on the four recordings available, yaw in
    that window is a property of the layout (U turns inside it, I does not),
    so it is two routes against two, not sixteen runs against each other.
    It is reported as consistent-with, never as established. ***

  5 THE REMEDY IS OUT OF REACH.  The scale-and-gravity refinement the method
    offers for case 4 is gated `if (mbMonocular)` in LocalMapping.cc, so on
    the stereo-inertial path the gravity direction gets no estimate after
    VIBA 2.  This is a code fact, not a measurement.

  6 WHAT THE TILT COSTS.  A gravity error theta tilts the estimator's
    horizontal plane, so vertical error should grow as d*sin(theta) with
    distance.  Fitting that line per run says where the mechanism actually
    operates.  It does NOT hold everywhere, and the script prints the runs
    where it fails so the two are not pooled.

  7 SCALE.  Stereo-inertial fixes s = 1 and removes it from Eq. 8, so no
    inertial path corrects a scale error.  Path length against the truth's
    LOOKS like one: analyse() reports every ORB-SLAM3 run 1.7-5.4 % long.
    It is not one.  exp1_path_scale.py shows the excess is the arc length of
    the pre-initialisation spike plus frame-to-frame jitter; sampled every
    second the local scale is 0.997-0.999.  This section prints the reported
    figure only to document the artefact; the chapter no longer quotes it as
    a scale bias.

Prints a table per check and the pooled statistic.  Writes nothing.
"""

import csv
import os
import sys
from itertools import combinations
from math import comb

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vio_metrics as vm                                       # noqa: E402

BAGS = ["dataset_allsensor_realistic_%03d" % i for i in range(4)]
ARMS = [("ORB-SLAM3", "loopclosing_cpu"),
        ("ORB-SLAM3 GPU", "loopclosing_gpu"),
        ("ORB-SLAM3 (no LC)", "noloopclosing_cpu"),
        ("ORB-SLAM3 GPU (no LC)", "noloopclosing_gpu")]
ORB = os.path.join(str(vm.ROOT), "output", "output_orb", "simulation")
# the two loop-closing arms of realistic_001 accept a false 14 m closure; they
# are excluded from the scale check, where that correction is the whole signal
FALSE_CLOSURE = {("001", "ORB-SLAM3"), ("001", "ORB-SLAM3 GPU")}


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def milestones(run_dir):
    """mTinit at IMU initialisation, VIBA 1 and VIBA 2, and the keyframe count.

    mTinit is counted by LocalMapping.cc from the map's FIRST keyframe, not
    from initialisation, which is why these do not land on the five and
    fifteen seconds after initialisation that the method's text gives."""
    path = os.path.join(run_dir, "local_mapping.csv")
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    t0 = int(rows[0]["timestamp_ns"]) / 1e9
    out = {"t_abs0": t0}
    for col in ("imu_initialized", "inertial_ba1", "inertial_ba2"):
        out[col] = (None, None)
        for r in rows:
            if r.get(col) in ("1", "True", "true"):
                out[col] = (int(r["timestamp_ns"]) / 1e9 - t0,
                            int(r["keyframes_in_map"]))
                break
    return out


def pose_step(run_dir, t_abs, half=0.6):
    """Largest single-sample rotation of the PUBLISHED pose within +-half s.

    ApplyScaledRotation rotates the map by the newly estimated Rwg, so a
    revision of the gravity direction shows up here.  Read from the pose
    stream alone: no ground truth enters."""
    if t_abs is None:
        return float("nan")
    rec = vm.load(os.path.join(run_dir, "vio.csv"))
    if rec is None or len(rec[0]) < 3:
        return float("nan")
    t, q = rec[0], rec[2]
    R = vm.quat_to_R(q)
    step = vm.geodesic_deg(R[:-1], R[1:])
    m = np.abs(t[1:] - t_abs) <= half
    return float(step[m].max()) if m.any() else float("nan")


def yaw_swept(gt, t_lo, t_hi):
    """Total |yaw| the ground truth sweeps in an absolute-time window, in deg."""
    t, q = gt[0], gt[2]
    m = (t >= t_lo) & (t <= t_hi)
    if m.sum() < 3:
        return float("nan"), 0
    y = np.unwrap(np.radians(vm.euler_zyx(vm.quat_to_R(q))[m, 0]))
    return float(np.degrees(np.abs(np.diff(y)).sum())), int(m.sum())


def first_yaw(gt, thresh_deg=5.0):
    """Seconds from the recording's start to the first turn worth the name."""
    t, q = gt[0], gt[2]
    y = np.degrees(np.unwrap(np.radians(vm.euler_zyx(vm.quat_to_R(q))[:, 0])))
    hit = np.where(np.abs(y - y[0]) > thresh_deg)[0]
    return float(t[hit[0]] - t[0]) if len(hit) else float("nan")


def exact_mannwhitney(a, b):
    """One-sided p for `a stochastically smaller than b`, by enumeration."""
    obs = sum(1 for x in a for y in b if x < y) + \
        0.5 * sum(1 for x in a for y in b if x == y)
    pool = np.concatenate([a, b])
    hit = tot = 0
    for idx in combinations(range(len(pool)), len(a)):
        aa = pool[list(idx)]
        bb = np.delete(pool, list(idx))
        u = sum(1 for x in aa for y in bb if x < y) + \
            0.5 * sum(1 for x in aa for y in bb if x == y)
        tot += 1
        hit += (u >= obs)
    return obs, hit / tot


def collect():
    rows = []
    for ds in BAGS:
        gt = vm.load_gt(os.path.join(vm.gt_candidates(ds)[0]
                                     if hasattr(vm, "gt_candidates") else "", ""))
        rows.append(ds)
    return rows


def main():
    recs = []
    for ds in BAGS:
        gt = None
        for cand in vm.gt_candidates(ds):
            if os.path.exists(cand):
                gt = vm.load_gt(cand)
                break
        if gt is None:
            print("  no ground truth for", ds)
            continue
        t_first = first_yaw(gt)
        for name, run in ARMS:
            d = os.path.join(ORB, ds, run)
            if not os.path.exists(os.path.join(d, "vio.csv")):
                continue
            ms = milestones(d)
            res = vm.analyse("simulation/" + ds,
                             systems=[(name, "output_orb", run)])
            if ms is None or res is None or not res[0]:
                continue
            row, al = res[0][0], res[1][0]

            t_abs = vm.world_frame_time(d)
            keep = (np.ones_like(al["t"], bool) if t_abs is None
                    else al["t"] >= t_abs - al["t0"])
            if keep.sum() < 30:
                keep = np.ones_like(al["t"], bool)
            de = wrap180(al["eul"] - al["refeul"])
            tilt = float(np.sqrt((de[keep, 1] ** 2 + de[keep, 2] ** 2).mean()))

            a0 = ms["t_abs0"]
            ti, kf = ms["imu_initialized"]
            b1 = ms["inertial_ba1"][0]
            b2 = ms["inertial_ba2"][0]
            yaw, nyaw = (yaw_swept(gt, a0 + ti, a0 + b2)
                         if (ti is not None and b2 is not None) else (float("nan"), 0))

            ref = al["ref"]
            dist = np.concatenate([[0.0], np.cumsum(
                np.linalg.norm(np.diff(ref, axis=0), axis=1))])
            z = (al["p"] - ref)[:, 2]
            x, y = dist[keep] - dist[keep][0], z[keep]
            slope = float(np.polyfit(x, y, 1)[0])
            recs.append(dict(
                ds=ds[-3:], arm=name, ti=ti, kf=kf, b1=b1, b2=b2,
                step_i=pose_step(d, a0 + ti if ti is not None else None),
                step_1=pose_step(d, a0 + b1 if b1 is not None else None),
                step_2=pose_step(d, a0 + b2 if b2 is not None else None),
                yaw=yaw, nyaw=nyaw, t_first=t_first, tilt=tilt,
                r_zd=float(np.corrcoef(x, y)[0, 1]),
                th_ramp=float(np.degrees(np.arcsin(np.clip(abs(slope), 0, 1)))),
                vert=float(np.sqrt((y ** 2).mean())),
                scale=100.0 * (row["path"] / row["ref"] - 1.0),
                along=float(np.sqrt((al["p"] - ref)[keep][:, 0].std() ** 2)) * 0 +
                float(np.sqrt(((al["p"] - ref)[keep] ** 2).sum(1).mean())) * 0,
                ate=row["rms"]))
    if not recs:
        print("no runs found"); return

    tr = {}
    for r in csv.DictReader(open(os.path.join(
            str(vm.ROOT), "output/compare/simulation/exp1-error-pattern/translation.csv"))):
        tr[(r["dataset"][-3:], r["name"])] = float(r["along_rms"])
    for r in recs:
        r["along"] = tr.get((r["ds"], r["arm"]), float("nan"))

    print("\n== 1-3  the recipe, the map rotation, and the bias prior")
    print("%-5s %-22s %6s %4s %7s %7s %8s %8s %8s" % (
        "bag", "arm", "tInit", "KFs", "mT_BA1", "mT_BA2",
        "step_ini", "step_BA1", "step_BA2"))
    for r in recs:
        print("%-5s %-22s %6.2f %4d %7.2f %7.2f %8.3f %8.3f %8.3f" % (
            r["ds"], r["arm"], r["ti"], r["kf"], r["b1"], r["b2"],
            r["step_i"], r["step_1"], r["step_2"]))
    kf = np.array([r["kf"] for r in recs])
    ti = np.array([r["ti"] for r in recs])
    si = np.array([r["step_i"] for r in recs])
    s1 = np.array([r["step_1"] for r in recs])
    s2 = np.array([r["step_2"] for r in recs])
    b1 = np.array([r["b1"] for r in recs])
    b2 = np.array([r["b2"] for r in recs])
    n = len(recs)
    print("  Eq.5 recipe: keyframes at initialisation = %s in %d of %d runs; "
          "t_init %.2f-%.2f s" % (set(kf.tolist()), (kf == 10).sum(), n,
                                  ti.min(), ti.max()))
    print("  map rotation at initialisation: %.2f-%.2f deg (camera optical "
          "frame to gravity)" % (si.min(), si.max()))
    print("  VIBA fires at mTinit %.2f-%.2f and %.2f-%.2f s, i.e. %.1f-%.1f and "
          "%.1f-%.1f s AFTER initialisation" % (
              b1.min(), b1.max(), b2.min(), b2.max(),
              (b1 - ti).min(), (b1 - ti).max(), (b2 - ti).min(), (b2 - ti).max()))
    k = int((s2 > s1).sum())
    p = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    print("  Eq.8 prior KEPT  (VIBA 1, priorA=1e5): %.3f-%.3f deg, median %.3f"
          % (s1.min(), s1.max(), np.median(s1)))
    print("  Eq.8 prior GONE  (VIBA 2, priors 0,0): %.3f-%.3f deg, median %.3f"
          % (s2.min(), s2.max(), np.median(s2)))
    print("  larger at VIBA 2 in %d of %d runs, one-sided sign test p = %.6f"
          % (k, n, p))

    print("\n== 4  observability: yaw swept between initialisation and VIBA 2")
    print("%-5s %8s %6s %10s %12s" % ("bag", "yaw[deg]", "n_gt", "1st>5deg", "tilt[deg]"))
    seen = set()
    for r in recs:
        if r["ds"] in seen:
            continue
        seen.add(r["ds"])
        t = [q["tilt"] for q in recs if q["ds"] == r["ds"]]
        print("%-5s %8.2f %6d %9.2fs  %5.3f-%5.3f"
              % (r["ds"], r["yaw"], r["nyaw"], r["t_first"], min(t), max(t)))
    exc = np.array([r["yaw"] > 1.0 for r in recs])
    tilt = np.array([r["tilt"] for r in recs])
    u, pm = exact_mannwhitney(tilt[exc], tilt[~exc])
    print("  excited  (n=%d): tilt %.3f-%.3f deg" % (exc.sum(), tilt[exc].min(), tilt[exc].max()))
    print("  straight (n=%d): tilt %.3f-%.3f deg" % ((~exc).sum(), tilt[~exc].min(), tilt[~exc].max()))
    print("  exact one-sided Mann-Whitney U=%g, p=%.5f" % (u, pm))
    print("  *** CONFOUNDED with layout: two routes against two.  Reportable as")
    print("      consistent-with only; see the script header. ***")

    print("\n== 6  does the tilt convert into vertical drift?  (per run)")
    print("%-5s %-22s %7s %8s %8s %8s" % ("bag", "arm", "r(d,z)", "th_ramp", "tilt", "vert[m]"))
    for r in recs:
        flag = "  <- line fits" if abs(r["r_zd"]) > 0.9 else ""
        print("%-5s %-22s %7.3f %8.3f %8.3f %8.3f%s"
              % (r["ds"], r["arm"], r["r_zd"], r["th_ramp"], r["tilt"], r["vert"], flag))
    fit = [r for r in recs if abs(r["r_zd"]) > 0.9]
    print("  the straight line fits in %d of %d runs; pooling the rest with them "
          "would be unsound" % (len(fit), n))

    print("\n== 7  scale: s is fixed to 1 and taken out of Eq. 8 for stereo")
    keep = [r for r in recs if (r["ds"], r["arm"]) not in FALSE_CLOSURE]
    sc = np.array([r["scale"] for r in keep])
    print("  analyse() path-length excess %+.2f to %+.2f %% in %d runs -- an ARTEFACT of the "
          "initialisation spike and jitter, not a scale bias; see exp1_path_scale.py "
          "(local scale 0.997-0.999)" % (sc.min(), sc.max(), len(sc)))


if __name__ == "__main__":
    main()
