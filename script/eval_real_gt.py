#!/usr/bin/env python3
"""Score real-robot VINS-Fusion / ORB-SLAM3 runs against the 2-D tracker ground truth.

Experiment 1, real-robot slot (chapter 4.1.2): do the visual estimators track the
robot's true path on the Gensurv floor? Ground truth is the optical tracker export
in dataset/gt/<dataset> (X, Y in centimetres at roughly 7 Hz, time relative to the
tracker's own start). The estimates are the vio.csv files the two nodes wrote while
replaying the dataset_realv2_* bags.

Three properties of this data decide the method, and each is handled explicitly
rather than assumed away:

TIME BASE. The robot's clock is not synchronised to anything (the bags are stamped
    14 September 2026; the tracker export says 1 October 2026), and the tracker's
    clock starts at zero when the operator presses record. There is no shared
    timestamp. The offset between the two is ESTIMATED by cross-correlating speed
    profiles: the tracker's speed (from its own positions) against the robot's wheel
    speed feedback (/feedback/velocity in the bag), which is independent of either
    estimator, so both estimators are scored on the same ground-truth timeline. The
    same correlation is repeated against each estimator's own speed as a cross-check,
    and the ATE is re-evaluated at +-0.1 s and +-0.2 s so the reader can see how much
    the number depends on the estimated offset. --time-offset pins a value instead.

TWO DIMENSIONS, NO HEADING. The tracker reports planar position only. Alignment is
    therefore rigid SE(2) -- yaw plus translation, scale fixed at one, which is the
    planar case of the SE(3) Umeyama fit used everywhere else in this project
    (vio_metrics.umeyama on points with z set to zero). Vertical error is not
    evaluated. Rotational RPE cannot be computed; translational RPE is computed on
    relative displacements in the globally aligned frame, i.e. the TUM definition
    with the reference orientation taken as identity, because the reference has none.
    A Sim(2) scale is fitted as a DIAGNOSTIC only and never applied: stereo-inertial
    estimates are metric.

LEVER ARM. The tracker follows a marker on the robot; the estimators report the pose
    of the IMU/camera origin. Unless the marker sits on that origin the two paths
    differ by a body-fixed offset that shows up as error during turns. Pass the
    measured marker offset with --lever-arm FORWARD,LEFT (metres, robot frame). The
    body axes of each vio.csv are detected from the data (the axis along which the
    body moves is "forward"), printed, and recorded. With no --lever-arm the offset is
    zero and every headline number is labelled provisional. --fit-lever-arm estimates
    the offset by least squares and reports it as a diagnostic; it is never used for
    the headline because fitting it would absorb genuine estimator error.

Validity rules are fixed here, before any number is read:
    ground truth    >= 50 samples, no gap > 1.0 s, no step > 0.5 m, duration within
                    25 % of the bag's;
    time alignment  wheel-based correlation peak r >= 0.9 and at least 0.05 above
                    the next peak more than 1 s away; failing that, peak r >= 0.5 with
                    the estimator-based offsets agreeing within 0.5 s;
    a run is        completed  : graceful shutdown, no estimator resets, >= 80 % of
                                 ground-truth samples matched, no jump, max error
                                 < 50 m;
                    partial    : completed's checks fail only on coverage or on a
                                 tracking loss / reset that the run recovered from;
                    failed     : any reset storm, divergence (> 50 m), < 20 %
                                 coverage, or an implausible trajectory (vertical
                                 range > 1 m on a floor run; rule added 2026-10-20
                                 after realv3_000, see status_of);
                    provisional: appended to any status while --lever-arm is absent
                                 or the time alignment is uncertain.
Failed runs are scored and plotted for diagnosis but excluded from the headline
LaTeX table, which says so in its caption.

Frame repair (4 November 2026, off by default): --join-imu-init rotates the poses an
    ORB-SLAM3 run wrote before each IMU initialisation into the frame it used after
    it (the rotation is measured at the initialisation frame), and --level-plane
    rotates each map's fitted plane to the horizontal (side chosen by --body-up).
    Both are recorded in the manifest and the metrics rows (frame_repair,
    tilt_before_deg, tilt_after_deg, imu_init_fraction, reset times and reasons).

Outputs, under --out/<dataset>/ (default output/compare/real/exp-estimator-baseline):
    manifest.json             provenance: paths, run metadata, offsets, git commit
    ground_truth_robotclock.csv  tracker samples re-stamped onto the robot clock
    time_offset.csv/.png      correlation-versus-offset curves (wheel, each estimator)
    aligned_<system>.csv      matched samples: t, gt xy, aligned estimate xy, error
    metrics.csv / rpe_metrics.csv / metrics.json
    trajectory_<theme>.png    xy overlay          error_<theme>.png   error & speed vs t
    rpe_<theme>.png           RPE versus interval
and at --out/: real_gt_metrics.csv, real_gt_rpe_metrics.csv, real_gt_table.tex,
real_gt_rpe_table.tex.

Example:
    python3 script/eval_real_gt.py --dataset dataset_realv2_000 --dataset dataset_realv2_001 \
        --run logging_20260916_03 --lever-arm 0.00,0.00
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import sqlite3
import struct
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vio_metrics as vm  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = vm.ROOT

# Same hues the rest of the project uses for these two estimators (viewer.py,
# plot_compare.py, analyse_proprio.py), so a reader does not re-learn the legend.
SYSTEMS = {"vins": ("VINS-Fusion", "output_vins", "#ffb74d"),
           "orb": ("ORB-SLAM3", "output_orb", "#e040fb"),
           # Same tree as "orb", a different run: the loop closer switched off. The
           # simulation campaign reports both, so the real slot can too; the
           # purple matches analyse_proprio.py's "ORB-SLAM3 (no LC)".
           "orb_nolc": ("ORB-SLAM3 (no LC)", "output_orb", "#ab47bc"),
           "vins_yolo": ("VINS-Fusion+YOLO", "output_vins_yolo", "#f57c00"),
           "orb_yolo": ("ORB-SLAM3+YOLO", "output_orb_yolo", "#8e24aa")}
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#dedcd4",
                  gt="#0b0b0b", wheel="#4fc3f7"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", grid="#3a3a37",
                 gt="#ffffff", wheel="#4fc3f7"),
}

WHEEL_TOPIC = "/feedback/velocity"
MAX_ASSOC_GAP_S = 0.15      # a GT sample is matched only if an estimate sample lies this close
MOVING_SPEED = 0.10         # m/s; below this the robot is treated as stationary
OFFSET_SEARCH_S = (-5.0, 20.0)
OFFSET_STEP_S = 0.01
DIVERGED_M = vm.DIVERGED_ERROR
PLAUSIBLE_Z_RANGE_M = 1.0       # a floor run that climbs more than this has failed (added 2026-10-20)
PLAUSIBLE_PATH_RATIO = (0.5, 2.0)  # outside this the ratio is reported as a reason
RPE_DELTAS = vm.RPE_DELTAS
RPE_TOL = 0.25


# --------------------------------------------------------------------------- inputs

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_kv_csv(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        next(f, None)
        for line in f:
            k, _, v = line.rstrip("\n").partition(",")
            out[k] = v
    return out


def load_gt(path):
    """Tracker export -> dict(t_rel [s], xy [m], n, checks)."""
    import csv
    t, x, y, stamps = [], [], [], []
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        for row in r:
            if len(row) < 4:
                continue
            stamps.append(row[0])
            t.append(float(row[1])); x.append(float(row[2])); y.append(float(row[3]))
    t, xy = np.array(t), np.stack([x, y], 1) / 100.0
    if not np.all(np.diff(t) > 0):
        order = np.argsort(t, kind="stable")
        t, xy = t[order], xy[order]
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    gaps = np.diff(t)
    return dict(path=os.path.abspath(path), header=header, t_rel=t, xy=xy, n=len(t),
                first_stamp=stamps[0] if stamps else "", last_stamp=stamps[-1] if stamps else "",
                duration=float(t[-1] - t[0]), gap_max=float(gaps.max()),
                gap_median=float(np.median(gaps)), step_max=float(step.max()),
                path_len=float(step.sum()))


def bag_db3(dataset_dir):
    import yaml
    meta = os.path.join(dataset_dir, "metadata.yaml")
    with open(meta) as f:
        info = yaml.safe_load(f)["rosbag2_bagfile_information"]
    rel = info["relative_file_paths"][0]
    return os.path.join(dataset_dir, rel), info


def load_wheel(db3, info):
    """/feedback/velocity (std_msgs/Float64) by receive time; CDR = 4-byte header + f64.

    Returned speed is |value| in whatever unit the firmware publishes. Only the SHAPE
    of the profile is used (Pearson correlation), so the unit does not matter here --
    and it should not be assumed to be m/s: on both realv2 bags the integral of this
    signal is about three times the tracker's path length.
    """
    con = sqlite3.connect(f"file:{db3}?mode=ro", uri=True)
    try:
        tops = {name: i for i, name in con.execute("select id,name from topics")}
        t0_ns = con.execute("select min(timestamp) from messages").fetchone()[0]
        t1_ns = con.execute("select max(timestamp) from messages").fetchone()[0]
        if WHEEL_TOPIC not in tops:
            return None, t0_ns, t1_ns
        rows = con.execute("select timestamp,data from messages where topic_id=? order by timestamp",
                           (tops[WHEEL_TOPIC],)).fetchall()
    finally:
        con.close()
    t = np.array([r[0] for r in rows], float) * 1e-9
    v = np.array([struct.unpack_from("<d", r[1], 4)[0] for r in rows])
    return dict(t=t, v=v, n=len(t)), t0_ns, t1_ns


# ------------------------------------------------------------------ time alignment

def resample_smooth(t, s, grid, smooth_s=0.25):
    """Linear resample of a signal onto `grid`, then a NaN-aware box filter.

    Outside the sampled span the result is NaN, not zero: zero would look like a
    genuine stop to the correlation.
    """
    out = np.interp(grid, t, s, left=np.nan, right=np.nan)
    k = max(1, int(round(smooth_s / np.median(np.diff(grid)))))
    if k > 1:
        pad = np.pad(out, (k // 2, k - 1 - k // 2), mode="edge")
        ker = np.ones(k) / k
        val = np.where(np.isnan(pad), 0.0, pad)
        cnt = np.convolve((~np.isnan(pad)).astype(float), ker, mode="valid")
        out = np.where(cnt > 0.5, np.convolve(val, ker, mode="valid") / np.maximum(cnt, 1e-9), np.nan)
    return out


def speed_profile(t, xy, grid, smooth_s=0.25):
    """|velocity| of a sampled planar path, resampled onto `grid` and box-smoothed.

    Speed is taken between consecutive samples (so an irregular 7 Hz tracker does not
    get differentiated at a rate it does not have) and placed at the mid-times.
    """
    dtm = np.diff(t)
    ok = dtm > 1e-6
    s = np.linalg.norm(np.diff(xy, axis=0), axis=1)[ok] / dtm[ok]
    tm = (t[1:][ok] + t[:-1][ok]) / 2
    return resample_smooth(tm, s, grid, smooth_s)


def correlate_offset(u_ref, s_ref, t_gt_rel, s_gt_fn, search=OFFSET_SEARCH_S, step=OFFSET_STEP_S,
                     min_overlap=0.5, gt_span=None):
    """Grid search for d such that GT(t_rel + d) best matches ref(u).

    u_ref are bag-relative seconds (receive time minus first bag message), s_ref the
    reference speed on that grid. For each candidate d the GT speed is read at
    u_ref - d and the Pearson r over the overlapping, finite samples is recorded. The
    curve is returned whole so it can be plotted and the second-best peak inspected.
    """
    ds = np.arange(search[0], search[1] + step / 2, step)
    r = np.full(len(ds), np.nan)
    span = gt_span if gt_span is not None else (t_gt_rel[-1] - t_gt_rel[0])
    for i, d in enumerate(ds):
        sg = s_gt_fn(u_ref - d)
        m = np.isfinite(sg) & np.isfinite(s_ref)
        if m.sum() < 20 or (u_ref[m][-1] - u_ref[m][0]) < min_overlap * span:
            continue
        a, b = s_ref[m], sg[m]
        if a.std() < 1e-9 or b.std() < 1e-9:
            continue
        r[i] = np.corrcoef(a, b)[0, 1]
    if not np.isfinite(r).any():
        return dict(offset=np.nan, peak_r=np.nan, second_r=np.nan, ds=ds, r=r)
    k = int(np.nanargmax(r))
    d_best, r_best = ds[k], r[k]
    # parabolic refinement on the three samples around the peak
    if 0 < k < len(ds) - 1 and np.isfinite(r[k - 1]) and np.isfinite(r[k + 1]):
        y0, y1, y2 = r[k - 1], r[k], r[k + 1]
        den = (y0 - 2 * y1 + y2)
        if abs(den) > 1e-12:
            d_best = ds[k] + 0.5 * (y0 - y2) / den * step
    # best peak outside +-1 s of the winner: how ambiguous is the match
    far = np.abs(ds - ds[k]) > 1.0
    second = float(np.nanmax(r[far])) if np.isfinite(r[far]).any() else np.nan
    return dict(offset=float(d_best), peak_r=float(r_best), second_r=second, ds=ds, r=r)


# ------------------------------------------------------------ geometry and metrics

def body_axes(t, p, q):
    """Detect which body axis points forward and which left, from the data.

    Returns (fwd_b, left_b, up_b, confidence). Confidence is the ratio between the
    dominant and the second body-velocity component; below ~3 the detection is weak
    (a robot that reverses as much as it advances, or that barely moved).
    """
    R = vm.quat_to_R(q)
    v = np.gradient(p, t, axis=0)
    speed = np.linalg.norm(v[:, :2], axis=1)
    m = speed > MOVING_SPEED
    if m.sum() < 10:
        return None
    vb = np.einsum("nji,nj->ni", R[m], v[m])          # R^T v
    mean = vb.mean(0)
    order = np.argsort(-np.abs(mean))
    fwd = np.zeros(3); fwd[order[0]] = np.sign(mean[order[0]])
    conf = abs(mean[order[0]]) / max(abs(mean[order[1]]), 1e-9)
    up_b = np.einsum("nji,j->ni", R[m], np.array([0.0, 0.0, 1.0])).mean(0)
    up_b /= max(np.linalg.norm(up_b), 1e-9)
    left = np.cross(up_b, fwd); left /= max(np.linalg.norm(left), 1e-9)
    return dict(fwd=fwd, left=left, up=up_b, confidence=float(conf),
                fwd_name=("xyz"[order[0]] if fwd[order[0]] > 0 else "-" + "xyz"[order[0]]))


def apply_lever_arm(p, q, axes, forward_m, left_m, up_m=0.0):
    """Move the estimated origin to the tracked marker. The vertical component matters
    only through body pitch/roll: on the site's 5-degree slope a marker 0.36 m above
    the camera moves about 3 cm horizontally relative to it, below today's errors but
    not below the 0.05 m target."""
    if axes is None or (forward_m == 0.0 and left_m == 0.0 and up_m == 0.0):
        return p
    r_b = forward_m * axes["fwd"] + left_m * axes["left"] + up_m * axes["up"]
    R = vm.quat_to_R(q)
    return p + np.einsum("nij,j->ni", R, r_b)


# ------------------------------------------------------------- frame repair (4 Nov 2026)
# ORB-SLAM3 writes its live poses in whatever world frame the active map has at
# that moment. Before the IMU initialises that world is the first camera frame
# (vertical along the camera's y axis); at the initialisation the whole map is
# rotated about its origin so that gravity points down -z (LocalMapping::
# InitializeIMU -> Map::ApplyScaledRotation; the scale is fixed for stereo), but
# the poses already written keep the old frame. A run whose initialisation came
# late, or never came (realv7: the upstream "insufficient motion" reset at 3-4 s
# throws the first map away and the second map's initialisation is starved by
# the local-mapping backlog), therefore lies in a tilted plane and carries a
# multi-metre step at the late initialisation. Two optional, GT-free repairs:
#   --join-imu-init  rotate the poses written BEFORE each initialisation by the
#                    rotation measured at the initialisation frame (orientation
#                    jump between consecutive poses), so a map is in one frame;
#   --level-plane    rotate each map's own fitted plane to the horizontal, the
#                    side chosen by the body's up axis (--body-up), which a floor
#                    robot permits and which takes nothing from the truth.
# Maps are delimited by the active-map resets recorded in performance.csv; a
# reset restarts the position at the origin and is NOT stitched (it is a loss).

BODY_UP = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}
JOIN_MIN_ANGLE_DEG = 10.0     # an initialisation rotation smaller than this is not trusted
PLANE_MIN_POSES = 20
PLANE_MIN_EXTENT_M = 0.15     # rms extent along the plane's second axis, else the fit is a line


def run_events(run_dir):
    """IMU-initialisation and active-map-reset events of an ORB-SLAM3 run, from the
    node's performance.csv (one row per frame). None for other estimators."""
    path = os.path.join(run_dir, "performance.csv")
    if not os.path.exists(path):
        return None
    inits, resets = [], []
    prev_init, max_cnt, n_ok, n_ok_init = None, None, 0, 0
    with open(path) as f:
        for r in csv.DictReader(f):
            if "imu_initialized" not in r or "active_map_reset_requests" not in r:
                return None
            try:
                state = int(r["tracking_state"])
            except ValueError:
                continue
            if state < 0:          # node-side rows (frame not handed to the tracker) carry zeros
                continue
            t = int(r["timestamp_ns"]) * 1e-9
            init = r["imu_initialized"] == "1"
            if init and prev_init is False:
                inits.append(t)
            cnt = int(r["active_map_reset_requests"])
            if max_cnt is not None and cnt > max_cnt:
                resets.append((t, r.get("last_reset_reason", "")))
            max_cnt = cnt if max_cnt is None else max(max_cnt, cnt)
            prev_init = init
            if state == 2:
                n_ok += 1
                n_ok_init += int(init)
    return dict(inits=inits, resets=resets, init_fraction=(n_ok_init / n_ok) if n_ok else float("nan"))


def R_to_quat(R):
    """(N,3,3) -> (N,4) wxyz, Shepperd's method."""
    N = len(R); q = np.zeros((N, 4))
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    for i in range(N):
        m = R[i]
        if tr[i] > 0:
            S = np.sqrt(tr[i] + 1.0) * 2
            q[i] = [0.25 * S, (m[2, 1] - m[1, 2]) / S, (m[0, 2] - m[2, 0]) / S, (m[1, 0] - m[0, 1]) / S]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            q[i] = [(m[2, 1] - m[1, 2]) / S, 0.25 * S, (m[0, 1] + m[1, 0]) / S, (m[0, 2] + m[2, 0]) / S]
        elif m[1, 1] > m[2, 2]:
            S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            q[i] = [(m[0, 2] - m[2, 0]) / S, (m[0, 1] + m[1, 0]) / S, 0.25 * S, (m[1, 2] + m[2, 1]) / S]
        else:
            S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            q[i] = [(m[1, 0] - m[0, 1]) / S, (m[0, 2] + m[2, 0]) / S, (m[1, 2] + m[2, 1]) / S, 0.25 * S]
    return q


def rot_a_to_b(a, b):
    """Rotation matrix taking unit vector a onto unit vector b (Rodrigues)."""
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); s = np.linalg.norm(v); c = float(np.dot(a, b))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        axis = np.cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1.0, 0])
        axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + 2 * K @ K
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s ** 2)


def rot_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def plane_normal(p):
    """Unit normal of the least-squares plane and the rms extents along the three
    principal axes (the third is the rms off-plane distance)."""
    c = p - p.mean(0)
    _, sv, vt = np.linalg.svd(c, full_matrices=False)
    return vt[2], sv / np.sqrt(len(p))


def plane_tilt_deg(p):
    n, _ = plane_normal(p)
    return float(np.degrees(np.arccos(abs(n[2]))))


def repair_frame(run_dir, t, p, q, join, level, body_up):
    """Apply the two frame repairs described above. Returns (p, q, report)."""
    p = p.copy(); q = q.copy()
    ev = run_events(run_dir) if join else None
    report = dict(join=join, level=level, body_up=None if body_up is None else [float(x) for x in body_up],
                  tilt_before_deg=plane_tilt_deg(p), events=ev, joins=[], levels=[])
    # segments = maps: split at the resets (a reset restarts the position at the origin)
    bounds = [0]
    if ev:
        for tr, _ in ev["resets"]:
            b = int(np.searchsorted(t, tr))
            if 0 < b < len(t) and b != bounds[-1]:
                bounds.append(b)
    bounds.append(len(t))
    segments = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1) if bounds[i + 1] > bounds[i]]
    report["segments"] = [dict(start_s=float(t[a] - t[0]), end_s=float(t[b - 1] - t[0]), poses=int(b - a)) for a, b in segments]
    e_up = None if body_up is None else np.asarray(body_up, float)

    for a, b in segments:
        if join and ev:
            for ti in ev["inits"]:
                k0 = int(np.searchsorted(t, ti))
                if not (a < k0 < b):
                    continue
                best = None
                for k in (k0 - 1, k0, k0 + 1):
                    if not (a < k < b):
                        continue
                    Rk = vm.quat_to_R(q[[k - 1, k]])
                    R = Rk[1] @ Rk[0].T
                    ang = rot_angle_deg(R)
                    if best is None or ang > best[1]:
                        best = (k, ang, R)
                if best is None:
                    continue
                k, ang, R = best
                step_before = float(np.linalg.norm(p[k] - p[k - 1]))
                if ang < JOIN_MIN_ANGLE_DEG:
                    report["joins"].append(dict(t_s=float(t[k] - t[0]), angle_deg=ang, applied=False,
                                               step_before_m=step_before, note="rotation below threshold"))
                    continue
                p[a:k] = p[a:k] @ R.T
                q[a:k] = R_to_quat(np.einsum("ij,njk->nik", R, vm.quat_to_R(q[a:k])))
                step_after = float(np.linalg.norm(p[k] - p[k - 1]))
                report["joins"].append(dict(t_s=float(t[k] - t[0]), angle_deg=ang, applied=True,
                                           step_before_m=step_before, step_after_m=step_after))
        if level:
            seg = p[a:b]
            n, ext = plane_normal(seg)
            Rb = vm.quat_to_R(q[a:b])
            u = None
            if e_up is not None:
                u = np.einsum("nij,j->ni", Rb, e_up).mean(0); u /= max(np.linalg.norm(u), 1e-9)
            tilt0 = float(np.degrees(np.arccos(abs(n[2]))))
            if (b - a) >= PLANE_MIN_POSES and ext[1] >= PLANE_MIN_EXTENT_M:
                method = "plane"
                if u is not None and np.dot(n, u) < 0:
                    n = -n
                elif u is None and n[2] < 0:
                    n = -n
                target = n
            elif u is not None:
                method = "body-up (plane undetermined)"
                target = u
            else:
                report["levels"].append(dict(segment=[a, b], method="none", tilt_before_deg=tilt0))
                continue
            Rl = rot_a_to_b(target, np.array([0.0, 0.0, 1.0]))
            p[a:b] = seg @ Rl.T
            q[a:b] = R_to_quat(np.einsum("ij,njk->nik", Rl, Rb))
            agree = None if u is None else float(np.degrees(np.arccos(np.clip(abs(np.dot(n, u)), -1, 1))))
            report["levels"].append(dict(segment=[a, b], method=method, tilt_before_deg=tilt0,
                                        tilt_after_deg=plane_tilt_deg(p[a:b]), rotation_deg=rot_angle_deg(Rl),
                                        body_up_vs_normal_deg=agree, off_plane_rms_m=float(ext[2])))
    report["tilt_after_deg"] = plane_tilt_deg(p)
    return p, q, report


def se2_align(src_xy, dst_xy):
    """Rigid planar fit (yaw + translation, scale 1) via the project's SE(3) Umeyama
    on z=0 points, so the algebra is literally the same routine as the simulation
    chapter's. Returns (R2, t2, yaw_deg)."""
    s3 = np.column_stack([src_xy, np.zeros(len(src_xy))])
    d3 = np.column_stack([dst_xy, np.zeros(len(dst_xy))])
    R, t = vm.umeyama(s3, d3)
    return R[:2, :2], t[:2], float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


def sim2_scale(src_xy, dst_xy):
    """Umeyama similarity scale, reported as a diagnostic only."""
    cs, cd = src_xy.mean(0), dst_xy.mean(0)
    S = (dst_xy - cd).T @ (src_xy - cs) / len(src_xy)
    U, D, Vt = np.linalg.svd(S)
    sgn = np.sign(np.linalg.det(U @ Vt))
    var = ((src_xy - cs) ** 2).sum(1).mean()
    return float((D[0] + sgn * D[1]) / max(var, 1e-12))


def fit_lever_arm(gt_xy, p, q, axes, iters=6):
    """Alternate SE(2) alignment and a linear least-squares solve for the marker
    offset (forward, left) in the detected body axes. Diagnostic only."""
    if axes is None:
        return None
    R = vm.quat_to_R(q)
    B = np.stack([np.einsum("nij,j->ni", R, axes["fwd"])[:, :2],
                  np.einsum("nij,j->ni", R, axes["left"])[:, :2]], axis=2)   # (n,2,2)
    fl = np.zeros(2)
    for _ in range(iters):
        pm = p[:, :2] + B @ fl
        R2, t2, _ = se2_align(pm, gt_xy)
        A = np.einsum("ij,njk->nik", R2, B).reshape(-1, 2)
        b = (gt_xy - (p[:, :2] @ R2.T + t2)).reshape(-1)
        fl, *_ = np.linalg.lstsq(A, b, rcond=None)
    pm = p[:, :2] + B @ fl
    R2, t2, _ = se2_align(pm, gt_xy)
    e = np.linalg.norm(pm @ R2.T + t2 - gt_xy, axis=1)
    return dict(forward_m=float(fl[0]), left_m=float(fl[1]),
                ate_rmse_with_fit=float(np.sqrt((e ** 2).mean())))


def rpe_2d(t, pa, g, delta, jumps):
    """Translational RPE on relative displacements in the aligned frame, with the
    pairing and tolerance rules of vio_metrics.rpe (pairs within 25 % of delta,
    overlapping, 'clean' excludes pairs spanning a jump)."""
    j = np.searchsorted(t, t + delta, side="left")
    i = np.arange(len(t)); keep = j < len(t); i, j = i[keep], j[keep]
    if len(i) == 0:
        return None
    ok = np.abs((t[j] - t[i]) - delta) <= RPE_TOL * delta
    i, j = i[ok], j[ok]
    if len(i) == 0:
        return None
    e = np.linalg.norm((pa[j] - pa[i]) - (g[j] - g[i]), axis=1)
    out = dict(delta_s=delta, pairs=int(len(i)), trans_rmse=float(np.sqrt((e ** 2).mean())),
               trans_median=float(np.median(e)), trans_max=float(e.max()))
    spans = np.zeros(len(t), int)
    if len(jumps):
        spans[np.asarray(jumps, int) + 1] = 1
    before = np.cumsum(spans)
    clean = before[j] == before[i]
    if clean.any():
        ec = e[clean]
        out.update(pairs_clean=int(clean.sum()), trans_rmse_clean=float(np.sqrt((ec ** 2).mean())),
                   trans_median_clean=float(np.median(ec)))
    else:
        out.update(pairs_clean=0, trans_rmse_clean=float("nan"), trans_median_clean=float("nan"))
    return out


def score(gt_abs_t, gt_xy, t_e, p_e, q_e, axes, lever, trim_start=0.0):
    """Match GT samples to the estimate, align, and compute every metric."""
    p_m = apply_lever_arm(p_e, q_e, axes, *lever)
    lo, hi = max(t_e[0], gt_abs_t[0]), min(t_e[-1], gt_abs_t[-1])
    if trim_start > 0:
        lo = max(lo, t_e[0] + trim_start)
    in_span = (gt_abs_t >= lo) & (gt_abs_t <= hi)
    # nearest estimate sample must be close: do not interpolate across a hole
    idx = np.clip(np.searchsorted(t_e, gt_abs_t), 1, len(t_e) - 1)
    near = np.minimum(np.abs(t_e[idx] - gt_abs_t), np.abs(t_e[idx - 1] - gt_abs_t))
    m = in_span & (near <= MAX_ASSOC_GAP_S)
    res = dict(gt_in_span=int(in_span.sum()), matched=int(m.sum()),
               unmatched_in_span=int((in_span & ~m).sum()))
    if m.sum() < 10:
        res.update(valid=False, reason="fewer than 10 matched samples")
        return res
    t = gt_abs_t[m]; g = gt_xy[m]
    pe = np.column_stack([np.interp(t, t_e, p_m[:, k]) for k in range(3)])
    R2, t2, yaw = se2_align(pe[:, :2], g)
    pa = pe[:, :2] @ R2.T + t2
    err = np.linalg.norm(pa - g, axis=1)
    # jumps on the matched estimate samples (same 10 m/s rule as vio_metrics)
    jumps, steps = vm.find_jumps(t, pe)
    rpe = {d: rpe_2d(t, pa, g, d, jumps) for d in RPE_DELTAS}
    gt_len = float(np.linalg.norm(np.diff(g, axis=0), axis=1).sum())
    est_len = float(np.linalg.norm(np.diff(pe[:, :2], axis=0), axis=1).sum())
    res.update(valid=True, t=t, g=g, pa=pa, pe=pe, err=err, yaw_align_deg=yaw,
               matched_duration_s=float(t[-1] - t[0]),
               ate_rmse=float(np.sqrt((err ** 2).mean())), ate_mean=float(err.mean()),
               ate_median=float(np.median(err)), ate_max=float(err.max()),
               ate_final=float(err[-1]), gt_path_m=gt_len, est_path_m=est_len,
               path_ratio=float(est_len / gt_len) if gt_len > 0 else float("nan"),
               scale_diag=sim2_scale(pe[:, :2], g), jumps=int(len(jumps)),
               biggest_jump_m=float(steps[jumps].max()) if len(jumps) else 0.0,
               z_range_m=float(pe[:, 2].max() - pe[:, 2].min()), rpe=rpe,
               diverged=bool(err.max() > DIVERGED_M))
    return res


# --------------------------------------------------------------------- reporting

def status_of(run_summary, pipeline, sc, align_ok, lever_known, lever_note="lever arm not provided"):
    reasons = []
    graceful = run_summary.get("shutdown_status", "") == "graceful"
    if not graceful:
        reasons.append("shutdown not graceful")
    resets = int(run_summary.get("resets", run_summary.get("active_map_reset_requests", 0)) or 0)
    if resets > 0:
        why = run_summary.get("last_reset_reason", "")
        reasons.append(f"{resets} estimator reset(s)" + (f" ({why})" if why else ""))
    loss = float(run_summary.get("tracking_loss_duration_s", 0) or 0)
    if loss > 0:
        reasons.append(f"tracking loss {loss:.2f} s")
    if not sc.get("valid"):
        return "failed", reasons + [sc.get("reason", "no valid matches")]
    cov = sc["matched"] / max(sc["gt_in_span"], 1)
    # Physical plausibility, added 2026-10-20 after dataset_realv3_000: VINS-Fusion
    # climbed 22 m vertically on a flat floor and travelled 60 m against 10 m of
    # truth, yet passed every rule above because its max planar error stayed under
    # the 50 m divergence threshold and it never reset. A run that leaves the floor
    # by more than PLAUSIBLE_Z_RANGE_M has failed whatever its ATE says. The path
    # ratio is only a reason, not a verdict: with the marker lever arm unmeasured,
    # a short path on a tight loop can be geometry rather than estimator error.
    implausible = float(sc.get("z_range_m", 0) or 0) > PLAUSIBLE_Z_RANGE_M
    if sc["diverged"] or implausible or (pipeline.startswith("vins") and resets > 5) or cov < 0.2:
        label = "failed"
        if sc["diverged"]:
            reasons.append(f"diverged, max error {sc['ate_max']:.1f} m")
        if implausible:
            reasons.append(f"implausible: z range {sc['z_range_m']:.1f} m on a planar run, "
                           f"path ratio {sc['path_ratio']:.2f}")
        if cov < 0.2:
            reasons.append(f"coverage {cov:.0%}")
    elif cov < 0.8 or sc["jumps"] > 0 or resets > 0 or loss > 0 or not graceful:
        label = "partial"
        if cov < 0.8:
            reasons.append(f"coverage {cov:.0%}")
        if sc["jumps"]:
            reasons.append(f"{sc['jumps']} jump(s)")
    else:
        label = "completed"
    # Bug fixed 2026-11-04: the "completed" label used to sit on the else branch of the
    # path-ratio test below, so any run with a plausible path ratio was relabelled
    # "completed" whatever the resets or coverage said (realv6/realv7 runs with
    # resets were reported as completed). Scorings before this date carry that label.
    if not (PLAUSIBLE_PATH_RATIO[0] <= float(sc.get("path_ratio", 1) or 1) <= PLAUSIBLE_PATH_RATIO[1]):
        reasons.append(f"path ratio {sc['path_ratio']:.2f}")
    prov = []
    if not lever_known:
        prov.append(lever_note)
    if not align_ok:
        prov.append("time alignment uncertain")
    if prov:
        label += " (provisional: " + "; ".join(prov) + ")"
    return label, reasons


def style(ax, th):
    ax.set_facecolor(th["surface"])
    for s in ax.spines.values():
        s.set_color(th["grid"])
    ax.tick_params(colors=th["ink2"], labelsize=8)
    ax.xaxis.label.set_color(th["ink2"]); ax.yaxis.label.set_color(th["ink2"])
    ax.title.set_color(th["ink"])
    ax.grid(True, color=th["grid"], lw=0.6)


def fig_trajectory(path, th, dataset, gt_xy, results, n_gt):
    fig, ax = plt.subplots(figsize=(7.5, 6.5), facecolor=th["surface"])
    style(ax, th)
    ax.plot(gt_xy[:, 0], gt_xy[:, 1], color=th["gt"], lw=2.0, label=f"tracker ground truth (n={n_gt})")
    ax.plot(gt_xy[0, 0], gt_xy[0, 1], "o", color=th["gt"], ms=8, mfc=th["surface"], mew=2)
    ax.plot(gt_xy[-1, 0], gt_xy[-1, 1], "s", color=th["gt"], ms=8, mfc=th["surface"], mew=2)
    for key, r in results.items():
        label, _, col = SYSTEMS[key]
        sc = r["score"]
        if not sc.get("valid"):
            ax.plot([], [], color=col, lw=2, label=f"{label}: {r['status']}")
            continue
        ax.plot(sc["pa"][:, 0], sc["pa"][:, 1], color=col, lw=2.0,
                label=f"{label}: ATE RMSE {sc['ate_rmse']:.3f} m, n={sc['matched']} [{r['status'].split(' (')[0]}]")
        ax.plot(sc["pa"][-1, 0], sc["pa"][-1, 1], "s", color=col, ms=7, mfc=th["surface"], mew=2)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x [m] (tracker frame)"); ax.set_ylabel("y [m] (tracker frame)")
    ax.set_title(f"{dataset}: estimates after rigid SE(2) alignment (scale fixed)\n"
                 "circle = start, square = end", fontsize=10)
    leg = ax.legend(fontsize=8, loc="best", facecolor=th["surface"], edgecolor=th["grid"])
    for t_ in leg.get_texts():
        t_.set_color(th["ink"])
    fig.tight_layout(); fig.savefig(path, dpi=160, facecolor=th["surface"]); plt.close(fig)


def fig_error(path, th, dataset, results, t_bag0, wheel_prof, grid, gt_prof_abs):
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True, facecolor=th["surface"])
    for ax in axes:
        style(ax, th)
    ax = axes[0]
    for key, r in results.items():
        label, _, col = SYSTEMS[key]
        sc = r["score"]
        if sc.get("valid"):
            ax.plot(sc["t"] - t_bag0, sc["err"], color=col, lw=1.6,
                    label=f"{label} (n={sc['matched']}, RMSE {sc['ate_rmse']:.3f} m)")
    ax.set_ylabel("position error after SE(2) alignment [m]")
    ax.set_title(f"{dataset}: error and speed on the robot-clock timeline", fontsize=10)
    leg = ax.legend(fontsize=8, facecolor=th["surface"], edgecolor=th["grid"])
    for t_ in leg.get_texts():
        t_.set_color(th["ink"])
    ax = axes[1]
    if wheel_prof is not None:
        w = wheel_prof / np.nanmax(wheel_prof)
        ax.plot(grid, w, color=th["wheel"], lw=1.2, label=f"{WHEEL_TOPIC} (normalised, unit unverified)")
    ax.plot(grid, gt_prof_abs, color=th["gt"], lw=1.8, label="tracker speed [m/s], shifted by fitted offset")
    for key, r in results.items():
        label, _, col = SYSTEMS[key]
        if "speed_grid" in r:
            ax.plot(grid, r["speed_grid"], color=col, lw=1.0, alpha=0.9, label=f"{label} speed [m/s]")
    ax.set_ylabel("speed"); ax.set_xlabel("time since first bag message [s]")
    leg = ax.legend(fontsize=8, facecolor=th["surface"], edgecolor=th["grid"])
    for t_ in leg.get_texts():
        t_.set_color(th["ink"])
    fig.tight_layout(); fig.savefig(path, dpi=160, facecolor=th["surface"]); plt.close(fig)


def fig_offset(path, th, dataset, curves):
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=th["surface"])
    style(ax, th)
    for name, (col, c) in curves.items():
        ax.plot(c["ds"], c["r"], color=col, lw=1.4,
                label=f"{name}: d = {c['offset']:.3f} s, r = {c['peak_r']:.2f}, next peak r = {c['second_r']:.2f}")
    ax.set_xlabel("candidate offset d [s]  (robot clock = bag start + d + tracker time)")
    ax.set_ylabel("Pearson r of speed profiles")
    ax.set_title(f"{dataset}: time-offset search", fontsize=10)
    leg = ax.legend(fontsize=8, facecolor=th["surface"], edgecolor=th["grid"])
    for t_ in leg.get_texts():
        t_.set_color(th["ink"])
    fig.tight_layout(); fig.savefig(path, dpi=160, facecolor=th["surface"]); plt.close(fig)


def fig_rpe(path, th, dataset, results):
    fig, ax = plt.subplots(figsize=(6.5, 4), facecolor=th["surface"])
    style(ax, th)
    for key, r in results.items():
        label, _, col = SYSTEMS[key]
        sc = r["score"]
        if not sc.get("valid"):
            continue
        xs = [d for d in RPE_DELTAS if sc["rpe"].get(d)]
        ax.plot(xs, [sc["rpe"][d]["trans_rmse"] for d in xs], "-o", color=col, lw=1.8, ms=6,
                label=f"{label} RMSE (pairs at 1 s: {sc['rpe'][1.0]['pairs'] if sc['rpe'].get(1.0) else 0})")
        ax.plot(xs, [sc["rpe"][d]["trans_median"] for d in xs], "--", color=col, lw=1.2,
                label=f"{label} median")
    ax.set_xscale("log"); ax.set_xticks(RPE_DELTAS); ax.set_xticklabels([str(d) for d in RPE_DELTAS])
    ax.set_xlabel("interval Δ [s]"); ax.set_ylabel("translational RPE [m]")
    ax.set_title(f"{dataset}: RPE in the aligned frame (no reference heading)", fontsize=10)
    leg = ax.legend(fontsize=8, facecolor=th["surface"], edgecolor=th["grid"])
    for t_ in leg.get_texts():
        t_.set_color(th["ink"])
    fig.tight_layout(); fig.savefig(path, dpi=160, facecolor=th["surface"]); plt.close(fig)


def tex_escape(s):
    return s.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def write_tables(out, rows, lever_known):
    def num(v, p=3):
        return "---" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{p}f}"
    excluded =[r for r in rows if r["status"].startswith("failed")]
    with open(os.path.join(out, "real_gt_table.tex"), "w") as f:
        w = f.write
        w("% Generated by script/eval_real_gt.py -- do not edit by hand.\n")
        w("\\begin{table}[htbp]\n\\centering\n")
        names = []
        for r in rows:
            if r["dataset"] not in names:
                names.append(r["dataset"])
        w("\\caption{Visual estimators against the optical-tracker ground truth on the real "
          + ", ".join("\\repo{%s}" % n for n in names) + " recording%s. Planar ATE after rigid $SE(2)$ alignment with "
          "scale fixed at one, evaluated at the tracker's own samples ($n$); the time offset "
          "between the unsynchronised robot clock and the tracker is estimated from wheel-speed "
          "correlation (\\cref{sec:method-trajectory}). Translational RPE at "
          "$\\Delta=\\SI{1}{\\second}$ compares relative displacements in the aligned frame; "
          "the tracker has no heading, so rotational RPE is not defined. "
          "``Path'' is the estimate's planar path length over the matched interval divided by "
          "the tracker's. " % ("" if len(names) == 1 else "s")
          + ("" if lever_known else "The marker-to-sensor lever arm is not measured (horizontal "
             "offset assumed zero), so every value is provisional. ")
          + ("Failed runs (%s) are excluded and reported in the text. "
             % ", ".join(f"\\repo{{{r['dataset']}}} {r['system']}" for r in excluded)
             if excluded else "")
          + "Lower is better.}\n")
        w("\\label{tab:real-gt-ate}\n\\small\n\\resizebox{\\textwidth}{!}{%\n")
        w("\\begin{tabular}{ll l r r r r r r r}\n\\toprule\n")
        w("Dataset & Estimator & Status & $n$ & Dur. [\\si{\\second}] & ATE RMSE [\\si{\\metre}] "
          "& ATE max [\\si{\\metre}] & RPE$_{1\\,\\mathrm{s}}$ [\\si{\\metre}] & Path & Jumps \\\\\n\\midrule\n")
        last = None
        for r in rows:
            if r["status"].startswith("failed"):
                continue
            ds = "\\repo{%s}" % r["dataset"].replace("dataset_", "") if r["dataset"] != last else ""
            last = r["dataset"]
            w("%s & %s & %s & %s & %s & %s & %s & %s & %s & %s \\\\\n" % (
                ds, r["system"], tex_escape(r["status"].split(" (")[0]), r.get("matched", "---"),
                num(r.get("matched_duration_s"), 1), num(r.get("ate_rmse")), num(r.get("ate_max")),
                num(r.get("rpe1_trans_rmse")), num(r.get("path_ratio"), 2), r.get("jumps", "---")))
        w("\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    with open(os.path.join(out, "real_gt_rpe_table.tex"), "w") as f:
        w = f.write
        w("% Generated by script/eval_real_gt.py -- do not edit by hand.\n")
        w("\\begin{table}[htbp]\n\\centering\n")
        w("\\caption{Translational RPE sweep on the real recordings, RMSE over all admissible "
          "pairs (realised separation within \\SI{25}{\\percent} of $\\Delta$), relative "
          "displacements compared in the aligned frame. Pair counts in parentheses. Failed runs "
          "excluded.}\n\\label{tab:real-gt-rpe}\n\\small\n")
        w("\\begin{tabular}{ll " + "r" * len(RPE_DELTAS) + "}\n\\toprule\n")
        w("Dataset & Estimator & " + " & ".join(f"$\\Delta={d:g}$\\,s" for d in RPE_DELTAS) + " \\\\\n\\midrule\n")
        last = None
        for r in rows:
            if r["status"].startswith("failed") or "rpe" not in r:
                continue
            ds = "\\repo{%s}" % r["dataset"].replace("dataset_", "") if r["dataset"] != last else ""
            last = r["dataset"]
            cells = []
            for d in RPE_DELTAS:
                x = r["rpe"].get(d)
                cells.append("---" if not x else f"{x['trans_rmse']:.3f} ({x['pairs']})")
            w("%s & %s & %s \\\\\n" % (ds, r["system"], " & ".join(cells)))
        w("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def git_rev(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "--short=12", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


# -------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", action="append", required=True, help="dataset name, repeatable")
    ap.add_argument("--dataset-root", default=os.path.join(ROOT, "dataset"))
    ap.add_argument("--gt-dir", default=os.path.join(ROOT, "dataset", "gt"))
    ap.add_argument("--run", default="logging_20260916_03",
                    help="run directory to pin, explicit and never newest: either ONE name for every "
                         "system, or per-system KEY=NAME pairs separated by commas, e.g. "
                         "vins=logging_20260916_01,orb=loopclosing,orb_nolc=noloopclosing. A system "
                         "without a pair falls back to the single name if one was given.")
    ap.add_argument("--systems", default="vins,orb", help="comma list of " + ",".join(SYSTEMS))
    ap.add_argument("--lever-arm", default=None,
                    help="FORWARD,LEFT[,UP] metres from the estimator body origin (cam0/IMU) to "
                         "the tracked marker, in the robot frame (UP positive above the camera). "
                         "Omit = 0,0,0 and provisional.")
    ap.add_argument("--lever-arm-assumed", action="store_true",
                    help="the --lever-arm value is an assumption, not a measurement: apply it but "
                         "keep every result labelled provisional and say so in the manifest")
    ap.add_argument("--fit-lever-arm", action="store_true", help="also fit the lever arm (diagnostic)")
    ap.add_argument("--time-offset", action="append", default=[],
                    help="DATASET=SECONDS: pin the tracker offset instead of estimating it")
    ap.add_argument("--bag-from-runs", action="store_true",
                    help="bag absent and no earlier manifest: take the bag's first-message time as the "
                         "earliest vio.csv timestamp of the scored systems and its duration as the longest "
                         "vio span; no wheel correlation, so the offset comes from the estimator-speed "
                         "correlation and the alignment is labelled as such in the manifest")
    ap.add_argument("--bag-manifest", default=None,
                    help="manifest.json of an earlier scoring of the SAME bag; used only when the bag "
                         "itself is absent AND the offset is pinned: supplies the bag's first-message "
                         "time and duration, the wheel correlation is skipped, and the manifest records "
                         "the substitution")
    ap.add_argument("--trim-start", type=float, default=0.0, help="seconds dropped from each estimate's start")
    ap.add_argument("--join-imu-init", action="store_true",
                    help="ORB-SLAM3 runs: rotate the poses written before each IMU initialisation by the "
                         "rotation measured at the initialisation frame, so that each map is in one frame "
                         "(the live log keeps pre-initialisation poses in the un-rotated camera frame)")
    ap.add_argument("--level-plane", action="store_true",
                    help="rotate each map's fitted plane to the horizontal before the planar alignment "
                         "(a floor robot's path is horizontal; GT-free; removes genuine vertical drift too)")
    ap.add_argument("--body-up", default="+y", choices=sorted(BODY_UP),
                    help="body axis that points up, used to pick the side of the levelled plane and as the "
                         "fallback when the plane is undetermined (this rig: +y, gravity reads on +y)")
    ap.add_argument("--experiment-id", default="exp-estimator-baseline")
    ap.add_argument("--out", default=None, help="default output/compare/real/<experiment-id>")
    ap.add_argument("--themes", default="light,dark")
    args = ap.parse_args()

    out_root = args.out or os.path.join(ROOT, "output", "compare", "real", args.experiment_id)
    os.makedirs(out_root, exist_ok=True)
    lever = (0.0, 0.0, 0.0)
    lever_known = args.lever_arm is not None and not args.lever_arm_assumed
    lever_note = ("lever arm assumed, not measured" if args.lever_arm_assumed else "lever arm not provided")
    if args.lever_arm is not None:
        parts = [float(x) for x in args.lever_arm.split(",")]
        if len(parts) not in (2, 3):
            sys.exit("--lever-arm takes FORWARD,LEFT or FORWARD,LEFT,UP")
        lever = tuple(parts + [0.0] * (3 - len(parts)))
    pinned = {k: float(v) for k, v in (s.split("=") for s in args.time_offset)}
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    # Per-system run names. The real-rig runs are not all named logging_*: ORB's two
    # realv3 runs are "loopclosing" and "noloopclosing" while VINS kept the stamp
    # scheme, so one shared name cannot address them. Pairs win; a bare name is the
    # fallback for every system without a pair; a system with neither is an error
    # rather than a silent "newest".
    run_pairs = [x for x in args.run.split(",") if "=" in x]
    run_default = [x for x in args.run.split(",") if "=" not in x and x.strip()]
    run_for = {k.strip(): v.strip() for k, v in (x.split("=", 1) for x in run_pairs)}
    for key in systems:
        if key not in run_for:
            if not run_default:
                sys.exit(f"--run gives no run name for system '{key}'")
            run_for[key] = run_default[0]
    themes = [t for t in args.themes.split(",") if t in THEMES]
    all_rows, all_rpe = [], []
    now = dt.datetime.now().isoformat(timespec="seconds")

    for dataset in args.dataset:
        print(f"\n=== {dataset}")
        out = os.path.join(out_root, dataset); os.makedirs(out, exist_ok=True)
        ds_dir = os.path.join(args.dataset_root, dataset)
        gt_path = os.path.join(args.gt_dir, dataset)
        if not os.path.exists(gt_path):
            print(f"  [skip] no ground truth at {gt_path}"); continue
        bag_substituted = None
        if os.path.exists(os.path.join(ds_dir, "metadata.yaml")):
            db3, info = bag_db3(ds_dir)
            wheel, t0_ns, t1_ns = load_wheel(db3, info)
            t_bag0, bag_dur = t0_ns * 1e-9, (t1_ns - t0_ns) * 1e-9
        elif args.bag_manifest and dataset in pinned:
            # The bag is not on this machine. With the offset pinned, the only facts the
            # scoring needs from it are its first-message time (to place the tracker on
            # the robot clock) and its duration (the ground-truth duration check); both
            # come from the earlier manifest and the substitution is recorded.
            bm = json.load(open(args.bag_manifest))["bag"]
            db3, info, wheel = bm["path"], None, None
            t_bag0, bag_dur = float(bm["first_message_robotclock_s"]), float(bm["duration_s"])
            bag_substituted = args.bag_manifest
            print(f"  bag  ABSENT; first message and duration taken from {args.bag_manifest}")
        elif args.bag_from_runs:
            firsts, lasts = [], []
            for key in systems:
                rd = vm.run_dir(SYSTEMS[key][1], os.path.join("real", dataset), run_for[key])
                rr = vm.load(os.path.join(rd, "vio.csv"))
                if rr is not None:
                    firsts.append(rr[0][0]); lasts.append(rr[0][-1])
            if not firsts:
                print(f"  [skip] no bag and no vio.csv to stand in for it"); continue
            db3, info, wheel = os.path.join(ds_dir, "(absent)"), None, None
            t_bag0, bag_dur = min(firsts), max(lasts) - min(firsts)
            bag_substituted = "estimator logs (--bag-from-runs)"
            print(f"  bag  ABSENT; first message and duration taken from the estimator logs (earliest vio.csv stamp)")
        else:
            print(f"  [skip] no bag under {ds_dir} (pass --time-offset and --bag-manifest, or --bag-from-runs, to score without it)"); continue
        gt = load_gt(gt_path)
        gt_checks = dict(samples_ok=gt["n"] >= 50, gap_ok=gt["gap_max"] <= 1.0,
                         step_ok=gt["step_max"] <= 0.5,
                         duration_ok=abs(gt["duration"] - bag_dur) <= 0.25 * bag_dur)
        gt_ok = all(gt_checks.values())
        print(f"  bag  {db3}\n       first message {t_bag0:.3f} "
              f"({dt.datetime.fromtimestamp(t_bag0, dt.timezone.utc):%Y-%m-%d %H:%M:%S} UTC, robot clock), "
              f"{bag_dur:.2f} s")
        print(f"  gt   {gt['path']}\n       {gt['n']} samples, {gt['duration']:.2f} s, median dt {gt['gap_median']:.3f} s, "
              f"max gap {gt['gap_max']:.3f} s, max step {gt['step_max']:.3f} m, path {gt['path_len']:.2f} m, "
              f"recorded '{gt['first_stamp']}'")
        print(f"       checks {gt_checks} -> {'OK' if gt_ok else 'INVALID'}")

        # ---- time offset -------------------------------------------------------
        grid = np.arange(0.0, bag_dur + 25.0, 0.05)        # bag-relative seconds
        gt_grid = np.arange(-30, bag_dur + 30, 0.05)        # tracker-relative seconds
        gt_prof_cache = speed_profile(gt["t_rel"], gt["xy"], gt_grid)
        gt_fn = lambda u: np.interp(u, gt_grid, gt_prof_cache, left=np.nan, right=np.nan)  # noqa: E731
        curves = {}
        wheel_prof = None
        if wheel is not None:
            wheel_prof = resample_smooth(wheel["t"] - t_bag0, np.abs(wheel["v"]), grid)
            curves["wheel " + WHEEL_TOPIC] = (THEMES["light"]["wheel"], correlate_offset(grid, wheel_prof, gt["t_rel"], gt_fn))
        # ---- estimators --------------------------------------------------------
        loaded = {}
        for key in systems:
            label, tree, col = SYSTEMS[key]
            rdir = vm.run_dir(tree, os.path.join("real", dataset), run_for[key])
            r = vm.load(os.path.join(rdir, "vio.csv"))
            if r is None:
                print(f"  [skip] {label}: no vio.csv in {rdir}"); continue
            t_e, p_e, q_e, _ = r
            repair = None
            if args.join_imu_init or args.level_plane:
                p_e, q_e, repair = repair_frame(rdir, t_e, p_e, q_e, args.join_imu_init, args.level_plane,
                                                BODY_UP[args.body_up])
            axes = body_axes(t_e, p_e, q_e)
            sp = speed_profile(t_e - t_bag0, p_e[:, :2], grid)
            curves[label] = (col, correlate_offset(grid, sp, gt["t_rel"], gt_fn))
            loaded[key] = dict(dir=rdir, t=t_e, p=p_e, q=q_e, axes=axes, speed_grid=sp, repair=repair,
                               meta=read_kv_csv(os.path.join(rdir, "run_metadata.csv")),
                               summary=read_kv_csv(os.path.join(rdir, "run_summary.csv")))
            ax_txt = ("forward=%s conf=%.1f" % (axes["fwd_name"], axes["confidence"])) if axes else "undetermined"
            print(f"  {label}: {rdir}\n       {len(t_e)} poses, {t_e[-1]-t_e[0]:.2f} s, body axes {ax_txt}")
            if repair:
                ev = repair["events"]
                if ev:
                    print(f"       events: IMU initialised at " + (", ".join(f"{x - t_e[0]:.1f}" for x in ev["inits"]) or "never")
                          + " s; resets at " + (", ".join(f"{x - t_e[0]:.1f} ({why})" for x, why in ev["resets"]) or "none")
                          + f"; IMU initialised for {100 * ev['init_fraction']:.0f} % of the tracked frames")
                for j in repair["joins"]:
                    print(f"       join at {j['t_s']:.1f} s: rotation {j['angle_deg']:.1f} deg, step {j['step_before_m']:.2f} m -> "
                          + (f"{j['step_after_m']:.2f} m" if j["applied"] else f"not applied ({j['note']})"))
                for lv in repair["levels"]:
                    print(f"       level segment {lv['segment']}: {lv['method']}, tilt {lv['tilt_before_deg']:.1f} -> "
                          + (f"{lv['tilt_after_deg']:.1f} deg, rotated {lv['rotation_deg']:.1f} deg, body-up vs normal "
                             f"{lv['body_up_vs_normal_deg']:.1f} deg, off-plane rms {lv['off_plane_rms_m']:.2f} m"
                             if "tilt_after_deg" in lv else "not levelled"))
                print(f"       plane tilt of the whole run {repair['tilt_before_deg']:.1f} -> {repair['tilt_after_deg']:.1f} deg")
        if not loaded:
            continue
        for name, (_, c) in curves.items():
            print(f"  offset via {name}: d = {c['offset']:.3f} s, r = {c['peak_r']:.2f}, next peak {c['second_r']:.2f}")
        wheel_curve = curves.get("wheel " + WHEEL_TOPIC)
        if dataset in pinned:
            offset, offset_src = pinned[dataset], "pinned by --time-offset"
        elif wheel_curve and np.isfinite(wheel_curve[1]["offset"]):
            offset, offset_src = wheel_curve[1]["offset"], "wheel-speed correlation"
        else:
            best = max((c for _, c in curves.values() if np.isfinite(c["offset"])), key=lambda c: c["peak_r"], default=None)
            offset, offset_src = (best["offset"], "estimator-speed correlation (no wheel topic)") if best else (float("nan"), "none")
        est_offsets = [c["offset"] for n, (_, c) in curves.items() if not n.startswith("wheel") and np.isfinite(c["offset"])]
        # The wheel-based peak is accepted on its own when it is sharp (r >= 0.9 and at
        # least 0.05 above the best peak more than 1 s away). The estimator-based
        # cross-check is only decisive when the wheel peak is weak: an estimator's
        # speed profile carries its own start-up transient and scale error, so on a
        # constant-speed run it can disagree by a second or more while the wheel and
        # tracker edges coincide exactly (seen on realv2_000/001, 6 October 2026).
        wc = None if wheel_curve is None else wheel_curve[1]
        wheel_sharp = (wc is not None and np.isfinite(wc["peak_r"]) and wc["peak_r"] >= 0.9
                       and (not np.isfinite(wc["second_r"]) or wc["peak_r"] - wc["second_r"] >= 0.05))
        wheel_fair = wc is not None and np.isfinite(wc["peak_r"]) and wc["peak_r"] >= 0.5
        align_ok = (dataset in pinned) or wheel_sharp or (
            wheel_fair and all(abs(o - offset) <= 0.5 for o in est_offsets))
        print(f"  => offset used: {offset:.3f} s ({offset_src}); alignment {'OK' if align_ok else 'UNCERTAIN'}")
        if not np.isfinite(offset):
            print("  [skip] no usable time offset"); continue
        gt_abs_t = t_bag0 + offset + gt["t_rel"]
        np.savetxt(os.path.join(out, "ground_truth_robotclock.csv"),
                   np.column_stack([gt_abs_t * 1e9, gt["xy"]]), delimiter=",", fmt=["%.0f", "%.5f", "%.5f"],
                   header="t_robotclock_ns,x_m,y_m  (tracker frame; offset %.4f s from %s)" % (offset, offset_src))
        with open(os.path.join(out, "time_offset.csv"), "w") as f:
            names = list(curves)
            f.write("offset_s," + ",".join("r_" + n.replace(" ", "_").replace("/", "") for n in names) + "\n")
            ds_ = curves[names[0]][1]["ds"]
            for i in range(len(ds_)):
                f.write(f"{ds_[i]:.3f}," + ",".join(f"{curves[n][1]['r'][i]:.4f}" for n in names) + "\n")

        # ---- score -------------------------------------------------------------
        results = {}
        for key, L in loaded.items():
            label = SYSTEMS[key][0]
            sc = score(gt_abs_t, gt["xy"], L["t"], L["p"], L["q"], L["axes"], lever, args.trim_start)
            sens = {}
            for dd in (-0.2, -0.1, 0.1, 0.2):
                s2 = score(gt_abs_t + dd, gt["xy"], L["t"], L["p"], L["q"], L["axes"], lever, args.trim_start)
                sens[f"{dd:+.1f}"] = s2.get("ate_rmse", float("nan"))
            fit = None
            if args.fit_lever_arm and sc.get("valid"):
                pe3 = np.column_stack([np.interp(sc["t"], L["t"], L["p"][:, k]) for k in range(3)])
                qe = vm.resample_quat(L["t"], L["q"], sc["t"])
                fit = fit_lever_arm(sc["g"], pe3, qe, L["axes"])
            status, reasons = status_of(L["summary"], L["meta"].get("pipeline", key), sc, align_ok,
                                        lever_known, lever_note)
            if not gt_ok:
                status = "failed (ground truth invalid)"
            results[key] = dict(score=sc, status=status, reasons=reasons, sens=sens, fit=fit,
                                speed_grid=L["speed_grid"], **{k: L[k] for k in ("dir", "meta", "summary", "axes", "repair", "t")})
            print(f"  {label}: {status}" + (f"  [{'; '.join(reasons)}]" if reasons else ""))
            if sc.get("valid"):
                r1 = sc["rpe"].get(1.0)
                print(f"       matched {sc['matched']}/{sc['gt_in_span']} GT samples over {sc['matched_duration_s']:.1f} s; "
                      f"ATE RMSE {sc['ate_rmse']:.3f} m, median {sc['ate_median']:.3f}, max {sc['ate_max']:.3f}, final {sc['ate_final']:.3f}; "
                      f"RPE(1 s) {r1['trans_rmse']:.3f} m over {r1['pairs']} pairs" if r1 else "RPE(1 s): no pairs")
                print(f"       path est/gt {sc['est_path_m']:.2f}/{sc['gt_path_m']:.2f} m (ratio {sc['path_ratio']:.2f}); "
                      f"Sim(2) scale diag {sc['scale_diag']:.3f}; yaw align {sc['yaw_align_deg']:.1f} deg; z range {sc['z_range_m']:.2f} m; "
                      f"ATE at offset -0.2/-0.1/+0.1/+0.2 s: " + "/".join(f"{sens[k]:.3f}" for k in ("-0.2", "-0.1", "+0.1", "+0.2")))
                if fit:
                    print(f"       fitted lever arm (diagnostic): forward {fit['forward_m']:+.3f} m, left {fit['left_m']:+.3f} m "
                          f"-> ATE RMSE would be {fit['ate_rmse_with_fit']:.3f} m")
                np.savetxt(os.path.join(out, f"aligned_{key}.csv"),
                           np.column_stack([sc["t"] * 1e9, sc["g"], sc["pa"], sc["err"]]), delimiter=",",
                           fmt=["%.0f"] + ["%.5f"] * 5,
                           header="t_robotclock_ns,gt_x,gt_y,est_x_aligned,est_y_aligned,error_m")
            row = dict(experiment_id=args.experiment_id, domain="real", dataset=dataset, system=label,
                       run=os.path.basename(L["dir"]), status=status, reasons="; ".join(reasons),
                       time_offset_s=offset, offset_source=offset_src, lever_forward_m=lever[0], lever_left_m=lever[1],
                       lever_up_m=lever[2],
                       **{k: sc.get(k) for k in ("gt_in_span", "matched", "unmatched_in_span", "matched_duration_s",
                                                  "ate_rmse", "ate_mean", "ate_median", "ate_max", "ate_final",
                                                  "gt_path_m", "est_path_m", "path_ratio", "scale_diag", "yaw_align_deg",
                                                  "jumps", "biggest_jump_m", "z_range_m", "diverged")},
                       **{f"ate_rmse_offset{k}s": v for k, v in sens.items()},
                       frames_received=L["summary"].get("stereo_pairs_received", L["summary"].get("images_received")),
                       frames_processed=L["summary"].get("stereo_pairs_processed", L["summary"].get("images_processed")),
                       frames_dropped=L["summary"].get("queue_dropped_frames", "0"),
                       resets=L["summary"].get("resets", L["summary"].get("active_map_reset_requests")),
                       tracking_loss_s=L["summary"].get("tracking_loss_duration_s"),
                       init_time_s=L["summary"].get("initialization_time_s"),
                       replay_rate=L["meta"].get("replay_rate"))
            if L.get("repair"):
                rp = L["repair"]; ev = rp["events"] or {}
                row.update(frame_repair=("join+level" if rp["join"] and rp["level"] else "join" if rp["join"] else "level"),
                           tilt_before_deg=rp["tilt_before_deg"], tilt_after_deg=rp["tilt_after_deg"],
                           imu_init_fraction=ev.get("init_fraction"), imu_init_times_s=[x - L["t"][0] for x in ev.get("inits", [])],
                           reset_times_s=[x - L["t"][0] for x, _ in ev.get("resets", [])],
                           reset_reasons=[w for _, w in ev.get("resets", [])],
                           joins_applied=sum(1 for j in rp["joins"] if j["applied"]), segments=len(rp["segments"]))
            if sc.get("valid"):
                row["rpe"] = sc["rpe"]
                r1 = sc["rpe"].get(1.0)
                row["rpe1_trans_rmse"] = r1["trans_rmse"] if r1 else None
                row["rpe1_pairs"] = r1["pairs"] if r1 else 0
                for d, x in sc["rpe"].items():
                    if x:
                        all_rpe.append(dict(dataset=dataset, system=label, **x))
            if fit:
                row.update(fit_lever_forward_m=fit["forward_m"], fit_lever_left_m=fit["left_m"],
                           ate_rmse_with_fitted_lever=fit["ate_rmse_with_fit"])
            all_rows.append(row)

        # ---- figures -------------------------------------------------------------
        gt_prof_abs = gt_fn(grid - offset)
        for th_name in themes:
            th = THEMES[th_name]
            fig_trajectory(os.path.join(out, f"trajectory_{th_name}.png"), th, dataset, gt["xy"], results, gt["n"])
            fig_error(os.path.join(out, f"error_{th_name}.png"), th, dataset, results, t_bag0, wheel_prof, grid, gt_prof_abs)
            fig_offset(os.path.join(out, f"time_offset_{th_name}.png"), th, dataset,
                       {n: (c[0], c[1]) for n, c in curves.items()})
            fig_rpe(os.path.join(out, f"rpe_{th_name}.png"), th, dataset, results)

        # ---- manifest ------------------------------------------------------------
        manifest = dict(
            experiment_id=args.experiment_id, domain="real", dataset=dataset, generated=now,
            command=" ".join(sys.argv), git_superproject=git_rev(ROOT),
            python=platform.python_version(), numpy=np.__version__, host=platform.node(),
            bag=dict(path=os.path.abspath(db3), size_bytes=(os.path.getsize(db3) if os.path.exists(db3) else None),
                     substituted_from=bag_substituted,
                     first_message_robotclock_s=t_bag0, duration_s=bag_dur,
                     message_count=(info or {}).get("message_count"),
                     wheel_topic=WHEEL_TOPIC, wheel_messages=None if wheel is None else wheel["n"]),
            ground_truth=dict(path=gt["path"], sha256=sha256(gt["path"]), header=gt["header"],
                              samples=gt["n"], duration_s=gt["duration"], median_dt_s=gt["gap_median"],
                              max_gap_s=gt["gap_max"], max_step_m=gt["step_max"], path_m=gt["path_len"],
                              first_stamp=gt["first_stamp"], last_stamp=gt["last_stamp"],
                              checks=gt_checks, valid=gt_ok, units="centimetres converted to metres",
                              dimensions=2, heading=False),
            time_alignment=dict(offset_s=offset, source=offset_src, ok=align_ok,
                                curves={n: dict(offset_s=c["offset"], peak_r=c["peak_r"], second_peak_r=c["second_r"])
                                        for n, (_, c) in curves.items()},
                                search_s=OFFSET_SEARCH_S, step_s=OFFSET_STEP_S),
            method=dict(alignment="rigid SE(2), scale fixed (vio_metrics.umeyama on z=0 points)",
                        association=f"estimate linearly interpolated at tracker samples; nearest estimate sample within {MAX_ASSOC_GAP_S} s",
                        rpe="translational, relative displacements in the aligned frame, pairs within 25 % of delta",
                        lever_arm=dict(forward_m=lever[0], left_m=lever[1], up_m=lever[2],
                                       measured=lever_known, assumed=bool(args.lever_arm_assumed)),
                        trim_start_s=args.trim_start, jump_speed_mps=vm.JUMP_SPEED, diverged_m=DIVERGED_M,
                        frame_repair=dict(join_imu_init=args.join_imu_init, level_plane=args.level_plane,
                                          body_up=args.body_up if (args.join_imu_init or args.level_plane) else None,
                                          join_min_angle_deg=JOIN_MIN_ANGLE_DEG, plane_min_poses=PLANE_MIN_POSES,
                                          plane_min_extent_m=PLANE_MIN_EXTENT_M)),
            runs={SYSTEMS[k][0]: dict(dir=r["dir"], status=r["status"], reasons=r["reasons"],
                                      body_axes=None if r["axes"] is None else dict(
                                          forward=r["axes"]["fwd_name"], confidence=r["axes"]["confidence"]),
                                      run_metadata=r["meta"], run_summary=r["summary"], frame_repair=r.get("repair"),
                                      ate_offset_sensitivity=r["sens"], fitted_lever_arm=r["fit"])
                  for k, r in results.items()})
        with open(os.path.join(out, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        with open(os.path.join(out, "metrics.json"), "w") as f:
            json.dump([r for r in all_rows if r["dataset"] == dataset], f, indent=2,
                      default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

    # ---- cross-dataset tables ----------------------------------------------------
    if all_rows:
        import csv
        keys = [k for k in all_rows[0] if k != "rpe"]
        for r in all_rows:
            for k in r:
                if k not in keys and k != "rpe":
                    keys.append(k)
        with open(os.path.join(out_root, "real_gt_metrics.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader()
            for r in all_rows:
                w.writerow(r)
        if all_rpe:
            with open(os.path.join(out_root, "real_gt_rpe_metrics.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(all_rpe[0])); w.writeheader()
                for r in all_rpe:
                    w.writerow(r)
        write_tables(out_root, all_rows, lever_known)
        print(f"\nwrote {out_root}/real_gt_metrics.csv, real_gt_table.tex, real_gt_rpe_table.tex")
    print("\nstatus summary:")
    for r in all_rows:
        print(f"  {r['dataset']:24s} {r['system']:14s} {r['status']}")


if __name__ == "__main__":
    main()
