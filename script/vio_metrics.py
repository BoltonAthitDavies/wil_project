#!/usr/bin/env python3
"""Trajectory metrics shared by plot_compare.py (one dataset) and plot_summary.py (all).

Kept in one place deliberately. These numbers appear in three outputs -- the console
report, the table baked into each comparison PNG, and the cross-dataset summary -- and
the moment the computation is duplicated they are free to disagree with each other.

WHY SE(3) ALIGNMENT
    Every trajectory is aligned to the reference with a full SE(3) Umeyama fit over the
    whole overlap, scale FIXED. That is the standard ATE convention: it removes the
    arbitrary choice of world frame, which is not a property of the estimator, while
    leaving drift and scale error fully visible, which are. It matters most for
    ORB-SLAM3, which re-bases its map when the IMU initialises and whose world frame is
    the first keyframe -- anchoring on a single pose of that produces a ~90 deg yaw
    error against a trajectory that is otherwise sound.
"""

import glob
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A ground robot cannot move this fast. Anything above it is a tracking jump --
# relocalisation into the wrong place, or a map merge -- not motion.
JUMP_SPEED = 10.0             # m/s

# Past this the run is not "inaccurate", it is gone -- indoors, a hundred metres of
# error means the estimate has no relationship to where the robot was.
DIVERGED_ERROR = 50.0         # m

# The sim car tops out at 4.0 m/s and the real rig crawls. Needs no reference frame,
# which is what makes it usable on the bags that have no ground truth.
PLAUSIBLE_MEAN_SPEED = 3.0    # m/s

# (display label, output tree) for every system a dataset might carry. SYSTEMS is the
# default pair every script uses unless told otherwise. SYSTEMS_YOLO adds the
# YOLO-masked variants -- run separately, on a handful of datasets, to test whether
# masking out detected dynamic objects before tracking recovers what plain VINS/ORB
# lose on those scenes. Any caller can pass its own `systems` list; nothing below
# assumes there are exactly two.
SYSTEMS = [("VINS-Fusion", "output_vins"), ("ORB-SLAM3", "output_orb")]
SYSTEMS_YOLO = SYSTEMS + [("VINS-Fusion+YOLO", "output_vins_yolo"),
                          ("ORB-SLAM3+YOLO", "output_orb_yolo")]
# The non-visual baselines from proprio_estimator.py. Separate from SYSTEMS because
# they answer a different question: not "which VIO is better" but "does either VIO
# beat doing no vision at all". Written in the same vio.csv format precisely so that
# every metric here applies to them unchanged.
SYSTEMS_PROPRIO = [("Wheel odometry", "output_wheel"),
                   ("IMU dead reckoning", "output_imu"),
                   ("EKF wheel+IMU", "output_ekf")]


def run_dir(tree, dataset, logging="never"):
    """Directory holding this system's artifacts for `dataset`.

    `logging` is "never", "auto", or an explicit run name such as
    "logging_20260916_02". An explicit name is the only safe choice once a dataset
    holds more than one run of the same estimator: "auto" takes the newest, so the
    day a second run appears every previously published number for that system
    changes underneath the report without anything in the call site changing.

    Two layouts coexist in output/. The estimator launch files write one `logging_*`
    directory per run, so the artifacts sit at <tree>/<dataset>/logging_<stamp>/;
    proprio_estimator.py and the older relogged copies write flat, at
    <tree>/<dataset>/. Callers that know they are reading the newer per-run layout
    pass logging="auto" and get the newest `logging_*` that actually contains a
    vio.csv.

    Default "never" on purpose: silently reaching into a run directory would make
    runs appear in plot_compare.py and plot_summary.py that those scripts have never
    reported before, changing published figures as a side effect of a helper added
    for something else. Opting in keeps that an explicit decision per caller.
    """
    base = os.path.join(ROOT, "output", tree, dataset)
    if logging not in ("never", "auto"):
        return os.path.join(base, logging)
    if logging == "auto" and not os.path.exists(os.path.join(base, "vio.csv")):
        runs = sorted(d for d in glob.glob(os.path.join(base, "logging_*"))
                      if os.path.exists(os.path.join(d, "vio.csv")))
        if runs:
            return runs[-1]
    return base


def split(entry):
    """A systems entry is (label, tree) or (label, tree, run).

    The optional third field pins one `logging_*` directory for that system, which
    is what lets two runs of the same estimator -- for instance ORB-SLAM3 with and
    without loop closure -- sit in one comparison as separate rows.
    """
    if len(entry) == 3:
        return entry[0], entry[1], entry[2]
    return entry[0], entry[1], None


def sources(dataset, systems=SYSTEMS, logging="never"):
    out = []
    for e in systems:
        label, tree, run = split(e)
        out.append((label, os.path.join(run_dir(tree, dataset, run or logging),
                                        "vio.csv")))
    return out


def gt_candidates(dataset, systems=SYSTEMS):
    # Order matters: the first tree that has ground_truth.csv wins. YOLO trees are
    # never the extraction target (extract_gt.py always writes under output_vins/),
    # so for SYSTEMS_YOLO this just checks the same two trees SYSTEMS would, plus two
    # that in practice never hold a copy -- harmless, and keeps this generic rather
    # than special-casing YOLO trees out.
    seen = dict.fromkeys(split(e)[1] for e in systems)
    return [os.path.join(ROOT, "output", tree, dataset, "ground_truth.csv")
            for tree in seen]


def discover(prefix="", contains=None, exclude=None, systems=SYSTEMS):
    """Every dataset with at least one vio.csv, as 'group/name'.

    Discovery beats a hand-kept list: a run shows up in every report as soon as it is
    processed, and one that is deleted stops being reported instead of erroring.

    `prefix` matches from the start of 'group/name' (e.g. "simulation/" for one tree).
    `contains` matches a substring anywhere in the name -- for pulling out one family
    of runs (e.g. "nofloortexture") regardless of which group they fall under.
    `exclude` is the converse: drop anything containing this substring, for keeping a
    family that is being reported separately out of the combined view.
    `systems` controls which output trees are scanned -- pass SYSTEMS_YOLO to also
    find datasets that only have a YOLO-masked run.
    """
    found = set()
    for tree in dict.fromkeys(split(e)[1] for e in systems):
        base = os.path.join(ROOT, "output", tree)
        if not os.path.isdir(base):
            continue
        for group in sorted(os.listdir(base)):
            gdir = os.path.join(base, group)
            if not os.path.isdir(gdir):
                continue
            for name in sorted(os.listdir(gdir)):
                if os.path.exists(os.path.join(gdir, name, "vio.csv")):
                    found.add(f"{group}/{name}")
    return sorted(d for d in found
                  if d.startswith(prefix) and (contains is None or contains in d)
                  and (exclude is None or exclude not in d))


def load(path):
    """VINS-format CSV -> (t_sec, position, quaternion_wxyz, velocity).

    Velocity is columns 9-11 where the writer emitted them, zeros otherwise.
    NOTE the frames differ by source and are reconciled in analyse(), not here:
    both estimators write velocity in their own WORLD frame, while the ground truth
    comes from a nav_msgs/Odometry twist, which REP-145 puts in the BODY frame of
    child_frame_id. Comparing the raw columns would put forward/lateral speed against
    east/north speed. None if absent/empty.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    d = np.loadtxt(path, delimiter=",")
    if d.ndim == 1:
        d = d.reshape(1, -1)
    if len(d) < 2:
        return None
    v = d[:, 8:11] if d.shape[1] >= 11 else np.zeros((len(d), 3))
    return d[:, 0] / 1e9, d[:, 1:4], d[:, 4:8], v


def quat_to_R(q):
    """(N,4) quaternions wxyz -> (N,3,3) rotation matrices."""
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def euler_zyx(R):
    """(N,3,3) -> (yaw, pitch, roll) in degrees, wrapped to (-180, 180].

    Deliberately NOT unwrapped. Unwrapping picks a branch from each series' own first
    sample, so two systems holding the SAME heading across a loop end up hundreds of
    degrees apart -- one wound to +270, the other to -90 -- showing a disagreement that
    does not exist. Wrapped, they overlay; the cost is a sawtooth at the wrap.
    """
    yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    pitch = np.arcsin(np.clip(-R[:, 2, 0], -1.0, 1.0))
    roll = np.arctan2(R[:, 2, 1], R[:, 2, 2])
    return np.degrees(np.stack([yaw, pitch, roll], axis=1))


def umeyama(src, dst):
    """Rigid SE(3) fit mapping src onto dst, scale fixed at 1. Returns (R, t)."""
    cs, cd = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - cs).T @ (dst - cd))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cd - R @ cs


def find_jumps(t, p):
    """Indices i where the step i -> i+1 implies an impossible speed."""
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    return np.where(d / np.maximum(np.diff(t), 1e-9) > JUMP_SPEED)[0], d


def longest_clean(t, p):
    """Slice of the longest run of poses containing no jump."""
    jumps, _ = find_jumps(t, p)
    cuts = [0, *(jumps + 1), len(p)]
    segs = [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b - a > 30]
    return max(segs, key=lambda s: s[1] - s[0]) if segs else (0, len(p))


def resample(t_src, p_src, t_dst):
    return np.stack([np.interp(t_dst, t_src, p_src[:, i]) for i in range(3)], axis=1)


def resample_deg(t_src, a_src, t_dst):
    """Resample wrapped angles. Interpolating straight across the +-180 seam would
    invent a full-range ramp between two samples that are actually 1 deg apart, so
    unwrap first and re-wrap after."""
    a = np.stack([np.unwrap(np.radians(a_src[:, i])) for i in range(3)], axis=1)
    out = resample(t_src, a, t_dst)
    return np.degrees((out + np.pi) % (2 * np.pi) - np.pi)


def resample_quat(t_src, q_src, t_dst):
    """Resample orientation onto t_dst by nlerp (linear interpolation + renormalise).

    Quaternions need one more step than resample_deg: q and -q represent the SAME
    rotation, so if the writer's sign convention flips between adjacent samples, linear
    interpolation swings the short way through the wrong hemisphere. Fix continuity
    first -- flip each sample so it lands in the same hemisphere as its predecessor --
    then interpolate. nlerp is not the constant-angular-velocity slerp, but at pose
    rates of 10+ Hz against inter-frame rotations of a few degrees the difference is
    far below the geodesic error this feeds into.
    """
    q = q_src.copy()
    flip = np.cumsum(np.sum(q[1:] * q[:-1], axis=1) < 0) % 2
    q[1:][flip.astype(bool)] *= -1
    out = np.stack([np.interp(t_dst, t_src, q[:, i]) for i in range(4)], axis=1)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def geodesic_deg(Ra, Rb):
    """(N,3,3), (N,3,3) -> (N,) angle in degrees between corresponding rotations.

    The rotational counterpart of ATE's position error: not a per-axis (yaw/pitch/roll)
    difference, which breaks down near gimbal singularities and double-counts a single
    tilt as error on two axes, but the single angle of the rotation that takes one
    orientation to the other -- via the standard trace formula on R_a^T @ R_b.
    """
    Rd = np.einsum("nji,njk->nik", Ra, Rb)
    tr = Rd[:, 0, 0] + Rd[:, 1, 1] + Rd[:, 2, 2]
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1.0, 1.0)))


# Intervals the RPE sweep reports. 1.0 s is the headline -- about 1.4 m at the sim
# robot's cruising speed -- and the rest show how error accumulates with interval.
RPE_DELTAS = (0.5, 1.0, 2.0, 5.0)


def rpe(t, p_est, q_est, p_ref, q_ref, delta_s=1.0, tol=0.25, jump_idx=None):
    """Relative pose error at a fixed time interval (Kuemmerle/TUM convention).

    For each pair (i, j) separated by delta_s seconds, the error transform is
    E = (Q_i^-1 Q_j)^-1 (P_i^-1 P_j), where P is the estimate and Q the reference.
    Translation error is ||trans(E)||; rotation error is the geodesic angle of
    rot(E), via geodesic_deg().

    RPE needs NO global alignment, which is the whole point of reporting it here.
    ATE on these runs is dominated by teleports: one discontinuity re-seats the
    rigid fit and every subsequent pose inherits the offset, so the headline number
    describes the jump rather than the tracking. RPE asks a local question instead
    -- over the next delta_s seconds, did the estimate move the way the reference
    moved -- and is immune to both the global frame and accumulated drift. It is
    the standard-named replacement for this module's longest-clean-segment
    heuristic, not an addition to it.

    A pair is accepted only when its realised separation is within `tol` of
    delta_s, so a gap in the log cannot silently widen the interval and inflate the
    error. Pairs are overlapping (every index starts one), which is the TUM default;
    `pairs` reports how many survived.

    jump_idx: indices from find_jumps(). When given, the returned dict also carries
    `*_clean` statistics computed over only those pairs that do not span a jump.
    The difference between the two isolates the teleport contribution from ordinary
    local error.

    Returns a dict, or None when no pair satisfies the interval.
    """
    t = np.asarray(t, float)
    if len(t) < 2:
        return None
    j = np.searchsorted(t, t + delta_s, side="left")
    i = np.arange(len(t))
    keep = j < len(t)
    i, j = i[keep], j[keep]
    if len(i) == 0:
        return None
    ok = np.abs((t[j] - t[i]) - delta_s) <= tol * delta_s
    i, j = i[ok], j[ok]
    if len(i) == 0:
        return None

    Re, Rr = quat_to_R(np.asarray(q_est)), quat_to_R(np.asarray(q_ref))
    pe, pr = np.asarray(p_est, float), np.asarray(p_ref, float)

    # Relative motion of each trajectory, expressed in its own frame at i.
    rel_e = np.einsum("nji,nj->ni", Re[i], pe[j] - pe[i])
    rel_r = np.einsum("nji,nj->ni", Rr[i], pr[j] - pr[i])
    # ||trans(E)|| reduces to this: E's translation is an orthonormal rotation of
    # (rel_e - rel_r), and rotation preserves norm.
    e_trans = np.linalg.norm(rel_e - rel_r, axis=1)

    rot_e = np.einsum("nji,njk->nik", Re[i], Re[j])
    rot_r = np.einsum("nji,njk->nik", Rr[i], Rr[j])
    e_rot = geodesic_deg(rot_r, rot_e)

    out = dict(delta_s=delta_s, pairs=int(len(i)),
               trans_rmse=float(np.sqrt((e_trans**2).mean())),
               trans_median=float(np.median(e_trans)),
               rot_rmse=float(np.sqrt((e_rot**2).mean())),
               rot_median=float(np.median(e_rot)))

    if jump_idx is None or len(jump_idx) == 0:
        # With no jumps the clean subset is the whole set; say so explicitly rather
        # than leaving the caller to decide what a missing field means.
        out.update(pairs_clean=out["pairs"], trans_rmse_clean=out["trans_rmse"],
                   trans_median_clean=out["trans_median"],
                   rot_rmse_clean=out["rot_rmse"], rot_median_clean=out["rot_median"])
        return out

    # A jump at index k is the step k -> k+1, so a pair (i, j) spans it when
    # i <= k < j. Counting jumps in each prefix makes that an O(n) test.
    spans = np.zeros(len(t), int)
    spans[np.asarray(jump_idx, int) + 1] = 1
    before = np.cumsum(spans)
    clean = before[j] == before[i]
    if not clean.any():
        out.update(pairs_clean=0, trans_rmse_clean=float("nan"),
                   trans_median_clean=float("nan"), rot_rmse_clean=float("nan"),
                   rot_median_clean=float("nan"))
        return out
    ct, cr = e_trans[clean], e_rot[clean]
    out.update(pairs_clean=int(clean.sum()),
               trans_rmse_clean=float(np.sqrt((ct**2).mean())),
               trans_median_clean=float(np.median(ct)),
               rot_rmse_clean=float(np.sqrt((cr**2).mean())),
               rot_median_clean=float(np.median(cr)))
    return out


def num(x, w=10, prec=2):
    """Number in at most w characters. A fully diverged run reports distances of 1e13 m;
    plain %f then blows the column apart and welds the whole row into one token. Keep the
    normal fixed-point form and fall back to %g only when it will not fit."""
    t = f"{x:.{prec}f}"
    if len(t) > w:
        t = f"{x:.{max(w - 6, 1)}g}"
    return f"{t:>{w}}"


def analyse(dataset, log=lambda *_: None, trim_start=0.0, systems=SYSTEMS,
            logging="never"):
    """Load a dataset and align every estimate to the reference.

    trim_start: seconds to discard from the FRONT of each estimate's own overlap with
    the reference, before any metric is computed -- position, rotation, jumps, path
    length, all of it. This is a per-estimator cut, not a per-dataset one: VINS and
    ORB-SLAM3 initialise at different wall-clock offsets and take different lengths of
    time to settle, so "the first 5 seconds" means the first 5 seconds each estimator
    had data, not 5 seconds of the bag. The point is to separate "this run is bad at
    steady state" from "this run needed a few seconds to initialise" -- the two look
    identical in headline ATE but call for different fixes.

    Returns (rows, aligned, info) or None when nothing loads. `rows` is a list of
    per-system metric dicts (one per system that had usable, overlapping data) --
    a dict rather than a positional tuple so a field can be added here without every
    caller's unpacking breaking. `aligned` carries the resampled series the plots draw.
    `info` holds the reference identity and the ground-truth arrays.

    Each row dict has: name, poses, secs, path, ref, rms, max, final, jumps (array of
    jump indices), clean_len, seg (longest-clean-segment summary or None), biggest
    (largest single jump, metres), rot_rms, rot_max (rotational error in degrees).
    """
    runs = []
    for name, path in sources(dataset, systems, logging):
        r = load(path)
        if r is None:
            log(f"  [skip] {name}: no usable data at {path}")
            continue
        runs.append([name, *r])
    if not runs:
        return None

    gt = None
    for c in gt_candidates(dataset, systems):
        gt = load(c)
        if gt is not None:
            log(f"  ground truth: {c}")
            break

    # Reference frame: ground truth when we have it, otherwise VINS -- on the real bags
    # the point is agreement between the two, and VINS is the established baseline.
    if gt is not None:
        ref_name, ref_t, ref_p = "ground truth", gt[0], gt[1]
        ref_R = quat_to_R(gt[2])
        # Ground truth arrives as a body-frame twist (see load()), so it needs its own
        # orientation applied to become the world-frame velocity the estimators report.
        ref_v = np.einsum("nij,nj->ni", ref_R, gt[3])
        ref_eul = euler_zyx(ref_R)
        ref_q = gt[2]
    else:
        ref_name, ref_t, ref_p = runs[0][0], runs[0][1], runs[0][2]
        ref_v = ref_eul = None
        # Rotational error follows the same "no GT -> compare against the first
        # estimator" convention translational error already uses (the console's
        # "dev rms" label): with nothing to call truth, the two systems' mutual
        # orientation agreement is the only rotational signal available.
        ref_q = runs[0][3]
        log(f"  no ground truth; using {ref_name} as the reference frame")

    rows, aligned = [], []
    for name, t, p, q, v in runs:
        lo, hi = max(t[0], ref_t[0]), min(t[-1], ref_t[-1])
        m = (t >= lo) & (t <= hi)
        if m.sum() < 10:
            log(f"  [skip] {name}: only {m.sum()} samples overlap the reference")
            continue
        tc, pc, qc, vc = t[m], p[m], q[m], v[m]
        if trim_start > 0:
            # Cut from THIS run's own first overlapping sample, not from the bag's
            # t=0 -- an estimator that only starts publishing at t=8s has no "first
            # 5 seconds" before that to discard.
            keep = tc >= tc[0] + trim_start
            if keep.sum() < 10:
                log(f"  [skip] {name}: only {keep.sum()} samples remain after "
                    f"trimming the first {trim_start:.1f}s")
                continue
            tc, pc, qc, vc = tc[keep], pc[keep], qc[keep], vc[keep]
        rp = resample(ref_t, ref_p, tc)
        R, tr = umeyama(pc, rp)
        pa = (R @ pc.T).T + tr
        # The same R that puts positions in the reference frame puts velocities and
        # orientations there too -- without it, "yaw" is measured from whichever
        # direction each estimator happened to be facing when it initialised.
        va = (R @ vc.T).T
        Ra = R @ quat_to_R(qc)
        eul = euler_zyx(Ra)
        # Rotational error: the geodesic angle between this estimate's orientation
        # (aligned into the reference frame) and the reference's own orientation at
        # the same instant -- see geodesic_deg(). Bounded to [0, 180] regardless of
        # how badly position has diverged, which is what lets it stay meaningful even
        # on a run whose translational ATE has blown up to nonsense.
        rot_err = geodesic_deg(Ra, quat_to_R(resample_quat(ref_t, ref_q, tc)))
        err = np.linalg.norm(pa - rp, axis=1)
        path_len = np.linalg.norm(np.diff(pc, axis=0), axis=1).sum()
        ref_len = np.linalg.norm(np.diff(rp, axis=0), axis=1).sum()
        # One teleport wrecks both the path length and the global fit, which then reads
        # as "diverged" for a run that was otherwise tracking well. Measure the clean
        # part separately rather than letting a single outlier set the headline.
        jumps, steps = find_jumps(tc, pc)
        clean_len = steps[steps / np.maximum(np.diff(tc), 1e-9) <= JUMP_SPEED].sum()
        a, b = longest_clean(tc, pc)
        ps, rs = pc[a:b], rp[a:b]
        if b - a > 30:
            Rc, trc = umeyama(ps, rs)
            ec = np.linalg.norm((Rc @ ps.T).T + trc - rs, axis=1)
            seg = (b - a, tc[b - 1] - tc[a], np.sqrt((ec**2).mean()))
        else:
            seg = None
        # Divergence is only meaningful against ground truth. With no GT the reference is
        # just whichever estimator ran first, and if THAT one is the broken one, the error
        # is large for the healthy run -- blaming it would be exactly backwards.
        aligned.append(dict(
            name=name, t=tc - tc[0], p=pa, ref=rp, err=err, v=va, eul=eul,
            roterr=rot_err, diverged=gt is not None and err.max() > DIVERGED_ERROR,
            refv=None if ref_v is None else resample(ref_t, ref_v, tc),
            refeul=None if ref_eul is None else resample_deg(ref_t, ref_eul, tc)))
        biggest = steps[jumps].max() if len(jumps) else 0.0
        rows.append(dict(
            name=name, poses=len(tc), secs=tc[-1] - tc[0], path=path_len, ref=ref_len,
            rms=np.sqrt((err**2).mean()), max=err.max(), final=err[-1], jumps=jumps,
            clean_len=clean_len, seg=seg, biggest=biggest,
            rot_rms=np.sqrt((rot_err**2).mean()), rot_max=rot_err.max()))

    info = dict(gt=gt, ref_name=ref_name, ref_t=ref_t, ref_p=ref_p,
                lbl="ATE rms" if gt is not None else "dev rms")
    return rows, aligned, info
