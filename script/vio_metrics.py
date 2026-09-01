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


def sources(dataset):
    return [("VINS-Fusion", os.path.join(ROOT, "output/output_vins", dataset, "vio.csv")),
            ("ORB-SLAM3", os.path.join(ROOT, "output/output_orb", dataset, "vio.csv"))]


def gt_candidates(dataset):
    return [os.path.join(ROOT, "output/output_vins", dataset, "ground_truth.csv"),
            os.path.join(ROOT, "output/output_orb", dataset, "ground_truth.csv")]


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


def num(x, w=10, prec=2):
    """Number in at most w characters. A fully diverged run reports distances of 1e13 m;
    plain %f then blows the column apart and welds the whole row into one token. Keep the
    normal fixed-point form and fall back to %g only when it will not fit."""
    t = f"{x:.{prec}f}"
    if len(t) > w:
        t = f"{x:.{max(w - 6, 1)}g}"
    return f"{t:>{w}}"


def analyse(dataset, log=lambda *_: None):
    """Load a dataset and align every estimate to the reference.

    Returns (rows, aligned, info) or None when nothing loads. `rows` is the per-system
    metric tuple the reports print; `aligned` carries the resampled series the plots
    draw; `info` holds the reference identity and the ground-truth arrays.
    """
    runs = []
    for name, path in sources(dataset):
        r = load(path)
        if r is None:
            log(f"  [skip] {name}: no usable data at {path}")
            continue
        runs.append([name, *r])
    if not runs:
        return None

    gt = None
    for c in gt_candidates(dataset):
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
    else:
        ref_name, ref_t, ref_p = runs[0][0], runs[0][1], runs[0][2]
        ref_v = ref_eul = None
        log(f"  no ground truth; using {ref_name} as the reference frame")

    rows, aligned = [], []
    for name, t, p, q, v in runs:
        lo, hi = max(t[0], ref_t[0]), min(t[-1], ref_t[-1])
        m = (t >= lo) & (t <= hi)
        if m.sum() < 10:
            log(f"  [skip] {name}: only {m.sum()} samples overlap the reference")
            continue
        tc, pc, qc, vc = t[m], p[m], q[m], v[m]
        rp = resample(ref_t, ref_p, tc)
        R, tr = umeyama(pc, rp)
        pa = (R @ pc.T).T + tr
        # The same R that puts positions in the reference frame puts velocities and
        # orientations there too -- without it, "yaw" is measured from whichever
        # direction each estimator happened to be facing when it initialised.
        va = (R @ vc.T).T
        eul = euler_zyx(R @ quat_to_R(qc))
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
            diverged=gt is not None and err.max() > DIVERGED_ERROR,
            refv=None if ref_v is None else resample(ref_t, ref_v, tc),
            refeul=None if ref_eul is None else resample_deg(ref_t, ref_eul, tc)))
        biggest = steps[jumps].max() if len(jumps) else 0.0
        rows.append((name, len(tc), tc[-1] - tc[0], path_len, ref_len,
                     np.sqrt((err**2).mean()), err.max(), err[-1], jumps, clean_len, seg,
                     biggest))

    info = dict(gt=gt, ref_name=ref_name, ref_t=ref_t, ref_p=ref_p,
                lbl="ATE rms" if gt is not None else "dev rms")
    return rows, aligned, info
