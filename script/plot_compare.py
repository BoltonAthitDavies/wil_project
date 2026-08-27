#!/usr/bin/env python3
"""Compare VINS-Fusion against ORB-SLAM3 on the same run, with ground truth if there is any.

    python3 script/plot_compare.py simulation/sim
    python3 script/plot_compare.py real/rosbag_realsense_imu_cambaseline95mm

The argument is a dataset path relative to output/output_vins and output/output_orb,
which are parallel trees. Ground truth is picked up from either side if present.
Figures land in output/compare/<dataset>/.

WHY NOT plot_vio_vs_gt.py
    That script compares ONE estimate against ground truth and anchors its alignment on
    the first pose. That is right for VINS, whose world frame is fixed from the start,
    but wrong for ORB-SLAM3, which re-bases its map when the IMU initialises and whose
    world frame is the first keyframe -- gravity-aligned in inertial mode, raw camera
    optical axes in pure stereo. Anchoring on one pose of that produces a ~90 deg yaw
    error against a trajectory that is otherwise sound.

    So every trajectory here is aligned to the reference with a full SE(3) Umeyama fit
    over the whole overlap, scale FIXED. That is the standard ATE convention: it removes
    the arbitrary choice of world frame, which is not a property of the estimator, while
    leaving drift and scale error fully visible, which are.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = sys.argv[1] if len(sys.argv) > 1 else "simulation/sim"
OUTDIR = os.path.join(ROOT, "output", "compare", DATASET)

SOURCES = [("VINS-Fusion", os.path.join(ROOT, "output/output_vins", DATASET, "vio.csv")),
           ("ORB-SLAM3",   os.path.join(ROOT, "output/output_orb",  DATASET, "vio.csv"))]
GT_CANDIDATES = [os.path.join(ROOT, "output/output_vins", DATASET, "ground_truth.csv"),
                 os.path.join(ROOT, "output/output_orb",  DATASET, "ground_truth.csv")]

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", ink3="#8a8983",
                  grid="#e3e2dd", series=("#1baf7a", "#2a78d6", "#eb6834")),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", ink3="#87867d",
                  grid="#33332f", series=("#199e70", "#3987e5", "#d95926")),
}


def load(path):
    """VINS-format CSV -> (t_sec, position, quaternion_wxyz, velocity).

    Velocity is columns 9-11 where the writer emitted them, zeros otherwise.
    NOTE the frames differ by source and are reconciled below, not here:
    both estimators write velocity in their own WORLD frame, while the ground
    truth comes from a nav_msgs/Odometry twist, which REP-145 puts in the BODY
    frame of child_frame_id. Plotting the raw columns against each other would
    compare forward/lateral speed against east/north speed.
    None if absent/empty.
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

    Deliberately NOT unwrapped. Unwrapping picks a branch from each series'
    own first sample, so two systems holding the SAME heading across a loop end
    up hundreds of degrees apart on the plot -- one wound to +270, the other to
    -90 -- and the panel shows a disagreement that does not exist. Wrapped, they
    overlay; the cost is a sawtooth at the wrap, which plot_wrapped() hides.
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


# A ground robot cannot move this fast. Anything above it is a tracking jump --
# relocalisation into the wrong place, or a map merge -- not motion.
JUMP_SPEED = 10.0     # m/s


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


# --- load --------------------------------------------------------------------
runs = []
for name, path in SOURCES:
    r = load(path)
    if r is None:
        print(f"  [skip] {name}: no usable data at {path}")
        continue
    runs.append([name, *r])
if not runs:
    sys.exit(f"no trajectories found for '{DATASET}'")

gt = None
for c in GT_CANDIDATES:
    gt = load(c)
    if gt is not None:
        print(f"  ground truth: {c}")
        break

# Reference frame: ground truth when we have it, otherwise VINS -- the point on the real
# bags is agreement between the two estimators, and VINS is the established baseline.
if gt is not None:
    ref_name, ref_t, ref_p = "ground truth", gt[0], gt[1]
else:
    ref_name, ref_t, ref_p = runs[0][0], runs[0][1], runs[0][2]
    print(f"  no ground truth; using {ref_name} as the reference frame")

# Reference velocity and attitude, expressed in the reference frame itself.
# Ground truth arrives as a body-frame twist (see load()), so it needs its own
# orientation applied to become the world-frame velocity the estimators report.
if gt is not None:
    ref_q, ref_v_body = gt[2], gt[3]
    ref_R = quat_to_R(ref_q)
    ref_v = np.einsum("nij,nj->ni", ref_R, ref_v_body)
    ref_eul = euler_zyx(ref_R)
else:
    ref_v = ref_eul = None

# --- align every trajectory to the reference over their common window --------
rows, aligned = [], []
for name, t, p, q, v in runs:
    lo, hi = max(t[0], ref_t[0]), min(t[-1], ref_t[-1])
    m = (t >= lo) & (t <= hi)
    if m.sum() < 10:
        print(f"  [skip] {name}: only {m.sum()} samples overlap the reference")
        continue
    tc, pc, qc, vc = t[m], p[m], q[m], v[m]
    rp = resample(ref_t, ref_p, tc)
    R, tr = umeyama(pc, rp)
    pa = (R @ pc.T).T + tr
    # The same R that puts positions in the reference frame puts velocities and
    # orientations there too -- without it, "yaw" is measured from whichever
    # direction each estimator happened to be facing when it initialised, and the
    # attitude panel compares nothing.
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
    aligned.append(dict(name=name, t=tc - tc[0], p=pa, ref=rp, err=err, v=va, eul=eul,
                        refv=None if ref_v is None else resample(ref_t, ref_v, tc),
                        refeul=None if ref_eul is None else resample_deg(ref_t, ref_eul, tc)))
    biggest = steps[jumps].max() if len(jumps) else 0.0
    rows.append((name, len(tc), tc[-1] - tc[0], path_len, ref_len,
                 np.sqrt((err**2).mean()), err.max(), err[-1], jumps, clean_len, seg,
                 biggest))

# --- report ------------------------------------------------------------------
lbl = "ATE rms" if gt is not None else "dev rms"
print(f"\n  {DATASET}   reference: {ref_name}\n")
print(f"  {'system':<14}{'poses':>7}{'secs':>8}{'path m':>10}{'ref m':>10}"
      f"{lbl:>11}{'max':>10}{'final':>10}")
for n, c, sec, pl, rl, rms, mx, fin, jm, cl, seg, bg in rows:
    self_ref = (gt is None and n == ref_name)
    cols = (f"{'--':>11}{'--':>10}{'--':>10}" if self_ref
            else f"{rms:>11.3f}{mx:>10.3f}{fin:>10.3f}")
    print(f"  {n:<14}{c:>7}{sec:>8.1f}{pl:>10.2f}{rl:>10.2f}{cols}"
          + ("   (reference)" if self_ref else ""))

# Plausibility first, and WITHOUT reference to the other system. On the real bags
# there is no ground truth, so the reference is just whichever estimator came first --
# and if that one has diverged, every "error vs reference" number is backwards, blaming
# the healthy run. Implied mean speed needs no reference and no common frame: a ground
# robot indoors does not average tens of m/s, whoever says otherwise is the broken one.
PLAUSIBLE_MEAN_SPEED = 3.0     # m/s; the sim car tops out at 4.0 and the real rig crawls
print()
for n, c, sec, pl, rl, rms, mx, fin, jm, cl, seg, bg in rows:
    v = pl / sec if sec > 0 else 0
    flag = "  <-- IMPLAUSIBLE" if v > PLAUSIBLE_MEAN_SPEED else ""
    print(f"  {n:<14} implied mean speed {v:7.2f} m/s over {sec:.0f}s{flag}")

# Distinguish the two ways a run goes wrong. A few impossible-speed steps are
# tracking JUMPS, and the numbers above understate a run that was otherwise fine.
# A long path with no jumps is genuine DIVERGENCE, and the numbers are real.
for n, c, sec, pl, rl, rms, mx, fin, jm, cl, seg, bg in rows:
    if len(jm):
        print(f"\n  [{n}] {len(jm)} tracking jump(s) >{JUMP_SPEED:.0f} m/s "
              f"(largest {bg:.1f} m).")
        print(f"        path excluding jumps: {cl:.2f} m   vs reference {rl:.2f} m")
        if seg:
            print(f"        longest clean segment: {seg[0]} poses / {seg[1]:.1f}s, "
                  f"{lbl} {seg[2]:.3f} m")
        print(f"        The table's {lbl} is set by the jump, not by tracking quality.")
    elif gt is not None and rl > 0 and (pl / rl > 3 or rl / pl > 3):
        print(f"\n  [warn] {n} path is {pl:.0f} m against {rl:.0f} m of GROUND TRUTH "
              f"({pl/rl:.1f}x) with no jumps -- genuine divergence.")

# --- plot --------------------------------------------------------------------
os.makedirs(OUTDIR, exist_ok=True)


def render(theme_name):
    th = THEMES[theme_name]
    fig = plt.figure(figsize=(14, 11.5), facecolor=th["surface"])
    outer = GridSpec(2, 1, figure=fig, height_ratios=[1.55, 1], hspace=0.28,
                     left=0.06, right=0.98, top=0.935, bottom=0.06)
    gs = outer[0].subgridspec(2, 2, width_ratios=[1.25, 1], hspace=0.32, wspace=0.22)
    bottom = outer[1].subgridspec(1, 3, wspace=0.26)

    def style(ax, xl, yl, title):
        ax.set_facecolor(th["surface"])
        ax.grid(True, color=th["grid"], lw=0.7, zorder=0)
        for s in ax.spines.values():
            s.set_color(th["grid"])
        ax.tick_params(colors=th["ink3"], labelsize=9)
        ax.set_xlabel(xl, color=th["ink2"], fontsize=9)
        ax.set_ylabel(yl, color=th["ink2"], fontsize=9)
        ax.set_title(title, color=th["ink"], fontsize=11, loc="left", pad=8)

    # top-down XY, the headline panel.
    # With no ground truth we cannot claim a common frame, and if one estimator has
    # diverged, plotting both on shared axes shrinks the good one to a dot. So give
    # each its own panel, each self-centred and auto-scaled.
    if gt is None and len(aligned) > 1:
        sub = gs[:, 0].subgridspec(len(runs), 1, hspace=0.38)
        for i, a in enumerate(aligned):
            name, praw = a["name"], a["p"]
            axi = fig.add_subplot(sub[i])
            pc = praw - praw[0]
            axi.plot(pc[:, 0], pc[:, 1], color=th["series"][i + 1], lw=1.5)
            axi.plot(0, 0, "o", color=th["series"][i + 1], ms=5)
            axi.set_aspect("equal", adjustable="datalim")
            span = np.linalg.norm(np.diff(praw, axis=0), axis=1).sum()
            style(axi, "x [m]", "y [m]", f"{name}   (path {span:.1f} m, own frame)")
        ax = None
    else:
        ax = fig.add_subplot(gs[:, 0])
    if ax is not None and aligned:
        rp0 = aligned[0]["ref"]
        ax.plot(rp0[:, 0], rp0[:, 1], color=th["series"][0], lw=3.0, alpha=0.55,
                label=ref_name, zorder=1)
    if ax is not None:
        for i, a in enumerate(aligned):
            name, pa = a["name"], a["p"]
            ax.plot(pa[:, 0], pa[:, 1], color=th["series"][i + 1], lw=1.6,
                    label=name, zorder=3 + i)
            ax.plot(pa[0, 0], pa[0, 1], "o", color=th["series"][i + 1], ms=5, zorder=6)
        # A run that has diverged by hundreds of metres would set the axes for
        # everyone, shrinking ground truth and the healthy estimator to a single dot.
        # The reference's own extent is the interesting region, so pin the view there
        # and let the diverged trace walk off the plot -- the error panel and the
        # table already quantify how far off it went.
        lo, hi = rp0[:, :2].min(0), rp0[:, :2].max(0)
        pad = 0.35 * max(hi - lo).max() if (hi - lo).max() > 0 else 1.0
        lo, hi = lo - pad, hi + pad
        off = [a["name"] for a in aligned
               if (a["p"][:, :2] < lo).any() or (a["p"][:, :2] > hi).any()]
        ax.set_aspect("equal", adjustable="datalim")
        if off:
            # adjustable="datalim" only ever WIDENS these to satisfy the aspect
            # ratio, so the clamp holds and the panel still fills its slot.
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
        style(ax, "x [m]", "y [m]", "Trajectory, top-down (SE(3)-aligned, scale fixed)")
        if off:
            ax.text(0.99, 0.01, f"{', '.join(off)} runs off the plot",
                    transform=ax.transAxes, ha="right", va="bottom",
                    color=th["ink3"], fontsize=9, zorder=7)
        ax.legend(facecolor=th["surface"], edgecolor=th["grid"], labelcolor=th["ink2"],
                  fontsize=9, loc="best")

    # error against the reference
    ax2 = fig.add_subplot(gs[0, 1])
    for i, a in enumerate(aligned):
        ax2.plot(a["t"], a["err"], color=th["series"][i + 1], lw=1.4, label=a["name"])
    style(ax2, "t [s]", "error [m]",
          f"Position error vs {ref_name}" if gt is not None else f"Difference from {ref_name}")
    ax2.legend(facecolor=th["surface"], edgecolor=th["grid"], labelcolor=th["ink2"],
               fontsize=9, loc="best")

    # z, which is where frame-convention problems announce themselves
    ax3 = fig.add_subplot(gs[1, 1])
    if aligned:
        tt0, rp0 = aligned[0]["t"], aligned[0]["ref"]
        ax3.plot(tt0, rp0[:, 2], color=th["series"][0], lw=2.4, alpha=0.55, label=ref_name)
    for i, a in enumerate(aligned):
        ax3.plot(a["t"], a["p"][:, 2], color=th["series"][i + 1], lw=1.4, label=a["name"])
    style(ax3, "t [s]", "z [m]", "Height")

    # --- per-component breakdown: position, velocity, attitude ---------------
    # Everything here is in the reference frame (see the alignment loop), so the
    # components are directly comparable across systems. Colour carries the SYSTEM,
    # matching every other panel; line style carries the COMPONENT. Two legends,
    # because one combined legend would be nine entries of mostly noise.
    COMPONENTS = (("x", "-"), ("y", "--"), ("z", ":"))
    ATTITUDE = (("yaw", "-"), ("pitch", "--"), ("roll", ":"))

    def masked(a, wrapped):
        """Break the line where a wrapped angle crosses +-180, so the wrap does not
        draw as a spurious full-height vertical stroke."""
        if not wrapped:
            return a
        a = a.astype(float).copy()
        a[:-1][np.abs(np.diff(a)) > 180.0] = np.nan
        return a

    def breakdown(slot, title, ylabel, key, labels, wrapped=False):
        axb = fig.add_subplot(bottom[slot])
        if gt is not None and aligned and aligned[0].get(f"ref{key}" if key != "p" else "ref") is not None:
            ref_arr = aligned[0]["ref" if key == "p" else f"ref{key}"]
            for c, (lab, ls) in enumerate(labels):
                axb.plot(aligned[0]["t"], masked(ref_arr[:, c], wrapped),
                         color=th["series"][0], lw=2.4, ls=ls, alpha=0.55, zorder=1)
        for i, a in enumerate(aligned):
            for c, (lab, ls) in enumerate(labels):
                axb.plot(a["t"], masked(a[key][:, c], wrapped),
                         color=th["series"][i + 1], lw=1.3, ls=ls, zorder=3 + i)
        style(axb, "t [s]", ylabel, title)
        # component key, drawn in a neutral ink so it reads as "line style", not
        # as a fourth system
        handles = [plt.Line2D([], [], color=th["ink3"], lw=1.4, ls=ls, label=lab)
                   for lab, ls in labels]
        axb.legend(handles=handles, facecolor=th["surface"], edgecolor=th["grid"],
                   labelcolor=th["ink2"], fontsize=8, loc="best", ncol=3,
                   handlelength=2.4, columnspacing=1.1)
        return axb

    breakdown(0, "Position", "[m]", "p", COMPONENTS)
    axv = breakdown(1, "Velocity  (reference frame)", "[m/s]", "v", COMPONENTS)
    axa = breakdown(2, "Attitude", "[deg]", "eul", ATTITUDE, wrapped=True)
    axa.set_yticks(np.arange(-180, 181, 90))

    # A relocalisation jump shows up in velocity as a spike orders of magnitude past
    # anything a ground robot does, and autoscaling to it flattens the real signal
    # into a flat line at zero. Scale to the bulk of the samples instead -- a high
    # percentile, not a fixed plausible-speed threshold, so the panel stays right for
    # a fast run as well as a slow one -- and label the panel when that hides a peak.
    vs = np.abs(np.concatenate([a["v"].ravel() for a in aligned])) if aligned else None
    if vs is not None and len(vs):
        m = max(np.percentile(vs, 99.0) * 1.6, 1e-3)
        if vs.max() > m:
            axv.set_ylim(-m, m)
            axv.text(0.99, 0.02, f"clipped; peak |v| {vs.max():.1f} m/s",
                     transform=axv.transAxes, ha="right", va="bottom",
                     color=th["ink3"], fontsize=8)

    fig.suptitle(f"VINS-Fusion vs ORB-SLAM3  --  {DATASET}",
                 color=th["ink"], fontsize=13, x=0.06, ha="left", y=0.978)
    out = os.path.join(OUTDIR, f"compare_{theme_name}.png")
    fig.savefig(out, dpi=160, facecolor=th["surface"])
    plt.close(fig)
    return out


print()
for t in ("light", "dark"):
    print("  wrote:", render(t))
