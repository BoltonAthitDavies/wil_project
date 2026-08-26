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
    """VINS-format CSV -> (t_sec, position, quaternion_wxyz). None if absent/empty."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    d = np.loadtxt(path, delimiter=",")
    if d.ndim == 1:
        d = d.reshape(1, -1)
    if len(d) < 2:
        return None
    return d[:, 0] / 1e9, d[:, 1:4], d[:, 4:8]


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

# --- align every trajectory to the reference over their common window --------
rows, aligned = [], []
for name, t, p, q in runs:
    lo, hi = max(t[0], ref_t[0]), min(t[-1], ref_t[-1])
    m = (t >= lo) & (t <= hi)
    if m.sum() < 10:
        print(f"  [skip] {name}: only {m.sum()} samples overlap the reference")
        continue
    tc, pc = t[m], p[m]
    rp = resample(ref_t, ref_p, tc)
    R, tr = umeyama(pc, rp)
    pa = (R @ pc.T).T + tr
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
    aligned.append((name, tc - tc[0], pa, rp, err))
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
    fig = plt.figure(figsize=(14, 7.5), facecolor=th["surface"])
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.25, 1], hspace=0.32, wspace=0.22,
                  left=0.06, right=0.98, top=0.90, bottom=0.09)

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
        sub = GridSpec(2, 2, figure=fig, width_ratios=[1.25, 1], hspace=0.32, wspace=0.22,
                       left=0.06, right=0.98, top=0.90, bottom=0.09)[:, 0].subgridspec(
                           len(runs), 1, hspace=0.38)
        for i, (name, _, pa, _, _) in enumerate(aligned):
            axi = fig.add_subplot(sub[i])
            _, _, praw, _, _ = aligned[i]
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
        _, _, _, rp0, _ = aligned[0]
        ax.plot(rp0[:, 0], rp0[:, 1], color=th["series"][0], lw=3.0, alpha=0.55,
                label=ref_name, zorder=1)
    if ax is not None:
        for i, (name, _, pa, _, _) in enumerate(aligned):
            ax.plot(pa[:, 0], pa[:, 1], color=th["series"][i + 1], lw=1.6,
                    label=name, zorder=3 + i)
            ax.plot(pa[0, 0], pa[0, 1], "o", color=th["series"][i + 1], ms=5, zorder=6)
        ax.set_aspect("equal", adjustable="datalim")
        style(ax, "x [m]", "y [m]", "Trajectory, top-down (SE(3)-aligned, scale fixed)")
        ax.legend(facecolor=th["surface"], edgecolor=th["grid"], labelcolor=th["ink2"],
                  fontsize=9, loc="best")

    # error against the reference
    ax2 = fig.add_subplot(gs[0, 1])
    for i, (name, tt, _, _, err) in enumerate(aligned):
        ax2.plot(tt, err, color=th["series"][i + 1], lw=1.4, label=name)
    style(ax2, "t [s]", "error [m]",
          f"Position error vs {ref_name}" if gt is not None else f"Difference from {ref_name}")
    ax2.legend(facecolor=th["surface"], edgecolor=th["grid"], labelcolor=th["ink2"],
               fontsize=9, loc="best")

    # z, which is where frame-convention problems announce themselves
    ax3 = fig.add_subplot(gs[1, 1])
    if aligned:
        _, tt0, _, rp0, _ = aligned[0]
        ax3.plot(tt0, rp0[:, 2], color=th["series"][0], lw=2.4, alpha=0.55, label=ref_name)
    for i, (name, tt, pa, _, _) in enumerate(aligned):
        ax3.plot(tt, pa[:, 2], color=th["series"][i + 1], lw=1.4, label=name)
    style(ax3, "t [s]", "z [m]", "Height")

    fig.suptitle(f"VINS-Fusion vs ORB-SLAM3  --  {DATASET}",
                 color=th["ink"], fontsize=13, x=0.06, ha="left", y=0.965)
    out = os.path.join(OUTDIR, f"compare_{theme_name}.png")
    fig.savefig(out, dpi=160, facecolor=th["surface"])
    plt.close(fig)
    return out


print()
for t in ("light", "dark"):
    print("  wrote:", render(t))
