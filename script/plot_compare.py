#!/usr/bin/env python3
"""Compare VINS-Fusion against ORB-SLAM3 on the same run, with ground truth if there is any.

    python3 script/plot_compare.py simulation/sim
    python3 script/plot_compare.py real/rosbag_realsense_imu_cambaseline95mm
    python3 script/plot_compare.py simulation/dataset_dynamic_nofloortexture_000 --yolo

The argument is a dataset path relative to output/output_vins and output/output_orb,
which are parallel trees. Ground truth is picked up from either side if present.
Figures land in output/compare/<dataset>/, as compare_{light,dark}.png -- or, with
--yolo, compare_{light,dark}_yolo.png, so the two-system and four-system views of the
same dataset sit side by side rather than one overwriting the other.

--yolo adds VINS-Fusion+YOLO and ORB-SLAM3+YOLO (see vio_metrics.SYSTEMS_YOLO) --
YOLO-masked variants run on a handful of datasets to test whether masking out detected
dynamic objects before tracking recovers what plain VINS/ORB lose on those scenes. A
system with no vio.csv for this dataset is silently absent, same as always -- most
datasets only have the plain pair even when --yolo is passed.

The metrics themselves live in vio_metrics.py, shared with plot_summary.py -- see that
module for why the alignment is a full SE(3) fit and why the numbers are computed once.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from vio_metrics import (DIVERGED_ERROR, JUMP_SPEED, PLAUSIBLE_MEAN_SPEED, ROOT,
                         SYSTEMS, SYSTEMS_YOLO, analyse, num)

args = sys.argv[1:]
YOLO = "--yolo" in args
if YOLO:
    args.remove("--yolo")
DATASET = args[0] if args else "simulation/sim"
SUFFIX = "_yolo" if YOLO else ""
OUTDIR = os.path.join(ROOT, "output", "compare", DATASET)

# Colours for the first two systems are unchanged from before --yolo existed, so a
# plain two-system figure renders identically to always. The two YOLO colours are
# appended, never inserted, so they only ever get used when a dataset actually has
# that data -- the index a system's colour comes from is its position in `aligned`,
# which just doesn't grow past 2 for every other dataset.
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", ink3="#8a8983",
                  grid="#e3e2dd",
                  series=("#1baf7a", "#2a78d6", "#eb6834", "#7c5cd6", "#c23b7a")),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", ink3="#87867d",
                  grid="#33332f",
                  series=("#199e70", "#3987e5", "#d95926", "#9678e8", "#e0568f")),
}

result = analyse(DATASET, print, systems=SYSTEMS_YOLO if YOLO else SYSTEMS)
if result is None:
    sys.exit(f"no trajectories found for '{DATASET}'")
rows, aligned, info = result
gt, ref_name, lbl = info["gt"], info["ref_name"], info["lbl"]
# Name column width: fixed at 14 (the old constant) unless a longer name is actually
# present, so every existing two-system dataset renders byte-identical to before this
# was added, and only a run that actually has e.g. "VINS-Fusion+YOLO" widens the table.
NAME_W = max(14, max((len(r["name"]) for r in rows), default=14) + 1)

# --- report ------------------------------------------------------------------
print(f"\n  {DATASET}   reference: {ref_name}\n")
print(f"  {'system':<{NAME_W}}{'poses':>7}{'secs':>8}{'path m':>10}{'ref m':>10}"
      f"{lbl:>11}{'max':>10}{'final':>10}")
for r in rows:
    self_ref = (gt is None and r["name"] == ref_name)
    cols = (f"{'--':>11}{'--':>10}{'--':>10}" if self_ref
            else f"{num(r['rms'], 11, 3)}{num(r['max'], 10, 3)}{num(r['final'], 10, 3)}")
    print(f"  {r['name']:<{NAME_W}}{r['poses']:>7}{r['secs']:>8.1f}{num(r['path'])}"
          f"{num(r['ref'])}{cols}" + ("   (reference)" if self_ref else ""))

# Plausibility first, and WITHOUT reference to the other system. On the real bags
# there is no ground truth, so the reference is just whichever estimator came first --
# and if that one has diverged, every "error vs reference" number is backwards, blaming
# the healthy run. Implied mean speed needs no reference and no common frame: a ground
# robot indoors does not average tens of m/s, whoever says otherwise is the broken one.
print()
for r in rows:
    v = r["path"] / r["secs"] if r["secs"] > 0 else 0
    flag = "  <-- IMPLAUSIBLE" if v > PLAUSIBLE_MEAN_SPEED else ""
    print(f"  {r['name']:<{NAME_W}} implied mean speed {num(v, 7)} m/s over {r['secs']:.0f}s{flag}")

# Distinguish the two ways a run goes wrong. A few impossible-speed steps are
# tracking JUMPS, and the numbers above understate a run that was otherwise fine.
# A long path with no jumps is genuine DIVERGENCE, and the numbers are real.
for r in rows:
    n, jm, seg = r["name"], r["jumps"], r["seg"]
    if len(jm):
        print(f"\n  [{n}] {len(jm)} tracking jump(s) >{JUMP_SPEED:.0f} m/s "
              f"(largest {num(r['biggest'], 8, 1).strip()} m).")
        print(f"        path excluding jumps: {num(r['clean_len'], 9).strip()} m   "
              f"vs reference {num(r['ref'], 9).strip()} m")
        if seg:
            print(f"        longest clean segment: {seg[0]} poses / {seg[1]:.1f}s, "
                  f"{lbl} {seg[2]:.3f} m")
        print(f"        The table's {lbl} is set by the jump, not by tracking quality.")
    elif gt is not None and r["ref"] > 0 and (r["path"] / r["ref"] > 3 or r["ref"] / r["path"] > 3):
        print(f"\n  [warn] {n} path is {r['path']:.0f} m against {r['ref']:.0f} m of "
              f"GROUND TRUTH ({r['path']/r['ref']:.1f}x) with no jumps -- genuine "
              f"divergence.")

# --- summary table, shared by the console report and the figures --------------
# Built once here rather than per-theme: the numbers do not depend on the theme,
# and the console and the PNG must not be able to disagree.
HDR = (f"  {'system':<{NAME_W - 1}}{'poses':>7}{'secs':>7}{'path m':>10}"
       f"{'GT m' if gt is not None else 'ref m':>9}{lbl:>10}{'max':>9}{'final':>9}"
       f"{'jumps':>7}{'clean':>9}")


def table_row(r):
    n, seg = r["name"], r["seg"]
    self_ref = (gt is None and n == ref_name)
    err_cols = (f"{'--':>10}{'--':>9}{'--':>9}" if self_ref
                else f"{num(r['rms'], 10, 3)}{num(r['max'], 9, 3)}{num(r['final'], 9, 3)}")
    clean = "--" if (self_ref or seg is None) else f"{seg[2]:.3f}"
    flag = "   DIVERGED" if diverged.get(n) else ""
    return (f"  {n:<{NAME_W - 1}}{r['poses']:>7}{r['secs']:>7.1f}{num(r['path'], 10)}"
            f"{num(r['ref'], 9)}{err_cols}{len(r['jumps']):>7}{clean:>9}{flag}")


# Footnotes: the two ways a run goes wrong, and the reference-free plausibility
# check. Kept to one line each -- the console report carries the long form.
diverged = {a["name"]: a["diverged"] for a in aligned}

notes = []
if gt is None:
    notes.append(f"  reference: {ref_name} (no ground truth for this dataset)")
for r in rows:
    n, jm = r["name"], r["jumps"]
    v = r["path"] / r["secs"] if r["secs"] > 0 else 0
    if diverged.get(n):
        notes.append(f"  [{n}] DIVERGED -- peak error {num(r['max'], 1).strip()} m against "
                     f"ground truth; shown in the trajectory panel, "
                     f"omitted from the time-series panels")
    elif v > PLAUSIBLE_MEAN_SPEED:
        notes.append(f"  [{n}] implied mean speed {num(v, 1).strip()} m/s -- IMPLAUSIBLE"
                     f" for a ground robot; the run has broken down")
    elif len(jm):
        notes.append(f"  [{n}] {len(jm)} tracking jump(s) >{JUMP_SPEED:.0f} m/s, largest "
                     f"{num(r['biggest'], 8, 1).strip()} m -- '{lbl}' is set by the jump, "
                     f"'clean' is the longest jump-free segment")
    elif gt is not None and r["ref"] > 0 and (r["path"] / r["ref"] > 3 or r["ref"] / r["path"] > 3):
        notes.append(f"  [{n}] path {num(r['path'], 1).strip()} m vs "
                     f"{num(r['ref'], 1).strip()} m of ground truth with no jumps -- "
                     f"genuine divergence")

# --- plot --------------------------------------------------------------------
os.makedirs(OUTDIR, exist_ok=True)


def render(theme_name):
    th = THEMES[theme_name]
    band = 0.055 + 0.022 * (len(rows) + len(notes))
    fig = plt.figure(figsize=(14, 15.6), facecolor=th["surface"])
    outer = GridSpec(3, 1, figure=fig, height_ratios=[band, 1.0, 1.15], hspace=0.16,
                     left=0.06, right=0.98, top=0.965, bottom=0.045)
    stats = fig.add_subplot(outer[0])
    gs = outer[1].subgridspec(2, 2, width_ratios=[1.25, 1], hspace=0.32, wspace=0.22)
    bottom = outer[2].subgridspec(3, 3, wspace=0.26, hspace=0.12)

    # --- summary table -------------------------------------------------------
    # The same numbers the script prints, carried into the PNG so a figure sent on
    # its own is still self-describing. Rows are tinted with each system's series
    # colour, which is what ties a row to its curve in the panels below.
    stats.axis("off")
    y, dy = 1.0, 1.0 / (len(rows) + len(notes) + 1.6)
    stats.text(0, y, HDR, transform=stats.transAxes, family="monospace",
               fontsize=10, color=th["ink2"], va="top", weight="bold")
    for i, r in enumerate(rows):
        y -= dy
        stats.text(0, y, table_row(r), transform=stats.transAxes, family="monospace",
                   fontsize=10, color=th["series"][i + 1], va="top")
    for note in notes:
        y -= dy
        stats.text(0, y, note, transform=stats.transAxes, family="monospace",
                   fontsize=9, color=th["ink3"], va="top")

    # A diverged run is still drawn everywhere -- you should be able to see it leave --
    # but only these set the scales.
    gone = [a for a in aligned if a["diverged"]]
    live = [a for a in aligned if not a["diverged"]]

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
        sub = gs[:, 0].subgridspec(len(aligned), 1, hspace=0.38)
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
    if live and gone:
        ax2.set_ylim(0, max(a["err"].max() for a in live) * 1.15 + 1e-6)
        # Bottom, not top: a legend placed with loc="best" tends to land top-right on
        # exactly this shape of curve (low near t=0, climbing later), which is the one
        # other spot a note here could land -- put this where "best" is unlikely to.
        ax2.text(0.99, 0.03, f"{', '.join(a['name'] for a in gone)} diverged, off scale",
                 transform=ax2.transAxes, ha="right", va="bottom", color=th["ink3"],
                 fontsize=8)
    ax2.legend(facecolor=th["surface"], edgecolor=th["grid"], labelcolor=th["ink2"],
               fontsize=9, loc="best")

    # z, which is where frame-convention problems announce themselves
    ax3 = fig.add_subplot(gs[1, 1])
    if aligned:
        tt0, rp0 = aligned[0]["t"], aligned[0]["ref"]
        ax3.plot(tt0, rp0[:, 2], color=th["series"][0], lw=2.4, alpha=0.55, label=ref_name)
    for i, a in enumerate(aligned):
        if a["diverged"]:
            continue
        ax3.plot(a["t"], a["p"][:, 2], color=th["series"][i + 1], lw=1.4, label=a["name"])
    style(ax3, "t [s]", "z [m]", "Height")

    # --- per-component breakdown: position, velocity, attitude ---------------
    # One subplot per axis, laid out as a 3x3 block: a column per quantity, a row
    # per component. Colour still carries the SYSTEM, matching every other panel,
    # and each subplot owns its own y scale -- which is the point of splitting them.
    # Sharing one axes per quantity made z invisible next to x and y, and roll
    # invisible next to yaw, because the components differ by an order of magnitude.
    # Everything here is in the reference frame (see the alignment loop), so the
    # components are directly comparable across systems.
    BLOCKS = (
        ("Position", "p", "ref", ("x", "y", "z"), "m", False),
        ("Velocity  (reference frame)", "v", "refv", ("vx", "vy", "vz"), "m/s", False),
        ("Attitude", "eul", "refeul", ("yaw", "pitch", "roll"), "deg", True),
    )

    def masked(a, wrapped):
        """Break the line where a wrapped angle crosses +-180, so the wrap does not
        draw as a spurious full-height vertical stroke."""
        if not wrapped:
            return a
        a = a.astype(float).copy()
        a[:-1][np.abs(np.diff(a)) > 180.0] = np.nan
        return a

    for j, (title, key, refkey, comps, unit, wrapped) in enumerate(BLOCKS):
        top_of_col = None
        for c, comp in enumerate(comps):
            axb = fig.add_subplot(bottom[c, j], sharex=top_of_col)
            top_of_col = top_of_col or axb
            ref_arr = aligned[0][refkey] if (gt is not None and aligned
                                             and aligned[0].get(refkey) is not None) else None
            if ref_arr is not None:
                axb.plot(aligned[0]["t"], masked(ref_arr[:, c], wrapped),
                         color=th["series"][0], lw=2.6, alpha=0.55, zorder=1,
                         label=ref_name)
            for i, a in enumerate(aligned):
                if a["diverged"]:
                    continue
                axb.plot(a["t"], masked(a[key][:, c], wrapped),
                         color=th["series"][i + 1], lw=1.3, zorder=3 + i, label=a["name"])
            # time axis only under the bottom row; the column is shared and aligned
            style(axb, "t [s]" if c == len(comps) - 1 else "",
                  f"{comp} [{unit}]", title if c == 0 else "")
            if c < len(comps) - 1:
                axb.tick_params(labelbottom=False)

            # Scale from the runs that are still tracking, plus the reference. A
            # diverged run reaching 1e15 m would otherwise flatten every other series
            # in this column to a line on the zero axis.
            scale_src = live or aligned
            dat = [a[key][:, c] for a in scale_src]
            if ref_arr is not None:
                dat.append(ref_arr[:, c])
            dat = np.concatenate(dat)

            if wrapped and comp == "yaw":
                # yaw is wrapped, so pin the full circle rather than letting the
                # range depend on which way the robot happened to turn
                axb.set_ylim(-190, 190)
                axb.set_yticks(np.arange(-180, 181, 90))

            elif key in ("v", "eul"):
                # Startup transients dominate these. A relocalisation jump differentiates
                # into a velocity spike orders of magnitude past anything a ground robot
                # does, and ORB-SLAM3's pre-IMU-initialisation attitude swings through
                # tens of degrees -- autoscaling to either flattens the real signal into
                # a line. Scale to the bulk of the samples instead: a high percentile
                # about the median, not a fixed threshold, so this stays right whether
                # the quantity is centred on zero (velocity) or on an offset (attitude).
                # 95th, not 99th: ORB-SLAM3's pre-initialisation attitude is a
                # sustained several-second block, not a spike, and at 25 Hz over a
                # 90 s run that is ~2.5% of the samples -- a 99th percentile lands
                # inside it and clips nothing.
                if len(dat):
                    mid = np.median(dat)
                    half = max(np.percentile(np.abs(dat - mid), 95.0) * 2.5, 1e-3)
                    axb.set_ylim(mid - half, mid + half)
                    allv = np.concatenate([a[key][:, c] for a in scale_src])
                    if (np.abs(allv - mid) > half).any():
                        peak = allv[np.argmax(np.abs(allv - mid))]
                        axb.text(0.99, 0.03, f"clipped; peak {peak:.1f}",
                                 transform=axb.transAxes, ha="right", va="bottom",
                                 color=th["ink3"], fontsize=8)


            if gone and c == 0:
                # Bottom-right in the Position column specifically: that column never
                # gets a "clipped; peak" note (only velocity/attitude do), so the slot
                # is free, and it is also where this column's legend (below) is NOT --
                # legend for the whole block only ever appears here too, via loc="best".
                y, va = (0.03, "bottom") if key == "p" else (0.96, "top")
                axb.text(0.99, y, f"{', '.join(a['name'] for a in gone)} diverged, "
                         "not shown", transform=axb.transAxes, ha="right", va=va,
                         color=th["ink3"], fontsize=8)

            # one legend for the whole block, in its first subplot
            if j == 0 and c == 0:
                axb.legend(facecolor=th["surface"], edgecolor=th["grid"],
                           labelcolor=th["ink2"], fontsize=8, loc="best")

    title = "VINS-Fusion vs ORB-SLAM3" + (" (+YOLO)" if YOLO else "")
    fig.suptitle(f"{title}  --  {DATASET}",
                 color=th["ink"], fontsize=13, x=0.06, ha="left", y=0.985)
    out = os.path.join(OUTDIR, f"compare_{theme_name}{SUFFIX}.png")
    fig.savefig(out, dpi=160, facecolor=th["surface"])
    plt.close(fig)
    return out


print()
for t in ("light", "dark"):
    print("  wrote:", render(t))
