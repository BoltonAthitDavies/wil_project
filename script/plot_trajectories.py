#!/usr/bin/env python3
"""Every dataset's top-down trajectory on one sheet, for comparing runs at a glance.

    python3 script/plot_trajectories.py                          # every dataset found
    python3 script/plot_trajectories.py simulation                # only under a prefix
    python3 script/plot_trajectories.py --filter nofloortexture   # only matching names
    python3 script/plot_trajectories.py --yolo --filter nofloortexture   # + YOLO runs

Writes output/compare/trajectories_{light,dark}.png, or with --filter and/or --yolo
appended to that name (trajectories_{light,dark}_yolo_nofloortexture.png) so a
filtered sheet never overwrites the full one.

--yolo adds VINS-Fusion+YOLO and ORB-SLAM3+YOLO (see vio_metrics.SYSTEMS_YOLO) to
every panel. A dataset that was never run through the YOLO variants just shows the
plain pair, same as always -- most datasets only have that.

Each panel is the same view as the headline panel of the per-dataset figure: every
estimate aligned to the reference with a full SE(3) Umeyama fit, scale FIXED, so what
you see is drift and scale error rather than the arbitrary world frame each estimator
happened to initialise in. Metrics come from vio_metrics.analyse(), the same call the
other two scripts make.
"""

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from vio_metrics import ROOT, SYSTEMS, SYSTEMS_YOLO, analyse, discover

args = sys.argv[1:]
FILTER = None
if "--filter" in args:
    i = args.index("--filter")
    FILTER = args[i + 1]
    del args[i:i + 2]
EXCLUDE = None
if "--exclude" in args:
    i = args.index("--exclude")
    EXCLUDE = args[i + 1]
    del args[i:i + 2]
YOLO = "--yolo" in args
if YOLO:
    args.remove("--yolo")
PREFIX = args[0] if args else ""
METRICS_SYSTEMS = SYSTEMS_YOLO if YOLO else SYSTEMS
_suffix_parts = (["yolo"] if YOLO else []) + ([FILTER] if FILTER else [])
SUFFIX = "_" + "_".join(_suffix_parts) if _suffix_parts else ""
OUT = os.path.join(ROOT, "output", "compare")

# Colours for the first two systems are unchanged, so a plain two-system sheet renders
# identically to always; the two YOLO colours are appended, only ever reached when a
# dataset's `aligned` list actually grows past 2.
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", ink3="#8a8983",
                  grid="#e3e2dd",
                  series=("#1baf7a", "#2a78d6", "#eb6834", "#7c5cd6", "#c23b7a"),
                  bad="#c0442a"),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", ink3="#87867d",
                  grid="#33332f",
                  series=("#199e70", "#3987e5", "#d95926", "#9678e8", "#e0568f"),
                  bad="#e0674a"),
}


def collect(dataset):
    res = analyse(dataset, systems=METRICS_SYSTEMS)
    if res is None:
        return None
    rows, aligned, info = res
    if not aligned:
        return None
    rms = {r["name"]: r["rms"] for r in rows}
    return dict(name=dataset.split("/", 1)[1], aligned=aligned, info=info, rms=rms)


def render(theme_name, data):
    th = THEMES[theme_name]
    # Colour by SYSTEM IDENTITY, not by position in a panel's own (possibly sparse)
    # aligned list. The legend is built once, from whichever panel happens to be
    # first, using METRICS_SYSTEMS' canonical order -- a panel that is missing an
    # earlier system (e.g. has ORB-SLAM3+YOLO but not VINS-Fusion) must still draw
    # ORB-SLAM3+YOLO in ORB-SLAM3+YOLO's colour, not whatever colour its LOCAL index
    # happens to land on, or its lines stop matching what the shared legend claims.
    color_of = {name: th["series"][i + 1] for i, (name, _) in enumerate(METRICS_SYSTEMS)}
    n = len(data)
    cols = min(4, max(1, math.ceil(math.sqrt(n))))
    rowsn = math.ceil(n / cols)

    # A fixed header in INCHES, not a fraction: a fraction that looks right at four
    # rows leaves a field of blank paper at one and collides with the titles at eight.
    head = 0.95
    fh = 4.0 * rowsn + head
    # A width floor for the header: with few datasets (few columns) the grid alone
    # makes a narrow figure, and the title/subtitle text -- sized for a full run of
    # ~4 columns -- overflows past the edge and collides with the legend. Panels just
    # end up a bit wider than 3.7in each when N is small, which is a better outcome
    # than a squeezed, unreadable header. The legend itself also grows with the system
    # count (--yolo puts 5 entries -- ground truth plus 4 systems -- on one row where
    # the default run only ever has 3), and a floor sized for 3 entries lets a 5-entry
    # legend run wide enough to collide with the title text on its left. Scale the
    # floor by how many entries the legend will actually hold.
    legend_entries = len(METRICS_SYSTEMS) + 1
    fig_w = max(3.7 * cols, 12.0, 3.0 + 2.2 * legend_entries)
    fig, axes = plt.subplots(rowsn, cols, figsize=(fig_w, fh),
                             facecolor=th["surface"], squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)

    for k, d in enumerate(data):
        ax = axes[k // cols][k % cols]
        ax.set_visible(True)
        ax.set_facecolor(th["surface"])
        ax.grid(True, color=th["grid"], lw=0.6, zorder=0)
        for sp in ax.spines.values():
            sp.set_color(th["grid"])
        ax.tick_params(colors=th["ink3"], labelsize=7)

        gt = d["info"]["gt"] is not None
        ref = d["aligned"][0]["ref"]
        if gt:
            ax.plot(ref[:, 0], ref[:, 1], color=th["series"][0], lw=3.0, alpha=0.5,
                    zorder=1, label="ground truth")
        for i, a in enumerate(d["aligned"]):
            c = color_of[a["name"]]
            ax.plot(a["p"][:, 0], a["p"][:, 1], color=c, lw=1.2,
                    zorder=3 + i, label=a["name"])
            ax.plot(a["p"][0, 0], a["p"][0, 1], "o", color=c, ms=3.5, zorder=8)

        # Pin the view to the reference's own extent. A diverged run reaching thousands
        # of metres would otherwise set the axes for the panel and shrink the reference
        # and the healthy estimate to a single dot -- the same clamp the per-dataset
        # figure uses, and the reason these panels stay comparable to each other.
        lo, hi = ref[:, :2].min(0), ref[:, :2].max(0)
        pad = 0.12 * max(hi - lo).max() if (hi - lo).max() > 0 else 1.0
        lo, hi = lo - pad, hi + pad
        off = [a["name"] for a in d["aligned"]
               if (a["p"][:, :2] < lo).any() or (a["p"][:, :2] > hi).any()]
        # adjustable="box", NOT "datalim". With "datalim" matplotlib satisfies the
        # equal aspect by EXPANDING the data limits set below, so a subplot wider
        # than it is tall silently widens x: dataset_static_nofloortexture_000 was
        # drawn over 55 m of x for a trajectory 17 m wide, and the ground truth sat
        # in a third of the panel with dead paper either side. "box" honours the
        # limits and shrinks the axes box instead, so the reference fills the panel
        # and the subtitle's claim that panels are pinned to the ground-truth extent
        # is actually true.
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal", adjustable="box")

        ax.set_title(d["name"], color=th["ink"], fontsize=9, loc="left", pad=6)
        # the two numbers that say whether the panel is worth trusting
        bits, colors = [], []
        for i, a in enumerate(d["aligned"]):
            short = a["name"].split("-")[0]
            if a["diverged"]:
                bits.append(f"{short} DIVERGED")
                colors.append(th["bad"])
            else:
                bits.append(f"{short} {d['rms'][a['name']]:.2f}")
                colors.append(color_of[a["name"]])
        y = 0.975
        for b, c in zip(bits, colors):
            ax.text(0.985, y, b, transform=ax.transAxes, ha="right", va="top",
                    color=c, fontsize=7.5, family="monospace")
            y -= 0.055
        if off and not any(a["diverged"] for a in d["aligned"]):
            ax.text(0.985, 0.02, "off-plot", transform=ax.transAxes, ha="right",
                    va="bottom", color=th["ink3"], fontsize=7)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(labels),
               facecolor=th["surface"], edgecolor=th["grid"], labelcolor=th["ink2"],
               fontsize=9, bbox_to_anchor=(0.995, 1 - 0.18 / fh))
    # The scope note goes on the SUBTITLE row, not the title row: the title row also
    # holds the legend (top-right), and a filter name appended to the title text can
    # run wide enough to collide with it once the figure is narrow (few datasets ->
    # few columns -> the width floor above, not a full-width grid).
    fig.suptitle("Trajectory, top-down (SE(3)-aligned, scale fixed)",
                 color=th["ink"], fontsize=14, x=0.012, ha="left",
                 y=1 - 0.30 / fh, va="top")
    scope = f"   --   datasets matching '{FILTER}'" if FILTER else ""
    yolo_note = "   (+YOLO)" if YOLO else ""
    fig.text(0.012, 1 - 0.60 / fh,
             "each panel pinned to the ground-truth extent, so panels are comparable; "
             f"ATE rms in metres, x/y in metres{scope}{yolo_note}",
             color=th["ink3"], fontsize=8.5, ha="left", va="top")
    fig.tight_layout(rect=[0, 0, 1, 1 - head / fh])

    path = os.path.join(OUT, f"trajectories_{theme_name}{SUFFIX}.png")
    fig.savefig(path, dpi=160, facecolor=th["surface"])
    plt.close(fig)
    return path


datasets = discover(PREFIX, contains=FILTER, exclude=EXCLUDE, systems=METRICS_SYSTEMS)
if not datasets:
    sys.exit(f"no datasets found under '{PREFIX}'")

data = []
for ds in datasets:
    d = collect(ds)
    if d is None:
        print(f"  [skip] {ds}")
        continue
    data.append(d)
    print(f"  {ds}")

print()
for t in ("light", "dark"):
    print("  wrote:", render(t, data))
