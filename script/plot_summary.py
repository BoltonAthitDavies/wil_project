#!/usr/bin/env python3
"""One table, every dataset: VINS-Fusion against ORB-SLAM3 across the whole collection.

    python3 script/plot_summary.py                          # every dataset found
    python3 script/plot_summary.py simulation                # only those under a prefix
    python3 script/plot_summary.py --trim 5                  # first 5s of each run cut
    python3 script/plot_summary.py --filter nofloortexture   # only names containing this
    python3 script/plot_summary.py --yolo --filter nofloortexture   # + YOLO-masked runs
    python3 script/plot_summary.py simulation --trim 5       # combine freely, any order

Writes output/compare/summary_{light,dark}.png. --trim, --filter and/or --yolo append
to that name (e.g. summary_{light,dark}_trim.png, summary_{light,dark}_nofloortexture.png,
summary_{light,dark}_yolo_nofloortexture.png) so a filtered run never overwrites the
full one -- run this once for the whole collection and again for just the datasets
you're asking about, and both stay on disk side by side.

--yolo widens the table to four systems (see vio_metrics.SYSTEMS_YOLO): VINS-Fusion,
ORB-SLAM3, and their YOLO-masked variants, which mask out detected dynamic objects
before tracking. Only datasets that were actually run with the YOLO variants get
numbers in those two columns; everything else shows '--' there, same as any other
system a given dataset was never run through.

Datasets are discovered by walking output/output_vins and output/output_orb for any
directory holding a vio.csv, so a newly added run appears here as soon as it is
processed -- nothing to register by hand. Metrics come from vio_metrics.analyse(), the
same call plot_compare.py makes, so a number here always matches the per-dataset figure.

--trim exists because a run's headline ATE can be set almost entirely by a few seconds
of pre-initialisation transient (see vio_metrics.analyse's trim_start), which reads as
"this estimator is bad at tracking" when the real story is "this estimator is bad at
STARTING" -- two different problems needing two different fixes. Comparing the plain
summary against the trimmed one is how you tell which is which for a given run.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vio_metrics import (DIVERGED_ERROR, PLAUSIBLE_MEAN_SPEED, ROOT, SYSTEMS,
                         SYSTEMS_YOLO, analyse, discover, num)

args = sys.argv[1:]
TRIM = 0.0
if "--trim" in args:
    i = args.index("--trim")
    TRIM = float(args[i + 1])
    del args[i:i + 2]
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
_suffix_parts = ((["trim"] if TRIM > 0 else []) + (["yolo"] if YOLO else [])
                 + ([FILTER] if FILTER else []))
SUFFIX = "_" + "_".join(_suffix_parts) if _suffix_parts else ""
OUT = os.path.join(ROOT, "output", "compare")

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", ink3="#8a8983",
                  rule="#d9d8d2", band="#f2f1ec",
                  good="#1b7f5a", warn="#b8781f", bad="#c0442a"),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", ink3="#87867d",
                  rule="#3a3a35", band="#232320",
                  good="#3fbf8e", warn="#d9a441", bad="#e0674a"),
}

SYSTEM_NAMES = [name for name, _ in METRICS_SYSTEMS]


def collect(dataset):
    """-> dict per system, plus whether this dataset has ground truth."""
    res = analyse(dataset, trim_start=TRIM, systems=METRICS_SYSTEMS)
    if res is None:
        return None
    rows, aligned, info = res
    diverged = {a["name"]: a["diverged"] for a in aligned}
    out = {}
    for r in rows:
        self_ref = (info["gt"] is None and r["name"] == info["ref_name"])
        seg = r["seg"]
        out[r["name"]] = dict(
            secs=r["secs"], path=r["path"], ref=r["ref"], rms=r["rms"],
            clean=None if seg is None else seg[2], jumps=len(r["jumps"]),
            rot_rms=r["rot_rms"], diverged=diverged.get(r["name"], False),
            self_ref=self_ref, speed=r["path"] / r["secs"] if r["secs"] > 0 else 0.0)
    return dict(systems=out, gt=info["gt"] is not None, ref=info["ref_name"])


def verdict(s, has_gt):
    """How a run should read at a glance, and in which colour.

    Colour follows the HEADLINE ATE, not the clean-segment figure. A run can have a
    healthy clean segment and still be useless overall -- emptybg_00's VINS is 0.86 m
    clean but 6.9 m in total -- and colouring by the flattering number would hide
    exactly the runs worth looking at.
    """
    if s["self_ref"]:
        return "reference", "ink3"
    if s["diverged"]:
        return "DIVERGED", "bad"
    if s["speed"] > PLAUSIBLE_MEAN_SPEED:
        return "IMPLAUSIBLE", "bad"
    label = f"{s['rms']:.3f}"
    if not has_gt:
        # Measured against the other estimator, not truth. If THAT one is the broken
        # one the number is large for the healthy run, so it grades nothing.
        return label, "ink3"
    return label, ("good" if s["rms"] < 1.0 else "warn")


def build_lines(data):
    """Flatten the table into one list of drawable lines, so the figure can be sized
    to exactly what it holds instead of guessing and leaving a field of blank paper."""
    lines = [("caption", None)]
    if TRIM > 0:
        lines.append(("trimnote", None))
    lines += [("spanhdr", None), ("hdr", None), ("rule", 1.2)]
    group = None
    for dataset, d in data:
        g = dataset.split("/")[0]
        if g != group:
            group = g
            lines.append(("group", g.upper()))
        if d["systems"]:
            lines.append(("row", (dataset, d)))
    lines += [("rule", 1.0), ("note", "trans: green < 1 m   amber >= 1 m   red = the "
                                      "run broke down   grey = no ground truth, so the "
                                      "number grades nothing"),
              ("note", "'--' under a system means it was not run on that dataset. "
                       "'clean' is the longest jump-free segment (translational); "
                       "'jmp' counts steps implying >10 m/s. 'rot' is not colour-graded.")]
    return lines


# Target ABSOLUTE column widths, in inches, for the n_sys != 2 layout below. Picked to
# roughly match what the original hand-tuned 2-system layout works out to in inches
# (name ~3.9in, each numeric subcolumn ~0.85-0.95in, ~0.3in of extra air between
# system blocks) -- the goal is that a subcolumn is exactly as legible whether the
# table has 2 systems or 4, which means computing the figure width FROM these targets
# rather than picking a width and hoping the fractions fit.
NAME_IN, SLOT_IN, GAP_IN, MARGIN_IN, AXES_FRAC = 4.0, 0.85, 0.35, 0.3, 0.94


def compute_layout(n_sys):
    """-> (xs, fig_w). xs are column positions in axes-fraction (0..1); fig_w is the
    figure width in inches that makes those fractions come out to the target widths.

    n_sys == 2 keeps the exact literal layout and fig_w=15.0 this always had, so the
    default, unfiltered summary renders byte-identical no matter what else this grows
    to support. Any other count (currently only 4, for --yolo) is laid out generically:
    n_sys blocks of 4 subcolumns (trans/rot/clean/jmp), then one trailing block of 3
    (secs/path m/GT m), each block separated by GAP_IN beyond the normal slot pitch.
    """
    if n_sys == 2:
        return ([0.008, 0.285, 0.345, 0.415, 0.475,
                 0.555, 0.615, 0.685, 0.745, 0.815, 0.875, 0.95], 15.0)
    blocks = [4] * n_sys + [3]
    n_slots = sum(blocks)
    n_gaps = len(blocks) - 1
    fig_w = (NAME_IN + n_slots * SLOT_IN + n_gaps * GAP_IN + 2 * MARGIN_IN) / AXES_FRAC
    axes_in = fig_w * AXES_FRAC
    xs, cursor = [0.008], NAME_IN
    for bi, blen in enumerate(blocks):
        if bi > 0:
            cursor += GAP_IN
        for _ in range(blen):
            xs.append(cursor / axes_in)
            cursor += SLOT_IN
    return xs, fig_w


def render(theme_name, data):
    th = THEMES[theme_name]
    lines = build_lines(data)
    n = len(lines)
    xs, fig_w = compute_layout(len(SYSTEM_NAMES))

    fig = plt.figure(figsize=(fig_w, 0.30 * n + 1.0), facecolor=th["surface"])
    ax = fig.add_axes([0.03, 0.01, 0.94, 0.90])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def yof(i):
        return 1.0 - (i + 0.5) / n

    def txt(x, i, s, color="ink", size=10, weight="normal", ha="left"):
        ax.text(x, yof(i), s, color=th[color], fontsize=size, family="monospace",
                weight=weight, ha=ha, va="center", transform=ax.transAxes)

    scope = f"datasets matching '{FILTER}'" if FILTER else "all datasets"
    sysline = " vs ".join(SYSTEM_NAMES)
    title = f"{sysline}  --  {scope}"
    if TRIM > 0:
        title += f"   (first {TRIM:g}s of each run trimmed)"
    fig.suptitle(title, color=th["ink"], fontsize=15, x=0.03, ha="left", y=0.985)

    for i, (kind, payload) in enumerate(lines):
        if kind == "caption":
            txt(xs[0], i, "trans = position ATE rms in metres against ground truth, "
                          "SE(3)-aligned, scale fixed.   rot = orientation error rms in "
                          f"degrees (geodesic).   DIVERGED = peak position error "
                          f"> {DIVERGED_ERROR:.0f} m.", "ink3", 9)
        elif kind == "trimnote":
            txt(xs[0], i, f"Every number below excludes the first {TRIM:g}s each "
                          "estimator had data -- compare against the untrimmed summary "
                          "to see how much of a run's error is startup transient.",
                "ink3", 9)
        elif kind == "spanhdr":
            # xs[1 + 4*k] is the first subcolumn of system k; the trailing block starts
            # right after the last system's 4 subcolumns.
            starts = [1 + 4 * k for k in range(len(SYSTEM_NAMES))] + [1 + 4 * len(SYSTEM_NAMES)]
            for k, name in enumerate(SYSTEM_NAMES):
                a, b = xs[starts[k]], xs[starts[k + 1]]
                txt((a + b) / 2, i, name, "ink2", 11, "bold", ha="center")
                ax.plot([a, b - 0.012], [yof(i) - 0.4 / n] * 2, color=th["rule"], lw=1.0,
                        transform=ax.transAxes, clip_on=False)
        elif kind == "hdr":
            txt(xs[0], i, "dataset", "ink2", 10, "bold")
            per_sys = ["trans", "rot", "clean", "jmp"] * len(SYSTEM_NAMES)
            for x, h in zip(xs[1:], per_sys + ["secs", "path m", "GT m"]):
                txt(x, i, h, "ink2", 10, "bold", ha="right")
        elif kind == "rule":
            ax.plot([0, 1], [yof(i)] * 2, color=th["rule"], lw=payload,
                    transform=ax.transAxes, clip_on=False)
        elif kind == "group":
            txt(xs[0], i, payload, "ink3", 9, "bold")
        elif kind == "note":
            txt(xs[0], i, payload, "ink3", 9)
        elif kind == "row":
            dataset, d = payload
            # Truncate with an ellipsis rather than a hard cut -- at this row's
            # narrower name column (shrunk to fit the new rot columns) the two real-bag
            # names are long enough to run into the first value column otherwise.
            nm = dataset.split("/", 1)[1]
            txt(xs[0], i, nm if len(nm) <= 34 else nm[:33] + "…", "ink", 10)
            for si, sysname in enumerate(SYSTEM_NAMES):
                s = d["systems"].get(sysname)
                base = 1 + si * 4
                if s is None:
                    for k in range(4):
                        txt(xs[base + k], i, "--", "ink3", 10, ha="right")
                    continue
                label, col = verdict(s, d["gt"])
                txt(xs[base], i, label, col, 10, ha="right")
                # Rotation is bounded to [0, 180] deg even on a translationally diverged
                # run (see geodesic_deg), so unlike 'trans' it is a real number to show
                # even then -- only the self-reference row has nothing to report.
                txt(xs[base + 1], i, "--" if s["self_ref"] else f"{s['rot_rms']:.2f}",
                    "ink2", 10, ha="right")
                txt(xs[base + 2], i,
                    "--" if s["clean"] is None or s["self_ref"] else num(s["clean"], 9, 3).strip(),
                    "ink2", 10, ha="right")
                txt(xs[base + 3], i, str(s["jumps"]), "ink3", 10, ha="right")
            any_s = next(iter(d["systems"].values()))
            tsecs, tpath, tgt = xs[-3], xs[-2], xs[-1]
            txt(tsecs, i, f"{any_s['secs']:.0f}", "ink2", 10, ha="right")
            txt(tpath, i, num(any_s["path"], 9, 1).strip(), "ink2", 10, ha="right")
            txt(tgt, i, f"{any_s['ref']:.1f}" if d["gt"] else "--", "ink2", 10,
                ha="right")

    path = os.path.join(OUT, f"summary_{theme_name}{SUFFIX}.png")
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
    data.append((ds, d))
    flags = " ".join(f"{n}={'DIV' if s['diverged'] else format(s['rms'], '.3f')}"
                     for n, s in d["systems"].items())
    print(f"  {ds:<48}{flags}")

print()
for t in ("light", "dark"):
    print("  wrote:", render(t, data))
