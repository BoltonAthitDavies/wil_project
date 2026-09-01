#!/usr/bin/env python3
"""One table, every dataset: VINS-Fusion against ORB-SLAM3 across the whole collection.

    python3 script/plot_summary.py                 # every dataset found
    python3 script/plot_summary.py simulation      # only those under a prefix

Writes output/compare/summary_{light,dark}.png.

Datasets are discovered by walking output/output_vins and output/output_orb for any
directory holding a vio.csv, so a newly added run appears here as soon as it is
processed -- nothing to register by hand. Metrics come from vio_metrics.analyse(), the
same call plot_compare.py makes, so a number here always matches the per-dataset figure.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vio_metrics import DIVERGED_ERROR, PLAUSIBLE_MEAN_SPEED, ROOT, analyse, num

PREFIX = sys.argv[1] if len(sys.argv) > 1 else ""
OUT = os.path.join(ROOT, "output", "compare")

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", ink3="#8a8983",
                  rule="#d9d8d2", band="#f2f1ec",
                  good="#1b7f5a", warn="#b8781f", bad="#c0442a"),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", ink3="#87867d",
                  rule="#3a3a35", band="#232320",
                  good="#3fbf8e", warn="#d9a441", bad="#e0674a"),
}

SYSTEMS = ["VINS-Fusion", "ORB-SLAM3"]


def discover():
    """Every dataset with at least one vio.csv, as 'group/name'."""
    found = set()
    for tree in ("output_vins", "output_orb"):
        base = os.path.join(ROOT, "output", tree)
        for group in sorted(os.listdir(base)) if os.path.isdir(base) else []:
            gdir = os.path.join(base, group)
            if not os.path.isdir(gdir):
                continue
            for name in sorted(os.listdir(gdir)):
                if os.path.exists(os.path.join(gdir, name, "vio.csv")):
                    found.add(f"{group}/{name}")
    return sorted(d for d in found if d.startswith(PREFIX))


def collect(dataset):
    """-> dict per system, plus whether this dataset has ground truth."""
    res = analyse(dataset)
    if res is None:
        return None
    rows, aligned, info = res
    diverged = {a["name"]: a["diverged"] for a in aligned}
    out = {}
    for n, c, sec, pl, rl, rms, mx, fin, jm, cl, seg, bg in rows:
        self_ref = (info["gt"] is None and n == info["ref_name"])
        out[n] = dict(secs=sec, path=pl, ref=rl, rms=rms, clean=None if seg is None else seg[2],
                      jumps=len(jm), diverged=diverged.get(n, False), self_ref=self_ref,
                      speed=pl / sec if sec > 0 else 0.0)
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
    lines = [("caption", None), ("spanhdr", None), ("hdr", None), ("rule", 1.2)]
    group = None
    for dataset, d in data:
        g = dataset.split("/")[0]
        if g != group:
            group = g
            lines.append(("group", g.upper()))
        if d["systems"]:
            lines.append(("row", (dataset, d)))
    lines += [("rule", 1.0), ("note", "green < 1 m   amber >= 1 m   red = the run broke "
                                      "down   grey = no ground truth, so the number "
                                      "grades nothing"),
              ("note", "'--' under a system means it was not run on that dataset. "
                       "'clean' is the longest jump-free segment; 'jmp' counts steps "
                       "implying >10 m/s.")]
    return lines


def render(theme_name, data):
    th = THEMES[theme_name]
    lines = build_lines(data)
    n = len(lines)
    # Columns: dataset, then per system (ATE, clean, jmp), then the run's own size.
    xs = [0.008, 0.30, 0.375, 0.445, 0.505, 0.585, 0.66, 0.72, 0.80, 0.90]

    fig = plt.figure(figsize=(15, 0.30 * n + 1.0), facecolor=th["surface"])
    ax = fig.add_axes([0.03, 0.01, 0.94, 0.90])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def yof(i):
        return 1.0 - (i + 0.5) / n

    def txt(x, i, s, color="ink", size=10, weight="normal", ha="left"):
        ax.text(x, yof(i), s, color=th[color], fontsize=size, family="monospace",
                weight=weight, ha=ha, va="center", transform=ax.transAxes)

    fig.suptitle("VINS-Fusion vs ORB-SLAM3  --  all datasets",
                 color=th["ink"], fontsize=15, x=0.03, ha="left", y=0.985)

    for i, (kind, payload) in enumerate(lines):
        if kind == "caption":
            txt(xs[0], i, "ATE rms in metres against ground truth, SE(3)-aligned, scale "
                          f"fixed.   DIVERGED = peak error > {DIVERGED_ERROR:.0f} m.",
                "ink3", 9)
        elif kind == "spanhdr":
            txt((xs[1] + xs[4]) / 2, i, "VINS-Fusion", "ink2", 11, "bold", ha="center")
            txt((xs[4] + xs[7]) / 2, i, "ORB-SLAM3", "ink2", 11, "bold", ha="center")
            for a, b in ((xs[1], xs[4] - 0.012), (xs[4], xs[7] - 0.012)):
                ax.plot([a, b], [yof(i) - 0.4 / n] * 2, color=th["rule"], lw=1.0,
                        transform=ax.transAxes, clip_on=False)
        elif kind == "hdr":
            txt(xs[0], i, "dataset", "ink2", 10, "bold")
            for x, h in zip(xs[1:], ["ATE", "clean", "jmp", "ATE", "clean", "jmp",
                                     "secs", "path m", "GT m"]):
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
            txt(xs[0], i, dataset.split("/", 1)[1][:44], "ink", 10)
            for si, sysname in enumerate(SYSTEMS):
                s = d["systems"].get(sysname)
                base = 1 + si * 3
                if s is None:
                    txt(xs[base], i, "--", "ink3", 10, ha="right")
                    continue
                label, col = verdict(s, d["gt"])
                txt(xs[base], i, label, col, 10, ha="right")
                txt(xs[base + 1], i,
                    "--" if s["clean"] is None or s["self_ref"] else f"{s['clean']:.3f}",
                    "ink2", 10, ha="right")
                txt(xs[base + 2], i, str(s["jumps"]), "ink3", 10, ha="right")
            any_s = next(iter(d["systems"].values()))
            txt(xs[7], i, f"{any_s['secs']:.0f}", "ink2", 10, ha="right")
            txt(xs[8], i, num(any_s["path"], 9, 1).strip(), "ink2", 10, ha="right")
            txt(xs[9], i, f"{any_s['ref']:.1f}" if d["gt"] else "--", "ink2", 10,
                ha="right")

    path = os.path.join(OUT, f"summary_{theme_name}.png")
    fig.savefig(path, dpi=160, facecolor=th["surface"])
    plt.close(fig)
    return path


datasets = discover()
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
