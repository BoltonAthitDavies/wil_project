#!/usr/bin/env python3
"""Render relogged trajectory metrics as LaTeX tables for the minor report.

Two tables in Chapter 4 used to be matplotlib-rendered PNG images. Their numbers
are reproduced here as real LaTeX tabulars so that they are typeset in the
document font, stay selectable and searchable in the PDF, and remain traceable
to the CSV that produced them.

This is a formatter, not an analysis: it reads the finished metric CSV and
writes .tex files. It never touches a rosbag, estimator, or raw run directory.

Usage:
    python3 script/make_result_tables.py
"""

from __future__ import annotations

import csv
import sys

import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
METRICS = REPO / "output/compare/relogged_20260916/trajectory_metrics.csv"
RPE_SWEEP = REPO / "output/compare/relogged_20260916/rpe_metrics.csv"
RPE_DELTAS = (0.5, 1.0, 2.0, 5.0)
TABLE_DIR = REPO / "docs/report/minor_report/tables"

# Column order matches the retired figure so the two can be compared directly.
SYSTEMS = ["VINS-Fusion", "ORB-SLAM3", "VINS-Fusion+YOLO", "ORB-SLAM3+YOLO"]
SYSTEM_HEADINGS = {
    "VINS-Fusion": "VINS-Fusion",
    "ORB-SLAM3": "ORB-SLAM3",
    "VINS-Fusion+YOLO": r"VINS-Fusion+YOLO$^{*}$",
    "ORB-SLAM3+YOLO": "ORB-SLAM3+YOLO",
}

# The example run shown alongside the per-run trajectory diagnostic figure.
EXAMPLE_DATASET = "dataset_dynamic_nofloortexture_00_000"
EXAMPLE_REPEAT = "logging_20260916_01"

MISSING = "--"


def short_dataset(name: str) -> str:
    """dataset_dynamic_nofloortexture_00_000 -> dynamic 00\\_000.

    Every dataset in this campaign is a no-floor-texture world, so that shared
    part of the name carries no information inside the table and is stated in
    the caption instead.
    """
    stem = name.removeprefix("dataset_")
    for family in ("dynamic", "static"):
        prefix = f"{family}_nofloortexture_"
        if stem.startswith(prefix):
            return f"{family} {tex_escape(stem[len(prefix):])}"
    return tex_escape(stem)


def short_repeat(name: str) -> str:
    """logging_20260916_01 -> 01."""
    return name.rsplit("_", 1)[-1]


def tex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def fmt(value: str, decimals: int) -> str:
    """Format one metric, or MISSING when the run does not exist."""
    if value is None or value == "" or value.lower() == "nan":
        return MISSING
    number = float(value)
    if decimals == 0:
        return f"{round(number):d}"
    return f"{number:.{decimals}f}"


def read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def group_by_run(rows: list[dict[str, str]]):
    """Collapse per-system rows into one record per (dataset, repeat).

    Insertion order is preserved so the table keeps the campaign's own ordering:
    the four dynamic datasets first, then the three static ones.
    """
    runs: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in rows:
        runs.setdefault((row["dataset"], row["repeat"]), {})[row["system"]] = row
    return runs


def per_run_table(runs) -> str:
    """The wide per-run table: four metrics for each of four requested modes."""
    metrics = [
        ("trans_rmse_m", 2),
        ("rot_rmse_deg", 1),
        ("clean_rmse_m", 2),
        ("jumps", 0),
    ]

    lines = [
        r"\begin{tabular}{ll" + "rrrr" * len(SYSTEMS) + "}",
        r"\toprule",
    ]

    # Two-deep header: requested mode above, metric and unit below.
    spanned = " & ".join(
        rf"\multicolumn{{4}}{{c}}{{{SYSTEM_HEADINGS[system]}}}" for system in SYSTEMS
    )
    lines.append(rf"& & {spanned} \\")
    lines.append(
        "".join(
            rf"\cmidrule(lr){{{3 + 4 * index}-{6 + 4 * index}}}"
            for index in range(len(SYSTEMS))
        )
    )
    lines.append(
        "Dataset & Rep & "
        + " & ".join(["ATE & rot & clean & jmp"] * len(SYSTEMS))
        + r" \\"
    )
    lines.append(
        "& & "
        + " & ".join([r"(m) & (deg) & (m) & (n)"] * len(SYSTEMS))
        + r" \\"
    )
    lines.append(r"\midrule")

    previous_family = None
    for (dataset, repeat), by_system in runs.items():
        family = "static" if "_static_" in dataset else "dynamic"
        if previous_family is not None and family != previous_family:
            lines.append(r"\midrule")
        previous_family = family

        cells = [short_dataset(dataset), short_repeat(repeat)]
        for system in SYSTEMS:
            row = by_system.get(system)
            for column, decimals in metrics:
                cells.append(fmt(row[column], decimals) if row else MISSING)
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def example_run_table(runs) -> str:
    """The small per-system summary that belonged to the example-run figure."""
    by_system = runs[(EXAMPLE_DATASET, EXAMPLE_REPEAT)]
    columns = [
        ("poses", 0),
        ("overlap_s", 1),
        ("path_m", 2),
        ("trans_rmse_m", 3),
        ("rot_rmse_deg", 2),
        ("clean_rmse_m", 3),
        ("jumps", 0),
    ]

    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Requested mode & Poses & Overlap & Path & ATE RMSE & Rot RMSE"
        r" & Clean ATE & Jumps \\",
        r"& (n) & (s) & (m) & (m) & (deg) & (m) & (n) \\",
        r"\midrule",
    ]
    for system in SYSTEMS:
        row = by_system[system]
        cells = [SYSTEM_HEADINGS[system]]
        cells += [fmt(row[column], decimals) for column, decimals in columns]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def _num(row, key):
    v = row.get(key, "")
    return float(v) if v not in ("", "nan", None) else float("nan")


def _med_iqr(values, decimals):
    """Median (IQR). These distributions are strongly right-skewed on the dynamic
    runs -- one teleport sets the mean -- so median and IQR are reported instead of
    mean and standard deviation."""
    a = np.array([v for v in values if not np.isnan(v)], float)
    if a.size == 0:
        return MISSING
    med = np.median(a)
    iqr = np.percentile(a, 75) - np.percentile(a, 25)
    return f"{med:.{decimals}f}\\,({iqr:.{decimals}f})"


def _scene(dataset):
    return "static" if "_static_" in dataset else "dynamic"


def rpe_summary_table(rows):
    """RPE at the headline interval, split by scene and requested mode."""
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"& & & \multicolumn{2}{c}{Translational (m)} &"
             r" \multicolumn{2}{c}{Rotational (deg)} \\",
             r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"Scene & Requested mode & Jumps & all pairs & jump-free"
             r" & all pairs & jump-free \\", r"\midrule"]
    for scene in ("static", "dynamic"):
        for system in SYSTEMS:
            sel = [r for r in rows
                   if _scene(r["dataset"]) == scene and r["system"] == system]
            if not sel:
                continue
            cells = [scene.capitalize(), SYSTEM_HEADINGS[system],
                     _med_iqr([_num(r, "jumps") for r in sel], 0),
                     _med_iqr([_num(r, "rpe1_trans_rmse_m") for r in sel], 3),
                     _med_iqr([_num(r, "rpe1_trans_rmse_clean_m") for r in sel], 3),
                     _med_iqr([_num(r, "rpe1_rot_rmse_deg") for r in sel], 2),
                     _med_iqr([_num(r, "rpe1_rot_rmse_clean_deg") for r in sel], 2)]
            lines.append(" & ".join(cells) + r" \\")
        if scene == "static":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def rpe_sweep_table(path):
    """Jump-free RPE against interval: the error-growth curve."""
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    lines = [r"\begin{tabular}{ll" + "r" * len(RPE_DELTAS) + "r}", r"\toprule",
             r"Scene & Requested mode & "
             + " & ".join(rf"\SI{{{d}}}{{\second}}" for d in RPE_DELTAS)
             + r" & Growth \\", r"\midrule"]
    for scene in ("static", "dynamic"):
        for system in SYSTEMS:
            vals = []
            for d in RPE_DELTAS:
                sel = [_num(r, "trans_rmse_clean") for r in rows
                       if _scene(r["dataset"]) == scene and r["system"] == system
                       and abs(float(r["delta_s"]) - d) < 1e-9]
                a = np.array([v for v in sel if not np.isnan(v)], float)
                vals.append(np.median(a) if a.size else float("nan"))
            if np.isnan(vals[0]):
                continue
            growth = vals[-1] / vals[0] if vals[0] else float("nan")
            lines.append(" & ".join([scene.capitalize(), SYSTEM_HEADINGS[system]]
                                    + [f"{v:.3f}" for v in vals]
                                    + [f"{growth:.1f}$\\times$"]) + r" \\")
        if scene == "static":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> int:
    if not METRICS.is_file():
        print(f"missing metrics file: {METRICS}", file=sys.stderr)
        return 1

    runs = group_by_run(read_metrics(METRICS))
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    banner = (
        "% Generated by script/make_result_tables.py from\n"
        "% output/compare/relogged_20260916/trajectory_metrics.csv.\n"
        "% Do not edit by hand; regenerate instead.\n"
    )
    raw = read_metrics(METRICS)
    outputs = {
        "trajectory_per_run.tex": per_run_table(runs),
        "example_run_summary.tex": example_run_table(runs),
        "rpe_summary.tex": rpe_summary_table(raw),
        "rpe_sweep.tex": rpe_sweep_table(RPE_SWEEP),
    }
    for name, body in outputs.items():
        target = TABLE_DIR / name
        target.write_text(banner + body)
        print(f"wrote {target.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
