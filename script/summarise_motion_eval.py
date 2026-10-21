#!/usr/bin/env python3
"""Aggregate motion-gate verifications and render report figures.

    python3 script/summarise_motion_eval.py

Reads output/motion_eval/<estimator>/<bag>/motion_summary.csv (written by
script/motion_votes_eval.py --out) for every run present and writes, next to
them in output/motion_eval/:

  summary_metrics.csv   one row per (estimator, bag) plus a pooled row per
                        estimator -- the companion of output/yolo_eval/
                        summary_metrics.csv for the motion gate
  summary_{light,dark}.png
                        left: what the mask was built from -- the fraction of
                        truly-moving and truly-static box-frames the gate
                        painted (class mode is 1.0 / 1.0, no filter 0 / 0);
                        right: per-vote recall and false alarm of the census
                        at the run's own thresholds

The two panels answer different questions. The left one is the operating
point the SLAM tracker actually experienced, inherited verdicts and all. The
right one is the quality of the signal on the frames where a vote was cast.
A gate can have a clean census and still mask little, if few boxes ever reach
the vote (VINS), or a noisy census and mask a lot (ORB-SLAM3).
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "motion_eval"
ESTIMATORS = ("orb", "vins")
# Two world FAMILIES, pooled separately and never together: dyn_00_* are the
# retired `_nofloortexture` recordings (also used by Experiment 2) and
# realistic_* are the current generated warehouses. They differ in layout, prop
# inventory, mover count and route, so a pooled row across both would be an
# average over two different experiments (skill section 4).
FAMILIES = {
    "nofloortexture": ("dyn_00_000", "dyn_00_001"),
    "realistic": ("realistic_002", "realistic_003"),
}
BAG_ORDER = tuple(b for bags in FAMILIES.values() for b in bags)

POOL_INT = ("box_frames", "moving_unknown", "moving_static", "moving_moving",
            "static_unknown", "static_static", "static_moving",
            "none_unknown", "none_static", "none_moving",
            "vote_n_moving", "vote_n_static")


def load() -> list[dict]:
    rows = []
    for est in ESTIMATORS:
        for bag in BAG_ORDER:
            p = OUT / est / bag / "motion_summary.csv"
            if not p.exists():
                continue
            r = next(csv.DictReader(p.open()))
            r = {k: v for k, v in r.items()}
            r["estimator"], r["dataset"] = est, bag
            r["family"] = next(f for f, bs in FAMILIES.items() if bag in bs)
            rows.append(r)
    return rows


def pooled(est: str, family: str, rows: list[dict]) -> dict | None:
    mine = [r for r in rows if r["estimator"] == est and r["dataset"] in FAMILIES[family]]
    if not mine:
        return None
    p = {"estimator": est, "dataset": f"Pooled {family}"}
    for k in POOL_INT:
        p[k] = sum(int(r[k]) for r in mine)
    n_mov = p["moving_unknown"] + p["moving_static"] + p["moving_moving"]
    n_sta = p["static_unknown"] + p["static_static"] + p["static_moving"]
    p["moving_masked_frac"] = p["moving_moving"] / n_mov if n_mov else float("nan")
    p["static_masked_frac"] = p["static_moving"] / n_sta if n_sta else float("nan")
    # Per-vote metrics pool by their counts: recall = sum(tp)/sum(n_moving).
    tp = sum(float(r["vote_recall"]) * int(r["vote_n_moving"]) for r in mine)
    fs = sum(float(r["vote_false_alarm"]) * int(r["vote_n_static"]) for r in mine)
    p["vote_recall"] = tp / p["vote_n_moving"] if p["vote_n_moving"] else float("nan")
    p["vote_false_alarm"] = fs / p["vote_n_static"] if p["vote_n_static"] else float("nan")
    for k in ("min_points", "ratio", "min_box_age"):
        p[k] = mine[0][k]
    return p


def main() -> None:
    rows = load()
    if not rows:
        raise SystemExit(f"no motion_summary.csv under {OUT}")
    for est in ESTIMATORS:
        for family in FAMILIES:
            p = pooled(est, family, rows)
            if p:
                rows.append(p)

    cols = ["estimator", "dataset", "family", "box_frames",
            "moving_moving", "moving_static", "moving_unknown",
            "static_moving", "static_static", "static_unknown",
            "moving_masked_frac", "static_masked_frac",
            "vote_n_moving", "vote_n_static", "vote_recall", "vote_false_alarm",
            "min_points", "ratio", "min_box_age"]
    with (OUT / "summary_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    themes = {
        "light": {"bg": "#ffffff", "ink": "#18212b", "grid": "#d9e0e6"},
        "dark": {"bg": "#10151c", "ink": "#eef3f7", "grid": "#39434d"},
    }
    est_color = {"orb": "#3575d3", "vins": "#cf4f83"}
    bags = []
    for family, members in FAMILIES.items():
        present = [b for b in members if any(r["dataset"] == b for r in rows)]
        bags += present
        if present:
            bags.append(f"Pooled {family}")
    labels = [b.replace("dyn_", "dyn ").replace("realistic_", "real ").replace("_", "/")
              for b in bags]
    x = np.arange(len(bags))

    def series(est, key):
        by = {r["dataset"]: r for r in rows if r["estimator"] == est}
        return [float(by[b][key]) if b in by else np.nan for b in bags]

    for name, theme in themes.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=theme["bg"])
        for ax in axes:
            ax.set_facecolor(theme["bg"])
            ax.tick_params(colors=theme["ink"], labelsize=8)
            ax.grid(axis="y", color=theme["grid"], linewidth=0.7, zorder=0)
            for spine in ax.spines.values():
                spine.set_color(theme["grid"])

        width = 0.19
        # left: masked fractions
        slots = [("orb", "moving_masked_frac", "ORB: moving masked", 1.0),
                 ("orb", "static_masked_frac", "ORB: static masked", 0.45),
                 ("vins", "moving_masked_frac", "VINS: moving masked", 1.0),
                 ("vins", "static_masked_frac", "VINS: static masked", 0.45)]
        for offset, (est, key, title, alpha) in zip((-1.5, -0.5, 0.5, 1.5), slots):
            axes[0].bar(x + offset * width, series(est, key), width, label=title,
                        color=est_color[est], alpha=alpha, zorder=2)
        axes[0].axhline(1.0, color=theme["grid"], linewidth=0.9, linestyle="--", zorder=1)
        axes[0].text(x[-1] + 0.45, 0.97, "class mode = 1.0 / 1.0", ha="right", va="top",
                     fontsize=7, color=theme["ink"])
        # Families are separate experiments; mark the boundary rather than
        # letting the eye average across it.
        n_first = sum(1 for b in bags if b.startswith(("dyn", "Pooled nofloor")))
        if 0 < n_first < len(bags):
            for ax in axes:
                ax.axvline(n_first - 0.5, color=theme["grid"], linewidth=1.1, zorder=1)
        axes[0].set_ylim(0, 1.05)
        axes[0].set_ylabel("fraction of box-frames painted", color=theme["ink"])
        axes[0].legend(frameon=False, fontsize=8, labelcolor=theme["ink"], ncol=2)
        axes[0].set_title("What the mask was built from", color=theme["ink"], loc="left")

        # right: per-vote quality
        slots = [("orb", "vote_recall", "ORB: vote recall", 1.0),
                 ("orb", "vote_false_alarm", "ORB: vote false alarm", 0.45),
                 ("vins", "vote_recall", "VINS: vote recall", 1.0),
                 ("vins", "vote_false_alarm", "VINS: vote false alarm", 0.45)]
        for offset, (est, key, title, alpha) in zip((-1.5, -0.5, 0.5, 1.5), slots):
            axes[1].bar(x + offset * width, series(est, key), width, label=title,
                        color=est_color[est], alpha=alpha, zorder=2)
        axes[1].set_ylim(0, 1.05)
        axes[1].set_ylabel("per-vote rate", color=theme["ink"])
        axes[1].legend(frameon=False, fontsize=8, labelcolor=theme["ink"], ncol=2)
        axes[1].set_title("Census quality on frames that were voted", color=theme["ink"], loc="left")

        for ax in axes:
            ax.set_xticks(x, labels, rotation=18, ha="right")
        fig.suptitle("Motion-gated masking scored against /clock-synchronized Gazebo world poses",
                     x=0.06, ha="left", color=theme["ink"], fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(OUT / f"summary_{name}.png", dpi=180, facecolor=theme["bg"], bbox_inches="tight")
        plt.close(fig)

    print(f"wrote {OUT / 'summary_metrics.csv'} and summary_{{light,dark}}.png")
    for r in rows:
        if str(r["dataset"]).startswith("Pooled"):
            print(f"  {r['estimator']:5s} {r['dataset']:22s}: moving masked {r['moving_masked_frac']:.3f}  "
                  f"static masked {r['static_masked_frac']:.3f}  vote recall {r['vote_recall']:.3f}  "
                  f"false alarm {r['vote_false_alarm']:.3f}  ({r['box_frames']} box-frames)")


if __name__ == "__main__":
    main()
