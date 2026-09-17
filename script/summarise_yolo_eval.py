#!/usr/bin/env python3
"""Aggregate completed YOLO/Gazebo evaluations and render report figures."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "yolo_eval"
RUNS = {
    "dynamic_00_000": OUT / "dyn_00_000" / "yolo_eval.csv",
    "dynamic_00_001": OUT / "dyn_00_001" / "yolo_eval.csv",
    "dynamic_01_000": OUT / "dyn_01_000" / "yolo_eval.csv",
    "dynamic_02_000": OUT / "dyn_02_000" / "yolo_eval.csv",
}


def aggregate(label: str, paths: list[Path]) -> dict[str, float | int | str]:
    rows = []
    for path in paths:
        with path.open(newline="") as stream:
            rows.extend(csv.DictReader(stream))

    integer_columns = (
        "n_gt", "n_gt_moving", "n_gt_static", "n_ignored", "n_pred",
        "tp", "fp", "fn", "tp_moving", "fn_moving",
    )
    sums = {key: sum(int(row[key]) for row in rows) for key in integer_columns}
    precision = sums["tp"] / (sums["tp"] + sums["fp"])
    recall = sums["tp"] / (sums["tp"] + sums["fn"])
    moving_recall = sums["tp_moving"] / (sums["tp_moving"] + sums["fn_moving"])
    f1 = 2 * precision * recall / (precision + recall)
    mean = lambda key: sum(float(row[key]) for row in rows) / len(rows)
    return {
        "dataset": label,
        "frames": len(rows),
        **sums,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "moving_recall": moving_recall,
        "gt_area_frac": mean("gt_area_frac"),
        "gt_moving_area_frac": mean("gt_area_frac_moving"),
        "pred_area_frac": mean("pred_area_frac"),
    }


per_run = [aggregate(label, [path]) for label, path in RUNS.items()]
pooled = aggregate("Pooled", list(RUNS.values()))
summary = per_run + [pooled]

columns = list(summary[0])
with (OUT / "summary_metrics.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(summary)

themes = {
    "light": {"bg": "#ffffff", "ink": "#18212b", "grid": "#d9e0e6"},
    "dark": {"bg": "#10151c", "ink": "#eef3f7", "grid": "#39434d"},
}
colors = ["#3575d3", "#7450c8", "#cf4f83", "#df7b2f", "#252b33"]
labels = [row["dataset"].replace("dynamic_", "dyn ").replace("_", "/") for row in summary]
x = np.arange(len(summary))

for name, theme in themes.items():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=theme["bg"])
    for ax in axes:
        ax.set_facecolor(theme["bg"])
        ax.tick_params(colors=theme["ink"], labelsize=8)
        ax.grid(axis="y", color=theme["grid"], linewidth=0.7, zorder=0)
        for spine in ax.spines.values():
            spine.set_color(theme["grid"])

    width = 0.19
    metrics = [("precision", "Precision"), ("recall", "Recall"),
               ("f1", "F1"), ("moving_recall", "Moving recall")]
    for offset, (key, title) in zip((-1.5, -0.5, 0.5, 1.5), metrics):
        axes[0].bar(x + offset * width, [row[key] for row in summary], width,
                    label=title, zorder=2)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("score", color=theme["ink"])
    axes[0].legend(frameon=False, fontsize=8, labelcolor=theme["ink"], ncol=2)
    axes[0].set_title("IoU-matched detector performance", color=theme["ink"], loc="left")

    coverage = [("gt_area_frac", "GT all"),
                ("gt_moving_area_frac", "GT moving"),
                ("pred_area_frac", "YOLO")]
    for offset, (key, title) in zip((-1, 0, 1), coverage):
        axes[1].bar(x + offset * 0.25, [100 * row[key] for row in summary], 0.24,
                    label=title, zorder=2)
    axes[1].set_ylabel("mean image area [%]", color=theme["ink"])
    axes[1].legend(frameon=False, fontsize=8, labelcolor=theme["ink"])
    axes[1].set_title("Projected and predicted box coverage", color=theme["ink"], loc="left")

    for ax in axes:
        ax.set_xticks(x, labels, rotation=18, ha="right")
    fig.suptitle("YOLO evaluation using /clock-synchronized Gazebo world poses",
                 x=0.06, ha="left", color=theme["ink"], fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / f"summary_{name}.png", dpi=180, facecolor=theme["bg"], bbox_inches="tight")
    plt.close(fig)

print(f"wrote {OUT / 'summary_metrics.csv'}")
print(f"pooled frames={pooled['frames']} TP={pooled['tp']} FP={pooled['fp']} FN={pooled['fn']}")
print(f"precision={pooled['precision']:.6f} recall={pooled['recall']:.6f} "
      f"F1={pooled['f1']:.6f} moving_recall={pooled['moving_recall']:.6f}")

# Pool the separate class-agnostic IoU associations. Rows are actual classes;
# columns are predictions. Background in the final column means a missed object,
# while background in the final row means an unmatched detection.
classes = ["bin", "box", "bucket", "background"]
matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
for path in (p.parent / "yolo_confusion.csv" for p in RUNS.values()):
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            matrix[classes.index(row["ground_truth"]),
                   classes.index(row["predicted"])] += int(row["count"])

with (OUT / "confusion_matrix.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["actual\\predicted", *classes])
    for label, values in zip(classes, matrix):
        writer.writerow([label, *values])

row_den = matrix.sum(axis=1, keepdims=True)
row_norm = np.divide(matrix, row_den, out=np.zeros_like(matrix, dtype=float),
                     where=row_den != 0)

for name, theme in themes.items():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=theme["bg"])
    displays = ((np.log10(matrix + 1), "Pooled counts", "log10(count + 1)"),
                (row_norm, "Row-normalized", "fraction of actual class"))
    for ax, (values, title, colorbar_label) in zip(axes, displays):
        ax.set_facecolor(theme["bg"])
        image = ax.imshow(values, cmap="Blues", vmin=0)
        ax.set_xticks(range(len(classes)), classes, rotation=20, ha="right")
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("predicted class", color=theme["ink"])
        ax.set_ylabel("actual ground-truth class", color=theme["ink"])
        ax.set_title(title, color=theme["ink"], loc="left")
        ax.tick_params(colors=theme["ink"], labelsize=9)
        for i in range(len(classes)):
            for j in range(len(classes)):
                label = f"{matrix[i, j]:,}" if ax is axes[0] else f"{100 * row_norm[i, j]:.1f}%"
                threshold = 0.55 * values.max() if values.max() else 0
                ax.text(j, i, label, ha="center", va="center", fontsize=9,
                        color="white" if values[i, j] > threshold else "#17212b")
        bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        bar.set_label(colorbar_label, color=theme["ink"], fontsize=8)
        bar.ax.tick_params(colors=theme["ink"], labelsize=7)
    fig.suptitle("YOLO object-detection confusion matrix (IoU >= 0.50)",
                 x=0.06, ha="left", color=theme["ink"], fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / f"confusion_matrix_{name}.png", dpi=180,
                facecolor=theme["bg"], bbox_inches="tight")
    plt.close(fig)

print(f"wrote {OUT / 'confusion_matrix.csv'}")

# Object-class assignment is conditional on successful class-agnostic
# localization. Background is deliberately excluded: it is a detection outcome,
# not one of the trained object classes.
object_classes = classes[:-1]
class_matrix = matrix[:-1, :-1]
class_support = class_matrix.sum(axis=1)
class_predicted = class_matrix.sum(axis=0)
class_tp = np.diag(class_matrix)
class_precision = np.divide(class_tp, class_predicted, out=np.zeros(3),
                            where=class_predicted != 0)
class_recall = np.divide(class_tp, class_support, out=np.zeros(3),
                         where=class_support != 0)
class_f1 = np.divide(2 * class_precision * class_recall,
                     class_precision + class_recall, out=np.zeros(3),
                     where=(class_precision + class_recall) != 0)
class_accuracy = class_tp.sum() / class_matrix.sum()

with (OUT / "classification_confusion_matrix.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["actual\\predicted", *object_classes])
    for label, values in zip(object_classes, class_matrix):
        writer.writerow([label, *values])

with (OUT / "classification_metrics.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["class", "support", "precision", "recall", "f1"])
    for i, label in enumerate(object_classes):
        writer.writerow([label, class_support[i], class_precision[i],
                         class_recall[i], class_f1[i]])
    writer.writerow(["macro_average", class_matrix.sum(), class_precision.mean(),
                     class_recall.mean(), class_f1.mean()])
    writer.writerow(["overall_accuracy", class_matrix.sum(), class_accuracy, "", ""])

class_row_norm = class_matrix / class_support[:, None]
for name, theme in themes.items():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), facecolor=theme["bg"])
    displays = ((np.log10(class_matrix + 1), "Localized-object counts", "log10(count + 1)"),
                (class_row_norm, "Recall-normalized by actual class", "fraction"))
    for ax, (values, title, colorbar_label) in zip(axes, displays):
        image = ax.imshow(values, cmap="Purples", vmin=0)
        ax.set_facecolor(theme["bg"])
        ax.set_xticks(range(3), object_classes, rotation=15, ha="right")
        ax.set_yticks(range(3), object_classes)
        ax.set_xlabel("predicted object class", color=theme["ink"])
        ax.set_ylabel("actual object class", color=theme["ink"])
        ax.set_title(title, color=theme["ink"], loc="left")
        ax.tick_params(colors=theme["ink"], labelsize=9)
        for i in range(3):
            for j in range(3):
                label = f"{class_matrix[i, j]:,}" if ax is axes[0] else f"{100 * class_row_norm[i, j]:.1f}%"
                threshold = 0.55 * values.max() if values.max() else 0
                ax.text(j, i, label, ha="center", va="center", fontsize=10,
                        color="white" if values[i, j] > threshold else "#17212b")
        bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        bar.set_label(colorbar_label, color=theme["ink"], fontsize=8)
        bar.ax.tick_params(colors=theme["ink"], labelsize=7)
    fig.suptitle("YOLO object-class assignment, conditional on IoU-matched localization",
                 x=0.04, ha="left", color=theme["ink"], fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(OUT / f"classification_confusion_{name}.png", dpi=180,
                facecolor=theme["bg"], bbox_inches="tight")
    plt.close(fig)

print(f"classification associations={class_matrix.sum()} "
      f"accuracy={class_accuracy:.6f} errors={class_matrix.sum() - class_tp.sum()}")
