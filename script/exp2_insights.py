#!/usr/bin/env python3
"""Experiment 2 insight analysis: WHERE the detector fails, from the logged scores.

Reads the per-frame and per-prediction files that script/yolo_eval.py wrote for
the four scripted dynamic recordings (output/yolo_eval/dyn_*/) and tests, within
each recording, the mechanisms a reader would propose for the pooled scores:

  false positives   right object with a box below the 0.5 IoU threshold (near
                    miss), a second box on an object already detected (duplicate),
                    a box on unlabelled scenery (ignored by the scorer, but still
                    a detection the filter would act on), or a box on nothing
                    (hallucination); and where in the image they sit.
  score threshold   precision, recall and moving-object recall if the operating
                    point were raised above the logged 0.35 (predictions below
                    the threshold were never retained, so the sweep goes up only).
  box size          recall against the size of the truth boxes, at frame level
                    (the per-frame mean truth-box area) and at object level (each
                    object's median matched-box area), split moving / static.
  miss structure    whether an object's misses come as long runs (it is not seen
                    for seconds) or as flicker (single frames between hits),
                    from the gaps between consecutive matches of the same object.
  per class         which class carries the misses and the false positives.

The ground-truth boxes themselves are not logged per box, only per frame as
counts and union area, so every false-negative size here is a proxy: at frame
level the mean truth area of the frame, at object level the object's own box in
the frames where it WAS matched.  Gaps between matches of one object are read
as misses only when shorter than GAP_MAX_S, because an object can also leave the
view.  Unmatched predictions include the ones the scorer ignored as scenery;
they are separated by the per-frame ignored count, not per box.

Outputs (output/compare/simulation/exp2-insights/):
  fp_anatomy.csv / fp_anatomy.tex        what each unmatched prediction is
  threshold_sweep.csv / threshold_sweep.tex
  recall_by_size.csv / recall_by_size.tex  frame-level and object-level
  miss_structure.csv / miss_structure.tex  gaps and never-seen objects
  per_class.tex                            misses and FPs by class
  box_shape.csv / box_shape.tex            matched vs near-miss box shape, score, IoU
  fig_fp_anatomy_light.png, fig_threshold_light.png, fig_size_light.png,
  fig_gaps_light.png
"""
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = ROOT / "output" / "yolo_eval"
OUT = ROOT / "output" / "compare" / "simulation" / "exp2-insights"
# 7 Nov 2026: the four realistic recordings, 4-class checkpoint (see summarise_yolo_eval.py).
DATASETS = ["dataset_allsensor_realistic_%03d_4class" % i for i in range(4)]
SHORT = {d: "realistic_%03d" % i for i, d in enumerate(DATASETS)}
W, H = 1280.0, 720.0
IOU_MATCH = 0.5
NEAR_MISS_IOU = 0.2        # unmatched prediction with best IoU in [0.2, 0.5): the right object, wrong box
DUP_IOU = 0.5              # unmatched prediction overlapping a MATCHED same-class prediction by this much
GAP_MAX_S = 2.0            # a gap between two matches of one object counts as misses only below this
FRAME_DT = 1 / 30.0
THRESHOLDS = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
SIZE_EDGES = [0, 24, 32, 48, 64, 96, 128, 192, 10000]   # sqrt(area) in px
COL = dict(zip(DATASETS, ["#2a78d6", "#cc8319", "#458554", "#a347b2"]))


def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load(ds):
    frames = list(csv.DictReader(open(SRC / ds / "yolo_eval.csv")))
    dets = list(csv.DictReader(open(SRC / ds / "yolo_detections.csv")))
    for r in dets:
        r["t"] = int(r["timestamp_ns"]); r["score"] = float(r["score"]); r["matched"] = int(r["matched"])
        r["box"] = (float(r["x0"]), float(r["y0"]), float(r["x1"]), float(r["y1"]))
        r["iou"] = float(r["iou"]) if r["iou"] else 0.0
        r["w"], r["h"] = r["box"][2] - r["box"][0], r["box"][3] - r["box"][1]
        r["size"] = float(np.sqrt(max(r["w"] * r["h"], 0.0)))
        r["moving"] = (r["gt_moving"] == "1") if r["matched"] else None
    for f in frames:
        for k in ("n_gt", "n_gt_moving", "n_gt_static", "n_ignored", "n_pred", "tp", "fp", "fn", "tp_moving", "fn_moving"):
            f[k] = int(f[k])
        f["t"] = int(f["timestamp_ns"]); f["gt_area_frac"] = float(f["gt_area_frac"])
    return frames, dets


# ------------------------------------------------------------ false positives

def fp_anatomy(ds, frames, dets):
    by_t = defaultdict(list)
    for r in dets:
        by_t[r["t"]].append(r)
    n_unm = n_dup = n_near = n_tiny = n_zero = n_border = 0
    fp_centres = []
    for t, rs in by_t.items():
        matched = [r for r in rs if r["matched"]]
        for r in rs:
            if r["matched"]:
                continue
            n_unm += 1
            dup = any(r["pred_class"] == m["pred_class"] and iou(r["box"], m["box"]) >= DUP_IOU for m in matched)
            b = r["box"]
            if b[0] <= 1 or b[1] <= 1 or b[2] >= W - 1 or b[3] >= H - 1:
                n_border += 1
            fp_centres.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
            if dup:
                n_dup += 1
            elif r["iou"] >= NEAR_MISS_IOU:
                n_near += 1
            elif r["iou"] > 0:
                n_tiny += 1
            else:
                n_zero += 1
    n_ign = sum(f["n_pred"] - f["tp"] - f["fp"] for f in frames)
    n_fp = sum(f["fp"] for f in frames)
    return dict(dataset=ds, unmatched=n_unm, scored_fp=n_fp, ignored_scenery=n_ign,
                duplicate=n_dup, near_miss=n_near, slight_overlap=n_tiny, no_overlap=n_zero,
                border=n_border, centres=np.array(fp_centres))


def box_shape(ds, frames, dets):
    """Matched against near-miss predictions: box side, aspect ratio, score, IoU;
    and the share of frames with a false negative in which at least one near-miss
    prediction of the frame exists."""
    m = [r for r in dets if r["matched"]]
    near = [r for r in dets if not r["matched"] and NEAR_MISS_IOU <= r["iou"] < IOU_MATCH]
    none = [r for r in dets if not r["matched"] and r["iou"] == 0]
    def med(rs, key):
        return float(np.median([key(r) for r in rs])) if rs else np.nan
    near_t = {r["t"] for r in near}
    fn_frames = [f for f in frames if f["fn"] > 0]
    return dict(dataset=ds,
                matched_iou_med=med(m, lambda r: r["iou"]), matched_iou_p25=float(np.percentile([r["iou"] for r in m], 25)),
                near_iou_med=med(near, lambda r: r["iou"]),
                side_matched=med(m, lambda r: r["size"]), side_near=med(near, lambda r: r["size"]), side_none=med(none, lambda r: r["size"]),
                aspect_matched=med(m, lambda r: r["w"] / max(r["h"], 1)), aspect_near=med(near, lambda r: r["w"] / max(r["h"], 1)),
                score_matched=med(m, lambda r: r["score"]), score_near=med(near, lambda r: r["score"]), score_none=med(none, lambda r: r["score"]),
                fn_frames=len(fn_frames), fn_frames_with_near=sum(1 for f in fn_frames if f["t"] in near_t),
                near_count=len(near), fn_count=sum(f["fn"] for f in frames))


def tex_shape(rows, path):
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption[Matched boxes against near misses.]{Matched predictions (IoU $\ge 0.5$) against near misses "
         r"(best IoU in $[0.2, 0.5)$ with an unmatched truth box of the class): median IoU, box side, width-to-height "
         r"ratio and score. The last columns give how many of the scorer's false negatives have a near miss standing "
         r"next to them: the share of frames with at least one false negative that also hold at least one near-miss "
         r"prediction, and the near-miss count against the false-negative count. Source: \path{box_shape.csv}.}",
         r"\label{tab:e2i-shape}",
         r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
         r"Recording & \multicolumn{2}{c}{IoU median} & \multicolumn{2}{c}{side [px]} & \multicolumn{2}{c}{w/h} & \multicolumn{2}{c}{score} & FN frames with \\",
         r" & matched & near & matched & near & matched & near & matched & near & a near miss; near/FN \\", r"\midrule"]
    for r in rows:
        L.append(f"\\repo{{{r['dataset']}}} & {fmt(r['matched_iou_med'])} & {fmt(r['near_iou_med'])} & {fmt(r['side_matched'],0)} & {fmt(r['side_near'],0)} & "
                 f"{fmt(r['aspect_matched'])} & {fmt(r['aspect_near'])} & {fmt(r['score_matched'])} & {fmt(r['score_near'])} & "
                 f"{100*r['fn_frames_with_near']/max(r['fn_frames'],1):.0f}\\,\\%; {r['near_count']}/{r['fn_count']} \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    path.write_text("\n".join(L) + "\n")


# ------------------------------------------------------------ score threshold

def threshold_sweep(ds, frames, dets):
    tot_gt = sum(f["tp"] + f["fn"] for f in frames)
    tot_mv = sum(f["tp_moving"] + f["fn_moving"] for f in frames)
    tp_all = sum(f["tp"] for f in frames)
    fp_all = sum(f["fp"] for f in frames)
    # matched predictions whose truth was ignored are in dets as matched but not in tp;
    # scale by the scored share so that the sweep reproduces the pooled numbers at 0.35
    m = [r for r in dets if r["matched"]]
    u = [r for r in dets if not r["matched"]]
    k_tp = tp_all / max(len(m), 1)
    k_fp = fp_all / max(len(u), 1)
    rows = []
    for th in THRESHOLDS:
        tp = sum(1 for r in m if r["score"] >= th) * k_tp
        tpm = sum(1 for r in m if r["score"] >= th and r["moving"]) * (sum(f["tp_moving"] for f in frames) / max(sum(1 for r in m if r["moving"]), 1))
        fp = sum(1 for r in u if r["score"] >= th) * k_fp
        rows.append(dict(dataset=ds, threshold=th, precision=tp / max(tp + fp, 1e-9), recall=tp / max(tot_gt, 1),
                         moving_recall=tpm / max(tot_mv, 1), n_pred=tp + fp))
    return rows


# ------------------------------------------------------------------ box size

def recall_by_size_frame(ds, frames):
    """Frame-level: recall of the frame against the mean truth-box side (sqrt of
    mean truth area), with the frame's moving/static split."""
    rows = []
    for f in frames:
        n = f["tp"] + f["fn"]
        if n == 0:
            continue
        mean_area = f["gt_area_frac"] * W * H / max(f["n_gt"], 1)
        rows.append(dict(size=np.sqrt(max(mean_area, 0.0)), tp=f["tp"], fn=f["fn"], tpm=f["tp_moving"], fnm=f["fn_moving"]))
    out = []
    for lo, hi in zip(SIZE_EDGES[:-1], SIZE_EDGES[1:]):
        sel = [r for r in rows if lo <= r["size"] < hi]
        tp, fn = sum(r["tp"] for r in sel), sum(r["fn"] for r in sel)
        tpm, fnm = sum(r["tpm"] for r in sel), sum(r["fnm"] for r in sel)
        out.append(dict(dataset=ds, level="frame", lo=lo, hi=hi, n_boxes=tp + fn, recall=tp / (tp + fn) if tp + fn else np.nan,
                        n_moving=tpm + fnm, moving_recall=tpm / (tpm + fnm) if tpm + fnm else np.nan,
                        static_recall=(tp - tpm) / ((tp + fn) - (tpm + fnm)) if (tp + fn) - (tpm + fnm) else np.nan))
    return out


def tracks(ds, frames, dets):
    """Per object: matched frame indices, their sizes and moving flags."""
    ts = sorted(f["t"] for f in frames)
    idx = {t: i for i, t in enumerate(ts)}
    tr = defaultdict(list)
    for r in dets:
        if r["matched"] and r["t"] in idx:
            tr[r["gt_name"]].append((idx[r["t"]], r["size"], r["moving"], r["gt_class"]))
    for k in tr:
        tr[k].sort()
    return tr, len(ts)


def recall_by_size_object(ds, frames, dets):
    """Object-level: for each object, misses are the frames inside gaps shorter
    than GAP_MAX_S between consecutive matches; recall = matches / (matches + misses)."""
    tr, nf = tracks(ds, frames, dets)
    gap_frames = int(GAP_MAX_S / FRAME_DT)
    objs = []
    for name, seq in tr.items():
        f = np.array([s[0] for s in seq]); sizes = np.array([s[1] for s in seq])
        mv = np.array([s[2] for s in seq], bool)
        gaps = np.diff(f) - 1
        miss = int(gaps[(gaps > 0) & (gaps <= gap_frames)].sum())
        objs.append(dict(dataset=ds, name=name, cls=seq[0][3], n_match=len(f), n_miss=miss,
                         recall=len(f) / (len(f) + miss), size=float(np.median(sizes)),
                         moving_share=float(mv.mean()), gaps=gaps))
    rows = []
    for lo, hi in zip(SIZE_EDGES[:-1], SIZE_EDGES[1:]):
        for label, pred in (("moving", lambda o: o["moving_share"] >= 0.5), ("static", lambda o: o["moving_share"] < 0.5)):
            sel = [o for o in objs if lo <= o["size"] < hi and pred(o)]
            nm, nmi = sum(o["n_match"] for o in sel), sum(o["n_miss"] for o in sel)
            rows.append(dict(dataset=ds, level="object", motion=label, lo=lo, hi=hi, n_objects=len(sel),
                             n_frames=nm + nmi, recall=nm / (nm + nmi) if nm + nmi else np.nan))
    return rows, objs


# ------------------------------------------------------------ miss structure

def miss_structure(ds, frames, objs):
    all_gaps = np.concatenate([o["gaps"] for o in objs]) if objs else np.array([])
    gap_frames = int(GAP_MAX_S / FRAME_DT)
    g = all_gaps[(all_gaps > 0) & (all_gaps <= gap_frames)]
    n_gt_total = sum(f["tp"] + f["fn"] for f in frames)
    n_fn_total = sum(f["fn"] for f in frames)
    return dict(dataset=ds, n_objects=len(objs), gaps=int(len(g)), gap_frames_total=int(g.sum()),
                gap_median=float(np.median(g)) if len(g) else np.nan, gap_p90=float(np.percentile(g, 90)) if len(g) else np.nan,
                single_frame_share=float((g == 1).mean()) if len(g) else np.nan,
                long_gap_share_of_misses=float(g[g >= 30].sum() / g.sum()) if len(g) and g.sum() else np.nan,
                fn_total=n_fn_total, fn_in_short_gaps=int(g.sum()), gt_total=n_gt_total,
                objects_low_recall=sum(1 for o in objs if o["recall"] < 0.5 and o["n_match"] + o["n_miss"] >= 30))


# ------------------------------------------------------------------ per class

def per_class(ds):
    conf = list(csv.DictReader(open(SRC / ds / "yolo_confusion.csv")))
    c = {(r["ground_truth"], r["predicted"]): int(r["count"]) for r in conf}
    classes = sorted({k[0] for k in c} - {"background"})
    rows = []
    for cl in classes:
        hit = sum(v for (g, p), v in c.items() if g == cl and p != "background")
        miss = c.get((cl, "background"), 0)
        fp = c.get(("background", cl), 0)
        rows.append(dict(dataset=ds, cls=cl, gt=hit + miss, recall=hit / (hit + miss) if hit + miss else np.nan,
                         fp=fp, fp_share=np.nan))
    tot_fp = sum(r["fp"] for r in rows)
    for r in rows:
        r["fp_share"] = r["fp"] / tot_fp if tot_fp else np.nan
    return rows


# --------------------------------------------------------------------- tables

def fmt(v, p=2):
    return "---" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{p}f}"


def tex_fp(rows, path):
    L = [r"\begin{table}[htbp]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\caption[What the unmatched predictions are.]{What the unmatched predictions are, per recording. "
         r"An unmatched prediction is a \emph{duplicate} when it overlaps a matched prediction of the same class by "
         r"IoU $\ge 0.5$; a \emph{near miss} when its best IoU with an unmatched truth box of its class is in "
         r"$[0.2, 0.5)$; \emph{slight overlap} when that IoU is in $(0, 0.2)$; \emph{no overlap} when it touches no "
         r"truth box. The scorer excludes the predictions that land on unlabelled scenery (\emph{ignored}); they are "
         r"counted here from the per-frame totals because the per-box file does not flag them, so the four categories "
         r"sum to unmatched, not to scored false positives. ``At border'': the box touches an image edge. Source: "
         r"\path{output/compare/simulation/exp2-insights/fp_anatomy.csv}.}",
         r"\label{tab:e2i-fp}",
         r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
         r"Recording & unmatched & scored FP & ignored & duplicate & near miss & slight & none & at border \\",
         r" & & & (scenery) & & [0.2,0.5) & (0,0.2) & 0 & \\", r"\midrule"]
    for r in rows:
        u = r["unmatched"]
        L.append(f"\\repo{{{r['dataset']}}} & {u} & {r['scored_fp']} & {r['ignored_scenery']} & "
                 f"{r['duplicate']} ({100*r['duplicate']/u:.0f}\\,\\%) & {r['near_miss']} ({100*r['near_miss']/u:.0f}\\,\\%) & "
                 f"{r['slight_overlap']} ({100*r['slight_overlap']/u:.0f}\\,\\%) & {r['no_overlap']} ({100*r['no_overlap']/u:.0f}\\,\\%) & "
                 f"{100*r['border']/u:.0f}\\,\\% \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    path.write_text("\n".join(L) + "\n")


def tex_threshold(rows, path):
    L = [r"\begin{table}[htbp]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\caption[Score-threshold sweep.]{Score-threshold sweep over the retained predictions. Precision, recall and "
         r"moving-object recall if every prediction below the threshold were discarded; the logged operating point "
         r"is \num{0.35}, and nothing below it was retained, so the sweep can only go up. Matching is not re-run, so "
         r"a discarded higher-scored box cannot free its truth for a lower-scored one; the values are therefore "
         r"lower bounds on recall by a small margin. Source: \path{threshold_sweep.csv}.}",
         r"\label{tab:e2i-threshold}",
         r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{l" + "rrr" * len(DATASETS) + "}", r"\toprule",
         r"Threshold & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{\\repo{{{SHORT[d]}}}}}" for d in DATASETS) + r" \\",
         r" & " + " & ".join("P & R & R$_{\\mathrm{mov}}$" for _ in DATASETS) + r" \\", r"\midrule"]
    for th in THRESHOLDS:
        cells = []
        for d in DATASETS:
            r = next(x for x in rows if x["dataset"] == d and x["threshold"] == th)
            cells += [fmt(r["precision"]), fmt(r["recall"]), fmt(r["moving_recall"])]
        L.append(f"{th:.2f} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    path.write_text("\n".join(L) + "\n")


def tex_size(frame_rows, obj_rows, path):
    L = [r"\begin{table}[htbp]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\caption[Recall against truth-box size.]{Recall against truth-box size, pooled over the four recordings. "
         r"Frame level: the frame's truth boxes are binned by the frame's mean truth-box side (square root of the mean "
         r"truth area; the per-box truth is not logged), recall is TP/(TP+FN) of those frames. Object level: each "
         r"object's median matched-box side; a miss is a frame inside a gap shorter than \SI{2}{\second} between two "
         r"matches of that object, so objects that leave the view are not charged. Moving: object moving in at "
         r"least half of its matched frames. Source: \path{recall_by_size.csv}.}",
         r"\label{tab:e2i-size}",
         r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lrrrrrrr}", r"\toprule",
         r"Side [px] & \multicolumn{3}{c}{frame level} & \multicolumn{4}{c}{object level} \\",
         r" & boxes & recall & R$_{\mathrm{mov}}$ / R$_{\mathrm{stat}}$ & objects mov/stat & frames & R$_{\mathrm{mov}}$ & R$_{\mathrm{stat}}$ \\", r"\midrule"]
    for lo, hi in zip(SIZE_EDGES[:-1], SIZE_EDGES[1:]):
        fr = [r for r in frame_rows if r["lo"] == lo]
        nb = sum(r["n_boxes"] for r in fr)
        tp = sum(r["recall"] * r["n_boxes"] for r in fr if np.isfinite(r["recall"]))
        nm = sum(r["n_moving"] for r in fr)
        tpm = sum(r["moving_recall"] * r["n_moving"] for r in fr if np.isfinite(r["moving_recall"]))
        ns = nb - nm; tps = tp - tpm
        om = [r for r in obj_rows if r["lo"] == lo and r["motion"] == "moving"]
        os_ = [r for r in obj_rows if r["lo"] == lo and r["motion"] == "static"]
        nfm, nfs = sum(r["n_frames"] for r in om), sum(r["n_frames"] for r in os_)
        rm = sum(r["recall"] * r["n_frames"] for r in om if np.isfinite(r["recall"])) / nfm if nfm else np.nan
        rs = sum(r["recall"] * r["n_frames"] for r in os_ if np.isfinite(r["recall"])) / nfs if nfs else np.nan
        hi_s = f"{hi}" if hi < 10000 else "$\\infty$"
        L.append(f"{lo}--{hi_s} & {nb} & {fmt(tp/nb) if nb else '---'} & {fmt(tpm/nm) if nm else '---'} / {fmt(tps/ns) if ns else '---'} & "
                 f"{sum(r['n_objects'] for r in om)}/{sum(r['n_objects'] for r in os_)} & {nfm + nfs} & {fmt(rm)} & {fmt(rs)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    path.write_text("\n".join(L) + "\n")


def tex_miss(rows, path):
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption[Structure of the misses.]{Structure of the misses, per recording, from the gaps between "
         r"consecutive matches of the same object (gaps under \SI{2}{\second} only). ``Single-frame'' is the share "
         r"of gaps one frame long; ``in long gaps'' the share of the gap frames that lie in gaps of \SI{1}{\second} "
         r"or more. ``Short-gap misses / FN'' compares the misses this reading finds with the scorer's total false "
         r"negatives: the remainder are misses at the ends of tracks or in gaps longer than \SI{2}{\second}, which "
         r"this reading cannot separate from objects leaving the view. ``Objects under 0.5'': objects with at least "
         r"30 frames whose in-track recall is below one half. Source: \path{miss_structure.csv}.}",
         r"\label{tab:e2i-miss}",
         r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lrrrrrrr}", r"\toprule",
         r"Recording & objects & gaps & median & p90 & single-frame & in long gaps & short-gap misses / FN \\",
         r" & & & [frames] & [frames] & & & \\", r"\midrule"]
    for r in rows:
        L.append(f"\\repo{{{r['dataset']}}} & {r['n_objects']} ({r['objects_low_recall']} under 0.5) & {r['gaps']} & {fmt(r['gap_median'],0)} & {fmt(r['gap_p90'],0)} & "
                 f"{fmt(100*r['single_frame_share'],0)}\\,\\% & {fmt(100*r['long_gap_share_of_misses'],0)}\\,\\% & "
                 f"{r['fn_in_short_gaps']} / {r['fn_total']} ({100*r['fn_in_short_gaps']/max(r['fn_total'],1):.0f}\\,\\%) \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    path.write_text("\n".join(L) + "\n")


def tex_class(rows, path):
    L = [r"\begin{table}[htbp]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\caption[Misses and false positives by class.]{Misses and false positives by class, per recording, from "
         r"the scorer's class-agnostic confusion counts. Recall is the share of truth boxes of the class associated "
         r"with any prediction; FP share is the class's share of the recording's background-row predictions. "
         r"The pallet jack is not a class of the loaded 3-class weights and is an ignore region, so it does not "
         r"appear. Source: \path{output/yolo_eval/dataset_allsensor_realistic_*_4class/yolo_confusion.csv}.}",
         r"\label{tab:e2i-class}",
         r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{ll" + "rrr" * len(DATASETS) + "}", r"\toprule",
         r" & & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{\\repo{{{SHORT[d]}}}}}" for d in DATASETS) + r" \\",
         r"Class & & " + " & ".join("truth & recall & FP share" for _ in DATASETS) + r" \\", r"\midrule"]
    for cl in sorted({r["cls"] for r in rows}):
        cells = []
        for d in DATASETS:
            r = next((x for x in rows if x["dataset"] == d and x["cls"] == cl), None)
            cells += [str(r["gt"]), fmt(r["recall"]), fmt(r["fp_share"])] if r else ["---"] * 3
        L.append(f"{cl} & & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    path.write_text("\n".join(L) + "\n")


# -------------------------------------------------------------------- figures

def fig_fp(anat, dets_by_ds, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    for ds in DATASETS:
        u = np.array([r["iou"] for r in dets_by_ds[ds] if not r["matched"]])
        ax.hist(u, bins=np.linspace(0, 0.5, 26), histtype="step", lw=1.4, color=COL[ds], label=SHORT[ds], density=True)
    ax.axvline(NEAR_MISS_IOU, color="#888", ls=":", lw=0.9)
    ax.set_xlabel("best IoU of an unmatched prediction with an unmatched truth box of its class")
    ax.set_ylabel("density"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title("Unmatched predictions: how close to a truth box", fontsize=10, loc="left")
    ax = axes[1]
    c = np.concatenate([a["centres"] for a in anat])
    hb = ax.hexbin(c[:, 0], c[:, 1], gridsize=28, extent=(0, W, 0, H), cmap="Purples", mincnt=1)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal")
    ax.set_xlabel("x [px]"); ax.set_ylabel("y [px]")
    ax.set_title("Where the unmatched boxes sit (all recordings)", fontsize=10, loc="left")
    fig.colorbar(hb, ax=ax, shrink=0.8, label="boxes")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def fig_threshold(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ds in DATASETS:
        rr = [r for r in rows if r["dataset"] == ds]
        axes[0].plot([r["recall"] for r in rr], [r["precision"] for r in rr], "o-", color=COL[ds], label=SHORT[ds], ms=4)
        for r in rr:
            axes[0].annotate(f"{r['threshold']:.2f}", (r["recall"], r["precision"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
        axes[1].plot([r["threshold"] for r in rr], [r["moving_recall"] for r in rr], "o-", color=COL[ds], ms=4, label=f"{SHORT[ds]} moving")
        axes[1].plot([r["threshold"] for r in rr], [r["precision"] for r in rr], "s--", color=COL[ds], ms=3, alpha=0.6)
    axes[0].set_xlabel("recall"); axes[0].set_ylabel("precision"); axes[0].grid(True, alpha=0.3); axes[0].legend(fontsize=8)
    axes[0].set_title("Precision against recall as the threshold rises from 0.35", fontsize=10, loc="left")
    axes[1].set_xlabel("score threshold"); axes[1].set_ylabel("moving-object recall (solid), precision (dashed)")
    axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=7)
    axes[1].set_title("What a higher threshold costs in moving-object recall", fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def fig_size(frame_rows, obj_rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    centres = [(lo + min(hi, 256)) / 2 for lo, hi in zip(SIZE_EDGES[:-1], SIZE_EDGES[1:])]
    for ds in DATASETS:
        fr = [r for r in frame_rows if r["dataset"] == ds]
        axes[0].plot(centres, [r["recall"] for r in fr], "o-", color=COL[ds], ms=4, label=SHORT[ds])
    axes[0].set_xlabel("mean truth-box side of the frame [px]"); axes[0].set_ylabel("recall")
    axes[0].set_title("Frame level", fontsize=10, loc="left"); axes[0].grid(True, alpha=0.3); axes[0].legend(fontsize=8)
    for motion, ls in (("moving", "-"), ("static", "--")):
        for ds in DATASETS:
            rr = [r for r in obj_rows if r["dataset"] == ds and r["motion"] == motion]
            axes[1].plot(centres, [r["recall"] for r in rr], ls, marker="o", color=COL[ds], ms=3, alpha=0.8,
                         label=f"{SHORT[ds]} {motion}")
    axes[1].set_xlabel("object's median matched-box side [px]")
    axes[1].set_title("Object level (solid moving, dashed static)", fontsize=10, loc="left"); axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=6, ncol=2)
    fig.suptitle("Recall against truth-box size", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(path, dpi=160); plt.close(fig)


def fig_gaps(objs_by_ds, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    gap_frames = int(GAP_MAX_S / FRAME_DT)
    for ds in DATASETS:
        g = np.concatenate([o["gaps"] for o in objs_by_ds[ds]])
        g = g[(g > 0) & (g <= gap_frames)]
        vals, counts = np.unique(g, return_counts=True)
        ax.plot(vals, np.cumsum(counts * vals) / (counts * vals).sum(), color=COL[ds], lw=1.4, label=SHORT[ds])
    ax.set_xscale("log"); ax.set_xlabel("gap length between two matches of one object [frames]")
    ax.set_ylabel("cumulative share of missed frames"); ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=8)
    ax.set_title("Are the misses flicker or long absences?", fontsize=11, loc="left")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


# ----------------------------------------------------------------------- main

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    anat, sweep, fr_rows, ob_rows, miss_rows, cls_rows, shape_rows = [], [], [], [], [], [], []
    dets_by_ds, objs_by_ds = {}, {}
    for ds in DATASETS:
        frames, dets = load(ds)
        dets_by_ds[ds] = dets
        anat.append(fp_anatomy(ds, frames, dets))
        shape_rows.append(box_shape(ds, frames, dets))
        sweep += threshold_sweep(ds, frames, dets)
        fr_rows += recall_by_size_frame(ds, frames)
        rows, objs = recall_by_size_object(ds, frames, dets)
        ob_rows += rows; objs_by_ds[ds] = objs
        miss_rows.append(miss_structure(ds, frames, objs))
        cls_rows += per_class(ds)

    def dump(name, rows, drop=()):
        keys = [k for k in rows[0] if k not in drop]
        with open(OUT / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore"); w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in keys})
    dump("fp_anatomy.csv", anat, drop=("centres",))
    dump("threshold_sweep.csv", sweep)
    dump("recall_by_size.csv", fr_rows + [dict(r, n_boxes=r["n_frames"], n_moving="", moving_recall="", static_recall="") for r in ob_rows])
    dump("miss_structure.csv", miss_rows)
    dump("per_class.csv", cls_rows)
    dump("box_shape.csv", shape_rows)
    tex_shape(shape_rows, OUT / "box_shape.tex")
    with open(OUT / "objects.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["dataset", "name", "cls", "n_match", "n_miss", "recall", "size_px", "moving_share"])
        for ds in DATASETS:
            for o in objs_by_ds[ds]:
                w.writerow([ds, o["name"], o["cls"], o["n_match"], o["n_miss"], f"{o['recall']:.3f}", f"{o['size']:.1f}", f"{o['moving_share']:.2f}"])

    tex_fp(anat, OUT / "fp_anatomy.tex")
    tex_threshold(sweep, OUT / "threshold_sweep.tex")
    tex_size(fr_rows, ob_rows, OUT / "recall_by_size.tex")
    tex_miss(miss_rows, OUT / "miss_structure.tex")
    tex_class(cls_rows, OUT / "per_class.tex")
    fig_fp(anat, dets_by_ds, OUT / "fig_fp_anatomy_light.png")
    fig_threshold(sweep, OUT / "fig_threshold_light.png")
    fig_size(fr_rows, ob_rows, OUT / "fig_size_light.png")
    fig_gaps(objs_by_ds, OUT / "fig_gaps_light.png")

    # digest
    for a in anat:
        u = a["unmatched"]
        print(f"{a['dataset']}: unmatched {u}, scored FP {a['scored_fp']}, ignored {a['ignored_scenery']}, dup {a['duplicate']} "
              f"({100*a['duplicate']/u:.0f}%), near {a['near_miss']} ({100*a['near_miss']/u:.0f}%), slight {a['slight_overlap']} "
              f"({100*a['slight_overlap']/u:.0f}%), none {a['no_overlap']} ({100*a['no_overlap']/u:.0f}%), border {100*a['border']/u:.0f}%")
    for r in sweep:
        if r["threshold"] in (0.35, 0.55, 0.75):
            print(f"  {r['dataset']} th {r['threshold']:.2f}: P {r['precision']:.3f} R {r['recall']:.3f} Rmov {r['moving_recall']:.3f}")
    for r in miss_rows:
        print(f"  {r['dataset']}: objects {r['n_objects']} (low {r['objects_low_recall']}), gaps {r['gaps']}, median {r['gap_median']}, p90 {r['gap_p90']}, "
              f"single {100*r['single_frame_share']:.0f}%, long-gap share {100*r['long_gap_share_of_misses']:.0f}%, short-gap misses {r['fn_in_short_gaps']}/{r['fn_total']}")
    for lo, hi in zip(SIZE_EDGES[:-1], SIZE_EDGES[1:]):
        fr = [r for r in fr_rows if r["lo"] == lo]; nb = sum(r["n_boxes"] for r in fr)
        tp = sum(r["recall"] * r["n_boxes"] for r in fr if np.isfinite(r["recall"]))
        print(f"  size {lo}-{hi}: frame-level boxes {nb} recall {tp/nb if nb else float('nan'):.3f}")


if __name__ == "__main__":
    main()
