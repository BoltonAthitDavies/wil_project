#!/usr/bin/env python3
"""Generalization gap for the warehouse detector: training distribution vs ours.

    python3 script/yolo_generalization.py --print-commands
    python3 script/yolo_generalization.py

RESEARCH QUESTION
    The detector reaches precision 0.958 and recall 0.882 on the validation
    split of the data it was trained on, and precision 0.684 / recall 0.593 on
    our Gazebo recordings. How much of that difference is a real failure to
    generalize, and how much is an artefact of measuring the two sides
    differently?

HYPOTHESIS (H5), FALSIFIABLE
    H5: the drop is a measurement artefact -- once the two sides use the same
    weights, the same operating point and the same matching rules, the residual
    gap is small.

    H5 is falsified if a protocol-matched comparison still shows a substantial
    drop. The decision threshold is stated before the numbers: a residual
    F1 drop of more than 0.10 counts as a real generalization gap.

WHY THE NAIVE COMPARISON IS INVALID
    Four things differ between the two numbers, any one of which could produce
    a gap of this size on its own.

    1. DIFFERENT WEIGHTS. weight/best.pt (2026-09-02) produced every published
       detector number. weight/runs/detect/train/ (2026-09-20) is a separate
       training run of the same recipe -- same yolo26m.pt, same
       warehouse-1/data.yaml, same 20 epochs, same {bin, box, bucket} -- but the
       checkpoints differ (md5 13f70de5... vs 5e3b7f6e...). The training curves
       belong to the second model; the field results belong to the first.

    2. DIFFERENT OPERATING POINT. Ultralytics reports precision and recall at
       the confidence that maximises F1, chosen per run. yolo_eval.py runs at a
       FIXED conf=0.35. Two different points on the same PR curve are not a
       comparison.

    3. DIFFERENT MATCHING RULES. yolo_eval.py declares a ground-truth box
       "ignore" when it is smaller than 12 px, or covered more than 70% by a
       nearer object, and drops predictions that sit on unlabelled scenery.
       Over the five recordings that removes 34,476 of 96,758 objects -- 36% of
       all ground truth -- from scoring. The training split has no such concept:
       every labelled box counts. Our recall is therefore computed over the
       easier two thirds of the objects.

    4. DIFFERENT LABEL SOURCE. The training labels are human boxes drawn on
       images. Ours are projected Gazebo mesh extents. Even a perfect detector
       scores below 1.0 against the other convention. This one cannot be
       controlled without relabelling and is reported as a residual.

    (1)-(3) are controllable with the runs this script prints. (4) is not.

WHAT THIS SCRIPT COMPUTES
    * source-domain metrics, read from the training run's results.csv;
    * target-domain metrics, pooled and per recording, from output/yolo_eval/;
    * a confidence sweep over our detections, giving our own max-F1 operating
      point so the two sides can be compared at like points rather than at one
      fixed threshold;
    * the protocol bound: recall recomputed with every ignored object counted as
      a miss, which is what the training-split convention would do;
    * the gap at each stage, so it is visible how much each control removes.

THE CONFIDENCE SWEEP IS TRUNCATED UNLESS THE EVALUATION IS RE-RUN
    yolo_detections.csv only contains boxes that already passed conf=0.35, so a
    sweep over it cannot go below 0.35 and cannot see the low-confidence tail
    where recall is won. If the detections start at 0.35 this script says so and
    marks every swept figure as a partial curve. --print-commands emits the
    re-run at conf=0.001 that makes the sweep complete; average precision is
    only reported when that re-run is present.

LIMITATIONS
    1. The training dataset itself is not in this repository -- data.yaml points
       at /content/datasets/warehouse-1, a Colab path. Without it the source
       side cannot be re-scored at our operating point or checked for
       train/val leakage, and the source numbers have to be taken as the
       training run reported them. Retrieving that export is the one thing that
       would make this experiment complete.
    2. n=1 training run. No seed variation, so none of the source-side figures
       carry uncertainty and a difference of a few points means nothing.
    3. The two domains differ in content as well as rendering. This measures
       that the detector transfers poorly; it does not say which factor --
       renderer, object meshes, viewpoint, scale, occlusion -- is responsible.
"""

import argparse
import csv
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_RUN = os.path.join(ROOT, 'weight/runs/detect/train')
EVAL_DIR = os.path.join(ROOT, 'output/yolo_eval')
OUT = os.path.join(ROOT, 'output/compare/simulation/exp-yolo-generalization')

# The conf yolo_eval.py used for the published run. A sweep cannot go below it.
PUBLISHED_CONF = 0.35
# Stated before the numbers: what counts as a real gap rather than a measurement
# artefact. See H5 above.
H5_F1_THRESHOLD = 0.10


def _f(row, key):
    for k in row:
        if k.strip() == key:
            try:
                return float(row[k])
            except (TypeError, ValueError):
                return None
    return None


def train_metrics(d):
    rc = os.path.join(d, 'results.csv')
    if not os.path.exists(rc):
        raise SystemExit('no results.csv under %s' % d)
    rows = list(csv.DictReader(open(rc)))
    last = rows[-1]
    args = {}
    ay = os.path.join(d, 'args.yaml')
    if os.path.exists(ay):
        for line in open(ay):
            if ':' in line:
                k, v = line.split(':', 1)
                args[k.strip()] = v.strip()
    return dict(
        epochs=len(rows),
        precision=_f(last, 'metrics/precision(B)'),
        recall=_f(last, 'metrics/recall(B)'),
        map50=_f(last, 'metrics/mAP50(B)'),
        map5095=_f(last, 'metrics/mAP50-95(B)'),
        base_model=args.get('model'), data=args.get('data'),
        imgsz=args.get('imgsz'), val_iou=args.get('iou'))


def target_pooled(d):
    """Pooled counts from summary_metrics.csv, which yolo_eval.py wrote."""
    p = os.path.join(d, 'summary_metrics.csv')
    if not os.path.exists(p):
        raise SystemExit('no summary_metrics.csv under %s -- run yolo_eval.py first' % p)
    rows = {r['dataset']: r for r in csv.DictReader(open(p))}
    if 'Pooled' not in rows:
        raise SystemExit('summary_metrics.csv has no Pooled row')
    r = rows['Pooled']
    i = lambda k: int(float(r[k]))
    return dict(frames=i('frames'), n_gt=i('n_gt'), n_ignored=i('n_ignored'),
                n_pred=i('n_pred'), tp=i('tp'), fp=i('fp'), fn=i('fn'),
                tp_moving=i('tp_moving'), fn_moving=i('fn_moving'),
                per_dataset={k: v for k, v in rows.items() if k != 'Pooled'})


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def sweep(eval_dir):
    """P/R/F1 against confidence, from every yolo_detections.csv.

    Returns (points, floor, total_gt). `floor` is the lowest score present: when
    it is at or above PUBLISHED_CONF the curve is truncated and the caller must
    say so rather than quietly reporting a partial AP as if it were AP.
    """
    dets, floor = [], 1.0
    for f in sorted(glob.glob(os.path.join(eval_dir, '*', 'yolo_detections.csv'))):
        for row in csv.DictReader(open(f)):
            try:
                s = float(row['score'])
            except (KeyError, ValueError):
                continue
            dets.append((s, row['matched'] == '1'))
            floor = min(floor, s)
    if not dets:
        return [], 1.0, 0
    dets.sort(key=lambda t: -t[0])
    pooled = target_pooled(eval_dir)
    total_gt = pooled['tp'] + pooled['fn']

    pts, tp = [], 0
    for i, (s, m) in enumerate(dets, 1):
        tp += 1 if m else 0
        # Only sample the curve; every detection would be 58k rows of output.
        if i % 200 == 0 or i == len(dets):
            p = tp / i
            r = tp / total_gt if total_gt else 0.0
            pts.append(dict(conf=s, n_pred=i, tp=tp,
                            precision=p, recall=r,
                            f1=(2 * p * r / (p + r) if p + r else 0.0)))
    return pts, floor, total_gt


def print_commands():
    print("""# Section 7 of the skill: the user runs these.

# ---------------------------------------------------------------- CONTROL 1
# Same weights on both sides. The training curves belong to the 2026-09-20 run,
# so that run's checkpoint is the one that must be scored on our recordings.
# NOTE the separate --out tree: this must not overwrite output/yolo_eval/, which
# holds the published numbers for weight/best.pt.
for B in dataset_dynamic_nofloortexture_01_000 ; do
  python3 script/yolo_eval.py $HOME/wil_project/dataset/$B/$B_0.db3 \\
    --weights $HOME/wil_project/weight/runs/detect/train/weights/best.pt \\
    --out $HOME/wil_project/output/yolo_eval_train20260920/$B \\
    --conf 0.001 --overlay 20
done
# --conf 0.001 is deliberate: it is what makes the PR curve and average
# precision computable. Scoring at 0.35 again would reproduce the same truncated
# curve this script has to refuse to integrate.

# ---------------------------------------------------------------- CONTROL 3
# Protocol-matched: disable the ignore machinery so every ground-truth object
# counts, as it does in the training split.
#   (requires the --no-ignore flag added to yolo_eval.py)
python3 script/yolo_eval.py <same bag> \\
  --weights $HOME/wil_project/weight/runs/detect/train/weights/best.pt \\
  --out $HOME/wil_project/output/yolo_eval_noignore/<dataset> \\
  --conf 0.001 --no-ignore

# ---------------------------------------------------------------- SOURCE SIDE
# Needs the training dataset, which is NOT in this repository -- data.yaml
# points at /content/datasets/warehouse-1 on Colab. Export it from Roboflow to
# $HOME/wil_project/dataset/warehouse-1/ and then:
yolo val model=$HOME/wil_project/weight/runs/detect/train/weights/best.pt \\
  data=$HOME/wil_project/dataset/warehouse-1/data.yaml \\
  split=val conf=0.001 iou=0.5 save_json=True \\
  project=$HOME/wil_project/output/yolo_eval_source name=train20260920
# Without this the source side can only be quoted as the training run reported
# it, and cannot be checked for train/val leakage.

# ---------------------------------------------------------------- THEN
python3 script/yolo_generalization.py""")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--train-run', default=TRAIN_RUN)
    ap.add_argument('--eval-dir', default=EVAL_DIR)
    ap.add_argument('--print-commands', action='store_true')
    a = ap.parse_args()
    if a.print_commands:
        return print_commands()

    src = train_metrics(a.train_run)
    tgt = target_pooled(a.eval_dir)
    os.makedirs(OUT, exist_ok=True)

    p, r, f1 = prf(tgt['tp'], tgt['fp'], tgt['fn'])
    # The training split counts every labelled object. Applying that convention
    # here means the ignored objects become misses. Precision cannot be bounded
    # the same way: predictions dropped for sitting on unlabelled scenery would
    # become either TP or FP and the artifacts do not record which, so this is
    # a recall bound only and is labelled as such.
    r_matched = tgt['tp'] / (tgt['tp'] + tgt['fn'] + tgt['n_ignored'])
    f1_matched = (2 * p * r_matched / (p + r_matched)) if p + r_matched else 0.0
    ignored_preds = tgt['n_pred'] - tgt['tp'] - tgt['fp']

    pts, floor, total_gt = sweep(a.eval_dir)
    truncated = floor >= PUBLISHED_CONF - 1e-9
    best = max(pts, key=lambda d: d['f1']) if pts else None

    print('SOURCE DOMAIN -- %s, %s, %d epochs' % (
        os.path.relpath(a.train_run, ROOT), src['base_model'], src['epochs']))
    print('   data            %s' % src['data'])
    print('   precision %.3f   recall %.3f   mAP50 %.3f   mAP50-95 %.3f'
          % (src['precision'], src['recall'], src['map50'], src['map5095']))
    print('   operating point: the confidence that maximises F1, chosen by '
          'Ultralytics per run')

    print('\nTARGET DOMAIN -- %s, %d frames, %d recordings'
          % (os.path.relpath(a.eval_dir, ROOT), tgt['frames'],
             len(tgt['per_dataset'])))
    print('   scored GT %d   ignored GT %d   predictions %d (%d ignored)'
          % (tgt['n_gt'], tgt['n_ignored'], tgt['n_pred'], ignored_preds))
    print('   precision %.3f   recall %.3f   F1 %.3f      at fixed conf %.2f'
          % (p, r, f1, PUBLISHED_CONF))
    print('   recall %.3f   F1 %.3f   with every ignored object counted as a '
          'miss' % (r_matched, f1_matched))

    if best:
        print('\nCONFIDENCE SWEEP over %d detections, floor %.3f%s'
              % (pts[-1]['n_pred'], floor, '  [TRUNCATED]' if truncated else ''))
        print('   our max-F1 point: conf %.3f   P %.3f   R %.3f   F1 %.3f'
              % (best['conf'], best['precision'], best['recall'], best['f1']))
        if truncated:
            print('   The curve starts at the published conf, so the '
                  'low-confidence tail is missing.\n'
                  '   Average precision is NOT reported: integrating this curve '
                  'would understate it\n'
                  '   by an unknown amount. Re-run per --print-commands to '
                  'complete it.')

    src_f1 = 2 * src['precision'] * src['recall'] / (src['precision'] + src['recall'])
    print('\nGAP, target minus source (negative means our data is worse)')
    print('   %-38s precision %+.3f   recall %+.3f   F1 %+.3f'
          % ('as published (nothing controlled)',
             p - src['precision'], r - src['recall'], f1 - src_f1))
    print('   %-38s precision %+.3f   recall %+.3f   F1 %+.3f'
          % ('with the ignore convention matched',
             p - src['precision'], r_matched - src['recall'], f1_matched - src_f1))
    print('\n   H5 (the drop is a measurement artefact): %s'
          % ('FALSIFIED -- residual F1 drop %.3f exceeds the %.2f threshold'
             % (abs(f1_matched - src_f1), H5_F1_THRESHOLD)
             if abs(f1_matched - src_f1) > H5_F1_THRESHOLD else 'not falsified'))
    print('   Controls still outstanding: same weights on both sides, matched '
          'operating point,\n   and the source split rescored locally. Until '
          'those are run this verdict is indicative.')

    with open(os.path.join(OUT, 'generalization_metrics.csv'), 'w') as fh:
        fh.write('side,condition,precision,recall,f1,n_gt,n_pred,note\n')
        fh.write('source,train-val as reported,%.6f,%.6f,%.6f,,,%s\n'
                 % (src['precision'], src['recall'], src_f1, src['data']))
        fh.write('target,fixed conf %.2f,%.6f,%.6f,%.6f,%d,%d,published run\n'
                 % (PUBLISHED_CONF, p, r, f1, tgt['n_gt'], tgt['n_pred']))
        fh.write('target,ignore convention matched,%.6f,%.6f,%.6f,%d,%d,'
                 'recall bound only\n'
                 % (p, r_matched, f1_matched, tgt['n_gt'] + tgt['n_ignored'],
                    tgt['n_pred']))
    print('\nwrote %s' % os.path.relpath(
        os.path.join(OUT, 'generalization_metrics.csv'), ROOT))

    if pts:
        with open(os.path.join(OUT, 'pr_sweep.csv'), 'w') as fh:
            fh.write('conf,n_pred,tp,precision,recall,f1,truncated\n')
            for d in pts:
                fh.write('%.4f,%d,%d,%.6f,%.6f,%.6f,%d\n'
                         % (d['conf'], d['n_pred'], d['tp'], d['precision'],
                            d['recall'], d['f1'], int(truncated)))
        print('wrote %s' % os.path.relpath(os.path.join(OUT, 'pr_sweep.csv'), ROOT))


if __name__ == '__main__':
    sys.exit(main())
