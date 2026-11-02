#!/usr/bin/env python3
"""Score ORB-SLAM3's motion gate against Gazebo ground truth, pooled over runs.

    python3 script/analyse_motion_gate.py \
        --bag /media/ambushee/32E4AAB1E4AA772F/dataset_dynamic_nofloortexture_00_000 \
        --runs output/output_orb_yolo/simulation/dataset_dynamic_nofloortexture_00_000/motion_*/ \
        --out output/compare/simulation/exp3-motion-gate

WHAT THIS ADDS OVER script/motion_votes_eval.py

    That script scores ONE run at the thresholds that run used, and prints a
    coarse sweep. It answers "how did this run do". This one answers the prior
    question -- "is there a usable signal here at all" -- by

      * pooling every labelled box-frame across runs, so the static class is
        large enough to estimate a false-alarm rate from (a single run at the
        shipped `motion_min_box_age` leaves ~19 static rows, and a separation
        measured on 19 rows did not survive 1044 -- see the note below);
      * sweeping BOTH gates, `motion_min_box_age` and `motion_inlier_ratio`,
        since loosening the first is the only way to raise coverage;
      * reporting AUC and Youden's J instead of F1. F1 is actively misleading
        here: roughly three quarters of labelled box-frames are moving, so a
        rule that calls everything moving scores F1 ~0.86 while being useless.
        AUC 0.5 and J 0 are chance.

    Direction convention: the gate declares MOVING when the box's inlier ratio
    is BELOW a threshold, so AUC is computed as P(moving ranks below static).
    An AUC under 0.5 therefore means the signal is inverted, not merely absent.

VALIDITY

    Reports `filter_status` and `detection_match_rate` for every run it pools.
    A run whose mask was intermittent produces more `unknown` verdicts, which
    changes coverage but not the separability of the rows that were scored.
    Both are written to the summary so a reader can discount accordingly.

OUTPUTS, under --out

    motion_gate_pooled.csv     every labelled box-frame, all runs, with truth
    motion_gate_summary.csv    one row per (box_age, ratio) cell of the sweep
    motion_gate_headline.csv   confusion, census quantiles, AUC, best J
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import subprocess
import sys
import tempfile

WORKSPACE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def label_run(bag: str, run: str, min_points: int, min_box_age: int) -> list[dict]:
    """Run motion_votes_eval.py --out into a temp dir and read back its labels."""
    votes = os.path.join(run, 'motion_votes.csv')
    if not os.path.exists(votes):
        return []
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [sys.executable, os.path.join(WORKSPACE, 'script', 'motion_votes_eval.py'),
               bag, votes, '--min-points', str(min_points),
               '--min-box-age', str(min_box_age), '--out', tmp]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        rows = list(csv.DictReader(open(os.path.join(tmp, 'motion_eval.csv'))))
    for r in rows:
        r['run'] = os.path.basename(os.path.normpath(run))
    return rows


def run_validity(run: str) -> dict:
    out = {'run': os.path.basename(os.path.normpath(run))}
    for name in ('run_summary.csv', 'filter_summary.csv'):
        p = os.path.join(run, name)
        if not os.path.exists(p):
            continue
        for r in csv.reader(open(p)):
            if len(r) == 2 and r[0] in ('filter_status', 'detection_match_rate',
                                        'queue_dropped_frames', 'stereo_pairs_received',
                                        'boxes_voted_moving', 'boxes_voted_static'):
                out[r[0]] = r[1]
    return out


def ratio_of(r: dict) -> float | None:
    try:
        return float(r['rel_ratio'])
    except (KeyError, ValueError):
        return None


def auc_moving_below(moving: list[float], static: list[float]) -> float:
    """P(a moving box scores BELOW a static one), ties counted as half."""
    if not moving or not static:
        return float('nan')
    marks = sorted([(v, 0) for v in moving] + [(v, 1) for v in static])
    n_static = len(static)
    seen_static = 0          # statics strictly BELOW the current value
    wins = 0.0
    i = 0
    while i < len(marks):
        j = i
        while j < len(marks) and marks[j][0] == marks[i][0]:
            j += 1
        tie_static = sum(1 for k in range(i, j) if marks[k][1] == 1)
        above = n_static - seen_static - tie_static   # statics strictly ABOVE
        for k in range(i, j):
            if marks[k][1] == 0:
                wins += above + 0.5 * tie_static
        seen_static += tie_static
        i = j
    return wins / (len(moving) * n_static)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', required=True)
    ap.add_argument('--runs', nargs='+', required=True,
                    help='run directories (shell glob is fine)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--min-points', type=int, default=15,
                    help='feature gate held fixed while box_age is swept')
    a = ap.parse_args()

    runs = [d for pat in a.runs for d in sorted(glob.glob(pat))]
    runs = [d for d in runs if os.path.exists(os.path.join(d, 'motion_votes.csv'))]
    if not runs:
        raise SystemExit('no run directories with motion_votes.csv')
    os.makedirs(a.out, exist_ok=True)

    print(f'bag  : {a.bag}')
    validity = [run_validity(d) for d in runs]
    for v in validity:
        print(f"  {v['run']:34s} status={v.get('filter_status','?'):12s} "
              f"match={v.get('detection_match_rate','?'):6s} "
              f"dropped={v.get('queue_dropped_frames','?')}")

    # Label with box_age 0 so NOTHING is pre-filtered; the sweep applies the
    # gates itself. Passing the shipped 30 here would discard the rows the
    # sweep exists to examine.
    rows: list[dict] = []
    for d in runs:
        got = label_run(a.bag, d, a.min_points, 0)
        print(f"  {os.path.basename(os.path.normpath(d)):34s} labelled {len(got)} box-frames")
        rows += got
    scored = [r for r in rows if r['truth'] in ('moving', 'static')]
    n_mov = sum(1 for r in scored if r['truth'] == 'moving')
    print(f'\npooled: {len(rows)} box-frames, {len(scored)} matched to a known prop '
          f'({n_mov} moving, {len(scored)-n_mov} static)')

    with open(os.path.join(a.out, 'motion_gate_pooled.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- confusion as the runs actually decided it -------------------------
    conf = {}
    for r in scored:
        v = {'0': 'unknown', '1': 'static', '2': 'moving'}[r['verdict']]
        conf[(r['truth'], v)] = conf.get((r['truth'], v), 0) + 1
    print('\n== what the mask was built from (pooled, shipped thresholds) ==')
    print(f"{'truth / verdict':16s}{'unknown':>9s}{'static':>9s}{'moving':>9s}{'masked':>9s}")
    headline = {'bag': a.bag, 'runs': ';'.join(v['run'] for v in validity),
                'pooled_box_frames': len(rows), 'labelled': len(scored)}
    for truth in ('moving', 'static'):
        tot = sum(conf.get((truth, v), 0) for v in ('unknown', 'static', 'moving'))
        frac = conf.get((truth, 'moving'), 0) / tot if tot else float('nan')
        print(f'{truth:16s}' + ''.join(f"{conf.get((truth, v), 0):9d}"
              for v in ('unknown', 'static', 'moving')) + f'{frac:8.1%}')
        headline[f'{truth}_masked_frac'] = f'{frac:.4f}'
        headline[f'{truth}_total'] = tot

    # ---- sweep -------------------------------------------------------------
    print('\n== sweep: both gates, pooled ==')
    print(f"{'box_age':>8}{'ratio':>7}{'voted':>8}{'cover':>7}"
          f"{'n_mov':>7}{'recall':>8}{'n_stat':>7}{'false':>8}{'J':>8}")
    sweep = []
    for age in (0, 3, 5, 10, 15, 20, 30):
        pool_m, pool_s = [], []
        for r in scored:
            if int(r['n_kp']) < a.min_points or int(r.get('box_age') or 0) < age:
                continue
            v = ratio_of(r)
            if v is None:
                continue
            (pool_m if r['truth'] == 'moving' else pool_s).append(v)
        if len(pool_m) < 20 or len(pool_s) < 10:
            continue
        auc = auc_moving_below(pool_m, pool_s)
        for th in (0.25, 0.5, 0.75, 1.0):
            rec = sum(1 for v in pool_m if v < th) / len(pool_m)
            fa = sum(1 for v in pool_s if v < th) / len(pool_s)
            cover = (len(pool_m) + len(pool_s)) / len(scored)
            print(f'{age:8d}{th:7.2f}{len(pool_m)+len(pool_s):8d}{cover:6.1%}'
                  f'{len(pool_m):7d}{rec:8.3f}{len(pool_s):7d}{fa:8.3f}{rec-fa:+8.3f}')
            sweep.append({'min_box_age': age, 'inlier_ratio': th,
                          'n_moving': len(pool_m), 'n_static': len(pool_s),
                          'coverage': f'{cover:.4f}', 'recall': f'{rec:.4f}',
                          'false_alarm': f'{fa:.4f}', 'youden_j': f'{rec-fa:+.4f}',
                          'auc': f'{auc:.4f}'})
        headline[f'auc_box_age_{age}'] = f'{auc:.4f}'
        print(f'{"":8s}{"":7s}{"":8s}{"":7s}  AUC {auc:.3f}   '
              f'(0.5 = chance; this is P(moving ranks below static), so above '
              f'0.5 = signal in the expected direction)')

    with open(os.path.join(a.out, 'motion_gate_summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0]))
        w.writeheader()
        w.writerows(sweep)

    best = max(sweep, key=lambda s: float(s['youden_j']))
    headline['best_j'] = best['youden_j']
    headline['best_j_box_age'] = best['min_box_age']
    headline['best_j_ratio'] = best['inlier_ratio']
    headline['best_j_recall'] = best['recall']
    headline['best_j_false_alarm'] = best['false_alarm']
    print(f"\nbest separation anywhere in the sweep: J = {best['youden_j']} "
          f"(box_age {best['min_box_age']}, ratio {best['inlier_ratio']}): "
          f"recall {best['recall']}, false alarm {best['false_alarm']}")

    with open(os.path.join(a.out, 'motion_gate_headline.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['key', 'value'])
        for k, v in headline.items():
            w.writerow([k, v])
        for v in validity:
            for k2, v2 in v.items():
                if k2 != 'run':
                    w.writerow([f"{v['run']}.{k2}", v2])
    print(f"\nwrote {a.out}/motion_gate_{{pooled,summary,headline}}.csv")


if __name__ == '__main__':
    main()
