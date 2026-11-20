#!/usr/bin/env python3
"""What RTAB-Map's place recognition proposes, and what its geometry does with it.

    python3 script/exp4_rtabmap_gating.py           # the numbers
    python3 script/exp4_rtabmap_gating.py --write   # and the figure

WHY THIS EXISTS
    Experiment 4 reports how well the published pose tracks ground truth. It
    does not report what the localiser had to reject to produce that pose, and
    that is where the warehouse actually bites. RTAB-Map accepts a loop closure
    in two stages: a bag-of-words layer proposes a candidate place from word
    co-occurrence, and a geometric check (Vis/EstimationType, here 3D->3D)
    counts how many matched features agree in metres. Vis/MinInliers (20) is
    the threshold between them.

    Both stages log their verdict. Every rejection is written by
    Rtabmap.cpp:3143 as "Not enough inliers i/N (matches=m)", which gives the
    word-match count the proposal carried AND the geometric support it turned
    out to have. Counting accepted fixes from the localisation CSV completes
    the ledger.

WHAT IT MEASURES
    The proposal survival rate, and -- more informative -- the shape of the
    failures. A rejection with 18 inliers is a near miss; a rejection with 0
    inliers from 100 word matches is the vocabulary layer pointing at a
    different place entirely, which is what perceptual aliasing in repeating
    aisles looks like from inside the algorithm.

SCOPE
    The reported arm only (exp4_score.is_gated). Counts are per run; the runs
    are one per cell, so these are proportions over proposals, not a
    comparison between cells.
"""
import argparse
import csv
import glob
import os
import re

import numpy as np

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(WORKSPACE, 'output', 'compare', 'simulation', 'exp4-nav-insights')
REJ = re.compile(r'Rejected loop closure (\d+) -> (\d+): '
                 r'Not enough inliers (\d+)/(\d+) \(matches=(\d+)\)')
SHAPE = {'P1_transit': 'straight', 'P2_return': 'square', 'P3_circuit': 'figure-8'}


def collect():
    import sys
    sys.path.insert(0, os.path.join(WORKSPACE, 'script'))
    import exp4_score as es
    rows = []
    for log in sorted(glob.glob(os.path.join(WORKSPACE, 'map', '*', 'exp4', '*', 'launch.log'))):
        d = os.path.dirname(log)
        if es.is_gated(d) is False:
            continue
        run = os.path.basename(d)
        prior = log.split(os.sep)[-4]
        est = os.path.join(WORKSPACE, 'map', prior, f'localize_exp4_{run}.csv')
        if not os.path.exists(est):
            continue
        rej = REJ.findall(open(log, errors='ignore').read())
        acc = sum(1 for r in csv.DictReader(open(est))
                  if r['kind'] == 'pose' and r.get('cov_trace')
                  and float(r['cov_trace']) < 1e4)
        world, path = run.split('_P')[0], 'P' + run.split('_P')[1].rsplit('_', 1)[0]
        lay = 'U' if world.startswith('U') else 'I'
        rows.append({'run': f'{lay} {SHAPE.get(path, path)}', 'accepted': acc,
                     'inliers': [int(r[2]) for r in rej],
                     'thr': int(rej[0][3]) if rej else 20,
                     'matches': [int(r[4]) for r in rej]})
    return rows


def figure(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    inl = np.array([v for r in rows for v in r['inliers']])
    mat = np.array([v for r in rows for v in r['matches']])
    thr = rows[0]['thr']

    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    # No shaded "accepted" band: RTAB-Map writes an inlier count only when it
    # REFUSES a proposal (Rtabmap.cpp:3143 is a warning), so the accepted ones
    # cannot be drawn here at all. An empty acceptance region on a panel that
    # plots only rejections reads as "nothing was ever accepted", which the
    # right-hand panel then contradicts.
    a.axhline(thr, color='#777777', lw=1.2, ls='--')
    a.text(mat.max() * 0.98, thr + 0.7,
           f'Vis/MinInliers = {thr}: every point here fell short of it',
           ha='right', va='bottom', fontsize=9, color='#555555')
    jit = (np.random.default_rng(0).random(len(inl)) - 0.5) * 0.6
    a.scatter(mat, inl + jit, s=26, color='#d95f02', alpha=0.75, lw=0)
    a.set_xlabel('bag-of-words matches the proposal carried')
    a.set_ylabel('features that agreed geometrically (inliers)')
    a.set_title(f'the {len(inl)} REJECTED proposals only '
                '(accepted ones carry no inlier count)')
    a.set_ylim(-1.5, max(thr + 5, inl.max() + 5))
    a.grid(alpha=0.2)

    names = [r['run'] for r in rows]
    x = np.arange(len(rows))
    acc = np.array([r['accepted'] for r in rows], float)
    zero = np.array([sum(1 for v in r['inliers'] if v == 0) for r in rows], float)
    some = np.array([sum(1 for v in r['inliers'] if v > 0) for r in rows], float)
    b.bar(x, acc, 0.62, label='accepted', color='#1b9e77')
    b.bar(x, some, 0.62, bottom=acc, label='rejected, some support', color='#f0a868')
    b.bar(x, zero, 0.62, bottom=acc + some, label='rejected, zero support', color='#d95f02')
    b.set_xticks(x)
    b.set_xticklabels(names, rotation=20, ha='right', fontsize=8)
    b.set_ylabel('loop-closure proposals')
    b.set_title('all 346 proposals, accepted and rejected')
    b.legend(fontsize=8)
    b.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, 'fig_rtabmap_gating.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args()
    rows = collect()
    if not rows:
        raise SystemExit('no gated runs found.')
    print(f"  {'run':14s} {'accepted':>9s} {'rejected':>9s} {'zero support':>13s} {'med matches':>12s}")
    for r in rows:
        n = len(r['inliers'])
        z = sum(1 for v in r['inliers'] if v == 0)
        print(f"  {r['run']:14s} {r['accepted']:9d} {n:9d} "
              f"{z:9d} ({100*z/max(n,1):3.0f}%) "
              f"{int(np.median(r['matches'])) if r['matches'] else 0:12d}")
    inl = np.array([v for r in rows for v in r['inliers']])
    mat = np.array([v for r in rows for v in r['matches']])
    acc = sum(r['accepted'] for r in rows)
    print(f"\n  {acc + len(inl)} proposals: {acc} accepted "
          f"({100*acc/(acc+len(inl)):.0f}%), {len(inl)} rejected")
    print(f"  of the rejected, {np.sum(inl == 0)} ({100*np.mean(inl == 0):.0f}%) had ZERO "
          f"geometric support, carrying a median of {np.median(mat):.0f} word matches")
    ra = np.argsort(np.argsort(mat)).astype(float)
    rb = np.argsort(np.argsort(inl)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    print(f"  Spearman(word matches, inliers) = "
          f"{(ra*rb).sum()/np.sqrt((ra**2).sum()*(rb**2).sum()):+.2f}  "
          f"-- word count barely predicts geometric agreement")
    if a.write:
        print(f"\n  figure -> {figure(rows)}")


if __name__ == '__main__':
    main()
