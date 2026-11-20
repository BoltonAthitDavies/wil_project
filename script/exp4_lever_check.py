#!/usr/bin/env python3
"""Along/cross-track decomposition of the Experiment 4 pose error.

    python3 script/exp4_lever_check.py

WHY THIS EXISTS
    The slot's Frames paragraph claims that applying the 0.338 m lever arm to
    ground truth in the COMPARISON put a constant along-track bias into every
    run, and that removing it leaves an along-track mean that is small and
    changes sign. That claim was originally a one-off analysis recorded only in
    a docstring, so it could not be re-derived when the reported run set
    changed from seven runs to the six of the ungated arm. This script is that
    analysis, kept runnable.

    It reports both states for every run of the reported arm: the error
    projected onto the ground-truth direction of travel (along) and onto its
    normal (cross).

WHAT IT CONFIRMS
    With the lever arm applied the along-track mean is large, negative and the
    same sign in every run while the cross-track mean sits near zero -- a
    constant displacement in the direction of travel, which is what a frame
    error looks like. Without it the along-track mean is small and changes
    sign, which is what an unbiased estimate looks like.

SELECTION
    The ungated arm, via exp4_score.is_gated, for the reason given there: the
    rep index is a configuration index, not a repeat index.
"""
import csv
import os
import sys

import numpy as np
import yaml

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKSPACE, 'script'))

import exp4_goals as eg                                        # noqa: E402
import exp4_score as es                                        # noqa: E402

GATE_M = 2.0            # a diverged sample is a different failure; see decompose()


def stats(spec, lever, arm='gated'):
    out = []
    for world in spec['worlds']:
        lay = spec['worlds'][world]['layout']
        prior = spec['priors'][lay]['database']
        al = eg._align(prior)
        for path in spec['priors'][lay]['paths']:
            for rep in es.reps_present(prior, world, path):
                run = f'{world}_{path}_{rep}'
                rund = os.path.join(WORKSPACE, 'map', prior, 'exp4', run)
                if arm != 'all' and es.is_gated(rund) != (arm == 'gated'):
                    continue
                rob = os.path.join(rund, 'robot_gt.csv')
                est = os.path.join(WORKSPACE, 'map', prior,
                                   f'localize_exp4_{run}.csv')
                if not (os.path.exists(rob) and os.path.exists(est)):
                    continue
                t, gt = es.load_robot_gt(rob, lever=lever)
                gm = es.world_to_map(al, gt)
                rec = [r for r in csv.DictReader(open(est))
                       if r['kind'] == 'tf' and r.get('x') and r.get('y')]
                et = np.array([float(r['stamp']) for r in rec])
                ex = np.array([float(r['x']) for r in rec])
                ey = np.array([float(r['y']) for r in rec])
                m = (et >= t[0]) & (et <= t[-1])
                et, ex, ey = et[m], ex[m], ey[m]
                gx = np.interp(et, t, gm[:, 0])
                gy = np.interp(et, t, gm[:, 1])
                dx, dy = ex - gx, ey - gy
                vx, vy = np.gradient(gx), np.gradient(gy)
                n = np.hypot(vx, vy)
                k = n > 1e-6
                along = (dx[k] * vx[k] + dy[k] * vy[k]) / n[k]
                cross = (dx[k] * vy[k] - dy[k] * vx[k]) / n[k]
                keep = np.hypot(dx[k], dy[k]) < GATE_M
                out.append({'run': run,
                            'along': float(along[keep].mean()),
                            'cross': float(cross[keep].mean()),
                            'median': float(np.median(np.hypot(dx, dy)))})
    return out


def main():
    spec = yaml.safe_load(open(os.path.join(WORKSPACE, 'script',
                                            'exp4_protocol.yaml')))
    for lever in (True, False):
        rows = stats(spec, lever)
        if not rows:
            sys.exit('no runs found for the reported arm.')
        print(f'--- lever arm applied to ground truth: {lever} '
              f'({len(rows)} runs) ---')
        print(f'  {"run":30s} {"along":>8s} {"cross":>8s} {"median":>8s}')
        for r in rows:
            print(f'  {r["run"]:30s} {r["along"]:+8.3f} {r["cross"]:+8.3f} '
                  f'{r["median"]:8.3f}')
        a = [r['along'] for r in rows]
        c = [r['cross'] for r in rows]
        d = [r['median'] for r in rows]
        print(f'  along {min(a):+.2f} to {max(a):+.2f} | '
              f'cross {min(c):+.2f} to {max(c):+.2f} | '
              f'median {min(d):.2f} to {max(d):.2f}\n')


if __name__ == '__main__':
    main()
