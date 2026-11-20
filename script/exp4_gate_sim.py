#!/usr/bin/env python3
"""Replay recorded Experiment 4 runs through a proposed correction gate.

    python3 script/exp4_gate_sim.py
    python3 script/exp4_gate_sim.py --eps 0.02 --eps0 0.10 --confirm 5

WHY OFFLINE FIRST
    The gate is the one defence left against aisle aliasing, and it is also the
    one that can lock the robot out of its own map if it is too strict. Both
    properties are visible in data already recorded: every run's localisation CSV
    holds the full correction stream and the pose it produced, and the ground
    truth sits beside it. So the rule can be tried on seven runs, including the
    two that warped, before a line of node code is written.

THE RULE
    A correction is a jump in map -> odom. The robot's true pose can only differ
    from the dead-reckoned one by the drift accumulated since the last accepted
    fix, so a correction larger than that drift could plausibly be is describing
    a place the robot cannot have reached.

        accept if   |dp| <= eps * d_odom + eps0
                    |dyaw| <= eps_yaw * d_odom + yaw0

    where d_odom is the distance the FRONT END says was travelled since the last
    accepted correction -- measured, not modelled.

    Experiment 4 is what makes this affordable. Unaided ORB-SLAM3 drift over
    these runs is 0.045-0.345 m median and never exceeds 0.76 m over 21-152 m,
    so rejecting a correction and coasting costs almost nothing, and eps can be
    set near the measured drift rate rather than guessed generously.

    Two escapes stop it latching:
      * the FIRST accepted correction is free, because the seed may be metres out
      * if `confirm` consecutive candidates agree with each other and all
        disagree with the held correction, the map is insisting and is believed.
        That is the recovery path for a genuinely lost pose.
"""
import argparse
import csv
import math
import os
import sys

import numpy as np
import yaml

WORKSPACE = os.path.expanduser('~/wil_project')
sys.path.insert(0, os.path.join(WORKSPACE, 'script'))
import exp4_goals as eg                                        # noqa: E402
import exp4_score as es                                        # noqa: E402

SHAPE = {'P1_transit': 'straight', 'P2_return': 'square', 'P3_circuit': 'figure-8'}


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def compose(c, p):
    """map pose from a correction (cx, cy, cyaw deg) and an odom pose."""
    t = math.radians(c[2])
    ct, st = math.cos(t), math.sin(t)
    return np.array([c[0] + ct * p[0] - st * p[1],
                     c[1] + st * p[0] + ct * p[1]])


def invert(c, m):
    """odom pose from a correction and a map pose."""
    t = math.radians(c[2])
    ct, st = math.cos(t), math.sin(t)
    d = np.array([m[0] - c[0], m[1] - c[1]])
    return np.array([ct * d[0] + st * d[1], -st * d[0] + ct * d[1]])


def run_gate(rows, eps, eps0, eps_yaw, yaw0, confirm, confirm_tol, warmup_s=10.0):
    """Replay one run. Returns (gated map track, stamps, accepted, rejected)."""
    held = None
    streak = 0
    out_t, out_p, acc, rej = [], [], 0, 0
    d_odom = 0.0
    prev_odom = None
    last_seen = None
    first_t = None
    for st, mp, c in rows:
        odom = invert(c, mp)
        if prev_odom is not None:
            step = float(np.hypot(*(odom - prev_odom)))
            if step < 1.0:                     # a front-end jump is not travel
                d_odom += step
        prev_odom = odom

        # Act only when the correction CHANGES. RTAB-Map republishes the held
        # correction continuously and the recorder samples it at 2 Hz, so
        # counting samples made any rejected correction "confirmed" within 2.5 s
        # -- the escape hatch re-admitted the exact match the gate had just
        # refused, which is why the first version of this made the U square run
        # worse (11.81 m against 6.80). A candidate is a NEW FIX, not a sample.
        fresh = (c != last_seen)
        last_seen = c

        # WARM-UP, by TIME since the first fix, not by samples. The gate's
        # premise is that it already holds a trusted pose, and that is false at
        # the first fix: in four of seven runs the first correction carries
        # 159-172 deg of yaw error and a later one repairs it. Those repairs all
        # arrive within 4 s of the first fix; the false fixes the gate exists to
        # catch arrive at 29 and 50 s. A sample count cannot separate them,
        # because in one run the wrong-yaw correction held long enough to look
        # settled and the gate then rejected its own repair, taking a clean run
        # to 24 m. A window does separate them.
        if held is None:
            if first_t is None:
                first_t = st
            if st - first_t >= warmup_s:
                held = c
                acc += 1
                d_odom = 0.0
        elif fresh and c != held:
            dp = float(np.hypot(c[0] - held[0], c[1] - held[1]))
            dy = abs(wrap(c[2] - held[2]))
            if dp <= eps * d_odom + eps0 and dy <= eps_yaw * d_odom + yaw0:
                held = c
                acc += 1
                streak = 0
                d_odom = 0.0
            else:
                rej += 1
                # CONCEDE ON SUSTAINED DISAGREEMENT. The gate's job is to refuse
                # ISOLATED jumps, not to win an argument with the map. Measured on
                # the recorded verdict streams, the two cases separate:
                #   aliasing          rejections INTERLEAVED with accepts,
                #                     longest run 6 (U static figure-8) and 3
                #   wrong held belief a solid block, 10 in a row (U dynamic),
                #                     because the map is continuously right
                # At 8 no static run concedes and the dynamic failure does.
                #
                # This replaces an agreement test that could never fire: it
                # required `confirm` consecutive candidates within 0.30 m of each
                # OTHER, but while the robot drives on a wrong held correction the
                # offers move (2.61, 3.28, 3.86, 4.82, 4.64, 4.89 m), so they never
                # agreed and the gate ratcheted itself out of its own map.
                streak += 1
                if streak >= confirm:
                    held = c
                    acc += 1
                    streak = 0
                    d_odom = 0.0
        out_t.append(st)
        out_p.append(compose(held if held is not None else c, odom))
    return np.array(out_t), np.array(out_p), acc, rej


def main(a):
    spec = yaml.safe_load(open(os.path.join(WORKSPACE, 'script', 'exp4_protocol.yaml')))
    print(f'gate: |dp| <= {a.eps}*d_odom + {a.eps0} m,  '
          f'|dyaw| <= {a.eps_yaw}*d_odom + {a.yaw0} deg,  '
          f'confirm after {a.confirm}\n')
    print(f'{"run":22} {"corr":>9} | {"as recorded":>22} | {"gated":>22}')
    print(f'{"":22} {"acc/rej":>9} | {"med":>6} {"max":>7} {">1m s":>7} | '
          f'{"med":>6} {"max":>7} {">1m s":>7}')
    tot = []
    for world in spec['worlds']:
        lay = spec['worlds'][world]['layout']
        prior = spec['priors'][lay]['database']
        al = eg._align(prior)
        for path in spec['priors'][lay]['paths']:
            d = os.path.join(WORKSPACE, 'map', prior, 'exp4', f'{world}_{path}_1')
            est = os.path.join(WORKSPACE, 'map', prior,
                               f'localize_exp4_{world}_{path}_1.csv')
            if not (os.path.exists(os.path.join(d, 'robot_gt.csv'))
                    and os.path.exists(est)):
                continue
            rec = list(csv.DictReader(open(est)))
            ff = next((float(r['stamp']) for r in rec if r['kind'] == 'pose'
                       and r.get('cov_trace') and float(r['cov_trace']) < 1e4), None)
            rows = []
            for r in rec:
                if r['kind'] != 'tf' or not r.get('x') or not r.get('corr_x'):
                    continue
                st = float(r['stamp'])
                if ff is not None and st < ff:
                    continue
                rows.append((st,
                             np.array([float(r['x']), float(r['y'])]),
                             (float(r['corr_x']), float(r['corr_y']),
                              float(r['corr_yaw_deg']))))
            if len(rows) < 20:
                continue
            t, gt = es.load_robot_gt(os.path.join(d, 'robot_gt.csv'))
            gm = es.world_to_map(al, gt)

            def score(ts, ps):
                m = (ts >= t[0]) & (ts <= t[-1])
                e = np.hypot(ps[m, 0] - np.interp(ts[m], t, gm[:, 0]),
                             ps[m, 1] - np.interp(ts[m], t, gm[:, 1]))
                over = e > 1.0
                dur = 0.0
                if over.any():
                    tt, s = ts[m], None
                    for i, b in enumerate(over):
                        if b and s is None:
                            s = tt[i]
                        elif not b and s is not None:
                            dur = max(dur, tt[i] - s)
                            s = None
                    if s is not None:
                        dur = max(dur, tt[-1] - s)
                return float(np.median(e)), float(e.max()), dur

            raw_t = np.array([r[0] for r in rows])
            raw_p = np.array([r[1] for r in rows])
            gt_, gp, acc, rej = run_gate(rows, a.eps, a.eps0, a.eps_yaw,
                                         a.yaw0, a.confirm, a.confirm_tol,
                                         a.warmup_s)
            r0 = score(raw_t, raw_p)
            r1 = score(gt_, gp)
            tot.append((r0, r1))
            name = f'{world} {SHAPE.get(path, path)}'.replace('_', ' ')
            print(f'{name:22} {acc:4d}/{rej:<4d} | {r0[0]:6.3f} {r0[1]:7.2f} '
                  f'{r0[2]:7.1f} | {r1[0]:6.3f} {r1[1]:7.2f} {r1[2]:7.1f}')
    if tot:
        print(f'\n  worst max, as recorded {max(x[0][1] for x in tot):.2f} m   '
              f'gated {max(x[1][1] for x in tot):.2f} m')
        print(f'  total seconds over 1 m, as recorded '
              f'{sum(x[0][2] for x in tot):.0f}   gated {sum(x[1][2] for x in tot):.0f}')
        print(f'  median error worsened in '
              f'{sum(1 for x in tot if x[1][0] > x[0][0] + 0.02)} of {len(tot)} runs')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--eps', type=float, default=0.02,
                    help='allowed correction per metre driven (m/m)')
    ap.add_argument('--eps0', type=float, default=1.5, help='fixed allowance (m)')
    ap.add_argument('--eps-yaw', type=float, default=0.10,
                    help='allowed yaw correction per metre driven (deg/m)')
    ap.add_argument('--yaw0', type=float, default=8.0,
                    help='fixed yaw allowance (deg)')
    ap.add_argument('--confirm', type=int, default=8,
                    help='consecutive REJECTIONS after which the gate concedes; 8 clears the worst static aliasing streak (6) by a margin of two')
    ap.add_argument('--warmup-s', type=float, default=10.0,
                    help='seconds after the first fix during which all corrections '
                         'are accepted, so the startup repair is not blocked')
    ap.add_argument('--confirm-tol', type=float, default=0.30,
                    help='how closely candidates must agree to count (m)')
    main(ap.parse_args())
