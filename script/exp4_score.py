#!/usr/bin/env python3
"""Score Experiment 4 from the recorded data, not from the action client.

    python3 script/exp4_score.py                 # the table
    python3 script/exp4_score.py --run U_static_P2_return_1   # one run in detail

WHY THIS EXISTS RATHER THAN JUST READING outcome.json
    The runs were operator-driven, so nothing in the record says whether a goal
    was reached -- outcome.json carries verdict 'driven' and an empty goal list.
    But ground truth was logged, and the goals are known poses, so whether the
    robot got to each one is a measurement rather than a judgement: the minimum
    GT distance to the goal over the run, and when. That is stricter than the
    action client's own opinion and it works for a driven run and an autonomous
    one alike.

    It also fills the other gap. --drive never subscribed to /localization_pose,
    so the blind-pose columns in outcome.json read 0/0 for every driven run. The
    launch's own localisation CSV has all of it, so availability is recovered from
    there instead of being absent.

FRAMES, AND THE ONE CORRECTION THAT MATTERS
    Ground truth is in the Gazebo world frame; goals and the localised pose are in
    the map frame. They are reconciled with the per-layout map->world fit solved in
    exp4_goals._align (yaw -179.93 / -179.44 deg), whose residual is the 0.338 m
    lever arm and nothing else.

    That lever arm is applied here and never to recorded data: /ground_truth/
    odometry reports the MODEL ORIGIN while the prior's nodes and the localised
    pose are the camera/IMU body, 0.338 m ahead and 0.192 m up. Comparing them raw
    puts a third of a metre of bias into every number.

WHAT 'reached' MEANS
    The GT body point came within goal_tolerance_m of the goal at some moment in
    the run. It does NOT mean Nav2 declared success, and it does not check the
    heading: a goal pose's yaw comes from the mapping run's heading at that node,
    which is not a heading the robot has any reason to adopt on arrival.
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import yaml

WORKSPACE = os.path.expanduser('~/wil_project')
sys.path.insert(0, os.path.join(WORKSPACE, 'script'))
import exp4_goals as eg                                        # noqa: E402

LEVER = np.array([0.338, 0.0, 0.192])      # vio_metrics.IMU_LEVER_M


def _quat_R(qw, qx, qy, qz):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qx * qw + qy * qz), 1 - 2 * (qx * qx + qy * qy)]])


def load_robot_gt(path, lever=False):
    """(t, xy_world) from robot_gt.csv. NO lever arm by default -- see below.

    THE LEVER ARM DOES NOT BELONG IN THIS COMPARISON, and applying it put a
    systematic 0.338 m bias into every number this module produced before
    12 November 2026. The recorder samples map -> base_link, and the published
    base_link sits where the model origin is, so ground truth needs no
    correction to meet it. Decomposing the error along and across the direction
    of travel is what showed it: with the lever arm applied the along-track mean
    was -0.28 to -0.38 m in all seven runs and the cross-track mean was near
    zero -- a constant offset in the direction of travel, of exactly the lever
    arm's length, in the same direction every time. Without it the along-track
    mean is -0.16 to +0.04 m and changes sign between runs, which is what an
    unbiased estimate looks like.

    It does still belong in exp4_goals._align, which fits DATABASE NODE POSES
    against ground truth. Those are body poses, and leaving the lever arm out
    there made the fit markedly worse (residual 0.337 m against 0.103 m). The
    two are different quantities and the frames legitimately differ; the
    alignment is body-to-body, this comparison is base-to-origin.
    """
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None, None
    t = np.array([float(r['stamp']) for r in rows])
    p = np.array([[float(r['x']), float(r['y']), float(r['z'])] for r in rows])
    if not lever:
        return t, p[:, :2]
    out = np.empty((len(rows), 3))
    for i, r in enumerate(rows):
        R = _quat_R(float(r['qw']), float(r['qx']), float(r['qy']), float(r['qz']))
        out[i] = p[i] + R @ LEVER
    return t, out[:, :2]


def map_to_world(al, xy):
    xy = np.atleast_2d(np.asarray(xy, float))
    return (al['_R'] @ xy.T).T + al['_t']


def world_to_map(al, xy):
    xy = np.atleast_2d(np.asarray(xy, float))
    return (al['_R'].T @ (xy - al['_t']).T).T


def localisation(prior, run_name, crit):
    """First fix, blind fraction and pose jumps, from the launch's own CSV."""
    p = os.path.join(WORKSPACE, 'map', prior, f'localize_{run_name}.csv')
    if not os.path.exists(p):
        return {'note': 'no localisation CSV'}
    rec = list(csv.DictReader(open(p)))
    P = [r for r in rec if r['kind'] == 'pose' and r.get('cov_trace')]
    T = [r for r in rec if r['kind'] == 'tf' and r.get('x')]
    if not P:
        return {'note': 'no localisation poses'}
    t0 = float(P[0]['stamp'])
    inf = [float(r['stamp']) - t0 for r in P if float(r['cov_trace']) < crit['cov_trace_blind']]
    blind = sum(1 for r in P if float(r['cov_trace']) >= crit['cov_trace_blind'])
    out = {'first_fix_s': round(inf[0], 1) if inf else None,
           'blind_poses': blind, 'poses': len(P),
           'blind_fraction': round(blind / len(P), 3)}
    # pose jumps, the warp measure: the robot's own map-frame step, excluding the
    # seed transition and the acquisition snap (see aisle_sweep for why the
    # correction's translation is the wrong thing to measure)
    jumps = []
    acq = inf[0] if inf else None
    seq = [(float(r['stamp']) - t0, float(r['x']), float(r['y'])) for r in T]
    driven = 0.0
    for i in range(1, len(seq)):
        t, d = seq[i][0], math.dist(seq[i][1:], seq[i - 1][1:])
        if t < 8.0:
            continue
        if d > crit['pose_jump_fail_m']:
            if acq is None or abs(t - acq) >= 2.0:
                jumps.append(round(d, 2))
        else:
            driven += d
    out.update({'pose_jumps': len(jumps), 'worst_jump_m': max(jumps, default=0.0),
                'tf_path_m': round(driven, 1)})
    return out


def score_run(spec, world, path, rep, verbose=False):
    w = spec['worlds'][world]
    layout = w['layout']
    prior = spec['priors'][layout]['database']
    d = os.path.join(WORKSPACE, 'map', prior, 'exp4', f'{world}_{path}_{rep}')
    rob = os.path.join(d, 'robot_gt.csv')
    if not os.path.exists(rob):
        return None
    t, gt = load_robot_gt(rob)
    if t is None:
        return None
    al = eg._align(prior)
    crit = spec['criteria']

    goals = spec['priors'][layout]['paths'][path]
    gw = map_to_world(al, [g['map_xy'] for g in goals])
    results = []
    for g, (gx, gy) in zip(goals, gw):
        dist = np.hypot(gt[:, 0] - gx, gt[:, 1] - gy)
        k = int(np.argmin(dist))
        results.append({'node': g['node'], 'closest_m': round(float(dist[k]), 2),
                        'at_s': round(float(t[k] - t[0]), 1),
                        'reached': bool(dist[k] <= crit['goal_tolerance_m'])})
    # driven distance, and the straight-line sum of the legs for comparison
    step = np.hypot(*np.diff(gt, axis=0).T)
    driven = float(step[step < 1.0].sum())
    start = gt[0]
    legs, prev = 0.0, start
    for (gx, gy) in gw:
        legs += math.dist(prev, (gx, gy))
        prev = (gx, gy)

    # ATE of the localised pose against GT, both in the map frame
    est_csv = os.path.join(WORKSPACE, 'map', prior, f'localize_exp4_{world}_{path}_{rep}.csv')
    ate = None
    if os.path.exists(est_csv):
        E = [r for r in csv.DictReader(open(est_csv))
             if r['kind'] == 'pose' and r.get('x') and r.get('cov_trace')
             and float(r['cov_trace']) < crit['cov_trace_blind']]
        if len(E) > 5:
            et = np.array([float(r['stamp']) for r in E])
            ep = np.array([[float(r['x']), float(r['y'])] for r in E])
            gm = world_to_map(al, gt)
            inside = (et >= t[0]) & (et <= t[-1])
            if inside.sum() > 5:
                gx = np.interp(et[inside], t, gm[:, 0])
                gy = np.interp(et[inside], t, gm[:, 1])
                err = np.hypot(ep[inside, 0] - gx, ep[inside, 1] - gy)
                ate = {'n': int(inside.sum()), 'median_m': round(float(np.median(err)), 3),
                       'rms_m': round(float(np.sqrt((err ** 2).mean())), 3),
                       'max_m': round(float(err.max()), 3)}

    out = {'world': world, 'path': path, 'rep': rep, 'goals': results,
           'reached': sum(1 for r in results if r['reached']), 'n_goals': len(results),
           'driven_m': round(driven, 1), 'legs_m': round(legs, 1),
           'detour': round(driven / legs, 2) if legs > 1 else None,
           'duration_s': round(float(t[-1] - t[0]), 1), 'ate': ate,
           'loc': localisation(prior, f'exp4_{world}_{path}_{rep}', crit)}
    man = os.path.join(d, 'outcome.json')
    if os.path.exists(man):
        m = json.load(open(man))
        out['min_clearance_m'] = m.get('min_clearance_m')
        out['min_actor_clearance_m'] = m.get('min_actor_clearance_m')
        out['nearest_actor'] = m.get('nearest_actor')
    if verbose:
        print(json.dumps(out, indent=2))
    return out


def tracking(spec):
    """How well the published pose tracks ground truth. Goals are not involved.

    TWO ERRORS, AND THE SECOND IS THE ONE THAT MATTERS
      fix     /localization_pose, published only when RTAB-Map accepts a
              localisation. How good each accepted fix is.
      pose    the map -> base_link transform, sampled at 2 Hz: the pose RViz draws
              and Nav2 consumes, which includes the front end's dead reckoning
              between fixes. A stack can have excellent fixes and an unusable
              pose if the fixes are rare.

    Both are compared against /ground_truth/odometry with the 0.338 / 0.192 m
    lever arm applied to the ground truth, since it reports the model origin and
    the estimators report the camera/IMU body. Both are in the map frame, via the
    per-layout fit whose residual is that same lever arm.
    """
    rows = []
    for world in spec['worlds']:
        lay = spec['worlds'][world]['layout']
        prior = spec['priors'][lay]['database']
        al = eg._align(prior)
        crit = spec['criteria']
        for path in spec['priors'][lay]['paths']:
            for rep in (1, 2, 3):
                d = os.path.join(WORKSPACE, 'map', prior, 'exp4',
                                 f'{world}_{path}_{rep}')
                rob = os.path.join(d, 'robot_gt.csv')
                est = os.path.join(WORKSPACE, 'map', prior,
                                   f'localize_exp4_{world}_{path}_{rep}.csv')
                if not (os.path.exists(rob) and os.path.exists(est)):
                    continue
                t, gt = load_robot_gt(rob)
                if t is None:
                    continue
                gm = world_to_map(al, gt)
                rec = list(csv.DictReader(open(est)))

                # The first informative pose. Everything before it is the seed
                # composed with raw odometry, which can sit far outside the
                # building: I_static/P3_circuit's 16.51 m "maximum" was at
                # t = 1.0 s against a first fix at 4.7 s, i.e. the acquisition
                # transient, not a tracking failure. The figures already draw
                # that segment dashed for the same reason; the table must not
                # count it either.
                first_fix = None
                for r in rec:
                    if (r['kind'] == 'pose' and r.get('cov_trace')
                            and float(r['cov_trace']) < crit['cov_trace_blind']):
                        first_fix = float(r['stamp'])
                        break

                def series(kind, require_cov):
                    out = []
                    for r in rec:
                        if r['kind'] != kind or not r.get('x'):
                            continue
                        if require_cov and (not r.get('cov_trace')
                                            or float(r['cov_trace']) >= crit['cov_trace_blind']):
                            continue
                        st = float(r['stamp'])
                        if first_fix is not None and st < first_fix:
                            continue
                        out.append((st, float(r['x']), float(r['y']),
                                    float(r['yaw_deg']) if r.get('yaw_deg') else None))
                    return out

                def decompose(seq):
                    """Split the error into a constant offset and scatter about it.

                    They mean different things for navigation. A constant offset
                    is the prior map being locally wrong where the robot is: the
                    pose is consistent, just displaced, and a planner can live
                    with it. Scatter is the estimate moving under the robot, which
                    a controller feels directly.

                    Measured over the static runs: scatter 0.06-0.25 m while the
                    offset reaches 0.59 m, so the error is mostly displacement.
                    And the offset is region-dependent -- the straight
                    single-aisle runs carry 0.3-0.6 m while the building-wide
                    circuits average to 0.02-0.14 m -- which is what local map
                    distortion looks like, not estimator noise.
                    """
                    if len(seq) < 5:
                        return None
                    et = np.array([x[0] for x in seq])
                    ex = np.array([x[1] for x in seq])
                    ey = np.array([x[2] for x in seq])
                    m = (et >= t[0]) & (et <= t[-1])
                    if m.sum() < 5:
                        return None
                    dx = ex[m] - np.interp(et[m], t, gm[:, 0])
                    dy = ey[m] - np.interp(et[m], t, gm[:, 1])
                    # a diverged tail is a different failure and would swamp the
                    # offset; it is reported by p95 and the half-by-half columns
                    k = np.hypot(dx, dy) < 2.0
                    if k.sum() < 5:
                        return None
                    dx, dy = dx[k], dy[k]
                    mx, my = float(dx.mean()), float(dy.mean())
                    return {'offset_m': float(math.hypot(mx, my)),
                            'offset_xy': (round(mx, 3), round(my, 3)),
                            'scatter_m': float(np.hypot(dx - mx, dy - my).std()),
                            'n_used': int(k.sum()), 'n_total': int(m.sum())}

                def excursion(seq, thr=1.0):
                    """Longest unbroken stretch, in seconds, with error over thr.

                    The half-by-half medians hide an excursion that straddles the
                    midpoint, and the maximum hides how long one lasted. Both U
                    runs crossed 2 m for tens of seconds and then recovered to
                    about 0.11 m: the square from t=41 to 76 s, the figure-of-eight
                    from t=29 to 52 s. A table reporting only medians and a maximum
                    called the first 'lost the pose' and the second clean.
                    """
                    if len(seq) < 5:
                        return None
                    et = np.array([x[0] for x in seq])
                    ex = np.array([x[1] for x in seq])
                    ey = np.array([x[2] for x in seq])
                    m = (et >= t[0]) & (et <= t[-1])
                    if m.sum() < 5:
                        return None
                    e = np.hypot(ex[m] - np.interp(et[m], t, gm[:, 0]),
                                 ey[m] - np.interp(et[m], t, gm[:, 1]))
                    tt, bad = et[m], e > thr
                    best = cur = 0.0
                    start = None
                    for i, b in enumerate(bad):
                        if b and start is None:
                            start = tt[i]
                        elif not b and start is not None:
                            cur = tt[i] - start
                            best = max(best, cur)
                            start = None
                    if start is not None:
                        best = max(best, tt[-1] - start)
                    tail = e[int(0.9 * len(e)):]
                    return {'longest_s': round(float(best), 1),
                            'recovered_to_m': round(float(np.median(tail)), 3)}

                def err_of(seq):
                    if len(seq) < 5:
                        return None
                    et = np.array([s[0] for s in seq])
                    ex = np.array([s[1] for s in seq])
                    ey = np.array([s[2] for s in seq])
                    m = (et >= t[0]) & (et <= t[-1])
                    if m.sum() < 5:
                        return None
                    gx = np.interp(et[m], t, gm[:, 0])
                    gy = np.interp(et[m], t, gm[:, 1])
                    e = np.hypot(ex[m] - gx, ey[m] - gy)
                    # drift: does the error grow with distance, or stay bounded?
                    half = len(e) // 2
                    return {'n': int(m.sum()), 'median': float(np.median(e)),
                            'p95': float(np.percentile(e, 95)), 'max': float(e.max()),
                            'first_half': float(np.median(e[:half])),
                            'second_half': float(np.median(e[half:]))}

                fix = err_of(series('pose', True))
                pose = err_of(series('tf', False))
                split = decompose(series('tf', False))
                exc = excursion(series('tf', False))
                step = np.hypot(*np.diff(gt, axis=0).T)
                rows.append({'world': world, 'path': path, 'rep': rep,
                             'driven_m': float(step[step < 1.0].sum()),
                             'dur': float(t[-1] - t[0]), 'fix': fix, 'pose': pose,
                             'split': split, 'exc': exc,
                             'loc': localisation(prior, f'exp4_{world}_{path}_{rep}',
                                                 crit)})
    if not rows:
        sys.exit('no runs with both ground truth and a localisation CSV.')
    print('pose = map->base_link at 2 Hz, what RViz draws and Nav2 consumes')
    print('fix  = /localization_pose, only where RTAB-Map accepted a localisation')
    print('both against /ground_truth/odometry (map->base_link vs the model origin;\nno lever arm -- see load_robot_gt)\n')
    print(f'{"world":10} {"shape":11} {"rep":>3} {"driven":>7} {"s":>5} | '
          f'{"pose med":>8} {"p95":>6} {"max":>6} {"1st/2nd half":>13} | '
          f'{"offset":>6} {"scatter":>7} | {"exc>1m":>6} {"ended":>6} | '
          f'{"blind":>6} {"jumps":>5}')
    for r in rows:
        p_, f_ = r['pose'], r['fix']
        half = (f'{p_["first_half"]:.2f}/{p_["second_half"]:.2f}' if p_ else '-')
        print(f'{r["world"]:10} {r["path"]:11} {r["rep"]:3d} {r["driven_m"]:7.1f} '
              f'{r["dur"]:5.0f} | '
              f'{(p_["median"] if p_ else float("nan")):8.3f} '
              f'{(p_["p95"] if p_ else float("nan")):6.3f} '
              f'{(p_["max"] if p_ else float("nan")):6.2f} {half:>13} | '
              f'{(r["split"]["offset_m"] if r["split"] else float("nan")):6.3f} '
              f'{(r["split"]["scatter_m"] if r["split"] else float("nan")):7.3f} | '
              f'{(r["exc"]["longest_s"] if r["exc"] else float("nan")):6.1f} '
              f'{(r["exc"]["recovered_to_m"] if r["exc"] else float("nan")):6.3f} | '
              f'{r["loc"].get("blind_fraction", float("nan")):6.3f} '
              f'{r["loc"].get("pose_jumps", 0):5d}')
    print('\n  1st/2nd half  median error over the first and second half of the run.')
    print('                Equal means the error is BOUNDED by the map, which is the')
    print('                whole claim of localising against a prior; growing means it')
    print('                is dead reckoning and the fixes are not holding it.')
    print('  offset        length of the MEAN error vector: the prior map being')
    print('                locally wrong where the robot is. A planner tolerates it.')
    print('  scatter       spread about that offset: the estimate moving under the')
    print('                robot, which a controller feels directly.')
    print('  exc>1m        longest unbroken stretch with error over 1 m; "ended" is')
    print('                the median over the last tenth of the run, i.e. whether it')
    print('                came back. A half-by-half median hides an excursion that')
    print('                straddles the midpoint, and a maximum hides its duration.')
    print('  max           post-first-fix only: the pre-fix segment is the seed plus')
    print('                raw odometry and is excluded, not a tracking error')
    print('  jumps         map-frame pose steps over 1 m after that snap')
    allp = [r['pose']['median'] for r in rows if r['pose']]
    print(f'\n  median over {len(allp)} runs: {np.median(allp):.3f} m'
          f'   worst run {max(allp):.3f} m')


def main(only=None):
    spec = yaml.safe_load(open(os.path.join(WORKSPACE, 'script', 'exp4_protocol.yaml')))
    rows = []
    for world in spec['worlds']:
        for path in spec['priors'][spec['worlds'][world]['layout']]['paths']:
            for rep in (1, 2, 3):
                if only and only != f'{world}_{path}_{rep}':
                    continue
                r = score_run(spec, world, path, rep, verbose=bool(only))
                if r:
                    rows.append(r)
    if not rows:
        sys.exit('no runs with ground truth found.')
    if only:
        return
    tol = spec['criteria']['goal_tolerance_m']
    print(f'goal reached = GT body point within {tol} m of the goal at some point\n')
    print(f'{"world":10} {"path":11} {"rep":>3} {"goals":>6} {"worst goal":>10} '
          f'{"driven":>7} {"detour":>6} {"ATE med":>8} {"first fix":>9} '
          f'{"blind":>7} {"jumps":>5} {"clear":>6}')
    for r in rows:
        worst = max((g['closest_m'] for g in r['goals']), default=float('nan'))
        # A detour ratio below 1 is geometrically impossible -- you cannot drive
        # less than the straight line -- so it is the signal that the route was
        # never driven, not an efficiency figure. Three P1 runs read 0.64-0.68
        # while ending 26-28 m from their single goal: the robot went down its own
        # aisle and stopped. Flagged rather than printed as if it meant something.
        if worst > 5.0:
            r['detour'] = 'n/a'
        ate = r['ate']['median_m'] if r['ate'] else float('nan')
        loc = r['loc']
        print(f'{r["world"]:10} {r["path"]:11} {r["rep"]:3d} '
              f'{r["reached"]:3d}/{r["n_goals"]:<2d} {worst:10.2f} '
              f'{r["driven_m"]:7.1f} {str(r["detour"]):>6} {ate:8.3f} '
              f'{str(loc.get("first_fix_s")):>9} '
              f'{loc.get("blind_fraction", float("nan")):7.3f} '
              f'{loc.get("pose_jumps", 0):5d} '
              f'{r.get("min_clearance_m", float("nan")):6.2f}')
    print('\n  worst goal  the furthest any goal in that path was ever approached')
    print('  detour      driven distance over the straight-line sum of the legs;'
          '\n              1.3-1.6 is a normal aisle route. n/a where a goal was never'
          '\n              approached, because the ratio then describes a route the'
          '\n              robot did not take')
    print('  ATE med     median error of the localised pose against ground truth,')
    print('              both in the map frame, lever arm applied')
    print('  blind       fraction of localisation poses carrying no information')
    print('  jumps       map-frame pose steps over 1 m after the acquisition snap')
    print('  clear       min GT centre-to-centre clearance to a static model')
    ok = sum(1 for r in rows if r['reached'] == r['n_goals'])
    print(f'\n  {ok} of {len(rows)} runs reached every goal in their path')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', metavar='WORLD_PATH_REP')
    ap.add_argument('--tracking', action='store_true',
                    help='pose accuracy against ground truth only; no goals')
    a = ap.parse_args()
    if a.tracking:
        tracking(yaml.safe_load(open(os.path.join(WORKSPACE, 'script',
                                                  'exp4_protocol.yaml'))))
    else:
        main(a.run)
