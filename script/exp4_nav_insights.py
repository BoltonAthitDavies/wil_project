#!/usr/bin/env python3
"""Insight analysis for Experiment 4 (navigation): rotation, the prior's value,
and the anatomy of the two excursions.

    python3 script/exp4_nav_insights.py            # print the findings
    python3 script/exp4_nav_insights.py --write    # also write tables and figures

Named exp4_NAV_insights so as not to collide with script/exp4_insights.py, which
is the MAP insight script under the pre-11-November numbering (maps were
Experiment 4 and are now Experiment 3).

WHAT THE RAW WRITE-UP LEAVES OPEN
    insights/experiment_4/ reports how well the pose tracks ground truth. It does
    not say why the error is the size it is, why two runs left the map for tens of
    seconds, or whether localising on the prior is better than not localising at
    all. This script asks those three.

    Q1  Is the error rotational? Nothing in the slot measures heading, and heading
        error is what generates cross-track error downstream. The November
        acceptance sweep found every false fix to be yaw-dominated, so if the
        excursions carry tens of degrees this slot and that one share a mechanism.
    Q2  What does the prior buy? The slot's premise is that localising against a
        map bounds the error where dead reckoning would not. Every run already
        holds the unaided ORB-SLAM3 trajectory, so the comparison needs no new
        data.
    Q3  What happens at an excursion? Two runs crossed a metre for 36 and 23.5 s
        and both recovered abruptly.

    One run per cell: these are mechanisms, not statistics, and the output says so.
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

OUT = os.path.join(WORKSPACE, 'output', 'compare', 'simulation', 'exp4-nav-insights')
SHAPE = {'P1_transit': 'straight', 'P2_return': 'square', 'P3_circuit': 'figure-8'}


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def gt_yaw(path):
    rows = list(csv.DictReader(open(path)))
    t = np.array([float(r['stamp']) for r in rows])
    qw = np.array([float(r['qw']) for r in rows])
    qx = np.array([float(r['qx']) for r in rows])
    qy = np.array([float(r['qy']) for r in rows])
    qz = np.array([float(r['qz']) for r in rows])
    yaw = np.degrees(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2)))
    vyaw = np.array([float(r.get('vyaw') or 0.0) for r in rows])
    return t, yaw, vyaw


def runs(spec):
    for world in spec['worlds']:
        lay = spec['worlds'][world]['layout']
        prior = spec['priors'][lay]['database']
        for path in spec['priors'][lay]['paths']:
            for rep in (1, 2, 3):
                d = os.path.join(WORKSPACE, 'map', prior, 'exp4',
                                 f'{world}_{path}_{rep}')
                est = os.path.join(WORKSPACE, 'map', prior,
                                   f'localize_exp4_{world}_{path}_{rep}.csv')
                if os.path.exists(os.path.join(d, 'robot_gt.csv')) and os.path.exists(est):
                    yield world, path, rep, lay, prior, d, est


def series(est_csv, first_fix_only=True, crit_blind=1e4):
    rec = list(csv.DictReader(open(est_csv)))
    ff = next((float(r['stamp']) for r in rec if r['kind'] == 'pose'
               and r.get('cov_trace') and float(r['cov_trace']) < crit_blind), None)
    T = [r for r in rec if r['kind'] == 'tf' and r.get('x')
         and (not first_fix_only or ff is None or float(r['stamp']) >= ff)]
    return rec, ff, T


# ---------------------------------------------------------------- Q1 rotation
def rotation(spec):
    out = []
    for world, path, rep, lay, prior, d, est in runs(spec):
        al = eg._align(prior)
        t, ywld, vyaw = gt_yaw(os.path.join(d, 'robot_gt.csv'))
        # GT heading in the MAP frame: the fit is map->world, so subtract its yaw
        ymap = wrap(ywld - al['yaw_deg'])
        _rec, _ff, T = series(est)
        if len(T) < 5:
            continue
        et = np.array([float(r['stamp']) for r in T])
        ey = np.array([float(r['yaw_deg']) for r in T])
        m = (et >= t[0]) & (et <= t[-1])
        # interpolate GT yaw through the unwrapped signal so 180 does not alias
        gi = np.interp(et[m], t, np.unwrap(np.radians(ymap)))
        err = wrap(ey[m] - np.degrees(gi))
        rate = np.abs(np.interp(et[m], t, vyaw))
        # position error on the same samples, to ask whether they move together
        _t, gt = es.load_robot_gt(os.path.join(d, 'robot_gt.csv'))
        gm = es.world_to_map(al, gt)
        ex = np.array([float(r['x']) for r in T])[m]
        ez = np.array([float(r['y']) for r in T])[m]
        perr = np.hypot(ex - np.interp(et[m], _t, gm[:, 0]),
                        ez - np.interp(et[m], _t, gm[:, 1]))
        turning = rate > 0.15
        big = perr > 1.0
        # The first sample after the first fix carries the correction still
        # settling: both 160-degree "maxima" sat there, 163.3 and 166.9 deg at
        # t = 0.0 s, with 6 of 205 and 1 of 61 samples over 30 deg in those runs.
        # It is the acquisition transient again, one sample wide, so the maximum
        # is taken from the second sample on and p95 is the figure to quote.
        if len(err) > 2:
            err, rate, perr = err[1:], rate[1:], perr[1:]
            turning, big = turning[1:], big[1:]
        out.append({
            'run': f'{world} {SHAPE.get(path, path)}', 'world': world, 'path': path,
            'median': float(np.median(np.abs(err))),
            'p95': float(np.percentile(np.abs(err), 95)),
            'max': float(np.abs(err).max()),
            'bias': float(np.mean(err)),
            'straight': float(np.median(np.abs(err[~turning]))) if (~turning).sum() else np.nan,
            'turning': float(np.median(np.abs(err[turning]))) if turning.sum() else np.nan,
            'in_exc': float(np.median(np.abs(err[big]))) if big.sum() else np.nan,
            'out_exc': float(np.median(np.abs(err[~big]))) if (~big).sum() else np.nan,
            'n_exc': int(big.sum()),
            'corr': float(np.corrcoef(np.abs(err), perr)[0, 1]) if len(err) > 5 else np.nan,
        })
    return out


# ------------------------------------------------------- Q2 what the prior buys
def umeyama_2d(P, Q):
    """Rigid planar transform taking P onto Q, no scale."""
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return R, qc - R @ pc


def prior_value(spec):
    out = []
    for world, path, rep, lay, prior, d, est in runs(spec):
        vio = os.path.join(d, 'orb', 'vio.csv')
        if not os.path.exists(vio):
            continue
        al = eg._align(prior)
        t, gt = es.load_robot_gt(os.path.join(d, 'robot_gt.csv'))
        gm = es.world_to_map(al, gt)
        _rec, ff, T = series(est)
        et = np.array([float(r['stamp']) for r in T])
        ex = np.array([float(r['x']) for r in T])
        ey = np.array([float(r['y']) for r in T])
        m = (et >= t[0]) & (et <= t[-1])
        loc = np.hypot(ex[m] - np.interp(et[m], t, gm[:, 0]),
                       ey[m] - np.interp(et[m], t, gm[:, 1]))

        v = np.loadtxt(vio, delimiter=',')
        vt = v[:, 0] / 1e9
        vp = v[:, 1:3]
        k = (vt >= max(t[0], ff or t[0])) & (vt <= t[-1])
        if k.sum() < 20:
            continue
        # The front end's frame is arbitrary, so the fair comparison is after the
        # best rigid alignment onto ground truth -- i.e. pure DRIFT, with the
        # unknown initial pose given away for free. That is the kindest possible
        # reading of unaided odometry.
        G = np.column_stack([np.interp(vt[k], t, gm[:, 0]), np.interp(vt[k], t, gm[:, 1])])
        R, tr = umeyama_2d(vp[k], G)
        va = (R @ vp[k].T).T + tr
        drift = np.hypot(*(va - G).T)
        out.append({'run': f'{world} {SHAPE.get(path, path)}', 'world': world,
                    'path': path, 'driven': float(np.hypot(*np.diff(gm, axis=0).T)
                                                  [np.hypot(*np.diff(gm, axis=0).T) < 1].sum()),
                    'loc_med': float(np.median(loc)), 'loc_max': float(loc.max()),
                    'vio_med': float(np.median(drift)), 'vio_max': float(drift.max()),
                    'vio_final': float(drift[-1])})
    return out


# ------------------------------------------------------- Q3 excursion anatomy
def excursions(spec, thr=1.0):
    out = []
    for world, path, rep, lay, prior, d, est in runs(spec):
        al = eg._align(prior)
        t, gt = es.load_robot_gt(os.path.join(d, 'robot_gt.csv'))
        gm = es.world_to_map(al, gt)
        rec, ff, T = series(est)
        et = np.array([float(r['stamp']) for r in T])
        ex = np.array([float(r['x']) for r in T])
        ey = np.array([float(r['y']) for r in T])
        cy = np.array([float(r['corr_yaw_deg']) if r.get('corr_yaw_deg') else np.nan
                       for r in T])
        cx = np.array([float(r['corr_x']) if r.get('corr_x') else np.nan for r in T])
        m = (et >= t[0]) & (et <= t[-1])
        err = np.hypot(ex[m] - np.interp(et[m], t, gm[:, 0]),
                       ey[m] - np.interp(et[m], t, gm[:, 1]))
        tt, cyy, cxx = et[m], cy[m], cx[m]
        bad = err > thr
        i = 0
        while i < len(bad):
            if not bad[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(bad) and bad[j + 1]:
                j += 1
            dur = tt[j] - tt[i]
            if dur >= 5.0:
                out.append({
                    'run': f'{world} {SHAPE.get(path, path)}',
                    'start_s': float(tt[i] - tt[0]), 'dur_s': float(dur),
                    'peak_m': float(err[i:j + 1].max()),
                    'corr_yaw_before': float(cyy[max(i - 1, 0)]),
                    'corr_yaw_during': float(np.nanmedian(cyy[i:j + 1])),
                    'corr_yaw_after': float(cyy[min(j + 1, len(cyy) - 1)]),
                    'corr_x_before': float(cxx[max(i - 1, 0)]),
                    'corr_x_during': float(np.nanmedian(cxx[i:j + 1])),
                    'err_after': float(err[min(j + 1, len(err) - 1)]),
                })
            i = j + 1
    return out


def _mpl():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def fig_rotation(spec, rot):
    """Heading error per run, and the absence of a turn-rate effect.

    Left: median and 95th percentile side by side, the dynamic run shaded. The
    point is the gap, not the absolute level -- every static run is under
    1.2 deg and the one run with traffic is 4.85.
    Right: each run's degradation relative to the best static run, in heading
    and in position. Traffic moves heading about ten times and position about
    three, so a position-only report understates it.
    """
    plt = _mpl()
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    names = [r['run'].replace('_', ' ') for r in rot]
    x = np.arange(len(rot))
    dyn = [i for i, r in enumerate(rot) if 'dynamic' in r['world']]
    a.bar(x - 0.2, [r['median'] for r in rot], 0.4, label='median',
          color=['#d95f02' if i in dyn else '#1b9e77' for i in x])
    a.bar(x + 0.2, [r['p95'] for r in rot], 0.4, label='95th percentile',
          color=['#f0a868' if i in dyn else '#8fd3c0' for i in x])
    a.set_xticks(x)
    a.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    a.set_ylabel('heading error (deg)')
    a.set_title('heading error per run (orange: the run with traffic)')
    a.legend(fontsize=8)
    a.grid(axis='y', alpha=0.2)

    base_h = min(r['median'] for r in rot if 'dynamic' not in r['world'])
    pos = {r['run']: None for r in rot}
    for r in rot:
        pos[r['run']] = r
    b.scatter([r['straight'] for r in rot], [r['turning'] for r in rot],
              c=['#d95f02' if 'dynamic' in r['world'] else '#1b9e77' for r in rot],
              s=70, zorder=3)
    lim = max(max(r['straight'] for r in rot), max(r['turning'] for r in rot)) * 1.15
    b.plot([0, lim], [0, lim], color='#7f8c8d', ls='--', lw=1,
           label='equal: no turn-rate effect')
    for r in rot:
        b.annotate(r['run'].replace('_', ' ').replace(' static', '').replace(' dynamic', ''),
                   (r['straight'], r['turning']), fontsize=7,
                   xytext=(4, 4), textcoords='offset points')
    b.set_xlim(0, lim)
    b.set_ylim(0, lim)
    b.set_xlabel('heading error while driving straight (deg)')
    b.set_ylabel('heading error while turning (deg)')
    b.set_title('turning does not degrade heading')
    b.legend(fontsize=8)
    b.grid(alpha=0.2)
    fig.tight_layout()
    out = os.path.join(OUT, 'fig_rotation.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_excursions(spec):
    """The two excursions side by side, each with what the correction did.

    Top row is position error, bottom row the correction's yaw and translation on
    twin axes. The figure-of-eight's yaw trace steps 11 deg and its translation
    does not; the square's translation steps 5 m and its yaw does not. That is
    the whole of Insight 2 and it is far clearer drawn than tabulated.
    """
    plt = _mpl()
    cases = [('U_static', 'P3_circuit', 'figure-of-eight: a rotation'),
             ('U_static', 'P2_return', 'square: a translation')]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 6.4), sharex='col')
    for col, (world, path, title) in enumerate(cases):
        lay = spec['worlds'][world]['layout']
        prior = spec['priors'][lay]['database']
        al = eg._align(prior)
        d = os.path.join(WORKSPACE, 'map', prior, 'exp4', f'{world}_{path}_1')
        t, gt = es.load_robot_gt(os.path.join(d, 'robot_gt.csv'))
        gm = es.world_to_map(al, gt)
        _rec, _ff, T = series(os.path.join(WORKSPACE, 'map', prior,
                                           f'localize_exp4_{world}_{path}_1.csv'))
        et = np.array([float(r['stamp']) for r in T])
        ex = np.array([float(r['x']) for r in T])
        ey = np.array([float(r['y']) for r in T])
        cy = np.array([float(r['corr_yaw_deg']) if r.get('corr_yaw_deg') else np.nan
                       for r in T])
        cx = np.array([float(r['corr_x']) if r.get('corr_x') else np.nan for r in T])
        m = (et >= t[0]) & (et <= t[-1])
        err = np.hypot(ex[m] - np.interp(et[m], t, gm[:, 0]),
                       ey[m] - np.interp(et[m], t, gm[:, 1]))
        tt = et[m] - et[m][0]
        # Heading error is the discriminator, not the correction alone: BOTH
        # excursions move the correction in yaw and in translation, and what
        # separates them is whether the robot's heading went with it. Drawing
        # only the corrections made them look alike.
        tgt, ywld, _v = gt_yaw(os.path.join(d, 'robot_gt.csv'))
        ymap = wrap(ywld - al['yaw_deg'])
        eyaw = np.array([float(r['yaw_deg']) for r in T])[m]
        gi = np.degrees(np.interp(et[m], tgt, np.unwrap(np.radians(ymap))))
        herr = np.abs(wrap(eyaw - gi))

        top, bot = axes[0, col], axes[1, col]
        top.plot(tt, err, color='#2c3e50', lw=1.6, label='position error (m)')
        top.axhline(1.0, color='#7f8c8d', ls=':', lw=1)
        top.set_ylabel('position error (m)' if col == 0 else '')
        top.set_title(title)
        top.grid(alpha=0.2)
        th = top.twinx()
        th.plot(tt, herr, color='#8e44ad', lw=1.5, label='heading error (deg)')
        th.set_ylim(0, max(14, herr.max() * 1.1))
        th.set_ylabel('heading error (deg)' if col == 1 else '')
        h1, l1 = top.get_legend_handles_labels()
        h2, l2 = th.get_legend_handles_labels()
        top.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper left')
        bad = err > 1.0
        if bad.any():
            top.axvspan(tt[bad][0], tt[bad][-1], color='#d95f02', alpha=0.12)
            bot.axvspan(tt[bad][0], tt[bad][-1], color='#d95f02', alpha=0.12)
        bot.plot(tt, cy[m], color='#d95f02', lw=1.8, label='correction yaw (deg)')
        bot.set_ylabel('correction yaw (deg)' if col == 0 else '')
        bot.set_xlabel('time since the first fix (s)')
        bot.grid(alpha=0.2)
        tw = bot.twinx()
        tw.plot(tt, cx[m], color='#1b9e77', lw=1.8, label='correction x (m)')
        tw.set_ylabel('correction x (m)' if col == 1 else '')
        h1, l1 = bot.get_legend_handles_labels()
        h2, l2 = tw.get_legend_handles_labels()
        bot.legend(h1 + h2, l1 + l2, fontsize=8, loc='best')
    fig.suptitle('Two excursions, two mechanisms: the heading error separates them',
                 y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(OUT, 'fig_excursions.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_prior_value(pv):
    """What the prior buys, as a paired comparison.

    Left: each run's median, localised against unaided front end, joined. A line
    sloping down is the prior helping, up is it hurting; they split three to
    three with one level, which is the finding.
    Right: the worst instant of each, logarithmic. The front end never exceeds
    0.76 m; the localised pose reaches 6.80. That asymmetry, not the medians, is
    what the prior costs.
    """
    plt = _mpl()
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    names = [r['run'].replace('_', ' ') for r in pv]
    y = np.arange(len(pv))
    for i, r in enumerate(pv):
        better = r['loc_med'] < r['vio_med']
        a.plot([r['vio_med'], r['loc_med']], [i, i],
               color='#1b9e77' if better else '#d95f02', lw=2.5, zorder=2)
    a.scatter([r['vio_med'] for r in pv], y, s=60, color='#7f8c8d', zorder=3,
              label='unaided ORB-SLAM3 (drift, aligned to GT)')
    a.scatter([r['loc_med'] for r in pv], y, s=60, color='#2c3e50', zorder=3,
              marker='D', label='localised on the prior')
    a.set_yticks(y)
    a.set_yticklabels(names, fontsize=8)
    a.set_xlabel('median error (m)')
    a.set_title('green: the prior helps   orange: it hurts')
    a.legend(fontsize=8, loc='lower right')
    a.grid(axis='x', alpha=0.2)

    w = 0.38
    b.barh(y - w / 2, [r['vio_med'] for r in pv], w, color='#95a5a6',
           label='front end, median')
    b.barh(y + w / 2, [r['loc_med'] for r in pv], w, color='#34495e',
           label='localised, median')
    b.scatter([r['vio_max'] for r in pv], y - w / 2, marker='|', s=160,
              color='#7f8c8d', label='front end, worst instant')
    b.scatter([r['loc_max'] for r in pv], y + w / 2, marker='|', s=160,
              color='#c0392b', label='localised, worst instant')
    b.set_xscale('log')
    b.set_yticks(y)
    b.set_yticklabels([])
    b.set_xlabel('error (m), logarithmic')
    b.set_title('the worst instant is where they differ')
    b.legend(fontsize=7, loc='lower right')
    b.grid(axis='x', alpha=0.2, which='both')
    fig.tight_layout()
    out = os.path.join(OUT, 'fig_prior_value.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main(write):
    spec = yaml.safe_load(open(os.path.join(WORKSPACE, 'script', 'exp4_protocol.yaml')))

    print('=' * 78)
    print('Q1  ROTATION: heading error, degrees')
    print('=' * 78)
    print(f'{"run":22} {"med":>6} {"p95":>6} {"max":>7} {"bias":>7} '
          f'{"straight":>8} {"turning":>8} {"in exc":>7} {"out exc":>7} {"r(pos)":>7}')
    rot = rotation(spec)
    for r in rot:
        print(f'{r["run"]:22} {r["median"]:6.2f} {r["p95"]:6.2f} {r["max"]:7.2f} '
              f'{r["bias"]:+7.2f} {r["straight"]:8.2f} {r["turning"]:8.2f} '
              f'{r["in_exc"]:7.2f} {r["out_exc"]:7.2f} {r["corr"]:7.2f}')

    print()
    print('=' * 78)
    print('Q2  WHAT THE PRIOR BUYS: localised pose against unaided ORB-SLAM3 drift')
    print('    (VIO rigidly aligned to GT first, so this is drift alone)')
    print('=' * 78)
    print(f'{"run":22} {"driven":>7} {"loc med":>8} {"loc max":>8} '
          f'{"vio med":>8} {"vio max":>8} {"vio end":>8}')
    pv = prior_value(spec)
    for r in pv:
        print(f'{r["run"]:22} {r["driven"]:7.1f} {r["loc_med"]:8.3f} {r["loc_max"]:8.2f} '
              f'{r["vio_med"]:8.3f} {r["vio_max"]:8.2f} {r["vio_final"]:8.2f}')

    print()
    print('=' * 78)
    print('Q3  EXCURSION ANATOMY (stretches over 1 m lasting 5 s or more)')
    print('=' * 78)
    ex = excursions(spec)
    if not ex:
        print('  none')
    for r in ex:
        print(f'  {r["run"]:22} start {r["start_s"]:6.1f}s  for {r["dur_s"]:5.1f}s  '
              f'peak {r["peak_m"]:5.2f} m')
        print(f'      correction yaw  before {r["corr_yaw_before"]:+7.2f}  '
              f'during {r["corr_yaw_during"]:+7.2f}  after {r["corr_yaw_after"]:+7.2f} deg')
        print(f'      correction x    before {r["corr_x_before"]:+7.2f}  '
              f'during {r["corr_x_during"]:+7.2f} m      '
              f'error after recovery {r["err_after"]:.3f} m')
    if write:
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, 'rotation.tex'), 'w') as f:
            f.write('\\begin{table}[htbp]\n\\centering\\small\n'
                    '\\caption{Heading error against ground truth, degrees. The first '
                    'sample after the first fix is excluded: it carries the correction '
                    'still settling and held both runs\' apparent 160-degree maxima. '
                    '``In exc.\'\' is the median while the position error is over '
                    '\\SI{1}{\\metre}, ``out\'\' while it is not, and $r$ their '
                    'correlation.}\n\\label{tab:insight4nav-rotation}\n'
                    '\\begin{tabular}{lrrrrrrrr}\n\\toprule\n'
                    'Run & Med. & p95 & Max & Bias & Straight & Turning & '
                    'In exc. & $r$ \\\\\n\\midrule\n')
            for r in rot:
                ie = f'{r["in_exc"]:.2f}' if r['in_exc'] == r['in_exc'] else '--'
                f.write(f'{r["run"].replace("_", " ")} & {r["median"]:.2f} & '
                        f'{r["p95"]:.2f} & {r["max"]:.2f} & {r["bias"]:+.2f} & '
                        f'{r["straight"]:.2f} & {r["turning"]:.2f} & {ie} & '
                        f'{r["corr"]:.2f} \\\\\n')
            f.write('\\bottomrule\n\\end{tabular}\n\\end{table}\n')
        with open(os.path.join(OUT, 'prior_value.tex'), 'w') as f:
            f.write('\\begin{table}[htbp]\n\\centering\\small\n'
                    '\\caption{The localised pose against unaided ORB-SLAM3 over the '
                    'same run, metres. The front end is rigidly aligned to ground truth '
                    'first, so its column is DRIFT alone with the unknown initial pose '
                    'given away free -- the kindest possible reading of odometry, and '
                    'the one thing the prior actually supplies.}\n'
                    '\\label{tab:insight4nav-prior}\n'
                    '\\begin{tabular}{lrrrrrr}\n\\toprule\n'
                    'Run & Driven & Loc. med. & Loc. max & VIO med. & VIO max & '
                    'VIO end \\\\\n\\midrule\n')
            for r in pv:
                f.write(f'{r["run"].replace("_", " ")} & {r["driven"]:.1f} & '
                        f'{r["loc_med"]:.3f} & {r["loc_max"]:.2f} & '
                        f'{r["vio_med"]:.3f} & {r["vio_max"]:.2f} & '
                        f'{r["vio_final"]:.2f} \\\\\n')
            f.write('\\bottomrule\n\\end{tabular}\n\\end{table}\n')
        with open(os.path.join(OUT, 'excursions.tex'), 'w') as f:
            f.write('\\begin{table}[htbp]\n\\centering\\small\n'
                    '\\caption{The two excursions, and the correction that produced '
                    'each. Yaw in degrees, $x$ in metres.}\n'
                    '\\label{tab:insight4nav-exc}\n'
                    '\\begin{tabular}{lrrrrrrr}\n\\toprule\n'
                    'Run & Start & Dur. & Peak & Yaw before & Yaw during & '
                    '$x$ before & $x$ during \\\\\n\\midrule\n')
            for r in ex:
                f.write(f'{r["run"].replace("_", " ")} & {r["start_s"]:.0f}\\,s & '
                        f'{r["dur_s"]:.1f}\\,s & {r["peak_m"]:.2f} & '
                        f'{r["corr_yaw_before"]:+.2f} & {r["corr_yaw_during"]:+.2f} & '
                        f'{r["corr_x_before"]:+.2f} & {r["corr_x_during"]:+.2f} '
                        f'\\\\\n')
            f.write('\\bottomrule\n\\end{tabular}\n\\end{table}\n')
        for f in (fig_rotation(spec, rot), fig_excursions(spec), fig_prior_value(pv)):
            print(f'  figure -> {f}')
        print(f'\n3 tables and 3 figures -> {OUT}')
    return rot, pv, ex


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--write', action='store_true')
    main(ap.parse_args().write)
