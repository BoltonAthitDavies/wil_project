#!/usr/bin/env python3
"""Score the VSLAM measured-IMU-noise treatment against the hand-set baseline.

    python3 script/analyse_measured_noise_vslam.py
    python3 script/analyse_measured_noise_vslam.py --csv out.csv

WHAT THIS CLOSES
    Table 4.3 compares an EKF whose IMU noise was MEASURED from each bag against
    VSLAM whose IMU noise was four hand-set constants. Tuning was therefore
    confounded with the estimator: a win for either family could be the
    estimator or could be the tuning.

    analyse_noise_transfer.py does the cheap half -- the EKF re-run at the
    VSLAM constants, offline. This does the other half, which needed Gazebo:
    VSLAM re-run at the EKF's measured per-bag model. The `_new` run directories
    hold that treatment; the unsuffixed ones are the hand-set baseline.

WHAT TRANSFERS, AND WHAT CANNOT
    The filter consumes exactly two measured IMU sigmas -- gyro z and accel x --
    plus a gyro bias random walk. Those three move to the VSLAM configs, divided
    by sqrt(200) where the filter's per-sample sigma has to become a continuous
    density. AccWalk/acc_w has no counterpart because the filter carries no
    accelerometer-bias state, so it stays SDF-derived in every condition and is
    NOT part of the treatment.

METRICS
    ate_rms   aligned ATE, rigid SE(3) with scale fixed, three-dimensional [m]
    rot_rms   geodesic rotation error under the same alignment [deg]
    rpe_t     RPE at delta = 1 s on the UNALIGNED poses, translation RMSE [m]
    rpe_r     the same pairs, rotation MEDIAN [deg]

    The rotation column is a median, not an RMSE, deliberately. ORB-SLAM3
    re-aligns to gravity a couple of seconds after initialisation; that one
    rotation-only step is not a position jump, so the `clean` filter does not
    remove it and it inflates an RPE rotation RMSE without describing steady
    state. See the rpe-clean-misses-rotation-only-jumps note.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                             # noqa: E402
import vio_metrics as vm                                       # noqa: E402

DATASETS = ['dataset_allsensor_realistic_%03d' % i for i in range(4)]

# (label, tree, baseline run, measured-noise run). A run of None is a flat tree
# -- the proprio estimators write straight into <tree>/<dataset>/.
ARMS = [
    ('VINS-Fusion',           'output_vins', 'cpu',                 'cpu_new'),
    ('ORB-SLAM3',             'output_orb',  'loopclosing_gpu',     'loopclosing_gpu_new'),
    ('ORB-SLAM3 (no LC)',     'output_orb',  'noloopclosing_gpu',   'noloopclosing_gpu_new'),
    ('ORB-SLAM3 CPU',         'output_orb',  'loopclosing_cpu',     'loopclosing_cpu_new'),
    ('ORB-SLAM3 CPU (no LC)', 'output_orb',  'noloopclosing_cpu',   'noloopclosing_cpu_new'),
]
# Reference arms, scored identically so the columns mean the same thing.
REFS = [('EKF @ VSLAM noise',      'output_ekf',          None),
        ('EKF @ measured noise',   'output_ekf_measured', None),
        ('Wheel odometry',         'output_wheel',        None),
        ('IMU dead reckoning',     'output_imu',          None)]


def _rpe(d, g, t_from=None):
    """RPE at delta = 1 s against ground truth, optionally from t_from onwards."""
    e = vm.load(os.path.join(d, 'vio.csv'))
    if e is None or g is None:
        return float('nan'), float('nan')
    t, p, q = e[0], e[1], e[2]
    lo, hi = max(t[0], g[0][0]), min(t[-1], g[0][-1])
    m = (t >= lo) & (t <= hi)
    if t_from is not None:
        m &= t >= t_from
    if m.sum() < 10:
        return float('nan'), float('nan')
    tt, pp, qq = t[m], p[m], q[m]
    r = vm.rpe(tt, pp, qq, vm.resample(g[0], g[1], tt),
               vm.resample_quat(g[0], g[2], tt), delta_s=1.0,
               jump_idx=vm.find_jumps(tt, pp)[0])
    return (r['trans_rmse'], r['rot_median']) if r else (float('nan'), float('nan'))


def score(dataset, label, tree, run):
    """Metrics for one run, on the full overlap AND anchored at IMU init.

    The anchored columns exist because ORB-SLAM3 publishes in the camera
    optical frame until its gravity-aligned world frame exists -- see
    vio_metrics.imu_init_time(). An estimator that logs no such flag is scored
    once and reports the same numbers in both column sets, so a table can mix
    the families without the anchor meaning something different per row.
    """
    sysent = [(label, tree, run)] if run else [(label, tree)]
    # output_vins is appended only to put it on the ground-truth candidate list;
    # it never loads a run, because the runs live in named subdirectories.
    probe = sysent + [('__gt__', 'output_vins')]
    res = vm.analyse('simulation/' + dataset, systems=probe)
    if res is None or not res[0]:
        return None
    row, al = res[0][0], res[1][0]

    d = vm.run_dir(tree, 'simulation/' + dataset, run or 'never')
    g = None
    for c in vm.gt_candidates('simulation/' + dataset, sysent + vm.SYSTEMS):
        g = vm.load(c)
        if g is not None:
            break
    rpe_t, rpe_r = _rpe(d, g)

    out = dict(ate_rms=row['rms'], ate_max=row['max'], rot_rms=row['rot_rms'],
               secs=row['secs'], poses=row['poses'], rpe_t=rpe_t, rpe_r=rpe_r)

    t_init = vm.world_frame_time(d)
    out['t_init'] = float('nan') if t_init is None else t_init - al['t0']
    if t_init is None or t_init <= al['t0']:
        # No initialisation transient to anchor past: the anchored columns are
        # the same measurement, not a missing one.
        out.update(a_ate=row['rms'], a_rot=row['rot_rms'],
                   a_rpe_t=rpe_t, a_rpe_r=rpe_r, a_secs=row['secs'],
                   anchored=False)
        return out

    res2 = vm.analyse('simulation/' + dataset, systems=probe,
                      trim_start=t_init - al['t0'])
    if res2 is None or not res2[0]:
        out.update(a_ate=float('nan'), a_rot=float('nan'), a_rpe_t=float('nan'),
                   a_rpe_r=float('nan'), a_secs=float('nan'), anchored=False)
        return out
    r2 = res2[0][0]
    a_rpe_t, a_rpe_r = _rpe(d, g, t_from=t_init)
    out.update(a_ate=r2['rms'], a_rot=r2['rot_rms'], a_rpe_t=a_rpe_t,
               a_rpe_r=a_rpe_r, a_secs=r2['secs'], anchored=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv')
    a = ap.parse_args()

    recs = []
    for ds in DATASETS:
        for label, tree, base, new in ARMS:
            for cond, run in (('hand-set', base), ('measured', new)):
                s = score(ds, label, tree, run)
                if s is None:
                    print('  [skip] %s / %s / %s' % (ds, label, cond), file=sys.stderr)
                    continue
                recs.append(dict(dataset=ds, arm=label, cond=cond, run=run, **s))
        for label, tree, run in REFS:
            s = score(ds, label, tree, run)
            if s is not None:
                recs.append(dict(dataset=ds, arm=label, cond='n/a', run=run, **s))

    hdr = ('%-24s %-9s | %7s %7s %7s %7s | %6s %7s %7s %7s %7s'
           % ('arm', 'noise', 'ATE m', 'rot', 'RPE m', 'RPE d',
              'init s', 'ATE m', 'rot', 'RPE m', 'RPE d'))
    def line(lbl, cond, r):
        return ('%-24s %-9s | %7.3f %7.2f %7.3f %7.2f | %6s %7.3f %7.2f %7.3f %7.2f'
                % (lbl, cond, r['ate_rms'], r['rot_rms'], r['rpe_t'], r['rpe_r'],
                   ('%.2f' % r['t_init']) if r['t_init'] == r['t_init'] else '--',
                   r['a_ate'], r['a_rot'], r['a_rpe_t'], r['a_rpe_r']))

    for ds in DATASETS:
        print('\n=== %s ===' % ds)
        print('%-34s | %31s | %39s' % ('', 'FULL OVERLAP', 'ANCHORED AT IMU INIT'))
        print(hdr)
        print('-' * len(hdr))
        for label, _, _, _ in ARMS:
            for cond in ('hand-set', 'measured'):
                r = next((x for x in recs if x['dataset'] == ds and x['arm'] == label
                          and x['cond'] == cond), None)
                if r:
                    print(line(label, cond, r))
        print('-' * len(hdr))
        for label, _, _ in REFS:
            r = next((x for x in recs if x['dataset'] == ds and x['arm'] == label), None)
            if r:
                print(line(label, '--', r))

    print('\n=== what the anchor does to the rotation column ===')
    print('%-24s %9s %9s %9s' % ('arm', 'rot full', 'rot anch', 'factor'))
    for label, _, _, _ in ARMS:
        f, a_ = [], []
        for ds in DATASETS:
            for cond in ('hand-set', 'measured'):
                r = next((x for x in recs if x['dataset'] == ds and x['arm'] == label
                          and x['cond'] == cond), None)
                if r and r['anchored']:
                    f.append(r['rot_rms']); a_.append(r['a_rot'])
        if f:
            print('%-24s %9.2f %9.2f %8.1fx'
                  % (label, np.mean(f), np.mean(a_), np.mean(f) / np.mean(a_)))

    # Paired effect of the treatment, per arm, across the four recordings.
    print('\n=== treatment effect: measured - hand-set, per arm ===')
    print('%-24s %7s %22s %22s' % ('arm', 'bags', 'ATE m (mean, range)', 'RPE m (mean, range)'))
    for label, _, _, _ in ARMS:
        d_ate, d_rpe = [], []
        for ds in DATASETS:
            b = next((x for x in recs if x['dataset'] == ds and x['arm'] == label and x['cond'] == 'hand-set'), None)
            n = next((x for x in recs if x['dataset'] == ds and x['arm'] == label and x['cond'] == 'measured'), None)
            if b and n:
                d_ate.append(n['ate_rms'] - b['ate_rms'])
                d_rpe.append(n['rpe_t'] - b['rpe_t'])
        if d_ate:
            print('%-24s %7d %+9.3f [%+.3f, %+.3f] %+9.3f [%+.3f, %+.3f]'
                  % (label, len(d_ate), np.mean(d_ate), min(d_ate), max(d_ate),
                     np.nanmean(d_rpe), np.nanmin(d_rpe), np.nanmax(d_rpe)))

    if a.csv:
        import csv
        with open(a.csv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
            w.writeheader()
            w.writerows(recs)
        print('\nwrote %s (%d rows)' % (a.csv, len(recs)))


if __name__ == '__main__':
    main()
