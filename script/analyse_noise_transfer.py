#!/usr/bin/env python3
"""Score the EKF noise-transfer experiment and write its auditable artifacts.

    # 1. build the transferred and ablated noise models
    python3 script/make_vslam_noise_yaml.py --preset inflated \
        --noise output/compare/simulation/exp-estimator-baseline/noise
    python3 script/analyse_noise_transfer.py --emit-ablation-models

    # 2. run the EKF once per condition per bag (see --print-commands)
    python3 script/analyse_noise_transfer.py --print-commands

    # 3. score everything
    python3 script/analyse_noise_transfer.py

RESEARCH QUESTION
    Is the EKF's accuracy advantage over VINS-Fusion and ORB-SLAM3 in
    tab:proprio-accuracy an estimator effect, or an effect of the noise model?

    The two families are not tuned alike. The EKF's noise model is measured from
    each bag by `proprio_estimator.py calibrate`; the visual estimators read four
    constants from config/wil_sim/stereo_imu.yaml. Tuning is therefore an
    uncontrolled variable in that comparison, confounded with the estimator.

HYPOTHESIS (H4), FALSIFIABLE
    H4: the EKF's advantage survives being given the visual systems' own IMU
    noise configuration. H4 is falsified if, on the transferred configuration,
    the mean aligned ATE of the EKF over the five bags is no better than
    VINS-Fusion's.

    Decision rule, and the direction of the test, were fixed before scoring.
    They were NOT fixed before the first exploratory run -- see LIMITATIONS.

INDEPENDENT VARIABLE
    The four IMU noise terms the filter consumes, at three levels:
      measured   per-bag values from `calibrate`            (output_ekf)
      inflated   the values stereo_imu.yaml declares        (output_ekf_vslamcfg)
      ablation   one term at a time moved to `inflated`     (output_ekf_ablation)

    Everything else is held fixed: the same five bags, the same ground truth,
    the same filter code and commit, the same bicycle yaw-rate model, the same
    50 Hz wheel-update stride, and the same alignment convention.

    The accelerometer and gyro biases and the entire wheel block stay at their
    measured values in every condition, so the IMU noise model is the only
    treatment. The one exception is the `Ronly` ablation, which deliberately
    raises the wheel yaw-rate sigma -- see below.

WHY THE `Ronly` ABLATION EXISTS
    The gyro sigma enters the filter in two places: the heading process noise in
    Q, and the bias observation's R as sw^2 + sg^2. Those cannot be separated by
    editing sg, because editing it moves both. Raising sw so that

        sw_new^2 + sg_measured^2  ==  sw_old^2 + sg_inflated^2

    reproduces the R change EXACTLY while leaving Q at the measured value. A
    null result there localises the whole effect to Q.

UNITS
    stereo_imu.yaml states densities per sqrt(Hz); the filter stores a per-sample
    sigma, because its process-noise block multiplies by dt rather than sqrt(dt).
    sigma = density * sqrt(f), with f taken from each yaml's own imu_dt_s rather
    than assumed. gyro_bias_rw is the exception: it enters through G[4,2]=sqrt(dt),
    already the density convention, so gyr_w transfers unchanged.

LIMITATIONS, STATED BEFORE THE NUMBERS
    1. The exploratory runs were scored before the hypothesis above was written
       down. This is a post-hoc formalisation of an exploratory finding, and the
       unanimity across five bags is what makes it worth reporting -- not a
       pre-registered test. A confirmatory repeat on new bags is the honest
       follow-up.
    2. Equal handicap is not equal effort. Both families now run on a declared
       noise model that nobody fitted to the data. The mirror experiment --
       giving the visual systems the measured values -- needs re-recording and
       has not been run to a reportable standard.
    3. The transfer is not symmetric in role: the filter's sigma_g is a scalar in
       a planar five-state model, whereas gyr_n feeds three-axis pre-integration.
    4. n = 1 run per bag per condition. The filter is deterministic given a bag
       and a noise model, so repeats would be identical and no run-to-run
       uncertainty can be quoted. Uncertainty here is across BAGS, not runs, and
       is reported as a range rather than a standard deviation.
"""

import argparse
import copy
import csv
import math
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vio_metrics as vm                                       # noqa: E402

ROOT = vm.ROOT
DOMAIN = 'simulation'
EXPERIMENT = 'exp-ekf-noise-transfer'
BAGS = ('000', '001', '002', '003', '004')
NOISE_DIR = os.path.join(ROOT, 'output/compare/simulation/'
                               'exp-estimator-baseline/noise')
ABL_DIR = os.path.join(ROOT, 'output/compare', DOMAIN, EXPERIMENT, 'noise')

# Published aligned ATE for the visual estimators, read from the baseline
# experiment rather than recomputed, so this script cannot quietly restate them.
BASELINE_METRICS = os.path.join(ROOT, 'output/compare/simulation/'
                                      'exp-estimator-baseline/proprio_metrics.csv')

# (condition key, tree, human label). The tree is where proprio_estimator.py was
# told to write, so a condition is identified by its artifacts, not by a name in
# this file.
CONDITIONS = [
    ('measured',  'output_ekf',           'measured throughout'),
    ('inflated',  'output_ekf_vslamcfg',  'all four (EKF @ VSLAM noise)'),
    ('gyroonly',  'output_ekf_ablation',  'gyroscope density'),
    ('acconly',   'output_ekf_ablation',  'accelerometer density'),
    ('rwonly',    'output_ekf_ablation',  'gyroscope bias random walk'),
    ('Ronly',     'output_ekf_ablation',  'gyroscope density through R only'),
]
ABLATIONS = ('gyroonly', 'acconly', 'rwonly', 'Ronly')

# stereo_imu.yaml, the file both visual estimators actually read.
VSLAM_GYR_N = 0.002        # rad/s/sqrt(Hz)
VSLAM_ACC_N = 0.02         # m/s^2/sqrt(Hz)
VSLAM_GYR_W = 0.0001       # rad/s^2/sqrt(Hz)


# --------------------------------------------------------------- noise models

def emit_ablation_models(dry=False):
    """One noise model per (bag, ablation), each moving exactly one term."""
    for i in BAGS:
        src = os.path.join(NOISE_DIR, 'dataset_allsensor_%s' % i,
                           'sensor_noise.yaml')
        if not os.path.exists(src):
            raise SystemExit('missing %s -- run `proprio_estimator.py calibrate` '
                             'for this bag first' % src)
        base = yaml.safe_load(open(src))
        f = math.sqrt(1.0 / float(base['window']['imu_dt_s']))
        sg_inf, sa_inf = VSLAM_GYR_N * f, VSLAM_ACC_N * f
        sg_meas = float(base['imu']['gyro_sigma_xyz'][2])
        sw_meas = float(base['wheel']['yaw_rate_sigma_bicycle'])

        out = os.path.join(ABL_DIR, 'dataset_allsensor_%s' % i)
        if not dry:
            os.makedirs(out, exist_ok=True)
        for key in ABLATIONS:
            n = copy.deepcopy(base)
            if key == 'gyroonly':
                n['imu']['gyro_sigma_xyz'] = [sg_inf] * 3
            elif key == 'acconly':
                n['imu']['accel_sigma_xyz'] = [sa_inf] * 3
            elif key == 'rwonly':
                n['imu']['gyro_bias_rw'] = VSLAM_GYR_W
                n['imu']['gyro_bias_rw_source'] = 'VSLAM config stereo_imu.yaml'
            elif key == 'Ronly':
                # Reproduce the gyro term's effect on R exactly, leaving Q alone.
                n['wheel']['yaw_rate_sigma_bicycle'] = math.sqrt(
                    sw_meas ** 2 + sg_inf ** 2 - sg_meas ** 2)
            n['provenance']['note'] = (
                'ablation %s for %s: exactly one IMU noise term moved to the '
                'value config/wil_sim/stereo_imu.yaml declares. Generated by '
                'script/analyse_noise_transfer.py --emit-ablation-models.' % (key, EXPERIMENT))
            n['provenance']['override_preset'] = 'ablation:' + key
            p = os.path.join(out, 'sensor_noise_%s.yaml' % key)
            print('%s  %s' % ('would write' if dry else 'wrote',
                              os.path.relpath(p, ROOT)))
            if not dry:
                yaml.safe_dump(n, open(p, 'w'), sort_keys=False,
                               default_flow_style=False)


def print_commands():
    """Exact commands, in run order. This script never launches the estimator."""
    print('# Section 7 of the skill: the user runs these, not the assistant.\n')
    for key, tree, _ in CONDITIONS:
        if key == 'measured':
            print('# `measured` is the existing baseline in output_ekf/ -- '
                  'do not re-run it, it is a published condition.\n')
            continue
        for i in BAGS:
            if key == 'inflated':
                noise = os.path.join(NOISE_DIR, 'dataset_allsensor_%s' % i,
                                     'sensor_noise_inflated.yaml')
                out = os.path.join(ROOT, 'output', tree, DOMAIN,
                                   'dataset_allsensor_%s' % i)
            else:
                noise = os.path.join(ABL_DIR, 'dataset_allsensor_%s' % i,
                                     'sensor_noise_%s.yaml' % key)
                out = os.path.join(ROOT, 'output', tree, DOMAIN,
                                   'dataset_allsensor_%s_%s' % (i, key))
            print('python3 script/proprio_estimator.py run --estimator ekf \\\n'
                  '  --bag %s/dataset/dataset_allsensor_%s/dataset_allsensor_%s_0.db3 \\\n'
                  '  --noise %s \\\n'
                  '  --experiment-id %s \\\n'
                  '  --out %s --allow-foreign-noise'
                  % (ROOT, i, i, noise, EXPERIMENT, out))
        print()


# ------------------------------------------------------------------- scoring

def traj(path):
    a = np.loadtxt(path, delimiter=',', ndmin=2)
    return dict(t=a[:, 0] * 1e-9, p=a[:, 1:4], q=a[:, 4:8],
                yaw=np.unwrap(2.0 * np.arctan2(a[:, 7], a[:, 4])))


def run_path(key, tree, bag):
    ds = 'dataset_allsensor_%s' % bag
    if key not in ('measured', 'inflated'):
        ds += '_' + key
    return os.path.join(ROOT, 'output', tree, DOMAIN, ds)


def score_one(d, gt_dir):
    """Aligned ATE, anchored error, RPE and the matched-sample bookkeeping 6.1
    requires. Returns None when the run is absent."""
    v = os.path.join(d, 'vio.csv')
    if not os.path.exists(v) or os.path.getsize(v) == 0:
        return None
    e = traj(v)
    g = traj(os.path.join(gt_dir, 'ground_truth.csv'))

    # Common interval only -- 6.1 requires it to be stated, not assumed.
    lo, hi = max(e['t'][0], g['t'][0]), min(e['t'][-1], g['t'][-1])
    m = (e['t'] >= lo) & (e['t'] <= hi)
    t = e['t'][m]
    gp = vm.resample(g['t'], g['p'], t)
    gq = vm.resample_quat(g['t'], g['q'], t)
    gyaw = np.interp(t, g['t'], g['yaw'])

    err = np.linalg.norm(e['p'][m][:, :2] - gp[:, :2], axis=1)
    yerr = np.degrees((e['yaw'][m] - gyaw + np.pi) % (2 * np.pi) - np.pi)
    R, tr = vm.umeyama(e['p'][m], gp)
    al = (R @ e['p'][m].T).T + tr
    aerr = np.linalg.norm(al[:, :2] - gp[:, :2], axis=1)
    arot = vm.geodesic_deg(vm.quat_to_R(_wxyz(e['q'][m], R)), vm.quat_to_R(gq))

    r = dict(matched_samples=int(m.sum()), matched_duration_s=float(t[-1] - t[0]),
             anchored_rms=float(np.sqrt((err ** 2).mean())),
             anchored_final=float(err[-1]),
             aligned_ate_rms=float(np.sqrt((aerr ** 2).mean())),
             aligned_ate_max=float(aerr.max()),
             aligned_rot_rms=float(np.sqrt((arot ** 2).mean())),
             yaw_rms_deg=float(np.sqrt((yerr ** 2).mean())),
             yaw_final_deg=float(yerr[-1]),
             path_len_m=float(np.linalg.norm(np.diff(gp[:, :2], axis=0),
                                             axis=1).sum()))
    r['drift_pct'] = 100.0 * r['anchored_final'] / max(1e-9, r['path_len_m'])
    for delta in (1.0,):
        out = vm.rpe(t, e['p'][m], e['q'][m], gp, gq, delta_s=delta)
        r['rpe%g_trans' % delta] = float(out['trans_rmse'])
        r['rpe%g_rot' % delta] = float(out['rot_rmse'])
        r['rpe%g_pairs' % delta] = int(out['pairs'])
    return r


def _wxyz(q, R):
    """Apply the alignment rotation to a (w,x,y,z) quaternion array, returning
    the same layout. Done through matrices so the convention cannot drift from
    vio_metrics'."""
    Re = vm.quat_to_R(q)
    Ra = np.einsum('ij,njk->nik', R, Re)
    # Back to quaternions via the same helper the rest of the chapter uses is not
    # available, so return matrices wrapped for quat_to_R's caller instead.
    return _R_to_quat(Ra)


def _R_to_quat(R):
    w = np.sqrt(np.maximum(0.0, 1.0 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2])) / 2
    w = np.maximum(w, 1e-12)
    x = (R[:, 2, 1] - R[:, 1, 2]) / (4 * w)
    y = (R[:, 0, 2] - R[:, 2, 0]) / (4 * w)
    z = (R[:, 1, 0] - R[:, 0, 1]) / (4 * w)
    return np.stack([w, x, y, z], 1)


def visual_baseline():
    """Published aligned ATE per visual system, read from the baseline CSV."""
    out = {}
    if not os.path.exists(BASELINE_METRICS):
        return out
    for row in csv.DictReader(open(BASELINE_METRICS)):
        if not row['aligned_ate_rms']:
            continue
        out.setdefault(row['name'], {})[row['dataset'].split('_')[-1]] = \
            float(row['aligned_ate_rms'])
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--emit-ablation-models', action='store_true')
    ap.add_argument('--print-commands', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    if args.emit_ablation_models:
        return emit_ablation_models(dry=args.dry_run)
    if args.print_commands:
        return print_commands()

    outdir = os.path.join(ROOT, 'output/compare', DOMAIN, EXPERIMENT)
    os.makedirs(outdir, exist_ok=True)

    recs, missing = [], []
    for key, tree, label in CONDITIONS:
        for i in BAGS:
            gt = run_path('measured', 'output_ekf', i)
            r = score_one(run_path(key, tree, i), gt)
            if r is None:
                missing.append((key, i))
                continue
            r.update(condition=key, label=label, dataset='dataset_allsensor_%s' % i)
            recs.append(r)

    if missing:
        print('MISSING RUNS -- these conditions are incomplete and are reported '
              'as such rather than averaged over what exists:')
        for key, i in missing:
            print('   %-10s %s' % (key, i))
        print()

    cols = ['condition', 'label', 'dataset', 'matched_samples',
            'matched_duration_s', 'path_len_m', 'anchored_rms', 'anchored_final',
            'drift_pct', 'aligned_ate_rms', 'aligned_ate_max', 'aligned_rot_rms',
            'yaw_rms_deg', 'yaw_final_deg', 'rpe1_trans', 'rpe1_rot', 'rpe1_pairs']
    csv_path = os.path.join(outdir, 'noise_transfer_metrics.csv')
    with open(csv_path, 'w') as f:
        f.write(','.join(cols) + '\n')
        for r in recs:
            f.write(','.join(('%.6g' % r[c]) if isinstance(r.get(c), float)
                             else str(r.get(c, '')) for c in cols) + '\n')
    print('wrote %s  (%d rows)' % (os.path.relpath(csv_path, ROOT), len(recs)))

    by = {}
    for r in recs:
        by.setdefault(r['condition'], []).append(r)

    def mean(key, col):
        v = [r[col] for r in by.get(key, [])]
        return float(np.mean(v)) if len(v) == len(BAGS) else float('nan')

    print('\n%-34s %9s %9s %9s %9s' % ('condition', 'anchRMS', 'alATE', 'alRot',
                                       'yawRMS'))
    for key, _, label in CONDITIONS:
        print('%-34s %9.3f %9.3f %9.2f %9.2f'
              % (label, mean(key, 'anchored_rms'), mean(key, 'aligned_ate_rms'),
                 mean(key, 'aligned_rot_rms'), mean(key, 'yaw_rms_deg')))

    vis = visual_baseline()
    if vis and len(by.get('inflated', [])) == len(BAGS):
        inf = {r['dataset'].split('_')[-1]: r['aligned_ate_rms']
               for r in by['inflated']}
        print('\nH4 test -- EKF on the transferred config against each visual '
              'system, aligned ATE:')
        for name in ('VINS-Fusion', 'ORB-SLAM3', 'ORB-SLAM3 (no LC)'):
            if name not in vis:
                continue
            wins = sum(1 for i in BAGS if vis[name].get(i, np.inf) < inf[i])
            mv = float(np.mean([vis[name][i] for i in BAGS if i in vis[name]]))
            print('   %-20s beats it on %d of %d bags   mean %.3f vs %.3f m'
                  % (name, wins, len(BAGS), mv, float(np.mean(list(inf.values())))))
        mv = float(np.mean([vis['VINS-Fusion'][i] for i in BAGS]))
        verdict = ('FALSIFIED' if mv <= float(np.mean(list(inf.values())))
                   else 'not falsified')
        print('   H4 (the advantage survives the transfer): %s' % verdict)

    write_tex(os.path.join(outdir, 'noise_transfer_ablation.tex'), by, mean)


def write_tex(path, by, mean):
    """The Chapter 4 ablation table, generated so its numbers are auditable
    against noise_transfer_metrics.csv rather than retyped."""
    if any(len(by.get(k, [])) != len(BAGS) for k, _, _ in CONDITIONS):
        print('not writing the LaTeX table: at least one condition is incomplete')
        return
    order = [('measured', 'measured throughout'),
             ('gyroonly', 'gyroscope density, $\\times 6.5$'),
             ('acconly', 'accelerometer density'),
             ('rwonly', 'gyroscope bias random walk, $\\times 10$'),
             ('inflated', 'all four (\\emph{EKF @ VSLAM noise})'),
             (None, None),
             ('Ronly', 'gyroscope density through $R$ only')]
    with open(path, 'w') as f:
        f.write('% Generated by script/analyse_noise_transfer.py -- '
                'do not edit by hand.\n')
        f.write('\\begin{tabular}{lrrr}\n\\toprule\n')
        f.write('EKF, one term moved & Anchored RMS & Aligned ATE & '
                'Anchored yaw RMS \\\\\n')
        f.write(' & [\\si{\\metre}] & [\\si{\\metre}] & [\\si{\\degree}] \\\\\n'
                '\\midrule\n')
        for key, label in order:
            if key is None:
                f.write('\\midrule\n')
                continue
            f.write('%s & %.3f & %.3f & %.2f \\\\\n'
                    % (label, mean(key, 'anchored_rms'),
                       mean(key, 'aligned_ate_rms'), mean(key, 'yaw_rms_deg')))
        f.write('\\bottomrule\n\\end{tabular}\n')
    print('wrote %s' % os.path.relpath(path, ROOT))


if __name__ == '__main__':
    sys.exit(main())
