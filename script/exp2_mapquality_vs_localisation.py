#!/usr/bin/env python3
"""Does the localised pose get worse where the prior map is worse?

Experiment 2 part B (navigation on the prior) against part A (map quality).
For every post-first-fix pose sample of the six reported runs (rep 5, gate
eps0 1.5) this looks up the map quality around the robot's TRUE position and
asks whether the localisation error is larger where that quality is lower.

Local quality, in a disc of radius R around the robot (world frame, 5 cm cells,
classified exactly as exp4_insights.score_map does):
  precision   correct / occupied           (occupied cell within 0.25 m of a true surface)
  recall      1 - missed / seen true edges (true surface the map left more than 0.25 m away)
  bad_frac    (thickened + phantom) / occupied
  node_res    median node-pose residual of the graph nodes inside the disc,
              after the body-to-body Procrustes fit (exp4_goals._align): how
              wrong the prior's own POSES are there, as opposed to its grid.

Statistics. Samples are 2 Hz and strongly autocorrelated, so the sample count
is not the evidence. Three tests that respect that:
  1. circular-shift permutation: shift each run's error series against its own
     quality series by a random lag (at least 10 % of the run), recompute the
     pooled within-run Spearman rho; the null keeps both autocorrelations.
  2. low- vs high-quality contrast (below / above the pooled median), median
     error difference, with a moving-block bootstrap CI (10 s blocks, within run).
  3. per-run sign: how many of the six runs have rho in the same direction.

Prints to stdout, writes one CSV of per-sample values, and writes the per-run
map-bound table docs/report/final_report/tables/exp2_map_bound.tex
(tab:exp4-map-bound): node error near the route against localisation error.
"""
import csv
import math
import os
import re
import sqlite3
import struct
import sys
import tempfile

import numpy as np
from scipy.stats import spearmanr, rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp4_score as es          # noqa: E402
import exp4_goals as eg          # noqa: E402
import exp4_insights as ei       # noqa: E402
from validate_map import load_gt  # noqa: E402

ROOT = ei.ROOT
OUT = ROOT / 'output' / 'compare' / 'simulation' / 'exp2-mapquality-vs-loc'
TABLE = ROOT / 'docs' / 'report' / 'final_report' / 'tables' / 'exp2_map_bound.tex'
RADII = (2.0, 3.0, 5.0)
MAIN_R = 3.0
RNG = np.random.default_rng(20261124)
LAYOUTS = {
    'U': dict(prior='dataset_allsensor_realistic_map', world='U_static'),
    'I': dict(prior='dataset_allsensor_realistic_01_map', world='I_static'),
}
PATHS = ('P1_transit', 'P2_return', 'P3_circuit')
REP = 5
SHAPE = {'P1_transit': 'straight', 'P2_return': 'square', 'P3_circuit': 'figure-8'}


def run_series(prior, world, path, al, crit):
    d = ROOT / 'map' / prior / 'exp4' / f'{world}_{path}_{REP}'
    t, gt = es.load_robot_gt(str(d / 'robot_gt.csv'))
    rec = list(csv.DictReader(open(ROOT / 'map' / prior / f'localize_exp4_{world}_{path}_{REP}.csv')))
    first_fix = next(float(r['stamp']) for r in rec if r['kind'] == 'pose' and r.get('cov_trace')
                     and float(r['cov_trace']) < crit['cov_trace_blind'])
    S = np.array([(float(r['stamp']), float(r['x']), float(r['y'])) for r in rec
                  if r['kind'] == 'tf' and r.get('x') and float(r['stamp']) >= first_fix])
    S = S[(S[:, 0] >= t[0]) & (S[:, 0] <= t[-1])]
    gw = np.column_stack([np.interp(S[:, 0], t, gt[:, 0]), np.interp(S[:, 0], t, gt[:, 1])])
    gm = es.world_to_map(al, gw)
    err = np.hypot(S[:, 1] - gm[:, 0], S[:, 2] - gm[:, 1])
    step = np.r_[0, np.hypot(*np.diff(gw, axis=0).T)]
    return S[:, 0], gw, err, np.cumsum(step)


def node_residuals(prior, al):
    """World position and Procrustes residual of every graph node."""
    gt = np.loadtxt(ROOT / 'dataset' / 'gt' / prior / 'ground_truth.csv', delimiter=',')
    tg = gt[:, 0] / 1e9
    qw, qx, qy, qz = gt[:, 4], gt[:, 5], gt[:, 6], gt[:, 7]
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))
    xg = gt[:, 1] + eg.LEVER_FWD_M * np.cos(yaw)
    yg = gt[:, 2] + eg.LEVER_FWD_M * np.sin(yaw)
    nodes = eg._nodes_with_stamp(prior)
    P = np.array([[n[2], n[3]] for n in nodes])
    st = [n[1] for n in nodes]
    Q = np.column_stack([np.interp(st, tg, xg), np.interp(st, tg, yg)])
    W = es.map_to_world(al, P)
    return Q, np.linalg.norm(W - Q, axis=1)


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def _qyaw(qw, qx, qy, qz):
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))


def node_headings(prior, al):
    """World position and signed heading error [deg] of every graph node: node
    yaw carried into the world by the body-to-body fit, minus the mapping run's
    ground-truth yaw at the node's stamp."""
    gt = np.loadtxt(ROOT / 'dataset' / 'gt' / prior / 'ground_truth.csv', delimiter=',')
    tg = gt[:, 0] / 1e9
    gyaw = np.unwrap(_qyaw(gt[:, 4], gt[:, 5], gt[:, 6], gt[:, 7]))
    xg = gt[:, 1] + eg.LEVER_FWD_M * np.cos(gyaw)
    yg = gt[:, 2] + eg.LEVER_FWD_M * np.sin(gyaw)
    db = sqlite3.connect(f"file:{ROOT / 'map' / prior / (prior + '.db')}?mode=ro", uri=True)
    N = []
    for _nid, st, blob in db.execute('select id, stamp, pose from Node order by id'):
        if blob and len(blob) >= 48:
            m = struct.unpack('12f', blob[:48])
            N.append((st, math.degrees(math.atan2(m[4], m[0]))))
    db.close()
    N = np.array(N)
    err = _wrap(N[:, 1] + al['yaw_deg'] - np.degrees(np.interp(N[:, 0], tg, gyaw)))
    return np.column_stack([np.interp(N[:, 0], tg, xg), np.interp(N[:, 0], tg, yg)]), err


def heading_series(prior, world, path, al, crit, nxy, nherr, R=MAIN_R):
    """Signed localised heading error [deg] after the first fix, minus that
    first sample (the correction still settling, as the heading figure does),
    and the median signed node heading error within R of the true position."""
    d = ROOT / 'map' / prior / 'exp4' / f'{world}_{path}_{REP}'
    rows = list(csv.DictReader(open(d / 'robot_gt.csv')))
    t = np.array([float(r['stamp']) for r in rows])
    gxy = np.array([[float(r['x']), float(r['y'])] for r in rows])
    gy = np.unwrap(_qyaw(*(np.array([float(r[k]) for r in rows]) for k in ('qw', 'qx', 'qy', 'qz'))))
    rec = list(csv.DictReader(open(ROOT / 'map' / prior / f'localize_exp4_{world}_{path}_{REP}.csv')))
    first_fix = next(float(r['stamp']) for r in rec if r['kind'] == 'pose' and r.get('cov_trace')
                     and float(r['cov_trace']) < crit['cov_trace_blind'])
    S = np.array([(float(r['stamp']), float(r['yaw_deg'])) for r in rec
                  if r['kind'] == 'tf' and r.get('x') and float(r['stamp']) >= first_fix])
    S = S[(S[:, 0] >= t[0]) & (S[:, 0] <= t[-1])][1:]
    herr = _wrap(S[:, 1] + al['yaw_deg'] - np.degrees(np.interp(S[:, 0], t, gy)))
    pw = np.column_stack([np.interp(S[:, 0], t, gxy[:, 0]), np.interp(S[:, 0], t, gxy[:, 1])])
    near = []
    for p in pw:
        dn = np.hypot(*(nxy - p).T)
        m = dn <= R
        near.append(np.median(nherr[m]) if m.any() else nherr[np.argmin(dn)])
    return herr, np.array(near)


def local_quality(s, scene, pos, R, nodes_xy, nodes_res):
    res = scene.res
    k = int(math.ceil(R / res))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    disc = (xx ** 2 + yy ** 2) * res * res <= R * R
    occ, cor, bad = s['M_occ'], s['correct'], s['thick'] | s['phantom']
    seen, missed = s['seen'], s['missed']
    out = []
    for p in pos:
        c, r = np.round((p - scene.lo) / res).astype(int)
        r0, r1, c0, c1 = r - k, r + k + 1, c - k, c + k + 1
        if r0 < 0 or c0 < 0 or r1 > scene.gh or c1 > scene.gw:
            out.append((np.nan,) * 5)
            continue
        sl = (slice(r0, r1), slice(c0, c1))
        n_occ = (occ[sl] & disc).sum()
        n_seen = (seen[sl] & disc).sum()
        prec = (cor[sl] & disc).sum() / n_occ if n_occ >= 20 else np.nan
        badf = (bad[sl] & disc).sum() / n_occ if n_occ >= 20 else np.nan
        rec = 1 - (missed[sl] & disc).sum() / n_seen if n_seen >= 20 else np.nan
        dn = np.linalg.norm(nodes_xy - p, axis=1)
        m = dn <= R
        nr = float(np.median(nodes_res[m])) if m.any() else float(nodes_res[np.argmin(dn)])
        out.append((prec, rec, badf, nr, n_occ))
    return np.array(out, float)


def pooled_within_rho(runs, key):
    """Spearman on within-run ranks: each run's x and y ranked inside the run
    (fractional ranks), pooled. Removes run-level offsets (layout, route length)."""
    xs, ys = [], []
    for r in runs:
        x, y = r[key], r['err']
        ok = np.isfinite(x)
        if ok.sum() < 10:
            continue
        xs.append((rankdata(x[ok]) - 0.5) / ok.sum())
        ys.append((rankdata(y[ok]) - 0.5) / ok.sum())
    return float(np.corrcoef(np.concatenate(xs), np.concatenate(ys))[0, 1])


def shift_test(runs, key, n=5000):
    obs = pooled_within_rho(runs, key)
    null = np.empty(n)
    for i in range(n):
        sh = []
        for r in runs:
            L = len(r['err'])
            lag = RNG.integers(int(0.1 * L), int(0.9 * L) + 1)
            sh.append({key: r[key], 'err': np.roll(r['err'], lag)})
        null[i] = pooled_within_rho(sh, key)
    # two-sided
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n + 1)
    return obs, p, np.percentile(null, [2.5, 97.5])


def low_high(runs, key, low_is_bad=True, n=4000, block=20):
    """Median error where quality is in the worse half vs the better half (pooled
    median split), bootstrap CI by moving blocks within each run."""
    allx = np.concatenate([r[key][np.isfinite(r[key])] for r in runs])
    thr = np.median(allx)

    def stat(pairs):
        x = np.concatenate([p[0] for p in pairs]); e = np.concatenate([p[1] for p in pairs])
        worse = x < thr if low_is_bad else x > thr
        if worse.sum() < 5 or (~worse).sum() < 5:
            return np.nan
        return float(np.median(e[worse]) - np.median(e[~worse]))

    base = [(r[key][np.isfinite(r[key])], r['err'][np.isfinite(r[key])]) for r in runs]
    obs = stat(base)
    boots = []
    for _ in range(n):
        res = []
        for x, e in base:
            L = len(x)
            nb = int(math.ceil(L / block))
            st = RNG.integers(0, max(L - block, 1), nb)
            idx = np.concatenate([np.arange(s, min(s + block, L)) for s in st])[:L]
            res.append((x[idx], e[idx]))
        boots.append(stat(res))
    boots = np.array(boots)
    worse = np.concatenate([b[0] for b in base])
    w = worse < thr if low_is_bad else worse > thr
    e = np.concatenate([b[1] for b in base])
    return dict(thr=thr, obs=obs, ci=np.nanpercentile(boots, [2.5, 97.5]),
                med_worse=float(np.median(e[w])), med_better=float(np.median(e[~w])),
                n_worse=int(w.sum()), n_better=int((~w).sum()))


def write_table(runs, pooled, path=TABLE):
    """Per-run medians of the node error within MAIN_R of the true position and
    of the localisation error, their ratio, and the within-run Spearman rho
    between the two. The localisation medians must equal tab:exp4-tracking."""
    k = f'node{MAIN_R:g}'
    rows, ratios, nodes = [], [], []
    for r in runs:
        x, e = r[k], r['err']
        ok = np.isfinite(x)
        node, loc = float(np.median(x[ok])), float(np.median(e))
        rho = spearmanr(x[ok], e[ok])[0]
        lay, shape = r['name'].split(' ', 1)
        hn, hl = r['hnode'], r['herr']
        rows.append(f'{lay} & {shape:8} & {len(e):3d} & {node:.3f} & {loc:.3f} & '
                    f'{loc / node:.2f} & ${rho:+.2f}$ & '
                    f'{np.median(np.abs(hn)):.2f} (${hn.mean():+.2f}$) & '
                    f'{np.median(np.abs(hl)):.2f} (${hl.mean():+.2f}$) \\\\')
        ratios.append(loc / node); nodes.append(node)
    cap = (r"""Localisation error against the prior's own node error along each
reported route, in position and heading. Node error is the median position residual of the prior's
graph nodes within \SI{%g}{\metre} of the robot's true position, after the
body-to-body fit, taken over the run's samples; localisation error is the median
of the same samples (\cref{tab:exp4-tracking}). The ratio of the two medians is
%.2f--%.2f. $\rho$ is the within-run Spearman correlation between the two
series. Pooled over the six runs it is $%+.2f$, $p = %.2f$ under a
circular-shift permutation test; the samples of a run are strongly
autocorrelated, so a single run's $\rho$ is not evidence on its own. Heading
is compared the same way: node heading is the prior's node yaw carried into
the world by the same fit, against the mapping run's ground truth; each cell
gives the median absolute error and, in brackets, the signed mean. The first
localised sample after the first fix is excluded from heading, as in
\cref{fig:exp4-heading}."""
           % (MAIN_R, min(ratios), max(ratios), pooled[0], pooled[1]))
    with open(path, 'w') as f:
        f.write('% Generated by script/exp2_mapquality_vs_localisation.py -- do not edit.\n')
        f.write('\\begin{table}[htbp]\n\\centering\n')
        f.write('\\caption[Localisation error against the prior\'s node error along each route, in position and heading.]{'
                + cap + '}\n\\label{tab:exp4-map-bound}\n\\small\n\\setlength{\\tabcolsep}{4pt}\n')
        f.write('\\begin{tabular}{llrrrrrrr}\n\\toprule\n')
        f.write(' & & & \\multicolumn{4}{c}{Position} & \\multicolumn{2}{c}{Heading} \\\\\n'
                '\\cmidrule(lr){4-7}\\cmidrule(lr){8-9}\n')
        f.write('Layout & Route & Samples & Node & Localised & Ratio & $\\rho$ & Node & Localised \\\\\n')
        f.write(' & & & [\\si{\\metre}] & [\\si{\\metre}] & & & [\\si{\\degree}] & [\\si{\\degree}] \\\\\n\\midrule\n')
        f.write('\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')
    print(f'table: {path}  node {min(nodes):.3f}-{max(nodes):.3f} m, ratio {min(ratios):.2f}-{max(ratios):.2f}')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    spec = eg.load_protocol() if hasattr(eg, 'load_protocol') else None
    import yaml
    proto = yaml.safe_load(open(ROOT / 'script' / 'exp4_protocol.yaml'))
    crit = proto['criteria']
    tmp = tempfile.mkdtemp(prefix='mq_loc.')
    runs = []
    for lay, L in LAYOUTS.items():
        prior = L['prior']
        sp = next(x for x in ei.REALISTIC if x['name'] == prior)
        gtp = load_gt(str(sp['gt']))
        scene = ei.Scene(sp['world'], 0.05, gtp[:, :2])
        s = ei.score_map(prior, scene, gtp, tmp)
        al = eg._align(prior)
        nxy, nres = node_residuals(prior, al)
        hxy, hnode = node_headings(prior, al)
        print(f'{lay}: map P {100*s["precision"]:.1f} %  node residual median {np.median(nres):.3f} m '
              f'(p90 {np.percentile(nres, 90):.3f}, max {nres.max():.3f})')
        for path in PATHS:
            t, gw, err, dist = run_series(prior, L['world'], path, al, crit)
            run = dict(name=f'{lay} {SHAPE[path]}', lay=lay, t=t, err=err, dist=dist, pos=gw)
            run['herr'], run['hnode'] = heading_series(prior, L['world'], path, al, crit, hxy, hnode)
            for R in RADII:
                q = local_quality(s, scene, gw, R, nxy, nres)
                run[f'prec{R:g}'], run[f'rec{R:g}'], run[f'bad{R:g}'] = q[:, 0], q[:, 1], q[:, 2]
                run[f'node{R:g}'], run[f'nocc{R:g}'] = q[:, 3], q[:, 4]
            runs.append(run)

    with open(OUT / 'samples.csv', 'w', newline='') as f:
        w = csv.writer(f)
        keys = [k for k in runs[0] if re.fullmatch(r'(prec|rec|bad|node|nocc)[0-9.]+', k)]
        w.writerow(['run', 't', 'x', 'y', 'dist_m', 'err_m'] + keys)
        for r in runs:
            for i in range(len(r['t'])):
                w.writerow([r['name'], f"{r['t'][i]:.3f}", f"{r['pos'][i,0]:.3f}", f"{r['pos'][i,1]:.3f}",
                            f"{r['dist'][i]:.2f}", f"{r['err'][i]:.4f}"] + [f'{r[k][i]:.4f}' for k in keys])

    print('\nPer run (R = 3 m): samples, median error, local quality median [min-max], Spearman rho(quality, error)')
    for r in runs:
        line = f'  {r["name"]:12} n={len(r["err"]):4d} err med {np.median(r["err"]):.3f} m | '
        for key, lab in (('prec3', 'P'), ('rec3', 'R'), ('bad3', 'bad'), ('node3', 'node')):
            x = r[key]; ok = np.isfinite(x)
            rho = spearmanr(x[ok], r['err'][ok])[0] if ok.sum() > 10 else np.nan
            line += f'{lab} {np.nanmedian(x):.2f} [{np.nanmin(x):.2f}-{np.nanmax(x):.2f}] rho {rho:+.2f} | '
        print(line)

    print('\nPooled within-run Spearman, circular-shift permutation (5000), two-sided')
    pooled = None
    for R in RADII:
        for key, lab, lowbad in (('prec', 'precision', True), ('rec', 'recall', True),
                                 ('bad', 'thick+phantom frac', False), ('node', 'node residual', False)):
            k = f'{key}{R:g}'
            obs, p, nci = shift_test(runs, k)
            if k == f'node{MAIN_R:g}':
                pooled = (obs, p)
            signs = []
            for r in runs:
                x = r[k]; ok = np.isfinite(x)
                signs.append(spearmanr(x[ok], r['err'][ok])[0] if ok.sum() > 10 else np.nan)
            signs = np.array(signs)
            # "worse quality goes with larger error" direction
            want = -1 if lowbad else 1
            agree = int(np.sum(np.sign(signs) == want))
            print(f'  R={R:g} m  {lab:20} rho {obs:+.3f}  p={p:.4f}  null95 [{nci[0]:+.3f},{nci[1]:+.3f}]  '
                  f'runs in the "worse map -> larger error" direction: {agree}/6')

    print('\nWorse half vs better half of local quality (R = 3 m), median error, block-bootstrap 95 % CI')
    for key, lab, lowbad in (('prec3', 'precision', True), ('rec3', 'recall', True),
                             ('bad3', 'thick+phantom frac', False), ('node3', 'node residual', False)):
        d = low_high(runs, key, lowbad)
        print(f'  {lab:20} split at {d["thr"]:.3f}: worse {d["med_worse"]:.3f} m (n={d["n_worse"]}) vs better '
              f'{d["med_better"]:.3f} m (n={d["n_better"]})  diff {d["obs"]:+.3f} [{d["ci"][0]:+.3f}, {d["ci"][1]:+.3f}]')
    print(f'\nper-sample CSV: {OUT / "samples.csv"}')
    write_table(runs, pooled)


if __name__ == '__main__':
    main()
