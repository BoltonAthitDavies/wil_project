#!/usr/bin/env python3
"""Experiment 4, simulation slot: score the three realistic-warehouse RTAB-Map
databases for geometry and for navigability.

    python3 script/eval_realistic_maps.py --print-commands
    python3 script/eval_realistic_maps.py --fit          # step 1, placement
    python3 script/eval_realistic_maps.py                # step 2, scoring
    python3 script/eval_realistic_maps.py --table        # step 3, LaTeX + CSV

RESEARCH QUESTION
    The tiny warehouse is the only map Experiment 4 has scored. Two databases
    now exist for the realistic worlds, one per layout, which are four times
    the area and carry a different obstacle layout. Do they reconstruct
    geometry as well, and could a planner drive them?

HYPOTHESIS (H6), FALSIFIABLE, THRESHOLDS FIXED BEFORE SCORING
    H6: the realistic maps match the tiny map's quality, which was occupied-cell
    precision 96.6 %, observed-surface recall 89.5 %, and 0.0 % false obstacles
    under the true footprint.

    H6 is falsified for a map if ANY of these fails:
        precision          < 90 %
        recall, observed   < 85 %
        false obstacle     > 1 % of probe poses
    A false-obstacle rate above 1 % is the one that matters operationally: it
    means a planner would refuse a route the robot has already driven.

    Whole-building recall is NOT in the criterion. On the tiny map it was 67.2 %
    and that was route coverage, not mapping error -- the run never entered the
    side aisles. It is reported, not tested.

THE INDEPENDENT-ROUTE RULE, WHICH IS THE POINT OF THE DESIGN
    Drivability MUST be scored against a route the map did not see built. Scoring
    it against the build run asks whether the map contains the obstacles it was
    made from, which it does by construction; that circularity already cost this
    project one retracted drivability claim. Each map is therefore probed with a
    different recording through the same static world:

        map_realistic      U world  <- probe dataset_allsensor_realistic_000
        map_realistic_01   I world  <- probe dataset_allsensor_realistic_001

    The worlds are static, so a route driven through one at any time is a valid
    probe. This is a geometry and navigability test, never a trajectory-error
    measurement.

WHAT IS NOT CONTROLLED, AND SO WHAT MAY NOT BE CONCLUDED
    The two databases differ in world, route, duration and path length at the
    same time. This is a characterisation of two maps, n = 1 per layout, not a
    controlled comparison, and no factor may be credited for a difference
    between them.

PLACEMENT IS MEASURED, NOT GUESSED
    world->map is a constant per database and must be fitted with
    fit_map_pose.py before anything is scored; a wrong placement makes every
    number below meaningless while still looking plausible. The realistic worlds
    spawn at (5.250, 19.995, -90), NOT at the tiny map's (1.8, 9.0, -90) default,
    so --spawn must be passed. A fit whose agreement margin over the best rival
    more than 5 deg away is thin should be rejected, not rounded off.

DIAGNOSTICS READ STRAIGHT FROM THE DATABASES (no run needed)
    Reported beside the metrics because they bound what the metrics can mean:
    node, link and loop-closure counts, session count, stamp span, and the
    optimised path length. See the notes printed by --inspect.
"""

import argparse
import csv
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORLDS = os.path.join(ROOT, 'aws-robomaker-small-warehouse-world/worlds/'
                            'small_warehouse_static')
OUT = os.path.join(ROOT, 'output/compare/simulation/exp4-realistic-maps')

# The realistic worlds spawn here; fit_map_pose's default is the tiny map's.
SPAWN = '5.250,19.995,-90'

# Pass criteria, fixed before any map was scored. See H6.
PASS = dict(precision=0.90, recall_seen=0.85, false_obstacle=0.01)

# (key, database, world, independent probe GT, build bag on the flash drive)
MAPS = [
    # The database named map_realistic is the build formerly held as
    # map_realistic_v2 (165 nodes, 417 MB, stamps 13.2-299 s); it was renamed
    # on 26 October 2026 and the earlier 114-node map_realistic was deleted.
    # Its artefacts are kept under exp4-realistic-maps/superseded/.
    dict(key='map_realistic',
         db=os.path.join(ROOT, 'map/map_realistic/map_realistic.db'),
         world=os.path.join(WORLDS,
                            'small_warehouse_static_realistic_nofloortexture.world'),
         probe=os.path.join(ROOT, 'output/output_vins/simulation/'
                                  'dataset_allsensor_realistic_000/ground_truth.csv'),
         probe_name='dataset_allsensor_realistic_000',
         build_bag='/media/ambushee/32E4AAB1E4AA772F/dataset_allsensor_realistic_mapv2'),
    dict(key='map_realistic_01',
         db=os.path.join(ROOT, 'map/map_realistic_01/map_realistic_01.db'),
         world=os.path.join(WORLDS,
                            'small_warehouse_static_realistic_nofloortexture_01.world'),
         probe=os.path.join(ROOT, 'output/output_vins/simulation/'
                                  'dataset_allsensor_realistic_001/ground_truth.csv'),
         probe_name='dataset_allsensor_realistic_001',
         build_bag='/media/ambushee/32E4AAB1E4AA772F/dataset_allsensor_realistic_01_map'),
]

# RTAB-Map Link::Type. Only 1-5 are closures; 0 is the odometry chain and 9 is
# the IMU gravity prior, which is added once per node and is not a constraint
# between places.
LINK_TYPE = {0: 'neighbour', 1: 'global closure', 2: 'local closure (space)',
             3: 'local closure (time)', 4: 'user closure', 5: 'virtual closure',
             6: 'neighbour merged', 7: 'pose prior', 8: 'landmark', 9: 'gravity'}


def placement_path(key):
    return os.path.join(OUT, '%s.mappose' % key)


def inspect(m):
    """Counts and spans straight out of the database. Read-only."""
    import sqlite3
    import struct

    import numpy as np
    uri = 'file:%s?mode=ro' % m['db']
    c = sqlite3.connect(uri, uri=True)
    nodes = list(c.execute('SELECT id,stamp,map_id FROM Node ORDER BY id'))
    links = dict(c.execute('SELECT type,COUNT(*) FROM Link GROUP BY type'))
    closures = [(f, t) for f, t, _ in
                c.execute('SELECT from_id,to_id,type FROM Link WHERE type BETWEEN 1 AND 5')]
    P = []
    for (pose,) in c.execute('SELECT pose FROM Node ORDER BY id'):
        if pose and len(pose) >= 48:
            v = struct.unpack('12f', pose[:48])
            P.append((v[3], v[7], v[11]))
    P = np.asarray(P)
    path = float(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1).sum()) if len(P) > 1 else 0.0
    spans = sorted(abs(f - t) for f, t in closures)
    return dict(
        key=m['key'], nodes=len(nodes),
        sessions=len({n[2] for n in nodes}),
        stamp_first=nodes[0][1], stamp_last=nodes[-1][1],
        duration_s=nodes[-1][1] - nodes[0][1],
        words=list(c.execute('SELECT COUNT(*) FROM Word'))[0][0],
        links_total=sum(links.values()), links_by_type=links,
        closures=len(closures), closure_spans=spans,
        # A closure between adjacent keyframes closes no loop. Separating them
        # matters: a database can report "closures" and still have had no place
        # recognised twice, which is the only kind that corrects drift.
        closures_spanning=sum(1 for s in spans if s > 3),
        path_m=path, db_mb=os.path.getsize(m['db']) / 1e6)


def do_inspect(args):
    rows = [inspect(m) for m in MAPS if os.path.exists(m['db'])]
    os.makedirs(OUT, exist_ok=True)
    for r in rows:
        print('=== %s' % r['key'])
        print('   %d nodes, %d sessions, %.1f s, %.1f m optimised path, %.0f MB'
              % (r['nodes'], r['sessions'], r['duration_s'], r['path_m'], r['db_mb']))
        print('   stamps %.2f -> %.2f s   %d visual words'
              % (r['stamp_first'], r['stamp_last'], r['words']))
        print('   links %d: %s' % (r['links_total'], ', '.join(
            '%s %d' % (LINK_TYPE.get(k, k), v) for k, v in sorted(r['links_by_type'].items()))))
        print('   closures %d, of which spanning >3 nodes: %d   spans=%s'
              % (r['closures'], r['closures_spanning'], r['closure_spans']))
        if r['closures'] and not r['closures_spanning']:
            print('   NOTE every closure is between neighbouring keyframes, so no '
                  'place was\n        recognised twice and the graph carries no '
                  'drift correction. The map is\n        an odometry '
                  'reconstruction and inherits that odometry\'s error.')
    with open(os.path.join(OUT, 'map_inventory.json'), 'w') as fh:
        json.dump(rows, fh, indent=2, sort_keys=True, default=str)
    print('\nwrote %s' % os.path.relpath(os.path.join(OUT, 'map_inventory.json'), ROOT))


def print_commands(args):
    print('# Section 7: the user runs these. Three steps, in order.\n')
    print('cd %s\n' % ROOT)
    print('# ---- step 1: measure where each database sits in its world.')
    print('#      --spawn matters: the realistic worlds spawn at %s, not at the' % SPAWN)
    print('#      tiny map default. Check the printed agreement margin before going on.')
    for m in MAPS:
        print('python3 script/fit_map_pose.py %s \\\n'
              '  --world %s \\\n'
              '  --spawn %s --overlay %s/fit_%s.png'
              % (os.path.relpath(m['db'], ROOT), os.path.relpath(m['world'], ROOT),
                 SPAWN, os.path.relpath(OUT, ROOT), m['key']))
    print('\n# Record each printed "--map-pose X,Y,YAW" into its own file:')
    for m in MAPS:
        print("echo '<paste the X,Y,YAW printed above>' > %s"
              % os.path.relpath(placement_path(m['key']), ROOT))
    print('\n# ---- step 2: score geometry and drivability.')
    print('python3 script/eval_realistic_maps.py')
    print('\n# ---- step 3: build the CSV and the LaTeX table.')
    print('python3 script/eval_realistic_maps.py --table')
    print('\n# ---- optional diagnostic: the trajectory the map was built on.')
    print('#      Only needed if a map fails H6 and the cause has to be placed on')
    print('#      the odometry rather than on the mapping. The bags are on the')
    print('#      flash drive and are large; extraction is a few minutes each.')
    for m in MAPS:
        print('python3 script/extract_gt.py %s/$(basename %s)_0.db3 \\\n'
              '  %s/gt_build_%s.csv'
              % (m['build_bag'], m['build_bag'], os.path.relpath(OUT, ROOT), m['key']))


def do_score(args):
    os.makedirs(OUT, exist_ok=True)
    missing = []
    for m in MAPS:
        if not os.path.exists(placement_path(m['key'])):
            missing.append(m['key'])
    if missing:
        raise SystemExit(
            'no fitted placement for: %s\n'
            'Run step 1 first (--print-commands). Scoring with a guessed '
            'placement produces plausible numbers that mean nothing.'
            % ', '.join(missing))

    for m in MAPS:
        pose = open(placement_path(m['key'])).read().strip()
        png = os.path.join(OUT, 'validate_%s.png' % m['key'])
        js = os.path.join(OUT, 'validate_%s.json' % m['key'])
        cmd = [sys.executable, os.path.join(ROOT, 'script/validate_map.py'), m['db'],
               '--world', m['world'], '--gt', m['probe'],
               '--map-pose', pose, '--out', png, '--json', js]
        print('\n=== %s   (probe route: %s)' % (m['key'], m['probe_name']))
        print('    ' + ' '.join(cmd))
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode:
            print('    FAILED, exit %d -- reported as a failed scoring, not skipped'
                  % r.returncode)


def do_table(args):
    ipath = os.path.join(OUT, 'map_inventory.json')
    inv = ({r['key']: r for r in json.load(open(ipath))}
           if os.path.exists(ipath) else {})
    rows = []
    for m in MAPS:
        js = os.path.join(OUT, 'validate_%s.json' % m['key'])
        if not os.path.exists(js):
            print('MISSING %s -- reported as unscored, not averaged over' % m['key'])
            continue
        d = json.load(open(js))
        d['key'] = m['key']
        d['probe'] = m['probe_name']
        d['world'] = os.path.basename(m['world'])
        v = inv.get(m['key'], {})
        d['nodes'] = v.get('nodes')
        d['closures_spanning'] = v.get('closures_spanning')
        d['verdict'] = ('pass' if (d['precision'] >= PASS['precision']
                                   and d['recall_seen'] >= PASS['recall_seen']
                                   and d['false_obstacle'] <= PASS['false_obstacle'])
                        else 'FAILS H6')
        rows.append(d)
    if not rows:
        raise SystemExit('nothing scored yet; run step 2')

    cols = ['key', 'world', 'probe', 'nodes', 'closures_spanning', 'map_pose',
            'occupied_cells', 'gt_poses', 'precision', 'recall_seen',
            'recall_whole', 'drivable', 'false_obstacle', 'unmapped',
            'clearance_median_m', 'clearance_p05_m', 'verdict']
    cpath = os.path.join(OUT, 'realistic_map_metrics.csv')
    with open(cpath, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print('wrote %s' % os.path.relpath(cpath, ROOT))

    tpath = os.path.join(OUT, 'realistic_map_quality.tex')
    with open(tpath, 'w') as f:
        f.write('% Generated by script/eval_realistic_maps.py -- do not edit by hand.\n')
        f.write('\\begin{tabular}{lrrrrrr}\n\\toprule\n')
        f.write('Database & Nodes & Precision & Recall & Recall & False & '
                'Clearance \\\\\n')
        f.write(' & & & observed & building & obstacle & median \\\\\n')
        f.write(' & & [\\si{\\percent}] & [\\si{\\percent}] & [\\si{\\percent}] '
                '& [\\si{\\percent}] & [\\si{\\metre}] \\\\\n\\midrule\n')
        for d in rows:
            f.write('\\repo{%s} & %s & %.1f & %.1f & %.1f & %.1f & %.2f \\\\\n'
                    % (d['key'],
                       d['nodes'] if d['nodes'] else '--',
                       100 * d['precision'], 100 * d['recall_seen'],
                       100 * d['recall_whole'], 100 * d['false_obstacle'],
                       d['clearance_median_m']))
        f.write('\\bottomrule\n\\end{tabular}\n')
    print('wrote %s' % os.path.relpath(tpath, ROOT))

    print('\n%-20s %9s %9s %9s %9s  %s'
          % ('database', 'prec', 'rec(obs)', 'falseObs', 'clear', 'H6'))
    for d in rows:
        print('%-20s %8.1f%% %8.1f%% %8.2f%% %8.2f m  %s'
              % (d['key'], 100 * d['precision'], 100 * d['recall_seen'],
                 100 * d['false_obstacle'], d['clearance_median_m'], d['verdict']))
    bad = [d['key'] for d in rows if d['verdict'] != 'pass']
    print('\nH6 %s' % ('holds for all three maps' if not bad
                       else 'FALSIFIED for: ' + ', '.join(bad)))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--print-commands', action='store_true')
    ap.add_argument('--inspect', action='store_true',
                    help='database counts only; reads nothing but the .db files')
    ap.add_argument('--table', action='store_true')
    a = ap.parse_args()
    if a.print_commands:
        return print_commands(a)
    if a.inspect:
        return do_inspect(a)
    if a.table:
        return do_table(a)
    return do_score(a)


if __name__ == '__main__':
    sys.exit(main())
