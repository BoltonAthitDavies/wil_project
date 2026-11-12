#!/usr/bin/env python3
"""Controlled sweep of RTAB-Map's localisation ACCEPTANCE settings, for the
simulation aisle-warping problem.

    python3 script/aisle_sweep.py --list
    python3 script/aisle_sweep.py --run B          # launches; you drive; Ctrl-C
    python3 script/aisle_sweep.py --analyse

WHY THIS EXISTS
    Five runs were made on 11-12 Nov 2026 and none of them can be compared with
    another. aisle_03 and aisle_04 share a run_name prefix but the launch file was
    edited between them (RGBD/MaxOdomCacheSize changed under the runs, file mtime
    08:52:56, one minute after aisle_04 finished). aisle_05 changed TWO settings at
    once. The route is driven by hand, so it differs every time. Nothing here is
    reportable.

    This script fixes the provenance half of that: every run writes a manifest
    beside its data recording the exact settings, the command, and the git commit,
    so a result can never again be attributed to the wrong configuration. It
    cannot fix the route; see DRIVING below.

THE QUESTION
    Which acceptance setting rejects a false match in a repeating aisle WITHOUT
    also rejecting the true ones? Measured so far:

        settings                        corrections/min   no-information poses
        defaults                             (warps)      --
        optimize_max_error 3.0  (aisle_03)       11.9     1 of 108
        optimize_max_error 3.0  (aisle_04)        2.8     45 of 94   + a 6.1 m warp
        inliers 35 + distribution 0.08            0.9     81 of 81   (nothing localised)

    So the thresholds that stop the warps also stop localisation. The sweep walks
    one factor at a time from the defaults to find the separation, if one exists.

DRIVING
    The route is the uncontrolled variable and this script cannot hold it. Drive
    the SAME route every time: out of the spawn, down the aisle where the warp
    appeared, round, and back. Keep each run over 90 s. A run whose path length
    differs by more than ~20 % from the others is not comparable and --analyse
    flags it.
"""
import argparse, json, os, shlex, subprocess, sys, time, math, csv

WORKSPACE = os.path.expanduser('~/wil_project')
MAP = 'dataset_allsensor_realistic_map'
WORLD = os.path.join(
    WORKSPACE, 'aws-robomaker-small-warehouse-world', 'worlds', 'small_warehouse',
    'small_warehouse_static_realistic_nofloortexture.world')
SEED = '-7.839 5.875 0 0 0 1.5696'

# One factor changed from the defaults per variant. The defaults ARE variant A,
# so every other row differs from A in exactly one argument.
DEFAULTS = {'optimize_max_error': '10.0', 'min_inliers': '20',
            'min_inliers_distribution': '0.0', 'odom_cache_size': '10',
            'geometric_check': 'false', 'estimation_type': '1',
            'vis_inlier_distance': '0.1', 'localize_delay_s': '0.0'}
VARIANTS = {
    'A': ({}, 'defaults; the baseline that warps'),
    'B': ({'optimize_max_error': '3.0'}, 'stricter graph-consistency test'),
    'C': ({'min_inliers_distribution': '0.03'}, 'inlier spread, gentle'),
    'D': ({'min_inliers_distribution': '0.05'}, 'inlier spread, moderate'),
    'E': ({'odom_cache_size': '30'}, 'anti-teleport context, physical not visual'),
    'F': ({'min_inliers': '28'}, 'stronger geometric match'),
    # G and H change TWO settings, deliberately, and the one-factor runs are what
    # licenses it. E (cache 30, error 10.0) changed nothing at all: 10 warps against
    # the baseline's 8. The odometry cache is not itself an acceptance test -- it is
    # the evidence the graph-consistency test is run against, and at
    # optimize_max_error 10.0 that test accepts almost anything however much
    # evidence it has. So cache size can only act THROUGH the error threshold, and
    # a cache variant with the threshold left loose was never a test of anything.
    # B (error 3.0, cache 10) removed every warp but took 165 s to acquire its
    # first fix. aisle_03 (error 3.0, cache 20) had both: 0 warps and 11.9
    # corrections/min from the start. These two re-test that at fixed provenance.
    'G': ({'optimize_max_error': '3.0', 'odom_cache_size': '20'},
          'graph-consistency test WITH enough odometry to run it on'),
    'H': ({'optimize_max_error': '3.0', 'odom_cache_size': '30'},
          'same, more context: does acquisition get slower or faster?'),
    # I is one factor from the defaults again, and it is the first variant to test
    # anything ABSOLUTE. A through H all tested either appearance (how good the 2-D
    # match looks) or the graph residual (how well it fits recent odometry), and a
    # rigid misplacement of the whole odometry window is invisible to both. Reg/
    # Strategy was 0 = Vis for every one of those twelve runs, so the depth stored
    # against all 148 prior nodes was never consulted. The four false fixes we
    # could measure were 73, 143, 23 and 24 deg of yaw, none of them a symmetry of
    # an aisle grid, so the structure cannot agree with them.
    'I': ({'geometric_check': 'true'}, 'ICP verification against the 3-D structure'),
    # I localised NOTHING in two runs, and not because ICP was too strict: RTAB-Map
    # ICP registers Data.scan and every node of this prior has image, depth and
    # calibration but no scan, so there was nothing to register. I is now refused
    # by the launch until the prior is reprocessed. J is the same idea without that
    # dependency -- 3D->3D registration makes the matched feature POSITIONS agree
    # in metres instead of merely reprojecting consistently, and the feature depths
    # are already stored.
    'J': ({'estimation_type': '0'}, 'sparse-geometry veto: 3D->3D, no scan needed'),
    'K': ({'estimation_type': '0', 'vis_inlier_distance': '0.05'},
          'same, half the tolerance'),
    # J's three runs showed the biggest remaining jump is NOT aisle aliasing: J1 and
    # J2 both latched a 73.8 / 73.7 deg wrong correction on the front end's first
    # unstable second and held it ~16 s. L removes the opportunity rather than
    # trying to reject the match.
    'L': ({'localize_delay_s': '5.0'}, 'no localising during the front end first 5 s'),
    'M': ({'estimation_type': '0', 'localize_delay_s': '5.0'},
          'J plus L: sparse-geometry veto AND no early latch'),
}


def run_name(v, rep):
    return f'sweep_{v}{rep}'


def build_cmd(v, rep):
    name = run_name(v, rep)
    args = dict(DEFAULTS)
    args.update(VARIANTS[v][0])
    cmd = ['ros2', 'launch', 'vslam_navigation', 'navigate_sim.launch.py',
           f'map_name:={MAP}', f'run_name:={name}', f'start_pose:={SEED}',
           f'world:={WORLD}',
           f'orb_output_path:={WORKSPACE}/output/output_orb/simulation/{name}']
    cmd += [f'{k}:={val}' for k, val in sorted(args.items())]
    return name, args, cmd


def git_commit():
    try:
        return subprocess.run(['git', '-C', WORKSPACE, 'rev-parse', '--short', 'HEAD'],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return 'unknown'


def do_run(v, rep):
    name, args, cmd = build_cmd(v, rep)
    man = os.path.join(WORKSPACE, 'map', MAP, f'sweep_{name}.json')
    # A leftover from an earlier run does not announce itself in the new one -- it
    # produces quietly wrong data. sweep_M4 localised with covariance 0.0056 and
    # recorded ZERO map->odom transforms over 34 s because a Gazebo from 21
    # minutes earlier was still publishing /clock; two clocks make sim time jump
    # and every TF lookup returns nothing, while the pose topic is unaffected
    # because it is published rather than looked up.
    sys.path.insert(0, os.path.join(WORKSPACE, 'script'))
    import sim_guard
    if not sim_guard.require_clean():
        sys.exit(1)
    if os.path.exists(os.path.join(WORKSPACE, 'map', MAP, f'localize_{name}.csv')):
        sys.exit(f'{name} already exists. Use a different --rep, or delete it deliberately.')
    print(f'variant {v}: {VARIANTS[v][1]}')
    print('  ' + ' \\\n    '.join(cmd) + '\n')
    print('Drive the SAME route as the other variants. Ctrl-C when done.\n')
    started = time.time()
    log = os.path.join(WORKSPACE, 'map', MAP, f'sweep_{name}.log')
    # Through tee so the output is both live (you are watching for the first fix)
    # and kept. ros2 launch's own log directory held only launch.log for the M4
    # run, so the node output that would have shown the clock conflict was gone.
    teed = f'{shlex.join(cmd)} 2>&1 | tee {shlex.quote(log)}'
    print(f'[sweep] output -> {log}\n')
    try:
        subprocess.run(['bash', '-c', teed])
    except KeyboardInterrupt:
        pass
    json.dump({'variant': v, 'run': name, 'note': VARIANTS[v][1], 'settings': args,
               'command': cmd, 'git': git_commit(), 'log': log,
               'started': time.strftime('%F %T', time.localtime(started)),
               'wall_seconds': round(time.time() - started, 1)},
              open(man, 'w'), indent=2)
    print(f'\nmanifest -> {man}')


def _tilt_fraction(name):
    """Fraction of front-end poses that were NOT gravity-aligned. A run with more
    than a few per cent was not a fair test of the back end."""
    p = f'{WORKSPACE}/output/output_orb/simulation/{name}/vio.csv'
    if not os.path.exists(p):
        return None
    try:
        import numpy as np
        d = np.loadtxt(p, delimiter=',')
        tilt = np.degrees(np.arccos(np.clip(1 - 2 * (d[:, 5] ** 2 + d[:, 6] ** 2), -1, 1)))
        return float((tilt > 30).mean())
    except Exception:
        return None


def analyse():
    """Two different failures look identical in a whole-run average, and reading one
    as the other is what made aisle_04 look like a rejection problem:

      ACQUISITION  the run cannot get its FIRST fix. Poses carry no information and
                   the robot dead reckons on the seed. Costly but recoverable --
                   drive further and it may lock on.
      REJECTION    the run localises, but accepts false matches (warps) or stops
                   accepting true ones part-way through.

    So everything below is reported from the first informative pose onwards, with
    the acquisition cost stated separately. A run that never acquires is reported as
    such and has no steady state to judge."""
    mdir = os.path.join(WORKSPACE, 'map', MAP)
    rows = []
    for f in sorted(os.listdir(mdir)):
        if not (f.startswith('sweep_') and f.endswith('.json')):
            continue
        man = json.load(open(os.path.join(mdir, f)))
        name = man['run']
        csvp = os.path.join(mdir, f'localize_{name}.csv')
        if not os.path.exists(csvp):
            continue
        rec = list(csv.DictReader(open(csvp)))
        T = [r for r in rec if r['kind'] == 'tf']
        P = [r for r in rec if r['kind'] == 'pose' and r.get('cov_trace')]
        if len(T) < 10:
            continue
        t0 = float(T[0]['stamp'])
        dur = float(T[-1]['stamp']) - t0

        # acquisition: the first pose that carries information at all
        informative = [float(r['stamp']) - t0 for r in P if float(r['cov_trace']) < 1e4]
        acq = informative[0] if informative else None

        # The robot's map-frame pose jump, which is what a warp IS. corr_step_m is
        # the TRANSLATION of the map->odom step and under-reports exactly the
        # corrections that displace the robot most: in sweep_B2 at t=38.5 s the
        # correction's translation moved 0.49 m while its yaw moved 73 deg and the
        # pose moved 35 m. Every warp figure reported before 12 Nov 2026 was the
        # translation-only one and understated rotational warps.
        # 1 m between 0.5 s samples is already past the robot's ~1.5 m/s, so a step
        # over it is not travel. Jumps inside the first 8 s are the identity ->
        # seed transition (the spawn sits ~9.8 m from the prior's first node), and
        # a jump within 2 s of the first informative pose is the acquisition snap:
        # the drift accumulated while dead reckoning, not a false match.
        jumps, pose_path = [], 0.0
        seq = [(float(r['stamp']) - t0, float(r['x']), float(r['y']))
               for r in T if r.get('x')]
        for i in range(1, len(seq)):
            t, d = seq[i][0], math.dist(seq[i][1:], seq[i - 1][1:])
            if t < 8.0:
                continue
            if d > 1.0:
                snap = acq is not None and abs(t - acq) < 2.0
                if not snap:
                    jumps.append((t, d))
            else:
                pose_path += d

        # distinct corrections; the first is the identity -> seed transition
        hist, prev = [], None
        for r in T:
            cur = (r['corr_x'], r['corr_y'], r['corr_yaw_deg'])
            if cur != prev:
                hist.append((float(r['stamp']) - t0,
                             float(r['corr_step_m']) if r.get('corr_step_m') else 0.0))
                prev = cur
        start = acq if acq is not None else 0.0
        post = [h for h in hist[2:] if h[0] >= start]
        post_dur = max(dur - start, 1e-6)
        warps = [h for h in post if h[1] > 1.0]
        # The size of the accepted corrections matters more than how many were over
        # 1 m, because a warp COUNT divided by near-zero acceptance is meaningless:
        # H looked best in the warp column (1) while accepting only 2 corrections in
        # 76 s, both of them jumps. A healthy localiser corrects continuously and
        # gently -- B's eleven corrections were all under 0.17 m. So report the
        # distribution, and flag a variant that bought its low warp count by not
        # localising.
        sizes = sorted(h[1] for h in post)
        blind_post = [r for r in P if float(r['stamp']) - t0 >= start]
        blind_n = sum(1 for r in blind_post if float(r['cov_trace']) >= 1e4)

        # path driven before and after the first fix; jumps are not travel
        pre_m = post_m = 0.0
        pts = [(float(r['stamp']) - t0, float(r['x']), float(r['y']))
               for r in T if r.get('x')]
        for i in range(1, len(pts)):
            d = math.dist(pts[i - 1][1:], pts[i][1:])
            if d >= 1.0:
                continue
            if acq is not None and pts[i][0] < acq:
                pre_m += d
            else:
                post_m += d
        med_step = sizes[len(sizes) // 2] if sizes else 0.0
        p90_step = sizes[int(0.9 * (len(sizes) - 1))] if sizes else 0.0
        # Per METRE, not per minute. A correction is triggered by travel into new
        # views, so a run driven slowly accepts fewer per minute while being no less
        # well localised: B's 5.7/min was 0.32 per metre against the baseline's
        # 0.24, i.e. MORE often per metre driven, and the per-minute figure had it
        # flagged as acceptance-starved. G and H are starved on either measure.
        rows.append(dict(v=man['variant'], run=name, dur=dur, acq=acq,
                         med=med_step, p90=p90_step,
                         pre=pre_m, path=post_m,
                         rate=10 * len(post) / post_m if post_m > 1 else 0.0,
                         blind=blind_n, n=len(blind_post),
                         warps=len(jumps),
                         worst=max([d for _, d in jumps], default=0.0),
                         jump_path=pose_path,
                         tilt=_tilt_fraction(name),
                         changed=' '.join(f'{k}={v}' for k, v in sorted(man['settings'].items())
                                          if DEFAULTS.get(k) != v) or 'defaults'))
    if not rows:
        sys.exit('no sweep runs found. Use --run first.')
    paths = [r['path'] for r in rows if r['path'] > 0]
    med = sorted(paths)[len(paths) // 2] if paths else 0
    print('  ' + ' ' * 44 + 'acquisition |        after the first fix')
    base = max((r['rate'] for r in rows if r['changed'] == 'defaults'), default=0.0)
    print(f'{"var":>3} {"changed from defaults":34} {"at":>6} {"blind m":>7} | '
          f'{"s":>4} {"path":>6} {"corr/10m":>8} {"med":>5} {"p90":>5} {"worst":>6} '
          f'{"warps":>5} {"fe_bad":>6}')
    for r in sorted(rows, key=lambda r: (r['v'], r['run'])):
        far = '*' if med and abs(r['path'] - med) / med > 0.20 else ' '
        tilt = f"{100*r['tilt']:.0f}%" if r['tilt'] is not None else '  --'
        if r['acq'] is None:
            print(f"{r['v']:>3} {r['changed'][:34]:34} {'NEVER':>6} {r['pre']:7.1f} | "
                  f"{'':>4} {'':>6} {'':>8} {'':>5} {'':>5} {'':>6} {'':>5} {tilt:>6}")
            continue
        starved = '!' if base and r['rate'] < 0.4 * base else ' '
        print(f"{r['v']:>3} {r['changed'][:34]:34} {r['acq']:6.1f} {r['pre']:7.1f} | "
              f"{r['dur']-r['acq']:4.0f} {r['path']:5.0f}{far} {r['rate']:7.1f}{starved} "
              f"{r['med']:5.2f} {r['p90']:5.2f} {r['worst']:6.2f} "
              f"{r['warps']:5d} {tilt:>6}")
    print('\n  at / blind m  when the first informative pose arrived, and how far the')
    print('                robot drove dead-reckoning on the seed before it did')
    print('  corr/10m      corrections accepted per 10 m driven AFTER the first fix;')
    print('                ! marks under 40 % of the baseline rate -- acceptance')
    print('                starved, so its warp count is not a measure of anything.')
    print('                Per metre, not per minute: driving slowly is not a fault.')
    print('  med/p90       size of the accepted corrections; worst is the worst POSE')
    print('                jump. med and p90 are the real signal:')
    print('                a healthy localiser corrects often and gently, so med and')
    print('                p90 both stay well under a metre. A large p90 means the')
    print('                pose is dead reckoning between snaps.')
    print('  warps         POSE jumps over 1 m, excluding the seed transition and the')
    print('                acquisition snap -- what the robot does, not what the')
    print('                correction translation does. Read it ONLY')
    print('                beside corr/10m, never on its own.')
    print('  fe_bad        front-end poses not gravity-aligned; over ~5 % voids the run')
    print('  NEVER         no fix in the whole run: pure dead reckoning, nothing to judge')
    print('  *             path differs from the median by over 20 %: not comparable')
    # --- grouped summary -------------------------------------------------------
    # Added after three repeats of variant B gave 0, 2 and 2 warps with p90
    # corrections of 0.14, 0.91 and 3.58 m at IDENTICAL settings. The run-to-run
    # spread of one configuration is as large as every between-variant difference
    # in the table above, so no single-run comparison in it means anything. This
    # block exists to make that visible instead of letting a lucky run be read as
    # an effect. Warps are per 100 m because the routes differ in length.
    groups = {}
    for r in rows:
        if r['acq'] is None:
            continue
        groups.setdefault(r['changed'], []).append(r)
    multi = [g for g in groups.values() if len(g) > 1]
    print('\n  repeats of one configuration (warps per 100 m driven):')
    print(f'    {"changed from defaults":34} {"n":>2} {"warps/100m":>22} {"p90 (m)":>20} '
          f'{"acquired (s)":>20}')
    for key in sorted(groups):
        g = groups[key]
        w = [100 * r['warps'] / r['jump_path'] if r['jump_path'] > 1 else 0.0
             for r in g]
        fmt = lambda xs, f='.1f': (f'{sum(xs)/len(xs):{f}}  [' +  # noqa: E731
                                   ' '.join(f'{x:{f}}' for x in sorted(xs)) + ']')
        print(f'    {key[:34]:34} {len(g):2d} {fmt(w):>22} '
              f'{fmt([r["p90"] for r in g], ".2f"):>20} '
              f'{fmt([r["acq"] for r in g]):>20}')
    if multi:
        rate = lambda r: 100 * r['warps'] / r['jump_path']  # noqa: E731
        spread = max(max(rate(r) for r in g if r['jump_path'] > 1)
                     - min(rate(r) for r in g if r['jump_path'] > 1)
                     for g in multi)
        print(f'\n  Widest spread within ONE configuration: {spread:.1f} warps/100 m.')
        print('  Any difference between variants smaller than that is not evidence.')

    print('\nA variant counts if its corrections stay small (p90 well under a metre) at a')
    print('rate per metre near the baseline. Acquisition latency is a separate, lesser cost: it is')
    print('paid once, and the robot can be driven through it.')



if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--run', metavar='VARIANT')
    ap.add_argument('--rep', type=int, default=1, help='repeat number (default 1)')
    ap.add_argument('--analyse', action='store_true')
    a = ap.parse_args()
    if a.list:
        for k, (chg, note) in VARIANTS.items():
            s = ' '.join(f'{x}={y}' for x, y in chg.items()) or '(defaults)'
            print(f'  {k}  {s:38} {note}')
    elif a.run:
        if a.run not in VARIANTS:
            sys.exit(f'unknown variant {a.run!r}; try --list')
        do_run(a.run, a.rep)
    elif a.analyse:
        analyse()
    else:
        ap.print_help()
