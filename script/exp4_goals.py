#!/usr/bin/env python3
"""Freeze Experiment 4's goal set: 4 worlds x 2 priors x 3 paths.

    python3 script/exp4_goals.py --show        # inspect the selection
    python3 script/exp4_goals.py --freeze      # write script/exp4_protocol.yaml
    python3 script/exp4_goals.py --preflight   # checks that must pass before a run

WHY GOALS ARE NODE IDS, NOT COORDINATES
    Nav2 plans in the map frame, and the map frame's relationship to the Gazebo
    world frame is not something this repository records anywhere. The priors'
    first node sits at (0.09, 0.25) in both layouts while the four worlds spawn
    the robot at (13.0, 14.0) or (5.25, 20.0), so the two frames plainly do not
    coincide and a hand-typed world coordinate is not a map-frame goal.

    Every goal here is therefore a NODE ID in the prior, and its pose is read
    from that database at run time. That makes each goal (a) unambiguously in the
    map frame, (b) a place the mapping run actually observed, so the planner has
    coverage and the localiser has signatures, and (c) checkable after the fact.

    It also means the goal set is a property of the PRIOR, not of the world: the
    U static and U dynamic worlds share the U prior and therefore the same three
    paths. That is the comparison Experiment 4 wants -- identical route, traffic
    added -- and it is why 4 worlds and 2 priors give 6 goal sequences, driven as
    12 runs.

PATH DESIGN, in increasing difficulty
    P1 transit   one goal, the mapped node farthest from the mapping start.
                 Tests relocalise -> plan -> drive, once, one way.
    P2 return    that node, then the node nearest the start. Tests whether the
                 pose survives a round trip, which is where accumulated error
                 and a stale correction show up.
    P3 circuit   four well-separated nodes by farthest-point sampling. Forces
                 repeated aisle-to-aisle transitions, which is where the +-25 deg
                 aliasing that survived the 11-12 Nov sweep actually lives.

TWO DEFECTS THIS SCRIPT REFUSES TO PAPER OVER (see --preflight)
    1. The four worlds do not share a spawn. small_warehouse_static_realistic_
       nofloortexture.world was hand-edited to (13.0, 14.0, -1.5708) and its
       original (5.25, 20, -1.5708) commented out; the other three still carry
       the original. Three paths driven from two different start points are not
       three paths driven under one condition.
    2. All 24 sweep runs of 11-12 Nov passed start_pose:=-7.839 5.875 0 0 0
       1.5696, which matches no world's spawn and no prior's first node. The
       seed is only an initial guess, so the runs still localised, but it is not
       a recorded condition and must not be carried into Experiment 4.
       start_pose:=origin (the launch default, meaning the prior's first node)
       is the only seed with a definition.
"""
import argparse
import math
import os
import re
import sqlite3
import struct
import subprocess
import sys

WORKSPACE = os.path.expanduser('~/wil_project')
# vio_metrics.IMU_LEVER_M is [0.338, 0, 0.192]; only the forward component moves a
# 2-D position.
LEVER_FWD_M = 0.338
WORLD_DIR = os.path.join(WORKSPACE, 'aws-robomaker-small-warehouse-world',
                         'worlds', 'small_warehouse')

# layout -> the closure-free prior for it. map_realistic and map_realistic_01 are
# NOT used: they carry 3 and 4 loop-closure links, and the realistic mapping bags
# are the ones where a false closure moved the map by 3.4 to 28.9 m. A navigation
# prior has to be one whose geometry was scored, which is these two.
PRIORS = {
    'U': 'dataset_allsensor_realistic_map',       # 148 nodes, 294 neighbour links
    'I': 'dataset_allsensor_realistic_01_map',    # 127 nodes, 248 neighbour links
}

# The 4 worlds. 'traffic' is the factor Experiment 4 adds on top of Experiment 3:
# the dynamic worlds carry 4 walking actors (person_H1..H4) over the same layout.
WORLDS = [
    ('U_static',  'small_warehouse_static_realistic_nofloortexture.world',     'U', False),
    ('I_static',  'small_warehouse_static_realistic_nofloortexture_01.world',  'I', False),
    ('U_dynamic', 'small_warehouse_dynamic_realistic_nofloortexture.world',    'U', True),
    ('I_dynamic', 'small_warehouse_dynamic_realistic_nofloortexture_01.world', 'I', True),
]

# The configuration every Experiment 4 run is driven with, and why. Fixed here so
# that a run can never again be attributed to the wrong settings.
CONFIG = {
    # The 11-12 Nov sweep, 24 driven runs: 3D->3D registration makes the matched
    # feature positions agree in metres instead of merely reprojecting, which PnP
    # allows a misplaced match to do.
    'estimation_type': '0',
    # Holds RTAB-Map back for the front end's unstable first second, which is
    # where it otherwise latches a 73-76 deg wrong correction and holds it ~16 s.
    # Confirmed: max |correction yaw| fell from 110-177 deg to 25 deg over three
    # runs. A fixed delay is a sim-specific guess; the transferable fix is an
    # orientation-stability condition in slam_wrapper.cpp.
    'localize_delay_s': '5.0',
    # start_pose is NOT here: it is per world, under worlds:<name>:start_pose,
    # because the U worlds spawn at (7.839, 5.875) and the I worlds at (5.25, 20).
    # Each one is its own world spawn pushed through that layout's map->world fit,
    # so the seed is the robot's actual pose in the frame Nav2 plans in. 'origin'
    # (the prior's first node) was right only while every world spawned where the
    # mapping run started; the U spawn is now 14.0 m from node 1, so 'origin'
    # there would seed the localiser 14 m from the truth.
    'frontend': 'orb',
    'image_transport': 'compressed',
    # Instrumentation, recorded here rather than hardcoded in the runner so a run's
    # outcome.json says what was being logged. All three publish topics only -- no
    # transform -- so none of them competes with the VSLAM frame chain.
    #   model_poses   where the four walkers actually were: the ground truth the
    #                 dynamic condition is about, and the only way a close approach
    #                 to a person can be scored at all.
    #   wheel_odom    an independent dead-reckoning track, so a navigation failure
    #                 can be attributed to the estimate rather than to the chassis.
    #   joint_states  the true wheel and steering angles, which is what makes the
    #                 commanded-versus-achieved steering readable.
    'bridge_model_poses': 'true',
    'bridge_wheel_odom': 'true',
    'bridge_joint_states': 'true',
}

# Success criteria, fixed in code because chapter 4 records that none exists.
# A run is scored per goal and as a whole; a path succeeds only if every goal in
# it does, in order.
CRITERIA = {
    # 1.0 m because that IS nav2_sim.yaml's xy_goal_tolerance: the controller stops
    # when it is within a metre and reports success, so scoring at 0.5 would fail
    # runs that did exactly what they were asked. The measured distance is reported
    # alongside, so a tighter reading is always available from the data.
    'goal_tolerance_m': 1.0,
    'goal_tolerance_deg': 30.0,
    'goal_timeout_s': 180.0,        # per goal; the longest leg is ~40 m at ~1 m/s
    'clearance_fail_m': 0.25,       # GT distance to any object surface: a collision
    'clearance_warn_m': 0.50,
    # A localisation the run cannot be scored on: RTAB-Map's no-information value
    # is a covariance trace of 3e4, and a pose jump over 1 m is not travel at
    # 1.5 m/s with 0.5 s sampling.
    'pose_jump_fail_m': 1.0,
    'cov_trace_blind': 1.0e4,
    'max_blind_fraction': 0.10,     # over a tenth of the run unlocalised = void
}

# Ground truth. The robot's pose is already bridged; object poses are not, and
# the two sources differ in kind:
#   * static objects -- parsed once from the .world file. Exact, no logging.
#   * actors -- must be logged during the run, because they move. gz pose/info
#     carries zero stamps, so sim time has to be rebuilt from /clock.
GROUND_TRUTH = {
    'robot_topic': '/ground_truth/odometry',
    'robot_note': 'published by models/ackermann_robot/model.sdf, already bridged',
    'object_static_source': 'the .world file: every <include> and <model> pose',
    'object_actor_topic': '/world/default/pose/info',
    'object_actor_bridge': ('navigate_sim bridge_model_poses:=true, which starts the '
                            'warehouse launch own model_pose_bridge'),
    'object_actor_note': ('gz.msgs.Pose_V with zero header stamps -- rebuild sim '
                          'time from /clock, do not trust the wall offset'),
    'lever_arm_m': 0.338,
    'lever_arm_note': ('GT is the model origin, the estimators are the IMU. The '
                       'correction is applied in scoring (vm.load_gt), never in '
                       'the recorded data.'),
}


def _world_spawn(path):
    """The ACTIVE robot pose in a world file, skipping commented-out ones."""
    txt = open(path).read()
    i = txt.find('<name>ackermann_robot_001</name>')
    if i < 0:
        return None
    tail = txt[i:i + 2000]
    tail = re.sub(r'<!--.*?-->', '', tail, flags=re.S)   # drop commented poses
    m = re.search(r'<pose>([^<]+)</pose>', tail)
    if not m:
        return None
    return [float(v) for v in m.group(1).split()]


def _align(prior):
    """The map->world rigid transform, solved against the mapping run's ground truth.

    fit_map_pose.py grid-searches the occupancy grid against the world's wall
    geometry; on this prior it returned agreement 0.169 with a +4.4 % margin and
    a result 40 m from the spawn, i.e. it found a wrong local optimum. There is a
    far better constraint available: dataset/gt/<prior>/ground_truth.csv holds the
    mapping run's GT on the same timebase as the Node table's stamp column, so the
    transform is a closed-form 2-D Procrustes fit over every node.

    It comes out at yaw -179.9 deg, t (5.15, 19.87) for U and -179.4 deg,
    (4.74, 20.43) for I -- the map frame is the spawn, turned around. The median
    residual is 0.337 m and 0.338 m, which is the known GT lever arm (GT is the
    model origin, the nodes are the camera/IMU body) and not misalignment. That
    agreement is the reason to trust the fit.

    Needed for two things: converting the world spawn into map coordinates to
    check the seed, and putting GT and the localised pose in one frame so ATE can
    be computed at all.
    """
    import numpy as np
    gt = np.loadtxt(os.path.join(WORKSPACE, 'dataset', 'gt', prior,
                                 'ground_truth.csv'), delimiter=',')
    tg = gt[:, 0] / 1e9
    # Move GT from the model origin to the camera/IMU body BEFORE fitting. The
    # nodes are body poses, so fitting them against the raw origin track makes the
    # rigid transform absorb part of the 0.338 m lever arm as a translation -- it
    # cannot absorb all of it, because the offset rotates with the robot, which is
    # exactly why the residual came out at the lever arm's length. Scoring then
    # applied the lever arm a second time on top of a transform already biased by
    # it, putting a fraction of a metre of my own making into every ATE.
    # Columns are t_ns, x, y, z, qw, qx, qy, qz (see script/extract_gt.py).
    qw, qx, qy, qz = gt[:, 4], gt[:, 5], gt[:, 6], gt[:, 7]
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))
    xg = gt[:, 1] + LEVER_FWD_M * np.cos(yaw)
    yg = gt[:, 2] + LEVER_FWD_M * np.sin(yaw)
    nodes = _nodes_with_stamp(prior)
    P = np.array([[n[2], n[3]] for n in nodes])
    Q = np.column_stack([np.interp([n[1] for n in nodes], tg, xg),
                         np.interp([n[1] for n in nodes], tg, yg)])
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = qc - R @ pc
    res = np.linalg.norm((R @ P.T).T + t - Q, axis=1)
    return {'yaw_deg': math.degrees(math.atan2(R[1, 0], R[0, 0])),
            'tx': float(t[0]), 'ty': float(t[1]),
            'residual_median_m': float(np.median(res)),
            'residual_p90_m': float(np.percentile(res, 90)),
            'residual_max_m': float(res.max()),
            'n_nodes': len(nodes),
            '_R': R, '_t': t}


def world_to_map(align, x, y):
    """A world point in map coordinates. R is map->world, so invert it."""
    import numpy as np
    v = np.array([x, y]) - align['_t']
    m = align['_R'].T @ v
    return float(m[0]), float(m[1])


def _nodes_with_stamp(prior):
    db = os.path.join(WORKSPACE, 'map', prior, f'{prior}.db')
    c = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    out = []
    for nid, st, blob in c.execute('select id, stamp, pose from Node order by id'):
        if not blob or len(blob) < 48:
            continue
        m = struct.unpack('12f', blob[:48])
        out.append((nid, st, m[3], m[7]))
    c.close()
    return out


def _nodes(prior):
    """(id, x, y, yaw) for every node in a prior, in the map frame."""
    db = os.path.join(WORKSPACE, 'map', prior, f'{prior}.db')
    c = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    out = []
    for nid, blob in c.execute('select id, pose from Node order by id'):
        if not blob or len(blob) < 48:
            continue
        m = struct.unpack('12f', blob[:48])
        out.append((nid, m[3], m[7], math.atan2(m[4], m[0])))
    c.close()
    return out


def _spawn_in_map(layout):
    """Where the robot actually starts, in the map frame, for a layout.

    Both worlds of a layout must agree, or 'the same three paths in a static and
    a dynamic world' is not one condition; _preflight refuses otherwise.
    """
    al = _align(PRIORS[layout])
    spawns = {tuple(_world_spawn(os.path.join(WORLD_DIR, w)))
              for _, w, lay, _ in WORLDS if lay == layout}
    sp = sorted(spawns)[0]
    x, y = world_to_map(al, sp[0], sp[1])
    yaw = math.degrees(sp[5]) - al['yaw_deg']
    while yaw > 180:
        yaw -= 360
    while yaw < -180:
        yaw += 360
    return x, y, yaw, len(spawns) == 1


def _paths(prior, layout=None):
    """The three goal sequences for a prior, as node ids. Deterministic.

    Measured from the SPAWN, not from the mapping start. The two were the same
    thing while every world spawned where mapping began; once the U worlds moved
    to (7.839, 5.875) they are 14.0 m apart, and 'farthest from the start' has to
    mean farthest from where the robot is put down, or P1 is not a transit and
    P2's return leg does not return anywhere.
    """
    nodes = _nodes(prior)
    if layout is None:
        layout = next(k for k, v in PRIORS.items() if v == prior)
    sx, sy, _, _ = _spawn_in_map(layout)
    ref = (0, sx, sy, 0.0)
    d = lambda n: math.dist(n[1:3], ref[1:3])       # noqa: E731
    far = max(nodes, key=d)
    # 'nearest the start' must be a real return leg, not somewhere the robot can
    # already see from where it is standing
    near = min((n for n in nodes if d(n) > 3.0), key=d)

    # farthest-point sampling seeded at the spawn
    chosen = []
    while len(chosen) < 4:
        pool = [ref] + chosen
        nxt = max(nodes, key=lambda n: min(math.dist(n[1:3], p[1:3]) for p in pool))
        chosen.append(nxt)
    return {
        'P1_transit': [far[0]],
        'P2_return': [far[0], near[0]],
        'P3_circuit': [n[0] for n in chosen],
    }


def show():
    for layout, prior in PRIORS.items():
        nodes = {n[0]: n for n in _nodes(prior)}
        print(f'--- {layout}  {prior}  ({len(nodes)} nodes)')
        sx, sy, syaw, _ = _spawn_in_map(layout)
        print(f'    spawn in map: ({sx:.2f}, {sy:.2f}) yaw {syaw:.1f} deg')
        ref = (0, sx, sy, 0.0)
        for name, ids in _paths(prior).items():
            legs = []
            prev = ref
            total = 0.0
            for nid in ids:
                n = nodes[nid]
                total += math.dist(prev[1:3], n[1:3])
                legs.append(f'node {nid} ({n[1]:6.2f},{n[2]:6.2f})')
                prev = n
            print(f'    {name:11} {total:5.1f} m straight-line  ' + '  ->  '.join(legs))


def preflight():
    """Everything that must be true before a single Experiment 4 run is driven."""
    ok = True
    spawns = {}
    for name, world, layout, _ in WORLDS:
        p = os.path.join(WORLD_DIR, world)
        if not os.path.isfile(p):
            print(f'  FAIL  {name}: {world} missing')
            ok = False
            continue
        spawns[name] = _world_spawn(p)
    uniq = {tuple(v) for v in spawns.values() if v}
    print('1. the two worlds of a LAYOUT share a spawn')
    print('   (not all four: the U worlds spawn at (7.839, 5.875) and the I worlds at')
    print('    (5.25, 20). Layout is already a factor -- different buildings -- so a')
    print('    per-layout spawn is tolerable where a per-world one is not, because the')
    print('    static/dynamic comparison is what has to hold the route fixed.)')
    for layout in PRIORS:
        sp = {n: tuple(spawns[n]) for n, _, lay, _ in WORLDS if lay == layout and n in spawns}
        agree = len(set(sp.values())) == 1
        if not agree:
            ok = False
        for n, v in sp.items():
            print(f'      {n:11} {list(v)}')
        print(f'      {layout}: {"pass" if agree else "FAIL -- static and dynamic disagree"}')

    print('2. the spawn is in mapped space and in free space')
    print('   (an earlier version compared a world coordinate against a map one and')
    print('    reported a 20 m error that was its own bug. Both are map-frame here.)')
    sys.path.insert(0, os.path.join(WORKSPACE, 'script'))
    import exp4_gt_logger as gtl
    for layout in PRIORS:
        sx, sy, syaw, agree = _spawn_in_map(layout)
        nodes = _nodes(PRIORS[layout])
        near = sorted((math.dist((sx, sy), (n[1], n[2])), n[0]) for n in nodes)[:3]
        world = next(w for _, w, lay, _ in WORLDS if lay == layout)
        objs = [o for o in gtl._poses_from_world(os.path.join(WORLD_DIR, world))
                if not o[0].startswith(('aws_robomaker_warehouse_Ground',
                                        'aws_robomaker_warehouse_Roof', 'ackermann'))]
        sp = _world_spawn(os.path.join(WORLD_DIR, world))
        clear = sorted((math.hypot(sp[0] - o[2], sp[1] - o[3]), o[0]) for o in objs)[:1]
        mapped = near[0][0] < 2.0
        if not mapped:
            ok = False
        print(f'      {layout}  spawn map ({sx:6.2f},{sy:6.2f}) yaw {syaw:5.1f}  '
              f'nearest node n{near[0][1]} at {near[0][0]:.2f} m '
              f'{"pass" if mapped else "FAIL unmapped"}')
        # Free space, two ways. A spawn the MAPPING RUN used is demonstrated free --
        # a vehicle started there and drove the whole route -- and that beats any
        # inference from model origins. Only a spawn no run has used falls back to
        # centre-to-centre clearance, which over-reports near wall-like models: the
        # I spawn sits 0.77 m from a DockDoor whose origin is in the wall, and the
        # mapping run started from that exact pose regardless.
        import numpy as _np
        gt0 = _np.loadtxt(os.path.join(WORKSPACE, 'dataset', 'gt', PRIORS[layout],
                                       'ground_truth.csv'), delimiter=',')[0]
        used = math.hypot(sp[0] - gt0[1], sp[1] - gt0[2])
        if used < 1.0:
            print(f'         free: the mapping run started here ({used:.2f} m from its '
                  f'first GT sample)  pass')
        elif clear[0][0] > 1.0:
            print(f'         free: nearest object {clear[0][0]:.2f} m '
                  f'({clear[0][1]})  pass')
        else:
            ok = False
            print(f'         FAIL blocked: {clear[0][0]:.2f} m from {clear[0][1]}, and '
                  f'no run has started here ({used:.1f} m from the mapping start)')
    print('   start_pose is therefore the spawn in map coordinates, per world, not')
    print("   'origin'. The U spawn is 14.0 m from node 1, so 'origin' would seed there.")

    print('3. priors are closure-free and carry a vocabulary')
    for layout, prior in PRIORS.items():
        db = os.path.join(WORKSPACE, 'map', prior, f'{prior}.db')
        c = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
        lt = dict(c.execute('select type, count(*) from Link group by type').fetchall())
        w = c.execute('select count(*) from Word').fetchone()[0]
        c.close()
        closures = lt.get(1, 0) + lt.get(6, 0)
        verdict = 'pass' if closures == 0 and w > 1000 else 'FAIL'
        if verdict == 'FAIL':
            ok = False
        print(f'      {layout}  closures {closures}  words {w}  {verdict}')

    print('4. ground truth is reachable')
    logger = os.path.join(WORKSPACE, 'script', 'exp4_gt_logger.py')
    have_logger = os.path.isfile(logger)
    bridge = subprocess.run(['ros2', 'pkg', 'prefix', 'ros_gz_bridge'],
                            capture_output=True, text=True).returncode == 0
    print(f'      robot   {GROUND_TRUTH["robot_topic"]}  (bridged by the robot model)')
    print(f'      static  parsed from the .world file, 130 models per world')
    print(f'      actors  {GROUND_TRUTH["object_actor_topic"]} -> /exp4/object_poses')
    print(f'      logger  {"present" if have_logger else "MISSING"}  '
          f'ros_gz_bridge {"present" if bridge else "MISSING"}')
    if not (have_logger and bridge):
        ok = False

    print('5. the map->world fit is trustworthy')
    for layout, prior in PRIORS.items():
        al = _align(prior)
        # Now that the lever arm is applied BEFORE fitting, the residual is true
        # misalignment plus node-pose error, and should be SMALL. It was 0.337 and
        # 0.338 m while the fit was absorbing the lever arm instead -- a figure that
        # looked like confirmation and was really the bias it was hiding.
        good = al['residual_median_m'] < 0.20
        if not good:
            ok = False
        print(f'      {layout}  yaw {al["yaw_deg"]:7.2f}  t ({al["tx"]:6.2f},{al["ty"]:6.2f})'
              f'  residual median {al["residual_median_m"]:.3f} m '
              f'(lever arm {GROUND_TRUTH["lever_arm_m"]})  '
              f'{"pass" if good else "FAIL"}')

    print('\n' + ('ALL CHECKS PASS' if ok else 'NOT READY -- fix the FAILs above first'))
    return ok


def freeze():
    out = os.path.join(WORKSPACE, 'script', 'exp4_protocol.yaml')
    lines = ['# Experiment 4: autonomous navigation on the reconstructed map.',
             '# GENERATED by script/exp4_goals.py --freeze. Do not hand-edit; change the',
             '# script so the selection stays reproducible.',
             '#',
             '# 4 worlds x 3 paths = 12 runs. The 2 priors give 6 goal sequences, each',
             '# driven in a static and a dynamic world, so traffic is a within-route',
             '# factor rather than a separate experiment.',
             '',
             'config:']
    for k, v in CONFIG.items():
        lines.append(f'  {k}: "{v}"')
    lines += ['', 'criteria:']
    for k, v in CRITERIA.items():
        lines.append(f'  {k}: {v}')
    lines += ['', 'ground_truth:']
    for k, v in GROUND_TRUTH.items():
        lines.append(f'  {k}: "{v}"' if isinstance(v, str) else f'  {k}: {v}')
    lines += ['', 'priors:']
    for layout, prior in PRIORS.items():
        nodes = {n[0]: n for n in _nodes(prior)}
        lines.append(f'  {layout}:')
        lines.append(f'    database: {prior}')
        lines.append(f'    nodes: {len(nodes)}')
        al = _align(prior)
        lines.append('    map_to_world:   # solved against the mapping run GT')
        for k in ('yaw_deg', 'tx', 'ty', 'residual_median_m', 'residual_p90_m',
                  'residual_max_m'):
            lines.append(f'      {k}: {al[k]:.4f}')
        lines.append('    paths:')
        for name, ids in _paths(prior).items():
            lines.append(f'      {name}:')
            for nid in ids:
                n = nodes[nid]
                lines.append(f'        - node: {nid}')
                lines.append(f'          map_xy: [{n[1]:.3f}, {n[2]:.3f}]')
                lines.append(f'          map_yaw_deg: {math.degrees(n[3]):.1f}')
    lines += ['', 'worlds:']
    for name, world, layout, traffic in WORLDS:
        spawn = _world_spawn(os.path.join(WORLD_DIR, world))
        lines.append(f'  {name}:')
        lines.append(f'    file: {world}')
        lines.append(f'    layout: {layout}')
        lines.append(f'    traffic: {str(traffic).lower()}')
        lines.append(f'    spawn: {spawn}            # Gazebo world frame')
        sx, sy, syaw, _ = _spawn_in_map(layout)
        lines.append(f'    start_pose: "{sx:.3f} {sy:.3f} 0 0 0 '
                     f'{math.radians(syaw):.4f}"   # the same pose, map frame')
    lines.append('')
    open(out, 'w').write('\n'.join(lines))
    print(f'wrote {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--freeze', action='store_true')
    ap.add_argument('--preflight', action='store_true')
    a = ap.parse_args()
    if a.show:
        show()
    elif a.freeze:
        freeze()
    elif a.preflight:
        sys.exit(0 if preflight() else 1)
    else:
        ap.print_help()
