#!/usr/bin/env python3
"""Drive one Experiment 4 run: relocalise on a prior, then reach a goal sequence.

    python3 script/exp4_navigate.py --list
    python3 script/exp4_navigate.py --run U_static P1_transit
    python3 script/exp4_navigate.py --run U_dynamic P3_circuit --rep 2
    python3 script/exp4_navigate.py --score                     # all runs so far

The robot drives itself. Nothing is teleoperated here, which is the point: the
24 sweep runs of 11-12 Nov were hand-driven, so route length was an uncontrolled
variable and no two runs were strictly comparable. A goal sequence is a
condition; a hand-driven lap is not.

WHAT ONE RUN PRODUCES, under map/<prior>/exp4/<world>_<path>_<rep>/
    outcome.json      per-goal result, the whole-run verdict and why
    robot_gt.csv      /ground_truth/odometry, raw (the 0.338 m lever arm is a
                      scoring correction, never applied to recorded data)
    object_gt.csv     the four actors, live, in dynamic worlds
    objects_static.csv the 130-model layout, parsed from the world file
    localize_*.csv    the localisation record, written by the launch as usual

VERDICTS
    reached      every goal accepted by Nav2 inside its timeout
    missed       Nav2 gave up or timed out on a goal
    collided     GT clearance fell below the failure threshold
    lost         more than max_blind_fraction of the run carried no information
    void         the stack did not come up, or no fix was ever obtained

A run can be both 'reached' and 'collided'; the verdict reports the worst.
"""
import argparse
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time

import yaml

WORKSPACE = os.path.expanduser('~/wil_project')
PROTOCOL = os.path.join(WORKSPACE, 'script', 'exp4_protocol.yaml')
WORLD_DIR = os.path.join(WORKSPACE, 'aws-robomaker-small-warehouse-world',
                         'worlds', 'small_warehouse')
GT_LOGGER = os.path.join(WORKSPACE, 'script', 'exp4_gt_logger.py')
# All four worlds declare <world name="default">, so this is where the warehouse
# launch's model_pose_bridge publishes. Not remapped: the bridge keeps the gz name.
GZ_POSE_TOPIC = '/world/default/pose/info'


def load():
    with open(PROTOCOL) as f:
        return yaml.safe_load(f)


def run_dir(spec, world, path, rep):
    prior = spec['priors'][spec['worlds'][world]['layout']]['database']
    return os.path.join(WORKSPACE, 'map', prior, 'exp4', f'{world}_{path}_{rep}')


# --------------------------------------------------------------- the ROS side
def _navigate(spec, world, path, out, deadline_s):
    """Wait for a fix, then send the goals one at a time. Returns the record."""
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
    from nav2_msgs.action import NavigateToPose

    crit = spec['criteria']
    goals = spec['priors'][spec['worlds'][world]['layout']]['paths'][path]

    class Driver(Node):
        def __init__(self):
            super().__init__('exp4_driver')
            self.set_parameters([rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])
            self.poses = []
            self.create_subscription(PoseWithCovarianceStamped, '/localization_pose',
                                     self.on_pose, 20)
            self.ac = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        def on_pose(self, m):
            # cov[0] + cov[7] + cov[14] is the trace RTAB-Map fills with 9999
            # per axis when it has no information.
            c = m.pose.covariance
            self.poses.append((self.get_clock().now().nanoseconds / 1e9,
                               c[0] + c[7] + c[14]))

        def sim_now(self):
            return self.get_clock().now().nanoseconds / 1e9

        def informed(self):
            return [p for p in self.poses if p[1] < crit['cov_trace_blind']]

    rclpy.init()
    n = Driver()
    rec = {'goals': [], 'first_fix_s': None, 'notes': []}
    t_start = None
    try:
        # 1. wait for the stack and the first informative pose
        t0 = time.time()
        while time.time() - t0 < deadline_s:
            rclpy.spin_once(n, timeout_sec=0.5)
            if n.informed():
                break
        if not n.informed():
            rec['notes'].append(f'no informative pose within {deadline_s:g}s wall')
            rec['verdict'] = 'void'
            return rec
        t_start = n.sim_now()
        rec['first_fix_s'] = round(n.informed()[0][0] - (n.poses[0][0] if n.poses else 0), 2)
        n.get_logger().info(f'localised; sending {len(goals)} goal(s)')

        if not n.ac.wait_for_server(timeout_sec=30.0):
            rec['notes'].append('navigate_to_pose action server never appeared')
            rec['verdict'] = 'void'
            return rec

        # 2. the goals, in order, each with its own timeout
        for i, g in enumerate(goals):
            ps = PoseStamped()
            ps.header.frame_id = 'map'
            ps.header.stamp = n.get_clock().now().to_msg()
            ps.pose.position.x, ps.pose.position.y = g['map_xy']
            yaw = math.radians(g['map_yaw_deg'])
            ps.pose.orientation.z = math.sin(yaw / 2)
            ps.pose.orientation.w = math.cos(yaw / 2)
            n.get_logger().info(f'goal {i+1}/{len(goals)}: node {g["node"]} '
                                f'at {g["map_xy"]}')
            sent = n.ac.send_goal_async(NavigateToPose.Goal(pose=ps))
            rclpy.spin_until_future_complete(n, sent, timeout_sec=30.0)
            handle = sent.result()
            if handle is None or not handle.accepted:
                rec['goals'].append({'node': g['node'], 'result': 'rejected'})
                rec['verdict'] = 'missed'
                return rec
            gt0 = n.sim_now()
            fut = handle.get_result_async()
            while n.sim_now() - gt0 < crit['goal_timeout_s']:
                rclpy.spin_once(n, timeout_sec=0.2)
                if fut.done():
                    break
            if not fut.done():
                handle.cancel_goal_async()
                rclpy.spin_once(n, timeout_sec=2.0)
                rec['goals'].append({'node': g['node'], 'result': 'timeout',
                                     'elapsed_s': round(n.sim_now() - gt0, 1)})
                rec['verdict'] = 'missed'
                return rec
            status = fut.result().status
            ok = status == 4   # STATUS_SUCCEEDED
            rec['goals'].append({'node': g['node'],
                                 'result': 'succeeded' if ok else f'status_{status}',
                                 'elapsed_s': round(n.sim_now() - gt0, 1)})
            if not ok:
                rec['verdict'] = 'missed'
                return rec

        rec['verdict'] = 'reached'
        return rec
    finally:
        if t_start is not None:
            rec['duration_s'] = round(n.sim_now() - t_start, 1)
            blind = [p for p in n.poses if p[1] >= crit['cov_trace_blind']
                     and p[0] >= t_start]
            after = [p for p in n.poses if p[0] >= t_start]
            rec['blind_poses'] = len(blind)
            rec['poses_after_fix'] = len(after)
            if after and len(blind) / len(after) > crit['max_blind_fraction']:
                rec['notes'].append(f'{len(blind)}/{len(after)} poses carried no '
                                    f'information after the first fix')
                rec['verdict'] = 'lost'
        json.dump(rec, open(os.path.join(out, 'outcome_partial.json'), 'w'), indent=2)
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# ------------------------------------------------------------------- the driver
def _prepare(world, path, rep):
    """Everything both --run and --drive need: directories, the static object
    dump, and the three commands. No process is started here."""
    spec = load()
    if world not in spec['worlds']:
        sys.exit(f'unknown world {world!r}; try --list')
    if path not in spec['priors'][spec['worlds'][world]['layout']]['paths']:
        sys.exit(f'unknown path {path!r} for {world}; try --list')
    w = spec['worlds'][world]
    prior = spec['priors'][w['layout']]['database']
    out = run_dir(spec, world, path, rep)
    prev = os.path.join(out, 'outcome.json')
    if os.path.exists(prev):
        try:
            verdict = json.load(open(prev)).get('verdict')
        except Exception:
            verdict = None
        # 'void' means the stack never came up or no fix was ever obtained. That is
        # not a result, so it must not block the retry; anything else is.
        if verdict != 'void':
            sys.exit(f'{out} already holds a scored run ({verdict}). '
                     f'Use --rep <n> for another.')
        print(f'[exp4] replacing an earlier void attempt in {os.path.basename(out)}')
    os.makedirs(out, exist_ok=True)
    # A leftover from an earlier run does not announce itself in the new one -- it
    # produces quietly wrong data. sweep_M4 recorded 28 localisation poses with
    # covariance 0.0056 and ZERO map->odom transforms over 34 s because a Gazebo
    # from an abort 21 minutes earlier was still publishing /clock. Two clocks make
    # sim time jump and every TF lookup returns nothing, while the pose topic is
    # unaffected because it is published rather than looked up. That abort was one
    # of this script's own, which is why the guard belongs here too.
    sys.path.insert(0, os.path.join(WORKSPACE, 'script'))
    import sim_guard
    if not sim_guard.require_clean():
        sys.exit(1)
    world_file = os.path.join(WORLD_DIR, w['file'])

    # the static layout, exact and free
    sys.path.insert(0, os.path.dirname(GT_LOGGER))
    import exp4_gt_logger as gtl
    gtl.dump_static(world_file, os.path.join(out, 'objects_static.csv'))

    cmd = ['ros2', 'launch', 'vslam_navigation', 'navigate_sim.launch.py',
           f'map_name:={prior}', f'run_name:=exp4_{world}_{path}_{rep}',
           f'world:={world_file}',
           f'orb_output_path:={out}/orb',
           # Per world, not from config: each layout spawns somewhere different, and
           # the seed is that spawn expressed in the frame Nav2 plans in.
           f'start_pose:={w["start_pose"]}',
           # The launch refuses to replace an existing localisation CSV, which is
           # right -- it stopped me overwriting three of the operator's runs on
           # 11 Nov. But here the authority on "this run is recorded" is exp4's own
           # outcome.json, checked just above, and an aborted attempt leaves a
           # partial CSV that would otherwise block every retry. Measured: an
           # attempt at 13:28 left a 1040-byte CSV and the 13:43 retry died on it
           # after Gazebo had already loaded.
           'overwrite:=true'] + \
          [f'{k}:={v}' for k, v in sorted(spec['config'].items())]
    # No bridge process here: navigate_sim starts the warehouse launch's own
    # model-pose bridge via bridge_model_poses:=true (added 12 Nov 2026). A second
    # parameter_bridge on the same topic is one more thing to go wrong and one more
    # process to survive an abort -- and a surviving process is what corrupted
    # sweep_M4's clock.
    logger = ['python3', GT_LOGGER, '--ros-args',
              '-p', 'use_sim_time:=true',
              '-p', f'robot_out:={out}/robot_gt.csv',
              '-p', f'object_out:={out}/object_gt.csv']

    return spec, w, prior, out, world_file, cmd, logger


def _launch(cmd, log):
    """Start the launch with its output both live and kept.

    ros2 launch's own log directory held only launch.log for the runs that failed
    on 12 Nov, so the node output naming the cause was gone by the time anyone
    looked. tee keeps it beside the run's own data.
    """
    print(f'[exp4] launch output -> {log}')
    teed = f'{shlex.join(cmd)} 2>&1 | tee {shlex.quote(log)}'
    return subprocess.Popen(['bash', '-c', teed], preexec_fn=os.setsid)


def _settle(proc, settle_s):
    """Wait for Gazebo and the stack, watching for the launch dying.

    A plain sleep here reported 'ground-truth logging started; drive the goals'
    45 s after the launch had already aborted, then wrote a void record with no
    indication of why. The launch prints its own reason; this just has to stop
    and point at it.
    """
    deadline = time.time() + settle_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        time.sleep(0.5)
    return proc.poll() is None


def _shutdown(procs):
    """SIGINT the process groups, then SIGKILL whatever is left. Process groups
    and pidfiles only: pkill -f matches this script's own command line."""
    for p in reversed(procs):
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGINT)
        except Exception:
            pass
    time.sleep(4)
    for p in reversed(procs):
        if p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:
                pass


def _finish(out, rec, world, path, rep, prior):
    """Clearance screen, then the outcome record. Shared so a driven run and an
    autonomous one are scored by exactly the same code."""
    sys.path.insert(0, os.path.join(WORKSPACE, 'script'))
    import exp4_gt_logger as gtl
    rob = os.path.join(out, 'robot_gt.csv')
    stat = os.path.join(out, 'objects_static.csv')
    obj = os.path.join(out, 'object_gt.csv')
    if os.path.exists(rob) and os.path.getsize(rob) > 80:
        try:
            cl = gtl.clearance(rob, stat)
            rec.update(cl)
            if cl['min_clearance_m'] < load()['criteria']['clearance_fail_m']:
                rec['notes'] = rec.get('notes', []) + [
                    f'clearance {cl["min_clearance_m"]} m to {cl["nearest_object"]}']
                rec['verdict'] = 'collided'
        except Exception as e:
            rec['notes'] = rec.get('notes', []) + [f'clearance not computed: {e}']
        rec['robot_gt_rows'] = sum(1 for _ in open(rob)) - 1
    else:
        rec['notes'] = rec.get('notes', []) + ['no robot GT recorded']
        rec['verdict'] = 'void'
    rec['object_gt_rows'] = (sum(1 for _ in open(obj)) - 1) if os.path.exists(obj) else 0
    if os.path.exists(rob) and os.path.getsize(rob) > 80:
        try:
            ac = gtl.actor_clearance(rob, obj)
            rec.update(ac)
            lim = load()['criteria']['clearance_fail_m']
            if ac.get('min_actor_clearance_m') is not None and \
                    ac['min_actor_clearance_m'] < lim:
                rec['notes'] = rec.get('notes', []) + [
                    f'came within {ac["min_actor_clearance_m"]} m of '
                    f'{ac["nearest_actor"]} at t={ac["actor_at_stamp"]} s while doing '
                    f'{ac["robot_speed_then_m_s"]} m/s']
                rec['verdict'] = 'collided'
        except Exception as e:
            rec['notes'] = rec.get('notes', []) + [f'actor clearance not computed: {e}']
    if rec['object_gt_rows'] == 0:
        rec['notes'] = rec.get('notes', []) + [
            'no actor poses logged: expected in a static world, a bridge failure '
            'in a dynamic one']
    rec.update({'world': world, 'path': path, 'rep': rep, 'prior': prior,
                'config': load()['config'],
                'git': subprocess.run(['git', '-C', WORKSPACE, 'rev-parse', '--short',
                                       'HEAD'], capture_output=True,
                                      text=True).stdout.strip(),
                'when': time.strftime('%F %T')})
    json.dump(rec, open(os.path.join(out, 'outcome.json'), 'w'), indent=2)
    print(f'\n    verdict: {rec["verdict"]}')
    print(f'    robot GT {rec.get("robot_gt_rows", 0)} rows, '
          f'actor GT {rec["object_gt_rows"]} rows')
    print(f'    clearance: static {rec.get("min_clearance_m", float("nan"))} m'
          f'   actors {rec.get("min_actor_clearance_m", "n/a")} m'
          + (f' ({rec["nearest_actor"]})' if rec.get('nearest_actor') else ''))
    for note in rec.get('notes', []):
        print(f'      note: {note}')
    print(f'    -> {out}/outcome.json')


def do_run(world, path, rep, settle_s, deadline_s):
    spec, w, prior, out, world_file, cmd, logger = _prepare(world, path, rep)
    print(f'\n=== {world} / {path} / rep {rep}   (autonomous)')
    print(f'    prior  {prior}')
    print(f'    spawn  world {w["spawn"][:2]}  ->  map seed "{w["start_pose"]}"')
    print(f'    goals  ' + ' -> '.join(
        f'node {g["node"]}'
        for g in spec['priors'][w['layout']]['paths'][path]))
    print(f'    out    {out}\n')

    procs = []
    rec = {'verdict': 'void', 'notes': ['driver aborted before scoring']}
    try:
        procs.append(_launch(cmd, os.path.join(out, 'launch.log')))
        if not _settle(procs[0], settle_s):
            rec = {'verdict': 'void', 'notes': [
                f'the launch exited during startup (code {procs[0].returncode}); '
                f'its own output above says why']}
            raise SystemExit
        procs.append(subprocess.Popen(logger, preexec_fn=os.setsid))
        rec = _navigate(spec, world, path, out, deadline_s)
    except KeyboardInterrupt:
        rec = {'verdict': 'void', 'notes': ['interrupted by the operator']}
    except SystemExit:
        pass
    finally:
        _shutdown(procs)
    _finish(out, rec, world, path, rep, prior)


def do_drive(world, path, rep, settle_s):
    """Launch the stack, log ground truth, and let the operator drive.

    The same shape as aisle_sweep.py --run: print the command, start it, stand
    back, Ctrl-C when done, write a manifest. --run does the same setup and then
    sends the goals through Nav2's action itself; this leaves that to you.

    The recorded data is identical -- robot GT, actor poses, the static layout,
    the clearance screen -- so a driven run is scored by the same --score table.
    Its verdict is 'driven' rather than reached/missed, because whether a goal
    counts as reached is a judgement the action client makes and an operator does
    not.
    """
    spec, w, prior, out, world_file, cmd, logger = _prepare(world, path, rep)
    goals = spec['priors'][w['layout']]['paths'][path]

    print(f'\n=== {world} / {path} / rep {rep}   (operator-driven)')
    print(f'    prior  {prior}')
    print(f'    spawn  world {w["spawn"][:2]}  ->  map seed "{w["start_pose"]}"')
    print(f'    out    {out}')
    print('\n    ' + ' \\\n        '.join(cmd))
    print('\n    Goals to set in RViz (2D Nav Goal), in the map frame, in order:')
    for i, g in enumerate(goals):
        print(f'      {i + 1}. node {g["node"]:<4} x {g["map_xy"][0]:7.2f}  '
              f'y {g["map_xy"][1]:7.2f}  yaw {g["map_yaw_deg"]:7.1f} deg')
    print('\n    Wait for the first fix before setting a goal: the stack holds RTAB-Map')
    print(f'    back {spec["config"]["localize_delay_s"]} s on purpose, and a goal sent '
          f'before the pose settles')
    print('    is a test of nothing. Ctrl-C when the sequence is done or has failed.\n')

    procs = []
    started = time.time()
    rec = {'verdict': 'driven', 'goals': [],
           'notes': [f'operator-driven; {len(goals)} goal(s) set by hand']}
    try:
        procs.append(_launch(cmd, os.path.join(out, 'launch.log')))
        if not _settle(procs[0], settle_s):
            print(f'\n[exp4] the launch exited during startup '
                  f'(code {procs[0].returncode}). Its output above says why; nothing '
                  f'was recorded.')
            rec['notes'].append(f'launch exited during startup, code '
                                f'{procs[0].returncode}')
            raise SystemExit
        procs.append(subprocess.Popen(logger, preexec_fn=os.setsid))
        print(f'[exp4] ground-truth logging started; drive the {len(goals)} goal(s), '
              f'then Ctrl-C')
        procs[0].wait()
    except (KeyboardInterrupt, SystemExit):
        print('\n[exp4] stopping')
    finally:
        _shutdown(procs)
    rec['duration_s'] = round(time.time() - started, 1)
    _finish(out, rec, world, path, rep, prior)


def score():
    spec = load()
    rows = []
    for world in spec['worlds']:
        layout = spec['worlds'][world]['layout']
        for path in spec['priors'][layout]['paths']:
            for rep in (1, 2, 3):
                f = os.path.join(run_dir(spec, world, path, rep), 'outcome.json')
                if os.path.exists(f):
                    rows.append(json.load(open(f)))
    if not rows:
        sys.exit('no scored runs yet.')
    print(f'{"world":11} {"path":11} {"rep":>3} {"verdict":9} {"goals":>7} '
          f'{"dur":>6} {"clear":>7} {"blind":>9}')
    for r in rows:
        g = r.get('goals', [])
        okn = sum(1 for x in g if x.get('result') == 'succeeded')
        print(f'{r["world"]:11} {r["path"]:11} {r["rep"]:3d} {r["verdict"]:9} '
              f'{okn:3d}/{len(g):<3d} {r.get("duration_s", 0):6.0f} '
              f'{r.get("min_clearance_m", float("nan")):7.2f} '
              f'{r.get("blind_poses", 0):4d}/{r.get("poses_after_fix", 0):<4d}')
    n_drv = sum(1 for r in rows if r['verdict'] == 'driven')
    if n_drv:
        print(f'\n  {n_drv} run(s) were operator-driven: the recorded data is the same,')
        print('  but whether each goal was reached is not in the record, so those rows')
        print('  carry no goal count. Judge them from the trajectory and clearance.')
    n_ok = sum(1 for r in rows if r['verdict'] == 'reached')
    print(f'\n  {n_ok} of {len(rows)} runs reached every goal')
    print('  clear = min GT centre-to-centre clearance, a screening measure; confirm a')
    print('          flagged run against the recording before calling it a collision')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--run', nargs=2, metavar=('WORLD', 'PATH'),
                    help='autonomous: the goals are sent through Nav2 for you')
    ap.add_argument('--drive', nargs=2, metavar=('WORLD', 'PATH'),
                    help='operator-driven: the stack and the ground-truth logging '
                         'come up, you set the goals in RViz, Ctrl-C to finish. '
                         'Same recorded data, same --score table.')
    ap.add_argument('--rep', type=int, default=1)
    ap.add_argument('--settle', type=float, default=45.0,
                    help='seconds for Gazebo and the stack before the bridge starts')
    ap.add_argument('--deadline', type=float, default=240.0,
                    help='wall seconds to wait for the first informative pose')
    ap.add_argument('--score', action='store_true')
    a = ap.parse_args()
    if a.list:
        spec = load()
        for world, w in spec['worlds'].items():
            paths = spec['priors'][w['layout']]['paths']
            print(f'{world:11} layout {w["layout"]}  traffic {w["traffic"]}  '
                  f'prior {spec["priors"][w["layout"]]["database"]}')
            for p, goals in paths.items():
                print(f'    {p:11} ' + ' -> '.join(f'node {g["node"]}' for g in goals))
    elif a.run:
        do_run(a.run[0], a.run[1], a.rep, a.settle, a.deadline)
    elif a.drive:
        do_drive(a.drive[0], a.drive[1], a.rep, a.settle)
    elif a.score:
        score()
    else:
        ap.print_help()
