#!/usr/bin/env python3
"""Ground truth for Experiment 4: the robot's pose and every object's pose.

    # one-shot, no ROS: the static layout straight out of the world file
    python3 script/exp4_gt_logger.py --world <file.world> --static-out objects_static.csv

    # during a run, as a node
    python3 script/exp4_gt_logger.py --ros-args \\
        -p use_sim_time:=true -p robot_out:=robot_gt.csv -p object_out:=object_gt.csv

TWO KINDS OF OBJECT, AND THEY NEED DIFFERENT TREATMENT
    The static layout -- 130 <include> blocks per world, racks, walls, pallet
    jacks -- never moves, so logging it at 10 Hz for 100 s would write a hundred
    thousand identical rows. It is parsed once from the .world file instead,
    which is also exact rather than sampled.

    The four actors (person_H1..H4) in each dynamic world do move, and they are
    the whole point of the dynamic condition, so those are logged live.

WHY THE STAMP COMES FROM THE CLOCK AND NOT THE MESSAGE
    gz's pose/info messages arrive with zero header stamps. Taking the stamp
    from the message would put every object at t=0; taking it from wall time
    drifts against sim time. So this node runs with use_sim_time:=true and
    stamps each sample from its own ROS clock, which is /clock. That is the only
    one of the three that is both non-zero and on the same timebase as
    /ground_truth/odometry and the localisation CSVs.

WHAT IS *NOT* CORRECTED HERE
    /ground_truth/odometry is the model origin; the estimators report the IMU,
    0.338 m ahead of it. The correction belongs in scoring, not in recorded
    data, so this writes the raw GT pose and the scorer applies the lever arm.
"""
import argparse
import csv
import math
import os
import re
import sys


# ---------------------------------------------------------------- static layout
def _poses_from_world(path):
    """(name, uri, x, y, z, roll, pitch, yaw) for every model in a world file.

    Commented-out blocks are stripped first: the U static world carries a
    commented spawn pose beside the live one, and a regex that ignores comments
    reports the wrong one.
    """
    txt = re.sub(r'<!--.*?-->', '', open(path).read(), flags=re.S)
    out = []
    for block in re.findall(r'<(?:include|model)\b.*?</(?:include|model)>', txt, flags=re.S):
        name = re.search(r'<name>([^<]+)</name>', block)
        uri = re.search(r'<uri>([^<]+)</uri>', block)
        pose = re.search(r'<pose>([^<]+)</pose>', block)
        if not name:
            name = re.search(r'<model name="([^"]+)"', block)
        if not name:
            continue
        vals = [float(v) for v in pose.group(1).split()] if pose else [0.0] * 6
        vals += [0.0] * (6 - len(vals))
        out.append((name.group(1), uri.group(1) if uri else '', *vals[:6]))
    return out


def dump_static(world, out):
    rows = _poses_from_world(world)
    os.makedirs(os.path.dirname(os.path.abspath(out)) or '.', exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['name', 'uri', 'x', 'y', 'z', 'roll', 'pitch', 'yaw'])
        w.writerows(rows)
    print(f'{len(rows)} objects -> {out}')
    return rows


# ------------------------------------------------------------------- live logger
def run_node(argv):
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Odometry
    from tf2_msgs.msg import TFMessage

    class GtLogger(Node):
        def __init__(self):
            super().__init__('exp4_gt_logger')
            self.declare_parameter('robot_out', '')
            self.declare_parameter('object_out', '')
            self.declare_parameter('object_topic', '/world/default/pose/info')
            # Only the movers are worth logging live; everything else is in the
            # static dump. '' logs every frame the bridge publishes.
            self.declare_parameter('object_filter', 'person_')
            self.declare_parameter('object_rate', 10.0)
            g = lambda k: self.get_parameter(k).value  # noqa: E731

            self.filt = g('object_filter')
            self.min_dt = 1.0 / max(float(g('object_rate')), 0.1)
            self._last_obj = -1e9
            self.n_robot = self.n_obj = 0
            self._names_written = False
            self.names_path = (os.path.join(os.path.dirname(g('object_out')),
                                            'pose_feed_entities.txt')
                               if g('object_out') else '/tmp/pose_feed_entities.txt')

            self.rf = self.of = None
            if g('robot_out'):
                self.rf = csv.writer(open(g('robot_out'), 'w', newline='', buffering=1))
                self.rf.writerow(['stamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw',
                                  'vx', 'vy', 'vyaw'])
                self.create_subscription(Odometry, '/ground_truth/odometry',
                                         self.on_robot, 50)
            if g('object_out'):
                self.of = csv.writer(open(g('object_out'), 'w', newline='', buffering=1))
                self.of.writerow(['stamp', 'name', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
                self.create_subscription(TFMessage, g('object_topic'),
                                         self.on_objects, 50)
            self.create_timer(10.0, self.report)
            self.get_logger().info(
                f'logging robot={bool(self.rf)} objects={bool(self.of)} '
                f'filter={self.filt!r} on {g("object_topic")}')

        def now(self):
            # /clock, because the gz messages carry zero stamps.
            return self.get_clock().now().nanoseconds / 1e9

        def on_robot(self, m):
            p, o = m.pose.pose.position, m.pose.pose.orientation
            t = m.twist.twist
            self.rf.writerow([f'{self.now():.6f}', f'{p.x:.5f}', f'{p.y:.5f}',
                              f'{p.z:.5f}', f'{o.x:.6f}', f'{o.y:.6f}', f'{o.z:.6f}',
                              f'{o.w:.6f}', f'{t.linear.x:.4f}', f'{t.linear.y:.4f}',
                              f'{t.angular.z:.4f}'])
            self.n_robot += 1

        def on_objects(self, msg):
            now = self.now()
            # Record every entity name ONCE, before any filtering. An empty
            # object_gt.csv in a dynamic world is otherwise ambiguous: it could mean
            # the bridge was off, or that <actor> entities are not carried in
            # pose/info at all (the upstream note about pose/info vs
            # dynamic_pose/info concerns PROPS, and says nothing about actors). With
            # this, the run itself says which, and the fallback -- a different gz
            # topic, or a different filter -- is a reading rather than a guess.
            if not self._names_written:
                self._names_written = True
                names = sorted(tr.child_frame_id for tr in msg.transforms)
                try:
                    with open(self.names_path, 'w') as f:
                        f.write('\n'.join(names) + '\n')
                except Exception:
                    pass
                hits = [n for n in names if self.filt and self.filt in n]
                self.get_logger().info(
                    f'{len(names)} entities on the pose feed; '
                    f'{len(hits)} match filter {self.filt!r}'
                    + (f': {hits[:6]}' if hits else
                       f'. Names written to {self.names_path}'))
                if self.filt and not hits:
                    self.get_logger().warn(
                        'nothing matches the filter, so no actor ground truth will be '
                        'recorded. Check that file for the names actually published.')
            if now - self._last_obj < self.min_dt:
                return
            self._last_obj = now
            for tr in msg.transforms:
                name = tr.child_frame_id
                if self.filt and self.filt not in name:
                    continue
                v, q = tr.transform.translation, tr.transform.rotation
                self.of.writerow([f'{now:.6f}', name, f'{v.x:.5f}', f'{v.y:.5f}',
                                  f'{v.z:.5f}', f'{q.x:.6f}', f'{q.y:.6f}',
                                  f'{q.z:.6f}', f'{q.w:.6f}'])
                self.n_obj += 1

        def report(self):
            self.get_logger().info(f'robot {self.n_robot} rows, objects {self.n_obj} rows')

    rclpy.init(args=argv)
    n = GtLogger()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.get_logger().info(f'final: robot {n.n_robot} rows, objects {n.n_obj} rows')
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# ---------------------------------------------------------------------- scoring
def actor_clearance(robot_csv, object_csv, robot_radius=0.35, person_radius=0.30):
    """Closest approach between the robot and any walking actor, over a run.

    The static screen cannot see this, and in a dynamic world it is the thing the
    condition is about. The four actors in each dynamic world walk scripted lines
    -- two along x = -6.54 and x = -1.98 from y -14 to ~6, two along y = 15.40 --
    and the U spawn at (13.0, 14.0) sits 1.4 m off person_H4's line, so the robot
    can meet one before it has moved.

    Read the result with one caveat: a close approach is not automatically the
    robot's fault. These actors follow a fixed script and do not avoid anything,
    so a pedestrian can walk into a stationary robot. Pair a flagged approach with
    the robot's own speed at that moment, which robot_gt.csv carries.
    """
    import numpy as np
    rob = list(csv.DictReader(open(robot_csv)))
    if not rob:
        return {'min_actor_clearance_m': None, 'note': 'no robot GT'}
    obj = list(csv.DictReader(open(object_csv))) if os.path.exists(object_csv) else []
    if not obj:
        return {'min_actor_clearance_m': None,
                'note': 'no actor poses logged (a static world, or the bridge was off)'}
    rt = np.array([float(r['stamp']) for r in rob])
    rx = np.array([float(r['x']) for r in rob])
    ry = np.array([float(r['y']) for r in rob])
    rv = np.array([abs(float(r.get('vx') or 0.0)) for r in rob])
    worst = (1e9, None, None, None)
    names = sorted({r['name'] for r in obj})
    for name in names:
        rows = [r for r in obj if r['name'] == name]
        at = np.array([float(r['stamp']) for r in rows])
        order = np.argsort(at)
        at = at[order]
        ax = np.array([float(r['x']) for r in rows])[order]
        ay = np.array([float(r['y']) for r in rows])[order]
        # the actor's position at each robot sample, from its own 10 Hz track
        inside = (rt >= at[0]) & (rt <= at[-1])
        if not inside.any():
            continue
        ix = np.interp(rt[inside], at, ax)
        iy = np.interp(rt[inside], at, ay)
        d = np.hypot(rx[inside] - ix, ry[inside] - iy) - robot_radius - person_radius
        k = int(np.argmin(d))
        if d[k] < worst[0]:
            worst = (float(d[k]), name, float(rt[inside][k]), float(rv[inside][k]))
    if worst[1] is None:
        return {'min_actor_clearance_m': None,
                'note': 'actor and robot GT do not overlap in time'}
    return {'min_actor_clearance_m': round(worst[0], 3),
            'nearest_actor': worst[1], 'actor_at_stamp': round(worst[2], 2),
            'robot_speed_then_m_s': round(worst[3], 2),
            'actors_tracked': len(names)}


def clearance(robot_csv, static_csv, object_csv=None, radius=0.35):
    """Minimum GT distance from the robot to any object centre, over a run.

    This is a CENTRE-to-centre distance minus a nominal object radius, not a
    surface distance: the world gives model origins, and the meshes' extents are
    frozen in their DAE vertex data rather than declared anywhere readable. So
    read it as a screening measure -- it reliably separates "drove down the
    middle of the aisle" from "drove into a rack" -- and confirm a flagged run
    against the recording before calling it a collision.
    """
    rob = list(csv.DictReader(open(robot_csv)))
    stat = [(r['name'], float(r['x']), float(r['y'])) for r in csv.DictReader(open(static_csv))
            if not r['name'].startswith(('aws_robomaker_warehouse_Ground',
                                         'aws_robomaker_warehouse_Roof',
                                         'ackermann_robot'))]
    worst = (1e9, None, None)
    for r in rob:
        x, y, t = float(r['x']), float(r['y']), float(r['stamp'])
        for name, ox, oy in stat:
            d = math.hypot(x - ox, y - oy) - radius
            if d < worst[0]:
                worst = (d, name, t)
    return {'min_clearance_m': round(worst[0], 3), 'nearest_object': worst[1],
            'at_stamp': worst[2], 'static_objects_considered': len(stat)}


if __name__ == '__main__':
    if '--ros-args' in sys.argv:
        run_node(sys.argv[1:])
        sys.exit(0)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--world')
    ap.add_argument('--static-out')
    ap.add_argument('--clearance', nargs=2, metavar=('ROBOT_CSV', 'STATIC_CSV'))
    a = ap.parse_args()
    if a.world and a.static_out:
        dump_static(a.world, a.static_out)
    elif a.clearance:
        print(clearance(*a.clearance))
    else:
        ap.print_help()
