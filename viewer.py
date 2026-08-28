#!/usr/bin/env python3
"""Lightweight 2D top-down viewer for the warehouse sim: watch and drive, no 3D.

    ros2 launch aws_robomaker_small_warehouse_world small_warehouse.launch.py headless:=True
    python3 ~/wil_project/viewer.py

WHY THIS EXISTS
    The Gazebo GUI is what starves the sim. The package README measures RTF 0.90
    headless against 0.40 with the GUI, and concludes it is the GUI, not the GPU
    choice. Every sensor rate scales with RTF, so the GUI directly costs you IMU
    and camera samples -- measured on this machine at headless RTF ~0.70: /imu
    139 Hz of a nominal 200, /cam0 13.9 of 20.

    So: run gz with -s, and get situational awareness from ROS topics instead. This
    draws the floor with its real texture, the obstacles from the world file (at
    their live pose, with --live-poses, if you have made any of them move), and
    the robot from /ground_truth/odometry, for about 0.1 of one core and zero GPU.
    It never subscribes to an image topic unless you ask for a thumbnail.

    The RTF sparkline in the panel is the point of the whole exercise: it is how
    you confirm, live and on your own machine, that this costs less than what it
    replaces.

CONTROLS
    w/up throttle   s/down brake+reverse   a,d / left,right steer   space handbrake
    - / +           lower / raise the speed cap      , / .  the steering-angle cap
    0               reset both caps to drive.py's defaults
    left-drag       set a Nav2 goal (drag = heading, like RViz's 2D Goal Pose)
    shift-left-drag append a WAYPOINT to a route instead of going now. Shift-click
                    leaves the heading automatic; drag to pin it.
    Enter           plan the route without moving (ComputePathThroughPoses); press
                    again to drive it (NavigateThroughPoses)
    Backspace       drop the last waypoint      Delete  clear the route
    Esc             cancel the goal (or the drag in progress)
    middle-drag     pan          wheel  zoom          f  fit to window
    g grid   o obstacles   m map overlay   t trails   c camera thumbnail
    l                floor markings (bay outlines + walkway lines)
    v VINS overlay   b ORB-SLAM3 overlay
    [ ]             rotate the view by 90 deg (see --rotate)
    r  re-align every estimator to ground truth     q / Ctrl-C  quit

RELATION TO drive.py
    drive.py is untouched and remains the no-GUI option. The teleop here is the same
    integrator and the same constants, imported from it rather than copied, so the
    two cannot drift apart. What differs is key handling: a raw TTY reports only key
    PRESSES, so drive.py has to infer "held" from auto-repeat within a HOLD_WINDOW.
    Qt delivers real releases, so that hack is gone -- but Qt also emits SYNTHETIC
    auto-repeat press/release pairs, so isAutoRepeat() has to be checked in both
    handlers or a held key stutters. See TeleopState.

    To avoid two publishers fighting over /cmd_vel, this one is created LAZILY on
    the first real drive key, and goes quiet after 2 s idle. Leave the viewer alone
    and it never appears in `ros2 topic info /cmd_vel` at all, so you can run
    drive.py alongside it. --no-teleop disables it outright.

THE QoS RULE, WRITTEN OUT BECAUSE IT IS THE BUG YOU WILL HIT
    A subscription gets data only if what it REQUESTS is no stronger than what the
    publisher OFFERS. Requesting RELIABLE from a BEST_EFFORT publisher yields
    silence with no error. So every streaming subscription here asks for
    BEST_EFFORT + VOLATILE, which is compatible with anything.

    Two exceptions, both load-bearing:
      * /map is published ONCE, latched. It must be requested RELIABLE +
        TRANSIENT_LOCAL or you get an empty map forever.
      * the /cmd_vel PUBLISHER must be RELIABLE. ros_gz_bridge's ROS->gz direction
        creates a RELIABLE subscription (it does not use a sensor-data profile), and
        a BEST_EFFORT publisher would be incompatible -- the robot would never move.

    Every subscription also carries an incompatible-QoS callback, and the panel
    separates "no publisher" from "publisher present, 0 Hz", so a mismatch names
    itself instead of looking like a dead sim.
"""

import argparse
import math
import os
import signal
import sys
import threading
import time
import traceback
from collections import deque

import numpy as np

# --- geometry, shared with the map baker ---------------------------------------
# bake_map.py lives next to this file and already solves world parsing, COLLADA
# loading and the node-matrix/unit transform chain -- including the fact that
# several meshes declare Z_UP while their raw vertices are Y-up, which their node
# matrix rotates. Importing it means the viewer and the map nav2 plans on cannot
# disagree about where anything is. A hardcoded footprint table would rot silently.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bake_map  # noqa: E402

# drive.py is imported, not copied, for the same reason: one definition of the
# vehicle. It guards its own main() behind __main__, so importing is side-effect
# free -- it does not touch rclpy or the terminal at import time.
import drive  # noqa: E402

PKG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'aws-robomaker-small-warehouse-world')

# Floor extent and texture mapping, fitted from the top face of the GroundB visual
# DAE:  u = y/TILE_U + U0,  v = x/TILE_V + V0  (refit residual 6.7e-07).
#
# The tile is ANISOTROPIC since scale_warehouse.py squared the building: geometry
# went x3.0105741651 in x and x2 in y, while the concrete UVs went x2 uniformly.
# So y keeps the stock 6.090153 m tile and x is stretched by 3.0105741651/2 =
# 1.5052871.  Re-run the fit if you rescale again -- these are not derived at
# runtime. Stock values were FLOOR_X +-6.990235, FLOOR_Y +-10.453333,
# TILE 6.090153 both axes, U0/V0 2.938774/3.719847.
FLOOR_X0, FLOOR_X1 = -21.044621, 21.044621
FLOOR_Y0, FLOOR_Y1 = -20.906665, 20.906665
TILE_U = 6.090153                 # metres per tile along y
TILE_V = 9.167420                 # metres per tile along x (= TILE_U * 1.5052871)
U0, V0 = 5.877548, 7.439694
TEXTURE = os.path.join(PKG, 'models', 'aws_robomaker_warehouse_GroundB_01',
                       'materials', 'textures', 'aws_robomaker_warehouse_GroundB_01.png')
# The painted floor markings (bay outlines, walkway lines) are a SECOND material on
# the GroundB visual mesh, texture-mapped into a 4-band colour atlas. They are real
# geometry, not part of the concrete tile, so they have to be rasterised separately.
GROUND_DAE = os.path.join(PKG, 'models', 'aws_robomaker_warehouse_GroundB_01', 'meshes',
                          'aws_robomaker_warehouse_GroundB_01_visual.DAE')
ATLAS_TEX = os.path.join(PKG, 'models', 'aws_robomaker_warehouse_GroundB_01',
                         'materials', 'textures', 'aws_robomaker_warehouse_GroundB_02.png')
MARK_MATERIAL = 'Material #946569'
WORLD = os.path.join(PKG, 'worlds', 'small_warehouse', 'small_warehouse.world')
MODELS = os.path.join(PKG, 'models')

# Robot collision extent about base_footprint, from model.sdf: the +x reach is the
# camera boom, the width is the wheels (0.42 separation + 0.065 wheel length).
ROBOT_X0, ROBOT_X1 = -0.300, 0.506
ROBOT_HALF_W = 0.2425

# Only obstacles that matter at robot height are drawn solid. Same band the map is
# baked with, so the drawing and the costmap agree.
Z_LO, Z_HI = 0.05, 2.00

# Bounds for the runtime teleop limit keys. drive.py's TOP_SPEED/STEER_LIMIT are the
# starting values; these are how far the keys may move them.
LIMIT_STEP = 1.1                  # per keypress, like teleop_twist_keyboard's 10%
SPEED_CAP_MIN = 0.2               # m/s
SPEED_CAP_MAX = 10.0              # m/s, the launch arg max_speed's own default. Past
                                  # that the AckermannSteering plugin clamps anyway,
                                  # so raise max_speed first if you want more.
STEER_CAP_MIN = 2.0 * math.pi / 180.0
# No STEER_CAP_MAX constant: the ceiling is drive.STEER_LIMIT, which IS the model's
# <steering_limit>. Above it the plugin silently clamps and the robot under-turns
# with no error anywhere, so there is nothing up there worth having.


def hull2d(points):
    """Convex hull, Andrew's monotone chain. Avoids a scipy import for 20 lines."""
    pts = sorted(set(map(tuple, np.round(points, 4))))
    if len(pts) < 3:
        return np.array(pts)

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2:
                (ax, ay), (bx, by) = out[-2], out[-1]
                if (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax) > 0:
                    break
                out.pop()
            out.append(p)
        return out[:-1]

    return np.array(half(pts) + half(reversed(pts)))


def load_obstacles(world_path=WORLD, models=MODELS):
    """[(label, Nx2 world-frame polygon, (x0, y0, yaw0)), ...] at robot height.

    The third element is the pose the WORLD FILE authored, which is what the polygon
    was baked at. Keeping it is what lets a live pose feed move the footprint later
    (MapView._live_poly) without re-reading any meshes: for a rigid body, polygon at
    the new pose = polygon - p0, rotated by yaw1-yaw0, + p1.

    Each mesh instance becomes its own convex hull. That is exact for the crates and
    correct-under-rotation, and it beats an axis-aligned box for the several models
    the world places at a yaw of ~+-90 degrees.

    Two deliberate imprecisions, both cosmetic and both visible in the --map overlay
    if you care: ShelfF's 0.15 m mid-aisle gap gets filled in by the hull, and the
    wall is drawn from its bounding box as a ring rather than hulled (its hull would
    be a filled rectangle covering the entire floor, which is worse than useless).
    """
    out = []
    wall = None
    for model, name, pose in bake_map.load_world(world_path):
        if 'ackermann_robot' in model or 'GroundB' in model or 'RoofB' in model:
            continue
        for col in bake_map.load_collisions(models, model):
            if col[0] == 'mesh':
                _, path, scale, off = col
                if not os.path.isfile(path):
                    continue
                verts, _ = bake_map.load_dae(path)
                wv = bake_map.transform(verts, off, scale, pose)
                if wv[:, 2].max() < Z_LO or wv[:, 2].min() > Z_HI:
                    continue
                if 'WallB' in model:
                    wall = (wv[:, 0].min(), wv[:, 1].min(), wv[:, 0].max(), wv[:, 1].max())
                    continue
                h = hull2d(wv[:, :2])
                if len(h) >= 3:
                    out.append((name, h, (pose[0], pose[1], pose[5])))
            elif col[0] == 'sphere':
                _, r, off = col
                cx, cy, cz = pose[0] + off[0], pose[1] + off[1], pose[2] + off[2]
                dz = abs(min(max(cz, Z_LO), Z_HI) - cz)
                if dz >= r:
                    continue
                rr = math.sqrt(r * r - dz * dz)
                a = np.linspace(0, 2 * math.pi, 24, endpoint=False)
                out.append((name, np.stack([cx + rr * np.cos(a),
                                            cy + rr * np.sin(a)], 1),
                            (pose[0], pose[1], pose[5])))
    return out, wall


# --- rate / RTF meters ---------------------------------------------------------

class RateMeter(object):
    """Arrival rate over a sliding WALL-clock window.

    Wall, not sim. This measures what actually reaches this process. Measured on sim
    time, /imu would read a flat 200 Hz even at RTF 0.1 -- which is precisely the
    information the panel exists to show.
    """

    def __init__(self, expected, window=3.0):
        self.expected = expected
        self.window = window
        self.buf = deque()

    def tick(self):
        now = time.monotonic()
        self.buf.append(now)
        while self.buf and now - self.buf[0] > self.window:
            self.buf.popleft()

    def hz(self):
        if len(self.buf) < 3:
            return None
        span = self.buf[-1] - self.buf[0]
        return (len(self.buf) - 1) / span if span > 1e-6 else None

    def stale(self):
        return not self.buf or (time.monotonic() - self.buf[-1]) > 2.0


class RtfMeter(object):
    """Real-time factor = d(sim)/d(wall).

    The wall clock must be monotonic(): time.time() steps under NTP and would show
    RTF spikes that never happened. The sim clock must come from the /clock message
    itself rather than node.get_clock().now(), because the latter gives no way to
    tell "paused" from "never connected".
    """

    WINDOW = 2.0

    def __init__(self):
        self.buf = deque()
        self.history = deque(maxlen=120)     # ~60 s of sparkline at 2 Hz
        self._last_hist = 0.0

    def add(self, sim_t):
        now = time.monotonic()
        # Sim time can go BACKWARDS -- gz reset, a relaunch while the viewer stays
        # up, or a bag jumping. Without this the window straddles the discontinuity
        # and reports a large negative RTF, which then propagates into the panel's
        # expected_hz * RTF and prints rows like "100% of -9018".
        if self.buf and sim_t < self.buf[-1][1]:
            self.buf.clear()
            self.history.clear()
        self.buf.append((now, sim_t))
        while len(self.buf) > 1 and now - self.buf[0][0] > self.WINDOW:
            self.buf.popleft()
        if now - self._last_hist >= 0.5:
            self._last_hist = now
            r = self.rtf()
            if r is not None:
                self.history.append(r)

    def rtf(self):
        if len(self.buf) < 2:
            return None
        dw = self.buf[-1][0] - self.buf[0][0]
        ds = self.buf[-1][1] - self.buf[0][1]
        if dw <= 0.2 or ds < 0.0:
            return None
        return ds / dw

    def stale(self):
        return not self.buf or (time.monotonic() - self.buf[-1][0]) > 1.5


class Trail(object):
    """Breadcrumbs, decimated by distance so standing still costs nothing."""

    def __init__(self, min_step=0.02, maxlen=6000):
        self.pts = deque(maxlen=maxlen)
        self.min_step = min_step

    def add(self, x, y):
        if self.pts:
            lx, ly = self.pts[-1]
            if (x - lx) ** 2 + (y - ly) ** 2 < self.min_step ** 2:
                return False
        self.pts.append((x, y))
        return True

    def clear(self):
        self.pts.clear()


# --- ROS side ------------------------------------------------------------------

import rclpy                                                        # noqa: E402
from geometry_msgs.msg import PoseStamped, Twist                    # noqa: E402
from nav_msgs.msg import OccupancyGrid, Odometry, Path              # noqa: E402
from rclpy.executors import SingleThreadedExecutor                  # noqa: E402
from rclpy.time import Time                                         # noqa: E402
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,  # noqa: E402
                       ReliabilityPolicy)
from sensor_msgs.msg import CameraInfo, Image, Imu                  # noqa: E402

try:
    from tf2_msgs.msg import TFMessage
    HAVE_TF_MSGS = True
except ImportError:
    HAVE_TF_MSGS = False

try:
    from nav2_msgs.action import (ComputePathThroughPoses, NavigateThroughPoses,
                                  NavigateToPose)
    from rclpy.action import ActionClient
    HAVE_NAV2 = True
except ImportError:                       # nav2_msgs absent: lose nav, keep the rest
    HAVE_NAV2 = False

# BEST_EFFORT is compatible with any publisher; see the module docstring.
SENSOR = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=5,
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                    durability=DurabilityPolicy.VOLATILE)
LATEST = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                    durability=DurabilityPolicy.VOLATILE)
# Where each estimator's odometry is looked for when its topic is 'auto', in order.
EST_PREFER = {
    'vins': ('/vins_estimator/odometry', '/odometry'),
    # orbslam3_node.cpp publishes the relative "~/odometry", so the topic is
    # /<NODE NAME>/odometry -- and the node name is not the one in the source. All
    # three launch files in orbslam3_ros2/launch pass name='orbslam3', which
    # overrides Node("orbslam3_node"), so the launched node gives /orbslam3/odometry
    # and only a bare `ros2 run` gives /orbslam3_node/odometry. Both are probed.
    'orb': ('/orbslam3/odometry', '/orbslam3_node/odometry'),
}

# /map is latched and published once. VOLATILE here means no map, ever.
MAPQOS = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                    reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL)


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Estimator(object):
    """One pose-estimate overlay: a topic, a trail, and a rigid fit to ground truth.

    VINS-Fusion and ORB-SLAM3 need identical treatment and get one class rather than
    two code paths: both publish nav_msgs/Odometry in their OWN start frame, with an
    arbitrary origin and heading, so neither can be drawn on the warehouse map until
    it has been put there. The fit is the same for both -- wait until the estimate
    has actually moved, then solve the yaw and translation that place it on ground
    truth at that instant, and hold it. Anything else publishing Odometry works too.

    Held under SharedState.lock like everything else the ROS thread writes.
    """

    def __init__(self, key, label, topic, align_mode, trail_min_step):
        self.key = key                    # also the palette key on the Qt side
        self.label = label                # panel row label
        self.topic = topic
        self.align_mode = align_mode
        self.show = True
        self.raw = None                   # (x, y, yaw) in the estimator's own frame
        self.trail = Trail(trail_min_step)
        self.err = None                   # metres from ground truth, aligned
        self.align = None                 # (cos, sin, tx, ty), estimator -> world

    def reset(self):
        self.align = None
        self.err = None
        self.trail.clear()


class SharedState(object):
    """Everything the GUI reads and the ROS thread writes, under one lock.

    Deliberately not Qt signals: /imu alone is 200 msg/s, and marshalling each one
    into the event loop would cost more than the entire renderer. The GUI polls a
    snapshot at 30 Hz instead.
    """

    def __init__(self, args):
        self.lock = threading.Lock()
        self.gt = None                    # (x, y, yaw, vx, wz)
        self.gt_ring = deque(maxlen=400)  # (sim_t, x, y, yaw), for estimator align
        self.sim_t = 0.0
        self.rtf = RtfMeter()
        self.rates = {}
        self.gt_trail = Trail(args.trail_min_step)
        # Topics are resolved later (RosLink probes for 'auto'), so these start with
        # whatever the command line asked for and are corrected in place.
        self.estimators = [
            Estimator('vins', 'VINS', args.vins_topic, args.align,
                      args.trail_min_step),
            Estimator('orb', 'ORB3', args.orb_topic, args.align,
                      args.trail_min_step),
        ]
        self.plan = None                  # Nx2
        self.grid = None                  # (np.uint8 HxW, res, ox, oy)
        self.thumb = None                 # HxWx3 uint8
        self.goal = None                  # (x, y, yaw)
        # A route queued up before anything moves. yaw None means "auto": resolved
        # at send time to the bearing THROUGH the point, because a Dubins planner
        # has to arrive at a specific heading and an arbitrary via-point yaw is what
        # makes an otherwise reachable route unplannable. Qt thread owns this list.
        self.route = []                   # [(x, y, yaw or None)]
        self.route_preview = None         # Nx2, result of ComputePathThroughPoses
        self.route_msg = ''
        # Live model poses, name -> (x, y, yaw), from the gz dynamic pose feed. Only
        # non-static entities appear, so this is normally tiny. Empty dict = the feed
        # is off, and every footprint stays where the world file put it.
        self.model_poses = {}
        self.nav_state = 'idle'
        self.nav_msg = ''
        self.warnings = deque(maxlen=6)
        self.teleop = 'off'

    def warn(self, text):
        with self.lock:
            if not self.warnings or self.warnings[-1] != text:
                self.warnings.append(text)
        print('[viewer] ' + text, file=sys.stderr)


class RosLink(object):
    """The rclpy node. Every method here runs on the ROS thread except send_goal /
    cancel_goal, which only set flags that the ROS thread acts on."""

    def __init__(self, args, state):
        self.args = args
        self.state = state
        # use_sim_time is deliberately NOT set, and we do not subscribe to /clock.
        # MEASURED on this machine: rclpy costs ~2.2 ms of CPU per message, and gz
        # publishes /clock at ~700 Hz real-time-equivalent -- so a single /clock
        # subscription is 70% of a core on its own, and use_sim_time=True creates
        # one internally whether you want it or not. That one line was the whole
        # reason an early build of this viewer cost 118% of a core and pushed RTF
        # from 0.44 down to 0.32, i.e. it was doing the very damage it exists to
        # diagnose.
        #
        # Sim time instead comes from the odometry header stamps, which are already
        # sim-stamped and which we already subscribe to for the pose. 50 Hz is ample
        # for both the teleop integrator (which ticks at 50 Hz) and for RTF.
        self.node = rclpy.create_node('warehouse_viewer')
        log = self.node.get_logger()
        self.t_start = time.monotonic()

        self.expected = {}
        self._sub(Odometry, args.gt_topic, self.on_gt, LATEST, 50.0)
        if args.watch_imu:
            # 200 Hz ~ 45% of a core just to count messages. Opt-in, because RTF
            # already tells you what this row would: every rate scales with it
            # (measured 100-103% of expected*RTF across all of them).
            self._sub(Imu, args.imu_topic, self.on_imu, SENSOR, 200.0)
        for t in args.cam_info_topics.split(','):
            t = t.strip()
            if t:
                # bind the topic into the callback; deriving it from frame_id at
                # runtime would attribute both cameras to whichever guessed first
                self._sub(CameraInfo, t, lambda m, tt=t: self._rate(tt), SENSOR, 20.0)

        # Resolve estimator topics rather than guess. For VINS: vins_estimator.cpp
        # constructs rclcpp::Node("vins_estimator") with NO namespace and publishes
        # the relative name "odometry", which resolves to /odometry -- but
        # carla_rtab_direct.cpp uses the absolute /vins_estimator/odometry. Both are
        # in circulation. ORB-SLAM3 is unambiguous by comparison: orbslam3_node.cpp
        # publishes "~/odometry" from Node("orbslam3_node").
        #
        # 'auto' probing must never let two estimators land on the SAME topic, so
        # each probe skips every other estimator's resolved or requested topic --
        # otherwise running ORB-SLAM3 without VINS would have VINS's fallback branch
        # ('any other Odometry publisher') quietly adopt ORB-SLAM3's feed and draw
        # the same trail twice under two names.
        # Probing here and giving up would be wrong: __init__ runs before the
        # executor spins, so DDS discovery has not finished and count_publishers()
        # reports 0 for topics that are in fact being published. Subscribe to the
        # explicit topics now and keep re-probing the 'auto' ones on a timer, which
        # also covers the ordinary case of starting the viewer before the estimator.
        with state.lock:
            ests = list(state.estimators)
        self._pending = []
        for est in ests:
            if est.topic == 'auto':
                self._pending.append(est)
            elif est.topic:
                self._sub(Odometry, est.topic,
                          lambda m, e=est: self.on_est(m, e), LATEST, 20.0)
                log.info('%s topic: %s' % (est.label, est.topic))
        self._resolve_until = time.monotonic() + 20.0
        if self._pending:
            self.node.create_timer(1.0, self._resolve_estimators)

        if args.live_poses:
            if not HAVE_TF_MSGS:
                state.warn('tf2_msgs missing; --live-poses ignored')
            else:
                # ~60 Hz nominal, and it scales with RTF like every other sim-driven
                # rate, so the panel judges it against expected*RTF as usual.
                self._sub(TFMessage, args.pose_topic, self.on_poses, SENSOR, 60.0)

        if args.map_spec == 'ros':
            self._sub(OccupancyGrid, '/map', self.on_map, MAPQOS, None)
        self._sub(Path, args.plan_topic, self.on_plan, LATEST, None)

        self.cmd_pub = None               # created lazily; see the docstring
        self.thumb_sub = None

        self.nav = None
        self.nav_route = None
        self.plan_route = None
        if HAVE_NAV2 and not args.no_nav:
            self.nav = ActionClient(self.node, NavigateToPose, args.nav_action)
            self.nav_route = ActionClient(self.node, NavigateThroughPoses,
                                          args.nav_through_action)
            self.plan_route = ActionClient(self.node, ComputePathThroughPoses,
                                           args.compute_route_action)
        self._goal_req = None
        self._route_req = None            # ('preview'|'execute', [(x, y, yaw), ...])
        self._cancel_req = False
        self._goal_handle = None
        self.node.create_timer(0.5, self._service_nav)
        self.node.create_timer(2.0, self._watchdog)

    # -- subscription helper ----------------------------------------------------
    def _sub(self, msg_type, topic, cb, qos, expected_hz, rate=False):
        # expected_hz None + rate True: show the rate but pass no judgement on it.
        # /clock is the case -- gz publishes it at a rate that is a function of the
        # step size and its own throttling, not something with a nominal value.
        if expected_hz is not None or rate:
            with self.state.lock:
                self.state.rates[topic] = RateMeter(expected_hz)
            self.expected[topic] = expected_hz

        def guarded(m, _cb=cb, _t=topic):
            # A raise here kills the executor thread, and the viewer then paints a
            # frozen world with no sign anything is wrong. Never let that happen.
            try:
                _cb(m)
            except Exception:
                self.state.warn('%s callback: %s' % (
                    _t, traceback.format_exc(limit=1).strip().replace('\n', ' ')))

        try:
            from rclpy.event_handler import SubscriptionEventCallbacks
        except ImportError:
            from rclpy.qos_event import SubscriptionEventCallbacks

        def on_bad_qos(status, _t=topic):
            self.state.warn('QoS MISMATCH on %s (%d incompatible publisher(s)) -- '
                            'this subscription will receive NOTHING' % (_t, status.total_count))

        return self.node.create_subscription(
            msg_type, topic, guarded, qos,
            event_callbacks=SubscriptionEventCallbacks(incompatible_qos=on_bad_qos))

    def _resolve_estimators(self):
        """Re-probe 'auto' estimator topics until they appear, then give up quietly."""
        if not self._pending:
            return
        if time.monotonic() > self._resolve_until:
            for est in self._pending:
                with self.state.lock:
                    est.topic = None
                self.node.get_logger().info('%s: no topic found' % est.label)
            self._pending = []
            return
        with self.state.lock:
            taken = set(e.topic for e in self.state.estimators
                        if e.topic not in (None, 'auto'))
        still = []
        for est in self._pending:
            found = self._find_odom(EST_PREFER[est.key], taken)
            if not found:
                still.append(est)
                continue
            with self.state.lock:
                est.topic = found
            taken.add(found)
            self._sub(Odometry, found,
                      lambda m, e=est: self.on_est(m, e), LATEST, 20.0)
            self.node.get_logger().info('%s topic: %s' % (est.label, found))
        self._pending = still

    def _find_odom(self, prefer, taken):
        """First of `prefer` that has a publisher, else any other Odometry topic.

        `taken` is the other estimators' topics: never hand two overlays one feed.
        """
        for cand in prefer:
            if cand not in taken and self.node.count_publishers(cand):
                return cand
        skip = {self.args.gt_topic, '/odom', '/odometry_filtered'} | set(taken)
        skip |= set(c for p in EST_PREFER.values() for c in p) - set(prefer)
        for name, types in self.node.get_topic_names_and_types():
            if 'nav_msgs/msg/Odometry' in types and name not in skip:
                return name
        return None

    # -- callbacks --------------------------------------------------------------
    def _rate(self, topic):
        m = self.state.rates.get(topic)
        if m:
            m.tick()

    def on_gt(self, msg):
        p = msg.pose.pose.position
        if not (math.isfinite(p.x) and math.isfinite(p.y)):
            return                        # one NaN would freeze the trail forever
        yaw = yaw_of(msg.pose.pose.orientation)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self.state.lock:
            self.state.gt = (p.x, p.y, yaw,
                             msg.twist.twist.linear.x, msg.twist.twist.angular.z)
            self.state.gt_ring.append((t, p.x, p.y, yaw))
            self.state.gt_trail.add(p.x, p.y)
            self.state.sim_t = t          # this IS the clock; see __init__
            self.state.rtf.add(t)
        self._rate(self.args.gt_topic)

    def on_est(self, msg, est):
        p = msg.pose.pose.position
        if not (math.isfinite(p.x) and math.isfinite(p.y)):
            return
        yaw = yaw_of(msg.pose.pose.orientation)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self.state.lock:
            st = self.state
            est.raw = (p.x, p.y, yaw)
            if est.align is None and est.align_mode != 'none':
                # Wait until the estimate has actually moved: the first samples are
                # initialisation garbage sitting at the origin. ORB-SLAM3 in
                # particular sits at exactly (0,0,0) until its map initialises.
                if (p.x * p.x + p.y * p.y) >= 0.25 and st.gt_ring:
                    g = min(st.gt_ring, key=lambda r: abs(r[0] - t))
                    d = g[3] - yaw
                    est.align = (math.cos(d), math.sin(d), 0.0, 0.0)
                    ax, ay = self._apply_align(est.align, p.x, p.y)
                    est.align = (math.cos(d), math.sin(d), g[1] - ax, g[2] - ay)
            if est.align is not None:
                wx, wy = self._apply_align(est.align, p.x, p.y)
                est.trail.add(wx, wy)
                if st.gt_ring:
                    # interpolate GT at the ESTIMATE's stamp; comparing against
                    # "latest GT" would add a spurious v*dt term -- 0.4 m at 4 m/s
                    # with 100 ms of lag
                    g = min(st.gt_ring, key=lambda r: abs(r[0] - t))
                    est.err = math.hypot(g[1] - wx, g[2] - wy)
        self._rate(est.topic)

    @staticmethod
    def _apply_align(a, x, y):
        c, s, tx, ty = a
        return c * x - s * y + tx, s * x + c * y + ty

    def on_imu(self, msg):
        self._rate(self.args.imu_topic)

    def on_poses(self, msg):
        self._rate(self.args.pose_topic)
        # child_frame_id is the gz entity name. Link-level entries ('link', a wheel)
        # share names across models and are not obstacles, but they are harmless: no
        # obstacle is called that, so nothing matches and they are simply ignored.
        out = {}
        for tr in msg.transforms:
            t, q = tr.transform.translation, tr.transform.rotation
            out[tr.child_frame_id] = (
                t.x, t.y, math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                     1.0 - 2.0 * (q.y * q.y + q.z * q.z)))
        with self.state.lock:
            self.state.model_poses = out

    def on_map(self, msg):
        w, h = msg.info.width, msg.info.height
        a = np.array(msg.data, dtype=np.int8).reshape(h, w)
        with self.state.lock:
            self.state.grid = (a, msg.info.resolution,
                               msg.info.origin.position.x, msg.info.origin.position.y)

    def on_plan(self, msg):
        pts = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses]) \
            if msg.poses else None
        with self.state.lock:
            self.state.plan = pts

    def on_image(self, msg):
        try:
            n = 3 if msg.encoding in ('rgb8', 'bgr8') else 1
            a = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step // n if n == 1 else msg.step)[:, :msg.width * n]
            a = a.reshape(msg.height, msg.width, n)
            if msg.encoding == 'bgr8':
                a = a[:, :, ::-1]
            k = max(1, self.args.thumb_decimate)
            with self.state.lock:
                self.state.thumb = np.ascontiguousarray(a[::k, ::k])
        except Exception:
            self.state.warn('thumbnail decode failed for encoding %r' % msg.encoding)

    # -- teleop / nav, called from the GUI thread -------------------------------
    def ensure_cmd_pub(self):
        """Lazy so an idle viewer never competes with drive.py for /cmd_vel.
        Publisher creation is not thread-safe against a spinning executor in
        general, but rclpy guards the node's entity list; this is called at most
        once, on a keypress, and matches what drive.py does."""
        if self.cmd_pub is None and not self.args.no_teleop:
            self.cmd_pub = self.node.create_publisher(Twist, self.args.cmd_vel_topic, 10)
        return self.cmd_pub

    def publish_cmd(self, v, wz):
        if self.cmd_pub is not None:
            tw = Twist()
            tw.linear.x = float(v)
            tw.angular.z = float(wz)
            self.cmd_pub.publish(tw)

    def toggle_thumb(self, topic):
        """Create/destroy the image subscription so the bytes only move when the
        thumbnail is actually open."""
        if self.thumb_sub is not None:
            self.node.destroy_subscription(self.thumb_sub)
            self.thumb_sub = None
            with self.state.lock:
                self.state.thumb = None
            return False
        self.thumb_sub = self._sub(Image, topic, self.on_image, LATEST, None)
        return True

    def request_goal(self, x, y, yaw):
        self._goal_req = (x, y, yaw)

    def request_route(self, mode, wps):
        self._route_req = (mode, list(wps))

    def request_cancel(self):
        self._cancel_req = True

    def _service_nav(self):
        if self.nav is None:
            return
        if self._cancel_req:
            self._cancel_req = False
            if self._goal_handle is not None:
                self._goal_handle.cancel_goal_async()
                self._set_nav('cancelling', '')
        if self._route_req is not None:
            mode, wps = self._route_req
            self._route_req = None
            self._service_route(mode, wps)
        req = self._goal_req
        if req is None:
            return
        self._goal_req = None
        if not self.nav.server_is_ready():
            self._set_nav('no server', 'is nav2 running? (%s)' % self.args.nav_action)
            return
        g = NavigateToPose.Goal()
        g.pose = self._pose_stamped(*req)
        self._set_nav('sending', '')
        self.nav.send_goal_async(g, feedback_callback=self._on_fb).add_done_callback(
            self._on_accept)

    def _pose_stamped(self, x, y, yaw):
        ps = PoseStamped()
        ps.header.frame_id = self.args.nav_frame
        # Zero stamp = "use the latest available transform". This node runs on the
        # wall clock (see __init__) while nav2 runs on sim time, so stamping with
        # our own clock would hand nav2 a far-future time and the TF lookup would
        # fail. Zero sidesteps the mismatch and is the normal thing to send.
        ps.header.stamp = Time().to_msg()
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.orientation.z = math.sin(yaw / 2.0)
        ps.pose.orientation.w = math.cos(yaw / 2.0)
        return ps

    def _service_route(self, mode, wps):
        """'preview' plans without moving; 'execute' drives it.

        ComputePathThroughPoses is a query on planner_server -- MEASURED: a 3
        waypoint route returned 281 poses with /ground_truth/odometry byte-identical
        before and after and nothing on /cmd_vel, so previewing is free. Watch the
        field names: the planner action calls them `goals`, the navigator action
        calls them `poses`, and mixing them up is a silent type error at send time.
        """
        poses = [self._pose_stamped(x, y, yaw) for x, y, yaw in wps]
        if mode == 'preview':
            cli = self.plan_route
            if cli is None or not cli.server_is_ready():
                return self._set_route('no planner (%s)'
                                       % self.args.compute_route_action)
            g = ComputePathThroughPoses.Goal()
            g.goals = poses
            g.use_start = False           # plan from wherever the robot is now
            g.planner_id = self.args.planner_id
            self._set_route('planning %d waypoints...' % len(poses))
            cli.send_goal_async(g).add_done_callback(self._on_preview_accept)
        else:
            cli = self.nav_route
            if cli is None or not cli.server_is_ready():
                return self._set_route('no navigator (%s)'
                                       % self.args.nav_through_action)
            g = NavigateThroughPoses.Goal()
            g.poses = poses
            self._set_nav('sending', '%d waypoints' % len(poses))
            self._set_route('executing %d waypoints' % len(poses))
            cli.send_goal_async(
                g, feedback_callback=self._on_route_fb).add_done_callback(
                    self._on_accept)

    def _set_route(self, msg):
        with self.state.lock:
            self.state.route_msg = msg

    def _on_preview_accept(self, fut):
        try:
            gh = fut.result()
        except Exception as e:
            return self._set_route('plan failed: %s' % e)
        if not gh.accepted:
            return self._set_route('plan rejected by planner_server')
        gh.get_result_async().add_done_callback(self._on_preview_result)

    def _on_preview_result(self, fut):
        try:
            res = fut.result()
        except Exception as e:
            return self._set_route('plan failed: %s' % e)
        if res.status != 4:               # 4 = STATUS_SUCCEEDED
            with self.state.lock:
                self.state.route_preview = None
            # Position is rarely the problem; a via-point heading a Dubins path
            # cannot leave from is. See the README's heading-feasibility note.
            return self._set_route('no path -- try a different waypoint heading')
        pts = np.array([[q.pose.position.x, q.pose.position.y]
                        for q in res.result.path.poses], dtype=np.float64)
        if len(pts) < 2:
            with self.state.lock:
                self.state.route_preview = None
            return self._set_route('planner returned an empty path')
        length = float(np.hypot(*(pts[1:] - pts[:-1]).T).sum())
        with self.state.lock:
            self.state.route_preview = pts
        self._set_route('path ok: %.1f m, %d poses -- Enter again to drive'
                        % (length, len(pts)))

    def _on_route_fb(self, fb):
        f = fb.feedback
        self._set_nav('active', '%.1f m left, %d wp to go' % (
            f.distance_remaining, f.number_of_poses_remaining))

    def _set_nav(self, st, msg):
        with self.state.lock:
            self.state.nav_state = st
            self.state.nav_msg = msg

    def _on_accept(self, fut):
        try:
            gh = fut.result()
        except Exception as e:
            return self._set_nav('failed', str(e))
        if not gh.accepted:
            return self._set_nav('rejected', '')
        self._goal_handle = gh
        self._set_nav('active', '')
        gh.get_result_async().add_done_callback(self._on_result)

    def _on_fb(self, fb):
        f = fb.feedback
        self._set_nav('active', '%.1f m left, %.0f s' % (
            f.distance_remaining, f.navigation_time.sec))

    def _on_result(self, fut):
        try:
            status = fut.result().status
        except Exception as e:
            return self._set_nav('failed', str(e))
        self._goal_handle = None
        self._set_nav({4: 'succeeded', 5: 'aborted', 6: 'canceled'}.get(status,
                                                                        'done'), '')

    def _watchdog(self):
        """Distinguish 'nobody is publishing' from 'somebody is, but we get nothing'
        -- the second is a QoS mismatch and the first is a dead sim."""
        if time.monotonic() - self.t_start < 8.0:
            return                        # cameras take a few seconds to come up
        for topic, meter in list(self.state.rates.items()):
            if meter.stale() and self.node.count_publishers(topic) > 0:
                self.state.warn('%s: publisher present but no data '
                                '(QoS? try `ros2 topic info -v %s`)' % (topic, topic))


# --- Qt side -------------------------------------------------------------------

from PyQt5.QtCore import QEvent, QPointF, QRectF, Qt, QTimer          # noqa: E402
from PyQt5.QtGui import (QBrush, QColor, QFont, QImage, QPainter,     # noqa: E402
                         QPen, QPixmap, QPolygonF)
from PyQt5.QtWidgets import QApplication, QWidget                     # noqa: E402

BG = QColor('#22252a')
C_FLOOR_FALLBACK = QColor('#3a3a3a')
C_OBST = QColor(40, 44, 52, 235)
C_OBST_EDGE = QColor(120, 130, 145)
C_WALL = QColor(70, 76, 86)
C_GT = QColor('#4fc3f7')
C_VINS = QColor('#ffb74d')
C_ORB = QColor('#e040fb')      # magenta: cyan reads as the ground-truth trail
                               # (#4fc3f7) at trail width, which is the one
                               # comparison this overlay exists to support
# Palette and dash pattern per estimator key. Distinct dashes as well as distinct
# hues, so the two trails stay tellable apart where they overlap and in a screenshot
# that has lost its colour.
EST_STYLE = {'vins': (C_VINS, Qt.DashLine),
             'orb': (C_ORB, Qt.DotLine)}
C_PLAN = QColor('#81c784')
C_GOAL = QColor('#e57373')
C_ROUTE = QColor('#ba9bf0')
C_ROBOT = QColor('#ffffff')
C_GRID = QColor(255, 255, 255, 26)
C_OK, C_WARN, C_BAD = QColor('#66bb6a'), QColor('#ffa726'), QColor('#ef5350')


def _marking_faces(dae_path, material):
    """[(xy 3x2 metres, uv 3x2), ...] for one material's triangles."""
    import xml.etree.ElementTree as ET
    ns = '{http://www.collada.org/2005/11/COLLADASchema}'
    root = ET.parse(dae_path).getroot()
    unit = root.find('.//%sunit' % ns)
    m_per = float(unit.get('meter')) if unit is not None else 1.0
    arr = {fa.get('id'): [float(x) for x in fa.text.split()]
           for fa in root.iter(ns + 'float_array')}
    pos = next(v for k, v in arr.items() if k.endswith('POSITION-array'))
    uvs = next(v for k, v in arr.items() if k.endswith('UV0-array'))
    out = []
    for tri in root.iter(ns + 'triangles'):
        if tri.get('material') != material:
            continue
        inp = {i.get('semantic'): int(i.get('offset')) for i in tri.findall(ns + 'input')}
        idx = [int(x) for x in tri.find(ns + 'p').text.split()]
        st = max(inp.values()) + 1
        for i in range(0, len(idx), st * 3):
            xy, uv = [], []
            for k in range(3):
                vi = idx[i + k * st + inp['VERTEX']]
                ti = idx[i + k * st + inp['TEXCOORD']]
                xy.append((pos[vi * 3] * m_per, pos[vi * 3 + 1] * m_per))
                uv.append((uvs[ti * 2], uvs[ti * 2 + 1]))
            out.append((np.array(xy), np.array(uv)))
    return out


def paint_markings(img, ppm, flip, dae_path=None, atlas_path=None):
    """Texture-map the floor markings into an already-tiled floor array, in place.

    Barycentric scan-convert per triangle: the strips are thin (0.145 m -> ~6 px at
    the default ppm 40) and there are only ~150 of them, so a per-face bounding-box
    rasteriser is far simpler than anything clever and still runs in milliseconds.

    u is the atlas's band axis -- 0.000-0.342 hazard stripes, 0.342-0.590 green,
    0.590-0.820 blue, 0.820-1.000 yellow -- so sampling with wrap reproduces the
    real paint colours rather than approximating them.
    """
    from PIL import Image as PILImage
    tex = np.asarray(PILImage.open(atlas_path or ATLAS_TEX).convert('RGB'))
    th, tw = tex.shape[:2]
    h, w = img.shape[:2]
    n = 0
    for xy, uv in _marking_faces(dae_path or GROUND_DAE, MARK_MATERIAL):
        cx = (xy[:, 0] - FLOOR_X0) * ppm - 0.5
        ry = (FLOOR_Y1 - xy[:, 1]) * ppm - 0.5
        c0, c1 = max(0, int(np.floor(cx.min()))), min(w - 1, int(np.ceil(cx.max())))
        r0, r1 = max(0, int(np.floor(ry.min()))), min(h - 1, int(np.ceil(ry.max())))
        if c1 < c0 or r1 < r0:
            continue
        cc, rr = np.meshgrid(np.arange(c0, c1 + 1), np.arange(r0, r1 + 1))
        (x1, y1), (x2, y2), (x3, y3) = zip(cx, ry)
        den = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if abs(den) < 1e-9:
            continue
        b1 = ((y2 - y3) * (cc - x3) + (x3 - x2) * (rr - y3)) / den
        b2 = ((y3 - y1) * (cc - x3) + (x1 - x3) * (rr - y3)) / den
        b3 = 1.0 - b1 - b2
        m = (b1 >= -1e-6) & (b2 >= -1e-6) & (b3 >= -1e-6)
        if not m.any():
            continue
        u = b1 * uv[0, 0] + b2 * uv[1, 0] + b3 * uv[2, 0]
        v = b1 * uv[0, 1] + b2 * uv[1, 1] + b3 * uv[2, 1]
        if flip in ('u', 'uv'):
            u = -u
        if flip in ('v', 'uv'):
            v = -v
        uc = (np.mod(u, 1.0) * tw).astype(np.int64) % tw
        vr = (np.mod(v, 1.0) * th).astype(np.int64) % th
        img[rr[m], cc[m]] = tex[vr[m], uc[m]]
        n += 1
    return n


def build_floor_array(path, ppm, flip):
    """Tile the GroundB concrete texture across the floor rect, in world orientation.

    The UV fit says u depends on y and v depends on x, so the texture is effectively
    TRANSPOSED relative to the image as stored -- hence the np.ix_ + transpose rather
    than a plain tile. Row 0 of the result is max y, i.e. already screen orientation.

    The V direction is the one thing about this that is not certain: COLLADA's UV
    origin is bottom-left while an image's row 0 is the top, so whether +v is +row
    depends on the loader. Since this is a repeating floor pattern the consequence
    is a mirrored pattern, not a wrong scale or offset -- --floor-flip settles it
    against a Gazebo screenshot if you care.
    """
    from PIL import Image as PILImage
    tex = np.asarray(PILImage.open(path).convert('RGB'))
    th, tw = tex.shape[:2]

    w = int(round((FLOOR_X1 - FLOOR_X0) * ppm))
    h = int(round((FLOOR_Y1 - FLOOR_Y0) * ppm))
    xs = FLOOR_X0 + (np.arange(w) + 0.5) / ppm
    ys = FLOOR_Y1 - (np.arange(h) + 0.5) / ppm          # row 0 = max y

    u = ys / TILE_U + U0
    v = xs / TILE_V + V0
    if flip in ('u', 'uv'):
        u = -u
    if flip in ('v', 'uv'):
        v = -v
    ucol = (np.mod(u, 1.0) * tw).astype(np.int64) % tw   # -> texture column, per ROW
    vrow = (np.mod(v, 1.0) * th).astype(np.int64) % th   # -> texture row,    per COL

    img = tex[np.ix_(vrow, ucol)].transpose(1, 0, 2)     # (h, w, 3)
    return np.ascontiguousarray(img)


def build_floor_pixmap(path, ppm, flip, markings=False):
    img = build_floor_array(path, ppm, flip)
    if markings:
        paint_markings(img, ppm, flip)
    h, w = img.shape[:2]
    img = np.ascontiguousarray(img)
    qim = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qim)


class View(object):
    """World <-> screen. +y is up on screen, so the y term is negated."""

    def __init__(self):
        self.ppm = 40.0
        self.cx, self.cy = 0.0, 0.0
        self.w, self.h = 1, 1
        # View rotation in degrees, applied by rotating the PAINTER at draw time
        # rather than by baking it into to_screen(). That way the world layers turn
        # while the panel, scale bar and camera thumbnail stay upright, and the
        # cached trail polygons stay valid because they are built in unrotated
        # screen space and rotated along with everything else.
        self.rot = 0.0

    def fit(self, w, h, margin=1.0):
        self.w, self.h = w, h
        if self.rot % 180 == 90:      # quarter turn: the floor lands on its side
            w, h = h, w
        sx = w / (FLOOR_X1 - FLOOR_X0 + margin)
        sy = h / (FLOOR_Y1 - FLOOR_Y0 + margin)
        self.ppm = min(sx, sy)
        self.cx = 0.5 * (FLOOR_X0 + FLOOR_X1)
        self.cy = 0.5 * (FLOOR_Y0 + FLOOR_Y1)

    def to_screen(self, x, y):
        return (self.w * 0.5 + (x - self.cx) * self.ppm,
                self.h * 0.5 - (y - self.cy) * self.ppm)

    def unrotate(self, sx, sy):
        """Widget point -> the unrotated screen space to_screen() works in.
        Qt's rotate(a) maps (x,y)->(x cos a - y sin a, x sin a + y cos a) about the
        origin, so undoing it is the same rotation by -a about the widget centre."""
        if not self.rot:
            return sx, sy
        a = math.radians(-self.rot)
        c, sn = math.cos(a), math.sin(a)
        dx, dy = sx - self.w * 0.5, sy - self.h * 0.5
        return self.w * 0.5 + c * dx - sn * dy, self.h * 0.5 + sn * dx + c * dy

    def rotate_pt(self, sx, sy):
        """Inverse of unrotate(): unrotated screen space -> widget point. Needed to
        put UPRIGHT text (the waypoint numbers) over a rotated world layer."""
        if not self.rot:
            return sx, sy
        a = math.radians(self.rot)
        c, sn = math.cos(a), math.sin(a)
        dx, dy = sx - self.w * 0.5, sy - self.h * 0.5
        return self.w * 0.5 + c * dx - sn * dy, self.h * 0.5 + sn * dx + c * dy

    def unrotate_delta(self, dx, dy):
        """Same, for a drag vector: rotate it, do not translate it."""
        if not self.rot:
            return dx, dy
        a = math.radians(-self.rot)
        c, sn = math.cos(a), math.sin(a)
        return c * dx - sn * dy, sn * dx + c * dy

    def to_world(self, sx, sy):
        sx, sy = self.unrotate(sx, sy)
        return (self.cx + (sx - self.w * 0.5) / self.ppm,
                self.cy - (sy - self.h * 0.5) / self.ppm)

    def poly(self, pts):
        p = QPolygonF()
        cx, cy, ppm = self.cx, self.cy, self.ppm
        hw, hh = self.w * 0.5, self.h * 0.5
        for x, y in pts:
            p.append(QPointF(hw + (x - cx) * ppm, hh - (y - cy) * ppm))
        return p


class MapView(QWidget):

    def __init__(self, args, state, ros):
        super().__init__()
        self.args, self.state, self.ros = args, state, ros
        self.setWindowTitle('warehouse viewer')
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        # We repaint every pixel from static_pm, so stop Qt erasing first.
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setAutoFillBackground(False)
        # Constructed once. Building a QFont per drawText makes fontconfig do a
        # family lookup on every call, and the panel has ~20 of them per frame.
        self._font = QFont('monospace', 9)
        self._font_big = QFont('monospace', 13, QFont.Bold)
        self._gt_poly = QPolygonF()
        self._gt_n = 0
        with state.lock:
            keys = [e.key for e in state.estimators]
        self._est_poly = dict((k, [QPolygonF(), 0]) for k in keys)
        self.resize(1200, 900)

        self.view = View()
        self.view.rot = float(args.rotate)
        self.show_grid = True
        self.show_obst = True
        self.show_map = args.map_spec != 'none'
        self.show_trails = True

        self.obstacles, self.wall = [], None
        try:
            self.obstacles, self.wall = load_obstacles(args.world)
            print('[viewer] %d obstacle footprints from %s'
                  % (len(self.obstacles), os.path.basename(args.world)))
        except Exception as e:
            state.warn('could not parse world (%s); obstacles hidden' % e)

        self.floor_pm = None
        self.floor_marked_pm = None
        self.show_marks = not args.no_markings
        if not args.no_floor:
            try:
                self.floor_pm = build_floor_pixmap(args.texture, args.floor_ppm,
                                                   args.floor_flip)
            except Exception as e:
                state.warn('floor texture unavailable (%s); flat fill' % e)
            if self.floor_pm is not None and not args.no_markings:
                # Painted markings are baked into a SECOND pixmap rather than drawn
                # per frame: they are static, and blitting one image costs nothing
                # while scan-converting 150 triangles every repaint would not.
                try:
                    img = build_floor_array(args.texture, args.floor_ppm, args.floor_flip)
                    n = paint_markings(img, args.floor_ppm, args.floor_flip)
                    h, w = img.shape[:2]
                    img = np.ascontiguousarray(img)
                    self.floor_marked_pm = QPixmap.fromImage(
                        QImage(img.data, w, h, 3 * w, QImage.Format_RGB888).copy())
                    print('[viewer] %d floor-marking triangles rasterised' % n)
                except Exception as e:
                    state.warn('floor markings unavailable (%s)' % e)
                    self.show_marks = False

        self.static_pm = None
        self.dirty = True
        self._drag = None                 # ('goal'|'pan', ...)
        self._goal_drag = None
        self._drag_wp = False             # shift held: append to the route instead
        self._live = frozenset()          # obstacle names the pose feed is moving
        self._cursor = None

        # teleop, ported from drive.py -- same constants, same integrator
        self.held = set()
        self.repeat_seen = set()
        self.last_evt = {}
        self.speed = 0.0
        self.steer = 0.0
        # Copies, not module mutation: drive.py is imported, not forked, and writing
        # its globals here would change what a drive.py running alongside does.
        self.max_speed = drive.TOP_SPEED
        self.steer_cap = drive.STEER_LIMIT
        self._rev_ratio = drive.REVERSE_SPEED / drive.TOP_SPEED
        self.prev_sim = None
        self.quiet = True
        self.armed = False
        self._idle_since = None
        self._warned_clock = False

        self.paint_ms = deque(maxlen=60)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_tick)
        self.timer.start(int(drive.TICK * 1000))
        self.repaint_timer = QTimer(self)
        self.repaint_timer.timeout.connect(self.update)
        self.repaint_timer.start(int(1000 / max(1, args.fps)))

    # -- view -------------------------------------------------------------------
    def _rotate_painter(self, p):
        """Turn the painter about the widget centre for the world-space layers."""
        if self.view.rot:
            p.translate(self.width() * 0.5, self.height() * 0.5)
            p.rotate(self.view.rot)
            p.translate(-self.width() * 0.5, -self.height() * 0.5)

    def resizeEvent(self, e):
        if self.static_pm is None:
            self.view.fit(self.width(), self.height())
        self.view.w, self.view.h = self.width(), self.height()
        self.dirty = True

    def rebuild_static(self, smooth=True):
        """Everything that only changes on pan/zoom/toggle, drawn once into a
        screen-space pixmap. paintEvent then blits it 1:1, which is the difference
        between 0.6 ms and 13 ms a frame."""
        pm = QPixmap(max(1, self.width()), max(1, self.height()))
        pm.fill(BG)
        p = QPainter(pm)
        p.setRenderHint(QPainter.SmoothPixmapTransform, smooth)
        p.setRenderHint(QPainter.Antialiasing, smooth)
        v = self.view
        p.save()
        self._rotate_painter(p)

        floor_pm = (self.floor_marked_pm
                    if self.show_marks and self.floor_marked_pm is not None
                    else self.floor_pm)
        if floor_pm is not None:
            tl = v.to_screen(FLOOR_X0, FLOOR_Y1)
            br = v.to_screen(FLOOR_X1, FLOOR_Y0)
            p.drawPixmap(QRectF(tl[0], tl[1], br[0] - tl[0], br[1] - tl[1]),
                         floor_pm, QRectF(floor_pm.rect()))
        else:
            p.fillRect(v.poly([(FLOOR_X0, FLOOR_Y0), (FLOOR_X1, FLOOR_Y0),
                               (FLOOR_X1, FLOOR_Y1), (FLOOR_X0, FLOOR_Y1)]
                              ).boundingRect(), C_FLOOR_FALLBACK)

        if self.show_grid:
            p.setPen(QPen(C_GRID, 1))
            x = math.ceil(FLOOR_X0)
            while x <= FLOOR_X1:
                a, b = v.to_screen(x, FLOOR_Y0), v.to_screen(x, FLOOR_Y1)
                p.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
                x += 1
            y = math.ceil(FLOOR_Y0)
            while y <= FLOOR_Y1:
                a, b = v.to_screen(FLOOR_X0, y), v.to_screen(FLOOR_X1, y)
                p.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
                y += 1

        if self.show_obst:
            live = self._live
            p.setPen(QPen(C_OBST_EDGE, 1))
            p.setBrush(QBrush(C_OBST))
            for name, poly, _ in self.obstacles:
                if name in live:          # drawn per-frame instead; see paintEvent
                    continue
                p.drawPolygon(v.poly(poly))
            if self.wall:
                x0, y0, x1, y1 = self.wall
                t = 0.13
                p.setBrush(QBrush(C_WALL))
                p.setPen(Qt.NoPen)
                for rect in (((x0, y0), (x1, y0), (x1, y0 + t), (x0, y0 + t)),
                             ((x0, y1 - t), (x1, y1 - t), (x1, y1), (x0, y1)),
                             ((x0, y0), (x0 + t, y0), (x0 + t, y1), (x0, y1)),
                             ((x1 - t, y0), (x1, y0), (x1, y1), (x1 - t, y1))):
                    p.drawPolygon(v.poly(rect))

        # On top of the obstacles, not under them: the point of this layer is to
        # check the grid nav2 plans on against the true footprints, and underneath
        # it the obstacles hide precisely the cells you came to look at.
        if self.show_map:
            self._draw_grid_overlay(p)
        p.restore()
        p.end()
        self.static_pm = pm
        self.dirty = False

    def _draw_grid_overlay(self, p):
        with self.state.lock:
            g = self.state.grid
        if g is None:
            return
        a, res, ox, oy = g
        occ = (a > 50)
        if not occ.any():
            return
        h, w = occ.shape
        rgba = np.zeros((h, w, 4), np.uint8)
        rgba[occ] = (255, 90, 90, 90)
        rgba = np.ascontiguousarray(rgba[::-1])        # row 0 -> max y
        qim = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
        tl = self.view.to_screen(ox, oy + h * res)
        br = self.view.to_screen(ox + w * res, oy)
        p.drawPixmap(QRectF(tl[0], tl[1], br[0] - tl[0], br[1] - tl[1]),
                     QPixmap.fromImage(qim), QRectF(qim.rect()))

    # -- painting ---------------------------------------------------------------
    def paintEvent(self, e):
        t0 = time.monotonic()
        with self.state.lock:
            poses = dict(self.state.model_poses) if self.state.model_poses else {}
        # Which obstacles the feed is moving decides what belongs in the cached
        # static layer, so this has to be settled BEFORE the rebuild below -- settle
        # it after and the frame a model starts moving draws it twice, once at the
        # authored pose from the stale pixmap and once live. Only a CHANGE of the set
        # costs a rebuild; a model that keeps moving does not.
        live = frozenset()
        if self.show_obst:
            live = self._live_names(poses)
            if live != self._live:
                self._live = live
                self.dirty = True
        if self.dirty or self.static_pm is None:
            self.rebuild_static(smooth=self._drag is None)
            # the trail caches are screen-space, so a view change invalidates them
            self._gt_poly, self._gt_n = QPolygonF(), 0
            for c in self._est_poly.values():
                c[0], c[1] = QPolygonF(), 0
        p = QPainter(self)
        # Blit only what Qt says is damaged. On a full update that is the whole
        # widget, but partial exposes (another window moving over ours) get cheap.
        r = e.rect()
        p.drawPixmap(r, self.static_pm, r)
        p.setRenderHint(QPainter.Antialiasing, True)
        v = self.view

        with self.state.lock:
            st = self.state
            gt = st.gt
            gt_pts = list(st.gt_trail.pts) if self.show_trails else []
            est_pts = [(e.key, list(e.trail.pts))
                       for e in st.estimators if self.show_trails and e.show]
            plan = st.plan
            goal = st.goal
            route = list(st.route)
            route_prev = st.route_preview
            thumb = st.thumb

        p.save()
        self._rotate_painter(p)          # world layers turn; the panel below does not
        if self.show_trails and len(gt_pts) > 1:
            self._gt_n = self._extend(self._gt_poly, self._gt_n, gt_pts)
            p.setPen(QPen(C_GT, 2))
            p.drawPolyline(self._gt_poly)
        for key, pts in est_pts:
            if len(pts) < 2:
                continue
            cache = self._est_poly[key]
            cache[1] = self._extend(cache[0], cache[1], pts)
            colour, dash = EST_STYLE.get(key, (C_VINS, Qt.DashLine))
            p.setPen(QPen(colour, 2, dash))
            p.drawPolyline(cache[0])
        if plan is not None and len(plan) > 1:
            p.setPen(QPen(C_PLAN, 3))
            p.drawPolyline(v.poly(plan))

        if live:
            p.setPen(QPen(C_OBST_EDGE, 1))
            p.setBrush(QBrush(C_OBST))
            for name, poly, p0 in self.obstacles:
                if name in live:
                    p.drawPolygon(v.poly(self._live_poly(poly, p0, poses[name])))

        if route_prev is not None and len(route_prev) > 1:
            p.setPen(QPen(C_ROUTE, 3, Qt.DashLine))
            p.drawPolyline(v.poly(route_prev))
        if route:
            wps = self._draw_route(p, route, gt)
        if goal is not None:
            gx, gy, gyaw = goal
            sx, sy = v.to_screen(gx, gy)
            p.setPen(QPen(C_GOAL, 2))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(sx, sy), 9, 9)
            ex, ey = v.to_screen(gx + 0.7 * math.cos(gyaw), gy + 0.7 * math.sin(gyaw))
            p.drawLine(QPointF(sx, sy), QPointF(ex, ey))

        if self._goal_drag is not None:
            (x0, y0), (x1, y1) = self._goal_drag
            p.setPen(QPen(C_GOAL, 2, Qt.DashLine))
            p.drawLine(QPointF(*v.to_screen(x0, y0)), QPointF(*v.to_screen(x1, y1)))

        if gt is not None:
            self._draw_robot(p, gt[0], gt[1], gt[2])
        else:
            self._draw_robot(p, 1.8, 9.0, -math.pi / 2, ghost=True)

        p.restore()

        if route:
            p.setFont(self._font)
            p.setPen(QPen(C_ROUTE))
            for i, (x, y, _) in enumerate(wps):
                sx, sy = v.rotate_pt(*v.to_screen(x, y))
                p.drawText(QPointF(sx + 10, sy - 9), str(i + 1))

        if thumb is not None:
            self._draw_thumb(p, thumb)
        self._draw_panel(p)
        self._draw_scalebar(p)
        p.end()
        self.paint_ms.append((time.monotonic() - t0) * 1000.0)

    def _extend(self, poly, n, pts):
        """Append only the points added since last frame. The Trail deque is
        bounded, so when it wraps, n exceeds len(pts) and we rebuild from scratch."""
        if n > len(pts):
            poly.clear()
            n = 0
        v = self.view
        hw, hh, ppm = v.w * 0.5, v.h * 0.5, v.ppm
        for x, y in pts[n:]:
            poly.append(QPointF(hw + (x - v.cx) * ppm, hh - (y - v.cy) * ppm))
        return len(pts)

    @staticmethod
    def _live_poly(poly, p0, p1):
        """Rigid-transform a footprint baked at authored pose p0 to live pose p1.

        p1 None (no feed, or this model is static and so never appears in the feed)
        returns the polygon untouched, which is the normal case.
        """
        if p1 is None:
            return poly
        dyaw = p1[2] - p0[2]
        c, sn = math.cos(dyaw), math.sin(dyaw)
        dx, dy = poly[:, 0] - p0[0], poly[:, 1] - p0[1]
        return np.stack([p1[0] + c * dx - sn * dy,
                         p1[1] + sn * dx + c * dy], 1)

    def _live_names(self, poses):
        """Obstacles the feed is actually moving.

        Compared against the AUTHORED pose, not against the previous frame: a model
        that is non-static but sitting still is in the feed every tick, and treating
        it as live would move it out of the cached static pixmap for no reason.
        """
        if not poses:
            return frozenset()
        out = set()
        for name, _, p0 in self.obstacles:
            p1 = poses.get(name)
            if p1 is not None and (abs(p1[0] - p0[0]) > 0.02 or
                                   abs(p1[1] - p0[1]) > 0.02 or
                                   abs(p1[2] - p0[2]) > 0.02):
                out.add(name)
        return frozenset(out)

    def _draw_route(self, p, route, gt):
        """Waypoints as clicked, with the leg order dotted in so the route reads
        before it has been planned. Returns the resolved list for the label pass."""
        v = self.view
        wps = self._resolve_route(route, gt)
        pts = [v.to_screen(x, y) for x, y, _ in wps]
        chain = ([v.to_screen(gt[0], gt[1])] if gt is not None else []) + pts
        p.setPen(QPen(C_ROUTE, 1, Qt.DotLine))
        for a, b in zip(chain, chain[1:]):
            p.drawLine(QPointF(*a), QPointF(*b))
        p.setBrush(Qt.NoBrush)
        for (x, y, yaw), (sx, sy), src in zip(wps, pts, route):
            # dashed ring = heading was auto-derived, solid = you dragged it
            p.setPen(QPen(C_ROUTE, 2,
                          Qt.DashLine if src[2] is None else Qt.SolidLine))
            p.drawEllipse(QPointF(sx, sy), 7, 7)
            ex, ey = v.to_screen(x + 0.6 * math.cos(yaw), y + 0.6 * math.sin(yaw))
            p.drawLine(QPointF(sx, sy), QPointF(ex, ey))
        return wps

    def _draw_robot(self, p, x, y, yaw, ghost=False):
        c, s = math.cos(yaw), math.sin(yaw)
        box = [(ROBOT_X0, -ROBOT_HALF_W), (ROBOT_X1, -ROBOT_HALF_W),
               (ROBOT_X1, ROBOT_HALF_W), (ROBOT_X0, ROBOT_HALF_W)]
        world = [(x + c * bx - s * by, y + s * bx + c * by) for bx, by in box]
        p.setPen(QPen(C_ROBOT, 2, Qt.DashLine if ghost else Qt.SolidLine))
        p.setBrush(Qt.NoBrush if ghost else QBrush(QColor(255, 255, 255, 70)))
        p.drawPolygon(self.view.poly(world))
        # heading tick from the rear axle forward, so "which way is front" is
        # unambiguous even when the robot is only a few pixels long
        a = self.view.to_screen(x, y)
        b = self.view.to_screen(x + 0.9 * c, y + 0.9 * s)
        p.setPen(QPen(C_ROBOT, 2))
        p.drawLine(QPointF(*a), QPointF(*b))
        if ghost:
            p.setPen(QPen(QColor(200, 200, 200)))
            p.drawText(QPointF(a[0] + 12, a[1] - 12), 'no pose (spawn)')

    def _draw_thumb(self, p, img):
        h, w = img.shape[:2]
        if img.ndim == 2:
            img = np.stack([img] * 3, -1)
        img = np.ascontiguousarray(img)
        qim = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        tw = 320
        th = int(tw * h / float(w))
        x, y = self.width() - tw - 12, self.height() - th - 12
        p.setPen(QPen(QColor(0, 0, 0, 160)))
        p.setBrush(QBrush(QColor(0, 0, 0, 160)))
        p.drawRect(x - 3, y - 3, tw + 6, th + 6)
        p.drawPixmap(QRectF(x, y, tw, th), QPixmap.fromImage(qim), QRectF(qim.rect()))

    def _draw_scalebar(self, p):
        ppm = self.view.ppm
        for metres in (0.5, 1, 2, 5, 10, 20):
            if metres * ppm > 70:
                break
        px = metres * ppm
        x, y = 14, self.height() - 18
        p.setPen(QPen(QColor(235, 235, 235), 2))
        p.drawLine(int(x), int(y), int(x + px), int(y))
        p.drawLine(int(x), int(y - 4), int(x), int(y + 4))
        p.drawLine(int(x + px), int(y - 4), int(x + px), int(y + 4))
        p.setFont(self._font)
        p.drawText(int(x), int(y - 8), '%g m' % metres)
        if self._cursor:
            p.drawText(int(x + px + 16), int(y),
                       'cursor  x %+.2f  y %+.2f' % self._cursor)

    def _draw_panel(self, p):
        with self.state.lock:
            st = self.state
            rtf = st.rtf.rtf()
            stale = st.rtf.stale()
            hist = list(st.rtf.history)
            rows = [(t, m.hz(), m.expected) for t, m in sorted(st.rates.items())]
            gt = st.gt
            est_rows = [(e.key, e.label, e.err, e.align_mode, e.show,
                         e.align is not None, e.topic, e.raw is not None)
                        for e in st.estimators]
            nav_state, nav_msg = st.nav_state, st.nav_msg
            n_route, route_msg = len(st.route), st.route_msg
            warns = list(st.warnings)
            teleop = st.teleop

        pad, lh, w = 10, 15, 330
        n = (len(rows) + 9 + len(warns) + (1 if (n_route or route_msg) else 0)
             + sum(1 for r in est_rows if r[6]))
        p.setBrush(QBrush(QColor(18, 20, 24, 215)))
        p.setPen(Qt.NoPen)
        p.drawRect(8, 8, w, n * lh + 2 * pad + 26)
        p.setFont(self._font)
        y = 8 + pad + 12

        # RTF headline: this is the number the whole exercise is about
        col = C_BAD if (stale or rtf is None) else (
            C_OK if rtf >= 0.8 else (C_WARN if rtf >= 0.4 else C_BAD))
        p.setPen(QPen(col))
        p.setFont(self._font_big)
        p.drawText(16, y + 4, 'RTF  %s' % ('--' if rtf is None else '%.2f' % rtf))
        p.setFont(self._font)
        if hist and len(hist) > 1:
            x0, y0, sw, sh = 130, y - 8, 190, 16
            p.setPen(QPen(QColor(255, 255, 255, 40)))
            p.drawRect(x0, y0, sw, sh)
            poly = QPolygonF()
            for i, r in enumerate(hist):
                poly.append(QPointF(x0 + sw * i / float(len(hist) - 1),
                                    y0 + sh - sh * min(r, 1.2) / 1.2))
            p.setPen(QPen(col, 1))
            p.drawPolyline(poly)
        y += lh + 8

        p.setPen(QPen(QColor(150, 158, 170)))
        p.drawText(16, y, 'topic                 Hz      of expected')
        y += lh
        for topic, hz, exp in rows:
            # Judge against expected*RTF: a topic keeping perfect pace with a
            # half-speed sim is healthy, not half-broken.
            if hz is None:
                col, txt = C_BAD, '   --   no data'
            elif exp is None or not rtf or rtf <= 0.0:
                # no usable RTF yet: show the rate, pass no judgement on it
                col, txt = QColor(150, 158, 170), '%7.1f  --' % hz
            else:
                target = exp * rtf
                frac = hz / target if target > 0 else 1.0
                col = C_OK if frac >= 0.85 else (C_WARN if frac >= 0.6 else C_BAD)
                txt = '%7.1f  %3.0f%% of %.0f' % (hz, 100 * frac, target)
            p.setPen(QPen(col))
            p.drawText(16, y, '%-20s %s' % (topic[:20], txt))
            y += lh

        y += 4
        p.setPen(QPen(QColor(210, 215, 225)))
        if gt:
            p.drawText(16, y, 'pose  x %+.2f  y %+.2f  yaw %+.1f deg'
                       % (gt[0], gt[1], math.degrees(gt[2])))
            y += lh
            p.drawText(16, y, 'vel   %+.2f m/s   yaw rate %+.2f rad/s' % (gt[3], gt[4]))
        else:
            p.drawText(16, y, 'pose  waiting for %s' % self.args.gt_topic)
            y += lh
        y += lh
        for key, label, err, mode, shown, aligned, topic, has_data in est_rows:
            if not topic:
                continue                  # nothing publishing; do not clutter
            colour = EST_STYLE.get(key, (C_VINS, None))[0]
            p.setPen(QPen(colour if shown else QColor(110, 116, 128)))
            if topic == 'auto':
                txt = '%-5s searching for a topic' % label
            elif not has_data:
                # subscribed but nothing arriving -- the rate row above names the
                # topic, and this says plainly that alignment is not the holdup
                txt = '%-5s no data' % label
            elif err is not None:
                txt = '%-5s err %.3f m  (%s)' % (label, err, mode)
            elif aligned:
                txt = '%-5s aligned, no GT to compare' % label
            else:
                txt = '%-5s waiting to move 0.5 m' % label
            p.drawText(16, y, txt + ('' if shown else '  [hidden]'))
            y += lh
        p.setPen(QPen(QColor(210, 215, 225)))
        p.drawText(16, y, 'teleop %s   speed %+.2f  steer %+.0f deg'
                   % (teleop, self.speed, math.degrees(self.steer)))
        y += lh
        # Dim when both caps are stock, so a changed limit is what draws the eye.
        stock = (abs(self.max_speed - drive.TOP_SPEED) < 1e-9 and
                 abs(self.steer_cap - drive.STEER_LIMIT) < 1e-9)
        p.setPen(QPen(QColor(120, 126, 138) if stock else C_WARN))
        p.drawText(16, y, 'limits %.2f m/s  %.1f deg  wz<=%.2f rad/s'
                   % (self.max_speed, math.degrees(self.steer_cap),
                      self.max_speed * math.tan(self.steer_cap) / drive.WHEEL_BASE))
        y += lh
        p.setPen(QPen(QColor(210, 215, 225)))
        p.setPen(QPen(C_PLAN if nav_state in ('active', 'succeeded') else
                      QColor(190, 195, 205)))
        p.drawText(16, y, 'nav2  %s %s' % (nav_state, nav_msg))
        y += lh
        if n_route or route_msg:
            p.setPen(QPen(C_ROUTE))
            p.drawText(16, y, 'route %d wp  %s' % (n_route, route_msg[:32]))
            y += lh
        if self.paint_ms:
            p.setPen(QPen(QColor(120, 126, 138)))
            p.drawText(16, y, 'paint %.2f ms avg' % (sum(self.paint_ms) / len(self.paint_ms)))
            y += lh
        for wmsg in warns:
            p.setPen(QPen(C_BAD))
            p.drawText(16, y, wmsg[:46])
            y += lh

    # -- teleop -----------------------------------------------------------------
    DRIVE_KEYS = {Qt.Key_W: 'throttle', Qt.Key_Up: 'throttle',
                  Qt.Key_S: 'brake', Qt.Key_Down: 'brake',
                  Qt.Key_A: 'left', Qt.Key_Left: 'left',
                  Qt.Key_D: 'right', Qt.Key_Right: 'right'}

    def keyPressEvent(self, e):
        act = self.DRIVE_KEYS.get(e.key())
        if act is None:
            return self._hotkey(e)
        if e.isAutoRepeat():
            # Qt emits synthetic repeat PRESS/RELEASE pairs. Re-adding here is what
            # makes a held key stutter, because the synthetic release lands between
            # repeats. Just note that repeat is active and refresh the watchdog.
            self.repeat_seen.add(act)
            self.last_evt[act] = time.monotonic()
            return
        # A keyboard repeats the newest key and only that one, and it never RESUMES
        # repeating an older one -- releasing d does not restart w's repeat stream.
        # So every key already down has just lost its repeats permanently, and their
        # silence stops being evidence of anything. Drop them from repeat_seen, which
        # is what the watchdog gates on, or releasing d kills the throttle the
        # instant the held count falls back to one.
        self.repeat_seen.difference_update(self.held)
        self.held.add(act)
        self.last_evt[act] = time.monotonic()
        self._arm()
        e.accept()

    def keyReleaseEvent(self, e):
        act = self.DRIVE_KEYS.get(e.key())
        if act is None:
            return super(MapView, self).keyReleaseEvent(e)
        if e.isAutoRepeat():
            return                        # synthetic; the key is still physically down
        self.held.discard(act)
        e.accept()

    def _hotkey(self, e):
        k = e.key()
        if k in (Qt.Key_Q,) or (k == Qt.Key_C and e.modifiers() & Qt.ControlModifier):
            self.close()
        elif k == Qt.Key_Space:
            self.speed = 0.0
            self.held.clear()
            self._arm()
        elif k == Qt.Key_Escape:
            if self._goal_drag is not None:
                self._goal_drag = None
            else:
                self.ros.request_cancel()
        elif k in (Qt.Key_Return, Qt.Key_Enter):
            self._send_route()
        elif k == Qt.Key_Backspace:
            with self.state.lock:
                gone = bool(self.state.route) and self.state.route.pop()
                n = len(self.state.route)
                self.state.route_preview = None
            self.state.warn('waypoint dropped, %d left' % n if gone
                            else 'no waypoints to drop')
            self.update()
        elif k == Qt.Key_Delete:
            with self.state.lock:
                n = len(self.state.route)
                self.state.route = []
                self.state.route_preview = None
                self.state.route_msg = ''
            self.state.warn('route cleared (%d waypoints)' % n)
            self.update()
        elif k in (Qt.Key_Equal, Qt.Key_Plus):
            self._scale_limits(LIMIT_STEP, 1.0)
        elif k in (Qt.Key_Minus, Qt.Key_Underscore):
            self._scale_limits(1.0 / LIMIT_STEP, 1.0)
        elif k in (Qt.Key_Period, Qt.Key_Greater):
            self._scale_limits(1.0, LIMIT_STEP)
        elif k in (Qt.Key_Comma, Qt.Key_Less):
            self._scale_limits(1.0, 1.0 / LIMIT_STEP)
        elif k == Qt.Key_0:
            self._scale_limits(None, None)
        elif k == Qt.Key_F:
            self.view.fit(self.width(), self.height())
            self.dirty = True
        elif k == Qt.Key_G:
            self.show_grid = not self.show_grid; self.dirty = True
        elif k == Qt.Key_O:
            self.show_obst = not self.show_obst; self.dirty = True
        elif k == Qt.Key_M:
            self.show_map = not self.show_map; self.dirty = True
            with self.state.lock:
                have = self.state.grid is not None
            if self.show_map and not have:
                self.state.warn('no map loaded (--map ros or --map <map.yaml>)')
        elif k in (Qt.Key_BracketLeft, Qt.Key_BracketRight):
            step = -90 if k == Qt.Key_BracketLeft else 90
            self.view.rot = (self.view.rot + step) % 360
            self.view.fit(self.width(), self.height())
            self.dirty = True
            self.state.warn('view rotated to %d deg' % self.view.rot)
        elif k == Qt.Key_L:
            if self.floor_marked_pm is None:
                self.state.warn('no floor markings (--no-markings, or none parsed)')
            else:
                self.show_marks = not self.show_marks
                self.dirty = True
                self.state.warn('floor markings %s' % ('on' if self.show_marks else 'off'))
        elif k == Qt.Key_T:
            self.show_trails = not self.show_trails
        elif k == Qt.Key_V:
            self._toggle_est('vins')
        elif k == Qt.Key_B:
            self._toggle_est('orb')
        elif k == Qt.Key_R:
            with self.state.lock:
                for e in self.state.estimators:
                    e.reset()
            for c in self._est_poly.values():
                c[0], c[1] = QPolygonF(), 0
            self.state.warn('estimator alignment reset')
        elif k == Qt.Key_C:
            topic = self.args.thumb_topic or '/cam0/image_raw'
            on = self.ros.toggle_thumb(topic)
            self.state.warn('thumbnail %s (%s)' % ('on' if on else 'off', topic))

    def _toggle_est(self, key):
        with self.state.lock:
            for e in self.state.estimators:
                if e.key == key:
                    e.show = not e.show
                    on, label, topic = e.show, e.label, e.topic
                    break
            else:
                return
        if not topic:
            self.state.warn('%s: no topic (is it running?)' % label)
        else:
            self.state.warn('%s overlay %s' % (label, 'on' if on else 'off'))

    def _scale_limits(self, fs, fa):
        """Scale the teleop speed / steering caps. fs=fa=None resets both.

        These are CAPS on the pedal, not a commanded value: nothing moves until you
        press a drive key, and the current speed is re-clamped on the next tick, so
        winding the limit down while driving slows the robot rather than waiting for
        the next acceleration.

        The angular cap is a STEERING ANGLE, because that is the only angular freedom
        an Ackermann car has. /cmd_vel gets the yaw rate this implies at the current
        speed, wz = v*tan(steer)/wheel_base, so the achievable yaw rate scales with
        BOTH knobs and is zero when stopped -- this cannot make the robot spin on the
        spot, and nothing here can.
        """
        if fs is None:
            self.max_speed, self.steer_cap = drive.TOP_SPEED, drive.STEER_LIMIT
        else:
            self.max_speed = max(SPEED_CAP_MIN,
                                 min(SPEED_CAP_MAX, self.max_speed * fs))
            self.steer_cap = max(STEER_CAP_MIN,
                                 min(drive.STEER_LIMIT, self.steer_cap * fa))
        at = '' if not self.steer_cap else (
            '  wz<=%.2f rad/s' % (self.max_speed * math.tan(self.steer_cap)
                                  / drive.WHEEL_BASE))
        self.state.warn('limits %.2f m/s  steer %.1f deg%s'
                        % (self.max_speed, math.degrees(self.steer_cap), at))

    def focusOutEvent(self, e):
        self._panic('focus lost')

    def changeEvent(self, e):
        if e.type() == QEvent.WindowDeactivate:
            self._panic('window deactivated')

    def _panic(self, why):
        """Stop rather than coast. A robot that keeps rolling while you alt-tab is
        the exact complaint the README makes about teleop_twist_keyboard."""
        if not self.held and self.speed == 0.0:
            return
        self.held.clear()
        self.repeat_seen.clear()
        self.speed = 0.0
        self.steer = 0.0
        if self.ros.cmd_pub is not None:
            for _ in range(3):
                self.ros.publish_cmd(0.0, 0.0)
        self.state.warn('teleop stop: %s' % why)

    def _arm(self):
        if self.args.no_teleop:
            return
        with self.state.lock:
            active = self.state.nav_state in ('active', 'sending')
        if active:
            # The human wins, immediately. We start publishing now and let the
            # cancel land when it lands; making a keypress wait ~200 ms for a
            # round trip is the worse failure.
            self.ros.request_cancel()
        self.ros.ensure_cmd_pub()
        self.armed = True
        self.quiet = False

    def on_tick(self):
        with self.state.lock:
            sim = self.state.sim_t
            clock_dead = self.state.rtf.stale()
        now = time.monotonic()
        if clock_dead or sim <= 0.0:
            if not self._warned_clock:
                self._warned_clock = True
                self.state.warn('no /clock -- integrating on the wall clock')
            sim = now
        if self.prev_sim is None:
            self.prev_sim = sim
        # clamp: sim time jumps when gz unpauses, and an unclamped dt would slam
        # the throttle to top speed in a single tick
        dt = min(max(sim - self.prev_sim, 0.0), 0.2)
        self.prev_sim = sim

        # If auto-repeat is active for a key we believe is held but no event has
        # arrived for 0.6 s, we lost the release (window manager grab, etc.).
        # Only trust this when repeat has actually been observed -- with `xset r
        # off` silence proves nothing.
        #
        # And only when ONE key is held. A keyboard repeats the most recently
        # pressed key and only that one, so pressing d while holding w stops w's
        # repeats immediately. With several keys down, silence on one of them means
        # "the keyboard is busy repeating another", not "this key came up" -- and
        # expiring it here dropped the throttle 0.6 s into every turn, coasted the
        # speed to zero, and with v=0 the yaw rate v*tan(steer)/L is zero however
        # hard the wheel is turned. That is why w+d could not drive a circle.
        #
        # Multi-key therefore relies on Qt's real releases, plus the focus-out and
        # window-deactivate panics below, which is what those exist for.
        if len(self.held) == 1:
            for act in list(self.held):
                if act in self.repeat_seen and now - self.last_evt.get(act, now) > 0.6:
                    self.held.discard(act)

        held = {k: (k in self.held) for k in ('throttle', 'brake', 'left', 'right')}
        if held['throttle']:
            self.speed += drive.THROTTLE_ACCEL * dt
        elif held['brake']:
            self.speed -= drive.BRAKE_DECEL * dt
        else:
            drag = drive.DRAG_DECEL * dt
            self.speed = (max(0.0, self.speed - drag) if self.speed > 0
                          else min(0.0, self.speed + drag))
        self.speed = max(-self.max_speed * self._rev_ratio,
                         min(self.max_speed, self.speed))

        if held['left']:
            self.steer += drive.STEER_RATE * dt
        elif held['right']:
            self.steer -= drive.STEER_RATE * dt
        else:
            ctr = drive.CENTER_RATE * dt
            self.steer = (max(0.0, self.steer - ctr) if self.steer > 0
                          else min(0.0, self.steer + ctr))
        self.steer = max(-self.steer_cap, min(self.steer_cap, self.steer))

        idle = (not self.held) and abs(self.speed) < 1e-6 and abs(self.steer) < 1e-6
        if self.armed and not self.quiet:
            # Ackermann bicycle model: a steering ANGLE held constant, converted to
            # the yaw rate /cmd_vel actually wants. See drive.py's docstring.
            self.ros.publish_cmd(self.speed,
                                 self.speed * math.tan(self.steer) / drive.WHEEL_BASE)
            if idle:
                self._idle_since = self._idle_since or now
                if now - self._idle_since > 2.0:
                    for _ in range(3):
                        self.ros.publish_cmd(0.0, 0.0)
                    self.quiet = True     # stop competing with drive.py
            else:
                self._idle_since = None

        with self.state.lock:
            self.state.teleop = ('off' if self.args.no_teleop else
                                 'idle' if not self.armed else
                                 'quiet' if self.quiet else 'ARMED')

    # -- mouse ------------------------------------------------------------------
    def mousePressEvent(self, e):
        if e.button() == Qt.MiddleButton:
            self._drag = ('pan', e.pos(), self.view.cx, self.view.cy)
        elif e.button() == Qt.LeftButton:
            w = self.view.to_world(e.pos().x(), e.pos().y())
            self._goal_drag = (w, w)
            self._drag_wp = bool(e.modifiers() & Qt.ShiftModifier)
        self.setFocus()

    def mouseMoveEvent(self, e):
        self._cursor = self.view.to_world(e.pos().x(), e.pos().y())
        if self._drag and self._drag[0] == 'pan':
            _, p0, cx0, cy0 = self._drag
            ddx, ddy = self.view.unrotate_delta(e.pos().x() - p0.x(),
                                                e.pos().y() - p0.y())
            self.view.cx = cx0 - ddx / self.view.ppm
            self.view.cy = cy0 + ddy / self.view.ppm
            self.dirty = True
        elif self._goal_drag is not None:
            self._goal_drag = (self._goal_drag[0], self._cursor)

    def mouseReleaseEvent(self, e):
        if self._drag:
            self._drag = None
            QTimer.singleShot(150, self._resmooth)
            return
        if self._goal_drag is None:
            return
        (x0, y0), (x1, y1) = self._goal_drag
        self._goal_drag = None
        dx, dy = x1 - x0, y1 - y0
        dragged = math.hypot(dx, dy) * self.view.ppm >= 12
        wp, self._drag_wp = self._drag_wp, False
        why = self._goal_rejected(x0, y0)
        if why:
            self.state.warn('%s refused: %s' % ('waypoint' if wp else 'goal', why))
            return
        if wp:
            # A bare shift-click leaves yaw None (auto). Dragging pins the heading.
            with self.state.lock:
                self.state.route.append((x0, y0, math.atan2(dy, dx) if dragged
                                         else None))
                n = len(self.state.route)
                self.state.route_preview = None
            self.state.warn('waypoint %d queued -- Enter to plan it' % n)
            self.update()
            return
        if dragged:
            yaw = math.atan2(dy, dx)
        else:
            # Bearing from the robot TO the goal, not the robot's current heading.
            # The planner is Dubins and has to ARRIVE on whatever yaw you send, and
            # the current heading is arbitrary with respect to where you clicked.
            # MEASURED from one pose: a goal whose yaw matched the approach bearing
            # SUCCEEDED in ~3 s and 5 replans; the identical goal 180 deg round never
            # converged in 75 s and 74 replans, because a Dubins path can only shed
            # that much heading with a loop and the aisle has no room for one.
            # Same rule as _resolve_route uses for click-placed waypoints.
            with self.state.lock:
                gt = self.state.gt
            yaw = math.atan2(y0 - gt[1], x0 - gt[0]) if gt else 0.0
        with self.state.lock:
            self.state.goal = (x0, y0, yaw)
        self.ros.request_goal(x0, y0, yaw)

    def _resolve_route(self, route, gt):
        """Fill in yaw for click-placed waypoints.

        The planner is Dubins and forward-only (allow_reversing: false), so it has
        to LEAVE each via-point on the heading you give it. An arbitrary yaw there
        is the usual reason a route whose positions are all reachable will not plan.
        The bearing through the point -- towards the next waypoint, or along the
        incoming leg for the last one -- is the one that keeps the legs stitchable.
        """
        out = []
        for i, (x, y, yaw) in enumerate(route):
            if yaw is None:
                if i + 1 < len(route):
                    nx, ny = route[i + 1][0], route[i + 1][1]
                elif i:
                    nx, ny = 2 * x - route[i - 1][0], 2 * y - route[i - 1][1]
                elif gt:
                    nx, ny = 2 * x - gt[0], 2 * y - gt[1]
                else:
                    nx, ny = x + 1.0, y
                yaw = math.atan2(ny - y, nx - x)
            out.append((x, y, yaw))
        return out

    def _send_route(self):
        """First Enter previews (plans, never moves), second Enter drives it."""
        with self.state.lock:
            route = list(self.state.route)
            previewed = self.state.route_preview is not None
            gt = self.state.gt
        if not route:
            return self.state.warn('no waypoints -- shift-left-drag to add some')
        self.ros.request_route('execute' if previewed else 'preview',
                               self._resolve_route(route, gt))

    def _goal_rejected(self, x, y):
        """Turn 'nav2 aborted instantly and I do not know why' into a named reason."""
        if not (FLOOR_X0 <= x <= FLOOR_X1 and FLOOR_Y0 <= y <= FLOOR_Y1):
            return 'outside the floor'
        with self.state.lock:
            live = dict(self.state.model_poses)
        for name, poly, p0 in self.obstacles:
            if _point_in_poly(x, y, self._live_poly(poly, p0, live.get(name))):
                return 'inside %s' % name
        return None

    def _resmooth(self):
        self.dirty = True
        self.update()

    def wheelEvent(self, e):
        wx, wy = self.view.to_world(e.pos().x(), e.pos().y())
        f = 1.15 ** (e.angleDelta().y() / 120.0)
        self.view.ppm = max(4.0, min(400.0, self.view.ppm * f))
        nx, ny = self.view.to_world(e.pos().x(), e.pos().y())
        self.view.cx += wx - nx           # zoom about the cursor
        self.view.cy += wy - ny
        self.dirty = True
        QTimer.singleShot(150, self._resmooth)

    def closeEvent(self, e):
        self._panic('closing')
        e.accept()


def _point_in_poly(x, y, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            xin = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if x < xin:
                inside = not inside
    return inside


# --- main ----------------------------------------------------------------------

def parse_args(argv):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gt-topic', default='/ground_truth/odometry')
    ap.add_argument('--vins-topic', default='auto',
                    help="'auto' probes /vins_estimator/odometry then /odometry then "
                         'any other Odometry publisher. "" disables the overlay.')
    ap.add_argument('--orb-topic', default='auto',
                    help="ORB-SLAM3 odometry. 'auto' probes /orbslam3/odometry (what "
                         "the launch files produce, they set name='orbslam3') then "
                         '/orbslam3_node/odometry (a bare `ros2 run`). "" disables '
                         'the overlay.')
    ap.add_argument('--cmd-vel-topic', default='/cmd_vel')
    ap.add_argument('--imu-topic', default='/imu')
    ap.add_argument('--cam-info-topics', default='/cam0/camera_info,/cam1/camera_info')
    ap.add_argument('--live-poses', action='store_true',
                    help='draw MOVING models at their live gz pose instead of the '
                         'pose the world file authored. Needs the launch bridge: '
                         'bridge_model_poses:=True. Measured ~57 Hz at RTF 0.98, so '
                         'about 13%% of a core; off by default because nothing in '
                         'the stock world moves.')
    ap.add_argument('--pose-topic', default='/world/default/dynamic_pose/info',
                    help='bridged gz dynamic pose feed (tf2_msgs/TFMessage) used by '
                         '--live-poses')
    ap.add_argument('--plan-topic', default='/plan')
    ap.add_argument('--nav-action', default='/navigate_to_pose')
    ap.add_argument('--nav-through-action', default='/navigate_through_poses',
                    help='action used to DRIVE a queued waypoint route')
    ap.add_argument('--compute-route-action', default='/compute_path_through_poses',
                    help='planner_server action used to PREVIEW a route without '
                         'moving the robot')
    ap.add_argument('--planner-id', default='GridBased',
                    help='planner plugin name for route previews; must match '
                         "planner_plugins in nav2_ackermann.yaml")
    ap.add_argument('--nav-frame', default='map')
    ap.add_argument('--no-nav', action='store_true')
    ap.add_argument('--no-teleop', action='store_true',
                    help='never create a /cmd_vel publisher at all')
    ap.add_argument('--thumb', choices=('none', 'cam0', 'cam1'), default='none')
    ap.add_argument('--thumb-topic', default=None)
    ap.add_argument('--thumb-decimate', type=int, default=4)
    ap.add_argument('--map', dest='map_spec', default='none',
                    help="none | ros | path to a map.yaml")
    ap.add_argument('--world', default=WORLD)
    ap.add_argument('--texture', default=TEXTURE)
    ap.add_argument('--floor-ppm', type=float, default=100.0)
    ap.add_argument('--floor-flip', choices=('none', 'u', 'v', 'uv'), default='none')
    ap.add_argument('--no-floor', action='store_true')
    ap.add_argument('--no-markings', action='store_true',
                    help='skip the painted bay/walkway markings on the floor')
    ap.add_argument('--rotate', type=int, choices=(0, 90, 180, 270), default=0,
                    help='rotate the map view by this many degrees. Only the world '
                         'layers turn -- the panel, scale bar and thumbnail stay '
                         'upright. Also bound to [ and ] at runtime.')
    ap.add_argument('--fps', type=int, default=15,
                    help='repaint rate. Measured ~8.5 ms per frame at 1200x900, so '
                         'this is the single biggest knob on viewer CPU; 15 is '
                         'smooth for a map and halves the cost of 30.')
    ap.add_argument('--trail-min-step', type=float, default=0.02)
    ap.add_argument('--align', dest='align', choices=('first', 'none'),
                    default='first',
                    help='how each estimator overlay is placed on the map. "first" '
                         'fits a rigid yaw+translation the first time the estimate '
                         'has moved 0.5 m; "none" draws it in its own frame. Applies '
                         'to every estimator.')
    ap.add_argument('--vins-align', dest='align', choices=('first', 'none'),
                    help=argparse.SUPPRESS)   # old spelling, same destination
    ap.add_argument('--no-sim-time', action='store_true',
                    help='ignored; kept so old command lines do not break')
    ap.add_argument('--watch-imu', action='store_true',
                    help='add /imu to the rate panel. Costs ~45%% of a core at '
                         '200 Hz (rclpy is ~2.2 ms/msg), and RTF already tells you '
                         'the same thing, so it is off by default.')
    return ap.parse_args(argv)


def load_map_file(path, state):
    """Read a nav2 map.yaml/pgm straight off disk, so the overlay works with no
    map_server running."""
    import yaml
    from PIL import Image as PILImage
    with open(path) as f:
        meta = yaml.safe_load(f)
    for k in ('image', 'resolution', 'origin'):
        if k not in meta:
            raise ValueError('map.yaml is missing %r' % k)
    img_path = meta['image']
    if not os.path.isabs(img_path):
        img_path = os.path.join(os.path.dirname(os.path.abspath(path)), img_path)
    a = np.asarray(PILImage.open(img_path).convert('L'))
    occ_thresh = 255 * (1.0 - float(meta.get('occupied_thresh', 0.65)))
    occ = np.where(a <= occ_thresh, 100, 0).astype(np.int8)
    occ = occ[::-1]                       # image row 0 is max y; grids are min y first
    with state.lock:
        state.grid = (occ, float(meta['resolution']),
                      float(meta['origin'][0]), float(meta['origin'][1]))


def main(argv=None):
    # line-buffered: under nohup/redirect these are fully buffered otherwise,
    # and the startup diagnostics are exactly what you need when it misbehaves
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.thumb != 'none' and not args.thumb_topic:
        args.thumb_topic = '/%s/image_raw' % args.thumb

    try:
        app = QApplication(sys.argv[:1])
    except Exception as e:
        print('viewer: cannot open a display (%s).\n'
              'This is a GUI. For headless use drive.py plus `ros2 topic hz`.' % e,
              file=sys.stderr)
        return 1

    # SignalHandlerOptions.NO is required: otherwise rclpy installs a handler that
    # only sets a shutdown flag, and Qt keeps running. And while app.exec_() is
    # blocked in C++ no Python bytecode runs, so a bare signal.signal() would never
    # fire either -- hence the idle QTimer below.
    rclpy.init(signal_handler_options=rclpy.signals.SignalHandlerOptions.NO)
    state = SharedState(args)
    ros = RosLink(args, state)

    if args.map_spec not in ('none', 'ros'):
        try:
            load_map_file(args.map_spec, state)
        except Exception as e:
            state.warn('map overlay unavailable: %s' % e)

    executor = SingleThreadedExecutor()
    executor.add_node(ros.node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()

    win = MapView(args, state, ros)
    win.view.fit(win.width(), win.height())
    win.show()
    if args.thumb != 'none':
        ros.toggle_thumb(args.thumb_topic)

    signal.signal(signal.SIGINT, lambda *_: app.quit())
    signal.signal(signal.SIGTERM, lambda *_: app.quit())
    idle = QTimer(app)                    # lets the interpreter run so Ctrl-C lands
    idle.timeout.connect(lambda: None)
    idle.start(100)

    try:
        rc = app.exec_()
    finally:
        if ros.cmd_pub is not None:
            for _ in range(5):
                ros.publish_cmd(0.0, 0.0)
        executor.shutdown()
        thread.join(timeout=2.0)
        ros.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return rc


if __name__ == '__main__':
    sys.exit(main())
