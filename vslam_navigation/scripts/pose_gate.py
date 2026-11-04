#!/usr/bin/env python3
"""Decide whether the localisation is sound enough to drive on, and start or
pause Nav2 accordingly.

RTAB-Map's localize mode publishes map -> odom whenever it recognises a place,
and a covariance below its threshold is NOT a correctness signal (design
record, Section 6d: 98 % "localised" with poses 161 m outside the building).
This node applies the three checks the record asks for and turns them into one
boolean, /localization_ok:

  1. init transient over   -- at least `init_ignore_s` seconds since the first
                              correction. ORB-SLAM3 re-aligns its world during
                              IMU initialisation and RTAB-Map re-seats the
                              correction with it (4 Nov: two re-seats at +4.5
                              and +5.5 s, 4.2 m apart, then 85 s stable).
  2. correction is stable  -- the last `stable_samples` map->odom samples each
                              stepped less than `max_step` metres.
  3. robot is on the map   -- map -> base_frame lies inside the prior map's node
                              extent (from the database) plus `margin`.

When all three hold the node (a) publishes /localization_ok true and
/localization_status with the reason, (b) relays /localization_pose to
/localization_pose_gated, and (c) if `manage_lifecycle` is true, asks the Nav2
lifecycle manager to STARTUP the stack the first time and RESUME it after a
pause. When any check fails after a pass, Nav2 is PAUSED, which stops the
controller; ackermann_bridge.py also zeroes the robot on /localization_ok
false, so the two guards are independent.
"""
import math
import os
import sqlite3
import struct
import sys

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav2_msgs.srv import ManageLifecycleNodes
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformException, TransformListener


def map_extent(db_path):
    try:
        c = sqlite3.connect(db_path)
        xs, ys = [], []
        for (blob,) in c.execute('select pose from Node'):
            v = struct.unpack('12f', blob[:48])
            xs.append(v[3])
            ys.append(v[7])
        return (min(xs), max(xs), min(ys), max(ys)) if xs else None
    except Exception:
        return None


class PoseGate(Node):
    def __init__(self):
        super().__init__('pose_gate')
        self.declare_parameter('database', '')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'orbslam3_world')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('margin', 2.0)
        self.declare_parameter('init_ignore_s', 8.0)
        self.declare_parameter('max_step', 1.0)
        self.declare_parameter('stable_samples', 6)
        self.declare_parameter('rate', 2.0)
        self.declare_parameter('manage_lifecycle', True)
        self.declare_parameter('lifecycle_service', '/lifecycle_manager_navigation/manage_nodes')
        self.declare_parameter('log_file', '')     # transitions + lifecycle commands, so a run leaves evidence
        g = lambda k: self.get_parameter(k).value  # noqa: E731
        self.map_frame, self.odom_frame, self.base_frame = g('map_frame'), g('odom_frame'), g('base_frame')
        self.margin = float(g('margin'))
        self.init_ignore = float(g('init_ignore_s'))
        self.max_step = float(g('max_step'))
        self.stable_n = int(g('stable_samples'))
        self.extent = map_extent(os.path.expanduser(g('database'))) if g('database') else None
        if self.extent:
            x0, x1, y0, y1 = self.extent
            self.get_logger().info(f'map extent x {x0:.1f}..{x1:.1f} y {y0:.1f}..{y1:.1f} m, margin {self.margin:g} m')
        else:
            self.get_logger().warn('no map extent: the on-map check is disabled')

        self.buf = Buffer()
        self.listener = TransformListener(self.buf, self)
        self.pub_ok = self.create_publisher(Bool, '/localization_ok', 10)
        self.pub_status = self.create_publisher(String, '/localization_status', 10)
        self.pub_pose = self.create_publisher(PoseWithCovarianceStamped, '/localization_pose_gated', 10)
        self.create_subscription(PoseWithCovarianceStamped, '/localization_pose', self.on_pose, 10)
        self.ok = False
        self.first_corr_t = None
        self.prev_corr = None
        self.stable = 0
        self.last_reason = ''
        self.manage = bool(g('manage_lifecycle'))
        self.lifecycle_state = 'inactive'      # inactive | active | paused
        self.cli = self.create_client(ManageLifecycleNodes, g('lifecycle_service')) if self.manage else None
        self.pending = None
        self.logf = None
        if g('log_file'):
            os.makedirs(os.path.dirname(os.path.abspath(os.path.expanduser(g('log_file')))), exist_ok=True)
            self.logf = open(os.path.expanduser(g('log_file')), 'w')
            self.logf.write('sim_time,event\n')
        self.create_timer(1.0 / float(g('rate')), self.tick)
        self.get_logger().info('gate: init_ignore %.0f s, max step %.2f m over %d samples'
                               % (self.init_ignore, self.max_step, self.stable_n))

    def inside(self, x, y):
        if not self.extent:
            return True
        x0, x1, y0, y1 = self.extent
        m = self.margin
        return x0 - m <= x <= x1 + m and y0 - m <= y <= y1 + m

    def on_pose(self, msg):
        if self.ok:
            self.pub_pose.publish(msg)

    def log(self, text):
        if self.logf:
            self.logf.write(f'{self.get_clock().now().nanoseconds * 1e-9:.3f},{text}\n')
            self.logf.flush()

    def set_status(self, ok, reason):
        if ok != self.ok or reason != self.last_reason:
            self.get_logger().info(f'localization_ok={ok}: {reason}')
            self.log(f'localization_ok={ok}: {reason}')
        self.ok, self.last_reason = ok, reason
        self.pub_ok.publish(Bool(data=ok))
        self.pub_status.publish(String(data=reason))

    def tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if now <= 0.0:
            return
        try:
            corr = self.buf.lookup_transform(self.map_frame, self.odom_frame, rclpy.time.Time())
        except TransformException:
            self.transition(False, 'no map->odom correction yet')
            return
        if self.first_corr_t is None:
            self.first_corr_t = now
        c = corr.transform.translation
        if self.prev_corr is not None:
            step = math.hypot(c.x - self.prev_corr[0], c.y - self.prev_corr[1])
            self.stable = self.stable + 1 if step < self.max_step else 0
        self.prev_corr = (c.x, c.y)
        since = now - self.first_corr_t
        if since < self.init_ignore:
            self.transition(False, f'init transient ({since:.0f}/{self.init_ignore:.0f} s)')
            return
        if self.stable < self.stable_n:
            self.transition(False, f'correction not stable ({self.stable}/{self.stable_n} samples)')
            return
        try:
            rb = self.buf.lookup_transform(self.map_frame, self.base_frame, rclpy.time.Time())
        except TransformException:
            self.transition(False, 'no map->base transform')
            return
        b = rb.transform.translation
        if not self.inside(b.x, b.y):
            self.transition(False, f'robot off the map at ({b.x:.1f}, {b.y:.1f})')
            return
        self.transition(True, 'localised and stable')

    def transition(self, ok, reason):
        was = self.ok
        self.set_status(ok, reason)
        if not self.manage or self.cli is None:
            return
        if ok and not was:
            if self.lifecycle_state == 'inactive':
                self.command(ManageLifecycleNodes.Request.STARTUP, 'active')
            elif self.lifecycle_state == 'paused':
                self.command(ManageLifecycleNodes.Request.RESUME, 'active')
        elif was and not ok and self.lifecycle_state == 'active':
            self.command(ManageLifecycleNodes.Request.PAUSE, 'paused')

    def command(self, cmd, new_state):
        if self.pending is not None and not self.pending.done():
            return
        if not self.cli.service_is_ready():
            self.get_logger().warn('lifecycle manager service not available yet')
            return
        req = ManageLifecycleNodes.Request()
        req.command = cmd
        name = {0: 'STARTUP', 1: 'PAUSE', 2: 'RESUME'}.get(cmd, str(cmd))
        self.get_logger().info(f'nav2 lifecycle: {name}')
        self.log(f'nav2 lifecycle: {name}')
        self.pending = self.cli.call_async(req)
        self.pending.add_done_callback(lambda f: self._done(f, name, new_state))

    def _done(self, fut, name, new_state):
        try:
            res = fut.result()
            if res.success:
                self.lifecycle_state = new_state
                self.get_logger().info(f'nav2 lifecycle: {name} done -> {new_state}')
                self.log(f'nav2 lifecycle: {name} done -> {new_state}')
            else:
                self.get_logger().error(f'nav2 lifecycle: {name} FAILED')
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f'nav2 lifecycle: {name} error {e}')


def main():
    rclpy.init()
    node = PoseGate()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
