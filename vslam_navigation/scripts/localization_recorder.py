#!/usr/bin/env python3
"""Record what RTAB-Map's localize mode actually produced, with a spatial sanity
bound, so a relocalisation run leaves evidence instead of an impression.

Two things are logged to one CSV:

  kind=pose   every /localization_pose (geometry_msgs/PoseWithCovarianceStamped):
              stamp, x y z, yaw, covariance trace, inside_extent.
  kind=tf     at `rate` Hz, the correction map -> odom_frame RTAB-Map publishes
              (translation, yaw) and the robot's map -> base_frame pose, with the
              step since the previous tf row.

`inside_extent` is the design record's rule (Section 6d): a covariance below
9999 is NOT a correctness signal -- a sim run reported 98 % localised with poses
161 m outside a 14 x 21 m building -- so every pose is checked against the
prior map's node extent (from the database's Node table) plus `margin` metres.

At shutdown a summary is printed and appended as kind=summary: number of
localisation poses, time to first fix (sim seconds from the first tf sample),
share inside the extent, and the largest map->odom step between consecutive
samples (a stable correction steps by centimetres; a re-localisation to a
wrong place steps by metres).

    python3 localization_recorder.py --ros-args -p use_sim_time:=true \
        -p csv:=map/<map>/localize_<run>.csv -p database:=map/<map>/<map>.db \
        -p odom_frame:=orbslam3_world -p base_frame:=base_link
"""
import csv
import math
import os
import sqlite3
import struct
import sys
import time

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener, TransformException


def yaw_of(q):
    """Yaw of a geometry_msgs quaternion."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def map_extent(db_path):
    """(xmin, xmax, ymin, ymax) of the prior map's node poses, or None."""
    try:
        c = sqlite3.connect(db_path)
        xs, ys = [], []
        for (blob,) in c.execute('select pose from Node'):
            v = struct.unpack('12f', blob[:48])   # row-major 3x4
            xs.append(v[3])
            ys.append(v[7])
        if not xs:
            return None
        return min(xs), max(xs), min(ys), max(ys)
    except Exception:
        return None


class Recorder(Node):
    def __init__(self):
        super().__init__('localization_recorder')
        self.declare_parameter('csv', '')
        self.declare_parameter('database', '')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'orbslam3_world')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('margin', 2.0)
        self.declare_parameter('rate', 2.0)
        path = os.path.expanduser(self.get_parameter('csv').value)
        if not path:
            self.get_logger().error('csv parameter is required')
            sys.exit(1)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.map_frame = self.get_parameter('map_frame').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.margin = float(self.get_parameter('margin').value)
        self.extent = map_extent(os.path.expanduser(self.get_parameter('database').value))
        if self.extent:
            x0, x1, y0, y1 = self.extent
            self.get_logger().info(
                f'prior map extent x {x0:.1f}..{x1:.1f} y {y0:.1f}..{y1:.1f} m (+{self.margin:g} m margin)')
        else:
            self.get_logger().warn('no map extent (database unreadable): inside_extent will be blank')

        self.f = open(path, 'w', newline='')
        self.w = csv.writer(self.f)
        self.w.writerow(['kind', 'stamp', 'x', 'y', 'z', 'yaw_deg', 'cov_trace', 'inside_extent',
                         'corr_x', 'corr_y', 'corr_yaw_deg', 'corr_step_m'])
        self.f.flush()
        self.path = path
        self.n_pose = 0
        self.n_inside = 0
        self.first_tf_t = None
        self.first_pose_t = None
        self.prev_corr = None
        self.max_step = 0.0
        self.tf_rows = 0

        self.buf = Buffer()
        self.listener = TransformListener(self.buf, self)
        self.create_subscription(PoseWithCovarianceStamped, '/localization_pose', self.on_pose, 10)
        self.create_timer(1.0 / float(self.get_parameter('rate').value), self.on_timer)
        self.get_logger().info(f'recording to {path}')

    def inside(self, x, y):
        if not self.extent:
            return ''
        x0, x1, y0, y1 = self.extent
        m = self.margin
        return int(x0 - m <= x <= x1 + m and y0 - m <= y <= y1 + m)

    def on_pose(self, msg):
        p = msg.pose.pose.position
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        cov = msg.pose.covariance
        trace = cov[0] + cov[7] + cov[14]
        ins = self.inside(p.x, p.y)
        self.n_pose += 1
        self.n_inside += 1 if ins == 1 else 0
        if self.first_pose_t is None:
            self.first_pose_t = t
            self.get_logger().info(
                f'first localization pose at ({p.x:.2f}, {p.y:.2f}) inside_extent={ins} cov_trace={trace:.3g}')
        self.w.writerow(['pose', f'{t:.3f}', f'{p.x:.3f}', f'{p.y:.3f}', f'{p.z:.3f}',
                         f'{math.degrees(yaw_of(msg.pose.pose.orientation)):.1f}', f'{trace:.4g}', ins,
                         '', '', '', ''])
        self.f.flush()

    def on_timer(self):
        now = self.get_clock().now()
        t = now.nanoseconds * 1e-9
        if t <= 0.0:
            return
        try:
            corr = self.buf.lookup_transform(self.map_frame, self.odom_frame, rclpy.time.Time())
        except TransformException:
            return                      # no correction yet: nothing recognised
        if self.first_tf_t is None:
            self.first_tf_t = t
        c = corr.transform.translation
        cyaw = math.degrees(yaw_of(corr.transform.rotation))
        step = ''
        if self.prev_corr is not None:
            step_m = math.hypot(c.x - self.prev_corr[0], c.y - self.prev_corr[1])
            self.max_step = max(self.max_step, step_m)
            step = f'{step_m:.3f}'
        self.prev_corr = (c.x, c.y)
        x = y = z = yaw = ''
        ins = ''
        try:
            rb = self.buf.lookup_transform(self.map_frame, self.base_frame, rclpy.time.Time())
            b = rb.transform.translation
            x, y, z = f'{b.x:.3f}', f'{b.y:.3f}', f'{b.z:.3f}'
            yaw = f'{math.degrees(yaw_of(rb.transform.rotation)):.1f}'
            ins = self.inside(b.x, b.y)
        except TransformException:
            pass
        self.w.writerow(['tf', f'{t:.3f}', x, y, z, yaw, '', ins,
                         f'{c.x:.3f}', f'{c.y:.3f}', f'{cyaw:.1f}', step])
        self.tf_rows += 1
        if self.tf_rows % 20 == 0:
            self.f.flush()

    def finish(self):
        share = (self.n_inside / self.n_pose * 100.0) if self.n_pose else 0.0
        ttf = '' if self.first_tf_t is None else f'{self.first_tf_t:.3f}'
        summary = (f'localization poses {self.n_pose}, inside extent {share:.0f} %, '
                   f'first correction at sim t={ttf or "never"}, '
                   f'max map->odom step {self.max_step:.2f} m over {self.tf_rows} samples')
        self.get_logger().info('SUMMARY: ' + summary)
        self.w.writerow(['summary', '', '', '', '', '', '', f'{share:.1f}', '', '', '', f'{self.max_step:.3f}'])
        self.w.writerow(['summary_text', summary] + [''] * 10)
        self.f.close()


def main():
    rclpy.init()
    node = Recorder()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.finish()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
