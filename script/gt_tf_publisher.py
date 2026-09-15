#!/usr/bin/env python3
"""Broadcast ground_truth.csv as the odometry TF a SLAM back-end expects.

Stands in for VINS or ORB-SLAM3 when the question under test is about MAPPING
rather than about trajectory error. Three reasons it is worth having:

  * It isolates the map from odometry drift. Any grid defect that survives a
    perfect trajectory is a depth or integration problem, not a SLAM one.
  * It lets the bag play at --rate 2.0 instead of the 0.3-0.5 VINS needs to keep
    its backlog down, so an experiment is ~1 min rather than ~6.
  * It is deterministic, so two runs differ only by the parameter under test.
    An earlier A/B was invalidated because the runs processed different numbers
    of frames; with this, node counts match.

The published frames default to VINS' names (world -> body), so
rtabmap_ros/rtab_sim.launch.py needs no changes.

    python3 script/gt_tf_publisher.py --ros-args -p use_sim_time:=true \
        -p csv:=output/output_vins/simulation/dataset_static_tiny_repeat/ground_truth.csv

CSV format is script/extract_gt.py's: t_ns, x, y, z, qw, qx, qy, qz, vx, vy, vz.
Poses are interpolated at the current /clock time, so the TF is continuous
rather than stepped at the 40 Hz sample rate.
"""
import bisect
import os
import sys

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster


def slerp(q0, q1, t):
    """Shortest-arc quaternion interpolation, (w,x,y,z) order."""
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:                      # nearly parallel: lerp and renormalise
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th0 = np.arccos(d)
    th = th0 * t
    q2 = q1 - q0 * d
    q2 /= np.linalg.norm(q2)
    return q0 * np.cos(th) + q2 * np.sin(th)


class GtTf(Node):
    def __init__(self):
        super().__init__('gt_tf_publisher')
        self.declare_parameter('csv', '')
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('body_frame', 'body')
        self.declare_parameter('publish_odom', True)
        self.declare_parameter('odom_topic', '/gt/odometry')

        path = os.path.expanduser(self.get_parameter('csv').value)
        if not path or not os.path.exists(path):
            self.get_logger().error(f'csv not found: {path!r}')
            sys.exit(1)

        a = np.loadtxt(path, delimiter=',')
        self.t = (a[:, 0] * 1e-9).tolist()
        self.p = a[:, 1:4]
        self.q = a[:, 4:8]                       # w x y z
        self.world = self.get_parameter('world_frame').value
        self.body = self.get_parameter('body_frame').value

        self.br = TransformBroadcaster(self)
        self.pub = None
        if self.get_parameter('publish_odom').value:
            self.pub = self.create_publisher(
                Odometry, self.get_parameter('odom_topic').value, 20)

        # 100 Hz is comfortably above the 4 Hz the shim republishes images at,
        # so rtabmap never waits on TF (its wait_for_transform is 5 s).
        self.create_timer(0.01, self.tick)
        self.n = 0
        self.warned = False
        self.get_logger().info(
            f'{len(self.t)} gt poses, {self.t[0]:.2f}..{self.t[-1]:.2f} s -> '
            f'TF {self.world} -> {self.body}')

    def pose_at(self, now):
        if now <= self.t[0]:
            return self.p[0], self.q[0]
        if now >= self.t[-1]:
            return self.p[-1], self.q[-1]
        i = bisect.bisect_left(self.t, now)
        t0, t1 = self.t[i - 1], self.t[i]
        u = 0.0 if t1 <= t0 else (now - t0) / (t1 - t0)
        return (self.p[i - 1] + u * (self.p[i] - self.p[i - 1]),
                slerp(self.q[i - 1], self.q[i], u))

    def tick(self):
        stamp = self.get_clock().now()
        now = stamp.nanoseconds * 1e-9
        # Before the bag starts, sim time sits at 0 and there is nothing to say.
        if now < self.t[0] - 1.0:
            if not self.warned:
                self.get_logger().info('waiting for /clock to reach the bag window')
                self.warned = True
            return
        p, q = self.pose_at(now)

        t = TransformStamped()
        t.header.stamp = stamp.to_msg()
        t.header.frame_id = self.world
        t.child_frame_id = self.body
        t.transform.translation.x = float(p[0])
        t.transform.translation.y = float(p[1])
        t.transform.translation.z = float(p[2])
        t.transform.rotation.w = float(q[0])
        t.transform.rotation.x = float(q[1])
        t.transform.rotation.y = float(q[2])
        t.transform.rotation.z = float(q[3])
        self.br.sendTransform(t)

        if self.pub is not None:
            o = Odometry()
            o.header = t.header
            o.child_frame_id = self.body
            o.pose.pose.position.x = float(p[0])
            o.pose.pose.position.y = float(p[1])
            o.pose.pose.position.z = float(p[2])
            o.pose.pose.orientation = t.transform.rotation
            self.pub.publish(o)

        self.n += 1
        if self.n % 1000 == 0:
            self.get_logger().info(f'{self.n} transforms, t={now:.1f}s')


def main():
    rclpy.init()
    node = GtTf()
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
