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

Replaying a FRONT-END's vio.csv (ORB-SLAM3 on the real rig) needs two more
things, both off by default so ground-truth replay is unchanged:

  start_time   seconds after the first row before anything is broadcast. ORB's
               first ~1 s after IMU init is a frame transient (the world flips
               ~90 deg at the inertial re-alignment and may flip back on a map
               reset); RTAB-Map would anchor its map in the wrong frame. Rows
               before this are dropped. The first KEPT pose is broadcast from
               the moment /clock runs (hold_before_start, default true), so the
               back-end never sees a frame with no odometry: RTAB-Map blocks
               wait_for_transform (5 s) on every such frame, the queue behind it
               backs up past its 10 s TF buffer, and the run yields 0 nodes
               (2026-10-29, three reproductions). Holding is correct here because
               the robot is stationary through ORB's init transient.
  hold_before_start  false sends nothing before start_time (the old behaviour).
  world_fix    'optical_to_flu' rotates the whole world so that a camera-optical
               world (x right, y down, z forward -- what the ORB node publishes
               on the real rig: the active map's frame is the camera at map
               creation, and the IMU body there is x-right/y-up/z-back, not FLU)
               becomes forward-left-up with z up. RTAB-Map projects its grid onto
               the odometry xy plane and takes roll/pitch from it as gravity
               (Mem/UseOdomGravity), so a y-down world gives a map standing on
               its side and no grid. Measured on realv5: body y (physical up)
               -> world -y for the whole run; after the fix -> +z.

  Both of the above are RUN-DEPENDENT: ORB's published world is the active
  map's frame, so one run ends camera-aligned (realv5 newIMU_TO_CAM_TF: body up
  -> world -y for 130 s) and another ends gravity-aligned (realv6: body up ->
  +z after a 9.5 s transient). Guessing them per run is what put a realv6 map
  on its side. Hence the data-driven options, recommended for any vio.csv:

  world_fix:=auto        average the body's up axis (body_up, default 0,1,0 =
                         the real rig's IMU frame; sim FLU would be 0,0,1) over
                         the kept rows and rotate the world so it points to +z.
                         Yaw is preserved. On a flat floor this also levels a
                         constant mount pitch, which fixed frames cannot.
  start_after_transient  true: start_time becomes 0.5 s after the last
                         orientation jump > 45 deg inside the first 30 s, i.e.
                         after ORB's inertial re-alignment(s) and any early map
                         reset. Logged, so the choice is visible.
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


def quat_from_matrix(R):
    """Rotation matrix -> (w, x, y, z)."""
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2.0
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


def rot_from_quat(q):
    """(w, x, y, z) -> 3x3."""
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def quat_mul(a, b):
    """Hamilton product, (w, x, y, z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw])


# Camera-optical world (x right, y down, z forward) -> FLU: x'=z, y'=-x, z'=-y.
WORLD_FIX = {
    'none': np.eye(3),
    'optical_to_flu': np.array([[0.0, 0.0, 1.0],
                                [-1.0, 0.0, 0.0],
                                [0.0, -1.0, 0.0]]),
}


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
        self.declare_parameter('start_time', 0.0)
        self.declare_parameter('world_fix', 'none')
        self.declare_parameter('hold_before_start', True)
        self.declare_parameter('start_after_transient', False)
        self.declare_parameter('body_up', [0.0, 1.0, 0.0])

        path = os.path.expanduser(self.get_parameter('csv').value)
        if not path or not os.path.exists(path):
            self.get_logger().error(f'csv not found: {path!r}')
            sys.exit(1)

        a = np.loadtxt(path, delimiter=',')
        start = float(self.get_parameter('start_time').value)
        if bool(self.get_parameter('start_after_transient').value):
            start = max(start, self._transient_end(a))
        if start > 0.0:
            keep = a[:, 0] * 1e-9 >= a[0, 0] * 1e-9 + start
            self.get_logger().info(
                f'start_time {start:g} s: dropping {int((~keep).sum())} of {len(a)} rows')
            a = a[keep]
            if len(a) < 2:
                self.get_logger().error('start_time leaves fewer than 2 rows')
                sys.exit(1)
        self.t = (a[:, 0] * 1e-9).tolist()
        self.p = a[:, 1:4]
        self.q = a[:, 4:8]                       # w x y z
        fix = self.get_parameter('world_fix').value
        if fix not in WORLD_FIX and fix != 'auto':
            self.get_logger().error(f'world_fix must be auto or one of {sorted(WORLD_FIX)}')
            sys.exit(1)
        if fix != 'none':
            R = self._auto_world_fix() if fix == 'auto' else WORLD_FIX[fix]
            qf = quat_from_matrix(R)
            self.p = self.p @ R.T
            self.q = np.array([quat_mul(qf, q) for q in self.q])
            self.get_logger().info(f'world_fix {fix}: world rotated, z is now up')
        self.hold_before_start = (start <= 0.0
                                  or bool(self.get_parameter('hold_before_start').value))
        if start > 0.0:
            self.get_logger().info(
                'holding the first kept pose before start_time' if self.hold_before_start
                else 'NO TF before start_time (hold_before_start:=false)')
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

    def _transient_end(self, a):
        """Seconds after the first row at which ORB's frame stops jumping."""
        t = (a[:, 0] - a[0, 0]) * 1e-9
        q = a[:, 4:8]
        last = 0.0
        for i in range(1, len(q)):
            if t[i] > 30.0:
                break
            R0, R1 = rot_from_quat(q[i - 1]), rot_from_quat(q[i])
            ang = np.degrees(np.arccos(np.clip((np.trace(R0.T @ R1) - 1.0) / 2.0, -1.0, 1.0)))
            if ang > 45.0:
                last = t[i]
                self.get_logger().info(f'orientation jump {ang:.0f} deg at +{t[i]:.1f} s')
        start = last + 0.5 if last > 0.0 else 0.0
        self.get_logger().info(f'start_after_transient: start_time = {start:.1f} s')
        return start

    def _auto_world_fix(self):
        """Rotation taking the mean body-up direction (in the csv's world) to +z."""
        up_b = np.array(self.get_parameter('body_up').value, float)
        up_b /= np.linalg.norm(up_b)
        ups = np.array([rot_from_quat(q) @ up_b for q in self.q])
        u = ups.mean(0)
        u /= np.linalg.norm(u)
        spread = np.degrees(np.arccos(np.clip(ups @ u, -1.0, 1.0)))
        z = np.array([0.0, 0.0, 1.0])
        v = np.cross(u, z)
        s_, c_ = np.linalg.norm(v), float(u @ z)
        if s_ < 1e-9:
            R = np.eye(3) if c_ > 0 else np.diag([1.0, -1.0, -1.0])
        else:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + vx + vx @ vx * ((1 - c_) / s_ ** 2)
        self.get_logger().info(
            f'world_fix auto: body up {up_b} points to world {np.round(u, 3)} '
            f'(p95 spread {np.percentile(spread, 95):.1f} deg); rotating by '
            f'{np.degrees(np.arccos(np.clip(c_, -1, 1))):.1f} deg so it is +z')
        return R

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
        # With hold_before_start the first pose goes out as soon as sim time is
        # running at all; without it, nothing until the first kept row.
        if now <= 0.0 or (not self.hold_before_start and now < self.t[0]) \
                or (self.hold_before_start and now < self.t[0] - 3600.0):
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
