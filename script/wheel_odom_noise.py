#!/usr/bin/env python3
"""Republish the wheel odometry with realistic dead-reckoning error.

    python3 ~/wil_project/script/wheel_odom_noise.py
    python3 ~/wil_project/script/wheel_odom_noise.py --sigma-v 0.03 --scale-v 1.02

Needs the wheel odometry bridged, which is off by default:

    ros2 launch aws_robomaker_small_warehouse_world small_warehouse.launch.py \\
        bridge_wheel_odom:=True

WHY A NODE AND NOT A PLUGIN PARAMETER
    Gazebo Fortress's AckermannSteering exposes 16 SDF parameters -- left_joint,
    wheel_base, max_velocity and so on -- and not one of them is noise. Checked
    against the library's own symbol table, not the documentation. So the only
    place to add it is downstream, which is also where it belongs: the thing being
    corrupted must not be the thing you measure against.

    /ground_truth/odometry is untouched. It comes from a different system
    (OdometryPublisher) reading the model's true pose out of the ECM, so nothing
    done here can reach it.

THE NOISE MODEL
    Per-message jitter alone is wrong for odometry. Real dead reckoning is wrong
    because the errors INTEGRATE: a 2% wheel-radius error is a metre per fifty
    travelled, and it never averages out. So this corrupts the twist and then
    integrates the corrupted twist into its own pose:

        v' = v * scale_v + N(0, sigma_v)
        w' = w * scale_w + N(0, sigma_w)
        x' += v' * cos(th') * dt   ...   th' += w' * dt

    scale_* are systematic (wheel radius / track width being slightly off) and
    produce steady drift. sigma_* are random and produce a random walk. Defaults
    are deliberately mild: 2% long on the wheels, 1% wide on the track.

    The published pose therefore DIVERGES from truth and never recovers, which is
    the point. The twist stays honest-ish; it is the integral that rots.

    Covariance is filled in rather than left at zero: twist from the sigmas, pose
    growing with elapsed time, so a consumer that reads covariance sees the
    uncertainty grow instead of trusting a drifting pose absolutely.
"""
import argparse, math, random, sys

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry

DEFAULT_IN = '/model/ackermann_robot_001/odometry'


class WheelOdomNoise(Node):
    def __init__(self, a):
        super().__init__('wheel_odom_noise')
        self.a = a
        self.rng = random.Random(a.seed)
        self.x = self.y = self.th = 0.0
        self.t0 = None
        self.last = None
        self.n = 0

        # BEST_EFFORT to match what ros_gz_bridge offers on a sensor-ish stream;
        # a RELIABLE subscription would simply never connect and the node would
        # sit silent with no error anywhere.
        qos = QoSProfile(depth=10, history=HistoryPolicy.KEEP_LAST,
                         reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pub = self.create_publisher(Odometry, a.out, 10)
        self.create_subscription(Odometry, a.inp, self.on_odom, qos)
        self.create_timer(5.0, self.report)
        self.get_logger().info('%s -> %s  scale_v=%.4f scale_w=%.4f '
                               'sigma_v=%.4f sigma_w=%.4f'
                               % (a.inp, a.out, a.scale_v, a.scale_w,
                                  a.sigma_v, a.sigma_w))

    def on_odom(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last is None:
            self.t0 = t
            self.x = msg.pose.pose.position.x
            self.y = msg.pose.pose.position.y
            q = msg.pose.pose.orientation
            self.th = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            self.last = t
            return
        dt = t - self.last
        self.last = t
        # A bridged stream can repeat or reorder a stamp; integrating a negative
        # or zero dt would step the pose backwards for no physical reason.
        if not 0.0 < dt < 1.0:
            return

        v = msg.twist.twist.linear.x * self.a.scale_v + self.rng.gauss(0, self.a.sigma_v)
        w = msg.twist.twist.angular.z * self.a.scale_w + self.rng.gauss(0, self.a.sigma_w)
        self.th += w * dt
        self.x += v * math.cos(self.th) * dt
        self.y += v * math.sin(self.th) * dt
        self.n += 1

        out = Odometry()
        out.header = msg.header
        out.child_frame_id = msg.child_frame_id
        out.pose.pose.position.x = self.x
        out.pose.pose.position.y = self.y
        out.pose.pose.position.z = msg.pose.pose.position.z
        out.pose.pose.orientation.z = math.sin(self.th / 2.0)
        out.pose.pose.orientation.w = math.cos(self.th / 2.0)
        out.twist.twist.linear.x = v
        out.twist.twist.angular.z = w

        age = max(t - self.t0, 1e-3)
        pc = [0.0] * 36
        pc[0] = pc[7] = (self.a.sigma_v ** 2) * age      # x, y grow with time
        pc[35] = (self.a.sigma_w ** 2) * age             # yaw
        out.pose.covariance = pc
        tc = [0.0] * 36
        tc[0] = self.a.sigma_v ** 2
        tc[35] = self.a.sigma_w ** 2
        out.twist.covariance = tc
        self.pub.publish(out)

    def report(self):
        if self.n:
            self.get_logger().info('%d msgs, drifted pose now (%.3f, %.3f, %.1f deg)'
                                   % (self.n, self.x, self.y, math.degrees(self.th)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in', dest='inp', default=DEFAULT_IN)
    # NOT /odom and NOT anything viewer.py might adopt by accident: its
    # _find_odom() falls back to "any other nav_msgs/Odometry topic" and would
    # quietly draw this drifting feed as if it were a VIO estimate.
    ap.add_argument('--out', default='/wheel_odom_noisy')
    ap.add_argument('--sigma-v', type=float, default=0.02,
                    help='m/s, random speed error (default 0.02)')
    ap.add_argument('--sigma-w', type=float, default=0.01,
                    help='rad/s, random yaw-rate error (default 0.01)')
    ap.add_argument('--scale-v', type=float, default=1.02,
                    help='systematic speed scale, e.g. wheel radius 2%% long')
    ap.add_argument('--scale-w', type=float, default=0.99,
                    help='systematic yaw-rate scale, e.g. track 1%% wide')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    rclpy.init()
    node = WheelOdomNoise(a)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
