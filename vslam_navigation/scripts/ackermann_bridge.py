#!/usr/bin/env python3
"""Turn Nav2's /cmd_vel (geometry_msgs/Twist) into the real rig's two command
topics, /cmd_velocity and /cmd_steering (std_msgs/Float64), with the safety
behaviour the rest of the stack relies on.

Bicycle model, as in script/drive.py and the simulated AckermannSteering:
    steering = atan(wheel_base * angular.z / linear.x), clamped to +-steering_limit
At v ~ 0 with w != 0 the model asks for full lock at a standstill; the sim
notes call this the one trap of the platform, so steering is held at zero
then instead.

Safety:
  * publishes at `rate` Hz regardless of input; if no /cmd_vel for `timeout`
    seconds, both outputs go to zero (Nav2 crashing mid-goal must not leave the
    last command running);
  * with `require_localization` true, outputs are zero while /localization_ok
    is false or has not been seen for `timeout` seconds (pose_gate.py decides);
  * `dry_run` true logs what would be sent and publishes nothing.

UNVERIFIED, deliberately parameters: the real chassis geometry and the units
of the two command topics. The recorded bags show /cmd_velocity set once to
1.25 and held, and /feedback/velocity is known not to be metres per second
(realv2 record), so `velocity_scale` and `steering_scale`/`steering_sign` are
here to be measured on the robot, not assumed. Defaults are the simulated car.
"""
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, Float64


class Bridge(Node):
    def __init__(self):
        super().__init__('ackermann_bridge')
        self.declare_parameter('wheel_base', 0.42)         # UNVERIFIED on the real chassis
        self.declare_parameter('steering_limit', 0.6109)   # rad, UNVERIFIED
        self.declare_parameter('velocity_scale', 1.0)      # /cmd_velocity units per m/s, UNVERIFIED
        self.declare_parameter('steering_scale', 1.0)      # /cmd_steering units per rad, UNVERIFIED
        self.declare_parameter('steering_sign', 1.0)       # +1 if positive = left (REP-103), UNVERIFIED
        self.declare_parameter('max_speed', 0.6)           # m/s hard cap, whatever Nav2 asks
        self.declare_parameter('require_localization', True)
        self.declare_parameter('timeout', 0.5)
        self.declare_parameter('rate', 20.0)
        self.declare_parameter('dry_run', False)
        g = lambda k: self.get_parameter(k).value  # noqa: E731
        self.L = float(g('wheel_base'))
        self.lim = float(g('steering_limit'))
        self.vs, self.ss, self.sign = float(g('velocity_scale')), float(g('steering_scale')), float(g('steering_sign'))
        self.vmax = float(g('max_speed'))
        self.require = bool(g('require_localization'))
        self.timeout = float(g('timeout'))
        self.dry = bool(g('dry_run'))
        self.pub_v = self.create_publisher(Float64, '/cmd_velocity', 10)
        self.pub_s = self.create_publisher(Float64, '/cmd_steering', 10)
        self.create_subscription(Twist, '/cmd_vel', self.on_cmd, 10)
        self.create_subscription(Bool, '/localization_ok', self.on_ok, 10)
        self.last_cmd, self.last_cmd_t = None, 0.0
        self.ok, self.ok_t = False, 0.0
        self.last_sent = None
        self.create_timer(1.0 / float(g('rate')), self.tick)
        self.get_logger().info(f'wheel_base {self.L} m, steering limit {math.degrees(self.lim):.0f} deg, '
                               f'max speed {self.vmax} m/s, require_localization {self.require}, dry_run {self.dry}')

    def on_cmd(self, msg):
        self.last_cmd, self.last_cmd_t = msg, time.monotonic()

    def on_ok(self, msg):
        self.ok, self.ok_t = bool(msg.data), time.monotonic()

    def tick(self):
        now = time.monotonic()
        v = w = 0.0
        reason = ''
        if self.last_cmd is None or now - self.last_cmd_t > self.timeout:
            reason = 'no cmd_vel'
        elif self.require and (not self.ok or now - self.ok_t > self.timeout):
            reason = 'localization not ok'
        else:
            v = max(-self.vmax, min(self.vmax, self.last_cmd.linear.x))
            w = self.last_cmd.angular.z
        if abs(v) > 1e-3:
            steer = max(-self.lim, min(self.lim, math.atan(self.L * w / v)))
        else:
            steer = 0.0
        out_v, out_s = v * self.vs, steer * self.ss * self.sign
        sent = (round(out_v, 3), round(out_s, 3))
        if sent != self.last_sent:
            self.last_sent = sent
            if self.dry:
                self.get_logger().info(f'DRY RUN would send velocity {out_v:+.3f} steering {out_s:+.3f}'
                                       + (f' ({reason})' if reason else ''))
            elif reason:
                self.get_logger().info(f'zero command: {reason}')
        if not self.dry:
            self.pub_v.publish(Float64(data=float(out_v)))
            self.pub_s.publish(Float64(data=float(out_s)))


def main():
    rclpy.init()
    node = Bridge()
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
