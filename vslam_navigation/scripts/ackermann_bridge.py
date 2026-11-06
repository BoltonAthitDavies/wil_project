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

Chassis and command interface (6 Nov 2026):
  * wheel_base 0.46 m, steering limit 20 deg, footprint: user measurements.
  * /cmd_steering is in DEGREES and POSITIVE TURNS RIGHT: on dataset_realv7_001
    every one of ten +-30 deg commands produced a body turn rate of the opposite
    sign (about 0.2 rad/s at 0.29 m/s), hence steering_scale 57.3 (rad -> deg)
    and steering_sign -1. Commands beyond +-20 saturate at the wheel limit.
  * /cmd_velocity: 1.25 held for 112 s gave 0.29 m/s median (0.37 at p90) from
    the ORB trajectory, i.e. ~4.3 units per m/s -- but one operating point only,
    and /feedback/velocity on the realv2 bags read 3.0 x the tracker speed, so
    the unit is not known and the response may be nonlinear. velocity_scale 4.3
    is PROVISIONAL and max_command_velocity caps what is ever published until
    the mapping is measured at two or three more command values.
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
        self.declare_parameter('wheel_base', 0.46)         # m, measured 6 Nov 2026
        self.declare_parameter('steering_limit', 0.349)    # rad (20 deg), measured
        self.declare_parameter('velocity_scale', 4.3)      # /cmd_velocity units per m/s, PROVISIONAL (one point)
        self.declare_parameter('max_command_velocity', 2.5)  # cap on the published command units (~0.58 m/s at 4.3)
        self.declare_parameter('steering_scale', 57.2958)  # /cmd_steering is in degrees
        self.declare_parameter('steering_sign', -1.0)      # positive command turns RIGHT (measured)
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
        self.cmax = float(g('max_command_velocity'))
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
                               f'max speed {self.vmax} m/s (command cap {self.cmax} units, scale {self.vs}/m/s, '
                               f'steering sign {self.sign:+.0f} in degrees), require_localization {self.require}, dry_run {self.dry}')

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
        out_v = max(-self.cmax, min(self.cmax, v * self.vs))
        out_s = steer * self.ss * self.sign
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
