#!/usr/bin/env python3
"""Put a real motor between /cmd_vel and the simulator.

    python3 ~/wil_project/script/drivetrain_sim.py
    ros2 launch ... small_warehouse.launch.py cmd_vel_bridge_topic:=/cmd_vel_exec

Sits in the command path: teleop or nav2 publishes /cmd_vel, this republishes
/cmd_vel_exec, and the bridge is pointed at /cmd_vel_exec instead.

WHAT AckermannSteering ALREADY DOES, AND WHAT IT DOES NOT
    It clamps acceleration (max_acceleration 3 m/s^2 in model.sdf), so a step
    command does not become a step in speed. That much is already realistic.

    What it has no notion of:

      DEADBAND   a real drivetrain does not move at all below some command. Static
                 friction, gearbox stiction and PWM floor mean 0.02 m/s asked for
                 is 0 m/s delivered. In sim, any nonzero command produces motion,
                 so a controller can creep to a goal in a way hardware cannot.

      LAG        current takes time to build in an inductive load and the loop has
                 a finite bandwidth. Modelled as first order with time constant
                 tau: v_out += (v_in - v_out) * dt/tau. At the default 0.15 s the
                 output reaches 63% of a step in 0.15 s and 95% in 0.45 s.

      STEERING   the rack has a slew limit; the wheels cannot snap to a new angle.
                 Limited here in yaw-rate terms, which is what /cmd_vel carries.

    Deadband is applied to the INPUT, before the lag. Applying it after would
    chop the tail off every deceleration and leave the robot stopping abruptly,
    which is the opposite of the effect being modelled.

    Nothing here touches /ground_truth/odometry, which reads the true pose from
    the ECM. The robot really does move differently, though, so ground truth
    describes a different trajectory: that is the point of the exercise, and it is
    why this is a separate opt-in node rather than a change to model.sdf.
"""
import argparse

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class Drivetrain(Node):
    def __init__(self, a):
        super().__init__('drivetrain_sim')
        self.a = a
        self.v = self.w = 0.0
        self.target = (0.0, 0.0)
        self.last_cmd = None

        self.pub = self.create_publisher(Twist, a.out, 10)
        self.create_subscription(Twist, a.inp, self.on_cmd, 10)
        self.dt = 1.0 / a.rate
        self.create_timer(self.dt, self.tick)
        self.get_logger().info(
            '%s -> %s  deadband v=%.3f w=%.3f  tau=%.3fs  steer slew=%.2f rad/s^2'
            % (a.inp, a.out, a.deadband_v, a.deadband_w, a.tau, a.slew_w))

    def on_cmd(self, msg):
        v, w = msg.linear.x, msg.angular.z
        if abs(v) < self.a.deadband_v:
            v = 0.0
        if abs(w) < self.a.deadband_w:
            w = 0.0
        self.target = (v, w)
        self.last_cmd = self.get_clock().now()

    def tick(self):
        tv, tw = self.target
        # Watchdog: a publisher that stops mid-command must not leave the robot
        # driving forever. Real hardware e-stops on a lost command link.
        if self.last_cmd is not None and self.a.timeout > 0:
            age = (self.get_clock().now() - self.last_cmd).nanoseconds * 1e-9
            if age > self.a.timeout:
                tv = tw = 0.0
        k = min(self.dt / self.a.tau, 1.0)
        self.v += (tv - self.v) * k
        # Steering is rate limited rather than lagged: a rack has a maximum speed,
        # and a first-order lag would let tiny corrections arrive instantly.
        dw = tw - self.w
        cap = self.a.slew_w * self.dt
        self.w += max(-cap, min(cap, dw))
        if abs(self.v) < 1e-4:
            self.v = 0.0
        if abs(self.w) < 1e-4:
            self.w = 0.0
        out = Twist()
        out.linear.x = self.v
        out.angular.z = self.w
        self.pub.publish(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in', dest='inp', default='/cmd_vel')
    ap.add_argument('--out', default='/cmd_vel_exec')
    ap.add_argument('--deadband-v', type=float, default=0.05,
                    help='m/s below which nothing moves (default 0.05)')
    ap.add_argument('--deadband-w', type=float, default=0.03,
                    help='rad/s below which the rack does not shift')
    ap.add_argument('--tau', type=float, default=0.15,
                    help='motor time constant, seconds (default 0.15)')
    ap.add_argument('--slew-w', type=float, default=2.0,
                    help='max yaw-rate change, rad/s^2 (default 2.0)')
    ap.add_argument('--timeout', type=float, default=0.5,
                    help='stop if no command for this long; 0 disables')
    ap.add_argument('--rate', type=float, default=50.0)
    a = ap.parse_args()
    if a.tau <= 0:
        raise SystemExit('--tau must be > 0')

    rclpy.init()
    node = Drivetrain(a)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
