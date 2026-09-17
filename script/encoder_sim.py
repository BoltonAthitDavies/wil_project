#!/usr/bin/env python3
"""Turn the simulator's exact joint angles into something an encoder could report.

    python3 ~/wil_project/script/encoder_sim.py --cpr 1024

Needs the encoders bridged, which is off by default:

    ros2 launch aws_robomaker_small_warehouse_world small_warehouse.launch.py \\
        bridge_joint_states:=True

WHAT IS MISSING FROM THE RAW FEED
    JointStatePublisher reports the solver's continuous float joint angle. A real
    incremental encoder reports COUNTS: with N counts per revolution the angle is
    known only to 2*pi/N, and nothing finer exists to be read.

    That quantisation is not a rounding detail, it is the dominant error at low
    speed. Velocity from differencing two quantised positions has a floor of
    (2*pi/N) / dt rad/s -- at 1024 CPR and 50 Hz that is 0.31 rad/s, i.e. 1.8 cm/s
    on a 0.0585 m wheel. Below that the reported speed is 0 or one whole tick, and
    nothing in between. A controller tuned on the exact feed will not behave the
    same way on this one.

    So: position is floored to a whole tick, and velocity is RE-DERIVED from the
    quantised positions rather than passed through. Passing the true velocity
    alongside a quantised position would be the worst of both, and is the mistake
    worth avoiding: it looks quantised while still carrying the exact answer.

    Effort is dropped. A real encoder does not measure torque.
"""
import argparse, math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState

DEFAULT_IN = '/model/ackermann_robot_001/joint_state'
WHEEL_RADIUS = 0.0585           # matches <wheel_radius> in model.sdf


class EncoderSim(Node):
    def __init__(self, a):
        super().__init__('encoder_sim')
        self.a = a
        self.tick = 2.0 * math.pi / a.cpr
        self.prev_pos = {}
        self.prev_t = None
        self.n = 0

        qos = QoSProfile(depth=10, history=HistoryPolicy.KEEP_LAST,
                         reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pub = self.create_publisher(JointState, a.out, 10)
        self.create_subscription(JointState, a.inp, self.on_js, qos)
        self.create_timer(5.0, self.report)
        self.get_logger().info(
            '%s -> %s   %d CPR = %.6f rad/tick = %.2f mm at the rim'
            % (a.inp, a.out, a.cpr, self.tick, self.tick * WHEEL_RADIUS * 1000.0))

    def on_js(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        out = JointState()
        out.header = msg.header
        out.name = list(msg.name)
        pos, vel = [], []
        dt = None if self.prev_t is None else t - self.prev_t
        if dt is not None and not 0.0 < dt < 1.0:
            dt = None                       # repeated or reordered stamp

        for i, name in enumerate(msg.name):
            p = msg.position[i] if i < len(msg.position) else 0.0
            q = math.floor(p / self.tick) * self.tick
            pos.append(q)
            if dt:
                vel.append((q - self.prev_pos.get(name, q)) / dt)
            else:
                vel.append(0.0)
            self.prev_pos[name] = q
        self.prev_t = t
        out.position, out.velocity, out.effort = pos, vel, []
        self.pub.publish(out)
        self.n += 1

    def report(self):
        if not self.n:
            return
        floor = self.tick * 50.0            # one tick per 50 Hz sample
        self.get_logger().info(
            '%d msgs; velocity floor %.3f rad/s = %.1f mm/s at the wheel'
            % (self.n, floor, floor * WHEEL_RADIUS * 1000.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in', dest='inp', default=DEFAULT_IN)
    ap.add_argument('--out', default='/joint_states_encoder')
    ap.add_argument('--cpr', type=int, default=1024,
                    help='counts per revolution (default 1024)')
    a = ap.parse_args()
    if a.cpr < 1:
        raise SystemExit('--cpr must be >= 1')

    rclpy.init()
    node = EncoderSim(a)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
