#!/usr/bin/env python3
"""Publish map -> odom only when the correction is physically reachable.

    ros2 run vslam_navigation correction_gate.py --ros-args \
        -p odom_frame:=odom -p base_frame:=base_link

WHAT IT DEFENDS AGAINST
    A warehouse aisle presents the same rack faces from several places, and
    RTAB-Map accepts a match at the wrong one. Measured on
    exp4/U_static_P2_return_1 at sim t = 71.0: the correction stepped 13.92 m
    and -21.6 deg in a single sample, the reported pose landed 6.27 m away, and
    it was held for 36 s. The front end's largest step in that whole run was
    0.42 m, so nothing physical moved. Two of seven Experiment 4 runs did this,
    for 36 s and 23.5 s, peaking at 6.80 m and 3.99 m.

WHY THE EXISTING DEFENCES DO NOT
    * Covariance cannot screen it: the false match reported 0.0048 against 0.106
      for the honest ones. More confident, and wrong.
    * RGBD/OptimizeMaxError is structurally blind: odometry links are RELATIVE,
      so a rigid displacement of the whole cached window leaves their residual
      near zero. That is a null space, not a threshold, which is why
      RGBD/MaxOdomCacheSize was inert and why tightening the error only halved
      the rate at the cost of 165 s to the first fix.
    * Vis/EstimationType 0 (adopted) tests one match's geometry. These matches
      are geometrically self-consistent -- with the wrong place.
    * pose_gate.py already detects this and only pauses Nav2. RTAB-Map publishes
      map -> odom itself, so the wrong transform still reaches RViz, the
      costmaps and the planner. THIS node is why rtabmap must run with
      publish_tf:=false; otherwise there are two publishers on one edge.

THE RULE
    A correction claims the robot is somewhere other than where dead reckoning
    puts it. It can only be right by as much as the front end has drifted since
    the last accepted fix, and the front end reports the distance it covered:

        accept if  |dp|   <= eps * d_odom + eps0
                   |dyaw| <= eps_yaw * d_odom + yaw0

    eps0 = 1.5 m and yaw0 = 8 deg, raised from 0.5 and 3 after the first live
    runs (15 Nov 2026). The verdict logs separate cleanly into two populations:
    honest fix-to-fix refinements at 0.57-1.99 m and 1.3-5.4 deg, and aliasing
    false fixes at 5.30-8.11 m and 14-34 deg. At 0.5 m the threshold sat INSIDE
    the honest band, so the gate refused real corrections: on
    I_static/P3_circuit it rejected the same ~1.05 m refinement five times and
    ended the run holding a 0.600 m offset with 14.5 s over a metre -- an
    excursion the ungated run of that route did not have. Anything between 2 and
    5 m separates the two populations; 1.5 m plus the per-metre term lands there
    by the time it matters.

    d_odom is measured from the odom -> base transform, not modelled.
    Experiment 4 is what makes the thresholds affordable: unaided ORB-SLAM3
    drift over those runs never exceeded 0.76 m across 21-152 m, so refusing a
    correction and coasting costs almost nothing.

TWO ESCAPES, both needed
    warmup_s   Every correction is accepted for this long after the first one.
               In three of seven runs the correction passes through about
               +-165 deg in its first 1.5 s while the robot still sits near the
               odom origin -- harmless to the POSE, which stays within 0.13 m,
               because the lever is short, but enough to trap a gate that reads
               the transform rather than the pose. Without this a clean run went
               to 24 m: the gate latched the transient and then rejected its own
               repair.
    confirm    `confirm` consecutive REJECTIONS make the gate concede and accept
               the latest candidate. The gate's job is to refuse ISOLATED jumps,
               not to win an argument with the map, and the two cases separate
               cleanly in the recorded verdict streams: aliasing rejections come
               INTERLEAVED with accepts (longest run 6, then 3), while a wrong
               held belief produces a solid block (10 in a row) because the map
               is continuously right. 8 clears the worst static streak by two.

               This replaces an agreement test that could never fire. It asked
               for `confirm` candidates within 0.30 m of EACH OTHER, but while
               the robot drives on a wrong held correction the offers move --
               2.61, 3.28, 3.86, 4.82, 4.64, 4.89 m were logged in one run -- so
               they never agreed and the gate ratcheted itself out of its own
               map. Measured cost of that bug: the front end was drifting only
               0.244 m while the gated pose reached 4.526 m, because every
               correction that would have fixed it was refused.

VALIDATION
    Replayed on all seven recorded Experiment 4 runs by
    script/exp4_gate_sim.py: total time with the error over 1 m falls from 62 s
    to 4 s, no run's median worsens, and four of the seven are untouched because
    the gate rejects nothing in them.

WHAT IT DOES NOT SOLVE
    A false match that is also physically reachable -- a slow walk into the
    wrong aisle -- passes. Nothing in the seven runs shows that happening.
"""
import math
import os

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


class Correction:
    """A planar map -> odom transform. z, roll and pitch ride along unchanged."""

    def __init__(self, x, y, yaw, z=0.0, qz_full=None):
        self.x, self.y, self.yaw, self.z = x, y, yaw, z

    def dist(self, o):
        return math.hypot(self.x - o.x, self.y - o.y)

    def dyaw(self, o):
        return abs(wrap(self.yaw - o.yaw))

    def close_to(self, o, tol_m, tol_rad):
        return self.dist(o) < tol_m and self.dyaw(o) < tol_rad


class CorrectionGate(Node):
    def __init__(self):
        super().__init__('correction_gate')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('pose_topic', '/localization_pose')
        self.declare_parameter('publish_rate', 20.0)
        # thresholds; see THE RULE
        self.declare_parameter('eps', 0.02)          # m of correction per m driven
        self.declare_parameter('eps0', 1.5)          # m, fixed allowance
        self.declare_parameter('eps_yaw_deg', 0.10)  # deg per m driven
        self.declare_parameter('yaw0_deg', 8.0)      # deg, fixed allowance
        self.declare_parameter('warmup_s', 10.0)
        self.declare_parameter('confirm', 8)
        self.declare_parameter('cov_blind', 1.0e4)   # RTAB-Map's no-information value
        self.declare_parameter('log_csv', '')
        g = lambda k: self.get_parameter(k).value    # noqa: E731

        self.map_frame = g('map_frame')
        self.odom_frame = g('odom_frame')
        self.base_frame = g('base_frame')
        self.eps, self.eps0 = float(g('eps')), float(g('eps0'))
        self.eps_yaw = math.radians(float(g('eps_yaw_deg')))
        self.yaw0 = math.radians(float(g('yaw0_deg')))
        self.warmup_s = float(g('warmup_s'))
        self.confirm = int(g('confirm'))
        self.cov_blind = float(g('cov_blind'))

        self.buf = Buffer()
        self.listener = TransformListener(self.buf, self)
        self.br = TransformBroadcaster(self)

        self.held = None            # the correction currently published
        self.first_t = None         # when the first candidate arrived
        self.last_cand = None       # to tell a new fix from a repeated sample
        self.streak = 0             # consecutive rejections
        self.d_odom = 0.0           # metres driven since the last accepted fix
        self.prev_odom_xy = None
        self.n_acc = self.n_rej = self.n_blind = 0

        latched = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_status = self.create_publisher(String, '/correction_gate/status', latched)
        self.pub_held = self.create_publisher(Bool, '/correction_gate/holding', latched)

        self.csv = None
        if g('log_csv'):
            os.makedirs(os.path.dirname(os.path.abspath(g('log_csv'))) or '.',
                        exist_ok=True)
            self.csv = open(g('log_csv'), 'w', buffering=1)
            self.csv.write('stamp,verdict,cand_x,cand_y,cand_yaw_deg,dp,dyaw_deg,'
                           'd_odom,allow_p,allow_yaw_deg\n')

        self.create_subscription(PoseWithCovarianceStamped, g('pose_topic'),
                                 self.on_pose, 20)
        self.create_timer(1.0 / float(g('publish_rate')), self.on_publish)
        self.create_timer(5.0, self.on_report)
        self.get_logger().info(
            f'gating {self.map_frame} -> {self.odom_frame}: '
            f'|dp| <= {self.eps}*d + {self.eps0} m, '
            f'|dyaw| <= {g("eps_yaw_deg")}*d + {g("yaw0_deg")} deg, '
            f'warm-up {self.warmup_s} s, confirm {self.confirm}. '
            f'rtabmap MUST run with publish_tf:=false.')

    # ------------------------------------------------------------------ helpers
    def odom_base(self, stamp=None):
        """odom -> base as (x, y, yaw), or None."""
        try:
            t = self.buf.lookup_transform(self.odom_frame, self.base_frame,
                                          stamp or rclpy.time.Time())
        except TransformException:
            return None
        v = t.transform.translation
        return v.x, v.y, yaw_of(t.transform.rotation)

    def accept(self, cand, why, now):
        self.held = cand
        self.n_acc += 1
        self.streak = 0
        self.d_odom = 0.0
        self.get_logger().info(f'accepted ({why}) at t={now:.2f}')

    # -------------------------------------------------------------------- input
    def on_pose(self, msg):
        c = msg.pose.covariance
        if c[0] + c[7] + c[14] >= self.cov_blind:
            self.n_blind += 1
            return
        ob = self.odom_base(rclpy.time.Time.from_msg(msg.header.stamp))
        if ob is None:
            ob = self.odom_base()
            if ob is None:
                return
        ox, oy, oyaw = ob
        p = msg.pose.pose.position
        myaw = yaw_of(msg.pose.pose.orientation)
        # map->odom = map->base * (odom->base)^-1, planar
        cyaw = wrap(myaw - oyaw)
        ct, st = math.cos(cyaw), math.sin(cyaw)
        cand = Correction(p.x - (ct * ox - st * oy),
                          p.y - (st * ox + ct * oy), cyaw, p.z)

        now = self.get_clock().now().nanoseconds / 1e9
        if self.first_t is None:
            self.first_t = now

        # A new FIX, not a repeated sample of the same one.
        fresh = (self.last_cand is None
                 or not cand.close_to(self.last_cand, 1e-3, 1e-4))
        self.last_cand = cand

        # Every row is logged BEFORE accept(), because accept() resets d_odom to 0.
        # Until 20 Nov 2026 the order was reversed, so gate_verdicts.csv files
        # written before then carry d_odom = 0 on every accepted row; recover it
        # from the logged allowance, d_odom = (allow_p - eps0) / eps.
        if self.held is None or now - self.first_t < self.warmup_s:
            self.log(now, 'warmup', cand, 0.0, 0.0, 0.0)
            self.accept(cand, 'warm-up', now)
            return
        if not fresh:
            return

        dp, dy = cand.dist(self.held), cand.dyaw(self.held)
        allow_p = self.eps * self.d_odom + self.eps0
        allow_y = self.eps_yaw * self.d_odom + self.yaw0
        if dp <= allow_p and dy <= allow_y:
            self.log(now, 'accept', cand, dp, dy, allow_p, allow_y)
            self.accept(cand, f'dp={dp:.2f}<={allow_p:.2f}', now)
            return

        self.n_rej += 1
        self.streak += 1
        self.log(now, 'reject', cand, dp, dy, allow_p, allow_y)
        self.get_logger().warn(
            f'rejected: dp={dp:.2f} m (allow {allow_p:.2f}), '
            f'dyaw={math.degrees(dy):.1f} deg (allow {math.degrees(allow_y):.1f}), '
            f'd_odom={self.d_odom:.1f} m, rejection {self.streak}/{self.confirm}')
        if self.streak >= self.confirm:
            # Its own row since 20 Nov 2026: the same candidate as the reject row
            # just above, now accepted. Before that a concession left no trace in
            # gate_verdicts.csv beyond the eighth 'reject'.
            self.log(now, 'concede', cand, dp, dy, allow_p, allow_y)
            self.accept(cand, f'conceded after {self.confirm} consecutive '
                              f'rejections -- sustained disagreement', now)

    def log(self, now, verdict, c, dp, dy, allow_p, allow_y=0.0):
        """One row of gate_verdicts.csv. verdict is one of:
        warmup   accepted unchecked inside warmup_s (dp, allowances logged as 0)
        accept   within both allowances
        reject   outside one of them; the held correction stays
        concede  the confirm-th consecutive reject, accepted anyway; it repeats
                 the candidate and numbers of the reject row just before it
        Always written BEFORE accept(), which resets d_odom."""
        if self.csv:
            self.csv.write(f'{now:.3f},{verdict},{c.x:.3f},{c.y:.3f},'
                           f'{math.degrees(c.yaw):.2f},{dp:.3f},'
                           f'{math.degrees(dy):.2f},{self.d_odom:.2f},'
                           f'{allow_p:.3f},{math.degrees(allow_y):.2f}\n')

    # ------------------------------------------------------------------- output
    def on_publish(self):
        ob = self.odom_base()
        if ob is not None:
            if self.prev_odom_xy is not None:
                step = math.dist(ob[:2], self.prev_odom_xy)
                if step < 1.0:          # a front-end jump is not travel
                    self.d_odom += step
            self.prev_odom_xy = ob[:2]
        if self.held is None:
            return
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.odom_frame
        t.transform.translation.x = self.held.x
        t.transform.translation.y = self.held.y
        t.transform.translation.z = self.held.z
        t.transform.rotation.z = math.sin(self.held.yaw / 2.0)
        t.transform.rotation.w = math.cos(self.held.yaw / 2.0)
        self.br.sendTransform(t)

    def on_report(self):
        holding = self.streak > 0
        self.pub_held.publish(Bool(data=holding))
        s = (f'accepted {self.n_acc}, rejected {self.n_rej}, '
             f'blind {self.n_blind}, d_odom {self.d_odom:.1f} m'
             + (f', HOLDING against {self.streak} consecutive rejections'
                if holding else ''))
        self.pub_status.publish(String(data=s))
        self.get_logger().info(s)


def main():
    rclpy.init()
    n = CorrectionGate()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.get_logger().info(f'final: accepted {n.n_acc}, rejected {n.n_rej}')
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
