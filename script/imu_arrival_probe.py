#!/usr/bin/env python3
"""Measure how an IMU topic ARRIVES at a subscriber: rate, burstiness, loss.

Why: on 2026-10-29 the ORB-SLAM3 stereo-inertial node on dataset_realv5_000 saw
the 200 Hz IMU arrive in bursts of up to ~900 samples separated by gaps of up to
11 s, with 17 % of samples never delivered, although the bag itself is a clean
200 Hz stream. This probe subscribes with the SAME QoS as the node
(RELIABLE, KEEP_LAST 2000) and reports arrival statistics, so the cause can be
localised: run it (a) alone while the bag plays, then (b) with the node running.

Usage (two terminals):
  ros2 bag play "/media/ambushee/New Volume/wil_rescue/<bag>" --rate 0.5   # datasets moved to this root on 2026-11-03
  python3 script/imu_arrival_probe.py --topic /hwt101ct_yaw_publisher --seconds 40

Prints, every 5 s and at the end: samples received, wall-clock arrival gaps
(median / p99 / max), number of bursts (>= 20 samples arriving within 2 ms),
largest burst, and the header-stamp gaps > 0.1 s (= samples lost or delayed).
"""
import argparse, time, sys
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu


class Probe(Node):
    def __init__(self, topic, best_effort):
        super().__init__('imu_arrival_probe')
        qos = QoSProfile(depth=2000, history=HistoryPolicy.KEEP_LAST,
                         reliability=ReliabilityPolicy.BEST_EFFORT if best_effort else ReliabilityPolicy.RELIABLE)
        self.sub = self.create_subscription(Imu, topic, self.cb, qos)
        self.arr, self.stamps = [], []

    def cb(self, m):
        self.arr.append(time.monotonic())
        self.stamps.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)


def report(p, label):
    import statistics as st
    a, s = p.arr, p.stamps
    if len(a) < 3:
        print(f'[{label}] {len(a)} samples'); return
    gaps = [b - c for b, c in zip(a[1:], a[:-1])]
    gaps_sorted = sorted(gaps)
    bursts, largest, run = 0, 0, 1
    for g in gaps:
        if g < 0.002:
            run += 1
        else:
            if run >= 20: bursts += 1; largest = max(largest, run)
            run = 1
    sgaps = [b - c for b, c in zip(s[1:], s[:-1])]
    lost = sum(1 for g in sgaps if g > 0.1)
    span = s[-1] - s[0]
    expect = span * 200.0
    print(f'[{label}] samples {len(a)}  stamp span {span:.1f} s  (expected at 200 Hz: {expect:.0f}, received {100*len(a)/max(expect,1):.0f} %)  '
          f'arrival gap median {1e3*st.median(gaps):.1f} ms  p99 {1e3*gaps_sorted[int(0.99*len(gaps))]:.0f} ms  max {1e3*max(gaps):.0f} ms  '
          f'bursts(>=20 in 2 ms) {bursts}  largest burst {largest}  stamp gaps >0.1 s {lost}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topic', default='/hwt101ct_yaw_publisher')
    ap.add_argument('--seconds', type=float, default=40)
    ap.add_argument('--best-effort', action='store_true', help='subscribe BEST_EFFORT instead of RELIABLE')
    args = ap.parse_args()
    rclpy.init()
    p = Probe(args.topic, args.best_effort)
    t0 = time.monotonic(); last = t0
    try:
        while rclpy.ok() and time.monotonic() - t0 < args.seconds:
            rclpy.spin_once(p, timeout_sec=0.05)
            if time.monotonic() - last >= 5:
                report(p, f'{time.monotonic()-t0:4.0f} s'); last = time.monotonic()
    except KeyboardInterrupt:
        pass
    report(p, 'final')
    p.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
