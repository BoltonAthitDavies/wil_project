#!/usr/bin/env python3
"""Extract /ground_truth/odometry from a rosbag2 sqlite3 bag into a vio.csv-shaped CSV.

Output columns: t_ns, x, y, z, qw, qx, qy, qz, vx, vy, vz
Timestamps come from the message HEADER (sim time), matching VINS' vio.csv.
"""
import sqlite3, sys
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry

BAG = sys.argv[1] if len(sys.argv) > 1 else "/home/ambushee/wil_project/sim/sim_0.db3"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/home/ambushee/output/wil_sim_stereo_imu/ground_truth.csv"
TOPIC = "/ground_truth/odometry"

con = sqlite3.connect(BAG)
tid = con.execute("SELECT id FROM topics WHERE name=?", (TOPIC,)).fetchone()
if tid is None:
    sys.exit(f"topic {TOPIC} not in {BAG}")

rows = []
frames = set()
for (blob,) in con.execute("SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp", (tid[0],)):
    m = deserialize_message(bytes(blob), Odometry)
    frames.add((m.header.frame_id, m.child_frame_id))
    t = m.header.stamp.sec * 10**9 + m.header.stamp.nanosec
    p, q = m.pose.pose.position, m.pose.pose.orientation
    v = m.twist.twist.linear
    rows.append((t, p.x, p.y, p.z, q.w, q.x, q.y, q.z, v.x, v.y, v.z))
con.close()

with open(OUT, "w") as f:
    for r in rows:
        f.write("%d," % r[0] + ",".join("%.5f" % x for x in r[1:]) + "\n")

print(f"{len(rows)} msgs -> {OUT}")
print("frames (header, child):", frames)
print("header stamp range: %.3f .. %.3f s" % (rows[0][0] / 1e9, rows[-1][0] / 1e9))
