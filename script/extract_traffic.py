#!/usr/bin/env python3
"""Extract the moving-model (traffic) poses of a simulation bag.

Reads `/world/default/pose/info` (tf2_msgs/TFMessage, every entity pose that
Gazebo publishes) from a rosbag2 sqlite3 bag and writes one CSV row per
(sim time, model) for every model that MOVES during the recording, the robot
excluded.  Gazebo stamps these messages with zero, so sim time is rebuilt from
the bag's own `/clock` topic: each pose/info message is mapped from its receive
time to the clock value received around the same instant (linear interpolation
between clock messages), the same reconstruction the ground truth extraction
relied on.

Usage:  python3 script/extract_traffic.py <bag.db3> <out.csv> [--min-move 1.0]

Output columns: t_ns, model, x, y, z, yaw_deg
A model counts as moving when its planar position range over the bag exceeds
--min-move metres.  Static scenery is dropped; the robot (any frame whose name
contains "ackermann") is dropped.
"""
import argparse
import math
import sqlite3

import numpy as np
from rclpy.serialization import deserialize_message
from rosgraph_msgs.msg import Clock
from tf2_msgs.msg import TFMessage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag"); ap.add_argument("out")
    ap.add_argument("--min-move", type=float, default=1.0)
    ap.add_argument("--topic", default="/world/default/pose/info")
    a = ap.parse_args()

    con = sqlite3.connect(a.bag)
    tid = {n: i for i, n in con.execute("SELECT id, name FROM topics")}
    if a.topic not in tid or "/clock" not in tid:
        raise SystemExit(f"need {a.topic} and /clock in {a.bag}; have {sorted(tid)}")

    # receive time -> sim time, from /clock
    rt, st = [], []
    for recv, blob in con.execute("SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp", (tid["/clock"],)):
        c = deserialize_message(bytes(blob), Clock)
        rt.append(recv); st.append(c.clock.sec * 10**9 + c.clock.nanosec)
    rt, st = np.array(rt, float), np.array(st, float)

    per = {}      # model -> list of (t_ns, x, y, z, yaw)
    n_msgs = 0
    for recv, blob in con.execute("SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp", (tid[a.topic],)):
        m = deserialize_message(bytes(blob), TFMessage)
        t_ns = int(np.interp(recv, rt, st))
        n_msgs += 1
        for tr in m.transforms:
            name = tr.child_frame_id
            if "ackermann" in name:
                continue
            q = tr.transform.rotation
            yaw = math.degrees(math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))
            tl = tr.transform.translation
            per.setdefault(name, []).append((t_ns, tl.x, tl.y, tl.z, yaw))
    con.close()

    movers = {}
    for name, rows in per.items():
        arr = np.array(rows, float)
        if np.ptp(arr[:, 1]) > a.min_move or np.ptp(arr[:, 2]) > a.min_move:
            movers[name] = arr
    with open(a.out, "w") as fh:
        fh.write("t_ns,model,x,y,z,yaw_deg\n")
        for name, arr in sorted(movers.items()):
            for r in arr:
                fh.write(f"{int(r[0])},{name},{r[1]:.4f},{r[2]:.4f},{r[3]:.4f},{r[4]:.2f}\n")
    print(f"{n_msgs} pose/info msgs, {len(per)} frames, {len(movers)} moving -> {a.out}")
    print(f"sim time {st[0]/1e9:.2f} .. {st[-1]/1e9:.2f} s; clock msgs {len(st)}")
    for name, arr in sorted(movers.items()):
        print(f"  {name:40s} n={len(arr):5d}  x {arr[:,1].min():7.2f}..{arr[:,1].max():7.2f}  y {arr[:,2].min():7.2f}..{arr[:,2].max():7.2f}")


if __name__ == "__main__":
    main()
