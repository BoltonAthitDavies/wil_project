#!/usr/bin/env python3
"""Measure how much the wheels actually slipped in a recorded bag.

    python3 script/check_slip.py dataset/dataset_allsensor_000
    python3 script/check_slip.py dataset/dataset_*/            # several at once

WHAT IT MEASURES
    Encoder-integrated distance against true path length:

        encoder = mean(|d(rear wheel angle)|) * WHEEL_RADIUS, summed
        truth   = sum of |d(ground-truth position)|

    The joint angle Gazebo reports is the true solver angle, so the encoder
    over-reads exactly when the tyres break traction -- which is what a real
    encoder does. A ratio above 1 is slip. A ratio below 1 is NOT slip; it is a
    rolling-radius or wheel-radius error, and it means WHEEL_RADIUS here is
    slightly wrong for that run.

WHY IT EXISTS
    make_slippery_floor.py changes a friction coefficient. That is an input, not
    a result. This is the measurement that says what the change actually did, and
    it is the number that belongs in the report -- "we set mu=0.4" is a setting,
    "the encoder over-read by 4.1%" is evidence.

    Needs /model/<robot>/joint_state and /ground_truth/odometry in the bag.
"""

import glob
import os
import sqlite3
import sys

import numpy as np
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState

# model.sdf: wheel collision radius.
WHEEL_RADIUS = 0.0585
REAR = ('rear_left_wheel_joint', 'rear_right_wheel_joint')


def read(cur, topics, topic, cls):
    if topic not in topics:
        return []
    return [deserialize_message(b[0], cls) for b in cur.execute(
        'SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp',
        (topics[topic],))]


def measure(bagdir):
    db = sorted(glob.glob(os.path.join(bagdir, '*.db3')))
    if not db:
        return None, 'no .db3'
    con = sqlite3.connect('file:%s?mode=ro' % db[0], uri=True)
    cur = con.cursor()
    topics = {n: i for i, n in cur.execute('SELECT id,name FROM topics')}
    js_topic = next((t for t in topics if t.endswith('/joint_state')), None)
    if js_topic is None:
        return None, 'no joint_state topic'
    js = read(cur, topics, js_topic, JointState)
    gt = read(cur, topics, topics and '/ground_truth/odometry', Odometry)
    con.close()
    if not js or not gt:
        return None, 'joint_state=%d ground_truth=%d' % (len(js), len(gt))

    names = js[0].name
    if not all(j in names for j in REAR):
        return None, 'rear wheel joints absent'
    idx = [names.index(j) for j in REAR]
    pos = np.array([[m.position[i] for i in idx] for m in js])
    # Mean of the two rear wheels: on an Ackermann chassis they differ through a
    # turn, and their mean is the distance the axle centre rolled.
    enc = np.abs(np.diff(pos, axis=0)).mean(axis=1).sum() * WHEEL_RADIUS

    xy = np.array([[m.pose.pose.position.x, m.pose.pose.position.y] for m in gt])
    truth = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
    return (enc, truth, enc / truth), None


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    print('%-42s %9s %9s %8s  %s' % ('bag', 'encoder', 'truth', 'ratio', 'reading'))
    print('-' * 88)
    for a in args:
        for bagdir in sorted(glob.glob(a)):
            if not os.path.isdir(bagdir):
                continue
            res, err = measure(bagdir)
            name = os.path.basename(bagdir.rstrip('/'))
            if res is None:
                print('%-42s %s' % (name, err))
                continue
            enc, truth, ratio = res
            if ratio > 1.001:
                reading = 'slip: encoder over-reads %+.2f%%' % ((ratio - 1) * 100)
            elif ratio < 0.999:
                reading = 'NOT slip: under-reads %+.2f%% (radius error)' % ((ratio - 1) * 100)
            else:
                reading = 'no measurable slip'
            print('%-42s %9.2f %9.2f %8.4f  %s' % (name, enc, truth, ratio, reading))


if __name__ == '__main__':
    main()
