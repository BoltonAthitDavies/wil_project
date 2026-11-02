#!/usr/bin/env python3
"""Publish one or more .ply point clouds as latched PointCloud2 topics for RViz.

No extra viewer needed: RViz is already installed and this is the same display
used for /cloud_map. Each file becomes its own topic, so two clouds (say an
exported map and the same map with a different max range) can be overlaid and
toggled.

    python3 script/view_ply.py map/real_311026_selfcal/real_311026_selfcal_cloud.ply \
        [other.ply ...] [--frame map] [--stride 1]
    rviz2   # Fixed Frame: map; Add -> PointCloud2 -> topic /ply/<file stem>

Reads binary little-endian or ASCII PLY with x y z and optional red green blue
(what rtabmap-export writes); other properties are ignored. --stride N keeps
every Nth point if RViz struggles with a big cloud.
"""
import argparse
import os
import re
import struct
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

TMAP = {'float': '<f4', 'float32': '<f4', 'double': '<f8', 'uchar': 'u1', 'uint8': 'u1',
        'int': '<i4', 'uint': '<u4', 'short': '<i2', 'ushort': '<u2', 'char': 'i1'}


def read_ply(path):
    with open(path, 'rb') as f:
        header = b''
        while not header.endswith(b'end_header\n'):
            line = f.readline()
            if not line:
                sys.exit(f'{path}: no end_header')
            header += line
        h = header.decode()
        fmt = re.search(r'format (\w+)', h).group(1)
        vsec = h.split('element vertex')[1]
        vsec = vsec.split('element')[0] if 'element' in vsec else vsec
        n = int(re.match(r'\s*(\d+)', vsec).group(1))
        props = re.findall(r'property (\w+) (\w+)', vsec)
        if fmt == 'ascii':
            rows = np.loadtxt(f, max_rows=n)
            names = [nm for _, nm in props]
            return {nm: rows[:, i] for i, nm in enumerate(names)}
        dt = np.dtype([(nm, TMAP[t]) for t, nm in props])
        a = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
        return {nm: a[nm] for _, nm in props}


def to_msg(d, frame, stride):
    xyz = np.stack([d['x'], d['y'], d['z']], 1).astype(np.float32)[::stride]
    fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
              for i, n in enumerate('xyz')]
    if all(k in d for k in ('red', 'green', 'blue')):
        rgb = (d['red'].astype(np.uint32) << 16 | d['green'].astype(np.uint32) << 8
               | d['blue'].astype(np.uint32))[::stride]
        fields.append(PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1))
        data = np.empty(len(xyz), dtype=[('xyz', '<f4', 3), ('rgb', '<u4')])
        data['xyz'] = xyz
        data['rgb'] = rgb
        step = 16
    else:
        data = xyz
        step = 12
    msg = PointCloud2()
    msg.header = Header(frame_id=frame)
    msg.height, msg.width = 1, len(xyz)
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = step
    msg.row_step = step * len(xyz)
    msg.is_dense = True
    msg.data = data.tobytes()
    return msg, len(xyz)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ply', nargs='+')
    ap.add_argument('--frame', default='map')
    ap.add_argument('--stride', type=int, default=1)
    a, ros_args = ap.parse_known_args()
    rclpy.init(args=ros_args)
    node = Node('view_ply')
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pubs = []
    for p in a.ply:
        stem = re.sub(r'[^A-Za-z0-9_]', '_', os.path.splitext(os.path.basename(p))[0])
        msg, n = to_msg(read_ply(p), a.frame, a.stride)
        pub = node.create_publisher(PointCloud2, f'/ply/{stem}', qos)
        pub.publish(msg)
        pubs.append((pub, msg))
        node.get_logger().info(f'{p}: {n} points on /ply/{stem} (frame {a.frame})')
    # Latched via transient-local QoS; republish slowly anyway for late RViz starts.
    node.create_timer(5.0, lambda: [pub.publish(m) for pub, m in pubs])
    node.get_logger().info('open rviz2, Fixed Frame = %s, Add -> PointCloud2 -> /ply/...' % a.frame)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
