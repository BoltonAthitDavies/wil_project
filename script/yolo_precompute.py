#!/usr/bin/env python3
"""Run the YOLO detector over a bag ONCE and freeze the boxes to a CSV.

    python3 script/yolo_precompute.py dataset/dataset_allsensor_realistic_003 \
        --weights weight/best.pt --out dataset/dataset_allsensor_realistic_003/dets_best.csv

Then replay with the SLAM node reading the file instead of a live detector:

    ros2 launch orbslam3_ros2 wil_sim_stereo_imu.launch.py filter:=true \
        dets_csv:=/abs/path/dets_best.csv ...

WHY THIS EXISTS
    The live detector is a separate process racing the tracker for the same
    machine. On this laptop it runs one to four frames behind ORB-SLAM3 and the
    tracker refuses anything older than det_max_age (150 ms = four frames), so
    whether a given frame gets a mask depends on CPU temperature: 33% of frames
    masked on a warm run, 91% on a cool one, same bag, same config. A filter-ON
    run was therefore a different method every time it was run, and could not be
    compared against filter-OFF.

    This script takes the detector out of the real-time loop. Every frame gets
    the boxes the model produces for THAT frame, keyed by the frame's own header
    stamp, and the node looks them up by exact timestamp. Two runs on the same
    bag then mask exactly the same frames with exactly the same boxes, and any
    difference between them is the tracker, not the laptop.

OUTPUT FORMAT  (one row per BOX; frames with no boxes still get one row so the
node can tell "nothing detected" from "frame not in file")

    # weights=... conf=... iou=... imgsz=... classes=...   <- provenance, '#' lines
    timestamp_ns,class,score,x0,y0,x1,y1
    87748000000,box,0.9123,412.0,200.5,530.2,318.9
    87748000000,bucket,0.6611,...
    87781000000,,0,0,0,0,0          <- frame seen, nothing detected

    Pixel coordinates are in the ORIGINAL image resolution, x0<x1, y0<y1, the
    same convention vision_msgs boxes are converted to inside the nodes.

The class set follows the weight, exactly as the live node's `auto` does: every
class the checkpoint knows is written. Narrow with --classes if you must, but
the recorded provenance line then says so.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

WORKSPACE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

COMPRESSED = '/cam0/image_raw/compressed'
RAW = '/cam0/image_raw'


def iter_frames(uri: str, topic: str | None):
    """Yield (timestamp_ns, bgr) for every cam0 frame in the bag, in bag order.

    Picks the compressed topic when the bag has it (every bag in this project so
    far) and falls back to raw Image otherwise. Stamps come from the HEADER, not
    the bag's receive time: the header is what the SLAM node keys its lookup on.
    """
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CompressedImage, Image

    r = SequentialReader()
    r.open(StorageOptions(uri=uri, storage_id='sqlite3'), ConverterOptions('cdr', 'cdr'))
    have = {t.name for t in r.get_all_topics_and_types()}
    if topic is None:
        topic = COMPRESSED if COMPRESSED in have else RAW
    if topic not in have:
        raise SystemExit(f'{topic} not in bag (topics: {sorted(have)})')
    r.set_filter(StorageFilter(topics=[topic]))
    compressed = topic.endswith('/compressed')
    print(f'topic    : {topic}')
    while r.has_next():
        _, data, _ = r.read_next()
        if compressed:
            m = deserialize_message(data, CompressedImage)
            bgr = cv2.imdecode(np.frombuffer(m.data, np.uint8), cv2.IMREAD_COLOR)
        else:
            m = deserialize_message(data, Image)
            buf = np.frombuffer(m.data, np.uint8)
            if m.encoding in ('rgb8', 'bgr8'):
                img = buf.reshape(m.height, m.width, 3)
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if m.encoding == 'rgb8' else img
            elif m.encoding == 'mono8':
                bgr = cv2.cvtColor(buf.reshape(m.height, m.width), cv2.COLOR_GRAY2BGR)
            else:
                raise SystemExit(f'unsupported encoding {m.encoding}')
        if bgr is None:
            continue
        ts = int(m.header.stamp.sec) * 10**9 + int(m.header.stamp.nanosec)
        yield ts, bgr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('bag')
    ap.add_argument('--out', required=True, help='CSV to write')
    ap.add_argument('--weights', default=os.path.join(WORKSPACE, 'weight', 'best.pt'))
    ap.add_argument('--classes', nargs='*', default=None,
                    help='class NAMES to keep; default = every class the weight knows')
    ap.add_argument('--conf', type=float, default=0.35)
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--topic', default=None, help='override the image topic')
    ap.add_argument('--limit', type=int, default=0, help='only the first N frames')
    a = ap.parse_args()

    from ultralytics import YOLO
    import torch
    model = YOLO(a.weights)
    names = {int(k): str(v) for k, v in dict(model.names).items()}
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu') \
        if a.device == 'auto' else a.device

    # Same resolution rule as dynamic_detector_node.py: names, case-insensitive,
    # against the checkpoint's own names dict; unknown names are an error here
    # rather than a warning, because a precomputed file is meant to be trusted.
    if a.classes:
        by_name = {v.lower(): k for k, v in names.items()}
        unknown = [c for c in a.classes if c.lower() not in by_name]
        if unknown:
            raise SystemExit(f'not in this weight: {unknown} (knows {list(names.values())})')
        keep = sorted(by_name[c.lower()] for c in a.classes)
    else:
        keep = sorted(names)
    print(f'weights  : {a.weights}')
    print(f'device   : {device}')
    print(f'classes  : {[names[i] for i in keep]}  ({len(keep)} of {len(names)})')

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + '.partial'
    n_frames = n_boxes = 0
    t_start = time.monotonic()
    with open(tmp, 'w') as f:
        f.write(f'# weights={os.path.abspath(a.weights)} conf={a.conf} iou={a.iou} '
                f'imgsz={a.imgsz} classes={",".join(names[i] for i in keep)} '
                f'bag={os.path.abspath(a.bag)}\n')
        f.write('timestamp_ns,class,score,x0,y0,x1,y1\n')
        for ts, bgr in iter_frames(a.bag, a.topic):
            res = model.predict(bgr, imgsz=a.imgsz, conf=a.conf, iou=a.iou,
                                classes=keep, device=device, verbose=False)[0]
            rows = []
            if res.boxes is not None and len(res.boxes):
                xy = res.boxes.xyxy.cpu().numpy()
                sc = res.boxes.conf.cpu().numpy()
                cl = res.boxes.cls.cpu().numpy().astype(int)
                for (x0, y0, x1, y1), s, c in zip(xy, sc, cl):
                    rows.append(f'{ts},{names[int(c)]},{s:.4f},{x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f}\n')
            if not rows:
                rows.append(f'{ts},,0,0,0,0,0\n')
            f.writelines(rows)
            n_frames += 1
            n_boxes += sum(1 for r in rows if not r.split(',')[1] == '')
            if n_frames % 500 == 0:
                print(f'  {n_frames} frames, {n_boxes} boxes, '
                      f'{n_frames / (time.monotonic() - t_start):.1f} fps', flush=True)
            if a.limit and n_frames >= a.limit:
                break
    # Rename only on success: a half-written file must never be mistaken for a
    # complete one by a node that will happily mask nothing past its last row.
    os.replace(tmp, a.out)
    dt = time.monotonic() - t_start
    print(f'\nframes   : {n_frames}\nboxes    : {n_boxes}  ({n_boxes / max(n_frames, 1):.2f} per frame)'
          f'\ntime     : {dt:.0f} s  ({n_frames / max(dt, 1e-9):.1f} fps)\nwrote    : {a.out}')


if __name__ == '__main__':
    main()
