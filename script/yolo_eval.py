#!/usr/bin/env python3
"""Score the YOLO dynamic-object detector against Gazebo ground truth.

    python3 script/yolo_eval.py dataset/dataset_dynamic_nofloortexture_00_000 \
        --out output/yolo_eval/dyn00 --overlay 40

Ground truth comes from `/world/default/pose/info`, which carries the TRUE world
pose of every model. Each prop's collision mesh is projected into cam0 and its 2D
extent taken, giving a box a human annotator would have drawn. YOLO is then run on
the same frame and matched to those boxes by IoU.

OUTPUTS (both keyed by the image timestamp in ns, exactly like vio.csv, so they
join straight onto a trajectory with a merge on column 0):

  yolo_eval.csv        one row per frame: counts, TP/FP/FN, precision, recall,
                       mask coverage, and the moving/static split
  yolo_detections.csv  one row per predicted box: class, score, matched or not,
                       IoU, and whether the GT it matched was moving
  yolo_confusion.csv   object-detection confusion counts after class-agnostic
                       IoU association; includes a background row and column
  overlay/*.jpg        GREEN = ground truth, RED = YOLO, YELLOW = ignored
                       (occluded). LOOK AT THESE FIRST -- if the green boxes do
                       not sit on the objects, every number below is meaningless.

VALIDATE THE PROJECTION BEFORE TRUSTING A METRIC. A wrong extrinsic or a missed
mesh unit produces confident, plausible, completely wrong numbers.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yolo_gt as G  # noqa: E402

WORKSPACE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Ideal gz pinhole: fx = fy = (width/2)/tan(hfov/2), principal point dead centre,
# every distortion term zero. Taken from config/wil_sim/stereo_imu.yaml.
FX = FY = 640.0
CX, CY = 640.0, 360.0
W, H = 1280, 720


def read_extrinsic(path: str, key: str) -> np.ndarray:
    """4x4 !!opencv-matrix out of a VINS/ORB yaml.

    Parsed by hand: cv2.FileStorage mis-reads this particular block (the entries
    are written as bare integers under dt: d) and returns garbage on the order of
    1e18 for the rotation while leaving the translation column plausible -- which
    is exactly the kind of wrong that survives a glance.
    """
    txt = open(path).read()
    m = re.search(rf'^{re.escape(key)}:\s*!!opencv-matrix.*?data:\s*\[(.*?)\]',
                  txt, re.S | re.M)
    if not m:
        raise SystemExit(f'{key} not found in {path}')
    v = np.fromstring(m.group(1).replace('\n', ' '), sep=',')
    if v.size != 16:
        raise SystemExit(f'{key}: expected 16 values, got {v.size}')
    return v.reshape(4, 4)


def quat_xyzw_to_R(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w),     s * (x * z + y * w)],
        [s * (x * y + z * w),     1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w),     s * (y * z + x * w),     1 - s * (x * x + y * y)],
    ])


class PoseTrack:
    """World pose of one model over time, sampled at pose/info rate (~58 Hz)."""

    def __init__(self):
        self.t, self.p, self.q = [], [], []

    def finish(self):
        self.t = np.asarray(self.t)
        self.p = np.asarray(self.p)
        self.q = np.asarray(self.q)
        o = np.argsort(self.t)
        self.t, self.p, self.q = self.t[o], self.p[o], self.q[o]

    def at(self, ts: float):
        """(position, R) at ts. Position linearly interpolated, rotation nearest.

        At 58 Hz the worst nearest-rotation error is ~8.6 ms, which at this rig's
        turn rates is a fraction of a degree -- a few pixels at f=640. Not worth a
        slerp.
        """
        i = int(np.searchsorted(self.t, ts))
        i = max(1, min(i, len(self.t) - 1))
        t0, t1 = self.t[i - 1], self.t[i]
        a = 0.0 if t1 <= t0 else (ts - t0) / (t1 - t0)
        a = float(np.clip(a, 0.0, 1.0))
        p = self.p[i - 1] * (1 - a) + self.p[i] * a
        q = self.q[i] if a > 0.5 else self.q[i - 1]
        return p, quat_xyzw_to_R(q)

    def speed(self, ts: float, win: float = 0.5) -> float:
        """Max world-frame speed within +/- win seconds. Labels moving vs static."""
        lo = np.searchsorted(self.t, ts - win)
        hi = np.searchsorted(self.t, ts + win)
        if hi - lo < 2:
            return 0.0
        seg_p, seg_t = self.p[lo:hi], self.t[lo:hi]
        d = np.linalg.norm(np.diff(seg_p, axis=0), axis=1)
        dt = np.maximum(np.diff(seg_t), 1e-6)
        return float((d / dt).max())


def load_bag(uri: str):
    """-> (pose tracks keyed by model name, [(image_stamp_ns, jpeg bytes)]).

    THE CLOCK PROBLEM. `/world/default/pose/info` arrives from the gz->ROS bridge
    with header.stamp LEFT AT ZERO -- all 5785 messages in the dynamic bags claim
    t=0. Using it verbatim pins every prop and the robot to their pose at the
    start of the run, which produces ground-truth boxes that are the right SIZE
    and in the wrong PLACE, and makes every prop look stationary. It looks like a
    calibration error and is not one.

    The bag's per-message RECEIVE stamp is wall clock, and wall does not track sim
    by a constant offset (measured std 6.65 s over 100 s -- the RTF wandered). So
    we rebuild sim time from `/clock`, which carries sim time in its payload and
    wall time in its receive stamp, giving a dense (~770 Hz) wall->sim map to
    interpolate the pose stamps through.
    """
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import CompressedImage
    from tf2_msgs.msg import TFMessage

    r = SequentialReader()
    r.open(StorageOptions(uri=uri, storage_id='sqlite3'), ConverterOptions('cdr', 'cdr'))
    tracks: dict[str, PoseTrack] = {}
    frames = []
    clock_wall, clock_sim = [], []
    pose_wall = []
    while r.has_next():
        topic, data, recv = r.read_next()
        if topic == '/clock':
            m = deserialize_message(data, Clock)
            clock_wall.append(recv * 1e-9)
            clock_sim.append(m.clock.sec + m.clock.nanosec * 1e-9)
        elif topic == '/world/default/pose/info':
            m = deserialize_message(data, TFMessage)
            w = recv * 1e-9
            for tr in m.transforms:
                k = tracks.setdefault(tr.child_frame_id, PoseTrack())
                t_ = tr.transform.translation
                q_ = tr.transform.rotation
                k.t.append(w)                      # wall for now; remapped below
                k.p.append((t_.x, t_.y, t_.z))
                k.q.append((q_.x, q_.y, q_.z, q_.w))
            pose_wall.append(w)
        elif topic == '/cam0/image_raw/compressed':
            m = deserialize_message(data, CompressedImage)
            ts_ns = int(m.header.stamp.sec) * 10**9 + int(m.header.stamp.nanosec)
            frames.append((ts_ns, bytes(m.data)))

    if len(clock_wall) < 2:
        raise SystemExit('/clock missing or too short -- cannot rebuild sim time '
                         'for pose/info, whose own stamps are zero')
    cw = np.asarray(clock_wall); cs = np.asarray(clock_sim)
    o = np.argsort(cw); cw, cs = cw[o], cs[o]
    for k in tracks.values():                      # wall -> sim
        k.t = list(np.interp(np.asarray(k.t), cw, cs))
    for k in tracks.values():
        k.finish()
    frames.sort(key=lambda x: x[0])
    return tracks, frames


def iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bag')
    ap.add_argument('--out', required=True, help='output directory (created)')
    ap.add_argument('--weights', default=os.path.join(WORKSPACE, 'weight', 'best.pt'))
    ap.add_argument('--config', default=os.path.join(
        WORKSPACE, 'vins_fusion_ros2', 'config', 'wil_sim', 'stereo_imu.yaml'))
    ap.add_argument('--conf', type=float, default=0.35)
    ap.add_argument('--iou-nms', type=float, default=0.5)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--device', default='auto',
                    help="Ultralytics device (for example 'cuda:0' or 'cpu'); "
                         "'auto' selects CUDA when available, otherwise CPU")
    ap.add_argument('--iou-match', type=float, default=0.5,
                    help='IoU for a prediction to count as a true positive')
    ap.add_argument('--min-box-px', type=float, default=12.0,
                    help='ignore GT boxes smaller than this on a side')
    ap.add_argument('--ignore-cover', type=float, default=0.5,
                    help="prediction this covered by unlabelled box-like scenery "
                         "(shelves etc) is ignored rather than counted a FP")
    ap.add_argument('--no-ignore', action='store_true',
                    help="score EVERY ground-truth object and every prediction, "
                         "disabling the small-box, occlusion and unlabelled-"
                         "scenery exemptions. The exemptions exist because a "
                         "projected mesh extent is not a box a human would have "
                         "drawn, so scoring them is unfair to the detector -- but "
                         "the training split has no such concept, so a comparison "
                         "against training metrics needs this mode. Use it ONLY "
                         "for that comparison; the headline numbers stay in the "
                         "default mode. NOTE it does not touch --min-box-px: a "
                         "sub-12-pixel speck goes unlabelled in a human-annotated "
                         "set too, so dropping it is not the same kind of "
                         "exemption. Lower that separately if the comparison "
                         "needs it, and say which value was used.")
    ap.add_argument('--occlusion', type=float, default=0.70,
                    help='GT covered more than this by NEARER boxes becomes "ignore"')
    ap.add_argument('--move-thresh', type=float, default=0.05,
                    help='m/s above which a prop counts as moving')
    ap.add_argument('--overlay', type=int, default=0, help='dump N annotated frames')
    ap.add_argument('--limit', type=int, default=0, help='only process N frames')
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    body_T_cam0 = read_extrinsic(a.config, 'body_T_cam0')

    print(f'bag      : {a.bag}')
    tracks, frames = load_bag(a.bag)
    print(f'frames   : {len(frames)}   pose tracks: {len(tracks)}')

    props, ignores = [], []
    for name, tr in tracks.items():
        p = G.model_prefix(name)
        if p:
            props.append((name, p, G.CLASS_OF[p], tr))
            continue
        q = G.ignore_prefix(name)
        if q:
            ignores.append((name, q, tr))
    print(f'GT props : {len(props)}   ignore-region models: {len(ignores)}')
    if not props:
        raise SystemExit('no props matched CLASS_OF -- check the model naming')

    robot = tracks.get('ackermann_robot_001')
    if robot is None:
        raise SystemExit('ackermann_robot_001 not in pose/info')

    from ultralytics import YOLO
    import torch
    model = YOLO(a.weights)
    names = dict(model.names)
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu') \
        if a.device == 'auto' else a.device
    print(f'model    : {a.weights}  device={device}  classes={list(names.values())}')

    ov_dir = os.path.join(a.out, 'overlay')
    if a.overlay:
        os.makedirs(ov_dir, exist_ok=True)
    ov_every = max(1, len(frames) // a.overlay) if a.overlay else 0

    fe = open(os.path.join(a.out, 'yolo_eval.csv'), 'w')
    fe.write('timestamp_ns,n_gt,n_gt_moving,n_gt_static,n_ignored,n_pred,tp,fp,fn,'
             'precision,recall,tp_moving,fn_moving,tp_static,fp_static,'
             'gt_area_frac,gt_area_frac_moving,pred_area_frac\n')
    fd = open(os.path.join(a.out, 'yolo_detections.csv'), 'w')
    fd.write('timestamp_ns,pred_class,score,x0,y0,x1,y1,matched,iou,gt_class,gt_moving,gt_name\n')

    tot = dict(tp=0, fp=0, fn=0, tpm=0, fnm=0, tps=0, fps=0, gt=0, gtm=0, pred=0,
               ign=0, pign=0)
    eval_classes = sorted(set(G.CLASS_OF.values()) | set(names.values()))
    confusion = {(gt, pred): 0
                 for gt in eval_classes + ['background']
                 for pred in eval_classes + ['background']}
    todo = frames[: a.limit] if a.limit else frames

    for fi, (ts_ns, jpg) in enumerate(todo):
        ts = ts_ns * 1e-9
        p_wb, R_wb = robot.at(ts)
        T_wb = np.eye(4)
        T_wb[:3, :3], T_wb[:3, 3] = R_wb, p_wb
        T_wc = T_wb @ body_T_cam0
        R_cw = T_wc[:3, :3].T
        t_cw = -R_cw @ T_wc[:3, 3]

        gts = []
        for name, prefix, cls, tr in props:
            p_wo, R_wo = tr.at(ts)
            v_w = (R_wo @ G.vertices(prefix).T).T + p_wo
            v_c = (R_cw @ v_w.T).T + t_cw
            z = v_c[:, 2]
            if not np.any(z > 0.05):
                continue                                  # entirely behind camera
            v_c = v_c[z > 0.05]
            u = FX * v_c[:, 0] / v_c[:, 2] + CX
            v = FY * v_c[:, 1] / v_c[:, 2] + CY
            x0, x1 = float(u.min()), float(u.max())
            y0, y1 = float(v.min()), float(v.max())
            cx0, cy0 = max(0.0, x0), max(0.0, y0)
            cx1, cy1 = min(float(W), x1), min(float(H), y1)
            if cx1 - cx0 < a.min_box_px or cy1 - cy0 < a.min_box_px:
                continue                                  # offscreen or a speck
            gts.append(dict(box=(cx0, cy0, cx1, cy1), cls=cls, name=name,
                            depth=float(np.median(v_c[:, 2])),
                            moving=tr.speed(ts) > a.move_thresh, ignore=False,
                            matched=False))

        # Occlusion: a prop hidden behind a nearer one still projects to a perfect
        # box, and YOLO rightly will not detect it. Counting that as a miss would
        # punish the detector for the ground truth's blindness. Approximate the
        # test with a coarse depth-ordered stencil and mark heavily covered boxes
        # "ignore" -- they then neither reward nor penalise anything.
        if gts:
            S = 4
            stencil = np.full((H // S, W // S), np.inf, np.float32)
            for g in sorted(gts, key=lambda d: d['depth']):
                x0, y0, x1, y1 = (int(c / S) for c in g['box'])
                if x1 > x0 and y1 > y0:
                    reg = stencil[y0:y1, x0:x1]
                    np.minimum(reg, g['depth'], out=reg)
            for g in gts:
                x0, y0, x1, y1 = (int(c / S) for c in g['box'])
                if x1 <= x0 or y1 <= y0:
                    continue
                reg = stencil[y0:y1, x0:x1]
                covered = float((reg < g['depth'] - 1e-3).mean())
                if covered > a.occlusion and not a.no_ignore:
                    g['ignore'] = True

        # Unlabelled but box-like scenery (loaded shelves, pallet jacks, desks).
        ign_boxes = []
        for name, prefix, tr in ignores:
            p_wo, R_wo = tr.at(ts)
            v_w = (R_wo @ G.vertices(prefix).T).T + p_wo
            v_c = (R_cw @ v_w.T).T + t_cw
            z = v_c[:, 2]
            if not np.any(z > 0.05):
                continue
            v_c = v_c[z > 0.05]
            u = FX * v_c[:, 0] / v_c[:, 2] + CX
            v = FY * v_c[:, 1] / v_c[:, 2] + CY
            bx = (max(0.0, float(u.min())), max(0.0, float(v.min())),
                  min(float(W), float(u.max())), min(float(H), float(v.max())))
            if bx[2] - bx[0] > 2 and bx[3] - bx[1] > 2:
                ign_boxes.append(bx)

        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        res = model.predict(img, imgsz=a.imgsz, conf=a.conf, iou=a.iou_nms,
                            device=device, verbose=False)[0]
        preds = []
        if res.boxes is not None and len(res.boxes):
            xy = res.boxes.xyxy.cpu().numpy()
            sc = res.boxes.conf.cpu().numpy()
            cl = res.boxes.cls.cpu().numpy().astype(int)
            for (x0, y0, x1, y1), s, c in zip(xy, sc, cl):
                preds.append(dict(box=(float(x0), float(y0), float(x1), float(y1)),
                                  cls=names.get(int(c), str(c)), score=float(s)))
        preds.sort(key=lambda d: -d['score'])

        # Greedy highest-confidence-first matching, same class required.
        tp = fp = 0
        for p in preds:
            best, bg = 0.0, None
            for g in gts:
                if g['matched'] or g['cls'] != p['cls']:
                    continue
                o = iou(p['box'], g['box'])
                if o > best:
                    best, bg = o, g
            if bg is not None and best >= a.iou_match:
                bg['matched'] = True
                p['gt'] = bg
                p['iou'] = best
                if not bg['ignore']:
                    tp += 1
                    tot['tpm' if bg['moving'] else 'tps'] += 1
            else:
                p['gt'] = None
                p['iou'] = best
                # Before calling it a false positive, ask whether it landed on
                # unlabelled box-like scenery. Intersection over the PREDICTION's
                # own area, not IoU: a small box on a big shelf is fully explained
                # by the shelf even though their IoU is tiny.
                pa = max((p['box'][2] - p['box'][0]) * (p['box'][3] - p['box'][1]), 1e-6)
                cov = 0.0
                for b in ign_boxes:
                    iw = max(0.0, min(p['box'][2], b[2]) - max(p['box'][0], b[0]))
                    ih = max(0.0, min(p['box'][3], b[3]) - max(p['box'][1], b[1]))
                    cov = max(cov, iw * ih / pa)
                if cov > a.ignore_cover and not a.no_ignore:
                    p['ignored'] = True
                    tot['pign'] += 1
                else:
                    fp += 1
                    tot['fps'] += 1
            fd.write(f"{ts_ns},{p['cls']},{p['score']:.4f},"
                     f"{p['box'][0]:.1f},{p['box'][1]:.1f},{p['box'][2]:.1f},{p['box'][3]:.1f},"
                     f"{int(p['gt'] is not None)},{p['iou']:.3f},"
                     f"{p['gt']['cls'] if p['gt'] else ''},"
                     f"{int(p['gt']['moving']) if p['gt'] else ''},"
                     f"{p['gt']['name'] if p['gt'] else ''}\n")

        scored = [g for g in gts if not g['ignore']]
        fn = sum(1 for g in scored if not g['matched'])
        n_move = sum(1 for g in scored if g['moving'])
        fnm = sum(1 for g in scored if g['moving'] and not g['matched'])

        # Diagnostic class confusion uses a separate, class-agnostic IoU match.
        # This exposes wrong-class localisations that the headline detector
        # scoring correctly represents as one FP plus one FN. Predictions already
        # excluded by the occlusion/scenery policy remain excluded here as well.
        eligible_preds = [
            p for p in preds
            if not p.get('ignored')
            and not (p.get('gt') is not None and p['gt']['ignore'])
        ]
        used_gt = set()
        for p in eligible_preds:  # already sorted by descending confidence
            best_i, best_iou = None, 0.0
            for gi, g in enumerate(scored):
                if gi in used_gt:
                    continue
                overlap = iou(p['box'], g['box'])
                if overlap > best_iou:
                    best_i, best_iou = gi, overlap
            if best_i is not None and best_iou >= a.iou_match:
                used_gt.add(best_i)
                confusion[(scored[best_i]['cls'], p['cls'])] += 1
            else:
                confusion[('background', p['cls'])] += 1
        for gi, g in enumerate(scored):
            if gi not in used_gt:
                confusion[(g['cls'], 'background')] += 1

        def area(bs):
            m = np.zeros((H // 4, W // 4), np.uint8)
            for b in bs:
                x0, y0, x1, y1 = (int(c / 4) for c in b)
                m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 1
            return float(m.mean())

        prec = tp / (tp + fp) if tp + fp else float('nan')
        rec = tp / (tp + fn) if tp + fn else float('nan')
        fe.write(f"{ts_ns},{len(scored)},{n_move},{len(scored)-n_move},"
                 f"{sum(1 for g in gts if g['ignore'])},{len(preds)},{tp},{fp},{fn},"
                 f"{prec:.4f},{rec:.4f},"
                 f"{sum(1 for g in scored if g['moving'] and g['matched'])},{fnm},"
                 f"{sum(1 for g in scored if not g['moving'] and g['matched'])},{fp},"
                 f"{area([g['box'] for g in scored]):.4f},"
                 f"{area([g['box'] for g in scored if g['moving']]):.4f},"
                 f"{area([p['box'] for p in preds]):.4f}\n")

        tot['tp'] += tp; tot['fp'] += fp; tot['fn'] += fn
        tot['fnm'] += fnm; tot['gt'] += len(scored); tot['gtm'] += n_move
        tot['pred'] += len(preds); tot['ign'] += sum(1 for g in gts if g['ignore'])

        if ov_every and fi % ov_every == 0:
            c = img.copy()
            for g in gts:
                col = (0, 220, 255) if g['ignore'] else (0, 220, 0)
                x0, y0, x1, y1 = (int(v) for v in g['box'])
                cv2.rectangle(c, (x0, y0), (x1, y1), col, 2)
                cv2.putText(c, g['cls'] + ('*' if g['moving'] else ''), (x0, max(12, y0 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
            for b in ign_boxes:
                cv2.rectangle(c, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                              (200, 200, 200), 1)
            for p in preds:
                x0, y0, x1, y1 = (int(v) for v in p['box'])
                col = (160, 160, 255) if p.get('ignored') else (0, 0, 255)
                cv2.rectangle(c, (x0, y0), (x1, y1), col, 2)
                cv2.putText(c, f"{p['cls']} {p['score']:.2f}", (x0, min(H - 4, y1 + 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            cv2.putText(c, 'GREEN=truth (*=moving)  RED=yolo  YELLOW=occluded/ignored',
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imwrite(os.path.join(ov_dir, f'{fi:05d}.jpg'), c)

        if fi % 200 == 0:
            print(f'  {fi}/{len(todo)}', flush=True)

    fe.close(); fd.close()

    with open(os.path.join(a.out, 'yolo_confusion.csv'), 'w') as fc:
        fc.write('ground_truth,predicted,count\n')
        for gt in eval_classes + ['background']:
            for pred in eval_classes + ['background']:
                fc.write(f'{gt},{pred},{confusion[(gt, pred)]}\n')

    P = tot['tp'] / max(tot['tp'] + tot['fp'], 1)
    R = tot['tp'] / max(tot['tp'] + tot['fn'], 1)
    Rm = tot['tpm'] / max(tot['gtm'], 1)
    print(f"""
frames scored     : {len(todo)}
GT boxes          : {tot['gt']}   (moving {tot['gtm']}, static {tot['gt']-tot['gtm']}, ignored/occluded {tot['ign']})
predictions       : {tot['pred']}   (ignored on unlabelled scenery: {tot['pign']})
TP / FP / FN      : {tot['tp']} / {tot['fp']} / {tot['fn']}
precision         : {P:.3f}
recall (all)      : {R:.3f}
recall (MOVING)   : {Rm:.3f}   <- what the SLAM filter actually needs
wrote             : {a.out}/yolo_eval.csv, yolo_detections.csv, yolo_confusion.csv{', overlay/' if a.overlay else ''}
""")


if __name__ == '__main__':
    main()
