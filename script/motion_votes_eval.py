#!/usr/bin/env python3
"""Score motion-gate votes against Gazebo ground truth.

    python3 script/motion_votes_eval.py dataset/dataset_allsensor_realistic_003 \
        output/.../motion_votes.csv [--ratio 0.5 --min-points 8]

A filter run with filter_mode:=motion writes motion_votes.csv: one row per
detected box per scored frame, with the raw census (keypoints inside the box,
how many were pose-optimisation inliers, and the same for the background) and
the verdict the node reached. This script projects the true props into cam0 at
each row's stamp (same machinery as yolo_eval.py), matches the voted box to a
prop by IoU, labels that prop MOVING or STATIC from its world-frame speed, and
reports:

  * the confusion of verdict vs truth, for the thresholds the run used;
  * the distribution of (box inlier ratio / background inlier ratio) separately
    for truly-moving and truly-static boxes -- the plot that chooses the
    threshold;
  * a sweep of motion_inlier_ratio and motion_min_points over the SAME census,
    so the next run's parameters are chosen from data, not guessed.

Only rows whose box actually received a vote (n_kp >= min_points) are scored for
the census; inherited verdicts are reported separately, because they are what
the mask was built from.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yolo_eval as E  # noqa: E402
import yolo_gt as G    # noqa: E402


def project(tr, prefix, R_cw, t_cw, min_px):
    p_wo, R_wo = tr
    v_w = (R_wo @ G.vertices(prefix).T).T + p_wo
    v_c = (R_cw @ v_w.T).T + t_cw
    z = v_c[:, 2]
    if not np.any(z > 0.05):
        return None
    v_c = v_c[z > 0.05]
    u = E.FX * v_c[:, 0] / v_c[:, 2] + E.CX
    v = E.FY * v_c[:, 1] / v_c[:, 2] + E.CY
    x0, x1 = max(0.0, float(u.min())), min(float(E.W), float(u.max()))
    y0, y1 = max(0.0, float(v.min())), min(float(E.H), float(v.max()))
    if x1 - x0 < min_px or y1 - y0 < min_px:
        return None
    return (x0, y0, x1, y1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('bag')
    ap.add_argument('votes', help='motion_votes.csv from a filter_mode:=motion run')
    ap.add_argument('--config', default=os.path.join(
        E.WORKSPACE, 'vins_fusion_ros2', 'config', 'wil_sim', 'stereo_imu.yaml'))
    ap.add_argument('--iou', type=float, default=0.3, help='box-to-prop association')
    ap.add_argument('--move-thresh', type=float, default=0.05, help='m/s')
    ap.add_argument('--ratio', type=float, default=0.5, help='motion_inlier_ratio the run used')
    ap.add_argument('--min-points', type=int, default=8, help='motion_min_points the run used')
    ap.add_argument('--min-box-age', type=int, default=0,
                    help='only score census rows whose box chain is at least this old '
                         '(needs the box_age column; mirrors motion_min_box_age)')
    ap.add_argument('--out', default=None,
                    help='directory to write motion_eval.csv (every box-frame with its '
                         'truth label) and motion_summary.csv (the metrics printed '
                         'below, one row), for script/summarise_motion_eval.py')
    a = ap.parse_args()

    body_T_cam0 = E.read_extrinsic(a.config, 'body_T_cam0')
    tracks, _ = E.load_bag(a.bag)
    props = [(n, G.model_prefix(n), tr) for n, tr in tracks.items() if G.model_prefix(n)]
    robot = tracks['ackermann_robot_001']

    rows = list(csv.DictReader(open(a.votes)))
    print(f'votes    : {len(rows)} box-frames in {a.votes}')
    by_t = {}
    for r in rows:
        by_t.setdefault(int(r['timestamp_ns']), []).append(r)

    # label every voted box
    labelled = []           # (row, truth) with truth in {'moving','static','none'}
    for ts_ns, group in by_t.items():
        ts = ts_ns * 1e-9
        p_wb, R_wb = robot.at(ts)
        T_wb = np.eye(4); T_wb[:3, :3], T_wb[:3, 3] = R_wb, p_wb
        T_wc = T_wb @ body_T_cam0
        R_cw = T_wc[:3, :3].T
        t_cw = -R_cw @ T_wc[:3, 3]
        gts = []
        for name, prefix, tr in props:
            bx = project(tr.at(ts), prefix, R_cw, t_cw, 12.0)
            if bx is not None:
                gts.append((bx, tr.speed(ts) > a.move_thresh, name))
        for r in group:
            box = (float(r['x0']), float(r['y0']), float(r['x1']), float(r['y1']))
            best, truth, who = 0.0, 'none', ''
            for bx, moving, name in gts:
                o = E.iou(box, bx)
                if o > best:
                    best, truth, who = o, ('moving' if moving else 'static'), name
            if best < a.iou:
                truth, who = 'none', ''
            labelled.append((r, truth, who))

    def ratio_rel(r):
        kp, inl, bkp, binl = (int(r[k]) for k in ('n_kp', 'n_inlier', 'bg_kp', 'bg_inlier'))
        if kp == 0 or bkp == 0 or binl == 0:
            return None
        return (inl / kp) / (binl / bkp)

    # ---- 1. verdict vs truth as the run decided it
    print('\n== verdicts the run reached (what the mask was built from) ==')
    conf = {}
    for r, truth, _ in labelled:
        v = {'0': 'unknown', '1': 'static', '2': 'moving'}[r['verdict']]
        conf[(truth, v)] = conf.get((truth, v), 0) + 1
    hdr = 'truth / verdict'
    print(f"{hdr:16s}{'unknown':>9s}{'static':>9s}{'moving':>9s}")
    for truth in ('moving', 'static', 'none'):
        print(f'{truth:16s}' + ''.join(f'{conf.get((truth, v), 0):9d}' for v in ('unknown', 'static', 'moving')))
    summary = {'votes_file': a.votes, 'bag': a.bag, 'box_frames': len(labelled),
               'min_points': a.min_points, 'ratio': a.ratio, 'min_box_age': a.min_box_age}
    for truth in ('moving', 'static', 'none'):
        for v in ('unknown', 'static', 'moving'):
            summary[f'{truth}_{v}'] = conf.get((truth, v), 0)
    n_mov = sum(conf.get(('moving', v), 0) for v in ('unknown', 'static', 'moving'))
    n_sta = sum(conf.get(('static', v), 0) for v in ('unknown', 'static', 'moving'))
    # What the mask was built from: fraction of truly-moving / truly-static
    # box-frames that ended up painted. These are the numbers that compare
    # directly with class mode (1.0 / 1.0) and with no filter (0 / 0).
    summary['moving_masked_frac'] = conf.get(('moving', 'moving'), 0) / n_mov if n_mov else float('nan')
    summary['static_masked_frac'] = conf.get(('static', 'moving'), 0) / n_sta if n_sta else float('nan')

    # ---- 2. the census itself, voted rows only
    def old_enough(r):
        return a.min_box_age <= 0 or int(r.get('box_age', 0) or 0) >= a.min_box_age
    voted = [(r, truth) for r, truth, _ in labelled
             if int(r['n_kp']) >= a.min_points and old_enough(r)]
    print(f'\n== census on voted rows (n_kp >= {a.min_points}): {len(voted)} of {len(labelled)} ==')
    for truth in ('moving', 'static'):
        vals = sorted(v for r, t in voted if t == truth for v in [ratio_rel(r)] if v is not None)
        summary[f'census_{truth}_n'] = len(vals)
        if not vals:
            summary[f'census_{truth}_median'] = float('nan')
            print(f'  {truth:7s}: no rows'); continue
        q = lambda p: vals[int(p * (len(vals) - 1))]
        print(f'  {truth:7s} n={len(vals):5d}  box/bg inlier ratio  p10 {q(.1):.2f}  p25 {q(.25):.2f}  '
              f'median {q(.5):.2f}  p75 {q(.75):.2f}  p90 {q(.9):.2f}')
        summary[f'census_{truth}_median'] = q(.5)

    # ---- 3a. the run's OWN operating point, always recorded
    def rates(mp, th):
        tp = fm = ts_ = fs = 0
        for r, truth, _ in labelled:
            if int(r['n_kp']) < mp or truth == 'none' or not old_enough(r):
                continue
            v = ratio_rel(r)
            if v is None:
                continue
            moving = v < th
            if truth == 'moving':
                tp += moving; fm += 1
            else:
                fs += moving; ts_ += 1
        return tp, fm, fs, ts_

    tp, fm, fs, ts_ = rates(a.min_points, a.ratio)
    summary['vote_n_moving'] = fm
    summary['vote_n_static'] = ts_
    summary['vote_recall'] = tp / fm if fm else float('nan')
    summary['vote_false_alarm'] = fs / ts_ if ts_ else float('nan')

    # ---- 3b. sweep around it
    print('\n== sweep: fraction of truly-MOVING boxes voted moving (recall) vs truly-STATIC voted moving (false alarm) ==')
    print(f"{'min_pts':>8s}{'ratio':>7s}{'n_mov':>7s}{'recall':>8s}{'n_stat':>8s}{'false':>8s}")
    for mp in sorted({8, 15, 25, 40, a.min_points}):
        for th in sorted({0.25, 0.5, 0.75, a.ratio}):
            tp, fm, fs, ts_ = rates(mp, th)
            mark = ' <-- run' if mp == a.min_points and abs(th - a.ratio) < 1e-9 else ''
            print(f'{mp:8d}{th:7.2f}{fm:7d}{(tp / fm if fm else float("nan")):8.3f}'
                  f'{ts_:8d}{(fs / ts_ if ts_ else float("nan")):8.3f}{mark}')

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, 'motion_eval.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['timestamp_ns', 'box', 'x0', 'y0', 'x1', 'y1', 'n_kp', 'n_inlier',
                        'bg_kp', 'bg_inlier', 'inherited', 'verdict', 'box_age',
                        'truth', 'gt_name', 'rel_ratio'])
            for r, truth, who in labelled:
                v = ratio_rel(r)
                w.writerow([r['timestamp_ns'], r['box'], r['x0'], r['y0'], r['x1'], r['y1'],
                            r['n_kp'], r['n_inlier'], r['bg_kp'], r['bg_inlier'],
                            r['inherited'], r['verdict'], r.get('box_age', ''),
                            truth, who, '' if v is None else f'{v:.4f}'])
        with open(os.path.join(a.out, 'motion_summary.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(summary))
            w.writeheader(); w.writerow(summary)
        print(f'\nwrote     : {a.out}/motion_eval.csv, motion_summary.csv')


if __name__ == '__main__':
    main()
