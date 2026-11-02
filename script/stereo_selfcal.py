#!/usr/bin/env python3
"""Self-calibrate the stereo extrinsic ROTATION (and baseline direction) from a
rosbag, and score how well any extrinsic rectifies that bag.

Why this exists (2026-11-02): dataset_realv6_000 produced an unidentifiable
dense map while realv5 was fine, with the SAME shim and the same 28 Oct
extrinsic. The difference was rectification quality -- with the 28 Oct
extrinsic only ~1 % of cross-checked ORB matches lay on the epipolar line on
realv6 against ~46 % on realv5 -- because the mount had moved: the baseline
direction differs by 6.3 deg. No chequerboard was available, so the rotation
and translation DIRECTION are recovered from the images themselves (essential
matrix over several frames, fisheye-undistorted points). The baseline LENGTH is
unobservable from images and is copied from a reference .dat.

Validation of the method: on realv5 it lands within 0.2 deg of the chequerboard
28 Oct rotation and raises the on-line fraction from 46 % to 77 %.

    python3 script/stereo_selfcal.py <bag_dir_or_db3> \
        --config vins_fusion_ros2/config/wil \
        --ref camera_parameters_281026/281026camera1_rot_trans.dat \
        --compare camera_parameters_281026/281026camera1_rot_trans.dat \
        --out /path/021126camera1_rot_trans_selfcal.dat

Output .dat has the SAME convention as the OpenCV stereoCalibrate files in
camera_parameters*/: X_cam1 = R X_cam0 + T. Write it into the configs with the
usual inverse-form derivation (see config/wil/stereo_imu.yaml comments).
The "on-line" score is the share of cross-checked ORB matches between the
RECTIFIED images with |dy| < 2 px and positive disparity; anything under ~30 %
leaves SGBM producing noise along the rays.
"""
import argparse
import glob
import os
import re
import sqlite3
import sys

import cv2
import numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage

NUM = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'


def read_fisheye(path):
    txt = re.sub(r'#[^\n]*', '', open(path).read())

    def num(k):
        return float(re.search(r'\b' + k + r'\s*:\s*(' + NUM + ')', txt).group(1))
    K = np.array([[num('mu'), 0, num('u0')], [0, num('mv'), num('v0')], [0, 0, 1]])
    D = np.array([num('k2'), num('k3'), num('k4'), num('k5')])
    return K, D, (int(num('image_width')), int(num('image_height')))


def load_dat(path):
    t = open(path).read()
    R = np.array([float(x) for x in re.findall(NUM, t.split('R:')[1].split('T:')[0])]).reshape(3, 3)
    T = np.array([float(x) for x in re.findall(NUM, t.split('T:')[1])][:3])
    return R, T


def write_dat(path, R, T):
    with open(path, 'w') as fh:
        fh.write('R:\n' + '\n'.join(' '.join(repr(float(x)) for x in row) for row in R))
        fh.write('\nT:\n' + '\n'.join(repr(float(x)) for x in T) + '\n')


def bag_frames(bag, idxs, cam0='/cam0/image_raw/compressed', cam1='/cam1/image_raw/compressed'):
    db = bag if bag.endswith('.db3') else sorted(glob.glob(os.path.join(bag, '*.db3')))[0]
    c = sqlite3.connect(db)
    tops = {n: i for i, n in c.execute('select id,name from topics')}
    n = c.execute('select count(*) from messages where topic_id=?', (tops[cam0],)).fetchone()[0]
    out = []
    for k in idxs:
        if k >= n:
            continue
        pair = []
        for t in (cam0, cam1):
            row = c.execute('select data from messages where topic_id=? order by timestamp limit 1 offset ?',
                            (tops[t], k)).fetchone()
            pair.append(cv2.imdecode(np.frombuffer(deserialize_message(row[0], CompressedImage).data, np.uint8),
                                     cv2.IMREAD_GRAYSCALE))
        out.append(pair)
    return out, n


def angle(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def selfcal(pairs, K0, D0, K1, D1, edge_radius=800):
    """Essential matrix cam0 -> cam1 over all pairs. Returns R, unit t, inliers, total."""
    orb = cv2.ORB_create(4000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    p0, p1 = [], []
    for c0, c1 in pairs:
        k0, d0 = orb.detectAndCompute(c0, None)
        k1, d1 = orb.detectAndCompute(c1, None)
        if d0 is None or d1 is None:
            continue
        ms = bf.match(d0, d1)
        a = np.array([k0[m.queryIdx].pt for m in ms], np.float64).reshape(-1, 1, 2)
        b = np.array([k1[m.trainIdx].pt for m in ms], np.float64).reshape(-1, 1, 2)
        # The equidistant model is weakest at the rim; keep matches inside a radius.
        keep = (np.linalg.norm(a[:, 0] - K0[:2, 2], axis=1) < edge_radius) & \
               (np.linalg.norm(b[:, 0] - K1[:2, 2], axis=1) < edge_radius)
        p0.append(cv2.fisheye.undistortPoints(a[keep], K0, D0))
        p1.append(cv2.fisheye.undistortPoints(b[keep], K1, D1))
    p0, p1 = np.concatenate(p0), np.concatenate(p1)
    E, mask = cv2.findEssentialMat(p0, p1, np.eye(3), method=cv2.RANSAC, prob=0.9999, threshold=0.0015)
    _, R, t, _ = cv2.recoverPose(E, p0, p1, np.eye(3), mask=mask)
    return R, t.ravel(), int(mask.sum()), len(p0)


def epipolar_score(R10, T10, pairs, K0, D0, K1, D1, size, hfov=110.0):
    """Rectify (left = cam1, right = cam0, as on this rig) and measure alignment."""
    W, H = size
    f = (W / 2) / np.tan(np.radians(hfov / 2))
    R = R10.T
    T = -R10.T @ T10                      # X_cam0 = R X_cam1 + T  (left -> right)
    R1, R2, _, _, _ = cv2.fisheye.stereoRectify(K1, D1, K0, D0, size, R, T,
                                                flags=cv2.CALIB_ZERO_DISPARITY, balance=0, fov_scale=1)
    P1 = np.array([[f, 0, W / 2, 0], [0, f, H / 2, 0], [0, 0, 1, 0]])
    P2 = P1.copy()
    P2[0, 3] = -f * np.linalg.norm(T)
    m1 = cv2.fisheye.initUndistortRectifyMap(K1, D1, R1, P1, size, cv2.CV_16SC2)
    m2 = cv2.fisheye.initUndistortRectifyMap(K0, D0, R2, P2, size, cv2.CV_16SC2)
    orb = cv2.ORB_create(3000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    frac, dys = [], []
    for c0, c1 in pairs:
        rl = cv2.remap(c1, *m1, cv2.INTER_LINEAR)
        rr = cv2.remap(c0, *m2, cv2.INTER_LINEAR)
        kl, dl = orb.detectAndCompute(rl, None)
        kr, dr = orb.detectAndCompute(rr, None)
        ms = bf.match(dl, dr)
        pl = np.array([kl[m.queryIdx].pt for m in ms])
        pr = np.array([kr[m.trainIdx].pt for m in ms])
        dy = pl[:, 1] - pr[:, 1]
        dx = pl[:, 0] - pr[:, 0]
        inl = (np.abs(dy) < 2) & (dx > 0) & (dx < 300)
        frac.append(inl.mean() * 100)
        dys.append(np.median(dy[inl]) if inl.any() else np.nan)
    return float(np.mean(frac)), float(np.nanmedian(dys))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bag')
    ap.add_argument('--config', default='vins_fusion_ros2/config/wil',
                    help='dir with cam0_fisheye.yaml / cam1_fisheye.yaml')
    ap.add_argument('--ref', required=True, help='.dat whose baseline LENGTH is kept')
    ap.add_argument('--compare', nargs='*', default=[], help='.dat files to score on this bag')
    ap.add_argument('--frames', type=int, default=8, help='frames used for the estimate')
    ap.add_argument('--out', default='', help='write the self-calibrated .dat here')
    a = ap.parse_args()

    K0, D0, size = read_fisheye(os.path.join(a.config, 'cam0_fisheye.yaml'))
    K1, D1, _ = read_fisheye(os.path.join(a.config, 'cam1_fisheye.yaml'))
    _, n = bag_frames(a.bag, [])
    cal_idx = np.linspace(0.05 * n, 0.95 * n, a.frames).astype(int)
    test_idx = (cal_idx[:-1] + cal_idx[1:]) // 2            # disjoint from the fit
    cal, _ = bag_frames(a.bag, cal_idx)
    test, _ = bag_frames(a.bag, test_idx[:4])
    Rr, Tr = load_dat(a.ref)

    R, t, ninl, ntot = selfcal(cal, K0, D0, K1, D1)
    T = t * np.linalg.norm(Tr)
    np.set_printoptions(precision=5, suppress=True)
    print(f'self-calibration from {ninl}/{ntot} matches over {len(cal)} frames')
    print(f'  rotation vs ref {angle(Rr.T @ R):.2f} deg, baseline direction vs ref '
          f'{np.degrees(np.arccos(np.clip(Tr @ T / np.linalg.norm(Tr) / np.linalg.norm(T), -1, 1))):.2f} deg, '
          f'direction {t}')
    s, dy = epipolar_score(R, T, test, K0, D0, K1, D1, size)
    print(f'  on-line {s:.0f} %  dy median {dy:+.2f} px   <- self-calibrated')
    for p in [a.ref] + a.compare:
        Rc, Tc = load_dat(p)
        s, dy = epipolar_score(Rc, Tc, test, K0, D0, K1, D1, size)
        print(f'  on-line {s:.0f} %  dy median {dy:+.2f} px   <- {p}')
    if a.out:
        write_dat(a.out, R, T)
        print('written', a.out)


if __name__ == '__main__':
    main()
