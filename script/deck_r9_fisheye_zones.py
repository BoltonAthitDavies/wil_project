#!/usr/bin/env python3
"""Deck slide R9: where in the real fisheye frame VINS-Fusion gets into trouble.

    python3 script/deck_r9_fisheye_zones.py [--rec dataset_realv7_000] [--skip 400]

One left-camera frame from the real rig, with every pixel coloured by the ray
angle theta it sees, from the rig's own Kannala-Brandt calibration
(orbslam3_ros2/config/wil/stereo_imu.yaml, Camera1.*; cam1 reuses it):

  theta < 70 deg      kept by VINS's fov_mask_deg 70
  70 - 90 deg         masked: VINS stores x/z = tan(theta) > 2.7, unbounded at 90
  90 deg and beyond   the ray points behind the camera plane: z <= 0, x/z flips sign
  black band          beyond the polynomial's peak (theta 109.3 deg, r = 924 px),
                      no ray can be recovered at all

FAST corners detected on the same frame are drawn in the colour of their zone,
and the shares are printed.  The frame comes from the bag on the rescue drive.

Output: docs/report/insights/update_181126/fig/r9_fisheye_zones.png
"""
import argparse
import glob
import os
import sqlite3

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import deck_style as S

BAGS = "/media/ambushee/New Volume/wil_rescue"
FX, FY, CX, CY = 607.5108682653754, 609.9053834748036, 958.2558093864706, 539.7975091283472
KB = [-0.03645464928389926, -0.0008403394026266405, 0.00017697497379393103, -0.000382711572282332]
ZONE = [("kept (under 70°)", None),
        ("masked: 70–90°, x/z explodes", "#e67e22"),
        ("behind the camera plane: ≥ 90°", "#c0392b")]


def kb_r(t):
    return t * (1 + KB[0] * t ** 2 + KB[1] * t ** 4 + KB[2] * t ** 6 + KB[3] * t ** 8)


def frame(rec, skip):
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CompressedImage
    c = sqlite3.connect(glob.glob(f"{BAGS}/{rec}/*.db3")[0])
    tid = {n: i for i, n in c.execute("select id,name from topics")}["/cam0/image_raw/compressed"]
    row = c.execute("select data from messages where topic_id=? order by timestamp limit 1 offset ?",
                    (tid, skip)).fetchone()
    c.close()
    return cv2.imdecode(np.frombuffer(deserialize_message(row[0], CompressedImage).data, np.uint8), 0)


def zone_image(rec="dataset_realv7_000", skip=400):
    """(rgb, theta_deg, corner_xy, corner_zone, shares) for one real left frame."""
    img = frame(rec, skip)
    h, w = img.shape
    v, u = np.mgrid[0:h, 0:w]
    rd = np.hypot((u - CX) / FX, (v - CY) / FY)
    grid = np.linspace(0, np.radians(109.3), 5000)
    peak = kb_r(grid).max()
    theta = np.degrees(np.interp(rd, kb_r(grid), grid))
    dead = rd > peak
    rgb = np.dstack([img] * 3).astype(float) / 255.0
    for lo, hi, col in ((70, 90, ZONE[1][1]), (90, 999, ZONE[2][1])):
        m = (theta >= lo) & (theta < hi) & ~dead
        c = np.array(tuple(int(col[i:i + 2], 16) / 255 for i in (1, 3, 5)))
        rgb[m] = 0.45 * rgb[m] + 0.55 * c
    rgb[dead] = 0.0
    pts = np.array([k.pt for k in cv2.FastFeatureDetector_create(20).detect(img, None)])
    pi = np.clip(np.round(pts).astype(int), 0, [w - 1, h - 1])
    th = theta[pi[:, 1], pi[:, 0]]
    zk = np.where(th < 70, 0, np.where(th < 90, 1, 2))
    return rgb, theta, pts, zk, [100 * np.mean(zk == k) for k in range(3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", default="dataset_realv7_000")
    ap.add_argument("--skip", type=int, default=400)
    a = ap.parse_args()
    img = frame(a.rec, a.skip)
    h, w = img.shape
    v, u = np.mgrid[0:h, 0:w]
    rd = np.hypot((u - CX) / FX, (v - CY) / FY)              # distorted radius, normalised
    grid = np.linspace(0, np.radians(109.3), 5000)
    peak = kb_r(grid).max()
    theta = np.degrees(np.interp(rd, kb_r(grid), grid))
    dead = rd > peak

    rgb = np.dstack([img] * 3).astype(float) / 255.0
    for lo, hi, col in ((70, 90, ZONE[1][1]), (90, 999, ZONE[2][1])):
        m = (theta >= lo) & (theta < hi) & ~dead
        c = np.array(tuple(int(col[i:i + 2], 16) / 255 for i in (1, 3, 5)))
        rgb[m] = 0.45 * rgb[m] + 0.55 * c
    rgb[dead] = 0.0

    corners = cv2.FastFeatureDetector_create(20).detect(img, None)
    pts = np.array([k.pt for k in corners])
    pi = np.clip(np.round(pts).astype(int), 0, [w - 1, h - 1])
    th = theta[pi[:, 1], pi[:, 0]]
    zk = np.where(th < 70, 0, np.where(th < 90, 1, 2))
    share = [100 * np.mean(zk == k) for k in range(3)]
    print("%s frame %d: %d FAST corners; under 70 deg %.1f %%, 70-90 %.1f %%, >= 90 %.1f %%; "
          "dead band %.1f %% of the frame; theta at the left/right edge %.1f / %.1f deg, top %.1f deg"
          % (a.rec, a.skip, len(pts), *share, 100 * dead.mean(),
             theta[int(CY), 0], theta[int(CY), w - 1], theta[0, int(CX)]))

    S.setup()
    fig, ax = plt.subplots(figsize=(4.6, 2.75))
    fig.patch.set_facecolor(S.TH["surface"])
    ax.imshow(rgb)
    cols = ["#2ecc71", ZONE[1][1], ZONE[2][1]]
    sel = np.random.default_rng(0).permutation(len(pts))[:1500]
    ax.scatter(pts[sel, 0], pts[sel, 1], s=1.2, c=[cols[k] for k in zk[sel]], lw=0)
    cs = ax.contour(theta, levels=[70, 80, 90, 100], colors="white", linewidths=0.5)
    ax.clabel(cs, fmt=lambda x: "%d°" % x, fontsize=5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    leg = [Patch(color="#2ecc71", label="corner VINS keeps (%.0f %%)" % share[0]),
           Patch(color=ZONE[1][1], label="70–90°: x/z = tanθ > 2.7, masked (%.0f %%)" % share[1]),
           Patch(color=ZONE[2][1], label="≥ 90°: behind the camera plane, x/z flips sign (%.0f %%)" % share[2]),
           Patch(color="black", label="beyond the lens model: no ray at all")]
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False,
              fontsize=5.0, handlelength=1.0, columnspacing=1.0, labelcolor=S.TH["ink2"])
    S.title(ax, "one real fisheye frame, coloured by the angle each pixel sees")
    S.save(fig, "r9_fisheye_zones.png")


if __name__ == "__main__":
    main()
