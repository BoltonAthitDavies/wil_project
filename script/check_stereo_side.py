#!/usr/bin/env python3
"""Which physical side is /cam1 on, relative to /cam0?

Reads synchronised JPEG frame pairs straight out of a rosbag2 sqlite3 file (no ROS
deps), matches features between them, and reports the median horizontal disparity

    d = x_cam0 - x_cam1

For a rigid forward-looking stereo pair every static world point obeys
x_right = x_left - f*b/Z, i.e. the point sits FURTHER LEFT in the right camera.
So:
    d > 0  ->  cam1 is the RIGHT camera  (cam1 is to cam0's right)
    d < 0  ->  cam1 is the LEFT  camera  (cam1 is to cam0's left)

Matches are restricted to the central image region because the lenses are ~180 deg
fisheye and the sign argument above only holds cleanly near the optical axis.

Usage: python3 check_stereo_side.py [bag_dir] [n_pairs]
"""
import sys, sqlite3, glob, os
import numpy as np, cv2

bag = sys.argv[1] if len(sys.argv) > 1 else "dataset/rosbag_realsense_imu_cambaseline95mm"
n_pairs = int(sys.argv[2]) if len(sys.argv) > 2 else 12
db = sorted(glob.glob(os.path.join(bag, "*.db3")))[0]

con = sqlite3.connect(db)
tid = {n: i for n, i in con.execute("SELECT name,id FROM topics").fetchall()}


def frames(topic):
    return con.execute(
        "SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp",
        (tid[topic],)).fetchall()


def jpeg(blob):
    """CompressedImage CDR blob -> gray image. Slice from the JPEG SOI marker."""
    b = bytes(blob)
    i = b.find(b"\xff\xd8\xff")
    if i < 0:
        return None
    j = b.rfind(b"\xff\xd9")
    return cv2.imdecode(np.frombuffer(b[i:j + 2], np.uint8), cv2.IMREAD_GRAYSCALE)


a, b_ = frames("/cam0/image_raw/compressed"), frames("/cam1/image_raw/compressed")
print(f"bag: {db}\ncam0 frames: {len(a)}  cam1 frames: {len(b_)}")

ts_b = np.array([t for t, _ in b_])
orb = cv2.ORB_create(4000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# sample evenly through the bag, skipping the very start
idxs = np.linspace(len(a) * 0.1, len(a) * 0.9, n_pairs).astype(int)
medians = []
for k in idxs:
    t0, d0 = a[k]
    j = int(np.argmin(np.abs(ts_b - t0)))
    dt_ms = (ts_b[j] - t0) / 1e6
    i0, i1 = jpeg(d0), jpeg(b_[j][1])
    if i0 is None or i1 is None:
        continue
    h, w = i0.shape
    # central band only: fisheye distortion is mild here
    x0, x1, y0, y1 = int(w * .25), int(w * .75), int(h * .25), int(h * .75)
    mask = np.zeros_like(i0); mask[y0:y1, x0:x1] = 255
    k0, de0 = orb.detectAndCompute(i0, mask)
    k1, de1 = orb.detectAndCompute(i1, mask)
    if de0 is None or de1 is None or len(k0) < 20 or len(k1) < 20:
        continue
    m = bf.match(de0, de1)
    if len(m) < 20:
        continue
    p0 = np.float32([k0[x.queryIdx].pt for x in m])
    p1 = np.float32([k1[x.trainIdx].pt for x in m])
    # keep near-horizontal correspondences: a stereo pair has little vertical offset
    keep = np.abs(p0[:, 1] - p1[:, 1]) < 8
    if keep.sum() < 15:
        continue
    d = (p0[keep, 0] - p1[keep, 0])
    med = float(np.median(d))
    medians.append(med)
    print(f"  frame {k:5d}  dt={dt_ms:+7.2f}ms  matches={int(keep.sum()):4d}  "
          f"median disparity = {med:+8.2f} px")

if not medians:
    sys.exit("no usable pairs")

overall = float(np.median(medians))
pos = sum(1 for m in medians if m > 0)
print(f"\nusable pairs: {len(medians)}   positive: {pos}   negative: {len(medians)-pos}")
print(f"overall median disparity (x_cam0 - x_cam1) = {overall:+.2f} px")
print("VERDICT: cam1 is the " + ("RIGHT" if overall > 0 else "LEFT") +
      " camera -> cam1 lies to cam0's " + ("right" if overall > 0 else "left"))
