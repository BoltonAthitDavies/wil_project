#!/usr/bin/env python3
"""What the two datasets actually give the estimators: simulation against the real rig.

    python3 script/exp1_dataset_compare.py            # uses the cached measurements
    python3 script/exp1_dataset_compare.py --measure  # re-reads the bags (slow, needs the drive)

The accuracy figures of Experiment 1 compare estimators. This compares the INPUT:
the motion the robot makes, the images the cameras deliver, the depth the stereo
pair can resolve, and where in the fisheye frame the corners fall. Everything is
measured from the recordings and the ground truth, nothing is taken from a
datasheet.

Measured per domain (4 simulation recordings, 3 realv7 recordings):
  motion   speed, turn rate and standing time from the ground truth
  image    mean, contrast, gradient, blur and corner count on sampled frames
  depth    ORB stereo matches -> depth, and the share beyond ORB-SLAM3's close/far
           threshold (Stereo.ThDepth 90 x baseline 0.0936 m = 8.4 m), beyond which a
           point is treated as monocular and contributes no scale
  radial   the radius at which corners are detected, against the fisheye model's own
           limits (the 70 deg mask and the radius where the Kannala-Brandt polynomial
           stops being monotonic)

Writes output/compare/datasets/{measurements.json, fig_dataset_compare.png}.
Chat-first: nothing here is in the report.
"""
import argparse
import glob
import json
import os
import sqlite3
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np                                   # noqa: E402
import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.lines import Line2D                  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "docs/report/final_report/figures")
sys.path.insert(0, os.path.join(ROOT, "script"))
import vio_metrics as vm                             # noqa: E402
import eval_real_gt as E                             # noqa: E402

OUT = os.path.join(ROOT, "output/compare/datasets")
BAGS = "/media/ambushee/New Volume/wil_rescue"
SIM = ["dataset_allsensor_realistic_%03d" % i for i in range(4)]
REAL = ["dataset_realv7_%03d" % i for i in range(3)]
FX = {"sim": 640.0, "real": 607.5}          # Camera1.fx of each config
BASE = 0.0936                                # stereo baseline, both rigs
THDEPTH = 90.0                               # Stereo.ThDepth, both configs
KB = [-0.03645464928389926, -0.0008403394026266405, 0.00017697497379393103, -0.000382711572282332]
TH = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#dedcd4", sim="#4c72b0", real="#c44e52")


def kb_radius(deg):
    t = np.radians(deg)
    return FX["real"] * t * (1 + KB[0] * t ** 2 + KB[1] * t ** 4 + KB[2] * t ** 6 + KB[3] * t ** 8)


# ------------------------------------------------------------------ measuring
def motion(dom, ds):
    if dom == "sim":
        r = vm.load_gt(os.path.join(ROOT, "dataset/gt", ds, "ground_truth.csv"))
        t, xy = r[0] - r[0][0], r[1][:, :2]
    else:
        g = E.load_gt(os.path.join(ROOT, "dataset/gt", ds))
        t, xy = np.asarray(g["t_rel"]), np.asarray(g["xy"])
    d = np.diff(xy, axis=0); dt = np.diff(t); k = dt > 1e-6
    v = np.linalg.norm(d[k], axis=1) / dt[k]
    h = np.arctan2(d[k][:, 1], d[k][:, 0])
    w = np.degrees(np.abs((np.diff(h) + np.pi) % (2 * np.pi) - np.pi)) / dt[k][1:]
    mv = v > 0.05
    return dict(span=float(t[-1] - t[0]), route=float(np.sum(np.linalg.norm(d, axis=1))),
                v=v[mv].tolist()[::3], w=w[mv[1:]].tolist()[::3], still=float(1 - mv.mean()))


def _frames(ds, cam, n, skip, step):
    import cv2
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CompressedImage
    f = glob.glob(f"{BAGS}/{ds}/*.db3")[0]; c = sqlite3.connect(f)
    tid = {n_: i for i, n_ in c.execute("select id,name from topics")}[cam]
    rows = c.execute("select data from messages where topic_id=? order by timestamp", (tid,)).fetchall()
    out = [cv2.imdecode(np.frombuffer(deserialize_message(rows[i][0], CompressedImage).data, np.uint8), 0)
           for i in range(skip, min(len(rows), skip + n * step), step)]
    c.close(); return out


def images(ds, n=12, skip=400, step=30):
    import cv2
    L = _frames(ds, "/cam0/image_raw/compressed", n, skip, step)
    R = _frames(ds, "/cam1/image_raw/compressed", n, skip, step)
    fast = cv2.FastFeatureDetector_create(20)
    orb = cv2.ORB_create(2000); bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    st, D, rad = [], [], []
    for l, r in zip(L, R):
        g = cv2.GaussianBlur(l, (3, 3), 0)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, 3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)
        mag = np.sqrt(gx * gx + gy * gy)
        mpx = l.shape[0] * l.shape[1] / 1e6
        st.append([l.mean(), l.std(), mag.mean(), cv2.Laplacian(g, cv2.CV_32F).var(), len(fast.detect(l, None)) / mpx, mpx])
        k1, d1 = orb.detectAndCompute(l, None); k2, d2 = orb.detectAndCompute(r, None)
        if d1 is None or d2 is None:
            continue
        h, w = l.shape; cx, cy = w / 2, h / 2
        fx = FX["real" if "realv" in ds else "sim"]
        for m in bf.match(d1, d2):
            p1, p2 = k1[m.queryIdx].pt, k2[m.trainIdx].pt
            rad.append(float(np.hypot(p1[0] - cx, p1[1] - cy)))
            if abs(p1[1] - p2[1]) > 3:
                continue
            disp = p1[0] - p2[0]
            if disp <= 0.5:
                continue
            z = fx * BASE / disp
            if 0.1 < z < 80:
                D.append(float(z))
    s = np.mean(st, axis=0)
    return dict(mean=s[0], sd=s[1], grad=s[2], lapvar=s[3], fast_per_mpx=s[4], mpx=s[5],
                depth=D, radius=rad[::3], shape=[int(L[0].shape[1]), int(L[0].shape[0])])


def measure():
    M = {"sim": {}, "real": {}}
    for dom, dss in (("sim", SIM), ("real", REAL)):
        for ds in dss:
            M[dom][ds] = dict(motion=motion(dom, ds), **images(ds))
            print("  measured", ds)
    os.makedirs(OUT, exist_ok=True)
    json.dump(M, open(os.path.join(OUT, "measurements.json"), "w"))
    return M


# ------------------------------------------------------------------- drawing
def style(ax, title):
    ax.set_facecolor(TH["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=TH["grid"], lw=0.6); ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=9.5, color=TH["ink"])
    ax.tick_params(labelsize=8, colors=TH["ink2"])


def ecdf(ax, x, **kw):
    x = np.sort(np.asarray(x, float)); ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", **kw)


def draw(M):
    pool = lambda dom, k: np.concatenate([np.asarray(M[dom][d]["motion"][k] if k in ("v", "w") else M[dom][d][k], float) for d in M[dom]])
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.0)); fig.patch.set_facecolor("white")
    a, b, c, d = ax.ravel()

    # (a) motion
    for dom, col, lbl in (("sim", TH["sim"], "simulation"), ("real", TH["real"], "real rig")):
        ecdf(a, pool(dom, "v"), color=col, lw=1.8, label=lbl)
    a.set_xlim(0, 1.8); a.set_xlabel("ground-truth speed while moving [m/s]", fontsize=8.5)
    a.set_ylabel("fraction of samples", fontsize=8.5); a.legend(fontsize=8, frameon=False, loc="lower right")
    still = {dom: 100 * np.mean([M[dom][x]["motion"]["still"] for x in M[dom]]) for dom in M}
    rate = {dom: np.median(pool(dom, "w")) for dom in M}
    style(a, "(a) the robot moves %.1fx slower on the rig (medians %.2f vs %.2f m/s),\n    stands still %.0f%% of the time against %.0f%%, and turns at %.0f vs %.0f deg/s"
          % (np.median(pool("sim", "v")) / np.median(pool("real", "v")), np.median(pool("sim", "v")), np.median(pool("real", "v")),
             still["real"], still["sim"], rate["real"], rate["sim"]))

    # (b) image statistics, as a ratio to the simulation so one axis can carry all four
    keys = [("sd", "contrast\n(intensity SD)"), ("grad", "mean |gradient|"), ("lapvar", "sharpness\n(Laplacian var)"), ("fast_per_mpx", "FAST corners\nper megapixel")]
    x = np.arange(len(keys)); w = 0.36
    base = {k: np.mean([M["sim"][ds][k] for ds in M["sim"]]) for k, _ in keys}
    for i, (dom, col, lbl) in enumerate((("sim", TH["sim"], "simulation"), ("real", TH["real"], "real rig"))):
        raw = [np.mean([M[dom][ds][k] for ds in M[dom]]) for k, _ in keys]
        vals = [100 * v / base[k] for v, (k, _) in zip(raw, keys)]
        bars = b.bar(x + (i - 0.5) * w, vals, w, color=col, label=lbl)
        for r, v, rv in zip(bars, vals, raw):
            b.text(r.get_x() + r.get_width() / 2, v + 2, "%.0f" % rv if rv > 100 else "%.1f" % rv, ha="center", va="bottom", fontsize=7.5, color=col)
    b.axhline(100, color=TH["ink2"], lw=0.8, ls=":")
    b.set_xticks(x); b.set_xticklabels([n for _, n in keys], fontsize=8); b.set_ylim(0, 215)
    b.legend(fontsize=8, frameon=False, loc="upper left")
    b.set_ylabel("percent of the simulation value (bars labelled with the raw number)", fontsize=8)
    # (c) depth
    lim = THDEPTH * BASE
    for dom, col, lbl in (("sim", TH["sim"], "simulation"), ("real", TH["real"], "real rig")):
        D = pool(dom, "depth"); ecdf(c, D, color=col, lw=1.8, label="%s (%.0f%% beyond the line)" % (lbl, 100 * np.mean(D > lim)))
    c.axvline(lim, color=TH["ink2"], ls=":", lw=1.2)
    c.text(lim + 0.4, 0.02, "Stereo.ThDepth 90 x %.4f m = %.1f m:\nbeyond it a point is monocular" % (BASE, lim), fontsize=7.5, color=TH["ink2"])
    c.set_xlim(0, 40); c.set_xlabel("depth of an ORB stereo match [m]", fontsize=8.5); c.set_ylabel("fraction of matches", fontsize=8.5)
    c.legend(fontsize=8, frameon=False, loc="center right")
    style(c, "(c) the real scene is deeper, so more of it carries no stereo scale")

    # (d) where the corners are, against the fisheye model's limits
    R = pool("real", "radius"); half = M["real"][REAL[0]]["shape"][1] / 2
    ecdf(d, R, color=TH["real"], lw=1.8, label="real rig, detected corners")
    for deg, txt, col in ((70, "70 deg mask", "#2d8a5e"), (109.3, "polynomial peak", "#9b59b6")):
        r = kb_radius(deg); d.axvline(r, color=col, ls="--", lw=1.2)
        d.text(r + 8, 0.08 if deg == 70 else 0.2, "%s\n%.0f px" % (txt, r), fontsize=7.5, color=col)
    d.axvspan(924.4, 1101, color="#555555", alpha=0.18)
    d.text(935, 0.55, "black: outside the\nlens image circle\n(9.5 %% of the frame,\n%.1f %% of corners)" % (100 * np.mean(R > 924.4)), fontsize=7.5, color=TH["ink2"])
    d.set_xlim(0, 1101); d.set_xlabel("radius of a detected corner from the image centre [px]", fontsize=8.5)
    d.set_ylabel("fraction of corners", fontsize=8.5); d.legend(fontsize=8, frameon=False, loc="upper left")
    style(d, "(d) the \u201cunprojectable\u201d band carries no image at all;\n    the 70 deg mask throws away %.0f%% of the corners that do exist" % (100 * np.mean(R > kb_radius(70))))

    fig.suptitle("Experiment 1: what the two datasets give the estimators", fontsize=11.5, x=0.02, ha="left", color=TH["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for d in (OUT, FIGS):
        fig.savefig(os.path.join(d, "exp1_dataset_compare.png" if d == FIGS else "fig_dataset_compare.png"), dpi=165)
    plt.close(fig)
    print("wrote", os.path.join(OUT, "fig_dataset_compare.png"))

    # the numbers behind the panels
    print("\n%-26s %7s %7s %6s %6s %6s %7s %7s %7s" % ("recording", "span s", "route m", "med v", "still%", "med w", "corners", "depth50", "far %"))
    for dom in ("sim", "real"):
        for ds in M[dom]:
            m = M[dom][ds]; mo = m["motion"]; D = np.asarray(m["depth"])
            print("%-26s %7.1f %7.1f %6.2f %6.0f %6.1f %7.0f %7.2f %7.1f" % (
                ds.replace("dataset_allsensor_", "sim ").replace("dataset_", "real "), mo["span"], mo["route"],
                np.median(mo["v"]), 100 * mo["still"], np.median(mo["w"]), m["fast_per_mpx"], np.median(D), 100 * np.mean(D > lim)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--measure", action="store_true"); a = ap.parse_args()
    f = os.path.join(OUT, "measurements.json")
    M = measure() if (a.measure or not os.path.exists(f)) else json.load(open(f))
    draw(M)


if __name__ == "__main__":
    main()
