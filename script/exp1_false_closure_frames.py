#!/usr/bin/env python3
"""What the false loop closure on realistic_001 actually matched: the two frames.

    python3 script/exp1_false_closure_frames.py

WHY THIS EXISTS
  The chapter attributes the 14 m false closure to the I layout's repeated
  aisles.  loop_closing.csv names the two keyframes the closer joined, with
  their timestamps; the bag holds the images.  Putting the two frames side by
  side shows what the bag-of-words matched and what the geometric check then
  accepted -- the same repeated asset, the yellow labelled box-stacks on bins,
  seen from opposite sides of the warehouse 17.8 m apart.  A demonstration,
  not a measurement.

Outputs: output/compare/simulation/exp1-false-closure/
  closure_<run>_current_kf<N>.jpg, closure_<run>_matched_kf<M>.jpg
  fig_false_closure_light.png      the GPU front end's pair, side by side
"""

import csv
import os
import sqlite3
import struct
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np                                   # noqa: E402
import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vio_metrics as vm                             # noqa: E402

ROOT = str(vm.ROOT)
OUT = os.path.join(ROOT, "output/compare/simulation/exp1-false-closure")
DS = "dataset_allsensor_realistic_001"
BAG = os.path.join("/media/ambushee/New Volume/wil_rescue", DS, DS + "_0.db3")
TH = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e")


def main():
    os.makedirs(OUT, exist_ok=True)
    gp = [c for c in vm.gt_candidates(DS) if os.path.exists(c)][0]; g = vm.load_gt(gp); t0 = g[0][0]
    stamps = None
    if os.path.exists(BAG):
        con = sqlite3.connect("file:%s?mode=ro" % BAG.replace(" ", "%20"), uri=True)
        tid = [i for i, n in con.execute("select id,name from topics") if n == "/cam0/image_raw/compressed"][0]
        rows = con.execute("select id, substr(data,5,8) from messages where topic_id=?", (tid,)).fetchall()
        stamps = np.array([struct.unpack("<iI", bytes(r[1]))[0] + struct.unpack("<iI", bytes(r[1]))[1] * 1e-9 for r in rows])

    def grab(sec, path):
        if stamps is None:
            return os.path.exists(path)
        k = int(np.argmin(np.abs(stamps - sec)))
        blob = bytes(con.execute("select data from messages where id=?", (int(rows[k][0]),)).fetchone()[0])
        a, b = blob.find(b"\xff\xd8"), blob.rfind(b"\xff\xd9"); open(path, "wb").write(blob[a:b + 2]); return True

    pairs = {}
    for run in ("loopclosing_gpu", "loopclosing_cpu"):
        for r in csv.DictReader(open(os.path.join(ROOT, "output/output_orb/simulation", DS, run, "loop_closing.csv"))):
            if r["status"] == "accepted" and float(r["correction_translation_m"] or 0) > 1:
                tc, tm = int(r["current_timestamp_ns"]) / 1e9, int(r["matched_timestamp_ns"]) / 1e9
                ic, im = np.argmin(np.abs(g[0] - tc)), np.argmin(np.abs(g[0] - tm))
                cur = os.path.join(OUT, "closure_%s_current_kf%s.jpg" % (run, r["current_keyframe_id"]))
                mat = os.path.join(OUT, "closure_%s_matched_kf%s.jpg" % (run, r["matched_keyframe_id"]))
                ok = grab(tc, cur) and grab(tm, mat)
                sep = np.linalg.norm(g[1][ic, :2] - g[1][im, :2])
                print("  %-16s kf%s at %.2f s (%.1f, %.1f)  <-  kf%s at %.2f s (%.1f, %.1f) ; %.1f m apart, correction %.2f m ; frames %s"
                      % (run, r["current_keyframe_id"], tc - t0, g[1][ic, 0], g[1][ic, 1], r["matched_keyframe_id"], tm - t0,
                         g[1][im, 0], g[1][im, 1], sep, float(r["correction_translation_m"]), "ok" if ok else "missing"))
                pairs[run] = (cur, mat, tc - t0, tm - t0, g[1][ic, :2], g[1][im, :2], sep, r["current_keyframe_id"], r["matched_keyframe_id"])
    run = "loopclosing_gpu"
    if run in pairs and os.path.exists(pairs[run][0]):
        cur, mat, tc, tm, pc, pm, sep, kc, km = pairs[run]
        fig = plt.figure(figsize=(11.6, 3.9), facecolor=TH["surface"])
        for k, (path, title) in enumerate(((cur, "(a) the keyframe being closed: kf %s at %.1f s, west side (%.1f, %.1f)" % (kc, tc, *pc)),
                                           (mat, "(b) the keyframe it was matched to: kf %s at %.1f s, east side (%.1f, %.1f)" % (km, tm, *pm)))):
            ax = fig.add_axes([0.02 + 0.49 * k, 0.04, 0.47, 0.80]); ax.imshow(plt.imread(path)); ax.axis("off")
            ax.set_title(title, fontsize=9.5, loc="left", color=TH["ink"])
        fig.suptitle("Experiment 1, simulation: the false loop closure on realistic_001 joined these two views, %.1f m apart" % sep,
                     fontsize=11, x=0.012, y=0.975, ha="left", color=TH["ink"])
        path = os.path.join(OUT, "fig_false_closure_light.png"); fig.savefig(path, dpi=160, facecolor=TH["surface"]); plt.close(fig)
        print("  wrote", path)


if __name__ == "__main__":
    main()
