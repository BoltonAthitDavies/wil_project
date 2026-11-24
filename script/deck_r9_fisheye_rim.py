#!/usr/bin/env python3
"""Deck slide R9: on the real fisheye, VINS-Fusion divides by depth and the
corners near the rim of the lens explode.

    python3 script/deck_r9_fisheye_rim.py

  (a) where the real rig's corners are.  The ORB corners matched between the
      two cameras on sampled frames of realv7_000-002 (measurements.json of
      script/exp1_dataset_compare.py, radius from the image centre) are turned
      into the angle from the optical axis by inverting the rig's own
      Kannala-Brandt calibration (KB coefficients and fx as in that script).
      Bars: share of corners per 10 deg.  Line, right axis: tan(angle), the
      number VINS-1 turns each corner into (x/z), which grows without bound
      as the angle nears 90 deg.
  (b) how far each estimate ever gets from its start point, on each realv7
      recording: VINS-Fusion before the fix (run dir rate1), after the fix
      (rate1_002: 70 deg mask, estimate_extrinsic 0), ORB-SLAM3 (patch 15,
      closer on) and the tracker ground truth.  No alignment is needed for this
      measure, so a diverged run can be drawn beside a good one.
"""
import json
import os
import sys

import numpy as np
from matplotlib.patches import Patch

import deck_style as S

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_real_gt as E  # noqa: E402

FX = 607.5
KB = [-0.03645464928389926, -0.0008403394026266405, 0.00017697497379393103, -0.000382711572282332]
RECS = ["dataset_realv7_000", "dataset_realv7_001", "dataset_realv7_002"]


def kb_radius(t):
    return FX * t * (1 + KB[0] * t ** 2 + KB[1] * t ** 4 + KB[2] * t ** 6 + KB[3] * t ** 8)


def far(path):
    V = np.loadtxt(path, delimiter=",")
    P = V[:, 1:3]
    return float(np.linalg.norm(P - P[0], axis=1).max())


def main():
    M = json.load(open(os.path.join(S.ROOT, "output/compare/datasets/measurements.json")))
    R = np.concatenate([M["real"][d]["radius"] for d in RECS])
    grid = np.radians(np.linspace(0, 109.3, 4000))
    ang = np.degrees(np.interp(R, kb_radius(grid), grid))   # monotonic below the 109.3 deg peak
    beyond = 100 * np.mean(ang > 70)
    print("%d corners; beyond 70 deg: %.1f %%; beyond 80: %.1f %%" % (len(R), beyond, 100 * np.mean(ang > 80)))

    import deck_r9_fisheye_zones as Z
    rgb, theta, pts, zk, share = Z.zone_image()
    print("zone frame: corners under 70 %.1f %%, 70-90 %.1f %%, >= 90 %.1f %%" % tuple(share))
    fig, (ax, bx) = S.figure(2, 1, figsize=(S.W, 2.7),
                             gridspec_kw=dict(height_ratios=[1.5, 1.0], hspace=0.30))
    ax.imshow(rgb)
    cols = ["#2ecc71", Z.ZONE[1][1], Z.ZONE[2][1]]
    sel = np.random.default_rng(0).permutation(len(pts))[:1200]
    ax.scatter(pts[sel, 0], pts[sel, 1], s=0.5, c=[cols[k] for k in zk[sel]], lw=0)
    cs = ax.contour(theta, levels=[70, 90], colors="white", linewidths=0.5)
    ax.clabel(cs, fmt=lambda x: "%d\u00b0" % x, fontsize=4.6)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    S.title(ax, "(a) one real frame, coloured by each pixel's angle")

    S.style(bx, grid="y")
    rows = []
    for ds in RECS:
        g = E.load_gt(os.path.join(S.ROOT, "dataset/gt", ds))
        xy = np.asarray(g["xy"])
        rows.append((ds[-3:],
                     far(os.path.join(S.ROOT, "output/output_vins/real", ds, "rate1/vio.csv")),
                     far(os.path.join(S.ROOT, "output/output_vins/real", ds, "rate1_002/vio.csv")),
                     far(os.path.join(S.ROOT, "output/output_orb/real", ds, "loopclosing_rate1/vio.csv")),
                     float(np.linalg.norm(xy - xy[0], axis=1).max())))
    cols = [(S.TH["bad"], "VINS before the fix"), (S.VINS, "VINS after the fix"),
            (S.ORB, "ORB-SLAM3"), (S.TH["ink2"], "true (tracker)")]
    w = 0.2
    for i, r in enumerate(rows):
        print("realv7_%s farthest from start: VINS before %.0f m, after %.1f m, ORB %.1f m, truth %.1f m" % r)
        for k, (c, _) in enumerate(cols):
            bx.bar(i + (k - 1.5) * w, r[k + 1], w * 0.92, color=c)
    bx.set_yscale("log")
    bx.set_ylim(1, 3e8)
    bx.set_yticks([1, 10, 100, 1e3, 1e4, 1e5])
    bx.set_yticklabels(["1 m", "10 m", "100 m", "1 km", "10 km", "100 km"])
    bx.set_xticks(range(3))
    bx.set_xticklabels(["realv7_%s" % r[0] for r in rows])
    S.label(bx, y="farthest from the start")
    S.title(bx, "(b) how far each estimate wanders")
    S.legend(bx, [Patch(color=c, label=n) for c, n in cols], loc="upper center", ncol=2, frameon=False,
             columnspacing=0.8, handlelength=0.9)
    S.save(fig, "r9_fisheye_rim.png")


if __name__ == "__main__":
    main()
