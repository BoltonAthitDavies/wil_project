#!/usr/bin/env python3
"""Does a moving object in the camera's view raise the visual estimators' error?

    python3 script/exp1_movers_in_view.py

WHY THIS EXISTS
  The project is named for dynamic-object filtering, and the filtered arm could
  not test its premise.  The unfiltered runs on the two traffic recordings
  can: the bag carries every model's true pose (/world/default/pose/info), so
  which movers are in the camera's frustum, how large they are in the image,
  how fast they sweep across it, and whether they are themselves moving can
  be computed per one-second window from the truth, with no detector.

WHAT IT TESTS, in the order the chapter reports them
  1  presence     a mover anywhere in view vs none                (null)
  2  sweep        an object crossing the view laterally > 5 deg/s (ORB +2.6 mm,
                  8/8 runs) -- and the control: a MOVING object in view vs a
                  STATIONARY one (4/4 split), so this is near-field parallax
                  from the robot's own motion, not a dynamic-object effect
  3  co-moving    a pallet jack driving ahead in the same aisle: one episode,
                  realistic_003 31.5-34 s, 2-9x the per-second error in every
                  run; realistic_002 has none, which is the whole
                  disagreement between the recordings

METHOD
  Windows are built exactly as in exp1_local_error_drivers.py.  Each mover's
  true pose is interpolated to the window start, transformed into the camera
  frame through the truth pose of the IMU and the calibrated IMU-to-camera
  extrinsic, and kept if it lies inside the 90 x 59 deg frustum.  Its image
  fraction is its bounding size over distance squared, its bearing rate the
  change of its azimuth over the window, "crossing" lateral motion exceeding
  radial.  Tests are paired per run; a result counts by its sign across runs.

MOVER POSES
  output/compare/simulation/exp1-movers/movers_00{2,3}.csv, one row per
  pose/info message per model, sim time rebuilt from /clock (the stamps in
  pose/info are zero).  They are extracted with rclpy's deserialiser, which
  needs a sourced ROS environment; the cached tables are used when present so
  the analysis itself runs without ROS.

Outputs: output/compare/simulation/exp1-movers/
  fig_jack_ahead_light.png     the frame at 33 s and the five runs' error across it
  cam0_realistic_003_33s.jpg   that frame
"""

import os
import sqlite3
import struct
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402
from scipy.stats import spearmanr, binomtest         # noqa: E402
import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vio_metrics as vm                             # noqa: E402
import exp1_local_error_drivers as L                 # noqa: E402

ROOT = str(vm.ROOT)
OUT = os.path.join(ROOT, "output/compare/simulation/exp1-movers")
BAG_DIR = "/media/ambushee/New Volume/wil_rescue"
DS = {"002": "dataset_allsensor_realistic_002", "003": "dataset_allsensor_realistic_003"}
ORB = [("ORB cpu LC", "loopclosing_cpu"), ("ORB gpu LC", "loopclosing_gpu"),
       ("ORB cpu noLC", "noloopclosing_cpu"), ("ORB gpu noLC", "noloopclosing_gpu")]
VINS = {"002": ["cpu", "cpu_001"], "003": ["cpu"]}
# cam0: fx = fy = 640, cx = 640, cy = 360, 1280 x 720; IMU -> cam0 from the ORB config
FX, FY, CX, CY, W_, H_ = 640.0, 640.0, 640.0, 360.0, 1280, 720
T_B_C = np.array([[0, 0, 1, 0.158], [-1, 0, 0, 0.0936], [0, -1, 0, 0.2475], [0, 0, 0, 1]], float)
SIZE = {"PalletJack": (1.6, 0.8, 1.2), "Bucket": (0.6, 0.6, 0.6), "Cluttering": (1.2, 1.0, 1.0)}
TH = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#dedcd4")
EPISODE = (31.5, 35.6)                     # the co-moving jack on realistic_003


def movers(key):
    path = os.path.join(OUT, "movers_%s.csv" % key)
    if not os.path.exists(path):
        raise SystemExit("missing %s: extract it with a sourced ROS environment (see docstring)" % path)
    M = pd.read_csv(path)
    rng = M.groupby("model").agg(dx=("x", lambda s: s.max() - s.min()), dy=("y", lambda s: s.max() - s.min()))
    keep = rng[(rng.dx > 0.5) | (rng.dy > 0.5)].index          # the ones that actually move
    M = M[M.model.isin(keep)]
    return {m: (g.sim_t.values, g[["x", "y", "z"]].values) for m, g in M.groupby("model")}


def size_of(name):
    for k, v in SIZE.items():
        if k.lower() in name.lower():
            return v
    return (1.0, 1.0, 1.0)


def covariates(tracks, gt, t_abs):
    """n in view, largest image fraction, nearest range, any moving, fastest bearing rate, crossing."""
    i = int(np.clip(np.searchsorted(gt["t"], t_abs), 1, len(gt["t"]) - 1))
    R_wb = vm.quat_to_R(gt["q"][i:i + 1])[0]; R_wc = R_wb @ T_B_C[:3, :3]; p_wc = gt["p"][i] + R_wb @ T_B_C[:3, 3]
    n, frac, near, moving, rate, cross = 0, 0.0, np.inf, 0, 0.0, 0
    for name, (tt, xyz) in tracks.items():
        if t_abs - 0.5 < tt[0] or t_abs + 0.5 > tt[-1]:
            continue
        pa = np.array([np.interp(t_abs - 0.5, tt, xyz[:, k]) for k in range(3)])
        pb = np.array([np.interp(t_abs + 0.5, tt, xyz[:, k]) for k in range(3)])
        pc = R_wc.T @ ((pa + pb) / 2 - p_wc)
        if pc[2] < 0.3 or abs(pc[0] / pc[2]) > CX / FX * 1.15 or abs(pc[1] / pc[2]) > CY / FY * 1.15:
            continue
        w, d, h = size_of(name)
        n += 1; frac = max(frac, min(1.0, (w * FX / pc[2]) * (h * FY / pc[2]) / (W_ * H_))); near = min(near, pc[2])
        moving += np.linalg.norm((pb - pa)[:2]) > 0.1
        ca, cb = R_wc.T @ (pa - p_wc), R_wc.T @ (pb - p_wc)
        if ca[2] > 0.3 and cb[2] > 0.3:
            r = abs(np.degrees(np.arctan2(cb[0], cb[2]) - np.arctan2(ca[0], ca[2])))
            if r > rate:
                rate = r; cross = int(np.linalg.norm((cb - ca)[:2]) > abs((cb - ca)[2]))
    return n, frac, (near if n else np.nan), moving, rate, cross


def signs(name, vals):
    v = np.array([x for x in vals if np.isfinite(x)]); pos = int((v > 0).sum()); n = len(v)
    p = binomtest(max(pos, n - pos), n, 0.5).pvalue if n else 1.0
    print("    %-60s median %+.4f   %d+ / %d-   p = %.4f" % (name, np.median(v), pos, n - pos, p))


def frame_from_bag(key, sec, gt, path):
    bag = os.path.join(BAG_DIR, DS[key], DS[key] + "_0.db3")
    if not os.path.exists(bag):
        return os.path.exists(path)
    con = sqlite3.connect("file:%s?mode=ro" % bag.replace(" ", "%20"), uri=True)
    tid = [i for i, n in con.execute("select id,name from topics") if n == "/cam0/image_raw/compressed"][0]
    rows = con.execute("select id, substr(data,5,8) from messages where topic_id=?", (tid,)).fetchall()
    st = np.array([struct.unpack("<iI", bytes(r[1]))[0] + struct.unpack("<iI", bytes(r[1]))[1] * 1e-9 for r in rows])
    k = int(np.argmin(np.abs(st - (gt["t0"] + sec))))
    blob = bytes(con.execute("select data from messages where id=?", (int(rows[k][0]),)).fetchone()[0])
    a, b = blob.find(b"\xff\xd8"), blob.rfind(b"\xff\xd9"); open(path, "wb").write(blob[a:b + 2])
    return True


def main():
    os.makedirs(OUT, exist_ok=True)
    frames = []
    for key, ds in DS.items():
        gt = L.gt_for(ds); tr = movers(key)
        print("realistic_%s: %d models move: %s" % (key, len(tr), [m.replace("aws_robomaker_warehouse_", "") for m in tr]))
        for lbl, run in ORB:
            df = L.windows_orb(ds, run, gt); df["lbl"] = lbl; df["fam"] = "ORB"; frames.append((key, gt, tr, df))
        for n, run in enumerate(VINS[key]):
            df = L.windows_vins(ds, run, gt); df["lbl"] = "VINS" + (" rep" if n else ""); df["fam"] = "VINS"; frames.append((key, gt, tr, df))
    parts = []
    for key, gt, tr, df in frames:
        cov = np.array([covariates(tr, gt, t) for t in df.t.values], float)
        df["n_mov"], df["frac"], df["near"], df["moving"], df["rate"], df["cross"] = cov.T
        df["inview"] = df.n_mov > 0; df["ds"] = key; parts.append(df)
    D = pd.concat(parts, ignore_index=True)
    O, V = D[D.fam == "ORB"], D[D.fam == "VINS"]
    orb = list(O.groupby(["ds", "lbl"])); vins = list(V.groupby(["ds", "lbl"]))
    print("windows: %d ; a mover in view in %.0f%% of them" % (len(D), 100 * D.inview.mean()))

    print("\n== 1  presence: median RPE with a mover in view minus without, per run ==")
    signs("ORB (8 runs)", [np.median(h[h.inview].rpe) - np.median(h[~h.inview].rpe) for _, h in orb])
    signs("VINS (3 runs)", [np.median(h[h.inview].rpe) - np.median(h[~h.inview].rpe) for _, h in vins])
    print("    confound on realistic_002: VINS features with a mover in view %.0f vs without %.0f (the movers work the textured aisles)"
          % (V[(V.ds == "002") & V.inview].inliers.median(), V[(V.ds == "002") & ~V.inview].inliers.median()))

    print("\n== 2  sweep: an object crossing the view laterally faster than 5 deg/s, minus no mover ==")
    signs("ORB", [np.median(h[(h.cross > 0) & (h.rate > 5)].rpe) - np.median(h[~h.inview].rpe) for _, h in orb])
    signs("VINS", [np.median(h[(h.cross > 0) & (h.rate > 5)].rpe) - np.median(h[~h.inview].rpe) for _, h in vins])
    signs("ORB: fast sweep (>10 deg/s) minus slow (<2), object in view",
          [np.median(h[h.rate > 10].rpe) - np.median(h[h.inview & (h.rate < 2)].rpe) for _, h in orb if (h.rate > 10).sum() > 20])
    signs("ORB control: MOVING object in view minus STATIONARY object in view",
          [np.median(h[h.inview & (h.moving > 0)].rpe) - np.median(h[h.inview & (h.moving == 0)].rpe) for _, h in orb])

    print("\n== 3  the co-moving jack: realistic_003, %.1f-%.1f s ==" % EPISODE)
    gt3 = L.gt_for(DS["003"])
    rows = []
    for (key, lbl), h in list(O.groupby(["ds", "lbl"])) + list(V.groupby(["ds", "lbl"])):
        if key != "003":
            continue
        tg = h.t.values - gt3["t0"]; ep = h[(tg >= EPISODE[0]) & (tg <= EPISODE[1])]; base = h[(tg >= 15) & (tg <= 28)]
        rows.append((lbl, base.rpe.median(), ep.rpe.median(), base.inliers.median(), ep.inliers.median()))
        print("    %-13s base %.4f -> episode %.4f (x%.1f) ; tracked %s %.0f -> %.0f"
              % (lbl, base.rpe.median(), ep.rpe.median(), ep.rpe.median() / base.rpe.median(),
                 "features" if lbl.startswith("VINS") else "points", base.inliers.median(), ep.inliers.median()))
    for key in ("002", "003"):
        h = O[O.ds == key]; big = h.frac > 0.05
        print("    realistic_%s: objects >5%% of the image are %.1f%% of ORB windows and %.1f-%.1f%% of summed RPE across its four runs"
              % (key, 100 * big.mean(),
                 *[100 * x for x in (min(hh.rpe[hh.frac > 0.05].sum() / hh.rpe.sum() for _, hh in h.groupby("lbl")),
                                     max(hh.rpe[hh.frac > 0.05].sum() / hh.rpe.sum() for _, hh in h.groupby("lbl")))]))
    # does realistic_002 hold a co-moving jack ahead at all?
    tr2, gt2 = movers("002"), L.gt_for(DS["002"]); hits = 0
    for tg in np.arange(5, gt2["t"][-1] - gt2["t0"] - 1, 0.5):
        n, frac, near, moving, rate, cross = covariates({k: v for k, v in tr2.items() if "PalletJack" in k}, gt2, gt2["t0"] + tg)
        hits += (n > 0 and near < 5 and moving > 0 and frac > 0.03)
    print("    realistic_002: seconds with a moving jack ahead within 5 m filling >3%% of the image: %d" % (hits // 2))

    # ---- figure: the frame, and the five runs' per-second error across the episode
    frame = os.path.join(OUT, "cam0_realistic_003_33s.jpg"); have = frame_from_bag("003", 33.0, gt3, frame)
    fig = plt.figure(figsize=(11.6, 4.6), facecolor=TH["surface"])
    if have:
        axi = fig.add_axes([0.02, 0.12, 0.46, 0.74]); axi.imshow(plt.imread(frame)); axi.axis("off")
        axi.set_title("(a) cam0 at 33 s: a pallet jack 3.4 m ahead, driving the same way at 1.3 m/s",
                      fontsize=9.5, loc="left", color=TH["ink"])
    ax = fig.add_axes([0.56, 0.16, 0.42, 0.70]); ax.set_facecolor(TH["surface"])
    for sp in ax.spines.values(): sp.set_color(TH["grid"])
    ax.grid(True, color=TH["grid"], lw=0.6, alpha=0.8); ax.set_axisbelow(True); ax.tick_params(colors=TH["ink2"], labelsize=8)
    COL = {"ORB cpu LC": "#e040fb", "ORB gpu LC": "#f48fff", "ORB cpu noLC": "#a347b2", "ORB gpu noLC": "#c98fd9", "VINS": "#cc8319"}
    ax.axvspan(*EPISODE, color=TH["ink2"], alpha=0.10, lw=0)
    for (key, lbl), h in list(O.groupby(["ds", "lbl"])) + list(V.groupby(["ds", "lbl"])):
        if key != "003":
            continue
        tg = h.t.values - gt3["t0"]; m = (tg >= 24) & (tg <= 44)
        s = pd.Series(h.rpe.values[m], index=tg[m]).rolling(15, center=True, min_periods=5).median()
        ax.plot(s.index, s.values, lw=1.3, color=COL[lbl], label=lbl)
    ax.set_yscale("log"); ax.set_xlabel("time since the recording's start [s]", color=TH["ink2"], fontsize=9)
    ax.set_ylabel("per-second RPE [m, log], 0.5-s rolling median", color=TH["ink2"], fontsize=8.5)
    ax.legend(fontsize=7.5, frameon=False, ncol=2, labelcolor=TH["ink2"], loc="lower left")
    ax.set_title("(b) every run on realistic_003 across it (shaded)", fontsize=9.5, loc="left", color=TH["ink"])
    fig.suptitle("Experiment 1, simulation: the one co-moving object in the traffic recordings, and what it costs",
                 fontsize=11, x=0.012, y=0.975, ha="left", color=TH["ink"])
    path = os.path.join(OUT, "fig_jack_ahead_light.png"); fig.savefig(path, dpi=160, facecolor=TH["surface"]); plt.close(fig)
    print("\n  wrote", path)


if __name__ == "__main__":
    main()
