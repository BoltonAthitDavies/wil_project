#!/usr/bin/env python3
"""What raises the visual estimators' LOCAL error, within a run, and what does not.

    python3 script/exp1_local_error_drivers.py

WHY THIS EXISTS
  Every other Experiment 1 analysis compares runs.  This one looks inside them.
  Each run is cut into one-second windows, each scored by its translational
  relative pose error over the next second against the truth -- the error the
  estimator made right there, free of the rigid fit and of what it had already
  accumulated -- and labelled with what the estimator and the robot were doing
  when the window opened: tracked map points (ORB-SLAM3) or tracked features
  (VINS-Fusion), tracking time, frames dropped, yaw swept, speed.

HOW IT IS TESTED, AND WHY THAT WAY
  The windows of one run are not independent samples, so nothing here pools
  them.  A relationship is a Spearman rank correlation inside EACH run, taken
  inside ten-second blocks where a slow trend would otherwise masquerade as a
  relationship, and it is reported as the count of runs that agree on its sign
  with a binomial test over the sixteen ORB-SLAM3 runs.  VINS-Fusion has four
  runs on these recordings, so its results are descriptive and rest on their
  consistency and on the repeat of realistic_002.

  "Cause" below means a mechanism plus temporal order -- the covariate is read
  at the start of the window and the error is what follows -- not an
  intervention.

WHAT IT FINDS (the numbers the chapter quotes)
  1  VINS-Fusion's planar error grows in the seconds it tracks <= 120 of its
     150-feature ceiling, on all four recordings; the episodes are places
     where the camera faces the untextured perimeter wall; the one on
     realistic_002 is reproduced to the feature by a second run (cpu_001).
  2  ORB-SLAM3 shows the same threshold in its per-second error (negative in
     15 of 16 runs; a step below ~250 tracked points) but the collapses last
     under a second and leave no step in the accumulated error.
  3  Tracking time -> frames dropped holds within every run of both front
     ends; tracked points -> tracking time holds on the GPU front end only;
     tracked-point level does not reach the ATE.
  4  Turning does not raise the translational error (paired, not significant);
     the per-second rotational error rises in every run, by ~0.03 deg.

Outputs: output/compare/simulation/exp1-local-error/
  windows.csv                     the 51k windows with their labels
  fig_vins_starvation_light.png   realistic_002, both VINS runs, features and error
  cam0_realistic_002_23s.jpg      the camera frame at the dip, pulled from the bag
"""

import csv
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
import analyse_exp3_timelines as T                   # noqa: E402

ROOT = str(vm.ROOT)
OUT = os.path.join(ROOT, "output/compare/simulation/exp1-local-error")
BAG_DIR = "/media/ambushee/New Volume/wil_rescue"
BAGS = ["dataset_allsensor_realistic_%03d" % i for i in range(4)]
ORB = [("ORB cpu LC", "loopclosing_cpu"), ("ORB gpu LC", "loopclosing_gpu"),
       ("ORB cpu noLC", "noloopclosing_cpu"), ("ORB gpu noLC", "noloopclosing_gpu")]
DELTA, TOL = 1.0, 0.25
LOW_VINS, CEIL_VINS = 120, 149          # VINS tracks at most 150
TH = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#dedcd4")


# ------------------------------------------------------------ windows
def gt_for(ds):
    p = [c for c in vm.gt_candidates(ds) if os.path.exists(c)][0]
    g = vm.load_gt(p)
    yaw = np.unwrap(np.radians(vm.euler_zyx(vm.quat_to_R(g[2]))[:, 0]))
    return dict(path=p, t=g[0], p=g[1], q=g[2], yaw=yaw, t0=g[0][0])


def rpe_pairs(t, p, q, gp, gq):
    """Per-pair translational RPE at DELTA, the vio_metrics.rpe() convention."""
    j = np.searchsorted(t, t + DELTA, side="left"); i = np.arange(len(t))
    k = j < len(t); i, j = i[k], j[k]
    ok = np.abs((t[j] - t[i]) - DELTA) <= TOL * DELTA
    i, j = i[ok], j[ok]
    Rp, Rg = vm.quat_to_R(q[i]), vm.quat_to_R(gq[i])
    dp = np.einsum("nij,ni->nj", Rp, p[j] - p[i])
    dg = np.einsum("nij,ni->nj", Rg, gp[j] - gp[i])
    # rotational RPE: geodesic angle of (Qi^-1 Qj)^-1 (Pi^-1 Pj)
    Rpj, Rgj = vm.quat_to_R(q[j]), vm.quat_to_R(gq[j])
    rel_p = np.einsum("nji,njk->nik", Rp, Rpj); rel_g = np.einsum("nji,njk->nik", Rg, Rgj)
    E = np.einsum("nji,njk->nik", rel_g, rel_p)
    rot = np.degrees(np.arccos(np.clip((np.trace(E, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    return i, j, np.linalg.norm(dp - dg, axis=1), rot


def nearest(ts_src, vals, ts_q, tol=0.03):
    k = np.clip(np.searchsorted(ts_src, ts_q), 1, len(ts_src) - 1)
    k = np.where(np.abs(ts_q - ts_src[k - 1]) < np.abs(ts_q - ts_src[k]), k - 1, k)
    out = vals[k].astype(float); out[np.abs(ts_src[k] - ts_q) > tol] = np.nan
    return out


def windows_orb(ds, run, gt):
    d = os.path.join(ROOT, "output/output_orb/simulation", ds, run)
    e = vm.load(os.path.join(d, "vio.csv"))
    if e is None:
        return None
    t, p, q = e[0], e[1], e[2]
    m = (t >= max(t[0], gt["t"][0])) & (t <= min(t[-1], gt["t"][-1])); t, p, q = t[m], p[m], q[m]
    gp, gq = vm.resample(gt["t"], gt["p"], t), vm.resample_quat(gt["t"], gt["q"], t)
    i, j, err, rot = rpe_pairs(t, p, q, gp, gq)
    span = np.zeros(len(i), bool)
    for jj in vm.find_jumps(t, p)[0]:
        span |= (i <= jj) & (j > jj)
    sw = vm.world_frame_time(d)
    pre = (t[i] < sw + 1.0) if sw is not None else np.zeros(len(i), bool)
    keep = ~span & ~pre; i, j, err, rot = i[keep], j[keep], err[keep], rot[keep]
    fr = pd.read_csv(os.path.join(d, "tracking_frontend.csv")); ft = fr.timestamp_ns.values / 1e9
    pf = pd.read_csv(os.path.join(d, "performance.csv")); pt = pf.timestamp_ns.values / 1e9
    df = pd.DataFrame(dict(
        t=t[i], tgt=t[i] - gt["t0"], rpe=err, rot=rot,
        inliers=nearest(ft, fr.map_matches_inliers.values, t[i]),
        features=nearest(ft, fr.features.values, t[i]),
        track_ms=nearest(ft, fr.tracking_total_ms.values, t[i])))
    for col, name in (("queue_dropped", "drops"), ("keyframes_created", "kf_new")):
        df[name] = nearest(pt, pf[col].values, t[j]) - nearest(pt, pf[col].values, t[i])
    gy = np.interp(t, gt["t"], gt["yaw"])
    df["turn_deg"] = np.degrees(np.abs(gy[j] - gy[i]))
    df["speed"] = np.linalg.norm(gp[j] - gp[i], axis=1) / (t[j] - t[i])
    return df


def windows_vins(ds, run, gt):
    d = os.path.join(ROOT, "output/output_vins/simulation", ds, run)
    e = vm.load(os.path.join(d, "vio.csv"))
    if e is None:
        return None
    t, p, q = e[0], e[1], e[2]
    m = (t >= max(t[0], gt["t"][0])) & (t <= min(t[-1], gt["t"][-1])); t, p, q = t[m], p[m], q[m]
    gp, gq = vm.resample(gt["t"], gt["p"], t), vm.resample_quat(gt["t"], gt["q"], t)
    i, j, err, rot = rpe_pairs(t, p, q, gp, gq)
    keep = t[i] >= t[0] + 5.0; i, j, err, rot = i[keep], j[keep], err[keep], rot[keep]
    fr = pd.read_csv(os.path.join(d, "frontend.csv")); ft = fr.timestamp_ns.values / 1e9
    df = pd.DataFrame(dict(
        t=t[i], tgt=t[i] - gt["t0"], rpe=err, rot=rot,
        inliers=nearest(ft, fr.feature_count.values, t[i]),
        features=nearest(ft, fr.stereo_feature_count.values, t[i]),
        track_ms=nearest(ft, fr.feature_tracking_ms.values, t[i])))
    df["drops"] = np.nan; df["kf_new"] = np.nan
    gy = np.interp(t, gt["t"], gt["yaw"])
    df["turn_deg"] = np.degrees(np.abs(gy[j] - gy[i]))
    df["speed"] = np.linalg.norm(gp[j] - gp[i], axis=1) / (t[j] - t[i])
    return df


# ------------------------------------------------------------ statistics
def signs(name, vals):
    v = np.array([x for x in vals if np.isfinite(x)]); pos = int((v > 0).sum()); n = len(v)
    p = binomtest(max(pos, n - pos), n, 0.5).pvalue if n else 1.0
    print("    %-52s median %+.3f   %2d+ / %2d-   p = %.4f" % (name, np.median(v), pos, n - pos, p))
    return v


def block_rho(g, x, y, blk, minn=15):
    g = g.dropna(subset=[x, y]); b = (g.tgt // blk).astype(int); rs = []
    for _, h in g.groupby(b):
        if len(h) >= minn and h[x].nunique() > 2 and h[y].nunique() > 2:
            rs.append(spearmanr(h[x], h[y]).correlation)
    return float(np.nanmedian(rs)) if rs else float("nan")


def planar(ds, tree, run, gt):
    es = T.error_series(dict(dir=os.path.join(ROOT, "output", tree, "simulation", ds, run)), T.gt_series(gt["path"]))
    return es["t"] - gt["t0"], es["err"]


def vins_growth(ds, run, gt):
    """Per-second planar error growth split by the feature count at that second."""
    fr = pd.read_csv(os.path.join(ROOT, "output/output_vins/simulation", ds, run, "frontend.csv"))
    tf = fr.timestamp_ns.values / 1e9 - gt["t0"]; fc = fr.feature_count.values.astype(float)
    te, e = planar(ds, "output_vins", run, gt)
    grid = np.arange(max(te[0], tf[0]) + 6, min(te[-1], tf[-1]) - 1, 1.0)
    f1 = np.interp(grid, tf, fc); e1 = np.interp(grid, te, e); de = np.diff(e1)
    lo, hi = f1[:-1] <= LOW_VINS, f1[:-1] >= CEIL_VINS
    eps = []; s = None
    for k in range(len(grid)):
        if f1[k] <= LOW_VINS and s is None:
            s = k
        if (f1[k] > LOW_VINS or k == len(grid) - 1) and s is not None:
            if k - s >= 3:
                eps.append((grid[s], grid[k], float(f1[s:k].min()), float(e1[k] - e1[s])))
            s = None
    return dict(ceil_share=float(np.mean(f1 >= CEIL_VINS)), g_low=float(np.median(de[lo])),
                g_hi=float(np.median(de[hi])), n_low=int(lo.sum()), episodes=eps,
                peak=float(e.max()), t_peak=float(te[e.argmax()]), fr=fr, tf=tf, fc=fc, te=te, e=e)


def frame_from_bag(ds, sec, gt, path):
    """One cam0 frame at `sec` seconds into the recording, by header stamp."""
    bag = os.path.join(BAG_DIR, ds, ds + "_0.db3")
    if not os.path.exists(bag):
        return os.path.exists(path)
    con = sqlite3.connect("file:%s?mode=ro" % bag.replace(" ", "%20"), uri=True)
    tid = [i for i, n in con.execute("select id,name from topics") if n == "/cam0/image_raw/compressed"]
    if not tid:
        return False
    rows = con.execute("select id, substr(data,5,8) from messages where topic_id=?", (tid[0],)).fetchall()
    st = np.array([struct.unpack("<iI", bytes(r[1]))[0] + struct.unpack("<iI", bytes(r[1]))[1] * 1e-9 for r in rows])
    k = int(np.argmin(np.abs(st - (gt["t0"] + sec))))
    blob = bytes(con.execute("select data from messages where id=?", (int(rows[k][0]),)).fetchone()[0])
    a, b = blob.find(b"\xff\xd8"), blob.rfind(b"\xff\xd9")
    open(path, "wb").write(blob[a:b + 2])
    return True


# ------------------------------------------------------------ main
def main():
    os.makedirs(OUT, exist_ok=True)
    GT = {ds: gt_for(ds) for ds in BAGS}
    frames = []
    for ds in BAGS:
        for lbl, run in ORB:
            df = windows_orb(ds, run, GT[ds])
            if df is not None:
                df["ds"], df["lbl"], df["fam"] = ds[-3:], lbl, "ORB " + ("GPU" if "gpu" in run else "CPU"); frames.append(df)
        df = windows_vins(ds, "cpu", GT[ds])
        if df is not None:
            df["ds"], df["lbl"], df["fam"] = ds[-3:], "VINS", "VINS"; frames.append(df)
    D = pd.concat(frames, ignore_index=True)
    D.to_csv(os.path.join(OUT, "windows.csv"), index=False)
    O = D[D.fam.str.startswith("ORB")]
    runs = list(O.groupby(["ds", "lbl"]))
    print("windows: %d over %d runs" % (len(D), D.groupby(["ds", "lbl"]).ngroups))

    # ---- 1 VINS: feature starvation
    print("\n== 1  VINS-Fusion: planar error growth per second, features <= %d vs at the ceiling ==" % LOW_VINS)
    V = {}
    for ds in BAGS:
        V[ds] = vins_growth(ds, "cpu", GT[ds])
        r = V[ds]
        print("    %s  at ceiling %2.0f%% of seconds | growth low %+.4f (n=%d) vs ceiling %+.4f m/s | peak %.3f m at %.1f s"
              % (ds[-3:], 100 * r["ceil_share"], r["g_low"], r["n_low"], r["g_hi"], r["peak"], r["t_peak"]))
        for a, b, mn, st in r["episodes"]:
            print("         dip %3.0f-%3.0f s  min %3.0f features  error step %+.3f m" % (a, b, mn, st))
    rep = os.path.join(ROOT, "output/output_vins/simulation", BAGS[2], "cpu_001")
    R2 = vins_growth(BAGS[2], "cpu_001", GT[BAGS[2]]) if os.path.exists(rep) else None
    if R2:
        res = vm.analyse("simulation/" + BAGS[2], systems=[("VINS-Fusion", "output_vins", "cpu_001")])
        res1 = vm.analyse("simulation/" + BAGS[2], systems=[("VINS-Fusion", "output_vins", "cpu")])
        print("    realistic_002 REPEAT (cpu_001): growth low %+.4f vs ceiling %+.4f | peak %.3f m at %.1f s | ATE %.3f vs first run %.3f"
              % (R2["g_low"], R2["g_hi"], R2["peak"], R2["t_peak"], res[0][0]["rms"], res1[0][0]["rms"]))
        for a, b, mn, st in R2["episodes"]:
            print("         dip %3.0f-%3.0f s  min %3.0f features  error step %+.3f m" % (a, b, mn, st))
        fr = V[BAGS[2]]["fr"]; tf = V[BAGS[2]]["tf"]
        for lo, hi, lbl in ((10, 17, "before"), (20, 26, "dip")):
            w = (tf >= lo) & (tf <= hi)
            print("         %-6s features %3.0f  stereo matches %3.0f  (match rate %.0f%%)" % (
                lbl, fr.feature_count[w].median(), fr.stereo_feature_count[w].median(),
                100 * fr.stereo_feature_count[w].median() / fr.feature_count[w].median()))

    # ---- 2 ORB: the threshold, and the corner
    print("\n== 2  ORB-SLAM3: per-second RPE against tracked map points ==")
    signs("rho(tracked points, RPE), whole run", [spearmanr(g.dropna(subset=["inliers"]).inliers, g.dropna(subset=["inliers"]).rpe).correlation for _, g in runs])
    signs("rho(tracked points, RPE), inside 10-s blocks", [block_rho(g, "inliers", "rpe", 10.0) for _, g in runs])
    X = O[~((O.ds == "001") & O.lbl.isin(["ORB cpu LC", "ORB gpu LC"]))]     # false-closure runs out
    X = X.assign(b=pd.cut(X.inliers, [0, 250, 300, 400, 10000]))
    s = X.groupby("b", observed=True).rpe.agg(n="size", median="median", p90=lambda x: x.quantile(.9))
    print("    median / p90 RPE by tracked-point bin (false-closure runs excluded):")
    print(s.round(4).to_string())
    print("    windows below 250: %.2f%% of all ORB windows" % (100 * (O.inliers < 250).mean()))
    g0 = GT[BAGS[0]]; d = np.linalg.norm(g0["p"][:, :2] - np.array([-12.0, -16.3]), axis=1)
    tc = g0["t"][np.argmin(d)] - g0["t0"]
    print("    realistic_000 south-west corner at %.1f s:" % tc)
    for lbl, run in ORB:
        h = O[(O.ds == "000") & (O.lbl == lbl)]; hh = h[(h.tgt > tc - 6) & (h.tgt < tc + 6)]
        base = h[(h.tgt > tc - 20) & (h.tgt < tc - 8)].inliers.median()
        te, e = planar(BAGS[0], "output_orb", run, g0)
        a, b = np.median(e[(te >= tc - 9) & (te <= tc - 1)]), np.median(e[(te >= tc + 2) & (te <= tc + 10)])
        print("      %-13s %3.0f -> min %3d points, <300 for %.1f s | planar error %.3f -> %.3f (step %+.3f m)"
              % (lbl, base, hh.inliers.min(), (hh.inliers < 300).sum() / 30.0, a, b, b - a))

    # ---- 3 the chain
    print("\n== 3  tracking time -> frames dropped -> ... (inside 20-s blocks) ==")
    for fam, g in O.groupby("fam"):
        rr = list(g.groupby(["ds", "lbl"])); print("    [%s]" % fam)
        signs("rho(tracking ms, frames dropped)", [block_rho(h, "track_ms", "drops", 20.0) for _, h in rr])
        signs("rho(tracked points, tracking ms)", [block_rho(h, "inliers", "track_ms", 20.0) for _, h in rr])
    print("    median drops per 1-s window: %s" % O.groupby("fam").drops.median().round(1).to_dict())
    R = [r for r in csv.DictReader(open(os.path.join(ROOT, "output/compare/simulation/exp-realistic-ab/runs_metrics.csv")))
         if "realistic_00" in r["dataset"] and r["tree"] == "output_orb"]
    key = dict(loopclosing_cpu="ORB cpu LC", loopclosing_gpu="ORB gpu LC", noloopclosing_cpu="ORB cpu noLC", noloopclosing_gpu="ORB gpu noLC")
    Tb = pd.DataFrame([dict(ds=r["dataset"][-3:], lbl=key[r["run"]], ate=float(r["ate_rms"]), kept=100 * float(r["of_bag"]),
                            track=O[(O.ds == r["dataset"][-3:]) & (O.lbl == key[r["run"]])].track_ms.median(),
                            inl=O[(O.ds == r["dataset"][-3:]) & (O.lbl == key[r["run"]])].inliers.median()) for r in R])
    print("    across 16 runs: rho(median tracking ms, frames kept) = %+.3f ; rho(median tracked points, ATE) = %+.3f (held runs %+.3f)"
          % (spearmanr(Tb.track, Tb.kept).correlation, spearmanr(Tb.inl, Tb.ate).correlation,
             spearmanr(Tb[Tb.ate < 1].inl, Tb[Tb.ate < 1].ate).correlation))
    dif = [Tb[(Tb.ds == d) & (Tb.lbl == "ORB gpu " + lc)].inl.values[0] - Tb[(Tb.ds == d) & (Tb.lbl == "ORB cpu " + lc)].inl.values[0]
           for d in ("000", "001", "002", "003") for lc in ("LC", "noLC")]
    print("    GPU minus CPU median tracked points, 8 paired: %s" % [int(round(x)) for x in dif])

    # ---- 4 turning, and the nulls
    print("\n== 4  turning, and what does not matter ==")
    for fam, g in D.groupby("fam"):
        dd = [np.median(h[h.turn_deg > 10].rpe) - np.median(h[h.turn_deg < 2].rpe) for _, h in g.groupby(["ds", "lbl"])
              if (h.turn_deg > 10).sum() > 30 and (h.turn_deg < 2).sum() > 30]
        signs("%-8s translational RPE, turning minus straight [m]" % fam, dd)
    for fam, g in D.groupby("fam"):
        dd = [np.median(h[h.turn_deg > 10].rot) - np.median(h[h.turn_deg < 2].rot) for _, h in g.groupby(["ds", "lbl"])
              if (h.turn_deg > 10).sum() > 30 and (h.turn_deg < 2).sum() > 30]
        signs("%-8s ROTATIONAL RPE, turning minus straight [deg]" % fam, dd)
    signs("ORB rho(frames dropped, RPE), 20-s blocks", [block_rho(h, "drops", "rpe", 20.0) for _, h in runs])
    signs("ORB rho(speed, RPE), 20-s blocks", [block_rho(h, "speed", "rpe", 20.0) for _, h in runs])
    signs("ORB rho(extracted features, tracked points), 20-s blocks", [block_rho(h, "features", "inliers", 20.0) for _, h in runs])

    # ---- figure: realistic_002, both VINS runs
    if R2:
        fig_vins(V[BAGS[2]], R2, GT[BAGS[2]])


def fig_vins(r1, r2, gt):
    frame = os.path.join(OUT, "cam0_realistic_002_23s.jpg")
    have = frame_from_bag(BAGS[2], 23.0, gt, frame)
    fig = plt.figure(figsize=(11.0, 5.6), facecolor=TH["surface"])
    axf = fig.add_axes([0.075, 0.57, 0.60, 0.33]); axe = fig.add_axes([0.075, 0.12, 0.60, 0.33], sharex=axf)
    for ax in (axf, axe):
        ax.set_facecolor(TH["surface"]); ax.grid(True, color=TH["grid"], lw=0.6, alpha=0.8); ax.set_axisbelow(True)
        for sp in ax.spines.values(): sp.set_color(TH["grid"])
        ax.tick_params(colors=TH["ink2"], labelsize=8)
    for r, col, lbl in ((r1, "#cc8319", "first run"), (r2, "#1f8a70", "second run")):
        axf.plot(r["tf"], r["fc"], lw=1.0, color=col, label=lbl, alpha=0.9)
        axe.plot(r["te"], r["e"], lw=1.1, color=col, label=lbl, alpha=0.9)
    for a, b, _, _ in r1["episodes"]:
        axf.axvspan(a, b, color=TH["ink2"], alpha=0.10, lw=0); axe.axvspan(a, b, color=TH["ink2"], alpha=0.10, lw=0)
    axf.axhline(LOW_VINS, color=TH["ink2"], ls=":", lw=1.0)
    axf.set_ylabel("tracked features", color=TH["ink2"], fontsize=9); axf.set_ylim(0, 160)
    axf.set_title("(a) VINS-Fusion on realistic_002, two runs: tracked features (ceiling 150, dotted = 120)",
                  fontsize=9.5, loc="left", color=TH["ink"])
    axe.set_ylabel("start-anchored planar error [m]", color=TH["ink2"], fontsize=9)
    axe.set_xlabel("time since the recording's start [s]", color=TH["ink2"], fontsize=9)
    axe.set_title("(b) the same two runs, planar error: the shaded seconds are the only ones below 120 features",
                  fontsize=9.5, loc="left", color=TH["ink"])
    axe.legend(fontsize=8, frameon=False, loc="upper right", labelcolor=TH["ink2"])
    plt.setp(axf.get_xticklabels(), visible=False)
    if have:
        axi = fig.add_axes([0.70, 0.40, 0.29, 0.40]); axi.imshow(plt.imread(frame)); axi.axis("off")
        axi.set_title("(c) cam0 at 23 s, inside the dip:\nthe untextured perimeter wall and floor",
                      fontsize=9, loc="left", color=TH["ink"])
    fig.suptitle("Experiment 1, simulation: VINS-Fusion's excursion on realistic_002 is feature starvation, and it repeats",
                 fontsize=11, x=0.012, y=0.975, ha="left", color=TH["ink"])
    path = os.path.join(OUT, "fig_vins_starvation_light.png")
    fig.savefig(path, dpi=160, facecolor=TH["surface"]); plt.close(fig)
    print("\n  wrote", path, "(frame %s)" % ("embedded" if have else "unavailable"))


if __name__ == "__main__":
    main()
