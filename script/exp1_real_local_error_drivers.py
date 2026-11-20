#!/usr/bin/env python3
"""What raises the visual estimators' LOCAL error on the REAL rig, within a run.

    python3 script/exp1_real_local_error_drivers.py

WHY THIS EXISTS
  script/exp1_local_error_drivers.py asks, of the simulation runs, what the
  estimator and the robot were doing in the seconds its error grew.  This is
  the same question put to the real recordings, with the same windows, the
  same covariates where the rig can supply them, and the same statistics --
  imported from that script rather than copied, so the two cannot drift.

WHERE THE REAL RIG FORCES A DIFFERENCE, AND WHAT WAS DONE
  * The tracker exports position only.  Per-window RPE is therefore the
    translational TUM form on relative displacements in the SE(2)-aligned
    frame (the eval_real_gt.py convention, so it agrees with tables 4.17 and
    4.18); rotational RPE does not exist.  Turn rate comes from the truth's own
    direction of travel, and is undefined (NaN) below 0.15 m/s.
  * Windows are cut on the scorer's aligned series (aligned_*.csv), which is
    the truth's own 5-6 Hz on realv7 and 1.6 Hz on realv6, already matched to
    the estimator by the scorer's time alignment.  Those stamps sit on the
    estimator's clock (median 15 ms from the nearest ORB frame, 46 ms for
    VINS), so per-frame covariates join by nearest stamp with that tolerance.
  * realv6_000's truth fails the validity rules, so its runs are computed and
    printed as diagnostics but never enter a sign test.  The tests run over the
    12 ORB-SLAM3 runs and 3 VINS-Fusion runs of realv7.
  * There is no mover truth on the real recordings, so the traffic analyses of
    the simulation have no counterpart here.

OUTPUT
  output/compare/real/exp1-local-error/windows.csv  every window, every run
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "script"))
import vio_metrics as vm                                              # noqa: E402
from exp1_local_error_drivers import DELTA, TOL, LOW_VINS, CEIL_VINS, nearest, signs, block_rho  # noqa: E402

OUT = os.path.join(ROOT, "output/compare/real/exp1-local-error")
DSS = ["dataset_realv6_000", "dataset_realv7_000", "dataset_realv7_001", "dataset_realv7_002"]
VALID = {d for d in DSS if "realv7" in d}
CONF = {"lc_gpu": ("ORB gpu LC", "loopclosing_rate1"), "nolc_gpu": ("ORB gpu noLC", "noloopclosing_rate1"),
        "lc_cpu": ("ORB cpu LC", "loopclosing_rate1_cpu"), "nolc_cpu": ("ORB cpu noLC", "noloopclosing_rate1_cpu")}
VINS_RUN = "rate1_002"
INIT_S, MIN_SPEED, TURN_DEG = 5.0, 0.15, 10.0
TOL_ORB, TOL_VINS = 0.12, 0.25


def aligned(path):
    a = np.loadtxt(path, delimiter=",", comments="#")
    return a[:, 0] / 1e9, a[:, 1:3], a[:, 3:5], a[:, 5]


def windows(t, g, e, fr, ft, cols, tol, pf=None, pt=None):
    """One row per admissible 1-s window on the aligned series."""
    j = np.searchsorted(t, t + DELTA, side="left"); i = np.arange(len(t))
    k = j < len(t); i, j = i[k], j[k]
    ok = np.abs((t[j] - t[i]) - DELTA) <= TOL * DELTA; i, j = i[ok], j[ok]
    rpe = np.linalg.norm((e[j] - e[i]) - (g[j] - g[i]), axis=1)
    step = np.linalg.norm(np.diff(e, axis=0), axis=1) / np.diff(t)          # estimator jump test
    jumps = np.where(step > vm.JUMP_SPEED)[0]
    span = np.zeros(len(i), bool)
    for jj in jumps:
        span |= (i <= jj) & (j > jj)
    keep = ~span & (t[i] >= t[0] + INIT_S); i, j, rpe = i[keep], j[keep], rpe[keep]
    # truth's direction of travel, unwrapped, for the turn covariate
    d = np.diff(g, axis=0); head = np.unwrap(np.arctan2(d[:, 1], d[:, 0])); head = np.r_[head[0], head]
    speed = np.linalg.norm(g[j] - g[i], axis=1) / (t[j] - t[i])
    turn = np.degrees(np.abs(head[j] - head[i])); turn[speed < MIN_SPEED] = np.nan
    df = pd.DataFrame(dict(t=t[i], tgt=t[i] - t[0], rpe=rpe, speed=speed, turn_deg=turn,
                           inliers=nearest(ft, fr[cols[0]].values, t[i], tol),
                           features=nearest(ft, fr[cols[1]].values, t[i], tol),
                           track_ms=nearest(ft, fr[cols[2]].values, t[i], tol)))
    if pf is not None:
        for col, name in (("queue_dropped", "drops"), ("keyframes_created", "kf_new")):
            df[name] = nearest(pt, pf[col].values, t[j], 0.3) - nearest(pt, pf[col].values, t[i], 0.3)
    else:
        df["drops"] = np.nan; df["kf_new"] = np.nan
    return df


def collect():
    rows = []
    for ds in DSS:
        short = ds[8:]
        for key, (lbl, run) in CONF.items():
            al = os.path.join(ROOT, "output/compare/real", f"{short}_{key}_p15", ds, "aligned_orb.csv")
            d = os.path.join(ROOT, "output/output_orb/real", ds, run)
            if not os.path.exists(al):
                continue
            t, g, e, _ = aligned(al)
            fr = pd.read_csv(os.path.join(d, "tracking_frontend.csv")); ft = fr.timestamp_ns.values / 1e9
            pf = pd.read_csv(os.path.join(d, "performance.csv")); pt = pf.timestamp_ns.values / 1e9
            w = windows(t, g, e, fr, ft, ("map_matches_inliers", "features", "tracking_total_ms"), TOL_ORB, pf, pt)
            w["ds"], w["lbl"], w["fam"], w["key"], w["valid"] = short, lbl, "ORB", key, ds in VALID
            rows.append(w)
        al = os.path.join(ROOT, "output/compare/real", f"{short}_vins_{VINS_RUN}", ds, "aligned_vins.csv")
        d = os.path.join(ROOT, "output/output_vins/real", ds, VINS_RUN)
        if os.path.exists(al):
            t, g, e, _ = aligned(al)
            fr = pd.read_csv(os.path.join(d, "frontend.csv")); ft = fr.timestamp_ns.values / 1e9
            w = windows(t, g, e, fr, ft, ("feature_count", "stereo_feature_count", "feature_tracking_ms"), TOL_VINS)
            w["ds"], w["lbl"], w["fam"], w["key"], w["valid"] = short, "VINS", "VINS", "vins", ds in VALID
            rows.append(w)
    return pd.concat(rows, ignore_index=True)


def along_cross(ds, key):
    """Full-run error split along / across the truth's direction of travel."""
    sub = f"{ds[8:]}_{key}_p15" if key != "vins" else f"{ds[8:]}_vins_{VINS_RUN}"
    al = os.path.join(ROOT, "output/compare/real", sub, ds, f"aligned_{'orb' if key != 'vins' else 'vins'}.csv")
    t, g, e, _ = aligned(al)
    d = np.gradient(g, axis=0); n = np.linalg.norm(d, axis=1); k = n > 1e-6
    ux, uy = d[k, 0] / n[k], d[k, 1] / n[k]; err = (e - g)[k]
    along = err[:, 0] * ux + err[:, 1] * uy; cross = -err[:, 0] * uy + err[:, 1] * ux
    a2, c2 = np.mean(along ** 2), np.mean(cross ** 2)
    return float(np.sqrt(a2)), float(np.sqrt(c2)), float(c2 / max(a2 + c2, 1e-12))


def vins_growth(ds):
    """Per-second planar error growth split by VINS feature count, as in the simulation."""
    al = os.path.join(ROOT, "output/compare/real", f"{ds[8:]}_vins_{VINS_RUN}", ds, "aligned_vins.csv")
    t, g, e, err = aligned(al)
    fr = pd.read_csv(os.path.join(ROOT, "output/output_vins/real", ds, VINS_RUN, "frontend.csv"))
    tf = fr.timestamp_ns.values / 1e9; fc = fr.feature_count.values.astype(float)
    grid = np.arange(max(t[0], tf[0]) + INIT_S + 1, min(t[-1], tf[-1]) - 1, 1.0)
    f1 = np.interp(grid, tf, fc); e1 = np.interp(grid, t, err); de = np.diff(e1)
    lo, hi = f1[:-1] <= LOW_VINS, f1[:-1] >= CEIL_VINS
    return dict(ceil=float(np.mean(f1 >= CEIL_VINS)), low=float(np.mean(f1 <= LOW_VINS)),
                g_low=float(np.median(de[lo])) if lo.sum() else float("nan"),
                g_hi=float(np.median(de[hi])) if hi.sum() else float("nan"),
                n_low=int(lo.sum()), n_hi=int(hi.sum()), fmin=float(f1.min()), fmed=float(np.median(f1)))


def main():
    D = collect()
    os.makedirs(OUT, exist_ok=True); D.to_csv(os.path.join(OUT, "windows.csv"), index=False)
    V = D[D.valid]; O = V[V.fam == "ORB"]; W = V[V.fam == "VINS"]
    runs = list(O.groupby(["ds", "lbl"])); vruns = list(W.groupby(["ds", "lbl"]))
    print("windows: %d over %d runs (%d ORB + %d VINS on realv7 in the tests; realv6 diagnostic only)"
          % (len(D), D.groupby(["ds", "lbl"]).ngroups, len(runs), len(vruns)))
    print("per-run window counts, realv7 ORB: %s" % sorted(int(len(g)) for _, g in runs))

    print("\n== 1  VINS-Fusion: feature starvation on the real rig ==")
    for ds in DSS:
        r = vins_growth(ds)
        tag = "" if ds in VALID else "  [diagnostic: truth invalid]"
        print("    %-10s features median %3.0f min %3.0f | at ceiling %2.0f%% of seconds, <= %d: %2.0f%% | "
              "growth low %+.4f (n=%d) vs ceiling %+.4f (n=%d) m/s%s"
              % (ds[8:], r["fmed"], r["fmin"], 100 * r["ceil"], LOW_VINS, 100 * r["low"],
                 r["g_low"], r["n_low"], r["g_hi"], r["n_hi"], tag))
    print("    rho(tracked features, RPE) per VINS run, 10-s blocks: %s"
          % ["%s %+.2f" % (d, block_rho(g, "inliers", "rpe", 10.0)) for (d, _), g in vruns])

    print("\n== 2  ORB-SLAM3: per-second RPE against tracked map points (12 realv7 runs) ==")
    signs("rho(tracked points, RPE), whole run",
          [spearmanr(g.dropna(subset=["inliers"]).inliers, g.dropna(subset=["inliers"]).rpe).correlation for _, g in runs])
    signs("rho(tracked points, RPE), inside 10-s blocks", [block_rho(g, "inliers", "rpe", 10.0) for _, g in runs])
    bins = [0, 100, 150, 200, 250, 300, 400, 10000]
    s = O.assign(b=pd.cut(O.inliers, bins)).groupby("b", observed=True).rpe.agg(["median", lambda x: x.quantile(.9), "size"])
    s.columns = ["median", "p90", "n"]
    print("    median / p90 RPE by tracked-point bin:"); print(s.round(4).to_string())
    print("    windows below 250 tracked points: %.1f%% of realv7 ORB windows" % (100 * (O.inliers < 250).mean()))
    print("    realv6 (diagnostic) rho(tracked points, RPE) 10-s blocks: %s"
          % ["%s %+.2f" % (l, block_rho(g, "inliers", "rpe", 10.0)) for (_, l), g in D[(~D.valid) & (D.fam == "ORB")].groupby(["ds", "lbl"])])

    print("\n== 3  the chain: tracking time -> frames dropped; tracked points -> tracking time (20-s blocks) ==")
    for fam, G in (("ORB", O), ("VINS", W)):
        rr = list(G.groupby(["ds", "lbl"])); print("    [%s]" % fam)
        if fam == "ORB":
            signs("rho(tracking ms, frames dropped)", [block_rho(h, "track_ms", "drops", 20.0) for _, h in rr])
        signs("rho(tracked points, tracking ms)", [block_rho(h, "inliers", "track_ms", 20.0) for _, h in rr])
    print("    ORB median drops per 1-s window: %.1f ; median tracking ms GPU %.0f / CPU %.0f"
          % (O.drops.median(), O[O.key.str.endswith("gpu")].track_ms.median(), O[O.key.str.endswith("cpu")].track_ms.median()))
    med = O.groupby(["ds", "key"]).agg(pts=("inliers", "median"), ms=("track_ms", "median"), rpe=("rpe", "median"))
    dif = [(med.loc[(d, "lc_gpu")].pts - med.loc[(d, "lc_cpu")].pts, med.loc[(d, "nolc_gpu")].pts - med.loc[(d, "nolc_cpu")].pts)
           for d in sorted({d for d, _ in med.index})]
    print("    GPU minus CPU median tracked points, paired within recording and LC state: %s"
          % [int(round(x)) for pr in dif for x in pr])
    drpe = [(med.loc[(d, "lc_gpu")].rpe - med.loc[(d, "lc_cpu")].rpe, med.loc[(d, "nolc_gpu")].rpe - med.loc[(d, "nolc_cpu")].rpe)
            for d in sorted({d for d, _ in med.index})]
    signs("GPU minus CPU median RPE, 6 pairs [m]", [x for pr in drpe for x in pr])
    dlc = [(med.loc[(d, "lc_gpu")].rpe - med.loc[(d, "nolc_gpu")].rpe, med.loc[(d, "lc_cpu")].rpe - med.loc[(d, "nolc_cpu")].rpe)
           for d in sorted({d for d, _ in med.index})]
    signs("LC on minus LC off median RPE, 6 pairs [m]", [x for pr in dlc for x in pr])

    print("\n== 4  turning, and what does not matter ==")
    for fam, G in (("ORB", O), ("VINS", W)):
        dd = []
        for _, h in G.groupby(["ds", "lbl"]):
            h = h.dropna(subset=["turn_deg"]); tu, st = h[h.turn_deg > TURN_DEG].rpe, h[h.turn_deg <= TURN_DEG].rpe
            dd.append(tu.median() - st.median() if len(tu) >= 10 and len(st) >= 10 else np.nan)
        signs("%-8s translational RPE, turning (>%.0f deg/s) minus straight [m]" % (fam, TURN_DEG), dd)
    signs("ORB rho(turn rate, RPE), 20-s blocks", [block_rho(h, "turn_deg", "rpe", 20.0) for _, h in runs])
    signs("ORB rho(frames dropped, RPE), 20-s blocks", [block_rho(h, "drops", "rpe", 20.0) for _, h in runs])
    signs("ORB rho(speed, RPE), 20-s blocks", [block_rho(h, "speed", "rpe", 20.0) for _, h in runs])
    signs("ORB rho(extracted features, tracked points), 20-s blocks", [block_rho(h, "features", "inliers", 20.0) for _, h in runs])
    signs("VINS rho(speed, RPE), 20-s blocks", [block_rho(h, "speed", "rpe", 20.0) for _, h in vruns])

    print("\n== 5  which way the error points: along / across the direction of travel (full run, rms) ==")
    print("    %-10s %-13s %8s %8s %11s" % ("recording", "run", "along", "cross", "cross share"))
    cs = {"ORB": [], "VINS": []}
    for ds in DSS:
        for key, (lbl, _) in list(CONF.items()) + [("vins", ("VINS", None))]:
            try:
                a, c, sh = along_cross(ds, key)
            except OSError:
                continue
            tag = "" if ds in VALID else "  [diagnostic]"
            print("    %-10s %-13s %8.3f %8.3f %11.2f%s" % (ds[8:], lbl, a, c, sh, tag))
            if ds in VALID:
                cs["VINS" if key == "vins" else "ORB"].append(sh)
    print("    cross share, realv7: ORB median %.2f (range %.2f-%.2f, n=%d) | VINS median %.2f (range %.2f-%.2f, n=%d)"
          % (np.median(cs["ORB"]), min(cs["ORB"]), max(cs["ORB"]), len(cs["ORB"]),
             np.median(cs["VINS"]), min(cs["VINS"]), max(cs["VINS"]), len(cs["VINS"])))


if __name__ == "__main__":
    main()
