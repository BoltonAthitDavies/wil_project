#!/usr/bin/env python3
"""Experiment 1, real robot: a wheel+gyro dead-reckoning baseline with a FITTED speed
scale, scored against the optical tracker exactly as the visual runs are.

    source /opt/ros/humble/setup.bash
    python3 script/real_wheel_baseline.py --fit dataset_realv7_000 \
        --hold dataset_realv7_001 --hold dataset_realv7_002 --hold dataset_realv6_000

Why this exists.  The real rig publishes /feedback/velocity (std_msgs/Float64, 50 Hz)
in a firmware unit that is not metres per second, /feedback/steering that is a
two-valued command echo (0 or -30) and so carries no yaw rate, and a WitMotion
HWT101CT on /hwt101ct_yaw_publisher whose orientation field is identity and whose
angular_velocity.z is a live gyro (and, as the fit shows, its y axis is the
one that carries the yaw rate: the sensor is mounted with y vertical).  The
rig's wheelbase is 0.60 m (user, 7 Nov 2026; an earlier 0.46 was corrected the
same day); it is NOT used here because the
steering echo carries no angle to put through a bicycle model.  The simulation baselines (wheel odometry, IMU
dead reckoning, EKF) therefore cannot be run as they are; the one non-visual
estimator the signals support is

    heading  psi(t) = psi0 + sign * integral (g_axis - b) dt,   b = gyro bias at rest,
    position x,y  += k * v_feedback * [cos psi, sin psi] dt,

with the speed scale k the unknown.  k is fitted on ONE recording (--fit) by
minimising the planar ATE after rigid SE(2) alignment (scale fixed, so k is
identifiable through path length) and then HELD on the others (--hold); the fit
recording's own score is a calibration, the held recordings' scores are the
result.  The gyro AXIS and sign are chosen on the fit recording too (the sensor's mounting
is not recorded: its y axis carries three times the spread of its z axis on
realv7_000, and a wrong sign mirrors the path, which a proper rotation cannot
align) and held.  The scale search is seeded by the path-length ratio
(tracker path / integral of the feedback) and refined on a grid around it.  Without a yaw-rate
measurement an EKF of the simulation design would reduce to the same integration
with a covariance, so no EKF row is produced; this is stated in the write-up.

Time offset tracker<->bag: the same wheel-speed cross-correlation as
script/eval_real_gt.py (only the SHAPE of the feedback is used there, so the
unknown unit does not matter).  Scoring: eval_real_gt.score (planar ATE at the
tracker's own samples, path ratio, RPE at 1 s), lever arm zero and unmeasured, so
every value is provisional in the same way as the visual rows.  The realv6 export
has one off-path sample (184.9 s) which the realv6 analysis removed; the same rule
(an isolated sample more than 2 m from both neighbours) is applied here and
recorded in the manifest.

Outputs under output/compare/real/exp1-wheel/: wheel_metrics.csv, wheel_table.tex,
wheel_paths_light.png, manifest.json; per-recording estimates under
output/output_wheel/real/<dataset>/gyro_wheel/wheel.csv.
"""
import argparse, json, os, sys, datetime as dt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_real_gt as E   # noqa: E402
import vio_metrics as vm   # noqa: E402

ROOT = vm.ROOT
DRIVE = "/media/ambushee/New Volume/wil_rescue"
IMU_TOPIC = "/hwt101ct_yaw_publisher"
OUT = os.path.join(ROOT, "output", "compare", "real", "exp1-wheel")


def read_bag(ds):
    """Receive-time series of the feedback speed and the gyro, bag-relative seconds."""
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Imu
    from std_msgs.msg import Float64
    uri = os.path.join(DRIVE, ds)
    if not os.path.isdir(uri):
        uri = os.path.join(ROOT, "dataset", ds)
    r = SequentialReader(); r.open(StorageOptions(uri=uri, storage_id="sqlite3"), ConverterOptions("cdr", "cdr"))
    tv, v, tg, g = [], [], [], []
    t0 = None; t1 = 0
    while r.has_next():
        topic, data, ts = r.read_next()
        t0 = ts if t0 is None else min(t0, ts); t1 = max(t1, ts)
        if topic == E.WHEEL_TOPIC:
            tv.append(ts); v.append(deserialize_message(data, Float64).data)
        elif topic == IMU_TOPIC:
            m = deserialize_message(data, Imu); tg.append(ts)
            g.append([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z])
    return dict(uri=uri, t0_ns=t0, t1_ns=t1, tv=np.array(tv, float) * 1e-9, v=np.array(v),
                tg=np.array(tg, float) * 1e-9, g=np.array(g))


def clean_gt(gt, jump_m=2.0):
    """Drop isolated samples more than jump_m from BOTH neighbours (the realv6 rule)."""
    xy, t = gt["xy"], gt["t_rel"]
    keep = np.ones(len(t), bool)
    for i in range(1, len(t) - 1):
        if np.linalg.norm(xy[i] - xy[i - 1]) > jump_m and np.linalg.norm(xy[i] - xy[i + 1]) > jump_m:
            keep[i] = False
    dropped = [(float(t[i]), float(xy[i, 0]), float(xy[i, 1])) for i in np.flatnonzero(~keep)]
    gt = dict(gt); gt["t_rel"], gt["xy"] = t[keep], xy[keep]; gt["n"] = int(keep.sum()); gt["dropped"] = dropped
    return gt


def offset(bag, gt):
    u = np.arange(0.0, (bag["t1_ns"] - bag["t0_ns"]) * 1e-9, 0.05)
    t0 = bag["t0_ns"] * 1e-9
    s_ref = E.resample_smooth(bag["tv"] - t0, np.abs(bag["v"]), u)
    gt_grid = np.arange(gt["t_rel"][0], gt["t_rel"][-1], 0.05)
    s_gt = E.speed_profile(gt["t_rel"], gt["xy"], gt_grid)
    fn = lambda x: np.interp(x, gt_grid, s_gt, left=np.nan, right=np.nan)
    return E.correlate_offset(u, s_ref, gt["t_rel"], fn)


def integrate(bag, k, sign, bias, axis=2):
    """Dead reckoning on the feedback sample times; gyro resampled onto them."""
    t = bag["tv"]; v = bag["v"]
    gz = np.interp(t, bag["tg"], bag["g"][:, axis]) - bias
    dtm = np.diff(t, prepend=t[0])
    psi = np.cumsum(sign * gz * dtm)
    dx = k * v * np.cos(psi) * dtm; dy = k * v * np.sin(psi) * dtm
    p = np.column_stack([np.cumsum(dx), np.cumsum(dy), np.zeros(len(t))])
    return t, p, psi


def rest_bias(bag, axis, min_s=1.0):
    """Mean gyro (axis) while the feedback speed is zero at the head of the bag."""
    t, v = bag["tv"], bag["v"]
    moving = np.flatnonzero(np.abs(v) > 1e-6)
    t_end = t[moving[0]] if len(moving) else t[-1]
    m = (bag["tg"] >= t[0]) & (bag["tg"] <= t_end)
    if m.sum() < 50 or (t_end - t[0]) < min_s:    # no usable rest: fall back to the whole-run mean
        return float(np.mean(bag["g"][:, axis])), float(t_end - t[0]), False
    return float(np.mean(bag["g"][m, axis])), float(t_end - t[0]), True


def score_est(gt_abs_t, gt, t, p):
    q = np.tile([1.0, 0.0, 0.0, 0.0], (len(t), 1))
    return E.score(gt_abs_t, gt["xy"], t, p, q, None, (0.0, 0.0, 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True, help="recording the speed scale and gyro sign are fitted on")
    ap.add_argument("--hold", action="append", default=[], help="recordings scored with the held scale")
    ap.add_argument("--gt-dir", default=os.path.join(ROOT, "dataset", "gt"))
    ap.add_argument("--k-grid", default="0.5,1.5,101", help="search range as multiples of the path-length scale, and steps")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    k0, k1, kn = a.k_grid.split(","); kgrid = np.linspace(float(k0), float(k1), int(kn))
    rows, paths, manifest = [], {}, dict(generated=dt.datetime.now().isoformat(timespec="seconds"), fit=a.fit,
                                         hold=a.hold, lever_arm="0,0,0 (unmeasured)", wheelbase_m=0.60, wheelbase_used=False, gt_clean_rule="isolated sample >2 m from both neighbours")
    k_fit = sign_fit = axis_fit = None
    for ds in [a.fit] + a.hold:
        bag = read_bag(ds)
        gt = clean_gt(E.load_gt(os.path.join(a.gt_dir, ds)))
        off = offset(bag, gt)
        gt_abs = bag["t0_ns"] * 1e-9 + gt["t_rel"] + off["offset"]
        v_int = float(np.sum(np.abs(bag["v"]) * np.diff(bag["tv"], prepend=bag["tv"][0])))
        k_len = gt["path_len"] / v_int
        if ds == a.fit:
            # tracker heading change, for the axis diagnostic
            d = np.diff(gt["xy"], axis=0); hd = np.unwrap(np.arctan2(d[:, 1], d[:, 0])[np.linalg.norm(d, axis=1) > 0.02])
            print("fit %s: tracker heading sweep %.1f rad (abs. increments %.1f rad); path-length k=%.4f" % (ds, hd[-1] - hd[0], np.abs(np.diff(hd)).sum(), k_len))
            best = None
            for axis in (0, 1, 2):
                bias, _, _ = rest_bias(bag, axis)
                gi = np.trapz(bag["g"][:, axis] - bias, bag["tg"])
                print("  gyro axis %d: integrated %.2f rad over the bag, bias %.4f rad/s" % (axis, gi, bias))
                for sign in (+1, -1):
                    for k in kgrid * k_len:
                        t, p, _ = integrate(bag, k, sign, bias, axis)
                        sc = score_est(gt_abs, gt, t, p)
                        if sc.get("valid") and (best is None or sc["ate_rmse"] < best[0]):
                            best = (sc["ate_rmse"], k, sign, axis)
            _, k_fit, sign_fit, axis_fit = best
            print("  chosen: axis=%d sign=%+d k=%.4f (ATE %.3f m)" % (axis_fit, sign_fit, k_fit, best[0]))
        bias, rest_s, rest_ok = rest_bias(bag, axis_fit)
        t, p, psi = integrate(bag, k_fit, sign_fit, bias, axis_fit)
        sc = score_est(gt_abs, gt, t, p)
        rdir = os.path.join(ROOT, "output", "output_wheel", "real", ds, "gyro_wheel"); os.makedirs(rdir, exist_ok=True)
        np.savetxt(os.path.join(rdir, "wheel.csv"), np.column_stack([t * 1e9, p[:, 0], p[:, 1], psi]), delimiter=",",
                   header="t_ns,x,y,yaw_rad  (k=%.4f gyro axis %s sign=%+d bias=%.5f rad/s)" % (k_fit, "xyz"[axis_fit], sign_fit, bias), comments="# ")
        row = dict(dataset=ds, role="fit" if ds == a.fit else "held", k=float(k_fit), gyro_axis="xyz"[axis_fit], gyro_sign=sign_fit,
                   gyro_bias_rad_s=bias, rest_s=rest_s, rest_ok=rest_ok, k_pathlen=k_len, offset_s=off["offset"], offset_r=off["peak_r"],
                   offset_second_r=off["second_r"], gt_n=gt["n"], gt_dropped=len(gt["dropped"]), gt_path_m=gt["path_len"],
                   matched=sc.get("matched"), valid=sc.get("valid"))
        if sc.get("valid"):
            row.update(ate_rmse=sc["ate_rmse"], ate_median=sc["ate_median"], ate_max=sc["ate_max"], ate_final=sc["ate_final"],
                       path_ratio=sc["path_ratio"], matched_s=sc["matched_duration_s"], yaw_align_deg=sc["yaw_align_deg"],
                       rpe1_m=(sc["rpe"].get(1.0) or {}).get("trans_rmse", float("nan")) if isinstance(sc["rpe"].get(1.0), dict) else float("nan"))
            paths[ds] = (sc["g"], sc["pa"])
        rows.append(row); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    # table
    with open(os.path.join(OUT, "wheel_metrics.csv"), "w") as fh:
        keys = sorted({k for r in rows for k in r}); fh.write(",".join(keys) + "\n")
        for r in rows: fh.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    with open(os.path.join(OUT, "wheel_table.tex"), "w") as fh:
        fh.write("% Generated by script/real_wheel_baseline.py -- do not edit by hand.\n\\begin{table}[htbp]\n\\centering\n")
        fh.write("\\caption[Wheel+gyro dead reckoning against the optical tracker.]{Wheel+gyro dead reckoning on the "
                 "real recordings against the optical tracker: the feedback speed "
                 "scaled by $k$ (fitted on \\repo{%s} by minimising planar ATE, held on the others), heading from the "
                 "integrated gyro with its rest bias removed, rigid $SE(2)$ alignment with scale fixed, lever arm zero and "
                 "unmeasured. ATE at the tracker's samples; path ratio is estimated over true path length.}\n"
                 "\\label{tab:exp1-real-wheel}\n\\small\n\\resizebox{\\textwidth}{!}{%%\n\\begin{tabular}{llrrrrrrr}\n\\toprule\n"
                 "Recording & Role & $k$ & Offset [\\si{\\second}] ($r$) & Samples & ATE RMSE [\\si{\\metre}] & ATE max [\\si{\\metre}] & Path ratio & RPE$_{1\\,\\mathrm{s}}$ [\\si{\\metre}] \\\\\n\\midrule\n" % a.fit.replace("dataset_", ""))
        for r in rows:
            fh.write("\\repo{%s} & %s & %.3f & %.2f (%.2f) & %s & %s & %s & %s & %s \\\\\n" % (
                r["dataset"].replace("dataset_", ""), r["role"], r["k"], r["offset_s"], r["offset_r"], r["matched"],
                "%.3f" % r["ate_rmse"] if r.get("valid") else "---", "%.2f" % r["ate_max"] if r.get("valid") else "---",
                "%.3f" % r["path_ratio"] if r.get("valid") else "---", ("%.3f" % r["rpe1_m"]) if r.get("valid") and np.isfinite(r["rpe1_m"]) else "---"))
        fh.write("\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    # figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    th = vm.THEMES["light"] if hasattr(vm, "THEMES") else dict(surface="white", ink="#222", ink2="#555", grid="#ddd")
    n = len(paths); fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.4), facecolor=th["surface"], squeeze=False)
    for ax, (ds, (g, pa)) in zip(axes[0], paths.items()):
        ax.plot(g[:, 0], g[:, 1], color="#222", lw=1.6, label="tracker"); ax.plot(pa[:, 0], pa[:, 1], color="#1f77b4", lw=1.1, label="wheel+gyro")
        r = next(x for x in rows if x["dataset"] == ds)
        ax.set_title("%s (%s): ATE %.2f m, path ratio %.2f" % (ds.replace("dataset_", ""), r["role"], r["ate_rmse"], r["path_ratio"]), fontsize=9)
        ax.set_aspect("equal"); ax.grid(alpha=.3); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "wheel_paths_light.png"), dpi=150, facecolor=th["surface"]); plt.close(fig)
    manifest["rows"] = rows; json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1, default=float)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
