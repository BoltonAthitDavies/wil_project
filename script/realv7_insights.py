#!/usr/bin/env python3
"""Real-robot figures and tables for the three dataset_realv7_* recordings (3 Nov 2026).

Reads the scorer's outputs (script/eval_real_gt.py, raw tracker export, which
passes the validity rules on all three recordings):
  output/compare/real/realv7_<ds>_{lc_gpu,nolc_gpu,lc_cpu,nolc_cpu,vins}_raw/
realv7_000 was scored with its bag present (wheel-speed time offset); 001 and 002
were scored with --bag-from-runs (bag absent: offset from the estimator's own
speed, labelled uncertain), and must be re-scored when the bags are mounted.

Adds one diagnostic the scorer does not have: the TILT of each estimate.  A
stereo-inertial estimate of a floor robot should lie in a horizontal plane; the
angle between the plane fitted to the estimate's 3-D positions and the horizontal
measures how far the gravity direction found at initialisation is from vertical.
This needs no time alignment and no ground truth.

Writes to output/compare/real/realv7-insights/:
  tilt_table.tex        every ORB run as written in the live log: plane tilt, rms off-plane, z range, resets
                        (the 3 Nov wrong-frame scores were dropped from it on 4 Nov; scores live in results_table.tex)
  results_table.tex     (--levelled, 4 Nov) every ORB run after the frame repair: IMU-initialised share,
                        reset times, raw tilt, ATE, median, RPE, path ratio, status (from realv7_<ds>_<key>_lvl)
  fig_paths_lvl_light.png  (--levelled) the repaired ORB estimates against the truth
  vins_table.tex        the three VINS runs
  fig_paths_light.png   2-D path of every run against the tracker truth, 3 recordings x 5 runs
  fig_tilt_light.png    side view (x-z, after rotating the plane fit) of every ORB run
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vio_metrics as vm                      # noqa: E402

import matplotlib                             # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt               # noqa: E402

ROOT = Path(vm.ROOT)
R = ROOT / "output" / "compare" / "real"
OUT = R / "realv7-insights"
DSS = ("dataset_realv7_000", "dataset_realv7_001", "dataset_realv7_002")
CONF = {"lc_gpu": ("GPU, LC on", "loopclosing_rate1"), "nolc_gpu": ("GPU, LC off", "noloopclosing_rate1"),
        "lc_cpu": ("CPU, LC on", "loopclosing_rate1_cpu"), "nolc_cpu": ("CPU, LC off", "noloopclosing_rate1_cpu")}
COL = {"lc_gpu": "#2a78d6", "nolc_gpu": "#a347b2", "lc_cpu": "#0097a7", "nolc_cpu": "#e040fb", "vins": "#cc8319"}
TILT_OK_DEG = 10.0


def metrics(d, system):
    m = json.load(open(d / "metrics.json")); entries = m if isinstance(m, list) else list(m.values())
    for e in entries:
        if isinstance(e, dict) and system in str(e.get("system", "")):
            return e
    return entries[0] if entries else {}


def manifest(d):
    return json.load(open(d / "manifest.json"))


def tilt(vio_path):
    a = np.loadtxt(vio_path, delimiter=","); p = a[:, 1:4]; c = p - p.mean(0)
    _, s, vt = np.linalg.svd(c, full_matrices=False)
    n = vt[2]; ang = float(np.degrees(np.arccos(abs(n[2])))); thick = float(s[2] / np.sqrt(len(p)))
    return ang, thick, float(np.ptp(a[:, 3])), c, vt


def summary(rdir):
    p = rdir / "run_summary.csv"
    return dict(x.split(",", 1) for x in open(p).read().splitlines() if "," in x) if p.exists() else {}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in DSS:
        for key, (label, run) in CONF.items():
            d = R / f"realv7_{ds[-3:]}_{key}_raw" / ds
            rdir = ROOT / "output" / "output_orb" / "real" / ds / run
            ang, thick, zr, c, vt = tilt(rdir / "vio.csv")
            m = metrics(d, "ORB"); ta = manifest(d)["time_alignment"]
            # reset times from the repaired scoring's event log (performance.csv), which exists for every run;
            # run_summary.csv is missing where the node was not shut down gracefully (three realv7_000 runs)
            dl = R / f"realv7_{ds[-3:]}_{key}_lvl" / ds
            if (dl / "metrics.json").exists():
                rt = [t for t in (metrics(dl, "ORB").get("reset_times_s") or []) if t >= 0]
                rtxt = ", ".join(f"{t:.0f} s" for t in rt) if rt else "none"
            else:
                rtxt = m.get("resets", "---")
            rows.append(dict(ds=ds, key=key, label=label, tilt=ang, thick=thick, zr=zr, ate=m["ate_rmse"], med=m["ate_median"],
                             ratio=m["path_ratio"], resets=rtxt, dropped=m.get("frames_dropped", "---"),
                             src=("wheel" if ta.get("source", "").startswith("wheel") else "estimator"), ok=ta.get("ok"), c=c, vt=vt,
                             status=m.get("status", "")))
    # ---- tilt table
    L = [r"\begin{table}[htbp]", r"\centering", r"\scriptsize\setlength{\tabcolsep}{3pt}",
         r"\caption[Realv7: the plane of every ORB-SLAM3 estimate as written in the live log.]{ORB-SLAM3 on the three "
         r"\repo{dataset_realv7} recordings, 3 November 2026 afternoon, same setting as the realv6 runs: the plane of "
         r"each estimate as written in the live log, before any repair. ``Tilt'' is the angle between the plane fitted "
         r"to the estimate's 3-D positions and the horizontal; ``off-plane'' the rms distance of the positions from "
         r"that plane. A floor robot's estimate should be horizontal: a tilt near \SI{90}{\degree} means the poses "
         r"were written while the map was not gravity-aligned (IMU not initialised; the vertical is then the camera's "
         r"$y$ axis). The scores are in \cref{tab:realv7-results}, after the repair.}",
         r"\label{tab:realv7-tilt}",
         r"\begin{tabular}{llrrrr}", r"\toprule",
         r"Recording & Run & tilt [\si{\degree}] & off-plane [m] & z range [m] & resets \\", r"\midrule"]
    last = None
    for r in rows:
        if last and r["ds"] != last:
            L.append(r"\addlinespace")
        last = r["ds"]
        flag = "" if r["tilt"] <= TILT_OK_DEG else r"\textcolor{flagred}{"
        end = "" if r["tilt"] <= TILT_OK_DEG else "}"
        L.append(f"\\repo{{{r['ds'][8:]}}} & {r['label']} & {flag}{r['tilt']:.1f}{end} & {r['thick']:.2f} & {r['zr']:.2f} & "
                 f"{r['resets']} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "tilt_table.tex").write_text("\n".join(L) + "\n")

    # ---- VINS table
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption[Realv7: the VINS-Fusion runs.]{VINS-Fusion on the three \repo{dataset_realv7} recordings "
         r"(\repo{config/wil/stereo_imu.yaml}, IMU noise $\times 10$, gravity norm \num{9.2642}). Frames received "
         r"against the bag's count (\SI{30}{\hertz} over the recording), IMU use, resets, and where the estimate ended.}",
         r"\label{tab:realv7-vins}",
         r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Recording & frames received & IMU received / used & resets & z range [m] & $|p|$ at end [m] \\", r"\midrule"]
    for ds in DSS:
        rdir = ROOT / "output" / "output_vins" / "real" / ds / "rate1"; s = summary(rdir)
        a = np.loadtxt(rdir / "vio.csv", delimiter=","); r_end = float(np.hypot(a[-1, 1], a[-1, 2]))
        span = (a[-1, 0] - a[0, 0]) * 1e-9
        L.append(f"\\repo{{{ds[8:]}}} & {s.get('images_received', '?')} of about {int(round(span * 30))} & "
                 f"{s.get('imu_messages_received', '?')} / {s.get('imu_samples_used', '?')} & {s.get('resets', '?')} & "
                 f"{np.ptp(a[:, 3]):.0f} & {r_end:.0f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (OUT / "vins_table.tex").write_text("\n".join(L) + "\n")

    # ---- path grid: 3 recordings x 5 runs
    fig, axes = plt.subplots(3, 5, figsize=(17, 10.5))
    for i, ds in enumerate(DSS):
        gtr = np.loadtxt(R / f"realv7_{ds[-3:]}_lc_gpu_raw" / ds / "ground_truth_robotclock.csv", delimiter=",")
        items = [(key, CONF[key][0], R / f"realv7_{ds[-3:]}_{key}_raw" / ds, "orb") for key in CONF] + \
                [("vins", "VINS-Fusion", R / f"realv7_{ds[-3:]}_vins_raw" / ds, "vins")]
        for j, (key, label, d, sysk) in enumerate(items):
            ax = axes[i, j]
            a = np.loadtxt(d / f"aligned_{sysk}.csv", delimiter=","); m = metrics(d, "ORB" if sysk == "orb" else "VINS")
            ax.plot(gtr[:, 1], gtr[:, 2], color="#2a2a2a", lw=1.8, label="tracker truth")
            ax.plot(a[:, 3], a[:, 4], color=COL[key], lw=1.1, label="estimate, SE(2) aligned")
            ax.plot(a[0, 3], a[0, 4], "o", color=COL[key], ms=5); ax.plot(a[-1, 3], a[-1, 4], "s", color=COL[key], ms=5)
            extra = ""
            if sysk == "orb":
                rr = next(x for x in rows if x["ds"] == ds and x["key"] == key); extra = f", tilt {rr['tilt']:.0f}°"
            ax.set_title(f"{ds[8:]} {label}: ATE {m['ate_rmse']:.2f} m{extra}", fontsize=8.5, loc="left")
            ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
            ax.set_xlim(gtr[:, 1].min() - 1, gtr[:, 1].max() + 1); ax.set_ylim(gtr[:, 2].min() - 1, gtr[:, 2].max() + 1)
            if i == 2: ax.set_xlabel("x [m]")
            if j == 0: ax.set_ylabel("y [m]")
    axes[0, 0].legend(fontsize=7, loc="lower right")
    fig.suptitle("dataset_realv7_000/001/002: every run against the tracker truth (circle = start, square = end; VINS clipped to the axes)", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(OUT / "fig_paths_light.png", dpi=150); plt.close(fig)

    # ---- tilt figure: side views, estimate positions projected on (horizontal, vertical)
    fig, axes = plt.subplots(3, 4, figsize=(14, 8))
    for i, ds in enumerate(DSS):
        for j, key in enumerate(CONF):
            r = next(x for x in rows if x["ds"] == ds and x["key"] == key); ax = axes[i, j]
            c = r["c"]; h = np.hypot(c[:, 0], c[:, 1])
            ax.plot(c[:, 0], c[:, 2], color=COL[key], lw=0.9)
            ax.set_title(f"{ds[8:]} {CONF[key][0]}: tilt {r['tilt']:.0f}°, z range {r['zr']:.1f} m", fontsize=8.5, loc="left")
            ax.set_aspect("equal"); ax.grid(True, alpha=0.3); ax.axhline(0, color="#888", lw=0.6)
            if i == 2: ax.set_xlabel("x [m] (estimate, centred)")
            if j == 0: ax.set_ylabel("z [m]")
    fig.suptitle("ORB-SLAM3 estimates seen from the side (x against z): a floor robot's path should be a flat line", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(OUT / "fig_tilt_light.png", dpi=150); plt.close(fig)

    for r in rows:
        print(f"{r['ds'][8:]} {r['label']:12s} tilt {r['tilt']:5.1f}°  off-plane {r['thick']:.2f}  z {r['zr']:.2f}  ATE {r['ate']:.2f}  ratio {r['ratio']:.2f}  {r['src']}")
    ok = [r for r in rows if r["tilt"] <= TILT_OK_DEG]
    print(f"{len(ok)} of {len(rows)} ORB runs within {TILT_OK_DEG:.0f} deg of horizontal:", [(r['ds'][8:], r['label'], round(r['ate'], 2)) for r in ok])


def levelled():
    """Results of the frame-repaired scoring (eval_real_gt.py --join-imu-init --level-plane
    with the bags on the dataset drive): results_table.tex and fig_paths_lvl_light.png."""
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in DSS:
        items = [(key, CONF[key][0], "ORB") for key in CONF] + [("vins", "VINS-Fusion", "VINS")]
        for key, label, sysk in items:
            d = R / f"realv7_{ds[-3:]}_{key}_lvl" / ds
            if not (d / "metrics.json").exists():
                continue
            m = metrics(d, sysk); ta = manifest(d)["time_alignment"]
            rows.append(dict(ds=ds, key=key, label=label, sysk=sysk, d=d, m=m,
                             src=("wheel" if ta.get("source", "").startswith("wheel") else "estimator")))
    def num(v, p=2):
        return "---" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{p}f}"
    L = [r"\begin{table}[htbp]", r"\centering", r"\scriptsize\setlength{\tabcolsep}{3pt}",
         r"\caption[Realv7: every run after the frame repair, with its initialisation history.]{ORB-SLAM3 on the "
         r"three \repo{dataset_realv7} recordings (3 November 2026, same setting as realv6), scored on 4 November "
         r"with the bags on the dataset drive (wheel-speed time offsets) and the two frame repairs of "
         r"\repo{script/eval_real_gt.py}: the poses written before each IMU initialisation rotated into the frame "
         r"used after it (\texttt{--join-imu-init}), and each map's fitted plane levelled (\texttt{--level-plane}). "
         r"``IMU'' is the share of tracked frames with the IMU initialised; ``resets'' the active-map resets after "
         r"the first pose, in seconds (a reset restarts the position at the origin and is not stitched; those "
         r"inside the route are tracking losses); ``tilt'' the plane tilt of the raw log, zero after the repair in "
         r"every run. ATE is planar after SE(2) alignment, RPE over \SI{1}{\second}. Status from the scorer's "
         r"rules (``partial'' = a reset or loss the run recovered from). Lever arm unmeasured (zero); VINS-Fusion "
         r"diverged on all three (\cref{tab:realv7-vins}) and is omitted. \textbf{Bold} marks the lowest ATE "
         r"among the four configurations of a recording; with one run per configuration that records which run "
         r"came lowest, not that the configuration is better.}",
         r"\label{tab:realv7-results}",
         r"\begin{tabular}{llrlrrrrrl}", r"\toprule",
         r"Recording & Run & IMU [\%] & resets [s] & tilt [\si{\degree}] & ATE [m] & median [m] & RPE$_{1\,\mathrm{s}}$ [m] & path ratio & status \\", r"\midrule"]
    orb = [r for r in rows if r["sysk"] == "ORB"]
    bestate = {}
    for ds in {r["ds"] for r in orb}:
        vals = [r["m"]["ate_rmse"] for r in orb if r["ds"] == ds
                and r["m"].get("ate_rmse") is not None and np.isfinite(r["m"]["ate_rmse"])]
        bestate[ds] = min(vals) if vals else None
    last = None
    for r in rows:
        if r["sysk"] != "ORB":
            continue
        if last and r["ds"] != last:
            L.append(r"\addlinespace")
        last = r["ds"]; m = r["m"]
        resets = [t for t in (m.get("reset_times_s") or []) if t >= 0]
        rtxt = ", ".join(f"{t:.0f}" for t in resets) if resets else "none"
        late = any(t > 10 for t in resets)
        if late:
            rtxt = r"\textcolor{flagred}{" + rtxt + "}"
        frac = m.get("imu_init_fraction")
        status = (m.get("status") or "").split(" (")[0]
        atetxt = num(m['ate_rmse'])
        _b = bestate.get(r["ds"])
        if _b is not None and m.get("ate_rmse") is not None and np.isfinite(m["ate_rmse"]) and abs(m["ate_rmse"] - _b) < 1e-9:
            atetxt = r"\textbf{" + atetxt + "}"
        L.append(f"\\repo{{{r['ds'][8:]}}} & {r['label']} & {num(100 * frac, 0) if frac is not None else '---'} & {rtxt} & "
                 f"{num(m.get('tilt_before_deg'), 0)} & {atetxt} & {num(m['ate_median'])} & "
                 f"{num(m.get('rpe1_trans_rmse'))} & {num(m['path_ratio'])} & {status} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "results_table.tex").write_text("\n".join(L) + "\n")

    # ---- path grid after the repair: 3 recordings x 4 ORB runs
    fig, axes = plt.subplots(3, 4, figsize=(14, 10.5))
    for i, ds in enumerate(DSS):
        for j, key in enumerate(CONF):
            r = next((x for x in rows if x["ds"] == ds and x["key"] == key), None); ax = axes[i, j]
            if r is None:
                ax.set_axis_off(); continue
            gtr = np.loadtxt(r["d"] / "ground_truth_robotclock.csv", delimiter=",")
            a = np.loadtxt(r["d"] / "aligned_orb.csv", delimiter=","); m = r["m"]
            ax.plot(gtr[:, 1], gtr[:, 2], color="#2a2a2a", lw=1.8, label="tracker truth")
            ax.plot(a[:, 3], a[:, 4], color=COL[key], lw=1.1, label="estimate, repaired + SE(2) aligned")
            ax.plot(a[0, 3], a[0, 4], "o", color=COL[key], ms=5); ax.plot(a[-1, 3], a[-1, 4], "s", color=COL[key], ms=5)
            frac = m.get("imu_init_fraction") or 0.0
            ax.set_title(f"{ds[8:]} {CONF[key][0]}: ATE {m['ate_rmse']:.2f} m, IMU {100 * frac:.0f} %", fontsize=8.5, loc="left")
            ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
            ax.set_xlim(gtr[:, 1].min() - 1, gtr[:, 1].max() + 1); ax.set_ylim(gtr[:, 2].min() - 1, gtr[:, 2].max() + 1)
            if i == 2: ax.set_xlabel("x [m]")
            if j == 0: ax.set_ylabel("y [m]")
    axes[0, 0].legend(fontsize=7, loc="lower right")
    fig.suptitle("dataset_realv7_000/001/002: ORB-SLAM3 runs after the frame repair (circle = start, square = end)", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(OUT / "fig_paths_lvl_light.png", dpi=150); plt.close(fig)
    for r in rows:
        if r["sysk"] == "ORB":
            m = r["m"]
            print(f"{r['ds'][8:]} {r['label']:12s} IMU {100 * (m.get('imu_init_fraction') or 0):3.0f} %  resets {[round(t) for t in (m.get('reset_times_s') or []) if t >= 0]}  "
                  f"tilt {m.get('tilt_before_deg', float('nan')):4.0f}  ATE {m['ate_rmse']:.2f}  med {m['ate_median']:.2f}  RPE1 {m.get('rpe1_trans_rmse') or float('nan'):.2f}  {m.get('status', '')[:9]}")


if __name__ == "__main__":
    import sys
    if "--levelled" in sys.argv:
        levelled()
    else:
        main()
