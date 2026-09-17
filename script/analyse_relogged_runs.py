#!/usr/bin/env python3
"""Analyse the 2026-09-16 repeated SLAM logs without modifying raw results.

This is deliberately separate from the legacy dataset-root plotters.  The new
campaign stores one experimental unit in each ``logging_*`` directory. Ground
truth is extracted once per exact-name rosbag into the derived output tree, then
associated with every repeat by source timestamp.

Outputs are written below ``output/compare/relogged_20260916``.  Numerical CSVs
are written before figures so every plotted summary is auditable.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "output"
OUT = RAW / "compare" / "relogged_20260916"
OUT.mkdir(parents=True, exist_ok=True)

SYSTEMS = {
    "output_vins": "VINS-Fusion",
    "output_orb": "ORB-SLAM3",
    "output_vins_yolo": "VINS-Fusion+YOLO",
    "output_orb_yolo": "ORB-SLAM3+YOLO",
}
ORDER = list(SYSTEMS.values())
COLORS = {
    "reference": "#79cfb2",
    "VINS-Fusion": "#2a78d6",
    "ORB-SLAM3": "#eb6834",
    "VINS-Fusion+YOLO": "#7c5cd6",
    "ORB-SLAM3+YOLO": "#c23b7a",
}
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e",
                  ink3="#8a8983", grid="#e3e2dd"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7",
                 ink3="#87867d", grid="#33332f"),
}


def read_kv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {r[0]: r[1] for r in csv.reader(f) if len(r) >= 2 and r[0] != "key"}


def read_table(path: Path) -> dict[str, np.ndarray]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    out = {}
    for key in rows[0]:
        vals = []
        numeric = True
        for row in rows:
            try:
                vals.append(float(row[key]))
            except (TypeError, ValueError):
                numeric = False
                break
        out[key] = np.asarray(vals) if numeric else np.asarray([r[key] for r in rows])
    return out


def fnum(d: dict[str, str], key: str, default=np.nan) -> float:
    try:
        return float(d.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def load_vio(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    a = np.loadtxt(path, delimiter=",")
    if a.ndim == 1:
        a = a[None, :]
    if len(a) < 10:
        return None
    return dict(t=a[:, 0] / 1e9, p=a[:, 1:4], q=a[:, 4:8],
                v=a[:, 8:11] if a.shape[1] >= 11 else np.zeros((len(a), 3)))


def quat_to_R(q):
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], -1),
        np.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], -1),
        np.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], -1),
    ], -2)


def euler_zyx(R):
    return np.degrees(np.stack([
        np.arctan2(R[:, 1, 0], R[:, 0, 0]),
        np.arcsin(np.clip(-R[:, 2, 0], -1, 1)),
        np.arctan2(R[:, 2, 1], R[:, 2, 2]),
    ], axis=1))


def umeyama(src, dst):
    cs, cd = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src-cs).T @ (dst-cd))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cd - R @ cs


def interp(t, x, ti):
    return np.stack([np.interp(ti, t, x[:, k]) for k in range(x.shape[1])], axis=1)


def interp_quat(t, q, ti):
    q = q.copy()
    for i in range(1, len(q)):
        if np.dot(q[i-1], q[i]) < 0:
            q[i] *= -1
    out = interp(t, q, ti)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def geodesic_deg(Ra, Rb):
    d = np.einsum("nji,njk->nik", Ra, Rb)
    tr = d[:, 0, 0] + d[:, 1, 1] + d[:, 2, 2]
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def discover_runs():
    runs = []
    for tree, label in SYSTEMS.items():
        for path in sorted((RAW / tree / "simulation").glob("*/logging_*")):
            runs.append(dict(path=path, tree=tree, system=label,
                             dataset=path.parent.name, repeat=path.name))
    return runs


def sensor_duration(perf):
    t = perf.get("timestamp_ns", np.array([]))
    return (t[-1] - t[0]) / 1e9 if len(t) > 1 else np.nan


def summarize_run(run):
    p = run["path"]
    s, m = read_kv(p / "run_summary.csv"), read_kv(p / "run_metadata.csv")
    fs = read_kv(p / "filter_summary.csv")
    perf = read_table(p / "performance.csv")
    orb = run["system"].startswith("ORB")
    received = fnum(s, "stereo_pairs_received" if orb else "images_received")
    processed = fnum(s, "stereo_pairs_processed" if orb else "images_processed")
    dropped = fnum(s, "queue_dropped_frames" if orb else "images_skipped_before_backend", 0)
    duration = sensor_duration(perf)
    cpu = np.nan
    wall = perf.get("wall_elapsed_ms", np.array([]))
    proc = perf.get("process_cpu_ms", np.array([]))
    if len(wall) > 1 and wall[-1] > wall[0]:
        cpu = 100 * (proc[-1] - proc[0]) / (wall[-1] - wall[0])
    mem = perf.get("resident_memory_kb", np.array([]))
    if orb:
        # Two graceful runs omit final inertial/map counters from run_summary.csv,
        # while every performance/local-mapping row still records the state. Treat
        # this as incomplete summary instrumentation, not as measured init failure.
        if "imu_initialized" in s:
            initialized = fnum(s, "imu_initialized", 0) == 1
            init_evidence = "run_summary"
        else:
            pi = perf.get("imu_initialized", np.array([]))
            initialized = bool(len(pi) and np.nanmax(pi) >= 1)
            init_evidence = "performance_fallback" if initialized else "missing"
    else:
        initialized = fnum(s, "initialization_events", 0) >= 1
        init_evidence = "run_summary"
    valid = (s.get("shutdown_status") == "graceful" and initialized
             and processed >= 10 and (p / "vio.csv").exists())
    filtered = run["system"].endswith("+YOLO")
    detection_matches = fnum(fs or s, "detection_frames_matched", 0)
    return {
        **{k: run[k] for k in ("tree", "system", "dataset", "repeat")},
        "status": "valid" if valid else "excluded",
        "treatment_status": ("filter_inactive" if filtered and detection_matches == 0
                             else "active" if filtered else "baseline"),
        "shutdown": s.get("shutdown_status", "missing"),
        "initialized": int(initialized),
        "initialization_evidence": init_evidence,
        "summary_schema_status": ("incomplete" if orb and "imu_initialized" not in s
                                  else "complete"),
        "sensor_duration_s": duration,
        "input_frames": received,
        "processed_frames": processed,
        "dropped_frames": dropped,
        "processed_fraction_pct": 100 * processed / received if received else np.nan,
        "throughput_hz": processed / duration if duration > 0 else np.nan,
        "poses": fnum(s, "poses_written"),
        "pose_rate_hz": fnum(s, "poses_written") / duration if duration > 0 else np.nan,
        "initialization_s": fnum(s, "initialization_time_s"),
        "tracking_losses_or_resets": (fnum(s, "tracking_loss_events") if orb
                                      else fnum(s, "resets")),
        "tracking_loss_duration_s": fnum(s, "tracking_loss_duration_s", 0),
        "keyframes": (fnum(s, "keyframes_created") if not orb or "keyframes_created" in s
                      else float(np.nanmax(perf.get("keyframes_created", [np.nan])))),
        "loop_closures": (fnum(s, "loop_closures", 0) if "loop_closures" in s
                          else float(np.nanmax(perf.get("loop_closures", [0])))),
        "map_merges": (fnum(s, "map_merges", 0) if "map_merges" in s
                       else float(np.nanmax(perf.get("map_merges", [0])))),
        "cpu_avg_pct_one_core": cpu,
        "memory_median_mb": np.median(mem) / 1024 if len(mem) else np.nan,
        "memory_peak_mb": np.max(mem) / 1024 if len(mem) else np.nan,
        "replay_rate": fnum(m, "replay_rate"),
        "detection_frames_matched": detection_matches,
        "detection_frames_missed": fnum(fs or s, "detection_frames_missed", 0),
        "masks_rejected": fnum(fs or s, "masks_rejected", 0),
        "dynamic_points_dropped": fnum(fs or s, "dynamic_points_dropped", 0),
    }


def module_values(run):
    p, sys = run["path"], run["system"]
    result = []
    if sys.startswith("ORB"):
        front, local, loop = (read_table(p / "tracking_frontend.csv"),
                              read_table(p / "local_mapping.csv"),
                              read_table(p / "loop_closing.csv"))
        specs = [("tracking_total_ms", front, "tracking_total_ms"),
                 ("feature_stereo_ms", front, "image_preprocessing_feature_stereo_ms"),
                 ("local_mapping_total_ms", local, "total_ms"),
                 ("local_BA_ms", local, "local_ba_ms"),
                 ("place_recognition_ms", loop, "duration_ms")]
    else:
        front, back = read_table(p / "frontend.csv"), read_table(p / "backend.csv")
        specs = [("feature_tracking_ms", front, "feature_tracking_ms"),
                 ("backend_total_ms", back, "backend_total_ms"),
                 ("solver_ms", back, "solver_ms"),
                 ("visual_update_ms", back, "visual_update_ms")]
    for module, table, key in specs:
        vals = table.get(key, np.array([])).astype(float)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            result.append((module, vals))
    return result


def trajectory_group(runs, dataset, repeat, valid_keys):
    selected = [r for r in runs if r["dataset"] == dataset and r["repeat"] == repeat
                and (r["system"], r["dataset"], r["repeat"]) in valid_keys]
    data = {r["system"]: load_vio(r["path"] / "vio.csv") for r in selected}
    data = {k: v for k, v in data.items() if v is not None}
    ref = load_vio(OUT / "ground_truth" / f"{dataset}.csv")
    if ref is None:
        return None
    aligned, rows = {}, []
    for name in ORDER:
        d = data.get(name)
        if d is None:
            continue
        lo, hi = max(ref["t"][0], d["t"][0]), min(ref["t"][-1], d["t"][-1])
        mask = (d["t"] >= lo) & (d["t"] <= hi)
        if mask.sum() < 10:
            continue
        t, p, q, v = d["t"][mask], d["p"][mask], d["q"][mask], d["v"][mask]
        rp = interp(ref["t"], ref["p"], t)
        R, tr = umeyama(p, rp)
        pa, va, Ra = (R @ p.T).T + tr, (R @ v.T).T, R @ quat_to_R(q)
        err = np.linalg.norm(pa-rp, axis=1)
        rq = interp_quat(ref["t"], ref["q"], t)
        rot_err = geodesic_deg(Ra, quat_to_R(rq))
        steps = np.linalg.norm(np.diff(p, axis=0), axis=1)
        dt = np.maximum(np.diff(t), 1e-9)
        jump_idx = np.where(steps / dt > 10.0)[0]
        jumps = int(len(jump_idx))
        cuts = [0, *(jump_idx + 1), len(p)]
        segments = [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b-a > 30]
        a, b = max(segments, key=lambda ab: ab[1]-ab[0]) if segments else (0, len(p))
        Rc, tc = umeyama(p[a:b], rp[a:b])
        clean_err = np.linalg.norm((Rc @ p[a:b].T).T + tc - rp[a:b], axis=1)
        aligned[name] = dict(t=t-t[0], raw_t=t, p=pa, v=va, eul=euler_zyx(Ra),
                             ref=rp, err=err, rot_err=rot_err)
        rows.append(dict(dataset=dataset, repeat=repeat, system=name,
                         reference="rosbag /ground_truth/odometry", poses=len(t),
                         overlap_s=t[-1]-t[0], path_m=steps.sum(),
                         trans_rmse_m=np.sqrt(np.mean(err**2)),
                         trans_max_m=np.max(err), trans_final_m=err[-1],
                         rot_rmse_deg=np.sqrt(np.mean(rot_err**2)),
                         rot_max_deg=np.max(rot_err), clean_rmse_m=np.sqrt(np.mean(clean_err**2)),
                         jumps=jumps, mean_speed_m_s=steps.sum()/(t[-1]-t[0])))
    return ref, aligned, rows


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def style(ax, th, xlabel="", ylabel="", title=""):
    ax.set_facecolor(th["surface"]); ax.grid(True, color=th["grid"], lw=.7)
    for sp in ax.spines.values(): sp.set_color(th["grid"])
    ax.tick_params(colors=th["ink3"], labelsize=8)
    ax.set_xlabel(xlabel, color=th["ink2"], fontsize=8)
    ax.set_ylabel(ylabel, color=th["ink2"], fontsize=8)
    ax.set_title(title, color=th["ink"], fontsize=10, loc="left")


def plot_detail(dataset, repeat, aligned, rows, theme):
    th = THEMES[theme]
    fig = plt.figure(figsize=(14, 15.6), facecolor=th["surface"])
    gs = fig.add_gridspec(5, 3, height_ratios=[.42, 1.1, .7, 1, 1],
                          left=.06, right=.98, top=.965, bottom=.05,
                          hspace=.38, wspace=.27)
    axh = fig.add_subplot(gs[0, :]); axh.axis("off")
    hdr = "system                 poses   secs   path m  ATE rms  rot rms  clean  jumps"
    axh.text(0, 1, hdr, va="top", family="monospace", fontsize=9, color=th["ink2"], weight="bold")
    for i, r in enumerate(rows):
        line = (f"{r['system']:<22}{r['poses']:>6}{r['overlap_s']:>7.1f}"
                f"{r['path_m']:>9.2f}{r['trans_rmse_m']:>9.3f}"
                f"{r['rot_rmse_deg']:>9.2f}{r['clean_rmse_m']:>8.3f}{r['jumps']:>7}")
        axh.text(0, .78-i*.18, line, va="top", family="monospace", fontsize=9,
                 color=COLORS[r["system"]])
    axh.text(0, .02, "Ground truth: rosbag /ground_truth/odometry; SE(3) alignment with scale fixed. VINS+YOLO*: zero matched masks.",
             color=th["ink3"], fontsize=8.5)
    ax = fig.add_subplot(gs[1:3, :2])
    refa = aligned["VINS-Fusion"]
    ax.plot(refa["ref"][:,0], refa["ref"][:,1], color=COLORS["reference"], lw=3, alpha=.6,
            label="ground truth")
    for name, a in aligned.items():
        ax.plot(a["p"][:,0], a["p"][:,1], color=COLORS[name], lw=1.4, label=name)
    ax.set_aspect("equal", adjustable="datalim"); style(ax, th, "x [m]", "y [m]", "Trajectory, top-down (SE(3)-aligned, scale fixed)")
    ax.legend(fontsize=8, facecolor=th["surface"], edgecolor=th["grid"], labelcolor=th["ink2"])
    ax = fig.add_subplot(gs[1, 2])
    for name, a in aligned.items(): ax.plot(a["t"], a["err"], color=COLORS[name], lw=1.2, label=name)
    style(ax, th, "t [s]", "error [m]", "Position error vs ground truth")
    ax = fig.add_subplot(gs[2, 2])
    for name, a in aligned.items(): ax.plot(a["t"], a["p"][:,2], color=COLORS[name], lw=1.2)
    style(ax, th, "t [s]", "z [m]", "Height")
    for col, (key, comps, unit, title) in enumerate([
        ("p", ("x","y","z"), "m", "Position"),
        ("v", ("vx","vy","vz"), "m/s", "Velocity (reference frame)"),
        ("eul", ("yaw","pitch","roll"), "deg", "Attitude")]):
        # Use three compact inset axes in the two available grid rows.
        host = fig.add_subplot(gs[3:, col]); host.axis("off")
        box = host.get_position(); gap=.012; h=(box.height-2*gap)/3
        for j, comp in enumerate(comps):
            ax = fig.add_axes([box.x0, box.y1-(j+1)*h-j*gap, box.width, h], facecolor=th["surface"])
            for name, a in aligned.items(): ax.plot(a["t"], a[key][:,j], color=COLORS[name], lw=1.0)
            style(ax, th, "t [s]" if j==2 else "", f"{comp} [{unit}]", title if j==0 else "")
            if j < 2: ax.tick_params(labelbottom=False)
    fig.suptitle(f"VINS-Fusion vs ORB-SLAM3 (+YOLO when available)  --  {dataset}/{repeat}",
                 color=th["ink"], fontsize=13, x=.06, ha="left", y=.987)
    path = OUT / "simulation" / dataset / repeat / f"compare_{theme}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=th["surface"]); plt.close(fig)


def grouped_points(ax, rows, metric, th, title, ylabel):
    labels = ORDER
    for i, label in enumerate(labels):
        vals = [float(r[metric]) for r in rows if r["system"] == label and "dynamic_" in r["dataset"] and np.isfinite(float(r[metric]))]
        if not vals: continue
        x = i + np.linspace(-.12, .12, len(vals))
        ax.scatter(x, vals, s=25, color=COLORS[label], alpha=.75, zorder=3)
        mean, sd = np.mean(vals), np.std(vals, ddof=1) if len(vals)>1 else 0
        ax.errorbar(i, mean, yerr=sd, fmt="_", ms=18, lw=2, color=th["ink"], capsize=4, zorder=4)
    short = [x.replace("-Fusion", "").replace("-SLAM3", "") for x in labels]
    short = [x + ("*" if x == "VINS+YOLO" else "") for x in short]
    ax.set_xticks(range(len(labels)), short, rotation=15, ha="right")
    style(ax, th, "", ylabel, title)


def plot_capacity(run_rows, theme):
    th=THEMES[theme]; fig, axes=plt.subplots(2,3,figsize=(14,8.5),facecolor=th["surface"])
    specs=[("throughput_hz","Processed throughput [Hz]","Achieved estimator throughput"),
           ("processed_fraction_pct","Processed / received [%]","Input retention"),
           ("poses","Pose rows [count]","Pose output"),
           ("cpu_avg_pct_one_core","CPU [% of one core]","Process CPU usage"),
           ("memory_peak_mb","Peak RSS [MiB]","Peak resident memory"),
           ("initialization_s","Initialization [s]","Initialization time")]
    for ax,(m,y,t) in zip(axes.ravel(),specs): grouped_points(ax,run_rows,m,th,t,y)
    fig.suptitle("Relogged dynamic runs: raw values; black = mean ± SD; *VINS+YOLO had zero matched masks",color=th["ink"],fontsize=14,x=.05,ha="left")
    fig.tight_layout(rect=[0,.02,1,.94]); p=OUT/f"capacity_resources_{theme}.png"; fig.savefig(p,dpi=160,facecolor=th["surface"]); plt.close(fig)


def plot_latency(module_rows, theme):
    th=THEMES[theme]
    wanted=[("ORB-SLAM3","tracking_total_ms"),("ORB-SLAM3+YOLO","tracking_total_ms"),
            ("VINS-Fusion","feature_tracking_ms"),("VINS-Fusion+YOLO","feature_tracking_ms"),
            ("ORB-SLAM3","local_mapping_total_ms"),("ORB-SLAM3+YOLO","local_mapping_total_ms"),
            ("VINS-Fusion","backend_total_ms"),("VINS-Fusion+YOLO","backend_total_ms")]
    fig,axes=plt.subplots(2,4,figsize=(16,8),facecolor=th["surface"])
    for ax,(system,module) in zip(axes.ravel(),wanted):
        rr=[r for r in module_rows if r["system"]==system and r["module"]==module and "dynamic_" in r["dataset"]]
        vals=[r["values"] for r in rr]
        if vals:
            pooled=np.concatenate(vals)
            ax.boxplot([pooled],positions=[0],widths=.45,showfliers=False,patch_artist=True,
                       boxprops=dict(facecolor=COLORS[system],alpha=.35,color=COLORS[system]),
                       medianprops=dict(color=th["ink"],lw=1.5),whiskerprops=dict(color=th["ink2"]),
                       capprops=dict(color=th["ink2"]))
            med=[np.median(v) for v in vals]
            ax.scatter(np.linspace(-.12,.12,len(med)),med,color=COLORS[system],edgecolor=th["ink"],s=25,zorder=3)
            ax.text(.03,.96,f"{len(vals)} runs, {len(pooled):,} events",transform=ax.transAxes,va="top",color=th["ink3"],fontsize=8)
        shown = system + ("*" if system == "VINS-Fusion+YOLO" else "")
        ax.set_xticks([]); style(ax,th,"","latency [ms]",f"{shown}\n{module.replace('_',' ')}")
    fig.suptitle("Per-module latency: pooled events + run medians; *VINS+YOLO had zero matched masks",color=th["ink"],fontsize=14,x=.04,ha="left")
    fig.tight_layout(rect=[0,.02,1,.92]); p=OUT/f"module_latency_{theme}.png"; fig.savefig(p,dpi=160,facecolor=th["surface"]); plt.close(fig)


def plot_robustness(run_rows, theme):
    th=THEMES[theme]; fig,axes=plt.subplots(1,3,figsize=(14,4.7),facecolor=th["surface"])
    for ax,(m,y,t) in zip(axes,[
        ("tracking_losses_or_resets","events [count]","Tracking losses (ORB) / resets (VINS)"),
        ("keyframes","keyframes [count]","Keyframes produced"),
        ("tracking_loss_duration_s","duration [s]","ORB tracking-loss duration")]):
        grouped_points(ax,run_rows,m,th,t,y)
    fig.suptitle("Robustness and map-process diagnostics (different event semantics are not equated)",color=th["ink"],fontsize=13,x=.04,ha="left")
    fig.tight_layout(rect=[0,.02,1,.9]); p=OUT/f"robustness_{theme}.png"; fig.savefig(p,dpi=160,facecolor=th["surface"]); plt.close(fig)


def plot_trajectory_montage(groups, theme):
    """Cross-run sheet matching the established trajectories_* figure family."""
    th = THEMES[theme]
    cols = 4
    rows_n = math.ceil(len(groups) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(15, 3.65 * rows_n + .9),
                             facecolor=th["surface"], squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for k, (dataset, repeat, aligned, rows) in enumerate(groups):
        ax = axes[k // cols, k % cols]
        ax.set_visible(True)
        ref = aligned["VINS-Fusion"]["ref"]
        ax.plot(ref[:, 0], ref[:, 1], color=COLORS["reference"], lw=2.5,
                alpha=.55, label="ground truth")
        for name in ORDER:
            a = aligned.get(name)
            if a is not None:
                ax.plot(a["p"][:, 0], a["p"][:, 1], color=COLORS[name], lw=1.05,
                        label=name)
        ax.set_aspect("equal", adjustable="datalim")
        style(ax, th, "x [m]", "y [m]",
              f"{dataset.replace('dataset_', '')} / {repeat.rsplit('_', 1)[-1]}")
        # Compact plausibility annotation alongside the ground-truth view.
        speed_bad = sum(r["mean_speed_m_s"] > 3 for r in rows)
        jumps = sum(r["jumps"] for r in rows)
        ax.text(.98, .02, f"implausible speed: {speed_bad}/{len(rows)}; jumps: {jumps}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
                color=th["ink3"])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(labels), fontsize=8,
               facecolor=th["surface"], edgecolor=th["grid"], labelcolor=th["ink2"])
    fig.suptitle("Relogged trajectories, top-down (SE(3)-aligned to rosbag ground truth)",
                 x=.025, ha="left", y=.992, fontsize=14, color=th["ink"])
    fig.text(.025, .972,
             "each panel is one repeat; fixed-scale SE(3) alignment to rosbag ground truth; VINS+YOLO* had zero matched masks",
             ha="left", va="top", fontsize=8.5, color=th["ink3"])
    fig.tight_layout(rect=[0, 0, 1, .955])
    p = OUT / f"trajectories_{theme}_yolo_nofloortexture.png"
    fig.savefig(p, dpi=160, facecolor=th["surface"]); plt.close(fig)


def plot_summary_table(groups, run_rows, theme):
    """Run-level table matching the established summary_* figure family."""
    th = THEMES[theme]
    run_status = {(r["system"], r["dataset"], r["repeat"]): r for r in run_rows}
    lines = []
    for dataset, repeat, _, rows in groups:
        by = {r["system"]: r for r in rows}
        lines.append((dataset, repeat, by))
    fig_h = 2.6 + .43 * len(lines)
    fig, ax = plt.subplots(figsize=(20, fig_h), facecolor=th["surface"])
    ax.set_facecolor(th["surface"]); ax.axis("off")
    # One dataset column, then the established trans/rot/clean/jump fields.
    left = .015; name_w = .235
    cell_w = (1 - left - name_w - .012) / 16
    xs = [left, *[left + name_w + i * cell_w for i in range(16)]]
    nrows = len(lines) + 5
    dy = .92 / nrows
    y = .94
    ax.text(left, y, "run", transform=ax.transAxes, family="monospace",
            fontsize=9, weight="bold", color=th["ink2"])
    for si, sys in enumerate(ORDER):
        start = 1 + si * 4
        label = sys + ("*" if sys == "VINS-Fusion+YOLO" else "")
        ax.text((xs[start] + xs[start+3]) / 2, y, label, transform=ax.transAxes,
                ha="center", family="monospace", fontsize=9, weight="bold",
                color=COLORS[sys])
    y -= dy
    ax.text(left, y, "dataset / repeat", transform=ax.transAxes, family="monospace",
            fontsize=8, color=th["ink2"])
    for si in range(4):
        for j, h in enumerate(("trans", "rot", "clean", "jmp")):
            ax.text(xs[1+si*4+j], y, h, transform=ax.transAxes, ha="right",
                    family="monospace", fontsize=8, color=th["ink2"])
    y -= .55 * dy
    ax.plot([left, .99], [y, y], transform=ax.transAxes, color=th["grid"], lw=1.2)
    y -= .65 * dy
    for dataset, repeat, by in lines:
        short = dataset.replace("dataset_", "") + "/" + repeat.rsplit("_", 1)[-1]
        ax.text(left, y, short, transform=ax.transAxes, family="monospace",
                fontsize=8, color=th["ink"])
        for si, sys in enumerate(ORDER):
            r = by.get(sys)
            status = run_status.get((sys, dataset, repeat), {})
            for j in range(4):
                x = xs[1+si*4+j]
                if r is None:
                    val, col = "--", th["ink3"]
                elif j == 0:
                    val = f"{r['trans_rmse_m']:.2f}"
                    col = "#c0442a" if r["trans_max_m"] > 50 else COLORS[sys]
                elif j == 1:
                    val = f"{r['rot_rmse_deg']:.1f}"
                    col = COLORS[sys]
                elif j == 2:
                    val = f"{r['clean_rmse_m']:.2f}"
                    col = COLORS[sys]
                else:
                    val = str(r["jumps"])
                    col = "#c0442a" if r["jumps"] else COLORS[sys]
                if status.get("status") == "excluded":
                    val, col = "EXC", "#c0442a"
                ax.text(x, y, val, transform=ax.transAxes, ha="right",
                        family="monospace", fontsize=8, color=col)
        y -= dy
    ax.plot([left, .99], [y+.35*dy, y+.35*dy], transform=ax.transAxes,
            color=th["grid"], lw=1)
    ax.text(left, y-.15*dy,
            "trans/clean [m] = global/longest-clean-segment ATE RMSE; rot [deg] = geodesic RMS; jmp = steps >10 m/s",
            transform=ax.transAxes, family="monospace", fontsize=8, color=th["ink3"])
    ax.text(left, y-.9*dy,
            "66/66 operational runs; 2 ORB run summaries incomplete; *VINS+YOLO had zero matched masks; only 1/12 ORB filter pairs passed input matching",
            transform=ax.transAxes, family="monospace", fontsize=8, color=th["ink3"])
    fig.suptitle("Relogged no-floor-texture summary: error against rosbag ground truth",
                 x=.025, ha="left", y=.99, fontsize=14, color=th["ink"])
    p = OUT / f"summary_{theme}_yolo_nofloortexture.png"
    fig.savefig(p, dpi=160, facecolor=th["surface"], bbox_inches="tight")
    plt.close(fig)


def main():
    runs=discover_runs()
    run_rows=[summarize_run(r) for r in runs]
    csv_write(OUT/"run_metrics.csv",run_rows)
    valid_keys={(r["system"],r["dataset"],r["repeat"]) for r in run_rows
                if r["status"] == "valid"}

    module_rows=[]; module_csv=[]
    for r in runs:
        for module,vals in module_values(r):
            rec={k:r[k] for k in ("system","dataset","repeat")}
            rec.update(module=module,values=vals); module_rows.append(rec)
            module_csv.append({k:rec[k] for k in ("system","dataset","repeat","module")} | {
                "events":len(vals),"median_ms":np.median(vals),"iqr_ms":np.percentile(vals,75)-np.percentile(vals,25),
                "p95_ms":np.percentile(vals,95),"max_ms":np.max(vals)})
    csv_write(OUT/"module_latency_summary.csv",module_csv)

    trajectory_rows=[]
    datasets=sorted({r["dataset"] for r in runs})
    repeats=sorted({r["repeat"] for r in runs})
    groups=[]
    for ds in datasets:
        for rep in repeats:
            g=trajectory_group(runs,ds,rep,valid_keys)
            if g is None: continue
            _,aligned,rows=g; groups.append((ds,rep,aligned,rows)); trajectory_rows.extend(rows)
            for theme in THEMES: plot_detail(ds,rep,aligned,rows,theme)
    csv_write(OUT/"trajectory_metrics.csv",trajectory_rows)
    for theme in THEMES:
        plot_capacity(run_rows,theme); plot_latency(module_rows,theme); plot_robustness(run_rows,theme)
        plot_trajectory_montage(groups, theme); plot_summary_table(groups, run_rows, theme)

    # Audit treatment pairing without converting unequal input streams into a causal result.
    audit=[]
    idx={(r["system"],r["dataset"],r["repeat"]):r for r in run_rows}
    for base,yolo in [("ORB-SLAM3","ORB-SLAM3+YOLO"),("VINS-Fusion","VINS-Fusion+YOLO")]:
        for ds in sorted(d for d in datasets if "dynamic_" in d):
            for rep in repeats:
                a,b=idx.get((base,ds,rep)),idx.get((yolo,ds,rep))
                if not a or not b:
                    audit.append(dict(estimator=base,dataset=ds,repeat=rep,status="missing_pair",
                                      input_difference_pct="",duration_difference_pct="")); continue
                di=100*abs(a["input_frames"]-b["input_frames"])/max(a["input_frames"],b["input_frames"])
                dd=100*abs(a["sensor_duration_s"]-b["sensor_duration_s"])/max(a["sensor_duration_s"],b["sensor_duration_s"])
                inactive = b["treatment_status"] == "filter_inactive"
                audit.append(dict(estimator=base,dataset=ds,repeat=rep,
                                  status=("filter_inactive" if inactive else
                                          "comparable" if di<=5 and dd<=5 else "descriptive_only"),
                                  input_difference_pct=di,duration_difference_pct=dd))
    csv_write(OUT/"pair_validity.csv",audit)
    print(f"wrote {len(run_rows)} run summaries, {len(trajectory_rows)} trajectory rows, {len(module_csv)} module rows")
    print(OUT)


if __name__ == "__main__":
    main()
