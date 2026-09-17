#!/usr/bin/env python3
"""Describe anomalies in the relogged SLAM campaign without diagnosing causes.

Reads existing raw logs and truth-based summaries. Writes auditable run-level
metrics, a catalogue separating observations from untested explanations, and a
compact diagnostic figure under output/compare/relogged_20260916/.
"""

from pathlib import Path
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "compare" / "relogged_20260916"
TREES = {
    "output_vins": "VINS-Fusion", "output_orb": "ORB-SLAM3",
    "output_vins_yolo": "VINS-Fusion+YOLO", "output_orb_yolo": "ORB-SLAM3+YOLO",
}
COLORS = {"VINS-Fusion":"#2a78d6", "ORB-SLAM3":"#eb6834",
          "VINS-Fusion+YOLO":"#7c5cd6", "ORB-SLAM3+YOLO":"#c23b7a"}
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", ink3="#8a8983", grid="#e3e2dd"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", ink3="#87867d", grid="#33332f"),
}


def table(path):
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def kv(path):
    if not path.exists(): return {}
    with path.open(newline="") as f:
        return {r[0]:r[1] for r in csv.reader(f) if len(r)>=2 and r[0] != "key"}


def write(path, rows):
    with path.open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


truth = {(r["system"],r["dataset"],r["repeat"]):r
         for r in table(OUT/"trajectory_metrics.csv")}
run_summary = {(r["system"],r["dataset"],r["repeat"]):r
               for r in table(OUT/"run_metrics.csv")}

rows=[]
for tree,system in TREES.items():
    for run in sorted((ROOT/"output"/tree/"simulation").glob("*/logging_*")):
        key=(system,run.parent.name,run.name)
        vio=np.loadtxt(run/"vio.csv",delimiter=",")
        if vio.ndim==1: vio=vio[None,:]
        t=vio[:,0]/1e9; step=np.linalg.norm(np.diff(vio[:,1:4],axis=0),axis=1)
        speed=step/np.maximum(np.diff(t),1e-9); ji=np.where(speed>10)[0]
        cuts=[0,*(ji+1),len(vio)]
        seg=max(zip(cuts[:-1],cuts[1:]),key=lambda x:x[1]-x[0])

        yr=table(run/"yolo_mask.csv")
        if yr:
            boxes=np.array([float(r["n_boxes"]) for r in yr]); matched=np.array([float(r["matched"]) for r in yr])
            coverage=np.array([float(r["mask_coverage"]) for r in yr]); age=np.array([float(r["det_age_ms"]) for r in yr])
            good_age=age[matched>0]
        else:
            boxes=matched=coverage=good_age=np.array([])

        perf=table(run/"performance.csv")
        inertial_transition=np.nan
        if system.startswith("ORB") and perf:
            it=[i for i,r in enumerate(perf) if float(r.get("imu_initialized",0))>=1]
            if it: inertial_transition=float(perf[it[0]]["timestamp_ns"])/1e9-t[0]

        tr=truth.get(key,{})
        rs=run_summary.get(key,{})
        rows.append(dict(
            system=system,dataset=run.parent.name,repeat=run.name,
            poses=len(vio),jumps=len(ji),jump_fraction=len(ji)/max(len(step),1),
            first_jump_s=(t[ji[0]]-t[0]) if len(ji) else "",
            last_jump_s=(t[ji[-1]]-t[0]) if len(ji) else "",
            largest_step_m=float(step.max()),largest_implied_speed_m_s=float(speed.max()),
            longest_clean_duration_s=float(t[seg[1]-1]-t[seg[0]]),
            inertial_transition_s=inertial_transition if np.isfinite(inertial_transition) else "",
            first_jump_minus_inertial_s=((t[ji[0]]-t[0]-inertial_transition)
                                         if len(ji) and np.isfinite(inertial_transition) else ""),
            ate_rmse_m=tr.get("trans_rmse_m",""),clean_ate_rmse_m=tr.get("clean_rmse_m",""),
            global_to_clean_ratio=((float(tr["trans_rmse_m"])/max(float(tr["clean_rmse_m"]),1e-12))
                                   if tr else ""),
            yolo_rows=len(yr),matched_detection_rows=int(matched.sum()) if len(matched) else 0,
            positive_box_rows=int((boxes>0).sum()) if len(boxes) else 0,
            nonzero_coverage_rows=int((coverage>0).sum()) if len(coverage) else 0,
            matched_fraction=float(matched.mean()) if len(matched) else "",
            positive_box_fraction=float((boxes>0).mean()) if len(boxes) else "",
            matched_age_median_ms=float(np.median(good_age)) if len(good_age) else "",
            matched_age_p95_ms=float(np.percentile(good_age,95)) if len(good_age) else "",
            summary_schema_status=rs.get("summary_schema_status",""),
            failure_detection_enabled=kv(run/"run_summary.csv").get("failure_detection_enabled","")))

write(OUT/"anomaly_run_metrics.csv",rows)

orb_yolo=[r for r in rows if r["system"]=="ORB-SLAM3+YOLO"]
vins_yolo=[r for r in rows if r["system"]=="VINS-Fusion+YOLO"]
dynamic=[r for r in rows if "dynamic" in r["dataset"]]
static_orb=[r for r in rows if r["system"]=="ORB-SLAM3" and "static" in r["dataset"]]
pair=table(OUT/"pair_validity.csv")
bad_orb=[r for r in pair if r["estimator"]=="ORB-SLAM3" and r["status"]!="comparable"]

catalog=[
 dict(id="A1",severity="critical",status="observed",anomaly="VINS requested filter inactive",
      affected="12/12 VINS-Fusion+YOLO runs",
      evidence=f"{sum(r['yolo_rows'] for r in vins_yolo)} mask rows; 0 matched messages; 0 positive-box rows; 0 nonzero-coverage rows",
      interpretation="These folders do not measure an active semantic-filter treatment.",
      untested_explanations="Detector production, topic connection, message transport, and timestamp matching remain untested."),
 dict(id="A2",severity="high",status="observed",anomaly="ORB detector-use repeat variability",
      affected="12/12 ORB-SLAM3+YOLO runs; strongest in repeated identical bags",
      evidence="Positive-box fractions vary from 1.4% to 99.2%; matched messages can contain zero boxes.",
      interpretation="Mask exposure is not repeat-stable, so treatment dose differs among repeats.",
      untested_explanations="Detector start/stop timing, inference scheduling, transport, and scene detections remain untested."),
 dict(id="A3",severity="critical",status="observed",anomaly="Dynamic trajectory discontinuity",
      affected=f"{sum(int(r['jumps'])>0 for r in dynamic)}/{len(dynamic)} dynamic runs",
      evidence="Median jump fractions: VINS 59.6%, ORB 30.3%, nominal VINS+YOLO 61.1%, ORB+YOLO 26.9%.",
      interpretation="Full dynamic trajectories are broken; global ATE cannot be read as ordinary drift.",
      untested_explanations="No calibration, timestamp, state-estimation, or logging cause is assigned."),
 dict(id="A4",severity="high",status="observed",anomaly="Global versus clean-segment error separation",
      affected="All four requested dynamic modes",
      evidence="Median global/clean ATE ratios: VINS 100x, ORB 146x, nominal VINS+YOLO 101x, ORB+YOLO 239x.",
      interpretation="Locally coherent intervals coexist with catastrophic full-run discontinuities.",
      untested_explanations="The transition mechanism and whether recovery/relocalization occurred remain untested."),
 dict(id="A5",severity="medium",status="observed",anomaly="Static ORB initialization-adjacent jump",
      affected=f"{sum(int(r['jumps'])>0 for r in static_orb)}/{len(static_orb)} static ORB runs",
      evidence="First jump occurs near 2.6 s; where a transition timestamp exists, it precedes inertial_initialized by about 0.10--0.17 s.",
      interpretation="The isolated static jump is temporally distinct from sustained dynamic breakdown.",
      untested_explanations="Temporal association does not establish an initialization/rebasing cause."),
 dict(id="A6",severity="medium",status="observed",anomaly="Incomplete ORB run-summary schema",
      affected="2 runs: static_nofloortexture_000 repeats 02 and 03",
      evidence="Final inertial/map counters are absent from run_summary.csv, while performance.csv and local_mapping.csv record initialized states.",
      interpretation="Runs remain operational; summary-only validation would falsely exclude them.",
      untested_explanations="Shutdown ordering and summary-writer behavior remain untested."),
 dict(id="A7",severity="high",status="observed",anomaly="ORB paired-input mismatch",
      affected=f"{len(bad_orb)}/12 baseline/filter pairs",
      evidence="Received-frame differences span 9.95%--30.06% in invalid pairs while duration differences stay below 0.13%.",
      interpretation="Most ORB baseline/filter differences are descriptive, not controlled treatment effects.",
      untested_explanations="Delivery, synchronization, compute contention, and startup ordering remain untested."),
 dict(id="A8",severity="medium",status="observed",anomaly="VINS reset evidence is non-discriminating",
      affected="33/33 VINS baseline and nominal-filter runs",
      evidence="failure_detection_enabled=0 and reset count=0 in every VINS run.",
      interpretation="Zero resets cannot be used as evidence that VINS remained healthy.",
      untested_explanations="The intended failure-detection configuration is not evaluated here."),
]
write(OUT/"anomaly_catalog.csv",catalog)


def style(ax,th,title,ylabel=""):
    ax.set_facecolor(th["surface"]); ax.grid(True,color=th["grid"],lw=.7,zorder=0)
    for s in ax.spines.values(): s.set_color(th["grid"])
    ax.tick_params(colors=th["ink3"],labelsize=8)
    ax.set_title(title,color=th["ink"],fontsize=10,loc="left")
    ax.set_ylabel(ylabel,color=th["ink2"],fontsize=8)


for theme,th in THEMES.items():
    fig,axs=plt.subplots(2,3,figsize=(15,8.5),facecolor=th["surface"])
    # YOLO exposure by run.
    ax=axs[0,0]
    for i,(sys,rr) in enumerate((("VINS-Fusion+YOLO",vins_yolo),("ORB-SLAM3+YOLO",orb_yolo))):
        vals=[100*float(r["matched_fraction"]) for r in rr]
        ax.scatter(np.full(len(vals),i)+np.linspace(-.12,.12,len(vals)),vals,color=COLORS[sys],s=28)
    ax.set_xticks([0,1],["VINS+YOLO*","ORB+YOLO"]); style(ax,th,"Matched detector-message exposure","matched rows [%]")
    ax=axs[0,1]
    for i,(sys,rr) in enumerate((("VINS-Fusion+YOLO",vins_yolo),("ORB-SLAM3+YOLO",orb_yolo))):
        vals=[100*float(r["positive_box_fraction"]) for r in rr]
        ax.scatter(np.full(len(vals),i)+np.linspace(-.12,.12,len(vals)),vals,color=COLORS[sys],s=28)
    ax.set_xticks([0,1],["VINS+YOLO*","ORB+YOLO"]); style(ax,th,"Positive-box exposure","rows with boxes [%]")
    # Dynamic jump fraction.
    ax=axs[0,2]
    for i,sys in enumerate(TREES.values()):
        rr=[r for r in dynamic if r["system"]==sys]; vals=[100*float(r["jump_fraction"]) for r in rr]
        ax.scatter(np.full(len(vals),i)+np.linspace(-.12,.12,len(vals)),vals,color=COLORS[sys],s=28)
    ax.set_xticks(range(4),["VINS","ORB","VINS+YOLO*","ORB+YOLO"],rotation=15,ha="right")
    style(ax,th,"Dynamic trajectory discontinuities","jump steps [%]")
    # Global vs clean ATE.
    ax=axs[1,0]
    for sys in TREES.values():
        rr=[r for r in dynamic if r["system"]==sys]
        ax.scatter([float(r["clean_ate_rmse_m"]) for r in rr],[float(r["ate_rmse_m"]) for r in rr],
                   color=COLORS[sys],label=sys,s=28,alpha=.8)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("clean ATE RMSE [m]",color=th["ink2"],fontsize=8)
    style(ax,th,"Global error versus clean interval","global ATE RMSE [m]")
    ax.legend(fontsize=7,facecolor=th["surface"],edgecolor=th["grid"],labelcolor=th["ink2"])
    # ORB input mismatch.
    ax=axs[1,1]
    orbpair=[r for r in pair if r["estimator"]=="ORB-SLAM3"]
    vals=[float(r["input_difference_pct"]) for r in orbpair]
    ax.scatter(range(1,len(vals)+1),vals,color=COLORS["ORB-SLAM3+YOLO"],s=30)
    ax.axhline(5,color="#c0442a",ls="--",lw=1,label="5% validity limit")
    ax.set_xlabel("paired repeat",color=th["ink2"],fontsize=8); style(ax,th,"ORB paired-input mismatch","received-frame difference [%]")
    ax.legend(fontsize=7,facecolor=th["surface"],edgecolor=th["grid"],labelcolor=th["ink2"])
    # Static ORB first jump and inertial transition.
    ax=axs[1,2]
    for i,r in enumerate(static_orb):
        fj=float(r["first_jump_s"]); ax.scatter(i,fj,color=COLORS["ORB-SLAM3"],s=28,label="first jump" if i==0 else None)
        if r["inertial_transition_s"]!="":
            ax.scatter(i,float(r["inertial_transition_s"]),marker="x",color=th["ink"],s=36,label="inertial state" if i==0 else None)
    ax.set_xlabel("static ORB run",color=th["ink2"],fontsize=8); style(ax,th,"Static ORB transition timing","time from first pose [s]")
    ax.legend(fontsize=7,facecolor=th["surface"],edgecolor=th["grid"],labelcolor=th["ink2"])
    fig.suptitle("Relogged anomaly diagnostics: observations only, root causes untested",
                 x=.04,ha="left",fontsize=14,color=th["ink"])
    fig.tight_layout(rect=[0,.02,1,.94])
    fig.savefig(OUT/f"anomaly_diagnostics_{theme}.png",dpi=160,facecolor=th["surface"])
    plt.close(fig)

print(f"wrote {len(rows)} anomaly run rows and {len(catalog)} anomaly classes to {OUT}")
