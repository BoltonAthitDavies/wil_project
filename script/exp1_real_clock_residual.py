#!/usr/bin/env python3
"""Residual clock offset in the real-rig scoring, per run, and what it costs.

    python3 script/exp1_real_clock_residual.py

WHY THIS EXISTS
  eval_real_gt.py aligns each estimate to the tracker truth in time by
  correlating speed profiles.  For the wheel+gyro baseline that correlation is
  against the wheel feedback itself (peak r ~ 0.98); for ORB-SLAM3 and
  VINS-Fusion no wheel topic reached the scorer and the estimator's own speed
  had to serve, over a search window of (-5, +20) s.  This script measures how
  far each run's chosen offset sits from the truth, by two criteria that do
  not share a mechanism:

    tau_am   the shift of the truth that zeroes the signed ALONG-track mean
             error.  A clock error of tau at speed v is an along-track offset
             of v*tau, so this mean is linear in tau with slope -v; the zero
             crossing is the residual.  (A forward lever arm would also give a
             constant along-track mean, but the same arm on every recording
             cannot produce a residual that differs between recordings.)
    tau_ate  the shift that minimises the planar ATE after SE(2) alignment.

  Agreement between the two, and between the five estimator runs of one
  recording, is what makes the residual a property of the recording's
  alignment and not of any estimator.

WHAT IT FOUND (17 Nov 2026)
  First pass, ORB scored without the bag (estimator-speed offsets, now in
  trash/exp1_real_p15_nobag_scoring_20261117): realv7_001 residuals of 1.4-5.3 s,
  one at the -5 s search bound.  Re-scored with the wheel topic (r 0.93-0.99)
  the ORB offsets equal the baseline's exactly, two realv7_001 ATEs fall from
  1.278/0.575 to 0.434/0.443 m, and the rest move by <= 0.02 m.
  BUT the residual survives the correct offset: +0.3 / +1.4 / +0.15 s on every
  estimator run of realv7_000/_001/_002 and 0.0 on the baseline with the same
  offset.  It is therefore not a clock error but a constant along-track lead of
  the estimators over the tracker marker (0.04-0.39 m), the same thing the
  scorer's diagnostic lever-arm fit reports (0.31/0.40/0.23 m forward).  The
  marker-to-IMU lever arm must be measured; neither tau* nor the fit is applied.

  ORB_SUFFIX=p15 (default) reads the wheel-aligned scoring; residual_p15_nobag.csv
  keeps the first pass for the record.
"""
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "script"))
import vio_metrics as vm                                              # noqa: E402
import eval_real_gt as E                                              # noqa: E402

OUT = os.path.join(ROOT, "output/compare/real/exp1-clock-residual")

DSS = ["dataset_realv7_000", "dataset_realv7_001", "dataset_realv7_002"]
CONF = {"lc_gpu": "GPU, LC on", "nolc_gpu": "GPU, LC off", "lc_cpu": "CPU, LC on", "nolc_cpu": "CPU, LC off"}
VINS_RUN = "rate1_002"
SUFFIX = os.environ.get("ORB_SUFFIX", "p15")   # p15 = scored without the bag (estimator-speed offset); p15w = with the wheel topic
SCAN = np.arange(-2.0, 6.01, 0.05)


def shifted(t, g, tau):
    return np.column_stack([np.interp(t + tau, t, g[:, 0]), np.interp(t + tau, t, g[:, 1])])


def along_mean(t, g, e, tau):
    gs = shifted(t, g, tau); d = np.gradient(gs, axis=0); n = np.linalg.norm(d, axis=1); k = n > 1e-6
    err = (e - gs)[k]; return float(np.mean(err[:, 0] * d[k, 0] / n[k] + err[:, 1] * d[k, 1] / n[k]))


def ate(t, g, e, tau):
    gs = shifted(t, g, tau); return float(np.sqrt(np.mean(np.sum((e - gs) ** 2, axis=1))))


def residual(t, g, e):
    taus = np.arange(-2.0, 2.01, 0.25)
    b = np.polyfit(taus, [along_mean(t, g, e, x) for x in taus], 1)
    A = np.array([ate(t, g, e, x) for x in SCAN]); k = int(np.argmin(A))
    return dict(tau_am=float(-b[1] / b[0]), slope=float(b[0]), tau_ate=float(SCAN[k]),
                ate0=ate(t, g, e, 0.0), ate_min=float(A[k]))


def estimator_series():
    """(ds, run label, system, t, gt_xy, est_xy, scorer offset) for every scored estimator run."""
    out = []
    for ds in DSS:
        for key, lbl in list(CONF.items()) + [("vins", "VINS-Fusion")]:
            sub = f"{ds[8:]}_{key}_{SUFFIX}" if key != "vins" else f"{ds[8:]}_vins_{VINS_RUN}"
            al = os.path.join(ROOT, "output/compare/real", sub, ds, f"aligned_{'orb' if key != 'vins' else 'vins'}.csv")
            mf = os.path.join(ROOT, "output/compare/real", sub, "real_gt_metrics.csv")
            if not os.path.exists(al):
                continue
            a = np.loadtxt(al, delimiter=",", comments="#")
            m = list(csv.DictReader(open(mf)))[0] if os.path.exists(mf) else {}
            out.append(dict(ds=ds, run=lbl, system="VINS" if key == "vins" else "ORB", t=a[:, 0] / 1e9, g=a[:, 1:3], e=a[:, 3:5],
                            offset_s=float(m.get("time_offset_s", "nan")), source=m.get("offset_source", "")[:20]))
    return out


def estimator_rows():
    rows = []
    for ds in DSS:
        for key, lbl in list(CONF.items()) + [("vins", "VINS-Fusion")]:
            sub = f"{ds[8:]}_{key}_{SUFFIX}" if key != "vins" else f"{ds[8:]}_vins_{VINS_RUN}"
            al = os.path.join(ROOT, "output/compare/real", sub, ds, f"aligned_{'orb' if key != 'vins' else 'vins'}.csv")
            mf = os.path.join(ROOT, "output/compare/real", sub, "real_gt_metrics.csv")
            if not os.path.exists(al):
                continue
            a = np.loadtxt(al, delimiter=",", comments="#"); t, g, e = a[:, 0] / 1e9, a[:, 1:3], a[:, 3:5]
            m = list(csv.DictReader(open(mf)))[0] if os.path.exists(mf) else {}
            r = residual(t, g, e)
            rows.append(dict(ds=ds[8:], run=lbl, system="VINS" if key == "vins" else "ORB",
                             offset_s=float(m.get("time_offset_s", "nan")), source=m.get("offset_source", "")[:20], **r))
    return rows


def baseline_series():
    """(dataset, t, gt_xy, aligned_est_xy) for the wheel baseline on each recording,
    rebuilt from wheel.csv with the baseline's own convention (gt_abs = bag_t0 + t_rel +
    offset) so a self-check against the stored ATE is possible."""
    man = json.load(open(os.path.join(ROOT, "output/compare/real/exp1-wheel/manifest.json")))
    out = []
    for row in man["rows"]:
        ds = row["dataset"]
        wf = os.path.join(ROOT, "output/output_wheel/real", ds, "gyro_wheel", "wheel.csv")
        if ds not in DSS or not os.path.exists(wf):
            continue
        w = np.loadtxt(wf, delimiter=","); tw = w[:, 0] / 1e9; pw = w[:, 1:3]
        gt = E.load_gt(os.path.join(ROOT, "dataset", "gt", ds)); tg = np.asarray(gt["t_rel"]); gxy = np.asarray(gt["xy"])
        tq = (tw - tw[0]) - row["offset_s"]
        k = (tq >= tg[0]) & (tq <= tg[-1])
        g = np.column_stack([np.interp(tq[k], tg, gxy[:, 0]), np.interp(tq[k], tg, gxy[:, 1])])
        src = np.c_[pw[k], np.zeros(k.sum())]; dst = np.c_[g, np.zeros(len(g))]
        R, tv = vm.umeyama(src, dst)[:2]; e = (src @ R.T + tv)[:, :2]
        out.append(dict(ds=ds, t=tw[k], g=g, e=e, offset_s=row["offset_s"], stored_ate=row["ate_rmse"]))
    return out


def baseline_rows():
    rows = []
    for b in baseline_series():
        r = residual(b["t"], b["g"], b["e"])
        rows.append(dict(ds=b["ds"][8:], run="wheel+gyro", system="baseline", offset_s=b["offset_s"],
                         source="wheel correlation", stored_ate=b["stored_ate"], **r))
    return rows


def main():
    rows = estimator_rows() + baseline_rows()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "residual_%s.csv" % SUFFIX), "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r})); wtr.writeheader(); wtr.writerows(rows)
    print("%-11s %-12s %-9s %8s %8s %8s %7s %7s  %s" % ("recording", "run", "system", "offset", "tau_am", "tau_ate", "ATE@0", "ATE@min", "source"))
    for r in rows:
        print("%-11s %-12s %-9s %+8.3f %+8.2f %+8.2f %7.3f %7.3f  %s%s" % (
            r["ds"], r["run"], r["system"], r["offset_s"], r["tau_am"], r["tau_ate"], r["ate0"], r["ate_min"], r["source"],
            ("  [self-check: stored %.3f]" % r["stored_ate"]) if "stored_ate" in r else ""))
    print("\nper recording, residual tau_ate over the five estimator runs, and the baseline's:")
    for ds in DSS:
        est = [r["tau_ate"] for r in rows if r["ds"] == ds[8:] and r["system"] != "baseline"]
        base = [r["tau_ate"] for r in rows if r["ds"] == ds[8:] and r["system"] == "baseline"]
        print("  %-11s estimators %+.2f .. %+.2f s   baseline %+.2f s" % (ds[8:], min(est), max(est), base[0] if base else float("nan")))
    print("\nwho leads, as scored vs re-aligned (best ORB-SLAM3 ATE against the baseline):")
    for ds in DSS:
        orb = [r for r in rows if r["ds"] == ds[8:] and r["system"] == "ORB"]
        b = [r for r in rows if r["ds"] == ds[8:] and r["system"] == "baseline"][0]
        print("  %-11s baseline %.3f | ORB best as scored %.3f -> re-aligned %.3f" % (
            ds[8:], b["ate0"], min(r["ate0"] for r in orb), min(r["ate_min"] for r in orb)))
    print("\nwrote", os.path.join(OUT, "residual_%s.csv" % SUFFIX))


if __name__ == "__main__":
    main()
