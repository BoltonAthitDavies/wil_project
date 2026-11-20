#!/usr/bin/env python3
"""Figures for the real-rig local-error analysis, drawn against the simulation.

    python3 script/exp1_real_vs_sim_figs.py

Reads the per-window tables both analyses already wrote
(output/compare/{simulation,real}/exp1-local-error/windows.csv) and the aligned
series via exp1_real_clock_residual, and writes to
output/compare/real/exp1-local-error/:

  fig_simreal_drivers.png   why the simulation's mechanisms do not carry over:
                            (a) VINS tracked features, (b) ORB tracked map points,
                            (c) robot speed, (d) RPE per metre travelled by tracked-point bin
  fig_clock_residual.png    ATE as a function of a time shift of the truth, per run and
                            recording, with the wheel baseline: the residual clock offset
  fig_turn_clock.png        turning-minus-straight RPE per run at the scorer's offset and
                            at the ATE-minimum offset: how much of the turning effect is clock
  fig_along_cross.png       along- and across-track rms per run and recording

Nothing here is written into the report; it is material for a decision.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "script"))
import exp1_real_clock_residual as C                                  # noqa: E402

OUT = os.path.join(ROOT, "output/compare/real/exp1-local-error")
SIM = pd.read_csv(os.path.join(ROOT, "output/compare/simulation/exp1-local-error/windows.csv"))
REAL = pd.read_csv(os.path.join(OUT, "windows.csv"))
REAL = REAL[REAL.valid.astype(str) == "True"]
for d in (SIM, REAL):
    d["sys"] = np.where(d.fam.str.contains("VINS"), "VINS", "ORB")
TH = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#dedcd4", sim="#4c72b0", real="#c44e52",
          orb=["#4c72b0", "#8fb0d9", "#2a4d7a", "#6b8cc0"], vins="#c44e52", base="#2d8a5e")
TURN_DEG, MIN_SPEED, INIT_S = 10.0, 0.15, 5.0
RECS = ["dataset_realv7_000", "dataset_realv7_001", "dataset_realv7_002"]


def style(ax, title):
    ax.set_facecolor(TH["surface"]); ax.grid(True, color=TH["grid"], lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, loc="left", fontsize=9.5, color=TH["ink"])
    ax.tick_params(labelsize=8, colors=TH["ink2"])


def ecdf(ax, x, **kw):
    x = np.sort(np.asarray(x)[~np.isnan(x)]); ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", **kw)


# ---------------------------------------------------------------- figure 1
def fig_drivers():
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.2)); fig.patch.set_facecolor("white")
    a, b, c, d = ax.ravel()
    sv, rv = SIM[SIM.sys == "VINS"], REAL[REAL.sys == "VINS"]
    so, ro = SIM[SIM.sys == "ORB"], REAL[REAL.sys == "ORB"]
    ecdf(a, sv.inliers, color=TH["sim"], lw=1.8, label="simulation (realistic_000-003)")
    ecdf(a, rv.inliers, color=TH["real"], lw=1.8, label="real rig (realv7_000-002)")
    a.axvline(120, color=TH["ink2"], ls=":", lw=1); a.text(121, 0.05, "120 = starvation line", fontsize=7.5, color=TH["ink2"])
    a.set_xlim(60, 155); a.set_xlabel("VINS-Fusion tracked features at window start (ceiling 150)", fontsize=8.5)
    a.set_ylabel("fraction of 1-s windows", fontsize=8.5); a.legend(fontsize=7.5, loc="upper left", frameon=False)
    style(a, "(a) VINS-Fusion: %.0f%% of sim windows below 120 features, %.0f%% real"
          % (100 * (sv.inliers <= 120).mean(), 100 * (rv.inliers <= 120).mean()))
    ecdf(b, so.inliers, color=TH["sim"], lw=1.8, label="simulation, 1500 extracted")
    ecdf(b, ro.inliers, color=TH["real"], lw=1.8, label="real rig, 2000 per camera (node counts both)")
    b.axvline(250, color=TH["ink2"], ls=":", lw=1); b.text(255, 0.05, "250", fontsize=7.5, color=TH["ink2"])
    b.set_xlim(100, 1300); b.set_xlabel("ORB-SLAM3 tracked map points at window start", fontsize=8.5)
    b.set_ylabel("fraction of 1-s windows", fontsize=8.5); b.legend(fontsize=7.5, loc="lower right", frameon=False)
    style(b, "(b) ORB-SLAM3: median %d tracked points in sim, %d real" % (so.inliers.median(), ro.inliers.median()))
    ecdf(c, SIM.speed, color=TH["sim"], lw=1.8, label="simulation"); ecdf(c, REAL.speed, color=TH["real"], lw=1.8, label="real rig")
    c.set_xlim(0, 1.7); c.set_xlabel("ground-truth speed over the window [m/s]", fontsize=8.5); c.set_ylabel("fraction of 1-s windows", fontsize=8.5)
    c.legend(fontsize=7.5, loc="lower right", frameon=False)
    style(c, "(c) the robot: %.2f m/s in simulation, %.2f m/s real (medians)" % (SIM.speed.median(), REAL.speed.median()))
    bins = [0, 250, 300, 400, 500, 600, 800, 1200]
    for df, col, lbl in ((so, TH["sim"], "simulation"), (ro, TH["real"], "real rig")):
        m = df[df.speed > MIN_SPEED].copy(); m["pm"] = m.rpe / m.speed; m["b"] = pd.cut(m.inliers, bins)
        g = m.groupby("b", observed=True).pm.agg(["median", lambda x: np.percentile(x, 90), "size"])
        x = [iv.mid for iv in g.index]
        d.plot(x, g["median"], "o-", color=col, lw=1.6, ms=4, label=lbl + " (median)")
        d.plot(x, g.iloc[:, 1], "o--", color=col, lw=1, ms=3, alpha=0.7, label=lbl + " (p90)")
        for xi, mi, n in zip(x, g["median"], g["size"]):
            d.text(xi, mi + (0.02 if lbl == "real rig" else -0.03), "n=%d" % n, fontsize=6, color=col, ha="center", va="bottom", alpha=0.9)
    d.set_xlabel("ORB-SLAM3 tracked map points (bin centre; counts beneath)", fontsize=8.5)
    d.set_ylabel("1-s RPE per metre travelled [m/m]", fontsize=8.5); d.legend(fontsize=7, frameon=False, ncol=2); d.set_ylim(-0.05, None)
    style(d, "(d) per metre travelled, the real rig drifts %.0fx faster"
          % ((ro[ro.speed > MIN_SPEED].rpe / ro[ro.speed > MIN_SPEED].speed).median() / (so[so.speed > MIN_SPEED].rpe / so[so.speed > MIN_SPEED].speed).median()))
    fig.suptitle("Why the simulation's local-error drivers do not transfer to the real rig", fontsize=11, color=TH["ink"], x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(os.path.join(OUT, "fig_simreal_drivers.png"), dpi=170); plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig_clock(est, base):
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.9), sharey=False); fig.patch.set_facecolor("white")
    taus = np.arange(-2.0, 6.01, 0.1)
    for ax, ds in zip(axs, RECS):
        runs = [r for r in est if r["ds"] == ds]; k = 0
        for r in runs:
            A = [C.ate(r["t"], r["g"], r["e"], x) for x in taus]
            col = TH["vins"] if r["system"] == "VINS" else TH["orb"][k % 4]; k += r["system"] == "ORB"
            ax.plot(taus, A, color=col, lw=1.5, label=r["run"]); i = int(np.argmin(A))
            ax.plot(taus[i], A[i], "o", color=col, ms=4); ax.plot(0, A[int(np.argmin(np.abs(taus)))], "s", color=col, ms=4, mfc="white")
        for bb in [b for b in base if b["ds"] == ds]:
            A = [C.ate(bb["t"], bb["g"], bb["e"], x) for x in taus]
            ax.plot(taus, A, color=TH["base"], lw=1.8, ls="-", label="wheel+gyro baseline"); i = int(np.argmin(A)); ax.plot(taus[i], A[i], "o", color=TH["base"], ms=4)
        ax.axvline(0, color=TH["ink2"], lw=0.8, ls=":")
        ax.set_xlabel("shift of the truth relative to the scorer's alignment [s]", fontsize=8.5); ax.set_ylim(0, None)
        style(ax, ds[8:] + ("   (two ORB offsets at the -5 s search bound)" if ds.endswith("001") else ""))
    axs[0].set_ylabel("planar ATE after SE(2) alignment [m]", fontsize=8.5)
    axs[1].legend(fontsize=7, frameon=False, ncol=2, loc="upper right")
    fig.text(0.02, 0.005, "square = as scored (shift 0); dot = minimum. The baseline's minimum sits at 0 on every recording with the same wheel-derived offset; the estimators' do not, "
             "and all five of a recording's runs move together: not a clock error but a lead along the path.", fontsize=7.5, color=TH["ink2"])
    fig.suptitle("Residual along-track lead of the estimators (time shift that would absorb it)", fontsize=11, color=TH["ink"], x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95)); fig.savefig(os.path.join(OUT, "fig_clock_residual.png"), dpi=170); plt.close(fig)


# ---------------------------------------------------------------- figure 3
def turn_effect(t, g, e, tau):
    """median 1-s translational RPE, turning (>10 deg/s) minus straight, on the aligned series
    with the truth shifted by tau. Windows start after INIT_S and need speed > MIN_SPEED."""
    gs = C.shifted(t, g, tau); t0 = t[0] + INIT_S
    tq = np.arange(t0, t[-1] - 1.0, 0.25)
    G0 = np.column_stack([np.interp(tq, t, gs[:, i]) for i in range(2)]); G1 = np.column_stack([np.interp(tq + 1, t, gs[:, i]) for i in range(2)])
    E0 = np.column_stack([np.interp(tq, t, e[:, i]) for i in range(2)]); E1 = np.column_stack([np.interp(tq + 1, t, e[:, i]) for i in range(2)])
    dg, de = G1 - G0, E1 - E0; rpe = np.linalg.norm(de - dg, axis=1); speed = np.linalg.norm(dg, axis=1)
    Gm = np.column_stack([np.interp(tq + 0.5, t, gs[:, i]) for i in range(2)])
    h0 = np.arctan2(*(Gm - G0)[:, ::-1].T); h1 = np.arctan2(*(G1 - Gm)[:, ::-1].T)
    turn = np.degrees(np.abs((h1 - h0 + np.pi) % (2 * np.pi) - np.pi)) / 0.5
    k = speed > MIN_SPEED; tr = k & (turn > TURN_DEG); st = k & (turn <= TURN_DEG)
    return float(np.median(rpe[tr]) - np.median(rpe[st])), int(tr.sum()), int(st.sum())


def fig_turn(est, res):
    orb = [r for r in est if r["system"] == "ORB"]
    rows = []
    for r in orb:
        tau = float(res[(res.ds == r["ds"][8:]) & (res.run == r["run"])].tau_ate.iloc[0])
        d0, n1, n2 = turn_effect(r["t"], r["g"], r["e"], 0.0); d1, _, _ = turn_effect(r["t"], r["g"], r["e"], tau)
        rows.append(dict(ds=r["ds"][8:], run=r["run"], tau=tau, d0=d0, d1=d1, n_turn=n1, n_straight=n2))
    T = pd.DataFrame(rows); T.to_csv(os.path.join(OUT, "turn_effect_vs_clock.csv"), index=False)
    fig, ax = plt.subplots(figsize=(8.5, 4.6)); fig.patch.set_facecolor("white")
    y = np.arange(len(T))[::-1]
    for yi, (_, r) in zip(y, T.iterrows()):
        ax.plot([r.d0, r.d1], [yi, yi], color=TH["grid"], lw=3, zorder=1)
        ax.plot(r.d0, yi, "s", color=TH["ink2"], ms=6, zorder=2, mfc="white"); ax.plot(r.d1, yi, "o", color=TH["real"], ms=6, zorder=3)
        ax.text(max(r.d0, r.d1) + 0.004, yi, "tau* %+.2f s" % r.tau, fontsize=7, va="center", color=TH["ink2"])
    ax.set_yticks(y); ax.set_yticklabels([f"{r.ds}  {r.run}" for _, r in T.iterrows()], fontsize=8)
    ax.axvline(0, color=TH["ink2"], lw=0.8); ax.set_xlabel("median 1-s RPE, turning (>10 deg/s) minus straight [m]", fontsize=8.5)
    ax.plot([], [], "s", color=TH["ink2"], mfc="white", label="at the scorer's offset: median %+.3f m, %d/%d positive" % (T.d0.median(), (T.d0 > 0).sum(), len(T)))
    ax.plot([], [], "o", color=TH["real"], label="at the ATE-minimum offset tau*: median %+.3f m, %d/%d positive" % (T.d1.median(), (T.d1 > 0).sum(), len(T)))
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    style(ax, "Turning raises ORB-SLAM3's local error on every run, but most of the measured size is the along-track lead, not turning")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_turn_clock.png"), dpi=170); plt.close(fig)
    return T


# ---------------------------------------------------------------- figure 4
def fig_along_cross(est, res):
    rows = []
    for r in est:
        for tag, tau in (("as scored", 0.0), ("at tau*", float(res[(res.ds == r["ds"][8:]) & (res.run == r["run"])].tau_ate.iloc[0]))):
            gs = C.shifted(r["t"], r["g"], tau); dd = np.gradient(gs, axis=0); n = np.linalg.norm(dd, axis=1); k = n > 1e-6
            u = dd[k] / n[k, None]; err = (r["e"] - gs)[k]; al = np.sum(err * u, axis=1); cr = err[:, 0] * -u[:, 1] + err[:, 1] * u[:, 0]
            rows.append(dict(ds=r["ds"][8:], run=r["run"], system=r["system"], tag=tag, along=np.sqrt(np.mean(al ** 2)), cross=np.sqrt(np.mean(cr ** 2))))
    A = pd.DataFrame(rows); A["cross_share"] = A.cross ** 2 / (A.along ** 2 + A.cross ** 2); A.to_csv(os.path.join(OUT, "along_cross_vs_clock.csv"), index=False)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True); fig.patch.set_facecolor("white")
    for ax, tag in zip(axs, ("as scored", "at tau*")):
        S = A[A.tag == tag]; x = 0; ticks, labels = [], []
        for ds in [d[8:] for d in RECS]:
            for _, r in S[S.ds == ds].iterrows():
                col = TH["vins"] if r.system == "VINS" else TH["orb"][0]
                ax.bar(x - 0.2, r.along, 0.38, color=col, alpha=0.95); ax.bar(x + 0.2, r.cross, 0.38, color=col, alpha=0.45, hatch="//", edgecolor=col)
                ticks.append(x); labels.append(r.run.replace("ORB ", "").replace("-Fusion", "")); x += 1
            ax.axvline(x - 0.5, color=TH["grid"], lw=1); ax.text(x - 3, ax.get_ylim()[1] * 0.0 + 0.95, ds, fontsize=8, color=TH["ink2"], ha="center")
            x += 0.6
        ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=60, fontsize=7, ha="right")
        style(ax, "error rms along (solid) and across (hatched) the direction of travel, " + tag)
    axs[0].set_ylabel("rms [m]", fontsize=8.5); axs[0].set_ylim(0, 1.3)
    fig.text(0.02, 0.005, "realv7_001's error is along-track as scored and is mostly gone at tau*: a constant along-track lead, which the wheel baseline does not show, is what an unmeasured lever arm looks like.",
             fontsize=7.5, color=TH["ink2"])
    fig.suptitle("Which way the real-rig error points, before and after removing the along-track lead", fontsize=11, color=TH["ink"], x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95)); fig.savefig(os.path.join(OUT, "fig_along_cross.png"), dpi=170); plt.close(fig)
    return A


def main():
    os.makedirs(OUT, exist_ok=True)
    res = pd.read_csv(os.path.join(ROOT, "output/compare/real/exp1-clock-residual/residual_p15.csv"))
    est = C.estimator_series(); base = C.baseline_series()
    fig_drivers(); fig_clock(est, base); T = fig_turn(est, res); A = fig_along_cross(est, res)
    print(T.round(3).to_string(index=False)); print()
    print(A.pivot_table(index=["ds", "run"], columns="tag", values=["along", "cross", "cross_share"]).round(3).to_string())
    print("\nwrote 4 figures to", OUT)


if __name__ == "__main__":
    main()
