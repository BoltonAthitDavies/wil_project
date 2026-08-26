#!/usr/bin/env python3
"""Compare VINS-Fusion vio.csv against /ground_truth/odometry from the sim bag.

Both logs carry sim-clock header stamps, so they need no time alignment.
Ground truth lives in the `odom` world frame; VIO lives in its own start-aligned
frame, so GT is rigidly mapped into the VIO frame (yaw + translation) using the
first overlapping pose. That keeps accumulated drift visible instead of
least-squares-fitting it away.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# One run directory holding vio.csv + ground_truth.csv; figures are written beside them
RUN = sys.argv[1] if len(sys.argv) > 1 else "output/simulation/sim"
RUN = os.path.abspath(RUN)
VIO = os.path.join(RUN, "vio.csv")
GT = os.path.join(RUN, "ground_truth.csv")
OUT = os.path.join(RUN, "vio_vs_gt")
LABEL = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(RUN)

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", ink3="#8a8983",
                  grid="#e3e2dd", series=("#2a78d6", "#eb6834", "#1baf7a")),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", ink3="#87867d",
                  grid="#33332f", series=("#3987e5", "#d95926", "#199e70")),
}


def yaw_of(q):
    """ZYX yaw from (w,x,y,z) rows, unwrapped."""
    w, x, y, z = q.T
    return np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2)))


def slerp(t_src, q_src, t_dst):
    """Piecewise SLERP of (w,x,y,z) rows onto t_dst."""
    i = np.clip(np.searchsorted(t_src, t_dst) - 1, 0, len(t_src) - 2)
    q0, q1 = q_src[i], q_src[i + 1]
    q1 = np.where((np.sum(q0 * q1, axis=1) < 0)[:, None], -q1, q1)  # shortest arc
    u = ((t_dst - t_src[i]) / (t_src[i + 1] - t_src[i]))[:, None]
    d = np.clip(np.sum(q0 * q1, axis=1), -1, 1)[:, None]
    th = np.arccos(d)
    lin = th < 1e-6
    s0 = np.where(lin, 1 - u, np.sin((1 - u) * th) / np.where(lin, 1, np.sin(th)))
    s1 = np.where(lin, u, np.sin(u * th) / np.where(lin, 1, np.sin(th)))
    q = s0 * q0 + s1 * q1
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# --- load & time-align -------------------------------------------------------
v = np.loadtxt(VIO, delimiter=",")
g = np.loadtxt(GT, delimiter=",")
tv, tg = v[:, 0] / 1e9, g[:, 0] / 1e9

keep = (tv >= tg[0]) & (tv <= tg[-1])          # only where GT can be interpolated
v, tv = v[keep], tv[keep]
t = tv - tv[0]

p_v, q_v, vel_v = v[:, 1:4], v[:, 4:8], v[:, 8:11]
p_g = np.stack([np.interp(tv, tg, g[:, i]) for i in (1, 2, 3)], axis=1)
q_g = slerp(tg, g[:, 4:8], tv)
# GT twist is body-frame (vy≈0, |vx| == |d/dt p|); VIO velocity is world-frame.
# Compare speed magnitude, which is frame-invariant.
sp_g = np.abs(np.interp(tv, tg, g[:, 8]))
sp_v = np.linalg.norm(vel_v, axis=1)

# --- rigid alignment: GT -> VIO frame, anchored on the first overlapping pose --
yaw_v, yaw_g = yaw_of(q_v), yaw_of(q_g)
R = rot_z(yaw_v[0] - yaw_g[0])
p_ga = (R @ (p_g - p_g[0]).T).T + p_v[0]
yaw_ga = yaw_g + (yaw_v[0] - yaw_g[0])

err = p_v - p_ga
err_n = np.linalg.norm(err, axis=1)
yaw_err = np.degrees(np.arctan2(np.sin(yaw_v - yaw_ga), np.cos(yaw_v - yaw_ga)))

dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p_ga, axis=0), axis=1))])
len_v = np.linalg.norm(np.diff(p_v, axis=0), axis=1).sum()

# --- ATE with a full SE(3) Umeyama fit (rotation+translation, scale fixed) ----
A, B = p_ga - p_ga.mean(0), p_v - p_v.mean(0)
U, _, Vt = np.linalg.svd(B.T @ A)          # H = B^T A, rotation R = V D U^T
D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
R_ate = Vt.T @ D @ U.T
ate = np.linalg.norm(A - (R_ate @ B.T).T, axis=1)
ate_rmse = np.sqrt((ate**2).mean())

def fmt_len(m):
    """Distance in <=7 chars, unit chosen by magnitude."""
    if abs(m) >= 1000:
        return f"{m/1000:.2f} km"
    if abs(m) >= 10:
        return f"{m:.1f} m"
    if abs(m) >= 1:
        return f"{m:.2f} m"
    return f"{m*100:.0f} cm"


STATS = [
    ("ATE RMSE",      fmt_len(ate_rmse)),
    ("mean error",    fmt_len(err_n.mean())),
    ("final drift",   fmt_len(err_n[-1])),
    ("drift / dist",  f"{err_n[-1]/dist[-1]*100:.2f} %"),
    ("path len err",  f"{len_v-dist[-1]:+.1f} m"),
    ("max yaw err",   f"{np.abs(yaw_err).max():.1f}°"),
]


def style_axes(ax, th):
    ax.set_facecolor(th["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(th["grid"]); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=th["ink2"], labelsize=8, length=3, width=0.8)
    ax.grid(True, color=th["grid"], linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)


def panel(ax, th, title, ylabel, series, xs=None):
    style_axes(ax, th)
    for label, y, c, *lw in series:
        ax.plot(t if xs is None else xs, y, color=c, linewidth=lw[0] if lw else 1.7,
                solid_capstyle="round", label=label)
    ax.set_title(title, color=th["ink"], fontsize=11, fontweight="700", loc="left", pad=9)
    ax.set_ylabel(ylabel, color=th["ink2"], fontsize=8.5)
    leg = ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=len(series),
                    frameon=False, fontsize=8.5, handlelength=1.4, handletextpad=0.5,
                    columnspacing=1.5, borderpad=0)
    for txt in leg.get_texts():
        txt.set_color(th["ink2"])
    ax.set_xlim(t[0], t[-1])


def render(mode):
    th = THEMES[mode]
    c1, c2, c3 = th["series"]
    fig = plt.figure(figsize=(15, 9.0), facecolor=th["surface"])
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.35, 1],
                  left=0.05, right=0.965, top=0.855, bottom=0.075,
                  wspace=0.22, hspace=0.30)

    head = f"VIO vs. simulator ground truth — {LABEL}"
    fig.text(0.05, 0.955, head, color=th["ink"], fontweight="700", va="top",
             fontsize=15 if len(head) <= 46 else 15 * 46 / len(head))
    fig.text(0.05, 0.915, f"{len(t)} poses · {t[-1]:.1f} s · GT mapped into the VIO "
             "frame on the first pose, not least-squares fitted",
             color=th["ink3"], fontsize=9, va="top")
    for i, (k, val) in enumerate(STATS):
        x = 0.550 + i * 0.070
        fig.text(x, 0.955, val, color=th["ink"], fontsize=13.5, fontweight="700", va="top")
        fig.text(x, 0.917, k, color=th["ink3"], fontsize=8, va="top")

    # --- trajectory overlay --------------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    style_axes(ax, th)
    ax.plot(p_ga[:, 0], p_ga[:, 1], color=c2, linewidth=2.4,
            solid_capstyle="round", label="ground truth", zorder=2)
    ax.plot(p_v[:, 0], p_v[:, 1], color=c1, linewidth=2.0,
            solid_capstyle="round", label="VIO estimate", zorder=3)
    ax.plot(p_v[0, 0], p_v[0, 1], "o", ms=9, mfc=th["surface"], mec=th["ink"],
            mew=1.8, zorder=5)
    ax.annotate("shared start", p_v[0, :2], textcoords="offset points",
                xytext=(10, 9), color=th["ink2"], fontsize=8.5)
    for pt, c, lab in ((p_ga[-1], c2, "GT end"), (p_v[-1], c1, "VIO end")):
        ax.plot(pt[0], pt[1], "o", ms=8, mfc=c, mec=th["surface"], mew=1.8, zorder=5)
    ax.annotate(f"final drift {fmt_len(err_n[-1])}",
                (0.5 * (p_v[-1, 0] + p_ga[-1, 0]), 0.5 * (p_v[-1, 1] + p_ga[-1, 1])),
                textcoords="offset points", xytext=(12, -16), color=th["ink2"], fontsize=8.5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.08)
    ax.set_title("Trajectory overlay, top-down (x–y)", color=th["ink"], fontsize=11.5,
                 fontweight="700", loc="left", pad=10)
    ax.set_xlabel("x  [m]", color=th["ink2"], fontsize=8.5)
    ax.set_ylabel("y  [m]", color=th["ink2"], fontsize=8.5)
    leg = ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2, frameon=False,
                    fontsize=9, handlelength=1.4, handletextpad=0.5, columnspacing=1.5,
                    borderpad=0)
    for txt in leg.get_texts():
        txt.set_color(th["ink2"])

    panel(fig.add_subplot(gs[1, 0]), th, "Position error  (VIO − GT)", "[m]",
          [("x", err[:, 0], c1), ("y", err[:, 1], c2), ("z", err[:, 2], c3)])
    panel(fig.add_subplot(gs[1, 1]), th, "Heading", "[deg]",
          [("ground truth", np.degrees(yaw_ga), c2, 2.6), ("VIO", np.degrees(yaw_v), c1, 1.6)])
    panel(fig.add_subplot(gs[1, 2]), th, "Speed", "[m/s]",
          [("ground truth", sp_g, c2, 2.6), ("VIO", sp_v, c1, 1.6)])
    for a in fig.axes[1:]:
        a.set_xlabel("time since first compared pose  [s]", color=th["ink2"], fontsize=8.5)

    out = f"{OUT}_{mode}.png"
    fig.savefig(out, dpi=160, facecolor=th["surface"])
    plt.close(fig)
    return out


paths = [render(m) for m in ("light", "dark")]

print(f"compared {len(t)} poses over {t[-1]:.2f} s "
      f"({dist[-1]:.2f} m ground-truth path)\n")
hdr = f"{'quantity':<22}{'min':>10}{'max':>10}{'mean':>10}{'rms':>10}{'final':>10}"
print(hdr)
for name, a in [("error x [m]", err[:, 0]), ("error y [m]", err[:, 1]),
                ("error z [m]", err[:, 2]), ("error norm [m]", err_n),
                ("ATE (SE3-fit) [m]", ate), ("yaw error [deg]", yaw_err),
                ("speed err [m/s]", sp_v - sp_g)]:
    print(f"{name:<22}" + "".join(f"{x:>10.3f}" for x in
          (a.min(), a.max(), a.mean(), np.sqrt((a**2).mean()), a[-1])))
print("\nwrote:", *paths, sep="\n  ")
