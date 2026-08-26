#!/usr/bin/env python3
"""Plot a VINS-Fusion vio.csv trajectory log.

Columns (no header): t_ns, px, py, pz, qw, qx, qy, qz, vx, vy, vz
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

CSV = sys.argv[1] if len(sys.argv) > 1 else "output/simulation/sim/vio.csv"
# default: write beside the csv; the run directory's name labels the figure
_DIR = os.path.dirname(os.path.abspath(CSV))
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_DIR, "vio")
LABEL = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(_DIR)

# --- palette -----------------------------------------------------------------
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", ink3="#8a8983",
                  grid="#e3e2dd", series=("#2a78d6", "#eb6834", "#1baf7a"),
                  seq=("#dce9f8", "#2a78d6", "#123253")),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", ink3="#87867d",
                  grid="#33332f", series=("#3987e5", "#d95926", "#199e70"),
                  seq=("#12314f", "#3987e5", "#bfd9f6")),
}

# --- load --------------------------------------------------------------------
d = np.loadtxt(CSV, delimiter=",")
t = (d[:, 0] - d[0, 0]) / 1e9
p, q, v = d[:, 1:4], d[:, 4:8], d[:, 8:11]
speed = np.linalg.norm(v, axis=1)
path_len = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()

w, qx, qy, qz = q.T
roll = np.degrees(np.arctan2(2 * (w * qx + qy * qz), 1 - 2 * (qx**2 + qy**2)))
pitch = np.degrees(np.arcsin(np.clip(2 * (w * qy - qz * qx), -1, 1)))
yaw = np.degrees(np.unwrap(np.arctan2(2 * (w * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))))

def fmt_len(m):
    """Distance in <=7 chars, unit chosen by magnitude."""
    if abs(m) >= 1000:
        return f"{m/1000:.2f} km"
    if abs(m) >= 10:
        return f"{m:.1f} m"
    if abs(m) >= 1:
        return f"{m:.2f} m"
    return f"{m*100:.0f} cm"


def fmt_spd(v):
    return f"{v:.0f} m/s" if abs(v) >= 100 else f"{v:.2f} m/s"


STATS = [
    ("duration",     f"{t[-1]:.1f} s"),
    ("path length",  fmt_len(path_len)),
    ("mean speed",   fmt_spd(speed.mean())),
    ("peak speed",   fmt_spd(speed.max())),
    ("net heading",  f"{yaw[-1] - yaw[0]:+.0f}°"),
    ("z excursion",  fmt_len(p[:, 2].ptp())),
]


def style_axes(ax, th):
    ax.set_facecolor(th["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(th["grid"])
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=th["ink2"], labelsize=8, length=3, width=0.8)
    ax.grid(True, color=th["grid"], linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)


def panel(ax, th, title, ylabel, series):
    """series: list of (label, values, color)"""
    style_axes(ax, th)
    for label, y, c in series:
        ax.plot(t, y, color=c, linewidth=1.7, solid_capstyle="round", label=label)
    ax.set_title(title, color=th["ink"], fontsize=11, fontweight="700",
                 loc="left", pad=9)
    ax.set_ylabel(ylabel, color=th["ink2"], fontsize=8.5)
    leg = ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=len(series),
                    frameon=False, fontsize=8.5, handlelength=1.4, handletextpad=0.5,
                    columnspacing=1.5, borderpad=0)
    for txt in leg.get_texts():
        txt.set_color(th["ink2"])
    ax.set_xlim(t[0], t[-1])


def render(mode):
    th = THEMES[mode]
    cmap = LinearSegmentedColormap.from_list("speed", th["seq"])
    fig = plt.figure(figsize=(15, 9.0), facecolor=th["surface"])
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.35, 1],
                  left=0.05, right=0.965, top=0.855, bottom=0.075,
                  wspace=0.20, hspace=0.30)

    # header
    head = f"Visual-inertial odometry — {LABEL}"
    fig.text(0.05, 0.955, head, color=th["ink"], fontweight="700", va="top",
             fontsize=15 if len(head) <= 46 else 15 * 46 / len(head))
    fig.text(0.05, 0.915, f"{len(t)} VIO poses · {1/np.diff(t).mean():.0f} Hz mean rate · "
             "no ground truth for this dataset — the estimate is shown unvalidated",
             color=th["ink3"], fontsize=9, va="top")
    for i, (k, val) in enumerate(STATS):
        x = 0.550 + i * 0.070
        fig.text(x, 0.955, val, color=th["ink"], fontsize=12.5, fontweight="700", va="top")
        fig.text(x, 0.917, k, color=th["ink3"], fontsize=8, va="top")

    # --- trajectory, colored by speed (sequential, single hue) ---------------
    ax = fig.add_subplot(gs[0, :])
    style_axes(ax, th)
    seg = np.stack([p[:-1, :2], p[1:, :2]], axis=1)
    lc = LineCollection(seg, cmap=cmap, linewidth=2.6, capstyle="round",
                        norm=plt.Normalize(0, speed.max()))
    lc.set_array(0.5 * (speed[:-1] + speed[1:]))
    ax.add_collection(lc)
    ax.plot(p[0, 0], p[0, 1], "o", ms=9, mfc=th["surface"], mec=th["ink"], mew=1.8, zorder=5)
    ax.plot(p[-1, 0], p[-1, 1], "o", ms=9, mfc=th["ink"], mec=th["surface"], mew=1.8, zorder=5)
    ax.annotate("start", p[0, :2], textcoords="offset points", xytext=(10, 8),
                color=th["ink2"], fontsize=8.5)
    ax.annotate("end", p[-1, :2], textcoords="offset points", xytext=(10, 8),
                color=th["ink2"], fontsize=8.5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.autoscale_view()
    ax.margins(0.09)
    ax.set_title("Trajectory, top-down (x–y)", color=th["ink"], fontsize=11.5,
                 fontweight="700", loc="left", pad=10)
    ax.set_xlabel("x  [m]", color=th["ink2"], fontsize=8.5)
    ax.set_ylabel("y  [m]", color=th["ink2"], fontsize=8.5)
    cb = fig.colorbar(lc, ax=ax, pad=0.012, fraction=0.020, aspect=24)
    cb.set_label("speed  [m/s]", color=th["ink2"], fontsize=8.5)
    cb.ax.tick_params(colors=th["ink2"], labelsize=8, length=3, width=0.8)
    cb.outline.set_visible(False)

    c1, c2, c3 = th["series"]
    for col, (title, ylab, series) in enumerate([
        ("Position", "[m]",
         [("x", p[:, 0], c1), ("y", p[:, 1], c2), ("z", p[:, 2], c3)]),
        ("Velocity", "[m/s]",
         [("vx", v[:, 0], c1), ("vy", v[:, 1], c2), ("vz", v[:, 2], c3)]),
        ("Attitude", "[deg]",
         [("yaw", yaw, c1), ("pitch", pitch, c2), ("roll", roll, c3)]),
    ]):
        axp = fig.add_subplot(gs[1, col])
        panel(axp, th, title, ylab, series)
        axp.set_xlabel("time since first pose  [s]", color=th["ink2"], fontsize=8.5)

    out = f"{OUT}_{mode}.png"
    fig.savefig(out, dpi=160, facecolor=th["surface"])
    plt.close(fig)
    return out


paths = [render(m) for m in ("light", "dark")]

# --- table view --------------------------------------------------------------
print(f"{len(t)} poses  |  {t[-1]:.2f} s  |  {path_len:.2f} m travelled\n")
hdr = ["quantity", "min", "max", "mean", "final"]
rows = [("x [m]", p[:, 0]), ("y [m]", p[:, 1]), ("z [m]", p[:, 2]),
        ("vx [m/s]", v[:, 0]), ("vy [m/s]", v[:, 1]), ("vz [m/s]", v[:, 2]),
        ("speed [m/s]", speed), ("roll [deg]", roll), ("pitch [deg]", pitch),
        ("yaw [deg]", yaw)]
print(f"{hdr[0]:<12}" + "".join(f"{h:>10}" for h in hdr[1:]))
for name, a in rows:
    print(f"{name:<12}" + "".join(f"{x:>10.3f}" for x in (a.min(), a.max(), a.mean(), a[-1])))
print("\nwrote:", *paths, sep="\n  ")
