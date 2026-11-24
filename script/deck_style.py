"""Shared look for the update_181126 deck figures (script/deck_r*.py).

Every deck figure is drawn at the size it is shown: the left column of a
\\findingdata slide is about 2.55 in wide and 2.3 in tall, so 6 pt here is 6 pt
on the slide.  Import, call setup(), draw, then save(fig, name).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "docs/report/insights/update_181126/fig")
W, H = 2.55, 2.3

TH = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#dedcd4",
          bad="#c0392b", good="#1e8449", grey="#b9b5a8", still="#d9d6cc")
SIM, ROBOT = "#8e44ad", "#1f7a8c"          # simulation / real robot
ORB, VINS = "#c2185b", "#d35400"           # the deck's front-end colours
FP = "#2e86c1"                              # a map after footprint clearing


def setup():
    plt.rcParams.update({"font.size": 6.0, "font.family": "DejaVu Sans",
                         "axes.titlesize": 6.0, "legend.fontsize": 4.9})


def style(ax, grid="both"):
    ax.set_facecolor(TH["surface"])
    for s in ax.spines.values():
        s.set_color(TH["grid"])
    ax.tick_params(colors=TH["ink2"], length=2, pad=1.5)
    if grid:
        ax.grid(axis=grid if grid != "both" else "both", color=TH["grid"], lw=0.4)
    ax.set_axisbelow(True)


def title(ax, text):
    ax.set_title(text, loc="left", fontsize=6.0, color=TH["ink"], pad=3)


def label(ax, x=None, y=None):
    if x:
        ax.set_xlabel(x, color=TH["ink2"], labelpad=1.0)
    if y:
        ax.set_ylabel(y, color=TH["ink2"], labelpad=1.0)


def legend(ax, handles, **kw):
    opts = dict(frameon=True, fancybox=False, framealpha=1.0, edgecolor=TH["grid"],
                handlelength=1.1, borderpad=0.3, labelspacing=0.25, labelcolor=TH["ink2"])
    opts.update(kw)
    return ax.legend(handles=handles, **opts)


def figure(nrows=2, ncols=1, **kw):
    setup()
    fig, axes = plt.subplots(nrows, ncols, figsize=kw.pop("figsize", (W, H)), **kw)
    fig.patch.set_facecolor(TH["surface"])
    return fig, axes


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, name)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=TH["surface"])
    print("wrote", out)
