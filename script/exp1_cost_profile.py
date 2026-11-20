#!/usr/bin/env python3
"""The two pipelines' resource profile: memory against processor time.

    python3 script/exp1_cost_profile.py

WHY THIS EXISTS
  The slot's conclusion about cost is that neither pipeline is cheaper, and
  that the choice is a memory-against-cores trade.  That claim is about the
  JOINT position of two columns of tab:mapping-resources, and a reader has to
  carry four numbers across the table to see it.  Plotted against each other
  the two families sit in opposite corners and the sentence needs no effort.

  Both axes are costs measured on the same runs, so this is a profile, not a
  correlation.  It is deliberately NOT accuracy against cost: that figure
  existed, showed nothing because its accuracy axis was dominated by two
  identified failure modes, and was removed.  Nothing here is plotted against
  an outcome.

WHAT IT CANNOT ESTABLISH
  Three repeats per configuration on one laptop, and the runs were not
  interleaved, so the error bars measure repeatability within a session and
  not the thermal state between sessions.  The caption carries that.

Output: output/compare/simulation/exp1-cost-profile/fig_cost_profile_light.png
"""

import collections
import csv
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np                                   # noqa: E402
import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "output/compare/simulation/exp3-rules/mapping_resources.csv")
OUT = os.path.join(ROOT, "output/compare/simulation/exp1-cost-profile")
TH = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#dedcd4")
COL = {"ORB-SLAM3": "#e040fb", "ORB-SLAM3 GPU": "#f48fff", "VINS-Fusion": "#cc8319"}
MARK = {"realistic_map": "o", "realistic_01_map": "s"}
LAY = {"realistic_map": "U layout", "realistic_01_map": "I layout"}


def main():
    g = collections.defaultdict(list)
    for r in csv.DictReader(open(SRC)):
        g[(r["dataset"].replace("dataset_allsensor_", ""), r["system"])].append(r)
    if not g:
        print("no rows"); return

    fig = plt.figure(figsize=(7.6, 5.0), facecolor=TH["surface"])
    ax = fig.add_axes([0.105, 0.135, 0.875, 0.70])
    ax.set_facecolor(TH["surface"])
    for sp in ax.spines.values():
        sp.set_color(TH["grid"])
    ax.grid(True, color=TH["grid"], lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TH["ink2"], labelsize=8)

    for (ds, sysn), rows in sorted(g.items()):
        cpu = np.array([float(r["cpu_pct"]) for r in rows]) / 100.0   # cores
        rss = np.array([float(r["rss_peak_mb"]) for r in rows])
        ax.errorbar(cpu.mean(), rss.mean(),
                    xerr=cpu.std(ddof=1), yerr=rss.std(ddof=1),
                    fmt=MARK[ds], ms=9, color=COL[sysn], ecolor=COL[sysn],
                    elinewidth=1.2, capsize=3, mew=1.2,
                    markerfacecolor=COL[sysn] if MARK[ds] == "o" else "none")
    # the gap between the two clusters is the result, so name what is in each
    # and the figure can be read without turning back to the table
    orb = [v for k, v in g.items() if k[1] != "VINS-Fusion"]
    vin = [v for k, v in g.items() if k[1] == "VINS-Fusion"]
    def band(rows_list, col, f=1.0):
        v = [f*float(r[col]) for rows in rows_list for r in rows]
        return min(v), max(v)
    oc, om = band(orb, "cpu_pct", 0.01), band(orb, "rss_peak_mb")
    vc, vm = band(vin, "cpu_pct", 0.01), band(vin, "rss_peak_mb")
    ax.annotate("ORB-SLAM3, both front ends\n%.1f-%.1f cores, %.0f-%.0f MiB" % (oc+om),
                xy=(3.6, 1770), xytext=(3.1, 1180), fontsize=8.5, color=TH["ink2"],
                arrowprops=dict(arrowstyle="->", color=TH["ink2"], lw=0.8))
    ax.annotate("VINS-Fusion\n%.1f-%.1f cores, %.0f-%.0f MiB" % (vc+vm),
                xy=(8.15, 500), xytext=(6.1, 950), fontsize=8.5, color=TH["ink2"],
                arrowprops=dict(arrowstyle="->", color=TH["ink2"], lw=0.8))
    ax.set_xlabel("processor time [cores of one, mean $\\pm$ SD over three repeats]",
                  color=TH["ink2"], fontsize=9)
    ax.set_ylabel("peak resident set [MiB]", color=TH["ink2"], fontsize=9)
    ax.set_xlim(2.5, 9.5)
    ax.set_ylim(0, 2100)

    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COL[s], marker="o", ls="", ms=8, label=s) for s in COL]
    h += [Line2D([], [], color=TH["ink2"], marker=MARK[d], ls="", ms=8,
                 markerfacecolor=TH["ink2"] if MARK[d] == "o" else "none", label=LAY[d])
          for d in MARK]
    ax.legend(handles=h, fontsize=8, ncol=2, frameon=False, loc="upper center",
              labelcolor=TH["ink2"])
    ax.set_title("Experiment 1, simulation: what each pipeline costs on the mapping\n"
                 "recordings. Neither is cheaper; they sit in opposite corners.",
                 fontsize=10.5, loc="left", color=TH["ink"], pad=12)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "fig_cost_profile_light.png")
    fig.savefig(path, dpi=160, facecolor=TH["surface"])
    plt.close(fig)
    print("  wrote", path)
    for (ds, sysn), rows in sorted(g.items()):
        cpu = np.mean([float(r["cpu_pct"]) for r in rows]) / 100
        rss = np.mean([float(r["rss_peak_mb"]) for r in rows])
        print("    %-18s %-14s %.2f cores, %4.0f MiB" % (ds, sysn, cpu, rss))


if __name__ == "__main__":
    main()
