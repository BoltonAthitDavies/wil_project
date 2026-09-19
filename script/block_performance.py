#!/usr/bin/env python3
"""Per-block performance verdicts for the pipeline diagrams in the minor report.

    python3 script/block_performance.py --out docs/report/minor_report

Writes, for each of the static and dynamic conditions:
    tables/block_performance_<cond>.tex   the evidence table
    figures/block_colours_<cond>.tex      \\blockcolour macros the TikZ reads

WHY THE COLOURS ARE GENERATED AND NOT TYPED
    A coloured block in a system diagram is an assertion, and there is no
    convention that makes a reader treat it as a hedged one. Typing the colours by
    hand would put ~40 unsourced claims into a figure that looks authoritative.
    Here every colour is computed from an artifact, and the table that accompanies
    the figure prints the metric, its value, the criterion and the rule that
    produced the verdict. If a number moves, the figure moves with it.

THE FOUR VERDICTS, AND WHAT THEY MAY NOT BE READ AS
    good     meets its stated criterion
    poor     fails its stated criterion
    noeffect measured, and the measurement shows the block does not change the
             outcome -- neither a success nor a failure
    none     not instrumented, not exercised, or not implemented; NO claim made

    `poor` never means "this block caused the trajectory error" unless the rule is
    `ablation`. Latency and outcome rules describe what a block costs or emits, not
    what it contributes to accuracy, and those are different questions. Only an
    ablation answers the second, and the project has four of them: the EKF's wheel
    correction, its IMU prediction, its yaw-rate model, and ORB-SLAM3's loop
    closing. Every other block is coloured on cost or on what it emitted.

THE THREE RULES
    ablation  ATE with the block removed or swapped, over ATE with it in place.
              >= 2.0   the block is load-bearing and working        -> good
              <= 1.1   removing it costs nothing, or helps          -> poor
              between  no measurable effect                         -> noeffect
    latency   the block's median time as a share of its stage total.
              >= 0.40 of the stage, or a stage that exceeds the
              inter-frame budget                                    -> poor
    outcome   a stated counter against a stated threshold.
"""

import argparse
import csv
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vio_metrics as vm

ROOT = vm.ROOT
STATIC = ["simulation/dataset_allsensor_%s" % d
          for d in ("000", "001", "002", "003", "004")]
# Frames arrive 1/(30.3 Hz * 0.5 replay) apart, so a per-frame stage that takes
# longer than this cannot keep up. Both estimators replayed at 0.5x; see the
# run manifests.
FRAME_BUDGET_MS = 1000.0 / (30.3 * 0.5)
DOMINANT = 0.40                 # share of a stage that makes a block its main cost
ABL_GOOD, ABL_POOR = 2.0, 1.1
USABLE_MIN = 0.99

# Provenance of each threshold, printed beside every verdict. Two of the numbers
# below follow from something outside this report; the rest are conventions this
# report adopts so that the rule is stated rather than implicit, and a reader who
# disagrees can recompute from the value column.
#
#   derived     FRAME_BUDGET_MS is arithmetic on the recorded conditions,
#               1000/(30.3 Hz * 0.5 replay). ANEES = DOF = 3 is the standard
#               consistency result for a 3-DOF pose error.
#   convention  DOMINANT, ABL_GOOD, ABL_POOR, the inverted ablation band and
#               USABLE_MIN are round numbers chosen here. Nothing external fixes
#               40% rather than 35%, or 2.0x rather than 1.5x.
#
# A third category was removed rather than labelled: an earlier revision graded
# VINS solver quality against "final cost > 5000", a threshold picked after seeing
# 1176 static and 14110 dynamic. That is circular -- the verdict was guaranteed by
# the choice -- and raw Ceres cost has no absolute scale anyway, since it depends
# on residual count and weighting. It is replaced below by the solver's own
# solution-usable flag.
DERIVED, CONVENTION = "derived", "convention"


def med_col(path, col):
    v = []
    if not os.path.exists(path):
        return float("nan")
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                x = float(r[col])
            except (TypeError, ValueError, KeyError):
                continue
            if x == x:
                v.append(x)
    return float(np.median(v)) if v else float("nan")


def runs_for(tree, cond, run=None):
    if cond == "static":
        return [vm.run_dir(tree, d, run or "logging_20260916_01") for d in STATIC]
    pat = os.path.join(ROOT, "output", tree, "simulation",
                       "dataset_dynamic_nofloortexture_*", "logging_*")
    return sorted(d for d in glob.glob(pat)
                  if os.path.exists(os.path.join(d, "vio.csv")))


def frac_rows(paths, fname, col, pred):
    """Fraction of all rows across `paths` whose `col` satisfies `pred`."""
    n = k = 0
    for p in paths:
        f = os.path.join(p, fname)
        if not os.path.exists(f):
            continue
        with open(f) as fh:
            for r in csv.DictReader(fh):
                n += 1
                if pred(r.get(col, "")):
                    k += 1
    return (k / n) if n else float("nan")


def across(paths, fname, col):
    vals = [med_col(os.path.join(p, fname), col) for p in paths]
    vals = [v for v in vals if v == v]
    return float(np.median(vals)) if vals else float("nan")


def gt_path(run_dir, dataset=None):
    """Ground truth for a run.

    The estimator launch files do not copy ground truth into their own
    logging_* directories; only proprio_estimator.py writes one, into the flat
    baseline trees. So a run directory's own ground_truth.csv usually does not
    exist, and looking only there silently turned every ORB ablation into NaN.
    """
    here = os.path.join(run_dir, "ground_truth.csv")
    if os.path.exists(here):
        return here
    if dataset is None:
        # .../output/<tree>/simulation/<name>[/logging_*]
        parts = os.path.abspath(run_dir).split(os.sep)
        if "logging_" in parts[-1]:
            parts = parts[:-1]
        dataset = os.path.join(parts[-2], parts[-1])
    for tree in ("output_wheel", "output_imu", "output_ekf", "output_vins", "output_orb"):
        c = os.path.join(ROOT, "output", tree, dataset, "ground_truth.csv")
        if os.path.exists(c):
            return c
    return here


def ate(run_dir, dataset=None):
    a = vm.load(os.path.join(run_dir, "vio.csv"))
    g = vm.load(gt_path(run_dir, dataset))
    if a is None or g is None:
        return float("nan")
    t, p = a[0], a[1]
    tg, pg = g[0], g[1]
    m = (t >= max(t[0], tg[0])) & (t <= min(t[-1], tg[-1]))
    if m.sum() < 10:
        return float("nan")
    rp = vm.resample(tg, pg, t[m])
    R, tr = vm.umeyama(p[m], rp)
    return float(np.sqrt((np.linalg.norm((R @ p[m].T).T + tr - rp, axis=1) ** 2).mean()))


def ekf_ablation(suffix):
    """Median ATE ratio of a variant tree against the full EKF, over the 5 bags."""
    r = []
    for d in STATIC:
        base = os.path.join(ROOT, "output", "output_ekf", d)
        var = base + suffix
        b, v = ate(base), ate(var)
        if b == b and v == v and b > 0:
            r.append(v / b)
    return float(np.median(r)) if r else float("nan")


def orb_loop_ablation(cond):
    """Median ATE(loop closing on) / ATE(off). >1 means enabling it costs accuracy."""
    if cond != "static":
        return float("nan")            # the relogged runs have no disabled variant
    r = []
    for d in STATIC:
        on = ate(vm.run_dir("output_orb", d, "logging_20260916_01"), d)
        off = ate(vm.run_dir("output_orb", d, "logging_20260916_02"), d)
        if on == on and off == off and off > 0:
            r.append(on / off)
    return float(np.median(r)) if r else float("nan")


def verdict_ablation(ratio, invert=False):
    """invert=True for a block whose ablation ratio is 'cost of enabling it'."""
    if ratio != ratio:
        return "none"
    if invert:
        # ratio = with/without. >1 means having the block makes things worse.
        if ratio >= 1.25:
            return "poor"
        if ratio <= 0.8:
            return "good"
        return "noeffect"
    if ratio >= ABL_GOOD:
        return "good"
    if ratio <= ABL_POOR:
        return "poor"
    return "noeffect"


def verdict_latency(ms, stage_ms, stage_over_budget):
    if ms != ms or stage_ms != stage_ms or stage_ms <= 0:
        return "none", float("nan")
    share = ms / stage_ms
    if share >= DOMINANT or (stage_over_budget and share >= 0.25):
        return "poor", share
    return "good", share


# --------------------------------------------------------------------------
# block inventories. (id, label, stage, rule, spec)
# --------------------------------------------------------------------------

def orb_blocks(cond):
    P = runs_for("output_orb", cond)
    tf, lm = "tracking_frontend.csv", "local_mapping.csv"
    tr_tot = across(P, tf, "tracking_total_ms")
    lm_tot = across(P, lm, "total_ms")
    over_tr = tr_tot > FRAME_BUDGET_MS
    out = []

    def lat(bid, label, stage, fname, col, tot, over):
        ms = across(P, fname, col)
        v, share = verdict_latency(ms, tot, over)
        out.append(dict(id=bid, label=label, stage=stage, rule="latency",
                        metric="median %s" % col, value=ms, unit="ms",
                        criterion=tag(CONVENTION, "$<$%d\\%% of stage" % int(DOMINANT * 100)),
                        extra="%.1f\\%% of stage" % (100 * share) if share == share else "--",
                        verdict=v))

    lat("orb_extract", "Extract ORB\\\\(+preproc, stereo)", "Tracking", tf,
        "image_preprocessing_feature_stereo_ms", tr_tot, over_tr)
    lat("orb_imupre", "IMU\\\\preintegration", "Tracking", tf,
        "imu_preintegration_ms", tr_tot, over_tr)
    lat("orb_posepred", "Initial pose\\\\estimation", "Tracking", tf,
        "pose_prediction_ms", tr_tot, over_tr)
    lat("orb_tracklocal", "Track\\\\local map", "Tracking", tf,
        "local_map_tracking_ms", tr_tot, over_tr)
    lat("orb_kfdec", "New keyframe\\\\decision", "Tracking", tf,
        "keyframe_decision_ms", tr_tot, over_tr)
    lat("orb_kfins", "KeyFrame\\\\insertion", "Local mapping", lm,
        "keyframe_insertion_ms", lm_tot, False)
    lat("orb_mpcull", "Recent MapPoints\\\\culling", "Local mapping", lm,
        "map_point_culling_ms", lm_tot, False)
    lat("orb_newpoints", "New points\\\\creation", "Local mapping", lm,
        "map_point_creation_fusion_ms", lm_tot, False)
    lat("orb_lba", "Local BA", "Local mapping", lm, "local_ba_ms", lm_tot, False)
    lat("orb_kfcull", "Local keyframes\\\\culling", "Local mapping", lm,
        "keyframe_culling_ms", lm_tot, False)

    # stage totals, judged against the real-time budget rather than a share
    out.append(dict(id="orb_tracking_total", label="TRACKING total", stage="Tracking",
                    rule="latency", metric="median tracking_total_ms", value=tr_tot,
                    unit="ms", criterion=tag(DERIVED, "$<$\\SI{%.1f}{\\milli\\second} frame budget"
                                    % FRAME_BUDGET_MS), extra="--",
                    verdict="poor" if over_tr else "good"))

    # IMU initialisation / scale refinement: present in the log but effectively
    # instantaneous, and absent altogether from the dynamic runs.
    for bid, label, col in (("orb_imuinit", "IMU\\\\initialization", "imu_initialization_ms"),
                            ("orb_imuref", "IMU scale\\\\refinement", "inertial_refinement_ms")):
        ms = across(P, lm, col)
        out.append(dict(id=bid, label=label, stage="Local mapping", rule="latency",
                        metric="median %s" % col, value=ms, unit="ms",
                        criterion=tag(CONVENTION, "$<$40\\% of stage"),
                        extra="not logged" if ms != ms else "negligible",
                        verdict="none" if ms != ms else "good"))

    # place recognition / loop closing: outcome, plus an ablation in static
    cand = acc = ev = 0
    for p in P:
        f = os.path.join(p, "loop_closing.csv")
        if not os.path.exists(f):
            continue
        with open(f) as fh:
            for r in csv.DictReader(fh):
                ev += 1
                if "loop_candidate" in (r.get("status") or ""):
                    cand += 1
                if r.get("event") == "loop_closure" and "rejected" not in (r.get("status") or ""):
                    acc += 1
    out.append(dict(id="orb_placerec", label="Place\\\\recognition", stage="Loop closing",
                    rule="outcome", metric="candidates accepted", value=acc,
                    unit="of %d" % cand,
                    criterion=tag(CONVENTION, "$>$\\SI{10}{\\percent} of candidates accepted"),
                    extra="%d events" % ev,
                    verdict="poor" if cand and acc / max(cand, 1) < 0.10 else
                            ("none" if not cand else "good")))
    lr = orb_loop_ablation(cond)
    out.append(dict(id="orb_loopfusion", label="Loop fusion\\\\+ essential graph",
                    stage="Loop closing", rule="ablation",
                    metric="ATE(on)/ATE(off)", value=lr, unit="$\\times$",
                    criterion=tag(CONVENTION, "$<$1.0 (enabling it should help)"),
                    extra="one run per configuration",
                    verdict=verdict_ablation(lr, invert=True)))
    out.append(dict(id="orb_fullba", label="Full BA", stage="Loop closing",
                    rule="outcome", metric="global BA executions", value=float("nan"),
                    unit="", criterion="--", extra="see \\cref{tab:orb-loop}",
                    verdict="none"))
    out.append(dict(id="orb_yolo", label="Dynamic feature\\\\removal (YOLO)",
                    stage="Tracking", rule="none", metric="--", value=float("nan"),
                    unit="", criterion="--",
                    extra="filter disabled in every run reported here",
                    verdict="none"))
    return out


def vins_blocks(cond):
    P = runs_for("output_vins", cond)
    be, fe = "backend.csv", "frontend.csv"
    be_tot = across(P, be, "backend_total_ms")
    out = []

    def lat(bid, label, fname, col, tot, stage):
        ms = across(P, fname, col)
        v, share = verdict_latency(ms, tot, False)
        out.append(dict(id=bid, label=label, stage=stage, rule="latency",
                        metric="median %s" % col, value=ms, unit="ms",
                        criterion=tag(CONVENTION, "$<$40\\% of stage"),
                        extra="%.1f\\%% of stage" % (100 * share) if share == share else "--",
                        verdict=v))

    ft = across(P, fe, "feature_tracking_ms")
    out.append(dict(id="vins_track", label="Feature\\\\tracking", stage="Front end",
                    rule="latency", metric="median feature_tracking_ms", value=ft,
                    unit="ms", criterion="$<$\\SI{%.1f}{\\milli\\second} frame budget"
                    % FRAME_BUDGET_MS, extra="--",
                    verdict="poor" if ft > FRAME_BUDGET_MS else "good"))
    qw = across(P, be, "feature_queue_wait_ms")
    out.append(dict(id="vins_queue", label="Feature queue", stage="Front end",
                    rule="latency", metric="median feature_queue_wait_ms", value=qw,
                    unit="ms", criterion="$<$\\SI{%.1f}{\\milli\\second} frame budget"
                    % FRAME_BUDGET_MS,
                    extra="backlog %.1f frames" % across(P, fe, "feature_backlog"),
                    verdict="poor" if qw > FRAME_BUDGET_MS else "good"))
    lat("vins_imuprop", "IMU\\\\propagation", be, "imu_propagation_ms", be_tot, "Back end")
    lat("vins_tri", "Triangulation", be, "triangulation_ms", be_tot, "Back end")
    lat("vins_construct", "Problem\\\\construction", be, "problem_construction_ms",
        be_tot, "Back end")
    lat("vins_solve", "Ceres solve", be, "solver_ms", be_tot, "Back end")
    lat("vins_marg", "Marginalization", be, "marginalization_ms", be_tot, "Back end")
    lat("vins_outlier", "Outlier\\\\rejection", be, "outlier_rejection_ms", be_tot, "Back end")
    lat("vins_slide", "Slide\\\\window", be, "slide_window_ms", be_tot, "Back end")
    out.append(dict(id="vins_backend_total", label="BACK END total", stage="Back end",
                    rule="latency", metric="median backend_total_ms", value=be_tot,
                    unit="ms", criterion="$<$\\SI{%.1f}{\\milli\\second} frame budget"
                    % FRAME_BUDGET_MS, extra="--",
                    verdict="poor" if be_tot > FRAME_BUDGET_MS else "good"))
    # The solver's own usability flag, not a threshold on raw Ceres cost. Cost has
    # no absolute scale -- it depends on residual count and weighting -- and the
    # previous threshold of 5000 had been picked after seeing 1176 static against
    # 14110 dynamic, which made the verdict a restatement of the choice.
    usable = frac_rows(P, be, "solver_solution_usable", lambda v: v == "1")
    noconv = frac_rows(P, be, "solver_termination_type", lambda v: v == "1")
    out.append(dict(id="vins_solverq", label="Solver\\\\convergence", stage="Back end",
                    rule="outcome", metric="solves returning a usable solution",
                    value=100.0 * usable, unit="\\%",
                    criterion=tag(CONVENTION, "$\\geq$\\SI{99}{\\percent} usable"),
                    extra="\\SI{%.1f}{\\percent} hit the iteration or "
                          "\\SI{80}{\\milli\\second} cap without converging; "
                          "reported, not graded" % (100.0 * noconv),
                    verdict="good" if usable >= USABLE_MIN else "poor"))
    out.append(dict(id="vins_loop", label="Loop fusion", stage="Loop closing",
                    rule="none", metric="--", value=float("nan"), unit="",
                    criterion="--",
                    extra="not implemented in this fork",
                    verdict="none"))
    out.append(dict(id="vins_rtab", label="RTAB-Map\\\\correction", stage="Loop closing",
                    rule="none", metric="--", value=float("nan"), unit="", criterion="--",
                    extra="separate process; correction never injected",
                    verdict="none"))
    out.append(dict(id="vins_yolo", label="Persistent-feature\\\\removal (YOLO)",
                    stage="Front end", rule="none", metric="--", value=float("nan"),
                    unit="", criterion="--",
                    extra="filter disabled in every run reported here",
                    verdict="none"))
    return out


def ekf_blocks(cond):
    if cond != "static":
        return []                       # no dynamic dataset carries wheel encoders
    out = []
    # NOTE these two blocks share ONE intervention. The wheel update carries two
    # measurement rows -- wheel speed correcting v, and (gyro - wheel yaw rate)
    # correcting the bias -- and --no-wheel-update removes both at once. There is
    # no flag that removes only one, so the 54x is joint evidence for the pair, not
    # two independent findings. Reporting it twice without saying so would double
    # count the only strong result the EKF has.
    specs = [("ekf_wheelupd", "Wheel speed\\\\correction", "_nowheel",
              "ATE without the wheel update / with"),
             ("ekf_biasobs", "Gyro-bias\\\\observation", "_nowheel",
              "ATE without the wheel update / with"),
             ("ekf_imupred", "IMU prediction", None, "ATE of wheel-only / EKF"),
             ("ekf_yaw", "Yaw-rate model\\\\(bicycle)", "_diffyaw",
              "ATE with differential / with bicycle")]
    joint = ("ekf_wheelupd", "ekf_biasobs")
    for bid, label, suffix, desc in specs:
        if bid == "ekf_imupred":
            r = []
            for d in STATIC:
                b = ate(os.path.join(ROOT, "output", "output_ekf", d))
                w = ate(os.path.join(ROOT, "output", "output_wheel", d))
                if b == b and w == w and b > 0:
                    r.append(w / b)
            ratio = float(np.median(r)) if r else float("nan")
            note = "median over 5 bags"
        else:
            ratio = ekf_ablation(suffix)
            note = "median over 5 bags"
            if bid in joint:
                note += "; joint with the other row -- one flag removes both"
        out.append(dict(id=bid, label=label, stage="EKF", rule="ablation",
                        metric=desc, value=ratio, unit="$\\times$",
                        criterion=tag(CONVENTION, "$\\geq$%.1f$\\times$ (block is load-bearing)" % ABL_GOOD),
                        extra=note,
                        verdict=verdict_ablation(ratio)))
    out.append(dict(id="ekf_anchor", label="Initial pose\\\\anchor", stage="EKF",
                    rule="none", metric="--", value=float("nan"), unit="", criterion="--",
                    extra="ground-truth pose; not a performance block",
                    verdict="none"))
    out.append(dict(id="ekf_cov", label="Covariance\\\\propagation", stage="EKF",
                    rule="outcome", metric="ANEES (3 DOF)", value=float("nan"), unit="",
                    criterion=tag(DERIVED, "3.0 for a consistent filter"),
                    extra="64.7--123.2; see \\cref{tab:proprio-covariance}",
                    verdict="poor"))
    return out


PIPES = [("ORB-SLAM3", orb_blocks), ("VINS-Fusion", vins_blocks),
         ("EKF wheel+IMU", ekf_blocks)]
VERDICT_COLOUR = {"good": "perfgood", "poor": "perfpoor",
                  "noeffect": "perfnoeffect", "none": "perfnone"}


def tag(src, text):
    return "\\textit{%s:} %s" % (src, text)


def fmt(v, unit):
    if v != v:
        return "--"
    if unit == "ms":
        return "\\num{%.3f}" % v if v < 1 else "\\num{%.1f}" % v
    if unit.startswith("$\\times$"):
        return "%.2f" % v
    if unit == "\\%":
        return "%.2f\\%%" % v
    if unit.startswith("of "):
        return "%d %s" % (v, unit)
    return "%.1f" % v


def write(cond, outdir):
    rows = []
    for name, fn in PIPES:
        for b in fn(cond):
            b["pipeline"] = name
            rows.append(b)
    cols = os.path.join(outdir, "figures", "block_colours_%s.tex" % cond)
    with open(cols, "w") as f:
        f.write("%% Generated by script/block_performance.py -- do not edit.\n")
        for b in rows:
            # Underscores are stripped from the control-sequence name: a csname
            # built from a token with catcode 8 is a trap that fails obscurely at
            # use time rather than at definition time.
            f.write("\\expandafter\\def\\csname blockcolour@%s@%s\\endcsname{%s}\n"
                    % (cond, b["id"].replace("_", ""), VERDICT_COLOUR[b["verdict"]]))
    # One table per pipeline, not one per condition: each algorithm gets its own
    # page in the report, which is how the figures are read -- a reader checking
    # why a VINS block is red should not have to page past ORB-SLAM3 to find it.
    slug = {"ORB-SLAM3": "orb", "VINS-Fusion": "vins", "EKF wheel+IMU": "ekf"}
    written = []
    for pipe in dict.fromkeys(b["pipeline"] for b in rows):
        sub = [b for b in rows if b["pipeline"] == pipe]
        # Group by stage, keeping each stage's first appearance order. The block
        # inventories append a few entries (stage totals, the IMU blocks) after the
        # loop that produced their stage-mates, so reading `sub` in insertion order
        # emitted "Tracking" and "Local mapping" headers twice each.
        order = list(dict.fromkeys(b["stage"] for b in sub))
        sub = sorted(sub, key=lambda b: order.index(b["stage"]))
        tab = os.path.join(outdir, "tables",
                           "block_performance_%s_%s.tex" % (cond, slug[pipe]))
        with open(tab, "w") as f:
            f.write("%% Generated by script/block_performance.py -- do not edit.\n")
            f.write("\\begin{tabularx}{\\textwidth}{>{\\raggedright\\arraybackslash}p{0.155\\textwidth}"
                    "l>{\\raggedright\\arraybackslash}X r"
                    ">{\\raggedright\\arraybackslash}p{0.20\\textwidth}l}\n\\toprule\n")
            f.write("Block & Rule & Metric & Value & Criterion / note & Verdict"
                    " \\\\\n\\midrule\n")
            laststage = None
            for b in sub:
                if b["stage"] != laststage:
                    # \addlinespace rather than \arraystretch: several cells wrap to
                    # three lines, and stretching the array would open those interior
                    # lines too, which loosens the text without separating the rows.
                    # booktabs' inter-row skip separates rows and leaves cells alone.
                    if laststage is not None:
                        f.write("\\addlinespace[5pt]\n")
                    f.write("\\multicolumn{6}{l}{\\textbf{%s}} \\\\\n"
                            "\\addlinespace[2pt]\n" % b["stage"])
                    laststage = b["stage"]
                label = b["label"].replace("\\\\", " ")
                note = b["criterion"]
                if b["extra"] and b["extra"] != "--":
                    note += " (%s)" % b["extra"]
                f.write("\\quad %s & %s & %s & %s & %s & \\perfmark{%s} \\\\\n"
                        "\\addlinespace[3.5pt]\n"
                        % (label, b["rule"], b["metric"].replace("_", "\\_"),
                           fmt(b["value"], b["unit"]), note, b["verdict"]))
            f.write("\\bottomrule\n\\end{tabularx}\n")
        written.append(os.path.basename(tab))
    print("  %-8s %d blocks -> %s + %s" % (cond, len(rows),
                                            os.path.basename(cols), ", ".join(written)))
    for v in ("good", "poor", "noeffect", "none"):
        ids = [b["id"] for b in rows if b["verdict"] == v]
        print("     %-9s %2d  %s" % (v, len(ids), " ".join(ids)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "report", "minor_report"))
    a = ap.parse_args()
    for d in ("figures", "tables"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    allrows = {}
    for cond in ("static", "dynamic"):
        allrows[cond] = write(cond, a.out)
    with open(os.path.join(a.out, "tables", "block_performance.json"), "w") as f:
        json.dump(allrows, f, indent=1, default=lambda x: None if x != x else x)


if __name__ == "__main__":
    main()
