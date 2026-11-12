#!/usr/bin/env python3
"""Regenerate the Experiment 1-3 sections of chapter 4 from the raw write-ups.

    python3 script/build_chapter4.py            # rewrite the chapter in place
    python3 script/build_chapter4.py --check    # print what would change, write nothing

Since 6 Nov 2026 the chapter's slots for Experiments 1-3 are the insight notes
docs/report/insights/experiment_N/experiment_N.tex transformed mechanically:
  * \section{Simulation|Real robot} -> \subsection, with alias labels the chapter-3
    fragments cross-reference (sec:results-proprio, sec:yolo-performance, ...);
  * the note's own \label{sec:vio-filter-method} is dropped (chapter 3 owns it);
  * \paragraph{Objective./Process./Dataset(s)./Reporting rules./Results./Reason*./
    Retired result, for the record./Next action.} -> \textbf{...} (the chapter's
    slot convention); Analysis/Summary/Future plan stay paragraphs;
  * \reporttable{x} -> \input{tables/x};  ../../../../output/ -> ../../../output/;
  * two cross-note sentences are rewritten (see XNOTE).
The chapter head (cutoff paragraph, slot-status table), the Experiment 4 section
(fragment m4_exp4_sim plus the realistic-map subsection) and section 4.5 are kept
from the existing chapter file verbatim: edit those in the chapter itself.
Edit a note, then run this; never hand-edit the generated sections.
"""
import re, sys, os, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CH = os.path.join(ROOT, "docs/report/final_report/chapters/chapter_4_experiments.tex")

BOLD = ("Objective.", "Process.", "Dataset.", "Datasets.", "Reporting rules.", "Reason.",
        "Reason the baselines are absent.", "Retired result, for the record.", "Next action.")
ALIASES = {1: (["sec:results-proprio", "sec:results-validity", "sec:realistic-ab", "sec:rpe",
                "sec:trajectory-results"], ["sec:exp1-real"]),
           2: (["sec:yolo-performance", "fig:yolo-ground-truth-overlay"], ["sec:exp2-real"])}
XNOTE = [
    ("The visual results are those of Experiment~3's\nreal-robot section, scored by the same procedure and not repeated here:",
     "The visual results are those of \\cref{sec:exp3-real},\nscored by the same procedure and not repeated here:"),
    ("(the dataset table of the Experiment~1 write-up)", "(\\cref{tab:exp1-datasets})"),
    ("frame and levels each map's fitted plane, both without using the truth.\nIt is kept for the record",
     "frame and levels each map's fitted plane, both without using the truth\n(\\cref{sec:method-realv7-repair}). It is kept for the record"),
    ("beside the real robot's much shorter ones,\nand it separates",
     "beside the real robot's much shorter ones\n(\\cref{sec:sim-vs-real}), and it separates"),
]
HEADS = {1: "\\section{Experiment 1: Estimator Comparison Against a Non-Visual Baseline}\n\\label{sec:exp1}\n",
         2: "\\section{Experiment 2: Offline YOLO Detector Performance}\n\\label{sec:exp2}\n"}
RULE = "% =====================================================================\n\\FloatBarrier\n"


def note(n):
    s = open(os.path.join(ROOT, "docs/report/insights/experiment_%d/experiment_%d.tex" % (n, n))).read()
    a = s.index("\\section{Simulation}"); b = s.index("\\section{Real robot}"); c = s.index("\\end{document}")
    return s[a:b], s[b:c]


def xf(s, aliases):
    for h in BOLD:
        s = s.replace("\\paragraph{%s}" % h, "\\textbf{%s}" % h)
    s = s.replace("\\reporttable{", "\\input{tables/").replace("../../../../output/", "../../../output/")
    s = s.replace("% =====================================================================\n", "")
    def sub(kind, m):
        own = m.group(1)
        labs = list(aliases) + ([own] if own not in aliases else [])
        return "\\FloatBarrier\n\\subsection{%s}\n" % kind + "".join("\\label{%s}\n" % l for l in labs)
    s = re.sub(r"\\section\{Simulation\}\n\\label\{([^}]*)\}\n", lambda m: sub("Simulation", m), s)
    s = re.sub(r"\\section\{Real robot\}\n\\label\{([^}]*)\}\n", lambda m: sub("Real robot", m), s)
    return s.strip() + "\n"


def build():
    old = open(CH).read()
    i1 = old.index(RULE + "\\section{Experiment 1")
    i4 = old.index(RULE + "\\section{Experiment 3")
    parts = [old[:i1]]
    for n in (1, 2):
        sim, real = note(n)
        sim_a, real_a = ALIASES[n]
        sim, real = xf(sim, sim_a), xf(real, real_a)
        for a, b in XNOTE:
            sim = sim.replace(a, b); real = real.replace(a, b)
        parts.append(RULE + HEADS[n] + "\n" + sim + "\n" + real + "\n")
    parts.append(old[i4:])
    new = "".join(parts)
    labs = re.findall(r"\\label\{([^}]*)\}", new)
    dup = [k for k, v in collections.Counter(labs).items() if v > 1]
    assert not dup, "duplicate labels: %s" % dup
    return old, new


if __name__ == "__main__":
    old, new = build()
    if "--check" in sys.argv:
        import difflib
        d = list(difflib.unified_diff(old.splitlines(), new.splitlines(), "chapter (current)", "chapter (from notes)", lineterm="", n=0))
        print("\n".join(d[:200])); print("... %d diff lines" % len(d))
    else:
        open(CH, "w").write(new)
        print("wrote", CH, new.count("\n"), "lines")
