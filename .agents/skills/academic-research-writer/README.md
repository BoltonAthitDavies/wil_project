# Academic Research Writer — KMUTT Faculty of Engineering

Writes and checks academic documents in the **KMUTT thesis / final-project
format, English edition**, with peer-reviewed sources and KMUTT-format
references.

Adapted from `endigo/claude-skills`' generic IEEE-based skill. The IEEE citation
machinery has been **replaced**, not supplemented: this skill emits the faculty
format and nothing else.

## What changed from upstream

| Upstream | Here |
|---|---|
| IEEE references (`A. Author, "Title," Journal, vol. 1, no. 2, 2020`) | KMUTT references (`Author, A.B., 2020, "Title," `**`Journal`**`, Vol. 1, No. 2, pp. 3-4.`) |
| `references/IEEE-CITATION-GUIDE.md` | `references/KMUTT-CITATION-GUIDE.md` |
| `scripts/ieee_formatter.py` | `scripts/kmutt_formatter.py` (plain + `\bibitem` output) |
| Generic paper structure (I. Introduction, A. subsection) | `references/KMUTT-FORMAT.md` + `assets/kmutt_report_outline.md` — the faculty volume order |
| No page-layout rules | Margins, type sizes, caption sides, front-matter order, abstract block, CV layout |
| Output as DOCX/PDF by default | LaTeX into `docs/report/final_report`, which already encodes the format |

## Contents

- `SKILL.md` — the non-negotiables, the workflow, and this repository's rules.
- `references/KMUTT-FORMAT.md` — page setup, volume order, abstract page,
  tables and figures, curriculum vitae, and how `docs/report/final_report`
  implements each one.
- `references/KMUTT-CITATION-GUIDE.md` — number and name–year systems, every
  reference entry type, the LaTeX form, and a pre-build checklist.
- `references/DE-RAMBLE.md` — the concision pass: cut clutter, shorten
  sentences, point-first paragraphs, rigor held intact, before/after
  breakdown. Encodes which hedges are load-bearing and must survive.
- `references/ACADEMIC-WRITING.md` — prose conventions (from upstream).
- `references/SOURCE-VERIFICATION.md` — peer review, databases, predatory
  publishers (from upstream).
- `scripts/kmutt_formatter.py` — reference formatter. Run with no arguments for
  a worked example of all eight entry types.
- `assets/kmutt_chapter_template.tex` — chapter skeleton for the existing build.
- `assets/kmutt_report_outline.md` — the whole volume in order.

## Source of truth

`docs/MyThesisInfo/27.8.67 Template เล่มโครงงาน (Eng Ver.)/` — the faculty part
files and `เอกสารประกอบ_template ENG.pdf` (Thesis/Dissertation Preparation
Manual, Faculty of Engineering, August 2015), with
`docs/MyThesisInfo/1. คู่มือการเขียนวิทยานิพนธ์-cr.มจธ.pdf` for anything the
English edition leaves open. Every rule in this skill traces to those files.

## Note

Local modification. Re-running `npx skills add endigo/claude-skills` overwrites
these files with the upstream IEEE version — commit them, or back them up, before
updating.
