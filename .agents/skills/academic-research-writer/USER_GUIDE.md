# User Guide

## What it does

Ask for a chapter, an abstract, a reference list or a format check, and the
skill writes it to the KMUTT Faculty of Engineering format (English edition):
Times New Roman 12 pt, 4/3/2/2 cm margins, 15/14/13 pt bold headings, bracketed
citation numbers, and reference entries in the faculty's own layout.

It also finds and vets sources: Google Scholar, IEEE Xplore, ACM, ScienceDirect,
PubMed, arXiv, each checked for peer review, author affiliation and venue
standing before it earns a citation number.

## Asking for things

Plain requests work. Useful detail to include: which part of the volume, thesis
or final project, and whether it joins the existing LaTeX build.

```
Draft section 2.3 on dynamic-environment SLAM, 600 words, cite from references.tex
where entries exist and tell me what you had to add.

Write the abstract from Chapter 4's results. 300 words, keywords sorted A-Z.

Check chapter_3_methodology.tex against the KMUTT manual.

De-ramble section 4.3. Show me before vs. after and what you held back.

Turn these five papers into KMUTT reference entries for references.tex.

Build the curriculum vitae page from my details.
```

## What it will not do

- Emit IEEE, APA or journal-house references. The faculty format only.
- Restyle the existing volume. `docs/report/final_report` already conforms; the
  skill matches it rather than reformatting it.
- Edit a Chapter 3 or 4 fragment in place — those are assembled, and the rule
  is in the `final-report-build` memory note.
- Invent a reference, a page range or a DOI, or cite a paper it has not read
  enough of to characterise.
- Decide what a result may claim. That is `wil-thesis-evaluation`'s job; this
  skill formats what that one licenses.

## De-rambling a draft

Ask it to tighten, shorten, filter or de-ramble, and it runs the pass in
`references/DE-RAMBLE.md`: clutter cut, long sentences broken, each paragraph's
point moved to its first sentence, every number and citation left untouched.

It reports a before/after table of the structural changes, the word count both
ways, and — the part worth reading — **what it held back**: the hedges that look
like padding but record a validity limit (`remains untested`, `supports no claim
yet`, `as configured`). Those never go, because a shorter paragraph that reads
more confidently than the evidence allows is a worse paragraph.

It will also flag, rather than fix, passages that ramble because the thought
underneath is unresolved. Editing cannot rescue those.

It does not reorder sections, drop results, or edit a Chapter 3 or 4 fragment in
place — it proposes those and stops.

## The reference formatter

```bash
cd .agents/skills/academic-research-writer

# worked example of all eight entry types, plain and LaTeX
python3 scripts/kmutt_formatter.py

# one entry, ready for chapters/references.tex
python3 scripts/kmutt_formatter.py --latex journal \
  --authors "Mur-Artal, R." "Tardós, J.D." --year 2017 \
  --title "ORB-SLAM2: An Open-Source SLAM System for Monocular, Stereo, and RGB-D Cameras" \
  --source "IEEE Transactions on Robotics" --volume 33 --number 5 --pages 1255-1262
```

Entry types: `journal`, `conference`, `book`, `inbook`, `thesis`, `newspaper`,
`patent`, `online`. `--help` on any of them lists its fields. The script warns
when an author name is not `Surname, I.I.`, converts accents to LaTeX control
sequences for the pdflatex build, and derives the citation key as surname+year.

## Format quick reference

| | |
|---|---|
| Body | Times New Roman 12 pt |
| `CHAPTER 1 INTRODUCTION` | 15 pt bold, centred, caps |
| `1.1 Head Topic` | 14 pt bold |
| `1.1.1 Sub Topic` | 13 pt bold |
| Margins | left 4, top 3, right 2, bottom 2 cm |
| Page number | top right, none on a chapter's first page |
| Table caption | **above**, `(Continue)` when split |
| Figure caption | **below** |
| Reproduced float | citation number in the caption + `(Source : …, year A.D.)` |
| Citation | `[1]`, `[1, 2, 3]`; number fixed for the whole volume |
| Reference | `Surname, I.I., year, "Title," `**`Source`**`, Vol. X, No. Y, pp. A-B.` |
| Keywords | ` / ` separated, A–Z (Thai: ก–ฮ, no English words) |
| Thai abstract | AngsanaUPC 16 pt, not a literal translation, no passive voice |
| Acknowledgements | not signed by the author |

Details in `references/KMUTT-FORMAT.md` and
`references/KMUTT-CITATION-GUIDE.md`.

## Building the volume

```bash
cd docs/report/final_report
latexmk -pdf main.tex
```

The preamble of `main.tex` is where the format lives. If a format change is
genuinely needed, it belongs there — once — not in a chapter file.

## If something looks wrong

- **An entry looks like IEEE** — it came from a publisher's "cite" button or a
  reference manager. Regenerate it with `kmutt_formatter.py`.
- **Citation numbers out of order** — the number system orders by first
  citation; a number, once assigned, never changes. Re-check after moving text
  between chapters.
- **A heading is the wrong size** — fix `\titleformat` in `main.tex`, not the
  chapter.
- **The skill disappeared after an update** — `npx skills add
  endigo/claude-skills` restored the upstream IEEE version over these files.
