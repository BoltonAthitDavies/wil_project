---
name: academic-research-writer
description: Write and revise academic documents in the KMUTT Faculty of Engineering thesis/final-project format (English edition) - Times New Roman 12 pt, 4/3/2/2 cm margins, 15/14/13 pt bold headings, numbered citations and KMUTT-format references, with sources verified as peer-reviewed. Use for thesis and final-project chapters, abstracts, literature reviews, reference lists, table and figure captions, curriculum vitae, and for checking a draft against the KMUTT manual.
---

# Academic Research Writer (KMUTT Faculty of Engineering)

Produces academic documents that conform to the KMUTT thesis-writing manual,
English edition, with verified peer-reviewed sources and KMUTT-format
references.

The format is not negotiable and not a matter of taste: it is set by
`docs/MyThesisInfo/27.8.67 Template เล่มโครงงาน (Eng Ver.)/` and the faculty
manual beside it. Where a general academic habit disagrees with the manual, the
manual wins.

## Non-negotiables

Get these right before anything else; they are the ones a reviewer checks first.

| | Rule |
|---|---|
| Font | Times New Roman **12 pt throughout** (Thai abstract page: AngsanaUPC 16 pt) |
| Margins | left **4 cm**, top **3 cm**, right **2 cm**, bottom **2 cm**, A4 |
| Headings | `CHAPTER n TITLE` 15 pt bold centred caps · `1.1` 14 pt bold · `1.1.1` 13 pt bold |
| Page numbers | top right; none on a chapter's first page; roman front matter from ii, arabic from Chapter 1 |
| Citations | number system `[1]`, `[1, 2, 3]`; a document keeps its number for the whole volume |
| References | KMUTT format — author, **year second**, "title," **bold source**, `Vol.`, `No.`, `pp.` |
| Tables | caption **above**, `Table 1.1`, `(Continue)` when split, `(Source : …, year A.D.)` |
| Figures | caption **below**, `Figure 1.1`, source line beneath |
| Keywords | separated by ` / `, sorted A–Z (Thai: ก–ฮ, no English words) |

**Never emit IEEE-style references.** `A. Author, "Title," Journal, vol. 1,
no. 2, pp. 3-4, 2020` is wrong here, however standard it looks. The KMUTT entry
is `Author, A.B., 2020, "Title," `**`Journal`**`, Vol. 1, No. 2, pp. 3-4.`

Full specifications:

- Layout, volume order, abstract block, captions, CV → [KMUTT-FORMAT.md](references/KMUTT-FORMAT.md)
- Citation systems and every entry type → [KMUTT-CITATION-GUIDE.md](references/KMUTT-CITATION-GUIDE.md)
- Prose conventions → [ACADEMIC-WRITING.md](references/ACADEMIC-WRITING.md)
- Cutting a rambling draft → [DE-RAMBLE.md](references/DE-RAMBLE.md)
- Judging a source → [SOURCE-VERIFICATION.md](references/SOURCE-VERIFICATION.md)

## Workflow

### 1. Establish what is being written, and into what

Settle before drafting:

- Which part of the volume — a numbered chapter, the abstract, the reference
  list, an appendix, the CV, or a standalone document.
- Thesis or bachelor's final project. The labels differ: a final project uses
  `Project Title` / `Project Credits` (6) / `Project Advisor`; a thesis uses
  `Thesis Title` / `Thesis Credits` (6 or 12) / `Thesis Advisors`.
- Whether it lands in an existing LaTeX volume or starts a new document. If a
  volume already exists, **read its preamble and an existing chapter first** and
  match them; do not restyle what is already conforming.

### 2. Find and verify sources

Search Google Scholar, IEEE Xplore, the ACM Digital Library, ScienceDirect,
PubMed and arXiv as the field requires. For each source confirm peer review,
author affiliation, venue standing and relevance before it earns a number; see
[SOURCE-VERIFICATION.md](references/SOURCE-VERIFICATION.md) for the protocol and
the predatory-publisher red flags. Prefer the last 5–7 years unless the point is
historical. Record the fields the KMUTT entry needs — surname and initials,
year, exact title, source title, volume, issue, page range — at the moment you
accept the source; recovering them later is the expensive part.

Do not cite what you have not read enough of to characterise. Never invent a
reference, a page range or a DOI.

### 3. Draft

Write the prose to the conventions in
[ACADEMIC-WRITING.md](references/ACADEMIC-WRITING.md): formal register,
objective stance, hedged claims, one idea per paragraph opened by a topic
sentence and supported by a cited source. Every factual claim taken from outside
carries its number.

Chapter skeleton, when a new chapter is being started:
`assets/kmutt_chapter_template.tex`. Whole-volume order:
`assets/kmutt_report_outline.md`.

### 4. Build the reference list

Assemble entries with `scripts/kmutt_formatter.py`, which emits both the plain
and the `\bibitem` form:

```bash
python3 scripts/kmutt_formatter.py journal \
  --authors "Campos, C." "Elvira, R." "Tardós, J.D." \
  --year 2021 --title "ORB-SLAM3: An Accurate Open-Source Library …" \
  --source "IEEE Transactions on Robotics" --volume 37 --number 6 --pages 1874-1890 \
  --latex
```

Run `--help` for the entry types: `journal`, `conference`, `book`, `inbook`,
`thesis`, `newspaper`, `patent`, `online`.

### 5. De-ramble

Run the concision pass of [DE-RAMBLE.md](references/DE-RAMBLE.md) over anything
drafted or revised, and whenever asked to tighten, shorten, filter or de-ramble
existing text. The five rules: cut the clutter, shorten sentences, put each
paragraph's point in its first sentence, keep every number and citation intact,
and report a before/after breakdown of the structural changes.

Two rules of that pass override ordinary concision:

- **Load-bearing hedges stay.** `remains untested`, `supports no claim yet`,
  `as configured` record what the evidence cannot support. An edit that reads
  more confidently than the draft has failed, however much shorter it is.
- **Passive stays where the agent is the author.** `five recordings were scored`
  is correct register here; `we scored` is not an improvement.

Always report what was held back and why, not only what was cut.

### 6. Check before declaring it done

Run the checklist at the end of
[KMUTT-CITATION-GUIDE.md](references/KMUTT-CITATION-GUIDE.md), then:

- [ ] De-ramble pass run; nothing load-bearing cut
- [ ] Heading sizes 15/14/13, body 12, Times throughout
- [ ] Margins 4/3/2/2 cm; page number top right; chapter first pages bare
- [ ] Front matter in the manual's order, roman numerals from ii
- [ ] Every table and figure numbered `chapter.n`, captioned on the correct
      side, and present in its list
- [ ] Reproduced tables and figures carry both a citation number and a
      `(Source : …)` line
- [ ] Abstract record block complete; keywords sorted A–Z
- [ ] Acknowledgements **not** signed with the author's name
- [ ] Years in A.D. everywhere
- [ ] If LaTeX: the document builds, and `main.log` has no new warnings about
      overfull boxes in text you added

Report what you checked. Do not claim conformance you have not verified.

## Working in this repository

The volume already exists and is building. Treat it as the reference
implementation, not as a draft to restyle.

- `docs/report/final_report/main.tex` — the preamble *is* the format spec in
  LaTeX. Build with `latexmk -pdf main.tex` from that directory.
- `docs/report/final_report/chapters/references.tex` — hand-written
  `thebibliography` in KMUTT format. Add entries in the same shape; do not
  switch the volume to BibTeX. `references.bib` is a record only.
- `docs/report/final_report/frontmatter/` — cover, approval, abstract,
  acknowledgements, symbols, and the Thai abstract as an included PDF.

Three project rules that override anything general in this skill:

1. **Chapters 3 and 4 are assembled from fragments — never edit a fragment in
   place.** See the `final-report-build` memory note.
2. **Chapter 4 slots start as an insight note** in
   `docs/report/insights/experiment_N/`, drafted before the chapter prose.
3. Experiment claims, result-slot status labels and reporting rules belong to
   the `wil-thesis-evaluation` skill. When writing results, follow it for *what*
   may be claimed; follow this skill for *how* it is formatted. If the two
   collide, `wil-thesis-evaluation` decides the content question.

## Output

- **LaTeX** for anything joining `docs/report/final_report` — the format is
  already encoded there.
- **DOCX** only when a Word deliverable is asked for; start from the faculty
  part files in `docs/MyThesisInfo/27.8.67 Template เล่มโครงงาน (Eng Ver.)/` so
  the styles come along, rather than building a document from nothing.
- **Markdown** for drafts and notes that are not yet part of the volume.
