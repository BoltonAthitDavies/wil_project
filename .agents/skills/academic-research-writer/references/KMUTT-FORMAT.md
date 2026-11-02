# KMUTT Volume Format (Faculty of Engineering, English Edition)

Authoritative sources, in order of precedence:

1. `docs/MyThesisInfo/27.8.67 Template เล่มโครงงาน (Eng Ver.)/เอกสารประกอบ_template ENG.pdf`
   — "Thesis/Dissertation Preparation Manual", Faculty of Engineering, KMUTT,
   August 2015. This is the English spec and carries the annotated page setup.
2. The six `.doc`/`.docx` part files beside it (cover, inner cover and approval,
   abstract, acknowledgements, body, references, tables and figures).
3. `docs/MyThesisInfo/1. คู่มือการเขียนวิทยานิพนธ์-cr.มจธ.pdf` — the full Thai
   manual, for anything the English spec leaves open.

Every number below is taken from those files. Do not substitute IEEE, APA or
journal-style conventions for any of them.

## 1. Page setup

| Property | Value |
|---|---|
| Paper | A4 (210 × 297 mm) |
| Left margin | **4 cm** (binding edge) |
| Top margin | **3 cm** |
| Right margin | **2 cm** |
| Bottom margin | **2 cm** |
| Page number | Top right |
| First page of a chapter | No page number printed |
| Front matter numbering | Lowercase roman, abstract starts at ii |
| Body numbering | Arabic, restarts at 1 in Chapter 1 |
| Sides | Single-sided |

The manual states these as a labelled diagram ("Figure 1.2 Preview Page Setup":
3 cm top, 4 cm left, 2 cm right, 2 cm bottom). The Word template carries the
same values as `w:pgMar left="2268" top="1701" right="1134" bottom="1134"`
twips, which is 4/3/2/2 cm exactly.

## 2. Type

**"Font Times New Roman 12 pt. at all places."** — the manual's own annotation
on the cover page. The one exception is the Thai abstract page, annotated
**"Font AngsanaUPC 16 pt. ทั้งหน้า"** (whole page).

| Element | Size | Weight |
|---|---|---|
| `CHAPTER n TITLE` | 15 pt | bold, centred, all caps |
| `1.1 Head Topic` | 14 pt | bold |
| `1.1.1 Sub Topic` | 13 pt | bold |
| Body text | 12 pt | regular |
| Front-matter headings (`CONTENTS`, `LIST OF TABLES`, …) | 15 pt | bold, centred |

Numbered sub-items below a subsection are dashes (`-`), indented, with
continuation lines aligned under the first word.

## 3. Order of the volume

| # | Part | Page |
|---|---|---|
| 1 | Outer cover, English, all caps (title / author / degree block / faculty / KMUTT / year) | unnumbered |
| 2 | Inner cover + approval page (committee signature lines, then `Copyright reserved`) | unnumbered |
| 3 | `ABSTRACT` (English) | ii |
| 4 | `THAI ABSTRACT` (บทคัดย่อ, AngsanaUPC 16 pt) | iii |
| 5 | `ACKNOWLEDGEMENTS` | iv |
| 6 | `CONTENTS` | v |
| 7 | `LIST OF TABLES` | vi |
| 8 | `LIST OF FIGURES` | vii |
| 9 | `LIST OF SYMBOLS` (where used) and `LIST OF TECHNICAL VOCABULARY AND ABBREVIATIONS` | viii |
| 10 | `CHAPTER 1 INTRODUCTION` | 1 |
| 11 | `CHAPTER 2` — theoretical issue / related work / background | — |
| 12 | `CHAPTER 3`, `CHAPTER 4` — own titles | — |
| 13 | `CHAPTER 5 CONCLUSION AND RECOMMENDATIONS` | — |
| 14 | `REFERENCES` | — |
| 15 | `APPENDIX A`, `B`, `C` … (omit the part entirely if there is none) | — |
| 16 | `CURRICULUM VITAE` | last |

Continuation pages of the contents and the lists repeat their heading with
`(Cont'd)` and repeat the column labels: `PAGE` for contents, `TABLE`/`PAGE`
and `FIGURE`/`PAGE` for the lists.

## 4. The abstract page

A record block, then the centred word `Abstract`, then the text, then keywords:

```
Thesis Title     <title, wrapping under itself>
Thesis Credits   12 or 6
Candidate        Mr./Mrs./Miss <Name Surname>
Thesis Advisors  <Title Dr. Name Surname>
                 <co-advisor on the next line>
Program          Master of Engineering
Field of Study   <field>
Department       <department>
Faculty          Engineering
Academic Year    <year of the examination>
```

- `Keywords:` are separated by ` / ` and **sorted A–Z**.
- For a bachelor's final project the labels read `Project Title`,
  `Project Credits`, `Project Advisor`; credits are 6.
- Academic Year is the year of the examination, not of submission.

Thai abstract page: same block in Thai (`หัวข้อวิทยานิพนธ์`, `หน่วยกิต`,
`ผู้เขียน`, `อาจารย์ที่ปรึกษา`, `หลักสูตร`, `สาขาวิชา`, `ภาควิชา`, `คณะ`,
`ปีการศึกษา`), heading `บทคัดย่อ`, keywords `คำสำคัญ :` sorted ก–ฮ. The manual
is explicit that the Thai abstract must **not** be a literal translation of the
English one and must avoid passive voice, and that **no English words** may
appear in the Thai keywords. Note `วิศวกรรมศาสตร` — no `์` — in degree names.

## 5. Acknowledgements

Free text. The template says explicitly: **do not sign the author's name at the
end** (`ไม่ต้องลงท้ายชื่อผู้เขียน`).

## 6. Tables and figures

Numbered `chapter.sequence` (1.1, 1.2, 2.1 …), independently for tables and
figures, and every one of them is listed in `LIST OF TABLES` / `LIST OF FIGURES`.

**Tables — caption above:**

```
Table 1.1  Name of table
           Name of table (if it runs over one line) [5]
           (Source : <author>, <year in A.D.>)
```

A table continuing onto the next page repeats its caption with `(Continue)`
appended.

**Figures — caption below, centred:**

```
Figure 1.1  Name of figure [6]
            (Source : <author>, <year in A.D.>)
```

Rules that apply to both:

- A table or figure reproduced or adapted from a source carries the citation
  number in its caption **and** a `(Source : …, <year>)` line beneath it.
- Years in a source line are given in A.D., never B.E.
- Refer to every table and figure from the body text by number.

## 7. Curriculum vitae

A label/value layout, labels in caps:

```
NAME                    Mr. <Name Surname>
Date of Birth           <d Month yyyy>
EDUCATIONAL RECORD
HIGH SCHOOL             <school, year>
BACHELOR'S DEGREE       Bachelor of Engineering (<field>)
                        King Mongkut's University of Technology Thonburi, <year>
MASTER'S DEGREE         <as above, if applicable>
SCHOLARSHIP/            <grant, body, year>
RESEARCH GRANT
EMPLOYMENT RECORD       <role, employer, years>
PUBLICATION             <full reference in the KMUTT format of §2 of
                         KMUTT-CITATION-GUIDE.md>
```

## 8. How this project implements the format

`docs/report/final_report/` is a working LaTeX realisation of everything above.
Read it before inventing anything:

- `main.tex` — the preamble is the format. `\geometry{left=40mm, right=20mm,
  top=30mm, bottom=20mm}`, `mathptmx` for Times, `\titleformat` for the
  15/14/13 pt bold headings, `fancyhdr` for the top-right number and the blank
  first chapter page, `tocloft` for the `(Cont'd)` headings and `PAGE` labels,
  `\captionsetup[table]{position=above}` / `[figure]{position=below}`.
- `frontmatter/` — `cover`, `approval`, `abstract`, `acknowledgements`,
  `symbols`, and `thai_abstract.fodt` → `thai_abstract.pdf` included as a page.
- `chapters/` — `chapter_1_introduction` … `chapter_5_conclusion`,
  `references.tex`, `curriculum_vitae.tex`.
- `chapters/references.tex` — a hand-written `thebibliography`, **not** BibTeX:
  no standard `.bst` produces the KMUTT layout. `references.bib` is kept only
  as a record.
- Build with `latexmk -pdf main.tex` from inside `docs/report/final_report`.

Two project-specific constraints that override any general writing advice:

- The Chapter 3 and 4 bodies are assembled from fragments. **Never edit a
  fragment in place** — see the `final-report-build` note in memory.
- Chapter 4 slots are drafted as an insight note in
  `docs/report/insights/experiment_N/` before the chapter prose is written.
