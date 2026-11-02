# KMUTT Citation and Reference Guide

From "6. การเขียนเอกสารอ้างอิง" and the citation section of the English
preparation manual (`เอกสารประกอบ_template ENG.pdf`). The manual's opening line
is binding: *"You must follow the following details in writing citations and
bibliography or references."*

**This is not IEEE.** The surface resemblance — a bracketed number — hides a
different entry format: the year comes second, the source title is bold, pages
are `pp.`, and volume and issue are spelled `Vol.` and `No.`. Never emit an
IEEE entry here.

## 1. In-text citation

Two systems are allowed. Choose one and use it consistently for the whole
volume.

### Number system (what this project uses)

A number in square brackets after the cited name or passage:

```
… as surveyed in [1].
… reported for both front ends [1, 2, 3].
```

- Several documents for the same passage: one bracket, numbers separated by
  commas.
- Once a number is assigned to a document it stays that document's number
  **throughout the thesis**.
- Numbering follows order of first citation.
- More than 3 authors → first author's last name + `et al.`:
  `Sugiura et al. [4] proposed that…`
- Exactly 2 authors → both last names: `Samit and Arter [5]`.

### Name–year system

Author and publication year at the end of the cited passage, years in the
Christian era:

```
Price, et al. (1999) found that …
```

- Multiple documents in one place are separated by semicolons.
- More than 3 authors → first author's last name, a comma, then `et al.`
- A direct quotation carries the page inside the parentheses:
  `Deuzen (1988, p. 29) argued that counselors must be involved …`
- A long quotation goes in its own paragraph, indented differently from
  ordinary text and set with tighter line spacing.

## 2. Reference list entries

Common shape: **author, year, title, source, location, pages.** The author is
`Surname, Initials.` with the initials run together (`Yeh, T.S.`), authors
separated by commas with `and` before the last. The **source title — book title
or journal name — is bold** (verified in the template's own character runs). In
the English edition the comma sits *inside* the closing quotation mark of an
article title.

### Book

```
Surname, Initials., year, <b>Title of Book</b>, edition, publisher, place, pp.
```

> Smiths, J.E., 1990, **Engineer**, 2nd ed., Amalin., Bangkok., pp. 120-150.

Editor rather than author: put `(Ed.)` after the name, `(Eds.)` for more than
one. `Hewitt, E.S. (Ed.), …`

### Article in a journal

```
Surname, Initials., year, "Title of Article," <b>Journal Name</b>, Vol. X, No. Y, pp.
```

> Yeh, T.S. and Sacks, M.D., 1988, "Effect of Particle Size Distribution on the
> Sintering of Alumina," **Journal of the American Ceramic Society**, Vol. 71,
> No. 12, pp. 484-487.

### Article in conference proceedings

```
Surname, Initials., year, "Title of Article," <b>Conference Name</b>, date, place, pp.
```

> Merin, U. and Daufin, G., 1989, "Separation Process Using Inorganic Membrane
> in the Food Industry," **International Conference on Inorganic Membranes**,
> 6 July 1989, Paris, pp. 272-278.

### Article in a book

```
Surname, Initials., year, "Title of Article," In <b>Book Title</b>, Editor, Initials. (Ed.), publisher, place, pp.
```

> Waechter, E.H., 1987, "How Families Cope : Accessing and Intervening," In
> **The Child and Family Facing Life-threatening Illness**, Krulik, T. (Ed.),
> Lippincott, Philadelphia, pp. 239-242.

### Article in a newspaper

```
Surname, Initials., year, "Title of Article," <b>Newspaper</b>, date, pp.
```

> Mena, N., 2014, "Ceramic Society of Japan," **Daily News**, May 6., pp. 3.

### Thesis

```
Surname, Initials., year, <b>Title of Thesis</b>, degree, major, faculty, university, pp.
```

> Mettripun, N., 2005, **Reduced VIC Code Length for Very Low Bit Rate Video
> Coding**, Master of Engineering Thesis, Computer Engineering, Faculty of
> Engineering, King Mongkut's University of Technology Thonburi, pp. 1-59.

### Patent

```
Patent owner, year granted, <b>Name of Invention</b>, country, patent number.
```

> Pinne, S., 2009, **Chemi Model**, Paris, 3421.352.110.

### Electronic document

```
Surname, Initials., year, <b>Title</b> [Online], Available : URL [year, Month day].
```

> Wu, K., **What is Nano** [Online], Available :
> http://www.nano.org.uk/nano.htm [2001, October 5].

An article read through a database takes `[Electronic]` and names the provider
instead of a URL:

> Peterson, S., 1999, "Growth of Penicillium Rogueforti," **Postharvest Biology
> and Technology** [Electronic], Vol. 17, No. 3, pp. 47-54, Available :
> Elsevier / Science Direct [2001, October 5].

The access date is `[year, Month day]` — year first, which is the opposite of
most styles. Keep the space before the colon in `Available :` as the template
sets it.

## 3. LaTeX form used in this project

`docs/report/final_report/chapters/references.tex` is a hand-written
`thebibliography`, in order of first citation, because no standard `.bst`
produces this layout. Match the existing entries exactly:

```latex
\bibitem{campos2021}
Campos, C., Elvira, R., G\'omez Rodr\'iguez, J.J., Montiel, J.M.M. and
Tard\'os, J.D., 2021, ``ORB-SLAM3: An Accurate Open-Source Library for Visual,
Visual-Inertial, and Multi-Map SLAM,'' \textbf{IEEE Transactions on Robotics},
Vol.~37, No.~6, pp.~1874--1890.
```

- `` `` `` and `''` for quotation marks, comma inside the closing pair.
- `\textbf{}` around the journal or book title.
- `Vol.~`, `No.~`, `pp.~` with non-breaking spaces; en-dash `--` in page ranges.
- Accented names written with LaTeX accents, not raw UTF-8, since the build is
  pdflatex.
- Cite with `\cite{key}`; the key is lowercase first author + year.

`scripts/kmutt_formatter.py` emits both the plain and the `\bibitem` form from
fielded input — use it rather than assembling entries by hand.

## 4. Checks before a build

- [ ] Every in-text number resolves to an entry, and every entry is cited.
- [ ] Numbers are in order of first citation and stable across chapters.
- [ ] Source titles bold; article titles quoted with the comma inside.
- [ ] `Vol.` / `No.` / `pp.` spelled that way, not `vol.`/`no.`/`p.`.
- [ ] Years in A.D. everywhere, including source lines under tables.
- [ ] Access dates as `[year, Month day]`.
- [ ] No IEEE-shaped entry (`A. Author, "Title," Journal, vol. 1, no. 2, 2020`)
      has crept in from a reference manager or a publisher's "cite" button.
