# Evidence rules

The ten modes differ in what they look for. They do not differ in how they
handle evidence. This file is the shared discipline; `SKILL.md` states the five
rules in short, and this expands each into something testable.

---

## 1. Never invent

The failure this guards against is fluent and plausible output that no page
supports. It is the only failure in this skill that cannot be noticed by
reading the analysis alone, which is why it comes first.

Concretely, never produce:

- a number the document does not contain, including one "implied" by rounding
  or by a chart read by eye without saying so;
- a page reference that was not checked;
- a quotation reconstructed from memory of the document;
- a citation, DOI or author list for a work the document only mentions in
  passing;
- an eleventh item because the mode asks for ten.

When a mode's schema demands a field the document cannot fill, write
`not stated` in that field. A schema with honest gaps is more useful than a
complete-looking one that cannot be trusted anywhere.

**Reading a chart.** A value taken off a plotted curve is an estimate. Mark it
as one: "about 0.4 m, read from figure 4.6" is honest; "0.42 m" is not.

---

## 2. Cite where it came from

Every finding carries a page or a section. The reader's test is whether they
can reach the same place in ten seconds.

**Printed page, not PDF page.** A document with roman front matter numbers its
body separately. In this repository's final report, PDF page 100 is printed
page 86. Cite the printed number the reader sees on the paper, and when the two
differ, give both once: "printed p. 86 (PDF p. 100)".

**Pages that carry no number.** A chapter's first page is unnumbered in this
format. Cite it by its chapter and section instead: "chapter 4 opening", not a
guessed number.

**Prefer the smallest locator that works.** Table 4.14 beats page 86; equation
3.2 beats section 3.7. A locator that survives repagination is worth more than
one that does not.

---

## 3. Separate fact from inference

Every item is marked:

- **document** — stated in the text, quotable, with a page;
- **analysis** — concluded by joining two or more places, or by applying
  outside knowledge.

Modes 2, 6, 8, 9 and 10 exist to produce *analysis* items; the mark is what
keeps them honest. Modes 1, 3 and 4 should be almost entirely *document*; if
they are not, the analysis has drifted into speculation and should be pulled
back.

An inference that rests on a second inference is worth stating only if both
links are shown. Two-step chains are where confident nonsense is manufactured.

---

## 4. An omission is not an error

Mode 5 looks for absence. Absence has three different meanings and they must
not be conflated:

| What you found | What it means |
|---|---|
| The document promised it and did not deliver | A defect. Report it as one. |
| The document never claimed to cover it | A gap in *your* knowledge, not a fault in the document. |
| The document says it is unavailable, untested or planned | A disclosure. Credit it; do not re-report it as a discovery. |

The third case matters most for a project's own documents, where status labels
and an Open Concerns document already record what is not known.

---

## 5. A clean result is a result

If mode 4 finds no contradiction, mode 6 no serious risk, or mode 7 no
unsupported conclusion, say so in one plain sentence and stop. Do not inflate a
rounding artefact, a stylistic inconsistency or a known limitation into a
finding.

The temptation is strongest when a mode promises ten items. Resist it: the
value of this skill collapses the first time a reader checks a finding and
discovers it was filler.

---

## Ranking

Modes 1, 5 and 6 rank by importance; mode 8 ranks by feasibility. Say what the
ranking is based on in one line before the list, because "important" means
different things to an author, a reviewer and a decision-maker.

Where two items are genuinely of equal weight, say so rather than imposing a
false order.

---

## Reporting coverage

Close every analysis with what was actually read: which chapters, which pages,
and what was skipped. An analysis of a 135-page report that silently read 30
pages is misleading even when every individual finding is correct.
