---
name: report-analysis-and-evaluation
description: Read a document harder than a summary does, through ten fixed analytical lenses - key information, hidden insight, data and statistics, contradictions, missing information, risk, critical thinking, opportunity, a 30-day action plan, and a decision brief. Every finding is tied to a page and separates what the document says from what the analysis infers. Use when asked to analyse, evaluate, audit, critique, fact-check or extract decisions from a PDF, report, paper or chapter, in Thai or English.
---

# Report Analysis and Evaluation

Ten lenses for reading a document properly. A summary tells you what a document
says; these tell you what it means, what it is missing, where it contradicts
itself, and what to do about it.

The modes are fixed and each has a fixed output schema. Do not invent an
eleventh, and do not quietly merge two.

| # | Mode | Thai | Answers |
|---|---|---|---|
| 1 | Key Information | ดึงข้อมูลสำคัญ | What must I not miss? |
| 2 | Hidden Insight | Insight ที่ซ่อนอยู่ | What does it imply but never state? |
| 3 | Data Detective | ตัวเลขและสถิติ | What do the numbers actually say? |
| 4 | Contradiction Finder | ข้อมูลที่ขัดแย้งกัน | Where does it disagree with itself? |
| 5 | Missing Information | ข้อมูลที่ยังไม่บอก | What is absent that I need? |
| 6 | Risk and Problem | ปัญหาและความเสี่ยง | What could go wrong? |
| 7 | Critical Thinking | ตรวจสอบข้อสรุป | Do the conclusions follow from the evidence? |
| 8 | Opportunity | ค้นหาโอกาส | What could be built or improved from this? |
| 9 | Action Plan | แผนปฏิบัติการ | What do I do in the next 30 days? |
| 10 | Decision Brief | สรุปเพื่อตัดสินใจ | What do I decide, and on what basis? |

Full output schema for every mode: [MODES.md](references/MODES.md).
The evidence discipline every mode obeys: [EVIDENCE-RULES.md](references/EVIDENCE-RULES.md).

## The spine: evidence discipline

These five rules outrank every instruction below them. They are what separates
this skill from an articulate guess.

1. **Never invent.** Not a number, not a page, not a quotation, not a source.
   If a mode asks for ten items and the document supports six, give six and say
   so. A padded list is a failed analysis.
2. **Cite where it came from.** Every finding carries a page or section. An
   uncited finding is an opinion.
3. **Separate fact from inference.** Mark each item as *document* (stated
   there) or *analysis* (concluded here). Modes 2, 6, 8, 9 and 10 mix the two
   by design, so the mark is mandatory in those.
4. **An omission is not an error.** Mode 5 finds what is absent. Absence is a
   gap to fill, not proof that the document is wrong, unless the document
   claimed to cover it.
5. **A clean result is a result.** If mode 4 finds no contradiction, say so
   plainly. Do not manufacture a defect to look thorough.

## Choosing a mode

Ask what the person will do with the answer.

- They must act, decide or brief someone → **10**, then **9** if they want the
  steps.
- They distrust the document, or it is someone else's work under review →
  **4** and **7**, then **5**.
- They own the document and want it stronger → **5**, **7**, **4**.
- They are mining it for ideas → **2**, **8**.
- They need the numbers → **3**.
- They have not read it yet → **1**, and stop there until they ask for more.

Run one or two modes, not ten. Ten modes over one document is a wall of text
nobody reads. If the request is broad ("analyse this"), run **1**, say which
other modes would pay off, and let them pick.

## Reading the document

**Prefer the text layer.** `pdftotext -layout file.pdf -` gives the whole
document with the table columns intact. The Read tool with `pages` renders
pages visually and is the better choice when the finding depends on a figure,
a layout or a scanned page.

**Page numbers are a trap, and this repository proves it.** In
`docs/report/final_report/main.pdf` the front matter is numbered in roman, so
**PDF page 100 is printed page 86**, a 14-page offset, and a chapter's first
page carries no printed number at all. Always cite the **printed** page the
reader will see, and say so when it differs from the PDF page. Verify the
offset on the document in front of you; never assume it.

**A table is evidence; its caption is a claim about the table.** When a mode
quotes a number, read it from the table, not from the prose that describes it.
Mode 4 exists largely because those two disagree.

## Working in this repository

The documents worth analysing here are the final report
(`docs/report/final_report/main.pdf`), the Open Concerns document
(`docs/report/concerns/concerns.pdf`), the per-experiment insight notes
(`docs/report/insights/experiment_N/`), the progress-update decks
(`docs/report/insights/update_*/`), and the literature in `docs/references/`.

Three rules apply when the document is the project's own work:

1. **`wil-thesis-evaluation` decides what may be claimed about an experiment.**
   This skill decides how to *read* the document. If an analysis seems to
   establish a new experimental result, it has not: it has found a question to
   put to the data. Say that, and name the run or table that would settle it.
2. **The Open Concerns document is the project's own mode 5 and mode 7.** Read
   it before reporting a gap or a weak conclusion, so the analysis adds to it
   instead of rediscovering it. An item already flagged there is confirmation,
   worth one line, not a finding.
3. **Status labels are evidence.** A slot marked *untested* or *planned* is not
   a weakness the analysis uncovered; it is a disclosure the document already
   made. Treat it as a fact about the project's state, not a criticism.

## Output language

Match the person. The ten prompts came from a Thai source: if they write in
Thai, answer in Thai with the technical terms left in English, which is how the
project's own speaker notes read. Keep quoted document text in its original
language whatever the surrounding language, and never translate a number, a
label or a file path.

## Before handing it over

- [ ] Every finding carries a page or section, and the page is the printed one
- [ ] Every item is marked *document* or *analysis*
- [ ] Nothing invented: each number, name and quotation traced back to the text
- [ ] The list is as long as the evidence allows, not as long as the mode asks
- [ ] Items ranked by importance or feasibility, as the mode requires
- [ ] A null result, where one was found, is stated plainly
- [ ] For a project document: nothing claimed that `wil-thesis-evaluation`
      would not allow, and nothing re-reported that Open Concerns already holds

State what was checked. Do not claim coverage that was not achieved: "I read
chapters 3 and 4, not the appendices" is a better report than silence.
