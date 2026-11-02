# De-Rambling Pass

Act as an elite academic editor: filter, streamline and de-ramble the draft.
This pass runs on request and on any draft before it is called finished. It
changes prose only — never the format, the numbers or the claims.

## The five rules

### 1. Cut the clutter

- Delete repetitive arguments. A point made in section 4.2 is not re-made in
  4.3; a value stated once is not restated two paragraphs later.
- Remove adverbs and adjectives that add no information: *significantly*
  (unless a test was run), *very*, *quite*, *extremely*, *clearly*,
  *interestingly*, *it is important to note that*, *as can be seen*.
- Prefer the active voice **where the agent is a thing in the system** — the
  detector, the filter, the robot, the estimator. `The detector masks features
  belonging to movable objects` beats `features belonging to movable objects
  are masked by the detector`.
- **Keep the passive where the agent is the author.** Academic register bars
  *I* and *we* here, so `five recordings were scored` is correct and
  `we scored five recordings` is not an improvement — it is a register error.
  Keep the passive too where the object is the topic of the paragraph and
  moving it would break the topic chain.

### 2. Shorten sentences

Break multi-clause sentences into direct statements. A sentence carrying three
subordinate clauses is usually three findings wearing one coat. Target: one
claim per sentence, the subject near the verb, the verb near its object.

Keep a long sentence when its length *is* the logic — a conditional with its
qualification, or a list whose items must be read together. Length is not the
defect; lost thread is.

### 3. Enhance scannability

**The first sentence of a paragraph states that paragraph's point.** Not its
topic, not its setup — its point. If the point only arrives at the end, move it
to the front and cut whatever was keeping the reader waiting.

Then check the transitions: each paragraph should follow from the one before by
something other than adjacency. If two paragraphs could swap places without
damage, at least one of them is not making a point.

### 4. Maintain rigor

Technical precision, data and citations survive the edit intact. Specifically,
**never** alter:

- Any measured value, unit, tolerance or range, including `\SI{}`,
  `\SIrange{}` and `\num{}` arguments.
- Citation keys, `\cite{}`, `\cref{}`, `\label{}`, or a float's number.
- Table and figure contents, captions' numbering, or `(Source : …)` lines.
- This project's markup: `\flag{}`, `\moved{}`, the `queried` environment,
  `\slotstatus{}`, `\status{}`, `\repo{}`. These are a record, not prose.
  `script/extract_concerns.py` parses the chapter source for them.

**Load-bearing hedges are not clutter.** This is the rule that matters most
here, because concision pressure attacks exactly the words that record what the
evidence cannot support. A hedge stays when it encodes a validity limit:

| Keep — states a limit | Cut — decorates |
|---|---|
| `remains untested` | `it may perhaps be the case that` |
| `supports no claim yet` | `it is interesting to note that` |
| `scored provisionally` | `it could possibly be argued` |
| `were not input-matched` | `somewhat`, `rather`, `fairly` |
| `as configured` | `in a sense` |
| `against projected simulator ground truth` | `basically`, `essentially` |

Stacked hedges collapse to the strongest single one: *may possibly suggest that
it could be* → *suggests*. But *suggests* never becomes *shows*, and
*untested* never becomes *weak*. **A de-ramble that strengthens a claim has
failed, however much shorter it is.** What a result may claim is governed by
the `wil-thesis-evaluation` skill; this pass may not overrule it.

### 5. Report before vs. after

Give a breakdown of the major structural changes, so the edit can be audited
rather than trusted. One row per change worth seeing — not per comma:

| # | Where | Before | After | Cut | Why |
|---|---|---|---|---|---|
| 1 | §4.3 ¶2 | 68 words, point in final clause | 31 words, point first | 2 hedges, 1 restatement | Paragraph argued toward its claim instead of opening with it |

Then two short lists, both required:

- **Held back** — what looked like clutter and was kept, with the reason.
  Every load-bearing hedge left in place belongs here. This list is how the
  author checks that nothing load-bearing went out with the trimming.
- **Flagged, not cut** — passages that ramble because the underlying thought is
  unresolved. Editing cannot fix those; name them and leave them.

Report the word count before and after. Do not pad the count by cutting
material the author needs.

## Procedure

Work one paragraph at a time, in order:

1. **Point test.** Read the first sentence alone. Does it state the paragraph's
   point? If not, find the sentence that does and promote it.
2. **Deletion test.** Delete each sentence in turn. If nothing is lost, it goes.
   Apply the same test to every adverb.
3. **Repetition scan.** Does this paragraph restate an earlier one, or quote a
   number already quoted in this slot? Keep the first statement, cut the echo,
   cross-reference instead with `\cref{}`.
4. **Hedge audit.** For each hedge: limit or decoration? Use the table above.
   Collapse stacks; keep limits verbatim.
5. **Voice check.** Agent is a system component → active. Agent is the author →
   leave passive.
6. **Thread check.** Re-read the paragraph openers in sequence. They should read
   as an argument on their own.

## In LaTeX

- Edit prose, not layout. Do not reflow a table, change column specs, or touch
  the preamble.
- Keep `~` non-breaking spaces (`Vol.~37`, `\cref{tab:x}`) and `--` en-dashes.
- A sentence split must not orphan a `\label{}` from its `\caption{}`.
- Source lines wrap under the caption text; leave their indentation alone.
- **Chapters 3 and 4 are assembled from fragments — never edit a fragment in
  place.** Edit at the level the build composes, as `final-report-build`
  requires. If a de-ramble would mean touching a fragment, say so and stop.

## Scope limits

- Rewriting prose is in scope. Reordering sections, merging chapters or dropping
  a result is not — propose it, do not do it.
- The abstract has its own constraint: the Thai abstract must not be a literal
  translation of the English one, and the manual bars passive voice there. A
  de-ramble of the English abstract does not propagate to the Thai page.
- Quotations are never edited. A long quotation keeps its indentation and its
  tighter spacing.
