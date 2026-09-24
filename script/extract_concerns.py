#!/usr/bin/env python3
"""Collect every flagged passage from the minor report into a standalone PDF.

    python3 script/extract_concerns.py
    latexmk -pdf docs/report/concerns/concerns.tex

WHAT IT COLLECTS
    Two markers, both defined in minor_report/main.tex:

      \\begin{queried}[Label] ... \\end{queried}   a whole queried paragraph
      \\flag{...}                                 an inline queried fragment

    Their shared meaning, from main.tex: a result, number or condition that is
    measured but NOT yet trustworthy as evidence -- an unexplained mechanism, a
    confound, an unfair comparison, or a figure whose provenance does not support
    the use it is being put to. Deliberately not "bad result": a large error that
    is correctly measured and understood is a finding, not a concern.

WHY A SEPARATE DOCUMENT
    In the report each flag sits beside the claim it qualifies, which is where it
    belongs -- a caveat separated from its result is how results get misquoted.
    This document does not remove them from the report; it COPIES them into one
    place so the whole set can be read as a checklist. Before a review it answers
    "what do I not yet stand behind?" in one pass instead of 68 pages.

CONTEXT
    An inline \\flag{6.481} means nothing alone, so each one is quoted inside its
    surrounding sentence with the flagged span still in red. Every entry records
    the chapter and section it came from, by title rather than by \\cref, so this
    document needs no cross-reference machinery and cannot go stale against the
    report's numbering.

    \\cref and \\ref inside an extracted body are rendered as a muted [label]:
    the target lives in the report, not here, and a dangling reference would
    either fail to compile or print a wrong number.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(ROOT, 'docs', 'report', 'minor_report')
CHAPTERS = [('Chapter 3 --- Methodology', 'chapters/chapter_3_methodology.tex'),
            ('Chapter 4 --- Results', 'chapters/chapter_4_results.tex')]
OUT_DIR = os.path.join(ROOT, 'docs', 'report', 'concerns')

# The colour legend is meta, not a concern; it is reproduced as the preamble
# instead of listed as an item.
SKIP_LABELS = {'Reading the colours'}


def match_brace(text, open_idx):
    """Index just past the '}' closing the '{' at open_idx. Brace-counting, so a
    flagged span containing its own groups survives."""
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == '{' and (i == 0 or text[i - 1] != '\\'):
            depth += 1
        elif text[i] == '}' and text[i - 1] != '\\':
            depth -= 1
            if depth == 0:
                return i + 1
    raise ValueError('unbalanced brace at %d' % open_idx)


def headings(text):
    """(position, level, title) for every section and subsection."""
    out = []
    for m in re.finditer(r'\\(section|subsection)\*?\{', text):
        end = match_brace(text, m.end() - 1)
        title = text[m.end():end - 1]
        out.append((m.start(), m.group(1), title))
    return out


def context_of(pos, heads):
    """The section and subsection in force at a position."""
    sec = sub = None
    for p, level, title in heads:
        if p > pos:
            break
        if level == 'section':
            sec, sub = title, None
        else:
            sub = title
    return sec, sub


def sentence_around(text, start, end):
    """Expand a span to its surrounding sentence, so a bare number reads."""
    lo = start
    while lo > 0:
        if text[lo - 1] == '\n' and text[lo - 2:lo - 1] == '\n':
            break
        if text[lo - 1] == ' ' and text[lo - 2:lo - 1] == '.' and \
                not text[lo - 4:lo - 1].endswith(('e.g', 'i.e')):
            break
        lo -= 1
    hi = end
    while hi < len(text):
        if text[hi] == '.' and (hi + 1 >= len(text) or text[hi + 1] in ' \n'):
            hi += 1
            break
        if text[hi] == '\n' and text[hi + 1:hi + 2] == '\n':
            break
        hi += 1
    return text[lo:start].strip(), text[start:end], text[end:hi].strip()


TABLE_ENVS = ('tabular', 'tabularx', 'longtable', 'table')


def in_table(text, pos):
    """True when pos sits inside a tabular-like environment.

    A flag in a table cell cannot be quoted as a sentence: expanding outward hits
    '&', '\\\\' and eventually '\\end{table}', and an unbalanced environment
    fragment will not compile. Those are extracted by ROW instead.
    """
    depth = 0
    for m in re.finditer(r'\\(begin|end)\{(\w+)\}', text[:pos]):
        if m.group(2) in TABLE_ENVS:
            depth += 1 if m.group(1) == 'begin' else -1
    return depth > 0


def row_around(text, start, end):
    """The table row containing a span, as readable text."""
    lo = text.rfind('\\\\', 0, start)
    lo = 0 if lo < 0 else lo + 2
    hi = text.find('\\\\', end)
    hi = len(text) if hi < 0 else hi
    row = text[lo:hi]
    for junk in ('\\midrule', '\\toprule', '\\bottomrule', '\\hline',
                 '\\endfirsthead', '\\endhead'):
        row = row.replace(junk, ' ')
    row = re.sub(r'\\(begin|end)\{\w+\}(\[[^]]*\])?(\{[^}]*\})*', ' ', row)
    cells = [c.strip() for c in row.split('&') if c.strip()]
    return ' \\textbar{} '.join(cells)


def balanced(frag):
    """Drop a fragment carrying an unmatched \\begin or \\end."""
    opens = len(re.findall(r'\\begin\{', frag))
    closes = len(re.findall(r'\\end\{', frag))
    return opens == closes


def clean(body):
    """Neutralise anything whose target lives in the report, not here.

    Cross-references have no target in this document, and figure inputs resolve
    against the report's directory, so both are printed as muted markers instead.
    """
    # Figures first: \resizebox{..}{..}{\input{..}} must lose the \input before
    # the brace-matching below sees a path it cannot resolve.
    body = re.sub(r'\\includegraphics(\[[^]]*\])?\{([^}]*)\}',
                  r'\\reportfigure{\2}', body)
    body = re.sub(r'\\input\{([^}]*)\}', r'\\reportfigure{\1}', body)
    body = re.sub(r'\\[Cc]refs?\{([^}]*)\}', r'\\reportref{\1}', body)
    body = re.sub(r'\\ref\{([^}]*)\}', r'\\reportref{\1}', body)
    body = re.sub(r'\\label\{[^}]*\}', '', body)
    return body.strip()


def harvest(path):
    text = open(path).read()
    heads = headings(text)
    items, spans = [], []

    for m in re.finditer(r'\\begin\{queried\}(\[([^]]*)\])?', text):
        end = text.index('\\end{queried}', m.end())
        label = (m.group(2) or 'Queried').strip()
        spans.append((m.start(), end))
        if label in SKIP_LABELS:
            continue
        sec, sub = context_of(m.start(), heads)
        items.append(dict(kind='block', label=label, sec=sec, sub=sub,
                          body=clean(text[m.end():end]), pos=m.start()))

    # \moved marks a queried sentence that was lifted OUT of the report body --
    # main.tex renders it as nothing, so this document is the only place it
    # appears. It is otherwise identical to \flag and is extracted the same way.
    for m in re.finditer(r'\\(?:flag|moved)\{', text):
        if any(a <= m.start() <= b for a, b in spans):
            continue  # already inside a queried block
        end = match_brace(text, m.end() - 1)
        sec, sub = context_of(m.start(), heads)
        if in_table(text, m.start()):
            items.append(dict(kind='row', label=None, sec=sec, sub=sub,
                              before='', span=clean(row_around(text, m.start(), end)),
                              after='', pos=m.start()))
            continue
        before, span, after = sentence_around(text, m.start(), end)
        if not (balanced(before) and balanced(after)):
            before, after = '', ''   # keep the flag, drop unsafe context
        items.append(dict(kind='inline', label=None, sec=sec, sub=sub,
                          before=clean(before), span=span,
                          after=clean(after), pos=m.start()))

    items.sort(key=lambda d: d['pos'])
    return items


def report_preamble():
    """Everything main.tex declares before \\begin{document}, minus its identity.

    The report's preamble is the single source of truth for \\flag, \\status,
    \\repo, the queried environment, the status colours and any macro added
    later -- \\perfmark was added after this script was first written and broke
    it. Inheriting rather than restating means a new macro in the report cannot
    silently break this document.

    Two adjustments are needed. \\title, \\author and \\date are removed by
    brace matching, not by line, because main.tex's title spans several lines and
    a line filter leaves the tail behind as stray body text. And the preamble
    itself \\inputs a TikZ definitions file by relative path, so \\input@path and
    \\graphicspath are pointed back at the report directory.
    """
    src = open(os.path.join(REPORT, 'main.tex')).read()
    pre = src[:src.index('\\begin{document}')]
    pre = re.sub(r'(?m)^\\documentclass.*$', '', pre)
    for cmd in ('title', 'author', 'date'):
        while True:
            m = re.search(r'\\%s\s*\{' % cmd, pre)
            if not m:
                break
            pre = pre[:m.start()] + pre[match_brace(pre, m.end() - 1):]
    # \\input@path must precede the preamble's own \\input; \\graphicspath must
    # follow it, since graphicx is loaded inside the preamble.
    before = ('\\makeatletter\n\\def\\input@path{{%s/}}\n\\makeatother\n'
              % REPORT)
    after = '\\graphicspath{{%s/}}\n' % REPORT
    return before + pre + after


PREAMBLE_HEAD = r"""\documentclass[11pt,a4paper]{article}
% Generated by script/extract_concerns.py -- do not edit by hand.
% The preamble below is inherited verbatim from minor_report/main.tex so that
% every macro the extracted passages use is defined the same way here.
"""

PREAMBLE_TAIL = r"""
\geometry{margin=22mm}
\usepackage{titlesec}

% The inherited preamble SUPPRESSES queried material, because the report no
% longer renders it -- that is the whole point of this document existing. Undo
% the suppression here, or every passage extracted below would be typeset into
% nothing and this document would silently come out empty of the very text it
% was built to carry.
\renewcommand{\flag}[1]{\textcolor{flagred}{#1}}
\renewcommand{\moved}[1]{\textcolor{flagred}{#1}}
% Cross-reference targets live in the report, so a reference is printed, not
% resolved -- a dangling \cref would either fail or print a wrong number.
\newcommand{\reportref}[1]{\textcolor{gray}{[\texttt{\detokenize{#1}}]}}
\newcommand{\reportfigure}[1]{\textcolor{gray}{[figure: \texttt{\detokenize{#1}}]}}
\titlespacing*{\section}{0pt}{2.2ex plus .2ex}{0.8ex}
\titlespacing*{\subsection}{0pt}{1.6ex plus .2ex}{0.5ex}

\title{Open Concerns}
\author{Bolton Athit Davies}
\date{@@DATE@@}

\begin{document}
\maketitle
\thispagestyle{empty}

\noindent
Every passage the minor progress report marks as \flag{queried}, collected into
one list. @@COUNT@@ items, from @@SOURCE@@.

\medskip
\noindent
A flag marks a result, number or condition that is \emph{measured but not yet
trustworthy as evidence} --- an unexplained mechanism, a confound, an unfair
comparison, or a figure whose provenance does not support the use it is being put
to. It deliberately does not mark a bad result: a large error that is correctly
measured and understood is a finding, not a concern.

\medskip
\noindent
These passages remain in the report beside the claims they qualify, which is
where they belong. This document copies rather than moves them, so the whole set
can be read as a checklist. References such as \reportref{sec:example} point into
the report.

\medskip
\hrule
"""



def emit(all_items, stamp, sources):
    n = sum(len(v) for v in all_items.values())
    head = ((PREAMBLE_HEAD + report_preamble() + PREAMBLE_TAIL).replace('@@DATE@@', stamp)
                    .replace('@@COUNT@@', str(n))
                    .replace('@@SOURCE@@', sources))
    out = [head]
    for chapter, items in all_items.items():
        if not items:
            continue
        out.append('\n\\section*{%s}\n' % chapter)
        last = None
        for i, it in enumerate(items, 1):
            where = it['sub'] or it['sec'] or ''
            if where != last:
                out.append('\n\\subsection*{%s}\n' % where)
                last = where
            if it['kind'] == 'row':
                out.append('\\noindent\\textbf{%d.}\\quad \\emph{table row:} %s\n\n'
                           % (i, it['span']))
            elif it['kind'] == 'block':
                out.append('\\noindent\\textbf{%d.\\ \\flag{[%s]}}\\quad %s\n\n'
                           % (i, it['label'], it['body']))
            else:
                body = ' '.join(x for x in (it['before'], it['span'], it['after']) if x)
                out.append('\\noindent\\textbf{%d.}\\quad %s\n\n' % (i, body))
            out.append('\\medskip\n')
    out.append('\n\\end{document}\n')
    return ''.join(out)


def main():
    import datetime
    all_items, total = {}, 0
    for title, rel in CHAPTERS:
        path = os.path.join(REPORT, rel)
        if not os.path.isfile(path):
            print('missing %s' % path, file=sys.stderr)
            return 1
        items = harvest(path)
        all_items[title] = items
        total += len(items)
        blocks = sum(1 for i in items if i['kind'] == 'block')
        print('%-28s %2d blocks + %2d inline = %2d'
              % (rel.split('/')[-1], blocks, len(items) - blocks, len(items)))

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.date.today().strftime('%d %B %Y')
    target = os.path.join(OUT_DIR, 'concerns.tex')
    open(target, 'w').write(emit(all_items, stamp,
                                 'the minor progress report'))
    print('wrote %s (%d items)' % (os.path.relpath(target, ROOT), total))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
