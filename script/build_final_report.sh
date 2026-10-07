#!/usr/bin/env bash
# Build the final report and the Open Concerns document.
#
#     script/build_final_report.sh
#
# Three steps, because the report has one page pdflatex cannot set:
#   1. LibreOffice typesets frontmatter/thai_abstract.fodt (the Thai abstract,
#      Norasi 16 pt) to thai_abstract.pdf, which abstract.tex places with
#      pdfpages.  pdflatex has no Thai font on this host and LuaLaTeX's font
#      database is broken, so this is the one dependable route to a Thai page.
#   2. latexmk builds docs/report/final_report/main.pdf.
#   3. script/extract_concerns.py regenerates docs/report/concerns/concerns.tex
#      from the report source (the report suppresses its queries and status
#      boxes; the concerns document is where they are read) and latexmk builds
#      it.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPORT="$ROOT/docs/report/final_report"
CONCERNS="$ROOT/docs/report/concerns"

echo "[1/3] Thai abstract page"
( cd "$REPORT/frontmatter" && timeout 180 soffice --headless --convert-to pdf \
    --outdir . thai_abstract.fodt >/dev/null 2>&1 )
test -s "$REPORT/frontmatter/thai_abstract.pdf" || { echo "thai_abstract.pdf not produced" >&2; exit 1; }

echo "[2/3] Report"
( cd "$REPORT" && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex >build.log 2>&1 ) \
  || { grep -nE '^!' "$REPORT/main.log" | head; exit 1; }
pdfinfo "$REPORT/main.pdf" | grep Pages

echo "[3/3] Open Concerns"
python3 "$ROOT/script/extract_concerns.py" | tail -1
( cd "$CONCERNS" && latexmk -pdf -interaction=nonstopmode -halt-on-error concerns.tex >build.log 2>&1 ) \
  || { grep -nE '^!' "$CONCERNS/concerns.log" | head; exit 1; }
pdfinfo "$CONCERNS/concerns.pdf" | grep Pages
rm -f "$REPORT/build.log" "$CONCERNS/build.log"
