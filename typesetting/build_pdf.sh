#!/bin/sh
# Typeset the current manuscript using its saved vector figures; no experiment is run.
set -eu
cd "$(dirname "$0")/.."
pandoc LAB.md --pdf-engine="${PDF_ENGINE:-xelatex}" -o LAB.pdf \
  --lua-filter=typesetting/layout.lua --include-in-header=typesetting/preamble.tex \
  -V papersize=a4 -V fontsize=10pt -V geometry:margin=22mm \
  -V mainfont=texgyretermes-regular.otf -V mathfont=texgyretermes-math.otf \
  -V colorlinks=true -V linkcolor=accent -V urlcolor=accent
