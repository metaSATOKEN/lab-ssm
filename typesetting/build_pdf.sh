#!/bin/sh
# Run from anywhere in a checkout containing this revision. No experiment is invoked.
set -eu
cd "$(dirname "$0")/.."
termfile=$(kpsewhich texgyretermes-regular.otf)
monofile=$(kpsewhich lmmono10-regular.otf)
test -f "$termfile" && test -f "$monofile"
pandoc LAB.md --pdf-engine=xelatex -o LAB.pdf \
  --lua-filter=typesetting/layout.lua --include-in-header=typesetting/preamble.tex \
  -V papersize=letter -V fontsize=10pt -V geometry:margin=0.9in \
  -V mainfont=texgyretermes-regular.otf \
  -V "mainfontoptions=Path=$(dirname "$termfile")/,BoldFont=texgyretermes-bold.otf,ItalicFont=texgyretermes-italic.otf,BoldItalicFont=texgyretermes-bolditalic.otf" \
  -V monofont=lmmono10-regular.otf -V "monofontoptions=Path=$(dirname "$monofile")/" -V colorlinks=true
