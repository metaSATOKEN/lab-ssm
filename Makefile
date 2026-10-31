# Typeset LAB from the five included vector figures.
PYTHON ?= python3

.PHONY: pdf figures all verify clean distclean

pdf:
	sh typesetting/build_pdf.sh

figures:
	$(PYTHON) figures/fig01_R_distribution.py
	$(PYTHON) figures/draw_saved.py

all: figures
	$(MAKE) pdf

verify:
	$(PYTHON) verification/verify_paper.py

clean:
	$(RM) LAB.pdf

distclean: clean
	$(RM) figures/fig*.pdf figures/fig*.png
