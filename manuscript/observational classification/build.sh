#!/bin/sh
# Builds observational_classification.pdf. Needs TeX Live with tikz and tcolorbox (texlive-latex-extra,
# texlive-pictures) and latexmk.
cd "$(dirname "$0")" || exit 1
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build observational_classification.tex || exit 1
cp build/observational_classification.pdf observational_classification.pdf
