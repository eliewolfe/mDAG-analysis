#!/bin/sh
set -eu
cd "$(dirname "$0")"
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build piggyback_rules.tex
cp build/piggyback_rules.pdf piggyback_rules.pdf
