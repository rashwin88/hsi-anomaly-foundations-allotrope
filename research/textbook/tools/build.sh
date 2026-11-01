#!/bin/bash
# Full build of the book: bash tools/build.sh   (from anywhere)
# 1. normalise worked-example labels, 2. merge answers/glossary/bib,
# 3. tectonic (writes main.idx), 4. make main.ind (no makeindex on this laptop), 5. tectonic again.
set -e
T="$(cd "$(dirname "$0")/.." && pwd)"; cd "$T"
PY=C:/av/Scripts/python.exe
TEC=/c/Users/FITT/tools/tectonic/tectonic.exe
$PY tools/normalize_labels.py
$PY tools/build_backmatter.py
$TEC --keep-logs --keep-intermediates main.tex > build/tectonic.log 2>&1
$PY tools/make_index.py
$TEC --keep-logs --keep-intermediates main.tex > build/tectonic.log 2>&1
grep "Output written" main.log
