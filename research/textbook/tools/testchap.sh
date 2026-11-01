#!/bin/bash
# Usage: bash tools/testchap.sh NN-slug [NN-slug ...]
# Compiles the given chapters (with their answers, glossary and bib files) into
# build/<first>/test-<first>.pdf and prints the errors and warnings that matter.
# Undefined references to OTHER chapters' labels are expected here; only fix
# undefined references to labels in your own chapters.
set -u
T="$(cd "$(dirname "$0")/.." && pwd)"
first="$1"
w="$T/test-$first.tex"
{
  echo '\documentclass[11pt,openany]{book}'
  echo '\input{preamble}'
  echo '\begin{document}'
  for c in "$@"; do echo "\input{chapters/$c}"; done
  echo '\appendix\chapter{Answers}'
  for c in "$@"; do [ -f "$T/answers/$c.tex" ] && echo "\input{answers/$c}"; done
  echo '\chapter{Glossary entries}\begin{description}'
  for c in "$@"; do [ -f "$T/glossary/$c.tex" ] && echo "\input{glossary/$c}"; done
  echo '\item[] \end{description}'
  echo '\bibliographystyle{plainnat}'
  bibs=""
  for c in "$@"; do [ -s "$T/bib/$c.bib" ] && bibs="$bibs,bib/$c"; done
  [ -n "$bibs" ] && echo "\bibliography{${bibs#,}}"
  echo '\end{document}'
} > "$w"
mkdir -p "$T/build/$first"
cd "$T"
/c/Users/FITT/tools/tectonic/tectonic.exe --outdir "build/$first" "test-$first.tex" > "build/$first/log.txt" 2>&1
rc=$?
grep -E "^error|! |Undefined control|LaTeX Error|Missing|Runaway|File .* not found|Overfull \\hbox \([0-9]{2,}" "build/$first/log.txt" | head -40
echo "--- undefined references (own labels only matter):"
grep -oE "Reference \`[^']*' on page" "build/$first/log.txt" | sort -u | head -40
echo "exit=$rc  pdf: build/$first/test-$first.pdf"
rm -f "$w"
