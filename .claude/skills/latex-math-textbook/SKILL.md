---
name: latex-math-textbook
description: Write a detailed, step-by-step mathematical textbook or long-form math note in LaTeX and compile it to PDF - for this project's training material, a method's mathematics, or any topic the user wants to "read deeply". Assumes a reader with basic maths who learns slowly; every derivation shown line by line with worked numeric examples, illustrations in TikZ/pgfplots or converted SVGs, compiled with Tectonic on this Windows laptop. Triggers on - latex, pdf, textbook, text book, math book, mathematical text, write up the math, derivation, step by step math, read deeply, typeset, course as a book.
---

# LaTeX math textbook

A book the user reads slowly and deeply, typically on a phone or tablet. It is not a paper and not a summary. The
standard: a reader with basic maths can follow every line without filling gaps in their head.

This skill sits on top of `teaching-a-new-hire`: follow its per-part skeleton, its "verify against source" rule and
its tone rules. This file adds what a typeset mathematical book needs on top of that.

## 0. Before writing: scope, sources, numbers

- **Sources.** Read all the material first: course Markdown, the code it cites, experiment READMEs. When a doc and the
  source disagree, the source wins - and say so in the text.
- **Numbers.** Every number from the project is tagged **[M]** measured (cite the file), **[C]** computed for the book
  (by a script kept next to the .tex, e.g. `compute_numbers.py`, with its output saved as `compute_numbers_output.txt`),
  or **[A]** assumed (state the assumption). Never type a number from memory.
- **Size.** Be honest about scope. A full course is 100+ pages. Build it so a long run always leaves a compiling book
  (section 4). If it cannot all be done, give the core mathematics full depth and mark the rest "abridged", with a list
  of what is missing.

## 1. Toolchain (Windows laptop)

- **Engine: Tectonic** (XeTeX, fetches packages on demand). Installed at `C:/Users/FITT/tools/tectonic/tectonic.exe`
  (0.17.0, from the GitHub release zip; `winget` has no package). It is not on PATH, so call it by full path. If it is
  missing, download the release zip into that folder. Do not change system settings.
- **Build:** `C:/Users/FITT/tools/tectonic/tectonic.exe main.tex` from the book folder. Tectonic reruns for references by
  itself. `imakeidx` works.
- **Looking at pages:** render PDF pages to PNG with `pypdfium2` (pip-installed into a venv under the session
  scratchpad, never into `C:/av`) and view the PNGs to check figures and layout.
- **SVG figures:** Tectonic cannot include SVG. Convert to PDF with `cairosvg` in that scratch venv; fall back to
  `rsvg-convert` or Inkscape. If a figure will not convert, redraw it in TikZ.
- **Fonts:** Libertinus or Latin Modern under XeTeX. Keep a fallback (`lmodern`) so a missing font never breaks the
  build.

## 2. Layout of a book

```
research/<book>/                 e.g. research/textbook/, research/math/<topic>/
  main.tex                       \documentclass{book} (or article for a <30-page note); \input{preamble}
  preamble.tex                   ALL packages and environments live here - never in chapters
  chapters/NN-name.tex           one chapter per file, \include-d by main.tex
  figures/                       converted PDFs + standalone TikZ sources
  figdata/*.dat                  data behind every pgfplots plot (written by the compute script)
  answers/                       worked solutions to Check-yourself, if kept out of the chapters
  compute_numbers.py (+ _output.txt)
  tools/testchap.sh              compiles one chapter alone through a wrapper (fast iteration)
```

Start `preamble.tex` from [`preamble-template.tex`](preamble-template.tex). It has the four teaching boxes, theorem
styles, colours, and `\code{}` / `\file{}` for paths.

## 3. How each chapter is written

Use the `teaching-a-new-hire` skeleton, in boxes:

1. `oneThing` box - "The one thing this chapter teaches", one sentence.
2. Motivation before mechanism: the problem, and why the obvious approach fails.
3. Definitions before use, including words that feel universal (vector, norm, covariance, likelihood, gradient, ...).
4. **Derivations, line by line.** Each line is one operation, and the text names it: "expand the square", "the cross
   terms cancel because ...", "differentiate with respect to $a$", "multiply both sides by $S^{-1}$". When a rule is
   used for the first time (transpose of a product, derivative of a quadratic form, Bayes' rule), state it and check it
   on a 2x2 or 3-element example before using it. Use `align` with one step per line, with a short `\text{}` comment on
   the right where it helps.
5. **A worked example after every result**, small enough to do on paper: 2-3 bands, 2x2 matrices, 6 tokens. Show every
   arithmetic operation on its own line: each difference, each square, the sum, then the division. Never
   "$0.070/3 = 0.023$" without the steps that produced 0.070.
6. **Pictures where a slow reader needs one:** vectors and angles, projections, a Gaussian with threshold areas, ROC and
   pAUC areas, mixing lines, attention matrices, masking grids, tensor shapes through a network. Draw them to scale from
   real numbers (pgfplots reading `figdata/`).
7. `confusion` box - "Common confusions", as question and answer.
8. `checkYourself` box - "Check yourself", about 5 questions, at least one hand calculation. Full worked answers, either in the
   chapter (collapsible style) or in `answers/`.

Tone: second person; plain hyphens; never "simply", "just", "obviously" or "clearly". Short paragraphs. Say when
something is hard, surprising, or an open question. One analogy per chapter that does real work.

Front and back matter for a book:
- "How to read this book" and a chapter dependency map;
- a notation table and a glossary;
- an index of worked examples, and an index (`imakeidx`);
- an annotated bibliography. If a citation detail (volume, page, article number) is uncertain, say so in the entry
  rather than invent it.

## 4. Build discipline

- **Chapter by chapter.** Write one chapter, compile it alone (`tools/testchap.sh`), fix it, then add it to `main.tex`
  and compile the whole book. After each part, the full book must compile.
- **Clean means:** no undefined references or citations, no missing figures, no overfull boxes worth seeing. Render a
  sample of pages to PNG and look at figures and tables at least once per part.
- **Do not commit** unless the user asks. Write only inside the book's folder: other agents and sessions share the
  working tree. If committing is asked for, follow the branch rule: check `git branch --show-current`, stage by
  explicit path, push `HEAD:<branch>` and confirm `origin/<branch>` moved.
- **No GPU or cloud** for a book.

## 5. Delivering

- Send the PDF to the user with `SendUserFile` (`display: "render"`), with a one-line caption saying what it is and the
  page count.
- Report: path, page count, chapter list (complete vs abridged), what was installed, how figures were made, every
  claim that could not be verified, and every place where the sources disagreed.

## Past books (for reference)

- `research/math/spectral-matching/spectral_matching.pdf`: a dense 20-page note. The user asked for more detail, so
  the default is the slow version.
- `research/math/spectral-matching/spectral_matching_step_by_step.pdf`: the step-by-step rewrite.
- `research/textbook/`: both courses (anomaly detection, Indradhanu) as one book.
