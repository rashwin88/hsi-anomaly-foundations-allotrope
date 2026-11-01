"""Regenerate the inline index of spectral_matching_step_by_step.tex (Tectonic does not run makeindex).

    C:/Users/FITT/tools/tectonic/tectonic.exe --keep-intermediates spectral_matching_step_by_step.tex
    C:/av/Scripts/python.exe make_index.py          # rewrites the theindex block in place
    C:/Users/FITT/tools/tectonic/tectonic.exe spectral_matching_step_by_step.tex

Reads the .idx, merges duplicate pages, sorts case-insensitively, separates initial letters with \\indexspace (the
format the block already had), and replaces everything between \\begin{theindex} and \\end{theindex}.
"""
import re
from collections import defaultdict
from pathlib import Path

D = Path(__file__).resolve().parent
TEX = D / "spectral_matching_step_by_step.tex"
IDX = D / "spectral_matching_step_by_step.idx"


def sortkey(k):
    return re.sub(r"\\[a-zA-Z]+\s*|[{}$\\]", "", k).strip().lower()


def main():
    pages = defaultdict(set)
    disp = {}
    for line in IDX.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"\\indexentry\{(.*)\|hyperpage\}\{(\d+)\}\s*$", line) or \
            re.match(r"\\indexentry\{(.*)\}\{(\d+)\}\s*$", line)
        if not m:
            continue
        term, page = m.group(1), int(m.group(2))
        k = sortkey(term)
        disp.setdefault(k, term)
        pages[k].add(page)
    out, letter = ["\\begin{theindex}\n"], None
    for k in sorted(pages):
        first = k[:1]
        if letter is not None and first != letter:
            out.append("  \\indexspace\n")
        letter = first
        out.append(f"  \\item {disp[k]}, " + ", ".join(f"\\hyperpage{{{p}}}" for p in sorted(pages[k])) + "\n")
    out.append("\\end{theindex}")
    s = TEX.read_text(encoding="utf-8")
    i, j = s.index("\\begin{theindex}"), s.index("\\end{theindex}") + len("\\end{theindex}")
    TEX.write_text(s[:i] + "".join(out) + s[j:], encoding="utf-8")
    print(f"{len(pages)} index terms")


if __name__ == "__main__":
    main()
