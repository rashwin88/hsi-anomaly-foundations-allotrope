"""Rewrite \\begin{worked}{Title}\\label{we:x} as \\begin{worked}[label={we:x}]{Title}.

A \\label placed after \\begin{worked}{...} attaches to the enclosing section, so
\\cref would print "Section 3.2" instead of "Worked example 3.4". The tcolorbox
label option attaches it to the box. Idempotent.
"""
import re
from pathlib import Path

T = Path(__file__).resolve().parent.parent
pat = re.compile(r"\\begin\{worked\}\{((?:[^{}]|\{[^{}]*\})*)\}\s*\\label\{([^}]*)\}")
total = 0
for f in sorted((T / "chapters").glob("*.tex")):
    s = f.read_text(encoding="utf-8")
    new, n = pat.subn(lambda m: f"\\begin{{worked}}[label={{{m.group(2)}}}]{{{m.group(1)}}}", s)
    if n:
        f.write_text(new, encoding="utf-8")
        total += n
        print(f"{f.name}: {n}")
print("converted", total)
