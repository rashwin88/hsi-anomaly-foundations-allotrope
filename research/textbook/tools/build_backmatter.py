"""Merge the per-chapter back-matter files into the three files main.tex reads.

    answers/NN-slug.tex  -> answers-all.tex   (chapter order, only chapters that exist)
    glossary/NN-slug.tex -> glossary-all.tex  (one alphabetical list, duplicate terms merged)
    bib/NN-slug.bib      -> refs.bib          (duplicate keys dropped, first one kept)

Run from anywhere: python research/textbook/tools/build_backmatter.py
"""
import re
from pathlib import Path

T = Path(__file__).resolve().parent.parent


def chapters():
    """Chapter stems that main.tex will include, in book order: read from its \\chap{...} lines, so a chapter
    inserted between two others (10a-convae, 19a-chakshu) lands in the right place, and a stray file in chapters/
    (a OneDrive conflict copy, say) is never picked up."""
    main = (T / "main.tex").read_text(encoding="utf-8")
    stems = re.findall(r"^\s*\\chap\{([^}]+)\}", main, flags=re.M)
    return [s for s in stems if (T / "chapters" / f"{s}.tex").exists()]


def build_answers(stems):
    parts = []
    for s in stems:
        f = T / "answers" / f"{s}.tex"
        if f.exists():
            parts.append(f"\\input{{answers/{s}}}\n")
    (T / "answers-all.tex").write_text("".join(parts) or "No answers yet.\n", encoding="utf-8")


def split_entries(text):
    """Split a file of \\glossentry{term}{definition} into (term, definition) pairs,
    respecting nested braces."""
    out, i = [], 0
    while True:
        j = text.find("\\glossentry", i)
        if j < 0:
            return out
        k = j + len("\\glossentry")
        args = []
        for _ in range(2):
            while text[k] != "{":
                k += 1
            depth, start = 0, k
            while True:
                if text[k] == "\\":
                    k += 2
                    continue
                if text[k] == "{":
                    depth += 1
                elif text[k] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            args.append(text[start + 1:k])
            k += 1
        out.append(tuple(args))
        i = k


def sort_key(term):
    t = re.sub(r"\\[a-zA-Z]+|[{}$\\^_]", "", term).strip().lower()
    return t


def build_glossary(stems):
    entries = {}
    for s in stems:
        f = T / "glossary" / f"{s}.tex"
        if not f.exists():
            continue
        for term, defn in split_entries(f.read_text(encoding="utf-8")):
            key = sort_key(term)
            if key in entries:
                # keep the first definition (earliest chapter); note later chapters
                entries[key][2].append(s)
            else:
                entries[key] = (term, defn, [s])
    lines = ["Each term is defined where it first appears; the chapter is given in brackets.\n",
             "\\begin{description}[style=nextline,leftmargin=1.2em,font=\\normalfont\\bfseries]\n"]
    for key in sorted(entries):
        term, defn, where = entries[key]
        ch = where[0]
        slug = ch.split("-", 1)[1]
        lines.append(f"\\item[{term}] {defn} (\\cref{{ch:{slug}}})\n")
    lines.append("\\end{description}\n")
    (T / "glossary-all.tex").write_text("".join(lines), encoding="utf-8")
    return len(entries)


def build_bib(stems):
    seen, out = set(), []
    files = [T / "bib" / f"{s}.bib" for s in stems] + [T / "bib" / "00-front.bib"]
    for f in files:
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8")
        for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
            pass
        # split on entry starts
        chunks = re.split(r"(?=^@)", text, flags=re.M)
        for c in chunks:
            m = re.match(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", c)
            if not m:
                continue
            key = m.group(2)
            if key in seen:
                continue
            seen.add(key)
            out.append(c.strip() + "\n\n")
    (T / "refs.bib").write_text("".join(out), encoding="utf-8")
    return len(seen)


if __name__ == "__main__":
    st = chapters()
    build_answers(st)
    g = build_glossary(st)
    b = build_bib(st)
    print(f"{len(st)} chapters, {g} glossary terms, {b} bib entries")
