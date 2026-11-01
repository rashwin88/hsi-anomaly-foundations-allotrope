"""A small makeindex replacement: main.idx -> main.ind.

Tectonic writes main.idx (with --keep-intermediates) but does not run makeindex,
and no makeindex binary is installed on this laptop. This script reads the
\\indexentry lines, groups entries and sub-entries (key!sub), honours key@display
and |format (|textbf, |( and |) ranges, |see{...}), sorts case-insensitively and
writes a theindex environment with letter groups. Rebuild with tectonic after it.
"""
import re
from collections import defaultdict
from pathlib import Path

T = Path(__file__).resolve().parent.parent


def split_top(s, ch):
    """Split s on ch at brace depth 0, ignoring "-quoted characters."""
    out, depth, cur, i = [], 0, "", 0
    while i < len(s):
        c = s[i]
        if c == '"' and i + 1 < len(s):
            cur += s[i + 1]
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        if c == ch and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += c
        i += 1
    out.append(cur)
    return out


def read_entries(idx):
    entries = []
    text = idx.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        m = re.match(r"\\indexentry\{(.*)\}\{(\d+|[ivxlcdm]+)\}\s*$", line)
        if not m:
            continue
        body, page = m.group(1), m.group(2)
        fmt = ""
        parts = split_top(body, "|")
        if len(parts) > 1:
            body, fmt = parts[0], parts[1]
        levels = []
        for lev in split_top(body, "!"):
            kd = split_top(lev, "@")
            key = kd[0].strip()
            disp = kd[1].strip() if len(kd) > 1 else key
            levels.append((key, disp))
        entries.append((levels[:3], page, fmt))
    return entries


def sortkey(k):
    k = re.sub(r"\\[a-zA-Z]+\s*|[{}$\\]", "", k).strip().lower()
    return k


def page_num(p):
    return int(p) if p.isdigit() else -1


def fmt_pages(plist):
    """plist: list of (page, fmt). Collapse duplicates and |( |) ranges."""
    see, items, seen = [], [], set()
    open_range = None
    for p, f in sorted(plist, key=lambda x: (page_num(x[0]), x[0])):
        if f.startswith("see"):
            see.append(f"\\{f}" if not f.startswith("\\") else f)
            continue
        if f.endswith("("):
            open_range = p
            continue
        if f.endswith(")") and open_range is not None:
            items.append(f"{open_range}--{p}" if open_range != p else p)
            seen.add(p)
            open_range = None
            continue
        if p in seen:
            continue
        seen.add(p)
        items.append(f"\\{f}{{{p}}}" if f else p)
    if open_range is not None and open_range not in seen:
        items.append(open_range)
    s = ", ".join(items)
    if see:
        see_txt = ", ".join(x.replace("\\see", "\\emph{see}~") for x in see)
        s = f"{s}, {see_txt}" if s else see_txt
    return s


def build():
    idx = T / "main.idx"
    if not idx.exists():
        print("no main.idx - build with --keep-intermediates first")
        return
    tree = defaultdict(lambda: {"disp": None, "pages": [], "kids": defaultdict(lambda: {"disp": None, "pages": [], "kids": defaultdict(lambda: {"disp": None, "pages": []})})})
    for levels, page, fmt in read_entries(idx):
        node = tree[sortkey(levels[0][0])]
        node["disp"] = node["disp"] or levels[0][1]
        if len(levels) == 1:
            node["pages"].append((page, fmt))
            continue
        kid = node["kids"][sortkey(levels[1][0])]
        kid["disp"] = kid["disp"] or levels[1][1]
        if len(levels) == 2:
            kid["pages"].append((page, fmt))
            continue
        g = kid["kids"][sortkey(levels[2][0])]
        g["disp"] = g["disp"] or levels[2][1]
        g["pages"].append((page, fmt))

    out = ["\\begin{theindex}\n"]
    letter = None
    for k in sorted(tree):
        first = k[:1].upper() if k[:1].isalpha() else "Symbols"
        if first != letter:
            letter = first
            out.append(f"\n  \\indexspace\n  {{\\bfseries {letter}}}\\nopagebreak\n\n")
        n = tree[k]
        out.append(f"  \\item {n['disp']}{', ' + fmt_pages(n['pages']) if n['pages'] else ''}\n")
        for kk in sorted(n["kids"]):
            c = n["kids"][kk]
            out.append(f"    \\subitem {c['disp']}{', ' + fmt_pages(c['pages']) if c['pages'] else ''}\n")
            for gg in sorted(c["kids"]):
                g = c["kids"][gg]
                out.append(f"      \\subsubitem {g['disp']}{', ' + fmt_pages(g['pages']) if g['pages'] else ''}\n")
    out.append("\n\\end{theindex}\n")
    (T / "main.ind").write_text("".join(out), encoding="utf-8")
    print(f"{len(tree)} top-level index entries")


if __name__ == "__main__":
    build()
