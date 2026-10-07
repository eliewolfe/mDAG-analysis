"""
Make the manuscript's cross-references clickable.

Run from anywhere: `python manuscript/refresh_links.py`. Idempotent: existing links of the two kinds below are
stripped and rebuilt, so re-running after the code or the headings change refreshes them.

1. Section references ("5.3", "Section 7", "Sections 5 and 6", "Appendix A", "A.3") become links to the heading
   anchors GitHub generates (lower case, punctuation removed, spaces to hyphens).
2. Backticked code references (`QmDAG.split_node`, `default_stages`, `known_QC_gaps.SEEDS`, `qc_gap_search.py`,
   `tests/test_fritz_entropic.py`) become links to the file in the repository, at the line where the symbol is
   defined (`#L<n>`), relative to the manuscript so that they work on GitHub and in local renderers. A backticked
   span that names no file or symbol of the repository is left alone.
Headings, fenced code blocks and inline math are never touched.
"""
import ast
import os
import re
import sys
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MANUSCRIPT = os.path.join(HERE, "piggybacks.md")
PY_DIRS = [ROOT, os.path.join(ROOT, "Special Applications"), os.path.join(ROOT, "tests")]


# --------------------------------------------------------------------------------------------------
# Symbol index: qualified name -> (path relative to ROOT, line)
# --------------------------------------------------------------------------------------------------

def index_symbols() -> Dict[str, Tuple[str, int]]:
    index: Dict[str, Tuple[str, int]] = {}
    bare: Dict[str, List[Tuple[str, int]]] = {}

    def add(name: str, path: str, line: int) -> None:
        index.setdefault(name, (path, line))

    for folder in PY_DIRS:
        for fname in sorted(os.listdir(folder)):
            if not fname.endswith(".py"):
                continue
            path = os.path.relpath(os.path.join(folder, fname), ROOT)
            module = fname[:-3]
            try:
                tree = ast.parse(open(os.path.join(folder, fname), encoding="utf-8").read())
            except SyntaxError:
                continue
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    add(f"{module}.{node.name}", path, node.lineno)
                    bare.setdefault(node.name, []).append((path, node.lineno))
                    if isinstance(node, ast.ClassDef):
                        for item in node.body:
                            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                add(f"{node.name}.{item.name}", path, item.lineno)
                                add(f"{module}.{node.name}.{item.name}", path, item.lineno)
                            elif isinstance(item, ast.Assign):
                                for t in item.targets:
                                    if isinstance(t, ast.Name):
                                        add(f"{node.name}.{t.id}", path, item.lineno)
                            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                                add(f"{node.name}.{item.target.id}", path, item.lineno)
                        # attributes assigned in __init__ (self.x = ...)
                        for item in ast.walk(node):
                            if isinstance(item, ast.Assign):
                                for t in item.targets:
                                    if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                                        add(f"{node.name}.{t.attr}", path, item.lineno)
                elif isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            add(f"{module}.{t.id}", path, node.lineno)
                            bare.setdefault(t.id, []).append((path, node.lineno))
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    add(f"{module}.{node.target.id}", path, node.lineno)
                    bare.setdefault(node.target.id, []).append((path, node.lineno))
    for name, places in bare.items():
        if len(places) == 1:
            index.setdefault(name, places[0])
    return index


def resolve_code(span: str, index: Dict[str, Tuple[str, int]]) -> Optional[str]:
    """Link target (relative to the manuscript) for a backticked span, or None."""
    text = span.strip()
    # Files of the repository.
    if re.fullmatch(r"[\w./ -]+\.(py|json|md|txt|nb)", text) and os.path.exists(os.path.join(ROOT, text)):
        return "../" + quote(text)
    name = re.split(r"[(\[]", text, maxsplit=1)[0].strip()
    if not re.fullmatch(r"[A-Za-z_][\w.]*", name):
        return None
    candidates = [name]
    parts = name.split(".")
    if len(parts) > 2:
        candidates.append(".".join(parts[-2:]))
    if len(parts) >= 2:
        candidates.append(parts[0])          # Class or module: fall back to the owner when the member is unknown
    for cand in candidates:
        if cand in index:
            path, line = index[cand]
            return f"../{quote(path)}#L{line}"
    return None


# --------------------------------------------------------------------------------------------------
# Heading anchors (GitHub's algorithm)
# --------------------------------------------------------------------------------------------------

def github_slug(heading: str) -> str:
    text = re.sub(r"`([^`]*)`", r"\1", heading)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = text.strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def heading_anchors(lines: List[str]) -> Dict[str, str]:
    """Reference key -> anchor. Keys: '0.4', '7.10', 'A.3' (subsections), '5' (sections), 'A' (the appendix)."""
    anchors: Dict[str, str] = {}
    seen: Dict[str, int] = {}
    for line in lines:
        m = re.match(r"^(#{2,3}) (.+)$", line)
        if not m:
            continue
        title = m.group(2).strip()
        slug = github_slug(title)
        if slug in seen:
            seen[slug] += 1
            slug = f"{slug}-{seen[slug]}"
        else:
            seen[slug] = 0
        key = None
        mm = re.match(r"^(\d+)\.\s", title)          # "## 7. Search ..."
        if mm and m.group(1) == "##":
            key = mm.group(1)
        mm = re.match(r"^([0-7]\.\d{1,2}|A\.\d)\s", title)   # "### 7.8 ..."
        if mm:
            key = mm.group(1)
        if title.startswith("Appendix A"):
            key = "A"
        if key is not None:
            anchors[key] = "#" + slug
    return anchors


# --------------------------------------------------------------------------------------------------
# Rewriting
# --------------------------------------------------------------------------------------------------

def strip_links(text: str) -> str:
    text = re.sub(r"\[((?:Sections? )?(?:[0-7](?:\.\d{1,2})?|A\.\d|Appendix A))\]\(#[^)]*\)", r"\1", text)
    text = re.sub(r"\[(`[^`]*`)\]\(\.\./[^)]*\)", r"\1", text)
    return text


def protected_spans(line: str) -> List[Tuple[int, int]]:
    """Character ranges of inline code and inline math, which are never rewritten."""
    spans = [(m.start(), m.end()) for m in re.finditer(r"`[^`]*`", line)]
    spans += [(m.start(), m.end()) for m in re.finditer(r"\$[^$]+\$", line)]
    return spans


def inside(pos: int, spans: List[Tuple[int, int]]) -> bool:
    return any(a <= pos < b for a, b in spans)


def link_sections(line: str, anchors: Dict[str, str]) -> str:
    spans = protected_spans(line)
    out = []
    i = 0
    pattern = re.compile(r"Sections (\d) and (\d)\b|Section (\d)\b|Appendix A\b|(?<![\w$.\\#/])([0-7]\.\d{1,2}|A\.\d)(?![\d])")
    for m in pattern.finditer(line):
        if inside(m.start(), spans):
            continue
        # "about 0.2, 0.5, 2 and 9 seconds" (6.6) is a measurement, not a reference.
        if m.group(4) in ("0.2", "0.5") and re.search(r"about 0\.2, 0\.5, 2 and 9", line):
            continue
        repl = None
        if m.group(1):
            a, b = m.group(1), m.group(2)
            if a in anchors and b in anchors:
                repl = f"Sections [{a}]({anchors[a]}) and [{b}]({anchors[b]})"
        elif m.group(3):
            if m.group(3) in anchors:
                repl = f"Section [{m.group(3)}]({anchors[m.group(3)]})"
        elif m.group(0) == "Appendix A":
            if "A" in anchors:
                repl = f"[Appendix A]({anchors['A']})"
        elif m.group(4) in anchors:
            repl = f"[{m.group(4)}]({anchors[m.group(4)]})"
        if repl is None:
            continue
        out.append(line[i:m.start()])
        out.append(repl)
        i = m.end()
    out.append(line[i:])
    return "".join(out)


def link_code(line: str, index: Dict[str, Tuple[str, int]]) -> str:
    math = [(m.start(), m.end()) for m in re.finditer(r"\$[^$]+\$", line)]

    def repl(m: re.Match) -> str:
        if inside(m.start(), math):
            return m.group(0)
        target = resolve_code(m.group(1), index)
        return f"[{m.group(0)}]({target})" if target else m.group(0)
    return re.sub(r"`([^`]+)`", repl, line)


def refresh(text: str, index: Dict[str, Tuple[str, int]]) -> str:
    text = strip_links(text)
    lines = text.split("\n")
    anchors = heading_anchors(lines)
    out = []
    in_fence = False
    for line in lines:
        if line.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence or re.match(r"^#{1,6} ", line):
            out.append(line)
            continue
        line = link_sections(line, anchors)
        line = link_code(line, index)
        out.append(line)
    return "\n".join(out)


def main(path: str = MANUSCRIPT) -> None:
    index = index_symbols()
    text = open(path, encoding="utf-8").read()
    new = refresh(text, index)
    open(path, "w", encoding="utf-8").write(new)
    n_sec = len(re.findall(r"\]\(#", new))
    n_code = len(re.findall(r"\]\(\.\./", new))
    unresolved = sorted({m.group(1) for m in re.finditer(r"(?<![\[`])`([^`]+)`(?![\]`])", new)
                         if re.fullmatch(r"[A-Za-z_][\w.]*(\(.*\))?", m.group(1).strip())})
    print(f"{n_sec} section links, {n_code} code links; backticked names without a target: {unresolved}")


if __name__ == "__main__":
    main(*sys.argv[1:])
