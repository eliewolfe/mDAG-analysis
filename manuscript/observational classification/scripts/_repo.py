"""Shared by the scripts of this directory: the repository on the import path, and where outputs go."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUTPUT = HERE / 'output'
for path in (ROOT, ROOT / 'Special Applications'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def save(name: str, data) -> Path:
    OUTPUT.mkdir(exist_ok=True)
    path = OUTPUT / name
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + '\n', encoding='utf-8')
    return path


def notation(md) -> str:
    """An mDAG in the manuscript's notation: its edges, then its sources, e.g. '0→1, 0→2; Q{0,1}, Q{0,2,3}'."""
    edges = sorted((int(a), int(b)) for a, b in md.directed_structure_instance.edge_list)
    facets = sorted(tuple(sorted(int(v) for v in f)) for f in md.simplicial_complex_instance.compressed_simplicial_complex)
    left = ', '.join(f'{a}→{b}' for a, b in edges) or 'no edges'
    right = ', '.join('Q{' + ','.join(map(str, f)) + '}' for f in facets) or 'no sources'
    return f'{left}; {right}'
