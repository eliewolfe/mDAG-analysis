"""
JSON cache of proven QC gaps, with the version of every piggyback a proof relied on.

Every entry is one proven input structure: its unlabelled id, its structure (edges, classical and quantum facets,
with the node labels of the stored representative), the seed its certificate ends at, the chain of transitions
(trick name, params, source and target ids) and the version of each trick that appears in the chain. When a
piggyback is corrected its version in `qc_gap_search.PIGGYBACK_VERSIONS` is bumped, and loading the cache drops
exactly the entries whose chain used that piggyback. The expensive stages of the search then run only on the inputs
that are neither cached nor proven by the cheap stage.
"""
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from qc_gap_search import GapReport, PIGGYBACK_VERSIONS, render_certificate

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache', 'known_gaps.json')


def _plain(obj: Any) -> Any:
    """JSON-friendly copy of params (tuples become lists, numpy ints become ints)."""
    if isinstance(obj, (list, tuple)):
        return [_plain(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if hasattr(obj, 'item'):
        return obj.item()
    return obj


def id_key(gid: Tuple[int, ...]) -> str:
    return ",".join(str(int(x)) for x in gid)


def key_id(key: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in key.split(","))


def structure_record(g) -> Dict[str, Any]:
    names = [str(v) for v in g.visible_nodes]
    return {
        'nodes': names,
        'edges': _plain(g.directed_structure_instance.edge_list),
        'classical_facets': _plain(sorted(sorted(f) for f in g.C_simplicial_complex_instance.simplicial_complex_as_sets)),
        'quantum_facets': _plain(sorted(sorted(f) for f in g.Q_simplicial_complex_instance.simplicial_complex_as_sets)),
    }


class GapCache:
    def __init__(self, path: str = CACHE_PATH) -> None:
        self.path = path
        self.entries: Dict[str, Dict[str, Any]] = {}
        self.dropped: int = 0          # entries invalidated at load because a piggyback version changed

    @classmethod
    def load(cls, path: str = CACHE_PATH, seeds: Optional[Iterable[str]] = None) -> "GapCache":
        """Loads the entries whose piggyback versions all match `PIGGYBACK_VERSIONS` and, when `seeds` (names) is
        given, whose ultimate seed is still among them; the rest are counted in `dropped`."""
        cache = cls(path)
        if not os.path.exists(path):
            return cache
        with open(path) as f:
            data = json.load(f)
        entries = data.get('entries', {})
        seed_names = None if seeds is None else set(seeds)
        for key, entry in entries.items():
            versions = entry.get('versions', {})
            if all(PIGGYBACK_VERSIONS.get(trick) == version for trick, version in versions.items()) \
                    and (seed_names is None or entry.get('seed') in seed_names):
                cache.entries[key] = entry
            else:
                cache.dropped += 1
        # An entry whose certificate ends at another cached entry inherits that entry's versions, so it is dropped
        # with it; this guards against an upstream entry missing from the file altogether.
        changed = True
        while changed:
            changed = False
            for key in list(cache.entries):
                upstream = cache.entries[key].get('via_cache')
                if upstream is not None and upstream not in cache.entries:
                    del cache.entries[key]
                    cache.dropped += 1
                    changed = True
        return cache

    def valid_ids(self) -> Set[Tuple[int, ...]]:
        return {key_id(k) for k in self.entries}

    def known(self) -> Dict[str, Tuple[int, ...]]:
        """name -> id, for use as extra known gaps of a search."""
        return {f"cache:{k}": key_id(k) for k in self.entries}

    def record(self, report: GapReport, only_new: bool = True) -> int:
        """Stores every proven input of the report whose certificate has at least one transition (seeds and
        already-cached inputs are skipped). Returns the number of entries added."""
        added = 0
        seed_names = {g.unique_unlabelled_id: name for name, g in report.seeds.items()}
        for gid, chain in report.proven.items():
            key = id_key(gid)
            if not chain or gid in seed_names or (only_new and key in self.entries):
                continue
            rep_structure = report.explorer.representatives.get(gid)
            seed = report.seed_hit.get(gid, '?')
            versions = {t.trick: PIGGYBACK_VERSIONS[t.trick] for t in chain}
            via_cache = None
            if seed.startswith('cache:'):
                # The certificate ends at a cached gap: this proof relies on that entry's proof too, so it inherits
                # its piggyback versions (transitively) and is invalidated with it.
                via_cache = seed[len('cache:'):]
                upstream = self.entries.get(via_cache, {})
                versions.update({t: v for t, v in upstream.get('versions', {}).items() if t not in versions})
                seed = upstream.get('seed', seed)
            self.entries[key] = {
                'id': list(map(int, gid)),
                'structure': structure_record(rep_structure) if rep_structure is not None else None,
                'seed': seed,
                'via_cache': via_cache,
                'chain': [{'trick': t.trick, 'params': _plain(t.params), 'source': list(map(int, t.source)),
                           'target': list(map(int, t.target))} for t in chain],
                'versions': versions,
                'certificate': render_certificate(report.explorer, chain, report.seed_hit.get(gid, '?')),
            }
            added += 1
        return added

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'w') as f:
            json.dump({'piggyback_versions': PIGGYBACK_VERSIONS, 'entries': self.entries}, f, indent=1, sort_keys=True)

    def __len__(self) -> int:
        return len(self.entries)
