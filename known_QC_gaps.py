"""
Quantum causal structures with a known quantum-classical (QC) gap, used as seeds for the piggyback search.

Every named structure below has a gap by a direct argument. The search's seeds (SEEDS) are only the *weakest*
ones: a structure is dropped when another named structure is a degradation of it (some of its quantum facets made
classical, see QmDAG.degradations), because the degradation piggyback then proves it from the weaker one. One
representative per relabelling class is kept. The stronger variants stay defined by name for tests and for the
manuscript.

Bell variants (four nodes). Two parties: settings 0 and 1, outcomes 2 and 3, the shared state Q{2,3} the only
facet containing an outcome apart from a classical facet between an outcome and its own setting. Each outcome
receives its setting through the edge (0→2, 1→3), through the edge plus a classical facet, or through a classical
facet alone (the setting is then a classical copy correlated with the outcome). The settings may be connected to
each other by an edge, a classical facet or both, but only when both outcomes read their settings through a bare
edge: a setting that shares a facet with its own outcome must have no other parent or facet, otherwise, conditional
on the two settings, that facet can leak the other party's setting to the outcome (0→2, 1→3; C{0,2}, C{0,1}, Q{2,3}
has no gap: 2 learns 1 from 0 and its shared facet). The gap argument is the usual one: in any classical model the
latent of Q{2,3} is independent of the settings and of every other latent, and conditional on the settings each
outcome's response depends on its own setting, the shared latent and private randomness only, so P(2,3|0,1) is
local; quantum models violate a Bell inequality.

Evans is treated as a QC gap only in the variant where both latents are quantum (QG_Evans); QG_Evansb is defined
for reference but is not a seed.
"""
from hypergraphs import Hypergraph
from directed_structures import DirectedStructure
from quantum_mDAG import QmDAG

# Known QC gaps with 3 visible nodes
# ---------------------------------------------------------------------------
QG_Instrumental1 = QmDAG(DirectedStructure([(0, 1), (1, 2)], 3), Hypergraph([], 3), Hypergraph([(1, 2)], 3))
QG_Instrumental2 = QmDAG(DirectedStructure([(0, 1), (1, 2)], 3), Hypergraph([], 3), Hypergraph([(0, 1), (1, 2)], 3))
QG_Instrumental3 = QmDAG(DirectedStructure([(1, 2)], 3), Hypergraph([], 3), Hypergraph([(0, 1), (1, 2)], 3))
QG_Instrumental2b = QmDAG(DirectedStructure([(0, 1), (1, 2)], 3), Hypergraph([(0, 1)], 3), Hypergraph([(1, 2)], 3))
QG_Instrumental3b = QmDAG(DirectedStructure([(1, 2)], 3), Hypergraph([(0, 1)], 3), Hypergraph([(1, 2)], 3))

IV_ids = set(special_QmDAG.unique_unlabelled_id for special_QmDAG in {QG_Instrumental1, QG_Instrumental2, QG_Instrumental3,
                                                                       QG_Instrumental2b, QG_Instrumental3b})

QG_Triangle1 = QmDAG(DirectedStructure([], 3), Hypergraph([], 3), Hypergraph([(1, 2), (2, 0), (0, 1)], 3))
QG_Triangle2 = QmDAG(DirectedStructure([], 3), Hypergraph([(1, 2)], 3), Hypergraph([(2, 0), (0, 1)], 3))
QG_Triangle3 = QmDAG(DirectedStructure([], 3), Hypergraph([(1, 2), (2, 0)], 3), Hypergraph([(0, 1)], 3))

Triangle_ids = set(special_QmDAG.unique_unlabelled_id for special_QmDAG in {QG_Triangle1, QG_Triangle2, QG_Triangle3})

QG_Evans = QmDAG(DirectedStructure([(0, 1), (0, 2)], 3), Hypergraph([], 3), Hypergraph([(0, 1), (0, 2)], 3))
QG_Evansb = QmDAG(DirectedStructure([(0, 1), (0, 2)], 3), Hypergraph([(0, 1)], 3), Hypergraph([(0, 2)], 3))
# Evans is treated as a QC gap only if both latents are quantum.
Evans_ids = set(special_QmDAG.unique_unlabelled_id for special_QmDAG in {QG_Evans})

known_QC_Gaps_QmDAGs_small_ids = set().union(IV_ids, Triangle_ids, Evans_ids)

# ---------------------------------------------------------------------------
# Known QC gaps with 4 visible nodes (Bell scenario variants). Legacy names first (several are relabellings or
# upgrades of one another), then the systematic family described in the module docstring.
# ---------------------------------------------------------------------------
QG_Bell1 = QmDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([], 4), Hypergraph([(2, 3)], 4))

QG_Bell2 = QmDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([(0, 2)], 4), Hypergraph([(2, 3)], 4))
QG_Bell2b = QmDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([], 4), Hypergraph([(0, 2), (2, 3)], 4))

QG_Bell3 = QmDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([(1, 3)], 4), Hypergraph([(2, 3)], 4))
QG_Bell3b = QmDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([], 4), Hypergraph([(2, 3), (1, 3)], 4))

QG_Bell4 = QmDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([(0, 2), (1, 3)], 4), Hypergraph([(2, 3)], 4))
QG_Bell4b = QmDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([(0, 2)], 4), Hypergraph([(1, 3), (2, 3)], 4))
QG_Bell4c = QmDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([(1, 3)], 4), Hypergraph([(0, 2), (2, 3)], 4))
QG_Bell4d = QmDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([], 4), Hypergraph([(0, 2), (1, 3), (2, 3)], 4))

QG_Bell6 = QmDAG(DirectedStructure([], 4), Hypergraph([(1, 3), (0, 2)], 4), Hypergraph([(2, 3)], 4))
QG_Bell6b = QmDAG(DirectedStructure([], 4), Hypergraph([(1, 3)], 4), Hypergraph([(0, 2), (2, 3)], 4))
QG_Bell6c = QmDAG(DirectedStructure([], 4), Hypergraph([(0, 2)], 4), Hypergraph([(1, 3), (2, 3)], 4))
QG_Bell6d = QmDAG(DirectedStructure([], 4), Hypergraph([], 4), Hypergraph([(0, 2), (1, 3), (2, 3)], 4))

QG_Bell5 = QmDAG(DirectedStructure([(1, 3)], 4), Hypergraph([(0, 2)], 4), Hypergraph([(2, 3)], 4))
QG_Bell5b = QmDAG(DirectedStructure([(1, 3)], 4), Hypergraph([], 4), Hypergraph([(2, 3), (0, 2)], 4))

QG_Bell7 = QmDAG(DirectedStructure([(1, 3)], 4), Hypergraph([(0, 2), (1, 3)], 4), Hypergraph([(2, 3)], 4))
QG_Bell7b = QmDAG(DirectedStructure([(1, 3)], 4), Hypergraph([(1, 3)], 4), Hypergraph([(0, 2), (2, 3)], 4))
QG_Bell7c = QmDAG(DirectedStructure([(1, 3)], 4), Hypergraph([(0, 2)], 4), Hypergraph([(1, 3), (2, 3)], 4))
QG_Bell7d = QmDAG(DirectedStructure([(1, 3)], 4), Hypergraph([], 4), Hypergraph([(0, 2), (1, 3), (2, 3)], 4))

QG_Bell8 = QmDAG(DirectedStructure([(0, 2)], 4), Hypergraph([(0, 2), (1, 3)], 4), Hypergraph([(2, 3)], 4))
QG_Bell8b = QmDAG(DirectedStructure([(0, 2)], 4), Hypergraph([(1, 3)], 4), Hypergraph([(0, 2), (2, 3)], 4))
QG_Bell8c = QmDAG(DirectedStructure([(0, 2)], 4), Hypergraph([(0, 2)], 4), Hypergraph([(1, 3), (2, 3)], 4))
QG_Bell8d = QmDAG(DirectedStructure([(0, 2)], 4), Hypergraph([], 4), Hypergraph([(0, 2), (1, 3), (2, 3)], 4))

QG_Bell9 = QmDAG(DirectedStructure([(0, 2)], 4), Hypergraph([(1, 3)], 4), Hypergraph([(2, 3)], 4))
QG_Bell9b = QmDAG(DirectedStructure([(0, 2)], 4), Hypergraph([], 4), Hypergraph([(1, 3), (2, 3)], 4))

# Settings connected to each other (both outcomes read their settings through a bare edge).
QG_Bell_SettingEdge = QmDAG(DirectedStructure([(0, 1), (0, 2), (1, 3)], 4), Hypergraph([], 4), Hypergraph([(2, 3)], 4))
QG_Bell_SettingsC = QmDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([(0, 1)], 4), Hypergraph([(2, 3)], 4))
QG_Bell_SettingEdgeC = QmDAG(DirectedStructure([(0, 1), (0, 2), (1, 3)], 4), Hypergraph([(0, 1)], 4), Hypergraph([(2, 3)], 4))

known_QC_Gaps_QmDAGs_big = {QG_Bell1, QG_Bell2, QG_Bell3, QG_Bell4, QG_Bell5, QG_Bell6, QG_Bell7, QG_Bell8, QG_Bell9,
                            QG_Bell2b, QG_Bell3b, QG_Bell4b, QG_Bell4c, QG_Bell4d, QG_Bell5b, QG_Bell6b, QG_Bell6c,
                            QG_Bell6d, QG_Bell7b, QG_Bell7c, QG_Bell7d, QG_Bell8b, QG_Bell8c, QG_Bell8d, QG_Bell9b,
                            QG_Bell_SettingEdge, QG_Bell_SettingsC, QG_Bell_SettingEdgeC}
known_QC_Gaps_QmDAGs_big_ids = set(special_QmDAG.unique_unlabelled_id for special_QmDAG in known_QC_Gaps_QmDAGs_big)

known_QC_Gaps_QmDAGs_ids = known_QC_Gaps_QmDAGs_small_ids.union(known_QC_Gaps_QmDAGs_big_ids)


def _named(prefix: str) -> dict:
    return {name: value for name, value in globals().items()
            if name.startswith(prefix) and isinstance(value, QmDAG)}


def weakest(named: dict) -> dict:
    """Keeps the structures none of whose degradations is (a relabelling of) another named structure, one per
    relabelling class (the first name in definition order)."""
    ids = {g.unique_unlabelled_id for g in named.values()}
    kept, seen = {}, set()
    for name, g in named.items():
        gid = g.unique_unlabelled_id
        if gid in seen or any(d.unique_unlabelled_id in ids for _, d in g.degradations()):
            continue
        kept[name] = g
        seen.add(gid)
    return kept


# Every named known gap, and the seeds: the weakest ones, one per relabelling class.
KNOWN_3_NODES = {name: g for name, g in _named('QG_').items() if g.number_of_visible == 3 and name != 'QG_Evansb'}
KNOWN_4_NODES = {name: g for name, g in _named('QG_Bell').items()}
KNOWN = {**KNOWN_3_NODES, **KNOWN_4_NODES}
SEEDS_3_NODES = weakest(KNOWN_3_NODES)
SEEDS_4_NODES = weakest(KNOWN_4_NODES)
SEEDS = {**SEEDS_3_NODES, **SEEDS_4_NODES}
