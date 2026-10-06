"""
Quantum causal structures with a known quantum-classical (QC) gap, used as seeds for the piggyback search.

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
# Known QC gaps with 4 visible nodes (Bell scenario variants)
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

known_QC_Gaps_QmDAGs_big = {QG_Bell1, QG_Bell2, QG_Bell3, QG_Bell4, QG_Bell5, QG_Bell6, QG_Bell7, QG_Bell8, QG_Bell9,
                            QG_Bell2b, QG_Bell3b, QG_Bell4b, QG_Bell4c, QG_Bell4d, QG_Bell5b, QG_Bell6b, QG_Bell6c,
                            QG_Bell6d, QG_Bell7b, QG_Bell7c, QG_Bell7d, QG_Bell8b, QG_Bell8c, QG_Bell8d, QG_Bell9b}
known_QC_Gaps_QmDAGs_big_ids = set(special_QmDAG.unique_unlabelled_id for special_QmDAG in known_QC_Gaps_QmDAGs_big)

known_QC_Gaps_QmDAGs_ids = known_QC_Gaps_QmDAGs_small_ids.union(known_QC_Gaps_QmDAGs_big_ids)


def _named(prefix: str) -> dict:
    return {name: value for name, value in globals().items()
            if name.startswith(prefix) and isinstance(value, QmDAG)}


# Seeds by name. Three-node seeds are the classic gaps; four-node seeds are the Bell-scenario variants.
SEEDS_3_NODES = {name: g for name, g in _named('QG_').items() if g.number_of_visible == 3 and name != 'QG_Evansb'}
SEEDS_4_NODES = {name: g for name, g in _named('QG_Bell').items()}
SEEDS = {**SEEDS_3_NODES, **SEEDS_4_NODES}
