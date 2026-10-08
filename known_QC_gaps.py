"""
Quantum causal structures with a known quantum-classical (QC) gap: the seeds of the piggyback search.

Only the WEAKEST structures are defined: every seed has a gap by a direct argument, no seed is a degradation of
another (some quantum facets made classical, QmDAG.degradation_steps), and there is one seed per relabelling class.
Everything above a seed in the upgrade order (quantum facets added inside its classical facets, classical facets
made quantum) is a gap by the degradation piggyback, which the search applies as a lookup, so such structures are
never listed here. `weakest_and_distinct` checks the two properties at import time.

Naming. Three-node seeds: the instrumental scenario, in which the treatment 1 receives the instrument 0 through an
edge (`QG_Instrumental_Edge`), through an edge plus a classical common cause (`QG_Instrumental_EdgeC`) or through a
classical common cause alone (`QG_Instrumental_C`), with the quantum state Q{1,2} between treatment and outcome; the
triangle with one quantum and two classical sources (`QG_Triangle`); the Evans structure with both sources quantum
(`QG_Evans`; with one source classical it is not known to have a gap and is not a seed).

Bell variants (four nodes): settings 0 and 1, outcomes 2 and 3, the shared state Q{2,3} the only facet containing
an outcome apart from a classical facet between an outcome and its own setting. Each outcome receives its setting
through a bare edge (`Edge`), through the edge plus a classical facet (`EdgeC`), or through a classical facet alone
(`C`, the setting is then a classical copy correlated with the outcome); `QG_Bell_<A>_<B>` names the two sides in
the order C, EdgeC, Edge. The settings may be connected to each other by an edge, a classical facet or both
(`_SettingsEdge`, `_SettingsC`, `_SettingsEdgeC`), but only when both sides are `Edge`: a setting that shares a
facet with its own outcome must have no other parent or facet, otherwise, conditional on the two settings, that
facet can leak the other party's setting to the outcome (0→2, 1→3; C{0,2}, C{0,1}, Q{2,3} has no gap: 2 learns 1
from 0 and the shared facet). The gap argument is the usual one: in any classical model the latent of Q{2,3} is
independent of the settings and of every other latent, and conditional on the settings each outcome's response
depends on its own setting, the shared latent and private randomness only, so P(2,3|0,1) is local; quantum models
violate a Bell inequality. A quantum facet between the settings is an upgrade of the classical one and is not
listed.
"""
from itertools import combinations_with_replacement
from typing import Dict

from hypergraphs import Hypergraph
from directed_structures import DirectedStructure
from quantum_mDAG import QmDAG

# Three visible nodes
# ---------------------------------------------------------------------------
QG_Instrumental_Edge = QmDAG(DirectedStructure([(0, 1), (1, 2)], 3), Hypergraph([], 3), Hypergraph([(1, 2)], 3))
QG_Instrumental_EdgeC = QmDAG(DirectedStructure([(0, 1), (1, 2)], 3), Hypergraph([(0, 1)], 3), Hypergraph([(1, 2)], 3))
QG_Instrumental_C = QmDAG(DirectedStructure([(1, 2)], 3), Hypergraph([(0, 1)], 3), Hypergraph([(1, 2)], 3))
QG_Triangle = QmDAG(DirectedStructure([], 3), Hypergraph([(1, 2), (2, 0)], 3), Hypergraph([(0, 1)], 3))
QG_Evans = QmDAG(DirectedStructure([(0, 1), (0, 2)], 3), Hypergraph([], 3), Hypergraph([(0, 1), (0, 2)], 3))

# Four visible nodes: the Bell family
# ---------------------------------------------------------------------------
_SIDE_ORDER = ('C', 'EdgeC', 'Edge')


def _bell(side_A: str, side_B: str, settings: str = '') -> QmDAG:
    """Settings 0 and 1, outcomes 2 and 3, Q{2,3}; each side's link is 'Edge', 'EdgeC' or 'C'; `settings` is '',
    'Edge', 'C' or 'EdgeC' (allowed only when both sides are 'Edge')."""
    edges, C = [], []
    for setting, outcome, side in ((0, 2, side_A), (1, 3, side_B)):
        if 'Edge' in side:
            edges.append((setting, outcome))
        if 'C' in side:
            C.append((setting, outcome))
    if settings:
        assert side_A == side_B == 'Edge', "settings may be connected only when both outcomes read a bare edge"
        if 'Edge' in settings:
            edges.append((0, 1))
        if 'C' in settings:
            C.append((0, 1))
    return QmDAG(DirectedStructure(sorted(edges), 4), Hypergraph(C, 4), Hypergraph([(2, 3)], 4))


def _bell_family() -> Dict[str, QmDAG]:
    family = {}
    for side_A, side_B in combinations_with_replacement(_SIDE_ORDER, 2):
        family[f"QG_Bell_{side_A}_{side_B}"] = _bell(side_A, side_B)
    for settings in ('Edge', 'C', 'EdgeC'):
        family[f"QG_Bell_Edge_Edge_Settings{settings}"] = _bell('Edge', 'Edge', settings)
    return family


globals().update(_bell_family())
QG_Bell_C_C: QmDAG
QG_Bell_C_EdgeC: QmDAG
QG_Bell_C_Edge: QmDAG
QG_Bell_EdgeC_EdgeC: QmDAG
QG_Bell_EdgeC_Edge: QmDAG
QG_Bell_Edge_Edge: QmDAG
QG_Bell_Edge_Edge_SettingsEdge: QmDAG
QG_Bell_Edge_Edge_SettingsC: QmDAG
QG_Bell_Edge_Edge_SettingsEdgeC: QmDAG


def weakest_and_distinct(named: Dict[str, QmDAG]) -> None:
    """Asserts that no structure is a degradation of another and that no two are relabellings of each other."""
    ids = {}
    for name, g in named.items():
        gid = g.unique_unlabelled_id
        assert gid not in ids, f"{name} is a relabelling of {ids[gid]}"
        ids[gid] = name
    for name, g in named.items():
        for _, d in g.degradation_steps():
            assert d.unique_unlabelled_id not in ids, f"{name} is an upgrade of {ids[d.unique_unlabelled_id]}"


def _named(prefix: str) -> Dict[str, QmDAG]:
    return {name: value for name, value in globals().items()
            if name.startswith(prefix) and isinstance(value, QmDAG)}


SEEDS_3_NODES = {name: g for name, g in _named('QG_').items() if g.number_of_visible == 3}
SEEDS_4_NODES = _named('QG_Bell')
SEEDS = {**SEEDS_3_NODES, **SEEDS_4_NODES}
weakest_and_distinct(SEEDS)
