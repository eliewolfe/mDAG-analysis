"""
Breadth-first search over causal structures connected by piggyback tricks.

A *piggyback* maps a QmDAG G to a QmDAG G' such that a quantum-classical (QC) gap in G' implies a QC gap in G. Every
transition is recorded with its provenance (which trick, with which parameters, from which structure), so that each
discovered gap comes with a human-readable certificate, and so that reachability restricted to any subset of tricks is
a cheap graph query over the recorded transitions. Structures are identified up to relabelling by their unlabelled id;
each id is expanded exactly once (all tricks are label-equivariant).
"""
from __future__ import annotations

import itertools
import warnings
from collections import Counter, deque
from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

from quantum_mDAG import QmDAG

UnlabelledId = Tuple[int, int, int, int]


@dataclass(frozen=True)
class Transition:
    trick: str
    params: Tuple
    source: UnlabelledId
    target: UnlabelledId


# --------------------------------------------------------------------------------------------------
# Per-trick transition functions: QmDAG -> iterable of (params, child QmDAG). Children may be any size;
# the explorer applies the visible-node bounds.
# --------------------------------------------------------------------------------------------------

def pd_trick(g: QmDAG) -> Iterable[Tuple[Tuple, QmDAG]]:
    """Fix one visible node to a point distribution (drop it)."""
    if g.number_of_visible <= 3:
        return
    for node in g.visible_nodes:
        yield (('drop', node),), g.fix_to_point_distribution_QmDAG(node)


def interruption(g: QmDAG) -> Iterable[Tuple[Tuple, QmDAG]]:
    if g.number_of_visible <= 3:
        return
    for node_with_no_children in sorted(g.vis_nodes_with_no_children):
        for node_with_no_parents in sorted(g.exogenous_visible_nodes):
            if node_with_no_children in g.directed_structure_instance.adjMat.descendantsplus_of(node_with_no_parents):
                continue
            yield ((('sink', node_with_no_children), ('source', node_with_no_parents)),
                   g.interruption_creation(node_with_no_children, node_with_no_parents))


def conditioning(g: QmDAG) -> Iterable[Tuple[Tuple, QmDAG]]:
    if g.number_of_visible <= 3:
        return
    for node in g.visible_nodes:
        if not g.has_grandparents_that_are_not_parents(node):
            yield (('condition', node),), g.condition(node)


def _marginalization(apply_teleportation: bool, districts_check: bool) -> Callable[[QmDAG], Iterable[Tuple[Tuple, QmDAG]]]:
    def marginalization(g: QmDAG) -> Iterable[Tuple[Tuple, QmDAG]]:
        if g.number_of_visible <= 3:
            return
        for node in g.visible_nodes:
            child = g.marginalize(node, districts_check=districts_check, apply_teleportation=apply_teleportation)
            if child is not None:
                yield (('marginalize', node),), child
    return marginalization


def _fritz(max_visible: int, keep_quantum_facets: bool, allow_childful_predictors: bool, max_predictors: int,
           districts_check: bool, apply_teleportation: bool) -> Callable[[QmDAG], Iterable[Tuple[Tuple, QmDAG]]]:
    def fritz(g: QmDAG) -> Iterable[Tuple[Tuple, QmDAG]]:
        pool = [y for y in g.visible_nodes
                if g.latent_siblings_of(y) and (allow_childful_predictors or y in g.vis_nodes_with_no_children)]
        for r in range(1, min(max_predictors, len(pool)) + 1):
            for predictors in itertools.combinations(pool, r):
                for params, child in g.fritz_transitions(predictors, max_visible=max_visible, min_visible=3,
                                                         keep_quantum_facets=keep_quantum_facets,
                                                         districts_check=districts_check,
                                                         allow_childful_predictors=allow_childful_predictors,
                                                         apply_teleportation=apply_teleportation):
                    yield (('predictors', predictors), ('predicted', params)), child
    return fritz


def default_tricks(max_visible: int = 5, keep_quantum_facets: bool = True, allow_childful_predictors: bool = True,
                   max_predictors: int = 2, districts_check: bool = False) -> Dict[str, Callable]:
    return {
        'PD': pd_trick,
        'interruption': interruption,
        'conditioning': conditioning,
        'naive_marginalization': _marginalization(apply_teleportation=False, districts_check=districts_check),
        'teleportation_marginalization': _marginalization(apply_teleportation=True, districts_check=districts_check),
        'Fritz': _fritz(max_visible, keep_quantum_facets, allow_childful_predictors, max_predictors,
                        districts_check, apply_teleportation=True),
    }


MARGINALIZATION_TRICKS = frozenset({'naive_marginalization', 'teleportation_marginalization'})

# For the report, "provable with T" uses the tricks in the group, and "provable only via T" removes just the trick
# itself from the full set. The Fritz trick is only meaningful together with marginalization (its outputs are
# typically larger than the original and need to be reduced), so its group includes the marginalizations.
TRICK_GROUPS_FOR_REPORT: Dict[str, Tuple[FrozenSet[str], FrozenSet[str]]] = {
    'PD': (frozenset({'PD'}), frozenset({'PD'})),
    'interruption': (frozenset({'interruption'}), frozenset({'interruption'})),
    'conditioning': (frozenset({'conditioning'}), frozenset({'conditioning'})),
    'naive_marginalization': (frozenset({'naive_marginalization'}), frozenset({'naive_marginalization'})),
    'teleportation_marginalization': (frozenset({'teleportation_marginalization'}),
                                      frozenset({'teleportation_marginalization'})),
    'Fritz (+ marginalization)': (frozenset({'Fritz'}) | MARGINALIZATION_TRICKS, frozenset({'Fritz'})),
}


# --------------------------------------------------------------------------------------------------
# The explorer
# --------------------------------------------------------------------------------------------------

class ClosureExplorer:
    def __init__(self, tricks: Dict[str, Callable], max_visible: int = 5, min_visible: int = 3,
                 max_states: int = 200000) -> None:
        self.tricks = tricks
        self.max_visible = max_visible
        self.min_visible = min_visible
        self.max_states = max_states
        self.representatives: Dict[UnlabelledId, QmDAG] = dict()
        self.edges: Dict[UnlabelledId, List[Transition]] = dict()  # populated once per expanded id

    def register(self, g: QmDAG) -> UnlabelledId:
        gid = g.unique_unlabelled_id
        self.representatives.setdefault(gid, g)
        return gid

    def _expand_one(self, gid: UnlabelledId) -> List[Transition]:
        if gid in self.edges:
            return self.edges[gid]
        g = self.representatives[gid]
        transitions: List[Transition] = []
        for trick_name, trick in self.tricks.items():
            for params, child in trick(g):
                if not (self.min_visible <= child.number_of_visible <= self.max_visible):
                    continue
                child_id = self.register(child)
                transitions.append(Transition(trick_name, params, gid, child_id))
        self.edges[gid] = transitions
        return transitions

    def expand(self, root: QmDAG) -> Set[UnlabelledId]:
        """Expands everything reachable from root under all tricks; returns the reachable ids (root included)."""
        root_id = self.register(root)
        reached = {root_id}
        frontier = deque([root_id])
        while frontier:
            current = frontier.popleft()
            for transition in self._expand_one(current):
                if transition.target not in reached:
                    reached.add(transition.target)
                    frontier.append(transition.target)
            if len(self.edges) > self.max_states:
                warnings.warn("ClosureExplorer exceeded max_states; results are incomplete.")
                break
        return reached

    def reachable(self, root_id: UnlabelledId, tricks: Optional[FrozenSet[str]] = None) -> Set[UnlabelledId]:
        """Ids reachable from an already-expanded root using only the given tricks (all tricks if None)."""
        reached = {root_id}
        frontier = deque([root_id])
        while frontier:
            current = frontier.popleft()
            for transition in self.edges.get(current, ()):
                if tricks is not None and transition.trick not in tricks:
                    continue
                if transition.target not in reached:
                    reached.add(transition.target)
                    frontier.append(transition.target)
        return reached

    def path(self, root_id: UnlabelledId, goals: Set[UnlabelledId],
             tricks: Optional[FrozenSet[str]] = None) -> Optional[List[Transition]]:
        """Shortest transition sequence from root to any goal id (empty list if root is itself a goal)."""
        if root_id in goals:
            return []
        parent: Dict[UnlabelledId, Transition] = dict()
        frontier = deque([root_id])
        seen = {root_id}
        while frontier:
            current = frontier.popleft()
            for transition in self.edges.get(current, ()):
                if tricks is not None and transition.trick not in tricks:
                    continue
                if transition.target in seen:
                    continue
                seen.add(transition.target)
                parent[transition.target] = transition
                if transition.target in goals:
                    chain = []
                    node = transition.target
                    while node != root_id:
                        chain.append(parent[node])
                        node = parent[node].source
                    return list(reversed(chain))
                frontier.append(transition.target)
        return None


# --------------------------------------------------------------------------------------------------
# Proving gaps
# --------------------------------------------------------------------------------------------------

@dataclass
class GapReport:
    inputs: List[QmDAG]
    seeds: Dict[str, QmDAG]
    proven: Dict[UnlabelledId, List[Transition]]        # input id -> certificate (chain of transitions to a seed)
    seed_hit: Dict[UnlabelledId, str]                   # input id -> name of the seed the certificate ends at
    remaining: List[QmDAG]
    provable_with: Dict[str, int]                        # per trick: number of inputs provable using its group alone
    only_via: Dict[str, int]                             # per trick: inputs no longer provable without it
    explorer: ClosureExplorer

    @property
    def counts(self) -> Dict[str, int]:
        """Counts are over input structures (several inputs may share an unlabelled id)."""
        return {'inputs': len(self.inputs), 'proven': len(self.inputs) - len(self.remaining),
                'remaining': len(self.remaining), 'proven_unique_ids': len(self.proven),
                **{'with ' + k: v for k, v in self.provable_with.items()},
                **{'only via ' + k: v for k, v in self.only_via.items()}}

    def certificate(self, g: QmDAG) -> str:
        return render_certificate(self.explorer, self.proven[g.unique_unlabelled_id], self.seed_hit[g.unique_unlabelled_id])


def _fixpoint(explorer: ClosureExplorer, input_ids: List[UnlabelledId], seed_ids: Set[UnlabelledId],
              tricks: Optional[FrozenSet[str]]) -> Set[UnlabelledId]:
    """Inputs provable when every proven input also counts as a known gap (closed under implication)."""
    proven = set(seed_ids)
    reach = {gid: explorer.reachable(gid, tricks) for gid in input_ids}
    unproven = set(input_ids)
    changed = True
    while changed:
        changed = False
        for gid in sorted(unproven):
            if not reach[gid].isdisjoint(proven):
                proven.add(gid)
                unproven.discard(gid)
                changed = True
    return proven.intersection(input_ids)


def prove_gaps(inputs: Iterable[QmDAG], seeds: Dict[str, QmDAG], tricks: Optional[Dict[str, Callable]] = None,
               max_visible: int = 5,
               trick_groups: Dict[str, Tuple[FrozenSet[str], FrozenSet[str]]] = TRICK_GROUPS_FOR_REPORT,
               verbose: bool = True) -> GapReport:
    inputs = list(inputs)
    if tricks is None:
        tricks = default_tricks(max_visible=max_visible)
    explorer = ClosureExplorer(tricks, max_visible=max_visible)
    seed_ids = {g.unique_unlabelled_id: name for name, g in seeds.items()}
    input_ids = []
    for i, g in enumerate(inputs):
        if verbose and i % 250 == 0:
            print(f"expanding {i} of {len(inputs)} inputs; {len(explorer.edges)} structures expanded so far")
        explorer.expand(g)
        input_ids.append(g.unique_unlabelled_id)
    unique_input_ids = list(dict.fromkeys(input_ids))
    multiplicity = Counter(input_ids)

    def count(ids: Set[UnlabelledId]) -> int:
        return sum(multiplicity[gid] for gid in ids)

    all_tricks = frozenset(tricks)
    proven_all = _fixpoint(explorer, unique_input_ids, set(seed_ids), all_tricks)
    provable_with = {name: count(_fixpoint(explorer, unique_input_ids, set(seed_ids), group))
                     for name, (group, _) in trick_groups.items()}
    only_via = {name: count(proven_all) - count(_fixpoint(explorer, unique_input_ids, set(seed_ids), all_tricks - removed))
                for name, (_, removed) in trick_groups.items()}

    # Certificates: a path to the nearest known gap (seed or already-certified input), chained down to a seed.
    proven: Dict[UnlabelledId, List[Transition]] = dict()
    seed_hit: Dict[UnlabelledId, str] = dict()
    known: Set[UnlabelledId] = set(seed_ids)
    pending = set(proven_all)
    while pending:
        progressed = False
        for gid in sorted(pending):
            if gid in seed_ids:
                proven[gid], seed_hit[gid] = [], seed_ids[gid]
            else:
                chain = explorer.path(gid, known, all_tricks)
                if chain is None:
                    continue
                last = chain[-1].target
                proven[gid] = chain + proven.get(last, [])
                seed_hit[gid] = seed_ids.get(last, seed_hit.get(last))
            known.add(gid)
            pending.discard(gid)
            progressed = True
        assert progressed, "certificate construction stalled"

    remaining = [g for g in inputs if g.unique_unlabelled_id not in proven_all]
    return GapReport(inputs, seeds, proven, seed_hit, remaining, provable_with, only_via, explorer)


def render_certificate(explorer: ClosureExplorer, chain: List[Transition], seed_name: str) -> str:
    if not chain:
        return f"is itself the known gap {seed_name}"
    lines = []
    for t in chain:
        source = explorer.representatives[t.source]
        lines.append(f"{t.trick}{t.params}:")
        lines.append("    " + source.as_string.replace("\n", "\n    ").rstrip())
    lines.append("== known gap " + seed_name + ":")
    lines.append("    " + explorer.representatives[chain[-1].target].as_string.replace("\n", "\n    ").rstrip())
    return "\n".join(lines)
