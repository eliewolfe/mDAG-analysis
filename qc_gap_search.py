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
from collections import deque
from dataclasses import dataclass, field
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


def _conditioning(strict_latents: bool) -> Callable[[QmDAG], Iterable[Tuple[Tuple, QmDAG]]]:
    def conditioning(g: QmDAG) -> Iterable[Tuple[Tuple, QmDAG]]:
        if g.number_of_visible <= 3:
            return
        for node in g.visible_nodes:
            if g.conditioning_is_justified(node, strict_latents=strict_latents):
                yield (('condition', node),), g.condition(node)
    return conditioning


conditioning = _conditioning(strict_latents=True)


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
           districts_check: bool, apply_teleportation: bool,
           predictor_mode: str = 'drop') -> Callable[[QmDAG], Iterable[Tuple[Tuple, QmDAG]]]:
    def fritz(g: QmDAG) -> Iterable[Tuple[Tuple, QmDAG]]:
        pool = [y for y in g.visible_nodes
                if g.latent_siblings_of(y) and (allow_childful_predictors or y in g.vis_nodes_with_no_children)]
        for r in range(1, min(max_predictors, len(pool)) + 1):
            for predictors in itertools.combinations(pool, r):
                for params, child in g.fritz_transitions(predictors, max_visible=max_visible, min_visible=3,
                                                         keep_quantum_facets=keep_quantum_facets,
                                                         districts_check=districts_check,
                                                         allow_childful_predictors=allow_childful_predictors,
                                                         apply_teleportation=apply_teleportation,
                                                         predictor_mode=predictor_mode):
                    yield (('predictors', predictors), ('predicted', params), ('predictor_mode', predictor_mode)), child
    return fritz


def _fritz_entropic(max_visible: int, keep_quantum_facets: bool, allow_childful_predictors: bool, max_predictors: int,
                    districts_check: bool, apply_teleportation: bool,
                    predictor_modes: Tuple[str, ...] = ('split',), extra_deletions: bool = True
                    ) -> Callable[[QmDAG], Iterable[Tuple[Tuple, QmDAG]]]:
    """The entropic Fritz piggyback. Only LP-reliant steps are emitted (certificate `markov` or `relabel`, or extra
    deletions): a step that plain d-separation certifies belongs to the Fritz trick."""
    def fritz_entropic(g: QmDAG) -> Iterable[Tuple[Tuple, QmDAG]]:
        pool = [y for y in g.visible_nodes
                if g.latent_siblings_of(y) and (allow_childful_predictors or y in g.vis_nodes_with_no_children)]
        for r in range(1, min(max_predictors, len(pool)) + 1):
            for predictors in itertools.combinations(pool, r):
                for params, child in g.fritz_entropic_transitions(
                        predictors, predictor_modes=predictor_modes, extra_deletions=extra_deletions,
                        max_visible=max_visible, min_visible=3, keep_quantum_facets=keep_quantum_facets,
                        districts_check=districts_check, allow_childful_predictors=allow_childful_predictors,
                        apply_teleportation=apply_teleportation, only_beyond_dsep=True,
                        base_predictor_modes=('drop', 'split')):
                    yield (('predictors', predictors),) + params, child
    return fritz_entropic


def entropic_tricks(max_visible: int = 5, keep_quantum_facets: bool = True, allow_childful_predictors: bool = True,
                    max_predictors: int = 1, districts_check: bool = False,
                    predictor_modes: Tuple[str, ...] = ('drop',), extra_deletions: bool = True) -> Dict[str, Callable]:
    """The LP-certified Fritz trick (not part of default_tricks: every emitted step needed LP solves). Predictors
    are dropped by default; predictor_modes=('split',) keeps them, which is the KPC construction (Section 7 of the
    manuscript) and is not run by default for cost."""
    return {'Fritz_entropic': _fritz_entropic(max_visible, keep_quantum_facets, allow_childful_predictors, max_predictors,
                                              districts_check, apply_teleportation=True,
                                              predictor_modes=predictor_modes, extra_deletions=extra_deletions)}


def elementary_tricks(max_visible: int = 5, districts_check: bool = False,
                      strict_conditioning: bool = True) -> Dict[str, Callable]:
    """The node-count-reducing piggybacks: point distribution, interruption, conditioning, marginalization."""
    return {
        'PD': pd_trick,
        'interruption': interruption,
        'conditioning': _conditioning(strict_latents=strict_conditioning),
        'naive_marginalization': _marginalization(apply_teleportation=False, districts_check=districts_check),
        'teleportation_marginalization': _marginalization(apply_teleportation=True, districts_check=districts_check),
    }


def fritz_tricks(max_visible: int = 5, keep_quantum_facets: bool = True, allow_childful_predictors: bool = True,
                 max_predictors: int = 2, districts_check: bool = False,
                 predictor_mode: str = 'drop') -> Dict[str, Callable]:
    """The Fritz piggyback with the d-separation certificate. 'drop' removes the predictors (trick `Fritz`), which
    is the composition of keeping them with marginalization and keeps the closure small; 'split' keeps them in
    place (trick `Fritz_kept`), which reaches more but makes an exhaustive closure far larger, so the default
    stages apply it to the still-unproven inputs only."""
    name = 'Fritz' if predictor_mode == 'drop' else 'Fritz_kept'
    return {name: _fritz(max_visible, keep_quantum_facets, allow_childful_predictors, max_predictors,
                         districts_check, apply_teleportation=True, predictor_mode=predictor_mode)}


def default_tricks(max_visible: int = 5, keep_quantum_facets: bool = True, allow_childful_predictors: bool = True,
                   max_predictors: int = 2, districts_check: bool = False,
                   predictor_mode: str = 'drop', strict_conditioning: bool = True) -> Dict[str, Callable]:
    """Elementary tricks plus Fritz (d-separation, dropped predictors): the tricks of the exhaustive closure."""
    return {**elementary_tricks(max_visible, districts_check, strict_conditioning),
            **fritz_tricks(max_visible, keep_quantum_facets, allow_childful_predictors, max_predictors,
                           districts_check, predictor_mode)}


Stage = Tuple[str, Dict[str, Callable], bool]   # (name, tricks, roots_only)


def default_stages(max_visible: int = 5, with_entropic: bool = True, with_kept: bool = False,
                   keep_quantum_facets: bool = True, allow_childful_predictors: bool = True, max_predictors: int = 2,
                   districts_check: bool = False, strict_conditioning: bool = True) -> List[Stage]:
    """The search runs in stages, cheapest first; every structure proven in a stage is a known gap for the next.
    (1) The elementary reductions, on everything reachable. (2) Fritz with the d-separation certificate, predictors
    dropped, on everything reachable. (3) The entropic Fritz piggyback, LP-certified steps only, predictors
    dropped, on the inputs still unproven. with_kept=True inserts a kept-predictor Fritz stage before (3) and lets
    (3) keep predictors too; both are applied to the unproven inputs only and are off by default for cost."""
    common = dict(max_visible=max_visible, keep_quantum_facets=keep_quantum_facets,
                  allow_childful_predictors=allow_childful_predictors, districts_check=districts_check)
    stages: List[Stage] = [
        ('elementary', elementary_tricks(max_visible, districts_check, strict_conditioning), False),
        ('Fritz', fritz_tricks(max_predictors=max_predictors, predictor_mode='drop', **common), False),
    ]
    if with_kept:
        stages.append(('Fritz_kept', fritz_tricks(max_predictors=1, predictor_mode='split', **common), True))
    if with_entropic:
        modes = ('drop', 'split') if with_kept else ('drop',)
        stages.append(('Fritz_entropic', entropic_tricks(max_predictors=1, predictor_modes=modes, **common), True))
    return stages


MARGINALIZATION_TRICKS = frozenset({'naive_marginalization', 'teleportation_marginalization'})

# For the report, "provable with T" uses the tricks in the group, and "provable only via T" removes just the trick
# itself from the full set. The Fritz trick is only meaningful together with marginalization (its outputs are
# typically larger than the original and need to be reduced), so its group includes the marginalizations.
FRITZ_TRICKS_ALL = frozenset({'Fritz', 'Fritz_kept', 'Fritz_entropic'})
TRICK_GROUPS_FOR_REPORT: Dict[str, Tuple[FrozenSet[str], FrozenSet[str]]] = {
    'PD': (frozenset({'PD'}), frozenset({'PD'})),
    'interruption': (frozenset({'interruption'}), frozenset({'interruption'})),
    'conditioning': (frozenset({'conditioning'}), frozenset({'conditioning'})),
    'naive_marginalization': (frozenset({'naive_marginalization'}), frozenset({'naive_marginalization'})),
    'teleportation_marginalization': (frozenset({'teleportation_marginalization'}),
                                      frozenset({'teleportation_marginalization'})),
    'Fritz (+ marginalization)': (frozenset({'Fritz'}) | MARGINALIZATION_TRICKS, frozenset({'Fritz'})),
}
KEPT_TRICK_GROUP = {
    'Fritz_kept (+ Fritz, marginalization)': (frozenset({'Fritz', 'Fritz_kept'}) | MARGINALIZATION_TRICKS,
                                              frozenset({'Fritz_kept'})),
}
ENTROPIC_TRICK_GROUP = {
    'Fritz_entropic (+ Fritz, marginalization)': (FRITZ_TRICKS_ALL | MARGINALIZATION_TRICKS,
                                                  frozenset({'Fritz_entropic'})),
}


# --------------------------------------------------------------------------------------------------
# The explorer
# --------------------------------------------------------------------------------------------------

class ClosureExplorer:
    def __init__(self, tricks: Dict[str, Callable], max_visible: int = 5, min_visible: int = 3,
                 max_states: int = 200000) -> None:
        self.tricks = dict(tricks)
        self.stage_tricks: List[FrozenSet[str]] = [frozenset(tricks)]   # cumulative trick sets, one per stage
        self.stage_names: List[str] = ['base']
        self.max_visible = max_visible
        self.min_visible = min_visible
        self.max_states = max_states
        self.representatives: Dict[UnlabelledId, QmDAG] = dict()
        self.edges: Dict[UnlabelledId, List[Transition]] = dict()  # populated once per expanded id
        self.applied: Dict[UnlabelledId, Set[str]] = dict()       # tricks already applied to each expanded id

    @property
    def base_tricks(self) -> FrozenSet[str]:
        """Tricks of the first stage (preferred in certificates)."""
        return self.stage_tricks[0]

    def register(self, g: QmDAG) -> UnlabelledId:
        gid = g.unique_unlabelled_id
        self.representatives.setdefault(gid, g)
        return gid

    def _expand_one(self, gid: UnlabelledId) -> List[Transition]:
        """Applies every trick not yet applied to this id; returns all of its transitions."""
        pending = [name for name in self.tricks if name not in self.applied.get(gid, set())]
        if not pending:
            return self.edges.get(gid, [])
        g = self.representatives[gid]
        for trick_name in pending:
            new_transitions = []
            for params, child in self.tricks[trick_name](g):   # collected first: a failure leaves no partial record
                if not (self.min_visible <= child.number_of_visible <= self.max_visible):
                    continue
                child_id = self.register(child)
                new_transitions.append(Transition(trick_name, params, gid, child_id))
            self.edges.setdefault(gid, []).extend(new_transitions)
            self.applied.setdefault(gid, set()).add(trick_name)
        return self.edges.get(gid, [])

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

    def begin_stage(self, name: str, extra_tricks: Dict[str, Callable]) -> None:
        """Records a new stage (for certificate preference and per-stage counts); idempotent per name."""
        if name not in self.stage_names:
            self.stage_names.append(name)
            self.stage_tricks.append(self.stage_tricks[-1] | frozenset(extra_tricks))

    def extend(self, extra_tricks: Dict[str, Callable], roots: Iterable[QmDAG], roots_only: bool = True,
               stage: Optional[str] = None) -> None:
        """Applies extra tricks. With roots_only, they are applied to the roots alone and the new children are
        expanded with the tricks already in the explorer; otherwise the extra tricks join the trick set and are
        applied to everything reachable from the roots. `stage` names the stage for reporting."""
        self.begin_stage(stage or '+'.join(sorted(extra_tricks)), extra_tricks)
        if roots_only:
            for root in roots:
                self.expand(root)   # no-op when already expanded; closes the root under the base tricks
                gid = self.register(root)
                representative = self.representatives[gid]   # params are stated in the representative's labels
                new_children = []
                for trick_name, trick in extra_tricks.items():
                    if trick_name in self.applied.get(gid, set()):
                        continue
                    transitions = []
                    for params, child in trick(representative):
                        if not (self.min_visible <= child.number_of_visible <= self.max_visible):
                            continue
                        transitions.append(Transition(trick_name, params, gid, self.register(child)))
                    self.edges.setdefault(gid, []).extend(transitions)
                    self.applied.setdefault(gid, set()).add(trick_name)
                    new_children.extend(t.target for t in transitions)
                for child_id in new_children:
                    self.expand(self.representatives[child_id])
            return
        self.tricks.update(extra_tricks)
        for root in roots:
            self.expand(root)

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
    """All counts are up to relabelling: an input structure is identified with its unlabelled id, so inputs that
    are relabellings of each other count once. `inputs` keeps the structures as given (with duplicates);
    `remaining` holds one representative per unproven unlabelled id."""
    inputs: List[QmDAG]
    seeds: Dict[str, QmDAG]
    proven: Dict[UnlabelledId, List[Transition]]        # input id -> certificate (chain of transitions to a seed)
    seed_hit: Dict[UnlabelledId, str]                   # input id -> name of the seed the certificate ends at
    remaining: List[QmDAG]
    provable_with: Dict[str, int]                        # per trick: number of input ids provable using its group alone
    only_via: Dict[str, int]                             # per trick: input ids no longer provable without it
    explorer: ClosureExplorer
    stage_counts: List[Tuple[str, int]] = field(default_factory=list)   # cumulative: proven after each stage

    @property
    def input_ids(self) -> List[UnlabelledId]:
        return list(dict.fromkeys(g.unique_unlabelled_id for g in self.inputs))

    @property
    def counts(self) -> Dict[str, int]:
        """Counts up to relabelling (distinct unlabelled ids)."""
        return {'inputs': len(self.input_ids), 'proven': len(self.proven), 'remaining': len(self.remaining),
                'labelled_inputs': len(self.inputs),
                **{'with ' + k: v for k, v in self.provable_with.items()},
                **{'only via ' + k: v for k, v in self.only_via.items()}}

    def certificate(self, g: QmDAG) -> str:
        return render_certificate(self.explorer, self.proven[g.unique_unlabelled_id], self.seed_hit[g.unique_unlabelled_id])

    def proven_structure_ids(self) -> Set[UnlabelledId]:
        """Every structure the search touched (inputs, intermediates, hybrids with classical facets) that reaches a
        seed: the known-gap database the search has established. A structure here is a QC gap by the chain of
        piggybacks from it to a seed; nothing is inferred from a structure to a weaker variant of it."""
        seed_ids = {g.unique_unlabelled_id for g in self.seeds.values()}
        reverse: Dict[UnlabelledId, List[UnlabelledId]] = {}
        for transitions in self.explorer.edges.values():
            for t in transitions:
                reverse.setdefault(t.target, []).append(t.source)
        known = set(seed_ids) & (set(self.explorer.representatives) | seed_ids)
        frontier = deque(known)
        while frontier:
            current = frontier.popleft()
            for source in reverse.get(current, ()):
                if source not in known:
                    known.add(source)
                    frontier.append(source)
        return known


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


def _trick_groups_for(explorer: ClosureExplorer, trick_groups):
    if trick_groups is not None:
        return trick_groups
    groups = dict(TRICK_GROUPS_FOR_REPORT)
    present = set(explorer.tricks).union(*explorer.stage_tricks)
    if 'Fritz_kept' in present:
        groups.update(KEPT_TRICK_GROUP)
    if 'Fritz_entropic' in present:
        groups.update(ENTROPIC_TRICK_GROUP)
    return groups


def prove_gaps(inputs: Iterable[QmDAG], seeds: Dict[str, QmDAG], tricks: Optional[Dict[str, Callable]] = None,
               max_visible: int = 5,
               trick_groups: Optional[Dict[str, Tuple[FrozenSet[str], FrozenSet[str]]]] = None,
               verbose: bool = True, stages: Optional[List[Stage]] = None, with_entropic: bool = True) -> GapReport:
    """Proves QC gaps for `inputs` from the known gaps `seeds`. With `tricks`, a single closure stage under those
    tricks. Otherwise the stages of `default_stages` (or the given `stages`) are run in order, cheapest first;
    every structure proven in a stage counts as a known gap for the later ones (reachability is transitive, so
    this holds whatever the order; the order decides cost and which tricks certificates prefer)."""
    inputs = list(inputs)
    if stages is None:
        stages = [('base', dict(tricks), False)] if tricks is not None \
            else default_stages(max_visible=max_visible, with_entropic=with_entropic)
    name, first, _ = stages[0]
    explorer = ClosureExplorer(first, max_visible=max_visible)
    explorer.stage_names[0] = name
    for i, g in enumerate(inputs):
        if verbose and i % 250 == 0:
            print(f"[{name}] expanding {i} of {len(inputs)} inputs; {len(explorer.edges)} structures expanded so far")
        explorer.expand(g)
    report = build_report(explorer, inputs, seeds, _trick_groups_for(explorer, trick_groups))
    for name, extra, roots_only in stages[1:]:
        report = add_stage(report, extra, trick_groups=trick_groups, verbose=verbose, roots_only=roots_only, name=name)
    return report


def add_stage(report: GapReport, extra_tricks: Dict[str, Callable],
              trick_groups: Optional[Dict[str, Tuple[FrozenSet[str], FrozenSet[str]]]] = None,
              verbose: bool = True, roots_only: bool = True, name: Optional[str] = None) -> GapReport:
    """Applies further tricks to the explorer of `report` and rebuilds the report. With roots_only the tricks are
    applied to the still-unproven inputs only (their children are expanded with the tricks already present);
    otherwise to everything reachable from every input."""
    explorer = report.explorer
    name = name or '+'.join(sorted(extra_tricks))
    if 'Fritz_entropic' in extra_tricks:
        try:
            import mosek  # noqa: F401
        except ImportError:
            warnings.warn("mosek is not installed; the entropic stage is skipped.")
            return report
    roots = list(dict.fromkeys(report.remaining)) if roots_only else list({g.unique_unlabelled_id: g for g in report.inputs}.values())
    for i, g in enumerate(roots):
        if verbose and i % 250 == 0:
            print(f"[{name}] expanding {i} of {len(roots)} roots; {len(explorer.edges)} structures")
        explorer.extend(extra_tricks, [g], roots_only=roots_only, stage=name)
    return build_report(explorer, report.inputs, report.seeds, _trick_groups_for(explorer, trick_groups))


def build_report(explorer: ClosureExplorer, inputs: List[QmDAG], seeds: Dict[str, QmDAG],
                 trick_groups: Dict[str, Tuple[FrozenSet[str], FrozenSet[str]]]) -> GapReport:
    seed_ids = {g.unique_unlabelled_id: name for name, g in seeds.items()}
    input_ids = [g.unique_unlabelled_id for g in inputs]
    unique_input_ids = list(dict.fromkeys(input_ids))
    tricks = dict(explorer.tricks)
    for transitions in explorer.edges.values():
        for t in transitions:
            tricks.setdefault(t.trick, None)
    def count(ids: Set[UnlabelledId]) -> int:
        return len(ids)   # up to relabelling

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
                # Prefer a certificate using the tricks of the earliest possible stage.
                chain = None
                for allowed in list(explorer.stage_tricks) + [all_tricks]:
                    chain = explorer.path(gid, known, allowed)
                    if chain is not None:
                        break
                if chain is None:
                    continue
                last = chain[-1].target
                proven[gid] = chain + proven.get(last, [])
                seed_hit[gid] = seed_ids.get(last, seed_hit.get(last))
            known.add(gid)
            pending.discard(gid)
            progressed = True
        assert progressed, "certificate construction stalled"

    remaining_by_id = {g.unique_unlabelled_id: g for g in inputs if g.unique_unlabelled_id not in proven_all}
    remaining = list(remaining_by_id.values())
    stage_counts = [(name, count(_fixpoint(explorer, unique_input_ids, set(seed_ids), allowed)))
                    for name, allowed in zip(explorer.stage_names, explorer.stage_tricks)]
    return GapReport(inputs, seeds, proven, seed_hit, remaining, provable_with, only_via, explorer, stage_counts)


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
