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

# Bump a piggyback's version when its map or its admissibility rule changes; cached proofs that relied on it are
# then discarded (gap_cache.py). Fritz-type tricks share the mechanism but are versioned separately.
PIGGYBACK_VERSIONS: Dict[str, int] = {
    'PD': 1,
    'interruption': 1,
    'conditioning': 3,                 # 1 visible grandparents; 2 latent grandparents; 3 guessing parents' children
    'naive_marginalization': 1,
    'teleportation_marginalization': 1,
    'Fritz': 3,                        # 1 original; 2 common/others formulation; 3 predictors removed soundly
    'Fritz_kept': 1,                   # childless kept untouched, childful split and copy marginalized
    'Fritz_entropic': 2,               # 1 rescue; 2 LP-only steps, relabel first, summed-row LP
}


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
           predictor_mode: str = 'drop', modes: Tuple[str, ...] = ('replace', 'copy')) -> Callable[[QmDAG], Iterable[Tuple[Tuple, QmDAG]]]:
    def fritz(g: QmDAG) -> Iterable[Tuple[Tuple, QmDAG]]:
        pool = [y for y in g.visible_nodes
                if g.latent_siblings_of(y) and (allow_childful_predictors or y in g.vis_nodes_with_no_children)]
        for r in range(1, min(max_predictors, len(pool)) + 1):
            for predictors in itertools.combinations(pool, r):
                for params, child in g.fritz_transitions(predictors, modes=modes, max_visible=max_visible, min_visible=3,
                                                         keep_quantum_facets=keep_quantum_facets,
                                                         districts_check=districts_check,
                                                         allow_childful_predictors=allow_childful_predictors,
                                                         apply_teleportation=apply_teleportation,
                                                         predictor_mode=predictor_mode):
                    yield (('predictors', predictors), ('predicted', params), ('predictor_mode', predictor_mode)), child
    return fritz


def _fritz_entropic(max_visible: int, keep_quantum_facets: bool, allow_childful_predictors: bool, max_predictors: int,
                    districts_check: bool, apply_teleportation: bool,
                    predictor_modes: Tuple[str, ...] = ('split',), extra_deletions: bool = False,
                    modes: Tuple[str, ...] = ('replace', 'copy')) -> Callable[[QmDAG], Iterable[Tuple[Tuple, QmDAG]]]:
    """The entropic Fritz piggyback. Only LP-reliant steps are emitted (certificate `markov` or `relabel`, or extra
    deletions): a step that plain d-separation certifies belongs to the Fritz trick."""
    def fritz_entropic(g: QmDAG) -> Iterable[Tuple[Tuple, QmDAG]]:
        pool = [y for y in g.visible_nodes
                if g.latent_siblings_of(y) and (allow_childful_predictors or y in g.vis_nodes_with_no_children)]
        for r in range(1, min(max_predictors, len(pool)) + 1):
            for predictors in itertools.combinations(pool, r):
                for params, child in g.fritz_entropic_transitions(
                        predictors, modes=modes, predictor_modes=predictor_modes, extra_deletions=extra_deletions,
                        max_visible=max_visible, min_visible=3, keep_quantum_facets=keep_quantum_facets,
                        districts_check=districts_check, allow_childful_predictors=allow_childful_predictors,
                        apply_teleportation=apply_teleportation, only_beyond_dsep=True,
                        base_predictor_modes=('drop', 'split')):
                    yield (('predictors', predictors),) + params, child
    return fritz_entropic


def entropic_tricks(max_visible: int = 5, keep_quantum_facets: bool = True, allow_childful_predictors: bool = True,
                    max_predictors: int = 1, districts_check: bool = False,
                    predictor_modes: Tuple[str, ...] = ('drop',), extra_deletions: bool = False,
                    modes: Tuple[str, ...] = ('replace', 'copy')) -> Dict[str, Callable]:
    """The LP-certified Fritz trick (not part of default_tricks: every emitted step needed LP solves). Predictors
    are dropped by default; predictor_modes=('split',) keeps them (the KPC construction); default_stages runs both.
    Steps that plain d-separation certifies are never emitted (base_predictor_modes covers both modes), so a search
    that wants d-separation-certified kept-predictor steps must also run fritz_tricks(predictor_mode='split')."""
    return {'Fritz_entropic': _fritz_entropic(max_visible, keep_quantum_facets, allow_childful_predictors, max_predictors,
                                              districts_check, apply_teleportation=True,
                                              predictor_modes=predictor_modes, extra_deletions=extra_deletions, modes=modes)}


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
                 max_predictors: int = 1, districts_check: bool = False,
                 predictor_mode: str = 'drop', modes: Tuple[str, ...] = ('replace', 'copy')) -> Dict[str, Callable]:
    """The Fritz piggyback with the d-separation certificate. 'drop' removes the predictors (trick `Fritz`), which
    is the composition of keeping them with marginalization and keeps the closure small; 'split' keeps them in
    place (trick `Fritz_kept`), which reaches more but makes an exhaustive closure far larger, so the default
    stages apply it to the still-unproven inputs only."""
    name = 'Fritz' if predictor_mode == 'drop' else 'Fritz_kept'
    return {name: _fritz(max_visible, keep_quantum_facets, allow_childful_predictors, max_predictors,
                         districts_check, apply_teleportation=True, predictor_mode=predictor_mode, modes=modes)}


def default_tricks(max_visible: int = 5, keep_quantum_facets: bool = True, allow_childful_predictors: bool = True,
                   max_predictors: int = 1, districts_check: bool = False,
                   predictor_mode: str = 'drop', strict_conditioning: bool = True) -> Dict[str, Callable]:
    """Elementary tricks plus Fritz (d-separation, dropped predictors): the tricks of the exhaustive closure."""
    return {**elementary_tricks(max_visible, districts_check, strict_conditioning),
            **fritz_tricks(max_visible, keep_quantum_facets, allow_childful_predictors, max_predictors,
                           districts_check, predictor_mode)}


Stage = Tuple[str, Dict[str, Callable], bool, Optional[FrozenSet[str]]]   # (name, tricks, roots_only, follow-up tricks)


def default_stages(max_visible: int = 5, with_entropic: bool = True, with_kept: bool = True,
                   keep_quantum_facets: bool = True, allow_childful_predictors: bool = True, max_predictors: int = 1,
                   districts_check: bool = False, strict_conditioning: bool = True) -> List[Stage]:
    """A cascade of stages, cheapest first; each runs only on the inputs the earlier ones left unproven, and every
    structure proven in a stage is a known gap for the next.
    (1) The elementary reductions, closed over everything reachable from every input.
    Then eight Fritz-type stages (the rungs of CASCADE), each applied once to each still-unproven input, its outputs
    reduced with the elementary tricks only ("depth one"): Fritz with dropped predictors in replace mode, then in
    copy mode; Fritz with kept predictors (childless untouched, childful split and the copy marginalized) in
    replace mode, then in copy mode; the entropic Fritz piggyback (LP-certified steps only) with dropped
    predictors in replace mode, then copy mode, then with kept predictors in replace mode, then copy mode. Joint
    predictor sets and LP-driven extra deletions are available (max_predictors, entropic_tricks) but off."""
    common = dict(max_visible=max_visible, keep_quantum_facets=keep_quantum_facets,
                  allow_childful_predictors=allow_childful_predictors, districts_check=districts_check)
    elementary = elementary_tricks(max_visible, districts_check, strict_conditioning)
    reductions = frozenset(elementary)
    names = [name for name, _ in CASCADE]
    stages: List[Stage] = [
        (names[0], elementary, False, None),
        (names[1], fritz_tricks(max_predictors=max_predictors, predictor_mode='drop', modes=('replace',), **common), True, reductions),
        (names[2], fritz_tricks(max_predictors=max_predictors, predictor_mode='drop', modes=('copy',), **common), True, reductions),
    ]
    if with_kept:
        stages.append((names[3], fritz_tricks(max_predictors=max_predictors, predictor_mode='split', modes=('replace',), **common), True, reductions))
        stages.append((names[4], fritz_tricks(max_predictors=max_predictors, predictor_mode='split', modes=('copy',), **common), True, reductions))
    if with_entropic:
        stages.append((names[5], entropic_tricks(max_predictors=max_predictors, predictor_modes=('drop',), modes=('replace',), **common), True, reductions))
        stages.append((names[6], entropic_tricks(max_predictors=max_predictors, predictor_modes=('drop',), modes=('copy',), **common), True, reductions))
        if with_kept:
            stages.append((names[7], entropic_tricks(max_predictors=max_predictors, predictor_modes=('split',), modes=('replace',), **common), True, reductions))
            stages.append((names[8], entropic_tricks(max_predictors=max_predictors, predictor_modes=('split',), modes=('copy',), **common), True, reductions))
    return stages


MARGINALIZATION_TRICKS = frozenset({'naive_marginalization', 'teleportation_marginalization'})

# For the report, "provable with T" uses the tricks in the group, and "provable only via T" removes just the trick
# itself from the full set. The Fritz trick is only meaningful together with marginalization (its outputs are
# typically larger than the original and need to be reduced), so its group includes the marginalizations.
FRITZ_TRICKS_ALL = frozenset({'Fritz', 'Fritz_kept', 'Fritz_entropic'})
# "Provable with T alone" and "provable only via T" are reported for the elementary reductions only; the expensive
# tricks are assessed by the cumulative ladder and the step categories of `fritz_breakdown`.
TRICK_GROUPS_FOR_REPORT: Dict[str, Tuple[FrozenSet[str], FrozenSet[str]]] = {
    'PD': (frozenset({'PD'}), frozenset({'PD'})),
    'interruption': (frozenset({'interruption'}), frozenset({'interruption'})),
    'conditioning': (frozenset({'conditioning'}), frozenset({'conditioning'})),
    'naive_marginalization': (frozenset({'naive_marginalization'}), frozenset({'naive_marginalization'})),
    'teleportation_marginalization': (frozenset({'teleportation_marginalization'}),
                                      frozenset({'teleportation_marginalization'})),
    'marginalization (either kind)': (MARGINALIZATION_TRICKS, MARGINALIZATION_TRICKS),
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
        self.current_stage: int = 0
        self.stage_of: Dict[Transition, int] = dict()   # the stage in which each transition was first recorded
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

    def _expand_one(self, gid: UnlabelledId, only: Optional[FrozenSet[str]] = None) -> List[Transition]:
        """Applies every trick (or every trick in `only`) not yet applied to this id; returns its transitions."""
        pending = [name for name in self.tricks if name not in self.applied.get(gid, set())
                   and (only is None or name in only)]
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
            self._record(gid, new_transitions)
            self.applied.setdefault(gid, set()).add(trick_name)
        return self.edges.get(gid, [])

    def _record(self, gid: UnlabelledId, transitions: List[Transition]) -> None:
        self.edges.setdefault(gid, []).extend(transitions)
        for t in transitions:
            self.stage_of.setdefault(t, self.current_stage)

    def stage_filter(self, stage_index: int) -> Callable[[Transition], bool]:
        """Predicate: transitions recorded in stages up to and including `stage_index`."""
        return lambda t: self.stage_of.get(t, 0) <= stage_index

    def expand(self, root: QmDAG, only: Optional[FrozenSet[str]] = None) -> Set[UnlabelledId]:
        """Expands everything reachable from root under all tricks (or under the tricks in `only`); returns the
        reachable ids (root included)."""
        root_id = self.register(root)
        reached = {root_id}
        frontier = deque([root_id])
        while frontier:
            current = frontier.popleft()
            for transition in self._expand_one(current, only):
                if only is not None and transition.trick not in only:
                    continue
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
        self.current_stage = self.stage_names.index(name)

    def extend(self, extra_tricks: Dict[str, Callable], roots: Iterable[QmDAG], roots_only: bool = True,
               stage: Optional[str] = None, followup: Optional[FrozenSet[str]] = None) -> None:
        """Applies extra tricks. With roots_only, they are applied to the roots alone and the new children are
        expanded with the tricks already in the explorer, or only with those in `followup`; otherwise the extra
        tricks join the trick set and are applied to everything reachable from the roots. `stage` names the stage."""
        self.begin_stage(stage or '+'.join(sorted(extra_tricks)), extra_tricks)
        if roots_only:
            for root in roots:
                self.expand(root)   # no-op when already expanded; closes the root under the base tricks
                gid = self.register(root)
                representative = self.representatives[gid]   # params are stated in the representative's labels
                new_children = []
                for trick_name, trick in extra_tricks.items():
                    # The same trick name may return in a later stage with other parameters (e.g. copy mode after
                    # replace mode), so roots-only applications are remembered per stage.
                    applied_key = f"{self.stage_names[self.current_stage]}:{trick_name}"
                    if applied_key in self.applied.get(gid, set()):
                        continue
                    transitions = []
                    for params, child in trick(representative):
                        if not (self.min_visible <= child.number_of_visible <= self.max_visible):
                            continue
                        transitions.append(Transition(trick_name, params, gid, self.register(child)))
                    self._record(gid, transitions)
                    self.applied.setdefault(gid, set()).add(applied_key)
                    new_children.extend(t.target for t in transitions)
                for child_id in new_children:
                    self.expand(self.representatives[child_id], only=followup)
            return
        self.tricks.update(extra_tricks)
        for root in roots:
            self.expand(root)

    def reachable(self, root_id: UnlabelledId, tricks: Optional[FrozenSet[str]] = None,
                  keep: Optional[Callable[[Transition], bool]] = None) -> Set[UnlabelledId]:
        """Ids reachable from an already-expanded root using only the given tricks (all tricks if None) and only
        transitions satisfying `keep` (all if None)."""
        reached = {root_id}
        frontier = deque([root_id])
        while frontier:
            current = frontier.popleft()
            for transition in self.edges.get(current, ()):
                if tricks is not None and transition.trick not in tricks:
                    continue
                if keep is not None and not keep(transition):
                    continue
                if transition.target not in reached:
                    reached.add(transition.target)
                    frontier.append(transition.target)
        return reached

    def path(self, root_id: UnlabelledId, goals: Set[UnlabelledId],
             tricks: Optional[FrozenSet[str]] = None,
             keep: Optional[Callable[[Transition], bool]] = None) -> Optional[List[Transition]]:
        """Shortest transition sequence from root to any goal id (empty list if root is itself a goal), using
        only the given tricks and only transitions satisfying `keep`."""
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
                if keep is not None and not keep(transition):
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
    known: Dict[str, UnlabelledId] = field(default_factory=dict)        # extra known gaps by id (cache), act as seeds

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
        seed_ids = {g.unique_unlabelled_id for g in self.seeds.values()} | set(self.known.values())
        reverse: Dict[UnlabelledId, List[UnlabelledId]] = {}
        for transitions in self.explorer.edges.values():
            for t in transitions:
                reverse.setdefault(t.target, []).append(t.source)
        known = set(seed_ids)
        frontier = deque(known)
        while frontier:
            current = frontier.popleft()
            for source in reverse.get(current, ()):
                if source not in known:
                    known.add(source)
                    frontier.append(source)
        return known & set(self.explorer.representatives)


def _fixpoint(explorer: ClosureExplorer, input_ids: List[UnlabelledId], seed_ids: Set[UnlabelledId],
              tricks: Optional[FrozenSet[str]], keep: Optional[Callable[[Transition], bool]] = None) -> Set[UnlabelledId]:
    """Inputs provable when every proven input also counts as a known gap (closed under implication), using the
    given tricks and the transitions satisfying `keep`."""
    proven = set(seed_ids)
    reach = {gid: explorer.reachable(gid, tricks, keep) for gid in input_ids}
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
    return TRICK_GROUPS_FOR_REPORT if trick_groups is None else trick_groups


def prove_gaps(inputs: Iterable[QmDAG], seeds: Dict[str, QmDAG], tricks: Optional[Dict[str, Callable]] = None,
               max_visible: int = 5,
               trick_groups: Optional[Dict[str, Tuple[FrozenSet[str], FrozenSet[str]]]] = None,
               verbose: bool = True, stages: Optional[List[Stage]] = None, with_entropic: bool = True,
               known: Optional[Dict[str, UnlabelledId]] = None) -> GapReport:
    """Proves QC gaps for `inputs` from the known gaps `seeds`. With `tricks`, a single closure stage under those
    tricks. Otherwise the stages of `default_stages` (or the given `stages`) are run in order, cheapest first;
    every structure proven in a stage counts as a known gap for the later ones (reachability is transitive, so
    this holds whatever the order; the order decides cost and which tricks certificates prefer)."""
    inputs = list(inputs)
    if stages is None:
        stages = [('base', dict(tricks), False, None)] if tricks is not None \
            else default_stages(max_visible=max_visible, with_entropic=with_entropic)
    name, first = stages[0][0], stages[0][1]
    explorer = ClosureExplorer(first, max_visible=max_visible)
    explorer.stage_names[0] = name
    for i, g in enumerate(inputs):
        if verbose and i % 250 == 0:
            print(f"[{name}] expanding {i} of {len(inputs)} inputs; {len(explorer.edges)} structures expanded so far")
        explorer.expand(g)
    report = build_report(explorer, inputs, seeds, _trick_groups_for(explorer, trick_groups), known=known)
    for stage in stages[1:]:
        name, extra, roots_only = stage[0], stage[1], stage[2]
        followup = stage[3] if len(stage) > 3 else None
        report = add_stage(report, extra, trick_groups=trick_groups, verbose=verbose, roots_only=roots_only, name=name,
                           followup=followup)
    return report


def add_stage(report: GapReport, extra_tricks: Dict[str, Callable],
              trick_groups: Optional[Dict[str, Tuple[FrozenSet[str], FrozenSet[str]]]] = None,
              verbose: bool = True, roots_only: bool = True, name: Optional[str] = None,
              followup: Optional[FrozenSet[str]] = None, seeds: Optional[Dict[str, QmDAG]] = None,
              inputs: Optional[List[QmDAG]] = None, known: Optional[Dict[str, UnlabelledId]] = None) -> GapReport:
    """Applies further tricks to the explorer of `report` and rebuilds the report. With roots_only the tricks are
    applied to the still-unproven inputs only, and their children are expanded with the tricks already present
    (or only with those in `followup`); otherwise to everything reachable from every input."""
    explorer = report.explorer
    name = name or '+'.join(sorted(extra_tricks))
    if 'Fritz_entropic' in extra_tricks:
        try:
            import mosek  # noqa: F401
        except ImportError:
            warnings.warn("mosek is not installed; the entropic stage is skipped.")
            return report
    seeds = report.seeds if seeds is None else seeds
    inputs = report.inputs if inputs is None else list(inputs)
    known = report.known if known is None else known
    if seeds is not report.seeds or inputs is not report.inputs or known is not report.known:
        report = build_report(explorer, inputs, seeds, _trick_groups_for(explorer, trick_groups), known=known)
    roots = list(dict.fromkeys(report.remaining)) if roots_only else list({g.unique_unlabelled_id: g for g in inputs}.values())
    for i, g in enumerate(roots):
        if verbose and i % 250 == 0:
            print(f"[{name}] expanding {i} of {len(roots)} roots; {len(explorer.edges)} structures")
        explorer.extend(extra_tricks, [g], roots_only=roots_only, stage=name, followup=followup)
    return build_report(explorer, inputs, seeds, _trick_groups_for(explorer, trick_groups), known=known)


def build_report(explorer: ClosureExplorer, inputs: List[QmDAG], seeds: Dict[str, QmDAG],
                 trick_groups: Dict[str, Tuple[FrozenSet[str], FrozenSet[str]]],
                 known: Optional[Dict[str, UnlabelledId]] = None) -> GapReport:
    """`known` are further known gaps given by unlabelled id (e.g. from the cache); they act as seeds."""
    known = known or {}
    seed_ids = {g.unique_unlabelled_id: name for name, g in seeds.items()}
    seed_ids.update({gid: name for name, gid in known.items()})
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
    goals: Set[UnlabelledId] = set(seed_ids)
    pending = set(proven_all)
    while pending:
        progressed = False
        for gid in sorted(pending):
            if gid in seed_ids:
                proven[gid], seed_hit[gid] = [], seed_ids[gid]
            else:
                # Prefer a certificate using the transitions of the earliest possible stage.
                chain = None
                for k in range(len(explorer.stage_names)):
                    chain = explorer.path(gid, goals, keep=explorer.stage_filter(k))
                    if chain is not None:
                        break
                if chain is None:
                    chain = explorer.path(gid, goals)
                if chain is None:
                    continue
                last = chain[-1].target
                proven[gid] = chain + proven.get(last, [])
                seed_hit[gid] = seed_ids.get(last, seed_hit.get(last))
            goals.add(gid)
            pending.discard(gid)
            progressed = True
        assert progressed, "certificate construction stalled"

    remaining_by_id = {g.unique_unlabelled_id: g for g in inputs if g.unique_unlabelled_id not in proven_all}
    remaining = list(remaining_by_id.values())
    # Cumulative: proven using the transitions recorded in stages up to k (a stage's elementary follow-up
    # transitions belong to that stage).
    stage_counts = [(name, count(_fixpoint(explorer, unique_input_ids, set(seed_ids), None, explorer.stage_filter(k))))
                    for k, name in enumerate(explorer.stage_names)]
    return GapReport(inputs, seeds, proven, seed_hit, remaining, provable_with, only_via, explorer, stage_counts, known)


# --------------------------------------------------------------------------------------------------
# Assessing the expensive (Fritz-type) steps: categories lost when removed, and the cumulative ladder
# --------------------------------------------------------------------------------------------------

def predicted_modes(t: Transition) -> Tuple:
    """((s, mode), ...) of a Fritz-type transition, () otherwise."""
    if t.trick in ('Fritz', 'Fritz_kept'):
        return dict(t.params)['predicted']
    if t.trick == 'Fritz_entropic':
        return t.params[1]
    return ()


def predictor_mode_of(t: Transition) -> Optional[str]:
    if t.trick == 'Fritz':
        return 'drop'
    if t.trick == 'Fritz_kept':
        return 'split'
    if t.trick == 'Fritz_entropic':
        return dict(t.params[2:]).get('predictor_mode')
    return None


def certificate_of(t: Transition) -> Optional[str]:
    return dict(t.params[2:]).get('certificate') if t.trick == 'Fritz_entropic' else ('dsep' if t.trick in ('Fritz', 'Fritz_kept') else None)


def uses_copy(t: Transition) -> bool:
    return any(mode == 'copy' for _, mode in predicted_modes(t))


def is_fritz_type(t: Transition) -> bool:
    return t.trick in FRITZ_TRICKS_ALL


STEP_CATEGORIES: Dict[str, Callable[[Transition], bool]] = {   # name -> predicate "this transition belongs to it"
    'Fritz (dropped predictors), copy mode': lambda t: t.trick == 'Fritz' and uses_copy(t),
    'Fritz_kept, copy mode': lambda t: t.trick == 'Fritz_kept' and uses_copy(t),
    'Fritz_entropic, kept predictors': lambda t: t.trick == 'Fritz_entropic' and predictor_mode_of(t) == 'split',
    'Fritz_entropic, dropped predictors': lambda t: t.trick == 'Fritz_entropic' and predictor_mode_of(t) == 'drop',
    'Fritz_entropic certified by relabel': lambda t: certificate_of(t) == 'relabel',
    'Fritz_entropic certified by markov': lambda t: certificate_of(t) == 'markov',
    'any copy-mode step': lambda t: is_fritz_type(t) and uses_copy(t),
    'all Fritz_kept steps': lambda t: t.trick == 'Fritz_kept',
    'all kept-predictor steps (Fritz_kept and Fritz_entropic with kept predictors)': lambda t: predictor_mode_of(t) == 'split',
    'all Fritz (dropped predictors) steps': lambda t: t.trick == 'Fritz',
    'all Fritz_entropic steps': lambda t: t.trick == 'Fritz_entropic',
    'all Fritz-type steps': is_fritz_type,
}

# The cascade, cheapest first: (stage name, predicate "this transition belongs to this rung or an earlier one").
# `default_stages` runs exactly these stages, so the cumulative stage counts of a report and `ladder` agree.
CASCADE: List[Tuple[str, Callable[[Transition], bool]]] = [
    ('elementary', lambda t: not is_fritz_type(t)),
    ('Fritz, dropped predictors, replace mode', lambda t: not is_fritz_type(t) or (t.trick == 'Fritz' and not uses_copy(t))),
    ('Fritz, dropped predictors, copy mode', lambda t: not is_fritz_type(t) or t.trick == 'Fritz'),
    ('Fritz_kept, replace mode', lambda t: t.trick not in ('Fritz_kept', 'Fritz_entropic') or (t.trick == 'Fritz_kept' and not uses_copy(t))),
    ('Fritz_kept, copy mode', lambda t: t.trick != 'Fritz_entropic'),
    ('Fritz_entropic, dropped predictors, replace mode', lambda t: t.trick != 'Fritz_entropic' or (predictor_mode_of(t) == 'drop' and not uses_copy(t))),
    ('Fritz_entropic, dropped predictors, copy mode', lambda t: t.trick != 'Fritz_entropic' or predictor_mode_of(t) == 'drop'),
    ('Fritz_entropic, kept predictors, replace mode', lambda t: t.trick != 'Fritz_entropic' or predictor_mode_of(t) == 'drop' or not uses_copy(t)),
    ('Fritz_entropic, kept predictors, copy mode', lambda t: True),
]
LADDER: List[Tuple[str, Callable[[Transition], bool]]] = [(('' if i == 0 else '+ ') + name, keep) for i, (name, keep) in enumerate(CASCADE)]


def _reachable_if(explorer: ClosureExplorer, root: UnlabelledId, keep: Callable[[Transition], bool]) -> Set[UnlabelledId]:
    seen = {root}
    frontier = deque([root])
    while frontier:
        current = frontier.popleft()
        for t in explorer.edges.get(current, ()):
            if keep(t) and t.target not in seen:
                seen.add(t.target)
                frontier.append(t.target)
    return seen


def _fixpoint_if(report: GapReport, keep: Callable[[Transition], bool]) -> Set[UnlabelledId]:
    goals = {g.unique_unlabelled_id for g in report.seeds.values()} | set(report.known.values())
    input_ids = report.input_ids
    reach = {gid: _reachable_if(report.explorer, gid, keep) for gid in input_ids}
    proven = set(goals)
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


def fritz_breakdown(report: GapReport) -> Dict[str, int]:
    """For each category of Fritz-type step, the number of inputs no longer proven when every transition of that
    category is removed (all other transitions kept)."""
    everything = len(_fixpoint_if(report, lambda t: True))
    return {name: everything - len(_fixpoint_if(report, lambda t, pred=pred: not pred(t))) for name, pred in STEP_CATEGORIES.items()}


def ladder(report: GapReport) -> List[Tuple[str, int, int]]:
    """Cumulative counts of proven inputs as more expensive categories of step are allowed: (rung, proven, new)."""
    rows = []
    previous = 0
    for name, keep in LADDER:
        proven = len(_fixpoint_if(report, keep))
        rows.append((name, proven, proven - previous))
        previous = proven
    return rows


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
