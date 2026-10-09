"""
Breadth-first search over causal structures connected by piggyback tricks.

A *piggyback* maps a QmDAG G to a QmDAG G' such that a quantum-classical (QC) gap in G' implies a QC gap in G. Every
transition is recorded with its provenance (which trick, with which parameters, from which structure), so that each
discovered gap comes with a human-readable certificate, and so that reachability restricted to any subset of tricks is
a cheap graph query over the recorded transitions. Structures are identified up to relabelling by their unlabelled id;
each id is expanded exactly once (all tricks are label-equivariant).
"""
from __future__ import annotations

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
    'node_stitching': 1,               # formerly 'interruption'
    'conditioning': 3,                 # 1 visible grandparents; 2 latent grandparents; 3 guessing parents' children
    'naive_marginalization': 1,
    'teleportation_marginalization': 1,
    'degradation': 1,                  # quantum source to classical source (lookup only)
    'Fritz': 6,                        # 1 original; 2 common/others; 3 predictors removed soundly; 4 unified trick:
                                       #   d-separation first, LP (relabel targets) on failure, both predictor modes;
                                       #   5 one deletion per (target, predictor set), no noise-only deletions,
                                       #   redundant sub-facets cleaned; 6 predictor-first with joint target sets
                                       #   (d-separation) and uniform params
}


@dataclass(frozen=True)
class Transition:
    trick: str
    params: Tuple
    source: UnlabelledId
    target: UnlabelledId


# --------------------------------------------------------------------------------------------------
# The trick registry: name -> (QmDAG -> iterable of (params, child QmDAG)). Every piggyback is implemented as a
# method of QmDAG (its "PIGGYBACKS AS THE SEARCH APPLIES THEM" section); this module only names them, groups them
# into stages and searches. Children may be any size; the explorer applies the visible-node bounds.
# --------------------------------------------------------------------------------------------------

Trick = Callable[[QmDAG], Iterable[Tuple[Tuple, QmDAG]]]


def degradation(g: QmDAG) -> Iterable[Tuple[Tuple, QmDAG]]:
    """Quantum source to classical source: a gap in any degradation of g is a gap in g. A *lookup* trick: the
    explorer registers the degradations of every structure it meets and never expands them (see ClosureExplorer)."""
    return g.degradation_steps()


def elementary_tricks(max_visible: int = 5, districts_check: bool = False,
                      strict_conditioning: bool = True) -> Dict[str, Trick]:
    """The node-count-reducing piggybacks: point distribution, node stitching, conditioning, marginalization."""
    return {
        'PD': lambda g: g.pd_steps(),
        'node_stitching': lambda g: g.node_stitching_steps(),
        'conditioning': lambda g: g.conditioning_steps(strict_latents=strict_conditioning),
        'naive_marginalization': lambda g: g.marginalization_steps(apply_teleportation=False, districts_check=districts_check),
        'teleportation_marginalization': lambda g: g.marginalization_steps(apply_teleportation=True, districts_check=districts_check),
    }


def fritz_tricks(max_visible: int = 5, predictor_mode: str = 'dropped', modes: Tuple[str, ...] = ('replace', 'copy'),
                 use_lp: bool = True, pool: str = 'siblings', allow_descendants: bool = False,
                 max_predictors: int = 1, max_targets: Optional[int] = None, lp_markov_target: bool = False) -> Dict[str, Trick]:
    """The Fritz trick (name 'Fritz'): QmDAG.fritz_steps once per predicted-node mode in `modes`. predictor_mode
    is 'dropped' or 'kept'; `pool` ('siblings' or 'siblings+parents') and `allow_descendants` widen the predictor
    pool (manuscript 8.4); `max_predictors` allows joint predictor sets; `max_targets` bounds the joint target sets of
    one predictor (all by default; 1 disables them); lp_markov_target=True also tries the
    `markov` LP target set after `relabel` fails (it never decided an input in the four-node census, 9.7)."""
    if use_lp:
        try:
            import mosek  # noqa: F401
        except ImportError:
            warnings.warn("mosek is not installed; the Fritz trick certifies by d-separation only.")
            use_lp = False

    def fritz(g: QmDAG) -> Iterable[Tuple[Tuple, QmDAG]]:
        for mode in modes:
            yield from g.fritz_steps(mode=mode, predictor_mode=predictor_mode, use_lp=use_lp, pool=pool,
                                     allow_descendants=allow_descendants, max_predictors=max_predictors,
                                     max_targets=max_targets, lp_markov_target=lp_markov_target, max_visible=max_visible)
    return {'Fritz': fritz}


def default_tricks(max_visible: int = 5, max_predictors: int = 1, districts_check: bool = False,
                   predictor_mode: str = 'dropped', strict_conditioning: bool = True) -> Dict[str, Trick]:
    """Elementary tricks plus the Fritz trick with dropped predictors and the d-separation certificate only (no
    LP): the tricks of a single exhaustive closure."""
    return {**elementary_tricks(max_visible, districts_check, strict_conditioning),
            **fritz_tricks(max_visible, predictor_mode=predictor_mode, max_predictors=max_predictors, use_lp=False)}


Stage = Tuple[str, Dict[str, Callable], bool, Optional[FrozenSet[str]]]   # (name, tricks, roots_only, follow-up tricks)


def default_stages(max_visible: int = 5, with_entropic: bool = True, with_kept: bool = True,
                   max_predictors: int = 1, districts_check: bool = False, strict_conditioning: bool = True,
                   pool: str = 'siblings', allow_descendants: bool = False, max_targets: Optional[int] = None,
                   lp_markov_target: bool = False) -> List[Stage]:
    """A cascade of stages, cheapest first; each runs only on the inputs the earlier ones left unproven, and every
    structure proven in a stage is a known gap for the next.
    (1) The elementary reductions, closed over everything reachable from every input.
    Then four Fritz stages (the rungs of CASCADE), each applied once to each still-unproven input, its outputs
    reduced with the elementary tricks only ("depth one"): replace mode with dropped predictors, replace mode with
    kept predictors, copy mode with dropped predictors, copy mode with kept predictors. Within a stage every
    candidate is certified by d-separation first and by the LP only where d-separation fails, and a root stops as
    soon as one of its outputs reaches a known gap; with_entropic=False disables the LP. The LP tries the
    `relabel` target set only (lp_markov_target=True turns the `markov` set back on; it never decided an input in
    the four-node census). The predictor pool is the latent siblings of the target that are not its descendants;
    pool='siblings+parents' adds the visible parents and allow_descendants=True keeps the descendants (the
    experiments of manuscript 1.2). Joint predictor sets are available (max_predictors) but off."""
    common = dict(max_visible=max_visible, use_lp=with_entropic, pool=pool, allow_descendants=allow_descendants,
                  max_predictors=max_predictors, max_targets=max_targets, lp_markov_target=lp_markov_target)
    elementary = elementary_tricks(max_visible, districts_check, strict_conditioning)
    reductions = frozenset(elementary)
    names = [name for name, _ in CASCADE]
    stages: List[Stage] = [(names[0], elementary, False, None)]
    plan = [(names[1], 'replace', 'dropped'), (names[2], 'replace', 'kept'),
            (names[3], 'copy', 'dropped'), (names[4], 'copy', 'kept')]
    for name, mode, predictor_mode in plan:
        if predictor_mode == 'kept' and not with_kept:
            continue
        stages.append((name, fritz_tricks(predictor_mode=predictor_mode, modes=(mode,), **common), True, reductions))
    return stages


MARGINALIZATION_TRICKS = frozenset({'naive_marginalization', 'teleportation_marginalization'})

# For the report, "provable with T" uses the tricks in the group, and "provable only via T" removes just the trick
# itself from the full set. The Fritz trick is only meaningful together with marginalization (its outputs are
# typically larger than the original and need to be reduced), so its group includes the marginalizations.
FRITZ_TRICKS_ALL = frozenset({'Fritz'})
LOOKUP_TRICKS = frozenset({'degradation'})   # always allowed: part of what "known" means
# "Provable with T alone" and "provable only via T" are reported for the elementary reductions only; the expensive
# trick is assessed by the cumulative ladder. The degradation lookup is part of every group.
TRICK_GROUPS_FOR_REPORT: Dict[str, Tuple[FrozenSet[str], FrozenSet[str]]] = {
    name: (frozenset(group) | LOOKUP_TRICKS, frozenset(removed)) for name, (group, removed) in {
        'PD': ({'PD'}, {'PD'}),
        'node_stitching': ({'node_stitching'}, {'node_stitching'}),
        'conditioning': ({'conditioning'}, {'conditioning'}),
        'naive_marginalization': ({'naive_marginalization'}, {'naive_marginalization'}),
        'teleportation_marginalization': ({'teleportation_marginalization'}, {'teleportation_marginalization'}),
        'marginalization (either kind)': (MARGINALIZATION_TRICKS, MARGINALIZATION_TRICKS),
    }.items()}


# --------------------------------------------------------------------------------------------------
# The explorer
# --------------------------------------------------------------------------------------------------

class ClosureExplorer:
    """Expands structures under `tricks`. `lookup_tricks` (by default the degradation piggyback) are applied once
    to every structure registered, their children are recorded as transitions but never expanded: they only serve
    to recognise a known gap (a structure is known as soon as one of its lookups is), so their cost is one id
    computation per child and no search."""
    def __init__(self, tricks: Dict[str, Callable], max_visible: int = 5, min_visible: int = 3,
                 max_states: int = 200000, lookup_tricks: Optional[Dict[str, Callable]] = None) -> None:
        self.tricks = dict(tricks)
        self.lookup_tricks: Dict[str, Callable] = {'degradation': degradation} if lookup_tricks is None else dict(lookup_tricks)
        self.lookup_only: Set[UnlabelledId] = set()   # registered through a lookup trick only: never expanded
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
        """Registers a structure reached by the search (or given as a root) and records its lookups."""
        gid = g.unique_unlabelled_id
        if gid in self.lookup_only:
            self.lookup_only.discard(gid)   # promoted: reached by a real trick, so it gets its own lookups
        elif gid in self.representatives:
            return gid
        self.representatives.setdefault(gid, g)
        self._lookup(gid)
        return gid

    def _lookup(self, gid: UnlabelledId) -> None:
        g = self.representatives[gid]
        transitions = []
        for name, trick in self.lookup_tricks.items():
            for params, child in trick(g):
                child_id = child.unique_unlabelled_id
                if child_id not in self.representatives:
                    self.representatives[child_id] = child
                    self.lookup_only.add(child_id)
                transitions.append(Transition(name, params, gid, child_id))
        self._record(gid, transitions)

    def _expand_one(self, gid: UnlabelledId, only: Optional[FrozenSet[str]] = None) -> List[Transition]:
        """Applies every trick (or every trick in `only`) not yet applied to this id; returns its transitions."""
        pending = [name for name in self.tricks if name not in self.applied.get(gid, set())
                   and (only is None or name in only)]
        if not pending or gid in self.lookup_only:
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
               stage: Optional[str] = None, followup: Optional[FrozenSet[str]] = None,
               stop_when_known: Optional[Set[UnlabelledId]] = None) -> None:
        """Applies extra tricks. With roots_only, they are applied to the roots alone and the new children are
        expanded with the tricks already in the explorer, or only with those in `followup`; otherwise the extra
        tricks join the trick set and are applied to everything reachable from the roots. `stage` names the stage.
        With `stop_when_known` (roots_only), each trick's generator is consumed lazily: every child is recorded and
        followed up as soon as it is emitted, and the root stops as soon as the child or its follow-up meets the
        known gaps (a root proven by a cheap step never pays for the expensive candidates that come later in the
        generator). Every recorded transition is a sound step, so the partial record is sound; the cumulative
        counts and the ladder do not depend on it (an unproven root still records everything)."""
        self.begin_stage(stage or '+'.join(sorted(extra_tricks)), extra_tricks)
        if roots_only:
            for root in roots:
                self.expand(root)   # no-op when already expanded; closes the root under the base tricks
                gid = self.register(root)
                representative = self.representatives[gid]   # params are stated in the representative's labels
                if stop_when_known is not None:
                    self._extend_root_lazily(gid, representative, extra_tricks, followup, stop_when_known)
                    continue
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

    def _extend_root_lazily(self, gid: UnlabelledId, representative: QmDAG, extra_tricks: Dict[str, Callable],
                            followup: Optional[FrozenSet[str]], goals: Set[UnlabelledId]) -> None:
        """The roots-only application of `extend` with early exit (see there). Degradation lookups are recorded
        transitions, so a child whose degradation is known counts as known through its recorded follow-up."""
        for trick_name, trick in extra_tricks.items():
            applied_key = f"{self.stage_names[self.current_stage]}:{trick_name}"
            if applied_key in self.applied.get(gid, set()):
                continue
            for params, child in trick(representative):
                if not (self.min_visible <= child.number_of_visible <= self.max_visible):
                    continue
                child_id = self.register(child)
                self._record(gid, [Transition(trick_name, params, gid, child_id)])
                self.expand(self.representatives[child_id], only=followup)
                if not goals.isdisjoint(self.reachable(child_id)):   # the child's lookups and follow-up included
                    return   # proven: the trick is left unfinished for this root (not marked as applied)
            self.applied.setdefault(gid, set()).add(applied_key)

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
        piggybacks from it to a seed; nothing is inferred from a structure to a weaker variant of it. Structures
        registered only as lookup targets (degradations) are not counted: a non-seed among them never reaches a
        seed, and a seed among them is a seed."""
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
        return (known & set(self.explorer.representatives)) - self.explorer.lookup_only


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
               known: Optional[Dict[str, UnlabelledId]] = None, early_exit: bool = True) -> GapReport:
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
                           followup=followup, early_exit=early_exit)
    return report


def add_stage(report: GapReport, extra_tricks: Dict[str, Callable],
              trick_groups: Optional[Dict[str, Tuple[FrozenSet[str], FrozenSet[str]]]] = None,
              verbose: bool = True, roots_only: bool = True, name: Optional[str] = None,
              followup: Optional[FrozenSet[str]] = None, seeds: Optional[Dict[str, QmDAG]] = None,
              inputs: Optional[List[QmDAG]] = None, known: Optional[Dict[str, UnlabelledId]] = None,
              early_exit: bool = True) -> GapReport:
    """Applies further tricks to the explorer of `report` and rebuilds the report. With roots_only the tricks are
    applied to the still-unproven inputs only, and their children are expanded with the tricks already present
    (or only with those in `followup`); otherwise to everything reachable from every input. With early_exit (and
    roots_only) a root stops as soon as one of its children, or that child's follow-up, is a known gap."""
    explorer = report.explorer
    name = name or '+'.join(sorted(extra_tricks))
    seeds = report.seeds if seeds is None else seeds
    inputs = report.inputs if inputs is None else list(inputs)
    known = report.known if known is None else known
    if seeds is not report.seeds or inputs is not report.inputs or known is not report.known:
        report = build_report(explorer, inputs, seeds, _trick_groups_for(explorer, trick_groups), known=known)
    roots = list(dict.fromkeys(report.remaining)) if roots_only else list({g.unique_unlabelled_id: g for g in inputs}.values())
    explorer.begin_stage(name, extra_tricks)   # the stage is counted even when nothing is left to expand
    # Early exit: a root stops as soon as it reaches a seed, a cached known gap or an input proven earlier (an input
    # proven in this very stage is not yet a goal; the next build_report chains the certificates anyway).
    goals = {g.unique_unlabelled_id for g in seeds.values()} | set(known.values()) | set(report.proven) if early_exit else None
    for i, g in enumerate(roots):
        if verbose and i % 250 == 0:
            print(f"[{name}] expanding {i} of {len(roots)} roots; {len(explorer.edges)} structures")
        explorer.extend(extra_tricks, [g], roots_only=roots_only, stage=name, followup=followup, stop_when_known=goals)
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
# Assessing the expensive (Fritz-type) steps: the cumulative ladder
# --------------------------------------------------------------------------------------------------

def mode_of(t: Transition) -> Optional[str]:
    """'replace' or 'copy' (the predicted-node mode) of a Fritz transition, None otherwise."""
    return dict(t.params).get('mode') if t.trick == 'Fritz' else None


def predictor_mode_of(t: Transition) -> Optional[str]:
    """'dropped' or 'kept' for a Fritz transition, None otherwise."""
    return dict(t.params).get('predictor_mode') if t.trick == 'Fritz' else None


def certificate_of(t: Transition) -> Optional[str]:
    """'dsep' or 'entropic' for a Fritz transition, None otherwise."""
    return dict(t.params).get('certificate') if t.trick == 'Fritz' else None


def uses_copy(t: Transition) -> bool:
    return mode_of(t) == 'copy'


def is_fritz_type(t: Transition) -> bool:
    return t.trick in FRITZ_TRICKS_ALL


def is_kept(t: Transition) -> bool:
    return predictor_mode_of(t) == 'kept'


def is_lp(t: Transition) -> bool:
    return certificate_of(t) == 'entropic'


# The cascade, cheapest first: (stage name, predicate "this transition belongs to this rung or an earlier one").
# `default_stages` runs exactly these stages. Each Fritz stage certifies by d-separation first and by the LP only
# where that fails, so LADDER splits every stage into a d-separation rung and an LP rung: the LP rung of a stage
# coincides with the stage count.
def _rung(mode: str, kept: bool, lp: bool) -> Callable[[Transition], bool]:
    order = [('replace', False), ('replace', True), ('copy', False), ('copy', True)]
    rank = order.index((mode, kept))
    def keep(t: Transition) -> bool:
        if not is_fritz_type(t):
            return True
        own = order.index(('copy' if uses_copy(t) else 'replace', is_kept(t)))
        return own < rank or (own == rank and (lp or not is_lp(t)))
    return keep


CASCADE: List[Tuple[str, Callable[[Transition], bool]]] = [
    ('elementary', lambda t: not is_fritz_type(t)),
    ('Fritz, replace mode, dropped predictors', _rung('replace', False, True)),
    ('Fritz, replace mode, kept predictors', _rung('replace', True, True)),
    ('Fritz, copy mode, dropped predictors', _rung('copy', False, True)),
    ('Fritz, copy mode, kept predictors', _rung('copy', True, True)),
]
LADDER: List[Tuple[str, Callable[[Transition], bool]]] = [('elementary', CASCADE[0][1])]
for _name, _mode, _kept in [('replace mode, dropped predictors', 'replace', False), ('replace mode, kept predictors', 'replace', True),
                            ('copy mode, dropped predictors', 'copy', False), ('copy mode, kept predictors', 'copy', True)]:
    LADDER.append((f'+ Fritz, {_name}, d-separation', _rung(_mode, _kept, False)))
    LADDER.append((f'+ Fritz, {_name}, LP', _rung(_mode, _kept, True)))


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
