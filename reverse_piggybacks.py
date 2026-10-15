"""
The piggybacks in reverse (manuscript Appendix A).

A piggyback maps a structure G to a structure G' such that a QC gap of G' implies a QC gap of G. Run forward, the
search starts from a candidate G and maps it down to a known gap. Run in reverse, a known gap G' generates gaps:
every G in the inverse image of a piggyback is a gap, with the forward step as its certificate. This module holds
one inverse generator per forward trick and the closure of the seeds under them (`ReverseClosure`), capped at a
number of visible nodes.

Soundness is by construction: every candidate G that an inverse generator proposes is confirmed by running the
forward primitive of `quantum_mDAG.QmDAG` on it (`forward_child`) and checking that the child has the class key
of G'. The generators may over-generate; the forward code stays the single source of truth for what a piggyback
does. A confirmed candidate is recorded as a forward `qc_gap_search.Transition` (trick, forward params in G's
labels, source = key(G), target = key(G'), via = G, produced = the forward child), so that the search's reports,
certificates and cache read the reverse closure exactly as they read forward transitions.

The new node is always appended as index n' = G'.number_of_visible: the forward reduction removes it, so for an
exact inverse the forward child is G' itself in G''s own labels.
"""
from __future__ import annotations

import itertools
import time
from collections import Counter, deque
from typing import Callable, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

from directed_structures import DirectedStructure
from hypergraphs import Hypergraph, hypergraph_full_cleanup
from qc_gap_search import Transition, UnlabelledId, group_seeds_by_key
from quantum_mDAG import QmDAG

Candidate = Tuple[str, Tuple, QmDAG]      # (forward trick name, forward params in the candidate's labels, candidate)
LOOKUP_INVERSES = ('degradation', 'upward')


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------

def edges_of(g: QmDAG) -> Set[Tuple[int, int]]:
    return set(map(tuple, g.directed_structure_instance.edge_list))


def c_facets_of(g: QmDAG) -> Set[FrozenSet[int]]:
    return set(g.C_simplicial_complex_instance.simplicial_complex_as_sets)


def q_facets_of(g: QmDAG) -> Set[FrozenSet[int]]:
    return set(g.Q_simplicial_complex_instance.simplicial_complex_as_sets)


def children_of(g: QmDAG, v: int) -> Set[int]:
    return {b for a, b in edges_of(g) if a == v}


def parents_of(g: QmDAG, v: int) -> Set[int]:
    return {a for a, b in edges_of(g) if b == v}


def descendants_plus(edges: Iterable[Tuple[int, int]], v: int) -> Set[int]:
    """v and everything reachable from it along the edges."""
    out: Dict[int, Set[int]] = {}
    for a, b in edges:
        out.setdefault(a, set()).add(b)
    seen = {v}
    stack = [v]
    while stack:
        u = stack.pop()
        for w in out.get(u, ()):
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return seen


def is_acyclic(n: int, edges: Iterable[Tuple[int, int]]) -> bool:
    edges = list(edges)
    return all(v not in (descendants_plus(edges, v) - {v}) or False for v in range(n)) and \
        not any(a in descendants_plus(edges, b) for a, b in edges)


def emit(trick: str, params: Tuple, n: int, edges, C, Q, hybrids: bool) -> Iterable[Candidate]:
    """One candidate, unless it has classical facets and hybrids are off."""
    g = assemble(n, edges, C, Q)
    if hybrids or g.is_all_quantum:
        yield trick, params, g


def assemble(n: int, edges: Iterable[Tuple[int, int]], C: Iterable[FrozenSet[int]], Q: Iterable[FrozenSet[int]]) -> QmDAG:
    """A QmDAG from plain parts; facets are cleaned (non-maximal faces dropped) before construction."""
    return QmDAG(DirectedStructure(sorted(set(map(tuple, edges))), n),
                 Hypergraph(hypergraph_full_cleanup(set(map(frozenset, C))), n),
                 Hypergraph(hypergraph_full_cleanup(set(map(frozenset, Q))), n))


def faces(facets: Iterable[FrozenSet[int]], min_size: int = 1) -> Set[FrozenSet[int]]:
    """Every subset of at least min_size nodes of some facet."""
    out: Set[FrozenSet[int]] = set()
    for f in facets:
        for r in range(min_size, len(f) + 1):
            out.update(map(frozenset, itertools.combinations(sorted(f), r)))
    return out


def antichains(options: Sequence[FrozenSet[int]], allow_empty: bool = True) -> Iterable[FrozenSet[FrozenSet[int]]]:
    """Every set of pairwise non-nested members of `options` (the facet sets a complex can have)."""
    options = sorted(set(options), key=lambda s: (len(s), sorted(s)))

    def rec(i: int, chosen: List[FrozenSet[int]]):
        if i == len(options):
            if chosen or allow_empty:
                yield frozenset(chosen)
            return
        yield from rec(i + 1, chosen)
        s = options[i]
        if all(not (s <= t or t <= s) for t in chosen):
            chosen.append(s)
            yield from rec(i + 1, chosen)
            chosen.pop()
    return rec(0, [])


def permutation_to(g: QmDAG, other: QmDAG) -> Optional[Tuple[int, ...]]:
    """The relabelling p with g relabelled by p equal to other (new index j holds g's node p[j]), or None.
    The three parts' integer permutations are listed in itertools.permutations order, so one scan suffices."""
    if g.number_of_visible != other.number_of_visible:
        return None
    target = other.unique_id[1:]
    perms = itertools.permutations(range(g.number_of_visible))
    for perm, d, c, q in zip(perms, g.directed_structure_instance.as_integer_permutations,
                             g.C_simplicial_complex_instance.as_integer_permutations,
                             g.Q_simplicial_complex_instance.as_integer_permutations):
        if (d, c, q) == target:
            return tuple(perm)
    return None


def translate_params(trick: str, params: Tuple, perm: Tuple[int, ...]) -> Tuple:
    """Forward params stated in g's labels restated for g relabelled by perm (new index j holds old node perm[j]):
    every node v becomes the j with perm[j] == v."""
    new_of = {v: j for j, v in enumerate(perm)}

    def node(v):
        return new_of[int(v)]

    if trick in ('PD', 'naive_marginalization', 'teleportation_marginalization', 'conditioning'):
        (name, v), = params
        return ((name, node(v)),)
    if trick == 'node_stitching':
        ((sink_key, sink), (source_key, source)), = params
        return (((sink_key, node(sink)), (source_key, node(source))),)
    if trick == 'degradation':
        (name, facets), = params
        return ((name, tuple(sorted(tuple(sorted(node(v) for v in f)) for f in facets))),)
    raise ValueError(f"no param translation for {trick}")


# --------------------------------------------------------------------------------------------------
# The forward primitives, by trick name (what a candidate is confirmed with)
# --------------------------------------------------------------------------------------------------

def forward_child(trick: str, params: Tuple, g: QmDAG) -> Optional[QmDAG]:
    """The structure the forward trick with these params produces from g, or None when the trick does not apply."""
    p = dict(params)
    if trick == 'PD':
        return g.fix_to_point_distribution_QmDAG(p['drop'])
    if trick == 'node_stitching':
        (pairs,) = params
        d = dict(pairs)
        sink, source = d['sink'], d['source']
        if sink not in g.vis_nodes_with_no_children or source not in g.exogenous_visible_nodes:
            return None
        if sink in g.directed_structure_instance.adjMat.descendantsplus_of(source):
            return None
        return g.node_stitching(sink, source)
    if trick in ('naive_marginalization', 'teleportation_marginalization'):
        return g.marginalize(p['marginalize'], apply_teleportation=(trick == 'teleportation_marginalization'))
    if trick == 'conditioning':
        v = p['condition']
        return g.condition(v) if g.conditioning_is_justified(v) else None
    if trick == 'degradation':
        chosen = set(map(frozenset, p['classical']))
        Q = q_facets_of(g)
        if not chosen <= Q or chosen == Q:
            return None
        n = g.number_of_visible
        return QmDAG(g.directed_structure_instance,
                     Hypergraph(hypergraph_full_cleanup(c_facets_of(g) | chosen), n),
                     Hypergraph(Q - chosen, n))
    raise ValueError(f"unknown trick {trick}")


# --------------------------------------------------------------------------------------------------
# The inverses
# --------------------------------------------------------------------------------------------------

def inverse_node_stitching(c: QmDAG, hybrids: bool = True) -> Iterable[Candidate]:
    """Interruption (A.5): the forward search stitches an exogenous x onto a childless sink y, so in G' the
    children of y are exactly the children x had. Inverse: for each node y, give all of y's children to a new
    exogenous node x = n' (no parents, no facets); y becomes childless."""
    n = c.number_of_visible
    x = n
    E = edges_of(c)
    for y in range(n):
        new_edges = {(x if a == y else a, b) for a, b in E}
        yield from emit('node_stitching', ((('sink', y), ('source', x)),), n + 1, new_edges, c_facets_of(c), q_facets_of(c), hybrids)


def inverse_degradation(c: QmDAG) -> Iterable[Candidate]:
    """Quantumification (A.1): a nonempty antichain of faces of the classical facets becomes quantum (the
    constructor absorbs the classical facets they dominate). A face strictly containing an existing quantum facet
    would absorb it, and the degradation could not give G' back, so such faces are not options; neither are faces
    inside an existing quantum facet (already quantum)."""
    Q = q_facets_of(c)
    options = [f for f in faces(c_facets_of(c), 2)
               if not any(f <= q for q in Q) and not any(q < f for q in Q)]
    for chosen in antichains(options, allow_empty=False):
        new_Q = hypergraph_full_cleanup(Q | set(chosen))
        params = (('classical', tuple(sorted(tuple(sorted(f)) for f in chosen))),)
        yield 'degradation', params, assemble(c.number_of_visible, edges_of(c), c_facets_of(c), new_Q)


def _bases(options: Iterable[FrozenSet[int]], exclude: Iterable[FrozenSet[int]] = ()) -> Iterable[FrozenSet[FrozenSet[int]]]:
    """Antichains of candidate bases (the facets of the new node minus the node), as sorted tuples of options."""
    options = set(options) - set(exclude)
    return antichains(sorted(options, key=lambda f: (len(f), sorted(f))))


def _edge_patterns(c: QmDAG) -> Iterable[Tuple[FrozenSet[int], FrozenSet[int]]]:
    """(parents, children) of a node inserted into c, every assignment of each node to parent, child or neither
    that keeps the structure acyclic (no parent is a descendant of a child)."""
    n = c.number_of_visible
    E = edges_of(c)
    desc = {v: descendants_plus(E, v) for v in range(n)}
    for roles in itertools.product((0, 1, 2), repeat=n):
        parents = frozenset(v for v, r in enumerate(roles) if r == 1)
        children = frozenset(v for v, r in enumerate(roles) if r == 2)
        if any(p in desc[ch] for p in parents for ch in children):
            continue
        yield parents, children


def inverse_PD(c: QmDAG, hybrids: bool = True) -> Iterable[Candidate]:
    """Node insertion (A.2). Point distribution removes x with its edges and its facet memberships, so G is G' with
    a node x = n' added: any parents and children (acyclic), and facets {x} u S whose base S is a face of a
    same-colour facet of G' or a single node (after the removal S is absorbed, or vanishes as a singleton); a
    classical base inside a quantum face is allowed too (the removal leaves a classical facet dominated by a
    quantum one, which the constructor drops). A facet of G' that x joins is replaced by the enlarged facet. Exact:
    removing x gives G' back for every candidate."""
    n = c.number_of_visible
    x = n
    C, Q = c_facets_of(c), q_facets_of(c)
    E = edges_of(c)
    singletons = {frozenset([v]) for v in range(n)}
    q_options = faces(Q, 1) | singletons
    c_options = (faces(C, 1) | faces(Q, 1) | singletons) if hybrids else set()
    facet_choices = []
    for q_bases in _bases(q_options):
        new_Q = Q | {S | {x} for S in q_bases}
        for c_bases in _bases(c_options):
            if any(any(S <= T for T in q_bases) for S in c_bases):
                continue                                       # dominated by a quantum facet of x: a duplicate
            new_C = C | {S | {x} for S in c_bases}
            facet_choices.append((new_C, new_Q))
    params = (('drop', x),)
    for parents, children in _edge_patterns(c):
        new_edges = E | {(p, x) for p in parents} | {(x, ch) for ch in children}
        for new_C, new_Q in facet_choices:
            yield from emit('PD', params, n + 1, new_edges, new_C, new_Q, hybrids)


def _subsets(items: Iterable[int]) -> Iterable[FrozenSet[int]]:
    items = sorted(items)
    for r in range(len(items) + 1):
        for comb in itertools.combinations(items, r):
            yield frozenset(comb)


def _inside(S: FrozenSet[int], facets: Iterable[FrozenSet[int]]) -> bool:
    return any(S <= F for F in facets)


def inverse_marginalization(c: QmDAG, apply_teleportation: bool, hybrids: bool = True) -> Iterable[Candidate]:
    """Un-marginalization (A.3). Marginalizing x relays its visible parents to its visible children Ch, turns every
    facet F containing x (classical or quantum) into the classical facet (F minus x) u Ch, adds the classical facet
    Ch when x is in no facet, keeps the quantum facets F minus x (naive) or enlarges them to (F minus x) u T with T the
    children sharing a quantum facet with x (teleportation). Inverse: choose the children Ch, the parents Pa among
    the common parents of Ch in G' (the relay edges Pa x Ch may then be absent in G), the bases S of x's quantum
    and classical facets, each of whose relays (S u Ch, and S u T for a quantum base) must be a face of G' or a
    single node, and any facets of G inside the relayed facets (absorbed in G'). Over-generates; every candidate
    is confirmed by the forward marginalization."""
    n = c.number_of_visible
    x = n
    C, Q = c_facets_of(c), q_facets_of(c)
    E = edges_of(c)
    faces_C, faces_Q = faces(C, 1), faces(Q, 1)
    all_faces = faces_C | faces_Q
    nodes = range(n)
    trick = 'teleportation_marginalization' if apply_teleportation else 'naive_marginalization'
    params = (('marginalize', x),)

    def face_or_point(S: FrozenSet[int], where: Set[FrozenSet[int]]) -> bool:
        return len(S) <= 1 or S in where

    for Ch in _subsets(nodes):
        common_parents = set(nodes) - Ch
        for ch in Ch:
            common_parents &= parents_of(c, ch)
        relay_target_ok = len(Ch) <= 1 or Ch in all_faces
        # quantum bases: an antichain of nonempty S; the relay S u Ch must be a face of G' (or a point) and S itself
        # (naive) or S u T (teleportation) must be a quantum face (or a point)
        q_options = [S for S in _subsets(nodes) if S and face_or_point(S | Ch, all_faces)]
        c_options = [S for S in _subsets(nodes) if S and face_or_point(S | Ch, all_faces)] if hybrids else []
        for q_bases in _bases(q_options):
            T = frozenset().union(*q_bases) & Ch if (apply_teleportation and q_bases) else frozenset()
            if not all(face_or_point(S | T, faces_Q) for S in q_bases):
                continue
            for c_bases in _bases(c_options):
                if any(any(S <= Tq for Tq in q_bases) for S in c_bases):
                    continue
                if not q_bases and not c_bases and not relay_target_ok:
                    continue
                produced_C = {S | Ch for S in (set(q_bases) | set(c_bases))} if (q_bases or c_bases) else {Ch}
                produced_C = {F for F in produced_C if len(F) >= 2}
                produced_Q = {S | T for S in q_bases if len(S | T) >= 2}
                # facets of G not containing x: those of G' that are not produced, plus any inside produced ones
                base_Q = Q - produced_Q
                base_C = C - produced_C
                extra_q_options = sorted((faces(produced_Q, 2) - base_Q) if produced_Q else set(), key=lambda f: (len(f), sorted(f)))
                extra_c_options = sorted(((faces(produced_C | produced_Q, 2) - base_C) if hybrids else set()), key=lambda f: (len(f), sorted(f)))
                x_facets_Q = {S | {x} for S in q_bases}
                x_facets_C = {S | {x} for S in c_bases}
                for extra_Q in antichains(extra_q_options):
                    if any(_inside(F, base_Q) or any(G < F for G in base_Q) for F in extra_Q):
                        continue
                    new_Q = base_Q | set(extra_Q) | x_facets_Q
                    for extra_C in antichains(extra_c_options):
                        if any(_inside(F, base_C) or any(G < F for G in base_C) for F in extra_C):
                            continue
                        new_C = base_C | set(extra_C) | x_facets_C
                        for Pa in _subsets(common_parents):
                            relay = {(p, ch) for p in Pa for ch in Ch}
                            for removed in _subsets(sorted(relay)) if relay else [frozenset()]:
                                new_edges = (E - set(removed)) | {(p, x) for p in Pa} | {(x, ch) for ch in Ch}
                                yield from emit(trick, params, n + 1, new_edges, new_C, new_Q, hybrids)


def inverse_naive_marginalization(c: QmDAG, hybrids: bool = True) -> Iterable[Candidate]:
    return inverse_marginalization(c, False, hybrids)


def inverse_teleportation_marginalization(c: QmDAG, hybrids: bool = True) -> Iterable[Candidate]:
    return inverse_marginalization(c, True, hybrids)


def inverse_conditioning(c: QmDAG, hybrids: bool = True) -> Iterable[Candidate]:
    """Un-swapping (A.4). Conditioning on x (where justified: every visible grandparent of x is a parent, and every
    facet meeting a parent of x contains x or lies within the parents) removes x with its edges, adds the classical
    facet over the parents B and all latent siblings, and the quantum facet over the quantum siblings. Inverse:
    choose the parents B (closed under visible parents), the children Ch, the bases of x's quantum facets (their
    union must be a quantum face of G' or a point) and of its classical facets (B with every base must be a face of
    G' or a point), plus any facets of G absorbed by the two produced facets; no facet of G may meet B without
    lying inside B. Over-generates; every candidate is confirmed by the forward conditioning."""
    n = c.number_of_visible
    x = n
    C, Q = c_facets_of(c), q_facets_of(c)
    E = edges_of(c)
    all_facets = C | Q
    nodes = frozenset(range(n))
    desc = {v: descendants_plus(E, v) for v in nodes}
    params = (('condition', x),)

    def closed_under_parents(B: FrozenSet[int]) -> bool:
        return all(parents_of(c, b) <= B for b in B)

    def respects_B(F: FrozenSet[int], B: FrozenSet[int]) -> bool:
        return F.isdisjoint(B) or F <= B

    for B in _subsets(nodes):
        if not closed_under_parents(B):
            continue
        for Ch in _subsets(nodes - B):
            if any(b in desc[ch] for b in B for ch in Ch):
                continue
            new_edges = E | {(b, x) for b in B} | {(x, ch) for ch in Ch}
            # quantum bases: nonempty subsets of one quantum facet of G', or single nodes
            q_options = {S for F in Q for S in faces([F], 1)} | {frozenset([v]) for v in nodes}
            for q_bases in _bases(sorted(q_options, key=lambda f: (len(f), sorted(f)))):
                Q_all = frozenset().union(*q_bases) if q_bases else frozenset()
                if len(Q_all) >= 2 and Q_all not in faces(Q, 2):
                    continue
                produced_Q = {Q_all} if len(Q_all) >= 2 else set()
                core = B | Q_all
                # classical bases: with B and the quantum bases they must fit in one facet of G'
                hosts = [F for F in all_facets if core <= F] if hybrids else []
                c_options = {S for F in hosts for S in faces([F], 1)} if hybrids else set()
                for c_bases in _bases(sorted(c_options, key=lambda f: (len(f), sorted(f)))):
                    if any(any(S <= T for T in q_bases) for S in c_bases):
                        continue
                    BS = core.union(*c_bases) if c_bases else core
                    if len(BS) >= 2 and BS not in faces(all_facets, 2):
                        continue
                    produced_C = {BS} if len(BS) >= 2 else set()
                    base_Q = Q - produced_Q
                    base_C = C - produced_C
                    if not all(respects_B(F, B) for F in base_Q | base_C):
                        continue
                    extra_q_options = sorted((faces(produced_Q, 2) - base_Q) if produced_Q else set(), key=lambda f: (len(f), sorted(f)))
                    extra_c_options = sorted(((faces(produced_C | produced_Q, 2) - base_C) if hybrids else set()), key=lambda f: (len(f), sorted(f)))
                    x_facets_Q = {S | {x} for S in q_bases}
                    x_facets_C = {S | {x} for S in c_bases}
                    for extra_Q in antichains(extra_q_options):
                        if any(not respects_B(F, B) or _inside(F, base_Q) or any(G < F for G in base_Q) for F in extra_Q):
                            continue
                        new_Q = base_Q | set(extra_Q) | x_facets_Q
                        for extra_C in antichains(extra_c_options):
                            if any(not respects_B(F, B) or _inside(F, base_C) or any(G < F for G in base_C) for F in extra_C):
                                continue
                            new_C = base_C | set(extra_C) | x_facets_C
                            yield from emit('conditioning', params, n + 1, new_edges, new_C, new_Q, hybrids)


INVERSES: Dict[str, Callable[..., Iterable[Candidate]]] = {
    'node_stitching': inverse_node_stitching,
    'PD': inverse_PD,
    'conditioning': inverse_conditioning,
    'naive_marginalization': inverse_naive_marginalization,
    'teleportation_marginalization': inverse_teleportation_marginalization,
}


# --------------------------------------------------------------------------------------------------
# The closure
# --------------------------------------------------------------------------------------------------

class ReverseClosure:
    """The known gaps closed under the inverses of the reversed tricks (`inverses`, by forward trick name) and,
    with `lookups`, the two lookup inverses (quantumification and the reverse upward rule), capped at
    `max_visible` visible nodes. Keys are those of `index` (equivalence_classes.ClassIndex): every colouring of a
    known key is expanded (the class members and the hybrids that lift into the class), every confirmed candidate
    is recorded as a forward Transition, and a new key enters the frontier. `stats` counts, per trick, the
    candidates generated, confirmed, rejected by the forward check, duplicate labelled structures, relabelled
    duplicates and new keys."""

    def __init__(self, index, inverses: Dict[str, Callable[[QmDAG], Iterable[Candidate]]],
                 max_visible: int = 4, hybrids: bool = True, lookups: bool = True, verbose: bool = False) -> None:
        self.index = index
        self.inverses = dict(inverses)
        self.max_visible = max_visible
        self.hybrids = hybrids
        self.lookups = lookups
        self.verbose = verbose
        self.known: Dict[UnlabelledId, QmDAG] = {}            # key -> representative
        self.representatives = self.known
        self.structures: Dict[UnlabelledId, QmDAG] = {}       # every labelled structure met, by its own id
        self.edges: Dict[UnlabelledId, List[Transition]] = {}
        self._edge_sets: Dict[UnlabelledId, Set[Transition]] = {}
        self.stats: Dict[str, Counter] = {}
        self.seconds: Dict[str, float] = Counter()
        self.origin: Dict[UnlabelledId, str] = {}             # key -> 'seed' / 'extra' / the trick that found it
        self._seen: Set[Tuple] = set()

    # -- bookkeeping ------------------------------------------------------------------------------------------

    def _store(self, g: QmDAG) -> QmDAG:
        return self.structures.setdefault(g.unique_unlabelled_id, g)

    def _record(self, t: Transition) -> bool:
        known = self._edge_sets.setdefault(t.source, set())
        if t in known:
            return False
        known.add(t)
        self.edges.setdefault(t.source, []).append(t)
        return True

    def _add_known(self, key: UnlabelledId, g: QmDAG, origin: str) -> bool:
        if key in self.known:
            return False
        rep = self.index.representative(key)
        self.known[key] = self._store(rep if rep is not None else g)
        self.origin[key] = origin
        return True

    def colourings(self, key: UnlabelledId) -> List[QmDAG]:
        """The structures with this key: the class members and their degradations that lift back into the class
        (a forward child is often such a hybrid, and its parents are only reached from it); a non-class key has
        its stored structure alone. Without `hybrids`, the all-quantum members only."""
        members = self.index.members(key, fallback=self.known[key])
        out = {m.unique_unlabelled_id: self._store(m) for m in members}
        if self.hybrids and self.index.is_class_key(key):
            for m in members:
                for _, d in m.degradation_steps():
                    if self.index.key(d) == key:
                        out.setdefault(d.unique_unlabelled_id, self._store(d))
        return list(out.values())

    # -- one candidate ----------------------------------------------------------------------------------------

    def _accept(self, trick: str, params: Tuple, g: QmDAG, key: UnlabelledId) -> Optional[UnlabelledId]:
        """Confirms a candidate with the forward trick, records the transition and returns the candidate's key
        (None when rejected)."""
        stat = self.stats.setdefault(trick, Counter())
        stat['candidates'] += 1
        mark = (g.unique_id, trick, key)
        if mark in self._seen:
            stat['duplicates'] += 1
            return None
        self._seen.add(mark)
        if not self.hybrids and not g.is_all_quantum:
            stat['hybrid skipped'] += 1
            return None
        child = forward_child(trick, params, g)
        if child is None or self.index.key(child) != key:
            stat['rejected'] += 1
            return None
        stat['confirmed'] += 1
        gk = self.index.key(g)
        if gk == key:
            stat['same class'] += 1
            return None
        stored = self.structures.get(g.unique_unlabelled_id)
        if stored is None:
            stored = self._store(g)
        elif stored.unique_id != g.unique_id:
            perm = permutation_to(g, stored)
            assert perm is not None, "two structures with one unlabelled id must be relabellings"
            params = translate_params(trick, params, perm)
            stat['relabelled'] += 1
            child_check = forward_child(trick, params, stored)
            assert child_check is not None and self.index.key(child_check) == key, "translated params must replay"
            child = child_check
        self._store(child)
        self._record(Transition(trick, params, gk, key, stored.unique_unlabelled_id, child.unique_unlabelled_id))
        if self._add_known(gk, stored, trick):
            stat['new keys'] += 1
            return gk
        return None

    def _reverse_upward(self, key: UnlabelledId, c: QmDAG) -> List[UnlabelledId]:
        """The classes structurally above `key` inside its classical class, through ClassIndex.above_steps; the
        recorded transition is the forward upward lookup, byte for byte."""
        stat = self.stats.setdefault('upward', Counter())
        new = []
        for params, above, rep in self.index.above_steps(c):
            stat['candidates'] += 1
            stat['confirmed'] += 1
            above_key = self.index.key(above)
            self._store(above)
            self._store(rep)
            self._record(Transition('upward', params, above_key, key, above.unique_unlabelled_id, rep.unique_unlabelled_id))
            if self._add_known(above_key, above, 'upward'):
                stat['new keys'] += 1
                new.append(above_key)
        return new

    # -- the closure ------------------------------------------------------------------------------------------

    def run(self, seeds: Dict[str, QmDAG], extra_known: Iterable[UnlabelledId] = ()) -> "ReverseClosure":
        t_start = time.time()
        queue: deque = deque()
        for key, name in group_seeds_by_key(self.index.key, seeds).items():
            g = next(s for s in seeds.values() if self.index.key(s) == key)
            self._store(g)
            if self._add_known(key, g, 'seed'):
                queue.append(key)
        for key in extra_known:
            rep = self.index.representative(key)
            if rep is not None and self._add_known(key, rep, 'extra'):
                queue.append(key)
        expanded = 0
        while queue:
            key = queue.popleft()
            expanded += 1
            if self.verbose and expanded % 1000 == 0:
                print(f"[reverse closure] {expanded} keys expanded, {len(self.known)} known, {len(queue)} queued")
            for c in self.colourings(key):
                n = c.number_of_visible
                if n > self.max_visible:
                    continue
                if self.lookups:
                    t0 = time.time()
                    for trick, params, g in inverse_degradation(c):
                        gk = self._accept(trick, params, g, key)
                        if gk is not None:
                            queue.append(gk)
                    self.seconds['degradation'] += time.time() - t0
                    t0 = time.time()
                    if c.is_all_quantum and self.index.is_class_key(key):
                        queue.extend(self._reverse_upward(key, c))
                    self.seconds['upward'] += time.time() - t0
                if n >= self.max_visible:
                    continue
                for name, inverse in self.inverses.items():
                    t0 = time.time()
                    for trick, params, g in inverse(c, hybrids=self.hybrids):
                        gk = self._accept(trick, params, g, key)
                        if gk is not None:
                            queue.append(gk)
                    self.seconds[name] += time.time() - t0
        self.seconds['total'] += time.time() - t_start
        return self

    # -- reporting --------------------------------------------------------------------------------------------

    def known_by_nodes(self) -> Dict[int, Tuple[int, int]]:
        """node count -> (class keys, other keys) among the known keys."""
        out: Dict[int, List[int]] = {}
        for key in self.known:
            n = key[0]
            slot = out.setdefault(n, [0, 0])
            slot[0 if self.index.is_class_key(key) else 1] += 1
        return {n: tuple(v) for n, v in sorted(out.items())}

    def summary(self) -> str:
        cols = ('candidates', 'confirmed', 'rejected', 'duplicates', 'relabelled', 'same class', 'hybrid skipped', 'new keys')
        lines = [f"reverse closure: {len(self.known)} known keys {self.known_by_nodes()} (class keys, other keys) in "
                 f"{self.seconds['total']:.1f} s; {sum(len(v) for v in self.edges.values())} recorded edges",
                 'trick'.ljust(28) + ''.join(c.rjust(12) for c in cols) + 'seconds'.rjust(10)]
        for trick in sorted(self.stats):
            st = self.stats[trick]
            lines.append(trick.ljust(28) + ''.join(str(st.get(c, 0)).rjust(12) for c in cols) + f"{self.seconds.get(trick, 0.0):10.1f}")
        return '\n'.join(lines)
