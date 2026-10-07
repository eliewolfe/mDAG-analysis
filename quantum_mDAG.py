from __future__ import absolute_import
import itertools
import warnings
import numpy as np
import numpy.typing as npt
from hypergraphs import Hypergraph, LabelledHypergraph, hypergraph_full_cleanup
from directed_structures import DirectedStructure, LabelledDirectedStructure
from mDAG_advanced import mDAG
from sys import version_info
assert version_info >= (3, 8), "Python 3.8+ is required for cached_property support."
from functools import total_ordering
from utilities import stringify_in_set as stringify
from typing import Any, Dict, Iterable, List, Set, Tuple
try:
    import networkx as nx
except ImportError:
    print("Functions which depend on networkx are not available.")

from functools import cached_property
from methodtools import lru_cache

BoolMatrix = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int_]


def C_facets_not_dominated_by_Q(c_facets: Set[frozenset], q_facets: Set[frozenset]) -> Set[frozenset]:
    c_facets_copy = c_facets.copy()
    for Q_facet in q_facets:
        dominated_by_quantum = set(filter(Q_facet.issuperset, c_facets_copy))
        c_facets_copy.difference_update(dominated_by_quantum)
    return c_facets_copy


def upgrade_to_QmDAG(mdag: mDAG) -> "QmDAG":
    return QmDAG(
        mdag.directed_structure_instance,
        Hypergraph([], mdag.number_of_visible),
        mdag.simplicial_complex_instance)

def as_classical_QmDAG(mdag: mDAG) -> "QmDAG":
    return QmDAG(
        mdag.directed_structure_instance,
        mdag.simplicial_complex_instance,
        Hypergraph([], mdag.number_of_visible))


_UNLABELLED_ID_MEMO: Dict[Tuple, Tuple[int, int, int, int]] = dict()
# Children of a structure under the piggybacks, keyed by its unlabelled id and the search options. All piggybacks
# are label-equivariant, so the children of a relabelled structure are relabellings of these.
_PIGGYBACK_CHILDREN_MEMO: Dict[Tuple, List["QmDAG"]] = dict()
# Entropic LPs (Shannon cone + local Markov equalities) per labelled structure; each holds a solver task.
_ENTROPIC_LP_CACHE: Dict[Tuple, Any] = dict()
_ENTROPIC_LP_CACHE_SIZE = 64
# Outcome tally of entropic certificates, keyed by ('single', outcome) for the per-target admissibility test
# (outcome in 'dsep', 'markov', 'relabel', 'fail'), ('joint', outcome) for multi-target candidates and
# ('deletions', outcome) for the greedy extra-deletion verification. Read it to see how often the LP fails.
ENTROPIC_STATS: Dict[Tuple[str, str], int] = dict()


def _tally(kind: str, outcome: str) -> None:
    ENTROPIC_STATS[(kind, outcome)] = ENTROPIC_STATS.get((kind, outcome), 0) + 1


# This class does NOT represent every possible quantum causal structure. It only represents the causal structures where every quantum latent is exogenized. This is the case, for example, of the known QC Gaps.
@total_ordering
class QmDAG:
    def __init__(self, directed_structure_instance: DirectedStructure, C_simplicial_complex_instance: Hypergraph, Q_simplicial_complex_instance: Hypergraph,
                 pp_restrictions: Tuple[int, ...] = tuple()) -> None:
        self.restricted_perfect_predictions_numeric = pp_restrictions
        self.directed_structure_instance = directed_structure_instance
        self.number_of_visible = self.directed_structure_instance.number_of_visible
        assert directed_structure_instance.number_of_visible == C_simplicial_complex_instance.number_of_visible, 'Different number of nodes in directed structure vs classical simplicial complex.'
        assert directed_structure_instance.number_of_visible == Q_simplicial_complex_instance.number_of_visible, 'Different number of nodes in directed structure vs quantum simplicial complex.'

        self.Q_simplicial_complex_instance = Q_simplicial_complex_instance
        if hasattr(C_simplicial_complex_instance, 'variable_names'):
            self.C_simplicial_complex_instance = LabelledHypergraph(
                C_simplicial_complex_instance.variable_names,
                C_facets_not_dominated_by_Q(
                C_simplicial_complex_instance.translated_simplicial_complex,
                Q_simplicial_complex_instance.translated_simplicial_complex
            ))
        else:
            self.C_simplicial_complex_instance = Hypergraph(C_facets_not_dominated_by_Q(
                C_simplicial_complex_instance.simplicial_complex_as_sets,
                Q_simplicial_complex_instance.simplicial_complex_as_sets
            ), self.number_of_visible)
        if hasattr(self.directed_structure_instance, 'variable_names'):
            self.variable_names = self.directed_structure_instance.variable_names
            if hasattr(self.C_simplicial_complex_instance, 'variable_names'):
                assert frozenset(self.variable_names) == frozenset(
                    self.C_simplicial_complex_instance.variable_names), 'Error: Inconsistent node names.'
                if not tuple(self.variable_names) == tuple(self.C_simplicial_complex_instance.variable_names):
                    print('Warning: Inconsistent node ordering. Following ordering of directed structure!')
        if hasattr(self.directed_structure_instance, 'variable_names'):
            self.variable_names = self.directed_structure_instance.variable_names
            if hasattr(self.Q_simplicial_complex_instance, 'variable_names'):
                assert frozenset(self.variable_names) == frozenset(
                    self.Q_simplicial_complex_instance.variable_names), 'Error: Inconsistent node names.'
                if not tuple(self.variable_names) == tuple(self.Q_simplicial_complex_instance.variable_names):
                    print('Warning: Inconsistent node ordering. Following ordering of directed structure!')
        self.visible_nodes = self.directed_structure_instance.visible_nodes
        self.classical_latent_nodes = tuple(
            range(self.number_of_visible, self.C_simplicial_complex_instance.number_of_visible_plus_latent))
        self.nonsingleton_classical_latent_nodes = tuple(range(self.number_of_visible,
                                                               self.C_simplicial_complex_instance.number_of_visible_plus_nonsingleton_latent))
        # it is not necessary to talk about quantum singletons in the first place:
        self.quantum_latent_nodes = tuple(range(self.C_simplicial_complex_instance.number_of_visible_plus_latent,
                                                self.Q_simplicial_complex_instance.number_of_visible_plus_nonsingleton_latent
                                                + self.C_simplicial_complex_instance.number_of_visible_plus_latent
                                                - self.number_of_visible))
        self.vis_nodes_with_no_children = set(self.directed_structure_instance.nodes_with_no_children)

    @cached_property
    def exogenous_visible_nodes(self) -> Set[int]:
        """Visible nodes with no visible parents and no (nonsingleton) latent parents, classical or quantum."""
        return set(self.directed_structure_instance.nodes_with_no_parents).intersection(
            self.C_simplicial_complex_instance.vis_nodes_with_singleton_latent_parents,
            self.Q_simplicial_complex_instance.vis_nodes_with_singleton_latent_parents)

    @cached_property
    def as_string(self) -> str:
        return 'Children'.ljust(10) + ': ' + self.directed_structure_instance.as_string \
               + '\nClassical'.ljust(11) + ': ' + self.C_simplicial_complex_instance.as_string \
               + '\nQuantum'.ljust(11) + ': ' + self.Q_simplicial_complex_instance.as_string + '\n'

    def __str__(self) -> str:
        return self.as_string

    def __repr__(self) -> str:
        return self.as_string

    #@cached_property
    @property
    def unique_id(self) -> Tuple[int, int, int, int, Tuple[int, ...]]:
        # Returns a unique identification tuple.
        return (
            self.number_of_visible,
            self.directed_structure_instance.as_integer,
            self.C_simplicial_complex_instance.as_integer,
            self.Q_simplicial_complex_instance.as_integer,
            tuple(self.restricted_perfect_predictions_numeric,))
    def __hash__(self) -> int:
        return hash(self.unique_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QmDAG):
            return False
        return self.unique_id == other.unique_id

    def __lt__(self, other: "QmDAG") -> bool:
        return self.unique_id < other.unique_id

    @cached_property
    def unique_unlabelled_id(self) -> Tuple[int, int, int, int]:
        # Returns a unique identification tuple up to relabelling.
        # Memoized across equal-but-distinct objects, since the search constructs many copies.
        key = self.unique_id
        try:
            return _UNLABELLED_ID_MEMO[key]
        except KeyError:
            unlabelled_id = (self.number_of_visible,) + min(zip(
                self.directed_structure_instance.as_integer_permutations,
                self.C_simplicial_complex_instance.as_integer_permutations,
                self.Q_simplicial_complex_instance.as_integer_permutations
            ))
            _UNLABELLED_ID_MEMO[key] = unlabelled_id
            return unlabelled_id

    #ON THE POINT DISTRIBUTION TRICK

    def subgraph(self, list_of_nodes: Tuple[int, ...]) -> "QmDAG":
        return QmDAG(
            LabelledDirectedStructure(list_of_nodes, self.directed_structure_instance.edge_list),
            LabelledHypergraph(list_of_nodes, self.C_simplicial_complex_instance.simplicial_complex_as_sets),
            LabelledHypergraph(list_of_nodes, self.Q_simplicial_complex_instance.simplicial_complex_as_sets),
        )

    def fix_to_point_distribution_QmDAG(self, node: int) -> "QmDAG":  # returns a smaller QmDAG
        return self.subgraph(self.visible_nodes[:node] + self.visible_nodes[(node + 1):])

    def _subgraphs_generator(self) -> Iterable["QmDAG"]:
        for r in range(3, self.number_of_visible):
            for to_keep in itertools.combinations(self.visible_nodes, r):
                yield self.subgraph(to_keep)

    @cached_property
    def subgraphs(self) -> Set["QmDAG"]:
        return {self.subgraph(to_keep) for to_keep in itertools.combinations(self.visible_nodes, self.number_of_visible-1)}
        # return set(self._subgraphs_generator())

    @cached_property
    def unique_unlabelled_ids_obtainable_by_PD_trick(self) -> Set[Tuple[int, int, int, int]]:
        return set(subQmDAG.unique_unlabelled_id for subQmDAG in self.subgraphs)

    # ON THE MARGINALIZATION TRICK

    def classical_sibling_sets_of(self, node: int) -> Set[frozenset]:
        return set(facet.difference({node}) for facet in self.C_simplicial_complex_instance.simplicial_complex_as_sets if
                node in facet)

    def quantum_sibling_sets_of(self, node: int) -> Set[frozenset]:
        return set(facet.difference({node}) for facet in self.Q_simplicial_complex_instance.simplicial_complex_as_sets if
                node in facet)

    def quantum_siblings_of(self, node: int) -> Set[Any]:
        return set(itertools.chain.from_iterable(self.quantum_sibling_sets_of(node)))

    @cached_property
    def as_mDAG(self) -> mDAG:
        return mDAG(
            self.directed_structure_instance,
            Hypergraph(C_facets_not_dominated_by_Q(
                self.C_simplicial_complex_instance.simplicial_complex_as_sets,
                self.Q_simplicial_complex_instance.simplicial_complex_as_sets
            ).union(
                self.Q_simplicial_complex_instance.simplicial_complex_as_sets
            ), self.number_of_visible),
            pp_restrictions=self.restricted_perfect_predictions_numeric
        )

    @cached_property
    def as_graph(self):
        return self.as_mDAG.as_graph

    def latent_sibling_sets_of(self, node: int) -> Set[frozenset]:
        return set(facet.difference({node}) for facet in self.as_mDAG.simplicial_complex_instance.simplicial_complex_as_sets if
                node in facet)

    def latent_siblings_of(self, node: int) -> Set[Any]:
        return set(itertools.chain.from_iterable(self.latent_sibling_sets_of(node)))
    
    def has_grandparents_that_are_not_parents(self, node: int) -> bool:
        visible_parents = set(np.flatnonzero(self.directed_structure_instance.as_bit_square_matrix[:, node]))
        for parent in visible_parents:
            grandparents=set(np.flatnonzero(self.directed_structure_instance.as_bit_square_matrix[:, parent]))
            if not grandparents.issubset(visible_parents):
                return True
        return False
    
    def parents_have_external_latents(self, node: int) -> bool:
        """True if some visible parent of `node` lies in a latent facet (classical or quantum) that neither contains
        `node` nor lies inside the set of visible parents of `node`. Conditioning on `node` then correlates that
        facet's outside children in a way the output structure cannot express, so the conditioning piggyback is
        not justified. (A facet contained in the parent block only redistributes the block and is harmless.)"""
        visible_parents = set(np.flatnonzero(self.directed_structure_instance.as_bit_square_matrix[:, node]))
        for facet in self.as_mDAG.simplicial_complex_instance.simplicial_complex_as_sets:
            if node not in facet and not facet.isdisjoint(visible_parents) and not facet.issubset(visible_parents):
                return True
        return False

    def conditioning_is_justified(self, node: int, strict_latents: bool = True) -> bool:
        """Three conditions. (1) Every visible grandparent of `node` is a parent of `node`. (2, strict_latents)
        every facet containing a visible parent of `node` contains `node` or lies within the parents. (3) A visible
        parent that shares no facet with `node` has to receive the post-selected common cause through its own
        output (it guesses it, and `node` post-selects on the guess); its output is then fine-grained, which is
        only harmless if every visible child of that parent is `node`, another parent, or a latent sibling of
        `node` (nodes that can read the common cause from the new facet). Then the block of parents is closed
        under all its inputs and post-selecting on `node` can be absorbed into one common cause of parents and
        latent siblings."""
        if self.has_grandparents_that_are_not_parents(node):
            return False
        if strict_latents and self.parents_have_external_latents(node):
            return False
        parents = set(np.flatnonzero(self.directed_structure_instance.as_bit_square_matrix[:, node]))
        siblings = set(self.latent_siblings_of(node))
        allowed_children = parents | siblings | {node}
        for p in parents:
            if p in siblings:
                continue
            children = set(np.flatnonzero(self.directed_structure_instance.as_bit_square_matrix[p]))
            if not children.issubset(allowed_children):
                return False
        return True

    def condition(self, node: int) -> "QmDAG":
        #assume we already checked that conditioning is justified (conditioning_is_justified)
        remaining_nodes = self.visible_nodes[:node] + self.visible_nodes[(node + 1):]
        # new_directed_edges = set(self.directed_structure_instance.edge_list)
        visible_parents = set(np.flatnonzero(self.directed_structure_instance.as_bit_square_matrix[:, node]))
        new_C_facets=self.C_simplicial_complex_instance.simplicial_complex_as_sets.copy()
        new_C_facets.add(frozenset(visible_parents.union(self.latent_siblings_of(node))))
        new_Q_facets=self.Q_simplicial_complex_instance.simplicial_complex_as_sets.copy()
        new_Q_facets.add(frozenset(self.quantum_siblings_of(node)))
        return QmDAG(
                LabelledDirectedStructure(remaining_nodes, self.directed_structure_instance.edge_list),
                LabelledHypergraph(remaining_nodes, new_C_facets),
                LabelledHypergraph(remaining_nodes, new_Q_facets),
                )

    def _subconditionals(self) -> Iterable["QmDAG"]:
        if self.number_of_visible > 3:
            for node in self.visible_nodes:
                if self.conditioning_is_justified(node):
                    conditional_QM = self.condition(node)
                    yield conditional_QM
                    for new_QmDAG in conditional_QM.subconditionals:
                        yield new_QmDAG

    @cached_property
    def subconditionals(self) -> Set["QmDAG"]:
        return set(self._subconditionals())
    
    @cached_property
    def unique_unlabelled_ids_obtainable_by_conditioning(self) -> Set[Tuple[int, int, int, int]]:
        return set(new_QmDAG.unique_unlabelled_id for new_QmDAG in self.subconditionals)


    def interruption_creation(self, node_with_no_children: int, node_with_no_parents: int) -> "QmDAG":
        """We make the children of the node with no parents into the children of the node that previously had no children, and then we remove the node with no parents."""
        remaining_nodes = self.visible_nodes[:node_with_no_parents] + self.visible_nodes[(node_with_no_parents + 1):]
        new_directed_edges = set(self.directed_structure_instance.edge_list)
        for edge in self.directed_structure_instance.edge_list:
            if edge[0] == node_with_no_parents:
                new_directed_edges.add((node_with_no_children, edge[1]))
                new_directed_edges.remove(edge)
        return QmDAG(
                LabelledDirectedStructure(remaining_nodes, list(new_directed_edges)),
                LabelledHypergraph(remaining_nodes, self.C_simplicial_complex_instance.simplicial_complex_as_sets),
                LabelledHypergraph(remaining_nodes, self.Q_simplicial_complex_instance.simplicial_complex_as_sets),
                )


    def _subinterruptions(self) -> Set["QmDAG"]:
        for node_with_no_children in self.vis_nodes_with_no_children:
            for node_with_no_parents in self.exogenous_visible_nodes:
                if node_with_no_children in self.directed_structure_instance.adjMat.descendantsplus_of(node_with_no_parents):
                    continue  # we don't want to create a cycle
                yield self.interruption_creation(node_with_no_children, node_with_no_parents)
    
    @cached_property
    def subinterruptions(self) -> Set["QmDAG"]:
        return set(self._subinterruptions())

    @cached_property
    def unique_unlabelled_ids_obtainable_by_interruption(self) -> Set[Tuple[int, int, int, int]]:
        return set(new_QmDAG.unique_unlabelled_id for new_QmDAG in self.subinterruptions)




    def marginalize(self, node: int, districts_check: bool = False, apply_teleportation: bool = True) -> "QmDAG":  # returns a smaller QmDAG
        remaining_nodes = self.visible_nodes[:node] + self.visible_nodes[(node + 1):]
        # Pass visible children on to visible children
        # Pass latent children on to visible children **classically**
        # Apply teleportation
        visible_children = set(np.flatnonzero(self.directed_structure_instance.as_bit_square_matrix[node]))
        visible_parents = set(np.flatnonzero(self.directed_structure_instance.as_bit_square_matrix[:, node]))
        new_directed_edges = set(self.directed_structure_instance.edge_list)
        for parent in visible_parents:
            for child in visible_children:
                new_directed_edges.add((parent, child))
        new_C_facets = self.C_simplicial_complex_instance.simplicial_complex_as_sets.copy()
        C_facets_to_grow = self.latent_sibling_sets_of(node)
        if len(C_facets_to_grow)>0:
            for C_facet_to_grow in C_facets_to_grow:
                new_C_facets.add(C_facet_to_grow.union(visible_children))
        else:
            new_C_facets.add(frozenset(visible_children))
        new_C_facets = hypergraph_full_cleanup(new_C_facets)
        new_C_simplicial_complex = LabelledHypergraph(remaining_nodes, new_C_facets)
        if not districts_check:
            ok_to_proceed = True
        else:
            previous_districts = [district.difference({node}) for district in self.as_mDAG.numerical_districts]
            new_districts = new_C_simplicial_complex.translated_districts
            ok_to_proceed = frozenset(map(frozenset, previous_districts)) == frozenset(map(frozenset, new_districts))
        if not ok_to_proceed:
            return None  # the marginalization trick does not apply when districts are not preserved
        else:
            if not apply_teleportation:
                return QmDAG(
                    LabelledDirectedStructure(remaining_nodes, list(new_directed_edges)),
                    LabelledHypergraph(remaining_nodes, new_C_facets),
                    LabelledHypergraph(remaining_nodes, self.Q_simplicial_complex_instance.simplicial_complex_as_sets),
                    )
            else:
                teleportable_children = self.quantum_siblings_of(node).intersection(visible_children)
                new_Q_facets = self.Q_simplicial_complex_instance.simplicial_complex_as_sets.copy()
                Q_facets_to_grow = self.quantum_sibling_sets_of(node)
                for Q_facet_to_grow in Q_facets_to_grow:
                    new_Q_facets.add(Q_facet_to_grow.union(teleportable_children))
                new_Q_facets = hypergraph_full_cleanup(new_Q_facets)
                return QmDAG(
                    LabelledDirectedStructure(remaining_nodes, list(new_directed_edges)),
                    LabelledHypergraph(remaining_nodes, new_C_facets),
                    LabelledHypergraph(remaining_nodes, new_Q_facets),
                    )


    def _submarginals(self, **kwargs):
        if self.number_of_visible > 3:
            for node in self.visible_nodes:
                marginalized_QM = self.marginalize(node, **kwargs)
                if marginalized_QM is None:
                    continue
                yield marginalized_QM
                for new_QmDAG in marginalized_QM.submarginals(**kwargs):
                    yield new_QmDAG

    def submarginals(self, **kwargs):
        return set(self._submarginals(**kwargs))

    def _unique_unlabelled_ids_obtainable_by_marginalization(self, **kwargs):
        return set(new_QmDAG.unique_unlabelled_id for new_QmDAG in self.submarginals(**kwargs))

    @lru_cache(maxsize=None)
    def unique_unlabelled_ids_obtainable_by_naive_marginalization(self, **kwargs):
        new_kwargs = kwargs.copy()
        new_kwargs['apply_teleportation'] = False
        return set(self._unique_unlabelled_ids_obtainable_by_marginalization(**new_kwargs))

    @lru_cache(maxsize=None)
    def unique_unlabelled_ids_obtainable_by_marginalization(self, **kwargs):
        new_kwargs = kwargs.copy()
        new_kwargs['apply_teleportation'] = True
        return set(self._unique_unlabelled_ids_obtainable_by_marginalization(**new_kwargs))

    def unique_unlabelled_ids_obtainable_by_reduction(self, **kwargs):
        subgraph_unlabelled_ids = set(self.unique_unlabelled_ids_obtainable_by_PD_trick)
        subgraph_unlabelled_ids.update(self.unique_unlabelled_ids_obtainable_by_conditioning)
        subgraph_unlabelled_ids.update(self.unique_unlabelled_ids_obtainable_by_marginalization(**kwargs))
        subgraph_unlabelled_ids.update(self.unique_unlabelled_ids_obtainable_by_interruption)
        return subgraph_unlabelled_ids

    # ------------------------------------------------------------------
    # THE FRITZ PIGGYBACK
    #
    # Let X1 be a set of visible nodes (the predictors, jointly) and s a visible node sharing a latent with some
    # member of X1 (a candidate predicted node). In the effective DAG (visible nodes, one node per latent facet, one
    # private-noise node per visible node) split the parents of s into common(s), those also seen by X1 (parents of
    # some predictor, or predictors themselves), and others(s). If X1 is d-separated from others(s) given common(s),
    # then:
    #   * classically, any model in which X1 perfectly predicts s can be rewritten so that s depends on common(s) only;
    #   * quantumly, any strategy for the reduced structure in which s is a deterministic function of its (classical)
    #     parents extends to the original structure with X1 outputting a copy of s.
    # Hence the structure G' obtained by deleting X1 and restricting s to common(s) satisfies: a QC gap in G' implies a
    # QC gap in G, with no caveat about perfect correlations in G'. Quantum facets read by s become classical for s
    # (the other children of the facet keep its quantum part).
    # In "copy" mode s is left untouched and a fresh node s_copy carrying the common (classical) part is added; this is
    # the node-splitting version of the trick (e.g. triangle -> Bell).
    # Predictors with children cannot simply be deleted: they are removed by the marginalization piggyback instead
    # (relaying their parents to their children, teleporting quantum shares when allowed) while still outputting the
    # copies of the predicted nodes, so the composition is sound whenever marginalization is. Since teleportation makes
    # marginalization order-dependent, every order of removing the predictors is enumerated.
    # ------------------------------------------------------------------

    @cached_property
    def effective_DAG_data(self) -> Tuple["nx.DiGraph", Dict[int, Tuple[str, frozenset]]]:
        """Directed graph on visible nodes plus one node per classical facet, quantum facet and private noise source.
        Returns the graph and a dict latent_index -> (kind, frozenset of visible children)."""
        n = self.number_of_visible
        edges = set(self.directed_structure_instance.as_set_of_tuples)
        latent_nodes: Dict[int, Tuple[str, frozenset]] = dict()
        idx = n
        for kind, hypergraph in (('C', self.C_simplicial_complex_instance), ('Q', self.Q_simplicial_complex_instance)):
            for facet in sorted(map(tuple, map(sorted, hypergraph.simplicial_complex_as_sets))):
                latent_nodes[idx] = (kind, frozenset(facet))
                edges.update((idx, v) for v in facet)
                idx += 1
        for v in range(n):
            latent_nodes[idx] = ('noise', frozenset({v}))
            edges.add((idx, v))
            idx += 1
        g = nx.DiGraph()
        g.add_nodes_from(range(idx))
        g.add_edges_from(edges)
        return g, latent_nodes

    def fritz_admissible_targets(self, predictors: Iterable[int],
                                 allow_childful_predictors: bool = True) -> Dict[int, Tuple[frozenset, frozenset]]:
        """Maps each admissible predicted node s to (common(s), others(s)) in effective-DAG indices.
        The predictors jointly predict s: common(s) are the parents of s seen by at least one predictor."""
        predictors = frozenset(predictors)
        if not allow_childful_predictors:
            assert predictors.issubset(self.vis_nodes_with_no_children), "Fritz predictors must be childless visible nodes."
        g, latent_nodes = self.effective_DAG_data
        seen_by_predictors = set(predictors)
        for y in predictors:
            seen_by_predictors.update(g.predecessors(y))
        candidates = set()
        for y in predictors:
            candidates.update(self.latent_siblings_of(y))
        candidates.difference_update(predictors)
        admissible = dict()
        for s in sorted(candidates):
            parents = set(g.predecessors(s))
            common = parents.intersection(seen_by_predictors)
            others = parents.difference(common)
            # `others` always contains the private noise of s, so a predictor downstream of s is never admissible.
            # A predictor that is itself a parent of s sits in common(s) and is conditioned on rather than tested
            # (networkx treats an empty first set as d-separated: conditioning on the predictors fixes s outright).
            if nx.is_d_separator(g, predictors.difference(common), others, common):
                admissible[s] = (frozenset(common), frozenset(others))
        return admissible

    def _fritz_kept_parents(self, admissible: Dict[int, Tuple[frozenset, ...]],
                            choices: Dict[int, str]) -> Dict[int, frozenset]:
        """The candidate reduced structure: predicted nodes keep common(s), every other node keeps all its parents.
        Keys are effective-DAG indices of visible nodes and latent facets (noise sources excluded)."""
        nodes, parents = self.lp_structure
        kept = dict(parents)
        for s in choices:
            kept[s] = frozenset(admissible[s][0])
        return kept

    def _fritz_build(self, predictors: frozenset, choices: Dict[int, str],
                     kept_parents: Dict[int, frozenset],
                     drop_predictors: bool = True, keep_quantum_facets: bool = False) -> Tuple["QmDAG", Dict[Any, int]]:
        """Builds the post-Fritz QmDAG. `choices` gives the mode per predicted node ('replace' or 'copy');
        `kept_parents` gives, for every visible node, the parents (visible nodes and latent facets, as effective-DAG
        indices) it keeps: common(s) for predicted nodes, possibly fewer for others after LP-certified deletions.
        In copy mode the original keeps all its parents and the copy gets kept_parents[s].
        Returns the QmDAG and the name -> index translation."""
        g, latent_nodes = self.effective_DAG_data
        copies = {s: str(s) + '_copy' for s, mode in choices.items() if mode == 'copy'}
        kept_originals = [v for v in self.visible_nodes if drop_predictors is False or v not in predictors]
        names = tuple(kept_originals) + tuple(copies[s] for s in sorted(copies))
        name_set = set(names)

        edges = set()
        for (a, b) in self.directed_structure_instance.as_set_of_tuples:
            if choices.get(b) == 'copy':
                edges.add((a, b))
                if a in kept_parents[b]:
                    edges.add((a, copies[b]))
            elif a in kept_parents[b]:
                edges.add((a, b))
        # A copy is a sub-output of s, so it feeds exactly the children that still see s.
        for s, s_copy in copies.items():
            for c in self.directed_structure_instance.adjMat.children_of(s):
                if choices.get(c) == 'copy':
                    edges.add((s_copy, c))
                    if s in kept_parents[c]:
                        edges.add((s_copy, copies[c]))
                elif s in kept_parents[c]:
                    edges.add((s_copy, c))
        edges = [(a, b) for (a, b) in edges if a in name_set and b in name_set]

        C_facets = set()
        Q_facets = set()
        for idx, (kind, facet) in latent_nodes.items():
            if kind == 'noise':
                continue
            quantum_readers = set()
            classical_readers = set()
            replaced_reader_present = False
            for v in facet:
                if drop_predictors and v in predictors:
                    continue
                keeps = idx in kept_parents[v]
                if choices.get(v) == 'copy':
                    quantum_readers.add(v)
                    if keeps:
                        classical_readers.add(copies[v])
                elif v in choices:
                    if keeps:
                        classical_readers.add(v)
                        replaced_reader_present = True
                elif keeps:
                    quantum_readers.add(v)
            if kind == 'C':
                C_facets.add(frozenset(quantum_readers.union(classical_readers)))
            else:
                if classical_readers:
                    C_facets.add(frozenset(quantum_readers.union(classical_readers)))
                if keep_quantum_facets or not replaced_reader_present:
                    Q_facets.add(frozenset(quantum_readers))
        new_ds = LabelledDirectedStructure(names, edges)
        new_QmDAG = QmDAG(new_ds, LabelledHypergraph(names, C_facets), LabelledHypergraph(names, Q_facets))
        return new_QmDAG, new_ds.translation_dict

    def fritz_transitions(self, predictors: Iterable[int], modes: Tuple[str, ...] = ('replace', 'copy'),
                          max_visible: int = None, min_visible: int = 3,
                          keep_quantum_facets: bool = True, districts_check: bool = False,
                          allow_childful_predictors: bool = True,
                          apply_teleportation: bool = True,
                          predictor_mode: str = 'drop', _presplit: bool = False) -> List[Tuple[Tuple[Tuple[int, str], ...], "QmDAG"]]:
        """All structures obtainable by the Fritz piggyback with the given (jointly predicting) predictors.
        predictor_mode 'drop' (default, cheap): X1 itself is removed, deleted if childless, otherwise by the
        marginalization piggyback (every removal order, since teleportation is order-dependent).
        predictor_mode 'split': every predictor is first split (node splitting, a copy with the same parents AND
        the same children) and the copies are the predictors, dropped as above; the originals stay. For a
        childless predictor this is the same as keeping it untouched. For a predictor with children it is NOT:
        keeping a childful predictor untouched is unsound (its children could read the prediction through the
        visible edge; e.g. 0->1 with quantum facets {0,1},{0,2},{1,2} is saturated, yet "0 predicts 2, 0 kept"
        would yield the instrumental gap), whereas marginalizing the copy relays what the children could learn.
        Returns (params, QmDAG) pairs where params = ((s, mode), ...) sorted by s."""
        predictors = frozenset(predictors)
        if max_visible is None:
            max_visible = self.number_of_visible + 1
        childless = predictors.issubset(self.vis_nodes_with_no_children)
        assert childless or allow_childful_predictors, "Fritz predictors must be childless visible nodes."
        childful = predictors.difference(self.vis_nodes_with_no_children)
        if predictor_mode == 'split' and childful and not _presplit:
            work, copies = self._split_predictors(childful)
            return work.fritz_transitions(predictors.difference(childful).union(copies), modes=modes,
                                          max_visible=max_visible, min_visible=min_visible,
                                          keep_quantum_facets=keep_quantum_facets, districts_check=districts_check,
                                          allow_childful_predictors=True, apply_teleportation=apply_teleportation,
                                          predictor_mode='split', _presplit=True)
        # Which predictors leave the structure: all of them in 'drop' mode; in 'split' mode only the childful ones
        # (the copies), which are marginalized; childless predictors are kept untouched.
        to_remove = predictors if predictor_mode == 'drop' else childful
        admissible = self.fritz_admissible_targets(predictors, allow_childful_predictors=allow_childful_predictors)
        targets = sorted(admissible)
        results = []
        for r in range(1, len(targets) + 1):
            for chosen in itertools.combinations(targets, r):
                for mode_choice in itertools.product(modes, repeat=r):
                    new_size = self.number_of_visible - len(to_remove) + mode_choice.count('copy')
                    if not (min_visible <= new_size <= max_visible):
                        continue
                    params = tuple(zip(chosen, mode_choice))
                    kept_parents = self._fritz_kept_parents(admissible, dict(params))
                    built = self._fritz_build(predictors, dict(params), kept_parents,
                                              drop_predictors=(predictor_mode == 'drop' and childless),
                                              keep_quantum_facets=keep_quantum_facets)
                    intermediate, to_nums = built
                    to_original = {num: self._fritz_original_of(name) for name, num in to_nums.items()}
                    to_marginalize = to_remove.difference(self.vis_nodes_with_no_children) if predictor_mode == 'split' \
                        else (frozenset() if childless else predictors)
                    if not to_marginalize:
                        candidates = [(intermediate, to_original)]
                    else:
                        candidates = [self._marginalize_predictors(intermediate, to_original, order,
                                                                   districts_check=districts_check,
                                                                   apply_teleportation=apply_teleportation)
                                      for order in itertools.permutations(sorted(to_marginalize))]
                    seen_here = set()
                    for new_QmDAG, new_to_original in candidates:
                        if new_QmDAG is None or new_QmDAG.unique_id in seen_here:
                            continue
                        if districts_check and not self._fritz_preserves_districts(to_remove, new_QmDAG, new_to_original):
                            continue
                        seen_here.add(new_QmDAG.unique_id)
                        results.append((params, new_QmDAG))
        return results

    def _split_predictors(self, predictors: frozenset) -> Tuple["QmDAG", Tuple[int, ...]]:
        """Splits every predictor into itself and a full copy (same parents, same children); returns the split
        structure and the indices of the copies, which become the predictors to be dropped."""
        work = self
        copies = []
        for x in sorted(predictors):
            work = work.split_node(x)
            copies.append(work.number_of_visible - 1)
        return work, tuple(copies)

    @staticmethod
    def _fritz_original_of(name: Any) -> int:
        return int(str(name).split('_copy')[0]) if isinstance(name, str) else name

    @staticmethod
    def _marginalize_predictors(qmdag: "QmDAG", to_original: Dict[int, int], order: Iterable[int],
                                districts_check: bool, apply_teleportation: bool):
        """Marginalizes the given original nodes out of qmdag in the given order, tracking which original node each
        remaining index refers to. Returns (QmDAG, to_original) or (None, None) if a marginalization is refused."""
        labels = [to_original[i] for i in range(qmdag.number_of_visible)]
        for y in order:
            idx = labels.index(y)
            qmdag = qmdag.marginalize(idx, districts_check=districts_check, apply_teleportation=apply_teleportation)
            if qmdag is None:
                return None, None
            labels.pop(idx)
        return qmdag, dict(enumerate(labels))

    def _fritz_preserves_districts(self, predictors: frozenset, new_QmDAG: "QmDAG", to_original: Dict[int, int]) -> bool:
        """Districts of the output (copies identified with their originals) equal the old districts minus predictors."""
        old_districts = set(frozenset(d.difference(predictors)) for d in self.as_mDAG.numerical_districts)
        old_districts.discard(frozenset())
        new_districts = set(frozenset(to_original[v] for v in d) for d in new_QmDAG.as_mDAG.numerical_districts)
        return old_districts == new_districts

    def fritz_intermediate_with_pp(self, predictors: Iterable[int], choices: Dict[int, str],
                                   keep_quantum_facets: bool = True, allow_childful_predictors: bool = True) -> "QmDAG":
        """The Fritz-reduced structure with the predictors retained, carrying perfect-prediction restrictions
        (each predicted node, or its copy, is a function of the predictors) for supports-based inference."""
        predictors = frozenset(predictors)
        admissible = self.fritz_admissible_targets(predictors, allow_childful_predictors=allow_childful_predictors)
        assert set(choices).issubset(admissible), "Some chosen node is not an admissible Fritz target."
        new_QmDAG, to_nums = self._fritz_build(predictors, choices, self._fritz_kept_parents(admissible, choices),
                                               drop_predictors=False, keep_quantum_facets=keep_quantum_facets)
        predictor_nums = tuple(sorted(to_nums[y] for y in predictors))
        pp = []
        for s, mode in sorted(choices.items()):
            predicted = to_nums[str(s) + '_copy'] if mode == 'copy' else to_nums[s]
            pp.append((predicted, predictor_nums))
        return QmDAG(new_QmDAG.directed_structure_instance, new_QmDAG.C_simplicial_complex_instance,
                     new_QmDAG.Q_simplicial_complex_instance, pp_restrictions=tuple(pp))

    # ------------------------------------------------------------------
    # THE ENTROPIC FRITZ PIGGYBACK (LP-certified edge deletion; Khanna, Pusey and Colbeck)
    #
    # Same output shapes as above, but the classical direction is certified by an entropy-vector LP instead of a
    # d-separation test. Hypotheses: Shannon inequalities over all nodes of G (latent facets as variables, no
    # explicit noise), the local Markov equalities of G, perfect prediction H(s | X1) = 0, and the elementary
    # conditional independences among visible nodes that hold by d-separation in the candidate G' (they hold for
    # free in the quantum lift, since the lifted distribution is Markov to G', yet are genuine extra hypotheses
    # classically). Two sound target sets are tried:
    #   'markov'  : the local Markov equalities of G' over G's own latents;
    #   'relabel' : when s keeps a single latent facet L, the Markov equalities of G'' = G' with L deleted and
    #               s made a parent of L's other children (a G''-model gives a G'-model by setting L := s).
    # The LP subsumes the d-separation test, so it is used as a rescue where d-separation fails. Because the
    # same hypotheses can certify deletions anywhere, a greedy loop may delete further parents of any node
    # (never of a predictor), re-deriving the d-separation hypotheses from the current candidate and verifying
    # the final structure as a whole.
    # ------------------------------------------------------------------

    @cached_property
    def lp_structure(self) -> Tuple[Tuple[int, ...], Dict[int, frozenset]]:
        """Nodes (visible nodes, then latent facets; effective-DAG indices, noise sources excluded) and their
        parent sets."""
        g, latent_nodes = self.effective_DAG_data
        nodes = tuple(v for v in sorted(g.nodes) if v < self.number_of_visible or latent_nodes[v][0] != 'noise')
        node_set = set(nodes)
        parents = {v: frozenset(p for p in g.predecessors(v) if p in node_set) for v in nodes}
        return nodes, parents

    def _entropic_lp(self):
        """The Shannon cone plus the local Markov equalities of this structure (cached per labelled id)."""
        from entropic_lp import EntropicLP, local_markov_rows
        key = self.unique_id
        lp = _ENTROPIC_LP_CACHE.get(key)
        if lp is None:
            nodes, parents = self.lp_structure
            lp = EntropicLP(len(nodes), [row for _, row in local_markov_rows(parents, nodes)])
            if len(_ENTROPIC_LP_CACHE) >= _ENTROPIC_LP_CACHE_SIZE:
                _ENTROPIC_LP_CACHE.pop(next(iter(_ENTROPIC_LP_CACHE))).close()
            _ENTROPIC_LP_CACHE[key] = lp
        return lp

    def _entropic_hypotheses(self, predictors: frozenset, kept_parents: Dict[int, frozenset],
                             predicted: Iterable[int]) -> list:
        from entropic_lp import cond_entropy_row, observable_dseparation_rows
        nodes, _ = self.lp_structure
        rows = [cond_entropy_row([s], predictors) for s in predicted]
        rows += observable_dseparation_rows(kept_parents, nodes, self.visible_nodes)
        return rows

    def _entropic_certificate(self, predictors: frozenset, kept_parents: Dict[int, frozenset],
                              predicted: Tuple[int, ...]):
        """Returns 'markov', 'relabel' or None: whether the LP certifies that a classical model of G in which the
        predictors perfectly predict the predicted nodes yields a classical model of the candidate."""
        from entropic_lp import local_markov_rows
        nodes, parents = self.lp_structure
        lp = self._entropic_lp()
        handle = lp.push_hypotheses(self._entropic_hypotheses(predictors, kept_parents, predicted))
        try:
            if lp.implies_all(row for _, row in local_markov_rows(kept_parents, nodes)):
                return 'markov'
            if len(predicted) == 1:
                s = predicted[0]
                common = kept_parents[s]
                if len(common) == 1 and min(common) >= self.number_of_visible:
                    lam = min(common)
                    relabelled = {v: (ps - {lam}) | {s} if lam in ps else ps
                                  for v, ps in kept_parents.items() if v != lam}
                    relabelled[s] = frozenset()
                    nodes_c = [v for v in nodes if v != lam]
                    if lp.implies_all(row for _, row in local_markov_rows(relabelled, nodes_c)):
                        return 'relabel'
            return None
        finally:
            lp.pop_to(handle)

    def fritz_entropic_admissible_targets(self, predictors: Iterable[int],
                                          allow_childful_predictors: bool = True
                                          ) -> Dict[int, Tuple[frozenset, frozenset, str]]:
        """Like fritz_admissible_targets, certified by the entropic LP. Maps s -> (common(s), others(s), certificate)
        where certificate is 'dsep' (already admissible by d-separation), 'markov' or 'relabel'."""
        predictors = frozenset(predictors)
        if not allow_childful_predictors:
            assert predictors.issubset(self.vis_nodes_with_no_children), "Fritz predictors must be childless visible nodes."
        by_dsep = self.fritz_admissible_targets(predictors, allow_childful_predictors=allow_childful_predictors)
        nodes, parents = self.lp_structure
        g, latent_nodes = self.effective_DAG_data
        noise_of = {next(iter(children)): idx for idx, (kind, children) in latent_nodes.items() if kind == 'noise'}
        seen_by_predictors = set(predictors)
        for y in predictors:
            seen_by_predictors.update(parents[y])
        candidates = set()
        for y in predictors:
            candidates.update(self.latent_siblings_of(y))
        candidates.difference_update(predictors)
        admissible = dict()
        for s in sorted(candidates):
            if s in by_dsep:
                admissible[s] = by_dsep[s] + ('dsep',)
                _tally('single', 'dsep')
                continue
            common = parents[s].intersection(seen_by_predictors)
            others = parents[s].difference(common) | {noise_of[s]}
            kept = dict(parents)
            kept[s] = frozenset(common)
            certificate = self._entropic_certificate(predictors, kept, (s,))
            _tally('single', certificate or 'fail')
            if certificate is not None:
                admissible[s] = (frozenset(common), frozenset(others), certificate)
        return admissible

    def _entropic_extra_deletions(self, predictors: frozenset, kept_parents: Dict[int, frozenset],
                                  predicted: Tuple[int, ...], max_lps: int = 400):
        """Greedy LP-certified deletion of further parents (of any non-predictor node), re-deriving the
        d-separation hypotheses from the current candidate after every deletion. Returns (kept_parents,
        certificate, deleted edges); on failure of the final verification the input candidate is returned."""
        from entropic_lp import cmi_row
        nodes, parents = self.lp_structure
        lp = self._entropic_lp()
        start = lp.lp_count
        kept = dict(kept_parents)
        deleted = []
        n = self.number_of_visible
        shared_facets = {y: {f for f in parents[y] if f >= n} for y in predictors}
        all_shared = set().union(*shared_facets.values()) if shared_facets else set()
        changed = True
        while changed and lp.lp_count - start < max_lps:
            changed = False
            handle = lp.push_hypotheses(self._entropic_hypotheses(predictors, kept, predicted))
            try:
                for t in self.visible_nodes:
                    if t in predictors:
                        continue
                    for p in sorted(kept[t]):
                        # A predicted node must keep a facet shared with a predictor: it carries the node's private
                        # randomness in the lift, otherwise the predictor could not predict it. Deleting the last
                        # shared facet would make the hypotheses (s = f(X1) and s ⊥ X1) contradictory, and the
                        # resulting "certificate" vacuous.
                        if t in predicted and p in all_shared and len(kept[t] & all_shared) == 1:
                            continue
                        if lp.implies(cmi_row([t], [p], kept[t] - {p})):
                            kept[t] = kept[t] - {p}
                            deleted.append((p, t))
                            changed = True
                            break
                    if changed:
                        break
            finally:
                lp.pop_to(handle)
        if not deleted:
            return kept_parents, None, []
        certificate = self._entropic_certificate(predictors, kept, predicted)
        _tally('deletions', certificate or 'fail')
        if certificate is None:
            return kept_parents, None, []
        return kept, certificate, deleted

    def split_node(self, node: int) -> "QmDAG":
        """Node splitting piggyback: `node` is replaced by itself and a copy (appended as the last visible node)
        with the same parents and children and a shared classical two-party latent (their common private noise).
        A QC gap in the split structure implies one in the original (the original node outputs the pair)."""
        n = self.number_of_visible
        copy = n
        edges = set(self.directed_structure_instance.as_set_of_tuples)
        for (a, b) in self.directed_structure_instance.as_set_of_tuples:
            if a == node:
                edges.add((copy, b))
            if b == node:
                edges.add((a, copy))
        C_facets = {f | {copy} if node in f else f for f in self.C_simplicial_complex_instance.simplicial_complex_as_sets}
        C_facets.add(frozenset({node, copy}))
        Q_facets = {f | {copy} if node in f else f for f in self.Q_simplicial_complex_instance.simplicial_complex_as_sets}
        return QmDAG(DirectedStructure(sorted(edges), n + 1), Hypergraph(hypergraph_full_cleanup(C_facets), n + 1),
                     Hypergraph(hypergraph_full_cleanup(Q_facets), n + 1))

    def fritz_entropic_transitions(self, predictors: Iterable[int], modes: Tuple[str, ...] = ('replace', 'copy'),
                                   predictor_modes: Tuple[str, ...] = ('split',), extra_deletions: bool = True,
                                   max_visible: int = None, min_visible: int = 3,
                                   keep_quantum_facets: bool = True, districts_check: bool = False,
                                   allow_childful_predictors: bool = True, apply_teleportation: bool = True,
                                   only_beyond_dsep: bool = True, max_lps: int = 60,
                                   max_lp_variables: int = 11,
                                   base_predictor_modes: Tuple[str, ...] = ('drop', 'split'),
                                   _presplit: bool = False) -> List[Tuple[Tuple, "QmDAG"]]:
        """Fritz transitions certified by the entropic LP.
        Copy mode is realised as node splitting followed by replace mode on the copy (so the LP sees the copy as a
        genuine node with its own shared noise). predictor_mode 'drop' removes the predictors as in fritz_transitions
        (deleted if childless, marginalized otherwise); 'split' splits each predictor into itself and a full copy
        and drops the copies (identical to keeping a childless predictor; sound, unlike keeping a childful one
        untouched, see fritz_transitions). With only_beyond_dsep, outputs in a predictor mode listed in
        base_predictor_modes whose certificate is plain d-separation and that delete nothing extra are skipped:
        by default every emitted step is LP-reliant, and d-separation-certified steps are left to fritz_transitions.
        params: (((s, mode), ...), ('predictor_mode', m), ('certificate', c), ('deleted', ((p, t), ...)))."""
        predictors = frozenset(predictors)
        if max_visible is None:
            max_visible = self.number_of_visible + 1
        childless = predictors.issubset(self.vis_nodes_with_no_children)
        assert childless or allow_childful_predictors, "Fritz predictors must be childless visible nodes."
        childful = predictors.difference(self.vis_nodes_with_no_children)
        if 'split' in predictor_modes and childful and not _presplit:
            # Kept predictors: childless ones stay untouched (sound); a childful one is split into itself and a full
            # copy (same children) and the copy, as predictor, is marginalized. Keeping a childful predictor
            # untouched is unsound (see fritz_transitions). 'drop' outputs are computed on the original structure.
            results = []
            if 'drop' in predictor_modes:
                results += self.fritz_entropic_transitions(
                    predictors, modes=modes, predictor_modes=('drop',), extra_deletions=extra_deletions,
                    max_visible=max_visible, min_visible=min_visible, keep_quantum_facets=keep_quantum_facets,
                    districts_check=districts_check, allow_childful_predictors=allow_childful_predictors,
                    apply_teleportation=apply_teleportation, only_beyond_dsep=only_beyond_dsep, max_lps=max_lps,
                    max_lp_variables=max_lp_variables, base_predictor_modes=base_predictor_modes)
            work, copies = self._split_predictors(childful)
            results += work.fritz_entropic_transitions(
                predictors.difference(childful).union(copies), modes=modes, predictor_modes=('split',),
                extra_deletions=extra_deletions, max_visible=max_visible, min_visible=min_visible,
                keep_quantum_facets=keep_quantum_facets, districts_check=districts_check,
                allow_childful_predictors=True, apply_teleportation=apply_teleportation,
                only_beyond_dsep=only_beyond_dsep, max_lps=max_lps, max_lp_variables=max_lp_variables,
                base_predictor_modes=base_predictor_modes, _presplit=True)
            return results
        if len(self.lp_structure[0]) > max_lp_variables:
            return []
        admissible = self.fritz_entropic_admissible_targets(predictors, allow_childful_predictors)
        targets = sorted(admissible)
        n = self.number_of_visible
        results = []
        admissibility_memo = {self.unique_id: admissible}
        for r in range(1, len(targets) + 1):
            for chosen in itertools.combinations(targets, r):
                for mode_choice in itertools.product(modes, repeat=r):
                    n_copies = mode_choice.count('copy')
                    sizes = {pm: n - (len(predictors) if pm == 'drop' else len(childful)) + n_copies for pm in predictor_modes}
                    if not any(min_visible <= size <= max_visible for size in sizes.values()):
                        continue
                    # Realise copies by node splitting; the predicted nodes in the working structure are the
                    # replace-mode originals and the copies (indices n, n+1, ... in order of splitting).
                    work = self
                    predicted = []
                    labels = {v: v for v in range(n)}
                    for s_node, mode in zip(chosen, mode_choice):
                        if mode == 'copy':
                            work = work.split_node(s_node)
                            predicted.append(work.number_of_visible - 1)
                            labels[predicted[-1]] = str(s_node) + '_copy'
                        else:
                            predicted.append(s_node)
                    predicted = tuple(predicted)
                    if len(work.lp_structure[0]) > max_lp_variables:
                        continue
                    adm_work = admissibility_memo.get(work.unique_id)
                    if adm_work is None:
                        adm_work = work.fritz_entropic_admissible_targets(predictors, allow_childful_predictors)
                        admissibility_memo[work.unique_id] = adm_work
                    if not set(predicted).issubset(adm_work):
                        continue
                    kept = work._fritz_kept_parents(adm_work, {t: 'replace' for t in predicted})
                    certificates = {adm_work[t][2] for t in predicted}
                    if certificates == {'dsep'}:
                        certificate = 'dsep'
                    elif len(predicted) == 1:
                        certificate = adm_work[predicted[0]][2]
                    else:
                        certificate = work._entropic_certificate(predictors, kept, predicted)
                        _tally('joint', certificate or 'fail')
                        if certificate is None:
                            continue
                    deleted = []
                    if extra_deletions:
                        kept2, certificate2, deleted = work._entropic_extra_deletions(predictors, kept, predicted, max_lps)
                        if deleted:
                            kept, certificate = kept2, certificate2
                    facet_label = {idx: 'L' + stringify(members) for idx, (kind, members) in work.effective_DAG_data[1].items()
                                   if kind != 'noise'}
                    def label(v):
                        return labels.get(v, v) if v < work.number_of_visible else facet_label.get(v, v)
                    deleted_labels = tuple((label(p_), label(t_)) for p_, t_ in deleted)
                    choices = {t: 'replace' for t in predicted}
                    for predictor_mode in predictor_modes:
                        if not (min_visible <= sizes[predictor_mode] <= max_visible):
                            continue
                        if only_beyond_dsep and certificate == 'dsep' and not deleted \
                                and predictor_mode in base_predictor_modes:
                            continue   # fritz_transitions already produces this output
                        params = (tuple(zip(chosen, mode_choice)), ('predictor_mode', predictor_mode),
                                  ('certificate', certificate), ('deleted', deleted_labels))
                        intermediate, to_nums = work._fritz_build(predictors, choices, kept,
                                                                  drop_predictors=(predictor_mode == 'drop' and childless),
                                                                  keep_quantum_facets=keep_quantum_facets)
                        to_original = {num: work._fritz_original_of(name) for name, num in to_nums.items()}
                        to_marginalize = predictors if (predictor_mode == 'drop' and not childless) \
                            else (childful if predictor_mode == 'split' else frozenset())
                        if to_marginalize:
                            candidates = [work._marginalize_predictors(intermediate, to_original, order,
                                                                       districts_check=districts_check,
                                                                       apply_teleportation=apply_teleportation)
                                          for order in itertools.permutations(sorted(to_marginalize))]
                        else:
                            candidates = [(intermediate, to_original)]
                        seen_here = set()
                        for new_QmDAG, new_to_original in candidates:
                            if new_QmDAG is None or new_QmDAG.unique_id in seen_here:
                                continue
                            if not (min_visible <= new_QmDAG.number_of_visible <= max_visible):
                                continue
                            removed = predictors if predictor_mode == 'drop' else frozenset()
                            if districts_check and not work._fritz_preserves_districts(removed, new_QmDAG, new_to_original):
                                continue
                            seen_here.add(new_QmDAG.unique_id)
                            results.append((params, new_QmDAG))
        return results

    # ------------------------------------------------------------------
    # COMPOSITION OF PIGGYBACKS
    # ------------------------------------------------------------------

    def piggyback_children(self, max_visible: int, min_visible: int = 3, districts_check: bool = False,
                           apply_teleportation: bool = True, include_Fritz: bool = True,
                           keep_quantum_facets: bool = True, allow_childful_predictors: bool = True,
                           max_predictors: int = 2, predictor_mode: str = 'drop',
                           strict_conditioning: bool = True) -> Iterable["QmDAG"]:
        """One application of every piggyback (PD, conditioning, marginalization, interruption, Fritz)."""
        n = self.number_of_visible
        if n > min_visible:
            yield from self.subgraphs
            for node in self.visible_nodes:
                if self.conditioning_is_justified(node, strict_latents=strict_conditioning):
                    yield self.condition(node)
                marginalized = self.marginalize(node, districts_check=districts_check,
                                                apply_teleportation=apply_teleportation)
                if marginalized is not None:
                    yield marginalized
        if n > min_visible:
            yield from self.subinterruptions
        if include_Fritz:
            predictor_pool = [y for y in self.visible_nodes
                              if self.latent_siblings_of(y) and (allow_childful_predictors or y in self.vis_nodes_with_no_children)]
            for r in range(1, min(max_predictors, len(predictor_pool)) + 1):
                for predictors in itertools.combinations(predictor_pool, r):
                    for params, new_QmDAG in self.fritz_transitions(predictors, max_visible=max_visible,
                                                                    min_visible=min_visible,
                                                                    keep_quantum_facets=keep_quantum_facets,
                                                                    districts_check=districts_check,
                                                                    allow_childful_predictors=allow_childful_predictors,
                                                                    apply_teleportation=apply_teleportation,
                                                                    predictor_mode=predictor_mode):
                        yield new_QmDAG

    def piggyback_closure(self, max_visible: int = None, min_visible: int = 3, max_states: int = 50000,
                          **kwargs) -> Dict[Tuple[int, int, int, int], "QmDAG"]:
        """Every structure reachable by composing piggybacks in any order, keyed by unlabelled id.
        Each unlabelled id is expanded once (all tricks are label-equivariant)."""
        if max_visible is None:
            max_visible = self.number_of_visible + 1
        reached = {self.unique_unlabelled_id: self}
        frontier = [self]
        options = (max_visible, min_visible, tuple(sorted(kwargs.items())))
        while frontier:
            current = frontier.pop()
            memo_key = (current.unique_unlabelled_id,) + options
            try:
                children = _PIGGYBACK_CHILDREN_MEMO[memo_key]
            except KeyError:
                children = list({child.unique_unlabelled_id: child for child in
                                 current.piggyback_children(max_visible=max_visible, min_visible=min_visible, **kwargs)
                                 if min_visible <= child.number_of_visible <= max_visible}.values())
                _PIGGYBACK_CHILDREN_MEMO[memo_key] = children
            for child in children:
                child_id = child.unique_unlabelled_id
                if child_id not in reached:
                    reached[child_id] = child
                    frontier.append(child)
                    if len(reached) > max_states:
                        warnings.warn("Piggyback closure exceeded max_states; the result is incomplete. "
                                      "Raise max_states or lower max_visible.")
                        return reached
        return reached

    @lru_cache(maxsize=None)
    def unique_unlabelled_ids_obtainable_by_Fritz_for_QC(self, max_visible: int = None,
                                                         keep_quantum_facets: bool = True,
                                                         allow_childful_predictors: bool = True,
                                                         max_predictors: int = 2) -> Set[Tuple[int, int, int, int]]:
        """Unlabelled ids reachable from self by any composition of the piggybacks including Fritz (self excluded)."""
        reached = self.piggyback_closure(max_visible=max_visible, districts_check=False, apply_teleportation=True,
                                         include_Fritz=True, keep_quantum_facets=keep_quantum_facets,
                                         allow_childful_predictors=allow_childful_predictors,
                                         max_predictors=max_predictors)
        return set(reached).difference({self.unique_unlabelled_id})

    @lru_cache(maxsize=None)
    def unique_unlabelled_ids_obtainable_by_Fritz_for_IC(self, max_visible: int = None,
                                                         keep_quantum_facets: bool = True,
                                                         allow_childful_predictors: bool = True,
                                                         max_predictors: int = 2) -> Set[Tuple[int, int, int, int]]:
        reached = self.piggyback_closure(max_visible=max_visible, districts_check=True, apply_teleportation=False,
                                         include_Fritz=True, keep_quantum_facets=keep_quantum_facets,
                                         allow_childful_predictors=allow_childful_predictors,
                                         max_predictors=max_predictors)
        return set(reached).difference({self.unique_unlabelled_id})


if __name__ == '__main__':
    ghost = QmDAG(DirectedStructure([(1, 2), (1, 3)], 4), Hypergraph([], 4), Hypergraph([(0, 2), (0, 3)], 4))
    print("All graphs obtainable from the Ghost by Interruption (should be Evans)")
    print(ghost.subinterruptions)
    print("Now assessing Fritz trick on the triangle (should reach Bell):")
    triangle = QmDAG(DirectedStructure([], 3), Hypergraph([], 3), Hypergraph([(0, 1), (1, 2), (0, 2)], 3))
    for params, post_Fritz in triangle.fritz_transitions((2,)):
        print(params)
        print(post_Fritz)
