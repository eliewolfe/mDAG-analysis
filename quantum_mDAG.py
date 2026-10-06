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
    
    def condition(self, node: int) -> "QmDAG":
        #assume we already checked that it doesn't have grandparents that are not parents
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
                if not self.has_grandparents_that_are_not_parents(node):
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
    # Let X1 be a set of childless visible nodes (the predictors) and s a visible node sharing a latent with some
    # member of X1 (a candidate predicted node). In the effective DAG (visible nodes, one node per latent facet, one
    # private-noise node per visible node) split the parents of s into common(s), those also seen by X1 (parents of
    # some predictor, or predictors themselves), and others(s). If X1 is d-separated from others(s) given common(s),
    # then:
    #   * classically, any model in which X1 perfectly predicts s can be rewritten so that s depends on common(s) only;
    #   * quantumly, any strategy for the reduced structure in which s is a deterministic function of its (classical)
    #     parents extends to the original structure with X1 outputting a copy of s.
    # Hence the structure G' obtained by deleting X1 and restricting s to common(s) satisfies: a QC gap in G' implies a
    # QC gap in G, with no caveat about perfect correlations in G'. Quantum facets read by s become classical for s.
    # In "copy" mode s is left untouched and a fresh node s_copy carrying the common (classical) part is added; this is
    # the node-splitting version of the trick (e.g. triangle -> Bell).
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

    def fritz_admissible_targets(self, predictors: Iterable[int]) -> Dict[int, Tuple[frozenset, frozenset]]:
        """Maps each admissible predicted node s to (common(s), others(s)) in effective-DAG indices."""
        predictors = frozenset(predictors)
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
            # Predictors are childless, so none of them is a parent of s and `others` always contains s's noise node.
            if nx.is_d_separator(g, predictors, others, common):
                admissible[s] = (frozenset(common), frozenset(others))
        return admissible

    def _fritz_build(self, predictors: frozenset, choices: Dict[int, str],
                     admissible: Dict[int, Tuple[frozenset, frozenset]],
                     drop_predictors: bool = True, keep_quantum_facets: bool = False) -> Tuple["QmDAG", Dict[Any, int]]:
        """Builds the post-Fritz QmDAG for the given mode per predicted node ('replace' or 'copy').
        Returns the QmDAG and the name -> index translation."""
        g, latent_nodes = self.effective_DAG_data
        copies = {s: str(s) + '_copy' for s, mode in choices.items() if mode == 'copy'}
        kept_originals = [v for v in self.visible_nodes if drop_predictors is False or v not in predictors]
        names = tuple(kept_originals) + tuple(copies[s] for s in sorted(copies))
        name_set = set(names)

        edges = set()
        for (a, b) in self.directed_structure_instance.as_set_of_tuples:
            if b in choices:
                common = admissible[b][0]
                if choices[b] == 'copy':
                    edges.add((a, b))
                    if a in common:
                        edges.add((a, copies[b]))
                elif a in common:
                    edges.add((a, b))
            else:
                edges.add((a, b))
        # A copy is a sub-output of s, so it feeds exactly the children that still see s.
        for s, s_copy in copies.items():
            for c in self.directed_structure_instance.adjMat.children_of(s):
                if c in choices:
                    if choices[c] == 'copy':
                        edges.add((s_copy, c))
                    if s in admissible[c][0]:
                        edges.add((s_copy, c if choices[c] == 'replace' else copies[c]))
                else:
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
                if v in choices:
                    is_common = idx in admissible[v][0]
                    if choices[v] == 'copy':
                        quantum_readers.add(v)
                        if is_common:
                            classical_readers.add(copies[v])
                    elif is_common:
                        classical_readers.add(v)
                        replaced_reader_present = True
                else:
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
                          keep_quantum_facets: bool = False, districts_check: bool = False) -> List[Tuple[Tuple[Tuple[int, str], ...], "QmDAG"]]:
        """All structures obtainable by the Fritz piggyback with the given childless predictors.
        Returns (params, QmDAG) pairs where params = ((s, mode), ...) sorted by s."""
        predictors = frozenset(predictors)
        if max_visible is None:
            max_visible = self.number_of_visible + 1
        admissible = self.fritz_admissible_targets(predictors)
        targets = sorted(admissible)
        results = []
        for r in range(1, len(targets) + 1):
            for chosen in itertools.combinations(targets, r):
                for mode_choice in itertools.product(modes, repeat=r):
                    new_size = self.number_of_visible - len(predictors) + mode_choice.count('copy')
                    if not (min_visible <= new_size <= max_visible):
                        continue
                    params = tuple(zip(chosen, mode_choice))
                    new_QmDAG, to_nums = self._fritz_build(predictors, dict(params), admissible,
                                                           drop_predictors=True, keep_quantum_facets=keep_quantum_facets)
                    if districts_check and not self._fritz_preserves_districts(predictors, dict(params), new_QmDAG, to_nums):
                        continue
                    results.append((params, new_QmDAG))
        return results

    def _fritz_preserves_districts(self, predictors: frozenset, choices: Dict[int, str], new_QmDAG: "QmDAG",
                                   to_nums: Dict[Any, int]) -> bool:
        """Districts of the output (copies identified with their originals) equal the old districts minus predictors."""
        old_districts = set(frozenset(d.difference(predictors)) for d in self.as_mDAG.numerical_districts)
        old_districts.discard(frozenset())
        to_original = dict()
        for name, num in to_nums.items():
            to_original[num] = int(str(name).split('_copy')[0]) if isinstance(name, str) else name
        new_districts = set(frozenset(to_original[v] for v in d) for d in new_QmDAG.as_mDAG.numerical_districts)
        return old_districts == new_districts

    def fritz_intermediate_with_pp(self, predictors: Iterable[int], choices: Dict[int, str],
                                   keep_quantum_facets: bool = False) -> "QmDAG":
        """The Fritz-reduced structure with the predictors retained, carrying perfect-prediction restrictions
        (each predicted node, or its copy, is a function of the predictors) for supports-based inference."""
        predictors = frozenset(predictors)
        admissible = self.fritz_admissible_targets(predictors)
        assert set(choices).issubset(admissible), "Some chosen node is not an admissible Fritz target."
        new_QmDAG, to_nums = self._fritz_build(predictors, choices, admissible,
                                               drop_predictors=False, keep_quantum_facets=keep_quantum_facets)
        predictor_nums = tuple(sorted(to_nums[y] for y in predictors))
        pp = []
        for s, mode in sorted(choices.items()):
            predicted = to_nums[str(s) + '_copy'] if mode == 'copy' else to_nums[s]
            pp.append((predicted, predictor_nums))
        return QmDAG(new_QmDAG.directed_structure_instance, new_QmDAG.C_simplicial_complex_instance,
                     new_QmDAG.Q_simplicial_complex_instance, pp_restrictions=tuple(pp))

    # ------------------------------------------------------------------
    # COMPOSITION OF PIGGYBACKS
    # ------------------------------------------------------------------

    def piggyback_children(self, max_visible: int, min_visible: int = 3, districts_check: bool = False,
                           apply_teleportation: bool = True, include_Fritz: bool = True,
                           keep_quantum_facets: bool = False) -> Iterable["QmDAG"]:
        """One application of every piggyback (PD, conditioning, marginalization, interruption, Fritz)."""
        n = self.number_of_visible
        if n > min_visible:
            yield from self.subgraphs
            for node in self.visible_nodes:
                if not self.has_grandparents_that_are_not_parents(node):
                    yield self.condition(node)
                marginalized = self.marginalize(node, districts_check=districts_check,
                                                apply_teleportation=apply_teleportation)
                if marginalized is not None:
                    yield marginalized
        if n > min_visible:
            yield from self.subinterruptions
        if include_Fritz:
            for y in sorted(self.vis_nodes_with_no_children):
                for params, new_QmDAG in self.fritz_transitions((y,), max_visible=max_visible, min_visible=min_visible,
                                                                keep_quantum_facets=keep_quantum_facets,
                                                                districts_check=districts_check):
                    yield new_QmDAG

    def piggyback_closure(self, max_visible: int = None, min_visible: int = 3, max_states: int = 50000,
                          **kwargs) -> Dict[Tuple[int, int, int, int], "QmDAG"]:
        """Every structure reachable by composing piggybacks in any order, keyed by unlabelled id.
        Each unlabelled id is expanded once (all tricks are label-equivariant)."""
        if max_visible is None:
            max_visible = self.number_of_visible + 1
        reached = {self.unique_unlabelled_id: self}
        frontier = [self]
        while frontier:
            current = frontier.pop()
            for child in current.piggyback_children(max_visible=max_visible, min_visible=min_visible, **kwargs):
                if not (min_visible <= child.number_of_visible <= max_visible):
                    continue
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
                                                         keep_quantum_facets: bool = False) -> Set[Tuple[int, int, int, int]]:
        """Unlabelled ids reachable from self by any composition of the piggybacks including Fritz (self excluded)."""
        reached = self.piggyback_closure(max_visible=max_visible, districts_check=False, apply_teleportation=True,
                                         include_Fritz=True, keep_quantum_facets=keep_quantum_facets)
        return set(reached).difference({self.unique_unlabelled_id})

    @lru_cache(maxsize=None)
    def unique_unlabelled_ids_obtainable_by_Fritz_for_IC(self, max_visible: int = None,
                                                         keep_quantum_facets: bool = False) -> Set[Tuple[int, int, int, int]]:
        reached = self.piggyback_closure(max_visible=max_visible, districts_check=True, apply_teleportation=False,
                                         include_Fritz=True, keep_quantum_facets=keep_quantum_facets)
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
