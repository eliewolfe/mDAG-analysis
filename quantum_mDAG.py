from __future__ import absolute_import
import itertools
import numpy as np
import numpy.typing as npt
# import networkx as nx
from hypergraphs import Hypergraph, LabelledHypergraph, hypergraph_full_cleanup
from directed_structures import DirectedStructure, LabelledDirectedStructure
# from radix import to_bits  # TODO: Make qmdaq from representation
from mDAG_advanced import mDAG
from merge import merge_intersection
from sys import version_info
assert version_info >= (3, 8), "Python 3.8+ is required for cached_property support."
from utilities import partsextractor, minimal_sets_within, maximal_sets_within, stringify_in_set, stringify_in_tuple
from functools import total_ordering
from typing import Any, DefaultDict, Dict, Iterable, List, Set, Tuple
try:
    import networkx as nx
except ImportError:
    print("Functions which depend on networkx are not available.")

from functools import cached_property
from collections import defaultdict
from methodtools import lru_cache

BoolMatrix = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int_]


def invert_dict(d: Dict[Any, Any]) -> DefaultDict[Any, List[Any]]:
    d_inv: DefaultDict[Any, List[Any]] = defaultdict(list)
    for k, v in d.items():
        d_inv[v].append(k)
    return d_inv


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
        self.Fritz_trick_has_been_applied_already = False
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


    def labelled_multi_marginalize(self,
                                   nodes_to_marginalize: Iterable[Any],
                                   all_nodes: Iterable[Any],
                                   directed_structure_list: List[Tuple[Any, Any]],
                                   C_simplicial_complex_instance_as_sets: Iterable[frozenset],
                                   Q_simplicial_complex_instance_as_sets: Iterable[frozenset],
                                   districts_check: bool = False) -> "QmDAG":  # returns a smaller QmDAG
        new_directed_structure_set = set(directed_structure_list).copy()
        new_C_simplicial_complex_instance_as_sets = set(C_simplicial_complex_instance_as_sets).copy()
        new_Q_simplicial_complex_instance_as_sets = set(Q_simplicial_complex_instance_as_sets).copy()
        remaining_nodes = set(all_nodes)

        for node in set(nodes_to_marginalize):
            remaining_nodes.discard(node)
            visible_children = set()
            visible_parents = set()
            for i in remaining_nodes:
                if (node, i) in new_directed_structure_set:
                    visible_children.add(i)
                    new_directed_structure_set.remove( (node, i) )
                if (i, node) in new_directed_structure_set:
                    visible_parents.add(i)
                    new_directed_structure_set.remove( (i, node))
            for parent in visible_parents:
                for child in visible_children:
                    new_directed_structure_set.add((parent, child))

            new_C_simplicial_complex_instance_as_sets.add(frozenset(visible_children))
            facets_to_kill = set()
            facets_to_add = set()
            for facet in new_C_simplicial_complex_instance_as_sets:
                if node in facet:
                    marginalized_facet = facet.difference({node}).union(visible_children)
                    facets_to_kill.add(facet)
                    facets_to_add.add(marginalized_facet)
            new_C_simplicial_complex_instance_as_sets.update(facets_to_add)
            new_C_simplicial_complex_instance_as_sets.difference_update(facets_to_kill)
            teleportable_children = set()
            facets_to_expand_by_teleportation = set()
            facets_to_kill = set()
            for facet in new_Q_simplicial_complex_instance_as_sets:
                if node in facet:
                    sub_qfacet = frozenset(facet).difference({node})
                    facets_to_expand_by_teleportation.add(sub_qfacet)
                    classical_marginalized_facet = sub_qfacet.union(visible_children)
                    teleportable_children.update(sub_qfacet.intersection(visible_children))
                    facets_to_kill.add(facet)
                    new_C_simplicial_complex_instance_as_sets.add(classical_marginalized_facet)
            new_Q_simplicial_complex_instance_as_sets.difference_update(facets_to_kill)
            for facet in facets_to_expand_by_teleportation:
                new_Q_simplicial_complex_instance_as_sets.add(facet.union(teleportable_children))
        remaining_nodes = tuple(remaining_nodes)
        new_C_simplicial_complex_instance_as_sets = hypergraph_full_cleanup(new_C_simplicial_complex_instance_as_sets)
        new_Q_simplicial_complex_instance_as_sets = hypergraph_full_cleanup(new_Q_simplicial_complex_instance_as_sets)
        if not districts_check:
            ok_to_proceed = True
        else:
            old_districts = merge_intersection(C_simplicial_complex_instance_as_sets.union(Q_simplicial_complex_instance_as_sets))
            new_districts = merge_intersection(new_C_simplicial_complex_instance_as_sets.union(new_Q_simplicial_complex_instance_as_sets))
            old_districts = [district.difference(nodes_to_marginalize) for district in old_districts]
            ok_to_proceed = frozenset(map(frozenset, new_districts)) == frozenset(map(frozenset, old_districts))
        if ok_to_proceed:
            return QmDAG(
                LabelledDirectedStructure(remaining_nodes, list(new_directed_structure_set)),
                LabelledHypergraph(remaining_nodes, new_C_simplicial_complex_instance_as_sets),
                LabelledHypergraph(remaining_nodes, new_Q_simplicial_complex_instance_as_sets)
            )
        else:
            return None  # the marginalization trick does not apply when districts are not preserved

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

    def _yield_from_Fritz_trick(self, choice_of_nodes,
                                new_directed_structure, new_C_simplicial_complex, new_Q_simplicial_complex,
                                nodes_relevant_for_pp, pprestrictions_if_present,
                                safe_for_inference=True, districts_check=False):
        nodes_to_marginalize_away = set(
            itertools.chain.from_iterable((nodes_relevant_for_pp[i] for i in choice_of_nodes)))
        if nodes_to_marginalize_away.issubset(choice_of_nodes):
            if safe_for_inference:
                coreQmDAG = self.labelled_multi_marginalize(
                    nodes_to_marginalize_away,
                    choice_of_nodes,
                    new_directed_structure,
                    new_C_simplicial_complex,
                    new_Q_simplicial_complex,
                    districts_check=districts_check)
                if coreQmDAG is None:
                    return None
                coreQmDAG.Fritz_trick_has_been_applied_already = True
                return coreQmDAG
            else:
                new_ds = LabelledDirectedStructure(choice_of_nodes, new_directed_structure)
                to_nums = new_ds.translation_dict
                pp_flat = list(itertools.chain.from_iterable((pprestrictions_if_present[i] for i in choice_of_nodes)))
                pp_flat_numeric = tuple(((to_nums[i], tuple(partsextractor(to_nums, j))) for i, j in pp_flat))
                coreQmDAG = QmDAG(
                    new_ds,
                    LabelledHypergraph(choice_of_nodes, new_C_simplicial_complex),
                    LabelledHypergraph(choice_of_nodes, new_Q_simplicial_complex),
                    pp_restrictions=pp_flat_numeric)
                coreQmDAG.restricted_perfect_predictions = pp_flat
                coreQmDAG.Fritz_trick_has_been_applied_already = True
                return coreQmDAG


    @lru_cache(maxsize=None)
    def apply_Fritz_trick(self, node_decomposition=True, safe_for_inference=True, districts_check=False, Sofia_extra=True):
        """Returns the frozenset of QmDAGs obtainable by one application of the Fritz trick."""
        return frozenset(filter(None, self._iter_Fritz_trick(node_decomposition=node_decomposition,
                                                             safe_for_inference=safe_for_inference,
                                                             districts_check=districts_check,
                                                             Sofia_extra=Sofia_extra)))

    def _iter_Fritz_trick(self, node_decomposition=True, safe_for_inference=True, districts_check=False, Sofia_extra=True):
        if not self.Fritz_trick_has_been_applied_already:
            expanded_edge_set = self.directed_structure_instance.as_set_of_tuples.copy()
            #Note that we use the expanded classical simplicial complex to ensure common cause in node decomposition.
            for i, children in zip(self.classical_latent_nodes, self.C_simplicial_complex_instance.extended_simplicial_complex_as_sets):
                expanded_edge_set.update(zip(itertools.repeat(i), children))
            # print("2. Nodes are: ", set(itertools.chain.from_iterable(expanded_edge_set)))
            for i, children in zip(self.quantum_latent_nodes, self.Q_simplicial_complex_instance.compressed_simplicial_complex):
                expanded_edge_set.update(zip(itertools.repeat(i), children))
            # num_quantum_nodes = self.Q_simplicial_complex_instance.number_of_nonsingleton_latent
            num_effective_nodes = self.Q_simplicial_complex_instance.number_of_visible_plus_nonsingleton_latent\
                                  + self.C_simplicial_complex_instance.number_of_visible_plus_latent\
                                  - self.number_of_visible
            assert all(isinstance(v, int) for v in set(itertools.chain.from_iterable(expanded_edge_set))), 'Somehow we have a non integer node!'
            effective_DAG = DirectedStructure(expanded_edge_set, num_effective_nodes)
            effective_nx_DAG = effective_DAG.as_networkx_graph
            # expanded_edge_set_of_tuples_of_strings = set([(str(i), str(j)) for (i,j) in expanded_edge_set])
            #We will make as subvariables as classical-common-cause connected only, so all quantum facets must be duplicated.
            #One for original vars, one for subvars.
            # for i, children in zip(self.quantum_latent_nodes, self.Q_simplicial_complex_instance.compressed_simplicial_complex):
            #     expanded_edge_set_of_tuples_of_strings.update(zip(itertools.repeat(str(i+num_quantum_nodes)), map(str,children)))
            common_cause_connected_sets = maximal_sets_within(effective_DAG.adjMat.descendantsplus_list)
            allnode_name_variants = dict()
            pprestrictions_if_present = dict()
            nodes_relevant_for_pp = dict()
            kept_parents_dict = dict()
            for target in self.visible_nodes:
                allnode_name_variants[target] = {target}
                pprestrictions_if_present[target] = set()
                nodes_relevant_for_pp[target] = set()
                effective_target_parents = effective_DAG.adjMat.parents_of(target)
                kept_parents_dict[target] = effective_target_parents
            for target in self.visible_nodes:
                effective_target_parents = set(kept_parents_dict[target].tolist())
                target_children = self.directed_structure_instance.adjMat.children_of(target)
                candidates_Yi = self.latent_siblings_of(target).union(self.directed_structure_instance.adjMat.parents_of(target))
                ### OLD CODE
                # singleton_edge_removals = {Yi: frozenset([v for v in effective_target_parents if not
                #                         any({v, Yi}.issubset(common_cause_connected_set) for common_cause_connected_set in common_cause_connected_sets)])
                #                            for Yi in candidates_Yi}
                ### Marina and TC's version which hold classically
                singleton_edge_removals = {Yi: frozenset([v for v in effective_target_parents.difference({Yi}) if
                                                          nx.is_d_separator(effective_nx_DAG, {Yi}, {v},
                                                                         effective_target_parents.difference({Yi,v}))])
                                           for Yi in candidates_Yi}

                collective_predicting_set_edge_removals = dict()
                for r in range(1, len(candidates_Yi) + 1):
                    for collective_predicting in map(frozenset, itertools.combinations(candidates_Yi, r)):
                        individual_edge_set_removals = [singleton_edge_removals[Yi] for Yi in
                                            collective_predicting]
                        collective_edge_removals = frozenset.union(*individual_edge_set_removals)
                        if len(collective_edge_removals)>=1:
                            collective_predicting_set_edge_removals[collective_predicting] = collective_edge_removals
                for (removed_edges, perfectly_predicting_sets) in invert_dict(collective_predicting_set_edge_removals).items():
                    minimal_pp_sets = minimal_sets_within(perfectly_predicting_sets)
                    for wasteful_pp_set in set(perfectly_predicting_sets).difference(minimal_pp_sets):
                        del collective_predicting_set_edge_removals[wasteful_pp_set]
                independently_predicting_set_edge_removals = dict()
                # independently_predicting_sets = set()
                collective_predicting_sets = collective_predicting_set_edge_removals.keys()
                max_r = len(collective_predicting_sets) + 1
                if not Sofia_extra:
                    max_r = 2
                for r in range(1, max_r):
                    for independently_predicting in map(frozenset, itertools.combinations(collective_predicting_sets, r)):
                        # independently_predicting_sets.add(independently_predicting)
                        individual_edge_set_removals = [collective_predicting_set_edge_removals[collective_predicting] for collective_predicting in
                                            independently_predicting]
                        independently_predicting_set_edge_removals[independently_predicting] = frozenset.union(*individual_edge_set_removals)
                for (removed_parents, perfectly_predicting_sets) in invert_dict(independently_predicting_set_edge_removals).items():
                    minimal_pp_sets = minimal_sets_within(perfectly_predicting_sets)
                    for wasteful_pp_set in set(perfectly_predicting_sets).difference(minimal_pp_sets):
                        del independently_predicting_set_edge_removals[wasteful_pp_set]
                # independently_predicting_sets = independently_predicting_set_edge_removals.keys()
                for minimal_pp_set, removed_parents in independently_predicting_set_edge_removals.items():
                    subtarget = str(target)+'_'+stringify_in_tuple(map(stringify_in_set, minimal_pp_set))
                    allnode_name_variants[target].add(subtarget)
                    pprestrictions_if_present[subtarget] = list(zip(itertools.repeat(subtarget), map(tuple, minimal_pp_set)))
                    nodes_relevant_for_pp[subtarget] = tuple(set(itertools.chain.from_iterable(minimal_pp_set)))
                    kept_parents = effective_target_parents.difference(removed_parents)
                    kept_parents_dict[subtarget] = kept_parents
                    for p in kept_parents:
                        if p in self.visible_nodes:
                            for p_variant in allnode_name_variants[p]:
                                expanded_edge_set.add((p_variant, subtarget))
                        else:
                            expanded_edge_set.add((p, subtarget))
                    for c in target_children:
                        if c in self.visible_nodes:
                            for c_variant in allnode_name_variants[c]:
                                if target in kept_parents_dict[c_variant]:
                                    expanded_edge_set.add((subtarget, c_variant))
                        else:
                            expanded_edge_set.add((subtarget, c))
            code_for_classical_latents = self.classical_latent_nodes + self.quantum_latent_nodes
            new_nodes = set(itertools.chain.from_iterable(allnode_name_variants.values()))
            new_directed_structure = [(i,j) for (i,j) in expanded_edge_set if i not in code_for_classical_latents]
            # print("New ds: ", new_directed_structure)
            new_C_simplicial_complex = [set([j for j in new_nodes if (i,j) in expanded_edge_set]) for i in code_for_classical_latents]
            new_C_simplicial_complex = hypergraph_full_cleanup(new_C_simplicial_complex)
            # print("New sc: ", new_C_simplicial_complex)
            new_Q_simplicial_complex = self.Q_simplicial_complex_instance.compressed_simplicial_complex.copy()
            # print("New qsc: ", new_Q_simplicial_complex)
            if not node_decomposition:
                for choice_of_nodes in itertools.product(*allnode_name_variants.values()):
                    if not set(choice_of_nodes).issubset(self.visible_nodes):
                        # print("Chosen nodes to explore:", choice_of_nodes)
                        nodes_to_marginalize_away = set(
                            itertools.chain.from_iterable((nodes_relevant_for_pp[i] for i in choice_of_nodes)))
                        if nodes_to_marginalize_away.issubset(choice_of_nodes):
                            yield self._yield_from_Fritz_trick(choice_of_nodes,
                                                    new_directed_structure, new_C_simplicial_complex, new_Q_simplicial_complex,
                                                    nodes_relevant_for_pp, pprestrictions_if_present,
                                                               safe_for_inference=safe_for_inference,
                                                               districts_check=districts_check)
            else:
                bonus_node_variants = [name_variants.difference(self.visible_nodes) for name_variants in allnode_name_variants.values() if
                                       len(name_variants) >= 2]
                bonus_node_variants = [name_variants.union({'-1'}) for name_variants in bonus_node_variants]
                for bonus_nodes in itertools.product(*bonus_node_variants):
                    actual_bonus_nodes = set(bonus_nodes).difference({'-1'})
                    choice_of_nodes = tuple(self.visible_nodes) + tuple(actual_bonus_nodes)
                    nodes_to_marginalize_away = set(
                        itertools.chain.from_iterable((nodes_relevant_for_pp[i] for i in choice_of_nodes)))
                    if nodes_to_marginalize_away.issubset(choice_of_nodes):
                        yield self._yield_from_Fritz_trick(choice_of_nodes,
                                                           new_directed_structure, new_C_simplicial_complex,
                                                           new_Q_simplicial_complex,
                                                           nodes_relevant_for_pp, pprestrictions_if_present,
                                                           safe_for_inference=safe_for_inference,
                                                           districts_check=districts_check)



    def _unique_unlabelled_ids_obtainable_by_Fritz_for_QC(self, **kwargs):
        for new_QmDAG in self.apply_Fritz_trick(**kwargs):
            yield new_QmDAG.unique_unlabelled_id
            for unlabelled_id in new_QmDAG.unique_unlabelled_ids_obtainable_by_reduction(districts_check=False, apply_teleportation=True):
                yield unlabelled_id
    
    @lru_cache(maxsize=None)
    def unique_unlabelled_ids_obtainable_by_Fritz_for_QC(self, **kwargs):
        return set(self._unique_unlabelled_ids_obtainable_by_Fritz_for_QC(**kwargs))

    def _unique_unlabelled_ids_obtainable_by_Fritz_for_IC(self, **kwargs):
        for new_QmDAG in self.apply_Fritz_trick(districts_check=True, **kwargs):
            yield new_QmDAG.unique_unlabelled_id
            for unlabelled_id in new_QmDAG.unique_unlabelled_ids_obtainable_by_reduction(districts_check=True, apply_teleportation=False):
                yield unlabelled_id
                    
    def unique_unlabelled_ids_obtainable_by_Fritz_for_IC(self, **kwargs):
        return set(self._unique_unlabelled_ids_obtainable_by_Fritz_for_IC(**kwargs))

if __name__ == '__main__':
    ghost = QmDAG(DirectedStructure([(1, 2), (1, 3)], 4), Hypergraph([], 4), Hypergraph([(0, 2), (0, 3)], 4))
    print("All graphs obtainable from the Ghost by Interruption (should be Evans)")
    print(ghost.subinterruptions)
    print("Now assessing Fritz trick...")
    Q1 = QmDAG(DirectedStructure([(0, 1), (1, 2), (2, 3)], 4), Hypergraph([], 4),
               Hypergraph([(0, 1), (0, 2), (0, 3), (1, 2, 3)], 4))
    post_Fritz_set = Q1.apply_Fritz_trick(node_decomposition=False, districts_check=True, safe_for_inference=True)
    print(post_Fritz_set)
    print([post_Fritz_qmDAG.number_of_visible for post_Fritz_qmDAG in post_Fritz_set])
