from __future__ import absolute_import
import numpy as np
import numpy.typing as npt
import itertools
# import networkx as nx
from sys import version_info
assert version_info >= (3, 8), "Python 3.8+ is required for cached_property support."
from typing import Tuple, List, Set, FrozenSet, Iterable, Any, Union, Optional

import functools
from functools import cached_property

BoolMatrix = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int_]

from radix import bitarray_to_int
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Visualization with matplotlib is disabled.")
try:
    import networkx as nx
except ImportError:
    print("Functions which depend on networkx are not available.")


from utilities import partsextractor
from collections import defaultdict
try:
    from supports_beyond_esep import SmartSupportTesting
except:
    print("Testing infeasible supports requires pysat.")
from hypergraphs import Hypergraph, LabelledHypergraph, UndirectedGraph, LabelledUndirectedGraph, hypergraph_full_cleanup
import methodtools
from directed_structures import LabelledDirectedStructure, DirectedStructure
from closure import closure as numeric_closure  # , is_this_subadjmat_densely_connected
from more_itertools import powerset
from merge import merge_intersection
from adjmat_class import AdjMat
import semigraphoid as sg

 
def mdag_to_int_pair(ds_bitarray: BoolMatrix, sc_bitarray: BoolMatrix) -> Tuple[int, int]:
    sc_int = bitarray_to_int(sc_bitarray)
    ds_int = bitarray_to_int(ds_bitarray)
    return (ds_int, sc_int)

def mdag_to_int(ds_bitarray: BoolMatrix, sc_bitarray: BoolMatrix) -> int:
    sc_int = bitarray_to_int(sc_bitarray)
    ds_int = bitarray_to_int(ds_bitarray)
    #The directed structure is always a square matrix of size n^2.
    offset = 2**ds_bitarray.size
    return ds_int + (sc_int * offset)



@functools.lru_cache(maxsize=None)
def _bipartition_patterns(length: int) -> Tuple[Tuple[int, ...], ...]:
    """Index sets of the second part of every bipartition of `length` items, in _all_bipartitions order."""
    return tuple(itertools.chain.from_iterable(
        itertools.combinations(range(length), r) for r in range(1, length)))


class mDAG:
    def __init__(self, directed_structure_instance: DirectedStructure, simplicial_complex_instance: Hypergraph, pp_restrictions: Tuple[int, ...] = tuple()) -> None:
        self.restricted_perfect_predictions_numeric = pp_restrictions
        self.directed_structure_instance = directed_structure_instance
        self.simplicial_complex_instance = simplicial_complex_instance
        assert directed_structure_instance.number_of_visible == simplicial_complex_instance.number_of_visible, 'Different number of nodes in directed structure vs simplicial complex.'
        if hasattr(self.directed_structure_instance, 'variable_names'):
            self.variable_names = self.directed_structure_instance.variable_names
            if hasattr(self.simplicial_complex_instance, 'variable_names'):
                assert frozenset(self.variable_names) == frozenset(self.simplicial_complex_instance.variable_names), 'Error: Inconsistent node names.'
                if not tuple(self.variable_names) == tuple(self.simplicial_complex_instance.variable_names):
                    print('Warning: Inconsistent node ordering. Following ordering of directed structure!')
        self.number_of_visible = self.directed_structure_instance.number_of_visible

    # The attributes below are read off the two parts on demand rather than stored: an mDAG then holds three
    # references and an integer (a few hundred bytes), which is what lets the five-node metagraph keep its 1.7
    # million nodes in memory.
    _CORE_ATTRIBUTES = frozenset(('restricted_perfect_predictions_numeric', 'directed_structure_instance',
                                  'simplicial_complex_instance', 'number_of_visible', 'variable_names'))

    @property
    def visible_nodes(self) -> List[int]:
        return self.directed_structure_instance.visible_nodes

    @property
    def latent_nodes(self) -> Tuple[int, ...]:
        return tuple(range(self.number_of_visible, self.simplicial_complex_instance.number_of_visible_plus_latent))

    @property
    def nof_nonsingleton_latents(self) -> int:
        return self.simplicial_complex_instance.number_of_nonsingleton_latent

    @property
    def total_number_of_nodes(self) -> int:
        return self.nof_nonsingleton_latents + self.number_of_visible

    @property
    def vis_nodes_with_no_children(self) -> Set[int]:
        return set(self.directed_structure_instance.nodes_with_no_children)

    @property
    def vis_nodes_with_no_parents(self) -> Set[int]:
        return set(self.simplicial_complex_instance.vis_nodes_with_singleton_latent_parents).intersection(
            self.directed_structure_instance.nodes_with_no_parents)

    @property
    def number_of_root_nodes(self) -> int:
        return self.simplicial_complex_instance.number_of_latent

    def release_cache(self) -> None:
        """Drop every cached property, keeping the two parts: the mDAG is recomputed from them when next used."""
        for key in [k for k in self.__dict__ if k not in self._CORE_ATTRIBUTES]:
            del self.__dict__[key]

    @cached_property
    def is_SWIG_like(self):
        is_SWIG_like = len(self.vis_nodes_with_no_children) and len(
            self.vis_nodes_with_no_parents.difference(
                self.vis_nodes_with_no_children))
        if not is_SWIG_like:
            return False
        else:
            for u in self.vis_nodes_with_no_children:
                for v in self.vis_nodes_with_no_parents.difference([u]):
                    to_contract_DAG = self.directed_structure_instance.as_networkx_graph
                    contracted_DAG = nx.contracted_nodes(to_contract_DAG, u, v, copy=True)
                    if nx.is_directed_acyclic_graph(contracted_DAG):
                        return True
        return False

    @cached_property
    def as_string(self) -> str:
        return self.directed_structure_instance.as_string + '|' + self.simplicial_complex_instance.as_string

    def __str__(self) -> str:
        return self.as_string

    def __repr__(self) -> str:
        return self.as_string

    # ------------------------------------------------------------------
    # The numeric DAG: visible nodes 0..n-1, then one latent per row of as_extended_bit_array (latent n + r is row
    # r: the lexsorted extended simplicial complex, singleton latents included), exactly the numbering that
    # parents_of_for_supports_analysis hands to supports.py. Every graph computation below (d-separation,
    # e-separation, predecessors, descendants, automorphisms) runs on these bitmasks; networkx is kept for plotting.
    # ------------------------------------------------------------------

    @cached_property
    def parents_masks(self) -> IntArray:
        """parents_masks[v] = bitmask of the parents of node v (visible and latent, latents numbered as above);
        length number_of_visible + number_of_latent."""
        n = self.number_of_visible
        sg.check_node_count(self.number_of_nodes_including_singleton_latents)
        masks = np.zeros(self.number_of_nodes_including_singleton_latents, dtype=np.int64)
        masks[:n] = self.directed_structure_instance.parents_masks | (
            self.simplicial_complex_instance.extended_latent_parents_masks << np.int64(n))
        return masks

    @cached_property
    def number_of_nodes_including_singleton_latents(self) -> int:
        return self.number_of_visible + self.simplicial_complex_instance.number_of_latent

    @cached_property
    def singleton_latent_mask(self) -> int:
        return self.simplicial_complex_instance.singleton_latent_rows_mask << self.number_of_visible

    @cached_property
    def nonsingleton_latent_nodes(self) -> Tuple[int, ...]:
        n = self.number_of_visible
        return tuple(n + r for r in range(self.simplicial_complex_instance.number_of_latent)
                     if not (self.simplicial_complex_instance.singleton_latent_rows_mask >> r) & 1)

    @cached_property
    def nonsingleton_parents_masks(self) -> IntArray:
        """parents_masks with the singleton latents removed (the parents as_graph shows)."""
        return self.parents_masks & ~np.int64(self.singleton_latent_mask)

    @cached_property
    def parentsplus_masks(self) -> IntArray:
        """Visible and non-singleton latent parents of each visible node, plus the node itself."""
        n = self.number_of_visible
        return self.nonsingleton_parents_masks[:n] | (np.int64(1) << np.arange(n, dtype=np.int64))

    @cached_property
    def descendant_masks(self) -> IntArray:
        return sg.descendant_masks(self.parents_masks)

    def _networkx_graph(self, nodes: Iterable[int]) -> 'nx.DiGraph':
        g = nx.DiGraph()
        nodes = list(nodes)
        g.add_nodes_from(nodes)
        keep = sg.mask_of(nodes)
        for v in nodes:
            for p in sg.bits_of(int(self.parents_masks[v]) & keep):
                g.add_edge(p, v)
        return g

    @cached_property
    def as_graph(self):
        """The DAG over the visible and non-singleton latent nodes (networkx; plotting and legacy callers)."""
        return self._networkx_graph(list(self.visible_nodes) + list(self.nonsingleton_latent_nodes))

    @cached_property
    def as_graph_with_singleton_latents(self):
        return self._networkx_graph(range(self.number_of_nodes_including_singleton_latents))

    @cached_property
    def visible_automorphisms(self) -> Tuple[Tuple[int, ...], ...]:
        """The relabellings of the visible nodes that fix the mDAG (as image tuples). A latent is determined by
        its set of children, so these are exactly the automorphisms of the DAG with latents restricted to the
        visible nodes; they are read off the relabelled ids computed for unique_unlabelled_id."""
        target = (self.directed_structure_instance.as_integer, self.simplicial_complex_instance.as_integer)
        return tuple(perm for perm, ds_int, sc_int in zip(
            itertools.permutations(self.visible_nodes),
            self.directed_structure_instance.as_integer_permutations,
            self.simplicial_complex_instance.as_integer_permutations) if (ds_int, sc_int) == target)

    @cached_property
    def automorphic_order(self) -> int:
        return len(self.visible_automorphisms)

    @cached_property
    def as_extended_bit_array(self) -> BoolMatrix:
        #NOTE THAT THIS IS REVERSED RELATIVE TO UNIQUE ID!
        return np.vstack((self.directed_structure_instance.as_bit_square_matrix, self.simplicial_complex_instance.as_extended_bit_array))

    @cached_property
    def n_of_edges(self) -> int:
        return self.directed_structure_instance.number_of_edges
    
    @cached_property
    def n_of_latents(self) -> int:
        return self.simplicial_complex_instance.number_of_nonsingleton_latent

    @cached_property
    def relative_complexity_for_sat_solver(self) -> Tuple[int, int, int, Tuple[int, ...], int]:  # choose eqclass representative which minimizes this
        return (self.number_of_root_nodes, self.as_extended_bit_array.sum(), self.n_of_edges, self.simplicial_complex_instance.tally, -self.automorphic_order)
                
    @cached_property
    def parents_of_for_supports_analysis(self) -> List[IntArray]:
        return list(map(np.flatnonzero, self.as_extended_bit_array.T))

    @cached_property
    def parents_of(self) -> List[List[int]]:
        """The parents of each visible node (visible parents, then latents numbered as in parents_masks: the
        supports.py convention, singleton latents included)."""
        return [sg.bits_of(int(m)) for m in self.parents_masks[:self.number_of_visible]]

    @cached_property
    def ds_bit_size(self) -> int:
        return 2**(self.number_of_visible**2)
    def mdag_int_pair_to_single_int(self, ds_int: int, sc_int: int) -> int:
        #ELIE: Note that this MUST MATCH the function mdag_int_pair_to_single_int in the metagraph class. All is good.
        return ds_int + sc_int*self.ds_bit_size


    @cached_property
    def unique_id(self) -> int:
        # Returns a unique identification number.
        return self.mdag_int_pair_to_single_int(self.directed_structure_instance.as_integer,
                                                self.simplicial_complex_instance.as_integer)

    def __hash__(self) -> int:
        return self.unique_id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, mDAG):
            return False
        return self.unique_id == other.unique_id

    @cached_property
    def ids_under_relabelling(self) -> Set[int]:
        return {self.mdag_int_pair_to_single_int(ds_int, sc_int) for (ds_int, sc_int) in
         zip(self.directed_structure_instance.as_integer_permutations,
             self.simplicial_complex_instance.as_integer_permutations)}


    @cached_property
    def unique_unlabelled_id(self) -> int:
        # Returns the unique id minimized over relabellings
        return min(self.ids_under_relabelling)

    @cached_property
    def skeleton_instance(self) -> Union[LabelledUndirectedGraph, UndirectedGraph]:
        if hasattr(self, 'variable_names'):
            return LabelledUndirectedGraph(self.variable_names, self.directed_structure_instance.edge_list_with_variable_names + self.simplicial_complex_instance.simplicial_complex_with_variable_names)
        else:
            return UndirectedGraph(self.directed_structure_instance.as_tuples + self.simplicial_complex_instance.as_tuples,
                               self.number_of_visible)

    @cached_property
    def skeleton(self) -> int:
        return self.skeleton_instance.as_edges_integer

    @cached_property
    def skeleton_adj_mat(self) -> BoolMatrix:
        return np.logical_or(self.simplicial_complex_instance.as_bidirected_adjmat,
                             np.logical_or(self.directed_structure_instance.as_bit_square_matrix,
                                           self.directed_structure_instance.as_bit_square_matrix.T))

    @cached_property
    def skeleton_nof_edges(self) -> int:
        return np.count_nonzero(self.skeleton_adj_mat)

    @cached_property
    def skeleton_unlabelled(self) -> int:
        return self.skeleton_instance.as_edges_unlabelled_integer

    def perfectly_correlated_subgraph(self, subnodes: Iterable[int]) -> bool:
        new_hypergraph = hypergraph_full_cleanup([
            facet.intersection(subnodes) for facet in
            self.simplicial_complex_instance.extended_simplicial_complex_as_sets])
        subnodes_as_list = list(subnodes)
        subindices = np.ix_(subnodes_as_list, subnodes_as_list)
        subadjmat = np.eye(self.number_of_visible)
        subadjmat[subindices] = self.directed_structure_instance.as_bit_square_matrix[subindices]
        subadjmat_closure = AdjMat(subadjmat).transitive_closure_plus
        for subfacet in new_hypergraph:
            subfact_as_list = list(subfacet)
            common_ancestor_check = np.any(subadjmat_closure[subfact_as_list], axis=0)[subnodes_as_list].all()
            if common_ancestor_check:
                return True
        return False

    @cached_property
    def perfectly_correlatable_sets_in_subgraph(self) -> Set[frozenset]:
        perfectly_correlatable_sets = []
        for r in range(2, self.number_of_visible+1):
            for subnodes in itertools.combinations(self.visible_nodes, r):
                if self.perfectly_correlated_subgraph(subnodes):
                        perfectly_correlatable_sets.append(subnodes)
        return hypergraph_full_cleanup(perfectly_correlatable_sets)


    @cached_property
    def common_cause_connected_sets(self) -> Set[FrozenSet[int]]:
        return hypergraph_full_cleanup([self.directed_structure_instance.adjMat.descendantsplus_of(list(facet))
                                       for facet in self.simplicial_complex_instance.extended_simplicial_complex_as_sets])

    @cached_property
    def all_visible_share_one_root_ancestor(self) -> bool:
        return len(self.common_cause_connected_sets) == 1

    @cached_property
    def interesting_via_no_dsep_but_no_common_ancestor(self) -> bool:
        return (self.no_dsep_relations and (not self.all_visible_share_one_root_ancestor))

    @cached_property
    def is_connected(self) -> bool:
        return (len(merge_intersection(self.common_cause_connected_sets)) == 1)

    @staticmethod
    def fake_frozenset(stuff: Iterable[Any]) -> Tuple[Any, ...]:
        return tuple(sorted(stuff))

    @cached_property
    def esep_model(self) -> "np.ndarray":
        """R[D, i, j, K]: the elementary e-separations among the visible nodes (semigraphoid.esep_all on
        parents_masks): i and j d-separated by K after deleting the visible nodes of D; R[0] is d-separation."""
        return sg.esep_all(self.number_of_nodes_including_singleton_latents, self.parents_masks,
                           m=self.number_of_visible)

    @cached_property
    def dsep_model(self) -> "np.ndarray":
        """E[i, j, K]: the elementary d-separations among the visible nodes (the D = 0 slice of esep_model, computed
        on its own when that is not cached, so that CI-only consumers do not pay for the deletion sets)."""
        if 'esep_model' in self.__dict__:
            return self.esep_model[0]
        return sg.dsep_all(self.number_of_nodes_including_singleton_latents, self.parents_masks,
                           m=self.number_of_visible)

    @cached_property
    def _esep_cells(self) -> "np.ndarray":
        """argwhere of the e-separation model over its meaningful cells (i < j), as rows (D, i, j, K)."""
        return np.argwhere(self.esep_model & sg.esep_valid_cells(self.number_of_visible))

    @cached_property
    def all_esep_numeric(self) -> Set[Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]]:
        bits = sg.bits_table(self.number_of_visible)
        return {((int(i), int(j)), bits[K], bits[D]) for D, i, j, K in self._esep_cells}

    @cached_property
    def all_CI_numeric(self) -> Set[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        m = self.number_of_visible
        bits = sg.bits_table(m)
        return {((int(i), int(j)), bits[K]) for i, j, K in np.argwhere(self.dsep_model & sg.dsep_valid_cells(m))}

    @cached_property
    def d_separable_pairs(self) -> Set[Tuple[int, ...]]:
        E = self.dsep_model
        return {(x, y) for x, y in itertools.combinations(self.visible_nodes, 2) if E[x, y].any()}

    @cached_property
    def d_unseparable_pairs(self) -> Set[Tuple[int, ...]]:
        return set(itertools.combinations(self.visible_nodes, 2)).difference(self.d_separable_pairs)

    @cached_property
    def d_unrestricted_sets(self) -> List[Tuple[int, ...]]:
        """Subsets of visible nodes no pair of which is d-separated by a subset of the complement."""
        E = self.dsep_model
        full = (1 << self.number_of_visible) - 1
        d_unrestricted_sets = []
        for r in range(2, self.number_of_visible + 1):
            for subnodes in itertools.combinations(self.visible_nodes, r):
                Zs = sg.submasks(full & ~sg.mask_of(subnodes))
                if not any(E[x, y, Zs].any() for x, y in itertools.combinations(subnodes, 2)):
                    d_unrestricted_sets.append(subnodes)
        return d_unrestricted_sets

    @cached_property
    def interesting_via_generalized_maximality(self):
        for subnodes in set(self.d_unrestricted_sets).difference(self.skeleton_instance.as_frozenset_edges):
            if not self.perfectly_correlated_subgraph(subnodes):
                return True
        return False

    @cached_property
    def no_dsep_relations(self) -> bool:
        return len(self.all_CI_numeric) == 0

    @cached_property
    def CI_count(self) -> int:
        return int(np.count_nonzero(self.dsep_model & sg.dsep_valid_cells(self.number_of_visible)))

    # Unlabelled (relabelling-invariant) keys: the lexicographically smallest relabelled model, packed to bytes
    # (semigraphoid.canonical_form), reported as the sorted relations of that relabelling. Only equality of these
    # keys is ever used (classification into CI / e-separation classes).

    def convert_to_named_set_of_tuples_of_tuples(self, attribute: str) -> Set[Tuple[Tuple[Any, ...], ...]]:
        if hasattr(self, 'variable_names'):
            return set(
                tuple(
                    self.fake_frozenset(partsextractor(self.variable_names, variable_set))
                    for variable_set in relation)
                for relation in self.__getattribute__(attribute))
        else:
            return self.__getattribute__(attribute)

    @cached_property
    def all_CI(self) -> Tuple[Tuple[Tuple[Any, ...], ...], ...]:
        return tuple(self.convert_to_named_set_of_tuples_of_tuples('all_CI_numeric'))

    @cached_property
    def all_CI_unlabelled(self) -> Tuple[Tuple[Tuple[int, ...], ...], ...]:
        m = self.number_of_visible
        bits = sg.bits_table(m)
        canonical = sg.canonical_form(self.dsep_model, m)
        return self.fake_frozenset(((int(i), int(j)), bits[K]) for i, j, K in
                                   np.argwhere(canonical & sg.dsep_valid_cells(m)))

    @cached_property
    def all_esep(self) -> Tuple[Tuple[Tuple[Any, ...], ...], ...]:
        return tuple(self.convert_to_named_set_of_tuples_of_tuples('all_esep_numeric'))

    @cached_property
    def all_esep_unlabelled(self) -> Tuple[Tuple[Tuple[int, ...], ...], ...]:
        m = self.number_of_visible
        bits = sg.bits_table(m)
        canonical = sg.canonical_form(self.esep_model, m)
        return self.fake_frozenset(((int(i), int(j)), bits[K], bits[D]) for D, i, j, K in
                                   np.argwhere(canonical & sg.esep_valid_cells(m)))

    @cached_property
    def e_separable_pairs(self) -> Set[Tuple[int, int]]:
        return set(itertools.combinations(self.visible_nodes, 2)).difference(self.skeleton_instance.as_edges)

    @cached_property
    def interesting_via_non_maximal(self) -> bool:
        return not self.d_unseparable_pairs.issubset(self.skeleton_instance.as_edges)

    @methodtools.lru_cache(maxsize=None, typed=False)
    def support_testing_instance_binary(self, n: int):
        if not hasattr(self, 'restricted_perfect_predictions_numeric'):
            self.restricted_perfect_predictions_numeric = tuple()
        return SmartSupportTesting(parents_of=self.parents_of_for_supports_analysis,
                                   observed_cardinalities=np.broadcast_to(2, self.number_of_visible),
                                   nof_events=n,
                                   esep_relations=self.all_esep_numeric,
                                   pp_relations=self.restricted_perfect_predictions_numeric,
                                   visible_automorphisms=self.visible_automorphisms
                                   )

    @methodtools.lru_cache(maxsize=None, typed=False)
    def support_testing_instance(self, observed_cardinalities: Iterable[int], n: int):
        if not hasattr(self, 'restricted_perfect_predictions_numeric'):
            self.restricted_perfect_predictions_numeric = tuple()
        return SmartSupportTesting(parents_of=self.parents_of_for_supports_analysis,
                                   observed_cardinalities=observed_cardinalities,
                                   nof_events=n,
                                   esep_relations=self.all_esep_numeric,
                                   pp_relations=self.restricted_perfect_predictions_numeric,
                                   visible_automorphisms=self.visible_automorphisms
                                   )
    def infeasible_4222_supports_n_events(self,n: int,**kwargs):
        return tuple(self.support_testing_instance((4, 2, 2, 2),n).unique_infeasible_supports_as_expanded_integers(**kwargs, name='mgh', use_timer=False))
    def infeasible_binary_supports_n_events(self, n: int, **kwargs):
        return tuple(self.support_testing_instance_binary(n).unique_infeasible_supports_as_expanded_integers(**kwargs, name='mgh', use_timer=False))
    def infeasible_binary_supports_n_events_beyond_esep(self, n: int, **kwargs):
        return tuple(self.support_testing_instance_binary(n).unique_infeasible_supports_beyond_esep_as_expanded_integers(**kwargs, name='mgh', use_timer=False))
    def infeasible_binary_supports_n_events_beyond_dsep(self, n: int, **kwargs):
        return tuple(self.support_testing_instance_binary(n).unique_infeasible_supports_beyond_dsep_as_expanded_integers(**kwargs, name='mgh', use_timer=False))

    def infeasible_binary_supports_n_events_as_matrices(self, n: int, **kwargs):
        return self.support_testing_instance_binary(n).unique_infeasible_supports_as_expanded_matrices(**kwargs, name='mgh', use_timer=False)
    def infeasible_binary_supports_n_events_beyond_esep_as_matrices(self, n: int, **kwargs):
        return self.support_testing_instance_binary(n).unique_infeasible_supports_beyond_esep_as_expanded_matrices(**kwargs, name='mgh', use_timer=False)
    def infeasible_binary_supports_n_events_beyond_dsep_as_matrices(self, n: int, **kwargs):
        return self.support_testing_instance_binary(n).unique_infeasible_supports_beyond_dsep_as_expanded_matrices(**kwargs, name='mgh', use_timer=False)
    def representative_infeasible_binary_supports_n_events_as_matrices(self, n: int, **kwargs):
        return self.support_testing_instance_binary(n).unique_infeasible_supports_as_compressed_matrices(**kwargs, name='mgh', use_timer=False)
    def representative_infeasible_binary_supports_n_events_beyond_esep_as_matrices(self, n: int, **kwargs):
        return self.support_testing_instance_binary(n).unique_infeasible_supports_beyond_esep_as_compressed_matrices(**kwargs, name='mgh', use_timer=False)
    def representative_infeasible_binary_supports_n_events_beyond_dsep_as_matrices(self, n: int, **kwargs):
        return self.support_testing_instance_binary(n).unique_infeasible_supports_beyond_dsep_as_compressed_matrices(**kwargs, name='mgh', use_timer=False)

    def infeasible_binary_supports_n_events_unlabelled(self, n: int, **kwargs):
        return tuple(self.support_testing_instance_binary(n).unique_infeasible_supports_as_integers_unlabelled(**kwargs, name='mgh', use_timer=False))
    def infeasible_binary_supports_n_events_beyond_esep_unlabelled(self, n: int, **kwargs):
        return tuple(self.support_testing_instance_binary(n).unique_infeasible_supports_beyond_esep_as_integers_unlabelled(**kwargs, name='mgh', use_timer=False))


    def no_infeasible_binary_supports_beyond_esep(self, n: int, **kwargs):
        return self.support_testing_instance_binary(n).no_infeasible_supports_beyond_esep(**kwargs, name='mgh', use_timer=False)
    def no_infeasible_binary_supports_beyond_esep_up_to(self, max_n: int, **kwargs):
        return all(self.no_infeasible_binary_supports_beyond_esep(n, **kwargs) for n in range(2, max_n + 1))
    def no_infeasible_binary_supports_beyond_dsep(self, n: int, **kwargs):
        return self.support_testing_instance_binary(n).no_infeasible_supports_beyond_dsep(**kwargs, name='mgh', use_timer=False)
    def no_infeasible_binary_supports_beyond_dsep_up_to(self, max_n: int, **kwargs):
        return all(self.no_infeasible_binary_supports_beyond_dsep(n, **kwargs) for n in range(2, max_n + 1))
    
    def no_infeasible_4222_supports_beyond_dsep(self,n: int,**kwargs):
        return self.support_testing_instance((4, 2, 2, 2),n).no_infeasible_supports_beyond_dsep(**kwargs, name='mgh', use_timer=False)
    def no_infeasible_4222_supports_beyond_dsep_up_to(self, max_n: int, **kwargs):
        return all(self.no_infeasible_4222_supports_beyond_dsep(n, **kwargs) for n in range(2, max_n + 1))
    
    def no_infeasible_8222_supports_beyond_dsep(self,n: int,**kwargs):
        return self.support_testing_instance((8, 2, 2, 2),n).no_infeasible_supports_beyond_dsep(**kwargs, name='mgh', use_timer=False)
    def no_infeasible_8222_supports_beyond_dsep_up_to(self, max_n: int, **kwargs):
        return all(self.no_infeasible_8222_supports_beyond_dsep(n, **kwargs) for n in range(2, max_n + 1))
    
    def no_infeasible_6222_supports_beyond_dsep(self,n: int,**kwargs):
        return self.support_testing_instance((6, 2, 2, 2),n).no_infeasible_supports_beyond_dsep(**kwargs, name='mgh', use_timer=False)
    def no_infeasible_6222_supports_beyond_dsep_up_to(self, max_n: int, **kwargs):
        return all(self.no_infeasible_6222_supports_beyond_dsep(n, **kwargs) for n in range(2, max_n + 1))
    
    
    def no_esep_beyond_dsep(self, n: int) -> bool:
        return not self.support_testing_instance_binary(n).interesting_due_to_esep
    def no_esep_beyond_dsep_up_to(self, max_n: int) -> bool:
        return all(self.no_esep_beyond_dsep(n) for n in range(2, max_n + 1))


    def infeasible_binary_supports_beyond_esep_up_to(self, max_n: int, **kwargs):
        return np.fromiter(itertools.chain.from_iterable(
            (self.infeasible_binary_supports_n_events_beyond_esep(n, **kwargs) for n in range(2, max_n + 1))),
            dtype=int)
    def infeasible_binary_supports_up_to(self, max_n: int, **kwargs):
        return np.fromiter(itertools.chain.from_iterable(
            (self.infeasible_binary_supports_n_events(n, **kwargs) for n in range(2, max_n + 1))),
            dtype=int)

    def infeasible_binary_supports_beyond_esep_as_matrices_up_to(self, max_n: int, **kwargs):
        return list(itertools.chain.from_iterable((self.infeasible_binary_supports_n_events_beyond_esep_as_matrices(n, **kwargs).astype(int) for n in range(2, max_n + 1))))
    def infeasible_binary_supports_beyond_dsep_as_matrices_up_to(self, max_n: int, **kwargs):
        return list(itertools.chain.from_iterable((self.infeasible_binary_supports_n_events_beyond_dsep_as_matrices(n, **kwargs).astype(int) for n in range(2, max_n + 1))))
    def infeasible_binary_supports_as_matrices_up_to(self, max_n: int, **kwargs):
        return list(itertools.chain.from_iterable((self.infeasible_binary_supports_n_events(n, **kwargs).astype(int) for n in range(2, max_n + 1))))

    def no_infeasible_binary_supports_n_events(self, n: int, **kwargs):
        return self.support_testing_instance_binary(n).no_infeasible_supports(**kwargs, name='mgh', use_timer=False)
    def no_infeasible_binary_supports_up_to(self, max_n: int, **kwargs):
        return all(self.no_infeasible_binary_supports_n_events(n, **kwargs) for n in range(2, max_n + 1))


    # ------------------------------------------------------------------
    # The HLP edge rule (Evans; Henson, Lal and Pusey): an edge a -> b between two members of a facet with
    # Pa(a) <= Pa(b) can be added or dropped without changing the classical model, since b sees everything a is
    # a function of and the latent can carry a's noise. Quantumly the rule holds when a has no visible parent
    # and exactly one latent parent (`is_premeasurable`): a's measurement is then fixed, so the facet's
    # source can perform it and hand the outcome to a and to b. The same argument applied to the *other* members of
    # a's facets gives the second form of the rule (quantum_hlp_origin_ok): when every other member is premeasurable,
    # the sources pre-measure them and a's remaining quantum resources are shared with b alone, so b computes a's
    # output. Otherwise the rule fails: Q{A,B,D}, Q{A,C,D}, Q{B,C} with A -> D lets D copy A's CHSH outcome against
    # B, which the monogamy of nonlocality forbids without the edge; Q{A,B,C} with S -> A, S -> B, T -> C and
    # A -> B likewise lets B copy A's outcome while A and C violate CHSH.
    # ------------------------------------------------------------------

    def is_premeasurable(self, node: int) -> bool:
        """No visible parent and exactly one (non-singleton) latent parent: the node's measurement does not depend
        on anything the facet's source does not know, so the source can perform it."""
        pa = int(self.nonsingleton_parents_masks[node])
        return pa & ((1 << self.number_of_visible) - 1) == 0 and bin(pa >> self.number_of_visible).count('1') == 1

    def quantum_hlp_origin_ok(self, a: int, b: int) -> bool:
        """Is the edge a -> b, with Pa(a) a subset of Pa(b), redundant for quantum model sets? Yes when b can obtain
        a's value without it, which the pre-measurement argument gives in two forms: a is premeasurable (its facet's
        source performs a's fixed measurement and hands the outcome to a and to b), or every other member of every
        facet of a is premeasurable (each source performs those members' fixed measurements first and tells a and b
        the outcomes; what is left of a's quantum resources is shared with b alone, and b, which knows every input
        of a, can then compute a's output from shared classical randomness). The second form includes the facet
        {a, b} with no other member, whatever visible parents a has."""
        if self.is_premeasurable(a):
            return True
        facets = self.simplicial_complex_instance.simplicial_complex_as_sets
        return all(self.is_premeasurable(v) for F in facets if a in F for v in F if v != a and v != b)

    def _hlp_candidates_ok(self, pairs: Iterable[Tuple[int, int]], rule: str) -> List[Tuple[int, int]]:
        assert rule in ('classical', 'quantum'), rule
        pa = self.nonsingleton_parents_masks.tolist()   # visible and latent parents, as as_graph shows them
        return [pair for pair in pairs if pa[pair[0]] & ~pa[pair[1]] == 0
                and (rule == 'classical' or self.quantum_hlp_origin_ok(pair[0], pair[1]))]

    @cached_property
    def droppable_edges(self) -> List[Tuple[int, int]]:
        candidates = [tuple(pair) for pair in self.directed_structure_instance.edge_list if
                      any(set(pair).issubset(hyperedge) for hyperedge in self.simplicial_complex_instance.compressed_simplicial_complex)]
        return self._hlp_candidates_ok(candidates, 'classical')

    @cached_property
    def addable_edges(self) -> List[Tuple[int, int]]:
        candidates = set(self.skeleton_instance.as_edges).difference(self.directed_structure_instance.edge_list)
        return self._hlp_candidates_ok(candidates, 'classical')

    @cached_property
    def quantum_droppable_edges(self) -> List[Tuple[int, int]]:
        """droppable_edges whose origin passes quantum_hlp_origin_ok: the quantum-valid HLP edges."""
        return self._hlp_candidates_ok(self.droppable_edges, 'quantum')

    @cached_property
    def quantum_addable_edges(self) -> List[Tuple[int, int]]:
        return self._hlp_candidates_ok(self.addable_edges, 'quantum')

    def hlp_edges(self, rule: str) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """(droppable, addable) under the classical or the quantum rule."""
        if rule == 'classical':
            return self.droppable_edges, self.addable_edges
        assert rule == 'quantum', rule
        return self.quantum_droppable_edges, self.quantum_addable_edges

    @cached_property
    def equivalent_under_edge_dropping(self) -> List["mDAG"]:
        equivalent_mdags = []
        for edges_to_drop in powerset(self.droppable_edges):
            if len(edges_to_drop):
                equivalent_mdags.append(mDAG(DirectedStructure(
                    self.directed_structure_instance.as_set_of_tuples.difference(edges_to_drop),
                    self.number_of_visible),
                     self.simplicial_complex_instance))
        return equivalent_mdags


    def weaker_mDAG_HLP(self, rule: str = 'classical'):
        """The id of this mDAG with all its HLP-droppable edges (under `rule`) dropped, when nothing is addable:
        the equivalence edge of the metagraph (the addable case is reached from the structure that has the edges)."""
        droppable, addable = self.hlp_edges(rule)
        if droppable and not addable:
            new_bit_square_matrix = self.directed_structure_instance.as_bit_square_matrix.copy()
            new_bit_square_matrix[tuple(np.asarray(droppable, dtype=int).reshape((-1, 2)).T)] = False
            yield self.mdag_int_pair_to_single_int(bitarray_to_int(new_bit_square_matrix),
                                                   self.simplicial_complex_instance.as_integer)

    @property
    def generate_weaker_mDAG_HLP(self):
        return self.weaker_mDAG_HLP('classical')

    @property
    def generate_slightly_weaker_mDAGs_HLP(self):
        for droppable_edge in self.droppable_edges:
            new_bit_square_matrix = self.directed_structure_instance.as_bit_square_matrix.copy()
            new_bit_square_matrix[droppable_edge] = False
            # new_bit_square_matrix[tuple(np.asarray(self.droppable_edges, dtype=int).reshape((-1,2)).T)] = False
            yield self.mdag_int_pair_to_single_int(bitarray_to_int(new_bit_square_matrix),
                                                   self.simplicial_complex_instance.as_integer)


    @staticmethod
    def _all_bipartitions(variables_to_partition: Iterable[int]):
        """(C, D) frozenset pairs, one per bipartition of the items (the order of _bipartition_patterns, which
        _facet_bipartitions_as_masks follows)."""
        items = tuple(variables_to_partition)
        for pattern in _bipartition_patterns(len(items)):
            yield (frozenset(items[i] for i in range(len(items)) if i not in pattern),
                   frozenset(items[i] for i in pattern))


    @cached_property
    def all_parentsplus_list(self) -> List[FrozenSet[int]]:
        """Each visible node with its visible and non-singleton latent parents (latents numbered as in
        parents_masks)."""
        return [frozenset(sg.bits_of(int(m))) for m in self.parentsplus_masks]

    def set_predecessors(self, X: Iterable[int]) -> FrozenSet[int]:
        # return frozenset().union(*partsextractor(self.directed_structure_instance.observable_parentsplus_list, X))
        return frozenset(itertools.chain.from_iterable(partsextractor(self.all_parentsplus_list, X)))
    def singleton_predecessors(self, x: int) -> FrozenSet[int]:
        return partsextractor(self.all_parentsplus_list, x)
    
    def singleton_visible_predecessors(self, x: int) -> FrozenSet[int]:
        return partsextractor(self.directed_structure_instance.observable_parentsplus_list, x)

    def children(self, node: int) -> List[int]:
        c = []
        for v in self.visible_nodes:
            if node in self.singleton_visible_predecessors(v) and node != v:
                c.append(v)
        return c
        
    def descendants(self, node: int) -> Set[int]:
        return set(sg.bits_of(int(self.descendant_masks[node])))

    def _facet_bipartitions_as_masks(self) -> Iterable[Tuple[int, int]]:
        """(C, D) bitmask pairs, one per bipartition of each facet, in the order _all_bipartitions yields them."""
        for facet in self.simplicial_complex_instance.compressed_simplicial_complex:
            facet = tuple(int(v) for v in facet)
            F = sg.mask_of(facet)
            for pattern in _bipartition_patterns(len(facet)):
                D = 0
                for i in pattern:
                    D |= 1 << facet[i]
                yield F & ~D, D

    @staticmethod
    def _masks_to_frozensets(pairs: Iterable[Tuple[int, int]]) -> List[Tuple[FrozenSet[int], FrozenSet[int]]]:
        return [(frozenset(sg.bits_of(C)), frozenset(sg.bits_of(D))) for C, D in pairs]

    @staticmethod
    def _union_of_masks(masks: List[int], C: int) -> int:
        u = 0
        for c in sg.bits_of(C):
            u |= masks[c]
        return u

    @cached_property
    def splittable_faces(self) -> List[Tuple[FrozenSet[int], FrozenSet[int]]]:
        """Bipartitions (C, D) of a facet with Pa+(C) a subset of Pa+(d) for every d in D (visible and non-singleton
        latent parents, plus the nodes themselves)."""
        pp = self.parentsplus_masks.tolist()
        result = []
        for C, D in self._facet_bipartitions_as_masks():
            u = self._union_of_masks(pp, C)
            if all(u & ~pp[d] == 0 for d in sg.bits_of(D)):
                result.append((C, D))
        return self._masks_to_frozensets(result)

    @cached_property
    def eventually_splittable_faces(self) -> List[Tuple[FrozenSet[int], FrozenSet[int]]]:
        pp = self.parentsplus_masks.tolist()
        result = []
        for C, D in self._facet_bipartitions_as_masks():
            u = self._union_of_masks(pp, C) & ~C
            if all(u & ~(pp[d] & ~D) == 0 for d in sg.bits_of(D)):
                result.append((C, D))
        return self._masks_to_frozensets(result)

    @cached_property
    def has_no_eventually_splittable_face(self) -> bool:
        return len(self.eventually_splittable_faces) == 0
    
    def set_visible_predecessors(self,X: Iterable[int]) -> FrozenSet[int]:
        return frozenset(itertools.chain.from_iterable(partsextractor(self.directed_structure_instance.observable_parentsplus_list, X)))
    def singleton_visible_predecessors(self, x: int) -> FrozenSet[int]:
        return partsextractor(self.directed_structure_instance.observable_parentsplus_list, x)

    def two_layer_circuit(self) -> bool:
        for node in self.visible_nodes:
            if len(self.children(node))>0 and len(self.singleton_predecessors(node))>1:
                return False
        return True

    @cached_property
    def numerical_districts(self) -> List[Set[int]]:
        return self.simplicial_complex_instance.districts

    @cached_property
    def districts(self) -> List[Set[Any]]:
        #Districts are returned as a list of sets
        if hasattr(self, 'variable_names'):
            return [partsextractor(self.variable_names, district) for district in self.numerical_districts]
        else:
            return self.numerical_districts

        
    def set_district(self, X: Iterable[int]) -> FrozenSet[int]:
        return frozenset(itertools.chain.from_iterable((district for district in self.districts if not district.isdisjoint(X))))
    def singleton_district(self, x: int) -> FrozenSet[int]:
        return frozenset(itertools.chain.from_iterable((district for district in self.districts if x in district)))

        
    def networkx_plot_mDAG(self) -> None:
        G=nx.DiGraph()
        G.add_nodes_from(self.visible_nodes)
        G.add_nodes_from(self.nonsingleton_latent_nodes)
        pos=nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, nodelist=self.visible_nodes, node_color="tab:blue")
        nx.draw_networkx_nodes(G, pos, nodelist=self.nonsingleton_latent_nodes, node_color="tab:red")
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G,pos,edgelist=self.directed_structure_instance.edge_list)
        latent_edges = [(p, v) for p, v in self.as_graph.edges if p >= self.number_of_visible]
        nx.draw_networkx_edges(G,pos,edgelist=latent_edges)
        plt.show()
        
    def plot_mDAG_indicating_one_node(self,node: int) -> None:
        G=nx.DiGraph()
        G.add_nodes_from(self.visible_nodes)
        G.add_nodes_from(self.nonsingleton_latent_nodes)
        pos=nx.spring_layout(G)
        v = [x for x in self.visible_nodes if x != node]
        nx.draw_networkx_nodes(G, pos, nodelist=v, node_color="tab:blue")
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color="tab:green")
        nx.draw_networkx_nodes(G, pos, nodelist=self.nonsingleton_latent_nodes, node_color="tab:red")
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, edgelist=self.directed_structure_instance.edge_list)
        latent_edges = [(p, v) for p, v in self.as_graph.edges if p >= self.number_of_visible]
        nx.draw_networkx_edges(G, pos, edgelist=latent_edges)
        plt.show()
    
    @cached_property
    def weak_splittable_faces(self) -> List[Tuple[FrozenSet[int], FrozenSet[int]]]:
        """Bipartitions (C, D) of a facet with the visible Pa+(C) a subset of the visible Pa+(d) for every d in D,
        and no node of C in another facet."""
        n = self.number_of_visible
        vpp_masks = sg.columns_to_masks(self.directed_structure_instance.as_bit_square_matrix_plus_eye).tolist()
        facets = [sg.mask_of(f) for f in self.simplicial_complex_instance.compressed_simplicial_complex]
        result = []
        for C, D in self._facet_bipartitions_as_masks():
            others = 0
            for f in facets:
                if f != C | D:
                    others |= f
            if C & others == 0 and all(self._union_of_masks(vpp_masks, C) & ~vpp_masks[d] == 0 for d in sg.bits_of(D)):
                result.append((C, D))
        return self._masks_to_frozensets(result)

    
    @property  
    def generate_weaker_mDAGs_FaceSplitting_Weak(self):
        for C, D in self.weak_splittable_faces:
            new_simplicial_complex = self.simplicial_complex_instance.simplicial_complex_as_sets.copy()
            new_simplicial_complex.discard(C.union(D))
            # setC = set(C)
            # setD = set(D)
            if not any(C.issubset(facet) for facet in new_simplicial_complex):
                new_simplicial_complex.add(C)
            if not any(D.issubset(facet) for facet in new_simplicial_complex):
                new_simplicial_complex.add(D)
            # new_simplicial_complex.sort()
            yield self.mdag_int_pair_to_single_int(
                self.directed_structure_instance.as_integer,
                Hypergraph(new_simplicial_complex, self.number_of_visible).as_integer)


    def generate_weaker_mDAGs_FaceSplitting(self, version: str):
        if version=='strong':
            return self.generate_weaker_mDAGs_FaceSplitting_Simultaneous
        if version=='moderate':
            return self.generate_weaker_mDAGs_FaceSplitting_Safe
        if version=='weak':
            return self.generate_weaker_mDAGs_FaceSplitting_Weak
        else:
            print('Version of Face Splitting unspecified')

    @property
    #FaceSplitting_Safe = Moderate Face Splitting
    def generate_weaker_mDAGs_FaceSplitting_Safe(self):
        for C, D in self.splittable_faces:
            new_simplicial_complex = self.simplicial_complex_instance.simplicial_complex_as_sets.copy()
            new_simplicial_complex.discard(C.union(D))
            # setC = set(C)
            # setD = set(D)
            if not any(C.issubset(facet) for facet in new_simplicial_complex):
                new_simplicial_complex.add(C)
            if not any(D.issubset(facet) for facet in new_simplicial_complex):
                new_simplicial_complex.add(D)
            # new_simplicial_complex.sort()
            yield self.mdag_int_pair_to_single_int(
                self.directed_structure_instance.as_integer,
                Hypergraph(new_simplicial_complex, self.number_of_visible).as_integer)

    @property  # Agressive conjucture of simultaneous face splitting
    def generate_weaker_mDAGs_FaceSplitting_Simultaneous(self):
        new_dict = defaultdict(list)
        for C, D in self.splittable_faces:
            new_dict[D].append(C)
        for D, Cs in new_dict.items():
            new_simplicial_complex = self.simplicial_complex_instance.simplicial_complex_as_sets.copy()
            #setD = frozenset(D)
            for C in Cs:
                new_simplicial_complex.discard(C.union(D))
            if not any(D.issubset(facet) for facet in new_simplicial_complex):
                new_simplicial_complex.add(D)
            for C in Cs:
                # setC = frozenset(C)
                if not any(C.issubset(facet) for facet in new_simplicial_complex):
                    new_simplicial_complex.add(C)
            # new_simplicial_complex.sort()
            yield self.mdag_int_pair_to_single_int(
                self.directed_structure_instance.as_integer,
                Hypergraph(new_simplicial_complex, self.number_of_visible).as_integer)

    #TODO: slightly speed up this by avoiding Hypergraph creation. That is, directly modify the bit array.
    #Or, if we are really fancy, we can modify the bits of the unique_id itself!!


    #Elie: I have integrated subgraph into both labelled directed structure and labelled hypergraph.
    def subgraph(self, list_of_nodes: Tuple[int, ...]):
        return mDAG(
            LabelledDirectedStructure(list_of_nodes, self.directed_structure_instance.edge_list),
            LabelledHypergraph(list_of_nodes, self.simplicial_complex_instance.simplicial_complex_as_sets)
        )

    def set_closure(self, X_int_or_list: Iterable[int], return_bidirectedQ: bool = False):
        return numeric_closure(core_B=X_int_or_list,
                               n=self.number_of_visible,
                               ds_adjmat= self.directed_structure_instance.as_bit_square_matrix_plus_eye,
                               sc_adjmat= self.simplicial_complex_instance.as_bidirected_adjmat,
                               return_bidirectedQ= return_bidirectedQ)

    @cached_property
    def singeleton_closures(self):
        return [frozenset(self.set_closure([i])) for i in range(self.number_of_visible)]

    def are_densely_connected(self, node1: int, node2: int):
        if node1 in self.set_visible_predecessors(self.singeleton_closures[node2]):
            return True
        if node2 in self.set_visible_predecessors(self.singeleton_closures[node1]):
            return True
        if self.set_closure([node1, node2], return_bidirectedQ=True)[-1]:
        # if is_this_subadjmat_densely_connected(self.simplicial_complex_instance.as_bidirected_adjmat, combined_closure):
            return True
        else:
            return False


# Evans 2021: It is possible to have a distribution with 2 variables identical to one another and independent of all others iff they are densely connected
# So, if node1 and node2 are densely connected in G1 but not in G2, we know that G1 is NOT equivalent to G2.

    @cached_property
    def all_densely_connected_pairs_numeric(self) -> Tuple[Tuple[int, int], ...]:
        return tuple([nodepair for nodepair in itertools.combinations(self.visible_nodes, 2) if self.are_densely_connected(*nodepair)])
    @cached_property
    def all_densely_connected_pairs(self) -> Tuple[Tuple[int, ...], ...]:
        if hasattr(self, 'variable_names'):
            return tuple(map(self.fake_frozenset, np.take(self.variable_names, self.all_densely_connected_pairs_numeric)))
        else:
            return self.all_densely_connected_pairs_numeric

    @cached_property
    def all_densely_connected_pairs_unlabelled(self) -> Tuple[Tuple[int, ...], ...]:
        n = self.number_of_visible
        adj = np.zeros((n, n), dtype=np.bool_)
        for x, y in self.all_densely_connected_pairs_numeric:
            adj[x, y] = adj[y, x] = True
        canonical = sg.canonical_form(adj, n)
        return self.fake_frozenset((int(x), int(y)) for x, y in np.argwhere(np.triu(canonical, 1)))


    @cached_property
    def fundamental_graphQ(self) -> bool:
        # Implement three conditions
        common_cause_districts = self.simplicial_complex_instance.nonsingleton_districts
        if len(common_cause_districts) != 1:
            return False
        district_vertices = set(common_cause_districts[0])
        non_district_vertices = set(range(self.number_of_visible)).difference(district_vertices)
        for v in non_district_vertices:
            # Does it have children outside the district
            if set(np.flatnonzero(self.directed_structure_instance.as_bit_square_matrix[v])).isdisjoint(district_vertices):
                return False
            # Does it have parents
            if self.directed_structure_instance.as_bit_square_matrix[:, v].any():
                return False
        return True
    
    @cached_property
    def latent_free_graphQ(self) -> bool:
        if self.simplicial_complex_instance.number_of_nonsingleton_latent==0:
            return True
        return False

    def marginalize_node(self, marginalized_node: int):
        variable_names=[node for node in self.visible_nodes if node!=marginalized_node]
        new_edge_list=[edge for edge in self.directed_structure_instance.edge_list if edge[0]!=marginalized_node and edge[1]!=marginalized_node]
        for parent_node in self.singleton_visible_predecessors(marginalized_node):
            if parent_node != marginalized_node:
                for child in self.children(marginalized_node):
                    new_edge_list.append((parent_node,child))
        new_simplicial_complex=[]                  
        marginalized_node_in_some_facet=False
        for facet in self.simplicial_complex_instance.simplicial_complex_as_sets:
            if len(facet)>1:
                if marginalized_node in facet:
                    marginalized_node_in_some_facet=True
                    new_facet=(element for element in facet if element!=marginalized_node)
                    new_facet=tuple(set(new_facet).union(set(self.children(marginalized_node))))
                else: 
                    new_facet=tuple(facet)
                already_there=False
                for facet_element in new_simplicial_complex:
                    if set(new_facet).issubset(set(facet_element)):
                        already_there=True
                if already_there==False and len(new_facet)>1:  
                    new_simp_comp_copy=new_simplicial_complex.copy()
                    for facet_element in new_simp_comp_copy:
                        if set(facet_element).issubset(set(new_facet)):
                            new_simplicial_complex.remove(facet_element)
                    new_simplicial_complex.append(new_facet)
        if marginalized_node_in_some_facet==False and len(self.children(marginalized_node))>1:
            new_simplicial_complex.append(tuple(self.children(marginalized_node)))
        return new_edge_list, new_simplicial_complex, variable_names
    
    def fix_to_point_distribution(self, node: int):  #returns a smaller mDAG
        return self.subgraph(self.visible_nodes[:node]+self.visible_nodes[(node+1):])



class Unlabelled_mDAG(mDAG):
    def __init__(self, directed_structure_instance: DirectedStructure, simplicial_complex_instance: Hypergraph):
        super().__init__(directed_structure_instance, simplicial_complex_instance)
    def __hash__(self):
        return self.unique_unlabelled_id
    def __eq__(self, other: object):
        if not isinstance(other, Unlabelled_mDAG):
            return False
        return self.unique_id == other.unique_unlabelled_id

# class labelled_mDAG(mDAG):
#     def __init__(self, labelled_directed_structure_instance, labelled_simplicial_complex_instance):
#         assert labelled_directed_structure_instance.variable_names == labelled_simplicial_complex_instance.variable_names, "Name conflict."
#         self.variable_names = labelled_directed_structure_instance.variable_names
#         super().__init__(labelled_directed_structure_instance, labelled_simplicial_complex_instance)

if __name__ == '__main__':
    G_triangle = mDAG(DirectedStructure([], 3),
                   Hypergraph([(0, 1), (0, 2), (1, 2)], 3))
    print(G_triangle)

    G58a = mDAG(DirectedStructure([(1, 2), (2, 3)], 4),
                Hypergraph([(0, 1), (1, 3)], 4))
    G58b = mDAG(DirectedStructure([(1, 2), (2, 3)], 4),
                Hypergraph([(0, 1), (1, 3), (2, 3)], 4))
    # print(G58a.all_esep_numeric)
    # print(G58b.all_esep_numeric)
    # print(G58a.all_densely_connected_pairs_numeric)
    # print(G58b.all_densely_connected_pairs_numeric)
    # print("G 5.8(a)")
    # print(G58a.infeasible_binary_supports_beyond_esep_as_matrices_up_to(15))
    # print("G 5.8(b)")
    # print(G58b.infeasible_binary_supports_beyond_esep_as_matrices_up_to(15))
    questionable_support = np.array([
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 1, 0, 1],
        [1, 0, 1, 1],
        [1, 1, 1, 1]
    ])
    # support_tester = G58a.support_testing_instance_binary(8)
    print(G58a.support_testing_instance_binary(8).feasibleQ_from_matrix(questionable_support, return_model=False))
    print(G58b.support_testing_instance_binary(8).feasibleQ_from_matrix(questionable_support, return_model=False))

    # rejected = G58a.support_testing_instance_binary(8).unique_infeasible_supports_as_expanded_matrices()
    # print(any(np.array_equiv(supp[:,0], supp[:,3]) for supp in rejected))
    
    
    G1= mDAG(DirectedStructure([(1,2), (2,3)], 4), Hypergraph([(0, 1),(1, 3)], 4))
    G2= mDAG(DirectedStructure([(1,2), (2,3)], 4), Hypergraph([(0, 1),(1, 3), (2,3)], 4))
    S=[[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1]]
    
    G1_supp=G1.support_testing_instance((2,2,2,2), 8).unique_infeasible_supports_as_expanded_matrices()
    G2_supp=G2.support_testing_instance((2,2,2,2), 8).unique_infeasible_supports_as_expanded_matrices()
    G1_supp_list=[[element.tolist() for element in s] for s in G1_supp]
    G2_supp_list=[[element.tolist() for element in s] for s in G2_supp]

            
    S in G1_supp_list
    
    G2.all_CI

    




