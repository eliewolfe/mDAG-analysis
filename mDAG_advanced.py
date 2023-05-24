from __future__ import absolute_import
import numpy as np
import itertools
# import networkx as nx
from sys import hexversion
if hexversion >= 0x3080000:
    from functools import cached_property
elif hexversion >= 0x3060000:
    from backports.cached_property import cached_property
else:
    cached_property = property

from radix import bitarray_to_int
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Visualization with matplotlib is disabled.")
try:
    import networkx as nx
    from networkx.algorithms.isomorphism import DiGraphMatcher
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

 
def mdag_to_int(ds_bitarray: np.ndarray, sc_bitarray: np.ndarray):
    sc_int = bitarray_to_int(sc_bitarray)
    ds_int = bitarray_to_int(ds_bitarray)
    #The directed structure is always a square matrix of size n^2.
    offset = 2**ds_bitarray.size
    # print(sc_int, ds_int, offset)
    return ds_int + (sc_int * offset)
    # return bitarray_to_int(np.vstack((sc_bitarray, ds_bitarray)))



class mDAG:
    def __init__(self, directed_structure_instance, simplicial_complex_instance, pp_restrictions=tuple()):
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
        self.visible_nodes = self.directed_structure_instance.visible_nodes
        self.latent_nodes = tuple(range(self.number_of_visible, self.simplicial_complex_instance.number_of_visible_plus_latent))
        self.nonsingleton_latent_nodes = tuple(range(self.number_of_visible, self.simplicial_complex_instance.number_of_visible_plus_nonsingleton_latent))
        self.total_number_of_nodes=len(self.nonsingleton_latent_nodes)+self.number_of_visible
        self.number_of_root_nodes = self.simplicial_complex_instance.number_of_latent

    @cached_property
    def as_string(self):
        return self.directed_structure_instance.as_string + '|' + self.simplicial_complex_instance.as_string

    # def extended_simplicial_complex(self):
    #  # Returns the simplicial complex extended to include singleton sets.
    #  return self.simplicial_complex + [(singleton,) for singleton in set(self.visible_nodes).difference(*self.simplicial_complex)]

    def __str__(self):
        return self.as_string

    def __repr__(self):
        return self.as_string

    @cached_property
    def as_graph(self):  # let DirectedStructure be a DAG initially without latents
        g = self.directed_structure_instance.as_networkx_graph.copy()
        g.add_nodes_from(self.nonsingleton_latent_nodes)
        g.add_edges_from(itertools.chain.from_iterable(
            zip(itertools.repeat(i), children) for i, children in
            zip(self.nonsingleton_latent_nodes, self.simplicial_complex_instance.compressed_simplicial_complex)))
        return g

    @cached_property
    def as_graph_with_singleton_latents(self):  # let DirectedStructure be a DAG initially without latents
        g = self.directed_structure_instance.as_networkx_graph.copy()
        g.add_nodes_from(self.latent_nodes)
        g.add_edges_from(itertools.chain.from_iterable(
            zip(itertools.repeat(i), children) for i, children in
            zip(self.latent_nodes, self.simplicial_complex_instance.extended_simplicial_complex_as_sets)))
        return g

    @cached_property
    def visible_automorphisms(self):
        discovered_automorphisms = set()
        for mapping in DiGraphMatcher(self.as_graph_with_singleton_latents,
                                      self.as_graph_with_singleton_latents).isomorphisms_iter():
            discovered_automorphisms.add(partsextractor(mapping, self.visible_nodes))
        return tuple(discovered_automorphisms)

    @cached_property
    def automorphic_order(self):
        return len(self.visible_automorphisms)

    @cached_property
    def as_extended_bit_array(self):
        #NOTE THAT THIS IS REVERSED RELATIVE TO UNIQUE ID!
        return np.vstack((self.directed_structure_instance.as_bit_square_matrix, self.simplicial_complex_instance.as_extended_bit_array))

    @cached_property
    def n_of_edges(self):
        return self.directed_structure_instance.number_of_edges

    @cached_property
    def relative_complexity_for_sat_solver(self):  # choose eqclass representative which minimizes this
        return (self.number_of_root_nodes, self.as_extended_bit_array.sum(), self.n_of_edges, self.simplicial_complex_instance.tally, -self.automorphic_order)
                
    @cached_property
    def parents_of_for_supports_analysis(self):
        return list(map(np.flatnonzero, self.as_extended_bit_array.T))

    @cached_property
    def parents_of(self):
        p=[]
        for visible_node in self.visible_nodes:
            p.append([])
        for edge in self.directed_structure_instance.edge_list:
            p[edge[1]].append(edge[0])
        latent_node=len(self.visible_nodes)
        for facet in self.simplicial_complex_instance.simplicial_complex_as_sets:
            for visible_node in facet:
                p[visible_node].append(latent_node)
            latent_node=latent_node+1
        return p

    @cached_property
    def unique_id(self):
        # Returns a unique identification number.
        return mdag_to_int(self.directed_structure_instance.as_bit_square_matrix, self.simplicial_complex_instance.as_bit_array)

    def __hash__(self):
        return self.unique_id

    def __eq__(self, other):
        return self.unique_id == other.unique_id

    @cached_property
    def unique_unlabelled_id(self):
        # Returns a unique identification number.
        return min(mdag_to_int(new_ds, new_sc) for (new_ds, new_sc) in zip(
            self.directed_structure_instance.bit_square_matrix_permutations,
            self.simplicial_complex_instance.bit_array_permutations
        ))

    @cached_property
    def skeleton_instance(self):
        if hasattr(self, 'variable_names'):
            return LabelledUndirectedGraph(self.variable_names, self.directed_structure_instance.edge_list_with_variable_names + self.simplicial_complex_instance.simplicial_complex_with_variable_names)
        else:
            return UndirectedGraph(self.directed_structure_instance.as_tuples + self.simplicial_complex_instance.as_tuples,
                               self.number_of_visible)

    @cached_property
    def skeleton(self):
        return self.skeleton_instance.as_edges_integer

    @cached_property
    def skeleton_unlabelled(self):
        return self.skeleton_instance.as_edges_unlabelled_integer

    @cached_property
    def cliques_with_common_ancestor_aside_from_external_visible(self):
        perfect_correlation_sets = set()
        for clique in self.skeleton_instance.cliques:
            if len(clique) > 2:
                common_ancestors = set(self.latent_nodes)
                for v in clique:
                    common_ancestors.intersection_update(nx.ancestors(self.as_graph_with_singleton_latents, v))
                for a in common_ancestors:
                    if len(set(self.as_graph_with_singleton_latents.successors(a)).intersection(clique))>0:
                        perfect_correlation_sets.add((self.fake_frozenset(clique),))
                        break
        return perfect_correlation_sets

    @cached_property
    def cliques_with_common_ancestor_aside_from_external_visible_unlabelled(self):
        return min(self._all_CI_like_unlabelled_generator('cliques_with_common_ancestor_aside_from_external_visible'))

    @cached_property
    def all_esep_plus_unlabelled(self):
        return self.all_esep_unlabelled + self.cliques_with_common_ancestor_aside_from_external_visible_unlabelled



    @cached_property
    def common_cause_connected_sets(self):
        return hypergraph_full_cleanup([self.directed_structure_instance.adjMat.descendantsplus_of(list(facet))
                                       for facet in self.simplicial_complex_instance.extended_simplicial_complex_as_sets])

    # @cached_property
    # def minimal_perfect_correlation_sets(self):
    #     admit_perfect_correlation_sets = set()
    #     for facet in self.simplicial_complex_instance.extended_simplicial_complex_as_sets:
    #         admit_perfect_correlation_sets.add(
    #             (self.fake_frozenset(
    #                 self.directed_structure_instance.adjMat.descendantsplus_of(list(facet))
    #             ),))
    #     return self.fake_frozenset(admit_perfect_correlation_sets)
    #
    # @cached_property
    # def minimal_perfect_correlation_sets_unlabelled(self):
    #     return min(self._all_CI_like_unlabelled_generator('minimal_perfect_correlation_sets'))
    #
    # @cached_property
    # def common_ancestors_sets_unlabelled_as_int(self):
    #     return self.fake_frozenset(self.common_cause_connected_sets)



    @staticmethod
    def _all_2_vs_any_partitions(variables_to_partition):
        # expect input in the form of a list
        length = len(variables_to_partition)
        subsetrange = range(length - 1)
        for x in subsetrange:
            for y in range(x + 1, length):
                complimentary_set = variables_to_partition[:x] + variables_to_partition[
                                                                 x + 1:y] + variables_to_partition[y + 1:]
                for r in subsetrange:
                    for complimentary_subset in itertools.combinations(complimentary_set, r):
                        yield variables_to_partition[x], variables_to_partition[y], complimentary_subset

    @staticmethod
    def fake_frozenset(stuff):
        return tuple(sorted(stuff))

    # We do not cache iterators, only their output, as iterators are consumed!
    @property
    def _all_CI_generator_numeric(self):
        for x, y, Z in self._all_2_vs_any_partitions(self.visible_nodes):
            if nx.d_separated(self.as_graph, {x}, {y}, set(Z)):
                yield (self.fake_frozenset([x, y]), self.fake_frozenset(Z))

    @cached_property
    def d_separable_pairs(self):
        discovered_pairs = set()
        for x, y, Z in self._all_2_vs_any_partitions(self.visible_nodes):
            if nx.d_separated(self.as_graph, {x}, {y}, set(Z)):
                discovered_pairs.add(self.fake_frozenset([x, y]))
        return discovered_pairs

    @cached_property
    def d_unseparable_pairs(self):
        return set(itertools.combinations(self.visible_nodes, 2)).difference(self.d_separable_pairs)

    @cached_property
    def all_CI_numeric(self):
        return set(self._all_CI_generator_numeric)

    @cached_property
    def CI_count(self):
        return len(self.all_CI_numeric)

    def _all_CI_like_unlabelled_generator(self, attribute):
        for perm in itertools.permutations(self.visible_nodes):
            yield self.fake_frozenset(
                tuple(
                    self.fake_frozenset(partsextractor(perm, variable_set))
                    for variable_set in relation)
                for relation in self.__getattribute__(attribute))

    def convert_to_named_set_of_tuples_of_tuples(self, attribute):
        if hasattr(self, 'variable_names'):
            return set(
                tuple(
                    self.fake_frozenset(partsextractor(self.variable_names, variable_set))
                    for variable_set in relation)
                for relation in self.__getattribute__(attribute))
        else:
            return self.__getattribute__(attribute)

    @cached_property
    def all_CI(self):
        return self.convert_to_named_set_of_tuples_of_tuples('all_CI_numeric')

    @cached_property
    def all_CI_unlabelled(self):
        return min(self._all_CI_like_unlabelled_generator('all_CI_numeric'))

    @property
    def _all_e_sep_generator_numeric(self):
        for r in range(self.number_of_visible - 1):
            for to_delete in itertools.combinations(self.visible_nodes, r):
                graph_copy = self.as_graph.copy()  # Don't forget to copy!
                graph_copy.remove_nodes_from(to_delete)
                remaining = set(self.visible_nodes).difference(to_delete)
                for x, y, Z in self._all_2_vs_any_partitions(tuple(remaining)):
                    if nx.d_separated(graph_copy, {x}, {y}, set(Z)):
                        yield (self.fake_frozenset([x, y]), self.fake_frozenset(Z), self.fake_frozenset(to_delete))
    @cached_property
    def all_esep_numeric(self):
        return set(self._all_e_sep_generator_numeric)

    @cached_property
    def all_esep(self):
        # return set(self._all_e_sep_generator_numeric)
        return self.convert_to_named_set_of_tuples_of_tuples('all_esep_numeric')

    @cached_property
    def all_esep_unlabelled(self):
        return min(self._all_CI_like_unlabelled_generator('all_esep_numeric'))

    # @property
    # def _e_separable_pairs(self):
    #     for to_delete in itertools.combinations(self.visible_nodes, self.number_of_visible-2):
    #         graph_copy = self.as_graph.copy()  # Don't forget to copy!
    #         graph_copy.remove_nodes_from(to_delete)
    #         remaining = set(self.visible_nodes).difference(to_delete)
    #         for x, y, Z in self._all_2_vs_any_partitions(tuple(remaining)):
    #             if nx.d_separated(graph_copy, {x}, {y}, set(Z)):
    #                 yield self.fake_frozenset([x, y])
    @cached_property
    def e_separable_pairs(self):
        return set(itertools.combinations(self.visible_nodes, 2)).difference(self.skeleton_instance.as_edges)

    @cached_property
    def interesting_via_e_sep_theorem(self):
        return not self.d_unseparable_pairs.issubset(self.skeleton_instance.as_edges)

    # @methodtools.lru_cache(maxsize=None, typed=False)
    # def support_testing_instance(self, n):
    #     return SupportTesting(self.parents_of_for_supports_analysis,
    #                           np.broadcast_to(2, self.number_of_visible),
    #                           n)

    #Let's use smart support testing for both smart and not.
    #@methodtools.lru_cache(maxsize=None, typed=False)
    
    # def smart_support_testing_instance_card_3(self, n):
    #     return SmartSupportTesting(self.parents_of_for_supports_analysis,
    #                                (3,2,2),
    #                                n, self.all_esep
    #                                )
    # def smart_infeasible_supports_n_events_card_3(self, n, **kwargs):
    #       return self.fake_frozenset(self.smart_support_testing_instance_card_3(n).unique_infeasible_supports_beyond_esep_as_integers(**kwargs, name='mgh', use_timer=False))
    #
    # def smart_support_testing_instance_card_4(self, n):
    #     return SmartSupportTesting(self.parents_of_for_supports_analysis,
    #                                (4,2,2),
    #                                n, self.all_esep
    #                                )
    # def smart_infeasible_supports_n_events_card_4(self, n, **kwargs):
    #       return self.fake_frozenset(self.smart_support_testing_instance_card_4(n).unique_infeasible_supports_beyond_esep_as_integers(**kwargs, name='mgh', use_timer=False))
    #

    @methodtools.lru_cache(maxsize=None, typed=False)
    def support_testing_instance_binary(self, n):
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
    def support_testing_instance(self, observed_cardinalities, n):
        if not hasattr(self, 'restricted_perfect_predictions_numeric'):
            self.restricted_perfect_predictions_numeric = tuple()
        return SmartSupportTesting(parents_of=self.parents_of_for_supports_analysis,
                                   observed_cardinalities=observed_cardinalities,
                                   nof_events=n,
                                   esep_relations=self.all_esep_numeric,
                                   pp_relations=self.restricted_perfect_predictions_numeric,
                                   visible_automorphisms=self.visible_automorphisms
                                   )
    def infeasible_4222_supports_n_events(self,n,**kwargs):
        return tuple(self.support_testing_instance((4, 2, 2, 2),n).unique_infeasible_supports_as_expanded_integers(**kwargs, name='mgh', use_timer=False))
    def infeasible_binary_supports_n_events(self, n, **kwargs):
        return tuple(self.support_testing_instance_binary(n).unique_infeasible_supports_as_expanded_integers(**kwargs, name='mgh', use_timer=False))
    def infeasible_binary_supports_n_events_beyond_esep(self, n, **kwargs):
        return tuple(self.support_testing_instance_binary(n).unique_infeasible_supports_beyond_esep_as_expanded_integers(**kwargs, name='mgh', use_timer=False))
    def infeasible_binary_supports_n_events_beyond_dsep(self, n, **kwargs):
        return tuple(self.support_testing_instance_binary(n).unique_infeasible_supports_beyond_dsep_as_expanded_integers(**kwargs, name='mgh', use_timer=False))

    def infeasible_binary_supports_n_events_as_matrices(self, n, **kwargs):
        return self.support_testing_instance_binary(n).unique_infeasible_supports_as_expanded_matrices(**kwargs, name='mgh', use_timer=False)
    def infeasible_binary_supports_n_events_beyond_esep_as_matrices(self, n, **kwargs):
        return self.support_testing_instance_binary(n).unique_infeasible_supports_beyond_esep_as_expanded_matrices(**kwargs, name='mgh', use_timer=False)
    def infeasible_binary_supports_n_events_beyond_dsep_as_matrices(self, n, **kwargs):
        return self.support_testing_instance_binary(n).unique_infeasible_supports_beyond_dsep_as_expanded_matrices(**kwargs, name='mgh', use_timer=False)
    def representative_infeasible_binary_supports_n_events_as_matrices(self, n, **kwargs):
        return self.support_testing_instance_binary(n).unique_infeasible_supports_as_compressed_matrices(**kwargs, name='mgh', use_timer=False)
    def representative_infeasible_binary_supports_n_events_beyond_esep_as_matrices(self, n, **kwargs):
        return self.support_testing_instance_binary(n).unique_infeasible_supports_beyond_esep_as_compressed_matrices(**kwargs, name='mgh', use_timer=False)
    def representative_infeasible_binary_supports_n_events_beyond_dsep_as_matrices(self, n, **kwargs):
        return self.support_testing_instance_binary(n).unique_infeasible_supports_beyond_dsep_as_compressed_matrices(**kwargs, name='mgh', use_timer=False)

    def infeasible_binary_supports_n_events_unlabelled(self, n, **kwargs):
        return tuple(self.support_testing_instance_binary(n).unique_infeasible_supports_as_integers_unlabelled(**kwargs, name='mgh', use_timer=False))
    def infeasible_binary_supports_n_events_beyond_esep_unlabelled(self, n, **kwargs):
        return tuple(self.support_testing_instance_binary(n).unique_infeasible_supports_beyond_esep_as_integers_unlabelled(**kwargs, name='mgh', use_timer=False))


    def no_infeasible_binary_supports_beyond_esep(self, n, **kwargs):
        return self.support_testing_instance_binary(n).no_infeasible_supports_beyond_esep(**kwargs, name='mgh', use_timer=False)
    def no_infeasible_binary_supports_beyond_esep_up_to(self, max_n, **kwargs):
        return all(self.no_infeasible_binary_supports_beyond_esep(n, **kwargs) for n in range(2, max_n + 1))
    def no_infeasible_binary_supports_beyond_dsep(self, n, **kwargs):
        return self.support_testing_instance_binary(n).no_infeasible_supports_beyond_dsep(**kwargs, name='mgh', use_timer=False)
    def no_infeasible_binary_supports_beyond_dsep_up_to(self, max_n, **kwargs):
        return all(self.no_infeasible_binary_supports_beyond_dsep(n, **kwargs) for n in range(2, max_n + 1))
    
    def no_infeasible_4222_supports_beyond_dsep(self,n,**kwargs):
        return self.support_testing_instance((4, 2, 2, 2),n).no_infeasible_supports_beyond_dsep(**kwargs, name='mgh', use_timer=False)
    def no_infeasible_4222_supports_beyond_dsep_up_to(self, max_n, **kwargs):
        return all(self.no_infeasible_4222_supports_beyond_dsep(n, **kwargs) for n in range(2, max_n + 1))
    
    def no_infeasible_8222_supports_beyond_dsep(self,n,**kwargs):
        return self.support_testing_instance((8, 2, 2, 2),n).no_infeasible_supports_beyond_dsep(**kwargs, name='mgh', use_timer=False)
    def no_infeasible_8222_supports_beyond_dsep_up_to(self, max_n, **kwargs):
        return all(self.no_infeasible_8222_supports_beyond_dsep(n, **kwargs) for n in range(2, max_n + 1))
    
    def no_infeasible_6222_supports_beyond_dsep(self,n,**kwargs):
        return self.support_testing_instance((6, 2, 2, 2),n).no_infeasible_supports_beyond_dsep(**kwargs, name='mgh', use_timer=False)
    def no_infeasible_6222_supports_beyond_dsep_up_to(self, max_n, **kwargs):
        return all(self.no_infeasible_6222_supports_beyond_dsep(n, **kwargs) for n in range(2, max_n + 1))
    
    
    def no_esep_beyond_dsep(self, n):
        return not self.support_testing_instance_binary(n).interesting_due_to_esep
    def no_esep_beyond_dsep_up_to(self, max_n):
        return all(self.no_esep_beyond_dsep(n) for n in range(2, max_n + 1))


    def infeasible_binary_supports_beyond_esep_up_to(self, max_n, **kwargs):
        return np.fromiter(itertools.chain.from_iterable(
            (self.infeasible_binary_supports_n_events_beyond_esep(n, **kwargs) for n in range(2, max_n + 1))),
            dtype=int)
    def infeasible_binary_supports_up_to(self, max_n, **kwargs):
        return np.fromiter(itertools.chain.from_iterable(
            (self.infeasible_binary_supports_n_events(n, **kwargs) for n in range(2, max_n + 1))),
            dtype=int)

    def infeasible_binary_supports_beyond_esep_as_matrices_up_to(self, max_n, **kwargs):
        return list(itertools.chain.from_iterable((self.infeasible_binary_supports_n_events_beyond_esep_as_matrices(n, **kwargs).astype(int) for n in range(2, max_n + 1))))
    def infeasible_binary_supports_beyond_dsep_as_matrices_up_to(self, max_n, **kwargs):
        return list(itertools.chain.from_iterable((self.infeasible_binary_supports_n_events_beyond_dsep_as_matrices(n, **kwargs).astype(int) for n in range(2, max_n + 1))))
    def infeasible_binary_supports_as_matrices_up_to(self, max_n, **kwargs):
        return list(itertools.chain.from_iterable((self.infeasible_binary_supports_n_events(n, **kwargs).astype(int) for n in range(2, max_n + 1))))

    def no_infeasible_binary_supports_n_events(self, n, **kwargs):
        return self.support_testing_instance_binary(n).no_infeasible_supports(**kwargs, name='mgh', use_timer=False)
    def no_infeasible_binary_supports_up_to(self, max_n, **kwargs):
        return all(self.no_infeasible_binary_supports_n_events(n, **kwargs) for n in range(2, max_n + 1))

    # def no_infeasible_binary_supports_up_to(self, max_n, **kwargs):
    #     return all(len(self.infeasible_binary_supports_n_events(n, **kwargs))==0 for n in range(2,max_n+1))




    @cached_property
    def droppable_edges(self):
        candidates = [tuple(pair) for pair in self.directed_structure_instance.edge_list if
                      any(set(pair).issubset(hyperedge) for hyperedge in self.simplicial_complex_instance.compressed_simplicial_complex)]
        candidates = [pair for pair in candidates if set(self.as_graph.predecessors(pair[0])).issubset(
            self.as_graph.predecessors(pair[1]))]  # as_graph includes both visible at latent parents
        return candidates

    @cached_property
    def equivalent_under_edge_dropping(self):
        equivalent_mdags = []
        for edges_to_drop in powerset(self.droppable_edges):
            if len(edges_to_drop):
                equivalent_mdags.append(mDAG(DirectedStructure(
                    self.directed_structure_instance.as_set_of_tuples.difference(edges_to_drop),
                    self.number_of_visible),
                     self.simplicial_complex_instance))
        return equivalent_mdags





    @property
    def generate_weaker_mDAG_HLP(self):
        if self.droppable_edges:
            new_bit_square_matrix = self.directed_structure_instance.as_bit_square_matrix.copy()
            new_bit_square_matrix[tuple(np.asarray(self.droppable_edges, dtype=int).reshape((-1,2)).T)] = False
            yield mdag_to_int(new_bit_square_matrix, self.simplicial_complex_instance.as_bit_array)

    @property
    def generate_slightly_weaker_mDAGs_HLP(self):
        for droppable_edge in self.droppable_edges:
            new_bit_square_matrix = self.directed_structure_instance.as_bit_square_matrix.copy()
            new_bit_square_matrix[droppable_edge] = False
            # new_bit_square_matrix[tuple(np.asarray(self.droppable_edges, dtype=int).reshape((-1,2)).T)] = False
            yield mdag_to_int(new_bit_square_matrix, self.simplicial_complex_instance.as_bit_array)


    @staticmethod
    def _all_bipartitions(variables_to_partition):
        #Yields tuples of frozensets
        # expect input in the form of a list
        length = len(variables_to_partition)
        integers = range(length)
        subsetrange = range(1, length)
        for r in subsetrange:
            for ints_to_mask in map(list, itertools.combinations(integers, r)):
                mask = np.ones(length, dtype=np.bool)
                mask[ints_to_mask] = False
                yield (frozenset(itertools.compress(variables_to_partition, mask.tolist())),
                       frozenset(itertools.compress(variables_to_partition, np.logical_not(mask).tolist())))

    # def setpredecessorsplus(self, X):
    #     result = set(X)
    #     for x in X:
    #         result.update(self.as_graph.predecessors(x))
    #     return result
    @cached_property
    def all_parentsplus_list(self):
        return [frozenset({}).union(v_par, l_par) for v_par,l_par in zip(
                self.directed_structure_instance.observable_parentsplus_list,
                self.simplicial_complex_instance.latent_parents_list)]

    def set_predecessors(self, X):
        # return frozenset().union(*partsextractor(self.directed_structure_instance.observable_parentsplus_list, X))
        return frozenset(itertools.chain.from_iterable(partsextractor(self.all_parentsplus_list, X)))
    def singleton_predecessors(self, x):
        return partsextractor(self.all_parentsplus_list, x)
    
    def singleton_visible_predecessors(self, x):
        return partsextractor(self.directed_structure_instance.observable_parentsplus_list, x)

    def children(self, node):
        c = []
        for v in self.visible_nodes:
            if node in self.singleton_visible_predecessors(v) and node != v:
                c.append(v)
        return c
        
    def descendants(self, node):
        return nx.descendants(self.as_graph, node)

    @cached_property
    def splittable_faces(self):
        candidates = itertools.chain.from_iterable(map(self._all_bipartitions, self.simplicial_complex_instance.compressed_simplicial_complex))
        candidates = [(C, D) for C, D in candidates if
                      all(self.set_predecessors(C).issubset(self.singleton_predecessors(d)) for d in D)]
        return candidates

    @cached_property
    def eventually_splittable_faces(self):
        candidates = itertools.chain.from_iterable(map(self._all_bipartitions, self.simplicial_complex_instance.compressed_simplicial_complex))
        result = []
        for C, D in candidates:
            if all((self.set_predecessors(C).difference(C)).issubset(
                self.singleton_predecessors(d).difference(D)) for d in D):
                result.append((C, D))
        return result

    @cached_property
    def has_no_eventually_splittable_face(self):
        return len(self.eventually_splittable_faces) == 0
    
    def set_visible_predecessors(self,X):
        return frozenset(itertools.chain.from_iterable(partsextractor(self.directed_structure_instance.observable_parentsplus_list, X)))
    def singleton_visible_predecessors(self, x):
        return partsextractor(self.directed_structure_instance.observable_parentsplus_list, x)

    def two_layer_circuit(self):
        for node in self.visible_nodes:
            if len(self.children(node))>0 and len(self.singleton_predecessors(node))>1:
                return False
        return True

    @cached_property
    def numerical_districts(self):
        return self.simplicial_complex_instance.districts

    @cached_property
    def districts(self):
        #Districts are returned as a list of sets
        if hasattr(self, 'variable_names'):
            return [partsextractor(self.variable_names, district) for district in self.numerical_districts]
        else:
            return self.numerical_districts

        
    def set_district(self, X):
        return frozenset(itertools.chain.from_iterable((district for district in self.districts if not district.isdisjoint(X))))
    def singleton_district(self, x):
        return frozenset(itertools.chain.from_iterable((district for district in self.districts if x in district)))



    #
    #
    # @cached_property
    # def districts_arbitrary_names(self):
    #     if hasattr(self.simplicial_complex_instance, 'variable_names'):
    #         return self.simplicial_complex_instance.translated_districts
    #     else:
    #         return self.simplicial_complex_instance.districts


        
    def networkx_plot_mDAG(self):
        G=nx.DiGraph()
        G.add_nodes_from(self.visible_nodes)
        G.add_nodes_from(self.nonsingleton_latent_nodes)
        pos=nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, nodelist=self.visible_nodes, node_color="tab:blue")
        nx.draw_networkx_nodes(G, pos, nodelist=self.nonsingleton_latent_nodes, node_color="tab:red")
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G,pos,edgelist=self.directed_structure_instance.edge_list)
        latent_edges=[]
        f=0
        for facet in self.simplicial_complex_instance.simplicial_complex_as_sets:
            for element in facet:
                latent_edges.append((self.latent_nodes[f],element))
            f=f+1
        nx.draw_networkx_edges(G,pos,edgelist=latent_edges)
        plt.show()
        
    def plot_mDAG_indicating_one_node(self,node):
        G=nx.DiGraph()
        G.add_nodes_from(self.visible_nodes)
        G.add_nodes_from(self.nonsingleton_latent_nodes)
        pos=nx.spring_layout(G)
        v=self.visible_nodes.copy()
        nx.draw_networkx_nodes(G, pos, nodelist=v.remove(node), node_color="tab:blue")
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color="tab:green")
        nx.draw_networkx_nodes(G, pos, nodelist=self.nonsingleton_latent_nodes, node_color="tab:red")
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, edgelist=self.directed_structure_instance.edge_list)
        latent_edges=[]
        f=0
        for facet in self.simplicial_complex_instance.simplicial_complex_as_sets:
            for element in facet:
                latent_edges.append((self.latent_nodes[f],element))
            f=f+1
        nx.draw_networkx_edges(G, pos, edgelist=latent_edges)
        plt.show()
    
    @cached_property
    def weak_splittable_faces(self):
        candidates = itertools.chain.from_iterable(map(self._all_bipartitions, self.simplicial_complex_instance.compressed_simplicial_complex))
        # candidates = [(C,D) for C,D in candidates if all(set(self.as_graph.predecessors(c).issubset(self.as_graph.predecessors(d)) for c in C for d in D)]
        simp_complex=[]
        for s in self.simplicial_complex_instance.compressed_simplicial_complex:
            simp_complex.append(tuple(s))
        candidates = [(C, D) for C, D in candidates if
                      all(self.set_visible_predecessors(C).issubset(self.singleton_visible_predecessors(d)) for d in D) and
                      all(c not in itertools.chain.from_iterable(set(simp_complex)-{tuple(set(C)|set(D))}) for c in C)]
        # print(candidates)
        return candidates
    
# =============================================================================
#     G_ev_prop=mDAG(DirectedStructure([(0,3),(0,1),(2,1),(2,3),(4,0),(4,3),(4,1),(5,2),(5,3),(5,1)],6),Hypergraph([(0,1,2,3),(4,),(5,)],6))
#     for mDAG in G_ev_prop.generate_weaker_mDAGs_FaceSplitting('weak'):
#         print (mDAG)
#     candidates = itertools.chain.from_iterable(map(G_ev_prop._all_bipartitions, G_ev_prop.simplicial_complex_instance.compressed_simplicial_complex))
#     [(C, D) for C, D in candidates if all(G_ev_prop.set_visible_predecessors(C).issubset(G_ev_prop.singleton_visible_predecessors(d)) for d in D) and all(c not in itertools.chain.from_iterable(set(G_ev_prop.simplicial_complex_instance.compressed_simplicial_complex)-{tuple(set(C)|set(D))}) for c in C)]  
#     [(C, D) for C, D in candidates]
#     C=frozenset({0,2})
#     D=frozenset({1,3})
#     all(c not in itertools.chain.from_iterable(set(G_ev_prop.simplicial_complex_instance.compressed_simplicial_complex)-{tuple(set(C)|set(D))}) for c in C)
#     
#     G_ev_prop3=mDAG(DirectedStructure([(0,1),(0,2),(1,2)],3),Hypergraph([(0,1),(1,2)],3))
#     for mDAG in G_ev_prop3.generate_weaker_mDAGs_FaceSplitting('weak'):
#         print(mDAG)
#     
# =============================================================================
# =============================================================================
#     G_ev_prop2=mDAG(DirectedStructure([(0,3),(0,2),(2,3),(3,1)],4),Hypergraph([(0,),(1,),(2,3)],4))
#     for mDAG in G_ev_prop2.generate_weaker_mDAGs_FaceSplitting('weak'):
#         print(mDAG)
# =============================================================================

    
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
            yield mdag_to_int(
                self.directed_structure_instance.as_bit_square_matrix,
                Hypergraph(new_simplicial_complex, self.number_of_visible).as_bit_array)
    
    
    def generate_weaker_mDAGs_FaceSplitting(self, version):
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
            yield mdag_to_int(
                self.directed_structure_instance.as_bit_square_matrix,
                Hypergraph(new_simplicial_complex, self.number_of_visible).as_bit_array)

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
            yield mdag_to_int(
                self.directed_structure_instance.as_bit_square_matrix,
                Hypergraph(new_simplicial_complex, self.number_of_visible).as_bit_array)

    #TODO: slightly speed up this by avoiding Hypergraph creation. That is, directly modify the bit array.
    #Or, if we are really fancy, we can modify the bits of the unique_id itself!!



    
    # @property
    # def districts_arbitrary_names(self):
    #     districts_translated=[]
    #     for d in self.districts:
    #         d_translated=set()
    #         for node in d:
    #             node_translated=list(self.simplicial_complex_instance.translation_dict.keys())[list(self.simplicial_complex_instance.translation_dict.values()).index(node)]
    #             d_translated.add(node_translated)
    #         districts_translated.append(d_translated)
    #     return districts_translated


    
    # def subgraph(self, list_of_nodes):
    #     new_edges=[]
    #     for edge in self.directed_structure_instance.edge_list:
    #         if edge[0] in list_of_nodes and edge[1] in list_of_nodes:
    #             new_edges.append(edge)
    #     new_hypergraph=[]
    #     for hyperedge in self.simplicial_complex_instance.simplicial_complex:
    #         new_hyperedge=hyperedge
    #         for node in hyperedge:
    #             if node not in list_of_nodes:
    #                 new_hyperedge_list=list(new_hyperedge)
    #                 new_hyperedge_list.remove(node)
    #                 new_hyperedge=tuple(new_hyperedge_list)
    #         subset_of_already_hyp=False
    #         for already_hyp in new_hypergraph:
    #             if set(new_hyperedge).issubset(set(already_hyp)) and new_hyperedge!=already_hyp:
    #                 subset_of_already_hyp=True
    #             if set(already_hyp).issubset(set(new_hyperedge)):
    #                 new_hypergraph.remove(already_hyp)
    #         if not subset_of_already_hyp:
    #             new_hypergraph.append(new_hyperedge)
    #     return mDAG(LabelledDirectedStructure(list_of_nodes,new_edges), LabelledHypergraph(list_of_nodes,new_hypergraph))

    #Elie: I have integrated subgraph into both labelled directed structure and labelled hypergraph.
    def subgraph(self, list_of_nodes):
        return mDAG(
            LabelledDirectedStructure(list_of_nodes, self.directed_structure_instance.edge_list),
            LabelledHypergraph(list_of_nodes, self.simplicial_complex_instance.simplicial_complex_as_sets)
        )

  

    # def closure_Marina(self, B):
    #     list_B=[set(self.visible_nodes)]
    #     graph=self.subgraph(list_B[0])
    #     next_B=set()
    #     for element in B:
    #         for dist in graph.districts_arbitrary_names:
    #             if element in dist:
    #                 next_B=set(next_B).union(dist)
    #     list_B.append(next_B)
    #     graph=self.subgraph(list_B[1])
    #     next_B=set()
    #     for element in B:
    #         next_B=set(list(next_B)+list(nx.ancestors(graph.directed_structure_instance.as_networkx_graph_arbitrary_names,element))+[element])
    #     list_B.append(next_B)
    #     i=2
    #     while any([list_B[i]!=list_B[i-1],list_B[i]!=list_B[i-2]]):
    #         graph=self.subgraph(list_B[-1])
    #         next_B=set()
    #         for element in B:
    #             for dist in graph.districts_arbitrary_names:
    #                 if element in dist:
    #                     next_B=set(next_B).union(dist)
    #         list_B.append(next_B)
    #         i=i+1
    #         graph=self.subgraph(list_B[-1])
    #         next_B=set()
    #         for element in B:
    #             next_B=set(list(next_B)+list(nx.ancestors(graph.directed_structure_instance.as_networkx_graph_arbitrary_names,element))+[element])
    #         list_B.append(next_B)
    #         i=i+1
    #     return list_B[-1]

    def set_closure(self, X_int_or_list, return_bidirectedQ=False):
        return numeric_closure(core_B=X_int_or_list,
                               n=self.number_of_visible,
                               ds_adjmat= self.directed_structure_instance.as_bit_square_matrix_plus_eye,
                               sc_adjmat= self.simplicial_complex_instance.as_bidirected_adjmat,
                               return_bidirectedQ= return_bidirectedQ)

    # @cached_property
    # def singeleton_closures_Marina(self):
    #     return [frozenset(self.closure_Marina([i])) for i in range(self.number_of_visible)]

    @cached_property
    def singeleton_closures(self):
        # list_of_closures = []
        # for i in range(self.number_of_visible):
        #     closure_Wolfe = frozenset(self.set_closure([i]))
        #     closure_Marina = self.singeleton_closures_Marina[i]
        #     if not closure_Wolfe == closure_Marina:
        #         print("Possible bug found in closure calculations!")
        #         print("mDAG: ", self.__repr__())
        #         print("node: ", i)
        #         print("closure_Marina: ", closure_Marina)
        #         print("closure_Wolfe: ", closure_Wolfe)
        #     list_of_closures.append(closure_Wolfe)
        # return list_of_closures
        return [frozenset(self.set_closure([i])) for i in range(self.number_of_visible)]



    # def are_densely_connected_Marina(self, node1, node2):
    #     if node1 in self.set_visible_predecessors(self.singeleton_closures[node2]):
    #         return True
    #     if node2 in self.set_visible_predecessors(self.singeleton_closures[node1]):
    #         return True
    #     double_closure = frozenset(self.set_closure([node1, node2]))
    #     for district in self.subgraph(double_closure).districts_arbitrary_names:
    #         if double_closure.issubset(district):
    #             return True
    #     return False

    # def are_densely_connected_Marina(self, node1, node2):
    #     if node1 in self.set_visible_predecessors(self.singeleton_closures_Marina[node2]):
    #         return True
    #     if node2 in self.set_visible_predecessors(self.singeleton_closures_Marina[node1]):
    #         return True
    #     combined_closure = list(set(itertools.chain.from_iterable([
    #         self.singeleton_closures_Marina[node1],
    #         self.singeleton_closures_Marina[node2]
    #     ])))
    #     # if self.set_closure([node1, node2], return_bidirectedQ=True)[-1]:
    #     if is_this_subadjmat_densely_connected(self.simplicial_complex_instance.as_bidirected_adjmat, combined_closure):
    #         return True
    #     else:
    #         return False

    def are_densely_connected(self, node1, node2):
        if node1 in self.set_visible_predecessors(self.singeleton_closures[node2]):
            return True
        if node2 in self.set_visible_predecessors(self.singeleton_closures[node1]):
            return True
        # combined_closure = list(set(itertools.chain.from_iterable([
        #     self.singeleton_closures[node1],
        #     self.singeleton_closures[node2]
        # ])))
        if self.set_closure([node1, node2], return_bidirectedQ=True)[-1]:
        # if is_this_subadjmat_densely_connected(self.simplicial_complex_instance.as_bidirected_adjmat, combined_closure):
            return True
        else:
            return False

    # #BUGFINDING!!
    # def are_densely_connected(self, node1, node2):
    #     result_Marina = self.are_densely_connected_Marina(node1, node2)
    #     result_Wolfe = self.are_densely_connected_Wolfe(node1, node2)
    #     if result_Marina != result_Wolfe:
    #         print("Possible bug found in densely connected test!")
    #         print("mDAG: ", self.__repr__())
    #         print("node pair: ", (node1, node2))
    #         print("Marina says: ", result_Marina)
    #         print("Elie says: ", result_Wolfe)
    #     return result_Wolfe





# Evans 2021: It is possible to have a distribution with 2 variables identical to one another and independent of all others iff they are densely connected
# So, if node1 and node2 are densely connected in G1 but not in G2, we know that G1 is NOT equivalent to G2.

    @cached_property
    def all_densely_connected_pairs_numeric(self):
        return np.array([nodepair for nodepair in itertools.combinations(self.visible_nodes, 2) if self.are_densely_connected(*nodepair)], dtype=int)
    @cached_property
    def all_densely_connected_pairs(self):
        if hasattr(self, 'variable_names'):
            [partsextractor(self.variable_names, nodepair) for nodepair in self.all_densely_connected_pairs_numeric]
            return set(map(self.fake_frozenset, np.take(self.variable_names, self.all_densely_connected_pairs_numeric)))
        else:
            return set(map(self.fake_frozenset, self.numerical_districts))

    def _all_densely_connected_pairs_unlabelled_generator(self):
        for perm in itertools.permutations(self.visible_nodes):
            yield self.fake_frozenset(map(self.fake_frozenset, np.take(perm, self.all_densely_connected_pairs_numeric)))
            
    @cached_property
    def all_densely_connected_pairs_unlabelled(self):
        return min(self._all_densely_connected_pairs_unlabelled_generator())


    @cached_property
    def fundamental_graphQ(self):
        # Implement three conditions
        common_cause_districts = self.simplicial_complex_instance.nonsingleton_districts
        # district_lengths = np.fromiter(map(len, self.districts), np.int_)
        # # district_lengths = np.asarray(list(map(len, self.districts)))
        # common_cause_district_positions = np.flatnonzero(district_lengths > 1)
        if len(common_cause_districts) != 1:
            return False
        # district_vertices = set(self.districts[common_cause_district_positions[0]])
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
    def latent_free_graphQ(self):
        if self.simplicial_complex_instance.number_of_nonsingleton_latent==0:
            return True
        return False

    def marginalize_node(self,marginalized_node):
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
    
    def fix_to_point_distribution(self, node):  #returns a smaller mDAG
        return self.subgraph(self.visible_nodes[:node]+self.visible_nodes[(node+1):])



class Unlabelled_mDAG(mDAG):
    def __init__(self, directed_structure_instance, simplicial_complex_instance):
        super().__init__(directed_structure_instance, simplicial_complex_instance)
    def __hash__(self):
        return self.unique_unlabelled_id
    def __eq__(self, other):
        return self.unique_id == other.unique_unlabelled_id

# class labelled_mDAG(mDAG):
#     def __init__(self, labelled_directed_structure_instance, labelled_simplicial_complex_instance):
#         assert labelled_directed_structure_instance.variable_names == labelled_simplicial_complex_instance.variable_names, "Name conflict."
#         self.variable_names = labelled_directed_structure_instance.variable_names
#         super().__init__(labelled_directed_structure_instance, labelled_simplicial_complex_instance)

if __name__ == '__main__':
    G_Evans=mDAG(DirectedStructure([(0,2)],3),Hypergraph([(0,1),(0,2)],3))
    G_triangle = mDAG(DirectedStructure([], 3),
                   Hypergraph([(0, 1), (0, 2), (1, 2)], 3))
    print(G_triangle.cliques_with_common_ancestor_aside_from_external_visible_unlabelled)
