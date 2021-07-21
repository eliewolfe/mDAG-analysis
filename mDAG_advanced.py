from __future__ import absolute_import
import numpy as np
import itertools
from sys import hexversion
if hexversion >= 0x3080000:
    from functools import cached_property
elif hexversion >= 0x3060000:
    from backports.cached_property import cached_property
else:
    cached_property = property

from radix import from_bits
import networkx as nx
from utilities import partsextractor
from collections import defaultdict
from supports import SupportTesting
from esep_support import SmartSupportTesting
from hypergraphs import undirected_graph, hypergraph
import methodtools

def mdag_to_int(ds_bitarray, sc_bitarray):
    return from_bits(np.vstack((sc_bitarray, ds_bitarray)).ravel()).astype(np.ulonglong).tolist()

class mDAG:
    def __init__(self, directed_structure_instance, simplicial_complex_instance):
        self.directed_structure_instance = directed_structure_instance
        self.simplicial_complex_instance = simplicial_complex_instance
        assert directed_structure_instance.number_of_visible == simplicial_complex_instance.number_of_visible, 'Different number of nodes in directed structure vs simplicial complex.'
        self.number_of_visible = directed_structure_instance.number_of_visible
        self.visible_nodes = directed_structure_instance.visible_nodes
        self.latent_nodes = tuple(range(self.number_of_visible, self.simplicial_complex_instance.number_of_visible_plus_latent))
        self.nonsingleton_latent_nodes = tuple(range(self.number_of_visible, self.simplicial_complex_instance.number_of_visible_plus_nonsingleton_latent))


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
    def as_graph(self):  # let directed_structure be a DAG initially without latents
        g = self.directed_structure_instance.as_networkx_graph.copy()
        g.add_nodes_from(self.nonsingleton_latent_nodes)
        g.add_edges_from(itertools.chain.from_iterable(
            zip(itertools.repeat(i), children) for i, children in
            zip(self.nonsingleton_latent_nodes, self.simplicial_complex_instance.compressed_simplicial_complex)))
        return g



    @cached_property
    def as_extended_bit_array(self):
        #NOTE THAT THIS IS REVERSED RELATIVE TO UNIQUE ID!
        return np.vstack((self.directed_structure_instance.as_bit_square_matrix, self.simplicial_complex_instance.as_extended_bit_array))

    @cached_property
    def relative_complexity_for_sat_solver(self):  # choose eqclass representative which minimizes this
        return self.as_extended_bit_array.sum()

    @cached_property
    def parents_of_for_supports_analysis(self):
        return list(map(np.flatnonzero, self.as_extended_bit_array.T))



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
        return undirected_graph(self.directed_structure_instance.as_tuples + self.simplicial_complex_instance.as_tuples)

    @cached_property
    def skeleton(self):
        return self.skeleton_instance.as_edges_integer

    @cached_property
    def skeleton_unlabelled(self):
        return self.skeleton_instance.as_edges_unlabelled_integer






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

    # We do not cache iterators, only their output, as iterators are consumed!
    @property
    def _all_CI_generator(self):
        for x, y, Z in self._all_2_vs_any_partitions(self.visible_nodes):
            if nx.d_separated(self.as_graph, {x}, {y}, set(Z)):
                yield frozenset([x, y]), frozenset(Z)

    @cached_property
    def all_CI(self):
        return set(self._all_CI_generator)

    def _all_CI_like_unlabelled_generator(self, attribute):
        for perm in itertools.permutations(self.visible_nodes):
            yield frozenset(
                tuple(
                    frozenset(partsextractor(perm, variable_set))
                    for variable_set in relation)
                for relation in self.__getattribute__(attribute))

    @cached_property
    def all_CI_unlabelled(self):
        return min(self._all_CI_like_unlabelled_generator('all_CI'))

    @property
    def _all_e_sep_generator(self):
        for r in range(self.number_of_visible - 1):
            for to_delete in itertools.combinations(self.visible_nodes, r):
                graph_copy = self.as_graph.copy()  # Don't forget to copy!
                graph_copy.remove_nodes_from(to_delete)
                remaining = set(self.visible_nodes).difference(to_delete)
                for x, y, Z in self._all_2_vs_any_partitions(tuple(remaining)):
                    if nx.d_separated(graph_copy, {x}, {y}, set(Z)):
                        yield frozenset([x, y]), frozenset(Z), frozenset(to_delete)

    @cached_property  # Behaving weird, get consumed unless wrapped in tuple
    def all_esep(self):
        return set(self._all_e_sep_generator)

    @cached_property
    def all_esep_unlabelled(self):
        return min(self._all_CI_like_unlabelled_generator('all_esep'))

    # @methodtools.lru_cache(maxsize=None, typed=False)
    # def support_testing_instance(self, n):
    #     return SupportTesting(self.parents_of_for_supports_analysis,
    #                           np.broadcast_to(2, self.number_of_visible),
    #                           n)

    #Let's use smart support testing for both smart and not.
    @methodtools.lru_cache(maxsize=None, typed=False)
    def smart_support_testing_instance(self, n):
        return SmartSupportTesting(self.parents_of_for_supports_analysis,
                                   np.broadcast_to(2, self.number_of_visible),
                                   n, self.all_esep
                                   )

    def infeasible_binary_supports_n_events(self, n):
        return frozenset(self.smart_support_testing_instance(n).unique_infeasible_supports(name='mgh', use_timer=False))

    def smart_infeasible_binary_supports_n_events(self, n):
        return frozenset(self.smart_support_testing_instance(n).smart_unique_infeasible_supports(name='mgh', use_timer=False))

    def infeasible_binary_supports_n_events_unlabelled(self, n):
        return frozenset(self.smart_support_testing_instance(n).unique_infeasible_supports_unlabelled(name='mgh', use_timer=False))

    def smart_infeasible_binary_supports_n_events_unlabelled(self, n):
        return frozenset(self.smart_support_testing_instance(n).smart_unique_infeasible_supports_unlabelled(name='mgh', use_timer=False))





    @cached_property
    def droppable_edges(self):
        candidates = [tuple(pair) for pair in self.directed_structure_instance.edge_list if
                      any(set(pair).issubset(hyperedge) for hyperedge in self.simplicial_complex_instance.compressed_simplicial_complex)]
        candidates = [pair for pair in candidates if set(self.as_graph.predecessors(pair[0])).issubset(
            self.as_graph.predecessors(pair[1]))]  # as_graph includes both visible at latent parents
        return candidates

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

    @cached_property
    def splittable_faces(self):
        candidates = itertools.chain.from_iterable(map(self._all_bipartitions, self.simplicial_complex_instance.compressed_simplicial_complex))
        # candidates = [(C,D) for C,D in candidates if all(set(self.as_graph.predecessors(c).issubset(self.as_graph.predecessors(d)) for c in C for d in D)]
        candidates = [(C, D) for C, D in candidates if
                      all(self.set_predecessors(C).issubset(self.singleton_predecessors(d)) for d in D)]
        # print(candidates)
        return candidates

    def generate_weaker_mDAGs_FaceSplitting(self, unsafe=True):
        if unsafe:
            return self.generate_weaker_mDAGs_FaceSplitting_Simultaneous
        else:
            return self.generate_weaker_mDAGs_FaceSplitting_Safe

    @property
    def generate_weaker_mDAGs_FaceSplitting_Safe(self):
        for C, D in self.splittable_faces:
            new_simplicial_complex = self.simplicial_complex_instance.simplicial_complex_as_sets.copy()
            new_simplicial_complex.discard(C.union(D))
            # setC = set(C)
            # setD = set(D)
            if not any(C.issubset(facet) for facet in new_simplicial_complex):
                new_simplicial_complex.append(C)
            if not any(D.issubset(facet) for facet in new_simplicial_complex):
                new_simplicial_complex.append(D)
            # new_simplicial_complex.sort()
            yield mdag_to_int(
                self.directed_structure_instance.bit_square_matrix,
                hypergraph(new_simplicial_complex).as_bit_array)

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
                hypergraph(new_simplicial_complex).as_bit_array)

    #TODO: slightly speed up this by avoiding hypergraph creation. That is, directly modify the bit array.
    #Or, if we are really fancy, we can modify the bits of the unique_id itself!!







    @property
    def districts(self):
        return self.simplicial_complex_instance.districts

    @cached_property
    def fundamental_graphQ(self):
        # Implement three conditions
        district_lengths = np.fromiter(map(len, self.districts), np.int_)
        # district_lengths = np.asarray(list(map(len, self.districts)))
        common_cause_district_positions = np.flatnonzero(district_lengths > 1)
        if len(common_cause_district_positions) != 1:
            return False
        district_vertices = set(self.districts[common_cause_district_positions[0]])
        non_district_vertices = set(range(self.number_of_visible)).difference(district_vertices)
        for v in non_district_vertices:
            # Does it have children outside the district
            if set(np.flatnonzero(self.directed_structure_instance.as_bit_square_matrix[v])).isdisjoint(district_vertices):
                return False
            # Does it have parents
            if self.directed_structure_instance.as_bit_square_matrix[:, v].any():
                return False
        return True