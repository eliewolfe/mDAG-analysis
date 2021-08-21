from __future__ import absolute_import
import numpy as np
import itertools
import networkx as nx
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
# from supports import SupportTesting
from supports_beyond_esep import SmartSupportTesting
from hypergraphs import UndirectedGraph, Hypergraph, LabelledHypergraph
import methodtools
from directed_structures import LabelledDirectedStructure, DirectedStructure

 
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
    def as_graph(self):  # let DirectedStructure be a DAG initially without latents
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
    def n_of_edges(self):
        return self.directed_structure_instance.number_of_edges

    @cached_property
    def relative_complexity_for_sat_solver(self):  # choose eqclass representative which minimizes this
        return (self.as_extended_bit_array.sum(), self.n_of_edges, self.simplicial_complex_instance.tally)
                
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
        return UndirectedGraph(self.directed_structure_instance.as_tuples + self.simplicial_complex_instance.as_tuples,
                               self.number_of_visible)

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
    #@methodtools.lru_cache(maxsize=None, typed=False)
    
    def smart_support_testing_instance_card_3(self, n):
        return SmartSupportTesting(self.parents_of_for_supports_analysis,
                                   (3,2,2),
                                   n, self.all_esep
                                   )
    def smart_infeasible_supports_n_events_card_3(self, n, **kwargs):
          return frozenset(self.smart_support_testing_instance_card_3(n).smart_unique_infeasible_supports(**kwargs, name='mgh', use_timer=False))
    
    def smart_support_testing_instance(self, n):
        return SmartSupportTesting(self.parents_of_for_supports_analysis,
                                   np.broadcast_to(2, self.number_of_visible),
                                   n, self.all_esep
                                   )

    def infeasible_binary_supports_n_events(self, n, **kwargs):
        return frozenset(self.smart_support_testing_instance(n).unique_infeasible_supports(**kwargs, name='mgh', use_timer=False))

    def smart_infeasible_binary_supports_n_events(self, n, **kwargs):
        return frozenset(self.smart_support_testing_instance(n).smart_unique_infeasible_supports(**kwargs, name='mgh', use_timer=False))

    def infeasible_binary_supports_n_events_unlabelled(self, n, **kwargs):
        return frozenset(self.smart_support_testing_instance(n).unique_infeasible_supports_unlabelled(**kwargs, name='mgh', use_timer=False))

    def smart_infeasible_binary_supports_n_events_unlabelled(self, n, **kwargs):
        return frozenset(self.smart_support_testing_instance(n).smart_unique_infeasible_supports_unlabelled(**kwargs, name='mgh', use_timer=False))

    def no_infeasible_supports_up_to(self, max_n, **kwargs):
        return all(self.smart_support_testing_instance(n).no_infeasible_supports(**kwargs, name='mgh', use_timer=False) for
                   n in range(2,max_n+1))

    def no_infeasible_supports_beyond_esep_up_to(self, max_n, **kwargs):
        return all(self.smart_support_testing_instance(n).no_infeasible_supports_beyond_esep(**kwargs, name='mgh', use_timer=False) for
                   n in range(2,max_n+1))
    def no_infeasible_supports_n_events(self, n, **kwargs):
        if len(self.all_esep)==0:
            return self.smart_support_testing_instance(n).no_infeasible_supports(**kwargs, name='mgh', use_timer=False)
        else:
            return False

    def no_infeasible_supports_up_to(self, max_n, **kwargs):
        return all(self.no_infeasible_supports_n_events(n, **kwargs) for n in range(2,max_n+1))

    # def no_infeasible_supports_up_to(self, max_n, **kwargs):
    #     return all(len(self.infeasible_binary_supports_n_events(n, **kwargs))==0 for n in range(2,max_n+1))




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


    @property
    def districts(self):
        return self.simplicial_complex_instance.districts
    
    @property
    def districts_arbitrary_names(self):
        districts_translated=[]
        for d in self.districts:
            d_translated=set()
            for node in d:
                node_translated=list(self.simplicial_complex_instance.translation_dict.keys())[list(self.simplicial_complex_instance.translation_dict.values()).index(node)]
                d_translated.add(node_translated)
            districts_translated.append(d_translated)
        return districts_translated
    
    def subgraph(self,list_of_nodes):
        new_edges=[]
        for edge in self.directed_structure_instance.edge_list:
            if edge[0] in list_of_nodes and edge[1] in list_of_nodes:
                new_edges.append(edge)
        new_hypergraph=[]
        for hyperedge in self.simplicial_complex_instance.simplicial_complex:
            new_hyperedge=hyperedge
            for node in hyperedge:
                if node not in list_of_nodes:
                    new_hyperedge_list=list(new_hyperedge)
                    new_hyperedge_list.remove(node)
                    new_hyperedge=tuple(new_hyperedge_list)
            subset_of_already_hyp=False
            for already_hyp in new_hypergraph:
                if set(new_hyperedge).issubset(set(already_hyp)) and new_hyperedge!=already_hyp:
                    subset_of_already_hyp=True
                if set(already_hyp).issubset(set(new_hyperedge)):
                    new_hypergraph.remove(already_hyp)
            if not subset_of_already_hyp:
                new_hypergraph.append(new_hyperedge)
        return mDAG(LabelledDirectedStructure(list_of_nodes,new_edges),LabelledHypergraph(list_of_nodes,new_hypergraph))
   
    def closure(self, B):
        list_B=[set(self.visible_nodes)]
        graph=self.subgraph(list_B[0])
        next_B=set()
        for element in B:
            for dist in graph.districts_arbitrary_names:
                if element in dist:
                    next_B=set(next_B).union(dist)
        list_B.append(next_B)
        graph=self.subgraph(list_B[1])
        next_B=set()
        for element in B:
            next_B=set(list(next_B)+list(nx.ancestors(graph.directed_structure_instance.as_networkx_graph_arbitrary_names,element))+[element])
        list_B.append(next_B)
        i=2
        while any([list_B[i]!=list_B[i-1],list_B[i]!=list_B[i-2]]):
            graph=self.subgraph(list_B[-1])
            for element in B:
                for dist in graph.districts_arbitrary_names:
                    if element in dist:
                        next_B=set(next_B).union(dist)
            list_B.append(next_B)
            i=i+1
            graph=self.subgraph(list_B[-1])
            next_B=set()
            for element in B:
                next_B=set(list(next_B)+list(nx.ancestors(graph.directed_structure_instance.as_networkx_graph_arbitrary_names,element))+[element])
            list_B.append(next_B)
            i=i+1
        return list_B[-1]
  
    def are_densely_connected(self,node1,node2):
        for closure_node in self.closure([node1]):
            if node2 in self.directed_structure_instance.as_networkx_graph.predecessors(closure_node):
                return True
        for closure_node in self.closure([node2]):
            if node1 in self.directed_structure_instance.as_networkx_graph.predecessors(closure_node):
                return True
        for district in self.districts:
            if self.closure([node1,node2]).issubset(district):
                return True
        return False
    
# Evans 2021: It is possible to have a distribution with 2 variables identical to one another and independent of all others iff they are densely connected
# So, if node1 and node2 are densely connected in G1 but not in G2, we know that G1 is NOT equivalent to G2.

    @property
    def all_densely_connected_pairs(self):
        all_densely_connected_pairs=set()
        for node1, node2 in itertools.combinations(self.visible_nodes,2):
            if self.are_densely_connected(node1,node2):
                all_densely_connected_pairs.add((node1,node2))
        return all_densely_connected_pairs   


    def _all_densely_connected_pairs_unlabelled_generator(self):
        for perm in itertools.permutations(self.visible_nodes):
            yield frozenset(
                tuple(partsextractor(perm, variable_set)
                    for variable_set in relation)
                for relation in self.__getattribute__('all_densely_connected_pairs'))
            
    @cached_property
    def all_densely_connected_pairs_unlabelled(self):
        return min(self._all_densely_connected_pairs_unlabelled_generator())
            
    
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

class labelled_mDAG(mDAG):
    def __init__(self, labelled_directed_structure_instance, labelled_simplicial_complex_instance):
        assert labelled_directed_structure_instance.variable_names == labelled_simplicial_complex_instance.variable_names, "Name conflict."
        self.variable_names = labelled_directed_structure_instance.variable_names
        super().__init__(labelled_directed_structure_instance, labelled_simplicial_complex_instance)
