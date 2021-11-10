from __future__ import absolute_import
import networkx as nx
import numpy as np
import itertools
from merge import merge_intersection
from more_itertools import ilen, chunked
from radix import bitarrays_to_ints
from utilities import partsextractor, nx_to_bitarray, hypergraph_to_bitarray, mdag_to_int, mdag_to_canonical_int
from utilities import stringify_in_tuple, stringify_in_list, stringify_in_set


from sys import hexversion

if hexversion >= 0x3080000:
    from functools import cached_property
elif hexversion >= 0x3060000:
    # with io.capture_output() as captured:
    #     !pip
    #     install
    #     backports.cached - property
    from backports.cached_property import cached_property
else:
    cached_property = property

from collections import defaultdict
from operator import itemgetter
from supports import SupportTesting
from supports_beyond_esep import SmartSupportTesting


class mDAG:
    def __init__(self, directed_structure, simplicial_complex, complex_extended=False):
        """
        Parameters
        ----------
        directed_structure : networkx.DiGraph
            The directed edges of the mDAG, given as a networkx digraph.
        simplicial_complex : list of lists
            The directed simplicial complex, given as a list of tuples.
        """
        self.directed_structure = directed_structure
        self.simplicial_complex = simplicial_complex
        # self.hypergraph_instance = Hypergraph(simplicial_complex)
        self.number_of_visible: int = self.directed_structure.number_of_nodes()

        self.directed_structure_as_list = sorted(self.directed_structure.edges())
        self.directed_structure_as_bitarray = nx_to_bitarray(self.directed_structure)
        # self.directed_structure_as_int = nx_to_int(self.DirectedStructure)
        self.visible_nodes = sorted(self.directed_structure.nodes())

        # Putting singleton latents on the nodes that originally have no latent parents:
        if complex_extended:
            self.extended_simplicial_complex = self.simplicial_complex
        else:
            self.extended_simplicial_complex = sorted(self.simplicial_complex + [(singleton,) for singleton in
                                                                                 set(self.visible_nodes).difference(
                                                                                     *self.simplicial_complex)])
        self.number_of_latent = len(self.extended_simplicial_complex)
        self.number_of_nodes = self.number_of_visible + self.number_of_latent

        self.latent_nodes = list(range(self.number_of_visible, self.number_of_nodes))
        self.all_nodes = self.visible_nodes + self.latent_nodes

        #self.stringify = lambda l: '(' + ','.join(map(str, l)) + ')'
        # self.directed_structure_string = self.stringify(map(self.stringify,self.directed_structure_as_list))
        # self.directed_structure_string = self.stringify(str(i)+':'+self.stringify(v) for i,v in nx.to_dict_of_lists(self.DirectedStructure).items() if len(v)>0)
        # self.simplicial_complex_string = self.stringify(map(self.stringify,sorted(self.simplicial_complex)))

    @cached_property
    def directed_structure_string(self):
        return stringify_in_set(
            str(i) + ':' + stringify_in_list(v) for i, v in nx.to_dict_of_lists(self.directed_structure).items() if
            len(v) > 0)

    @cached_property
    def simplicial_complex_string(self):
        return stringify_in_list(map(stringify_in_tuple, sorted(self.extended_simplicial_complex)))

    @cached_property
    def mDAG_string(self):
        return self.directed_structure_string + '|' + self.simplicial_complex_string

    # def extended_simplicial_complex(self):
    #  # Returns the simplicial complex extended to include singleton sets.
    #  return self.simplicial_complex + [(singleton,) for singleton in set(self.visible_nodes).difference(*self.simplicial_complex)]

    def __str__(self):
        return self.mDAG_string

    def __repr__(self):
        return self.mDAG_string

    @cached_property
    def as_graph(self):  # let DirectedStructure be a DAG initially without latents
        g = self.directed_structure.copy()
        g.add_nodes_from(self.latent_nodes, type="Latent")
        g.add_edges_from(itertools.chain.from_iterable(
            zip(itertools.repeat(i), children) for i, children in
            zip(self.latent_nodes, self.extended_simplicial_complex)))
        return g

    @cached_property
    def relative_complexity_for_sat_solver(self):  # choose eqclass representative which minimizes this
        return self.as_graph.number_of_edges()

    # @cached_property
    # def as_extended_graph(self):  #This fills in the singleton variables
    #     g = self.as_graph.copy()
    #     rootless_visible = set(self.visible_nodes).difference(*self.simplicial_complex)
    #     implicit_latent_nodes = list(range(self.number_of_nodes,self.number_of_nodes+len(rootless_visible)))
    #     g.add_nodes_from(implicit_latent_nodes, type="Latent")
    #     g.add_edges_from(zip(implicit_latent_nodes,rootless_visible))
    #     return g

    # @staticmethod
    # def partsextractor(thing_to_take_parts_of, indices):
    #     if len(indices) == 0:
    #         return tuple()
    #     elif len(indices) == 1:
    #         return (itemgetter(*indices)(thing_to_take_parts_of),)
    #     else:
    #         return itemgetter(*indices)(thing_to_take_parts_of)

    @cached_property
    def translation_dict(self):
        return dict(zip(self.as_graph.nodes(), range(self.as_graph.number_of_nodes())))

    def as_integer_labels(self, labels):
        return partsextractor(self.translation_dict, labels)

    @cached_property
    def parents_of_for_supports_analysis(self):
        return [self.as_integer_labels(tuple(self.as_graph.predecessors(n))) for n in self.visible_nodes]
        # to_translate = list(map(set, map(self.as_extended_graph.predecessors, self.visible_nodes)))
        # return [tuple(self.translation_dict[n] for n in parents) for parents in to_translate]

    # @cached_property
    # def simplicial_complex_as_mat(self):
    #     r = np.zeros((len(self.extended_simplicial_complex), self.number_of_visible), dtype = bool)
    #     for i,lp in enumerate(self.extended_simplicial_complex):
    #         r[i, self.as_integer_labels(lp)] = True
    #     return r[np.lexsort(r.T)]

    # @cached_property
    # def simplicial_complex_as_int(self):
    #     # return bitarray_to_int(self.simplicial_complex_as_mat).tolist()
    #     # return hypergraph_to_int(list(map(self.as_integer_labels, self.extended_simplicial_complex)))
    #     return hypergraph_to_int(self.extended_simplicial_complex) #Assumes observed variables are integers!

    @cached_property
    def simplicial_complex_as_bitarray(self):
        return hypergraph_to_bitarray(self.extended_simplicial_complex)  # Assumes observed variables are integers!

    @cached_property
    def unique_id(self):
        # Returns a unique identification number.
        # return int.from_bytes(bytearray(self.__repr__().encode('UTF-8')), byteorder='big', signed=True)
        return mdag_to_int(self.directed_structure_as_bitarray, self.simplicial_complex_as_bitarray)

    @cached_property
    def unique_unlabelled_id(self):
        # Returns a unique identification number.
        # return int.from_bytes(bytearray(self.__repr__().encode('UTF-8')), byteorder='big', signed=True)
        return mdag_to_canonical_int(self.directed_structure_as_bitarray, self.simplicial_complex_as_bitarray)

    def infeasible_binary_supports_n_events(self, n):
        return SupportTesting(self.parents_of_for_supports_analysis,
                              np.broadcast_to(2, self.number_of_visible),
                              n).unique_infeasible_supports_as_integers(name='mgh', use_timer=False)

    def smart_infeasible_binary_supports_n_events(self, n):
        return SmartSupportTesting(self.parents_of_for_supports_analysis,
                                   np.broadcast_to(2, self.number_of_visible),
                                   n, chunked(map(lambda vars: list(self.as_integer_labels(tuple(vars))),
                                                  itertools.chain.from_iterable(self.all_esep)), 3)
                                   ).unique_infeasible_supports_beyond_esep_as_integers(name='mgh', use_timer=False)

    def infeasible_binary_supports_n_events_unlabelled(self, n):
        return SupportTesting(self.parents_of_for_supports_analysis,
                              np.broadcast_to(2, self.number_of_visible),
                              n).unique_infeasible_supports_unlabelled(name='mgh', use_timer=False)

    def smart_infeasible_binary_supports_n_events_unlabelled(self, n):
        return SmartSupportTesting(self.parents_of_for_supports_analysis,
                                   np.broadcast_to(2, self.number_of_visible),
                                   n, chunked(map(lambda vars: list(self.as_integer_labels(tuple(vars))),
                                                  itertools.chain.from_iterable(self.all_esep)), 3)
                                   ).unique_infeasible_supports_beyond_esep_as_integers_unlabelled(name='mgh', use_timer=False)

    @cached_property
    def simplicial_complex_as_graph(self):  # let DirectedStructure be a DAG initially without latents
        g = nx.DiGraph()
        g.add_nodes_from(self.visible_nodes, type="Visible")
        g.add_nodes_from(self.latent_nodes, type="Latent")
        g.add_edges_from(itertools.chain.from_iterable(
            zip(itertools.repeat(i), children) for i, children in zip(self.latent_nodes, self.simplicial_complex)))
        return g

    @cached_property
    def districts(self):
        # Definitely this can be implemented way more efficiently, i.e. by working on the adjacency matrix
        # https://stackoverflow.com/questions/9110837/python-simple-list-merging-based-on-intersections
        # return [tuple(set(component).difference(self.latent_nodes)) for component in
        #         nx.weakly_connected_components(self.simplicial_complex_as_graph)]
        return merge_intersection(self.extended_simplicial_complex)

    # @staticmethod
    # def graph_to_string(g,order):
    #     numnodes = g.number_of_nodes()
    #     adjmat = nx.to_numpy_array(g, nodelist=order, dtype=np.bool_)
    #     flatmat = np.hstack([np.diag(adjmat, i) for i in np.arange(-numnodes+1,numnodes)])
    #     return tuple(flatmat)

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
            relabelling = dict(zip(self.visible_nodes, perm))
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


    @cached_property
    def droppable_edges(self):
        candidates = [tuple(pair) for pair in self.directed_structure.edges() if
                      any(set(pair).issubset(hyperredge) for hyperredge in self.simplicial_complex)]
        candidates = [pair for pair in candidates if set(self.as_graph.predecessors(pair[0])).issubset(
            self.as_graph.predecessors(pair[1]))]  # as graph included both visible at latent parents
        return candidates

    # @property
    # def generate_weaker_mDAGs_HLP(self):
    #     for droppable_edge in self.droppable_edges:
    #         new_directed_structure = self.DirectedStructure.copy()
    #         new_directed_structure.remove_edge(*droppable_edge)
    #         new_mDAG = self.__class__(new_directed_structure, self.extended_simplicial_complex, complex_extended=True)
    #         yield new_mDAG

    @property
    def generate_weaker_mDAG_HLP(self):
        if self.droppable_edges:
            new_directed_structure = self.directed_structure.copy()
            new_directed_structure.remove_edges_from(self.droppable_edges)
            new_mDAG = self.__class__(new_directed_structure, self.extended_simplicial_complex, complex_extended=True)
            yield new_mDAG

    @staticmethod
    def _all_bipartitions(variables_to_partition):
        # expect input in the form of a list
        length = len(variables_to_partition)
        integers = range(length)
        subsetrange = range(1, length)
        for r in subsetrange:
            for ints_to_mask in map(list, itertools.combinations(integers, r)):
                mask = np.ones(length, dtype=np.bool)
                mask[ints_to_mask] = False
                yield (set(itertools.compress(variables_to_partition, mask.tolist())),
                       set(itertools.compress(variables_to_partition, np.logical_not(mask).tolist())))

    def setpredecessorsplus(self, X):
        result = set(X)
        for x in X:
            result.update(self.as_graph.predecessors(x))
        return result

    @cached_property
    def splittable_faces(self):
        candidates = itertools.chain.from_iterable(map(self._all_bipartitions, self.simplicial_complex))
        # candidates = [(C,D) for C,D in candidates if all(set(self.as_graph.predecessors(c).issubset(self.as_graph.predecessors(d)) for c in C for d in D)]
        candidates = [(C, D) for C, D in candidates if
                      all(self.setpredecessorsplus(C).issubset(self.as_graph.predecessors(d)) for d in D)]
        # print(candidates)
        return candidates

    @property
    def generate_weaker_mDAGs_FaceSplitting(self):
        for C, D in self.splittable_faces:
            new_simplicial_complex = self.extended_simplicial_complex.copy()
            new_simplicial_complex.remove(tuple(sorted(list(C) + list(D))))
            setC = set(C)
            setD = set(D)
            if not any(setC.issubset(facet) for facet in new_simplicial_complex):
                new_simplicial_complex.append(tuple(sorted(C)))
            if not any(setD.issubset(facet) for facet in new_simplicial_complex):
                new_simplicial_complex.append(tuple(sorted(D)))
            new_simplicial_complex.sort()
            new_mDAG = self.__class__(self.directed_structure, new_simplicial_complex, complex_extended=True)
            yield new_mDAG

    @property  # Agressive conjucture of simultaneous face splitting
    # What does it mean to split simultanously? It means we consider intersections based on D...?
    def generate_weaker_mDAGs_FaceSplitting_Simultaneous(self):
        new_dict = defaultdict(list)
        for C, D in self.splittable_faces:
            new_dict[tuple(D)].append(C)
        for D, Cs in new_dict.items():
            new_simplicial_complex = self.extended_simplicial_complex.copy()
            setD = set(D)
            for C in Cs:
                new_simplicial_complex.remove(tuple(sorted(list(C) + list(D))))
            if not any(setD.issubset(facet) for facet in new_simplicial_complex):
                new_simplicial_complex.append(tuple(sorted(D)))
            for C in Cs:
                setC = set(C)
                if not any(setC.issubset(facet) for facet in new_simplicial_complex):
                    new_simplicial_complex.append(tuple(sorted(C)))
            new_simplicial_complex.sort()
            new_mDAG = self.__class__(self.directed_structure, new_simplicial_complex, complex_extended=True)
            yield new_mDAG

    def participates_in_only_one_facet(self, v):
        originating_hyperedge_count = 0
        allowed_by_Evans = True
        for hyperedge in self.extended_simplicial_complex:
            if v in hyperedge:
                originating_hyperedge_count += 1
            if originating_hyperedge_count >= 2:
                allowed_by_Evans = False
                break
        return allowed_by_Evans

    @property
    def generate_weaker_mDAGs_Evans_directed_structure(self):
        for droppable_edge in self.droppable_edges:
            if self.participates_in_only_one_facet(droppable_edge[0]):
                new_directed_structure = self.directed_structure.copy()
                new_directed_structure.remove_edge(*droppable_edge)
                new_mDAG = self.__class__(new_directed_structure, self.extended_simplicial_complex,
                                          complex_extended=True)
                yield new_mDAG

    @property
    def generate_weaker_mDAGs_Evans_simplicial_complex(self):
        for C, D in self.splittable_faces:
            if all(self.participates_in_only_one_facet(c) for c in C):
                new_simplicial_complex = self.extended_simplicial_complex.copy()
                new_simplicial_complex.remove(tuple(sorted(list(C) + list(D))))
                new_simplicial_complex.append(tuple(sorted(C)))
                setD = set(D)
                if not any(setD.issubset(facet) for facet in new_simplicial_complex):
                    new_simplicial_complex.append(tuple(sorted(D)))
                new_simplicial_complex.sort()
                new_mDAG = self.__class__(self.directed_structure, new_simplicial_complex, complex_extended=True)
                yield new_mDAG

    @cached_property
    def districts(self):
        return merge_intersection(self.extended_simplicial_complex)

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
            if set(self.directed_structure.successors(v)).isdisjoint(district_vertices):
                return False
            if ilen(self.directed_structure.predecessors(v)) > 0:
                return False
        return True

    @property
    def generate_symmetric_variants(self):
        node_order = self.visible_nodes.copy()
        node_order[:2] = reversed(node_order[:2])
        node_orders = [node_order]
        if self.number_of_visible > 2:
            node_orders.append(np.roll(self.visible_nodes, 1).tolist())
        for node_order in node_orders:
            new_directed_structure = nx.relabel_nodes(self.directed_structure, node_order.__getitem__)
            # print(self.extended_simplicial_complex)
            new_simplicial_complex = [tuple(sorted(np.take(node_order, hyperedge))) for hyperedge in
                                      self.extended_simplicial_complex]
            # print(new_simplicial_complex)
            new_mDAG = self.__class__(new_directed_structure, new_simplicial_complex, complex_extended=True)
            yield new_mDAG

    @cached_property
    def skeleton_bitarray(self):
        skeleton_graph = self.directed_structure.to_undirected()
        for hyperedge in self.simplicial_complex:
            skeleton_graph.add_edges_from(itertools.combinations(hyperedge,2))
        cliques =  list(nx.find_cliques(skeleton_graph))
        # print(cliques)
        # simplicial_complex_as_nontriv_sets = [set(clique) for clique in cliques if len(clique) > 1]
        return hypergraph_to_bitarray(cliques)
        # return hypergraph_to_bitarray(merge_intersection(self.directed_structure_as_list +  self.simplicial_complex))

    @cached_property
    def skeleton(self):
        # Concern about int64 overflow
        # return bitarray_to_int(self.skeleton_bitarray).astype(np.ulonglong).tolist()
        return bitarrays_to_ints(self.skeleton_bitarray).tolist()
        # return frozentset(map(frozenset,merge_intersection(self.directed_structure_as_list +  self.simplicial_complex)))

    @cached_property
    def skeleton_unlabelled(self):
        # return min([sorted(map(sorted,np.take(perm,self.skeleton.edges())))
        #              for perm in itertools.permutations(self.visible_nodes)])
        # return set([frozenset(map(frozenset, np.take(perm, self.skeleton))) for perm in
        # itertools.permutations(self.visible_nodes)]).pop()
        return bitarrays_to_ints(
            [self.skeleton_bitarray[np.lexsort(self.skeleton_bitarray[:, perm].T)][:, perm] for perm in
             map(list, itertools.permutations(range(self.number_of_visible)))]).min() #.astype(np.ulonglong)
