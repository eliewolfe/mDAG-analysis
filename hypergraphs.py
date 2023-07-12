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

from radix import bitarray_to_int
from merge import merge_intersection
from utilities import stringify_in_tuple, stringify_in_list, partsextractor

# from scipy.special import comb
import networkx as nx
from functools import lru_cache

@lru_cache(maxsize=5)
def empty_graph(n):
    baseg = nx.Graph()
    baseg.add_nodes_from(range(n))
    return baseg

def has_length_greater_than_one(stuff):
    return len(stuff)>1
def drop_singletons(hypergraph):
    return filter(has_length_greater_than_one, hypergraph)
def permute_bit_array(bitarray, perm):
    almost_new_sc = bitarray[:, list(perm)]
    return almost_new_sc[np.lexsort(almost_new_sc.T)]
def bit_array_permutations(bitarray):
    assert len(bitarray.shape)==2, 'Not a bitarray!'
    n = bitarray.shape[-1]
    return (permute_bit_array(bitarray, perm) for perm in map(list,itertools.permutations(range(n))))


def hypergraph_canonicalize_with_deduplication(hypergraph):
    # hypergraph_copy = set(map(frozenset, drop_singletons(hypergraph)))
    return set(map(frozenset, drop_singletons(hypergraph)))
def hypergraph_to_list_of_tuples(hypergraph):
    return sorted(map(lambda s: tuple(sorted(s)), hypergraph_canonicalize_with_deduplication(hypergraph)))
def hypergraph_full_cleanup(hypergraph):
    hypergraph_copy = set(map(frozenset, hypergraph))
    cleaned_hypergraph_copy = hypergraph_copy.copy()
    for dominating_hyperedge in hypergraph_copy:
        if dominating_hyperedge in cleaned_hypergraph_copy:
            dominated_hyperedges = []
            for dominated_hyperedge in cleaned_hypergraph_copy:
                if len(dominated_hyperedge) < len(dominating_hyperedge):
                    if dominated_hyperedge.issubset(dominating_hyperedge):
                        dominated_hyperedges.append(dominated_hyperedge)
            cleaned_hypergraph_copy.difference_update(dominated_hyperedges)
    return cleaned_hypergraph_copy









class Hypergraph:
    def __init__(self, any_simplicial_complex, n):
        self.number_of_visible = n

        self.simplicial_complex_as_sets = hypergraph_canonicalize_with_deduplication(any_simplicial_complex)
        self.compressed_simplicial_complex = hypergraph_to_list_of_tuples(self.simplicial_complex_as_sets)
        self.number_of_nonsingleton_latent = len(self.simplicial_complex_as_sets)
        self.vis_nodes_with_singleton_latent_parents = set(
            range(self.number_of_visible)).difference(
            itertools.chain.from_iterable(self.simplicial_complex_as_sets))
        self.singleton_hyperedges = set(frozenset({v}) for v in self.vis_nodes_with_singleton_latent_parents)
        self.extended_simplicial_complex_as_sets = self.simplicial_complex_as_sets.union(self.singleton_hyperedges)
        self.number_of_latent = len(self.extended_simplicial_complex_as_sets)


        if self.number_of_nonsingleton_latent:
            assert max(map(max, self.compressed_simplicial_complex)) + 1 <= self.number_of_visible, "More nodes referenced than expected."
        self.number_of_visible_plus_latent = self.number_of_visible + self.number_of_latent
        self.number_of_visible_plus_nonsingleton_latent = self.number_of_visible + self.number_of_nonsingleton_latent



        # self.max_number_of_latents = comb(self.number_of_visible, np.floor_divide(self.number_of_visible,2), exact=True)

    # @cached_property
    # def singleton_hyperedges(self):
    #     return set(frozenset({v}) for v in set(range(self.number_of_visible)).difference(self.simplicial_complex))
    #
    # @cached_property
    # def extended_simplicial_complex(self):
    #     return self.simplicial_complex_as_sets.union(*self.singleton_hyperedges)

    @cached_property
    def tally(self):
        # absent_latent_count = self.max_number_of_latents - self.number_of_latent
        # return tuple(np.pad(np.flip(sorted(map(len, self.simplicial_complex))),(0,absent_latent_count)))
        return tuple(np.flip(sorted(map(len, self.compressed_simplicial_complex))))

    # @cached_property
    # def compressed_simplicial_complex(self):
    #     return list(drop_singletons(self.simplicial_complex))
    #
    # @cached_property
    # def number_of_nonsingleton_latent(self):
    #     return len(self.compressed_simplicial_complex)

    # @cached_property
    # def number_of_visible_plus_nonsingleton_latent(self):
    #     return self.number_of_visible + self.number_of_nonsingleton_latent


    @cached_property
    def as_tuples(self):
        return tuple(self.compressed_simplicial_complex)

    @cached_property
    def as_bit_array(self):
        r = np.zeros((self.number_of_nonsingleton_latent, self.number_of_visible), dtype=bool)
        for i, lp in enumerate(self.compressed_simplicial_complex):
            r[i, tuple(lp)] = True
        return r[np.lexsort(r.T)]

    @cached_property
    def latent_parents_list(self):
        # return list(map(frozenset, map(np.flatnonzero, self.as_bit_array.T)))
        return [frozenset(np.flatnonzero(column)+self.number_of_visible) for column in self.as_bit_array.T]

    @cached_property
    def as_extended_bit_array(self):
        r = np.zeros((self.number_of_latent, self.number_of_visible), dtype=bool)
        for i, lp in enumerate(self.extended_simplicial_complex_as_sets):
            r[i, tuple(lp)] = True
        return r[np.lexsort(r.T)]

    @cached_property
    def nonsingleton_districts(self):
        return merge_intersection(self.compressed_simplicial_complex)

    @cached_property
    def districts(self):
        return self.nonsingleton_districts + list(map(set, self.singleton_hyperedges))

    @cached_property
    def as_integer(self):
        return bitarray_to_int(self.as_bit_array)

    @property
    def bit_array_permutations(self):
        return (permute_bit_array(self.as_bit_array, perm)
                for perm in map(list, itertools.permutations(range(self.number_of_visible))))

    @property
    def as_integer_permutations(self):
        return [bitarray_to_int(ba) for ba in self.bit_array_permutations]

    @cached_property
    def as_unlabelled_integer(self):
        return min(bitarray_to_int(ba) for ba in self.as_integer_permutations)

    @cached_property
    def as_string(self):
        return stringify_in_list(map(stringify_in_tuple, self.as_tuples))

    def can_S1_minimally_simulate_S2(S1, S2):
        """
        S1 and S2 are simplicial complices, in our data structure as lists of tuples.
        """
        # Modifying to restrict to minimal differences for speed
        # return all(any(s2.issubset(s1) for s1 in S1) for s2 in map(set, S2))
        dominance_count = 0
        so_far_so_good = True
        for s2 in S2.simplicial_complex_as_sets:
            contained = False
            for s1 in S1.simplicial_complex_as_sets:
                if s2.issubset(s1):
                    contained = True
                    if len(s2) > len(s1):
                        dominance_count += 1
                    break
            so_far_so_good = contained and (dominance_count <= 1)
            if not so_far_so_good:
                break
        return so_far_so_good

    def are_S1_facets_one_more_than_S2_facets(S1, S2):
        if S1.simplicial_complex_as_sets.issuperset(S2.simplicial_complex_as_sets):
            if S1.number_of_nonsingleton_latent == S2.number_of_nonsingleton_latent + 1:
                return True
        return False

    def is_S1_strictly_above_S2(S1, S2):
        """
        S1 and S2 are simplicial complices, in our data structure as lists of tuples.
        """
        dominance_count = 0
        for s2 in S2.simplicial_complex_as_sets:
            contained = False
            for s1 in S1.simplicial_complex_as_sets:
                if s2.issubset(s1):
                    contained = True
                    if len(s2) > len(s1):
                        dominance_count += 1
                    break
            if not contained:
                return False
        return ((dominance_count > 0) or (S1.number_of_nonsingleton_latent > S2.number_of_nonsingleton_latent))

    def __str__(self):
        return self.as_string

    def __repr__(self):
        return self.as_string

    @cached_property
    def as_bidirected_adjmat(self):
        adjmat = np.zeros((self.number_of_visible, self.number_of_visible), dtype=bool)
        for hyperedge in map(list, self.compressed_simplicial_complex):
            subindices = np.ix_(hyperedge, hyperedge)
            adjmat[subindices] = True
        return np.bitwise_and(adjmat, np.invert(np.identity(self.number_of_visible, dtype=bool)))



class LabelledHypergraph(Hypergraph):
    """
    This class is NOT meant to encode mDAGs. As such, we do not get into an implementation of predecessors or successors here.
    NEW: We can automatically extract SUBHYPERGRAPHS
    """
    def __init__(self, variable_names, simplicial_complex):
        """
        There may be MORE variable names than are referenced in the simplicial complex. These are considered unreferenced singletons.
        There may be FEWER variable names than are referenced in the simplicial complex. In that case we take a subhypergraph.
        """
        self.variable_names = tuple(variable_names)
        self.variable_names_as_frozenset = frozenset(self.variable_names)
        self.number_of_variables = len(variable_names)
        assert self.number_of_variables == len(self.variable_names_as_frozenset), "A variable name appears in duplicate."

        implicit_variable_names = set(itertools.chain.from_iterable(simplicial_complex))
        if implicit_variable_names.issubset(self.variable_names_as_frozenset):
            self.simplicial_complex_with_variable_names = hypergraph_canonicalize_with_deduplication(simplicial_complex)
        else:
            self.simplicial_complex_with_variable_names = hypergraph_full_cleanup(
                [self.variable_names_as_frozenset.intersection(hyperedge) for hyperedge in simplicial_complex])
        # self.simplicial_complex_with_variable_names_as_set = set(map(frozenset, simplicial_complex)) #To remove duplicates & partially canonicalize.
        # if self.number_of_variables<len(implicit_variable_names):
        #     #step 1: remove other variables from every hyperredge
        #     self.simplicial_complex_with_variable_names_as_set = set(self.variable_names_as_frozenset.intersection(hyperedge)
        #                                                    for hyperedge in simplicial_complex)
        #     #step 2: remove dominated hyperredges:
        #     for dominating_hyperedge in self.simplicial_complex_with_variable_names_as_set:
        #         for dominated_hyperedge in self.simplicial_complex_with_variable_names_as_set:
        #             if len(dominated_hyperedge) < len(dominating_hyperedge):
        #                 if dominated_hyperedge.issubset(dominating_hyperedge):
        #                     self.simplicial_complex_with_variable_names_as_set.discard(dominated_hyperedge)


        self.variables_as_range = tuple(range(self.number_of_variables))
        self.translation_dict = dict(zip(self.variable_names, self.variables_as_range))
        self.variable_are_range = False
        if all(isinstance(v, int) for v in self.variable_names):
            if np.array_equal(self.variable_names, self.variables_as_range):
                self.variable_are_range = True
                self.numerical_simplicial_complex = self.simplicial_complex_with_variable_names
        if not self.variable_are_range:
            self.variable_are_range = False
            self.numerical_simplicial_complex = [partsextractor(self.translation_dict, hyperedge) for hyperedge in self.simplicial_complex_with_variable_names]
        super().__init__(self.numerical_simplicial_complex, self.number_of_variables)
        # self.as_string = stringify_in_list(map(stringify_in_tuple, self.simplicial_complex_with_variable_names))

    @cached_property
    def as_string(self):
        return stringify_in_list(map(stringify_in_tuple, self.simplicial_complex_with_variable_names))

    @cached_property
    def translated_nonsingleton_districts(self):
        return [partsextractor(self.variable_names, district) for district in self.nonsingleton_districts]

    @cached_property
    def translated_districts(self):
        return [set(partsextractor(self.variable_names, district)) for district in self.districts]

    @cached_property
    def translated_extended_simplicial_complex(self):
        return set(frozenset(partsextractor(self.variable_names, hyperedge)) for hyperedge in self.extended_simplicial_complex_as_sets)

    @cached_property
    def translated_simplicial_complex(self):
        return set(frozenset(partsextractor(self.variable_names, hyperedge)) for hyperedge in self.simplicial_complex_as_sets)
        
    def __str__(self):
        return self.as_string

    def __repr__(self):
        return self.as_string




class UndirectedGraph:
    def __init__(self, hyperedges, n):
        self.nof_nodes = n
        if hyperedges:
            assert max(map(max, hyperedges)) + 1 <= self.nof_nodes, "More nodes referenced than expected."
        self.as_edges = tuple(set(itertools.chain.from_iterable(
            itertools.combinations(sorted(hyperedge), 2) for hyperedge in hyperedges)))
        self.as_string = stringify_in_list(map(stringify_in_tuple, self.as_edges))

    # @property
    # def as_edges(self):
    #     return itertools.chain.from_iterable(
    #         itertools.combinations(hyperedge, 2) for hyperedge in self.extended_simplicial_complex)
    def __str__(self):
        return self.as_string

    def __repr__(self):
        return self.as_string



    @property
    def as_edges_array(self):
        # return np.fromiter(self.as_edges, int).reshape((-1,2))
        return np.asarray(self.as_edges, dtype=int).reshape((-1,2))

    @property
    def as_adjacency_matrix(self):
        adjmat = np.zeros((self.nof_nodes, self.nof_nodes), dtype=bool)
        adjmat[tuple(self.as_edges_array.T)] = True
        return np.bitwise_or(adjmat, adjmat.T)

    @cached_property
    def as_edges_bit_array(self):
        r = np.zeros((len(self.as_edges_array), self.nof_nodes), dtype=bool)
        np.put_along_axis(r, self.as_edges_array, True, axis=1)
        return r[np.lexsort(r.T)]

    @staticmethod
    def edges_bit_array_to_integer(bitarray):
        """
        Quick base conversion algorithm.
        """
        assert len(bitarray.shape) == 2, 'Not a bitarray!'
        (d,n) = bitarray.shape
        if d:
            return np.frompyfunc(lambda a, b: n * a + b, 2, 1).reduce(np.where(bitarray)[-1])
        else:
            return 0

    @cached_property
    def as_edges_integer(self):
        return self.edges_bit_array_to_integer(self.as_edges_bit_array)

    @cached_property
    def as_edges_unlabelled_integer(self):
        return min(self.edges_bit_array_to_integer(ba) for ba in bit_array_permutations(self.as_edges_bit_array))

    @cached_property
    def as_networkx_graph(self):
        g = empty_graph(self.nof_nodes).copy()
        g.add_edges_from(self.as_edges)
        return g

    @cached_property
    def cliques(self):
        return list(nx.enumerate_all_cliques(self.as_networkx_graph))

class LabelledUndirectedGraph(UndirectedGraph):
     def __init__(self, variable_names, hyperedges):
        """
        There may be MORE variable names than are referenced in the simplicial complex. These are considered unreferenced singletons.
        There may be FEWER variable names than are referenced in the simplicial complex. In that case we take a subhypergraph.
        """
        self.variable_names = tuple(variable_names)
        self.variable_names_as_frozenset = frozenset(self.variable_names)
        self.number_of_variables = len(variable_names)
        assert self.number_of_variables == len(
            self.variable_names_as_frozenset), "A variable name appears in duplicate."
        self.edges_with_variable_names = tuple(set(itertools.chain.from_iterable(
            itertools.combinations(sorted(hyperedge), 2) for hyperedge in hyperedges)))
        # self.simplicial_complex_with_variable_names_as_set = set(map(frozenset, simplicial_complex)) #To remove duplicates & partially canonicalize.
        # if self.number_of_variables<len(implicit_variable_names):
        #     #step 1: remove other variables from every hyperredge
        #     self.simplicial_complex_with_variable_names_as_set = set(self.variable_names_as_frozenset.intersection(hyperedge)
        #                                                    for hyperedge in simplicial_complex)
        #     #step 2: remove dominated hyperredges:
        #     for dominating_hyperedge in self.simplicial_complex_with_variable_names_as_set:
        #         for dominated_hyperedge in self.simplicial_complex_with_variable_names_as_set:
        #             if len(dominated_hyperedge) < len(dominating_hyperedge):
        #                 if dominated_hyperedge.issubset(dominating_hyperedge):
        #                     self.simplicial_complex_with_variable_names_as_set.discard(dominated_hyperedge)

        self.variables_as_range = tuple(range(self.number_of_variables))
        self.translation_dict = dict(zip(self.variable_names, self.variables_as_range))
        self.variable_are_range = False
        if all(isinstance(v, int) for v in self.variable_names):
            if np.array_equal(self.variable_names, self.variables_as_range):
                self.variable_are_range = True
                self.numerical_edges = self.edges_with_variable_names
        if not self.variable_are_range:
            self.numerical_edges = [partsextractor(self.translation_dict, edge) for edge in
                                                 self.edges_with_variable_names]
        super().__init__(self.numerical_edges, self.number_of_variables)
        self.as_string = stringify_in_list(map(stringify_in_tuple, self.edges_with_variable_names))

