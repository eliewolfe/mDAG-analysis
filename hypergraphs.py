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

from scipy.special import comb

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








class Hypergraph:
    def __init__(self, extended_simplicial_complex, n):
        #TODO: Adjust code to handle non-integer-values hypergraphs!
        self.simplicial_complex = extended_simplicial_complex
        self.simplicial_complex_as_sets = set(map(frozenset, self.simplicial_complex))
        self.number_of_latent = len(self.simplicial_complex)
        self.number_of_visible = n
        if self.simplicial_complex:
            assert max(map(max, self.simplicial_complex)) + 1 <= self.number_of_visible, "More nodes referenced than expected."
        self.number_of_visible_plus_latent = self.number_of_visible + self.number_of_latent
        # self.max_number_of_latents = comb(self.number_of_visible, np.floor_divide(self.number_of_visible,2), exact=True)

    @cached_property
    def tally(self):
        # absent_latent_count = self.max_number_of_latents - self.number_of_latent
        # return tuple(np.pad(np.flip(sorted(map(len, self.simplicial_complex))),(0,absent_latent_count)))
        return tuple(np.flip(sorted(map(len, self.simplicial_complex))))

    @cached_property
    def compressed_simplicial_complex(self):
        return list(drop_singletons(self.simplicial_complex))

    @cached_property
    def number_of_nonsingleton_latent(self):
        return len(self.compressed_simplicial_complex)

    @cached_property
    def number_of_visible_plus_nonsingleton_latent(self):
        return self.number_of_visible + self.number_of_nonsingleton_latent


    @cached_property
    def as_tuples(self):
        return tuple(sorted(map(lambda s: tuple(sorted(s)), self.simplicial_complex)))

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
        for i, lp in enumerate(self.simplicial_complex):
            r[i, tuple(lp)] = True
        return r[np.lexsort(r.T)]

    @cached_property
    def districts(self):
        return merge_intersection(self.simplicial_complex)

    @staticmethod
    def bit_array_to_integer(bit_array):
        # Concern about int64 overflow
        # return from_bits(bitarray.ravel()).astype(np.ulonglong).tolist()
        return bitarray_to_int(bit_array)


    @cached_property
    def as_integer(self):
        return self.bit_array_to_integer(self.as_bit_array)

    @property
    def bit_array_permutations(self):
        return (permute_bit_array(self.as_bit_array, perm)
                for perm in map(list, itertools.permutations(range(self.number_of_visible))))

    @cached_property
    def as_unlabelled_integer(self):
        return min(self.bit_array_to_integer(ba) for ba in self.bit_array_permutations)

    @cached_property
    def as_string(self):
        return stringify_in_list(map(stringify_in_tuple, sorted(self.simplicial_complex)))

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
                        dominance_count += 0
                    break
            so_far_so_good = contained and (dominance_count <= 1)
            if not so_far_so_good:
                break
        return so_far_so_good

    def __str__(self):
        return self.as_string

    def __repr__(self):
        return self.as_string


class LabelledHypergraph(Hypergraph):
    """
    This class is NOT meant to encode mDAGs. As such, we do not get into an implementation of predecessors or successors here.
    """
    def __init__(self, variable_names, simplicial_complex):
        self.variable_names = variable_names
        self.number_of_variables = len(variable_names)
        self.variables_as_range = tuple(range(self.number_of_variables))
        self.translation_dict = dict(zip(self.variable_names, self.variables_as_range))
        if all(isinstance(v, int) for v in self.variable_names):
            if np.array_equal(self.variable_names, self.variables_as_range):
                self.variable_are_range = True
                self.numerical_simplicial_complex = simplicial_complex
            else:
                self.variable_are_range = False
                self.numerical_simplicial_complex = [partsextractor(self.translation_dict, hyperedge) for hyperedge in simplicial_complex]
        super().__init__(self.numerical_simplicial_complex, self.number_of_variables)
        self.as_string = stringify_in_list(map(stringify_in_tuple, simplicial_complex))
        
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

