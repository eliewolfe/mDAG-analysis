from __future__ import absolute_import
import numpy as np
import itertools
from radix import bitarray_to_int
from sys import hexversion
if hexversion >= 0x3080000:
    from functools import cached_property
elif hexversion >= 0x3060000:
    from backports.cached_property import cached_property
else:
    cached_property = property

from radix import from_bits
from merge import merge_intersection
import itertools
from utilities import stringify_in_tuple, stringify_in_list

def has_length_greater_than_one(stuff):
    return len(stuff)>1
def drop_singletons(hypergraph):
    return filter(has_length_greater_than_one,hypergraph)
def permute_bit_array(bitarray, perm):
    almost_new_sc = bitarray[:, list(perm)]
    return almost_new_sc[np.lexsort(almost_new_sc.T)]
def bit_array_permutations(bitarray):
    assert len(bitarray.shape)==2, 'Not a bitarray!'
    n = bitarray.shape[-1]
    return (permute_bit_array(bitarray, perm) for perm in itertools.permutations(range(n)))








class hypergraph:
    def __init__(self, extended_simplicial_complex):
        self.simplicial_complex = extended_simplicial_complex
        self.number_of_latent = len(self.simplicial_complex)
        self.nof_nodes = max(map(max, self.simplicial_complex)) + 1

    @cached_property
    def compressed_simplicial_complex(self):
        return list(drop_singletons(self.simplicial_complex))

    @cached_property
    def compressed_number_of_latent(self):
        return len(self.compressed_simplicial_complex)

    @cached_property
    def as_tuples(self):
        return tuple(sorted(map(lambda s: tuple(sorted(s)), self.simplicial_complex)))

    @cached_property
    def as_bit_array(self):
        r = np.zeros((self.compressed_number_of_latent, self.nof_nodes), dtype=bool)
        for i, lp in enumerate(self.compressed_simplicial_complex):
            r[i, lp] = True
        return r[np.lexsort(r.T)]

    @cached_property
    def districts(self):
        return merge_intersection(self.simplicial_complex)

    @cached_property
    def as_integer(self):
        return from_bits(self.as_bit_array.ravel()).astype(np.ulonglong).tolist()

    @staticmethod
    def bit_array_to_integer(bitarray):
        return from_bits(bitarray.ravel()).astype(np.ulonglong).tolist()

    @property
    def as_unlabelled_integer(self):
        return min(self.bit_array_to_integer(ba) for ba in bit_array_permutations(self.as_bit_array))

    @cached_property
    def as_string(self):
        return stringify_in_list(map(stringify_in_tuple, sorted(self.simplicial_complex)))

class undirected_graph:
    def __init__(self, hyperedges):
        self.nof_nodes = max(map(max, hyperedges)) + 1
        self.as_edges = tuple(itertools.chain.from_iterable(
            itertools.combinations(hyperedge, 2) for hyperedge in hyperedges))

    # @property
    # def as_edges(self):
    #     return itertools.chain.from_iterable(
    #         itertools.combinations(hyperedge, 2) for hyperedge in self.extended_simplicial_complex)

    @cached_property
    def as_edges_array(self):
        # return np.fromiter(self.as_edges, int).reshape((-1,2))
        return np.asarray(self.as_edges)

    @cached_property
    def as_edges_bit_array(self):
        r = np.zeros((len(self.as_edges_array), self.nof_nodes), dtype=bool)
        np.put_along_axis(r, self.as_edges_array)
        return r[np.lexsort(r.T)]

    @staticmethod
    def edges_bit_array_to_integer(bitarray):
        """
        Quick base conversion algorithm.
        """
        assert len(bitarray.shape) == 2, 'Not a bitarray!'
        n = bitarray.shape[-1]
        return np.frompyfunc(lambda a, b: n * a + b, 2, 1).reduce(np.where(bitarray)[-1])

    @property
    def as_edges_unlabelled_integer(self):
        return min(self.edges_bit_array_to_integer(ba) for ba in bit_array_permutations(self.as_edges_bit_array))

