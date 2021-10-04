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
from utilities import stringify_in_set, stringify_in_list, partsextractor
from more_itertools import chunked

from functools import lru_cache

@lru_cache(maxsize=5)
def empty_digraph(n):
    baseg = nx.DiGraph()
    baseg.add_nodes_from(range(n))
    return baseg

def transitive_closure(adjmat):
    n=len(adjmat)
    closure_mat = np.bitwise_or(np.asarray(adjmat, dtype=bool),np.identity(n, dtype=bool))
    while n>0:
        n = np.floor_divide(n,2)
        next_closure_mat = np.matmul(closure_mat, closure_mat)
        if np.array_equal(closure_mat,next_closure_mat):
            break
        else:
            closure_mat=next_closure_mat
    return np.bitwise_and(closure_mat,np.invert(np.identity(len(adjmat), dtype=bool)))

def transitive_reduction(adjmat):
    # n = len(adjmat)
    closure_mat=transitive_closure(adjmat)
    # closure_minus_identity = np.bitwise_and(transitive_closure(adjmat),np.invert(np.identity(n, dtype=bool)))
    return np.bitwise_and(closure_mat, np.invert(np.matmul(closure_mat, closure_mat)))






class DirectedStructure:
    """
    This class is NOT meant to encode mDAGs. As such, we do not get into an implementation of predecessors or successors here.
    """
    def __init__(self, numeric_edge_list, n):
        self.number_of_visible = n
        self.visible_nodes = list(range(self.number_of_visible))
        self.edge_list = numeric_edge_list
        self.number_of_edges = len(numeric_edge_list)
        if self.edge_list:
            assert max(map(max, self.edge_list)) + 1 <= self.number_of_visible, "More nodes referenced than expected."




    @cached_property
    def as_tuples(self):
        return tuple(sorted(map(nx.utils.to_tuple, self.edge_list)))

    @cached_property
    def as_set_of_tuples(self):
        return set(map(nx.utils.to_tuple, self.edge_list))

    @property
    def as_edges_array(self):
        return np.asarray(self.edge_list, dtype=int).reshape((-1,2)) #No sorting or deduplication.

    @cached_property
    def as_bit_square_matrix(self):
        r = np.zeros((self.number_of_visible, self.number_of_visible), dtype=bool)
        r[tuple(self.as_edges_array.T)] = True
        return r

    @cached_property
    def as_bit_square_matrix_plus_eye(self):
        #Used for computing parents_plus
        return np.bitwise_or(self.as_bit_square_matrix, np.identity(self.number_of_visible, dtype=bool))

    @cached_property
    def observable_parentsplus_list(self):
        return list(map(frozenset, map(np.flatnonzero, self.as_bit_square_matrix_plus_eye.T)))

    @staticmethod
    def bit_array_to_integer(bitarray):
        return from_bits(bitarray.ravel()).astype(np.ulonglong).tolist()

    @cached_property
    def as_integer(self):
        return self.bit_array_to_integer(self.as_bit_square_matrix)

    def permute_bit_square_matrix(self, perm):
        return self.as_bit_square_matrix[perm][:, perm]

    @property
    def bit_square_matrix_permutations(self):
        return (self.permute_bit_square_matrix(perm) for perm in map(list,itertools.permutations(self.visible_nodes)))

    @cached_property
    def as_unlabelled_integer(self):
        return min(self.bit_array_to_integer(ba) for ba in self.bit_square_matrix_permutations)

    @cached_property
    def as_string(self):
        return stringify_in_set(
            str(partsextractor(self.visible_nodes, i)) + ':' + stringify_in_list(partsextractor(self.visible_nodes, v))
                # for i, v in nx.to_dict_of_lists(self.DirectedStructure).items()
                for i, v in enumerate(map(np.flatnonzero, self.as_bit_square_matrix))
                )

    def __str__(self):
        return self.as_string

    def __repr__(self):
        return self.as_string

    # @cached_property
    # def parents_of_for_supports_analysis(self):
    #     return list(map(np.flatnonzero, self.as_bit_square_matrix.T))

    #I'm hoping to eventually transition away from networkx entirely.
    @cached_property
    def as_networkx_graph(self):
        g = empty_digraph(self.number_of_visible).copy()
        g.add_edges_from(self.edge_list)
        return g

    def can_D1_minimally_simulate_D2(D1, D2):
        """
        D1 and D2 are networkx.DiGraph objects.
        We say that D1 can 'simulate' D2 if the edges of D2 are contained within those of D1.
        """
        return D1.number_of_edges == D2.number_of_edges + 1 and D2.as_set_of_tuples.issubset(D1.as_set_of_tuples)



class LabelledDirectedStructure(DirectedStructure):
    """
    This class is NOT meant to encode mDAGs. As such, we do not get into an implementation of predecessors or successors here.
    """
    def __init__(self, variable_names, edge_list):
        self.variable_names = variable_names
        self.number_of_variables = len(variable_names)
        self.variables_as_range = tuple(range(self.number_of_variables))
        self.translation_dict = dict(zip(self.variable_names, self.variables_as_range))
        self.edge_list_variable_names=edge_list
        if all(isinstance(v, int) for v in self.variable_names):
            if np.array_equal(self.variable_names, self.variables_as_range):
                self.variable_are_range = True
                self.edge_list = edge_list
            else:
                self.variable_are_range = False
                self.edge_list = list(chunked(partsextractor(self.translation_dict, tuple(itertools.chain.from_iterable(edge_list))),2))
        super().__init__(self.edge_list, self.number_of_variables)

    @cached_property
    def as_string(self):
        return stringify_in_set(
            str(partsextractor(self.variable_names, i)) + ':' + stringify_in_list(partsextractor(self.variable_names, v))
                # for i, v in nx.to_dict_of_lists(self.DirectedStructure).items()
                for i, v in enumerate(map(np.flatnonzero, self.as_bit_square_matrix))
                )
    @cached_property
    def as_networkx_graph_arbitrary_names(self):
        g=nx.DiGraph()
        g.add_nodes_from(self.variable_names)
        g.add_edges_from(self.edge_list_variable_names)
        return g

    def __str__(self):
        return self.as_string

    def __repr__(self):
        return self.as_string
  
  
