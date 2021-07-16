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

def mdag_to_int(ds_bitarray, sc_bitarray):
    return from_bits(np.vstack((sc_bitarray, ds_bitarray)).ravel()).astype(np.ulonglong).tolist()

class mDAG
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
    def as_compressed_graph(self):  # let directed_structure be a DAG initially without latents
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
        return mdag_to_int(self.directed_structure_instance.as_bit_square_matrix, self.simplicial_complex_instance.as_extended_bit_array)

    @cached_property
    def unique_unlabelled_id(self):
        # Returns a unique identification number.
        return min(mdag_to_int(new_ds, new_sc) for (new_ds, new_sc) in zip(
            self.directed_structure_instance.bit_square_matrix_permutations,
            self.simplicial_complex_instance.bit_array_permutations
        ))

