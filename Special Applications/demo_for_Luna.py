from __future__ import absolute_import
import numpy as np
from mDAG_advanced import mDAG
from hypergraphs import LabelledHypergraph
from directed_structures import LabelledDirectedStructure
import itertools

ds = LabelledDirectedStructure(variable_names=["A", "B", "C"],
                                               edge_list=[])
sc = LabelledHypergraph(variable_names=["A", "B", "C"],
                                        simplicial_complex=[("A", "B"), ("A", "C"), ("B", "C")])
mDAG_triangle = mDAG(directed_structure_instance=ds,
              simplicial_complex_instance=sc)
for i in range(2,7):

    support_testing_instance = mDAG_triangle.support_testing_instance((3,3,3), i)
    infeasible_supports = support_testing_instance.unique_infeasible_supports_beyond_esep_as_expanded_matrices()
    for s in infeasible_supports:
        print(s)
