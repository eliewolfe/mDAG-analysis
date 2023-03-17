from __future__ import absolute_import
from hypergraphs import Hypergraph
from directed_structures import DirectedStructure
from mDAG_advanced import mDAG
import numpy as np

G_a = mDAG(DirectedStructure([(0, 1), (1, 2), (2, 3), (1, 3)], 4),
           Hypergraph([(0, 2), (1, 2), (0, 3)], 4))
G_b = mDAG(DirectedStructure([(0, 1), (1, 2), (2, 3), (1, 3)], 4),
           Hypergraph([(0, 2), (1, 2), (2, 3), (0, 3)], 4))
G_c = mDAG(DirectedStructure([(0, 1), (1, 2), (2, 3), (1, 3)], 4),
           Hypergraph([(0, 2), (1, 3), (0, 3)], 4))
G_d = mDAG(DirectedStructure([(0, 1), (1, 2), (2, 3), (1, 3)], 4),
           Hypergraph([(0, 2), (2, 3), (1, 3), (0, 3)], 4))
G_e = mDAG(DirectedStructure([(0, 1), (1, 2), (2, 3), (1, 3)], 4),
           Hypergraph([(0, 2), (1, 3), (1, 2), (0, 3)], 4))
G_f = mDAG(DirectedStructure([(0, 1), (1, 2), (2, 3), (1, 3)], 4),
           Hypergraph([(0, 2), (1, 3), (2, 3), (1, 2), (0, 3)], 4))
G_g = mDAG(DirectedStructure([(0, 1), (1, 2), (2, 3), (1, 3)], 4),
           Hypergraph([(0, 2), (1, 2, 3), (0, 3)], 4))

advanced_support = np.array([[0, 0, 0, 0],
                             [0, 0, 1, 0],
                             [0, 1, 0, 0],
                             [1, 0, 0, 0],
                             [1, 1, 0, 1],
                             [2, 0, 0, 1],
                             [2, 1, 1, 0]], dtype=int);

the_7_hard_graphs = [G_a, G_b, G_c, G_d, G_e, G_f, G_g]

# for g in [G_a, G_b, G_c, G_d, G_e, G_f, G_g]:
#     print(g.support_testing_instance((3, 2, 2, 2), 7).feasibleQ_from_matrix(
#         occurring_events=advanced_support))

# ST = G_a.support_testing_instance((3, 2, 2, 2), 7)
# advanced_support_as_int = ST.from_matrix_to_integer(advanced_support)
# print(advanced_support_as_int)
# canonical_int = ST.canonical_under_outcome_relabelling(advanced_support_as_int)
# print(ST.int_dtype)
# print(canonical_int)
# print(ST.from_integer_to_matrix(canonical_int))
# attempt = ST.attempt_to_find_one_infeasible_support(verbose=True)
# print(attempt)
if __name__ == '__main__':
    ST = G_a.support_testing_instance((3, 2, 2, 2), 7)
    print(ST.int_dtype)
    candidates = ST.unique_candidate_supports_as_compressed_integers
    print(len(candidates))
    attempt = ST.attempt_to_find_one_infeasible_support(verbose=True)
    print(attempt)

    ST = G_e.support_testing_instance((3, 3, 2, 2), 9)
    print(ST.int_dtype)
    candidates = ST.unique_candidate_supports_as_compressed_integers
    print(len(candidates))
    attempt = ST.attempt_to_find_one_infeasible_support(verbose=True)
    print(attempt)
