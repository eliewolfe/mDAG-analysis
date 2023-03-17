from __future__ import absolute_import
import numpy as np
from mDAG_advanced import mDAG
from hypergraphs import LabelledHypergraph
from directed_structures import LabelledDirectedStructure

# print("Reinforced_flag:", intermed_infeas)
# %%
# triangle_ds = LabelledDirectedStructure(variable_names=["A","B","C"],
#                                         edge_list=[])
# triangle_sc = LabelledHypergraph(variable_names=["A","B","C"],
#                            simplicial_complex=[("A","B"), ("A", "C")])
# triangle_mDAG = mDAG(directed_structure_instance=triangle_ds,
#                simplicial_complex_instance=triangle_sc)
# print(triangle_mDAG.infeasible_binary_supports_n_events_as_matrices(3))


UC_ds = LabelledDirectedStructure(variable_names=["A", "B", "C"],
                                  edge_list=[("A", "B"), ("A", "C")])
UC_sc = LabelledHypergraph(variable_names=["A", "B", "C"],
                           simplicial_complex=[("A", "B"), ("A", "C")])
UC_mDAG = mDAG(directed_structure_instance=UC_ds,
               simplicial_complex_instance=UC_sc)

IV_ds = LabelledDirectedStructure(variable_names=["A", "B", "C"],
                                  edge_list=[("B", "A"), ("A", "C")])
IV_sc = LabelledHypergraph(variable_names=["A", "B", "C"],
                           simplicial_complex=[("A", "C")])
IV_mDAG = mDAG(directed_structure_instance=IV_ds,
               simplicial_complex_instance=IV_sc)


import itertools
four_choose_two_game = []
for (i, vals) in enumerate(itertools.combinations(range(4), 2)):
    for val in vals:
        if i==0:
            four_choose_two_game.append([0, i, val])
        elif i==5:
            four_choose_two_game.append([1, i, val])
        else:
            four_choose_two_game.extend([[0, i, val], [1, i, val]])
four_choose_two_game = np.array(four_choose_two_game)
for event in four_choose_two_game.tolist():
    print(event)




# support_testing_instance = UC_mDAG.support_testing_instance(tuple(four_choose_two_game.max(axis=0)+1), len(four_choose_two_game))
support_testing_instance = IV_mDAG.support_testing_instance(tuple(four_choose_two_game.max(axis=0)+1), len(four_choose_two_game))
print(support_testing_instance.feasibleQ_from_matrix(four_choose_two_game))