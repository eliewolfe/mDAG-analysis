from __future__ import absolute_import
from hypergraphs import Hypergraph
from directed_structures import DirectedStructure
from mDAG_advanced import mDAG
import numpy as np

G_a=mDAG(DirectedStructure([(0,1),(1,2),(2,3),(1,3)],4),Hypergraph([(0,2),(1,2),(0,3)],4))
G_b=mDAG(DirectedStructure([(0,1),(1,2),(2,3),(1,3)],4),Hypergraph([(0,2),(1,2),(2,3),(0,3)],4))
G_c=mDAG(DirectedStructure([(0,1),(1,2),(2,3),(1,3)],4),Hypergraph([(0,2),(1,3),(0,3)],4))
G_d=mDAG(DirectedStructure([(0,1),(1,2),(2,3),(1,3)],4),Hypergraph([(0,2),(2,3),(1,3),(0,3)],4))
G_e=mDAG(DirectedStructure([(0,1),(1,2),(2,3),(1,3)],4),Hypergraph([(0,2),(1,3),(1,2),(0,3)],4))
G_f=mDAG(DirectedStructure([(0,1),(1,2),(2,3),(1,3)],4),Hypergraph([(0,2),(1,3),(2,3),(1,2),(0,3)],4))
G_g=mDAG(DirectedStructure([(0,1),(1,2),(2,3),(1,3)],4),Hypergraph([(0,2),(1,2,3),(0,3)],4))

advanced_support = np.array([[0, 0, 0, 0],
                      [0, 0, 1, 0],
                      [0, 1, 0, 0],
                      [1, 0, 0, 0],
                      [1, 1, 0, 1],
                      [2, 0, 0, 1],
                      [2, 1, 1, 0]], dtype=int);

for g in [G_a, G_b, G_c, G_d, G_e, G_f, G_g]:
    print(g.support_testing_instance((4,2,2,2),7).feasibleQ_from_matrix(
    occurring_events=advanced_support))
#
# print(G_a.support_testing_instance((4,2,2,2),7).feasibleQ_from_matrix(
#     occurring_events=[[0, 0, 0, 0],
#                       [0, 0, 1, 0],
#                       [0, 1, 0, 0],
#                       [1, 0, 0, 0],
#                       [1, 1, 0, 1],
#                       [2, 0, 0, 1],
#                       [2, 1, 1, 0]]))
#
# print(G_a.support_testing_instance((4,2,2,2),6).feasibleQ_from_matrix(
#     occurring_events=[[0, 0, 0, 0],
#                       [0, 0, 1, 0],
#                       [0, 1, 0, 0],
#                       [1, 0, 0, 0],
#                       [1, 1, 0, 1],
#                       [2, 0, 0, 1]]))