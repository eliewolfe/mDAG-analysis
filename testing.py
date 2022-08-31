from __future__ import absolute_import
from hypergraphs import Hypergraph
from directed_structures import DirectedStructure
from mDAG_advanced import mDAG


G3=mDAG(DirectedStructure([(0,1),(2,3)],4),Hypergraph([(0,1),(0,2,3)],4))
#Relabel:
# 1->2, 0->1, 2->0, 3->3
G4=mDAG(DirectedStructure([(0,3),(1,2)],4),Hypergraph([(1,2),(0,1,3)],4))
print(G3.unique_unlabelled_id, G4.unique_unlabelled_id)
l1 = G3.infeasible_binary_supports_n_events_unlabelled(3)
l2 = G4.infeasible_binary_supports_n_events_unlabelled(3)
print(G3.infeasible_binary_supports_n_events_unlabelled(3))
print(G4.infeasible_binary_supports_n_events_unlabelled(3))