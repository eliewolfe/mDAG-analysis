from __future__ import absolute_import
from quantum_mDAG import QmDAG
from hypergraphs import Hypergraph
from directed_structures import DirectedStructure

Fig_14a = QmDAG(DirectedStructure([(0,3), (1,2)],4),Hypergraph([], 4),Hypergraph([(0,1),(1,3),(3,2),(2,0)],4))
Fig_14b = QmDAG(DirectedStructure([(0,1), (1,3),(2,3)],4),Hypergraph([], 4),Hypergraph([(0,2),(1,2),(0,3)],4))
Fig_14c = QmDAG(DirectedStructure([(0,1), (1,2),(2,3)],4),Hypergraph([], 4),Hypergraph([(0,2),(1,3),(0,3)],4))

print(Fig_14a)
print("Can Fritz delete facets (0,2) and (0,1)? =",Fig_14a.assess_Fritz_Wolfe_style(0, {}, {}, {frozenset((0,2)),frozenset((0,1))}))
print(list(Fig_14a.apply_Fritz_trick(node_decomposition=False))[3])

