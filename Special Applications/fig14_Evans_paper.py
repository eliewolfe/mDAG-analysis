from __future__ import absolute_import
from quantum_mDAG import QmDAG, as_classical_QmDAG
from hypergraphs import Hypergraph
from directed_structures import DirectedStructure
from proving_IC_Gaps import not_interesting_with_infeasible_supports
from proving_QC_Gaps import QC_gap_by_Fritz_without_node_splitting

Fig_14a = QmDAG(DirectedStructure([(0,3), (1,2)],4),Hypergraph([], 4),Hypergraph([(0,1),(1,3),(3,2),(2,0)],4))
Fig_14b = QmDAG(DirectedStructure([(0,1), (1,3),(2,3)],4),Hypergraph([], 4),Hypergraph([(0,2),(1,2),(0,3)],4))
Fig_14c = QmDAG(DirectedStructure([(0,1), (1,2),(2,3)],4),Hypergraph([], 4),Hypergraph([(0,2),(1,3),(0,3)],4))

print("Fig 14a has a QC Gap by Fritz Trick=",Fig_14a in QC_gap_by_Fritz_without_node_splitting)
print("Fig 14b is not shown to have an IC Gap by any of the tricks=",as_classical_QmDAG(Fig_14b.as_mDAG).unique_unlabelled_id in set(qmdag.unique_unlabelled_id for qmdag in IC_remaining_representatives))
print("Fig 14c has a QC Gap by Fritz Trick=",Fig_14c in QC_gap_by_Fritz_without_node_splitting)

