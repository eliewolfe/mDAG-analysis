from __future__ import absolute_import
import numpy as np
from hypergraphs import Hypergraph
from directed_structures import DirectedStructure
from mDAG_advanced import mDAG


G_Bell=mDAG(DirectedStructure([(0,3),(1,2)],4),Hypergraph([(2,3)],4))

support=np.array([(0,0,0,0),(2,1,1,1),(2,0,1,1)])

#mDAG.support_testing_instance(observed cardinalities, number of events)

G_Bell.support_testing_instance((3,2,2,2),3).feasibleQ_from_matrix(support)


########################################

G_chain=mDAG(DirectedStructure([(0,1),(1,2)],3),Hypergraph([],3))

support=np.array([(0,0,0), (1,1,1)])

G_chain.support_testing_instance((2,2,2),2).feasibleQ_from_matrix(support)

########################################

G_reinforced_flag=mDAG(DirectedStructure([(2,1),(2,3),(1,3)],4),Hypergraph([(0,1),(0,3)],4))


support=np.array([(0,0,0,0),(2,0,0,0),(0,0,0,1),(2,0,0,1),(0,0,1,0),
                  (2,0,1,0),(0,0,1,1),(2,0,1,1),(1,1,0,0),(3,1,0,0),(1,1,0,1),
                  (3,1,0,1),(1,1,1,0),(3,1,1,0),(1,1,1,1),(3,1,1,1)])

G_reinforced_flag.support_testing_instance((4,2,2,2),16).feasibleQ_from_matrix(support)




