from __future__ import absolute_import
from hypergraphs import Hypergraph
from directed_structures import DirectedStructure
from mDAG_advanced import mDAG
from supports_beyond_esep import SmartSupportTesting


G_Evans=mDAG(DirectedStructure([(0,2)],3),Hypergraph([(0,1),(0,2)],3))
print(G_Evans.no_infeasible_binary_supports_beyond_dsep_up_to(4))

G_okay=mDAG(DirectedStructure([(0,3),(1,2)],4),Hypergraph([(2,3)],4))
print(G_okay.no_infeasible_binary_supports_beyond_dsep_up_to(4))


G_problem=mDAG(DirectedStructure([(0,2),(0,1)],4),Hypergraph([(0,), (1,3),(2,3)],4))
print(G_problem.no_infeasible_binary_supports_beyond_dsep_up_to(4, verbose=True))

