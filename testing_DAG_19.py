from __future__ import absolute_import
from hypergraphs import Hypergraph
from directed_structures import DirectedStructure
from mDAG_advanced import mDAG


G_19=mDAG(DirectedStructure([(0,1),(1,2),(1,3),(2,3)],4),Hypergraph([(0,2),(1,2,3),(0,3)],4))

print("Attempt to prove interestingness of this remaining one using TC's algorithm with cardinalities 8222:")
interesting=False
n=2
while n<11 and interesting==False:    #11 is a big number of events choosen randomly, just so it doesn't run forever
    if not G_19.no_infeasible_6222_supports_beyond_dsep_up_to(n):
        print("is shown interesting by TC's algorithm (cardinality 6222) up to",n,"events.")
        interesting=True
        n=n+1
    else: 
        print("is still not shown interesting by TC's algorithm (cardinality 8222) up to",n,"events.")
        n=n+1
        