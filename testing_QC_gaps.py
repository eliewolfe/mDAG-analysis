from __future__ import absolute_import
import networkx as nx
import numpy as np
from hypergraphs import Hypergraph
from directed_structures import DirectedStructure
from mDAG_advanced import mDAG

G_Bell=mDAG(DirectedStructure([(0,3),(1,2)],4),Hypergraph([(0,),(1,),(2,3)],4))
   
for c in Observable_mDAGs4.foundational_eqclasses:
    if G_Bell in c:
        for element in c:
            print(element in no_inf_sups_beyond_esep4)
            
#########################          
            
G_Triangle=mDAG(DirectedStructure( [],3),Hypergraph([(0,1),(1,2),(2,0)],3))
    
for c in Observable_mDAGs3.foundational_eqclasses:
    if G_Triangle in c:
        for element in c:
            print(element in no_inf_sups_beyond_esep3)
        
######################### 

G_Square=mDAG(DirectedStructure( [],4),Hypergraph([(0,1),(1,2),(2,3),(3,0)],4))
    
for c in Observable_mDAGs4.foundational_eqclasses:
    if G_Bell in c:
        for element in c:
            print(element in no_inf_sups_beyond_esep4)
            
######################### 

G_Instrumental=mDAG(DirectedStructure( [(0,1),(1,2)],3),Hypergraph([(0,),(1,2)],3))

for i in range(2,9):
    infeasible = G_Instrumental.smart_infeasible_binary_supports_n_events(i)
    print(infeasible)
    
for c in Observable_mDAGs3.foundational_eqclasses:
    if G_Instrumental in c:
        for element in c:
            print(element in no_inf_sups_beyond_esep3)

for i in range(2,11):
    infeasible3=G_problem.smart_infeasible_supports_n_events_card_3(i)
    print(infeasible3)

######################### 
    
G_Evans=mDAG(DirectedStructure( [(0,1),(0,2)],3),Hypergraph([(0,1),(0,2)],3))

for c in Observable_mDAGs3.foundational_eqclasses:
    if G_Evans in c:
        for element in c:
            print(element in no_inf_sups_beyond_esep3)

for i in range(2,9):
    infeasible = G_Evans.smart_infeasible_binary_supports_n_events(i)
    print(infeasible)

for i in range(2,9):
    infeasible3=G_Evans.smart_infeasible_supports_n_events_card_3(i)
    print(infeasible3)
    
######################### 


    
