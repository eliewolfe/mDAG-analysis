# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 15:08:55 2021

@author: Marina ♡
"""
from __future__ import absolute_import
import networkx as nx
import numpy as np
from hypergraphs import Hypergraph
from directed_structures import DirectedStructure
from mDAG_advanced import mDAG


G_problem=mDAG(DirectedStructure( [(0,1),(1,2)],3),Hypergraph([(0,),(1,2)],3))
print(G_problem)

for i in range(2,9):
    infeasible = G_problem.smart_infeasible_binary_supports_n_events(i)
    print(infeasible)
    
for c in Observable_mDAGs3.foundational_eqclasses:
    if G_problem in c:
        for element in c:
            print(element in no_inf_sups_beyond_esep3)

for i in range(2,11):
    infeasible3=G_problem.smart_infeasible_supports_n_events_card_3(i)
    print(infeasible3)

G_problem.smart_support_testing_instance(4).from_integer_to_matrix(list(infeasible))

G_triangle=mDAG(DirectedStructure( [],3),Hypergraph([(0,1),(1,2),(2,0)],3))

for i in range(2,9):
    infeasible = G_triangle.smart_infeasible_binary_supports_n_events(i)
    print(infeasible)

for i in range(2,9):
    infeasible3=G_triangle.smart_infeasible_supports_n_events_card_3(i)
    print(infeasible3)
    
G_evans=mDAG(DirectedStructure( [(0,1),(0,2)],3),Hypergraph([(0,1),(0,2)],3))

for i in range(2,9):
    infeasible = G_evans.smart_infeasible_binary_supports_n_events(i)
    print(infeasible)

for c in Observable_mDAGs3.foundational_eqclasses:
    if G_evans in c:
        for element in c:
            print(element in no_inf_sups_beyond_esep3)

for i in range(2,9):
    infeasible3=G_evans.smart_infeasible_supports_n_events_card_3(i)
    print(infeasible3)
    
G_Bell=mDAG(DirectedStructure([(0,3),(1,2)],4),Hypergraph([(0,),(1,),(2,3)],4))
G_Bell.fundamental_graphQ

for i in range(2,9):
    infeasible = G_Bell.smart_infeasible_binary_supports_n_events(i)
    print(infeasible)
    
for c in Observable_mDAGs.foundational_eqclasses:
    if G_Bell in c:
        for element in c:
            print(element in no_inf_sups_beyond_esep)

G_square=mDAG(DirectedStructure( [],4),Hypergraph([(0,1),(1,2),(2,3),(3,0)],4))

for i in range(2,9):
    infeasible = G_square.smart_infeasible_binary_supports_n_events(i)
    print(infeasible)
    
