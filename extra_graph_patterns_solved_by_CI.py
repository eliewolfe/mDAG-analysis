# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 20:32:01 2021

@author: Marina ♡
"""
from __future__ import absolute_import
import networkx as nx
import numpy as np
from hypergraphs import Hypergraph
from directed_structures import DirectedStructure
from mDAG_advanced import mDAG



G=mDAG(DirectedStructure([(0,1),(2,3),(2,1)],4),Hypergraph([(1,2),(2,3)],4))
G.all_CI   #symmetry: 2 <-> 3


G_relabeled=mDAG(DirectedStructure([(0,1),(3,2),(3,1)],4),Hypergraph([(1,3),(3,2)],4))
G_relabeled.all_CI

G.all_CI==G_relabeled.all_CI

G.all_esep==G_relabeled.all_esep

G.smart_infeasible_binary_supports_n_events(4)==G_relabeled.smart_infeasible_binary_supports_n_events(4)

G1=mDAG(DirectedStructure([],4),Hypergraph([(0,3),(1,2),(1,3),(2,3)],4))
G1.all_CI   #symmetry: 2 <-> 3


G1_relabeled=mDAG(DirectedStructure([],4),Hypergraph([(0,3),(2,1),(2,3),(1,3)],4))
G1_relabeled.all_CI

G1.all_CI==G1_relabeled.all_CI

G1.all_esep==G1_relabeled.all_esep

G1.smart_infeasible_binary_supports_n_events(4)==G1_relabeled.smart_infeasible_binary_supports_n_events(4)

