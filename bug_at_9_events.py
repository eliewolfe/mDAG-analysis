# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 15:08:55 2021

@author: Marina ♡
"""
from __future__ import absolute_import
import networkx as nx
import numpy as np
from mDAG_advanced import mDAG


G_problem=mDAG(directed_structure((0,1,2), [(1,0),(1,2)]),hypergraph([(0,1),(1,2)]))

infeasible = G_problem.smart_infeasible_binary_supports_n_events(9)

G_problem.smart_support_testing_instance(4).from_integer_to_matrix(list(infeasible))

G_problem
