from __future__ import absolute_import
import networkx as nx
import numpy as np
from hypergraphs import Hypergraph
from directed_structures import DirectedStructure
from mDAG_advanced import mDAG


G_problem=mDAG(DirectedStructure( [(1,0),(1,2)],3),Hypergraph([(0,1),(1,2)],3))

infeasible = G_problem.infeasible_binary_supports_n_events_beyond_esep(9)

G_problem.support_testing_instance_binary(4).from_integer_to_matrix(list(infeasible))

G_problem

