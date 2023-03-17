# This is a demo script for Robin Evans to check if an mDAG has any infeasible supports which are not infeasible due to e-separation
from __future__ import absolute_import
import numpy as np
from mDAG_advanced import mDAG
from hypergraphs import LabelledHypergraph
from directed_structures import LabelledDirectedStructure

# Preliminary example: There are no inequality constraints.
directed_structure = LabelledDirectedStructure(variable_names=["A", "B", "C", "D"],
                                               edge_list=[("A", "C"), ("A", "D"), ("B", "C"), ("C", "D")])
simplicial_complex = LabelledHypergraph(variable_names=["A", "B", "C", "D"],
                                        simplicial_complex=[("A", "C"), ("B", "C")])
mDAG_0 = mDAG(directed_structure_instance=directed_structure,
              simplicial_complex_instance=simplicial_complex)
print("Ordinary-Markov equivalent mDAG:", mDAG_0)
print("Confirming that is does NOT find any infeasible supports beyond dsep, up to 15 events!")
# print("Debugging: ", mDAG_0.infeasible_binary_supports_n_events_beyond_dsep_as_matrices(2))
print("CBOM proven? ", not mDAG_0.no_infeasible_binary_supports_beyond_dsep_up_to(15))
print("\n\n")


# FIRST EXAMPLE: There are nontrivial e-sep relations.
directed_structure = LabelledDirectedStructure(variable_names=["X", "A", "B", "C"],
                                               edge_list=[("X", "A"), ("A", "B")])
simplicial_complex = LabelledHypergraph(variable_names=["X", "A", "B", "C"],
                                        simplicial_complex=[("A", "C"), ("B", "C")])
mDAG_1 = mDAG(directed_structure_instance=directed_structure,
              simplicial_complex_instance=simplicial_complex)
print("FLAG mDAG:", mDAG_1)
print("Checking so-called FLAG mDAG, for supports over 2 events:")
print("CBOM proven? ", not mDAG_1.no_infeasible_binary_supports_beyond_dsep_up_to(2))
print("\n\n")

# SECOND EXAMPLE: No nontrivial e-sep relations.
directed_structure = LabelledDirectedStructure(variable_names=["X", "A", "B", "C"],
                                               edge_list=[("X", "A"), ("A", "B"), ("X", "B")])
simplicial_complex = LabelledHypergraph(variable_names=["X", "A", "B", "C"],
                                        simplicial_complex=[("A", "C"), ("B", "C")])
mDAG_2 = mDAG(directed_structure_instance=directed_structure,
              simplicial_complex_instance=simplicial_complex)
print("REINFORCED FLAG mDAG:", mDAG_2)
print("Checking so-called REINFORCED FLAG mDAG, for supports over 2 events:")
print("CBOM proven? ", not mDAG_2.no_infeasible_binary_supports_beyond_dsep_up_to(2))
print("Checking so-called REINFORCED FLAG mDAG, for supports over 4 events:")
print("CBOM proven? ", not mDAG_2.no_infeasible_binary_supports_beyond_dsep_up_to(4))
print("\n\n")

# FINAL EXAMPLE: Challenge DAG, very hard to prove not saturated!
directed_structure = LabelledDirectedStructure(variable_names=["X", "Y", "A", "B"],
                                               edge_list=[("X", "A"), ("Y", "B")])
simplicial_complex = LabelledHypergraph(variable_names=["X", "Y", "A", "B"],
                                        simplicial_complex=[("X", "A", "B"), ("Y", "A", "B"), ("X", "Y")])
mDAG_3 = mDAG(directed_structure_instance=directed_structure,
              simplicial_complex_instance=simplicial_complex)
print("'Challenge' mDAG:", mDAG_3)
print("Checking 'Challenge' mDAG, for supports over 2 events:")
print("CBOM proven? ", not mDAG_3.no_infeasible_binary_supports_beyond_dsep_up_to(2))
print("Checking 'Challenge' FLAG mDAG, for supports over 4 events:")
print("CBOM proven? ", not mDAG_3.no_infeasible_binary_supports_beyond_dsep_up_to(4))

