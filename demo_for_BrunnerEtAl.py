from __future__ import absolute_import
from hypergraphs import Hypergraph
from directed_structures import DirectedStructure
from mDAG_advanced import mDAG

square_mDAG = mDAG(DirectedStructure([], 4),
                   Hypergraph([(0, 1), (1, 2), (2, 3), (3, 0)], 4))
infeasible_patterns = square_mDAG.infeasible_binary_supports_beyond_dsep_as_matrices_up_to(15, verbose=True)
print(len(infeasible_patterns))
print(infeasible_patterns[-1])
# It turns out there are no infeasible supports for the square beyond 10 events.
# There are 163 classically infeasible patterns before compression due to dihedral symmetry.