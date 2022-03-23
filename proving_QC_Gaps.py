from __future__ import absolute_import
from hypergraphs import Hypergraph
from directed_structures import DirectedStructure
from mDAG_advanced import mDAG
from quantum_mDAG import QmDAG, upgrade_to_QmDAG
from metagraph_advanced import Observable_mDAGs_Analysis

# if __name__ == '__main__':
Observable_mDAGs4 = Observable_mDAGs_Analysis(nof_observed_variables=4, max_nof_events_for_supports=0)
mDAGs4_representatives = Observable_mDAGs4.representative_mDAGs_list
QmDAGs4_representatives = list(map(upgrade_to_QmDAG, mDAGs4_representatives))

print("Marina style: ")
G_Instrumental1 = mDAG(DirectedStructure([(0, 1), (1, 2)], 3), Hypergraph([(1, 2)], 3))
G_Instrumental2 = mDAG(DirectedStructure([(0, 1), (1, 2)], 3), Hypergraph([(0, 1), (1, 2)], 3))
G_Instrumental3 = mDAG(DirectedStructure([(1, 2)], 3), Hypergraph([(0, 1), (1, 2)], 3))
G_Bell = mDAG(DirectedStructure([(0, 1), (2, 3)], 4), Hypergraph([(1, 2)], 4))
G_Bell_wComm = mDAG(DirectedStructure([(0, 1), (2, 3), (1, 2)], 4), Hypergraph([], 4))
G_Triangle = mDAG(DirectedStructure([], 3), Hypergraph([(1, 2), (2, 0), (0, 1)], 3))
G_Evans = mDAG(DirectedStructure([(0, 1), (0, 2)], 3), Hypergraph([(0, 1), (0, 2)], 3))
known_QC_Gaps_mDAGs = [G_Instrumental1, G_Instrumental2, G_Instrumental3, G_Bell, G_Bell_wComm, G_Triangle]
known_QC_Gaps_mDAGs_id = set(special_mDAG.unique_unlabelled_id for special_mDAG in known_QC_Gaps_mDAGs)

# For the trick of fixing to point distribution, we can simply compare mDAGs. The QmDAG structure is going to be useful only in the marginalization case (where classical and quantum latents appear)
def reduces_to_knownQCGap_by_intervention(mDAG):
    for node in mDAG.visible_nodes:
        if mDAG.fix_to_point_distribution(node).unique_unlabelled_id in known_QC_Gaps_mDAGs_id:
            return True
    return False
QC_gap_by_Intervention = list(filter(reduces_to_knownQCGap_by_intervention, mDAGs4_representatives))
print(len(mDAGs4_representatives))
print(len(QC_gap_by_Intervention))


print("Elie style: ")
QG_Instrumental1 = QmDAG(DirectedStructure([(0, 1), (1, 2)], 3), Hypergraph([], 3), Hypergraph([(1, 2)], 3))
QG_Instrumental2 = QmDAG(DirectedStructure([(0, 1), (1, 2)], 3), Hypergraph([], 3), Hypergraph([(0, 1), (1, 2)], 3))
QG_Instrumental3 = QmDAG(DirectedStructure([(1, 2)], 3), Hypergraph([], 3), Hypergraph([(0, 1), (1, 2)], 3))
QG_Instrumental2b = QmDAG(DirectedStructure([(0, 1), (1, 2)], 3), Hypergraph([(0, 1)], 3), Hypergraph([(1, 2)], 3))
QG_Instrumental3b = QmDAG(DirectedStructure([(1, 2)], 3), Hypergraph([(0, 1)], 3), Hypergraph([(1, 2)], 3))
### WARNING, WE NEED THE ENTIRE EQUIVALENCE CLASS OF QUANTUM BELL = 5 + 2 + 1 = 8 qmDAGs
# QG_Bell = QmDAG(DirectedStructure([(0, 1), (2, 3)], 4), Hypergraph([], 4), Hypergraph([(1, 2)], 4))
# QG_Bell_wComm = QmDAG(DirectedStructure([(0, 1), (2, 3), (1, 2)], 4), Hypergraph([], 4), Hypergraph([(1, 2)], 4))
###
QG_Triangle1 = QmDAG(DirectedStructure([], 3), Hypergraph([], 3), Hypergraph([(1, 2), (2, 0), (0, 1)], 3))
QG_Triangle2 = QmDAG(DirectedStructure([], 3), Hypergraph([(1, 2)], 3), Hypergraph([(2, 0), (0, 1)], 3))
QG_Triangle3 = QmDAG(DirectedStructure([], 3), Hypergraph([(1, 2), (2, 0)], 3), Hypergraph([(0, 1)], 3))
known_QC_Gaps_QmDAGs = {QG_Instrumental1, QG_Instrumental2, QG_Instrumental3,
                        QG_Instrumental2b, QG_Instrumental3b,
                        QG_Triangle1, QG_Triangle2, QG_Triangle3}
known_QC_Gaps_QmDAGs_id = set(special_QmDAG.unique_unlabelled_id for special_QmDAG in known_QC_Gaps_QmDAGs)




print(len(QmDAGs4_representatives))

def reduces_to_knownQCGap_by_PD_trick(qmDAG):
    return not known_QC_Gaps_QmDAGs_id.isdisjoint(qmDAG.unique_unlabelled_ids_obtainable_by_PD_trick)
QC_gap_by_PD_trick = list(filter(reduces_to_knownQCGap_by_PD_trick, QmDAGs4_representatives))

print(len(QC_gap_by_PD_trick))

def reduces_to_knownQCGap_by_naive_marginalization(qmDAG):
    return not known_QC_Gaps_QmDAGs_id.isdisjoint(qmDAG.unique_unlabelled_ids_obtainable_by_naive_marginalization)
QC_gap_by_naive_marginalization = list(filter(reduces_to_knownQCGap_by_naive_marginalization, set(QmDAGs4_representatives).difference(QC_gap_by_PD_trick)))

print(len(QC_gap_by_naive_marginalization))
print(QC_gap_by_naive_marginalization)

# debug_QmDAG = QmDAG(
#         DirectedStructure([(0, 1), (1, 2), (2, 3)], 4),
#         Hypergraph([], 4),
#         Hypergraph([(0, 1), (1, 3), (2, 3)], 4)
#     )
# print("Is this even considered? ", debug_QmDAG in QmDAGs4_representatives)
# print("Is it detected by PD? ", debug_QmDAG in QC_gap_by_PD_trick)
# print("Is it detected by our code? ", debug_QmDAG in QC_gap_by_naive_marginalization)

def reduces_to_knownQCGap_by_marginalization(qmDAG):
    return not known_QC_Gaps_QmDAGs_id.isdisjoint(qmDAG.unique_unlabelled_ids_obtainable_by_marginalization)
QC_gap_by_marginalization = list(filter(reduces_to_knownQCGap_by_marginalization, set(QmDAGs4_representatives).difference(
    QC_gap_by_PD_trick,
    QC_gap_by_naive_marginalization
)))

print(len(QC_gap_by_marginalization))
print(QC_gap_by_marginalization)

# def reduces_to_knownQCGap_by_PD_or_teleportation(qmDAG):
#     return not known_QC_Gaps_QmDAGs_id.isdisjoint(
#         set(qmDAG.unique_unlabelled_ids_obtainable_by_marginalization).difference(
#             qmDAG.unique_unlabelled_ids_obtainable_by_marginalization
#         ))
# QC_gaps_where_teleportation_is_relevant = list(filter(reduces_to_knownQCGap_by_marginalization, QmDAGs4_representatives))
# print(len(QC_gaps_where_teleportation_is_relevant))


