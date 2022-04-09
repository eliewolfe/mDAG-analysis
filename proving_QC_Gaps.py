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
Observable_mDAGs3 = Observable_mDAGs_Analysis(nof_observed_variables=3, max_nof_events_for_supports=0)


n=0
HLP_6_node_representatives = set()
for eqclass in Observable_mDAGs4.foundational_eqclasses:
    for mdag in eqclass:
        if mdag.simplicial_complex_instance.number_of_nonsingleton_latent == 2:
            HLP_6_node_representatives.add(mdag)
            n+=1
            break
print("Number of 6-node mDAGs:", n)


# print("Marina style: ")
G_Instrumental1 = mDAG(DirectedStructure([(0, 1), (1, 2)], 3), Hypergraph([(1, 2)], 3))
G_Instrumental2 = mDAG(DirectedStructure([(0, 1), (1, 2)], 3), Hypergraph([(0, 1), (1, 2)], 3))
G_Instrumental3 = mDAG(DirectedStructure([(1, 2)], 3), Hypergraph([(0, 1), (1, 2)], 3))
# G_Bell = mDAG(DirectedStructure([(0, 1), (2, 3)], 4), Hypergraph([(1, 2)], 4))
# #G_Bell_wComm = mDAG(DirectedStructure([(0, 1), (2, 3), (1, 2)], 4), Hypergraph([], 4))
G_Triangle = mDAG(DirectedStructure([], 3), Hypergraph([(1, 2), (2, 0), (0, 1)], 3))
G_Evans = mDAG(DirectedStructure([(0, 1), (0, 2)], 3), Hypergraph([(0, 1), (0, 2)], 3))
known_interesting_mDAGs = [G_Instrumental1, G_Instrumental2, G_Instrumental3, G_Evans, G_Triangle]
known_interesting_ids = set(special_mDAG.unique_unlabelled_id for special_mDAG in known_interesting_mDAGs)
#
# # For the trick of fixing to point distribution, we can simply compare mDAGs. The QmDAG structure is going to be useful only in the marginalization case (where classical and quantum latents appear)
def reduces_to_knownQCGap_by_intervention(mDAG):
    for node in mDAG.visible_nodes:
        if mDAG.fix_to_point_distribution(node).unique_unlabelled_id in known_interesting_ids:
            return True
    return False
reducible_6_node_mDAGs_via_PD = set(filter(reduces_to_knownQCGap_by_intervention, HLP_6_node_representatives))
irreducible_6_node_mDAGs = HLP_6_node_representatives.difference(reducible_6_node_mDAGs_via_PD)
print("Number of irreducible 6 node mDAGs:", len(irreducible_6_node_mDAGs)+3)
with open('HLP_reproduction.txt', 'w+') as output_file:
    output_file.writelines(mdag.as_string+"\n" for mdag in [G_Instrumental1, G_Evans, G_Triangle])
    output_file.writelines(mdag.as_string+"\n" for mdag in irreducible_6_node_mDAGs)



print("Elie style: ")
QG_Instrumental1 = QmDAG(DirectedStructure([(0, 1), (1, 2)], 3), Hypergraph([], 3), Hypergraph([(1, 2)], 3))
QG_Instrumental2 = QmDAG(DirectedStructure([(0, 1), (1, 2)], 3), Hypergraph([], 3), Hypergraph([(0, 1), (1, 2)], 3))
QG_Instrumental3 = QmDAG(DirectedStructure([(1, 2)], 3), Hypergraph([], 3), Hypergraph([(0, 1), (1, 2)], 3))
QG_Instrumental2b = QmDAG(DirectedStructure([(0, 1), (1, 2)], 3), Hypergraph([(0, 1)], 3), Hypergraph([(1, 2)], 3))
QG_Instrumental3b = QmDAG(DirectedStructure([(1, 2)], 3), Hypergraph([(0, 1)], 3), Hypergraph([(1, 2)], 3))

QG_Evans = QmDAG(DirectedStructure([(0, 1), (0, 2)], 3), Hypergraph([], 3), Hypergraph([(0, 1), (0, 2)], 3))

QG_Bell1 = QmDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([], 4), Hypergraph([(2, 3)], 4))
QG_Bell2 = QmDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([(0,2)], 4), Hypergraph([(2, 3)], 4))
QG_Bell2b = QmDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([], 4), Hypergraph([(0,1),(2, 3)], 4))
QG_Bell3 = QmDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([(1,3)], 4), Hypergraph([(2, 3)], 4))
QG_Bell3b = QmDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([], 4), Hypergraph([(2, 3),(2,3)], 4))
QG_Bell4 = QmDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([(0, 2), (1, 3)], 4), Hypergraph([(2, 3)], 4))
QG_Bell4b = QmDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([(0,2)], 4), Hypergraph([(1, 3),(2,3)], 4))
QG_Bell4c = QmDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([(1,3)], 4), Hypergraph([(0,2),(2, 3)], 4))
QG_Bell4d = QmDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([], 4), Hypergraph([(0, 2), (1, 3),(2,3)], 4))
QG_Bell5 = QmDAG(DirectedStructure([(1, 3)], 4), Hypergraph([(0,2)], 4), Hypergraph([(2, 3)], 4))
QG_Bell5b = QmDAG(DirectedStructure([(1, 3)], 4), Hypergraph([], 4), Hypergraph([(2, 3),(0,2)], 4))
QG_Bell6 = QmDAG(DirectedStructure([], 4), Hypergraph([(1,3),(0,2)], 4), Hypergraph([(2, 3)], 4))
QG_Bell6b = QmDAG(DirectedStructure([], 4), Hypergraph([(1,3)], 4), Hypergraph([(0,2),(2, 3)], 4))
QG_Bell6c = QmDAG(DirectedStructure([], 4), Hypergraph([(0,2)], 4), Hypergraph([(1, 3),(2,3)], 4))
QG_Bell6d = QmDAG(DirectedStructure([], 4), Hypergraph([], 4), Hypergraph([(1, 3),(0,2),(2,3)], 4))
QG_Bell7 = QmDAG(DirectedStructure([(1, 3)], 4), Hypergraph([(0, 2), (1, 3)], 4), Hypergraph([(2, 3)], 4))
QG_Bell7b = QmDAG(DirectedStructure([(1, 3)], 4), Hypergraph([(1,3)], 4), Hypergraph([(0,2),(2, 3)], 4))
QG_Bell7c = QmDAG(DirectedStructure([(1, 3)], 4), Hypergraph([(0,2)], 4), Hypergraph([(1,3),(2,3)], 4))
QG_Bell7d = QmDAG(DirectedStructure([(1, 3)], 4), Hypergraph([], 4), Hypergraph([(0, 2), (1, 3),(2,3)], 4))
QG_Bell8 = QmDAG(DirectedStructure([(0, 2)], 4), Hypergraph([(0, 2), (1, 3)], 4), Hypergraph([(2, 3)], 4))
QG_Bell8b = QmDAG(DirectedStructure([(0, 2)], 4), Hypergraph([(1,3)], 4), Hypergraph([(0,2),(2, 3)], 4))
QG_Bell8c = QmDAG(DirectedStructure([(0, 2)], 4), Hypergraph([(0,2)], 4), Hypergraph([(1, 3),(2,3)], 4))
QG_Bell8d = QmDAG(DirectedStructure([(0, 2)], 4), Hypergraph([], 4), Hypergraph([(0, 2), (1, 3),(2,3)], 4))
QG_Bell9 = QmDAG(DirectedStructure([(0, 2)], 4), Hypergraph([(1,3)], 4), Hypergraph([(2, 3)], 4))
QG_Bell9b = QmDAG(DirectedStructure([(0, 2)], 4), Hypergraph([], 4), Hypergraph([(1, 3),(2,3)], 4))

# QG_Bell_wComm = QmDAG(DirectedStructure([(0, 1), (2, 3), (1, 2)], 4), Hypergraph([], 4), Hypergraph([(1, 2)], 4))
###
QG_Triangle1 = QmDAG(DirectedStructure([], 3), Hypergraph([], 3), Hypergraph([(1, 2), (2, 0), (0, 1)], 3))
QG_Triangle2 = QmDAG(DirectedStructure([], 3), Hypergraph([(1, 2)], 3), Hypergraph([(2, 0), (0, 1)], 3))
QG_Triangle3 = QmDAG(DirectedStructure([], 3), Hypergraph([(1, 2), (2, 0)], 3), Hypergraph([(0, 1)], 3))
known_QC_Gaps_QmDAGs = {QG_Evans,QG_Instrumental1, QG_Instrumental2, QG_Instrumental3,
                        QG_Instrumental2b, QG_Instrumental3b,
                        QG_Triangle1, QG_Triangle2, QG_Triangle3,
                        QG_Bell1,QG_Bell2,QG_Bell3,QG_Bell4,QG_Bell5,QG_Bell6,QG_Bell7,QG_Bell8,QG_Bell9,
                        QG_Bell2b,QG_Bell3b,QG_Bell4b,QG_Bell4c,QG_Bell4d,QG_Bell5b,QG_Bell6b,QG_Bell6c,
                        QG_Bell6d,QG_Bell7b,QG_Bell7c,QG_Bell7d,QG_Bell8b,QG_Bell8c,QG_Bell8d,QG_Bell9b}
known_QC_Gaps_QmDAGs_id = set(special_QmDAG.unique_unlabelled_id for special_QmDAG in known_QC_Gaps_QmDAGs)




print("Total number of qmDAGs to analyze: ", len(QmDAGs4_representatives))

def reduces_to_knownQCGap_by_PD_trick(qmDAG):
    return not known_QC_Gaps_QmDAGs_id.isdisjoint(qmDAG.unique_unlabelled_ids_obtainable_by_PD_trick)

QC_gap_by_PD_trick = list(filter(reduces_to_knownQCGap_by_PD_trick, set(QmDAGs4_representatives).difference(known_QC_Gaps_QmDAGs)))

print("# of QC gaps seen via PD trick: ", len(QC_gap_by_PD_trick))

def reduces_to_knownQCGap_by_naive_marginalization(qmDAG):
    return not known_QC_Gaps_QmDAGs_id.isdisjoint(qmDAG.unique_unlabelled_ids_obtainable_by_naive_marginalization)
QC_gap_by_naive_marginalization = list(filter(reduces_to_knownQCGap_by_naive_marginalization, set(QmDAGs4_representatives).difference(known_QC_Gaps_QmDAGs,QC_gap_by_PD_trick)))

print("# of ADDITIONAL QC gaps seen via naive marginalization: ", len(QC_gap_by_naive_marginalization))
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
    known_QC_Gaps_QmDAGs,
    QC_gap_by_PD_trick,
    QC_gap_by_naive_marginalization
)))

print("# of ADDITIONAL QC gaps seen via teleporation marginalization: ", len(QC_gap_by_marginalization))
print(QC_gap_by_marginalization)

def reduces_to_knownQCGap_by_conditioning(qmDAG):
    return not known_QC_Gaps_QmDAGs_id.isdisjoint(qmDAG.unique_unlabelled_ids_obtainable_by_conditioning)

QC_gap_by_conditioning = list(filter(reduces_to_knownQCGap_by_conditioning, set(QmDAGs4_representatives).difference(
    known_QC_Gaps_QmDAGs,
    QC_gap_by_PD_trick,
    QC_gap_by_naive_marginalization,
    QC_gap_by_marginalization
)))


updated_known_QC_Gaps_QmDAGs=set(list(known_QC_Gaps_QmDAGs)+QC_gap_by_PD_trick+QC_gap_by_naive_marginalization+QC_gap_by_marginalization+QC_gap_by_conditioning)
updated_known_QC_Gaps_QmDAGs_id=set(known_QmDAG.unique_unlabelled_id for known_QmDAG in updated_known_QC_Gaps_QmDAGs)
def reduces_to_knownQCGap_by_Fritz_without_node_splitting(qmDAG):
    obtained_ids = [debug_info[-1] for debug_info in qmDAG.unique_unlabelled_ids_obtainable_by_Fritz_without_node_splitting]
    # return not updated_known_QC_Gaps_QmDAGs_id.isdisjoint(obtained_ids)
    return not known_QC_Gaps_QmDAGs_id.isdisjoint(obtained_ids)

def reduces_to_knownQCGap_by_Fritz_without_node_splitting(qmDAG):
    # obtained_ids = [debug_info[-1] for debug_info in qmDAG.unique_unlabelled_ids_obtainable_by_Fritz_without_node_splitting]
    # return not updated_known_QC_Gaps_QmDAGs_id.isdisjoint(obtained_ids)
    return not known_QC_Gaps_QmDAGs_id.isdisjoint(qmDAG.unique_unlabelled_ids_obtainable_by_Fritz_with_node_splitting(node_decomposition=False))
def reduces_to_knownQCGap_by_Fritz_with_node_splitting(qmDAG):
    # obtained_ids = [debug_info[-1] for debug_info in qmDAG.unique_unlabelled_ids_obtainable_by_Fritz_without_node_splitting]
    # return not updated_known_QC_Gaps_QmDAGs_id.isdisjoint(obtained_ids)
    return not known_QC_Gaps_QmDAGs_id.isdisjoint(qmDAG.unique_unlabelled_ids_obtainable_by_Fritz_with_node_splitting(node_decomposition=True))


print("# of QC Gaps discovered so far: ", len(QC_gap_by_PD_trick+QC_gap_by_naive_marginalization+QC_gap_by_marginalization+QC_gap_by_conditioning))
remaining_representatives = set(QmDAGs4_representatives).difference(updated_known_QC_Gaps_QmDAGs)
print("# of QC Gaps still to be assessed: ", len(remaining_representatives))


QG_Square = QmDAG(DirectedStructure([], 4), Hypergraph([], 4), Hypergraph([(2,3),(1,3),(0,1),(0,2)], 4))
print("Are we going to discover the square? ", reduces_to_knownQCGap_by_Fritz_without_node_splitting(QG_Square))
# # reduces_to_knownQCGap_by_marginalization(QG_Square)
# print("Is the square in the remaining set? ", QG_Square in remaining_representatives)

QC_gap_by_Fritz_without_node_splitting = list(filter(reduces_to_knownQCGap_by_Fritz_without_node_splitting,remaining_representatives))
remaining_representatives.difference_update(QC_gap_by_Fritz_without_node_splitting)

print("# of QC Gaps discovered via Fritz without splitting: ", len(QC_gap_by_Fritz_without_node_splitting))

QC_gap_by_Fritz_with_node_splitting = list(filter(reduces_to_knownQCGap_by_Fritz_with_node_splitting,remaining_representatives))
remaining_representatives.difference_update(QC_gap_by_Fritz_with_node_splitting)

print("# of QC Gaps discovered via Fritz with splitting: ", len(QC_gap_by_Fritz_with_node_splitting))
print("# of QC Gaps still to be assessed: ", len(remaining_representatives))
print("Note that here, we ARE considering Evans as if it had a QC Gap")
# n=1
# for new_QmDAG in QC_gap_by_Fritz_without_node_splitting:
#     for i in range(0,len(new_QmDAG.unique_unlabelled_ids_obtainable_by_Fritz_without_node_splitting)):
#         (target, Y,set_of_visible_parents_to_delete,set_of_Q_facets_to_delete, new_qmDAG)=list(new_QmDAG.unique_unlabelled_ids_obtainable_by_Fritz_without_node_splitting)[i]
#         if new_qmDAG in updated_known_QC_Gaps_QmDAGs_id:
#             print(n,"target=",target)
#             print(n,"Y=",Y)
#     new_QmDAG.as_mDAG.networkx_plot_mDAG()
#     n=n+1


no_infeasible_supports=[]
for mDAG in mDAGs4_representatives:
    if mDAG.support_testing_instance((2,2,2,2),3).no_infeasible_supports():
        no_infeasible_supports.append(mDAG)

remaining_reps_with_infeasible_supports=remaining_representatives.copy()
for G in no_infeasible_supports:
    QG=upgrade_to_QmDAG(G)
    for QmDAG in remaining_representatives:
        if QG.unique_unlabelled_id==QmDAG.unique_unlabelled_id:
            remaining_reps_with_infeasible_supports.remove(QmDAG)
            break
        
for QmDAG in remaining_reps_with_infeasible_supports:
    QmDAG.as_mDAG.networkx_plot_mDAG()

for QmDAG in remaining_reps_with_infeasible_supports:
    for eqclass in Observable_mDAGs4.foundational_eqclasses:
        if QmDAG.as_mDAG in eqclass:
            print(len(eqclass))


