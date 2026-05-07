from __future__ import absolute_import
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypergraphs import Hypergraph
from directed_structures import DirectedStructure
from mDAG_advanced import mDAG
from quantum_mDAG import QmDAG, upgrade_to_QmDAG
from metagraph_temporally_ordered import Metagraph_temporally_ordered_mDAGs
from itertools import chain

# if __name__ == '__main__':
Metagraph4 = Metagraph_temporally_ordered_mDAGs(n=4, temporally_ordered=True)
print("Number of temporally-ordered equivalence classes:", len(Metagraph4.equivalence_classes_as_ids))
print("Number of temporally-ordered provably-algebraic equivalence classes:", len(Metagraph4.latent_free_equivalence_classes_as_ids))
temporally_ordered_not_latent_free_equivalent_classes_as_mDAGs = Metagraph4.NOT_latent_free_equivalence_classes_as_mDAGs
print("Number of temporally-ordered not-provably-algebraic equivalence classes:", len(temporally_ordered_not_latent_free_equivalent_classes_as_mDAGs))
mDAGs4_representatives = list(chain.from_iterable(temporally_ordered_not_latent_free_equivalent_classes_as_mDAGs))
QmDAGs4_representatives = list(map(upgrade_to_QmDAG, mDAGs4_representatives))

print("Number of temporally-ordered not-provably-algebraic mDAGs:", len(QmDAGs4_representatives))




# print("CATEGORIZATION 1: COUNTING BY WHICH 3 NODE DAG IS DISCOVERED: ")
QG_Instrumental1 = QmDAG(DirectedStructure([(0, 1), (1, 2)], 3), Hypergraph([], 3), Hypergraph([(1, 2)], 3))
QG_Instrumental2 = QmDAG(DirectedStructure([(0, 1), (1, 2)], 3), Hypergraph([], 3), Hypergraph([(0, 1), (1, 2)], 3))
QG_Instrumental3 = QmDAG(DirectedStructure([(1, 2)], 3), Hypergraph([], 3), Hypergraph([(0, 1), (1, 2)], 3))
QG_Instrumental2b = QmDAG(DirectedStructure([(0, 1), (1, 2)], 3), Hypergraph([(0, 1)], 3), Hypergraph([(1, 2)], 3))
QG_Instrumental3b = QmDAG(DirectedStructure([(1, 2)], 3), Hypergraph([(0, 1)], 3), Hypergraph([(1, 2)], 3))

IV_ids = set(special_QmDAG.unique_unlabelled_id for special_QmDAG in {QG_Instrumental1, QG_Instrumental2, QG_Instrumental3,
                        QG_Instrumental2b, QG_Instrumental3b})

# reduces_to_IV = set(new_QmDAG for new_QmDAG in set(QmDAGs4_representatives) if not new_QmDAG.unique_unlabelled_ids_obtainable_by_reduction(districts_check=False).isdisjoint(IV_ids))
# print("# that reduce to IV: ", len(reduces_to_IV))
# reduces_to_IV_by_PD_trick = set(new_QmDAG for new_QmDAG in set(QmDAGs4_representatives) if not new_QmDAG.unique_unlabelled_ids_obtainable_by_PD_trick.isdisjoint(IV_ids))
# print("# that reduce to IV by PD trick: ", len(reduces_to_IV_by_PD_trick))
# reduces_to_IV_by_naive_marginalization = set(new_QmDAG for new_QmDAG in set(QmDAGs4_representatives) if not new_QmDAG.unique_unlabelled_ids_obtainable_by_naive_marginalization(districts_check=False, apply_teleportation=False).isdisjoint(IV_ids))
# print("# that reduce to IV by naive marginalization: ", len(reduces_to_IV_by_naive_marginalization))
# reduces_to_IV_by_marginalization = set(new_QmDAG for new_QmDAG in set(QmDAGs4_representatives) if not new_QmDAG.unique_unlabelled_ids_obtainable_by_marginalization(districts_check=False, apply_teleportation=True).isdisjoint(IV_ids))
# print("# that reduce to IV by marginalization: ", len(reduces_to_IV_by_marginalization))
# reduces_to_IV_by_conditioning = set(new_QmDAG for new_QmDAG in set(QmDAGs4_representatives) if not new_QmDAG.unique_unlabelled_ids_obtainable_by_conditioning.isdisjoint(IV_ids))
# print("# that reduce to IV by conditioning: ", len(reduces_to_IV_by_conditioning))
# reduces_to_IV_by_Fritz_without_node_splitting = set(new_QmDAG for new_QmDAG in set(QmDAGs4_representatives) if not new_QmDAG.unique_unlabelled_ids_obtainable_by_Fritz_for_QC(node_decomposition=False).isdisjoint(IV_ids))
# print("# that reduce to IV by Fritz without node splitting: ", len(reduces_to_IV_by_Fritz_without_node_splitting))
# reduces_to_IV_by_Fritz_with_node_splitting = set(new_QmDAG for new_QmDAG in set(QmDAGs4_representatives) if not new_QmDAG.unique_unlabelled_ids_obtainable_by_Fritz_for_QC(node_decomposition=True).isdisjoint(IV_ids))
# print("# that reduce to IV by Fritz with node splitting: ", len(reduces_to_IV_by_Fritz_with_node_splitting))



QG_Triangle1 = QmDAG(DirectedStructure([], 3), Hypergraph([], 3), Hypergraph([(1, 2), (2, 0), (0, 1)], 3))
QG_Triangle2 = QmDAG(DirectedStructure([], 3), Hypergraph([(1, 2)], 3), Hypergraph([(2, 0), (0, 1)], 3))
QG_Triangle3 = QmDAG(DirectedStructure([], 3), Hypergraph([(1, 2), (2, 0)], 3), Hypergraph([(0, 1)], 3))

Triangle_ids = set(special_QmDAG.unique_unlabelled_id for special_QmDAG in {QG_Triangle1, QG_Triangle2, QG_Triangle3})

# reduces_to_Tri = set(new_QmDAG for new_QmDAG in set(QmDAGs4_representatives) if not new_QmDAG.unique_unlabelled_ids_obtainable_by_reduction(districts_check=False).isdisjoint(Triangle_ids))
# print("# that reduce to Tri: ", len(reduces_to_Tri))

QG_Evans = QmDAG(DirectedStructure([(0, 1), (0, 2)], 3), Hypergraph([], 3), Hypergraph([(0, 1), (0, 2)], 3))
QG_Evansb = QmDAG(DirectedStructure([(0, 1), (0, 2)], 3), Hypergraph([(0, 1)], 3), Hypergraph([(0, 2)], 3))
#Evans_ids = set(special_QmDAG.unique_unlabelled_id for special_QmDAG in {QG_Evans, QG_Evansb})
Evans_ids = set(special_QmDAG.unique_unlabelled_id for special_QmDAG in {QG_Evans})

# reduces_to_Evans = set(new_QmDAG for new_QmDAG in set(QmDAGs4_representatives) if not new_QmDAG.unique_unlabelled_ids_obtainable_by_reduction(districts_check=False).isdisjoint(Evans_ids))
# print("# that reduce to Evans: ", len(reduces_to_Evans))


known_QC_Gaps_QmDAGs_small_ids = set().union(IV_ids, Triangle_ids, Evans_ids) 

QG_Bell1 = QmDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([], 4), Hypergraph([(2, 3)], 4))

QG_Bell2 = QmDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([(0,2)], 4), Hypergraph([(2, 3)], 4))
QG_Bell2b = QmDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([], 4), Hypergraph([(0,2),(2, 3)], 4))

QG_Bell3 = QmDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([(1,3)], 4), Hypergraph([(2, 3)], 4))
QG_Bell3b = QmDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([], 4), Hypergraph([(2,3), (1,3)], 4))

QG_Bell4 = QmDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([(0, 2), (1, 3)], 4), Hypergraph([(2, 3)], 4))
QG_Bell4b = QmDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([(0,2)], 4), Hypergraph([(1,3),(2,3)], 4))
QG_Bell4c = QmDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([(1,3)], 4), Hypergraph([(0,2),(2,3)], 4))
QG_Bell4d = QmDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([], 4), Hypergraph([(0,2), (1,3),(2,3)], 4))

QG_Bell6 = QmDAG(DirectedStructure([], 4), Hypergraph([(1,3),(0,2)], 4), Hypergraph([(2, 3)], 4))
QG_Bell6b = QmDAG(DirectedStructure([], 4), Hypergraph([(1,3)], 4), Hypergraph([(0,2),(2, 3)], 4))
QG_Bell6c = QmDAG(DirectedStructure([], 4), Hypergraph([(0,2)], 4), Hypergraph([(1,3),(2,3)], 4))
QG_Bell6d = QmDAG(DirectedStructure([], 4), Hypergraph([], 4), Hypergraph([(1,3),(0,2),(2,3)], 4))
QG_Bell6d = QmDAG(DirectedStructure([], 4), Hypergraph([], 4), Hypergraph([(0,2),(1,3),(2,3)], 4))

QG_Bell5 = QmDAG(DirectedStructure([(1, 3)], 4), Hypergraph([(0,2)], 4), Hypergraph([(2,3)], 4))
QG_Bell5b = QmDAG(DirectedStructure([(1, 3)], 4), Hypergraph([], 4), Hypergraph([(2, 3),(0,2)], 4))

QG_Bell7 = QmDAG(DirectedStructure([(1, 3)], 4), Hypergraph([(0, 2), (1, 3)], 4), Hypergraph([(2,3)], 4))
QG_Bell7b = QmDAG(DirectedStructure([(1, 3)], 4), Hypergraph([(1,3)], 4), Hypergraph([(0,2),(2,3)], 4))
QG_Bell7c = QmDAG(DirectedStructure([(1, 3)], 4), Hypergraph([(0,2)], 4), Hypergraph([(1,3),(2,3)], 4))
QG_Bell7d = QmDAG(DirectedStructure([(1, 3)], 4), Hypergraph([], 4), Hypergraph([(0,2), (1,3),(2,3)], 4))

QG_Bell8 = QmDAG(DirectedStructure([(0, 2)], 4), Hypergraph([(0, 2), (1, 3)], 4), Hypergraph([(2, 3)], 4))
QG_Bell8b = QmDAG(DirectedStructure([(0, 2)], 4), Hypergraph([(1,3)], 4), Hypergraph([(0,2),(2,3)], 4))
QG_Bell8c = QmDAG(DirectedStructure([(0, 2)], 4), Hypergraph([(0,2)], 4), Hypergraph([(1,3),(2,3)], 4))
QG_Bell8d = QmDAG(DirectedStructure([(0, 2)], 4), Hypergraph([], 4), Hypergraph([(0,2), (1,3),(2,3)], 4))

QG_Bell9 = QmDAG(DirectedStructure([(0, 2)], 4), Hypergraph([(1,3)], 4), Hypergraph([(2,3)], 4))
QG_Bell9b = QmDAG(DirectedStructure([(0, 2)], 4), Hypergraph([], 4), Hypergraph([(1,3),(2,3)], 4))


known_QC_Gaps_QmDAGs_big = {QG_Bell1,QG_Bell2,QG_Bell3,QG_Bell4,QG_Bell5,QG_Bell6,QG_Bell7,QG_Bell8,QG_Bell9,
                        QG_Bell2b,QG_Bell3b,QG_Bell4b,QG_Bell4c,QG_Bell4d,QG_Bell5b,QG_Bell6b,QG_Bell6c,
                        QG_Bell6d,QG_Bell7b,QG_Bell7c,QG_Bell7d,QG_Bell8b,QG_Bell8c,QG_Bell8d,QG_Bell9b}
known_QC_Gaps_QmDAGs_big_ids = set(special_QmDAG.unique_unlabelled_id for special_QmDAG in known_QC_Gaps_QmDAGs_big)

known_QC_Gaps_QmDAGs_ids = known_QC_Gaps_QmDAGs_small_ids.union(known_QC_Gaps_QmDAGs_big_ids)




# QC_remaining_representatives = set(QmDAGs4_representatives).difference(known_QC_Gaps_QmDAGs_big)

QC_remaining_representatives = set([qmDAG for qmDAG in QmDAGs4_representatives if qmDAG.unique_unlabelled_id not in known_QC_Gaps_QmDAGs_big_ids])
things_we_got_rid_of = set(QmDAGs4_representatives).difference(QC_remaining_representatives)
print("Total number of qmDAGs to analyze: ", len(QC_remaining_representatives))
print("Number of representatives that are known QC Gaps: ", len(things_we_got_rid_of))

# Bell_scenarios_that_are_not_in_the_to_analyze_set = set([
#     QG_Bell1, QG_Bell2b, QG_Bell3b, QG_Bell4d, QG_Bell5b, QG_Bell6d, QG_Bell7d, QG_Bell8d, QG_Bell9b
# ]).difference(QmDAGs4_representatives)
# # Bell_scenarios_that_are_not_in_the_to_analyze_set = [m for m in known_QC_Gaps_QmDAGs_big.difference(QmDAGs4_representatives) if m.C_simplicial_complex_instance.number_of_nonsingleton_latent == 0]
# print("Bell scenarios missing from our starting list:")
# for Bell_scenario in Bell_scenarios_that_are_not_in_the_to_analyze_set:
#     print(Bell_scenario)
# print("Bell scenarios in our starting list:")
# for Bell_scenario in set(QmDAGs4_representatives).intersection(known_QC_Gaps_QmDAGs_big):
#     print(Bell_scenario)

# print("Is 7d (all latent) in our starting list? ", QG_Bell7d in QmDAGs4_representatives)
# print("Is 7d no nonsigleton classical latent? ", QG_Bell7d.C_simplicial_complex_instance.number_of_nonsingleton_latent == 0)
# print("Is 7d in the remaining representatives? ", QG_Bell7d in QC_remaining_representatives)
# print("Is 7d in the known intersection? ", QG_Bell7d in set(QmDAGs4_representatives).intersection(known_QC_Gaps_QmDAGs_big))
# print(QG_Bell7d)
# print("\n\n")





def reduces_to_knownQCGap_by_PD_trick(qmDAG):
    return not known_QC_Gaps_QmDAGs_small_ids.isdisjoint(qmDAG.unique_unlabelled_ids_obtainable_by_PD_trick)


QC_gap_by_PD_trick = list(filter(reduces_to_knownQCGap_by_PD_trick, QC_remaining_representatives))
print("# of ADDITIONAL QC gaps seen via PD trick: ", len(QC_gap_by_PD_trick))
QC_remaining_representatives.difference_update(QC_gap_by_PD_trick)

def reduces_to_knownQCGap_by_interruption(qmDAG):
    return not known_QC_Gaps_QmDAGs_small_ids.isdisjoint(qmDAG.unique_unlabelled_ids_obtainable_by_interruption)
QC_gap_by_interruption = list(filter(reduces_to_knownQCGap_by_interruption, QC_remaining_representatives))
print("# of ADDITIONAL QC gaps seen via interruption: ", len(QC_gap_by_interruption))
QC_remaining_representatives.difference_update(QC_gap_by_interruption)

def reduces_to_knownQCGap_by_naive_marginalization(qmDAG):
    return not known_QC_Gaps_QmDAGs_small_ids.isdisjoint(qmDAG.unique_unlabelled_ids_obtainable_by_naive_marginalization(districts_check=False))
QC_gap_by_naive_marginalization = list(filter(reduces_to_knownQCGap_by_naive_marginalization, QC_remaining_representatives))

print("# of ADDITIONAL QC gaps seen via naive marginalization: ", len(QC_gap_by_naive_marginalization))
# print(QC_gap_by_naive_marginalization)
QC_remaining_representatives.difference_update(QC_gap_by_naive_marginalization)
# debug_QmDAG = QmDAG(
#         DirectedStructure([(0, 1), (1, 2), (2, 3)], 4),
#         Hypergraph([], 4),
#         Hypergraph([(0, 1), (1, 3), (2, 3)], 4)
#     )
# print("Is this even considered? ", debug_QmDAG in QmDAGs4_representatives)
# print("Is it detected by PD? ", debug_QmDAG in QC_gap_by_PD_trick)
# print("Is it detected by our code? ", debug_QmDAG in QC_gap_by_naive_marginalization)

def reduces_to_knownQCGap_by_marginalization(qmDAG):
    return not known_QC_Gaps_QmDAGs_small_ids.isdisjoint(qmDAG.unique_unlabelled_ids_obtainable_by_marginalization(districts_check=False))
QC_gap_by_marginalization = list(filter(reduces_to_knownQCGap_by_marginalization, QC_remaining_representatives))

print("# of ADDITIONAL QC gaps seen via teleporation marginalization: ", len(QC_gap_by_marginalization))
# print(QC_gap_by_marginalization)
QC_remaining_representatives.difference_update(QC_gap_by_marginalization)

def reduces_to_knownQCGap_by_conditioning(qmDAG):
    return not known_QC_Gaps_QmDAGs_small_ids.isdisjoint(qmDAG.unique_unlabelled_ids_obtainable_by_conditioning)

QC_gap_by_conditioning = list(filter(reduces_to_knownQCGap_by_conditioning, QC_remaining_representatives))
print("# of ADDITIONAL QC gaps seen via conditioning: ", len(QC_gap_by_conditioning))
# print(QC_gap_by_conditioning)
QC_remaining_representatives.difference_update(QC_gap_by_conditioning)


updated_known_QC_Gaps_QmDAGs = set(QmDAGs4_representatives).difference(QC_remaining_representatives)
updated_known_QC_Gaps_QmDAGs_ids = set(known_QmDAG.unique_unlabelled_id for known_QmDAG in updated_known_QC_Gaps_QmDAGs)
print("Size of new database: ", len(updated_known_QC_Gaps_QmDAGs_ids))
updated_known_QC_Gaps_QmDAGs_ids.update(known_QC_Gaps_QmDAGs_ids)
print("Size of known database: ", len(updated_known_QC_Gaps_QmDAGs_ids))
print("Knows about Bell etc: ",  updated_known_QC_Gaps_QmDAGs_ids.issuperset(known_QC_Gaps_QmDAGs_ids))

# =============================================================================
# def reduces_to_knownQCGap_by_Fritz_without_node_splitting(qmDAG):
#     obtained_ids = [debug_info[-1] for debug_info in qmDAG.unique_unlabelled_ids_obtainable_by_Fritz]
#     # return not updated_known_QC_Gaps_QmDAGs_id.isdisjoint(obtained_ids)
#     return not updated_known_QC_Gaps_QmDAGs_ids.isdisjoint(obtained_ids)
# =============================================================================

def reduces_to_knownQCGap_by_Fritz_without_node_splitting(qmDAG):
    # obtained_ids = [debug_info[-1] for debug_info in qmDAG.unique_unlabelled_ids_obtainable_by_Fritz_without_node_splitting]
    # return not updated_known_QC_Gaps_QmDAGs_id.isdisjoint(obtained_ids)
    return not updated_known_QC_Gaps_QmDAGs_ids.isdisjoint(qmDAG.unique_unlabelled_ids_obtainable_by_Fritz_for_QC(node_decomposition=False))
def reduces_to_knownQCGap_by_Fritz_with_node_splitting(qmDAG):
    # obtained_ids = [debug_info[-1] for debug_info in qmDAG.unique_unlabelled_ids_obtainable_by_Fritz_without_node_splitting]
    # return not updated_known_QC_Gaps_QmDAGs_id.isdisjoint(obtained_ids)
    return not updated_known_QC_Gaps_QmDAGs_ids.isdisjoint(qmDAG.unique_unlabelled_ids_obtainable_by_Fritz_for_QC(node_decomposition=True))


print("# of QC Gaps discovered so far: ", len(QC_gap_by_PD_trick+QC_gap_by_interruption+QC_gap_by_naive_marginalization+QC_gap_by_marginalization+QC_gap_by_conditioning))
# QC_remaining_representatives = set(QmDAGs4_representatives).difference(updated_known_QC_Gaps_QmDAGs)
print("# of QC Gaps still to be assessed: ", len(QC_remaining_representatives))


# QG_Square = QmDAG(DirectedStructure([], 4), Hypergraph([], 4), Hypergraph([(2,3),(1,3),(0,1),(0,2)], 4))
# print("Are we going to discover the square? ", reduces_to_knownQCGap_by_Fritz_without_node_splitting(QG_Square))
# # reduces_to_knownQCGap_by_marginalization(QG_Square)
# print("Is the square in the remaining set? ", QG_Square in QC_remaining_representatives)

QC_gap_by_Fritz_without_node_splitting = list(filter(reduces_to_knownQCGap_by_Fritz_without_node_splitting, QC_remaining_representatives))
QC_remaining_representatives.difference_update(QC_gap_by_Fritz_without_node_splitting)

print("# of QC Gaps discovered via Fritz without splitting: ", len(QC_gap_by_Fritz_without_node_splitting))
print(QC_gap_by_Fritz_without_node_splitting)

QC_gap_by_Fritz_with_node_splitting = list(filter(reduces_to_knownQCGap_by_Fritz_with_node_splitting, QC_remaining_representatives))
QC_remaining_representatives.difference_update(QC_gap_by_Fritz_with_node_splitting)

print("# of QC Gaps discovered via Fritz with splitting: ", len(QC_gap_by_Fritz_with_node_splitting))
print(QC_gap_by_Fritz_with_node_splitting)
# =============================================================================
print("# of QC Gaps still to be assessed: ", len(QC_remaining_representatives))
print("Note that here, we ARE considering Evans as if it had a QC Gap (only if both latents go quantum).")

len(QC_remaining_representatives)

# =============================================================================
# # n=1
# # for new_QmDAG in QC_gap_by_Fritz_without_node_splitting:
# #     for i in range(0,len(new_QmDAG.unique_unlabelled_ids_obtainable_by_Fritz_without_node_splitting)):
# #         (target, Y,set_of_visible_parents_to_delete,set_of_Q_facets_to_delete, new_qmDAG)=list(new_QmDAG.unique_unlabelled_ids_obtainable_by_Fritz_without_node_splitting)[i]
# #         if new_qmDAG in updated_known_QC_Gaps_QmDAGs_id:
# #             print(n,"target=",target)
# #             print(n,"Y=",Y)
# #     new_QmDAG.as_mDAG.networkx_plot_mDAG()
# #     n=n+1
# 
# no_infeasible_supports=[]
# for mDAG in mDAGs4_representatives:
#     if mDAG.support_testing_instance((2,2,2,2),3).no_infeasible_supports():
#         no_infeasible_supports.append(mDAG)
#         
# 
# QC_remaining_reps_with_infeasible_supports=QC_remaining_representatives.copy()
# for G in no_infeasible_supports:
#     QG=upgrade_to_QmDAG(G)
#     for QmDAG in QC_remaining_representatives:
#         if QG.unique_unlabelled_id==QmDAG.unique_unlabelled_id:
#             QC_remaining_reps_with_infeasible_supports.remove(QmDAG)
#             break
# len(QC_remaining_reps_with_infeasible_supports)
# 
# 
#         
# # =============================================================================
# # for QmDAG in remaining_reps_with_infeasible_supports:
# #     QmDAG.as_mDAG.networkx_plot_mDAG()
# # 
# # for QmDAG in remaining_reps_with_infeasible_supports:
# #     for eqclass in Observable_mDAGs4.foundational_eqclasses:
# #         if QmDAG.as_mDAG in eqclass:
# #             print(len(eqclass))
# # =============================================================================
# 
# # =============================================================================
# # IC_is_subset_of_QC=[]
# # for element in IC_remaining_representatives:
# #     if upgrade_to_QmDAG(element.as_mDAG) not in QC_remaining_representatives:
# #         IC_is_subset_of_QC.append(element)
# # len(IC_is_subset_of_QC)
# # =============================================================================
# 
# known_interesting_supps={i:list(mDAG.infeasible_binary_supports_n_events_unlabelled(i) for mDAG in [G_Instrumental1, G_Evans, G_Triangle,G_Bell1]) for i in range(2,7)}
# def same_sup_as_known_QC_Gap(mDAG,n):
#     if mDAG.infeasible_binary_supports_n_events_unlabelled(n) in known_interesting_supps[n]:
#         return True
#     return False
# 
# unproven_QC_with_known_QC_support=False
# for QmDAG in QC_remaining_reps_with_infeasible_supports:
#     if same_sup_as_known_QC_Gap(QmDAG.as_mDAG,3):
#         unproven_QC_with_known_QC_support=True
#         print("The following  remaining representative still to be assessed for a QC Gap has the same support as a known QC Gap at 3 events:", QmDAG)
# if not unproven_QC_with_known_QC_support:
#     print("None of the remaining representatives still to be assessed for a QC Gap has the same support as a known QC Gap at 3 events.")
# 
# latent_free_supps={i:list(mDAG.infeasible_binary_supports_n_events_unlabelled(i) for mDAG in Observable_mDAGs4.latent_free_representative_mDAGs_list) for i in range(2,6)}
# def same_sup_as_latent_free(mDAG,n):
#     if mDAG.infeasible_binary_supports_n_events_unlabelled(n) in latent_free_supps[n]:
#         return True
#     return False
# 
# # =============================================================================
# # for mDAG in no_infeasible_supports:
# #     print(same_sup_as_latent_free(mDAG,5))
# # =============================================================================
# 
# unproven_QC_with_latent_free_support=False
# unproven_QC_with_latent_free_support_list_3events=[]
# for QmDAG in QC_remaining_reps_with_infeasible_supports:
#     if same_sup_as_latent_free(QmDAG.as_mDAG,3):
#         unproven_QC_with_latent_free_support=True
#         unproven_QC_with_latent_free_support_list_3events.append(QmDAG)
#         print("The following remaining representative still to be assessed for a QC Gap has the same support as a latent-free at 3 events:", QmDAG)
# if not unproven_QC_with_latent_free_support:
#     print("None of the remaining representatives still to be assessed for a QC Gap has the same support as a latent-free at 3 events.")
# 
# unproven_QC_with_latent_free_support=False
# for QmDAG in unproven_QC_with_latent_free_support_list_3events:
#     if same_sup_as_latent_free(QmDAG.as_mDAG,4):
#         unproven_QC_with_latent_free_support=True
#         print("The following remaining representative still to be assessed for a QC Gap has the same support as a latent-free at 3 and 4 events:", QmDAG)
# if not unproven_QC_with_latent_free_support:
#     print("None of the remaining representatives still to be assessed for a QC Gap has the same support as a latent-free at 3 and 4 events.")
#     
# 
# =============================================================================
