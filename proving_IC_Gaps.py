from __future__ import absolute_import
from hypergraphs import Hypergraph
from directed_structures import DirectedStructure
from mDAG_advanced import mDAG
from quantum_mDAG import QmDAG, as_classical_QmDAG
from metagraph_advanced import Observable_mDAGs_Analysis

# if __name__ == '__main__':
Observable_mDAGs4 = Observable_mDAGs_Analysis(nof_observed_variables=4, max_nof_events_for_supports=0)
mDAGs4_representatives = Observable_mDAGs4.representative_mDAGs_list
QmDAGs4_representatives = list(map(as_classical_QmDAG, mDAGs4_representatives))
Observable_mDAGs3 = Observable_mDAGs_Analysis(nof_observed_variables=3, max_nof_events_for_supports=0)

n = 0
HLP_6_node_representatives = set()
for eqclass in Observable_mDAGs4.foundational_eqclasses:
    for mdag in eqclass:
        if mdag.simplicial_complex_instance.number_of_nonsingleton_latent == 2:
            HLP_6_node_representatives.add(mdag)
            n += 1
            break
print("Number of 6-node mDAGs:", n)

# print("Marina style: ")
G_Instrumental1 = mDAG(DirectedStructure([(0, 1), (1, 2)], 3), Hypergraph([(1, 2)], 3))
G_Instrumental2 = mDAG(DirectedStructure([(0, 1), (1, 2)], 3), Hypergraph([(0, 1), (1, 2)], 3))
G_Instrumental3 = mDAG(DirectedStructure([(1, 2)], 3), Hypergraph([(0, 1), (1, 2)], 3))

G_Triangle = mDAG(DirectedStructure([], 3), Hypergraph([(1, 2), (2, 0), (0, 1)], 3))
G_Evans = mDAG(DirectedStructure([(0, 1), (0, 2)], 3), Hypergraph([(0, 1), (0, 2)], 3))

G_Bell1 = mDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([(2, 3)], 4))
G_Bell2 = mDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([(2, 3), (0, 2)], 4))
G_Bell2b = mDAG(DirectedStructure([(1, 3)], 4), Hypergraph([(2, 3), (0, 2)], 4))
G_Bell3 = mDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([(2, 3), (1, 3)], 4))
G_Bell3b = mDAG(DirectedStructure([(0, 2)], 4), Hypergraph([(2, 3), (1, 3)], 4))
G_Bell4 = mDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([(2, 3), (0, 2), (1, 3)], 4))
G_Bell4a = mDAG(DirectedStructure([(0, 2)], 4), Hypergraph([(2, 3), (0, 2), (1, 3)], 4))
G_Bell4b = mDAG(DirectedStructure([(1, 3)], 4), Hypergraph([(2, 3), (0, 2), (1, 3)], 4))
G_Bell4c = mDAG(DirectedStructure([], 4), Hypergraph([(2, 3), (0, 2), (1, 3)], 4))

known_interesting_mDAGs = list(map(as_classical_QmDAG, [G_Instrumental1, G_Instrumental2, G_Instrumental3, G_Evans, G_Triangle,
                                                        G_Bell1, G_Bell2, G_Bell2b, G_Bell3, G_Bell3b,
                                                        G_Bell4, G_Bell4a, G_Bell4b, G_Bell4c]))
known_interesting_ids = set(special_mDAG.unique_unlabelled_id for special_mDAG in known_interesting_mDAGs)


# #
# # # For the trick of fixing to point distribution, we can simply compare mDAGs. The QmDAG structure is going to be useful only in the marginalization case (where classical and quantum latents appear)
# def reduces_to_knownICGap_by_intervention(mDAG):
#     for node in mDAG.visible_nodes:
#         if mDAG.fix_to_point_distribution(node).unique_unlabelled_id in known_interesting_ids:
#             return True
#     return False
# 
# 
# reducible_6_node_mDAGs_via_PD = set(filter(reduces_to_knownICGap_by_intervention, HLP_6_node_representatives))
# irreducible_6_node_mDAGs = HLP_6_node_representatives.difference(reducible_6_node_mDAGs_via_PD)
# print("Number of irreducible 6 node mDAGs:", len(irreducible_6_node_mDAGs) + 3)
# with open('HLP_reproduction.txt', 'w+') as output_file:
#     output_file.writelines(mdag.as_string + "\n" for mdag in [G_Instrumental1, G_Evans, G_Triangle])
#     output_file.writelines(mdag.as_string + "\n" for mdag in irreducible_6_node_mDAGs)




remaining_representatives = set(QmDAGs4_representatives).copy()

print("Total number of qmDAGs to analyze: ", len(remaining_representatives))


def reduces_to_knownICGap_by_PD_trick(qmDAG):
    return not known_interesting_ids.isdisjoint(qmDAG.unique_unlabelled_ids_obtainable_by_PD_trick)


IC_gap_by_PD_trick = list(filter(reduces_to_knownICGap_by_PD_trick, remaining_representatives))
print("# of ADDITIONAL IC gaps seen via PD trick: ", len(IC_gap_by_PD_trick))
remaining_representatives.difference_update(IC_gap_by_PD_trick)


def reduces_to_knownICGap_by_naive_marginalization(qmDAG):
    return not known_interesting_ids.isdisjoint(
        qmDAG.unique_unlabelled_ids_obtainable_by_naive_marginalization)


IC_gap_by_naive_marginalization = list(
    filter(reduces_to_knownICGap_by_naive_marginalization, remaining_representatives))

print("# of ADDITIONAL IC gaps seen via naive marginalization: ", len(IC_gap_by_naive_marginalization))
# print(IC_gap_by_naive_marginalization)
remaining_representatives.difference_update(IC_gap_by_naive_marginalization)




def reduces_to_knownICGap_by_conditioning(qmDAG):
    return not known_interesting_ids.isdisjoint(qmDAG.unique_unlabelled_ids_obtainable_by_conditioning)


IC_gap_by_conditioning = list(filter(reduces_to_knownICGap_by_conditioning, remaining_representatives))
print("# of ADDITIONAL IC gaps seen via conditioning: ", len(IC_gap_by_conditioning))
# print(IC_gap_by_conditioning)
remaining_representatives.difference_update(IC_gap_by_conditioning)

# updated_known_IC_Gaps_QmDAGs = set(QmDAGs4_representatives).difference(remaining_representatives)
# updated_known_IC_Gaps_QmDAGs_ids = known_IC_Gaps_QmDAGs_ids.union(set(known_QmDAG.unique_unlabelled_id for known_QmDAG in updated_known_IC_Gaps_QmDAGs))
updated_known_IC_Gaps_QmDAGs_ids = known_interesting_ids.copy()
# print("Size of known database: ", len(updated_known_IC_Gaps_QmDAGs_ids))
# print("Knows about Bell etc: ", updated_known_IC_Gaps_QmDAGs_ids.issuperset(known_interesting_ids))


def reduces_to_knownICGap_by_Fritz_without_node_splitting(qmDAG):
    obtained_ids = [debug_info[-1] for debug_info in
                    qmDAG.unique_unlabelled_ids_obtainable_by_Fritz_without_node_splitting]
    # return not updated_known_IC_Gaps_QmDAGs_id.isdisjoint(obtained_ids)
    return not updated_known_IC_Gaps_QmDAGs_ids.isdisjoint(obtained_ids)


def reduces_to_knownICGap_by_Fritz_without_node_splitting(qmDAG):
    # obtained_ids = [debug_info[-1] for debug_info in qmDAG.unique_unlabelled_ids_obtainable_by_Fritz_without_node_splitting]
    # return not updated_known_IC_Gaps_QmDAGs_id.isdisjoint(obtained_ids)
    return not updated_known_IC_Gaps_QmDAGs_ids.isdisjoint(
        qmDAG.unique_unlabelled_ids_obtainable_by_Fritz_with_node_splitting(node_decomposition=False))


def reduces_to_knownICGap_by_Fritz_with_node_splitting(qmDAG):
    # obtained_ids = [debug_info[-1] for debug_info in qmDAG.unique_unlabelled_ids_obtainable_by_Fritz_without_node_splitting]
    # return not updated_known_IC_Gaps_QmDAGs_id.isdisjoint(obtained_ids)
    return not updated_known_IC_Gaps_QmDAGs_ids.isdisjoint(
        qmDAG.unique_unlabelled_ids_obtainable_by_Fritz_with_node_splitting(node_decomposition=True))


print("# of IC Gaps discovered so far: ",
      len(IC_gap_by_PD_trick + IC_gap_by_naive_marginalization + IC_gap_by_conditioning))
# remaining_representatives = set(QmDAGs4_representatives).difference(updated_known_IC_Gaps_QmDAGs)
print("# of IC Gaps still to be assessed: ", len(remaining_representatives))

# QG_Square = QmDAG(DirectedStructure([], 4), Hypergraph([(2, 3), (1, 3), (0, 1), (0, 2)], 4), Hypergraph([], 4))
# print("Are we going to discover the square? ", reduces_to_knownICGap_by_Fritz_without_node_splitting(QG_Square))
# # # reduces_to_knownICGap_by_marginalization(QG_Square)
# print("Is the square in the remaining set? ", QG_Square in remaining_representatives)

IC_gap_by_Fritz_without_node_splitting = list(
    filter(reduces_to_knownICGap_by_Fritz_without_node_splitting, remaining_representatives))
remaining_representatives.difference_update(IC_gap_by_Fritz_without_node_splitting)

print("# of IC Gaps discovered via Fritz without splitting: ", len(IC_gap_by_Fritz_without_node_splitting))

IC_gap_by_Fritz_with_node_splitting = list(
    filter(reduces_to_knownICGap_by_Fritz_with_node_splitting, remaining_representatives))
remaining_representatives.difference_update(IC_gap_by_Fritz_with_node_splitting)

print("# of IC Gaps discovered via Fritz with splitting: ", len(IC_gap_by_Fritz_with_node_splitting))
print("# of IC Gaps still to be assessed: ", len(remaining_representatives))
print(remaining_representatives)

