from __future__ import absolute_import
from hypergraphs import Hypergraph, LabelledHypergraph
from directed_structures import DirectedStructure, LabelledDirectedStructure
from mDAG_advanced import mDAG
from quantum_mDAG import QmDAG
from metagraph_advanced import Observable_mDAGs_Analysis


if __name__ == '__main__':
    # Observable_mDAGs2 = Observable_mDAGs_Analysis(nof_observed_variables=2, max_nof_events_for_supports=0)
    #Observable_mDAGs3 = Observable_mDAGs_Analysis(nof_observed_variables=3, max_nof_events_for_supports=0)
    Observable_mDAGs4 = Observable_mDAGs_Analysis(nof_observed_variables=4, max_nof_events_for_supports=0)
    
    G_Instrumental1=mDAG(DirectedStructure([(0,1),(1,2)],3),Hypergraph([(1,2)],3))
    G_Instrumental2=mDAG(DirectedStructure([(0,1),(1,2)],3),Hypergraph([(0,1),(1,2)],3))
    G_Instrumental3=mDAG(DirectedStructure([(1,2)],3),Hypergraph([(0,1),(1,2)],3))
    G_Bell= mDAG(DirectedStructure([(0,1),(2,3)],4),Hypergraph([(1,2)],4))
    G_Triangle=mDAG(DirectedStructure([],3),Hypergraph([(1,2),(2,0),(0,1)],3))
    G_Evans=mDAG(DirectedStructure([(0,1),(0,2)],3),Hypergraph([(0,1),(0,2)],3))
    known_QC_Gaps_mDAGs_id=[G_Instrumental1.unique_unlabelled_id,G_Instrumental2.unique_unlabelled_id,G_Instrumental3.unique_unlabelled_id,G_Bell.unique_unlabelled_id,G_Triangle.unique_unlabelled_id]
    #known_QC_Gaps_mDAGs_id=[G_Evans.unique_unlabelled_id,G_Instrumental.unique_unlabelled_id,G_Bell.unique_unlabelled_id,G_Triangle.unique_unlabelled_id]

    
    QG_Instrumental1=QmDAG(DirectedStructure([(0,1),(1,2)],3),Hypergraph([],3),Hypergraph([(1,2)],3))
    QG_Instrumental2=QmDAG(DirectedStructure([(0,1),(1,2)],3),Hypergraph([],3),Hypergraph([(0,1),(1,2)],3))
    QG_Instrumental3=QmDAG(DirectedStructure([(1,2)],3),Hypergraph([],3),Hypergraph([(0,1),(1,2)],3))
    QG_Bell= QmDAG(DirectedStructure([(0,1),(2,3)],4),Hypergraph([],4),Hypergraph([(1,2)],4))
    known_QC_Gaps_QmDAGs_id=[QG_Instrumental.unique_unlabelled_id,QG_Bell.unique_unlabelled_id]

# For the trick of fixing to point distribution, we can simply compare mDAGs. The QmDAG structure is going to be useful only in the marginalization case (where classical and quantum latents appear)
    def reduces_to_knownQCGap_by_intervention(mDAG):
        for node in mDAG.visible_nodes:
                if mDAG.fix_to_point_distribution(node).unique_unlabelled_id in known_QC_Gaps_mDAGs_id:
                    return True
        return False
    
# =============================================================================
#     QC_gap_by_Intervention=[]
#     for eqclass in Observable_mDAGs4.foundational_eqclasses:
#         smart_representative=Observable_mDAGs4.representative_mDAGs_list[Observable_mDAGs4.foundational_eqclasses.index(eqclass)]
#         for mDAG in eqclass:
#             if reduces_to_knownQCGap_by_intervention(mDAG):
#                 QC_gap_by_Intervention.append(smart_representative)
#                 break
# =============================================================================

    QC_gap_by_Intervention=[]
    for mDAG in Observable_mDAGs4.representative_mDAGs_list:
        if reduces_to_knownQCGap_by_intervention(mDAG):
            QC_gap_by_Intervention.append(mDAG)
    
    print(len(Observable_mDAGs4.representative_mDAGs_list))
    print(len(QC_gap_by_Intervention))

