from __future__ import absolute_import

import numpy as np

from metagraph import Observable_mDAGs
from mDAG import mDAG
import networkx as nx
import json


if __name__ == '__main__':
    # # Testing example for Ilya conjecture proof.
    # directed_dict_of_lists = {"C":["A", "B"], "X":["C"], "D":["A", "B"]}
    # directed_structure = nx.from_dict_of_lists(directed_dict_of_lists, create_using=nx.DiGraph)
    # simplicial_complex2 = [("A", "X", "D"), ("B", "X", "D")]
    # simplicial_complex1 = [("A", "X", "D"), ("B", "X", "D"), ("A", "C")]
    # esep1 = mDAG(directed_structure, simplicial_complex1).all_esep
    # esep2 = mDAG(directed_structure, simplicial_complex2).all_esep
    # print(esep1.difference(esep2))




    #Let's also think about fundamentality.
    n = 4
    Observable_mDAGs = Observable_mDAGs(n)
    print("Number of unlabelled graph patterns: ", len(Observable_mDAGs.all_unlabelled_mDAGs))
    fundamental_list = [mDAG.fundamental_graphQ for mDAG in Observable_mDAGs.all_unlabelled_mDAGs]
    print("Number of fundamental unlabelled graph patterns: ", len(np.flatnonzero(fundamental_list)))

    eqclasses = Observable_mDAGs.equivalence_classes
    print("Upper bound on number of equivalence classes: ", len(eqclasses))
    unlabelled_eqclasses = [eqclass.intersection(Observable_mDAGs.symmetry_representatives) for eqclass in eqclasses]


    # # foundational_eqclasses = [eqclass for eqclass in eqclasses4 if (len(eqclass)==1 and Observable_mDAGs4.lookup_mDAG(list(eqclass)).fundamental_graphQ) or (len(eqclass)>1 and all(mDAG.fundamental_graphQ for mDAG in Observable_mDAGs4.lookup_mDAG(list(eqclass))))]
    foundational_eqclasses = [eqclass for eqclass in unlabelled_eqclasses if all(mDAG.fundamental_graphQ for mDAG in Observable_mDAGs.lookup_mDAG(eqclass))]
    print("Upper bound on number of 100% foundational equivalence classes: ", len(foundational_eqclasses))

    # #Let's confirm that e-separation relations are invariant across every eq class
    # #CONFIRMED, hence commented out.
    # problem_found = False
    # for eqclass in eqclasses4:
    #     if len(eqclass)>1 and not len(dict.fromkeys(frozenset(mDAG.all_esep) for mDAG in Observable_mDAGs4.lookup_mDAG(list(eqclass))))==1:
    #         print(eqclass)
    #         problem_found = True
    #         break
    # print("Problem Found? ", problem_found)

    # print("Writing to file: 'eqclasses_info.json'")
    #
    # returntouser = {
    #     'metagraph': sorted(Observable_mDAGs4.meta_graph.edges()),
    #     'eqclasses': [list(eqclass) for eqclass in Observable_mDAGs4.equivalence_classes],
    #     'simplicial_complices': Observable_mDAGs4.all_simplicial_complices,
    #     'directed_structures': Observable_mDAGs4.all_directed_structures_as_tuples
    # }
    # f = open('eqclasses_info.json', 'w')
    # print(json.dumps(returntouser), file=f)
    # f.close()

    #We can then partition (unlabelled) eq-classes by determining if they have distinct (unlabelled) skeletons or if they have distinct unlabelled e-sep relations


