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

    Observable_mDAGs3 = Observable_mDAGs(3)
    Observable_mDAGs4 = Observable_mDAGs(4)

    fundamental_list = [mDAG.fundamental_graphQ for mDAG in Observable_mDAGs4.all_mDAGs]
    print(len(fundamental_list))
    print(len(np.flatnonzero(fundamental_list)))

    eqclasses3 = Observable_mDAGs3.equivalence_classes
    # eqclasses4 = Observable_mDAGs4.equivalence_classes
    print(len(eqclasses3))

    # # foundational_eqclasses = [eqclass for eqclass in eqclasses4 if (len(eqclass)==1 and Observable_mDAGs4.lookup_mDAG(list(eqclass)).fundamental_graphQ) or (len(eqclass)>1 and all(mDAG.fundamental_graphQ for mDAG in Observable_mDAGs4.lookup_mDAG(list(eqclass))))]
    # foundational_eqclasses = [eqclass for eqclass in eqclasses4 if all(mDAG.fundamental_graphQ for mDAG in Observable_mDAGs4.lookup_mDAGs(eqclass))]
    # print(len(foundational_eqclasses))

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


