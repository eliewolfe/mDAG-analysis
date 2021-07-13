from __future__ import absolute_import

import numpy as np

#from metagraph import Observable_mDAGs
from unlabelled_metagraph import Observable_unlabelled_mDAGs
# from mDAG import mDAG
# import networkx as nx
# import json


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
    Observable_mDAGs = Observable_unlabelled_mDAGs(n)
    print("Number of unlabelled graph patterns: ", len(Observable_mDAGs.all_unlabelled_mDAGs), flush=True)
    fundamental_list = [mDAG.fundamental_graphQ for mDAG in Observable_mDAGs.all_unlabelled_mDAGs]
    print("Number of fundamental unlabelled graph patterns: ", len(np.flatnonzero(fundamental_list)), flush=True)

    eqclasses = Observable_mDAGs.equivalence_classes
    print("Upper bound on number of equivalence classes: ", len(eqclasses), flush=True)
    #unlabelled_eqclasses = [eqclass.intersection(Observable_mDAGs.symmetry_representatives) for eqclass in eqclasses]


    # # foundational_eqclasses = [eqclass for eqclass in eqclasses4 if (len(eqclass)==1 and Observable_mDAGs4.lookup_mDAG(list(eqclass)).fundamental_graphQ) or (len(eqclass)>1 and all(mDAG.fundamental_graphQ for mDAG in Observable_mDAGs4.lookup_mDAG(list(eqclass))))]
    foundational_eqclasses = [eqclass for eqclass in eqclasses if all(mDAG.fundamental_graphQ for mDAG in Observable_mDAGs.lookup_mDAG(eqclass))]
    print("Upper bound on number of 100% foundational equivalence classes: ", len(foundational_eqclasses), flush=True)
    semi_foundational_eqclasses = [eqclass for eqclass in eqclasses if any(mDAG.fundamental_graphQ for mDAG in Observable_mDAGs.lookup_mDAG(eqclass))]
    print("Upper bound on number of partially foundational equivalence classes: ", len(semi_foundational_eqclasses), flush=True)




#TODO: Only compute dominance relations for symmetry representatives
#TODO: Provide construction for representative simplicial complices
#TODO: Bypass networkx metagraph entirely for connected component calculations?