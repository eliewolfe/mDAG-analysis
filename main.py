from __future__ import absolute_import

import numpy as np

#from metagraph import Observable_mDAGs
from metagraph_advanced import Observable_unlabelled_mDAGs
# from mDAG import mDAG
# import networkx as nx
# import json
from utilities import convert_eqclass_dict_to_representatives_dict


if __name__ == '__main__':
    # # Testing example for Ilya conjecture proof.
    # directed_dict_of_lists = {"C":["A", "B"], "X":["C"], "D":["A", "B"]}
    # DirectedStructure = nx.from_dict_of_lists(directed_dict_of_lists, create_using=nx.DiGraph)
    # simplicial_complex2 = [("A", "X", "D"), ("B", "X", "D")]
    # simplicial_complex1 = [("A", "X", "D"), ("B", "X", "D"), ("A", "C")]
    # esep1 = mDAG(DirectedStructure, simplicial_complex1).all_esep
    # esep2 = mDAG(DirectedStructure, simplicial_complex2).all_esep
    # print(esep1.difference(esep2))

    n = 4
    Observable_mDAGs = Observable_unlabelled_mDAGs(n)
    print("Number of unlabelled graph patterns: ", len(Observable_mDAGs.all_unlabelled_mDAGs), flush=True)
    fundamental_list = [mDAG.fundamental_graphQ for mDAG in Observable_mDAGs.all_unlabelled_mDAGs]
    print("Number of fundamental unlabelled graph patterns: ", len(np.flatnonzero(fundamental_list)), flush=True)

    eqclasses = Observable_mDAGs.equivalence_classes_as_mDAGs
    print("Upper bound on number of equivalence classes: ", len(eqclasses), flush=True)

    foundational_eqclasses = Observable_mDAGs.foundational_eqclasses
    print("Upper bound on number of 100% foundational equivalence classes: ", len(foundational_eqclasses),
          flush=True)

    print("Number of Foundational CI classes: ", len(Observable_mDAGs.CI_classes))
    print("Number of Foundational Skeleton classes: ", len(Observable_mDAGs.Skeleton_classes))
    print("Number of Foundational Skeleton+CI classes: ", len(Observable_mDAGs.Skeleton_and_CI))
    print("Number of Foundational ESEP classes: ", len(Observable_mDAGs.esep_classes))
    print("Number of Foundational Skeleton+ESEP classes: ", len(Observable_mDAGs.Skeleton_and_esep))

    same_esep_different_skeleton = Observable_mDAGs.groupby_then_split_by(
        ['all_esep_unlabelled'], ['skeleton_unlabelled'])
    same_skeleton_different_CI = Observable_mDAGs.groupby_then_split_by(
        ['skeleton_unlabelled'], ['all_CI_unlabelled'])
    same_CI_different_skeleton = Observable_mDAGs.groupby_then_split_by(
        ['all_CI_unlabelled'], ['skeleton_unlabelled'])

    for example_collection in (same_esep_different_skeleton, same_skeleton_different_CI, same_CI_different_skeleton):
        for example in example_collection:
            convert_eqclass_dict_to_representatives_dict(example)










#TODO: Only compute dominance relations for symmetry representatives
#TODO: Provide construction for representative simplicial complices
#TODO: Bypass networkx metagraph entirely for connected component calculations?