from __future__ import absolute_import
from hypergraphs import Hypergraph, LabelledHypergraph
from directed_structures import DirectedStructure, LabelledDirectedStructure
from mDAG_advanced import mDAG
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
import itertools

G_test=mDAG(DirectedStructure([(0,1),(1,2)],3),Hypergraph([(0,1)],3))
supps_to_be_checked=G_test.support_testing_instance((4, 2, 2),4).unique_candidate_supports_as_compressed_matrices
G_test.support_testing_instance((4, 2, 2),4).extract_support_matrices_satisfying_pprestrictions(supps_to_be_checked,((1,(0,)),))

G_efg_Fritz=mDAG(DirectedStructure([(0,1),(1,2),(2,3),(1,3)],4),Hypergraph([(0,2),(0,3)],4))

global inf_sup_obeying_pp  
def inf_sup_obeying_pp(t):
    (card,n)=t
    s=G_efg_Fritz.support_testing_instance(card,n)
    infeasible_supports=s.unique_infeasible_supports_as_expanded_matrices(name='mgh', use_timer=False)
    result=s.extract_support_matrices_satisfying_pprestrictions(infeasible_supports,((2,(0,)),(3,(0,))))
    print(result)
    return result

cards=[(2,2,2,2),(4,2,2,2),(4,4,2,2),(6,2,2,2)]
card_and_n=list(itertools.product(cards, list(range(3,7))))

checked_list=process_map(inf_sup_obeying_pp, card_and_n, tqdm_class=tqdm)
infsup_obey_pp=list(filter(lambda x: x is not None, checked_list))

print(infsup_obey_pp)

s=G_efg_Fritz.support_testing_instance((4,2,2,2),4)
obey_pp_matrix=s.extract_support_matrices_satisfying_pprestrictions(s.unique_candidate_supports_as_compressed_matrices,((2,(0,)),(3,(0,))))
obey_pp_integers=[]
for a in obey_pp_matrix:
    obey_pp_integers.append(s.from_matrix_to_integer(a))


a=obey_pp_matrix[0]
s.from_matrix_to_integer([[0, 0, 0, 0],
       [0, 0, 0, 1],
       [0, 1, 1, 0],
       [2, 0, 1, 0]])
s.from_integer_to_matrix(1234)
s.unique_infeasible_supports_as_integers_among(s.unique_candidate_supports_as_compressed_lists)
s.attempt_to_find_one_infeasible_support
