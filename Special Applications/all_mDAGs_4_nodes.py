from __future__ import absolute_import
from hypergraphs import Hypergraph
from directed_structures import DirectedStructure
from metagraph_advanced import Observable_unlabelled_mDAGs
from mDAG_advanced import mDAG
import string

Observable_unlabelled_mDAGs4=Observable_unlabelled_mDAGs(4)
unlabelled_mDAGs4=list(Observable_unlabelled_mDAGs4.all_unlabelled_mDAGs)

def dict_names(mDAG):
    dict_names={}
    for v in range(mDAG.number_of_visible):
        dict_names[v]=list(string.ascii_uppercase)[len(mDAG.nonsingleton_latent_nodes)+v]
    for l in range(mDAG.number_of_visible,mDAG.number_of_visible+len(mDAG.nonsingleton_latent_nodes)):
        dict_names[l]=list(string.ascii_uppercase)[l-mDAG.number_of_visible]
    return dict_names


def list_of_parents(mDAG):
    parents=[]
    for list_parents in mDAG.parents_of:
        new_names_list=[dict_names(mDAG)[k] for k in list_parents]
        parents.append(new_names_list)
    return(parents)


list_of_mDAGs=[]
for mDAG in unlabelled_mDAGs4:
    visible_nodes=[dict_names(mDAG)[i] for i in range(4)]
    parents=list_of_parents(mDAG)
    list_of_mDAGs.append([visible_nodes,parents])
    
# The variable list_of_mDAGs includes all 2809 mDAGs with 4 visibles in Shashaank's notation.
