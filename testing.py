from mDAG import mDAG
import networkx as nx
from radix import int_to_bitarray
# directed_structure = nx.from_dict_of_lists({0:[1,2,3], 1:[2]}, create_using=nx.DiGraph)
# simplicial_structure = [(0,1),(0,2),(0,3)]
# mDAG1 = mDAG(directed_structure, simplicial_structure)
# print(mDAG1.skeleton_bitarray.astype(int))
# print(mDAG1.skeleton)
# print(mDAG1.skeleton_unlabelled)
# print(int_to_bitarray(mDAG1.skeleton_unlabelled,4))

ds1 = nx.from_dict_of_lists({0:[1],1:[2],2:[3]}, create_using=nx.DiGraph)
ds2 = nx.from_dict_of_lists({0:[3],2:[3]}, create_using=nx.DiGraph)
ds2.add_nodes_from([1])
sc1 = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
sc2 = [(0,1,3),(0,2),(1,2),(2,3)]
mDAG1 = mDAG(ds1, sc1)
mDAG2 = mDAG(ds2, sc2)
# print(mDAG1.skeleton_bitarray.astype(int))
# print(mDAG2.skeleton_bitarray.astype(int))
print(int_to_bitarray(mDAG1.skeleton_unlabelled,4))
print(int_to_bitarray(mDAG2.skeleton_unlabelled,4))
print(mDAG1.skeleton_unlabelled)
print(mDAG2.skeleton_unlabelled)
print(mDAG1.all_esep)
print(mDAG2.all_esep)
