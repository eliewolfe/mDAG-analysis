from mDAG import mDAG
import networkx as nx
directed_structure = nx.from_dict_of_lists({0:[1,2,3], 1:[2]}, create_using=nx.DiGraph)
simplicial_structure = [(0,1),(0,2),(0,3)]
mDAG1 = mDAG(directed_structure, simplicial_structure)
print(mDAG1.skeleton_bitarray.astype(int))
print(mDAG1.skeleton)
print(mDAG1.skeleton_unlabelled)
from radix import int_to_bitarray
print(int_to_bitarray(mDAG1.skeleton_unlabelled,4))