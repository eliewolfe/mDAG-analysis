from __future__ import absolute_import
import networkx as nx
import numpy as np
from itertools import combinations, repeat, chain, product, islice, permutations
import itertools
import scipy.special #For binomial coefficient
from mDAG import mDAG
from operator import itemgetter
# from radix import bitarray_to_int
from utilities import partsextractor, nx_to_int, hypergraph_to_int
from collections import defaultdict

from sys import hexversion

if hexversion >= 0x3080000:
    from functools import cached_property
elif hexversion >= 0x3060000:
    # with io.capture_output() as captured:
    #     !pip
    #     install
    #     backports.cached - property
    from backports.cached_property import cached_property
else:
    cached_property = property

class Observable_unlabelled_mDAGs:
    def __init__(self, n):
        """
        Parameters
        ----------
        n : int
            The number of observed variables.
        """
        self.n = n
        self.setn = tuple(range(self.n))
        self.error_collection = []

    @cached_property
    def all_directed_structures(self):
        l = []
        baseg = nx.DiGraph()
        baseg.add_nodes_from(self.setn, type="Visible")
        possible_node_pairs = list(combinations(self.setn, 2))
        for list_of_directions in product(range(2), repeat=scipy.special.comb(self.n, 2, exact=True)):
            g = baseg.copy()  # Reset
            g.add_edges_from(
                choice for choice, direction in zip(possible_node_pairs, list_of_directions) if direction)
            l.append(g.copy())
        return l

    @cached_property
    def num_directed_structures(self):
        return len(self.all_directed_structures)

    @cached_property
    def all_directed_structures_as_tuples(self):
        # return list(map(nx.edges,self.all_directed_structures))
        return [tuple(sorted(d.edges())) for d in self.all_directed_structures]

    @cached_property
    def dict_directed_structures(self):
        # return list(map(nx.edges,self.all_directed_structures))
        #return [nx_to_int(d):i for i,d in enumerate(self.all_directed_structures)]
        #return dict(zip(map(nx_to_int, self.all_directed_structures), range(len(self.all_directed_structures))))
        return dict(zip(map(nx_to_int, self.all_directed_structures), self.all_directed_structures))


    # def lookup_directed_structure_index(self, mDAG):
    #     # return self.all_directed_structures_as_tuples.index(tuple(sorted(mDAG.directed_structure.edges())))
    #     return nx_to_int(mDAG.directed_structure)

    @cached_property
    def all_simplicial_complices(self):
        list_of_simplicial_complices = [[]]
        for ch in sorted(chain.from_iterable(combinations(self.setn, i) for i in range(2, self.n + 1))):
            ch_as_set = set(ch)
            list_of_simplicial_complices.extend(
                [simplicial_complex + [ch] for simplicial_complex in list_of_simplicial_complices if (
                        all((precedent <= ch) for precedent in simplicial_complex) and not any(
                    (ch_as_set.issubset(ch_list) or ch_list.issubset(ch)) for ch_list in map(set, simplicial_complex))
                )])
        return [tuple(
            sorted(simplicial_complex + [(singleton,) for singleton in set(self.setn).difference(*simplicial_complex)])) for
                simplicial_complex in list_of_simplicial_complices]

    @cached_property
    def dict_simplicial_complices(self):
        # return dict(zip(map(hypergraph_to_int, self.all_simplicial_complices), range(len(self.all_simplicial_complices))))
        return dict(zip(map(hypergraph_to_int, self.all_simplicial_complices), self.all_simplicial_complices))
    #
    # def lookup_simplicial_complex_index(self, mDAG):
    #     # return self.all_simplicial_complices.index(tuple(sorted(mDAG.extended_simplicial_complex)))
    #     return hypergraph_to_int(mDAG.extended_simplicial_complex)

    #
    # def lookup_indices(self, mDAG):
    #     d = self.lookup_directed_structure_index(mDAG)
    #     s = self.lookup_simplicial_complex_index(mDAG)
    #     return (s, d)

    @cached_property
    def num_simplicial_complices(self):
        return len(self.all_simplicial_complices)





    # @cached_property
    # def all_indexed_mDAGs(self):
    #     return [(mDAG(directed_structure, list(simplicial_complex), complex_extended=True), s, d)
    #             for s, simplicial_complex in enumerate(self.all_simplicial_complices)
    #             for d, directed_structure in enumerate(self.all_directed_structures)
    #             ]
    #
    @cached_property
    def all_labelled_mDAGs(self):
        return [mDAG(ds, list(sc), complex_extended=True)
                for sc in self.all_simplicial_complices
                for ds in self.all_directed_structures]

    @cached_property
    def dict_ind_labelled_mDAGs(self):
        return {mDAG.unique_id: mDAG for mDAG in self.all_labelled_mDAGs}

    @cached_property
    def dict_ind_unlabelled_mDAGs(self):
        return {mDAG.unique_unlabelled_id: mDAG for mDAG in self.all_labelled_mDAGs}

    def lookup_mDAG(self, indices):
        return partsextractor(self.dict_ind_unlabelled_mDAGs, indices)
        # return partsextractor(self.dict_ind_labelled_mDAGs, indices) #if we we working with labelled mDAGs

    @cached_property
    def dict_id_to_canonical_id(self):
        print("Computing canonical (unlabelled) graphs...", flush=True)
        return {mDAG.unique_id: mDAG.unique_unlabelled_id for mDAG in self.all_labelled_mDAGs}

    def mdag_int_pair_to_single_int(self, sc_int, ds_int):
        return ds_int + sc_int*(2**(self.n**2))

    def mdag_int_pair_to_canonical_int(self, sc_int, ds_int):
        #print(ds_int, sc_int, self.mdag_int_pair_to_single_int(ds_int, sc_int))
        return self.dict_id_to_canonical_id[self.mdag_int_pair_to_single_int(sc_int, ds_int)]

    @property
    def all_unlabelled_mDAGs(self):
        return self.dict_ind_unlabelled_mDAGs.values()

    @cached_property
    def labelled_meta_graph_nodes(self):
        return np.asarray(self.dict_ind_labelled_mDAGs.keys())

    @cached_property
    def meta_graph_nodes(self):
        return tuple(self.dict_ind_unlabelled_mDAGs.keys())

    @staticmethod
    def can_D1_simulate_D2(D1, D2):
        """
        D1 and D2 are networkx.DiGraph objects.
        We say that D1 can 'simulate' D2 if the edges of D2 are contained within those of D1.
        """
        # return set(D2.edges()).issubset(D1.edges())
        # Modifying to restrict to minimal differences for speed
        D1edges = D1.edges()
        D2edges = D2.edges()
        return len(D1edges) == len(D2edges) + 1 and set(D2.edges()).issubset(D1.edges())

    @staticmethod
    def can_S1_simulate_S2(S1, S2):
        """
        S1 and S2 are simplicial complices, in our data structure as lists of tuples.
        """
        # Modifying to restrict to minimal differences for speed
        # return all(any(s2.issubset(s1) for s1 in S1) for s2 in map(set, S2))
        dominance_count = 0
        so_far_so_good = True
        for s2 in map(set, S2):
            contained = False
            for s1 in S1:
                if s2.issubset(s1):
                    contained = True
                    if len(s2) > len(s1):
                        dominance_count += 0
                    break
            so_far_so_good = contained and (dominance_count <= 1)
            if not so_far_so_good:
                break
        return so_far_so_good

    @cached_property
    def directed_dominances(self):
        return [(first[0], second[0]) for first, second in permutations(self.dict_directed_structures.items(), 2) if
                self.can_D1_simulate_D2(first[1], second[1])]

    @cached_property
    def hypergraph_dominances(self):
        return [(first[0], second[0]) for first, second in permutations(self.dict_simplicial_complices.items(), 2) if
                self.can_S1_simulate_S2(first[1], second[1])]

    @property
    def meta_graph_directed_edges(self):
        # I'm experimenting with only using the directed dominances.
        for h in self.dict_simplicial_complices.keys():
            for d1, d2 in self.directed_dominances:
                yield (self.mdag_int_pair_to_canonical_int(h, d1), self.mdag_int_pair_to_canonical_int(h, d2))
        for d in self.dict_directed_structures.keys():
            for h1, h2 in self.hypergraph_dominances:
                yield (self.mdag_int_pair_to_canonical_int(h1, d), self.mdag_int_pair_to_canonical_int(h2, d))


    # def lookup_mDAG(self, indices):
    #     # Untested. I think this works well for for multiple sets of indices, may need some wrapping if only one set of indices is given.
    #     return itemgetter(
    #         *np.ravel_multi_index(np.array(indices).T, (self.num_simplicial_complices, self.num_directed_structures)))(
    #         self.all_mDAGs)
    #
    # def lookup_mDAGs(self, multiple_indices):
    #     as_array = np.array(list(multiple_indices))
    #     looked_up = itemgetter(
    #         *np.ravel_multi_index(as_array.T, (self.num_simplicial_complices, self.num_directed_structures)))(
    #         self.all_mDAGs)
    #     if as_array.__len__() == 1:
    #         return [looked_up]
    #     else:
    #         return looked_up


    #
    # @property
    # def meta_graph_symmetry_edges(self):
    #     for ((s, d), mDAG) in self.dict_ind_mDAGs.items():
    #         for mDAG2 in mDAG.generate_symmetric_variants:
    #             d2 = self.lookup_directed_structure_index(mDAG2)
    #             s2 = self.lookup_simplicial_complex_index(mDAG2)
    #             yield ((s2, d2), (s, d))
    #
    # @cached_property
    # def symmetry_graph(self):
    #     g = nx.DiGraph()
    #     g.add_nodes_from(self.meta_graph_nodes)
    #     print('Adding symmetry relations...')
    #     g.add_edges_from(self.meta_graph_symmetry_edges)
    #     return g
    #
    # @cached_property
    # def symmetry_classes(self):
    #     return list(nx.strongly_connected_components(self.symmetry_graph))
    #
    # @cached_property
    # def symmetry_representatives(self):
    #     return representatives(self.symmetry_classes)
    #
    # @cached_property
    # def dict_ind_unlabelled_mDAGs(self):
    #     return dict(zip(self.symmetry_representatives, self.lookup_mDAG(self.symmetry_representatives)))
    #
    # @property
    # def all_unlabelled_mDAGs(self):
    #     return self.dict_ind_unlabelled_mDAGs.values()






    @property
    def safe_meta_graph_undirected_edges(self):
        for (id, mDAG) in self.dict_ind_unlabelled_mDAGs.items():  # NEW: Over graph patterns only
            for mDAG2 in mDAG.generate_weaker_mDAG_HLP:
                yield (mDAG2.unique_unlabelled_id, id)
            for mDAG2 in mDAG.generate_weaker_mDAGs_FaceSplitting:
                yield (mDAG2.unique_unlabelled_id, id)

    @property
    def meta_graph_undirected_edges(self):
        for (id, mDAG) in self.dict_ind_unlabelled_mDAGs.items():  # NEW: Over graph patterns only
            for mDAG2 in mDAG.generate_weaker_mDAG_HLP:
                yield (mDAG2.unique_unlabelled_id, id)
            for mDAG2 in mDAG.generate_weaker_mDAGs_FaceSplitting_Simultaneous:
                yield (mDAG2.unique_unlabelled_id, id)

    @property
    def HLP_edges(self):
        for (id, mDAG) in self.dict_ind_unlabelled_mDAGs.items():  # NEW: Over graph patterns only
            for mDAG2 in mDAG.generate_weaker_mDAG_HLP:
                yield (mDAG2.unique_unlabelled_id, id)

    @property
    def FaceSplitting_edges(self):
        for (id, mDAG) in self.dict_ind_unlabelled_mDAGs.items():  # NEW: Over graph patterns only
            for mDAG2 in mDAG.generate_weaker_mDAGs_FaceSplitting_Simultaneous:
                yield (mDAG2.unique_unlabelled_id, id)



    @cached_property
    def safe_meta_graph(self):
        g = nx.DiGraph()
        g.add_nodes_from(self.meta_graph_nodes)
        g.add_edges_from(self.meta_graph_directed_edges)
        g.add_edges_from(self.safe_meta_graph_undirected_edges)
        return g

    @cached_property
    def meta_graph(self):
        g = nx.DiGraph()
        g.add_nodes_from(self.meta_graph_nodes)
        print('Adding dominance relations...')
        g.add_edges_from(self.meta_graph_directed_edges)
        print('Adding equivalence relations...')
        edge_count = g.number_of_edges()
        g.add_edges_from(self.HLP_edges)
        new_edge_count =  g.number_of_edges()
        print('Number of HLP equivalence relations added: ', new_edge_count-edge_count)
        edge_count = new_edge_count
        g.add_edges_from(self.FaceSplitting_edges)
        new_edge_count = g.number_of_edges()
        print('Number of FaceSplitting equivalence relations added: ', new_edge_count - edge_count)
        #print('Metagraph has been constructed. Number of nontrivial edges: ',g.number_of_edges() - edge_count)
        return g

    @cached_property
    def safe_equivalence_classes(self):
        return list(nx.strongly_connected_components(self.safe_meta_graph))

    @cached_property
    def equivalence_classes(self):
        return list(nx.strongly_connected_components(self.meta_graph))

    @cached_property
    def safe_equivalence_classes_as_mDAGs(self):
        return [self.lookup_mDAG(eqclass) for eqclass in self.safe_equivalence_classes]

    @cached_property
    def equivalence_classes_as_mDAGs(self):
        return [self.lookup_mDAG(eqclass) for eqclass in self.equivalence_classes]

    # def same_eq_class(self, mDAG1, mDAG2):
    #     (s1, d1) = self.lookup_indices(mDAG1)
    #     (s2, d2) = self.lookup_indices(mDAG2)
    #     for eq_class in self.equivalence_classes:
    #         if (s1, d1) in eq_class and (s2, d2) in eq_class:
    #             return True
    #             break
    #     return False

    @cached_property
    def safe_representative_mDAGs_list(self):
        return [eqclass[0] for eqclass in self.safe_equivalence_classes_as_mDAGs]
        # rep_mDAGs=[]
        # indexed_unlabelled_mDAGs=self.dict_ind_unlabelled_mDAGs
        # eq_classes=self.safe_equivalence_classes
        # for equivalence_class in eq_classes:
        #   representative_indices=next(iter(equivalence_class))   #just pick some mDAG of every equivalence class
        #   representative_mDAG=indexed_unlabelled_mDAGs[representative_indices]
        #   rep_mDAGs.append(representative_mDAG)
        # return rep_mDAGs

    @cached_property
    def representative_mDAGs_list(self):
        return [eqclass[0] for eqclass in self.equivalence_classes_as_mDAGs]
        # rep_mDAGs=[]
        # indexed_unlabelled_mDAGs=self.dict_ind_unlabelled_mDAGs
        # eq_classes=self.equivalence_classes
        # for equivalence_class in eq_classes:
        #   representative_indices=next(iter(equivalence_class))   #just pick some mDAG of every equivalence class
        #   representative_mDAG=indexed_unlabelled_mDAGs[representative_indices]
        #   rep_mDAGs.append(representative_mDAG)
        # return rep_mDAGs

    def classify_by_attributes(self, representatives, attributes):
        """

        :param choice of either self.safe_representative_mDAGs_list or self.representative_mDAGs_list:
        :param attributes: list of attributes to classify mDAGs by. For a single attribute, use a list with one item.
        Attributes are given as STRINGS, e.g. ['all_CI_unlabelled', 'skeleton_unlabelled']
        :return: dictionary of mDAGS according to attributes.
        """
        d = defaultdict(set)
        for mDAG in representatives:
            d[tuple(mDAG.__getattribute__(prop) for prop in attributes)].add(mDAG)
        return d

    def safe_CI_classes(self):
        return self.classify_by_attributes(self.safe_representative_mDAGs_list, ['all_CI_unlabelled'])
        # dict_CI_relations={}
        # for representative_mDAG in self.safe_representative_mDAGs_list():
        #   dict_CI_relations[representative_mDAG]=tuple(representative_mDAG.all_CI_unlabelled)
        #  #returns dictionary where the CI relations are the keys and a list of the representative mDAGs of the corresponding equivalence classes are the values
        # return {CI:[rep for rep in dict_CI_relations.keys() if dict_CI_relations[rep] == CI] for CI in set(dict_CI_relations.values())}
    
    def CI_classes(self):
        return self.classify_by_attributes(self.representative_mDAGs_list, ['all_CI_unlabelled'])
        # dict_CI_relations={}
        # for representative_mDAG in self.representative_mDAGs_list():
        #   dict_CI_relations[representative_mDAG]=tuple(sorted(representative_mDAG.all_CI_unlabelled))
        #  #returns dictionary where the CI relations are the keys and a list of the representative mDAGs of the corresponding equivalence classes are the values
        # return {CI:[rep for rep in dict_CI_relations.keys() if dict_CI_relations[rep] == CI] for CI in set(dict_CI_relations.values())}

    # Acknowledgement: Leonardo Lessa helped me improve CI_classes
    
    def safe_esep_classes(self):
        return self.classify_by_attributes(self.safe_representative_mDAGs_list, ['all_esep_unlabelled'])
        # dict_esep_relations={}
        # for representative_mDAG in self.safe_representative_mDAGs_list():
        #   dict_esep_relations[representative_mDAG]=tuple(representative_mDAG.all_esep_unlabelled)
        #  #returns dictionary where the esep relations are the keys and a list of the representative mDAGs of the corresponding equivalence classes are the values
        # return {esep:[rep for rep in dict_esep_relations.keys() if dict_esep_relations[rep] == esep] for esep in set(dict_esep_relations.values())}

    def esep_classes(self):
        return self.classify_by_attributes(self.representative_mDAGs_list, ['all_esep_unlabelled'])
        # dict_esep_relations={}
        # for representative_mDAG in self.representative_mDAGs_list():
        #   dict_esep_relations[representative_mDAG]=tuple(representative_mDAG.all_esep_unlabelled)
        #  #returns dictionary where the esep relations are the keys and a list of the representative mDAGs of the corresponding equivalence classes are the values
        # return {esep:[rep for rep in dict_esep_relations.keys() if dict_esep_relations[rep] == esep] for esep in set(dict_esep_relations.values())}

    def Skeleton_classes(self):
        return self.classify_by_attributes(self.representative_mDAGs_list, ['skeleton_unlabelled'])
        # eq_classes_representative_mDAGs=self.representative_mDAGs_list()
        # skeleton_classes=[[eq_classes_representative_mDAGs[0]]]
        # for k in range(1,len(eq_classes_representative_mDAGs)):
        #   graph=eq_classes_representative_mDAGs[k]
        #   already_there=False
        #   for same_skeleton in skeleton_classes:
        #     if graph.skeleton_unlabelled==same_skeleton[0].skeleton_unlabelled:
        #       same_skeleton.append(graph)
        #       already_there=True
        #   if not already_there:
        #     skeleton_classes.append([graph])
        # return skeleton_classes
    
    def Skeleton_and_CI(self):
        return self.classify_by_attributes(self.representative_mDAGs_list, ['skeleton_unlabelled', 'all_CI_unlabelled'])
        # Skeleton_and_CI = []
        # CI_classes=list(self.CI_classes().values())
        # sk_classes=self.Skeleton_classes()
        # for x in CI_classes:
        #   for y in sk_classes:
        #     x_and_y = list(filter(lambda g:g in x, y))
        #     if x_and_y:
        #       Skeleton_and_CI.append(x_and_y)
        # return Skeleton_and_CI
    
    def Skeleton_and_esep(self):
        return self.classify_by_attributes(self.representative_mDAGs_list, ['skeleton_unlabelled', 'all_esep_unlabelled'])
        # Skeleton_and_esep = []
        # esep_classes=list(self.esep_classes().values())
        # sk_classes=self.Skeleton_classes()
        # for x in esep_classes:
        #   for y in sk_classes:
        #     x_and_y = list(filter(lambda g:g in x, y))
        #     if x_and_y:
        #       Skeleton_and_esep.append(x_and_y)
        # return Skeleton_and_esep

    def groupby_then_split_by(self, level1attributes, level2attributes):
        d = defaultdict(set)
        joint_attributes = level1attributes + level2attributes
        critical_range = len(level1attributes)
        for mDAG in self.representative_mDAGs_list:
            d[tuple(mDAG.__getattribute__(prop) for prop in joint_attributes)].add(mDAG)
        d2 = defaultdict(dict)
        for key_tuple, partition in d.items():
            d2[key_tuple[:critical_range]][key_tuple] = tuple(partition)
        return [val for val in d2.values() if len(val)>1]



    
    # def is_mDAG_in_list(self, mDAG, l):
    #     mDAG_id=mDAG.unique_unlabelled_id
    #     for mDAG_l in l:
    #         mDAG_l_id=mDAG_l.unique_unlabelled_id
    #         if mDAG_l_id==mDAG_id:
    #             return True
    #     return False
    
    #examples of mDAGs that are shown inequivalent by comparison1 but not by comparison2
    # def Example_searcher(self, classes_comparison1, classes_comparison2):
    #     for (prop2_v1, prop2_v2) in itertools.combinations(classes_comparison1, 2):
    #         for prop1 in classes_comparison2:
    #             overlap_v1 = prop2_v1.intersection(prop1)
    #             if len(overlap_v1)>=1:
    #                 overlap_v2 = prop2_v2.intersection(prop1)
    #                 if len(overlap_v2)>=1:
    #                     for pair in itertools.product(overlap_v1, overlap_v2):
    #                         yield pair
    #                 else:
    #                     break
    #             else:
    #                 break
        # for class1 in classes_comparison1:
        #     if class1 not in classes_comparison2:
        #         for mDAG1 in class1:
        #             for class2 in classes_comparison2:
        #                 if self.is_mDAG_in_list(mDAG1,class2):
        #                     for mDAG2 in class2:
        #                         if not self.is_mDAG_in_list(mDAG2,class1):
        #                             yield (mDAG1, mDAG2)
        
    
    @cached_property
    def singleton_equivalence_classes(self):
        return [tuple(eqclass)[0] for eqclass in self.equivalence_classes if len(eqclass) == 1]
