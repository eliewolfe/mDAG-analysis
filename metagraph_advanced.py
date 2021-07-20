from __future__ import absolute_import
import networkx as nx
import numpy as np
import itertools
import scipy.special #For binomial coefficient


from operator import itemgetter
from utilities import partsextractor
from collections import defaultdict

from radix import int_to_bitarray

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

from hypergraphs import hypergraph
from directed_structures import directed_structure
from mDAG_advanced import mDAG


def evaluate_property_or_method(instance, attribute):
    if isinstance(attribute, str):
        return getattr(instance, attribute)
    elif isinstance(attribute, tuple):
        return getattr(instance, attribute[0])(*attribute[1:])

def classify_by_attributes(representatives, attributes):
    """
    :param choice of either self.safe_representative_mDAGs_list or self.representative_mDAGs_list:
    :param attributes: list of attributes to classify mDAGs by. For a single attribute, use a list with one item.
    Attributes are given as STRINGS, e.g. ['all_CI_unlabelled', 'skeleton_unlabelled']
    :return: dictionary of mDAGS according to attributes.
    """
    d = defaultdict(set)
    for mDAG in representatives:
        d[tuple(evaluate_property_or_method(mDAG, prop) for prop in attributes)].add(mDAG)
    return tuple(d.values())



class Observable_unlabelled_mDAGs:
    def __init__(self, n):
        """
        Parameters
        ----------
        n : int
            The number of observed variables.
        """
        self.n = n
        self.tuple_n = tuple(range(self.n))
        self.set_n = set(self.tuple_n)
        self.sort_by_length = lambda s: sorted(s, key=len)

    @cached_property
    def all_directed_structures(self):
        possible_node_pairs = list(itertools.combinations(self.tuple_n, 2))
        return [directed_structure(self.tuple_n, edge_list) for r
                                                    in range(0, len(possible_node_pairs)+1) for edge_list
                                                    in itertools.combinations(possible_node_pairs, r)]

    @cached_property
    def all_unlabelled_directed_structures(self):
        d = defaultdict(list)
        for ds in self.all_directed_structures:
            d[ds.as_unlabelled_integer].append(ds)
        return tuple(next(iter(eqclass)) for eqclass in d.values())
        # return [ds for ds in excessive_unlabelled_directed_structures
        #        if all(all(any((edge2[0]==edge[0]-i and edge2[1]==edge[1]-i) for edge2 in ds.edge_list) for i in range(1,edge[0])) for edge in ds.edge_list)]

    # @cached_property
    # def all_simplicial_complices_old(self):
    #     list_of_simplicial_complices = [[]]
    #     for ch in itertools.chain.from_iterable(itertools.combinations(self.tuple_n, i) for i in range(2, self.n + 1)):
    #         ch_as_set = set(ch)
    #         list_of_simplicial_complices.extend(
    #             [simplicial_complex + [ch] for simplicial_complex in list_of_simplicial_complices if (
    #                     # all((precedent <= ch) for precedent in simplicial_complex) and
    #                     not any(
    #                 (ch_as_set.issubset(ch_list) or ch_list.issubset(ch)) for ch_list in map(set, simplicial_complex))
    #             )])
    #     return [tuple(
    #         [(singleton,) for singleton in self.set_n.difference(*simplicial_complex)] + simplicial_complex) for
    #             simplicial_complex in list_of_simplicial_complices]


    @cached_property
    def all_simplicial_complices(self):
        list_of_simplicial_complices = [[]]
        for ch in map(frozenset, itertools.chain.from_iterable(itertools.combinations(self.tuple_n, i) for i in range(2, self.n + 1))):
            list_of_simplicial_complices.extend(
                [sc + [ch] for sc in list_of_simplicial_complices if (
                    # (len(simplicial_complex)==0 or max(map(tuple, simplicial_complex)) <= tuple(ch)) and
                    not any((ch.issubset(ch_list) or ch_list.issubset(ch)) for ch_list in sc))])
            #At this stage we have all the *compressed* simplicial complices.
        return sorted((hypergraph([frozenset({v}) for v in self.set_n.difference(*sc)]+sc) for sc in list_of_simplicial_complices), key = lambda sc:sc.tally)

    @cached_property
    def all_labelled_mDAGs(self):
        return [mDAG(ds, sc) for sc in self.all_simplicial_complices for ds in self.all_directed_structures]

    @cached_property
    def dict_id_to_canonical_id(self):
        print("Computing canonical (unlabelled) graphs...", flush=True)
        return {mDAG.unique_id: mDAG.unique_unlabelled_id for mDAG in self.all_labelled_mDAGs}

    def mdag_int_pair_to_single_int(self, sc_int, ds_int):
        return ds_int + sc_int*(2**(self.n**2))

    def mdag_int_pair_to_canonical_int(self, sc_int, ds_int):
        #print(ds_int, sc_int, self.mdag_int_pair_to_single_int(ds_int, sc_int))
        return self.dict_id_to_canonical_id[self.mdag_int_pair_to_single_int(sc_int, ds_int)]

    @cached_property
    def dict_ind_unlabelled_mDAGs(self):
        return {mDAG.unique_unlabelled_id: mDAG for mDAG in self.all_labelled_mDAGs}

    def lookup_mDAG(self, indices):
        return partsextractor(self.dict_ind_unlabelled_mDAGs, indices)

    @property
    def all_unlabelled_mDAGs(self):
        return self.dict_ind_unlabelled_mDAGs.values()

    @cached_property
    def meta_graph_nodes(self):
        return tuple(self.dict_ind_unlabelled_mDAGs.keys())


    @property
    def hypergraph_dominances(self):
        return [(S1.as_integer, S2.as_integer) for S1, S2 in itertools.permutations(self.all_simplicial_complices, 2) if
                S1.can_S1_minimally_simulate_S2(S2)]

    @property
    def directed_dominances(self):
        # return [(D1.as_integer, D2.as_integer) for D1, D2 in itertools.permutations(self.all_unlabelled_directed_structures, 2) if
        #         D1.can_D1_minimally_simulate_D2(D2)]
        return [(D1.as_integer, D2.as_integer) for D1, D2 in
                itertools.permutations(self.all_directed_structures, 2) if
                D1.can_D1_minimally_simulate_D2(D2)]

    @property
    def unlabelled_dominances(self):
        # I'm experimenting with only using the directed dominances.
        for h in map(lambda sc: sc.as_integer, self.all_simplicial_complices):
            for d1, d2 in self.directed_dominances:
                yield (self.mdag_int_pair_to_canonical_int(h, d1), self.mdag_int_pair_to_canonical_int(h, d2))
        for d in  map(lambda ds: ds.as_integer, self.all_unlabelled_directed_structures):
            for h1, h2 in self.hypergraph_dominances:
                yield (self.mdag_int_pair_to_canonical_int(h1, d), self.mdag_int_pair_to_canonical_int(h2, d))

    @property
    def HLP_edges(self):
        # return [(self.dict_id_to_canonical_id[mDAG2], id) for
        #  (id, mDAG) in self.dict_ind_unlabelled_mDAGs.items() for
        #  mDAG2 in mDAG.generate_weaker_mDAG_HLP]
        for (id, mDAG) in self.dict_ind_unlabelled_mDAGs.items():  # NEW: Over graph patterns only
            for mDAG2 in mDAG.generate_weaker_mDAG_HLP:
                yield (self.dict_id_to_canonical_id[mDAG2], id)
                # if (id, self.dict_id_to_canonical_id[mDAG2]) not in self.unlabelled_dominances:
                #     print((int_to_bitarray(id,4), int_to_bitarray(self.dict_id_to_canonical_id[mDAG2],4)))
                # yield (id, self.dict_id_to_canonical_id[mDAG2]) #Not needed if directed dominances included

    def FaceSplitting_edges(self, unsafe=True):
        # return [(self.dict_id_to_canonical_id[mDAG2], id) for
        #  (id, mDAG) in self.dict_ind_unlabelled_mDAGs.items() for
        #  mDAG2 in mDAG.generate_weaker_mDAGs_FaceSplitting(unsafe=unsafe)]
        for (id, mDAG) in self.dict_ind_unlabelled_mDAGs.items():  # NEW: Over graph patterns only
            for mDAG2 in mDAG.generate_weaker_mDAGs_FaceSplitting(unsafe=unsafe):
                yield (self.dict_id_to_canonical_id[mDAG2], id)
                # yield (id, self.dict_id_to_canonical_id[mDAG2]) #Not needed if hypergraph dominance relations included
    #
    # def meta_graph_undirected_edges(self, unsafe=True):
    #     for (id, mDAG) in self.dict_ind_unlabelled_mDAGs.items():  # NEW: Over graph patterns only
    #         for mDAG2 in mDAG.generate_weaker_mDAG_HLP:
    #             yield (self.dict_id_to_canonical_id[mDAG2], id)
    #         for mDAG2 in mDAG.generate_weaker_mDAGs_FaceSplitting(unsafe=unsafe):
    #             yield (self.dict_id_to_canonical_id[mDAG2], id)

    @cached_property
    def meta_graph(self):
        g = nx.DiGraph()
        g.add_nodes_from(self.meta_graph_nodes)
        print('Adding dominance relations...', flush=True)
        g.add_edges_from(self.unlabelled_dominances)
        edge_count = g.number_of_edges()
        print('Adding HLP equivalence relations...', flush=True)
        g.add_edges_from(self.HLP_edges)
        new_edge_count =  g.number_of_edges()
        print('Number of HLP equivalence relations added: ', new_edge_count-edge_count)
        edge_count = new_edge_count
        print('Adding FaceSplitting equivalence relations...', flush=True)
        g.add_edges_from(self.FaceSplitting_edges(unsafe=True))
        new_edge_count = g.number_of_edges()
        print('Number of FaceSplitting equivalence relations added: ', new_edge_count - edge_count)
        print('Metagraph has been constructed. Total edge count: ', g.number_of_edges(), flush=True)
        return g

    @cached_property
    def equivalence_classes_as_ids(self):
        return list(nx.strongly_connected_components(self.meta_graph))

    # @cached_property
    # def safe_equivalence_classes_as_mDAGs(self):
    #     return [self.lookup_mDAG(eqclass) for eqclass in self.safe_equivalence_classes]

    @cached_property
    def equivalence_classes_as_mDAGs(self):
        return [self.lookup_mDAG(eqclass) for eqclass in self.equivalence_classes_as_ids]


    @cached_property
    def singleton_equivalence_classes(self):
        return [next(iter(eqclass)) for eqclass in self.equivalence_classes_as_mDAGs if len(eqclass) == 1]

    @cached_property
    def foundational_eqclasses(self):
        return list(filter(lambda eqclass: all(mDAG.fundamental_graphQ for mDAG in eqclass),
                           self.equivalence_classes_as_mDAGs))

    @staticmethod
    def representatives(eqclasses):
        return [next(iter(eqclass)) for eqclass in eqclasses]

    @staticmethod
    def smart_representatives(eqclasses, attribute):
        return [min(eqclass, key = lambda mDAG: mDAG.__getattribute__(attribute)) for eqclass in eqclasses]


    # @cached_property
    # def safe_representative_mDAGs_list(self):
    #     return self.representatives(self.safe_equivalence_classes_as_mDAGs)

    # @cached_property
    # def representative_mDAGs_list(self):
    #     return self.representatives(self.equivalence_classes_as_mDAGs)
    @cached_property
    def representative_mDAGs_list(self):
        return self.smart_representatives(self.foundational_eqclasses, 'relative_complexity_for_sat_solver')

    @property
    def CI_classes(self):
        return classify_by_attributes(self.representative_mDAGs_list, ['all_CI_unlabelled'])

    @property
    def esep_classes(self):
        return classify_by_attributes(self.representative_mDAGs_list, ['all_esep_unlabelled'])

    @property
    def Skeleton_classes(self):
        return classify_by_attributes(self.representative_mDAGs_list, ['skeleton_unlabelled'])

    @property
    def Skeleton_and_CI(self):
        return classify_by_attributes(self.representative_mDAGs_list, ['skeleton_unlabelled', 'all_CI_unlabelled'])

    @property
    def Skeleton_and_esep(self):
        return classify_by_attributes(self.representative_mDAGs_list,
                                           ['skeleton_unlabelled', 'all_esep_unlabelled'])

    def groupby_then_split_by(self, level1attributes, level2attributes):
        d = defaultdict(set)
        joint_attributes = level1attributes + level2attributes
        critical_range = len(level1attributes)
        for mDAG in self.representative_mDAGs_list:
            d[tuple(mDAG.__getattribute__(prop) for prop in joint_attributes)].add(mDAG)
        d2 = defaultdict(dict)
        for key_tuple, partition in d.items():
            d2[key_tuple[:critical_range]][key_tuple] = tuple(partition)
        return [val for val in d2.values() if len(val) > 1]






if __name__ == '__main__':
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
    print("Number of ESEP+Supports2 classes: ", len(classify_by_attributes(
        Observable_mDAGs.representative_mDAGs_list,
        ['all_esep_unlabelled', ('smart_infeasible_binary_supports_n_events_unlabelled',2)])))

    same_esep_different_skeleton = Observable_mDAGs.groupby_then_split_by(
        ['all_esep_unlabelled'], ['skeleton_unlabelled'])
    same_skeleton_different_CI = Observable_mDAGs.groupby_then_split_by(
        ['skeleton_unlabelled'], ['all_CI_unlabelled'])
    same_CI_different_skeleton = Observable_mDAGs.groupby_then_split_by(
        ['all_CI_unlabelled'], ['skeleton_unlabelled'])