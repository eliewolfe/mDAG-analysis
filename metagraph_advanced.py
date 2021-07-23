from __future__ import absolute_import
import networkx as nx
import numpy as np
import itertools
from more_itertools import ilen
# import scipy.special #For binomial coefficient
import progressbar
# from operator import itemgetter
from utilities import partsextractor
from collections import defaultdict
# from radix import int_to_bitarray

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

def classify_by_attributes(representatives, attributes, verbose=False):
    """
    :param choice of either self.safe_representative_mDAGs_list or self.representative_mDAGs_list:
    :param attributes: list of attributes to classify mDAGs by. For a single attribute, use a list with one item.
    Attributes are given as STRINGS, e.g. ['all_CI_unlabelled', 'skeleton_unlabelled']
    :return: dictionary of mDAGS according to attributes.
    """
    if len(representatives)>1:
        d = defaultdict(set)
        if verbose:
            for mDAG in progressbar.progressbar(
                representatives, widgets=[progressbar.SimpleProgress(), progressbar.Bar(), ' (', progressbar.ETA(), ') ']):
                d[tuple(evaluate_property_or_method(mDAG, prop) for prop in attributes)].add(mDAG)
        else:
            for mDAG in representatives:
                d[tuple(evaluate_property_or_method(mDAG, prop) for prop in attributes)].add(mDAG)
        return tuple(d.values())
    else:
        return tuple(representatives)

def further_classify_by_attributes(eqclasses, attributes, verbose=False):
    return tuple(itertools.chain.from_iterable((classify_by_attributes(representatives, attributes, verbose) for representatives in eqclasses)))


def multiple_classifications(*args):
    return list(filter(None, itertools.starmap(set.intersection, itertools.product(*args))))

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
        for (id, mDAG) in self.dict_ind_unlabelled_mDAGs.items():  # NEW: Over graph patterns only
            for mDAG2 in mDAG.generate_weaker_mDAG_HLP:
                yield (self.dict_id_to_canonical_id[mDAG2], id)

    def FaceSplitting_edges(self, unsafe=True):
        for (id, mDAG) in self.dict_ind_unlabelled_mDAGs.items():  # NEW: Over graph patterns only
            for mDAG2 in mDAG.generate_weaker_mDAGs_FaceSplitting(unsafe=unsafe):
                yield (self.dict_id_to_canonical_id[mDAG2], id)


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

    @cached_property
    def representative_mDAGs_list(self):
    #     return self.representatives(self.equivalence_classes_as_mDAGs)
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



    @property
    def representatives_for_only_hypergraphs(self):
        #choosing representatives with the smaller number of edges will guarantee that hypergraph-only mDAGs are chosen
        return self.smart_representatives(self.foundational_eqclasses, 'n_of_edges')

    @cached_property
    def equivalent_to_only_hypergraph_representative(self):
        #diagnostic:
        for mDAG_by_hypergraph, mDAG_by_complexity in zip(
                self.representatives_for_only_hypergraphs, self.representative_mDAGs_list):
            if mDAG_by_hypergraph != mDAG_by_complexity and mDAG_by_hypergraph.n_of_edges==0:
                print(mDAG_by_hypergraph, " =!= ", mDAG_by_complexity)
        return {mDAG_by_complexity: mDAG_by_hypergraph.n_of_edges==0 for
                mDAG_by_hypergraph, mDAG_by_complexity in
                zip(self.representatives_for_only_hypergraphs, self.representative_mDAGs_list)}

    def effectively_all_singletons(self, eqclass):
        return all(self.equivalent_to_only_hypergraph_representative[mDAG] for mDAG in eqclass)

    def single_partition_lowerbound_count_accounting_for_hypergraph_inequivalence(self, eqclass):
        num_hypergraph_only_representatives = sum(partsextractor(self.equivalent_to_only_hypergraph_representative, eqclass))
        if num_hypergraph_only_representatives > 0:
            return num_hypergraph_only_representatives
        elif len(eqclass)>0:
            return 1
        else:
            return 0

    def lowerbound_count_accounting_for_hypergraph_inequivalence(self, eqclasses):
        return sum(map(self.single_partition_lowerbound_count_accounting_for_hypergraph_inequivalence, eqclasses))


    @property 
    def Only_Hypergraphs(self):
        # only_hypergraphs=[]
        # for i,mDAG in enumerate(self.representatives_for_only_hypergraphs):
        #     if mDAG.n_of_edges==0:
        #         only_hypergraphs.append(self.representative_mDAGs_list[i])
        # return only_hypergraphs
        return [mDAG_by_complexity for mDAG_by_hypergraph, mDAG_by_complexity in
            zip(self.representatives_for_only_hypergraphs, self.representative_mDAGs_list) if
                mDAG_by_hypergraph.n_of_edges==0]

    
    
    def Only_Hypergraphs_Rule(self, given_classification):  
        new_classification=list(given_classification)
        for classif in given_classification:
            only_hypergraphs_in_classif=list(set(classif) & set(self.Only_Hypergraphs))
            if len(only_hypergraphs_in_classif)>1:
                more_than_just_hypergraphs_in_classif= [element for element in classif if element not in only_hypergraphs_in_classif]
                new_classifs=[[only_hypergraph]+more_than_just_hypergraphs_in_classif for only_hypergraph in only_hypergraphs_in_classif]
                new_classification=new_classification+new_classifs
                new_classification.remove(classif)
        return tuple(new_classification)
            
        

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

class Observable_mDAGs_Analysis(Observable_unlabelled_mDAGs):
    def __init__(self, nof_observed_variables=4, max_nof_events_for_supports=3):
        super().__init__(nof_observed_variables)
        self.max_nof_events = max_nof_events_for_supports

        self.singletons_dict = dict({1: list(itertools.chain.from_iterable(
            filter(lambda eqclass: (len(eqclass) == 1 or self.effectively_all_singletons(eqclass)),
                   self.esep_classes)))})
        self. non_singletons_dict = dict({1: sorted(
            filter(lambda eqclass: (len(eqclass) > 1 and not self.effectively_all_singletons(eqclass)),
                   self.esep_classes), key=len)})
        print("# of singleton classes from ESEP+Prop 6.8: ", len(self.singletons_dict[1]))
        print("# of non-singleton classes from ESEP+Prop 6.8: ", self.lowerbound_count_accounting_for_hypergraph_inequivalence(self.non_singletons_dict[1]),
              ", comprising {} total foundational graph patterns (no repetitions)".format(
                  ilen(itertools.chain.from_iterable(self.non_singletons_dict[1]))))

        smart_supports_dict = dict()
        for k in range(2, self.max_nof_events + 1):
            print("[Working on nof_events={}]".format(k))
            smart_supports_dict[k] = further_classify_by_attributes(self.non_singletons_dict[k - 1],
                                                            [('smart_infeasible_binary_supports_n_events_unlabelled',
                                                              k)], verbose=False)
            self.singletons_dict[k] = list(itertools.chain.from_iterable(
                filter(lambda eqclass: (len(eqclass) == 1 or self.effectively_all_singletons(eqclass)),
                       smart_supports_dict[k]))) + self.singletons_dict[k - 1]
            self.non_singletons_dict[k] = sorted(
                filter(lambda eqclass: (len(eqclass) > 1 and not self.effectively_all_singletons(eqclass)),
                       smart_supports_dict[k]), key=len)
            print("# of singleton classes from ESEP+Supports Up To {}: ".format(k), len(self.singletons_dict[k]))
            print("# of non-singleton classes from ESEP+Supports Up To {}: ".format(k), self.lowerbound_count_accounting_for_hypergraph_inequivalence(self.non_singletons_dict[k]),
                  ", comprising {} total foundational graph patterns".format(
                      ilen(itertools.chain.from_iterable(self.non_singletons_dict[k]))))







if __name__ == '__main__':

    Observable_mDAGs = Observable_mDAGs_Analysis(nof_observed_variables=4, max_nof_events_for_supports=3)

    # n = 4
    # Observable_mDAGs = Observable_unlabelled_mDAGs(n)
    #
    # print("Number of unlabelled graph patterns: ", len(Observable_mDAGs.all_unlabelled_mDAGs), flush=True)
    # fundamental_list = [mDAG.fundamental_graphQ for mDAG in Observable_mDAGs.all_unlabelled_mDAGs]
    # print("Number of fundamental unlabelled graph patterns: ", len(np.flatnonzero(fundamental_list)), flush=True)
    #
    # eqclasses = Observable_mDAGs.equivalence_classes_as_mDAGs
    # print("Upper bound on number of equivalence classes: ", len(eqclasses), flush=True)
    #
    # foundational_eqclasses = Observable_mDAGs.foundational_eqclasses
    # print("Upper bound on number of 100% foundational equivalence classes: ", len(foundational_eqclasses),
    #       flush=True)
    #
    # print("Number of Foundational Only_Hypergraphs_Rule classes: ", len(Observable_mDAGs.Only_Hypergraphs))
    # print("Number of Foundational CI classes: ", len(Observable_mDAGs.CI_classes))
    # print("Number of Foundational Skeleton classes: ", len(Observable_mDAGs.Skeleton_classes))
    # print("Number of Foundational Skeleton+CI classes: ", len(multiple_classifications(Observable_mDAGs.CI_classes, Observable_mDAGs.Skeleton_classes)))
    # print("Number of Foundational ESEP classes: ", len(Observable_mDAGs.esep_classes))
    # print("Number of Foundational Skeleton+ESEP classes: ", len(multiple_classifications(Observable_mDAGs.esep_classes, Observable_mDAGs.Skeleton_classes)))
    # # print("Number of Foundational Only_Hypergraphs_Rule+ESEP classes: ",len(Observable_mDAGs.Only_Hypergraphs_Rule(Observable_mDAGs.esep_classes)))
    #
    # #MEMOIZATION TEST
    # print("Wasting time memoizing infeasible supports over 2 events...")
    # waste_time = classify_by_attributes(Observable_mDAGs.representative_mDAGs_list,
    #         [('smart_infeasible_binary_supports_n_events_unlabelled', 2)], verbose=True)
    # print("Finding instances with no infeasible supports over 2 events:")
    # no_infeasible_support_over_2_events = [mDAG for mDAG in  Observable_mDAGs.representative_mDAGs_list if
    #                                         mDAG.smart_support_testing_instance(2).no_infeasible_supports()]
    # print(len(no_infeasible_support_over_2_events), " such instances found.\n")
    #
    # # for i in Observable_mDAGs.Only_Hypergraphs:
    # #     print(i)
    #
    #
    # max_nof_events = 3
    # smart_supports_dict = dict()
    # singletons_dict = dict({1: list(filter(lambda eqclass:len(eqclass)==1, Observable_mDAGs.esep_classes))})
    # non_singletons_dict = dict({1: sorted(filter(lambda eqclass:len(eqclass)>1, Observable_mDAGs.esep_classes), key=len)})
    # print("# of singleton classes from ESEP: ", len(singletons_dict[1]))
    # print("# of non-singleton classes from ESEP: ", len(non_singletons_dict[1]),
    #       ", comprising {} total foundational graph patterns".format(ilen(itertools.chain.from_iterable(non_singletons_dict[1]))))
    # # singletons_dict = dict({1: list(itertools.chain.from_iterable(filter(lambda eqclass: (len(eqclass)==1 or Observable_mDAGs.effectively_all_singletons(eqclass)), Observable_mDAGs.Only_Hypergraphs_Rule(Observable_mDAGs.esep_classes))))})
    # # non_singletons_dict = dict({1: sorted(filter(lambda eqclass: (len(eqclass)>1 and not Observable_mDAGs.effectively_all_singletons(eqclass)),  Observable_mDAGs.Only_Hypergraphs_Rule(Observable_mDAGs.esep_classes)), key=len)})
    # singletons_dict = dict({1: list(itertools.chain.from_iterable(filter(lambda eqclass: (len(eqclass)==1 or Observable_mDAGs.effectively_all_singletons(eqclass)), Observable_mDAGs.esep_classes)))})
    # non_singletons_dict = dict({1: sorted(filter(lambda eqclass: (len(eqclass)>1 and not Observable_mDAGs.effectively_all_singletons(eqclass)),  Observable_mDAGs.esep_classes), key=len)})
    # print("# of singleton classes from ESEP+Prop 6.8: ", len(singletons_dict[1]))
    # print("# of non-singleton classes from ESEP+Prop 6.8: ", len(non_singletons_dict[1]),
    #       ", comprising {} total foundational graph patterns (no repetitions)".format(ilen(itertools.chain.from_iterable(non_singletons_dict[1]))))
    #
    # for k in range(2, max_nof_events+1):
    #     print("[Working on nof_events={}]".format(k))
    #     smart_supports_dict[k] = classify_by_attributes(itertools.chain.from_iterable(non_singletons_dict[k-1]),
    #         [('smart_infeasible_binary_supports_n_events_unlabelled', k)], verbose=True)
    #     singletons_dict[k] = list(itertools.chain.from_iterable(filter(lambda eqclass: (len(eqclass)==1 or Observable_mDAGs.effectively_all_singletons(eqclass)), smart_supports_dict[k]))) + singletons_dict[k-1]
    #     non_singletons_dict[k] = sorted(filter(lambda eqclass: (len(eqclass)>1 and not Observable_mDAGs.effectively_all_singletons(eqclass)), smart_supports_dict[k]), key=len)
    #     print("# of singleton classes from ESEP+Supports Up To {}: ".format(k), len(singletons_dict[k]))
    #     print("# of non-singleton classes from ESEP+Supports Up To {}: ".format(k), len(non_singletons_dict[k]),
    #           ", comprising {} total foundational graph patterns".format(ilen(itertools.chain.from_iterable(non_singletons_dict[k]))))

    for k in tuple(Observable_mDAGs.singletons_dict.keys())[1:]:
        for i in Observable_mDAGs.singletons_dict[k]:
            if i not in Observable_mDAGs.singletons_dict[k-1]:
                print(i)
        
        
        # print("Number of ESEP+Supports Up To {} classes: ".format(k), len(
        #     multiple_classifications(Observable_mDAGs.esep_classes,
        #                          *(smart_supports_dict[i] for i in range(2, k+1)))), flush=True)


    # same_esep_different_skeleton = Observable_mDAGs.groupby_then_split_by(
    #     ['all_esep_unlabelled'], ['skeleton_unlabelled'])
    # same_skeleton_different_CI = Observable_mDAGs.groupby_then_split_by(
    #     ['skeleton_unlabelled'], ['all_CI_unlabelled'])
    # same_CI_different_skeleton = Observable_mDAGs.groupby_then_split_by(
    #     ['all_CI_unlabelled'], ['skeleton_unlabelled'])