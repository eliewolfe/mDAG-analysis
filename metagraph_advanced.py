from __future__ import absolute_import
import networkx as nx
import numpy as np
import itertools
from more_itertools import ilen
# import scipy.special #For binomial coefficient
import progressbar
from utilities import partsextractor
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

from hypergraphs import Hypergraph, LabelledHypergraph
from directed_structures import DirectedStructure, LabelledDirectedStructure
from mDAG_advanced import mDAG
from functools import lru_cache

# @lru_cache(maxsize=None)
# def mDAG(directed_structure_instance, simplicial_complex_instance):
#     return uncached_mDAG(directed_structure_instance, simplicial_complex_instance)


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

def further_classify_by_attributes(eqclasses, attributes, verbose=True):
    if verbose:
        return tuple(itertools.chain.from_iterable(
            (classify_by_attributes(representatives, attributes, verbose=False) for representatives in progressbar.progressbar(
                eqclasses, widgets=[progressbar.SimpleProgress(), progressbar.Bar(), ' (', progressbar.ETA(), ') ']))))
    else:
        return tuple(itertools.chain.from_iterable((classify_by_attributes(representatives, attributes, verbose=False) for representatives in eqclasses)))


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
        return [DirectedStructure(edge_list, self.n) for r
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
        # return sorted((Hypergraph([frozenset({v}) for v in self.set_n.difference(*sc)] + sc, self.n) for sc in list_of_simplicial_complices), key = lambda sc:sc.tally)
        return sorted((Hypergraph(sc, self.n) for sc in list_of_simplicial_complices), key=lambda sc: sc.tally)

    @cached_property
    def all_labelled_mDAGs(self):
        return [mDAG(ds, sc) for sc in self.all_simplicial_complices for ds in self.all_directed_structures]

    @cached_property
    def dict_id_to_canonical_id(self):
        print("Computing canonical (unlabelled) graphs...", flush=True)
        return {mDAG.unique_id: mDAG.unique_unlabelled_id for mDAG in self.all_labelled_mDAGs}


    def mdag_int_pair_to_single_int(self, sc_int, ds_int):
        #ELIE: Note that this MUST MATCH the function mdag_to_int in the mDAG class. All is good.
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
    def all_unlabelled_mDAGs_faster(self):
        d = defaultdict(list)
        for ds in self.all_unlabelled_directed_structures:
            for sc in self.all_simplicial_complices:
                mdag = mDAG(ds, sc)
                d[mdag.unique_unlabelled_id].append(mdag)
        return tuple(next(iter(eqclass)) for eqclass in d.values())

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

    def FaceSplitting_edges(self):
        for (id, mDAG) in self.dict_ind_unlabelled_mDAGs.items():  # NEW: Over graph patterns only
            for mDAG2 in mDAG.generate_weaker_mDAGs_FaceSplitting('strong'):
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
        g.add_edges_from(self.FaceSplitting_edges())
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
    def foundational_eqclasses_picklist(self):
        return [all(mDAG.fundamental_graphQ for mDAG in eqclass) for eqclass in self.equivalence_classes_as_mDAGs]

    @cached_property
    def foundational_eqclasses(self):
        # return list(filter(lambda eqclass: all(mDAG.fundamental_graphQ for mDAG in eqclass),
        #                    self.equivalence_classes_as_mDAGs))
        return [eqclass for eqclass,foundational_Q in
                zip(self.equivalence_classes_as_mDAGs,self.foundational_eqclasses_picklist) if foundational_Q]

    @staticmethod
    def representatives(eqclasses):
        return [next(iter(eqclass)) for eqclass in eqclasses]

    @staticmethod
    def smart_representatives(eqclasses, attribute):
        return [min(eqclass, key = lambda mDAG: mDAG.__getattribute__(attribute)) for eqclass in eqclasses]

    @cached_property
    def representative_mDAGs_not_necessarily_foundational(self):
    #     return self.representatives(self.equivalence_classes_as_mDAGs)
        return self.smart_representatives(self.equivalence_classes_as_mDAGs, 'relative_complexity_for_sat_solver')

    @cached_property
    def representatives_not_even_one_fundamental_in_class(self):
        representatives_of_NOT_entirely_fundamental_classes=[element for element in self.representative_mDAGs_not_necessarily_foundational if element not in self.representative_mDAGs_list]   
        not_even_one_fundamental_in_class=[]
        for mDAG in representatives_of_NOT_entirely_fundamental_classes:
            for c in self.equivalence_classes_as_mDAGs:
                if mDAG in c:
                    not_one_fund=True
                    for equivalent_mDAG in c:
                        if equivalent_mDAG.fundamental_graphQ:
                            not_one_fund=False
                    if not_one_fund:
                        not_even_one_fundamental_in_class.append(mDAG)
        return not_even_one_fundamental_in_class


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
    def Dense_connectedness_classes(self):
        return classify_by_attributes(self.representative_mDAGs_list, ['all_densely_connected_pairs_unlabelled'])
    
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
    def Dense_connectedness_and_Skeleton(self):
        return classify_by_attributes(self.representative_mDAGs_list,
                                           ['all_densely_connected_pairs_unlabelled', 'skeleton_unlabelled'])
    @property
    def Dense_connectedness_and_CI(self):
        return classify_by_attributes(self.representative_mDAGs_list,
                                           ['all_densely_connected_pairs_unlabelled', 'all_CI_unlabelled'])
    @property
    def Dense_connectedness_and_esep(self):
        return classify_by_attributes(self.representative_mDAGs_list,
                                           ['all_densely_connected_pairs_unlabelled', 'all_esep_unlabelled'])


    @property
    def representatives_for_only_hypergraphs(self):
        #choosing representatives with the smaller number of edges will guarantee that Hypergraph-only mDAGs are chosen
        return self.smart_representatives(self.equivalence_classes_as_mDAGs, 'n_of_edges')

    @cached_property
    def equivalent_to_only_hypergraph_representative(self):
        # #diagnostic:
        # for mDAG_by_hypergraph, mDAG_by_complexity in zip(
        #         self.representatives_for_only_hypergraphs, self.representative_mDAGs_list):
        #     if mDAG_by_hypergraph != mDAG_by_complexity and mDAG_by_hypergraph.n_of_edges==0:
        #         print(mDAG_by_hypergraph, " =!= ", mDAG_by_complexity)
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

        print("Number of unlabelled graph patterns: ", len(self.all_unlabelled_mDAGs), flush=True)
        fundamental_list = [mDAG.fundamental_graphQ for mDAG in self.all_unlabelled_mDAGs]
        print("Number of fundamental unlabelled graph patterns: ", len(np.flatnonzero(fundamental_list)), flush=True)
        print("Upper bound on number of equivalence classes: ", len(self.equivalence_classes_as_mDAGs), flush=True)
        print("Upper bound on number of 100% foundational equivalence classes: ", len(self.foundational_eqclasses),
              flush=True)
        print("Number of Foundational CI classes: ", len(self.CI_classes))


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
        
        
        self.singletons_dict = dict({1: list(itertools.chain.from_iterable(
            filter(lambda eqclass: (len(eqclass) == 1 or self.effectively_all_singletons(eqclass)),
                   self.Dense_connectedness_and_esep)))})
        self.non_singletons_dict = dict({1: sorted(
            filter(lambda eqclass: (len(eqclass) > 1 and not self.effectively_all_singletons(eqclass)),
                   self.Dense_connectedness_and_esep), key=len)})
        print("# of singleton classes from also considering Dense Connectedness: ", len(self.singletons_dict[1]))
        print("# of non-singleton classes from also considering Dense Connectedness: ", self.lowerbound_count_accounting_for_hypergraph_inequivalence(self.non_singletons_dict[1]),
              ", comprising {} total foundational graph patterns (no repetitions)".format(
                  ilen(itertools.chain.from_iterable(self.non_singletons_dict[1]))))

        smart_supports_dict = dict()
        for k in range(2, self.max_nof_events + 1):
            print("[Working on nof_events={}]".format(k))
            # I changed the line below
            smart_supports_dict[k] = further_classify_by_attributes(self.non_singletons_dict[k - 1],
                                                            [('infeasible_binary_supports_n_events_beyond_esep_unlabelled',
                                                              k)], verbose=True)
            self.singletons_dict[k] = list(itertools.chain.from_iterable(
                filter(lambda eqclass: (len(eqclass) == 1 or self.effectively_all_singletons(eqclass)),
                       smart_supports_dict[k]))) + self.singletons_dict[k - 1]
            self.non_singletons_dict[k] = sorted(
                filter(lambda eqclass: (len(eqclass) > 1 and not self.effectively_all_singletons(eqclass)),
                       smart_supports_dict[k]), key=len)
            print("# of singleton classes from also considering Supports Up To {}: ".format(k), len(self.singletons_dict[k]))
            print("# of non-singleton classes from also considering Supports Up To {}: ".format(k), self.lowerbound_count_accounting_for_hypergraph_inequivalence(self.non_singletons_dict[k]),
                  ", comprising {} total foundational graph patterns".format(
                      ilen(itertools.chain.from_iterable(self.non_singletons_dict[k]))))     
   

if __name__ == '__main__':
    # Observable_mDAGs2 = Observable_mDAGs_Analysis(nof_observed_variables=2, max_nof_events_for_supports=0)
    Observable_mDAGs3 = Observable_mDAGs_Analysis(nof_observed_variables=3, max_nof_events_for_supports=0)
    Observable_mDAGs4 = Observable_mDAGs_Analysis(nof_observed_variables=4, max_nof_events_for_supports=0)

# =============================================================================
#     for i in range(len(Observable_mDAGs4.foundational_eqclasses)):
#          print(Observable_mDAGs4.foundational_eqclasses[i][0])
# 
#     G_Bell=mDAG(DirectedStructure([(0,3),(1,2)],4),Hypergraph([(0,),(1,),(2,3)],4))
#     import networkx as nx
#     G_Bell_nx = nx.DiGraph()
#     G_Bell_nx.add_nodes_from(G_Bell.visible_nodes)
#     l=len(G_Bell.visible_nodes)
#     for facet in G_Bell.simplicial_complex_instance.compressed_simplicial_complex:
#         G_Bell_nx.add_node(l)
#         for node in facet:
#             G_Bell_nx.add_edge(l,node)
#     nx.draw(G_Bell_nx)
#         
#     G_Bell_nx.add_nodes_from(G_Bell.latent_nodes)
#     G.add_edges_from([(1, 2), (1, 3)])
# =============================================================================
    


# =============================================================================
#     two_latents=[]
#     for i in range(len(Observable_mDAGs4.foundational_eqclasses)):
#         for mDAG in Observable_mDAGs4.equivalence_classes_as_mDAGs[i]:
#             if len(mDAG.latent_nodes)==2 or len(mDAG.latent_nodes)==1:
#                 two_latents.append(mDAG)
#                 break
#     len(two_latents)
#     
# 
#                 
#     G_Bell=mDAG(DirectedStructure([(0,3),(1,2)],4),Hypergraph([(0,),(1,),(2,3)],4))
#     G_HLP=mDAG(DirectedStructure([(0,1),(1,3),(2,3)],4),Hypergraph([(1,2),(2,3)],4))
# 
# 
#     (edge_list,simplicial_complex,variable_names)=G_HLP.marginalize_node(2)
#     marginalized_G_HLP=mDAG(LabelledDirectedStructure(variable_names,edge_list),LabelledHypergraph(variable_names,simplicial_complex))
#     
#     G_Instrumental=mDAG(DirectedStructure( [(0,1),(1,2)],3),Hypergraph([(1,2)],3))
#     for eqclass in Observable_mDAGs3.foundational_eqclasses:
#         if G_Instrumental in eqclass:
#             Instrumental_class=eqclass
# 
#     G= mDAG(DirectedStructure([(0,1),(1,2),(1,3)],4),Hypergraph([(0,1),(0,2),(2,3),(1,3)],4))
#     G.marginalize_node(2)    
#     
#     #APPLYING MARGINALIZATION TO REDUCE TO INSTRUMENTAL:    
#     mDAGs_that_reduce_to_Instrumental=[]
#     for representative_mDAG in Observable_mDAGs4.representative_mDAGs_list:
#         for marginalized_node in representative_mDAG.visible_nodes:
#             (d,s,variable_names)=representative_mDAG.marginalize_node(marginalized_node)
#             #analyzing if the marginalized mDAG is equivalent to Instrumental
#             if len(s)==1 and len(d)==2 and set(s).issubset(d) and s[0][0]==list(set(d)-set(s))[0][1]:
#                 mDAGs_that_reduce_to_Instrumental.append((representative_mDAG,marginalized_node))
#                 break
#             #alternatively, looking if it is equivalent to one of the other mDAGs in the instrumental equivalence class
#             elif len(s)==2 and len(d)==1 and set(d).issubset(s) and d[0][0]==list(set(s)-set(d))[0][1]:
#                 mDAGs_that_reduce_to_Instrumental.append((representative_mDAG,marginalized_node))
#                 break
#             elif len(s)==len(d)==2 and [set(ele) for ele in s]==[set(ele) for ele in d] and any((d[0][1]==d[1][0],d[1][1]==d[0][0])):
#                 mDAGs_that_reduce_to_Instrumental.append((representative_mDAG,marginalized_node))
#                 break
# 
#     len(mDAGs_that_reduce_to_Instrumental)
#     
#     len(Observable_mDAGs4.foundational_eqclasses)
#   
#   
#     quantumly_valid=[]
#     for (mDAG, marginalized_node) in mDAGs_that_reduce_to_Instrumental:
#         X_contains_latent_parents=False
#         for facet in mDAG.simplicial_complex_instance.simplicial_complex_as_sets:
#             if marginalized_node in facet:
#                 X_contains_latent_parents=True
#                 break
#         share_facet_with_children=[]
#         if X_contains_latent_parents:
#             for c in mDAG.children(marginalized_node):
#                 X_share_facet_with_c=False
#                 for facet in mDAG.simplicial_complex_instance.simplicial_complex_as_sets:
#                     if marginalized_node in facet and c in facet:
#                         X_share_facet_with_c=True
#                 share_facet_with_children.append(X_share_facet_with_c)
#         if X_contains_latent_parents==False or all(share_facet_with_children):
#             quantumly_valid.append((mDAG, marginalized_node))
#                 
#     len(quantumly_valid)
#         
#     not_quantumly_valid=list(set(mDAGs_that_reduce_to_Instrumental)-set(quantumly_valid))
#     
#     for (G,node) in not_quantumly_valid:
#         print("G=",G)
#         print("node=",node)
#         print(G.marginalize_node(3))
#         
#     no_children=[]
#     for (G,node) in mDAGs_that_reduce_to_Instrumental:
#         if len(G.children(node))==0:
#             no_children.append((G,node))
#             
#    with_children=list(set(mDAGs_that_reduce_to_Instrumental)-set(no_children))
# 
#     graph_eqgraph_eqnode=[]
#     for (G,node) in no_children:
#         for eqclass in Observable_mDAGs4.foundational_eqclasses:
#             if G in eqclass:
#                 eq_to_G=[G,node]
#                 for graph in eqclass:
#                     found_node=False
#                     for marginalized_node in graph.visible_nodes:
#                         if len(graph.children(marginalized_node))!=0:
#                             (d,s,variable_names)=graph.marginalize_node(marginalized_node)
#                             #analyzing if the marginalized mDAG is equivalent to Instrumental
#                             if len(s)==1 and len(d)==2 and set(s).issubset(d) and s[0][0]==list(set(d)-set(s))[0][1]:
#                                 eq_to_G.append((graph,marginalized_node))
#                                 break
#                             #alternatively, looking if it is equivalent to one of the other mDAGs in the instrumental equivalence class
#                             elif len(s)==2 and len(d)==1 and set(d).issubset(s) and d[0][0]==list(set(s)-set(d))[0][1]:
#                                 eq_to_G.append((graph,marginalized_node))
#                                 break
#                             elif len(s)==len(d)==2 and [set(ele) for ele in s]==[set(ele) for ele in d] and any((d[0][1]==d[1][0],d[1][1]==d[0][0])):
#                                  eq_to_G.append((graph,marginalized_node))
#                                  break
#                 if len(eq_to_G)>0:
#                     graph_eqgraph_eqnode.append(eq_to_G)
#                         
#                             
#                             len(graph_eqgraph_eqnode)
#     l=[]
#     for ele in graph_eqgraph_eqnode:
#         if len(ele)>2:
#             l.append(ele)
# len(l)
# 
# l[0]
# 
#     
#     G= mDAG(DirectedStructure([(1,2),(2,3)],4),Hypergraph([(0,2),(1,2),(2,3)],4))    
#     G.plot_mDAG_indicating_one_node(1)
#     
# 
#     G_HLP=mDAG(DirectedStructure([(0,1),(1,3),(2,3)],4),Hypergraph([(1,2),(2,3)],4))
#     G_HLP.networkx_plot_mDAG()
#     G_HLP.latent_nodes 
# =============================================================================
    
    
# =============================================================================
#     n=0
#     for ci_class in Observable_mDAGs4.esep_classes:
#         if len(ci_class)==1:
#             n=n+1
#     print(n)
# =============================================================================

# =============================================================================
#     G_Instr=mDAG(DirectedStructure([(1,2)],3),Hypergraph([(0,1),(1,2)],3))
#     G_UC=mDAG(DirectedStructure([(0,1),(0,2)],3),Hypergraph([(0,1),(0,2)],3))
#     print("G_Instr=",G_Instr)
#     print("G_UC=",G_UC)
#     print("Same esep=",G_Instr.all_esep_unlabelled== G_UC.all_esep_unlabelled)
#     for k in range(2,9):
#       print("Same Supports at",k,"events=",G_Instr.infeasible_binary_supports_n_events_unlabelled(k)==G_UC.infeasible_binary_supports_n_events_unlabelled(k))
#       print("Same Smart Supports at",k,"events=",G_Instr.infeasible_binary_supports_n_events_beyond_esep_unlabelled(k)==G_UC.infeasible_binary_supports_n_events_beyond_esep_unlabelled(k))
# =============================================================================



# =============================================================================
#     G1=mDAG(DirectedStructure([(2,0),(1,3)],4),Hypergraph([(0,1),(1,2),(2,3)],4))
#     G1.all_densely_connected_pairs_unlabelled
#     not_all_densely_connected=[]
#     for G in Observable_mDAGs4.representative_mDAGs_not_necessarily_foundational:
#         if G.all_densely_connected_pairs_unlabelled != G1.all_densely_connected_pairs_unlabelled:
#             not_all_densely_connected.append(G)
#             print(G.all_densely_connected_pairs_unlabelled)
#           
#     print("Number of mDAGs that have at least one pair of nodes that are not densely connected=",len(not_all_densely_connected))
# =============================================================================
  
  
# =============================================================================
#     i=0
#     for ob_class in Observable_mDAGs4.foundational_eqclasses:
#         p1=ob_class[0].all_densely_connected_pairs_unlabelled
#         for mDAG in ob_class:
#             if mDAG.all_densely_connected_pairs_unlabelled!=p1:
#                 print(i)
#         i=i+1
# No problem here: all classes are consistent in terms of Dense Connectedness
# =============================================================================

 
        
# =============================================================================
#     Skeleton_classes4=Observable_mDAGs4.Skeleton_classes
#     print("Number of Skeleton classes=",len(Skeleton_classes4))
#     CI_classes4=Observable_mDAGs4.CI_classes
#     print("Number of CI classes=",len(CI_classes4))
#     esep_classes4=Observable_mDAGs4.esep_classes
#     print("Number of esep classes=",len(esep_classes4))
#     Dense_con_classes=Observable_mDAGs4.Dense_connectedness_classes
#     print("Number of Dense Conectedness Classes=",len(Dense_con_classes))
#     
#     print("Skeleton+CI=",len(Observable_mDAGs4.Skeleton_and_CI))
#     print("Skeleton+esep=",len(Observable_mDAGs4.Skeleton_and_esep))
#     print("Dense Conectedness+Skeleton=",len(Observable_mDAGs4.Dense_connectedness_and_Skeleton))
#     print("Dense Conectedness+CI=",len(Observable_mDAGs4.Dense_connectedness_and_CI))
#     print("Dense Conectedness+esep=",len(Observable_mDAGs4.Dense_connectedness_and_esep))
# 
#     G_Pgraph=mDAG(DirectedStructure([(0,1),(1,2),(2,3)],5),Hypergraph([(0,),(1,3,4),(2,4)],5))
#     G_Pgraph.are_densely_connected(0,3)    
#     G_Pgraph.all_esep
#     G_Pgraph.fundamental_graphQ
#     G_modified_Pgraph=mDAG(DirectedStructure([(0,1),(1,2),(2,3),(1,3)],5),Hypergraph([(0,),(1,3,4),(2,4)],5))
#     G_Pgraph.all_esep==G_modified_Pgraph.all_esep
#     G_Pgraph4=mDAG(DirectedStructure([(0,1),(1,2),(2,3)],4),Hypergraph([(0,),(1,3),(2,)],4))
#     G_Pgraph4.all_esep
# =============================================================================
    
# ==================================

# for k in tuple(Observable_mDAGs.singletons_dict.keys())[1:]:
    #     print("mDAGs freshly recognized as singleton-classes from supports over ",k," events:")
    #     for i in Observable_mDAGs.singletons_dict[k]:
    #         if i not in Observable_mDAGs.singletons_dict[k-1]:
    #             print(i)

    # metagraph_adjmat = nx.to_numpy_array(Observable_mDAGs.meta_graph, nodelist=list(
    #     itertools.chain.from_iterable(Observable_mDAGs.equivalence_classes_as_ids)), dtype=bool)
    # for_codewords = np.add.accumulate(list(map(len,Observable_mDAGs.equivalence_classes_as_ids)))
    # codewords = itertools.starmap(np.arange,zip(np.hstack((0,for_codewords)), for_codewords))
    # #codewords = [np.arange(i1,i2) for i1,i2 in zip(np.hstack((0, for_codewords)), for_codewords)]
    # codewords = [codeword for codeword,foundational_Q in zip(codewords,Observable_mDAGs.foundational_eqclasses_picklist) if foundational_Q]
    # eqclass_adjmat = np.empty(np.broadcast_to(len(Observable_mDAGs.foundational_eqclasses),2), dtype=bool)
    # for i,c1 in enumerate(codewords):
    #     for j,c2 in enumerate(codewords):
    #         # eqclass_adjmat[i,j] = metagraph_adjmat[(c1,c2)].any()
    #         eqclass_adjmat[i,j] = metagraph_adjmat[c1].any(axis=0)[c2].any()
    # print(eqclass_adjmat.astype(int))


    
# =============================================================================
#     no_inf_sups_beyond_esep4=[]
#     i=0
#     for mDAG in Observable_mDAGs4.representative_mDAGs_list:
#      print(i,"of",len(Observable_mDAGs4.representative_mDAGs_list))
#      i=i+1
#      if mDAG.no_infeasible_binary_supports_beyond_esep_up_to(5):
#          no_inf_sups_beyond_esep4.append(mDAG)
#     
#     len(no_inf_sups_beyond_esep4)
#     
# 
#     no_inf_sups_beyond_esep3=[]
#     i=0
#     for mDAG in Observable_mDAGs3.representative_mDAGs_list:
#      print(i,"of",len(Observable_mDAGs3.representative_mDAGs_list))
#      i=i+1
#      if mDAG.no_infeasible_binary_supports_beyond_esep_up_to(5):
#          no_inf_sups_beyond_esep3.append(mDAG)
#     
#     len(no_inf_sups_beyond_esep3)
# =============================================================================
               
    
  
# =============================================================================
#     for k in tuple(Observable_mDAGs.singletons_dict.keys())[1:]:
#         print("mDAGs freshly recognized as singleton-classes from supports over ",k," events:")
#         for i in Observable_mDAGs.singletons_dict[k]:
#             if i not in Observable_mDAGs.singletons_dict[k-1]:
#                 print(i)
# =============================================================================


# =============================================================================
#     for i in Observable_mDAGs4.CI_classes:
#         if len(i)==1:
#              print("class of", i)
#              for c in Observable_mDAGs.foundational_eqclasses:
#                  for i0 in i:
#                      if i0 in c:
#                          print(c)
# =============================================================================
         

# =============================================================================
#     metagraph_adjmat = nx.to_numpy_array(Observable_mDAGs.meta_graph, nodelist=list(
#         itertools.chain.from_iterable(Observable_mDAGs.equivalence_classes_as_ids)), dtype=bool)
#     for_codewords = np.add.accumulate(list(map(len,Observable_mDAGs.equivalence_classes_as_ids)))
#     codewords = itertools.starmap(np.arange,zip(np.hstack((0,for_codewords)), for_codewords))
#     #codewords = [np.arange(i1,i2) for i1,i2 in zip(np.hstack((0, for_codewords)), for_codewords)]
#     codewords = [codeword for codeword,foundational_Q in zip(codewords,Observable_mDAGs.foundational_eqclasses_picklist) if foundational_Q]
#     eqclass_adjmat = np.empty(np.broadcast_to(len(Observable_mDAGs.foundational_eqclasses),2), dtype=bool)
#     for i,c1 in enumerate(codewords):
#         for j,c2 in enumerate(codewords):
#             # eqclass_adjmat[i,j] = metagraph_adjmat[(c1,c2)].any()
#             eqclass_adjmat[i,j] = metagraph_adjmat[c1].any(axis=0)[c2].any()
#     print(eqclass_adjmat.astype(int))
# 
#     [mDAG for mDAG in Observable_mDAGs.singletons_dict[3] if mDAG.no_infeasible_binary_supports_up_to(4)]
# 
#    [non_singleton_set for non_singleton_set in Observable_mDAGs.non_singletons_dict[3] if next(iter(non_singleton_set)).support_testing_instance_binary(5).no_infeasible_supports()]
# 
#     next(iter([non_singleton_set for non_singleton_set in Observable_mDAGs.non_singletons_dict[3] if next(iter(non_singleton_set)).no_infeasible_binary_supports_up_to(3)]))
# 
# =============================================================================