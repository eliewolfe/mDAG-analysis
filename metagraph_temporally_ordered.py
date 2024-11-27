from __future__ import absolute_import
import networkx as nx
import numpy as np
import itertools
from more_itertools import ilen
import progressbar
from utilities import partsextractor
from collections import defaultdict
from scipy import sparse

from sys import hexversion, stderr
import gc

if hexversion >= 0x3080000:
    from functools import cached_property
elif hexversion >= 0x3060000:
    from backports.cached_property import cached_property
else:
    cached_property = property

from hypergraphs import Hypergraph
from directed_structures import DirectedStructure
from mDAG_advanced import mDAG
from supports import explore_candidates

def eprint(*args, **kwargs):
    print(*args, file=stderr, **kwargs)


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
            for mdag in progressbar.progressbar(
                representatives, widgets=[progressbar.SimpleProgress(), progressbar.Bar(), ' (', progressbar.ETA(), ') ']):
                d[tuple(evaluate_property_or_method(mdag, prop) for prop in attributes)].add(mdag)
        else:
            for mdag in representatives:
                d[tuple(evaluate_property_or_method(mdag, prop) for prop in attributes)].add(mdag)
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


class Metagraph_temporally_ordered_mDAGs:
    def __init__(self, n: int, temporally_ordered,
                 verbose=True):
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
        self.verbose = verbose
        self.temporally_ordered=temporally_ordered
        
    @cached_property
    def ds_bit_size(self):
        return np.asarray(2**(self.n**2), dtype=object)

    def mdag_int_pair_to_single_int(self, ds_int, sc_int):
        #ELIE: Note that this MUST MATCH the function mdag_int_pair_to_single_int in the mDAG class. All is good.
        return ds_int + sc_int*self.ds_bit_size
        
    @cached_property
    def all_temporally_ordered_directed_structures(self):
        possible_node_pairs = list(itertools.combinations(self.tuple_n, 2))
        return tuple(DirectedStructure(edge_list, self.n) for r
                in range(0, len(possible_node_pairs)+1) for edge_list
                in itertools.combinations(possible_node_pairs, r))
    
    @cached_property
    def truly_all_directed_structures(self):
        truly_all = []
        for ds in self.all_temporally_ordered_directed_structures:
            truly_all.extend(ds.equivalents_under_symmetry)
        truly_all_no_repetitions=[]
        for ds in truly_all:
            if ds.edge_list not in [ds1.edge_list for ds1 in truly_all_no_repetitions]:
                truly_all_no_repetitions.append(ds)
        return truly_all_no_repetitions
    
    @cached_property
    def all_simplicial_complices(self):
        list_of_simplicial_complices = [[]]
        for ch in map(frozenset, itertools.chain.from_iterable(itertools.combinations(self.tuple_n, i) for i in range(2, self.n + 1))):
            list_of_simplicial_complices.extend(
                [sc + [ch] for sc in list_of_simplicial_complices if (not any((ch.issubset(ch_list) or ch_list.issubset(ch)) for ch_list in sc))])
            #At this stage we have all the *compressed* simplicial complices.
        return tuple(sorted((Hypergraph(sc, self.n) for sc in list_of_simplicial_complices), key=lambda sc: sc.tally))
    
    @cached_property
    def all_temporally_ordered_mDAGs(self):
        return [mDAG(ds, sc) for sc, ds in itertools.product(self.all_simplicial_complices, self.all_temporally_ordered_directed_structures)]
    
    @cached_property
    def truly_all_mDAGs(self):
        return [mDAG(ds, sc) for sc, ds in itertools.product(self.all_simplicial_complices,self.truly_all_directed_structures)]
    
    @cached_property
    def dict_ind_temporally_ordered_mDAGs(self):   # Dictionary from indices to temporally ordered mDAGs
        return {mdag.unique_id: mdag for mdag in self.all_temporally_ordered_mDAGs}
    
    @cached_property
    def dict_ind_mDAGs(self):   # Dictionary from indices to all mDAGs
        return {mdag.unique_id: mdag for mdag in self.truly_all_mDAGs}
    
    def lookup_mDAG(self, indices):   # Getting an mDAG from an index
        return partsextractor(self.dict_ind_mDAGs, indices)
    
    @cached_property
    def metagraph_nodes(self):    # This metagraph will be a graph whose nodes are all of the mDAGs
        return np.array(tuple(self.dict_ind_mDAGs.keys()))
    
    @cached_property
    def hypergraph_dominances(self):   
        return tuple((S1.as_integer, S2.as_integer) for S1, S2 in itertools.permutations(self.all_simplicial_complices, 2) if
                S1.can_S1_minimally_simulate_S2(S2))
    
    @cached_property
    def directed_dominances(self):
        return tuple((D1.as_integer, D2.as_integer) for D1, D2 in itertools.permutations(self.truly_all_directed_structures, 2) if
                D1.can_D1_minimally_simulate_D2(D2))
    
    @property
    def raw_HLP_edges(self):
        for (id1, mdag1) in explore_candidates(
                self.dict_ind_mDAGs.items(),
                verbose=self.verbose,
                message="Finding HLP Rule #4 type edges"):
            for id2 in mdag1.generate_weaker_mDAG_HLP:
                yield (id2, id1)

    @property
    def raw_FaceSplitting_edges(self):
        for (id1, mdag1) in explore_candidates(
                self.dict_ind_mDAGs.items(),
                verbose=self.verbose,
                message="Finding metagraph edges due to face splitting"):  
            for id2 in mdag1.generate_weaker_mDAGs_FaceSplitting('strong'):
                yield (id2, id1)
                
    @cached_property
    def all_dominances(self):
        directed_ids = np.array(tuple(ds.as_integer for ds in self.truly_all_directed_structures), dtype=object)[:,np.newaxis,np.newaxis]
        hypergraph_dominances = np.asarray(self.hypergraph_dominances, dtype=object)
        dominances_from_hypergraphs = np.reshape(self.mdag_int_pair_to_single_int(directed_ids, hypergraph_dominances),(-1,2))
        simplicial_ids = np.array(tuple(sc.as_integer for sc in self.all_simplicial_complices), dtype=object)[:,np.newaxis,np.newaxis]
        directed_dominances = np.asarray(self.directed_dominances, dtype=object)
        dominances_from_directed_edges = np.reshape(self.mdag_int_pair_to_single_int(directed_dominances, simplicial_ids),(-1,2))
        HLP_dominances = np.asarray(tuple(self.raw_HLP_edges), dtype=object)
        FaceSplitting_dominances = np.asarray(tuple(self.raw_FaceSplitting_edges), dtype=object)              
        return (list(dominances_from_hypergraphs) + list(dominances_from_directed_edges)+
                list(HLP_dominances)+list(FaceSplitting_dominances))
    
    @cached_property
    def metagraph(self):   # Graph where the nodes are (truly) all mDAGs.
        g = nx.DiGraph()
        g.add_nodes_from(self.metagraph_nodes.flat)
        g.add_edges_from(self.all_dominances)
        gc.collect(generation=2)
        return g
    
# =============================================================================
#     def partial_order_of_subset(self, subset):   #subset is a set of mDAG ids
#         g=self.metagraph.copy()
#         for n in self.metagraph.nodes:
#             if n not in subset:
#                 for p in self.metagraph.predecessors(n):
#                     for c in self.metagraph.successors(n):
#                         g.add_edge(p,c)
#         for n in self.metagraph.nodes:
#             if n not in subset:
#                 g.remove_node(n)
#         return g
#     
#     def metagraph_of_temporally_ordered_mDAGs(self):
#         temporally_ordered=self.dict_ind_temporally_ordered_mDAGs.keys()
#         return self.partial_order_of_subset(temporally_ordered)

#   PROBLEMATIC: There were dominances missing when I tried to get the equivalence classes of temporally ordered mDAGs in this way
# =============================================================================
  
    @cached_property
    def equivalence_classes_as_ids(self):     # Here, each "equivalence class" is actually a block of the proven-equivalence partition of mDAGs according to all of the known observational equivalence rules
        eq_classes_all_mDAGs=list(nx.strongly_connected_components(self.metagraph))    
        if not self.temporally_ordered:
            return eq_classes_all_mDAGs
        if self.temporally_ordered:
            temporally_ordered=set(self.dict_ind_temporally_ordered_mDAGs.keys())  
            return [s & temporally_ordered for s in eq_classes_all_mDAGs if s & temporally_ordered]  # Only the temporally ordered mDAGs

    @cached_property
    def equivalence_classes_as_mDAGs(self):    
        return [self.lookup_mDAG(eqclass) for eqclass in self.equivalence_classes_as_ids]
    
    def eqclass_of_mDAG(self, mDAG):
        for eqclass in self.equivalence_classes_as_mDAGs:
            if mDAG in eqclass:
                return eqclass
    
class Proven_Inequivalence_Partition_Analysis(Metagraph_temporally_ordered_mDAGs):
    def __init__(self, nof_observed_variables, temporally_ordered,
                 verbose=True):
        super().__init__(nof_observed_variables, temporally_ordered)   
        
    
    @staticmethod
    def smart_representatives(eqclasses, attribute):    # Picking one representative of each equivalence class according to some attribute
        return [min(eqclass, key = lambda mdag: mdag.__getattribute__(attribute)) for eqclass in eqclasses]

    @cached_property
    def representative_mDAGs_list(self):  # For each equivalence class, the representative is the one that performs better with the SAT solver (relevant for supports)
        return self.smart_representatives(self.equivalence_classes_as_mDAGs,
                                              'relative_complexity_for_sat_solver') 
    @property
    def CI_partition(self):   # Classifying the representative of each equivalence class into the proven-inequivalence partition according to d-separation
        return classify_by_attributes(self.representative_mDAGs_list, ['all_CI'])

    @property
    def Skeleton_partition(self):
        return classify_by_attributes(self.representative_mDAGs_list, ['skeleton'])

    @property
    def Skeleton_and_CI_partition(self):
        return classify_by_attributes(self.representative_mDAGs_list, ['skeleton', 'all_CI'])
    
    @property
    def esep_partition(self):
        return classify_by_attributes(self.representative_mDAGs_list, ['all_esep'])
 
    @property
    def Dense_connectedness_and_esep_partition(self):
        return classify_by_attributes(self.representative_mDAGs_list,
                                           ['all_densely_connected_pairs_numeric', 'all_esep'])
    
    @cached_property
    def representatives_equivalent_to_confounder_free(self):  # Dictionary that says which representatives are equivalent to a confounder free mDAG.
        representatives_with_smaller_number_of_latents=self.smart_representatives(self.equivalence_classes_as_mDAGs,'n_of_latents')
        return {mDAG_by_complexity: mDAG_by_n_latents.n_of_latents==0 for
                mDAG_by_n_latents, mDAG_by_complexity in
                zip(representatives_with_smaller_number_of_latents, self.representative_mDAGs_list)}

    @cached_property
    def representatives_equivalent_to_directed_edge_free(self):  # Dictionary that says which representatives are equivalent to a directed edge free mDAG.
        representatives_with_smaller_number_of_edges=self.smart_representatives(self.equivalence_classes_as_mDAGs,'n_of_edges')
        return {mDAG_by_complexity: mDAG_by_n_edges.n_of_edges==0 for
                mDAG_by_n_edges, mDAG_by_complexity in
                zip(representatives_with_smaller_number_of_edges, self.representative_mDAGs_list)}

    def effectively_solved_class_by_directed_edge_free(self, proven_inequivalence_block):   # Returns True if all of the elements of a given proven inequivalence block include one directed edge free mDAG. This means that each element is a solved equivalence class. 
        return all(self.representatives_equivalent_to_directed_edge_free[mdag] for mdag in proven_inequivalence_block)
    
    def completely_solved(self, proven_inequivalence_partition):  # Returns the list of representatives of equivalence classes that are completely solved by some rule, defined by the starting proven_inequivalence_partition
        return list(map(lambda x: {x}, list(itertools.chain.from_iterable(
                filter(lambda block: (len(block) == 1),proven_inequivalence_partition)))))
    
    def completely_solved_with_directededgefree_rule(self, proven_inequivalence_partition):  # Returns the list of representatives of equivalence classes that are completely solved by the directed-edge-free rule together with another rule, defined by the starting proven_inequivalence_partition
        return list(map(lambda x: {x}, list(itertools.chain.from_iterable(
                filter(lambda block: (len(block) == 1 or self.effectively_solved_class_by_directed_edge_free(block)),
                       proven_inequivalence_partition))))) 
    
    def not_completely_solved_with_directededgefree_rule(self, proven_inequivalence_partition):       
        return sorted(
            filter(lambda block: (len(block) > 1 and not self.effectively_solved_class_by_directed_edge_free(block)),
                   proven_inequivalence_partition), key=len)

    def supports_analysis(self,starting_proven_inequivalence_partition,max_nof_events):
        starting_solved_with_directededgefree_rule=self.completely_solved_with_directededgefree_rule(starting_proven_inequivalence_partition)
        starting_not_solved_with_directededgefree_rule=self.not_completely_solved_with_directededgefree_rule(starting_proven_inequivalence_partition)
        solved_dict=dict({1: starting_solved_with_directededgefree_rule})
        not_solved_dict=dict({1: starting_not_solved_with_directededgefree_rule})
        proven_inequivalence_partition_dict = dict({1: starting_solved_with_directededgefree_rule+starting_not_solved_with_directededgefree_rule})  # Dictionary where each key will be the maximum number of events for which the supports were tested
        for k in range(2, max_nof_events + 1):
            print("[Working on nof_events={}]".format(k))
            partition_blocks_previously_not_solved=list(further_classify_by_attributes(not_solved_dict[k - 1],
                            [('infeasible_binary_supports_n_events_beyond_esep',k)], verbose=True))  # Check the supports of k events only on the blocks that were not solved by supports of k-1 events
            proven_inequivalence_partition_dict[k] = partition_blocks_previously_not_solved + solved_dict[k-1]  
            solved_dict[k] = self.completely_solved_with_directededgefree_rule(partition_blocks_previously_not_solved) + solved_dict[k - 1]
            not_solved_dict[k] = self.not_completely_solved_with_directededgefree_rule(partition_blocks_previously_not_solved)
        return (proven_inequivalence_partition_dict, solved_dict)

    def supports_proven_inequivalence_partition_dict(self,max_nof_events):  # Returns a dictionary where the keys are the number of events of the supports tested, and each value is the proven-inequivalence partition obtained from support test up to that number of events
        (proven_inequivalence_partition_dict, solved_dict) = self.supports_analysis(self.Dense_connectedness_and_esep_partition,max_nof_events)   # This includes dense connectedness, e-separation and directed-edge-only rules
        return proven_inequivalence_partition_dict
    
    def supports_solved_classes_dict(self,max_nof_events):  # Returns a dictionary where the keys are the number of events of the supports tested, and each value is the set of completely solved classes
        (proven_inequivalence_partition_dict, solved_dict) = self.supports_analysis(self.Dense_connectedness_and_esep_partition,max_nof_events)  # This includes dense connectedness, e-separation and directed-edge-only rules
        return solved_dict    
    
    def certainly_non_algebraic_proven_inequivalence_blocks(self, proven_inequivalence_partition):   # If a certain proven-inequivalence block only contains representatives that are NOT equivalent to any confounder-free DAG, then all of its elements are certainly non-algebraic (i.e., have inequality constraints)
        if self.temporally_ordered:
            return "Non-Algebraicness should be checked based on the complete proven-inequivalence partition, not just the one of temporally ordered mDAGs."
        else:    
            maybe_algebraic=[]
            for block in proven_inequivalence_partition:
                if any(self.representatives_equivalent_to_confounder_free[representative] for representative in block):
                    maybe_algebraic.append(block)
        return [block for block in proven_inequivalence_partition if block not in maybe_algebraic]   # Gives list of proven-inequivalence blocks that are shown to be fully composed of non-algebraic representatives 
    
    def certainly_non_algebraic_representatives(self, proven_inequivalence_partition):   # Gives list of representatives that are shown to be non-algebraic by the analysis above
        if self.temporally_ordered:
            return "Non-Algebraicness should be checked based on the complete proven-inequivalence partition, not just the one of temporally ordered mDAGs."
        else:       
            return [element for block in self.certainly_non_algebraic_proven_inequivalence_blocks(proven_inequivalence_partition) for element in block]
       
    def certainly_algebraic_representatives(self):    #  If an mDAG is known to be observationally equivalent to a confounder-free mDAG, then it is certainly algebraic.   
        return [representative for representative in self.representative_mDAGs_list if self.representatives_equivalent_to_confounder_free[representative]]
       
    def filter_equivalent_to_temporally_ordered(self, set_of_mDAGs):  # Given a set of mDAGs, this function says which ones are equivalent to temporally ordered mDAGs. Useful for analyzing nonalgebraicness.
        temporally_ordered=set(self.dict_ind_temporally_ordered_mDAGs.values())  
        filtered=[]
        for mDAG in set_of_mDAGs:
            if set(self.eqclass_of_mDAG(mDAG)) & temporally_ordered:
                filtered.append(mDAG)
        return filtered
    
    def filter_temporally_ordered_proven_inequivalence_blocks(self, proven_inequivalence_partition):  # Given a proven-inequivalence partition, which in our notation is a set of sets of representatives, this function filters out the proven-inequivalence partition that holds only among temporally-ordered mDAGs. Useful for analyzing nonalgebraicness.
        proven_inequivalence_block_temporally_ordered=[]
        for block in proven_inequivalence_partition:
            filtered_block=self.filter_equivalent_to_temporally_ordered(block)
            if len(filtered_block)>0:
                proven_inequivalence_block_temporally_ordered.append(filtered_block)
        return proven_inequivalence_block_temporally_ordered
       
if __name__ == "__main__":
    
    # TEMPORALLY ORDERED ANALYSIS OF PROVEN-EQUIVALENCE AND PROVEN-INEQUIVALENCE PARTITIONS:
    mDAG_analysis= Proven_Inequivalence_Partition_Analysis(nof_observed_variables=4, temporally_ordered=True, verbose=2)
    print("Number of Temporally ordered mDAGs:", len(mDAG_analysis.all_temporally_ordered_mDAGs))
    print("Number of proven-equivalence blocks:", len(mDAG_analysis.equivalence_classes_as_ids))
    print("Number of proven-inequivalence blocks by skeletons:", len(mDAG_analysis.Skeleton_partition))
    print("Number of proven-inequivalence blocks by dsep:", len(mDAG_analysis.CI_partition))
    print("Number of proven-inequivalence blocks by skeletons and dsep:", len(mDAG_analysis.Skeleton_and_CI_partition))
    print("Number of proven-inequivalence blocks by esep:", len(mDAG_analysis.esep_partition))
    print("Number of proven-inequivalence blocks by Dense Connectedness+esep:", len(mDAG_analysis.Dense_connectedness_and_esep_partition))
    k=3
    supps_dict=mDAG_analysis.supports_proven_inequivalence_partition_dict(k)
    for n in range(1,k+1):
        print("Number of proven-inequivalence blocks by graphical methods + supports up to", n, "events:",len(supps_dict[n]))
    print("Number of completely solved classes by skeletons:", len(mDAG_analysis.completely_solved(mDAG_analysis.Skeleton_partition)))
    print("Number of completely solved classes by dsep:", len(mDAG_analysis.completely_solved(mDAG_analysis.CI_partition)))
    print("Number of completely solved classes by skeletons and dsep:", len(mDAG_analysis.completely_solved(mDAG_analysis.Skeleton_and_CI_partition)))
    print("Number of completely solved classes by esep:", len(mDAG_analysis.completely_solved(mDAG_analysis.esep_partition)))
    print("Number of completely solved classes by Dense Connectedness+esep:", len(mDAG_analysis.completely_solved(mDAG_analysis.Dense_connectedness_and_esep_partition)))
    supps_solved_dict=mDAG_analysis.supports_solved_classes_dict(k)
    for n in range(1,k+1):
        print("Number of completely solved classes by graphical methods + supports up to", n, "events:",len(supps_solved_dict[n]))
    
    
    # FINDING MINIMUM FRACTION OF NONALGEBRAIC TEMPORALLY-ORDERED CLASSES (must analyse all mDAGs, not only temporally ordered)
    nonalgebraicness_analysis= Proven_Inequivalence_Partition_Analysis(nof_observed_variables=4, temporally_ordered=False, verbose=2)
    # Every algebraic class must contain at least one confounder-free mDAG. Therefore:
    print("Maximum number of algebraic temporally ordered classes:", len(nonalgebraicness_analysis.filter_equivalent_to_temporally_ordered(nonalgebraicness_analysis.certainly_algebraic_representatives())))       
    # Obtaining the certainly nonalgebraic mDAGs from the proven-inequivalence partition of all mDAGs, and then restricting to temporally ordered mDAGs:
    nonalgebraic_all=nonalgebraicness_analysis.certainly_non_algebraic_proven_inequivalence_blocks(nonalgebraicness_analysis.supports_proven_inequivalence_partition_dict(3)[3])
    print("Minimum number of non-algebraic temporally ordered classes:", len(nonalgebraicness_analysis.filter_temporally_ordered_proven_inequivalence_blocks(nonalgebraic_all)))
    
