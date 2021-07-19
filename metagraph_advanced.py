from __future__ import absolute_import
import networkx as nx
import numpy as np
import itertools
import scipy.special #For binomial coefficient


from operator import itemgetter
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

from hypergraphs import hypergraph
from directed_structures import directed_structure
from mDAG_advanced import mDAG



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
    def all_unlabelled_directed_structures(self):
        possible_node_pairs = list(itertools.combinations(self.tuple_n, 2))
        excessive_unlabelled_directed_structures = [directed_structure(self.tuple_n, edge_list) for r
                                                    in range(1, len(possible_node_pairs)) for edge_list
                                                    in itertools.combinations(possible_node_pairs, r)]
        d = defaultdict(list)
        for ds in excessive_unlabelled_directed_structures:
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
        return [mDAG(ds, sc) for sc in self.all_simplicial_complices for ds in self.all_unlabelled_directed_structures]

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

    @property
    def all_unlabelled_mDAGs(self):
        return self.dict_ind_unlabelled_mDAGs.values()

    @cached_property
    def meta_graph_nodes(self):
        return tuple(self.dict_ind_unlabelled_mDAGs.keys())


    @cached_property
    def hypergraph_dominances(self):
        return [(S1.as_integer, S2.as_integer) for S1, S2 in itertools.permutations(self.all_simplicial_complices, 2) if
                S1.can_S1_minimally_simulate_S2(S2)]

    @cached_property
    def directed_dominances(self):
        return [(D1.as_integer, D2.as_integer) for D1, D2 in itertools.permutations(self.all_unlabelled_directed_structures, 2) if
                D1.can_D1_minimally_simulate_D2(D2)]

    @property
    def meta_graph_directed_edges(self):
        # I'm experimenting with only using the directed dominances.
        for h in map(lambda sc: sc.as_integer, self.all_simplicial_complices):
            for d1, d2 in self.directed_dominances:
                yield (self.mdag_int_pair_to_canonical_int(h, d1), self.mdag_int_pair_to_canonical_int(h, d2))
        for d in  map(lambda ds: ds.as_integer, self.all_unlabelled_directed_structures):
            for h1, h2 in self.hypergraph_dominances:
                yield (self.mdag_int_pair_to_canonical_int(h1, d), self.mdag_int_pair_to_canonical_int(h2, d))

    def meta_graph_undirected_edges(self, unsafe=True):
        for (id, mDAG) in self.dict_ind_unlabelled_mDAGs.items():  # NEW: Over graph patterns only
            for mDAG2 in mDAG.generate_weaker_mDAG_HLP:
                yield (self.dict_id_to_canonical_id[mDAG2], id)
            for mDAG2 in mDAG.generate_weaker_mDAGs_FaceSplitting(unsafe=unsafe):
                yield (self.dict_id_to_canonical_id[mDAG2], id)

    @cached_property
    def meta_graph(self):
        g = nx.DiGraph()
        g.add_nodes_from(self.meta_graph_nodes)
        print('Adding dominance relations...', flush=True)
        g.add_edges_from(self.meta_graph_directed_edges)
        print('Adding equivalence relations...', flush=True)
        g.add_edges_from(self.meta_graph_undirected_edges(unsafe=True))
        print('Metagraph has been constructed.', flush=True)
        return g

    @cached_property
    def equivalence_classes(self):
        return list(nx.strongly_connected_components(self.meta_graph))





if __name__ == '__main__':
    n = 4
    Observable_mDAGs = Observable_unlabelled_mDAGs(n)
    print("Number of unlabelled graph patterns: ", len(Observable_mDAGs.all_unlabelled_mDAGs), flush=True)
    fundamental_list = [mDAG.fundamental_graphQ for mDAG in Observable_mDAGs.all_unlabelled_mDAGs]
    print("Number of fundamental unlabelled graph patterns: ", len(np.flatnonzero(fundamental_list)), flush=True)

    eqclasses = Observable_mDAGs.equivalence_classes
    print("Upper bound on number of equivalence classes: ", len(eqclasses), flush=True)
