from __future__ import absolute_import
import networkx as nx
import numpy as np
from itertools import combinations, repeat, chain, product, islice, permutations
import itertools
import scipy.special #For binomial coefficient
from mDAG import mDAG
from operator import itemgetter

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

class Observable_mDAGs:
    def __init__(self, n):
        """
        Parameters
        ----------
        n : int
            The number of observed variables.
        """
        self.n = n
        self.setn = set(range(self.n))
        self.error_collection = []

    @cached_property
    def all_directed_structures(self):
        l = []
        debug = []
        baseg = nx.DiGraph()
        baseg.add_nodes_from(self.setn, type="Visible")
        possible_node_pairs = list(combinations(self.setn, 2))
        reversed_node_pairs = list(map(tuple, map(reversed, possible_node_pairs)))
        node_pair_choices = tuple(zip(possible_node_pairs, reversed_node_pairs))
        for list_of_directions in product(range(3), repeat=scipy.special.comb(self.n, 2, exact=True)):
            g = baseg.copy()  # Reset
            g.add_edges_from(
                tuple(choice[direction]) for choice, direction in zip(node_pair_choices, list_of_directions) if
                direction < 2)
            if nx.is_directed_acyclic_graph(g):
                l.append(g.copy())
        return l

    @cached_property
    def num_directed_structures(self):
        return len(self.all_directed_structures)

    @cached_property
    def all_directed_structures_as_tuples(self):
        # return list(map(nx.edges,self.all_directed_structures))
        return [tuple(sorted(d.edges())) for d in self.all_directed_structures]

    def lookup_directed_structure_index(self, mDAG):
        return self.all_directed_structures_as_tuples.index(tuple(sorted(mDAG.directed_structure.edges())))

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
            sorted(simplicial_complex + [(singleton,) for singleton in self.setn.difference(*simplicial_complex)])) for
                simplicial_complex in list_of_simplicial_complices]

    def lookup_simplicial_complex_index(self, mDAG):
        return self.all_simplicial_complices.index(tuple(sorted(mDAG.extended_simplicial_complex)))

    def lookup_indices(self, mDAG):
        d = self.lookup_directed_structure_index(mDAG)
        s = self.lookup_simplicial_complex_index(mDAG)
        return (s, d)

    @cached_property
    def num_simplicial_complices(self):
        return len(self.all_simplicial_complices)

    @cached_property
    def all_indexed_mDAGs(self):
        return [(mDAG(directed_structure, list(simplicial_complex), complex_extended=True), s, d)
                for s, simplicial_complex in enumerate(self.all_simplicial_complices)
                for d, directed_structure in enumerate(self.all_directed_structures)
                ]

    @cached_property
    def all_mDAGs(self):
        return [mDAG for (mDAG, s, d) in self.all_indexed_mDAGs]

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
        return all(any(s2.issubset(s1) for s1 in S1) for s2 in map(set, S2))
        dominance_count = 0
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
        return [(first[0], second[0]) for first, second in permutations(enumerate(self.all_directed_structures), 2) if
                self.can_D1_simulate_D2(first[1], second[1])]

    @cached_property
    def hypergraph_dominances(self):
        return [(first[0], second[0]) for first, second in permutations(enumerate(self.all_simplicial_complices), 2) if
                self.can_S1_simulate_S2(first[1], second[1])]

    def lookup_mDAG(self, indices):
        # Untested. I think this works well for for multiple sets of indices, may need some wrapping if only one set of indices is given.
        return itemgetter(
            *np.ravel_multi_index(np.array(indices).T, (self.num_simplicial_complices, self.num_directed_structures)))(
            self.all_mDAGs)

    def lookup_mDAGs(self, multiple_indices):
        as_array = np.array(list(multiple_indices))
        looked_up = itemgetter(
            *np.ravel_multi_index(as_array.T, (self.num_simplicial_complices, self.num_directed_structures)))(
            self.all_mDAGs)
        if as_array.__len__() == 1:
            return [looked_up]
        else:
            return looked_up

    @cached_property
    def meta_graph_nodes(self):
        return [(s, d)
                for s in range(self.num_simplicial_complices)
                for d in range(self.num_directed_structures)
                ]


    @property
    def meta_graph_directed_edges(self):
        # I'm experimenting with only using the directed dominances.
        for h in range(self.num_simplicial_complices):
            for d1, d2 in self.directed_dominances:
                yield ((h, d1), (h, d2))
        for d in range(self.num_directed_structures):
            for h1, h2 in self.hypergraph_dominances:
                yield ((h1, d), (h2, d))

    @property
    def safe_meta_graph_undirected_edges(self):
        for (mDAG, s, d) in self.all_indexed_mDAGs:
            for mDAG2 in mDAG.generate_weaker_mDAG_HLP:  # Generates AT MOST ONE alternate directed structure
                d2 = self.lookup_directed_structure_index(mDAG2)
                yield ((s, d2), (s, d))
            for mDAG2 in mDAG.generate_weaker_mDAGs_FaceSplitting:
                s2 = self.lookup_simplicial_complex_index(mDAG2)
                yield ((s2, d), (s, d))

    @property
    def meta_graph_undirected_edges(self):
        for (mDAG, s, d) in self.all_indexed_mDAGs:
            for mDAG2 in mDAG.generate_weaker_mDAG_HLP:  # Generates AT MOST ONE alternate directed structure
                d2 = self.lookup_directed_structure_index(mDAG2)
                yield ((s, d2), (s, d))
            for mDAG2 in mDAG.generate_weaker_mDAGs_FaceSplitting_Simultaneous:
                s2 = self.lookup_simplicial_complex_index(mDAG2)
                yield ((s2, d), (s, d))
            #And to account for relabellings...
            for mDAG2 in mDAG.generate_symmetric_variants:
                d2 = self.lookup_directed_structure_index(mDAG2)
                s2 = self.lookup_simplicial_complex_index(mDAG2)
                yield ((s2, d2), (s, d))



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
        g.add_edges_from(self.meta_graph_directed_edges)
        g.add_edges_from(self.meta_graph_undirected_edges)
        print('Metagraph has been constructed.')
        return g

    @cached_property
    def safe_equivalence_classes(self):
        return list(nx.strongly_connected_components(self.safe_meta_graph))

    @cached_property
    def equivalence_classes(self):
        return list(nx.strongly_connected_components(self.meta_graph))

    def same_eq_class(self, mDAG1, mDAG2):
        (s1, d1) = self.lookup_indices(mDAG1)
        (s2, d2) = self.lookup_indices(mDAG2)
        for eq_class in self.equivalence_classes:
            if (s1, d1) in eq_class and (s2, d2) in eq_class:
                return True
                break
        return False

    def dict_ind_mDAGs(self):
        return {(s, d): mDAG for (mDAG, s, d) in self.all_indexed_mDAGs}

    def CI_classes(self):
        dict_CI_relations = {}
        indexed_mDAGs = self.dict_ind_mDAGs()
        eq_classes = self.equivalence_classes
        for equivalence_class in eq_classes:
            representative_indices = next(iter(equivalence_class))  # just pick some mDAG of every equivalence class
            representative_mDAG = indexed_mDAGs[representative_indices]
            dict_CI_relations[representative_mDAG] = tuple(representative_mDAG.all_CI)
            # return a dictionary where the CI relations are the keys and a list of the representative mDAGs of the corresponding equivalence classes are the values
        return {CI: [equiv_class for equiv_class in dict_CI_relations.keys() if dict_CI_relations[equiv_class] == CI]
                for CI in set(dict_CI_relations.values())}

    # Acknowledgement: Leonardo Lessa helped me improve CI_classes

    @cached_property
    def singleton_equivalence_classes(self):
        return [tuple(eqclass)[0] for eqclass in self.equivalence_classes if len(eqclass) == 1]
