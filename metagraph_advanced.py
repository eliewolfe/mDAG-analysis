from __future__ import absolute_import
import networkx as nx
import numpy as np
import itertools
import array
from more_itertools import ilen
# import scipy.special #For binomial coefficient
import progressbar
from utilities import partsextractor
from collections import defaultdict
from scipy import sparse

from sys import stderr, version_info
assert version_info >= (3, 8), "Python 3.8+ is required for cached_property support."
import gc

from functools import cached_property

from hypergraphs import Hypergraph
from directed_structures import DirectedStructure
from mDAG_advanced import mDAG
from canonical_forms import CanonicalForms
# from functools import lru_cache
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


class IndexedLabels:
    """A read-only mapping id -> label backed by the node index and a label array (class_index_of)."""
    def __init__(self, index, labels) -> None:
        self.index, self.labels = index, labels

    def __getitem__(self, uid):
        return int(self.labels[self.index[uid]])

    def __contains__(self, uid) -> bool:
        return uid in self.index

    def __len__(self) -> int:
        return len(self.index)

    def __iter__(self):
        return iter(self.index)

    def get(self, uid, default=None):
        return self[uid] if uid in self.index else default

    def items(self):
        return ((uid, int(self.labels[i])) for uid, i in self.index.items())


class CanonicalIdView:
    """A read-only mapping from the id of any labelled mDAG (a tabulated directed structure with any complex) to
    its unlabelled id, canonicalising on demand: what dict_id_to_canonical_id used to hold as a dict over every
    labelled mDAG (seven million entries on five nodes)."""
    def __init__(self, forms: CanonicalForms) -> None:
        self.forms = forms

    def __getitem__(self, mdag_int):
        return self.forms.canonical_of_int(mdag_int)

    def __contains__(self, mdag_int) -> bool:
        return mdag_int in self.forms

    def get(self, mdag_int, default=None):
        return self[mdag_int] if mdag_int in self else default

class Observable_unlabelled_mDAGs:
    """The metagraph of the n-node mDAGs up to relabelling: nodes are the unlabelled mDAGs, edges are sound
    containments of model sets (structural dominance, face splitting, the HLP edge rule), and the strongly connected
    components are proven observational equivalence classes.

    `rule` selects the model sets the edges are sound for. 'classical' (the default) uses every rule. 'quantum'
    uses structural dominance and face splitting, which are quantum-valid as they stand, and the HLP edge rule only
    where its origin node is premeasurable (mDAG.is_premeasurable); its components are proven *quantum*
    observational equivalence classes, each inside one classical class. The two metagraphs share everything but the
    HLP edges: build the quantum one and derive the classical one with `ClassicalMetagraph.from_quantum`.
    """
    SHARED_CACHE = ('all_directed_structures', 'all_unlabelled_directed_structures', 'truly_all_directed_structures',
                    'all_noncanonical_directed_structures', 'all_simplicial_complices', 'most_labelled_mDAGs',
                    'all_labelled_mDAGs', 'canonical_forms', 'dict_id_to_canonical_id', 'dict_uniqind_unlabelled_mDAGs',
                    'dict_ind_unlabelled_mDAGs', 'metagraph_nodes', 'hypergraph_dominances',
                    'directed_dominances_up_to_symmetry', 'structural_dominances', 'face_splitting_edge_list',
                    'latent_free_DAGs_unlabelled', 'structural_order', '_structural_closure', '_dict_id_to_idx',
                    'num_metagraph_nodes')
    # What a sibling must share for the two metagraphs to agree (the rest is shared when already computed).
    FORCED_SHARED = ('canonical_forms', 'dict_uniqind_unlabelled_mDAGs', 'structural_dominances',
                     'face_splitting_edge_list', 'structural_order')

    def __init__(self, n: int,
                 fully_foundational=False,
                 all_dominances=False,
                 verbose=True,
                 rule: str = 'classical',
                 compact=None):
        """
        Parameters
        ----------
        n : int
            The number of observed variables.
        rule : 'classical' or 'quantum'
            Which model sets the metagraph's equivalences are sound for (see the class docstring).
        compact : bool, default n >= 5
            Keep the memory per node small: every mDAG's cached attributes are released once its edges have been
            read off (a few hundred bytes per node instead of a few kilobytes), and the dominances enter the
            classical graph without the CI filter of `boring_dominances`, which prunes edges that can never lie
            on a cycle and so leaves the components unchanged while costing a d-separation enumeration per node.
        """
        assert rule in ('classical', 'quantum'), rule
        self.n = n
        self.rule = rule
        self.compact = (n >= 5) if compact is None else bool(compact)
        self.tuple_n = tuple(range(self.n))
        self.set_n = set(self.tuple_n)
        self.sort_by_length = lambda s: sorted(s, key=len)
        self.fully_foundational = fully_foundational
        self.verbose = verbose
        # self.experimental_speedup = experimental_speedup
        self.all_dominances = all_dominances

    @classmethod
    def from_sibling(cls, other: "Observable_unlabelled_mDAGs", rule: str, **kwargs) -> "Observable_unlabelled_mDAGs":
        """A metagraph on the same nodes under another rule, sharing every rule-independent cached attribute."""
        new = cls.__new__(cls)
        Observable_unlabelled_mDAGs.__init__(new, other.n, fully_foundational=other.fully_foundational,
                                             all_dominances=other.all_dominances, verbose=other.verbose, rule=rule,
                                             compact=other.compact, **kwargs)
        for name in cls.FORCED_SHARED:
            getattr(other, name)                       # computed on `other` if need be, so that both share it
        for name in cls.SHARED_CACHE:
            if name in other.__dict__:
                new.__dict__[name] = other.__dict__[name]
        return new

    @cached_property
    def all_directed_structures(self):
        possible_node_pairs = list(itertools.combinations(self.tuple_n, 2))
        return tuple(DirectedStructure(edge_list, self.n) for r
                in range(0, len(possible_node_pairs)+1) for edge_list
                in itertools.combinations(possible_node_pairs, r))
    
    
    @cached_property
    def all_unlabelled_directed_structures(self):
        return tuple({ds.as_unlabelled_integer: ds for ds in explore_candidates(self.all_directed_structures,
                                     verbose=False,
                                     message="Iterating over directed structures.")}.values())

    @cached_property
    def truly_all_directed_structures(self):
        truly_all = []
        for ds in self.all_unlabelled_directed_structures:
            truly_all.extend(ds.equivalents_under_symmetry)
        return truly_all

    @cached_property
    def all_noncanonical_directed_structures(self):
        # return set(self.truly_all_directed_structures).difference(self.all_unlabelled_directed_structures)
        return set(self.all_directed_structures).difference(self.all_unlabelled_directed_structures)

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
        return tuple(sorted((Hypergraph(sc, self.n) for sc in list_of_simplicial_complices), key=lambda sc: sc.tally))

    @cached_property
    def all_unlabelled_simplicial_complices(self):
        return tuple({sc.as_unlabelled_integer: sc for sc in explore_candidates(self.all_simplicial_complices,
                                     verbose=False,
                                     message="Iterating over simplicial complices.")}.values())



    @cached_property
    def most_labelled_mDAGs(self):
        return [mDAG(ds, sc) for sc, ds in itertools.product(self.all_simplicial_complices,
                                                             self.all_unlabelled_directed_structures)]

    @cached_property
    def all_labelled_mDAGs(self):
        return self.most_labelled_mDAGs + [mDAG(ds, sc) for sc, ds in itertools.product(self.all_simplicial_complices,
                                                                 self.all_noncanonical_directed_structures)]

    @cached_property
    def canonical_forms(self):
        """Permutation tables of the labelled directed structures and complices (canonical_forms.CanonicalForms):
        canonical ids on demand, the orbit enumeration of the nodes and their lower neighbours."""
        if self.verbose:
            eprint("Tabulating directed structures and simplicial complices under relabelling...", flush=True)
        return CanonicalForms(self.n,
                              [ds.as_integer for ds in self.all_directed_structures],
                              [ds.as_bit_square_matrix for ds in self.all_directed_structures],
                              [sc.as_integer for sc in self.all_simplicial_complices],
                              [sc.as_tuples for sc in self.all_simplicial_complices])

    def canonical_id(self, ds_int, sc_int):
        """The unlabelled id (mDAG.unique_unlabelled_id) of the labelled mDAG with these parts, canonicalised on
        demand: the canonical complex first, then the smallest directed structure over the relabellings that
        achieve it (a bounded memo keeps recent answers)."""
        return self.canonical_forms.canonical(ds_int, sc_int)

    @cached_property
    def dict_id_to_canonical_id(self):
        """Labelled mDAG id -> unlabelled id, as a lazy view (CanonicalIdView) rather than a dict over every
        labelled mDAG."""
        return CanonicalIdView(self.canonical_forms)

    @cached_property
    def ds_bit_size(self):
        return np.asarray(2**(self.n**2), dtype=object)

    def mdag_int_pair_to_single_int(self, ds_int, sc_int):
        #ELIE: Note that this MUST MATCH the function mdag_int_pair_to_single_int in the mDAG class. All is good.
        return ds_int + sc_int*self.ds_bit_size

    def mdag_int_pair_to_canonical_int(self, ds_int, sc_int):
        return self.canonical_id(ds_int, sc_int)

    @cached_property
    def dict_uniqind_unlabelled_mDAGs(self):
        """unlabelled id -> one labelled representative mDAG, enumerated by orbits: for each unlabelled directed
        structure's representative, the complices up to its automorphism group (no non-canonical mDAG is ever
        built). Order and representatives are those of the former dict over complex x directed structure."""
        if self.verbose:
            eprint("Enumerating unlabelled mDAGs by orbits...", flush=True)
        ds_by_int = {ds.as_integer: ds for ds in self.all_unlabelled_directed_structures}
        sc_by_int = {sc.as_integer: sc for sc in self.all_simplicial_complices}
        return {canonical: mDAG(ds_by_int[ds_int], sc_by_int[sc_int]) for canonical, ds_int, sc_int in
                self.canonical_forms.nodes(list(ds_by_int.keys()))}

    @cached_property
    def dict_ind_unlabelled_mDAGs(self):
        return {mdag.unique_id: mdag for mdag in self.dict_uniqind_unlabelled_mDAGs.values()}


    def lookup_mDAG(self, indices):
        return partsextractor(self.dict_uniqind_unlabelled_mDAGs, indices)
    @cached_property
    def dict_labelled_to_unlabelled_mDAGs(self):
        return {mdag: self.lookup_mDAG(mdag.unique_unlabelled_id) for mdag in self.all_labelled_mDAGs}

    def convert_to_unlabelled(self, mdags):
        return partsextractor(self.dict_labelled_to_unlabelled_mDAGs, mdags)

    @property
    def all_unlabelled_mDAGs(self):
        return self.dict_uniqind_unlabelled_mDAGs.values()

    @cached_property
    def all_unlabelled_mDAGs_faster(self):
        "Warning: This function returns mDAGs who's id does not match unique_id."
        d = defaultdict(list)
        for mdag in self.most_labelled_mDAGs:
            d[mdag.unique_unlabelled_id].append(mdag)
        return tuple(next(iter(eqclass)) for eqclass in d.values())

    @cached_property
    def latent_free_DAGs_unlabelled(self):
        empty_sc = Hypergraph([], self.n)
        return {mDAG(ds, empty_sc) for ds in self.all_unlabelled_directed_structures}

    @cached_property
    def latent_free_DAGs_labelled(self):
        empty_sc = Hypergraph([], self.n)
        return {mDAG(ds, empty_sc) for ds in self.all_directed_structures}

    # @cached_property
    # def latent_free_DAG_ids(self):
    #     return set([m.unique_id for m in self.latent_free_DAGs_unlabelled])

    @cached_property
    def latent_free_DAG_ids_and_CI_unlabelled(self):
        return {m.unique_unlabelled_id: m.all_CI_unlabelled for m in sorted(
            self.latent_free_DAGs_unlabelled,
            key=lambda m: m.CI_count,
            reverse=False)}

    @cached_property
    def latent_free_DAG_ids_and_CI_labelled(self):
        return {m.unique_id: m.all_CI for m in sorted(
            self.latent_free_DAGs,
            key=lambda m: m.CI_count,
            reverse=False)}

    @cached_property
    def latent_free_CI_patterns(self):
        recognized_patterns = set()
        for m in self.latent_free_DAGs_unlabelled:
            recognized_patterns.add(m.all_CI_unlabelled)
        return recognized_patterns

    @cached_property
    def latent_free_esep_patterns(self):
        recognized_patterns = set()
        for m in self.latent_free_DAGs_unlabelled:
            recognized_patterns.add(m.all_esep_unlabelled)
        return recognized_patterns

    @cached_property
    def latent_free_esep_plus_patterns(self):
        recognized_patterns = set()
        for m in self.latent_free_DAGs_unlabelled:
            recognized_patterns.add(m.all_esep_plus_unlabelled)
        return recognized_patterns

    # @cached_property
    # def all_unlabelled_mDAGs_latent_free_equivalent(self):
    #     pass_count = 0
    #     known_collection_dict = dict()
    #     recently_discovered_latent_free = set(self.convert_to_unlabelled([mDAG(ds, Hypergraph([], self.n)) for ds in self.all_unlabelled_directed_structures]))
    #     while len(recently_discovered_latent_free) > 0:
    #         previously_discovered = recently_discovered_latent_free.copy()
    #         known_collection_dict.update({m.unique_unlabelled_id: m for m in previously_discovered})
    #         print(f"Pass #{pass_count+1}, Boring count: {len(known_collection_dict)}")
    #         #STEP ONE: INCREASE LATENT POWER
    #         after_adding_latents = dict()
    #         for m in previously_discovered:
    #             ds = m.directed_structure_instance
    #             orig_sc = m.simplicial_complex_instance
    #             CI_count = len(m.all_CI)
    #             for sc in self.all_simplicial_complices:
    #                 if sc.is_S1_strictly_above_S2(orig_sc):
    #                     m2 = mDAG(ds, sc)
    #                     if m2.unique_unlabelled_id not in known_collection_dict.keys():
    #                         if len(m2.all_CI) == CI_count:  # That is, no loss of CI relations
    #                             after_adding_latents[m2.unique_unlabelled_id] = self.convert_to_unlabelled(m2)
    #         #STEP TWO: DROP DIRECTED EDGES SAFELY
    #         recently_discovered_latent_free_dict = after_adding_latents.copy()
    #         for i, m in enumerate(after_adding_latents.values()):
    #             # print(f"Now trying to expand discovery on mDAG #{i} of {len(after_adding_latents)}, namely, {m}")
    #             newly_discovered_for_this_sc_dict = {m.unique_unlabelled_id: m}
    #             while len(newly_discovered_for_this_sc_dict) > 0:
    #                 # print("Recently discovered: ", newly_discovered_for_this_sc_dict)
    #                 previously_discovered = newly_discovered_for_this_sc_dict.copy()
    #                 newly_discovered_for_this_sc_dict = dict()
    #                 orig_ds = m.directed_structure_instance
    #                 orig_sc = m.simplicial_complex_instance
    #                 CI_count = len(m.all_CI)
    #                 for m_equiv in set(self.convert_to_unlabelled(m.equivalent_under_edge_dropping)).difference(recently_discovered_latent_free_dict.values()):
    #                     m_equiv_unlabelled = self.convert_to_unlabelled(m_equiv)
    #                     newly_discovered_for_this_sc_dict[m_equiv_unlabelled.unique_id] = m_equiv_unlabelled
    #                     recently_discovered_latent_free_dict[m_equiv_unlabelled.unique_id] = m_equiv_unlabelled
    #                     for ds_stronger in self.all_directed_structures:  # IDEA: CANNOT RELY ON ONLY UNLABELLED DAGS!
    #                         if ds_stronger.is_D1_strictly_above_D2(orig_ds):
    #                             m2 = mDAG(ds_stronger, orig_sc)
    #                             m2_unlabelled = self.convert_to_unlabelled(m2)
    #                             if m2_unlabelled.unique_id not in recently_discovered_latent_free_dict.keys():
    #                                 if len(m2_unlabelled.all_CI) == CI_count:
    #                                     newly_discovered_for_this_sc_dict[m2_unlabelled.unique_id] = self.convert_to_unlabelled(m2_unlabelled)
    #                                     recently_discovered_latent_free_dict[m2_unlabelled.unique_id] = self.convert_to_unlabelled(m2_unlabelled)
    #         recently_discovered_latent_free = set(m for (k, m) in recently_discovered_latent_free_dict.items()
    #                                               if k not in known_collection_dict.keys())
    #         pass_count += 1
    #     # print("Most recently discovered: ", previously_discovered)
    #     return set(known_collection_dict.values())
    #     #
    #     # print("Hardest to infer boring: ",hardest_to_infer)
    #     # d = defaultdict(list)
    #     # for m in new_collection:
    #     #     d[m.unique_unlabelled_id].append(m)
    #     # return tuple(next(iter(eqclass)) for eqclass in d.values())

    @cached_property
    def metagraph_nodes(self):
        return np.array(tuple(self.dict_uniqind_unlabelled_mDAGs.keys()))

    @cached_property
    def num_metagraph_nodes(self):
        return len(self.metagraph_nodes)

    @cached_property
    def hypergraph_dominances(self):
        return tuple((S1.as_integer, S2.as_integer) for S1, S2 in itertools.permutations(self.all_simplicial_complices, 2) if
                S1.can_S1_minimally_simulate_S2(S2))
        # if self.verbose:
        #     eprint("Number of hypergraph dominance relations: ", len(hdominances))
        # return hdominances

    @cached_property
    def hypergraph_dominances_up_to_symmetry(self):
        hdominances = tuple({min(zip(S1.as_integer_permutations,
                                     S2.as_integer_permutations)): (S1.as_integer,
                                                                    S2.as_integer)
               for S1, S2 in
               itertools.permutations(self.all_simplicial_complices, 2) if
               S1.can_S1_minimally_simulate_S2(S2)}.values())
        return hdominances

    # @property
    # def hypergraph_strong_dominances(self):
    #     return [(S1.as_integer, S2.as_integer) for S1, S2 in itertools.permutations(self.all_simplicial_complices, 2) if
    #             S1.are_S1_facets_one_more_than_S2_facets(S2)]

    @cached_property
    def directed_dominances_up_to_symmetry(self):
        # return tuple((D1.as_integer, D2.as_integer) for D1, D2 in
        #         itertools.permutations(self.all_directed_structures, 2) if
        #         D1.can_D1_minimally_simulate_D2(D2))
        return tuple({min(zip(D1.as_integer_permutations, D2.as_integer_permutations)): (D1.as_integer, D2.as_integer)
                for D1, D2 in
                itertools.permutations(self.all_directed_structures, 2) if
                D1.can_D1_minimally_simulate_D2(D2)}.values())

    @cached_property
    def directed_dominances(self):
        return tuple((D1.as_integer, D2.as_integer)
                for D1, D2 in
                itertools.permutations(self.truly_all_directed_structures, 2) if
                D1.can_D1_minimally_simulate_D2(D2))

    # @property
    # def raw_all_hypergraph_dominances(self):
    #     for d in map(lambda ds: ds.as_integer, explore_candidates(self.all_unlabelled_directed_structures,
    #                                                               verbose=self.verbose,
    #                                                               message="Enumerating all hypergraph dominances...")):
    #         for h1, h2 in self.hypergraph_dominances:
    #             yield (self.mdag_int_pair_to_single_int(d, h1), self.mdag_int_pair_to_single_int(d, h2))
    # @property
    # def _all_hypergraph_dominances(self):
    #     for d in map(lambda ds: ds.as_integer, explore_candidates(self.all_unlabelled_directed_structures,
    #                                                               verbose=self.verbose,
    #                                                               message="Enumerating all hypergraph dominances...")):
    #         for h1, h2 in self.hypergraph_dominances:
    #             yield (self.mdag_int_pair_to_canonical_int(d, h1), self.mdag_int_pair_to_canonical_int(d, h2))
    #
    # @property
    # def raw_all_directed_dominances(self):
    #     for h in map(lambda sc: sc.as_integer, explore_candidates(self.all_simplicial_complices,
    #                                                               verbose=self.verbose,
    #                                                               message="Enumerating all directed dominances...")):
    #         for d1, d2 in self.directed_dominances_up_to_symmetry:
    #             yield (self.mdag_int_pair_to_single_int(d1, h), self.mdag_int_pair_to_single_int(d2, h))
    #
    # @property
    # def _all_directed_dominances(self):
    #     for h in map(lambda sc: sc.as_integer, explore_candidates(self.all_simplicial_complices,
    #                                                               verbose=self.verbose,
    #                                                               message="Enumerating all directed dominances...")):
    #         for d1, d2 in self.directed_dominances_up_to_symmetry:
    #             yield (self.mdag_int_pair_to_canonical_int(d1, h), self.mdag_int_pair_to_canonical_int(d2, h))


    # ------------------------------------------------------------------
    # Edges. Every edge collection is an (E, 2) int32 array of node indices into metagraph_nodes (row = source,
    # column = target), deduplicated; `edge_ids` turns one back into (id, id) pairs. The raw_* generators below
    # yield labelled ids and are kept for callers that want them.
    # ------------------------------------------------------------------

    @property
    def raw_HLP_edges(self):
        for (id, mdag) in explore_candidates(
                self.dict_ind_unlabelled_mDAGs.items(),
                verbose=self.verbose,
                message=f"Finding HLP Rule #4 type edges ({self.rule} rule)"):  # NEW: Over graph patterns only
            for id2 in mdag.weaker_mDAG_HLP(self.rule):
                yield (id2, id)
            if self.compact:
                mdag.release_cache()

    @property
    def raw_FaceSplitting_edges(self):
        for (id, mdag) in explore_candidates(
                self.dict_ind_unlabelled_mDAGs.items(),
                verbose=self.verbose,
                message="Finding metagraph edges due to face splitting"):  # NEW: Over graph patterns only
            for id2 in mdag.generate_weaker_mDAGs_FaceSplitting('strong'):
                yield (id2, id)
            if self.compact:
                mdag.release_cache()

    @property
    def HLP_edges(self):
        for (id2, id) in self.raw_HLP_edges:
            yield partsextractor(self.dict_id_to_canonical_id, (id2, id))

    @property
    def FaceSplitting_edges(self):
        for (id2, id) in self.raw_FaceSplitting_edges:
            yield partsextractor(self.dict_id_to_canonical_id, (id2, id))

    def _canonical_pairs(self, pairs) -> set:
        return {(self.dict_id_to_canonical_id[a], self.dict_id_to_canonical_id[b]) for a, b in pairs}

    def _edge_array(self, rows, cols):
        """The deduplicated (E, 2) int32 edge array of two equal-length index sequences, first occurrences in
        order."""
        edges = np.empty((len(rows), 2), dtype=np.int32)
        edges[:, 0] = np.frombuffer(rows, dtype=np.int32) if isinstance(rows, array.array) else rows
        edges[:, 1] = np.frombuffer(cols, dtype=np.int32) if isinstance(cols, array.array) else cols
        if not len(edges):
            return edges
        keys = edges[:, 0].astype(np.int64) * self.num_metagraph_nodes + edges[:, 1]
        return edges[np.sort(np.unique(keys, return_index=True)[1])]

    def edge_ids(self, edges):
        """An edge array as a list of (source id, target id) pairs."""
        nodes = self.metagraph_nodes
        return [(nodes[a], nodes[b]) for a, b in np.asarray(edges, dtype=np.int64).tolist()]

    def _equivalence_edges(self, weaker_ids_of, message):
        """(weak, strong) edges from a per-node generator of the labelled ids of the weaker structures each
        node is equivalent to; the node itself is already canonical."""
        idx = self._dict_id_to_idx
        canonical_of = self.canonical_forms.canonical_of_int
        rows, cols = array.array('i'), array.array('i')
        for strong, mdag in explore_candidates(self.dict_uniqind_unlabelled_mDAGs.items(),
                                               verbose=self.verbose, message=message):
            j = idx[strong]
            for id2 in weaker_ids_of(mdag):
                rows.append(idx[canonical_of(id2)])
                cols.append(j)
            if self.compact:
                mdag.release_cache()
        return self._edge_array(rows, cols)

    @cached_property
    def structural_dominances(self):
        """(strong, weak) edges: weak is strong with one maximal face or one edge fewer (a containment of both
        the classical and the quantum model set). Generated from each node's lower neighbours (drop one edge;
        drop one maximal face) and canonicalised, which gives the same set as the former all-pairs
        hypergraph_dominances x directed structures and directed_dominances_up_to_symmetry x complices.
        Rule-independent, shared between the two metagraphs."""
        if self.verbose:
            eprint('Adding dominance relations...', flush=True)
        forms = self.canonical_forms
        idx = self._dict_id_to_idx
        rows, cols = array.array('i'), array.array('i')
        for strong, mdag in explore_candidates(self.dict_uniqind_unlabelled_mDAGs.items(),
                                               verbose=self.verbose, message="Structural dominances"):
            i = idx[strong]
            ds_int = mdag.directed_structure_instance.as_integer
            sc_int = mdag.simplicial_complex_instance.as_integer
            for d, s in forms.lower_neighbour_parts(ds_int, sc_int):
                rows.append(i)
                cols.append(idx[forms.canonical(d, s)])
        return self._edge_array(rows, cols)

    @cached_property
    def legacy_structural_dominances(self):
        """The structural dominances by the former all-pairs construction, as id pairs (kept to cross-check the
        new one)."""
        directed_ids = np.array(tuple(ds.as_integer for ds in self.all_unlabelled_directed_structures), dtype=object)[:, np.newaxis, np.newaxis]
        hypergraph_dominances = np.asarray(self.hypergraph_dominances, dtype=object)
        from_hypergraphs = np.reshape(self.mdag_int_pair_to_single_int(directed_ids, hypergraph_dominances), (-1, 2))
        simplicial_ids = np.array(tuple(sc.as_integer for sc in self.all_simplicial_complices), dtype=object)[:, np.newaxis, np.newaxis]
        directed_dominances = np.asarray(self.directed_dominances_up_to_symmetry, dtype=object)
        from_directed_edges = np.reshape(self.mdag_int_pair_to_single_int(directed_dominances, simplicial_ids), (-1, 2))
        return list(self._canonical_pairs(from_hypergraphs.tolist())) + list(self._canonical_pairs(from_directed_edges.tolist()))

    @cached_property
    def hlp_edge_list(self):
        """(weak, strong) edges of the HLP equivalence under this metagraph's rule."""
        rule = self.rule
        return self._equivalence_edges(lambda mdag: mdag.weaker_mDAG_HLP(rule),
                                       f"Finding HLP Rule #4 type edges ({rule} rule)")

    @cached_property
    def face_splitting_edge_list(self):
        """(split, original) edges of the face-splitting equivalence (quantum-valid as well: the source of the
        part D distributes one conditional state per value of what every member of D sees of C)."""
        return self._equivalence_edges(lambda mdag: mdag.generate_weaker_mDAGs_FaceSplitting('strong'),
                                       "Finding metagraph edges due to face splitting")

    @cached_property
    def all_HLP_relevant_dominances(self):
        """(structural dominances, HLP equivalence edges), both as edge arrays."""
        return (self.structural_dominances, self.hlp_edge_list)

    @cached_property
    def boring_dominances(self):
        """The HLP edges together with the structural dominances that can lie on a cycle (same skeleton and CI
        count at both ends), as one edge array; unfiltered under `all_dominances` or `compact`."""
        (equiv_breaking_doms, equiv_maybe_preserving_doms) = self.all_HLP_relevant_dominances
        if self.all_dominances or self.compact:
            # Unfiltered: the filter below only drops dominances that can never lie on a cycle (every edge is a
            # sound containment, so a cycle's members share skeleton and CI count), so the components are the
            # same; it is skipped when the CI enumeration per node is the expensive part.
            return self._edge_array(np.concatenate([equiv_maybe_preserving_doms[:, 0], equiv_breaking_doms[:, 0]]),
                                    np.concatenate([equiv_maybe_preserving_doms[:, 1], equiv_breaking_doms[:, 1]]))
        nodes = self.metagraph_nodes
        mdags = self.dict_uniqind_unlabelled_mDAGs
        keep = np.zeros(len(equiv_breaking_doms), dtype=bool)
        for k, (strong, weak) in enumerate(explore_candidates(equiv_breaking_doms.tolist(),
                                                              verbose=self.verbose,
                                                              message="Finding dominances which preserve CI")):
            strong_mDAG, weak_mDAG = mdags[nodes[strong]], mdags[nodes[weak]]
            # The skeleton (adjacency = no e-separation, Evans) and the CI count are invariants of observational
            # equivalence: a dominance between structures that differ in either is never an equivalence.
            keep[k] = (strong_mDAG.skeleton_nof_edges == weak_mDAG.skeleton_nof_edges
                       and strong_mDAG.CI_count == weak_mDAG.CI_count)
        return np.concatenate([equiv_maybe_preserving_doms, equiv_breaking_doms[keep]])

    # ------------------------------------------------------------------
    # The metagraph as sparse adjacency matrices (scipy.sparse): node i is metagraph_nodes[i], and a True entry
    # (i, j) is an edge from i to j. Strongly connected components, reachability and the topological order all
    # come from the CSR arrays; nothing is held as a networkx graph.
    # ------------------------------------------------------------------

    @cached_property
    def _dict_id_to_idx(self):
        return {id: i for i, id in enumerate(self.metagraph_nodes.flat)}

    def _csr_from_edges(self, *edge_arrays):
        """The boolean N x N CSR adjacency matrix of the given edge arrays."""
        N = self.num_metagraph_nodes
        edges = np.concatenate([np.asarray(e, dtype=np.int32).reshape(-1, 2) for e in edge_arrays]) if edge_arrays \
            else np.empty((0, 2), dtype=np.int32)
        m = sparse.coo_matrix((np.ones(len(edges), dtype=bool), (edges[:, 0], edges[:, 1])), shape=(N, N)).tocsr()
        m.data[:] = True
        return m

    @staticmethod
    def _reachable(csr, start, blocked=None):
        """Indices reachable from `start` along the rows of `csr`, `start` itself excluded, never entering an
        index marked in `blocked`."""
        indptr, indices = csr.indptr, csr.indices
        seen = {start}
        stack = [start]
        out = set()
        while stack:
            u = stack.pop()
            for v in indices[indptr[u]:indptr[u + 1]].tolist():
                if v not in seen and (blocked is None or not blocked[v]):
                    seen.add(v)
                    out.add(v)
                    stack.append(v)
        return out

    @cached_property
    def HLP_meta_graph(self):
        """The classical-rule graph without face splitting: the (CI-filtered) structural dominances and the HLP
        equivalence edges, as a CSR matrix."""
        return self._csr_from_edges(self.boring_dominances)

    @cached_property
    def HLP_meta_adjmat(self):
        """(dominances + HLP edges, transposed, with self-loops): the matrix whose powers HLP_meta_adjmat_closure
        iterates."""
        N = self.num_metagraph_nodes
        return (self._csr_from_edges(*self.all_HLP_relevant_dominances).T + sparse.identity(N, dtype=bool, format='csr')).tocoo()

    @cached_property
    def HLP_meta_adjmat_closure(self):
        n = self.num_metagraph_nodes
        closure_mat = self.HLP_meta_adjmat.tocsr()
        while n > 0:
            n = np.floor_divide(n, 2)
            next_closure_mat = closure_mat.multiply(closure_mat)
            if np.array_equal(closure_mat.indices, next_closure_mat.indices) and np.array_equal(closure_mat.indptr, next_closure_mat.indptr):
                break
            else:
                closure_mat = next_closure_mat
        return closure_mat

    @cached_property
    def boring_by_virtue_of_HLP(self):
        """The mDAGs dominated by (reaching, in HLP_meta_graph) a latent-free DAG, found latent-free DAG by
        latent-free DAG in order of CI count; what one DAG accounts for is removed before the next search."""
        incoming = self.HLP_meta_graph.T.tocsr()          # row v: the nodes with an edge into v
        removed = np.zeros(self.num_metagraph_nodes, dtype=bool)
        idx = self._dict_id_to_idx
        nodes = self.metagraph_nodes
        proven_boring = set()
        for lf_id, lf_CI in explore_candidates(self.latent_free_DAG_ids_and_CI_unlabelled.items(),
                                               verbose=self.verbose,
                                               message="Running HLP in reverse"):
            i = idx[lf_id]
            ids_dominated_by_LF = {lf_id}.union(nodes[list(self._reachable(incoming, i, removed))].tolist())
            dominated_by_LF = partsextractor(self.dict_uniqind_unlabelled_mDAGs, ids_dominated_by_LF)
            if self.all_dominances:
                dominated_by_LF = [m for m in dominated_by_LF if m.all_CI_unlabelled==lf_CI]
            for m in dominated_by_LF:
                removed[idx[m.unique_unlabelled_id]] = True
            proven_boring.update(dominated_by_LF)
        return proven_boring

    @cached_property
    def meta_graph(self):
        """The full metagraph under this rule, as a CSR matrix: structural dominances, the HLP edges of the rule
        and face splitting (the classical companion of a quantum metagraph adds the unrestricted HLP edges)."""
        if self.rule == 'quantum':
            # Every edge is a sound containment of quantum model sets; no CI filter is needed for soundness.
            return self._csr_from_edges(self.structural_dominances, self.hlp_edge_list, self.face_splitting_edge_list)
        if self.verbose:
            eprint('Adding FaceSplitting equivalence relations...', flush=True)
        return self._csr_from_edges(self.boring_dominances, self.face_splitting_edge_list)

    @cached_property
    def class_labels(self):
        """class_labels[i] = index (in equivalence_classes_as_ids) of the class of metagraph_nodes[i]: the
        strongly connected components of meta_graph, numbered by their smallest node index."""
        _, labels = sparse.csgraph.connected_components(self.meta_graph, directed=True, connection='strong')
        first = np.full(labels.max() + 1 if len(labels) else 0, len(labels), dtype=np.int64)
        np.minimum.at(first, labels, np.arange(len(labels)))
        return np.argsort(np.argsort(first, kind='stable'), kind='stable')[labels].astype(np.int32)

    @cached_property
    def equivalence_classes_as_ids(self):
        """The equivalence classes, as sets of unlabelled ids, in the order of class_labels."""
        labels = self.class_labels
        order = np.argsort(labels, kind='stable')
        groups = np.split(order, np.flatnonzero(np.diff(labels[order])) + 1) if len(order) else []
        nodes = self.metagraph_nodes
        return [set(nodes[g].tolist()) for g in groups]

    # ------------------------------------------------------------------
    # Class lookups shared by the quantum and the classical metagraph
    # ------------------------------------------------------------------

    @cached_property
    def class_index_of(self):
        """unlabelled mDAG id -> index of its equivalence class in equivalence_classes_as_ids (a view on
        class_labels, not a second dictionary)."""
        return IndexedLabels(self._dict_id_to_idx, self.class_labels)

    def class_of(self, uid):
        """The equivalence class (a set of unlabelled ids) of an unlabelled mDAG id."""
        return self.equivalence_classes_as_ids[self.class_index_of[uid]]

    @cached_property
    def latent_free_equivalent_ids(self):
        """Members of the classes that contain a latent-free DAG (under this metagraph's rule). Under the
        classical rule these are the provably algebraic structures, for which Q(G) = C(G)."""
        idx = self._dict_id_to_idx
        labels = self.class_labels
        hit = np.zeros(len(self.equivalence_classes_as_ids), dtype=bool)
        hit[[labels[idx[m.unique_unlabelled_id]] for m in self.latent_free_DAGs_unlabelled]] = True
        return set(self.metagraph_nodes[hit[labels]].tolist())

    @cached_property
    def structural_order(self):
        """The structural dominances as a CSR matrix (strong -> weak): u reaches v iff v is u with facets or
        edges removed, so model(u) contains model(v) classically and quantumly."""
        return self._csr_from_edges(self.structural_dominances)

    @cached_property
    def _structural_closure(self):
        """Transitive closure of the structural order as bitsets: (ids in topological order, index of each id in
        it, below[i] = bitmask of the indices structurally below the i-th id, above[i] likewise)."""
        g = self.structural_order
        gT = g.T.tocsr()
        N = g.shape[0]
        indegree = np.diff(gT.indptr).tolist()
        ready = [i for i in range(N) if indegree[i] == 0]
        topo = []
        while ready:                                          # Kahn's algorithm on the CSR arrays
            u = ready.pop()
            topo.append(u)
            for v in g.indices[g.indptr[u]:g.indptr[u + 1]].tolist():
                indegree[v] -= 1
                if indegree[v] == 0:
                    ready.append(v)
        assert len(topo) == N, "the structural order has a cycle"
        position = [0] * N                                    # node index -> position in topo
        for k, u in enumerate(topo):
            position[u] = k
        below = [0] * N
        for k in range(N - 1, -1, -1):
            mask = 0
            u = topo[k]
            for v in g.indices[g.indptr[u]:g.indptr[u + 1]].tolist():
                j = position[v]
                mask |= (1 << j) | below[j]
            below[k] = mask
        above = [0] * N
        for k in range(N):
            mask = 0
            u = topo[k]
            for v in gT.indices[gT.indptr[u]:gT.indptr[u + 1]].tolist():
                j = position[v]
                mask |= (1 << j) | above[j]
            above[k] = mask
        order = self.metagraph_nodes[topo].tolist()
        index = {uid: k for k, uid in enumerate(order)}
        return order, index, below, above

    def _ids_of_bits(self, mask: int):
        order = self._structural_closure[0]
        out = set()
        while mask:
            low = mask & -mask
            out.add(order[low.bit_length() - 1])
            mask ^= low
        return out

    def structurally_below(self, uid):
        """Unlabelled ids of the structures obtained from uid by removing facets or edges (uid excluded)."""
        order, index, below, above = self._structural_closure
        return self._ids_of_bits(below[index[uid]])

    def structurally_above(self, uid):
        order, index, below, above = self._structural_closure
        return self._ids_of_bits(above[index[uid]])

    @cached_property
    def equivalence_classes_as_mDAGs(self):
        return [self.lookup_mDAG(eqclass) for eqclass in self.equivalence_classes_as_ids]
    
    def eqclass_of_mDAG(self, mDAG):
        for eqclass in self.equivalence_classes_as_mDAGs:
            if mDAG in eqclass:
                return eqclass
            
    def eqclasses_of_subset(self, subset):       #subset is a set of mDAGs (i.e. nodes of the metagraph)
        partition_into_eqclasses=[]
        for mDAG in subset:
            element=[G for G in self.eqclass_of_mDAG(mDAG) if G in subset]
            if element not in partition_into_eqclasses:
                partition_into_eqclasses.append(element)
        return partition_into_eqclasses

    def partial_order_of_subset(self, subset):
        """The reachability order of meta_graph restricted to `subset` (mDAGs that are nodes of the metagraph),
        transitively reduced, as a networkx DiGraph for plotting: an edge a -> b means a reaches b in the
        metagraph and no third member of the subset lies between them."""
        subset = list(subset)
        idx = self._dict_id_to_idx
        incoming = self.meta_graph.T.tocsr()
        positions = {idx[g.unique_unlabelled_id]: k for k, g in enumerate(subset)}
        reaches = np.zeros((len(subset), len(subset)), dtype=bool)        # reaches[a, b]: a is an ancestor of b
        for b, child in enumerate(subset):
            for i in self._reachable(incoming, idx[child.unique_unlabelled_id]):
                if i in positions:
                    reaches[positions[i], b] = True
        G_simplified = nx.DiGraph()
        G_simplified.add_nodes_from(subset)
        for a in range(len(subset)):
            for b in range(len(subset)):
                if reaches[a, b] and not (reaches[a] & reaches[:, b]).any():
                    G_simplified.add_edge(subset[a], subset[b])
        return G_simplified


    @cached_property
    def singleton_equivalence_classes(self):
        return [next(iter(eqclass)) for eqclass in self.equivalence_classes_as_mDAGs if len(eqclass) == 1]
    
    @cached_property
    def latent_free_eqclasses_picklist(self):
        return [any(mdag.latent_free_graphQ for mdag in eqclass) for eqclass in self.equivalence_classes_as_mDAGs]

    @cached_property
    def latent_free_eqclasses(self):
        return [eqclass for eqclass,latent_free_Q in
                zip(self.equivalence_classes_as_mDAGs, self.latent_free_eqclasses_picklist) if latent_free_Q]
    
    @cached_property
    def latent_free_representative_mDAGs_list(self):
    #     return self.representatives(self.equivalence_classes_as_mDAGs)
        return self.smart_representatives(self.latent_free_eqclasses, 'relative_complexity_for_sat_solver')

    @cached_property
    def NOT_latent_free_eqclasses_picklist(self):
        return [not any(mdag.latent_free_graphQ for mdag in eqclass) for eqclass in self.equivalence_classes_as_mDAGs]

    @cached_property
    def NOT_latent_free_eqclasses(self):
        # return list(filter(lambda eqclass: all(mdag.fundamental_graphQ for mdag in eqclass),
        #                    self.equivalence_classes_as_mDAGs))
        return [eqclass for eqclass,not_latent_free_Q in
                zip(self.equivalence_classes_as_mDAGs, self.NOT_latent_free_eqclasses_picklist) if not_latent_free_Q]

    @cached_property
    def boring_by_virtue_of_Evans(self):
        return set().union(*self.latent_free_eqclasses)

    @cached_property
    def NOT_latent_free_representative_mDAGs_list(self):
    #     return self.representatives(self.equivalence_classes_as_mDAGs)
        return self.smart_representatives(self.NOT_latent_free_eqclasses, 'relative_complexity_for_sat_solver')


    @cached_property
    def foundational_eqclasses_picklist(self):
        return [all(mdag.fundamental_graphQ for mdag in eqclass) for eqclass in self.equivalence_classes_as_mDAGs]

    @cached_property
    def foundational_eqclasses(self):
        # return list(filter(lambda eqclass: all(mdag.fundamental_graphQ for mdag in eqclass),
        #                    self.equivalence_classes_as_mDAGs))
        return [eqclass for eqclass,foundational_Q in
                zip(self.equivalence_classes_as_mDAGs, self.foundational_eqclasses_picklist) if foundational_Q]

    @staticmethod
    def representatives(eqclasses):
        return [next(iter(eqclass)) for eqclass in eqclasses]

    @staticmethod
    def smart_representatives(eqclasses, attribute):
        return [min(eqclass, key = lambda mdag: mdag.__getattribute__(attribute)) for eqclass in eqclasses]

    @cached_property
    def representative_mDAGs_not_necessarily_foundational(self):
    #     return self.representatives(self.equivalence_classes_as_mDAGs)
        return self.smart_representatives(self.equivalence_classes_as_mDAGs, 'relative_complexity_for_sat_solver')

    @cached_property
    def representatives_not_even_one_fundamental_in_class(self):
        representatives_of_NOT_entirely_fundamental_classes=[element for element in self.representative_mDAGs_not_necessarily_foundational if element not in self.representative_mDAGs_list]   
        not_even_one_fundamental_in_class=[]
        for mdag in representatives_of_NOT_entirely_fundamental_classes:
            for c in self.equivalence_classes_as_mDAGs:
                if mdag in c:
                    not_one_fund=True
                    for equivalent_mDAG in c:
                        if equivalent_mDAG.fundamental_graphQ:
                            not_one_fund=False
                    if not_one_fund:
                        not_even_one_fundamental_in_class.append(mdag)
        return not_even_one_fundamental_in_class


    # @cached_property
    # def safe_representative_mDAGs_list(self):
    #     return self.representatives(self.safe_equivalence_classes_as_mDAGs)

    @cached_property
    def representative_mDAGs_list(self):
        if self.fully_foundational:
            return self.smart_representatives(self.foundational_eqclasses,
                                              'relative_complexity_for_sat_solver')
        else:
            return self.smart_representatives(self.equivalence_classes_as_mDAGs,
                                              'relative_complexity_for_sat_solver')
    
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
        if self.fully_foundational:
            return self.smart_representatives(self.foundational_eqclasses,
                                              'n_of_edges')
        else:
            return self.smart_representatives(self.equivalence_classes_as_mDAGs,
                                              'n_of_edges')
        # return self.smart_representatives(self.equivalence_classes_as_mDAGs, 'n_of_edges')

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
        return all(self.equivalent_to_only_hypergraph_representative[mdag] for mdag in eqclass)

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
        # for i, mdag in enumerate(self.representatives_for_only_hypergraphs):
        #     if mdag.n_of_edges==0:
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
        for mdag in self.representative_mDAGs_list:
            d[tuple(mdag.__getattribute__(prop) for prop in joint_attributes)].add(mdag)
        d2 = defaultdict(dict)
        for key_tuple, partition in d.items():
            d2[key_tuple[:critical_range]][key_tuple] = tuple(partition)
        return [val for val in d2.values() if len(val) > 1]

class Observable_labelled_mDAGs(Observable_unlabelled_mDAGs):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @cached_property
    def all_labelled_mDAGs(self):
        return [mDAG(ds, sc) for sc, ds in itertools.product(self.all_simplicial_complices,
                                                                 self.truly_all_directed_structures)]
    @cached_property
    def most_labelled_mDAGs(self):
        return self.all_labelled_mDAGs
    @cached_property
    def dict_id_to_canonical_id(self):
        # if self.verbose:
        #     print("Dictionary creations: mDAG ids to unlabelled ids...", flush=True)
        return {mdag.unique_id: mdag.unique_id for mdag in self.all_labelled_mDAGs}

    def canonical_id(self, ds_int, sc_int):
        return self.mdag_int_pair_to_single_int(ds_int, sc_int)    # labelled: every mDAG is its own class

    @cached_property
    def dict_uniqind_unlabelled_mDAGs(self):
        return {mdag.unique_id: mdag for mdag in explore_candidates(
            self.most_labelled_mDAGs,
            verbose=self.verbose,
            message="Mapping unlabelled ids to mDAGs"
        )}
    @cached_property
    def directed_dominances_up_to_symmetry(self):
        return self.directed_dominances

    @cached_property
    def truly_all_directed_structures(self):
        truly_all = []
        for ds in self.all_directed_structures:
            truly_all.extend(ds.equivalents_under_symmetry)
        return truly_all

    @cached_property
    def all_unlabelled_directed_structures(self):
        return self.truly_all_directed_structures

class Observable_mDAGs_Analysis(Observable_unlabelled_mDAGs):
    def __init__(self, nof_observed_variables=4, max_nof_events_for_supports=3, fully_foundational=False):
        super().__init__(nof_observed_variables, fully_foundational=fully_foundational)
        self.max_nof_events = max_nof_events_for_supports


        print("Number of unlabelled graph patterns: ", len(self.all_unlabelled_mDAGs), flush=True)
        fundamental_list = [mdag.fundamental_graphQ for mdag in self.all_unlabelled_mDAGs]
        if self.fully_foundational:
            print("Number of fundamental unlabelled graph patterns: ", len(np.flatnonzero(fundamental_list)), flush=True)
            print(
                "Upper bound on number of 100% foundational equivalence classes: ",
                len(self.foundational_eqclasses),
                flush=True)
            print("Number of Foundational CI classes: ", len(self.CI_classes))
        else:
            print("Number of distinct CI classes: ", len(self.CI_classes))
            print("Upper bound on number of equivalence classes: ", len(self.equivalence_classes_as_mDAGs), flush=True)



        self.singletons_dict = dict({1: list(itertools.chain.from_iterable(
            filter(lambda eqclass: (len(eqclass) == 1 or self.effectively_all_singletons(eqclass)),
                   self.esep_classes)))})
        self. non_singletons_dict = dict({1: sorted(
            filter(lambda eqclass: (len(eqclass) > 1 and not self.effectively_all_singletons(eqclass)),
                   self.esep_classes), key=len)})
        print("# of singleton classes from ESEP+Prop 6.8: ", len(self.singletons_dict[1]))
        print("# of non-singleton classes from ESEP+Prop 6.8: ", self.lowerbound_count_accounting_for_hypergraph_inequivalence(self.non_singletons_dict[1]),
              ", comprising {} total graph patterns (no repetitions)".format(
                  ilen(itertools.chain.from_iterable(self.non_singletons_dict[1]))))
        
        
        self.singletons_dict = dict({1: list(itertools.chain.from_iterable(
            filter(lambda eqclass: (len(eqclass) == 1 or self.effectively_all_singletons(eqclass)),
                   self.Dense_connectedness_and_esep)))})
        self.non_singletons_dict = dict({1: sorted(
            filter(lambda eqclass: (len(eqclass) > 1 and not self.effectively_all_singletons(eqclass)),
                   self.Dense_connectedness_and_esep), key=len)})
        print("# of singleton classes from also considering Dense Connectedness: ", len(self.singletons_dict[1]))
        print("# of non-singleton classes from also considering Dense Connectedness: ", self.lowerbound_count_accounting_for_hypergraph_inequivalence(self.non_singletons_dict[1]),
              ", comprising {} total graph patterns (no repetitions)".format(
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
                  ", comprising {} total graph patterns".format(
                      ilen(itertools.chain.from_iterable(self.non_singletons_dict[k]))))     
   




class QuantumMetagraph(Observable_unlabelled_mDAGs):
    """The metagraph under the quantum rule: its strongly connected components are proven quantum observational
    equivalence classes (each inside one classical class)."""
    def __init__(self, n: int, verbose: bool = False, **kwargs) -> None:
        super().__init__(n, verbose=verbose, rule='quantum', **kwargs)


class ClassicalMetagraph(Observable_unlabelled_mDAGs):
    """The metagraph under the classical rule (every HLP edge)."""
    def __init__(self, n: int, verbose: bool = False, **kwargs) -> None:
        super().__init__(n, verbose=verbose, rule='classical', **kwargs)

    @classmethod
    def from_quantum(cls, quantum: "Observable_unlabelled_mDAGs") -> "ClassicalMetagraph":
        """The classical metagraph on the nodes of `quantum`, sharing its dominances and face-splitting edges and
        adding the unrestricted HLP edges."""
        assert quantum.rule == 'quantum'
        return cls.from_sibling(quantum, 'classical')

    def within_class_above(self, uid):
        """Members of uid's class that are structurally above uid: gap(uid) implies gap of each of them, since
        their quantum model contains uid's and their classical model equals it."""
        return self.structurally_above(uid) & self.class_of(uid)

    def within_class_below(self, uid):
        """Members of uid's class that are structurally below uid: a gap of any of them implies gap(uid)."""
        return self.structurally_below(uid) & self.class_of(uid)


if __name__ == "__main__":
    test = Observable_unlabelled_mDAGs(3, verbose=2)
    print(len(test.all_unlabelled_mDAGs))
