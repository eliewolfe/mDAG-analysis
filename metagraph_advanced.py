from __future__ import absolute_import
import networkx as nx
import numpy as np
import itertools
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
                    'latent_free_DAGs_unlabelled', 'structural_order', '_structural_closure')
    # What a sibling must share for the two metagraphs to agree (the rest is shared when already computed).
    FORCED_SHARED = ('canonical_forms', 'dict_uniqind_unlabelled_mDAGs', 'structural_dominances',
                     'face_splitting_edge_list', 'structural_order')

    def __init__(self, n: int,
                 fully_foundational=False,
                 all_dominances=False,
                 verbose=True,
                 rule: str = 'classical'):
        """
        Parameters
        ----------
        n : int
            The number of observed variables.
        rule : 'classical' or 'quantum'
            Which model sets the metagraph's equivalences are sound for (see the class docstring).
        """
        assert rule in ('classical', 'quantum'), rule
        self.n = n
        self.rule = rule
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
                                             all_dominances=other.all_dominances, verbose=other.verbose, rule=rule, **kwargs)
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


    @property
    def raw_HLP_edges(self):
        for (id, mdag) in explore_candidates(
                self.dict_ind_unlabelled_mDAGs.items(),
                verbose=self.verbose,
                message=f"Finding HLP Rule #4 type edges ({self.rule} rule)"):  # NEW: Over graph patterns only
            for id2 in mdag.weaker_mDAG_HLP(self.rule):
                yield (id2, id)

    @property
    def raw_FaceSplitting_edges(self):
        for (id, mdag) in explore_candidates(
                self.dict_ind_unlabelled_mDAGs.items(),
                verbose=self.verbose,
                message="Finding metagraph edges due to face splitting"):  # NEW: Over graph patterns only
            for id2 in mdag.generate_weaker_mDAGs_FaceSplitting('strong'):
                yield (id2, id)

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

    @cached_property
    def structural_dominances(self):
        """(strong, weak) canonical-id pairs: weak is strong with one maximal face or one edge fewer (a containment
        of both the classical and the quantum model set). Generated from each node's lower neighbours (drop one
        edge; drop one maximal face) and canonicalised, which gives the same set as the former all-pairs
        hypergraph_dominances x directed structures and directed_dominances_up_to_symmetry x complices.
        Rule-independent, shared between the two metagraphs."""
        if self.verbose:
            eprint('Adding dominance relations...', flush=True)
        forms = self.canonical_forms
        pairs = set()
        for strong, mdag in explore_candidates(self.dict_uniqind_unlabelled_mDAGs.items(),
                                               verbose=self.verbose, message="Structural dominances"):
            ds_int = mdag.directed_structure_instance.as_integer
            sc_int = mdag.simplicial_complex_instance.as_integer
            for d, s in forms.lower_neighbour_parts(ds_int, sc_int):
                pairs.add((strong, self.canonical_id(d, s)))
        return list(pairs)

    @cached_property
    def legacy_structural_dominances(self):
        """The structural dominances by the former all-pairs construction (kept to cross-check the new one)."""
        directed_ids = np.array(tuple(ds.as_integer for ds in self.all_unlabelled_directed_structures), dtype=object)[:, np.newaxis, np.newaxis]
        hypergraph_dominances = np.asarray(self.hypergraph_dominances, dtype=object)
        from_hypergraphs = np.reshape(self.mdag_int_pair_to_single_int(directed_ids, hypergraph_dominances), (-1, 2))
        simplicial_ids = np.array(tuple(sc.as_integer for sc in self.all_simplicial_complices), dtype=object)[:, np.newaxis, np.newaxis]
        directed_dominances = np.asarray(self.directed_dominances_up_to_symmetry, dtype=object)
        from_directed_edges = np.reshape(self.mdag_int_pair_to_single_int(directed_dominances, simplicial_ids), (-1, 2))
        return list(self._canonical_pairs(from_hypergraphs.tolist())) + list(self._canonical_pairs(from_directed_edges.tolist()))

    @cached_property
    def hlp_edge_list(self):
        """(weak, strong) canonical-id pairs of the HLP equivalence under this metagraph's rule."""
        return list(self._canonical_pairs(self.raw_HLP_edges))

    @cached_property
    def face_splitting_edge_list(self):
        """(split, original) canonical-id pairs of the face-splitting equivalence (quantum-valid as well: the
        source of the part D distributes one conditional state per value of what every member of D sees of C)."""
        return list(self.FaceSplitting_edges)

    @cached_property
    def all_HLP_relevant_dominances(self):
        """(structural dominances, HLP equivalence edges), both as canonical-id pairs."""
        return (self.structural_dominances, self.hlp_edge_list)

    @property
    def boring_dominances(self):
        (equiv_breaking_doms, equiv_maybe_preserving_doms) = self.all_HLP_relevant_dominances
        equiv_maybe_preserving_doms = list(equiv_maybe_preserving_doms)   # a copy: the cached HLP list stays intact
        if self.all_dominances:
            equiv_maybe_preserving_doms.extend(equiv_breaking_doms)
            return equiv_maybe_preserving_doms
        for dom_ids in explore_candidates(equiv_breaking_doms,
                                          verbose=self.verbose,
                                          message="Finding dominances which preserve CI"):
            (strong_mDAG, weak_mDAG) = self.lookup_mDAG(dom_ids)
            # The skeleton (adjacency = no e-separation, Evans) and the CI count are invariants of observational
            # equivalence: a dominance between structures that differ in either is never an equivalence.
            if strong_mDAG.skeleton_nof_edges == weak_mDAG.skeleton_nof_edges:
                if strong_mDAG.CI_count == weak_mDAG.CI_count:
                    equiv_maybe_preserving_doms.append(dom_ids)
        return equiv_maybe_preserving_doms

    @cached_property
    def HLP_meta_graph(self):
        g = nx.DiGraph()
        g.add_nodes_from(self.metagraph_nodes.flat)
        # gc.collect(generation=2)
        # if self.verbose:
        #     eprint('Adding dominance relations...', flush=True)
        # # raw_dir_dom = tuple(self._all_directed_dominances)
        # # raw_hyp_dom = tuple(self._all_hypergraph_dominances)
        # # dedup_dir_dom = set(raw_dir_dom)
        # # dedup_hyp_dom = set(raw_hyp_dom)
        # # eprint(f"Directed dominances before and after deduplication: {len(raw_dir_dom)} vs {len(dedup_dir_dom)}")
        # # eprint(f"Hypergraph dominances before and after deduplication: {len(raw_hyp_dom)} vs {len(dedup_hyp_dom)}")
        # # g.add_edges_from(dedup_dir_dom.union(dedup_hyp_dom))
        # g.add_edges_from(set(self.boring_dominances))
        # gc.collect(generation=2)
        # # edge_count = g.number_of_edges()
        # if self.verbose:
        #     eprint('Adding HLP equivalence relations...', flush=True)
        # g.add_edges_from(zip(*self.all_HLP_relevant_dominances))
        g.add_edges_from(self.boring_dominances)
        # gc.collect(generation=2)
        # new_edge_count =  g.number_of_edges()
        # if self.verbose:
        #     print('Number of HLP equivalence relations added: ', new_edge_count-edge_count)
        # if self.verbose:
        #     print('HLP Metagraph has been constructed. Total edge count: ', g.number_of_edges(), flush=True)
        return g

    @cached_property
    def _dict_id_to_idx(self):
        return {id: i for i, id in enumerate(self.metagraph_nodes.flat)}




    @cached_property
    def HLP_meta_adjmat(self):
        (pre_row, pre_col) = self.all_HLP_relevant_dominances
        all_idxs = tuple(range(self.num_metagraph_nodes))
        col = partsextractor(self._dict_id_to_idx, pre_row) + all_idxs
        row = partsextractor(self._dict_id_to_idx, pre_col) + all_idxs
        return sparse.coo_matrix((np.ones(len(row), dtype=bool), (row, col)),
                                        shape=(self.num_metagraph_nodes, self.num_metagraph_nodes),
                                 dtype=bool)

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
        # proven_boring = set()
        # for lf_id, lf_CI in explore_candidates(self.latent_free_DAG_ids_and_CI_unlabelled.items(),
        #                                        verbose=self.verbose,
        #                                        message="Running HLP in reverse"):
        #
        #     idxs_dominated_by_LF = self.HLP_meta_adjmat_closure[self._dict_id_to_idx[lf_id]].tocoo().row
        #     ids_dominated_by_LF = self.metagraph_nodes[idxs_dominated_by_LF].tolist()
        #     mdags_dominated_by_LF = partsextractor(self.dict_uniqind_unlabelled_mDAGs, ids_dominated_by_LF)
        #     mdags_dominated_by_LF = [m for m in mdags_dominated_by_LF if m.all_CI_unlabelled==lf_CI]
        #     proven_boring.update(mdags_dominated_by_LF)
        # return proven_boring
        # HLP_meta_graph = nx.transitive_closure(self.HLP_meta_graph, reflexive=True)
        HLP_meta_graph = self.HLP_meta_graph.copy()
        proven_boring = set()
        for lf_id, lf_CI in explore_candidates(self.latent_free_DAG_ids_and_CI_unlabelled.items(),
                                               verbose=self.verbose,
                                               message="Running HLP in reverse"):
            ids_dominated_by_LF = {lf_id}.union(nx.ancestors(HLP_meta_graph, lf_id))
            # ids_dominated_by_LF = tuple(HLP_meta_graph.predecessors(lf_id))
            dominated_by_LF = partsextractor(self.dict_uniqind_unlabelled_mDAGs, ids_dominated_by_LF)
            if self.all_dominances:
                dominated_by_LF = [m for m in dominated_by_LF if m.all_CI_unlabelled==lf_CI]
            HLP_meta_graph.remove_nodes_from(m.unique_unlabelled_id for m in dominated_by_LF)
            proven_boring.update(dominated_by_LF)
        return proven_boring

    @cached_property
    def meta_graph(self):
        if self.rule == 'quantum':
            # Every edge is a sound containment of quantum model sets; no CI filter is needed for soundness, and the
            # classical companion (from_quantum) adds the unrestricted HLP edges to this graph.
            g = nx.DiGraph()
            g.add_nodes_from(self.metagraph_nodes.flat)
            g.add_edges_from(self.structural_dominances)
            g.add_edges_from(self.hlp_edge_list)
            g.add_edges_from(self.face_splitting_edge_list)
            return g
        g = self.HLP_meta_graph.copy()
        # edge_count = g.number_of_edges()
        if self.verbose:
            eprint('Adding FaceSplitting equivalence relations...', flush=True)
        g.add_edges_from(self.face_splitting_edge_list)
        gc.collect(generation=2)
        # new_edge_count = g.number_of_edges()
        # if self.verbose:
        #     print('Number of FaceSplitting equivalence relations added: ', new_edge_count - edge_count)
        #     print('Full metagraph has been constructed. Total edge count: ', g.number_of_edges(), flush=True)
        return g         

    @cached_property
    def equivalence_classes_as_ids(self):
        return list(nx.strongly_connected_components(self.meta_graph))

    # ------------------------------------------------------------------
    # Class lookups shared by the quantum and the classical metagraph
    # ------------------------------------------------------------------

    @cached_property
    def class_index_of(self):
        """unlabelled mDAG id -> index of its equivalence class in equivalence_classes_as_ids."""
        return {uid: k for k, cls in enumerate(self.equivalence_classes_as_ids) for uid in cls}

    def class_of(self, uid):
        """The equivalence class (a set of unlabelled ids) of an unlabelled mDAG id."""
        return self.equivalence_classes_as_ids[self.class_index_of[uid]]

    @cached_property
    def latent_free_equivalent_ids(self):
        """Members of the classes that contain a latent-free DAG (under this metagraph's rule). Under the
        classical rule these are the provably algebraic structures, for which Q(G) = C(G)."""
        latent_free = {m.unique_unlabelled_id for m in self.latent_free_DAGs_unlabelled}
        return set().union(*(cls for cls in self.equivalence_classes_as_ids if cls & latent_free)) \
            if self.equivalence_classes_as_ids else set()

    @cached_property
    def structural_order(self):
        """The structural dominances as a digraph (strong -> weak): u reaches v iff v is u with facets or edges
        removed, so model(u) contains model(v) classically and quantumly."""
        g = nx.DiGraph()
        g.add_nodes_from(self.metagraph_nodes.flat)
        g.add_edges_from(self.structural_dominances)
        return g

    @cached_property
    def _structural_closure(self):
        """Transitive closure of the structural order as bitsets: (index of each id, below[i] = bitmask of the
        indices structurally below id i, above[i] likewise), computed once in topological order."""
        order = list(nx.topological_sort(self.structural_order))
        index = {uid: i for i, uid in enumerate(order)}
        below = [0] * len(order)
        for uid in reversed(order):
            i = index[uid]
            mask = 0
            for child in self.structural_order.successors(uid):
                j = index[child]
                mask |= (1 << j) | below[j]
            below[i] = mask
        above = [0] * len(order)
        for uid in order:
            i = index[uid]
            mask = 0
            for parent in self.structural_order.predecessors(uid):
                j = index[parent]
                mask |= (1 << j) | above[j]
            above[i] = mask
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



    # @cached_property
    # def safe_equivalence_classes_as_mDAGs(self):
    #     return [self.lookup_mDAG(eqclass) for eqclass in self.safe_equivalence_classes]

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
    
    
    
    
    def partial_order_of_subset(self, subset):   #subset is a set of mDAGs (i.e. nodes of the metagraph)
        G = nx.DiGraph()
        for g in subset:
            G.add_node(g)
        for child in subset:
            for potential_ancestor in subset:
               if potential_ancestor.unique_unlabelled_id in nx.ancestors(self.meta_graph,child.unique_unlabelled_id):
                    G.add_edge(potential_ancestor,child)
        G_simplified = nx.DiGraph()                  # If there is a path g1->g2->g3, G_simplified gets rid of the direct arrow g1->g3
        for n1 in G.nodes():
            for n2 in G.nodes():
                number_paths=0
                for path in nx.all_simple_paths(G,n1,n2):
                    number_paths=number_paths+1
                if number_paths==1:
                    G_simplified.add_edge(n1,n2)
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
