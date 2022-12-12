from __future__ import absolute_import

import itertools
from sys import hexversion

import methodtools
import numpy as np
import progressbar
from pysat.formula import IDPool  # I wonder if we can't get away without this, but it is SO convenient
from pysat.solvers import Solver

from radix import from_digits, to_digits
from utilities import partsextractor

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
from typing import List, Tuple

try:
    import networkx as nx
    from networkx.algorithms.isomorphism import DiGraphMatcher
except ImportError:
    print("Functions which depend on networkx are not available.")

from scipy.special import comb

def infer_automorphisms(parents_of):
    visible_node_count = len(parents_of)
    total_node_count = max(itertools.chain.from_iterable(parents_of)) + 1
    g = nx.DiGraph()
    g.add_nodes_from(range(total_node_count))
    for i, parents in enumerate(parents_of):
        for p in parents:
            g.add_edge(p, i)
    observed_nodes = tuple(range(visible_node_count))
    visible_automorphisms = set()
    for mapping in DiGraphMatcher(g, g).isomorphisms_iter():
        visible_automorphisms.add(partsextractor(mapping, observed_nodes))
    return tuple(visible_automorphisms)

# def depth(l):
#     if hasattr(l, '__iter__'):
#         if len(l):
#             return 1 + max(depth(item) for item in l)
#         else:
#             return 1
#     else:
#         return 0

class SupportTester(object):
    def __init__(self,
                 parents_of: Tuple,
                 observed_cardinalities: np.ndarray,
                 nof_events: int, **kwargs):
        # For Victor Gitton:
        # if depth(parents_of) == 2:
        #     self.parents_of = parents_of
        #     self.causal_symmetries_to_enforce = [self.parents_of]
        # elif depth(parents_of) == 3:
        #     self.parents_of = parents_of[0]
        #     self.causal_symmetries_to_enforce = parents_of
        # else:
        #     assert False, "Ill formatted input!"

        self.parents_of = parents_of
        self.nof_events = nof_events
        self.nof_observed = len(self.parents_of)
        assert self.nof_observed == len(observed_cardinalities)
        self.nof_latent = max(itertools.chain.from_iterable(self.parents_of)) + 1 - self.nof_observed
        self.observed_cardinalities = np.asarray(observed_cardinalities, dtype=int)
        self.observed_cardinalities_ranges = list(map(range, self.observed_cardinalities))
        self.latent_cardinalities = np.repeat(self.nof_events, self.nof_latent)
        self.observed_and_latent_cardinalities = tuple(np.hstack((observed_cardinalities,
                                                                  self.latent_cardinalities)))

        self.max_conceivable_events = np.prod(self.observed_cardinalities)

        self.matrix_dtype = np.min_scalar_type(self.observed_cardinalities)
        self.list_dtype = np.min_scalar_type(self.max_conceivable_events)
        self.int_dtype = np.min_scalar_type(np.power(self.max_conceivable_events, self.nof_events))

        self.event_cardinalities = np.broadcast_to(self.max_conceivable_events, self.nof_events)

        self.repeated_observed_cardinalities = np.broadcast_to(np.expand_dims(self.observed_cardinalities, axis=-1),
                                                               (self.nof_observed, self.nof_events)).T.ravel()

        self.conceivable_events_range = np.arange(self.max_conceivable_events).astype(self.list_dtype)

        self.relevant_parent_cardinalities = [partsextractor(self.observed_and_latent_cardinalities,
                                                             self.parents_of[idx]) for idx in
                                              range(self.nof_observed)]

        self.binary_variables = set(np.flatnonzero(np.asarray(self.observed_cardinalities, dtype=np.intp) == 2))
        self.nonbinary_variables = set(range(self.nof_observed)).difference(self.binary_variables)
        self.vpool = IDPool(start_from=1)
        self.var = lambda idx, val, par: -self.vpool.id(
            'v[{0}]_{2}==0'.format(idx, val, par)) if idx in self.binary_variables and val == 1 else self.vpool.id(
            'v[{0}]_{2}=={1}'.format(idx, val, par))
        # self.cached_properties_computed_yet = False


    def reverse_var(self, id):
        str = self.vpool.obj(abs(id))
        if id < 0:
            return str.replace("==", "!=")
        else:
            return str

    def reverse_vars(self, stuff):
        if hasattr(stuff, '__iter__'):
            return [self.reverse_vars(substuff) for substuff in stuff]
        else:
            return self.reverse_var(stuff)

    @cached_property
    def at_least_one_outcome(self):
        return [[self.var(idx, val, par) for val in self.observed_cardinalities_ranges[idx]]
                for idx in self.nonbinary_variables
                for par in np.ndindex(self.relevant_parent_cardinalities[idx])]

    def from_list_to_matrix(self, supports_as_lists: np.ndarray) -> np.ndarray:
        return to_digits(supports_as_lists, self.observed_cardinalities).astype(self.matrix_dtype)

    def from_matrix_to_list(self, supports_as_matrices: np.ndarray) -> np.ndarray:
        return from_digits(supports_as_matrices, self.observed_cardinalities).astype(self.list_dtype)

    def from_list_to_integer(self, supports_as_lists: np.ndarray) -> np.ndarray:
        return from_digits(supports_as_lists, self.event_cardinalities).astype(self.int_dtype)

    def from_integer_to_list(self, supports_as_integers: np.ndarray) -> np.ndarray:
        return to_digits(supports_as_integers, self.event_cardinalities).astype(self.list_dtype)

    def from_matrix_to_integer(self, supports_as_matrices: np.ndarray) -> np.ndarray:
        supports_as_matrices_as_array = np.asarray(supports_as_matrices, dtype=self.matrix_dtype)
        shape = supports_as_matrices_as_array.shape
        return from_digits(
            supports_as_matrices_as_array.reshape(shape[:-2] + (np.prod(shape[-2:]),)),
            self.repeated_observed_cardinalities).astype(self.int_dtype)

    def from_integer_to_matrix(self, supports_as_integers: np.ndarray) -> np.ndarray:
        return np.reshape(to_digits(
            supports_as_integers, self.repeated_observed_cardinalities).astype(self.matrix_dtype),
            np.asarray(supports_as_integers, dtype=self.int_dtype).shape + (self.nof_events, self.nof_observed))

    @methodtools.lru_cache(maxsize=None, typed=False)
    def forbidden_event_clauses(self, event: int):
        """Get the clauses associated with a particular event not occurring anywhere in the off-diagonal worlds."""
        forbidden_event_as_row = self.from_list_to_matrix(event)
        forbidden_event_clauses = []
        observed_iterator_as_list = forbidden_event_as_row.tolist()
        for latent_iterator in itertools.product(
                range(self.nof_events),
                repeat=self.nof_latent):
            iterator = observed_iterator_as_list.copy()
            iterator.extend(latent_iterator)
            no_go_clause = tuple(sorted(-self.var(
                i, val, partsextractor(iterator, self.parents_of[i]))
                                        for i, val in enumerate(
                forbidden_event_as_row.flat)))
            forbidden_event_clauses.append(no_go_clause)
        return forbidden_event_clauses
    def forbidden_events_clauses(self, occurring_events: np.ndarray) -> List:
        """Get the clauses associated with all nonoccurring event as not occurring anywhere in the off-diagonal worlds."""
        occurring_events_as_list = self.from_matrix_to_list(occurring_events)
        forbidden_events_as_list = np.setdiff1d(
            self.conceivable_events_range,
            occurring_events_as_list,
            assume_unique=True)
        return list(itertools.chain.from_iterable([
            self.forbidden_event_clauses(event) for event in forbidden_events_as_list.flat
        ]))

    @methodtools.lru_cache(maxsize=None, typed=False)
    def positive_outcome_clause(self,
                                 world: int,
                                 outcomes: Tuple):
        clauses = []
        iterator_plus = tuple(outcomes) + tuple(np.repeat(world, self.nof_latent))
        for idx in self.nonbinary_variables:
            # NEXT LINE IS OPTIONAL, CAN BE COMMENTED OUT
            clauses.append( [self.var(idx, outcomes[idx], partsextractor(iterator_plus,
                                                               self.parents_of[
                                                                   idx]))] )
            wrong_values = set(range(self.observed_cardinalities[idx]))
            wrong_values.remove(outcomes[idx])
            for wrong_value in wrong_values:
                # if wrong_value != iterator[idx]:
                clauses.append( [-self.var(idx, wrong_value,
                                 partsextractor(iterator_plus,
                                                self.parents_of[idx]))] )
        for idx in self.binary_variables:
            clauses.append( [self.var(idx, outcomes[idx], partsextractor(iterator_plus,
                                                               self.parents_of[
                                                                   idx]))] )
        return clauses

    def array_of_positive_outcomes(self, occurring_events: np.ndarray) -> List:
        return [clause for world, outcomes in enumerate(occurring_events)
                for clause in self.positive_outcome_clause(world, tuple(outcomes))]
        # for world, outcomes in enumerate(occurring_events):
        #     for clause in self.positive_outcome_clause(world, tuple(outcomes)):
        #         yield clause

    def _sat_solver_clauses(self, occurring_events: np.ndarray) -> List:
        assert self.nof_events == len(occurring_events), 'The number of events does not match the expected number.'
        return self.array_of_positive_outcomes(occurring_events) + \
               self.at_least_one_outcome + \
               self.forbidden_events_clauses(occurring_events)

    def feasibleQ_from_matrix(self, occurring_events: np.ndarray, **kwargs) -> bool:
        with Solver(bootstrap_with=self._sat_solver_clauses(occurring_events), **kwargs) as s:
            # if not self.cached_properties_computed_yet:
            #     # print("Initialization of problem complete.")
            #     self.cached_properties_computed_yet = True
            return s.solve()

    @methodtools.lru_cache(maxsize=None, typed=False)
    def feasibleQ_from_tuple(self, occurring_events_as_tuple: np.ndarray, **kwargs) -> bool:
        return self.feasibleQ_from_matrix(self.from_list_to_matrix(occurring_events_as_tuple), **kwargs)

    @methodtools.lru_cache(maxsize=None, typed=False)
    def feasibleQ_from_integer(self, occurring_events_as_integer, **kwargs) -> bool:
        return self.feasibleQ_from_matrix(self.from_integer_to_matrix(occurring_events_as_integer), **kwargs)

    @methodtools.lru_cache(maxsize=None, typed=False)
    def subSupportTester(self, n: int):
        if n != self.nof_events:
            return self.__class__(self.parents_of, self.observed_cardinalities, n)
        else:
            return self


    def feasibleQ_from_matrix_CONSERVATIVE(self, occurring_events: np.ndarray,
                                           min_definite=2,
                                           max_definite=np.inf,
                                           always_include=tuple(),
                                           **kwargs) -> bool:
        sanitized_always_include = tuple(map(tuple, always_include))
        others_to_potentially_include = tuple(set(map(tuple, occurring_events)).difference(
            sanitized_always_include))
        fixed_count = len(always_include)
        flex_count = len(others_to_potentially_include)
        for n in range(
                max(min_definite, 2),
                min(max_definite, len(occurring_events)) + 1):
            subSupportTester = self.subSupportTester(n)
            max_to_check = comb(flex_count, n - fixed_count, exact=True)
            with progressbar.ProgressBar(max_value=max_to_check) as bar:
                for i, extra_occurring_events in enumerate(itertools.combinations(others_to_potentially_include, n-fixed_count)):
                    definitely_occurring_events = np.array(
                        sanitized_always_include + extra_occurring_events,
                        dtype=self.matrix_dtype)
                    passes_inflation_test = subSupportTester.potentially_feasibleQ_from_matrix_pair(
                        definitely_occurring_events_matrix=definitely_occurring_events,
                        potentially_occurring_events_matrix=occurring_events,
                        **kwargs)
                    bar.update(i)
                    if not passes_inflation_test:
                        print("Got one! Rejected a support of ", self.nof_events, " events using level ", n, " inflation.")
                        print("Rejected the support by requiring the following events to occur in diagonal worlds:")
                        print(definitely_occurring_events)
                        return passes_inflation_test

    def _sat_solver_clauses_bonus(self,
                                  definitely_occurring_events: np.ndarray,
                                  potentially_occurring_events: np.ndarray) -> List:
        """
        Intended to analyze many events with few-event hot_start. Infeasible test, but cannot guarantee feasible.
        (We could write another version that requires the missing events to show up somewhere in the off diagonal
         worlds, but that seems complicated.)
        """
        assert self.nof_events == len(
            definitely_occurring_events), 'The number of events does not match the expected number.'
        return list(self.array_of_positive_outcomes(definitely_occurring_events)) + \
               self.at_least_one_outcome + \
               self.forbidden_events_clauses(potentially_occurring_events)

    def potentially_feasibleQ_from_matrix_pair(self,
                                               definitely_occurring_events_matrix: np.ndarray,
                                               potentially_occurring_events_matrix: np.ndarray,
                                               return_model=False,
                                               **kwargs) -> bool:
        new_support_as_tuples = set(map(tuple, potentially_occurring_events_matrix))
        definite_events_as_tuples = set(map(tuple, definitely_occurring_events_matrix))
        assert definite_events_as_tuples.issubset(new_support_as_tuples), "Input is not of format where definite events are a subset of the potential events."
        with Solver(bootstrap_with=self._sat_solver_clauses_bonus(definitely_occurring_events_matrix,
                                                                  potentially_occurring_events_matrix), **kwargs) as s:
            # if not self.cached_properties_computed_yet:
            #     # print(f"Initialization of problem complete. ({self.nof_events} events)")
            #     self.cached_properties_computed_yet = True
            sol = s.solve()
            if (not return_model) or (not sol):
                return sol
            else:
                return s.get_model()


    @staticmethod
    def generate_supports_satisfying_pp_restriction(i: int, pp_vector: np.ndarray,
                                                    candidate_support_matrices: np.ndarray,
                                                    **kwargs) -> np.ndarray:
        for s in explore_candidates(candidate_support_matrices,
                                    message='Enforce ' + str(
                                        i) + ' is perf. pred. by ' + np.array_str(
                                        pp_vector),
                                    **kwargs):
            s_resorted = s[np.argsort(s[:, i])]
            partitioned = [np.vstack(tuple(g))[:, pp_vector] for k, g in
                           itertools.groupby(s_resorted, lambda e: e[i])]
            to_test_for_intersection = [set(map(tuple, pp_values)) for
                                        pp_values in partitioned]
            raw_length = np.sum(
                [len(vals) for vals in to_test_for_intersection])
            compressed_length = len(set().union(*to_test_for_intersection))
            if raw_length == compressed_length:
                yield s

    def extract_support_matrices_satisfying_pprestrictions(
            self,
            candidate_support_matrices_raw: np.ndarray,
            pp_restrictions: Tuple,
            verbose=True) -> np.ndarray:
        if len(pp_restrictions) > 1:
            candidate_support_matrices = candidate_support_matrices_raw.copy()
            for (i, pp_set) in pp_restrictions:
                print(
                    "Isolating candidates due to perfect prediction restriction...",
                    i, " by ", pp_set)
                pp_vector = np.array(pp_set)
                candidate_support_matrices = np.array(list(
                    self.generate_supports_satisfying_pp_restriction(i, pp_vector,
                                                                candidate_support_matrices,
                                                                verbose=verbose)),
                    dtype=self.matrix_dtype)
            return candidate_support_matrices
        else:
            return candidate_support_matrices_raw


def explore_candidates(candidates, verbose=False, message=''):
    if verbose:
        if not message:
            text = 'shape=' + str(np.asarray(candidates).shape)
        else:
            text = message
        return progressbar.progressbar(
            candidates, widgets=[
                '[', text, '] '
                , progressbar.SimpleProgress(), progressbar.Bar()
                , ' (', progressbar.ETA(), ') '])
    else:
        return candidates


class SupportTesting(SupportTester):
    def __init__(self, *args, pp_relations=tuple(), visible_automorphisms=tuple()):
        super().__init__(*args)
        self.must_perfectpredict = pp_relations
        self._canonical_under_outcome_relabelling = dict()
        self._orbit_under_external_party_relabelling = dict()
        self._orbit_under_internal_party_relabelling = dict()
        self.visible_automorphisms = visible_automorphisms


    @cached_property
    def outcome_relabelling_group(self) -> np.ndarray:
        if np.array_equiv(2, self.observed_cardinalities):
            return np.bitwise_xor.outer(self.conceivable_events_range, self.conceivable_events_range).astype(self.list_dtype)
        else:
            return self.from_matrix_to_list(np.fromiter(
                itertools.chain.from_iterable(itertools.chain.from_iterable(
                    itertools.starmap(
                        itertools.product,
                        itertools.product(*map(itertools.permutations, map(range, self.observed_cardinalities)))
                    ))),
                self.matrix_dtype).reshape((-1, self.max_conceivable_events, self.nof_observed)))

    def canonical_under_outcome_relabelling(self, n: int):
        try:
            return self._canonical_under_outcome_relabelling[n]
        except KeyError:
            l = self.from_integer_to_list(n)
            l_variants = np.take(self.outcome_relabelling_group, l, axis=-1)
            l_variants.sort(axis=-1)
            l_variants = np.unique(l_variants, axis=0)
            n_orbit = self.from_list_to_integer(l_variants)
            n_canonical = n_orbit.min()
            for n_new in n_orbit.flat:
                self._canonical_under_outcome_relabelling[n_new] = n_canonical
            return n_canonical

    @cached_property
    def internal_party_relabelling_group(self) -> np.ndarray:
        """
        Like the full_party_relabelling group, but restricted to relabellings
        of the parties consistent with the automorphism of the DAG itself.
        """
        to_reshape = self.conceivable_events_range.reshape(self.observed_cardinalities)
        perms_as_event_permutations = np.empty(
            (len(self.visible_automorphisms),
             self.max_conceivable_events),
            dtype=self.list_dtype
        )
        for i, perm in enumerate(self.visible_automorphisms):
            if np.array_equal(self.observed_cardinalities,
                              np.take(self.observed_cardinalities, perm)):
                perms_as_event_permutations[i] = to_reshape.transpose(
                    perm).ravel()
        return perms_as_event_permutations

    @cached_property
    def visible_nonautomorphisms(self):
        return [perm for perm in
                itertools.permutations(range(self.nof_observed))
                if perm not in self.visible_automorphisms]

    @cached_property
    def external_party_relabelling_group(self) -> np.ndarray:
        to_reshape = self.conceivable_events_range.reshape(self.observed_cardinalities)
        return np.fromiter(
            itertools.chain.from_iterable(
                to_reshape.transpose(perm).ravel()
                for perm in self.visible_nonautomorphisms
                if np.array_equal(
                    self.observed_cardinalities,
                    np.take(self.observed_cardinalities, perm))
            ), self.list_dtype
        ).reshape((-1, self.max_conceivable_events,))

    def orbit_under_external_party_relabelling(self, n: int):
        """
        Used for selecting optimal group element, so duplicates must be preserved.
        """
        try:
            return self._orbit_under_external_party_relabelling[n]
        except KeyError:
            l = self.from_integer_to_list(n)
            l_variants = np.take(self.external_party_relabelling_group, l, axis=-1)
            l_variants.sort(axis=-1)
            n_orbit = self.from_list_to_integer(l_variants)
            for n_new in n_orbit.flat:
                self._orbit_under_external_party_relabelling[n_new] = n_orbit
            return n_orbit

    def orbit_under_internal_party_relabelling(self, n: int):
        """
        Used for selecting canonical representative, so duplicates may be discarded.
        """
        try:
            return self._orbit_under_internal_party_relabelling[n]
        except KeyError:
            l = self.from_integer_to_list(n)
            l_variants = np.take(self.internal_party_relabelling_group, l, axis=-1)
            l_variants.sort(axis=-1)
            l_variants = np.unique(l_variants, axis=0)
            n_orbit = self.from_list_to_integer(l_variants)
            for n_new in n_orbit.flat:
                self._orbit_under_internal_party_relabelling[n_new] = n_orbit
            return n_orbit

    def compress_integers_into_canonical_under_independent_relabelling(self, set_of_integers: set) -> np.ndarray:
        """
        Since this is used to filter out supports which are implied by others
         plus graph symmetry, we use the internal party relabelling group as
         opposed to the full (S_N) party relabelling group.
        """
        if len(set_of_integers) and (len(self.visible_automorphisms) > 1):
            compressed = set()
            remaining_to_compress = set_of_integers.copy()
            while len(remaining_to_compress):
                m = remaining_to_compress.pop()
                m_party_variants = self.orbit_under_internal_party_relabelling(
                    m)
                m_party_variants = [self.canonical_under_outcome_relabelling(n)
                                  for n in m_party_variants.flat]
                compressed.add(min(m_party_variants))
                remaining_to_compress.difference_update(m_party_variants)
            # for m in set_of_integers:
            #     m_party_variants = self.orbit_under_internal_party_relabelling(m)
            #     canonical_rep = min(self.canonical_under_outcome_relabelling(n) for n in m_party_variants.flat)
            #     compressed.add(canonical_rep)
            return np.array(sorted(compressed), dtype=self.int_dtype)
        else:
            return np.array(sorted(set_of_integers), dtype=self.int_dtype)

    def expand_integers_from_canonical_via_internal_party_relabelling(self, list_of_integers: np.ndarray) -> np.ndarray:
        if len(list_of_integers) and (len(self.visible_automorphisms) > 1):
            expanded = set()
            remaining_to_expand = set(list_of_integers.flat)
            while len(remaining_to_expand):
                m = remaining_to_expand.pop()
                m_party_variants = self.orbit_under_internal_party_relabelling(
                    m)
                m_party_variants = [self.canonical_under_outcome_relabelling(n)
                                  for n in m_party_variants.flat]
                expanded.update(m_party_variants)
                remaining_to_expand.difference_update(m_party_variants)
            # for m in list_of_integers.flat:
            #     m_party_variants = self.orbit_under_internal_party_relabelling(m)
            #     for n in m_party_variants.flat:
            #         expanded.add(self.canonical_under_outcome_relabelling(n))
            return np.array(sorted(expanded), dtype=self.int_dtype)
        else:
            return list_of_integers

    def convert_integers_into_canonical_under_coherent_relabelling(self, list_of_integers: np.ndarray) -> np.ndarray:
        """
        This function is used to compare with other DAGs, so we invoke full S_N
        in a coherent fashion.
        Note that we can skip party relabellings which correspond to internal symmetries.
        """
        if len(list_of_integers) and len(self.external_party_relabelling_group):
            variant_lists_of_integers = np.empty(
                (len(self.external_party_relabelling_group),
                 len(list_of_integers)),
            dtype=self.int_dtype)
            for i, n in enumerate(list_of_integers.flat):
                variant_lists_of_integers[:, i] = self.orbit_under_external_party_relabelling(n)
            variant_lists_of_integers.sort(axis=-1)
            order = np.lexsort(np.rot90(variant_lists_of_integers))
            return variant_lists_of_integers[order[0]]
        else:
            return list_of_integers

    @cached_property
    def unique_candidate_supports_as_compressed_lists(self) -> np.ndarray:
        if self.max_conceivable_events > self.nof_events:
            candidates = np.pad(np.fromiter(itertools.chain.from_iterable(
                itertools.combinations(self.conceivable_events_range[1:], self.nof_events - 1)), self.list_dtype).reshape(
                (-1, self.nof_events - 1)), ((0, 0), (1, 0)), 'constant')
            if self.must_perfectpredict:
                to_filter = self.from_list_to_matrix(candidates)
                filtered = self.extract_support_matrices_satisfying_pprestrictions(to_filter, self.must_perfectpredict)
                candidates = np.array(list(map(self.from_matrix_to_list, filtered)), dtype=self.list_dtype)
            candidates_as_ints = self.from_list_to_integer(candidates)
            candidates_as_ints = set((self.canonical_under_outcome_relabelling(n) for n in candidates_as_ints.flat))
            candidates_as_ints = self.compress_integers_into_canonical_under_independent_relabelling(candidates_as_ints) # New!
            candidates = self.from_integer_to_list(candidates_as_ints)
            return candidates
        else:
            return np.empty((0, 0), dtype=self.list_dtype)

    @cached_property
    def unique_candidate_supports_as_compressed_integers(self) -> np.ndarray:
        return self.from_list_to_integer(self.unique_candidate_supports_as_compressed_lists)

    @cached_property
    def unique_candidate_supports_as_compressed_matrices(self) -> np.ndarray:
        return self.from_list_to_matrix(self.unique_candidate_supports_as_compressed_lists)

    @staticmethod
    def explore_candidates(candidates, **kwargs):
        return explore_candidates(candidates, **kwargs)

    @methodtools.lru_cache(maxsize=None, typed=False)
    def attempt_to_find_one_infeasible_support(self, **kwargs) -> np.ndarray:
        return self.attempt_to_find_one_infeasible_support_among(self.unique_candidate_supports_as_compressed_lists, **kwargs)

    def attempt_to_find_one_infeasible_support_among(
            self, candidates_as_lists: np.ndarray, verbose=False) -> np.ndarray:
        for occurring_events_as_tuple in self.explore_candidates(candidates_as_lists,
                                                                 verbose=verbose,
                                                                 message='Finding an infeasible support'):
            if not self.feasibleQ_from_tuple(occurring_events_as_tuple):
                return self.from_list_to_matrix(occurring_events_as_tuple)
        return np.empty((0, self.nof_observed), dtype=self.matrix_dtype)

    def no_infeasible_supports_among(self, candidates_as_lists, **kwargs) -> bool:
        if len(self.attempt_to_find_one_infeasible_support_among(candidates_as_lists, **kwargs)) == 0:
            return True
        else:
            return False

    @methodtools.lru_cache(maxsize=None, typed=False)
    def no_infeasible_supports(self, **kwargs) -> bool:
        return self.no_infeasible_supports_among(self.unique_candidate_supports_as_compressed_lists, **kwargs)

    def unique_infeasible_supports_as_integers_among(
            self, candidates_as_integers, verbose=False, **kwargs) -> np.ndarray:
        """This function does NOT apply expansion due to self symmetry."""
        return np.fromiter((occurring_events_as_int for occurring_events_as_int in
                            self.explore_candidates(candidates_as_integers, verbose=verbose) if
                            not self.feasibleQ_from_integer(occurring_events_as_int, **kwargs)), dtype=self.int_dtype)

    def unique_infeasible_supports_as_integers_expanded_among(self,
                                                              *args, **kwargs) -> np.ndarray:
        """This function DOES apply expansion due to self symmetry."""
        return self.expand_integers_from_canonical_via_internal_party_relabelling(
            self.unique_infeasible_supports_as_integers_among(*args, **kwargs)
        )

    @methodtools.lru_cache(maxsize=None, typed=False)
    def unique_infeasible_supports_as_expanded_integers(self, **kwargs) -> np.ndarray:
        """
        Return a signature of infeasible support for a given parents_of, observed_cardinalities, and nof_events
        """
        return self.unique_infeasible_supports_as_integers_expanded_among(self.unique_candidate_supports_as_compressed_integers, **kwargs)

    @methodtools.lru_cache(maxsize=None, typed=False)
    def unique_infeasible_supports_as_compressed_integers(self, **kwargs) -> np.ndarray:
        """
        Return infeasible support UP TO INTERNAL SYMMETRY for a given parents_of, observed_cardinalities, and nof_events
        """
        return self.unique_infeasible_supports_as_integers_among(self.unique_candidate_supports_as_compressed_integers, **kwargs)

    @methodtools.lru_cache(maxsize=None, typed=False)
    def unique_infeasible_supports_as_expanded_matrices(self, **kwargs) -> np.ndarray:
        return self.from_integer_to_matrix(self.unique_infeasible_supports_as_expanded_integers(**kwargs))
    @methodtools.lru_cache(maxsize=None, typed=False)
    def unique_infeasible_supports_as_compressed_matrices(self, **kwargs) -> np.ndarray:
        return self.from_integer_to_matrix(self.unique_infeasible_supports_as_compressed_integers(**kwargs))

    @methodtools.lru_cache(maxsize=None, typed=False)
    def unique_infeasible_supports_as_integers_unlabelled(self, **kwargs) -> np.ndarray:
        return self.convert_integers_into_canonical_under_coherent_relabelling(
            self.unique_infeasible_supports_as_expanded_integers(**kwargs))

    def unique_infeasible_supports_as_integers_independent_unlabelled(self, **kwargs) -> np.ndarray:
        return self.unique_infeasible_supports_as_compressed_integers(**kwargs)


def one_off(parents_of: Tuple, support_as_matrix: np.ndarray,
            definite_events=tuple(),
            stclass=SupportTester,
            **kwargs):
    support_as_array = np.asarray(support_as_matrix)
    (nrows, ncols) = support_as_array.shape
    new_support = np.empty((nrows, ncols), dtype=int)
    observed_cardinalities = np.empty(ncols, dtype=int)
    for i, col in enumerate(support_as_array.T):
        unique_vals, inverse = np.unique(col, return_inverse=True)
        observed_cardinalities[i] = len(unique_vals)
        new_support[:, i] = inverse
    n_definite = len(definite_events)
    if n_definite == 0:
        st = stclass(parents_of, observed_cardinalities, nof_events=nrows)
        return st.feasibleQ_from_matrix(new_support, **kwargs)
    else:
        st = stclass(parents_of, observed_cardinalities, nof_events=n_definite)
        return st.potentially_feasibleQ_from_matrix_pair(
            definitely_occurring_events_matrix=np.asarray(definite_events),
            potentially_occurring_events_matrix=new_support, **kwargs)



class CumulativeSupportTesting:
    # TODO: Add scrollbar
    def __init__(self, parents_of, observed_cardinalities, max_nof_events, **kwargs):
        self.parents_of = parents_of
        self.observed_cardinalities = observed_cardinalities
        self.nof_observed = len(self.parents_of)
        assert self.nof_observed == len(observed_cardinalities)
        self.max_nof_events = max_nof_events
        self.max_conceivable_events = np.prod(self.observed_cardinalities)
        self.multi_index_shape = np.broadcast_to(self.max_conceivable_events, self.max_nof_events)
        self.repeated_observed_cardinalities = np.broadcast_to(np.expand_dims(self.observed_cardinalities, axis=-1),
                                                               (self.nof_observed, self.max_nof_events)).T.ravel()
        self.matrix_dtype = np.min_scalar_type(self.observed_cardinalities)
        self.list_dtype = np.min_scalar_type(self.max_conceivable_events)
        self.int_dtype = np.min_scalar_type(np.power(self.max_conceivable_events, self.max_nof_events))
        self.kwargs = kwargs

    @property
    def _all_infeasible_supports(self):
        for nof_events in range(2, self.max_nof_events + 1):
            yield SupportTesting(self.parents_of, self.observed_cardinalities, nof_events, **self.kwargs
                                 ).unique_infeasible_supports_as_expanded_integers(name='mgh', use_timer=False)

    @property
    def _all_infeasible_supports_unlabelled(self):
        for nof_events in range(2, self.max_nof_events + 1):
            yield SupportTesting(self.parents_of, self.observed_cardinalities, nof_events, **self.kwargs
                                 ).unique_infeasible_supports_as_integers_unlabelled(name='mgh', use_timer=False)

    @property
    def _all_infeasible_supports_independent_unlabelled(self):
        for nof_events in range(2, self.max_nof_events + 1):
            yield SupportTesting(self.parents_of, self.observed_cardinalities, nof_events, **self.kwargs
                                 ).unique_infeasible_supports_as_integers_independent_unlabelled(name='mgh', use_timer=False)

    @cached_property
    def all_infeasible_supports(self):
        return np.fromiter(itertools.chain.from_iterable(self._all_infeasible_supports), self.int_dtype)

    @cached_property
    def all_infeasible_supports_unlabelled(self):
        return np.fromiter(itertools.chain.from_iterable(self._all_infeasible_supports_unlabelled), self.int_dtype)

    @cached_property
    def all_infeasible_supports_independent_unlabelled(self):
        return np.fromiter(itertools.chain.from_iterable(self._all_infeasible_supports_independent_unlabelled), self.int_dtype)

    def from_list_to_matrix(self, supports_as_lists: np.ndarray) -> np.ndarray:
        return to_digits(supports_as_lists, self.observed_cardinalities).astype(self.matrix_dtype)

    def from_matrix_to_list(self, supports_as_matrices: np.ndarray) -> np.ndarray:
        return from_digits(supports_as_matrices, self.observed_cardinalities).astype(self.list_dtype)

    def from_list_to_integer(self, supports_as_lists: np.ndarray) -> np.ndarray:
        return from_digits(supports_as_lists, self.multi_index_shape).astype(self.int_dtype)

    def from_integer_to_list(self, supports_as_integers: np.ndarray) -> np.ndarray:
        return to_digits(supports_as_integers, self.multi_index_shape).astype(self.list_dtype)

    def from_matrix_to_integer(self, supports_as_matrices: np.ndarray) -> np.ndarray:
        # return self.from_list_to_integer(self.from_matrix_to_list(supports_as_matrices))
        supports_as_matrices_as_array = np.asarray(supports_as_matrices)
        shape = supports_as_matrices_as_array.shape
        return from_digits(
            supports_as_matrices_as_array.reshape(shape[:-2] + (np.prod(shape[-2:]),)),
            self.repeated_observed_cardinalities)

    def from_integer_to_matrix(self, supports_as_integers: np.ndarray) -> np.ndarray:
        return np.reshape(to_digits(
            supports_as_integers, self.repeated_observed_cardinalities).astype(self.matrix_dtype),
            np.asarray(supports_as_integers).shape + (self.max_nof_events, self.nof_observed))



# TODO: Visualize an infeasible support as a SciPy sparce matrix
# TODO: Check n+k event support using n event code.

if __name__ == '__main__':
    parents_of = ([3, 4], [4, 5], [3, 5])
    print("One off: ", one_off(parents_of, [(1, 0, 0), (0, 1, 0), (0, 0, 1)]))
    print("One off: ", one_off(parents_of, [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
                               definite_events=[(1, 0, 0), (0, 1, 0)]))
    print("One off: ", one_off(parents_of, [(0, 0, 0), (0, 1, 0), (0, 0, 1)]))
    # nof_events = 3
    # st = SupportTesting(parents_of, observed_cardinalities, nof_events)
    #
    # occurring_events_temp = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    # print(st.feasibleQ_from_matrix(occurring_events_temp, name='mgh', use_timer=True))
    # occurring_events_temp = [(0, 0, 0), (0, 1, 0), (0, 0, 1)]
    # print(st.feasibleQ_from_matrix(occurring_events_temp, name='mgh', use_timer=True))

    parents_of = ([1, 3], [3, 4], [1, 4])
    visible_automorphisms = infer_automorphisms(parents_of)
    observed_cardinalities = (3, 3, 3)
    st = SupportTesting(parents_of, observed_cardinalities, 3, visible_automorphisms=visible_automorphisms)
    cst = CumulativeSupportTesting(parents_of, observed_cardinalities, 4, visible_automorphisms=visible_automorphisms)
    inf = st.unique_infeasible_supports_as_integers_unlabelled(name='mgh', use_timer=False)
    print("UC automorphisms:", st.visible_automorphisms)
    print("UC nonautomorphisms:", st.visible_nonautomorphisms)
    print(st.from_integer_to_matrix(inf))
    print(cst.all_infeasible_supports)
    print(cst.all_infeasible_supports_unlabelled)
    print(cst.all_infeasible_supports_independent_unlabelled)
    parents_of = ([3, 4], [4, 5], [5, 3])
    visible_automorphisms = infer_automorphisms(parents_of)
    cst = CumulativeSupportTesting(parents_of, observed_cardinalities, 4, visible_automorphisms=visible_automorphisms)
    print(cst.all_infeasible_supports)
    print(cst.all_infeasible_supports_unlabelled)
    print(cst.all_infeasible_supports_independent_unlabelled)

    # sample2 = st.unique_candidate_supports
    # print(st.unique_candidate_supports)
    # print(st.visualize_supports(st.unique_candidate_supports))
    # inf2 = st.unique_infeasible_supports_as_integers(name='mgh', use_timer=False)
    # print(inf2)
    # print(st.devisualize_supports(inf2))
    # print(st.extreme_devisualize_supports(inf2))
    # st = SupportTesting(parents_of, observed_cardinalities, 3)
    # # sample3 = st.unique_candidate_supports
    # inf3 = st.unique_infeasible_supports_as_integers(name='mgh', use_timer=False)
    # print(inf3)
    # print(st.devisualize_supports(inf3))
    # print(st.extreme_devisualize_supports(inf3))
    # st = SupportTesting(parents_of, observed_cardinalities, 4)
    # st = SupportTesting(parents_of, [2, 3, 3], 2)
    # print(st.outcome_relabelling_group.shape)
    # st = SupportTesting(parents_of, [2, 3, 4], 2)
    # print(st.outcome_relabelling_group.shape)
    # print(st._array_of_potentially_forbidden_events)
    # cst = CumulativeSupportTesting(parents_of, observed_cardinalities, 6)
    # print(cst.all_infeasible_supports)
    # print(cst.all_infeasible_supports_matrix)
    #
    # #Everything appears to work as desired.
    # #So let's pick a really hard problem, the square!
    #
    # parents_of = ([4, 5, 6], [4, 7, 8], [5, 7, 9], [6, 8, 9])
    # observed_cardinalities = (2, 2, 2, 2)
    # cst = CumulativeSupportTesting(parents_of, observed_cardinalities, 4)
    # print(cst.all_infeasible_supports)
    # print(cst.all_infeasible_supports_unlabelled)
    # print(cst.all_infeasible_supports_independent_unlabelled)
    # discovered = cst.from_integer_to_matrix(cst.all_infeasible_supports_unlabelled)
    # # print(discovered)
    # # print(cst.all_infeasible_supports_matrix)
    # # discovered = to_digits(cst.all_infeasible_supports_matrix, observed_cardinalities)
    # # print(discovered)
    # trulyvariable = discovered.compress(discovered.any(axis=-2).all(axis=-1),
    #                                     axis=0)  # makes every variable actually variable
    # print(trulyvariable)
    # TODO: it would be good to recognize PRODUCT support matrices. Will be required for d-sep and e-sep filter.
    # see https://www.researchgate.net/post/How-I-can-check-in-MATLAB-if-a-matrix-is-result-of-the-Kronecker-product/542ab19bd039b130378b469d/citation/download?
    # see https://math.stackexchange.com/a/321424

    # # Special problem for Victor Gitton
    # parents_of = np.array(([3, 4], [4, 5], [5, 3]))
    # # parents_of_subtracted = parents_of - 3
    # # party_and_source_sym = np.array([np.take(perm, parents_of_subtracted)+3 for perm in itertools.permutations(range(3))])
    # party_and_source_sym = np.array([np.take(parents_of, perm, axis=0) for perm in
    #                       itertools.permutations(range(3))])
    # party_sym = np.array([np.roll(parents_of, i, axis=0) for i in range(3)])
    # print(party_sym)
    # party_and_source_sym = np.vstack((party_sym, party_sym[:, :, [1, 0]]))
    # party_and_source_sym = [parents_of[list(p_perm)][:, s_perm]
    #                         for p_perm in itertools.permutations(range(3))
    #                         for s_perm in itertools.permutations(range(2))
    #                         ]
    # print(party_and_source_sym)
    # observed_cardinalities = (3, 3, 3)
    # occurring_events_temp = np.array([(0, 0, 0), (1, 1, 1), (2, 2, 2)]
    #                                  + list(itertools.permutations(range(3))))
    #
    #
    # st = SupportTester(parents_of=parents_of,
    #               observed_cardinalities=observed_cardinalities,
    #               nof_events=(3+6))
    # print("3 outcome with no symmetry: ", st.feasibleQ_from_matrix(occurring_events_temp, name='mgh'))
    #
    # st = SupportTester(parents_of=party_sym,
    #               observed_cardinalities=observed_cardinalities,
    #               nof_events=(3+6))
    # print("3 outcome with only party symmetry: ", st.feasibleQ_from_matrix(occurring_events_temp, name='mgh'))
    #
    # st = SupportTester(parents_of=party_and_source_sym,
    #               observed_cardinalities=observed_cardinalities,
    #               nof_events=(3+6))
    # print("3 outcome w/ transposition symmetry: ", st.feasibleQ_from_matrix_CONSERVATIVE(occurring_events_temp, name='mgh'))
    #
    #
    #
    # occurring_events_card_4 = np.array([(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)]
    #                                  + list(itertools.permutations(range(4), 3)))
    #
    # max_n = len(occurring_events_card_4)
    # rejected_yet = False
    # for n in range(7, max_n+1):
    #     if rejected_yet:
    #         break
    #     print(f"Working on cardinality 4 with {n} definite events...")
    #     st = SupportTester(parents_of=party_sym,
    #                        observed_cardinalities=(4, 4, 4),
    #                        nof_events=n)
    #     # for definitely_occurring_events in itertools.combinations(occurring_events_card_4, n):
    #     definitely_occurring_events = occurring_events_card_4[:n]
    #     rejected_yet = not st.potentially_feasibleQ_from_matrix_pair(
    #         definitely_occurring_events_matrix=definitely_occurring_events,
    #         potentially_occurring_events_matrix=occurring_events_card_4
    #     )
    #     if rejected_yet:
    #         print(f"search completed at {n} events")
    #         print(definitely_occurring_events)
    #         break
    # print("In the end, were we able to prove infeasibility? ", rejected_yet)


