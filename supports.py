from __future__ import absolute_import

import itertools
from functools import reduce
from sys import hexversion

# from tqdm import tqdm
import methodtools
import numpy as np
import progressbar
from pysat.formula import IDPool  # I wonder if we can't get away without this, but it is SO convenient
from pysat.solvers import Solver

# from pysat.card import CardEnc, EncType
# from operator import itemgetter
from radix import from_digits, to_digits, uniform_base_test
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


class SupportTester(object):
    def __init__(self, parents_of, observed_cardinalities, nof_events, pp_relations=tuple()):
        self.must_perfectpredict = pp_relations
        self.parents_of = parents_of
        self.nof_events = nof_events
        self.nof_observed = len(self.parents_of)
        assert self.nof_observed == len(observed_cardinalities)
        self.nof_latent = max(itertools.chain.from_iterable(self.parents_of)) + 1 - self.nof_observed
        self.observed_cardinalities = observed_cardinalities
        self.observed_cardinalities_ranges = list(map(range, self.observed_cardinalities))

        self.observed_and_latent_cardinalities = tuple(np.hstack((observed_cardinalities,
                                                                  np.repeat(self.nof_events, self.nof_latent))))

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
        self.subSupportTesters = {n: SupportTester(parents_of, observed_cardinalities, n) for n in
                                  range(2, self.nof_events)}
        self._orbit_under_outcome_relabelling = dict()
        self._orbit_under_party_relabelling = dict()

    @cached_property
    def at_least_one_outcome(self):
        return [[self.var(idx, val, par) for val in self.observed_cardinalities_ranges[idx]]
                for idx in self.nonbinary_variables
                for par in np.ndindex(self.relevant_parent_cardinalities[idx])]

    @cached_property
    def _array_of_potentially_forbidden_events(self):
        return np.asarray([[-self.var(idx, iterator[idx], partsextractor(iterator, self.parents_of[idx])) for idx
                            in range(self.nof_observed)]
                           for iterator in np.ndindex(*self.observed_and_latent_cardinalities)], dtype=np.intp).reshape(
            (np.prod(self.observed_cardinalities), -1, self.nof_events ** self.nof_latent, self.nof_observed))

    def from_list_to_matrix(self, supports_as_lists: np.ndarray) -> np.ndarray:
        return to_digits(supports_as_lists, self.observed_cardinalities).astype(self.matrix_dtype)

    def from_matrix_to_list(self, supports_as_matrices: np.ndarray) -> np.ndarray:
        return from_digits(supports_as_matrices, self.observed_cardinalities).astype(self.list_dtype)

    def from_list_to_integer(self, supports_as_lists: np.ndarray) -> np.ndarray:
        return from_digits(supports_as_lists, self.event_cardinalities).astype(self.int_dtype)

    def from_integer_to_list(self, supports_as_integers: np.ndarray) -> np.ndarray:
        return to_digits(supports_as_integers, self.event_cardinalities).astype(self.list_dtype)

    def from_matrix_to_integer(self, supports_as_matrices: np.ndarray) -> np.ndarray:
        supports_as_matrices_as_array = np.asarray(supports_as_matrices, dtype=np.intp)
        shape = supports_as_matrices_as_array.shape
        return from_digits(
            supports_as_matrices_as_array.reshape(shape[:-2] + (np.prod(shape[-2:]),)),
            self.repeated_observed_cardinalities).astype(self.int_dtype)

    def from_integer_to_matrix(self, supports_as_integers: np.ndarray) -> np.ndarray:
        return np.reshape(to_digits(
            supports_as_integers, self.repeated_observed_cardinalities).astype(self.matrix_dtype),
            np.asarray(supports_as_integers, dtype=self.int_dtype).shape + (self.nof_events, self.nof_observed))

    def forbidden_events_clauses(self, occurring_events: np.ndarray) -> List:
        return self._array_of_potentially_forbidden_events[
            np.setdiff1d(
                self.conceivable_events_range,
                self.from_matrix_to_list(np.asarray(occurring_events, dtype=self.matrix_dtype)),
                assume_unique=True
            )].reshape((-1, self.nof_observed)).tolist()

    def array_of_positive_outcomes(self, occurring_events: np.ndarray) -> List:
        for i, iterator in enumerate(occurring_events):
            iterator_plus = tuple(iterator) + tuple(np.repeat(i, self.nof_latent))
            for idx in self.nonbinary_variables:
                # NEXT LINE IS OPTIONAL, CAN BE COMMENTED OUT
                yield [self.var(idx, iterator[idx], partsextractor(iterator_plus, self.parents_of[idx]))]
                wrong_values = set(range(self.observed_cardinalities[idx]))
                wrong_values.remove(iterator[idx])
                for wrong_value in wrong_values:
                    # if wrong_value != iterator[idx]:
                    yield [-self.var(idx, wrong_value, partsextractor(iterator_plus, self.parents_of[idx]))]
            for idx in self.binary_variables:
                yield [self.var(idx, iterator[idx], partsextractor(iterator_plus, self.parents_of[idx]))]

    def _sat_solver_clauses(self, occurring_events: np.ndarray) -> List:
        assert self.nof_events == len(occurring_events), 'The number of events does not match the expected number.'
        return list(self.array_of_positive_outcomes(occurring_events)) + \
               self.at_least_one_outcome + \
               self.forbidden_events_clauses(occurring_events)

    def feasibleQ_from_matrix(self, occurring_events: np.ndarray, **kwargs) -> bool:
        with Solver(bootstrap_with=self._sat_solver_clauses(occurring_events), **kwargs) as s:
            return s.solve()

    @methodtools.lru_cache(maxsize=None, typed=False)
    def feasibleQ_from_tuple(self, occurring_events_as_tuple: np.ndarray, **kwargs) -> bool:
        return self.feasibleQ_from_matrix(self.from_list_to_matrix(occurring_events_as_tuple), **kwargs)

    @methodtools.lru_cache(maxsize=None, typed=False)
    def feasibleQ_from_integer(self, occurring_events_as_integer, **kwargs) -> bool:
        return self.feasibleQ_from_matrix(self.from_integer_to_matrix(occurring_events_as_integer), **kwargs)

    def feasibleQ_from_matrix_CONSERVATIVE(self, occurring_events: np.ndarray, **kwargs) -> bool:
        for n in range(2, self.nof_events):
            subSupportTester = self.subSupportTesters[n]
            for definitely_occurring_events in itertools.combinations(occurring_events, n):
                passes_inflation_test = subSupportTester.infeasibleQ_from_matrix_pair(definitely_occurring_events,
                                                                                      occurring_events, **kwargs)
                if not passes_inflation_test:
                    # print("Got on! Rejected a support of ", self.nof_events, " events using level ", n, " inflation.")
                    return passes_inflation_test
        with Solver(bootstrap_with=self._sat_solver_clauses(occurring_events), **kwargs) as s:
            return s.solve()

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

    def infeasibleQ_from_matrix_pair(self,
                                     definitely_occurring_events_matrix: np.ndarray,
                                     potentially_occurring_events_matrix: np.ndarray,
                                     **kwargs) -> bool:
        with Solver(bootstrap_with=self._sat_solver_clauses_bonus(definitely_occurring_events_matrix,
                                                                  potentially_occurring_events_matrix), **kwargs) as s:
            return s.solve()

    @methodtools.lru_cache(maxsize=None, typed=False)
    def infeasibleQ_from_tuple_pair(self,
                                    definitely_occurring_events_tuple: np.ndarray,
                                    potentially_occurring_events_tuple: np.ndarray,
                                    **kwargs) -> bool:
        return self.infeasibleQ_from_matrix_pair(self.from_list_to_matrix(definitely_occurring_events_tuple),
                                                 self.from_list_to_matrix(potentially_occurring_events_tuple), **kwargs)

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

    def orbit_under_outcome_relabelling(self, n: int):
        try:
            return self._orbit_under_outcome_relabelling[n]
        except KeyError:
            l = self.from_integer_to_list(n)
            l_variants = np.take(self.outcome_relabelling_group, l, axis=-1)
            l_variants.sort(axis=-1)
            n_orbit = self.from_list_to_integer(l_variants)
            for n_new in n_orbit.flat:
                self._orbit_under_outcome_relabelling[n_new] = n_orbit
            return n_orbit

    @cached_property
    def party_relabelling_group(self) -> np.ndarray:
        # assert uniform_base_test(
        #     self.observed_cardinalities), "Not meaningful to relabel parties with different cardinalities"
        to_reshape = self.conceivable_events_range.reshape(self.observed_cardinalities)
        return np.fromiter(
            itertools.chain.from_iterable(
                to_reshape.transpose(perm).ravel()
                for perm in itertools.permutations(range(self.nof_observed))
                if np.array_equal(
                    self.observed_cardinalities,
                    np.take(self.observed_cardinalities, perm))
            ), self.list_dtype
        ).reshape((-1, self.max_conceivable_events,))

    def orbit_under_party_relabelling(self, n: int):
        try:
            return self._orbit_under_party_relabelling[n]
        except KeyError:
            l = self.from_integer_to_list(n)
            l_variants = np.take(self.party_relabelling_group, l, axis=-1)
            l_variants.sort(axis=-1)
            n_orbit = self.from_list_to_integer(l_variants)
            for n_new in n_orbit.flat:
                self._orbit_under_party_relabelling[n_new] = n_orbit
            return n_orbit

    @cached_property
    def unique_candidate_supports_as_lists(self) -> np.ndarray:
        if self.max_conceivable_events > self.nof_events:
            candidates = np.pad(np.fromiter(itertools.chain.from_iterable(
                itertools.combinations(self.conceivable_events_range[1:], self.nof_events - 1)), self.list_dtype).reshape(
                (-1, self.nof_events - 1)), ((0, 0), (1, 0)), 'constant')
            if self.must_perfectpredict:
                to_filter = self.from_list_to_matrix(candidates)
                filtered = self.extract_support_matrices_satisfying_pprestrictions(to_filter, self.must_perfectpredict)
                candidates = np.array(list(map(self.from_matrix_to_list, filtered)), dtype=self.list_dtype)
            candidates_as_ints = self.from_list_to_integer(candidates)
            candidates_as_ints = set((self.orbit_under_outcome_relabelling(n).min() for n in candidates_as_ints.flat))
            candidates = self.from_integer_to_list(list(candidates_as_ints))
            return candidates
        else:
            return np.empty((0, 0), dtype=self.list_dtype)

    @cached_property
    def unique_candidate_supports_as_integers(self) -> np.ndarray:
        return self.from_list_to_integer(self.unique_candidate_supports_as_lists)

    @cached_property
    def unique_candidate_supports_as_matrices(self) -> np.ndarray:
        return self.from_list_to_matrix(self.unique_candidate_supports_as_lists)

    @staticmethod
    def explore_candidates(candidates, **kwargs):
        return explore_candidates(candidates, **kwargs)

    @methodtools.lru_cache(maxsize=None, typed=False)
    def attempt_to_find_one_infeasible_support(self, **kwargs) -> np.ndarray:
        return self.attempt_to_find_one_infeasible_support_among(self.unique_candidate_supports_as_lists, **kwargs)

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
        return self.no_infeasible_supports_among(self.unique_candidate_supports_as_lists, **kwargs)
        # return all(self.feasibleQ_from_integer(occurring_events_as_int, **kwargs)[0] for occurring_events_as_int in
        #            self.explore_candidates(self.unique_candidate_supports_as_integers, verbose=verbose))

    def unique_infeasible_supports_as_integers_among(
            self, candidates_as_integers, verbose=False, **kwargs) -> np.ndarray:
        return np.fromiter((occurring_events_as_int for occurring_events_as_int in
                            self.explore_candidates(candidates_as_integers, verbose=verbose) if
                            not self.feasibleQ_from_integer(occurring_events_as_int, **kwargs)), dtype=self.int_dtype)

    @methodtools.lru_cache(maxsize=None, typed=False)
    def unique_infeasible_supports_as_integers(self, **kwargs) -> np.ndarray:
        """
        Return a signature of infeasible support for a given parents_of, observed_cardinalities, and nof_events
        :param kwargs: optional arguments to pysat.Solver
        :param verbose: option to display progressbar
        """
        return self.unique_infeasible_supports_as_integers_among(self.unique_candidate_supports_as_integers, **kwargs)

    @methodtools.lru_cache(maxsize=None, typed=False)
    def unique_infeasible_supports_as_matrices(self, **kwargs) -> np.ndarray:
        return self.from_integer_to_matrix(self.unique_infeasible_supports_as_integers(**kwargs))

    def convert_integers_into_canonical_under_independent_relabelling(self, list_of_integers: np.ndarray) -> np.ndarray:
        if len(list_of_integers) > 0:
            compressed = set()
            for m in list_of_integers.flat:
                m_party_variants = self.orbit_under_party_relabelling(m)
                canonical_rep = min(self.orbit_under_outcome_relabelling(n).min() for n in m_party_variants.flat)
                compressed.add(canonical_rep)
            return np.fromiter((n for n in compressed), dtype=self.int_dtype)
        else:
            return list_of_integers

    def convert_integers_into_canonical_under_coherent_relabelling(self, list_of_integers: np.ndarray) -> np.ndarray:
        if len(list_of_integers) > 0:
            current_list_of_integers = list_of_integers.copy()
            current_list_of_lists = self.from_integer_to_list(current_list_of_integers)
            for g_parties in self.party_relabelling_group:
                temp_list_of_lists = g_parties[current_list_of_lists]
                temp_list_of_lists.sort(axis=-1)
                temp_list_of_integers = self.from_list_to_integer(temp_list_of_lists)
                temp_list_of_integers = np.fromiter(
                    (self.orbit_under_outcome_relabelling(n).min() for n in temp_list_of_integers.flat),
                    dtype=self.int_dtype)
                temp_list_of_integers.sort(axis=-1)
                if np.all(np.less_equal(temp_list_of_integers, current_list_of_integers)):
                    current_list_of_integers = temp_list_of_integers
                    current_list_of_lists = temp_list_of_lists
            return current_list_of_integers
        else:
            return list_of_integers

    @methodtools.lru_cache(maxsize=None, typed=False)
    def unique_infeasible_supports_as_integers_unlabelled(self, **kwargs) -> np.ndarray:
        return self.convert_integers_into_canonical_under_coherent_relabelling(
            self.unique_infeasible_supports_as_integers(**kwargs))

    @methodtools.lru_cache(maxsize=None, typed=False)
    def unique_infeasible_supports_as_integers_independent_unlabelled(self, **kwargs) -> np.ndarray:
        return self.convert_integers_into_canonical_under_independent_relabelling(
            self.unique_infeasible_supports_as_integers(**kwargs))


class CumulativeSupportTesting:
    # TODO: Add scrollbar
    def __init__(self, parents_of, observed_cardinalities, max_nof_events):
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

    @property
    def _all_infeasible_supports(self):
        for nof_events in range(2, self.max_nof_events + 1):
            yield SupportTesting(self.parents_of, self.observed_cardinalities, nof_events
                                 ).unique_infeasible_supports_as_integers(name='mgh', use_timer=False)

    @property
    def _all_infeasible_supports_unlabelled(self):
        for nof_events in range(2, self.max_nof_events + 1):
            yield SupportTesting(self.parents_of, self.observed_cardinalities, nof_events
                                 ).unique_infeasible_supports_as_integers_unlabelled(name='mgh', use_timer=False)

    @property
    def _all_infeasible_supports_independent_unlabelled(self):
        for nof_events in range(2, self.max_nof_events + 1):
            yield SupportTesting(self.parents_of, self.observed_cardinalities, nof_events
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
    parents_of = ([1, 3], [3, 4], [1, 4])
    observed_cardinalities = (3, 3, 3)
    # nof_events = 3
    # st = SupportTesting(parents_of, observed_cardinalities, nof_events)
    #
    # occurring_events_temp = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    # print(st.feasibleQ_from_matrix(occurring_events_temp, name='mgh', use_timer=True))
    # occurring_events_temp = [(0, 0, 0), (0, 1, 0), (0, 0, 1)]
    # print(st.feasibleQ_from_matrix(occurring_events_temp, name='mgh', use_timer=True))

    st = SupportTesting(parents_of, observed_cardinalities, 3)
    cst = CumulativeSupportTesting(parents_of, observed_cardinalities, 4)
    inf = st.unique_infeasible_supports_as_integers_unlabelled(name='mgh', use_timer=False)
    print(st.from_integer_to_matrix(inf))
    print(cst.all_infeasible_supports)
    print(cst.all_infeasible_supports_unlabelled)
    print(cst.all_infeasible_supports_independent_unlabelled)
    parents_of = ([3, 4], [4, 5], [3, 5])
    cst = CumulativeSupportTesting(parents_of, observed_cardinalities, 4)
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
