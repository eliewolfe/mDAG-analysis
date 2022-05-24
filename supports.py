from __future__ import absolute_import
import numpy as np
import itertools
from pysat.solvers import Solver
from pysat.formula import IDPool  # I wonder if we can't get away without this, but it is SO convenient
# from pysat.card import CardEnc, EncType
# from operator import itemgetter
from radix import from_digits, to_digits, uniform_base_test
from utilities import partsextractor
import progressbar
# from tqdm import tqdm
import methodtools

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

class SupportTester(object):
    def __init__(self, parents_of, observed_cardinalities, nof_events, pp_relations=tuple()):
        # print('Instantiating new object with nof_events=', nof_events, flush=True)
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

        self.event_cardinalities = np.broadcast_to(self.max_conceivable_events, self.nof_events)

        self.repeated_observed_cardinalities = np.broadcast_to(np.expand_dims(self.observed_cardinalities,axis=-1),
                                                               (self.nof_observed, self.nof_events)).T.ravel()

        self.conceivable_events_range = np.arange(self.max_conceivable_events, dtype=np.uint)

        self.relevant_parent_cardinalities = [partsextractor(self.observed_and_latent_cardinalities,
                                                                  self.parents_of[idx]) for idx in
                                              range(self.nof_observed)]

        self.binary_variables = set(np.flatnonzero(np.asarray(self.observed_cardinalities, dtype=np.intp) == 2))
        self.nonbinary_variables = set(range(self.nof_observed)).difference(self.binary_variables)
        self.vpool = IDPool(start_from=1)
        # self.var = lambda idx, val, par: self.vpool.id('v[{0}]_{2}=={1}'.format(idx, val, par))
        self.var = lambda idx, val, par: -self.vpool.id(
            'v[{0}]_{2}==0'.format(idx, val, par)) if idx in self.binary_variables and val == 1 else self.vpool.id(
            'v[{0}]_{2}=={1}'.format(idx, val, par))
        self.subSupportTesters = {n: SupportTester(parents_of, observed_cardinalities, n) for n in range(2, self.nof_events)}

    # @staticmethod
    # def partsextractor(thing_to_take_parts_of, indices):
    #     if len(indices) == 0:
    #         return tuple()
    #     elif len(indices) == 1:
    #         return (itemgetter(*indices)(thing_to_take_parts_of),)
    #     else:
    #         return itemgetter(*indices)(thing_to_take_parts_of)

    @cached_property
    def at_least_one_outcome(self):
        return [[self.var(idx, val, par) for val in self.observed_cardinalities_ranges[idx]]
                for idx in self.nonbinary_variables
                for par in np.ndindex(self.relevant_parent_cardinalities[idx])]

    @cached_property
    def _array_of_potentially_forbidden_events(self):
        return np.asarray([[-self.var(idx, iterator[idx], partsextractor(iterator, self.parents_of[idx])) for idx
                            in range(self.nof_observed)]
                           for iterator in np.ndindex(self.observed_and_latent_cardinalities)], dtype=np.intp).reshape(
            (np.prod(self.observed_cardinalities), -1, self.nof_events ** self.nof_latent, self.nof_observed))

    def from_list_to_matrix(self, supports_as_lists):
        return to_digits(supports_as_lists, self.observed_cardinalities)

    def from_matrix_to_list(self, supports_as_matrices):
        return from_digits(supports_as_matrices, self.observed_cardinalities)

    def from_list_to_integer(self, supports_as_lists):
        return from_digits(supports_as_lists, self.event_cardinalities)

    def from_integer_to_list(self, supports_as_integers):
        return to_digits(supports_as_integers, self.event_cardinalities)

    # def from_matrix_to_integer(self, supports_as_matrices):
    #     return self.from_list_to_integer(self.from_matrix_to_list(supports_as_matrices))
    #
    # def from_integer_to_matrix(self, supports_as_integers):
    #     return self.from_list_to_matrix(self.from_integer_to_list(supports_as_integers))

    def from_matrix_to_integer(self, supports_as_matrices):
        supports_as_matrices_as_array = np.asarray(supports_as_matrices, dtype=np.intp)
        shape = supports_as_matrices_as_array.shape
        return from_digits(
            supports_as_matrices_as_array.reshape(shape[:-2] + (np.prod(shape[-2:]),)),
            self.repeated_observed_cardinalities)

    def from_integer_to_matrix(self, supports_as_integers):
        # return self.from_list_to_matrix(self.from_integer_to_list(supports_as_integers))
        return np.reshape(to_digits(
            supports_as_integers, self.repeated_observed_cardinalities),
            np.asarray(supports_as_integers, dtype=np.intp).shape + (self.nof_events, self.nof_observed))

    def forbidden_events_clauses(self, occurring_events):
        return self._array_of_potentially_forbidden_events[
            np.setdiff1d(
                self.conceivable_events_range,
                self.from_matrix_to_list(np.asarray(occurring_events, dtype=np.intp)),
                assume_unique=True
            )].reshape((-1, self.nof_observed))

    def array_of_positive_outcomes(self, ocurring_events):
        for i, iterator in enumerate(ocurring_events):
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

    def _sat_solver_clauses(self, occurring_events):
        assert self.nof_events == len(occurring_events), 'The number of events does not match the expected number.'
        return list(self.array_of_positive_outcomes(occurring_events)) + \
               self.at_least_one_outcome + \
               self.forbidden_events_clauses(occurring_events).tolist()

    def feasibleQ_from_matrix(self, occurring_events, **kwargs):
        with Solver(bootstrap_with=self._sat_solver_clauses(occurring_events), **kwargs) as s:
            return s.solve()

    @methodtools.lru_cache(maxsize=None, typed=False)
    def feasibleQ_from_tuple(self, occurring_events_as_tuple, **kwargs):
        return self.feasibleQ_from_matrix(self.from_list_to_matrix(occurring_events_as_tuple), **kwargs)

    @methodtools.lru_cache(maxsize=None, typed=False)
    def feasibleQ_from_integer(self, occurring_events_as_integer, **kwargs):
        return self.feasibleQ_from_matrix(self.from_integer_to_matrix(occurring_events_as_integer), **kwargs)


    def feasibleQ_from_matrix_CONSERVATIVE(self, occurring_events, **kwargs):
        for n in range(2, self.nof_events):
            subSupportTester = self.subSupportTesters[n]
            for definitely_occurring_events in itertools.combinations(occurring_events, n):
                passes_inflation_test = subSupportTester.infeasibleQ_from_matrix_pair(definitely_occurring_events, occurring_events, **kwargs)
                if not passes_inflation_test:
                    # print("Got on! Rejected a support of ", self.nof_events, " events using level ", n, " inflation.")
                    return passes_inflation_test
        with Solver(bootstrap_with=self._sat_solver_clauses(occurring_events), **kwargs) as s:
            return s.solve()
    # def feasibleQ_from_matrix(self, occurring_events, **kwargs):
    #     return all(self._feasibleQ_from_matrix_conservative(occurring_events, **kwargs))


    def _sat_solver_clauses_bonus(self, definitely_occurring_events, potentially_occurring_events):
        """
        Intended to analyze many events with few-event hot_start. Infeasible test, but cannot guarantee feasible.
        (We could write another version that requires the missing events to show up somewhere in the off diagonal
         worlds, but that seems complicated.)
        """
        assert self.nof_events == len(definitely_occurring_events), 'The number of events does not match the expected number.'
        return list(self.array_of_positive_outcomes(definitely_occurring_events)) + \
               self.at_least_one_outcome + \
               self.forbidden_events_clauses(potentially_occurring_events).tolist()
    #
    # @methodtools.lru_cache(maxsize=None, typed=False)
    # def infeasibleQ_from_integer(self, definitely_occurring_events_as_integer, potentially_occurring_events_as_integer, **kwargs):
    #     return self.infeasibleQ(
    #         self.from_integer_to_matrix(definitely_occurring_events_as_integer)
    #         , self.from_integer_to_matrix(potentially_occurring_events_as_integer)
    #         , **kwargs)
    #
    def infeasibleQ_from_matrix_pair(self, definitely_occurring_events_matrix, potentially_occurring_events_matrix, **kwargs):
        with Solver(bootstrap_with=self._sat_solver_clauses_bonus(definitely_occurring_events_matrix, potentially_occurring_events_matrix), **kwargs) as s:
            return s.solve()

    @methodtools.lru_cache(maxsize=None, typed=False)
    def infeasibleQ_from_tuple_pair(self, definitely_occurring_events_tuple, potentially_occurring_events_tuple, **kwargs):
        return self.infeasibleQ_from_matrix_pair(self.from_list_to_matrix(definitely_occurring_events_tuple), self.from_list_to_matrix(potentially_occurring_events_tuple), **kwargs)


# def does_this_support_respect_this_pp_restriction(i, pp_set, s):
#     if len(pp_set):
#         s_resorted = s[np.argsort(s[:, i])]
#         # for k, g in itertools.groupby(s_resorted, lambda e: e[i]):
#         #     print((k,g))
#         partitioned = [np.vstack(tuple(g))[:, np.asarray(pp_set)] for k, g in itertools.groupby(s_resorted, lambda e: e[i])]
#         to_test_for_intersection=[set(map(tuple, pp_values)) for pp_values in partitioned]
#         raw_length = np.sum([len(vals) for vals in to_test_for_intersection])
#         compressed_length = len(set().union(*to_test_for_intersection))
#         return (raw_length == compressed_length)
#     else:
#         return True

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

def generate_supports_satisfying_pp_restriction(i, pp_vector, candidate_support_matrices, **kwargs):
    for s in explore_candidates(candidate_support_matrices, message='Enforce '+str(i)+' is perf. pred. by '+np.array_str(pp_vector), **kwargs):
        s_resorted = s[np.argsort(s[:, i])]
        # for k, g in itertools.groupby(s_resorted, lambda e: e[i]):
        #     print((k,g))
        partitioned = [np.vstack(tuple(g))[:, pp_vector] for k, g in itertools.groupby(s_resorted, lambda e: e[i])]
        to_test_for_intersection=[set(map(tuple, pp_values)) for pp_values in partitioned]
        raw_length = np.sum([len(vals) for vals in to_test_for_intersection])
        compressed_length = len(set().union(*to_test_for_intersection))
        if (raw_length == compressed_length):
            yield s

def extract_support_matrices_satisfying_pprestrictions(candidate_support_matrices_raw, pp_restrictions, verbose=True):
    candidate_support_matrices = candidate_support_matrices_raw.copy()
    for (i, pp_set) in pp_restrictions:
        print("Isolating candidates due to perfect prediction restriction...", i, " by ", pp_set)
        pp_vector = np.array(pp_set)
        candidate_support_matrices = list(generate_supports_satisfying_pp_restriction(i, pp_vector, candidate_support_matrices, verbose=verbose))
    return candidate_support_matrices


class SupportTesting(SupportTester):

    # def support_respects_perfect_prediction_restrictions(self, candidate_s):
    #     return all(does_this_support_respect_this_pp_restriction(
    #         *pp_restriction, candidate_s) for pp_restriction in self.must_perfectpredict)


    @cached_property
    def outcome_relabelling_group(self):
        if np.array_equiv(2, self.observed_cardinalities):
            return np.bitwise_xor.outer(self.conceivable_events_range, self.conceivable_events_range)
        else:
            return from_digits(np.fromiter(
                itertools.chain.from_iterable(itertools.chain.from_iterable(
                    itertools.starmap(
                        itertools.product,
                        itertools.product(*map(itertools.permutations, map(range, self.observed_cardinalities)))
                    ))),
                int).reshape((-1, self.max_conceivable_events, self.nof_observed)),
                               self.observed_cardinalities)

    @cached_property
    def party_relabelling_group(self):
        assert uniform_base_test(self.observed_cardinalities), "Not meaningful to relabel parties with different cardinalities"
        to_reshape = self.conceivable_events_range.reshape(self.observed_cardinalities)
        return np.fromiter(
            itertools.chain.from_iterable(
                to_reshape.transpose(perm).ravel() for perm in itertools.permutations(range(self.nof_observed))
            ), int
        ).reshape((-1, self.max_conceivable_events,))

    @cached_property
    def universal_relabelling_group(self):
        return np.vstack(np.take(self.outcome_relabelling_group, self.party_relabelling_group, axis=-1))
        # hugegroup = np.vstack(np.take(self.outcome_relabelling_group, self.party_relabelling_group, axis=-1))
        # hugegroup = hugegroup[np.lexsort(hugegroup.T)]
        # hugegroup2 = np.vstack(np.take(self.party_relabelling_group, self.outcome_relabelling_group, axis=-1))
        # hugegroup2 = hugegroup2[np.lexsort(hugegroup2.T)]
        # assert np.array_equal(hugegroup, hugegroup2), "subgroups should be commutative"

    # def unique_supports_under_group(self, candidates_raw, group, verbose=True):
    #     candidates = candidates_raw.copy()
    #     compressed_candidates = from_digits(candidates, self.event_cardinalities)
    #     for group_element in explore_candidates(group[1:], verbose=verbose):
    #         new_candidates = np.take(group_element, candidates)
    #         new_candidates.sort()
    #         np.minimum(compressed_candidates,
    #                    from_digits(new_candidates, self.event_cardinalities),
    #                    out=compressed_candidates)
    #         candidates = to_digits(compressed_candidates, self.event_cardinalities)
    #     return np.unique(candidates, axis=0)
    def unique_supports_under_group(self, candidates_raw, group, verbose=False):
        candidates = candidates_raw.copy()
        #
        for group_element in explore_candidates(group[1:], verbose=verbose, message='Getting unique under relabelling'):
            compressed_candidates = from_digits(candidates, self.event_cardinalities)
            new_candidates = np.take(group_element, candidates)
            new_candidates.sort()
            np.minimum(compressed_candidates,
                       from_digits(new_candidates, self.event_cardinalities),
                       out=compressed_candidates)
            candidates = to_digits(np.unique(compressed_candidates), self.event_cardinalities)
        return candidates


    @cached_property
    def unique_candidate_supports_as_lists(self):
        if self.max_conceivable_events > self.nof_events:
            candidates = np.pad(np.fromiter(itertools.chain.from_iterable(
                itertools.combinations(self.conceivable_events_range[1:], self.nof_events - 1)), np.intp).reshape(
                (-1, self.nof_events - 1)), ((0, 0), (1, 0)), 'constant')
            to_filter = self.from_list_to_matrix(candidates)
            filtered = extract_support_matrices_satisfying_pprestrictions(to_filter, self.must_perfectpredict)
            candidates = np.array(list(map(self.from_matrix_to_list, filtered)), dtype=int)
            candidates = self.unique_supports_under_group(candidates, self.outcome_relabelling_group)
            return candidates
        else:
            return np.empty((0, 0), dtype=np.intp)

    # @cached_property
    # def unique_candidate_supports_as_lists(self):
    #     print("Thus far: ", len(self._unique_candidate_supports_as_lists))
    #     print("Isolating due to perfect prediction restrictions.")
    #     to_filter = self.from_list_to_matrix(self._unique_candidate_supports_as_lists)
    #     picklist = np.fromiter(map(self.support_respects_perfect_prediction_restrictions, to_filter), bool)
    #     print("Thus far: ", np.sum(picklist.astype(int)))
    #     return self._unique_candidate_supports_as_lists[picklist]

    @cached_property
    def unique_candidate_supports_as_integers(self):
        return self.from_list_to_integer(self.unique_candidate_supports_as_lists)

    @cached_property
    def unique_candidate_supports_as_matrices(self):
        return self.from_list_to_matrix(self.unique_candidate_supports_as_lists)



    def explore_candidates(self, candidates, **kwargs):
        return explore_candidates(candidates, **kwargs)

    # def unique_candidate_supports_as_lists_to_iterate(self, verbose=False):
    #     if verbose:
    #         return progressbar.progressbar(
    #                     self.unique_candidate_supports_as_lists, widgets=[
    #                         '[nof_events=', str(self.nof_events), '] '
    #                         , progressbar.SimpleProgress(), progressbar.Bar()
    #                         , ' (', progressbar.ETA(), ') '])
    #     else:
    #         return self.unique_candidate_supports_as_lists



    # def unique_candidate_supports_as_integers_to_iterate(self, verbose=False):
    #     if verbose:
    #         return progressbar.progressbar(
    #                     self.unique_candidate_supports_as_integers, widgets=[
    #                         '[nof_events=', str(self.nof_events), '] '
    #                         , progressbar.SimpleProgress(), progressbar.Bar()
    #                         , ' (', progressbar.ETA(), ') '])
    #     else:
    #         return self.unique_candidate_supports_as_integers
        
    #To understand the function unique_candidate_supports:    
    #ch=itertools.chain.from_iterable(itertools.combinations(np.arange(16, dtype=np.uint),4))
    #l=np.fromiter(ch, np.uint).reshape((-1, 4))
    #p=np.pad(l,((0, 0), (1, 0)), 'constant')
    #print(p)
   




    @methodtools.lru_cache(maxsize=None, typed=False)
    def attempt_to_find_one_infeasible_support(self, **kwargs):
        return self.attempt_to_find_one_infeasible_support_among(self.unique_candidate_supports_as_lists, **kwargs)
    def attempt_to_find_one_infeasible_support_among(self, candidates_as_lists, verbose=False, **kwargs):
        # for n in range(2, self.nof_events):
        #     subSupportTester = self.subSupportTesters[n]
        #     for occurring_events_as_tuple in map(tuple, self.explore_candidates(candidates_as_lists, verbose=verbose)):
        #         for definitely_occurring_events_as_tuple in itertools.combinations(occurring_events_as_tuple, n):
        #             passes_inflation_test = subSupportTester.infeasibleQ_from_tuple_pair(definitely_occurring_events_as_tuple, occurring_events_as_tuple, **kwargs)
        #             if not passes_inflation_test:
        #                 # print("Got one! Rejected a support of ", self.nof_events, " events using level ", n, " inflation.")
        #                 return self.from_list_to_matrix(occurring_events_as_tuple)
        for occurring_events_as_tuple in map(tuple, self.explore_candidates(candidates_as_lists,
                                                                            verbose=verbose,
                                                                            message='Finding an infeasible support')):
            if not self.feasibleQ_from_tuple(occurring_events_as_tuple):
                return self.from_list_to_matrix(occurring_events_as_tuple)
        return np.empty((0, self.nof_observed), dtype=int)



    def no_infeasible_supports_among(self, candidates_as_lists, **kwargs):
        if len(self.attempt_to_find_one_infeasible_support_among(candidates_as_lists, **kwargs)) == 0:
            return True
        else:
            return False

    @methodtools.lru_cache(maxsize=None, typed=False)
    def no_infeasible_supports(self, **kwargs):
        return self.no_infeasible_supports_among(self.unique_candidate_supports_as_lists, **kwargs)
        # return all(self.feasibleQ_from_integer(occurring_events_as_int, **kwargs)[0] for occurring_events_as_int in
        #            self.explore_candidates(self.unique_candidate_supports_as_integers, verbose=verbose))

    def unique_infeasible_supports_as_integers_among(self, candidates_as_integers, verbose=False, **kwargs):
        return np.fromiter((occurring_events_as_int for occurring_events_as_int in self.explore_candidates(candidates_as_integers, verbose=verbose) if
                            not self.feasibleQ_from_integer(occurring_events_as_int, **kwargs)), dtype=int)

    @methodtools.lru_cache(maxsize=None, typed=False)
    def unique_infeasible_supports_as_integers(self, **kwargs):
        """
        Return a signature of infeasible support for a given parents_of, observed_cardinalities, and nof_events
        :param kwargs: optional arguments to pysat.Solver
        :param verbose: option to display progressbar
        """
        return self.unique_infeasible_supports_as_integers_among(self.unique_candidate_supports_as_integers, **kwargs)


    @methodtools.lru_cache(maxsize=None, typed=False)
    def unique_infeasible_supports_as_matrices(self, **kwargs):
        return self.from_integer_to_matrix(self.unique_infeasible_supports_as_integers(**kwargs))

    def convert_integers_into_canonical_under_relabelling(self, list_of_integers):
        if len(list_of_integers) > 0:
            labelled_infeasible_as_lists = self.from_integer_to_list(list_of_integers)
            labelled_variants = self.universal_relabelling_group[:, labelled_infeasible_as_lists]
            labelled_variants.sort(axis=-1)
            labelled_variants = self.from_list_to_integer(labelled_variants).astype(int)
            labelled_variants.sort(axis=-1)
            lexsort = np.lexsort(labelled_variants.T)
            return labelled_variants[lexsort[0]]
        else:
            return list_of_integers

    @methodtools.lru_cache(maxsize=None, typed=False)
    def unique_infeasible_supports_as_integers_unlabelled(self, **kwargs):
        return self.convert_integers_into_canonical_under_relabelling(
            self.unique_infeasible_supports_as_integers(**kwargs))






class CumulativeSupportTesting:
    #TODO: Add scrollbar
    def __init__(self, parents_of, observed_cardinalities, max_nof_events):
        self.parents_of = parents_of
        self.observed_cardinalities= observed_cardinalities
        self.nof_observed = len(self.parents_of)
        assert self.nof_observed == len(observed_cardinalities)
        self.max_nof_events = max_nof_events
        self.max_conceivable_events = np.prod(self.observed_cardinalities)
        self.multi_index_shape = np.broadcast_to(self.max_conceivable_events, self.max_nof_events)
        self.repeated_observed_cardinalities = np.broadcast_to(np.expand_dims(self.observed_cardinalities,axis=-1),
                                                               (self.nof_observed, self.max_nof_events)).T.ravel()

    @property
    def _all_infeasible_supports(self):
        for nof_events in range(2, self.max_nof_events+1):
            yield SupportTesting(self.parents_of, self.observed_cardinalities, nof_events
                                 ).unique_infeasible_supports_as_integers(name='mgh', use_timer=False)

    @property
    def _all_infeasible_supports_unlabelled(self):
        for nof_events in range(2, self.max_nof_events+1):
            yield SupportTesting(self.parents_of, self.observed_cardinalities, nof_events
                                 ).unique_infeasible_supports_as_integers_unlabelled(name='mgh', use_timer=False)

    @cached_property
    def all_infeasible_supports(self):
        return np.fromiter(itertools.chain.from_iterable(self._all_infeasible_supports), np.uint64)

    @cached_property
    def all_infeasible_supports_unlabelled(self):
        return np.fromiter(itertools.chain.from_iterable(self._all_infeasible_supports_unlabelled), np.uint64)

    def from_list_to_matrix(self, supports_as_lists):
        return to_digits(supports_as_lists, self.observed_cardinalities)

    def from_matrix_to_list(self, supports_as_matrices):
        return from_digits(supports_as_matrices, self.observed_cardinalities)

    def from_list_to_integer(self, supports_as_lists):
        return from_digits(supports_as_lists, self.multi_index_shape)

    def from_integer_to_list(self, supports_as_integers):
        return to_digits(supports_as_integers, self.multi_index_shape)

    def from_matrix_to_integer(self, supports_as_matrices):
        # return self.from_list_to_integer(self.from_matrix_to_list(supports_as_matrices))
        supports_as_matrices_as_array = np.asarray(supports_as_matrices)
        shape = supports_as_matrices_as_array.shape
        return from_digits(
            supports_as_matrices_as_array.reshape(shape[:-2] + (np.prod(shape[-2:]),)),
            self.repeated_observed_cardinalities)

    def from_integer_to_matrix(self, supports_as_integers):
        # return self.from_list_to_matrix(self.from_integer_to_list(supports_as_integers))
        return np.reshape(to_digits(
            supports_as_integers, self.repeated_observed_cardinalities),
            np.asarray(supports_as_integers).shape + (self.max_nof_events, self.nof_observed))

    # @cached_property
    # def all_infeasible_supports_as_list(self):
    #     return self.from_integer_to_list(self.all_infeasible_supports)


#TODO: Visualize an infeasible support as a SciPy sparce matrix
#TODO: Check n+k event support using n event code.

if __name__ == '__main__':
    parents_of = ([3, 4], [4, 5], [3, 5])
    observed_cardinalities = (2, 2, 2)
    nof_events = 3
    # st = SupportTesting(parents_of, observed_cardinalities, nof_events)
    #
    # occuring_events_temp = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    # print(st.feasibleQ_from_matrix(occuring_events_temp, name='mgh', use_timer=True))
    # occuring_events_temp = [(0, 0, 0), (0, 1, 0), (0, 0, 1)]
    # print(st.feasibleQ_from_matrix(occuring_events_temp, name='mgh', use_timer=True))

    st = SupportTesting(parents_of, observed_cardinalities, 3)
    inf = st.unique_infeasible_supports_as_integers_unlabelled(name='mgh', use_timer=False)
    print(inf)
    print(st.from_integer_to_matrix(inf))
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
    parents_of = ([4, 5, 6], [4, 7, 8], [5, 7, 9], [6, 8, 9])
    observed_cardinalities = (2, 2, 2,2)
    cst = CumulativeSupportTesting(parents_of, observed_cardinalities, 4)
    print(cst.all_infeasible_supports)
    print(cst.all_infeasible_supports_unlabelled)
    discovered=cst.from_integer_to_matrix(cst.all_infeasible_supports_unlabelled)
    #print(discovered)
    # print(cst.all_infeasible_supports_matrix)
    # discovered = to_digits(cst.all_infeasible_supports_matrix, observed_cardinalities)
    # print(discovered)
    trulyvariable = discovered.compress(discovered.any(axis=-2).all(axis=-1), axis=0) #makes every variable actually variable
    print(trulyvariable)
    #TODO: it would be good to recognize PRODUCT support matrices. Will be required for d-sep and e-sep filter.
    # see https://www.researchgate.net/post/How-I-can-check-in-MATLAB-if-a-matrix-is-result-of-the-Kronecker-product/542ab19bd039b130378b469d/citation/download?
    # see https://math.stackexchange.com/a/321424



