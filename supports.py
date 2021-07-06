from __future__ import absolute_import
import numpy as np
import itertools
from pysat.solvers import Solver
from pysat.formula import IDPool  # I wonder if we can't get away without this, but it is SO convenient
# from pysat.card import CardEnc, EncType
from operator import itemgetter
from radix import from_digits, to_digits

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


class SupportTester:
    def __init__(self, parents_of, observed_cardinalities, nof_events):
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

        self.conceivable_events_range = np.arange(self.max_conceivable_events, dtype=np.uint)

        self.relevant_parent_cardinalities = [self.partsextractor(self.observed_and_latent_cardinalities,
                                                                  self.parents_of[idx]) for idx in
                                              range(self.nof_observed)]

        self.binary_variables = set(np.flatnonzero(np.asarray(self.observed_cardinalities) == 2))
        self.nonbinary_variables = set(range(self.nof_observed)).difference(self.binary_variables)
        self.vpool = IDPool(start_from=1)
        # self.var = lambda idx, val, par: self.vpool.id('v[{0}]_{2}=={1}'.format(idx, val, par))
        self.var = lambda idx, val, par: -self.vpool.id(
            'v[{0}]_{2}==0'.format(idx, val, par)) if idx in self.binary_variables and val == 1 else self.vpool.id(
            'v[{0}]_{2}=={1}'.format(idx, val, par))

    @staticmethod
    def partsextractor(thing_to_take_parts_of, indices):
        if len(indices) == 0:
            return tuple()
        elif len(indices) == 1:
            return (itemgetter(*indices)(thing_to_take_parts_of),)
        else:
            return itemgetter(*indices)(thing_to_take_parts_of)

    @cached_property
    def at_least_one_outcome(self):
        return [[self.var(idx, val, par) for val in self.observed_cardinalities_ranges[idx]]
                for idx in self.nonbinary_variables
                for par in np.ndindex(self.relevant_parent_cardinalities[idx])]

    @cached_property
    def _array_of_potentially_forbidden_events(self):
        return np.asarray([[-self.var(idx, iterator[idx], self.partsextractor(iterator, self.parents_of[idx])) for idx
                            in range(self.nof_observed)]
                           for iterator in np.ndindex(self.observed_and_latent_cardinalities)]).reshape(
            (np.prod(self.observed_cardinalities), -1, self.nof_events ** self.nof_latent, self.nof_observed))


    def forbidden_events_clauses(self, occurring_events):
        return self._array_of_potentially_forbidden_events[
            np.setdiff1d(
                self.conceivable_events_range,
                from_digits(np.asarray(occurring_events), self.observed_cardinalities),
                assume_unique=True
            )].reshape((-1, self.nof_observed))

    def array_of_positive_outcomes(self, ocurring_events):
        for i, iterator in enumerate(ocurring_events):
            iterator_plus = tuple(iterator) + tuple(np.repeat(i, self.nof_latent))
            for idx in self.nonbinary_variables:
                # NEXT LINE IS OPTIONAL, CAN BE COMMENTED OUT
                yield [self.var(idx, iterator[idx], self.partsextractor(iterator_plus, self.parents_of[idx]))]
                wrong_values = set(range(self.observed_cardinalities[idx]))
                wrong_values.remove(iterator[idx])
                for wrong_value in wrong_values:
                    # if wrong_value != iterator[idx]:
                    yield [-self.var(idx, wrong_value, self.partsextractor(iterator_plus, self.parents_of[idx]))]
            for idx in self.binary_variables:
                yield [self.var(idx, iterator[idx], self.partsextractor(iterator_plus, self.parents_of[idx]))]

    def _sat_solver_clauses(self, occurring_events):
        assert self.nof_events == len(occurring_events), 'The number of events does not match the expected number.'
        return list(self.array_of_positive_outcomes(occurring_events)) + \
               self.at_least_one_outcome + \
               self.forbidden_events_clauses(occurring_events).tolist()

    def feasibleQ(self, occurring_events, **kwargs):
        with Solver(bootstrap_with=self._sat_solver_clauses(occurring_events), **kwargs) as s:
            return (s.solve(), s.time())

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

    def infeasibleQ(self, definitely_occurring_events, potentially_occurring_events, **kwargs):
        with Solver(bootstrap_with=self._sat_solver_clauses_bonus(definitely_occurring_events, potentially_occurring_events), **kwargs) as s:
            return (s.solve(), s.time())



class SupportTesting(SupportTester):
    @property
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
                np.uint).reshape((-1, self.max_conceivable_events, self.nof_observed)),
                               self.observed_cardinalities)

    # @cached_property
    @property
    def unique_candidate_supports(self):
        candidates = np.pad(np.fromiter(itertools.chain.from_iterable(
            itertools.combinations(self.conceivable_events_range[1:], self.nof_events - 1)), np.uint).reshape(
            (-1, self.nof_events - 1)), ((0, 0), (1, 0)), 'constant')
        compressed_candidates = from_digits(candidates, self.event_cardinalities)
        for group_element in self.outcome_relabelling_group[1:]:
            new_candidates = group_element[candidates]
            new_candidates.sort()
            np.minimum(compressed_candidates,
                       from_digits(new_candidates, self.event_cardinalities),
                       out=compressed_candidates)
            candidates = to_digits(compressed_candidates, self.event_cardinalities)
        return np.unique(candidates, axis=0)

    def visualize_supports(self, supports):
        return to_digits(supports, self.observed_cardinalities)

    def devisualize_supports(self, visualized_supports):
        return from_digits(visualized_supports, self.observed_cardinalities)

    def extreme_devisualize_supports(self, visualized_supports):
        return from_digits(self.devisualize_supports(visualized_supports), self.event_cardinalities)

    def unique_infeasible_supports(self, **kwargs):
        """
        Return a signature of infeasible support for a given parents_of, observed_cardinalities, and nof_events
        :param kwargs: optional arguments to pysat.Solver
        In this function I would like to add intelligent filter, such as removing supports which are equivalent under
        graph automorphisms, or which are identified as infeasible due to d-separation and e-separation
        """
        return np.array(
            [occuring_events for occuring_events in self.visualize_supports(self.unique_candidate_supports) if
             not self.feasibleQ(occuring_events, **kwargs)[0]])

    def unique_infeasible_supports_devisualized(self, **kwargs):
        return self.devisualize_supports(self.unique_infeasible_supports(**kwargs))

    def unique_infeasible_supports_extremely_devisualized(self, **kwargs):
        return self.extreme_devisualize_supports(self.unique_infeasible_supports(**kwargs))



class CumulativeSupportTesting:
    def __init__(self, parents_of, observed_cardinalities, max_nof_events):
        self.parents_of = parents_of
        self.observed_cardinalities= observed_cardinalities
        self.max_nof_events = max_nof_events
        self.max_conceivable_events = np.prod(self.observed_cardinalities)
        self.multi_index_shape = np.broadcast_to(self.max_conceivable_events, self.max_nof_events)

    @property
    def _all_infeasible_supports(self):
        for nof_events in range(2, self.max_nof_events+1):
            st = SupportTesting(self.parents_of, self.observed_cardinalities, nof_events)
            #yield st.extreme_devisualize_supports(st.unique_infeasible_supports(name='mgh', use_timer=False))
            yield st.unique_infeasible_supports_extremely_devisualized(name='mgh', use_timer=False)

    @cached_property
    def all_infeasible_supports(self):
        return np.fromiter(itertools.chain.from_iterable(self._all_infeasible_supports), np.uint64)

    @cached_property
    def all_infeasible_supports_matrix(self):
        return to_digits(self.all_infeasible_supports, self.multi_index_shape)


#TODO: Visualize an infeasible support as a SciPy sparce matrix
#TODO: Check n+k event support using n event code.

if __name__ == '__main__':
    parents_of = ([3, 4], [4, 5], [3, 5])
    observed_cardinalities = (2, 2, 2)
    nof_events = 3
    st = SupportTesting(parents_of, observed_cardinalities, nof_events)

    occuring_events_temp = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    print(st.feasibleQ(occuring_events_temp, name='mgh', use_timer=True))
    occuring_events_temp = [(0, 0, 0), (0, 1, 0), (0, 0, 1)]
    print(st.feasibleQ(occuring_events_temp, name='mgh', use_timer=True))

    # st = SupportTesting(parents_of, observed_cardinalities, 2)
    # sample2 = st.unique_candidate_supports
    # # print(st.unique_candidate_supports)
    # # print(st.visualize_supports(st.unique_candidate_supports))
    # inf2 = st.unique_infeasible_supports(name='mgh', use_timer=False)
    # print(inf2)
    # print(st.devisualize_supports(inf2))
    # print(st.extreme_devisualize_supports(inf2))
    # st = SupportTesting(parents_of, observed_cardinalities, 3)
    # sample3 = st.unique_candidate_supports
    # inf3 = st.unique_infeasible_supports(name='mgh', use_timer=False)
    # print(inf3)
    # print(st.devisualize_supports(inf3))
    # print(st.extreme_devisualize_supports(inf3))
    # st = SupportTesting(parents_of, observed_cardinalities, 4)
    # st = SupportTesting(parents_of, [2, 3, 3], 2)
    # print(st.outcome_relabelling_group.shape)
    # st = SupportTesting(parents_of, [2, 3, 4], 2)
    # print(st.outcome_relabelling_group.shape)
    # print(st._array_of_potentially_forbidden_events)
    cst = CumulativeSupportTesting(parents_of, observed_cardinalities, 6)
    print(cst.all_infeasible_supports)
    print(cst.all_infeasible_supports_matrix)

    #Everything appears to work as desired.
    #So let's pick a really hard problem, the square!

    parents_of = ([4, 5, 6], [4, 7, 8], [5, 7, 9], [6, 8, 9])
    observed_cardinalities = (2, 2, 2,2)
    cst = CumulativeSupportTesting(parents_of, observed_cardinalities, 3)
    print(cst.all_infeasible_supports)
    print(cst.all_infeasible_supports_matrix)
    discovered = to_digits(cst.all_infeasible_supports_matrix, observed_cardinalities)
    trulyvariable = discovered.compress(discovered.any(axis=-2).all(axis=-1), axis=0) #makes every variable actually variable
    print(trulyvariable)
    #TODO: it would be good to recognize PRODUCT support matrices. Will be required for d-sep and e-sep filter.
    # see https://www.researchgate.net/post/How-I-can-check-in-MATLAB-if-a-matrix-is-result-of-the-Kronecker-product/542ab19bd039b130378b469d/citation/download?
    # see https://math.stackexchange.com/a/321424



