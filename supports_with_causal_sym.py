from supports import SupportTester
import numpy as np
import itertools
from functools import cached_property
import methodtools

class SupportTester_PartyCausalSymmetry(SupportTester):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_cardinality = self.nof_observed * self.nof_events
        self.observed_cardinality = self.observed_cardinalities[0]
        self.visible_outcomes = list(range(self.observed_cardinality))
        self.latent_outcomes = list(range(self.latent_cardinality))
        del self.var

    def var(self, i, j, val):
        return self.vpool.id(f"A_[{i},{j}]=={val}")

    @property
    def at_least_one_outcome(self):
        # return list(map(sorted, set(map(frozenset, super().at_least_one_outcome))))
        return [[self.var(i, j, val) for val in self.visible_outcomes] for (i, j) in itertools.product(self.latent_outcomes, repeat=2)]

    @methodtools.lru_cache(maxsize=None, typed=False)
    def forbidden_event_clauses(self, event: int):
        """Get the clauses associated with a particular event not occurring anywhere in the off-diagonal worlds."""
        forbidden_event_as_row = self.from_list_to_matrix(event)
        [val1, val2, val3] = forbidden_event_as_row
        forbidden_event_clauses = set() # np.empty((self.latent_cardinality ** 3, 3), dtype=int)
        for idx, (i, j, k) in enumerate(itertools.combinations(self.latent_outcomes, 3)):
            no_go_clause = tuple(sorted([-self.var(i, j, val1), -self.var(j, k, val2), -self.var(k, i, val3)], key=abs))
            forbidden_event_clauses.add(no_go_clause)
        # forbidden_event_clauses.sort(axis=-1)
        # forbidden_event_clauses = np.unique(forbidden_event_clauses, axis=0).tolist()
        forbidden_event_clauses = list(map(list, forbidden_event_clauses))
        return forbidden_event_clauses

    # def forbidden_events_clauses(self, event: int):
    #     return list(map(sorted, set(map(frozenset, self.forbidden_events_clauses(event)))))

    @methodtools.lru_cache(maxsize=None, typed=False)
    def positive_outcome_clause(self,
                                 world: int,
                                 outcomes: tuple):
        i = world * self.observed_cardinality
        for p, val in enumerate(outcomes):
            if p == 0:
                left_s = i
                right_s = i + 1
            elif p == 1:
                left_s = i + 1
                right_s = i + 2
            elif p == 2:
                left_s = i + 2
                right_s = i
            yield [self.var(left_s, right_s, val)]
            wrong_values = set(range(self.observed_cardinalities[p]))
            wrong_values.remove(outcomes[p])
            for wrong_val in wrong_values:
                yield [-self.var(left_s, right_s, wrong_val)]




    # def array_of_positive_outcomes(self, occurring_events: np.ndarray):
    #     # return list(map(sorted,
    #     #          set(map(frozenset, super().array_of_positive_outcomes(occurring_events)))))
    #     wrong_values_template = set(self.visible_outcomes)
    #     for raw_i, iterator in enumerate(occurring_events):
    #         i = raw_i * self.observed_cardinality
    #         for p, val in enumerate(iterator):
    #             if p == 0:
    #                 left_s = i
    #                 right_s = i + 1
    #             elif p == 1:
    #                 left_s = i + 1
    #                 right_s = i + 2
    #             elif p == 2:
    #                 left_s = i + 2
    #                 right_s = i
    #             yield [self.var(left_s, right_s, val)]
    #             for wrong_val in wrong_values_template.difference({val}):
    #                 yield [-self.var(left_s, right_s, wrong_val)]







class SupportTester_FullCausalSymmetry(SupportTester_PartyCausalSymmetry):
    def var(self, i, j, val):
        [new_i, new_j] = sorted([i, j])
        return self.vpool.id(f"A_[{new_i},{new_j}]=={val}")

if __name__ == '__main__':
    import itertools

# Special problem for Victor Gitton
    parents_of = np.array(([3, 4], [4, 5], [5, 3]))
    observed_cardinalities = (3, 3, 3)
    occurring_events_card3 = np.array([(0, 0, 0), (1, 1, 1), (2, 2, 2)]
                                       + list(itertools.permutations(range(3))))


    st = SupportTester(parents_of=parents_of,
                       observed_cardinalities=observed_cardinalities,
                       nof_events=len(occurring_events_card3))
    print("3 outcome with no symmetry: ",
          st.feasibleQ_from_matrix(occurring_events_card3, name='mgh'))

    st = SupportTester_PartyCausalSymmetry(parents_of=parents_of,
                                           observed_cardinalities=observed_cardinalities,
                                           nof_events=len(occurring_events_card3))
    # for clause in st._sat_solver_clauses(occurring_events_card3):
    #     print(st.reverse_vars(clause))
    print("3 outcome with only party symmetry: ",
          st.feasibleQ_from_matrix(occurring_events_card3, name='mgh'))
    # print("At-least-one-outcome clauses: \n", sorted(st.reverse_vars(
    #     st.at_least_one_outcome)))
    # print("Occurring event clauses: \n", st.reverse_vars(st.array_of_positive_outcomes(occurring_events_card3)))
    # print("Some forbidden event clauses: \n", sorted(st.reverse_vars(st.forbidden_event_clauses(st.from_matrix_to_list([[0, 1, 1]])[0]))))
    # print("Some forbidden event clauses: \n", sorted(st.reverse_vars(next(iter(st._forbidden_event_clauses.values())))))
    st = SupportTester_FullCausalSymmetry(parents_of=parents_of,
                                           observed_cardinalities=observed_cardinalities,
                                           nof_events=len(occurring_events_card3))
    # for clause in st._sat_solver_clauses(occurring_events_card3):
    #     print(st.reverse_vars(clause))
    print("3 outcome with full party symmetry: ",
          st.feasibleQ_from_matrix_CONSERVATIVE(occurring_events_card3, name='mgh'))


    observed_cardinalities = (4, 4, 4)
    occurring_events_card4 = np.array([(0, 0, 0),
                                       (1, 1, 1),
                                       (2, 2, 2),
                                       (3, 3, 3)]
                                       + list(
        itertools.permutations(range(4), 3)))


    st = SupportTester_PartyCausalSymmetry(parents_of=parents_of,
                       observed_cardinalities=observed_cardinalities,
                       nof_events=len(occurring_events_card4))
    print("ready")
    print("4 outcome with Party symmetry: ",
          st.feasibleQ_from_matrix(occurring_events_card4, name='mgh'))

    # max_n = len(occurring_events_card4)
    # rejected_yet = False
    # for n in range(2, max_n+1):
    #     if rejected_yet:
    #         break
    #     print(f"Working on cardinality 4 with {n} definite events...")
    #     st = SupportTester_PartyCausalSymmetry(parents_of=parents_of,
    #                        observed_cardinalities=(4, 4, 4),
    #                        nof_events=n)
    #     # for definitely_occurring_events in itertools.combinations(occurring_events_card_4, n):
    #     definitely_occurring_events = occurring_events_card4[:n]
    #     rejected_yet = not st.potentially_feasibleQ_from_matrix_pair(
    #         definitely_occurring_events_matrix=definitely_occurring_events,
    #         potentially_occurring_events_matrix=occurring_events_card4
    #     )
    #     if rejected_yet:
    #         print(f"search completed at {n} events")
    #         print(definitely_occurring_events)
    #         break
    # print("In the end, were we able to prove infeasibility? ", rejected_yet)
