from __future__ import absolute_import
import numpy as np
import itertools
from radix import to_digits
from supports import SupportTesting
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

"""
Suppose that we see a d-sep relation of the form (a,b,C).
Then we take a support and partition into into values on C.
We then take every subset of partitions of cardinality strictly less than 2^|C|

Suppose that we see an e-sep relation of the form (a,b,C,D)
"""


def product_support_test(two_column_mat:np. ndarray) -> bool:
    # exploits the fact that np.unique also sorts
    return np.array_equiv(
        list(itertools.product(*map(np.unique, two_column_mat.T))),
        # np.fromiter(itertools.product(*map(np.unique, two_column_mat.T)),int).reshape((-1,2)),
        np.unique(two_column_mat, axis=0)
    )


def does_this_dsep_rule_out_this_support(ab: np.ndarray, C: np.ndarray, s: np.ndarray) -> bool:
    if len(C):
        s_resorted = s[np.lexsort(np.asarray(s)[:, list(C)].T)]
        partitioned = [np.vstack(tuple(g)) for k, g in itertools.groupby(s_resorted, lambda e: tuple(e.take(C)))]
        for submat in partitioned:
            if not product_support_test(submat[:, list(ab)]):
                return True
        return False
    else:
        return not product_support_test(np.asarray(s)[:, list(ab)])


def does_this_esep_rule_out_this_support(ab: np.ndarray, C: np.ndarray, D: np.ndarray, s: np.ndarray) -> bool:
    if len(D) == 0:
        return does_this_dsep_rule_out_this_support(ab, C, s)
    else:
        columns_D = np.asarray(s)[:, np.asarray(D)]
        if np.array_equiv(columns_D[0], columns_D):
            return does_this_dsep_rule_out_this_support(ab, C, s)
        else:
            return False


class SmartSupportTesting(SupportTesting):
    def __init__(self, parents_of, observed_cardinalities, nof_events, esep_relations, pp_relations=tuple()):
        super().__init__(parents_of, observed_cardinalities, nof_events, pp_relations=pp_relations)
        self.esep_relations = tuple(esep_relations)
        self.dsep_relations = tuple(((ab, C) for (ab, C, D) in self.esep_relations if len(D)==0))
        self.must_perfectpredict = pp_relations

    def infeasible_support_Q_due_to_esep_from_matrix(self, candidate_s: np.ndarray) -> bool:
        return any(does_this_esep_rule_out_this_support(*map(list, e_sep), candidate_s) for e_sep in self.esep_relations)

    def infeasible_support_Q_due_to_dsep_from_matrix(self, candidate_s: np.ndarray)-> bool:
        return any(does_this_dsep_rule_out_this_support(*map(list,d_sep), candidate_s) for d_sep in self.dsep_relations)

    #REDEFINITION!
    def feasibleQ_from_matrix(self, occurring_events: np.ndarray, **kwargs) -> bool:
        if not self.infeasible_support_Q_due_to_esep_from_matrix(occurring_events):
            return super().feasibleQ_from_matrix(occurring_events, **kwargs)
        else:
            # return (False, 0) #Timing of zero.
            return False

    # @cached_property
    # def _infeasible_support_respects_restrictions(self):
    #     to_filter = self.from_list_to_matrix(super().unique_candidate_supports_as_lists)
    #     return np.fromiter(map(self.support_respects_perfect_prediction_restrictions, to_filter), bool)

    @cached_property
    def _infeasible_support_due_to_esep_picklist(self) -> np.ndarray:
        to_filter = self.from_list_to_matrix(self.unique_candidate_supports_as_lists)
        return np.fromiter(map(self.infeasible_support_Q_due_to_esep_from_matrix, to_filter), bool)

    @cached_property
    def infeasible_supports_due_to_esep_as_integers(self) -> np.ndarray:
        return np.asarray(self.unique_candidate_supports_as_integers, dtype=self.int_dtype)[self._infeasible_support_due_to_esep_picklist]

    @cached_property
    def _infeasible_support_due_to_dsep_picklist(self) -> np.ndarray:
        to_filter = self.from_list_to_matrix(self.unique_candidate_supports_as_lists)
        return np.fromiter(map(self.infeasible_support_Q_due_to_dsep_from_matrix, to_filter), bool)

    @cached_property
    def infeasible_supports_due_to_dsep_as_integers(self) -> np.ndarray:
        return np.asarray(self.unique_candidate_supports_as_integers, dtype=self.int_dtype)[self._infeasible_support_due_to_dsep_picklist]

    @cached_property
    def infeasible_supports_due_to_esep_as_matrices(self) -> np.ndarray:
        return self.from_integer_to_matrix(self.infeasible_supports_due_to_esep_as_integers)

    @cached_property
    def infeasible_supports_due_to_dsep_as_matrices(self) -> np.ndarray:
        return self.from_integer_to_matrix(self.infeasible_supports_due_to_dsep_as_integers)

    @cached_property
    def unique_candidate_supports_not_infeasible_due_to_esep_as_integers(self) -> np.ndarray:
        return np.asarray(self.unique_candidate_supports_as_integers, dtype=self.int_dtype)[
            np.logical_not(self._infeasible_support_due_to_esep_picklist)]

    @cached_property
    def unique_candidate_supports_not_infeasible_due_to_dsep_as_integers(self) -> np.ndarray:
        return np.asarray(self.unique_candidate_supports_as_integers, dtype=self.int_dtype)[
            np.logical_not(self._infeasible_support_due_to_dsep_picklist)]

    @cached_property
    def unique_candidate_supports_not_infeasible_due_to_esep_as_lists(self) -> np.ndarray:
        return np.asarray(self.unique_candidate_supports_as_lists, dtype=self.int_dtype)[
            np.logical_not(self._infeasible_support_due_to_esep_picklist)]

    @cached_property
    def unique_candidate_supports_not_infeasible_due_to_dsep_as_lists(self) -> np.ndarray:
        return np.asarray(self.unique_candidate_supports_as_lists, dtype=np.intp)[
            np.logical_not(self._infeasible_support_due_to_dsep_picklist)]

    @methodtools.lru_cache(maxsize=None, typed=False)
    def unique_infeasible_supports_beyond_esep_as_integers(self, **kwargs):
        """
        Return a signature of infeasible support for a given parents_of, observed_cardinalities, and nof_events
        :param kwargs: optional arguments to pysat.Solver
        """
        return self.unique_infeasible_supports_as_integers_among(self.unique_candidate_supports_not_infeasible_due_to_esep_as_integers, **kwargs)
        # return np.fromiter((occuring_events_as_int for occuring_events_as_int in self.explore_candidates(self.unique_candidate_supports_not_infeasible_due_to_esep_as_integers, verbose=verbose) if
        #      not self.feasibleQ_from_integer(occuring_events_as_int, **kwargs)), dtype=int)

    @methodtools.lru_cache(maxsize=None, typed=False)
    def unique_infeasible_supports_beyond_dsep_as_integers(self, **kwargs):
        return self.unique_infeasible_supports_as_integers_among(self.unique_candidate_supports_not_infeasible_due_to_dsep_as_integers, **kwargs)

    @methodtools.lru_cache(maxsize=None, typed=False)
    def unique_infeasible_supports_beyond_esep_as_matrices(self, **kwargs):
        return self.from_integer_to_matrix(self.unique_infeasible_supports_beyond_esep_as_integers(**kwargs))
    @methodtools.lru_cache(maxsize=None, typed=False)
    def unique_infeasible_supports_beyond_dsep_as_matrices(self, **kwargs):
        return self.from_integer_to_matrix(self.unique_infeasible_supports_beyond_dsep_as_integers(**kwargs))

    @methodtools.lru_cache(maxsize=None, typed=False)
    def unique_infeasible_supports_beyond_esep_as_integers_unlabelled(self, **kwargs):
        return self.convert_integers_into_canonical_under_relabelling(
            self.unique_infeasible_supports_beyond_esep_as_integers(**kwargs))



    #REDEFINITION
    @methodtools.lru_cache(maxsize=None, typed=False)
    def unique_infeasible_supports_as_integers(self, **kwargs):
        """
        Return a signature of infeasible support for a given parents_of, observed_cardinalities, and nof_events
        :param kwargs: optional arguments to pysat.Solver
        CHANGED: Now returns each infeasible support as a single integer.
        """
        return np.sort(np.hstack((self.unique_infeasible_supports_beyond_esep_as_integers(**kwargs),
                          self.infeasible_supports_due_to_esep_as_integers)).astype(self.int_dtype)).astype(self.int_dtype)

    @methodtools.lru_cache(maxsize=None, typed=False)
    def unique_infeasible_supports_as_matrices(self, **kwargs):
        return self.from_integer_to_matrix(self.unique_infeasible_supports_as_integers(**kwargs))


    def attempt_to_find_one_infeasible_support_beyond_dsep(self, **kwargs):
        return self.attempt_to_find_one_infeasible_support_among(self.unique_candidate_supports_not_infeasible_due_to_dsep_as_lists, **kwargs)
    def attempt_to_find_one_infeasible_support_beyond_esep(self, **kwargs):
        return self.attempt_to_find_one_infeasible_support_among(self.unique_candidate_supports_not_infeasible_due_to_esep_as_lists, **kwargs)

    @methodtools.lru_cache(maxsize=None, typed=False)
    def no_infeasible_supports_beyond_esep(self, **kwargs):
        return self.no_infeasible_supports_among(self.unique_candidate_supports_not_infeasible_due_to_esep_as_lists, **kwargs)

    @methodtools.lru_cache(maxsize=None, typed=False)
    def no_infeasible_supports_beyond_dsep(self, **kwargs):
        return self.no_infeasible_supports_among(self.unique_candidate_supports_not_infeasible_due_to_dsep_as_lists,
                                                 **kwargs)

    @cached_property
    def interesting_due_to_esep(self):
        return not set(self.unique_candidate_supports_not_infeasible_due_to_dsep_as_integers).issubset(
            self.unique_candidate_supports_not_infeasible_due_to_esep_as_integers)




if __name__ == '__main__':
    from mDAG_advanced import mDAG
    from hypergraphs import LabelledHypergraph, Hypergraph
    from directed_structures import LabelledDirectedStructure, DirectedStructure

    MarinaTest = mDAG(directed_structure_instance=DirectedStructure([], 4),
                      simplicial_complex_instance=Hypergraph([], 4))
    st_instance = MarinaTest.support_testing_instance_binary(2)
    result = st_instance.feasibleQ_from_tuple((0, 14))
    # print("Result:", result)
    # print("Result:", result[0])
    print(st_instance.feasibleQ_from_tuple((0, 14)))
    print(st_instance.feasibleQ_from_integer(14))
    print(st_instance.feasibleQ_from_matrix(np.array([[0,0,0,0],[1,1,1,1]])))
    # print(st_instance.no_infeasible_supports())
    #
    print(does_this_dsep_rule_out_this_support([1,2], [0], np.asarray([[0,0,0,0],[0,1,1,0]])))
    print(does_this_dsep_rule_out_this_support([0,3], [1,2], np.asarray([[0, 0, 0, 0], [1, 1, 1, 1]])))

    # latent_free1 = mDAG(DirectedStructure([(0, 1), (0, 2), (1, 3), (2, 3)], 4), Hypergraph([], 4))
    # print(latent_free1.all_CI)
    # print("From dsep:",
    # latent_free1.support_testing_instance_binary(2).infeasible_supports_due_to_dsep_as_matrices)
    # print("From esep:",
    # latent_free1.support_testing_instance_binary(2).infeasible_supports_due_to_esep_as_matrices)
    # print(latent_free1.no_esep_beyond_dsep_up_to(2))
    #
    # latent_free2 = mDAG(DirectedStructure([(0, 3), (1, 3), (2, 3)], 4), Hypergraph([], 4))
    # latent_free2.no_infeasible_binary_supports_beyond_dsep_up_to(4)
    # from radix import to_bits
    # test_two_col_mat = np.asarray([[1, 0], [0, 1], [0, 1], [1, 1]])
    # instrumental = mDAG(
    #     LabelledDirectedStructure([0,1,2],[(0,1),(1,2)]),
    #     LabelledHypergraph([0,1,2], [(0,),(1,2)]))
    # UC = mDAG(LabelledDirectedStructure([0,1,2],[(1,0),(1,2)]),
    #           LabelledHypergraph([0,1,2],[(0,1),(1,2)]))
    #
    # print(instrumental.infeasible_binary_supports_n_events(3, verbose=False))
    # print(UC.infeasible_binary_supports_n_events(3, verbose=False))
    # print(instrumental.infeasible_binary_supports_n_events_beyond_esep(3, verbose=False))
    # print(UC.infeasible_binary_supports_n_events_beyond_esep(3, verbose=False))
    # print(instrumental.infeasible_binary_supports_n_events(4, verbose=False))
    # print(UC.infeasible_binary_supports_n_events(4, verbose=False))
    # print(instrumental.infeasible_binary_supports_n_events_beyond_esep(4, verbose=False))
    # print(UC.infeasible_binary_supports_n_events_beyond_esep(4, verbose=False))
    # print(to_bits(instrumental.infeasible_binary_supports_n_events(3, verbose=False), mantissa=3))
    # print(to_bits(UC.infeasible_binary_supports_n_events(3, verbose=False), mantissa=3))
    # print(to_bits(instrumental.infeasible_binary_supports_n_events_beyond_esep(3, verbose=False), mantissa=3))
    # print(to_bits(UC.infeasible_binary_supports_n_events_beyond_esep(3, verbose=False), mantissa=3))
    # print(to_bits(instrumental.infeasible_binary_supports_n_events(4, verbose=False), mantissa=3))
    # print(to_bits(UC.infeasible_binary_supports_n_events(4, verbose=False), mantissa=3))
    # print(to_bits(instrumental.infeasible_binary_supports_n_events_beyond_esep(4, verbose=False), mantissa=3))
    # print(to_bits(UC.infeasible_binary_supports_n_events_beyond_esep(4, verbose=False), mantissa=3))


    # print(set(instrumental.infeasible_binary_supports_n_events(4, verbose=False)).difference(
    #     UC.infeasible_binary_supports_n_events(4, verbose=False)))
    # print(set(instrumental.infeasible_binary_supports_n_events_beyond_esep(4, verbose=False)).difference(
    #     UC.infeasible_binary_supports_n_events_beyond_esep(4, verbose=False)))


    s = np.asarray(
        [[0, 0, 0, 0, 0],
         [1, 1, 0, 1, 0],
         [0, 0, 1, 1, 1],
         [1, 1, 0, 0, 0],
         [1, 1, 1, 1, 1]])
    C = [1, 2]
    ab = [0, 4]


    print(does_this_dsep_rule_out_this_support(ab, C, s))


    import networkx as nx


    #
    # # Testing example for Ilya conjecture proof.
    directed_dict_of_lists = {"C": ["A", "B"], "X": ["C"], "D": ["A", "B"]}
    ds = LabelledDirectedStructure(["X", "A", "B", "C", "D"], nx.from_dict_of_lists(directed_dict_of_lists, create_using=nx.DiGraph).edges())


    simplicial_complex1 = LabelledHypergraph(["X", "A", "B", "C", "D"], [("A", "X", "D"), ("B", "X", "D"), ("A", "C")])
    simplicial_complex2 = LabelledHypergraph(["X", "A", "B", "C", "D"], [("A", "X", "D"), ("B", "X", "D")])


    md1 = mDAG(ds, simplicial_complex1)
    md2 = mDAG(ds, simplicial_complex2)
    print("Hashing test: ", hash(simplicial_complex1), hash(ds), hash(md1))
    test = set([simplicial_complex1, simplicial_complex2])


    print(md2.all_esep)

    print(set(md2.infeasible_binary_supports_n_events_beyond_esep(4, verbose=True)).difference(md1.infeasible_binary_supports_n_events_beyond_esep(4, verbose=True)))
    print(to_digits(to_digits(9786, np.broadcast_to(2 ** 5, 4)), np.broadcast_to(2, 5)))
    # #Cool, it works great.

    #Now testing memoization
    print(set(md2.infeasible_binary_supports_n_events_beyond_esep_unlabelled(4, verbose=True)))
    print(set(md2.infeasible_binary_supports_n_events_unlabelled(4, verbose=True)))
