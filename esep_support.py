from __future__ import absolute_import
import numpy as np
import itertools
from radix import from_digits, to_digits
from supports import SupportTesting

"""
Suppose that we see a d-sep relation of the form (a,b,C).
Then we take a support and partition into into values on C.
We then take every subset of partitions of cardinality strictly less than 2^|C|

Suppose that we see an e-sep relation of the form (a,b,C,D)
"""


def product_support_test(two_column_mat):
    # exploits the fact that np.unique also sorts
    return np.array_equiv(
        list(itertools.product(*map(np.unique, two_column_mat.T))),
        # np.fromiter(itertools.product(*map(np.unique, two_column_mat.T)),int).reshape((-1,2)),
        np.unique(two_column_mat, axis=0)
    )


def does_this_dsep_rule_out_this_support(ab, C, s):
    if C:
        s_resorted = s[np.lexsort(s[:, C].T)]
        partitioned = [np.vstack(tuple(g)) for k, g in itertools.groupby(s_resorted, lambda e: tuple(e.take(C)))]
        conditioning_sizes = range(1, 2 ** 2)
        for r in conditioning_sizes:
            for partition_index in itertools.combinations(partitioned, r):
                submat = np.vstack(tuple(partition_index))
                if not product_support_test(submat[:, ab]):
                    # print((submat[:, C], submat[:, ab]))
                    return True
        return False
    else:
        return not product_support_test(s[:, ab])


def does_this_esep_rule_out_this_support(ab, C, D, s):
    # print(s)
    # print((ab, C, D))
    if D:
        s_resorted = s[np.lexsort(s[:, D].T)]
        partitioned = [np.vstack(tuple(g)) for k, g in itertools.groupby(s_resorted, lambda e: tuple(e.take(D)))]
        for fixedD in partitioned:
            if does_this_dsep_rule_out_this_support(ab, C, fixedD):
                return True
        return False
    else:
        return does_this_dsep_rule_out_this_support(ab, C, s)

class SmartSupportTesting(SupportTesting):
    def __init__(self, parents_of, observed_cardinalities, nof_events, eseprelations):
        super().__init__(parents_of, observed_cardinalities, nof_events)
        self.eseprelations = tuple(eseprelations)



    def trivial_infeasible_support_Q(self, candidate_s):
        return any(does_this_esep_rule_out_this_support(*e_sep, candidate_s) for e_sep in self.eseprelations)

    # @property
    # def _smart_unique_candidate_supports(self):
    #     to_filter = self.visualize_supports(self.unique_candidate_supports)
    #     for idx, candidate_s in enumerate(to_filter):
    #         if not self.trivial_infeasible_support_Q(candidate_s):
    #             yield idx


    @property
    def smart_unique_candidate_supports(self):
        to_filter = self.visualize_supports(self.unique_candidate_supports)
        indices_to_keep = [idx for idx, candidate_s in enumerate(to_filter) if
                           not self.trivial_infeasible_support_Q(candidate_s)]
        # indices_to_keep = list(self._smart_unique_candidate_supports)
        # print(self.extreme_devisualize_supports(to_filter[indices_to_keep]))
        return self.unique_candidate_supports[indices_to_keep]

    def smart_unique_infeasible_supports(self, **kwargs):
        """
        Return a signature of infeasible support for a given parents_of, observed_cardinalities, and nof_events
        :param kwargs: optional arguments to pysat.Solver
        In this function I would like to add intelligent filter, such as removing supports which are equivalent under
        graph automorphisms, or which are identified as infeasible due to d-separation and e-separation
        """
        return np.array(
            [occuring_events for occuring_events in self.visualize_supports(self.smart_unique_candidate_supports) if
             not self.feasibleQ(occuring_events, **kwargs)[0]])

    def smart_unique_infeasible_supports_devisualized(self, **kwargs):
        return self.devisualize_supports(self.smart_unique_infeasible_supports(**kwargs))

    def smart_unique_infeasible_supports_extremely_devisualized(self, **kwargs):
        return self.extreme_devisualize_supports(self.smart_unique_infeasible_supports(**kwargs))


if __name__ == '__main__':
    # test_two_col_mat = np.asarray([[1, 0], [0, 1], [0, 1], [1, 1]])
    #
    # s = np.asarray([[0, 0, 0, 0, 0], [1, 1, 0, 1, 0], [0, 0, 1, 1, 1], [1, 1, 0, 0, 0], [1, 1, 1, 1, 1]])
    # C = [1, 2]
    # ab = [0, 4]
    # s_resorted = s[np.lexsort(s[:,C].T)]
    # partitioned = [np.vstack(tuple(g)) for k, g in itertools.groupby(s_resorted, lambda e: tuple(e.take(C)))]
    # conditioning_sizes = range(1,2**2)
    # for r in conditioning_sizes:
    #     for partition_index in itertools.combinations(partitioned, r):
    #         submat = np.vstack(tuple(partition_index))
    #         if not product_support_test(submat[:,ab]):
    #             print((submat[:,C], submat[:,ab], product_support_test(submat[:,ab])))

    # print(does_this_dsep_rule_out_this_support(ab, C, s))

    from mDAG import mDAG
    import networkx as nx
    from more_itertools import chunked

    # # Testing example for Ilya conjecture proof.
    directed_dict_of_lists = {"C": ["A", "B"], "X": ["C"], "D": ["A", "B"]}
    directed_structure = nx.from_dict_of_lists(directed_dict_of_lists, create_using=nx.DiGraph)
    simplicial_complex1 = [("A", "X", "D"), ("B", "X", "D"), ("A", "C")]
    simplicial_complex2 = [("A", "X", "D"), ("B", "X", "D")]

    md1 = mDAG(directed_structure, simplicial_complex1)
    md2 = mDAG(directed_structure, simplicial_complex2)
    # print(md1.all_esep)
    # print(list(chunked(map(lambda vars: list(md2.as_integer_labels(tuple(vars))), itertools.chain.from_iterable(md2.all_esep)),3)))
    # print(md1.all_esep.difference(md2.all_esep))
    #print(md1.infeasible_binary_supports_n_events(4))
    print(md1.smart_infeasible_binary_supports_n_events(4))
    print(md2.smart_infeasible_binary_supports_n_events(4))
    print(to_digits(to_digits(9787, np.broadcast_to(2 ** 5, 4)), np.broadcast_to(2, 5)))
    #Cool, it works great.
