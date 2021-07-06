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
        conditioning_sizes = range(1, len(C) ** 2)
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
    # We check for D being a point distribution

    # print(s)
    # print((ab, C, D))
    columns_D = s[:, D]
    if len(D) > 0 and np.array_equiv(columns_D[0], columns_D):
        s_resorted = s[np.lexsort(columns_D.T)]
        partitioned = [np.vstack(tuple(g)) for k, g in itertools.groupby(s_resorted, lambda e: tuple(e.take(D)))]
        for fixedD in partitioned:
            if does_this_dsep_rule_out_this_support(ab, C, fixedD):
                return True
        return False
    else:
        return does_this_dsep_rule_out_this_support(ab, C, s)


class SmartSupportTesting(SupportTesting):
    def __init__(self, parents_of, observed_cardinalities, nof_events, esep_relations):
        super().__init__(parents_of, observed_cardinalities, nof_events)
        self.esep_relations = tuple(esep_relations)

    def trivial_infeasible_support_Q(self, candidate_s):
        return any(does_this_esep_rule_out_this_support(*e_sep, candidate_s) for e_sep in self.esep_relations)

    # @property
    # def _smart_unique_candidate_supports(self):
    #     to_filter = self.visualize_supports(self.unique_candidate_supports)
    #     for idx, candidate_s in enumerate(to_filter):
    #         if not self.trivial_infeasible_support_Q(candidate_s):
    #             yield idx

    @property
    def smart_unique_candidate_supports(self):
        to_filter = self.from_list_to_matrix(self.unique_candidate_supports)
        indices_to_keep = [idx for idx, candidate_s in enumerate(to_filter) if
                           not self.trivial_infeasible_support_Q(candidate_s)]
        # indices_to_keep = list(self._smart_unique_candidate_supports)
        # print(self.extreme_devisualize_supports(to_filter[indices_to_keep]))
        return self.unique_candidate_supports[indices_to_keep]

    def smart_unique_infeasible_supports(self, **kwargs):
        """
        Return a signature of infeasible support for a given parents_of, observed_cardinalities, and nof_events
        :param kwargs: optional arguments to pysat.Solver
        CHANGED: Now returns each infeasible support as a single integer.
        """
        return self.from_matrix_to_integer(
            [occuring_events for occuring_events in self.from_list_to_matrix(self.smart_unique_candidate_supports) if
             not self.feasibleQ(occuring_events, **kwargs)[0]])

    def smart_unique_infeasible_supports_unlabelled(self, **kwargs):
        return np.unique(np.amin(self.from_list_to_integer(
            np.sort(self.universal_relabelling_group[:, self.from_integer_to_list(self.smart_unique_infeasible_supports())])
        ), axis=0))




if __name__ == '__main__':
    # test_two_col_mat = np.asarray([[1, 0], [0, 1], [0, 1], [1, 1]])

    s = np.asarray(
        [[0, 0, 0, 0, 0],
         [1, 1, 0, 1, 0],
         [0, 0, 1, 1, 1],
         [1, 1, 0, 0, 0],
         [1, 1, 1, 1, 1]])
    C = [1, 2]
    ab = [0, 4]
    # s_resorted = s[np.lexsort(s[:,C].T)]
    # partitioned = [np.vstack(tuple(g)) for k, g in itertools.groupby(s_resorted, lambda e: tuple(e.take(C)))]
    # conditioning_sizes = range(1,2**2)
    # for r in conditioning_sizes:
    #     for partition_index in itertools.combinations(partitioned, r):
    #         submat = np.vstack(tuple(partition_index))
    #         if not product_support_test(submat[:,ab]):
    #             print((submat[:,C], submat[:,ab], product_support_test(submat[:,ab])))

    print(does_this_dsep_rule_out_this_support(ab, C, s))

    from mDAG import mDAG
    import networkx as nx

    #
    # # Testing example for Ilya conjecture proof.
    directed_dict_of_lists = {"C": ["A", "B"], "X": ["C"], "D": ["A", "B"]}
    directed_structure = nx.from_dict_of_lists(directed_dict_of_lists, create_using=nx.DiGraph)
    simplicial_complex1 = [("A", "X", "D"), ("B", "X", "D"), ("A", "C")]
    simplicial_complex2 = [("A", "X", "D"), ("B", "X", "D")]
    #
    md1 = mDAG(directed_structure, simplicial_complex1)
    md2 = mDAG(directed_structure, simplicial_complex2)
    # # print(md1.all_esep)
    # # print(md1.all_esep.difference(md2.all_esep))
    # #print(md1.infeasible_binary_supports_n_events(4))
    # print(md1.smart_infeasible_binary_supports_n_events(4))
    # print(md2.smart_infeasible_binary_supports_n_events(4))
    print(set(md2.smart_infeasible_binary_supports_n_events(4)).difference(md1.smart_infeasible_binary_supports_n_events(4)))
    print(to_digits(to_digits(9786, np.broadcast_to(2 ** 5, 4)), np.broadcast_to(2, 5)))
    # #Cool, it works great.
