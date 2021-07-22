from __future__ import absolute_import
import numpy as np
import itertools
from radix import to_digits
from supports import SupportTesting
import progressbar
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
        return any(does_this_esep_rule_out_this_support(*map(list,e_sep), candidate_s) for e_sep in self.esep_relations)

    # @property
    # def _smart_unique_candidate_supports(self):
    #     to_filter = self.visualize_supports(self.unique_candidate_supports)
    #     for idx, candidate_s in enumerate(to_filter):
    #         if not self.trivial_infeasible_support_Q(candidate_s):
    #             yield idx
    @cached_property
    def _trivially_infeasible_support_picklist(self):
        to_filter = self.from_list_to_matrix(self.unique_candidate_supports)
        return np.fromiter(map(self.trivial_infeasible_support_Q, to_filter), bool)

    @cached_property
    def trivially_infeasible_supports_as_integers(self):
        return np.asarray(self.unique_candidate_supports_as_integers, dtype=np.intp)[self._trivially_infeasible_support_picklist]

    @cached_property
    def trivially_infeasible_supports_as_integers(self):
        return self.from_list_to_integer(self._trivially_infeasible_support_picklist)

    @cached_property
    def smart_unique_candidate_supports_as_integers(self):
        return np.asarray(self.unique_candidate_supports_as_integers, dtype=np.intp)[
            np.logical_not(self._trivially_infeasible_support_picklist)]

    def smart_unique_candidate_supports_to_iterate(self, verbose=False):
        if verbose:
            return progressbar.progressbar(
                        self.smart_unique_candidate_supports_as_integers, widgets=[
                            '[nof_events=', str(self.nof_events), '] '
                            , progressbar.SimpleProgress(), progressbar.Bar()
                            , ' (', progressbar.ETA(), ') '])
        else:
            return self.smart_unique_candidate_supports_as_integers


    @methodtools.lru_cache(maxsize=None, typed=False)
    def smart_unique_infeasible_supports(self, verbose=False, **kwargs):
        """
        Return a signature of infeasible support for a given parents_of, observed_cardinalities, and nof_events
        :param kwargs: optional arguments to pysat.Solver
        CHANGED: Now returns each infeasible support as a single integer.
        """
        return [occuring_events_as_int for occuring_events_as_int in self.smart_unique_candidate_supports_to_iterate(verbose) if
             not self.feasibleQ_from_integer(occuring_events_as_int, **kwargs)[0]]

    @methodtools.lru_cache(maxsize=None, typed=False)
    def smart_unique_infeasible_supports_unlabelled(self, **kwargs):
        return np.unique(np.amin(self.from_list_to_integer(
            np.sort(self.universal_relabelling_group[:, self.from_integer_to_list(self.smart_unique_infeasible_supports(**kwargs))])
        ), axis=0))

    @methodtools.lru_cache(maxsize=None, typed=False)
    def no_infeasible_supports_beyond_esep(self, verbose=False, **kwargs):
        return all(self.feasibleQ_from_integer(occuring_events_as_int, **kwargs)[0] for occuring_events_as_int in
                   self.smart_unique_candidate_supports_to_iterate(verbose))




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

    from mDAG_advanced import mDAG
    import networkx as nx
    from hypergraphs import hypergraph
    from directed_structures import directed_structure

    #
    # # Testing example for Ilya conjecture proof.
    directed_dict_of_lists = {"C": ["A", "B"], "X": ["C"], "D": ["A", "B"]}
    ds = directed_structure(["X","A","B","C","D"], nx.from_dict_of_lists(directed_dict_of_lists, create_using=nx.DiGraph).edges())
    # simplicial_complex1 = hypergraph([("A", "X", "D"), ("B", "X", "D"), ("A", "C")])
    # simplicial_complex2 = hypergraph[("A", "X", "D"), ("B", "X", "D")])
    simplicial_complex1 = hypergraph([(1, 0, 4), (2, 0, 4), (1, 3)])
    simplicial_complex2 = hypergraph([(1, 0, 4), (2, 0, 4)])
    #

    md1 = mDAG(ds, simplicial_complex1)
    md2 = mDAG(ds, simplicial_complex2)

    # # print(md1.all_esep)
    # # print(md1.all_esep.difference(md2.all_esep))
    print(md2.all_esep)
    # #print(md1.infeasible_binary_supports_n_events(4))
    # print(md1.smart_infeasible_binary_supports_n_events(4))
    # print(md2.smart_infeasible_binary_supports_n_events(4))
    print(set(md2.smart_infeasible_binary_supports_n_events(4, verbose=True)).difference(md1.smart_infeasible_binary_supports_n_events(4, verbose=True)))
    print(to_digits(to_digits(9786, np.broadcast_to(2 ** 5, 4)), np.broadcast_to(2, 5)))
    # #Cool, it works great.

    #Now testing memoization
    print(set(md2.smart_infeasible_binary_supports_n_events_unlabelled(4, verbose=True)))
    print(set(md2.infeasible_binary_supports_n_events_unlabelled(4, verbose=True)))
