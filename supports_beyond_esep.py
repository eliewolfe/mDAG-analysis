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

def does_this_support_respect_this_pp_restriction(i, pp_set, s):
    if len(pp_set):
        s_resorted = s[np.argsort(s[:, i])]
        # for k, g in itertools.groupby(s_resorted, lambda e: e[i]):
        #     print((k,g))
        partitioned = [np.vstack(tuple(g))[:, np.asarray(pp_set)] for k, g in itertools.groupby(s_resorted, lambda e: e[i])]
        to_test_for_intersection=[set(map(tuple, pp_values)) for pp_values in partitioned]
        raw_length = np.sum([len(vals) for vals in to_test_for_intersection])
        compressed_length = len(set().union(*to_test_for_intersection))
        return (raw_length == compressed_length)
    else:
        return True


def does_this_dsep_rule_out_this_support(ab, C, s):
    if len(C):
        s_resorted = s[np.lexsort(s[:, C].T)]
        partitioned = [np.vstack(tuple(g)) for k, g in itertools.groupby(s_resorted, lambda e: tuple(e.take(C)))]
        conditioning_sizes = range(1, min(len(partitioned)+1, 2 ** len(C)))
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
    if len(D) == 0:
        return does_this_dsep_rule_out_this_support(ab, C, s)
    else:
        columns_D = s[:, D]
        if np.array_equiv(columns_D[0], columns_D):
            return does_this_dsep_rule_out_this_support(ab, C, s)
        else:
            return False


class SmartSupportTesting(SupportTesting):
    def __init__(self, parents_of, observed_cardinalities, nof_events, esep_relations, pp_relations=tuple()):
        super().__init__(parents_of, observed_cardinalities, nof_events)
        self.esep_relations = tuple(esep_relations)
        self.dsep_relations = tuple(((ab, C) for (ab, C, D) in self.esep_relations if len(D)==0))
        self.must_perfectpredict = pp_relations
        self.unique_candidate_supports_as_integers = np.asarray(
            self.unique_candidate_supports_as_integers, dtype=np.intp)[
            self._infeasible_support_respects_restrictions]
        self.unique_candidate_supports = np.asarray(
            self.unique_candidate_supports, dtype=np.intp)[
            self._infeasible_support_respects_restrictions]
        # print("Candidate supports:", self.from_list_to_matrix(self.unique_candidate_supports))


    def infeasible_support_Q_due_to_esep_from_matrix(self, candidate_s):
        return any(does_this_esep_rule_out_this_support(*map(list,e_sep), candidate_s) for e_sep in self.esep_relations)

    def infeasible_support_Q_due_to_dsep_from_matrix(self, candidate_s):
        return any(does_this_dsep_rule_out_this_support(*map(list,d_sep), candidate_s) for d_sep in self.dsep_relations)

    def support_respects_perfect_prediction_restrictions(self, candidate_s):
        return all(does_this_support_respect_this_pp_restriction(
            *pp_restriction, candidate_s) for pp_restriction in self.must_perfectpredict)

    #REDEFINITION!
    def feasibleQ_from_matrix(self, occurring_events, **kwargs):
        if not self.infeasible_support_Q_due_to_esep_from_matrix(occurring_events):
            return super().feasibleQ_from_matrix(occurring_events, **kwargs)
        else:
            return (False, 0) #Timing of zero.

    @cached_property
    def _infeasible_support_respects_restrictions(self):
        to_filter = self.from_list_to_matrix(super().unique_candidate_supports)
        return np.fromiter(map(self.support_respects_perfect_prediction_restrictions, to_filter), bool)

    @cached_property
    def _infeasible_support_due_to_esep_picklist(self):
        to_filter = self.from_list_to_matrix(self.unique_candidate_supports)
        return np.fromiter(map(self.infeasible_support_Q_due_to_esep_from_matrix, to_filter), bool)

    @cached_property
    def infeasible_supports_due_to_esep_as_integers(self):
        return np.asarray(self.unique_candidate_supports_as_integers, dtype=np.intp)[self._infeasible_support_due_to_esep_picklist]

    @cached_property
    def _infeasible_support_due_to_dsep_picklist(self):
        to_filter = self.from_list_to_matrix(self.unique_candidate_supports)
        return np.fromiter(map(self.infeasible_support_Q_due_to_dsep_from_matrix, to_filter), bool)

    @cached_property
    def infeasible_supports_due_to_dsep_as_integers(self):
        return np.asarray(self.unique_candidate_supports_as_integers, dtype=np.intp)[self._infeasible_support_due_to_dsep_picklist]

    @cached_property
    def infeasible_supports_due_to_esep_as_matrices(self):
        return self.from_integer_to_matrix(self.infeasible_supports_due_to_esep_as_integers)

    @cached_property
    def infeasible_supports_due_to_dsep_as_matrices(self):
        return self.from_integer_to_matrix(self.infeasible_supports_due_to_dsep_as_integers)

    @cached_property
    def unique_candidate_supports_not_infeasible_due_to_esep_as_integers(self):
        # print("Supports detected as trivially infeasible:", self.infeasible_supports_due_to_esep_as_integers)
        return np.asarray(self.unique_candidate_supports_as_integers, dtype=np.intp)[
            np.logical_not(self._infeasible_support_due_to_esep_picklist)]

    @cached_property
    def unique_candidate_supports_not_infeasible_due_to_dsep_as_integers(self):
        # print("Supports detected as trivially infeasible:", self.infeasible_supports_due_to_esep_as_integers)
        return np.asarray(self.unique_candidate_supports_as_integers, dtype=np.intp)[
            np.logical_not(self._infeasible_support_due_to_dsep_picklist)]

    def smart_unique_candidate_supports_to_iterate(self, verbose=False):
        if verbose:
            return progressbar.progressbar(
                        self.unique_candidate_supports_not_infeasible_due_to_esep_as_integers, widgets=[
                            '[nof_events=', str(self.nof_events), '] '
                            , progressbar.SimpleProgress(), progressbar.Bar()
                            , ' (', progressbar.ETA(), ') '])
        else:
            return self.unique_candidate_supports_not_infeasible_due_to_esep_as_integers


    @methodtools.lru_cache(maxsize=None, typed=False)
    def unique_infeasible_supports_beyond_esep_as_integers(self, verbose=False, **kwargs):
        """
        Return a signature of infeasible support for a given parents_of, observed_cardinalities, and nof_events
        :param kwargs: optional arguments to pysat.Solver
        CHANGED: Now returns each infeasible support as a single integer.
        """
        return np.fromiter((occuring_events_as_int for occuring_events_as_int in self.smart_unique_candidate_supports_to_iterate(verbose) if
             not self.feasibleQ_from_integer(occuring_events_as_int, **kwargs)[0]), dtype=int)

    @methodtools.lru_cache(maxsize=None, typed=False)
    def unique_infeasible_supports_beyond_dsep_as_integers(self, verbose=False, **kwargs):
        return np.fromiter((occuring_events_as_int for occuring_events_as_int in self.unique_candidate_supports_not_infeasible_due_to_dsep_as_integers if
             not self.feasibleQ_from_integer(occuring_events_as_int, **kwargs)[0]), dtype=int)

    @methodtools.lru_cache(maxsize=None, typed=False)
    def unique_infeasible_supports_beyond_esep_as_matrices(self, **kwargs):
        return self.from_integer_to_matrix(self.unique_infeasible_supports_beyond_esep_as_integers(**kwargs))
    @methodtools.lru_cache(maxsize=None, typed=False)
    def unique_infeasible_supports_beyond_dsep_as_matrices(self, **kwargs):
        return self.from_integer_to_matrix(self.unique_infeasible_supports_beyond_dsep_as_integers(**kwargs))

    @methodtools.lru_cache(maxsize=None, typed=False)
    def unique_infeasible_supports_beyond_esep_as_integers_unlabelled(self, **kwargs):
        # return np.unique(np.amin(self.from_list_to_integer(
        #     np.sort(self.universal_relabelling_group[:, self.from_integer_to_list(self.unique_infeasible_supports_beyond_esep_as_integers(**kwargs))])
        # ), axis=0))
        labelled_infeasible_as_integers = self.unique_infeasible_supports_beyond_esep_as_integers(**kwargs)
        if len(labelled_infeasible_as_integers) > 0:
            labelled_infeasible_as_lists = self.from_integer_to_list(labelled_infeasible_as_integers)
            labelled_variants = self.universal_relabelling_group[:, labelled_infeasible_as_lists]
            labelled_variants.sort(axis=-1)
            labelled_variants = self.from_list_to_integer(labelled_variants).astype(int)
            labelled_variants.sort(axis=-1)
            # print(labelled_variants.shape, labelled_variants.dtype)
            # print(np.unique(labelled_variants, axis=0))
            lexsort = np.lexsort(labelled_variants.T)
            # print(labelled_variants[lexsort[0]])
            return labelled_variants[lexsort[0]]
        else:
            return labelled_infeasible_as_integers




    @methodtools.lru_cache(maxsize=None, typed=False)
    def unique_infeasible_supports_as_integers(self, **kwargs):
        """
        Return a signature of infeasible support for a given parents_of, observed_cardinalities, and nof_events
        :param kwargs: optional arguments to pysat.Solver
        CHANGED: Now returns each infeasible support as a single integer.
        """
        # new_method = np.sort(np.hstack((self.unique_infeasible_supports_beyond_esep_as_integers(**kwargs),
        #                   self.infeasible_supports_due_to_esep_as_integers)))
        # old_method = np.fromiter((occuring_events_as_int for occuring_events_as_int in self.unique_candidate_supports_as_integers if
        #      not self.feasibleQ_from_integer(occuring_events_as_int, **kwargs)[0]), dtype=int)
        # assert np.array_equal(new_method, old_method), "We have a problem!"
        # return new_method
        return np.sort(np.hstack((self.unique_infeasible_supports_beyond_esep_as_integers(**kwargs),
                          self.infeasible_supports_due_to_esep_as_integers)))

    @methodtools.lru_cache(maxsize=None, typed=False)
    def unique_infeasible_supports_as_matrices(self, **kwargs):
        return self.from_integer_to_matrix(self.unique_infeasible_supports_as_integers(**kwargs))

    def unique_infeasible_supports_as_integers_unlabelled(self, **kwargs):
        # return np.unique(np.amin(self.from_list_to_integer(
        #     np.sort(self.universal_relabelling_group[:, self.from_integer_to_list(self.unique_infeasible_supports_as_integers(**kwargs))])
        # ), axis=0))
        labelled_infeasible_as_integers = self.unique_infeasible_supports_as_integers(**kwargs)
        if len(labelled_infeasible_as_integers) > 0:
            labelled_infeasible_as_lists = self.from_integer_to_list(labelled_infeasible_as_integers)
            labelled_variants = self.universal_relabelling_group[:, labelled_infeasible_as_lists]
            labelled_variants.sort(axis=-1)
            labelled_variants = self.from_list_to_integer(labelled_variants).astype(int)
            labelled_variants.sort(axis=-1)
            # print(labelled_variants.shape, labelled_variants.dtype)
            # print(np.unique(labelled_variants, axis=0))
            lexsort = np.lexsort(labelled_variants.T)
            # print(labelled_variants[lexsort[0]])
            return labelled_variants[lexsort[0]]
        else:
            return labelled_infeasible_as_integers


    @methodtools.lru_cache(maxsize=None, typed=False)
    def no_infeasible_supports_beyond_esep(self, **kwargs):
        return all(self.feasibleQ_from_integer(occuring_events_as_int, **kwargs)[0] for occuring_events_as_int in
                   self.unique_candidate_supports_not_infeasible_due_to_esep_as_integers)

    @methodtools.lru_cache(maxsize=None, typed=False)
    def no_infeasible_supports_beyond_dsep(self, **kwargs):
        return all(self.feasibleQ_from_integer(occuring_events_as_int, **kwargs)[0] for occuring_events_as_int in
                   self.unique_candidate_supports_not_infeasible_due_to_dsep_as_integers)

    @cached_property
    def interesting_due_to_esep(self):
        return not set(self.unique_candidate_supports_not_infeasible_due_to_dsep_as_integers).issubset(
            self.unique_candidate_supports_not_infeasible_due_to_esep_as_integers)




if __name__ == '__main__':
    from mDAG_advanced import mDAG
    from hypergraphs import LabelledHypergraph, Hypergraph
    from directed_structures import LabelledDirectedStructure, DirectedStructure

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
