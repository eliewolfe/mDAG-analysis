from __future__ import absolute_import
import networkx as nx
import numpy as np
from operator import itemgetter
from radix import bitarray_to_int

def bitarray_to_lex_int(bitarray: np.ndarray) -> int:
    for_viewing = np.packbits(bitarray.ravel(), axis=-1, bitorder='little')
    return int.from_bytes(for_viewing.tolist(), byteorder='little', signed=False)


def partsextractor(thing_to_take_parts_of, indices):
    if hasattr(indices, '__iter__'):
        if len(indices) == 0:
            return tuple()
        elif len(indices) == 1:
            return (itemgetter(*indices)(thing_to_take_parts_of),)
        else:
            return itemgetter(*indices)(thing_to_take_parts_of)
    else:
        return itemgetter(indices)(thing_to_take_parts_of)



def nx_to_tuples(g):
    return tuple(sorted(g.edges()))


def nx_to_bitarray(g):
    # ds_bitarray = nx.to_numpy_array(g, nodelist=sorted(g.nodes()), dtype=bool)
    # print(ds_bitarray)
    return nx.to_numpy_array(g, nodelist=sorted(g.nodes()), dtype=bool)


def nx_to_int(g):
    # return nx_to_tuples(g)
    # return bitarray_to_int(nx_to_bitarray(g)).astype(np.ulonglong).tolist()
    # Concern about int64 overflow.
    return bitarray_to_lex_int(nx_to_bitarray(g))


def hypergraph_to_canonical_tuples(hypergraph):
    return tuple(sorted(map(lambda s: tuple(sorted(s)), hypergraph)))


def hypergraph_to_bitarray(integers_hypergraph): #Now works with sets.
    #nof_nodes = np.hstack(integers_hypergraph).max() + 1
    nof_nodes = max(map(max, integers_hypergraph)) + 1
    integers_hypergraph_compressed = [list(lp) for lp in integers_hypergraph if len(lp)>1]
    r = np.zeros((len(integers_hypergraph_compressed), nof_nodes), dtype=bool)
    for i, lp in enumerate(integers_hypergraph_compressed):
        r[i, list(lp)] = True
    return r[np.lexsort(r.T)]


def hypergraph_to_int(integers_hypergraph):
    # Concern about int64 overflow
    # return bitarray_to_int(hypergraph_to_bitarray(integers_hypergraph)).astype(np.ulonglong).tolist()
    return bitarray_to_lex_int(hypergraph_to_bitarray(integers_hypergraph))


def representatives(eqclasses):
    return [next(iter(eqclass)) for eqclass in eqclasses]

def convert_eqclass_dict_to_representatives_dict(d):
        for k, v in d.items():
            d[k] = next(iter(v))

# def mdag_to_int(ds_bitarray, sc_bitarray):
#     # Note simplicial complex ABOVE directed structure, as SC is always square
#     # Concern about int64 overflow
#     # return bitarray_to_int(np.vstack((sc_bitarray, ds_bitarray))).astype(np.ulonglong).tolist()
#     return bitarray_to_int(np.vstack((sc_bitarray, ds_bitarray)))
#
#
# def bitarrays_permutations(ds_bitarray, sc_bitarray):
#     nof_observed = len(ds_bitarray)
#     for perm in map(list, itertools.permutations(range(nof_observed))):
#         new_ds = ds_bitarray[perm][:, perm]
#         almost_new_sc = sc_bitarray[:, perm]
#         new_sc = almost_new_sc[np.lexsort(almost_new_sc.T)]
#         # if not np.all(new_sc.sum(axis=-1)):
#         #     print(sc_bitarray)
#         yield mdag_to_int(new_ds, new_sc)
#
#
# def mdag_to_canonical_int(ds_bitarray, sc_bitarray) -> int:
#     # nof_observed = len(ds_bitarray)
#     # # print(sc_bitarray)
#     # return min(mdag_to_int(
#     #         ds_bitarray[perm][:, perm],
#     #         sc_bitarray[np.lexsort(sc_bitarray[:, perm].T)][:, perm]) for
#     #      perm in map(list, itertools.permutations(range(nof_observed))))
#     return min(bitarrays_permutations(ds_bitarray, sc_bitarray))

def stringify_in_tuple(l) -> str:
    return '(' + ','.join(map(str, l)) + ')'
def stringify_in_list(l) -> str:
    return '[' + ','.join(map(str, l)) + ']'
def stringify_in_set(l) -> str:
    return '{' + ','.join(map(str, l)) + '}'

def minimal_sets_within(list_of_sets):
    original_list_copy = list_of_sets.copy()
    verified_minimal = []
    for i in range(len(list_of_sets)):
        candidate_minimal_set = original_list_copy.pop()
        if not any(counterexample.issubset(candidate_minimal_set) for counterexample in (original_list_copy + verified_minimal)):
            verified_minimal.append(candidate_minimal_set)
    return verified_minimal

def maximal_sets_within(list_of_sets):
    original_list_copy = list_of_sets.copy()
    verified_maximal = []
    for i in range(len(list_of_sets)):
        candidate_maximal_set = original_list_copy.pop()
        if not any(counterexample.issuperset(candidate_maximal_set) for counterexample in (original_list_copy + verified_maximal)):
            verified_maximal.append(candidate_maximal_set)
    return verified_maximal



if __name__ == '__main__':
    # print(partsextractor({'a': 1, 'b': 2}, ['a', 'b']))
    print(maximal_sets_within([{1, 2}, {1, 3}, {1, 2, 3}]))
    print(maximal_sets_within([{1}, {1, 2}, {1, 3}]))
    print(minimal_sets_within([{1, 2}, {1, 3}, {1, 2, 3}]))
    print(minimal_sets_within([{1}, {1, 2}, {1, 3}]))
