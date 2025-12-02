from __future__ import absolute_import
import networkx as nx
import numpy as np
import numpy.typing as npt
from operator import itemgetter
from radix import bitarray_to_int
import itertools
from typing import Any, Iterable, Tuple, Union, List, Set

BoolMatrix = npt.NDArray[np.bool_]
IntVector = npt.NDArray[np.int_]


def partsextractor(thing_to_take_parts_of: Any, indices: Union[int, Iterable[int]]) -> Union[Any, Tuple[Any, ...]]:
    if hasattr(indices, '__iter__'):
        if len(indices) == 0:
            return tuple()
        elif len(indices) == 1:
            return (itemgetter(*indices)(thing_to_take_parts_of),)
        else:
            return itemgetter(*indices)(thing_to_take_parts_of)
    else:
        return itemgetter(indices)(thing_to_take_parts_of)



def nx_to_tuples(g: nx.Graph) -> Tuple[Tuple[Any, Any], ...]:
    return tuple(sorted(g.edges()))


def nx_to_bitarray(g: nx.Graph) -> BoolMatrix:
    # ds_bitarray = nx.to_numpy_array(g, nodelist=sorted(g.nodes()), dtype=bool)
    # print(ds_bitarray)
    return nx.to_numpy_array(g, nodelist=sorted(g.nodes()), dtype=bool)


def nx_to_int(g: nx.Graph) -> np.integer:
    # return nx_to_tuples(g)
    # return bitarray_to_int(nx_to_bitarray(g)).astype(np.ulonglong).tolist()
    # Concern about int64 overflow.
    return bitarray_to_int(nx_to_bitarray(g))


def hypergraph_to_canonical_tuples(hypergraph: Iterable[Iterable[Any]]) -> Tuple[Tuple[Any, ...], ...]:
    return tuple(sorted(map(lambda s: tuple(sorted(s)), hypergraph)))


def hypergraph_to_bitarray(integers_hypergraph: Iterable[Iterable[int]]) -> BoolMatrix: #Now works with sets.
    #nof_nodes = np.hstack(integers_hypergraph).max() + 1
    nof_nodes = max(map(max, integers_hypergraph)) + 1
    integers_hypergraph_compressed = [list(lp) for lp in integers_hypergraph if len(lp)>1]
    r = np.zeros((len(integers_hypergraph_compressed), nof_nodes), dtype=bool)
    for i, lp in enumerate(integers_hypergraph_compressed):
        r[i, list(lp)] = True
    return r[np.lexsort(r.T)]


def hypergraph_to_int(integers_hypergraph: Iterable[Iterable[int]]) -> np.integer:
    # Concern about int64 overflow
    # return bitarray_to_int(hypergraph_to_bitarray(integers_hypergraph)).astype(np.ulonglong).tolist()
    return bitarray_to_int(hypergraph_to_bitarray(integers_hypergraph))


def representatives(eqclasses: Iterable[Iterable[Any]]) -> List[Any]:
    return [next(iter(eqclass)) for eqclass in eqclasses]

def convert_eqclass_dict_to_representatives_dict(d: dict) -> None:
        for k, v in d.items():
            d[k] = next(iter(v))

def mdag_to_int(ds_bitarray: BoolMatrix, sc_bitarray: BoolMatrix) -> np.integer:
    # Note simplicial complex ABOVE directed structure, as SC is always square
    # Concern about int64 overflow
    # return bitarray_to_int(np.vstack((sc_bitarray, ds_bitarray))).astype(np.ulonglong).tolist()
    return bitarray_to_int(np.vstack((sc_bitarray, ds_bitarray)))


def bitarrays_permutations(ds_bitarray: BoolMatrix, sc_bitarray: BoolMatrix) -> Iterable[np.integer]:
    nof_observed = len(ds_bitarray)
    for perm in map(list, itertools.permutations(range(nof_observed))):
        new_ds = ds_bitarray[perm][:, perm]
        almost_new_sc = sc_bitarray[:, perm]
        new_sc = almost_new_sc[np.lexsort(almost_new_sc.T)]
        # if not np.all(new_sc.sum(axis=-1)):
        #     print(sc_bitarray)
        yield mdag_to_int(new_ds, new_sc)


def mdag_to_canonical_int(ds_bitarray: BoolMatrix, sc_bitarray: BoolMatrix) -> int:
    # nof_observed = len(ds_bitarray)
    # # print(sc_bitarray)
    # return min(mdag_to_int(
    #         ds_bitarray[perm][:, perm],
    #         sc_bitarray[np.lexsort(sc_bitarray[:, perm].T)][:, perm]) for
    #      perm in map(list, itertools.permutations(range(nof_observed))))
    return min(bitarrays_permutations(ds_bitarray, sc_bitarray))

def stringify_in_tuple(l: Iterable[Any]) -> str:
    return '(' + ','.join(map(str, l)) + ')'
def stringify_in_list(l: Iterable[Any]) -> str:
    return '[' + ','.join(map(str, l)) + ']'
def stringify_in_set(l: Iterable[Any]) -> str:
    return '{' + ','.join(map(str, l)) + '}'

def minimal_sets_within(list_of_sets: List[Set[Any]]) -> List[Set[Any]]:
    original_list_copy = list_of_sets.copy()
    verified_minimal = []
    for i in range(len(list_of_sets)):
        candidate_minimal_set = original_list_copy.pop()
        if not any(counterexample.issubset(candidate_minimal_set) for counterexample in (original_list_copy + verified_minimal)):
            verified_minimal.append(candidate_minimal_set)
    return verified_minimal

def maximal_sets_within(list_of_sets: List[Set[Any]]) -> List[Set[Any]]:
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
