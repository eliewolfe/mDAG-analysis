from __future__ import absolute_import
import numpy as np
import numpy.typing as npt
from sys import hexversion
from typing import List, Tuple, Union

if hexversion >= 0x3080000:
    from functools import cached_property
elif hexversion >= 0x3060000:
    from backports.cached_property import cached_property
else:
    cached_property = property

BoolMatrix = npt.NDArray[np.bool_]
IntVector = npt.NDArray[np.int_]


# Acknowledgement to https://digitalcommons.njit.edu/cgi/viewcontent.cgi?article=2820&context=theses
class AdjMat:
    @staticmethod
    def adjacency_plus_method(adjmat: BoolMatrix) -> BoolMatrix:
        return np.bitwise_or(np.asarray(adjmat, dtype=bool), np.identity(len(adjmat), dtype=bool))

    @staticmethod
    def adjacency_minus_method(adjmat: BoolMatrix) -> BoolMatrix:
        return np.bitwise_and(np.asarray(adjmat, dtype=bool), np.invert(np.identity(len(adjmat), dtype=bool)))

    @staticmethod
    def _children_list(adjmat: BoolMatrix) -> List[frozenset[int]]:
        return list(map(frozenset, map(np.flatnonzero, adjmat)))

    def _parents_list(self, adjmat: BoolMatrix) -> List[frozenset[int]]:
        return self._children_list(np.transpose(adjmat))

    @staticmethod
    def _children_of(adjmat: BoolMatrix, X_indices: Union[int, IntVector]) -> IntVector:
        if hasattr(X_indices, '__iter__'):
            return np.flatnonzero(np.any(adjmat[X_indices], axis=0))
        else:
            return np.flatnonzero(adjmat[X_indices])

    def _parents_of(self, adjmat: BoolMatrix, X_indices: Union[int, IntVector]) -> IntVector:
        return self._children_of(np.transpose(adjmat), X_indices)

    def __init__(self, adjmat: BoolMatrix) -> None:
        self.adjacency_plus = self.adjacency_plus_method(adjmat)
        self.adjacency_minus = self.adjacency_minus_method(adjmat)
        self.childrenplus_list = self._children_list(self.adjacency_plus)
        self.parentsplus_list = self._parents_list(self.adjacency_plus)
        self.children_list = self._children_list(self.adjacency_minus)
        self.parents_list = self._parents_list(self.adjacency_minus)
        self.len = len(adjmat)

    @cached_property
    def numeric_edges(self) -> List[Tuple[int, int]]:
        return [(parent, child) for parent in range(self.len) for child in self.children_list[parent]]

    def childrenplus_of(self, X_indices: Union[int, IntVector]) -> IntVector:
        return self._children_of(self.adjacency_plus, X_indices)

    def parentsplus_of(self, X_indices: Union[int, IntVector]) -> IntVector:
        return self._parents_of(self.adjacency_plus, X_indices)

    def children_of(self, X_indices: Union[int, IntVector]) -> IntVector:
        return self._children_of(self.adjacency_minus, X_indices)

    def parents_of(self, X_indices: Union[int, IntVector]) -> IntVector:
        return self._parents_of(self.adjacency_minus, X_indices)

    @cached_property
    def transitive_closure_plus(self) -> BoolMatrix:
        n = self.len
        closure_mat = self.adjacency_plus.copy()
        while n > 0:
            n = np.floor_divide(n, 2)
            next_closure_mat = np.matmul(closure_mat, closure_mat)
            if np.array_equal(closure_mat, next_closure_mat):
                break
            else:
                closure_mat = next_closure_mat
        return closure_mat

    @cached_property
    def transitive_closure(self) -> BoolMatrix:
        return self.adjacency_minus_method(self.transitive_closure_plus)

    @cached_property
    def transitive_reduction(self) -> BoolMatrix:
        return np.bitwise_and(self.transitive_closure, np.invert(np.matmul(self.transitive_closure, self.transitive_closure)))


    @cached_property
    def descendantsplus_list(self) -> List[frozenset[int]]:
        return self._children_list(self.transitive_closure_plus)

    @cached_property
    def ancestorsplus_list(self) -> List[frozenset[int]]:
        return self._parents_list(self.transitive_closure_plus)

    @cached_property
    def descendants_list(self) -> List[frozenset[int]]:
        return self._children_list(self.transitive_closure)

    @cached_property
    def ancestors_list(self) -> List[frozenset[int]]:
        return self._parents_list(self.transitive_closure)

    @cached_property
    def closure_numeric_edges(self) -> List[Tuple[int, int]]:
        return [(parent, child) for parent in range(self.len) for child in self.descendants_list[parent]]


    def descendantsplus_of(self, X_indices: Union[int, IntVector]) -> IntVector:
        return self._children_of(self.transitive_closure_plus, X_indices)

    def ancestorsplus_of(self, X_indices: Union[int, IntVector]) -> IntVector:
        return self._parents_of(self.transitive_closure_plus, X_indices)

    def descendants_of(self, X_indices: Union[int, IntVector]) -> IntVector:
        return self._children_of(self.transitive_closure, X_indices)

    def ancestors_of(self, X_indices: Union[int, IntVector]) -> IntVector:
        return self._parents_of(self.transitive_closure, X_indices)

    def subadjmat(self, X_indices: IntVector) -> BoolMatrix:
        subindices = np.ix_(X_indices, X_indices)
        submat = np.zeros((self.len, self.len),  dtype=bool)
        submat[subindices] = self.adjacency_minus[subindices]
        return submat


def subhypmat(hypmat: BoolMatrix, X_indices: IntVector) -> BoolMatrix:
    submat = np.zeros_like(hypmat)
    submat[:, X_indices] = hypmat[:, X_indices]
    return submat






if __name__ == '__main__':
    adjmat = np.asarray(np.array(
        [[1, 0, 1, 1, 0, 0],
         [0, 1, 0, 0, 1, 1],
         [0, 1, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 1]], dtype=int), dtype=bool) + np.identity(6, dtype=bool)

    test = AdjMat(adjmat)

    print(test.transitive_closure.astype(int))
    print(test.transitive_reduction.astype(int))
    # print(np.linalg.matrix_power(adjmat, 2).astype(np.int_))
