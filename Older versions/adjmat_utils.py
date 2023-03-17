import numpy as np
# from utilities import partsextractor
# import itertools


# Acknowledgement to https://digitalcommons.njit.edu/cgi/viewcontent.cgi?article=2820&context=theses

def adjacency_plus(adjmat):
    return np.bitwise_or(np.asarray(adjmat, dtype=bool), np.identity(len(adjmat), dtype=bool))
def adjacency_minus(adjmat):
    return np.bitwise_and(np.asarray(adjmat, dtype=bool), np.invert(np.identity(len(adjmat), dtype=bool)))

def _children_list(adjmat):
    return list(map(frozenset, map(np.flatnonzero, adjmat)))
def _parents_list(adjmat):
    return _children_list(np.transpose(adjmat))
def _children_of(adjmat, X_indices):
    if hasattr(X_indices, '__iter__'):
        return np.flatnonzero(np.any(adjmat[X_indices], axis=0))
    else:
        return np.flatnonzero(adjmat[X_indices])
def _parents_of(adjmat, X_indices):
    return _children_of(np.transpose(adjmat), X_indices)


def childrenplus_list(adjmat):
    return _children_list(adjacency_plus(adjmat))
def parentsplus_list(adjmat):
    return _parents_list(adjacency_plus(adjmat))
def children_list(adjmat):
    return _children_list(adjacency_minus(adjmat))
def parents_list(adjmat):
    return _parents_list(adjacency_minus(adjmat))

def childrenplus_of(adjmat, X_indices):
    return _children_of(adjacency_plus(adjmat), X_indices)
def parentsplus_of(adjmat, X_indices):
    return _parents_of(adjacency_plus(adjmat), X_indices)
def children_of(adjmat, X_indices):
    return _children_of(adjacency_minus(adjmat), X_indices)
def parents_of(adjmat, X_indices):
    return _parents_of(adjacency_minus(adjmat), X_indices)


def transitive_closure_plus(adjmat):
    n = len(adjmat)
    closure_mat = adjacency_plus(adjmat)
    while n > 0:
        n = np.floor_divide(n, 2)
        next_closure_mat = np.matmul(closure_mat, closure_mat)
        if np.array_equal(closure_mat, next_closure_mat):
            break
        else:
            closure_mat = next_closure_mat
    return closure_mat
def transitive_closure(adjmat):
    return adjacency_minus(transitive_closure_plus(adjmat))


def descendantsplus_list(adjmat):
    return _children_list(transitive_closure_plus(adjmat))
def ancestorsplus_list(adjmat):
    return _parents_list(transitive_closure_plus(adjmat))
def descendants_list(adjmat):
    return _children_list(transitive_closure(adjmat))
def ancestors_list(adjmat):
    return _parents_list(transitive_closure(adjmat))

def descendantsplus_of(adjmat, X_indices):
    return _children_of(transitive_closure_plus(adjmat), X_indices)
def ancestorsplus_of(adjmat, X_indices):
    return _parents_of(transitive_closure_plus(adjmat), X_indices)
def descendants_of(adjmat, X_indices):
    return _children_of(transitive_closure(adjmat), X_indices)
def ancestors_of(adjmat, X_indices):
    return _parents_of(transitive_closure(adjmat), X_indices)


def transitive_reduction(adjmat):
    n = len(adjmat)
    closure_mat = transitive_closure(adjmat)
    return np.bitwise_and(closure_mat, np.invert(np.matmul(closure_mat, closure_mat)))

def subadjmat(adjmat, X_indices):
    subindices = np.ix_(X_indices,X_indices)
    submat = np.zeros_like(adjmat)
    submat[subindices] = adjmat[subindices]
    return submat

def subhypmat(hypmat, X_indices):
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

    print(transitive_closure(adjmat).astype(int))
    print(transitive_reduction(adjmat).astype(int))
    # print(np.linalg.matrix_power(adjmat, 2).astype(np.int_))
    print(transitive_closure(adjmat[:4]).astype(int))