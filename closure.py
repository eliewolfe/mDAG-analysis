from __future__ import absolute_import
import numpy as np
from typing import List, Tuple, Union
from numpy.typing import NDArray
from sys import version_info
assert version_info >= (3, 8), "Python 3.8+ is required for cached_property support."

from functools import cached_property

from adjmat_class import AdjMat
from scipy.sparse.csgraph import connected_components


IntVector = NDArray[np.int_]
BoolMatrix = NDArray[np.bool_]


def closure_district_step(
    core_B: Union[List[int], IntVector],
    working_B: IntVector,
    ds_adjmat: BoolMatrix,
    sc_adjmat: BoolMatrix,
    return_bidirectedQ: bool = False,
) -> Union[IntVector, Tuple[IntVector, bool]]:
    num_components, component_labels = connected_components(sc_adjmat, directed=False, return_labels=True)
    labels_pertinant_to_core_B = np.unique(component_labels[core_B])
    num_components = len(labels_pertinant_to_core_B)
    new_working_B = np.flatnonzero(np.any(np.equal.outer(component_labels, labels_pertinant_to_core_B), axis=1))
    assert len(working_B) >= len(new_working_B), "Candidate closure should never grow."
    if len(working_B) == len(new_working_B):
        if not return_bidirectedQ:
            return working_B
        else:
            return working_B, (num_components==1)
    else:
        new_ds_adjmat = AdjMat(ds_adjmat).subadjmat(new_working_B)
        # new_sc_adjmat = adjmat_utils.subadjmat(sc_adjmat, new_working_B)
        return closure_ancestors_step(core_B, new_working_B, new_ds_adjmat, sc_adjmat, num_components, return_bidirectedQ=return_bidirectedQ)


def closure_ancestors_step(
    core_B: Union[List[int], IntVector],
    working_B: IntVector,
    ds_adjmat: BoolMatrix,
    sc_adjmat: BoolMatrix,
    num_components: int,
    return_bidirectedQ: bool = False,
) -> Union[IntVector, Tuple[IntVector, bool]]:
    new_working_B = AdjMat(ds_adjmat).ancestorsplus_of(core_B)
    assert len(working_B) >= len(new_working_B), "Candidate closure should never grow."
    if len(working_B) == len(new_working_B):
        if not return_bidirectedQ:
            return working_B
        else:
            return working_B, (num_components==1)
    else:
        # new_ds_adjmat = adjmat_utils.subadjmat(ds_adjmat, new_working_B)
        new_sc_adjmat = AdjMat(sc_adjmat).subadjmat(new_working_B)
        return closure_district_step(core_B, new_working_B, ds_adjmat, new_sc_adjmat, return_bidirectedQ=return_bidirectedQ)


def closure(
    core_B: Union[List[int], IntVector],
    n: int,
    ds_adjmat: BoolMatrix,
    sc_adjmat: BoolMatrix,
    return_bidirectedQ: bool = False,
) -> Union[IntVector, Tuple[IntVector, bool]]:
    return closure_district_step(core_B, np.arange(n+1), ds_adjmat, sc_adjmat, return_bidirectedQ=return_bidirectedQ)
    #Elie: I start with a working set LARGER than the number of visible nodes in order to avoid false termination.


def is_this_subadjmat_densely_connected(sc_adjmat: BoolMatrix, X_indices: Union[List[int], IntVector]) -> bool:
    return (1==connected_components(sc_adjmat[np.ix_(X_indices, X_indices)], directed=False, return_labels=False))
