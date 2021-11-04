from __future__ import absolute_import
import numpy as np
import itertools
from sys import hexversion
if hexversion >= 0x3080000:
    from functools import cached_property
elif hexversion >= 0x3060000:
    from backports.cached_property import cached_property
else:
    cached_property = property

import adjmat_utils
from scipy.sparse.csgraph import connected_components

def closure_district_step(core_B, working_B, ds_adjmat, sc_adjmat, return_bidirectedQ=False):
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
        new_ds_adjmat = adjmat_utils.subadjmat(ds_adjmat, new_working_B)
        new_sc_adjmat = adjmat_utils.subadjmat(sc_adjmat, new_working_B)
        return closure_ancestors_step(core_B, new_working_B, new_ds_adjmat, new_sc_adjmat, num_components, return_bidirectedQ=return_bidirectedQ)

def closure_ancestors_step(core_B, working_B, ds_adjmat, sc_adjmat, num_components, return_bidirectedQ=False):
    new_working_B = adjmat_utils.ancestorsplus_of(ds_adjmat, core_B)
    assert len(working_B) >= len(new_working_B), "Candidate closure should never grow."
    if len(working_B) == len(new_working_B):
        if not return_bidirectedQ:
            return working_B
        else:
            return working_B, (num_components==1)
    else:
        new_ds_adjmat = adjmat_utils.subadjmat(ds_adjmat, new_working_B)
        new_sc_adjmat = adjmat_utils.subadjmat(sc_adjmat, new_working_B)
        return closure_district_step(core_B, new_working_B, new_ds_adjmat, new_sc_adjmat, return_bidirectedQ=return_bidirectedQ)

def closure(core_B, n, ds_adjmat, sc_adjmat, return_bidirectedQ=False):
    return closure_district_step(core_B, np.arange(n+1), ds_adjmat, sc_adjmat, return_bidirectedQ=return_bidirectedQ)
    #Elie: I start with a working set LARGER than the number of visible nodes in order to avoid false termination.

def is_this_subadjmat_densely_connected(sc_adjmat, X_indices):
    return (connected_components(sc_adjmat[np.ix_(X_indices, X_indices)], directed=False, return_labels=False)==1)