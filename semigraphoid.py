"""
Semigraphoid closure over elementary triplets, and a bit-parallel d-separation enumerator.

A conditional-independence model over n variables (indices 0..n-1) is stored by its ELEMENTARY triplets <i,j|K>,
i != j, K a subset of the other variables, in a dense boolean array E[i, j, K] indexed by the bitmask of K (cells
with i or j in K are unused; E is kept symmetric in i, j). A semigraphoid is determined by its elementary triplets:
a general triplet <A,B|C> belongs to it iff <i,j|K> does for every i in A, j in B and every K with
C <= K <= (A u B u C) minus {i,j} (Studený). A set of elementary triplets is the elementary part of a semigraphoid
iff it is closed under the single exchange rule (Matúš; Studený, Probabilistic Conditional Independence
Structures, Lemma 2.2): for distinct i, j, k and K disjoint from them,

        <i,j|K> and <i,k|K+j>      if and only if      <i,k|K> and <i,j|K+k>,

so iterating that rule to a fixpoint computes the semigraphoid closure (`close`). The rule is the elementary form
of contraction, weak union and decomposition at once; every instance is a Shannon-type identity, so whatever the
closure derives, the entropic LP of entropic_lp.py derives too (the converse fails in general: Shannon implication
is stronger than semigraphoid implication). A functional dependence H(s|X) = 0 enters as the elementary triplets
<s,j|K> for all K containing X (`add_functional_dependence`): that is its whole content for conditional-mutual-
information targets, since H(s|X) is a sum of such terms along any ordering of the other variables.

`dsep_all` enumerates the elementary d-separations of a DAG given as parent bitmasks, by Bayes-ball reachability on
bitmasks, one walk per (source, conditioning set); the result is the d-separation model in the same layout, so
Markov properties are containment checks (`contains`).

Kernels are compiled with numba when it is installed (MDAG_SEMIGRAPHOID_NUMBA=0 forces the numpy fallback); the
numpy fallback is an independent implementation, which the tests compare cell by cell with the compiled one.
"""
from __future__ import annotations

import os
import time
from typing import Dict, Iterable, List, Optional

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = os.environ.get('MDAG_SEMIGRAPHOID_NUMBA', '1') != '0'
except ImportError:   # pragma: no cover - exercised on machines without numba
    njit = None
    HAVE_NUMBA = False

STATS: Dict[str, float] = {'closures': 0, 'sweeps': 0, 'dsep_models': 0, 'seconds': 0.0}


# --------------------------------------------------------------------------------------------------
# Layout helpers
# --------------------------------------------------------------------------------------------------

def mask_of(variables: Iterable[int]) -> int:
    m = 0
    for v in variables:
        m |= 1 << int(v)
    return m


def bits_of(mask: int) -> List[int]:
    out = []
    v = 0
    while mask:
        if mask & 1:
            out.append(v)
        mask >>= 1
        v += 1
    return out


def submasks(mask: int) -> np.ndarray:
    """All submasks of `mask` as an int64 array (2^popcount(mask) entries, ascending)."""
    bits = bits_of(mask)
    out = np.zeros(1, dtype=np.int64)
    for b in bits:
        out = np.concatenate([out, out | (1 << b)])
    out.sort()
    return out


def empty_model(n: int) -> np.ndarray:
    return np.zeros((n, n, 1 << n), dtype=np.bool_)


def valid_cells(n: int, excluded: int = 0) -> np.ndarray:
    """Boolean array marking the cells (i, j, K) that carry a triplet: i != j, K disjoint from {i, j}, and none of
    i, j, K in the `excluded` mask."""
    masks = np.arange(1 << n, dtype=np.int64)
    valid = np.zeros((n, n, 1 << n), dtype=np.bool_)
    for i in range(n):
        if (excluded >> i) & 1:
            continue
        for j in range(n):
            if j == i or (excluded >> j) & 1:
                continue
            valid[i, j] = (masks & ((1 << i) | (1 << j) | excluded)) == 0
    return valid


def parents_to_masks(parents: Dict[int, Iterable[int]], n: int) -> np.ndarray:
    pa = np.zeros(n, dtype=np.int64)
    for v, ps in parents.items():
        pa[int(v)] = mask_of(ps)
    return pa


# --------------------------------------------------------------------------------------------------
# Triplets
# --------------------------------------------------------------------------------------------------

def add_triplet(E: np.ndarray, A: int, B: int, C: int) -> None:
    """Adds every elementary component of <A,B|C> (A, B, C disjoint bitmasks)."""
    assert A & B == 0 and A & C == 0 and B & C == 0, "a triplet has disjoint sides"
    union = A | B | C
    for i in bits_of(A):
        for j in bits_of(B):
            extra = union & ~((1 << i) | (1 << j) | C)
            Ks = C | submasks(extra)
            E[i, j, Ks] = True
            E[j, i, Ks] = True


def add_functional_dependence(E: np.ndarray, s: int, X: int) -> None:
    """H(s|X) = 0 as elementary triplets: <s,j|K> for every K with X <= K, s not in K, j not in K + s."""
    n = E.shape[0]
    full = (1 << n) - 1
    assert not (X >> s) & 1, "a variable does not determine itself"
    for j in range(n):
        if j == s or (X >> j) & 1:
            continue
        extra = full & ~(X | (1 << s) | (1 << j))
        Ks = X | submasks(extra)
        E[s, j, Ks] = True
        E[j, s, Ks] = True


def holds(E: np.ndarray, A: int, B: int, C: int) -> bool:
    """Does <A,B|C> hold in the (closed) model E, i.e. do all its elementary components?"""
    assert A & B == 0 and A & C == 0 and B & C == 0, "a triplet has disjoint sides"
    if A == 0 or B == 0:
        return True
    union = A | B | C
    for i in bits_of(A):
        for j in bits_of(B):
            extra = union & ~((1 << i) | (1 << j) | C)
            if not E[i, j, C | submasks(extra)].all():
                return False
    return True


def contains(E: np.ndarray, F: np.ndarray) -> bool:
    """Is every triplet of F (on its valid cells, which are the only ones F sets) in E?"""
    return not np.any(F & ~E)


def restrict(E: np.ndarray, allowed: int) -> np.ndarray:
    """E with every cell touching a variable outside `allowed` cleared (a copy)."""
    n = E.shape[0]
    return E & valid_cells(n, excluded=((1 << n) - 1) & ~allowed)


# --------------------------------------------------------------------------------------------------
# The closure kernel
# --------------------------------------------------------------------------------------------------

def _close_python(E: np.ndarray, n: int) -> int:
    """Exchange rule to a fixpoint; returns the number of sweeps. Plain Python source, compiled by numba."""
    full = (1 << n) - 1
    sweeps = 0
    while True:
        changed = False
        for i in range(n):
            bi = 1 << i
            for j in range(n):
                if j == i:
                    continue
                bj = 1 << j
                for k in range(j + 1, n):
                    if k == i:
                        continue
                    bk = 1 << k
                    rest = full & ~(bi | bj | bk)
                    sub = rest
                    while True:
                        K = sub
                        left = E[i, j, K] and E[i, k, K | bj]
                        right = E[i, k, K] and E[i, j, K | bk]
                        if left != right:
                            E[i, j, K] = True
                            E[j, i, K] = True
                            E[i, k, K | bj] = True
                            E[k, i, K | bj] = True
                            E[i, k, K] = True
                            E[k, i, K] = True
                            E[i, j, K | bk] = True
                            E[j, i, K | bk] = True
                            changed = True
                        if sub == 0:
                            break
                        sub = (sub - 1) & rest
        sweeps += 1
        if not changed:
            return sweeps


def _close_numpy(E: np.ndarray, n: int) -> int:
    """The same rule, vectorised over K for each (i; j, k): the independent fallback."""
    full = (1 << n) - 1
    masks = np.arange(1 << n, dtype=np.int64)
    sweeps = 0
    while True:
        changed = False
        for i in range(n):
            bi = 1 << i
            for j in range(n):
                if j == i:
                    continue
                bj = 1 << j
                for k in range(j + 1, n):
                    if k == i:
                        continue
                    bk = 1 << k
                    Ks = masks[(masks & (bi | bj | bk)) == 0]
                    left = E[i, j, Ks] & E[i, k, Ks | bj]
                    right = E[i, k, Ks] & E[i, j, Ks | bk]
                    new = Ks[left != right]
                    if new.size:
                        for a, b, extra in ((i, j, 0), (i, k, bj), (i, k, 0), (i, j, bk)):
                            E[a, b, new | extra] = True
                            E[b, a, new | extra] = True
                        changed = True
        sweeps += 1
        if not changed:
            return sweeps


# --------------------------------------------------------------------------------------------------
# d-separation by Bayes ball on bitmasks
# --------------------------------------------------------------------------------------------------

def _dsep_python(E: np.ndarray, n: int, m: int, pa: np.ndarray, ch: np.ndarray) -> None:
    """E[i, j, K] = i and j are d-separated given K, for every source i, target j and conditioning set K among the
    first m of the n nodes (K not containing i or j); the walk itself runs over all n nodes.
    Bayes ball (Shachter): a ball reaching an unobserved node from a child goes on to its parents and children,
    one reaching an observed node from a child stops; a ball reaching an unobserved node from a parent goes on to
    its children, one reaching an observed node from a parent bounces to its parents."""
    full = (1 << m) - 1
    for i in range(m):
        bi = 1 << i
        rest = full & ~bi
        sub = rest
        while True:
            K = sub
            up = bi          # visited "from a child" (the source counts as such)
            down = 0         # visited "from a parent"
            frontier_up = bi
            frontier_down = 0
            while frontier_up != 0 or frontier_down != 0:
                new_up = 0
                new_down = 0
                f = frontier_up
                while f != 0:
                    v = 0
                    while not (f >> v) & 1:
                        v += 1
                    f &= f - 1
                    if not (K >> v) & 1:
                        new_up |= pa[v]
                        new_down |= ch[v]
                f = frontier_down
                while f != 0:
                    v = 0
                    while not (f >> v) & 1:
                        v += 1
                    f &= f - 1
                    if (K >> v) & 1:
                        new_up |= pa[v]
                    else:
                        new_down |= ch[v]
                frontier_up = new_up & ~up
                frontier_down = new_down & ~down
                up |= new_up
                down |= new_down
            reached = up | down
            for j in range(m):
                if j != i and not (K >> j) & 1:
                    E[i, j, K] = not (reached >> j) & 1
            if sub == 0:
                break
            sub = (sub - 1) & rest


def _dsep_numpy(E: np.ndarray, n: int, m: int, pa: np.ndarray, ch: np.ndarray) -> None:
    """The same walk, vectorised over all conditioning sets for each source."""
    masks = np.arange(1 << m, dtype=np.int64)
    for i in range(m):
        bi = 1 << i
        Ks = masks[(masks & bi) == 0]
        observed = [((Ks >> v) & 1).astype(bool) for v in range(n)]
        up = np.full(Ks.shape, bi, dtype=np.int64)
        down = np.zeros(Ks.shape, dtype=np.int64)
        frontier_up = up.copy()
        frontier_down = down.copy()
        while frontier_up.any() or frontier_down.any():
            new_up = np.zeros(Ks.shape, dtype=np.int64)
            new_down = np.zeros(Ks.shape, dtype=np.int64)
            for v in range(n):
                in_up = ((frontier_up >> v) & 1).astype(bool) & ~observed[v]
                new_up |= np.where(in_up, pa[v], 0)
                new_down |= np.where(in_up, ch[v], 0)
                in_down = ((frontier_down >> v) & 1).astype(bool)
                new_up |= np.where(in_down & observed[v], pa[v], 0)
                new_down |= np.where(in_down & ~observed[v], ch[v], 0)
            frontier_up = new_up & ~up
            frontier_down = new_down & ~down
            up |= new_up
            down |= new_down
        reached = up | down
        for j in range(m):
            if j == i:
                continue
            free = ~observed[j]
            E[i, j, Ks[free]] = ((reached[free] >> j) & 1) == 0


if HAVE_NUMBA:
    _close_kernel = njit(cache=True)(_close_python)
    _dsep_kernel = njit(cache=True)(_dsep_python)
else:
    _close_kernel = _close_numpy
    _dsep_kernel = _dsep_numpy


def close(E: np.ndarray, kernel=None) -> int:
    """Semigraphoid closure of E in place (the exchange rule to a fixpoint); returns the number of sweeps."""
    t0 = time.perf_counter()
    sweeps = int((kernel or _close_kernel)(E, E.shape[0]))
    STATS['closures'] += 1
    STATS['sweeps'] += sweeps
    STATS['seconds'] += time.perf_counter() - t0
    return sweeps


def dsep_all(n: int, parents_masks: np.ndarray, m: Optional[int] = None, kernel=None) -> np.ndarray:
    """The elementary d-separation model of the DAG with the given parent bitmasks (int64 array of length n). With
    `m`, only the first m nodes serve as sources, targets and conditioning variables (the observed block of a
    structure whose observed nodes come first): the result has shape (m, m, 2^m) and equals the leading block of
    the full model, at a cost exponential in m instead of n."""
    t0 = time.perf_counter()
    pa = np.asarray(parents_masks, dtype=np.int64)
    assert pa.shape == (n,)
    m = n if m is None else m
    assert 0 < m <= n
    ch = np.zeros(n, dtype=np.int64)
    for v in range(n):
        for p in bits_of(int(pa[v])):
            ch[p] |= 1 << v
    E = empty_model(m)
    (kernel or _dsep_kernel)(E, n, m, pa, ch)
    STATS['dsep_models'] += 1
    STATS['seconds'] += time.perf_counter() - t0
    return E


def dsep_model_of(n: int, parents: Dict[int, Iterable[int]], m: Optional[int] = None) -> np.ndarray:
    """dsep_all for a parent map {node: parents} over the nodes 0..n-1."""
    return dsep_all(n, parents_to_masks(parents, n), m=m)


def local_markov_model(n: int, parents_masks: np.ndarray) -> np.ndarray:
    """The elementary expansions of the local Markov triplets <v, ND(v) minus Pa(v) | Pa(v)> (a generating set of
    the d-separation model; its closure equals `dsep_all`)."""
    pa = np.asarray(parents_masks, dtype=np.int64)
    full = (1 << n) - 1
    desc = np.zeros(n, dtype=np.int64)
    changed = True
    ch = np.zeros(n, dtype=np.int64)
    for v in range(n):
        for p in bits_of(int(pa[v])):
            ch[p] |= 1 << v
    desc[:] = ch
    while changed:
        changed = False
        for v in range(n):
            new = desc[v]
            for d in bits_of(int(desc[v])):
                new |= desc[d]
            if new != desc[v]:
                desc[v] = new
                changed = True
    E = empty_model(n)
    for v in range(n):
        nondesc = full & ~(int(desc[v]) | (1 << v) | int(pa[v]))
        if nondesc:
            add_triplet(E, 1 << v, nondesc, int(pa[v]))
    return E


def warm_up() -> None:
    """Compiles the kernels (a few seconds the first time on a machine; cached afterwards)."""
    E = empty_model(3)
    close(E)
    dsep_all(3, np.array([0, 1, 2], dtype=np.int64))
