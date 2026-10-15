"""
Canonical forms of directed structures, simplicial complices and mDAGs under relabelling, by permutation tables.

An mDAG's unlabelled id (mDAG.unique_unlabelled_id) is min over the relabellings p of ds_int(p) + sc_int(p) * 2^(n^2),
where ds_int is the integer of the n x n adjacency matrix (radix.bitarray_to_int: row-major, first bit most
significant) and sc_int the integer of the lexsorted facet bit array (np.lexsort(rows.T): the last column is the
primary key). Since the simplicial complex occupies the high bits, the minimum is lexicographic: first the smallest
sc_int over all relabellings, then the smallest ds_int over the relabellings that achieve it (a coset of the
complex's automorphism group). The tables below give every labelled structure's integer under every permutation, so
that canonical ids, automorphism groups and orbits are table lookups; the kernels are compiled with numba when it is
available (semigraphoid.HAVE_NUMBA) and fall back to the same Python code otherwise. Integers must fit in 64 bits:
n^2 <= 62 for the directed structures (n <= 7) and L * n <= 62 for a complex with L facets (every complex of up to
five nodes); `fits_int64` says when the tables apply.
"""
from __future__ import annotations

import itertools
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from semigraphoid import HAVE_NUMBA, njit


def permutations_array(n: int) -> np.ndarray:
    """All permutations of range(n) in itertools order (the identity first), as an int64 array (n!, n)."""
    return np.array(list(itertools.permutations(range(n))), dtype=np.int64)


def fits_int64(n: int, max_facets: int) -> bool:
    return n * n <= 62 and max_facets * n <= 62


# --------------------------------------------------------------------------------------------------
# Kernels (plain Python, compiled by numba when available)
# --------------------------------------------------------------------------------------------------

def _ds_table_python(mats: np.ndarray, perms: np.ndarray) -> np.ndarray:
    """mats: (D, n, n) bool adjacency matrices; returns (D, n!) int64: the integer of mats[d][p][:, p] for every
    permutation p, with the first bit (row 0, column 0) most significant."""
    D, n, _ = mats.shape
    P = perms.shape[0]
    out = np.zeros((D, P), dtype=np.int64)
    for d in range(D):
        for k in range(P):
            value = 0
            for i in range(n):
                pi = perms[k, i]
                for j in range(n):
                    value = (value << 1) | (1 if mats[d, pi, perms[k, j]] else 0)
            out[d, k] = value
    return out


def _sorted_rows_int(row_keys: np.ndarray, count: int, n: int) -> int:
    """The integer of the bit array whose rows are the masks row_keys[:count] (bit c = column c) after sorting
    them as np.lexsort(rows.T) does: ascending by the key sum_c bit_c 2^c (the last column most significant), then
    encoded row-major with the first bit most significant (radix.bitarray_to_int)."""
    # insertion sort of the small key array (facets are distinct, so there are no ties)
    keys = row_keys[:count].copy()
    for i in range(1, count):
        x = keys[i]
        j = i - 1
        while j >= 0 and keys[j] > x:
            keys[j + 1] = keys[j]
            j -= 1
        keys[j + 1] = x
    value = 0
    for r in range(count):
        for c in range(n):
            value = (value << 1) | ((keys[r] >> c) & 1)
    return value


def _sc_table_python(facets: np.ndarray, counts: np.ndarray, perms: np.ndarray, n: int) -> np.ndarray:
    """facets: (S, Lmax) int64 masks (bit c = node c), counts[s] facets in complex s; returns (S, n!) int64: the
    integer of the relabelled, re-sorted facet bit array under every permutation p (new column c = old column
    p[c], as hypergraphs.permute_bit_array)."""
    S, Lmax = facets.shape
    P = perms.shape[0]
    out = np.zeros((S, P), dtype=np.int64)
    keys = np.zeros(Lmax, dtype=np.int64)
    for s in range(S):
        count = counts[s]
        for k in range(P):
            for r in range(count):
                mask = facets[s, r]
                new = 0
                for c in range(n):
                    if (mask >> perms[k, c]) & 1:
                        new |= 1 << c
                keys[r] = new
            out[s, k] = _sorted_rows_int(keys, count, n)
    return out


def _sc_int_python(facets: np.ndarray, count: int, n: int) -> int:
    """The integer of a complex given as facet masks (unsorted)."""
    return _sorted_rows_int(facets, count, n)


if HAVE_NUMBA:
    _ds_table = njit(cache=True)(_ds_table_python)
    _sorted_rows_int_jit = njit(cache=True)(_sorted_rows_int)

    # the compiled kernels must call the compiled row encoder, so they are separate copies of the Python ones
    @njit(cache=True)
    def _sc_table_impl(facets, counts, perms, n):
        S, Lmax = facets.shape
        P = perms.shape[0]
        out = np.zeros((S, P), dtype=np.int64)
        keys = np.zeros(Lmax, dtype=np.int64)
        for s in range(S):
            count = counts[s]
            for k in range(P):
                for r in range(count):
                    mask = facets[s, r]
                    new = 0
                    for c in range(n):
                        if (mask >> perms[k, c]) & 1:
                            new |= 1 << c
                    keys[r] = new
                out[s, k] = _sorted_rows_int_jit(keys, count, n)
        return out

    @njit(cache=True)
    def _sc_int_impl(facets, count, n):
        return _sorted_rows_int_jit(facets, count, n)

    _sc_table = _sc_table_impl

    def _sc_int(facets: np.ndarray, count: int, n: int) -> int:
        return int(_sc_int_impl(facets, count, n))
else:
    _ds_table = _ds_table_python
    _sc_table = _sc_table_python
    _sc_int = _sc_int_python


# --------------------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------------------

def directed_structure_table(matrices: Sequence[np.ndarray], n: int) -> np.ndarray:
    """(D, n!) integers of the given adjacency matrices under every permutation."""
    mats = np.array([np.asarray(m, dtype=np.bool_) for m in matrices], dtype=np.bool_).reshape(len(matrices), n, n)
    return _ds_table(mats, permutations_array(n))


def facet_masks(complexes: Iterable[Iterable[Iterable[int]]], n: int) -> Tuple[np.ndarray, np.ndarray]:
    """The facets of each complex as node bitmasks, padded: (S, Lmax) int64 and the counts (S,)."""
    rows = [[sum(1 << int(v) for v in facet) for facet in sc] for sc in complexes]
    Lmax = max((len(r) for r in rows), default=0)
    out = np.zeros((len(rows), max(Lmax, 1)), dtype=np.int64)
    counts = np.zeros(len(rows), dtype=np.int64)
    for s, r in enumerate(rows):
        out[s, :len(r)] = r
        counts[s] = len(r)
    return out, counts


def simplicial_complex_table(complexes: Sequence[Iterable[Iterable[int]]], n: int) -> np.ndarray:
    """(S, n!) integers of the given complexes (facets as node iterables) under every permutation."""
    masks, counts = facet_masks(complexes, n)
    return _sc_table(masks, counts, permutations_array(n), n)


def simplicial_complex_int(facets: Iterable[int], n: int) -> int:
    """The integer of a complex given by its facet masks (Hypergraph.as_integer of the same facets)."""
    masks = np.array(list(facets), dtype=np.int64)
    if masks.size == 0:
        return 0
    return _sc_int(masks, masks.size, n)


class CanonicalForms:
    """Canonical ids of the mDAGs over n visible nodes, from the permutation tables of the labelled directed
    structures (`ds_ints`, `ds_matrices`: in the metagraph, the 2^C(n,2) structures whose edges point up the node
    order; every directed structure is a relabelling of one of them) and of all labelled simplicial complices
    (`sc_ints`, `sc_facets`).

    canonical(ds_int, sc_int) is mDAG.unique_unlabelled_id of the mDAG with those parts, for any tabulated parts
    (a bounded memo keeps the recent answers). `nodes` enumerates one labelled representative per unlabelled mDAG
    the way the metagraph always has: for each unlabelled directed structure's representative d, the complices up
    to the automorphism group of d. `lower_neighbours` gives the canonical ids one edge or one maximal face
    below a node, which is what the structural dominances of the metagraph are."""

    MEMO_LIMIT = 1 << 20

    def __init__(self, n: int, ds_ints: Sequence[int], ds_matrices: Sequence[np.ndarray],
                 sc_ints: Sequence[int], sc_facets: Sequence[Iterable[Iterable[int]]]) -> None:
        assert fits_int64(n, max((len(list(f)) for f in sc_facets), default=0)), "the tables need 64-bit integers"
        self.n = n
        self.ds_offset = 2 ** (n * n)
        self.ds_ints = np.array([int(v) for v in ds_ints], dtype=np.int64)
        self.sc_ints = np.array([int(v) for v in sc_ints], dtype=np.int64)
        self.ds_index: Dict[int, int] = {int(v): i for i, v in enumerate(self.ds_ints)}
        self.sc_index: Dict[int, int] = {int(v): i for i, v in enumerate(self.sc_ints)}
        self.ds_table = directed_structure_table(ds_matrices, n)
        self.sc_masks, self.sc_counts = facet_masks(sc_facets, n)
        self.sc_table = _sc_table(self.sc_masks, self.sc_counts, permutations_array(n), n)
        assert np.array_equal(self.ds_table[:, 0], self.ds_ints), "directed structure integers disagree with radix"
        assert np.array_equal(self.sc_table[:, 0], self.sc_ints), "simplicial complex integers disagree with radix"
        self._memo: Dict[int, int] = {}
        self._facet_removals: Dict[int, Tuple[int, ...]] = {}

    # -- ids ----------------------------------------------------------------------------------------------

    def split(self, mdag_int: int) -> Tuple[int, int]:
        return mdag_int % self.ds_offset, mdag_int // self.ds_offset

    def join(self, ds_int: int, sc_int: int) -> int:
        return int(ds_int) + int(sc_int) * self.ds_offset

    def stabiliser_coset(self, sc_int: int) -> Tuple[int, np.ndarray]:
        """(canonical integer of the complex, the permutation indices that achieve it)."""
        row = self.sc_table[self.sc_index[int(sc_int)]]
        s = int(row.min())
        return s, np.flatnonzero(row == s)

    def canonical(self, ds_int: int, sc_int: int) -> int:
        """mDAG.unique_unlabelled_id of the mDAG (ds_int, sc_int): the complex occupies the high bits of the id,
        so the minimum is the canonical complex first, then the smallest directed structure over the relabellings
        that achieve it."""
        mdag_int = self.join(ds_int, sc_int)
        try:
            return self._memo[mdag_int]
        except KeyError:
            pass
        s, coset = self.stabiliser_coset(sc_int)
        d = int(self.ds_table[self.ds_index[int(ds_int)], coset].min())
        value = self.join(d, s)
        if len(self._memo) >= self.MEMO_LIMIT:
            self._memo.clear()
        self._memo[mdag_int] = value
        return value

    def canonical_of_int(self, mdag_int: int) -> int:
        return self.canonical(*self.split(int(mdag_int)))

    def __contains__(self, mdag_int: int) -> bool:
        ds_int, sc_int = self.split(int(mdag_int))
        return ds_int in self.ds_index and sc_int in self.sc_index

    # -- nodes --------------------------------------------------------------------------------------------

    def ds_automorphisms(self, ds_int: int) -> np.ndarray:
        """Indices of the permutations fixing the directed structure."""
        row = self.ds_table[self.ds_index[int(ds_int)]]
        return np.flatnonzero(row == int(ds_int))

    def sc_orbits(self, ds_int: int) -> List[Tuple[int, int]]:
        """The orbits of the complices under Aut(ds), as (first, last) table indices of each orbit, in the order
        of their first members."""
        aut = self.ds_automorphisms(ds_int)
        keys = self.sc_table[:, aut].min(axis=1)
        _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
        last = np.zeros(len(first), dtype=np.int64)
        last[inverse] = np.arange(len(keys))          # the last assignment to each orbit wins
        order = np.argsort(first)
        return [(int(first[k]), int(last[k])) for k in order]

    def nodes(self, ds_representatives: Sequence[int]) -> List[Tuple[int, int, int]]:
        """One labelled representative per unlabelled mDAG, as (canonical id, ds_int, sc_int), in the order and
        with the representatives of the metagraph's original construction (the dict over complex x directed
        structure keyed by unlabelled id: a key's position is that of its first labelled member, its value the
        last). `ds_representatives` are the integers of the unlabelled directed structures' representatives."""
        entries = []
        for position, ds_int in enumerate(ds_representatives):
            for first, last in self.sc_orbits(ds_int):
                sc_int = int(self.sc_ints[last])
                entries.append((first, position, self.canonical(ds_int, sc_int), int(ds_int), sc_int))
        entries.sort()
        return [entry[2:] for entry in entries]

    # -- lower neighbours ---------------------------------------------------------------------------------

    def facet_removals(self, sc_int: int) -> Tuple[int, ...]:
        """The integers of the complices with one maximal face of the complex removed (the face's maximal proper
        subsets of two or more nodes not covered by another facet become facets)."""
        try:
            return self._facet_removals[int(sc_int)]
        except KeyError:
            pass
        i = self.sc_index[int(sc_int)]
        facets = [int(m) for m in self.sc_masks[i, :self.sc_counts[i]]]
        out = []
        for F in facets:
            others = [G for G in facets if G != F]
            for bit in range(self.n):
                sub = F & ~(1 << bit)
                if sub != F and bin(sub).count('1') >= 2 and not any(sub & ~G == 0 for G in others):
                    others.append(sub)
            out.append(simplicial_complex_int(others, self.n))
        result = tuple(out)
        self._facet_removals[int(sc_int)] = result
        return result

    def lower_neighbour_parts(self, ds_int: int, sc_int: int) -> List[Tuple[int, int]]:
        """(ds_int, sc_int) of the labelled mDAGs one edge or one maximal face below (ds_int, sc_int)."""
        ds_int, sc_int = int(ds_int), int(sc_int)
        out = []
        rest = ds_int
        while rest:
            low = rest & -rest
            out.append((ds_int ^ low, sc_int))
            rest ^= low
        for s2 in self.facet_removals(sc_int):
            out.append((ds_int, s2))
        return out

    def lower_neighbours(self, ds_int: int, sc_int: int) -> List[int]:
        """Canonical ids of the mDAGs one edge or one maximal face below (ds_int, sc_int)."""
        return [self.canonical(d, s) for d, s in self.lower_neighbour_parts(ds_int, sc_int)]
