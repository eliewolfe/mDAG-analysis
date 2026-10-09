"""
Entropy-vector linear programming for conditional-independence implications.

An entropy vector over n random variables has one coordinate per nonempty subset S (column index: bitmask(S) - 1).
Every such vector satisfies the Shannon (elemental) inequalities, and a conditional independence X ⊥ Y | Z holds
iff the linear functional I(X:Y|Z) vanishes. Since the cone already makes every CMI and conditional entropy
nonnegative, a hypothesis "I(X:Y|Z) = 0" is encoded as the single inequality I(X:Y|Z) <= 0 (likewise H(X|Z) <= 0
for a functional dependence). Hence

    hypotheses imply target CMI = 0

whenever the LP {Shannon cone, hypotheses <= 0, target >= 1} is infeasible (the cone is scale-invariant, so
"target >= 1" is just "target > 0"). All constraints being inequalities, an infeasibility certificate is a
nonnegative combination of them (Farkas), which is the human-checkable proof. The implication direction proven
this way is sound for all distributions (Shannon inequalities are universally valid) and strictly stronger than the
semigraphoid axioms; it does not require full support (no intersection axiom).

Solver: Mosek Optimizer API (raw Task objects). Imported lazily so the rest of the package works without it.
"""
from __future__ import annotations

import itertools
from functools import lru_cache
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

Row = Tuple[np.ndarray, np.ndarray]  # (column indices, coefficients) of a sparse linear functional


# --------------------------------------------------------------------------------------------------
# Linear functionals on entropy vectors
# --------------------------------------------------------------------------------------------------

def mask_of(variables: Iterable[int]) -> int:
    m = 0
    for v in variables:
        m |= 1 << v
    return m


def _accumulate(terms: Dict[int, float], mask: int, coefficient: float) -> None:
    if mask:
        terms[mask] = terms.get(mask, 0.0) + coefficient


def _finalize(terms: Dict[int, float]) -> Row:
    cols = [mask - 1 for mask, c in terms.items() if c != 0.0]
    vals = [c for c in terms.values() if c != 0.0]
    return np.asarray(cols, dtype=np.int64), np.asarray(vals, dtype=float)


def cond_entropy_row(x: Iterable[int], z: Iterable[int]) -> Row:
    """H(X | Z) = H(XZ) - H(Z)."""
    xm, zm = mask_of(x), mask_of(z)
    terms: Dict[int, float] = dict()
    _accumulate(terms, xm | zm, 1.0)
    _accumulate(terms, zm, -1.0)
    return _finalize(terms)


def cmi_row(x: Iterable[int], y: Iterable[int], z: Iterable[int]) -> Row:
    """I(X : Y | Z) = H(XZ) + H(YZ) - H(XYZ) - H(Z)."""
    xm, ym, zm = mask_of(x), mask_of(y), mask_of(z)
    terms: Dict[int, float] = dict()
    _accumulate(terms, xm | zm, 1.0)
    _accumulate(terms, ym | zm, 1.0)
    _accumulate(terms, xm | ym | zm, -1.0)
    _accumulate(terms, zm, -1.0)
    return _finalize(terms)


def sum_rows(rows: Sequence[Row]) -> Row:
    """Coefficient-wise sum of sparse rows (duplicate columns combined)."""
    cols = np.concatenate([np.asarray(c, dtype=np.int64) for c, _ in rows])
    vals = np.concatenate([np.asarray(v, dtype=float) for _, v in rows])
    uniq, inverse = np.unique(cols, return_inverse=True)
    summed = np.zeros(len(uniq))
    np.add.at(summed, inverse, vals)
    keep = summed != 0
    return uniq[keep], summed[keep]


TIMEOUTS = [0]   # number of LP solves that hit the time limit, across all EntropicLP instances
SOLVES = [0]     # number of LP solves, across all EntropicLP instances


def rows_to_csr(rows: Sequence[Row], n_columns: int) -> sp.csr_matrix:
    if not rows:
        return sp.csr_matrix((0, n_columns))
    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum([len(cols) for cols, _ in rows])
    indices = np.concatenate([cols for cols, _ in rows]) if rows else np.zeros(0, dtype=np.int64)
    data = np.concatenate([vals for _, vals in rows]) if rows else np.zeros(0)
    return sp.csr_matrix((data, indices, indptr), shape=(len(rows), n_columns))


@lru_cache(maxsize=None)
def elemental_inequalities(n: int) -> sp.csr_matrix:
    """The elemental Shannon inequalities as a sparse matrix M with M h >= 0:
    H(i | all others) >= 0 for each i, and I(i : j | K) >= 0 for i < j and every K ⊆ [n] \\ {i, j}.
    Row count: n + C(n, 2) * 2^(n-2)."""
    full = (1 << n) - 1
    n_cols = full
    # H(i | all others) = H(all) - H(all \ i)  (for n == 1 this is just H(0) >= 0)
    rows_i, cols_i, vals_i = [], [], []
    for i in range(n):
        rest = full & ~(1 << i)
        rows_i.append(i); cols_i.append(full - 1); vals_i.append(1.0)
        if rest:
            rows_i.append(i); cols_i.append(rest - 1); vals_i.append(-1.0)
    blocks = [sp.coo_matrix((vals_i, (rows_i, cols_i)), shape=(n, n_cols))]
    if n >= 2:
        others_count = n - 2
        K_compact = np.arange(1 << others_count, dtype=np.int64)
        for i, j in itertools.combinations(range(n), 2):
            others = [v for v in range(n) if v not in (i, j)]
            K = np.zeros_like(K_compact)
            for bit, v in enumerate(others):
                K |= ((K_compact >> bit) & 1) << v
            ii, jj = 1 << i, 1 << j
            n_rows = len(K)
            r = np.arange(n_rows)
            cols = np.concatenate([(K | ii) - 1, (K | jj) - 1, (K | ii | jj) - 1, K - 1])
            vals = np.concatenate([np.ones(n_rows), np.ones(n_rows), -np.ones(n_rows), -np.ones(n_rows)])
            rr = np.concatenate([r, r, r, r])
            keep = cols >= 0  # drop the H(∅) term when K is empty
            blocks.append(sp.coo_matrix((vals[keep], (rr[keep], cols[keep])), shape=(n_rows, n_cols)))
    return sp.vstack(blocks).tocsr()


def expected_elemental_count(n: int) -> int:
    return n + (n * (n - 1) // 2) * (1 << max(n - 2, 0)) if n >= 2 else n


# --------------------------------------------------------------------------------------------------
# Hypotheses from DAGs
# --------------------------------------------------------------------------------------------------

def local_markov_rows(parents: Dict[int, FrozenSet[int]], nodes: Iterable[int]) -> List[Tuple[int, Row]]:
    """For each node v: I(v : nondescendants(v) \\ parents(v) | parents(v)) = 0. Returns (node, row) pairs,
    omitting nodes whose constraint is vacuous."""
    import networkx as nx
    g = nx.DiGraph()
    nodes = list(nodes)
    g.add_nodes_from(nodes)
    for child, pa in parents.items():
        g.add_edges_from((p, child) for p in pa)
    rows = []
    for v in nodes:
        pa = set(parents.get(v, frozenset()))
        nondesc = set(nodes) - nx.descendants(g, v) - {v} - pa
        if nondesc:
            rows.append((v, cmi_row([v], nondesc, pa)))
    return rows


def observable_dseparation_rows(parents: Dict[int, FrozenSet[int]], nodes: Iterable[int],
                                observed: Sequence[int]) -> List[Row]:
    """Elementary conditional independences I(x : y | Z) = 0 among observed nodes that hold by d-separation,
    enumerated with the bit-parallel walk of semigraphoid.dsep_all (the networkx version below is the oracle)."""
    import semigraphoid as sg
    nodes = sorted(nodes)
    n = len(nodes)
    assert nodes == list(range(n)), "lp_structure indices are 0..n-1"
    E = sg.dsep_all(n, sg.parents_to_masks(parents, n))
    observed = sorted(observed)
    obs_mask = sg.mask_of(observed)
    rows = []
    for a, x in enumerate(observed):
        for y in observed[a + 1:]:
            Zs = sg.submasks(obs_mask & ~((1 << x) | (1 << y)))
            for Z in Zs[E[x, y, Zs]]:
                rows.append(cmi_row([x], [y], sg.bits_of(int(Z))))
    return rows


def _observable_dseparation_rows_nx(parents: Dict[int, FrozenSet[int]], nodes: Iterable[int],
                                    observed: Sequence[int]) -> List[Row]:
    """The brute-force networkx enumeration, kept as the test oracle for observable_dseparation_rows."""
    import networkx as nx
    g = nx.DiGraph()
    nodes = list(nodes)
    g.add_nodes_from(nodes)
    for child, pa in parents.items():
        g.add_edges_from((p, child) for p in pa)
    rows = []
    observed = list(observed)
    for x, y in itertools.combinations(observed, 2):
        others = [o for o in observed if o not in (x, y)]
        for r in range(len(others) + 1):
            for Z in itertools.combinations(others, r):
                if nx.is_d_separator(g, {x}, {y}, set(Z)):
                    rows.append(cmi_row([x], [y], Z))
    return rows


# --------------------------------------------------------------------------------------------------
# The LP
# --------------------------------------------------------------------------------------------------

_ENV = None


def _shared_env():
    """One Mosek environment per process (licence check-out and thread pool)."""
    global _ENV
    if _ENV is None:
        import mosek
        _ENV = mosek.Env()
    return _ENV


class EntropicLP:
    """Shannon cone over n variables plus a fixed block of hypotheses (functionals forced to be <= 0, hence = 0).
    Additional hypothesis blocks can be pushed and popped; `implies` decides whether a functional is forced to zero.

    Implementation: one Mosek Task. Variables are the 2^n - 1 entropies (nonnegative). Constraints: elemental
    inequalities (>= 0), hypothesis rows (<= 0), and, during a query, the target row (>= 1). Infeasible means the
    target is implied to vanish."""

    def __init__(self, n: int, hypotheses: Sequence[Row] = (), optimizer: str = 'intpnt', presolve: bool = True,
                 max_time: float = 60.0) -> None:
        import mosek
        self._mosek = mosek
        self.n = n
        self.n_columns = (1 << n) - 1
        self.env = _shared_env()
        self.task = self.env.Task()
        self.task.putintparam(mosek.iparam.log, 0)
        # Wall-clock limit per solve; a solve that hits it counts as undecided (the implication is not claimed).
        self.task.putdouparam(mosek.dparam.optimizer_max_time, float(max_time))
        self.task.putintparam(mosek.iparam.optimizer, getattr(mosek.optimizertype, optimizer))
        if optimizer == 'intpnt':
            # Feasibility questions only: no basis identification needed. The interior-point method decides
            # infeasibility orders of magnitude faster than the simplex methods on these degenerate cones.
            self.task.putintparam(mosek.iparam.intpnt_basis, mosek.basindtype.never)
        if not presolve:
            self.task.putintparam(mosek.iparam.presolve_use, mosek.presolvemode.off)
        self.task.appendvars(self.n_columns)
        self.task.putvarboundsliceconst(0, self.n_columns, mosek.boundkey.lo, 0.0, 0.0)
        self.task.putobjsense(mosek.objsense.minimize)   # feasibility problems only: zero objective
        M = elemental_inequalities(n)
        self._append_rows(M, mosek.boundkey.lo, 0.0, 0.0)
        self.n_fixed_rows = M.shape[0]
        self.n_rows = self.n_fixed_rows
        self.lp_count = 0
        self.undecided = 0
        self.timeouts = 0
        self.push_hypotheses(hypotheses)
        self.n_fixed_rows = self.n_rows

    # -- rows ---------------------------------------------------------------------------------
    def _append_rows(self, M: sp.csr_matrix, boundkey, lower: float, upper: float) -> int:
        m = M.shape[0]
        if m == 0:
            return self.n_rows
        first = self.task.getnumcon()
        self.task.appendcons(m)
        self.task.putarowslice(first, first + m, np.asarray(M.indptr[:-1], dtype=np.int64), np.asarray(M.indptr[1:], dtype=np.int64), np.asarray(M.indices, dtype=np.int32), np.asarray(M.data, dtype=float))
        self.task.putconboundsliceconst(first, first + m, boundkey, lower, upper)
        self.n_rows = first + m
        return first

    def push_hypotheses(self, rows: Sequence[Row]) -> int:
        """Appends hypothesis rows bounded above by zero (CMIs and conditional entropies are nonnegative on the
        cone, so this forces them to vanish); returns the number of rows before the push (a handle)."""
        handle = self.n_rows
        self._append_rows(rows_to_csr(list(rows), self.n_columns), self._mosek.boundkey.up, 0.0, 0.0)
        return handle

    def pop_to(self, handle: int) -> None:
        """Removes every row appended after `handle`."""
        assert handle >= self.n_fixed_rows, "cannot pop the fixed hypothesis block"
        if self.n_rows > handle:
            self.task.removecons(list(range(handle, self.n_rows)))
            self.n_rows = handle

    # -- queries ------------------------------------------------------------------------------
    def _solve_with_target(self, row: Row, lower: float, want_dual: bool = False):
        """Solves {cone, hypotheses, row >= lower} (zero objective) with the target row appended temporarily.
        Returns (problem status, dual vector or None)."""
        mosek = self._mosek
        handle = self.n_rows
        try:
            self._append_rows(rows_to_csr([row], self.n_columns), mosek.boundkey.lo, lower, 0.0)
            rescode = self.task.optimize()
            self.lp_count += 1
            SOLVES[0] += 1
            if rescode == mosek.rescode.trm_max_time:
                self.timeouts += 1
                TIMEOUTS[0] += 1
            soltype = mosek.soltype.bas if self.task.solutiondef(mosek.soltype.bas) else mosek.soltype.itr
            prosta = self.task.getprosta(soltype)
            y = np.array(self.task.gety(soltype)) if want_dual else None
        finally:
            self.pop_to(handle)
        return prosta, y

    def is_feasible_with(self, row: Row, lower: float = 1.0) -> bool:
        """Is {cone, hypotheses, row >= lower} feasible? An undecided solver status counts as feasible, which is
        the conservative answer (the implication is then NOT claimed)."""
        mosek = self._mosek
        prosta, _ = self._solve_with_target(row, lower)
        if prosta in (mosek.prosta.prim_infeas, mosek.prosta.prim_infeas_or_unbounded):
            return False
        if prosta not in (mosek.prosta.prim_feas, mosek.prosta.prim_and_dual_feas, mosek.prosta.dual_infeas):
            self.undecided += 1
        return True

    def farkas_certificate(self, row: Row):
        """For an implied target, the dual infeasibility certificate: nonnegative multipliers (y_elemental,
        y_hypotheses, y_target) with  sum_i y_i * (elemental_i) + sum_j y'_j * (-hypothesis_j) >= y_target * target
        coefficient-wise, exhibiting the target as a nonnegative combination of elemental inequalities and
        hypotheses. Returns None when the target is not implied. Numerical; meant for inspection and exact
        re-verification."""
        mosek = self._mosek
        prosta, y = self._solve_with_target(row, 1.0, want_dual=True)
        if prosta not in (mosek.prosta.prim_infeas, mosek.prosta.prim_infeas_or_unbounded):
            return None
        # Mosek's certificate: y_i >= 0 on lower-bounded rows, <= 0 on upper-bounded rows. Report magnitudes.
        n_el = elemental_inequalities(self.n).shape[0]
        return {'elemental': y[:n_el], 'hypotheses': -y[n_el:-1], 'target': y[-1]}

    def implies(self, row: Row) -> bool:
        """True iff the hypotheses force the functional to zero (for functionals that are nonnegative on the
        cone, i.e. CMIs and conditional entropies)."""
        return not self.is_feasible_with(row, 1.0)

    def implies_all(self, rows: Iterable[Row]) -> bool:
        """True iff every row is forced to zero. For functionals that are nonnegative on the Shannon cone (CMIs,
        conditional entropies, and any sum of them) this is one LP, not one per row: the rows all vanish iff their
        sum vanishes, since a sum of nonnegative quantities is zero iff each term is. The Farkas certificate of the
        summed row therefore certifies every row at once."""
        rows = list(rows)
        if not rows:
            return True
        return self.implies(sum_rows(rows))

    def close(self) -> None:
        self.task.__exit__(None, None, None)

    def __enter__(self) -> "EntropicLP":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
