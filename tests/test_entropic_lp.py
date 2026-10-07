"""Tests for the entropy-vector LP (Shannon cone + conditional-independence hypotheses)."""
import numpy as np
import pytest

pytest.importorskip("mosek")

import entropic_lp as E


def test_elemental_inequalities_have_the_right_shape():
    for n in range(1, 9):
        M = E.elemental_inequalities(n)
        assert M.shape == (E.expected_elemental_count(n), (1 << n) - 1)
        assert np.all((M != 0).sum(axis=1) <= 4)


def test_elemental_inequalities_match_the_mathematica_notebook_for_three_variables():
    # ShanMb[3] from piggyback_check.nb, columns ordered {1},{2},{3},{1,2},{1,3},{2,3},{1,2,3}.
    notebook = np.array([[0, 0, 0, -1, 0, 0, 1], [0, 0, 0, 0, -1, 0, 1], [0, 0, 0, 0, 0, -1, 1],
                         [1, 1, 0, -1, 0, 0, 0], [0, 0, -1, 0, 1, 1, -1], [1, 0, 1, 0, -1, 0, 0],
                         [0, -1, 0, 1, 0, 1, -1], [0, 1, 1, 0, 0, -1, 0], [-1, 0, 0, 1, 1, 0, -1]])
    to_bitmask_order = [0, 1, 3, 2, 4, 5, 6]  # notebook column k is our column to_bitmask_order[k]
    permuted = np.zeros_like(notebook)
    permuted[:, to_bitmask_order] = notebook
    ours = E.elemental_inequalities(3).toarray()
    assert set(map(tuple, permuted)) == set(map(tuple, ours))


def test_functionals():
    cols, vals = E.cmi_row([0], [1], [2])
    # I(0:1|2) = H(02) + H(12) - H(012) - H(2): masks 5, 6, 7, 4 -> columns 4, 5, 6, 3
    assert dict(zip(cols.tolist(), vals.tolist())) == {4: 1.0, 5: 1.0, 6: -1.0, 3: -1.0}
    cols, vals = E.cond_entropy_row([0], [1])
    assert dict(zip(cols.tolist(), vals.tolist())) == {2: 1.0, 1: -1.0}
    cols, vals = E.cmi_row([0], [1], [])
    assert dict(zip(cols.tolist(), vals.tolist())) == {0: 1.0, 1: 1.0, 2: -1.0}


def test_shannon_cone_implies_the_semigraphoid_axioms_but_not_intersection():
    x, y, z, w = 0, 1, 2, 3
    with E.EntropicLP(4) as lp:
        h = lp.push_hypotheses([E.cmi_row([x], [y], [z]), E.cmi_row([x], [w], [y, z])])
        assert lp.implies(E.cmi_row([x], [y, w], [z]))            # contraction
        lp.pop_to(h)
        h = lp.push_hypotheses([E.cmi_row([x], [y, w], [z])])
        assert lp.implies(E.cmi_row([x], [y], [z, w]))            # weak union
        assert lp.implies(E.cmi_row([x], [y], [z]))               # decomposition
        lp.pop_to(h)
        h = lp.push_hypotheses([E.cmi_row([x], [y], [z])])
        assert not lp.implies(E.cmi_row([x], [y], []))            # conditioning cannot be dropped
        lp.pop_to(h)
        h = lp.push_hypotheses([E.cmi_row([x], [y], [z]), E.cmi_row([x], [y], [w])])
        assert not lp.implies(E.cmi_row([x], [y], []))            # intersection needs full support
        lp.pop_to(h)
        assert lp.n_rows == lp.n_fixed_rows


def test_local_markov_implies_exactly_the_d_separations():
    import networkx as nx
    rng = np.random.default_rng(3)
    n = 6
    with E.EntropicLP(n) as lp:
        for _ in range(4):
            parents = {v: frozenset(u for u in range(v) if rng.random() < 0.5) for v in range(n)}
            g = nx.DiGraph()
            g.add_nodes_from(range(n))
            g.add_edges_from((p, c) for c, ps in parents.items() for p in ps)
            h = lp.push_hypotheses([row for _, row in E.local_markov_rows(parents, range(n))])
            for _ in range(8):
                x, y, *rest = rng.permutation(n).tolist()
                Z = rest[:rng.integers(0, 3)]
                assert lp.implies(E.cmi_row([x], [y], Z)) == nx.is_d_separator(g, {x}, {y}, set(Z))
            lp.pop_to(h)


def test_perfect_prediction_transfers_independences():
    # If s is a function of x and x ⊥ y, then s ⊥ y.
    s, x, y = 0, 1, 2
    with E.EntropicLP(3) as lp:
        lp.push_hypotheses([E.cond_entropy_row([s], [x]), E.cmi_row([x], [y], [])])
        assert lp.implies(E.cmi_row([s], [y], []))


def test_implies_all_on_the_summed_row_agrees_with_the_per_row_loop():
    # Local Markov rows of a random DAG are CMIs, nonnegative on the cone; under the Markov hypotheses of a second
    # DAG they are all implied iff every d-separation of the second holds in the first. The summed row decides
    # this in one LP and must agree with the row-by-row loop, in both directions.
    import networkx as nx
    rng = np.random.default_rng(11)
    n = 6
    seen = set()
    with E.EntropicLP(n) as lp:
        for _ in range(12):
            hyp = {v: frozenset(u for u in range(v) if rng.random() < 0.5) for v in range(n)}
            if rng.random() < 0.5:    # a target DAG with more edges asserts fewer independences: implied
                tgt = {v: hyp[v] | frozenset(u for u in range(v) if rng.random() < 0.3) for v in range(n)}
            else:
                tgt = {v: frozenset(u for u in range(v) if rng.random() < 0.5) for v in range(n)}
            rows = [row for _, row in E.local_markov_rows(tgt, range(n))]
            h = lp.push_hypotheses([row for _, row in E.local_markov_rows(hyp, range(n))])
            per_row = all(lp.implies(row) for row in rows)
            assert lp.implies_all(rows) == per_row
            seen.add(per_row)
            lp.pop_to(h)
        # Degenerate cases: no rows, and a target equal to the hypotheses.
        assert lp.implies_all([])
        h = lp.push_hypotheses([row for _, row in E.local_markov_rows(hyp, range(n))])
        assert lp.implies_all(row for _, row in E.local_markov_rows(hyp, range(n)))
        lp.pop_to(h)
    assert seen == {True, False}, "the random DAGs should produce both outcomes"


def test_sum_rows_combines_duplicate_columns():
    cols, vals = E.sum_rows([E.cmi_row([0], [1], [2]), E.cmi_row([0], [1], [2]), E.cond_entropy_row([0], [1])])
    # 2*I(0:1|2) + H(0|1): columns 4,5 -> 2; 6,3 -> -2; H(01)=col 2 -> +1; H(1)=col 1 -> -1
    assert dict(zip(cols.tolist(), vals.tolist())) == {4: 2.0, 5: 2.0, 6: -2.0, 3: -2.0, 2: 1.0, 1: -1.0}


def test_time_limit_is_passed_to_mosek_and_counted():
    import mosek
    with E.EntropicLP(4, max_time=7.5) as lp:
        assert lp.task.getdouparam(mosek.dparam.optimizer_max_time) == 7.5
        assert lp.timeouts == 0
