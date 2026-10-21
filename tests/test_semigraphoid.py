"""The semigraphoid closure engine: layout, d-separation enumeration, the exchange rule, functional dependence, and
agreement with networkx and with the entropic LP."""
import importlib.util
import itertools
import math
import random

import networkx as nx

import numpy as np
import pytest

import semigraphoid as sg


def random_dag(n, p=0.4, rng=random):
    pa = np.zeros(n, dtype=np.int64)
    for v in range(n):
        for u in range(v):
            if rng.random() < p:
                pa[v] |= 1 << u
    return pa


def nx_dsep_model(n, pa):
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    for v in range(n):
        for u in sg.bits_of(int(pa[v])):
            g.add_edge(u, v)
    E = sg.empty_model(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for K in range(1 << n):
                if K & ((1 << i) | (1 << j)):
                    continue
                E[i, j, K] = nx.is_d_separator(g, {i}, {j}, set(sg.bits_of(K)))
    return E


@pytest.fixture(scope="module", autouse=True)
def compiled():
    sg.warm_up()


def test_dsep_all_matches_networkx_and_the_numpy_kernel():
    rng = random.Random(7)
    for n in (3, 4, 5, 6, 7):
        for _ in range(3):
            pa = random_dag(n, rng=rng)
            E = sg.dsep_all(n, pa)
            assert (E == nx_dsep_model(n, pa)).all()
            assert (E == sg.dsep_all(n, pa, kernel=sg._esep_numpy)).all()
            assert (E == E.transpose(1, 0, 2)).all()
            assert not (E & ~sg.valid_cells(n)).any()
            # Restricted to the first m nodes (sources, targets and conditioning sets): the leading block.
            m = max(2, n - 2)
            R = sg.dsep_all(n, pa, m=m)
            assert R.shape == (m, m, 1 << m) and (R == E[:m, :m, :1 << m]).all()
            assert (R == sg.dsep_all(n, pa, m=m, kernel=sg._esep_numpy)).all()


def nx_esep_model(n, m, pa, max_deleted):
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    for v in range(n):
        for u in sg.bits_of(int(pa[v])):
            g.add_edge(u, v)
    R = np.zeros((1 << m, m, m, 1 << m), dtype=bool)
    for D in range(1 << m):
        deleted = sg.bits_of(D)
        if len(deleted) > max_deleted:
            continue
        h = g.copy()
        h.remove_nodes_from(deleted)
        rest = [v for v in range(m) if v not in deleted]
        for i in rest:
            for j in rest:
                if i == j:
                    continue
                others = [v for v in rest if v not in (i, j)]
                for r in range(len(others) + 1):
                    for K in itertools.combinations(others, r):
                        R[D, i, j, sg.mask_of(K)] = nx.is_d_separator(h, {i}, {j}, set(K))
    return R


def random_dag_with_latents(m, n, rng):
    """Observed nodes 0..m-1 (in topological order), latent roots m..n-1 with random observed children."""
    pa = random_dag(m, rng=rng)
    pa = np.concatenate([pa, np.zeros(n - m, dtype=np.int64)])
    for latent in range(m, n):
        for child in rng.sample(range(m), rng.randint(1, m)):
            pa[child] |= 1 << latent
    return pa


def test_esep_all_matches_networkx_on_the_dag_with_the_deleted_nodes_removed():
    rng = random.Random(5)
    for m in (2, 3, 4, 5):
        for extra in (0, 2, 3):
            n = m + extra
            pa = random_dag_with_latents(m, n, rng)
            R = sg.esep_all(n, pa, m=m)
            assert R.shape == (1 << m, m, m, 1 << m)
            assert sg.canonical_form(R, m).shape == R.shape
            assert (R == nx_esep_model(n, m, pa, m - 2)).all()
            assert (R == sg.esep_all(n, pa, m=m, kernel=sg._esep_numpy)).all()
            assert (R[0] == sg.dsep_all(n, pa, m=m)).all()
            assert not (R & ~sg.esep_valid_cells(m) & ~sg.esep_valid_cells(m).transpose(0, 2, 1, 3)).any()
            # Only the requested deletion sets are filled.
            R1 = sg.esep_all(n, pa, m=m, max_deleted=1)
            assert (R1[0] == R[0]).all() and not R1[[D for D in range(1 << m) if bin(D).count("1") > 1]].any()


def test_canonical_form_is_relabelling_invariant_and_an_actual_relabelling():
    rng = random.Random(3)
    m = 4
    for _ in range(5):
        pa = random_dag_with_latents(m, m + 2, rng)
        R = sg.esep_all(m + 2, pa, m=m)
        canonical = sg.canonical_form(R, m)
        perms = [p for p, _ in sg._permutation_index_maps(m)]
        relabelled = [R[np.ix_(pm, p, p, pm)] for p, pm in sg._permutation_index_maps(m)]
        assert any((canonical == Rp).all() for Rp in relabelled)
        for Rp in relabelled:
            assert (sg.canonical_form(Rp, m) == canonical).all()
        # the same for the d-separation model and for a symmetric adjacency matrix
        E = R[0]
        assert all((sg.canonical_form(E[np.ix_(p, p, pm)], m) == sg.canonical_form(E, m)).all()
                   for p, pm in sg._permutation_index_maps(m))
        adj = np.zeros((m, m), dtype=bool)
        adj[0, 1] = adj[1, 0] = adj[1, 3] = adj[3, 1] = True
        for p in perms:
            assert (sg.canonical_form(adj[np.ix_(p, p)], m) == sg.canonical_form(adj, m)).all()


def test_is_d_separated_matches_networkx_and_dsep_all_allocates_one_slice():
    rng = random.Random(9)
    for _ in range(6):
        m = rng.randint(3, 5)
        n = m + rng.randint(0, 3)
        pa = random_dag_with_latents(m, n, rng)
        g = nx.DiGraph()
        g.add_nodes_from(range(n))
        for v in range(n):
            for u in sg.bits_of(int(pa[v])):
                g.add_edge(u, v)
        for _ in range(40):
            nodes = list(range(n))
            rng.shuffle(nodes)
            A = nodes[:rng.randint(1, 2)]
            B = nodes[len(A):len(A) + rng.randint(1, 2)]
            K = [v for v in nodes[len(A) + len(B):] if rng.random() < 0.4]
            assert sg.is_d_separated(pa, sg.mask_of(A), sg.mask_of(B), sg.mask_of(K)) == \
                nx.is_d_separator(g, set(A), set(B), set(K))
        assert sg.is_d_separated(pa, 0, 1, 0) and sg.is_d_separated(pa, 1, 0, 0)
        E = sg.dsep_all(n, pa)
        assert E.base is None or E.base.shape[0] == 1   # not a view into a (2^n, n, n, 2^n) array


def test_descendant_masks():
    pa = sg.parents_to_masks({1: [0], 2: [1], 3: [1, 4], 4: []}, 5)
    desc = sg.descendant_masks(pa)
    assert [sg.bits_of(int(d)) for d in desc] == [[1, 2, 3], [2, 3], [], [], [3]]


def test_closure_of_the_local_markov_triplets_is_the_d_separation_model():
    # Geiger, Verma and Pearl: the semigraphoid closure of the recursive basis is d-separation. Also the two kernels
    # agree cell by cell, and the closure is idempotent.
    rng = random.Random(11)
    for n in (4, 5, 6, 7):
        for _ in range(3):
            pa = random_dag(n, rng=rng)
            L = sg.local_markov_model(n, pa)
            L2 = L.copy()
            sweeps = sg.close(L)
            sg.close(L2, kernel=sg._close_numpy)
            assert (L == L2).all() and (L == sg.dsep_all(n, pa)).all()
            assert sweeps >= 1 and sg.close(L) == 1


def test_semigraphoid_axioms_hold_and_intersection_does_not():
    # <0,1|3> and <0,2|13> give contraction <0,12|3>, weak union <0,1|23> and decomposition <0,2|3>.
    E = sg.empty_model(4)
    sg.add_triplet(E, 0b0001, 0b0010, 0b1000)
    sg.add_triplet(E, 0b0001, 0b0100, 0b1010)
    sg.close(E)
    assert sg.holds(E, 0b0001, 0b0110, 0b1000)
    assert sg.holds(E, 0b0001, 0b0010, 0b1100)
    assert sg.holds(E, 0b0001, 0b0100, 0b1000)
    assert not sg.holds(E, 0b0001, 0b0010, 0)        # conditioning cannot be dropped
    # Intersection: <0,1|2> and <0,2|1> do not give <0,12|>.
    E = sg.empty_model(4)
    sg.add_triplet(E, 0b0001, 0b0010, 0b0100)
    sg.add_triplet(E, 0b0001, 0b0100, 0b0010)
    sg.close(E)
    assert not sg.holds(E, 0b0001, 0b0110, 0)


def test_general_triplets_are_their_elementary_components():
    E = sg.empty_model(5)
    sg.add_triplet(E, 0b00011, 0b01100, 0b10000)
    for i in (0, 1):
        for j in (2, 3):
            for extra in sg.submasks(0b01111 & ~((1 << i) | (1 << j))):
                assert E[i, j, 0b10000 | extra] and E[j, i, 0b10000 | extra]
    assert sg.holds(E, 0b00011, 0b01100, 0b10000) and sg.holds(E, 0b00001, 0b00100, 0b11010)
    assert not sg.holds(E, 0b00001, 0b00100, 0)


def test_functional_dependence_transfers_independences():
    # s = f(x) and x _||_ y give s _||_ y; s _||_ y | x holds outright; nothing says x _||_ s.
    x, y, s = 0, 1, 2
    E = sg.empty_model(3)
    sg.add_triplet(E, 1 << x, 1 << y, 0)
    sg.add_functional_dependence(E, s, 1 << x)
    sg.close(E)
    assert sg.holds(E, 1 << s, 1 << y, 0) and sg.holds(E, 1 << s, 1 << y, 1 << x)
    assert sg.holds(E, 1 << x, 1 << y, 1 << s)      # x _||_ y | f(x)
    assert not sg.holds(E, 1 << x, 1 << s, 0)


def test_restrict_and_contains():
    n = 4
    pa = np.array([0, 1, 3, 0], dtype=np.int64)        # 0 -> 1 -> 2 <- 0; 3 isolated
    E = sg.dsep_all(n, pa)
    R = sg.restrict(E, 0b0111)
    assert sg.contains(E, R) and not (R & ~sg.valid_cells(n, excluded=0b1000)).any()
    assert R[0, 3].sum() == 0 and R[0, 2, 0b0010] == E[0, 2, 0b0010]


@pytest.mark.skipif(importlib.util.find_spec("mosek") is None, reason="mosek not installed")
def test_closure_implies_only_what_the_lp_implies():
    # Soundness: every elementary triplet the closure derives from a DAG's Markov statements, a functional
    # dependence and a few extra independences is implied by the Shannon LP under the same hypotheses.
    from entropic_lp import EntropicLP, cmi_row, cond_entropy_row, local_markov_rows
    rng = random.Random(3)
    for _ in range(3):
        n = 5
        pa = random_dag(n, 0.3, rng=rng)
        parents = {v: frozenset(sg.bits_of(int(pa[v]))) for v in range(n)}
        lp = EntropicLP(n, [row for _, row in local_markov_rows(parents, range(n))])
        E = sg.dsep_all(n, pa)
        s, x = 0, 1
        extra = [(2, 3, 1 << 4)]
        hyps = [cond_entropy_row([s], [x])] + [cmi_row([a], [b], sg.bits_of(K)) for a, b, K in extra]
        sg.add_functional_dependence(E, s, 1 << x)
        for a, b, K in extra:
            sg.add_triplet(E, 1 << a, 1 << b, K)
        sg.close(E)
        handle = lp.push_hypotheses(hyps)
        try:
            for i, j in itertools.combinations(range(n), 2):
                for K in sg.submasks(((1 << n) - 1) & ~((1 << i) | (1 << j))):
                    if E[i, j, K]:
                        assert lp.implies(cmi_row([i], [j], sg.bits_of(int(K)))), (i, j, K)
        finally:
            lp.pop_to(handle)
            lp.close()


def test_observable_rows_match_the_networkx_oracle():
    from entropic_lp import observable_dseparation_rows, _observable_dseparation_rows_nx
    rng = random.Random(5)
    for _ in range(4):
        n = 7
        pa = random_dag(n, 0.35, rng=rng)
        parents = {v: frozenset(sg.bits_of(int(pa[v]))) for v in range(n)}
        new = {(tuple(r[0]), tuple(r[1])) for r in observable_dseparation_rows(parents, range(n), range(4))}
        old = {(tuple(r[0]), tuple(r[1])) for r in _observable_dseparation_rows_nx(parents, range(n), range(4))}
        assert new == old


def test_permutation_index_maps_are_cached_read_only_for_small_m_and_generated_beyond():
    for m in (3, 4):
        cached = sg._permutation_index_maps(m)
        assert isinstance(cached, tuple) and cached is sg._permutation_index_maps(m)
        generated = list(sg._generate_permutation_index_maps(m))
        assert len(cached) == len(generated) == math.factorial(m)
        for (p, pm), (q, qm) in zip(cached, generated):
            assert np.array_equal(p, q) and np.array_equal(pm, qm)
            assert not p.flags.writeable and not pm.flags.writeable
    beyond = sg._permutation_index_maps(sg._PERMUTATION_MAPS_CACHED_UP_TO + 1)
    assert not isinstance(beyond, tuple)
    p, pm = next(iter(beyond))
    assert p.shape == (sg._PERMUTATION_MAPS_CACHED_UP_TO + 1,)
