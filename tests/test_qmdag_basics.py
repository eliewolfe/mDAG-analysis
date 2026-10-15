"""Fast regression tests for QmDAG basics and the piggyback tricks (stage-1 maintenance pins)."""
from hypergraphs import Hypergraph, LabelledHypergraph
from directed_structures import DirectedStructure
from quantum_mDAG import QmDAG


def triangle():
    return QmDAG(DirectedStructure([], 3), Hypergraph([], 3), Hypergraph([(0, 1), (1, 2), (0, 2)], 3))


def bell6():
    return QmDAG(DirectedStructure([], 4), Hypergraph([(1, 3), (0, 2)], 4), Hypergraph([(2, 3)], 4))


def evans():
    return QmDAG(DirectedStructure([(0, 1), (0, 2)], 3), Hypergraph([], 3), Hypergraph([(0, 1), (0, 2)], 3))


def test_labelled_hypergraph_string_drops_singletons():
    assert LabelledHypergraph((0, 2), [frozenset({2, 3})]).as_string == "[]"
    bell = QmDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([], 4), Hypergraph([(2, 3)], 4))
    assert "(2)" not in bell.subgraph((0, 2)).as_string


def test_marginalize_returns_none_when_districts_break():
    # Marginalizing the middle node of a chain merges two districts into one.
    chain = QmDAG(DirectedStructure([(0, 1), (1, 2), (3, 2)], 4), Hypergraph([(0, 1), (2, 3)], 4), Hypergraph([], 4))
    assert chain.marginalize(1, districts_check=True) is None
    assert chain.marginalize(1, districts_check=False) is not None
    for sub in chain.submarginals(districts_check=True):
        assert sub is not None
        assert sub.number_of_visible >= 3


def test_exogenous_visible_nodes_and_interruption():
    ghost = QmDAG(DirectedStructure([(1, 2), (1, 3)], 4), Hypergraph([], 4), Hypergraph([(0, 2), (0, 3)], 4))
    assert ghost.exogenous_visible_nodes == {1}
    assert evans().unique_unlabelled_id in ghost.unique_unlabelled_ids_obtainable_by_interruption


def test_unlabelled_id_is_memoized_and_label_invariant():
    a = QmDAG(DirectedStructure([(0, 1)], 3), Hypergraph([(1, 2)], 3), Hypergraph([(0, 2)], 3))
    b = QmDAG(DirectedStructure([(2, 1)], 3), Hypergraph([(1, 0)], 3), Hypergraph([(2, 0)], 3))
    assert a.unique_unlabelled_id == b.unique_unlabelled_id
    assert a.unique_id != b.unique_id


def test_triangle_reaches_bell6_via_fritz():
    ids = triangle().unique_unlabelled_ids_obtainable_by_Fritz_for_QC()
    assert bell6().unique_unlabelled_id in ids


def test_unlabelled_id_fast_path_equals_the_permutation_minimum():
    """QmDAG.unique_unlabelled_id uses the permutation-table kernels of canonical_forms; it must equal the minimum
    over the three as_integer_permutations tuples (the definition) on random labelled structures, hybrids included."""
    import random
    from hypergraphs import hypergraph_full_cleanup
    from quantum_mDAG import _UNLABELLED_ID_MEMO
    rng = random.Random(2026)
    for n in (3, 4, 5):
        for _ in range(60):
            order = list(range(n))
            rng.shuffle(order)
            edges = [(order[i], order[j]) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.4]
            def facets():
                fs = {frozenset(rng.sample(range(n), rng.randint(2, n))) for _ in range(rng.randint(0, 3))}
                return hypergraph_full_cleanup(fs)
            g = QmDAG(DirectedStructure(edges, n), Hypergraph(facets(), n), Hypergraph(facets(), n))
            _UNLABELLED_ID_MEMO.pop(g.unique_id, None)
            expected = (n,) + min(zip(g.directed_structure_instance.as_integer_permutations,
                                      g.C_simplicial_complex_instance.as_integer_permutations,
                                      g.Q_simplicial_complex_instance.as_integer_permutations))
            assert g.unique_unlabelled_id == expected
