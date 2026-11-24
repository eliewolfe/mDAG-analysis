"""Fast regression tests for QmDAG basics and the piggyback tricks (stage-1 maintenance pins)."""
import pytest

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


def test_stitching_preserves_outcome_children_and_sources():
    graph = QmDAG(DirectedStructure([(1, 2), (1, 3), (0, 2), (0, 4), (3, 5)], 6),
                  Hypergraph([(0, 4)], 6), Hypergraph([(0, 5)], 6))
    # Removing setting 1 leaves vertices [0, 2, 3, 4, 5] in that order.
    expected = QmDAG(DirectedStructure([(0, 1), (0, 2), (0, 3), (2, 4)], 5),
                     Hypergraph([(0, 3)], 5), Hypergraph([(0, 4)], 5))
    steps = {(dict(params[0])['outcome'], dict(params[0])['source']): child.unique_id
             for params, child in graph.node_stitching_steps()}
    assert steps.keys() == {(0, 1), (4, 1)}
    assert steps[0, 1] == expected.unique_id
    assert {child.unique_id for child in graph.subinterruptions} == set(steps.values())
    for descendant in (1, 2, 3, 5):
        with pytest.raises(ValueError):
            graph.node_stitching(descendant, 1)
    for endogenous in (0, 2, 4, 5):
        with pytest.raises(ValueError):
            graph.node_stitching(1, endogenous)


def test_unlabelled_id_is_memoized_and_label_invariant():
    a = QmDAG(DirectedStructure([(0, 1)], 3), Hypergraph([(1, 2)], 3), Hypergraph([(0, 2)], 3))
    b = QmDAG(DirectedStructure([(2, 1)], 3), Hypergraph([(1, 0)], 3), Hypergraph([(2, 0)], 3))
    assert a.unique_unlabelled_id == b.unique_unlabelled_id
    assert a.unique_id != b.unique_id


def test_triangle_reaches_bell6_via_fritz():
    ids = triangle().unique_unlabelled_ids_obtainable_by_Fritz_for_QC()
    assert bell6().unique_unlabelled_id in ids
