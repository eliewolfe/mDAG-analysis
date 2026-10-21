"""The graph objects at their API boundaries (the October 2026 repository review): named components in any order,
equality and hashing of mDAGs and directed structures, iterator inputs, node-index validation, tuple arguments to the
closure and adjacency helpers, and the matrix-returning support wrapper."""
import itertools

import numpy as np
import pytest

from adjmat_class import AdjMat
from directed_structures import DirectedStructure, LabelledDirectedStructure
from hypergraphs import Hypergraph, LabelledHypergraph
from mDAG_advanced import mDAG, Unlabelled_mDAG
from quantum_mDAG import QmDAG

NAMES = ('A', 'B', 'C', 'D')
EDGES = [('A', 'B'), ('B', 'C')]
FACETS = [('A', 'C'), ('C', 'D')]


# -- finding 1: the order of the node names ------------------------------------------------------------------------

def test_named_components_in_any_order_give_the_same_numeric_graph():
    ds = LabelledDirectedStructure(NAMES, EDGES)
    reference = mDAG(ds, LabelledHypergraph(NAMES, FACETS))
    for perm in itertools.permutations(NAMES):
        md = mDAG(ds, LabelledHypergraph(perm, FACETS))
        assert md.simplicial_complex_instance.as_tuples == reference.simplicial_complex_instance.as_tuples
        assert md.unique_id == reference.unique_id and md.all_CI == reference.all_CI
        assert md.variable_names == NAMES
        # the named facets are preserved, whatever the order they were given in
        assert set(md.simplicial_complex_instance.simplicial_complex_with_variable_names) == {frozenset(f) for f in FACETS}
    # the quantum structure translates its classical and its quantum complex alike
    q_reference = QmDAG(ds, LabelledHypergraph(NAMES, [('A', 'B')]), LabelledHypergraph(NAMES, FACETS))
    for perm in itertools.permutations(NAMES):
        q = QmDAG(ds, LabelledHypergraph(perm, [('A', 'B')]), LabelledHypergraph(perm[::-1], FACETS))
        assert q.unique_id == q_reference.unique_id
        assert q.C_simplicial_complex_instance.as_tuples == q_reference.C_simplicial_complex_instance.as_tuples
        assert q.Q_simplicial_complex_instance.as_tuples == q_reference.Q_simplicial_complex_instance.as_tuples
    with pytest.raises(ValueError):
        mDAG(ds, LabelledHypergraph(('A', 'B', 'C', 'E'), FACETS))     # different names, not just another order


def test_mixed_named_and_numeric_components_have_a_skeleton():
    named = mDAG(LabelledDirectedStructure(('A', 'B', 'C'), [('A', 'B')]), Hypergraph([(1, 2)], 3))
    numeric = mDAG(DirectedStructure([(0, 1)], 3), Hypergraph([(1, 2)], 3))
    assert named.skeleton == numeric.skeleton and named.skeleton_unlabelled == numeric.skeleton_unlabelled
    labelled = mDAG(LabelledDirectedStructure(('A', 'B', 'C'), [('A', 'B')]), LabelledHypergraph(('A', 'B', 'C'), [('B', 'C')]))
    assert labelled.skeleton == numeric.skeleton and labelled.skeleton_unlabelled == numeric.skeleton_unlabelled
    assert set(labelled.skeleton_instance.edges_with_variable_names) == {('A', 'B'), ('B', 'C')}


@pytest.mark.parametrize('named_c,named_q', list(itertools.product((False, True), repeat=2)))
def test_quantum_dominance_uses_aligned_numeric_components(named_c, named_q):
    names = ('A', 'B', 'C')
    ds = LabelledDirectedStructure(names, [])
    reference = QmDAG(ds, Hypergraph([(0, 1), (1, 2)], 3), Hypergraph([(0, 1)], 3))
    for c_order in itertools.permutations(names):
        for q_order in itertools.permutations(names):
            c = (LabelledHypergraph(c_order, [('A', 'B'), ('B', 'C')]) if named_c
                 else Hypergraph([(0, 1), (1, 2)], 3))
            q = LabelledHypergraph(q_order, [('A', 'B')]) if named_q else Hypergraph([(0, 1)], 3)
            actual = QmDAG(ds, c, q)
            assert actual.unique_id == reference.unique_id
            assert actual.C_simplicial_complex_instance.as_tuples == ((1, 2),)
            if named_c:
                assert actual.C_simplicial_complex_instance.simplicial_complex_with_variable_names == {frozenset(('B', 'C'))}


def test_named_mdag_converts_to_a_classical_quantum_structure():
    from quantum_mDAG import as_classical_QmDAG
    md = mDAG(LabelledDirectedStructure(('A', 'B', 'C'), [('A', 'B')]),
              LabelledHypergraph(('A', 'B', 'C'), [('B', 'C')]))
    assert as_classical_QmDAG(md).as_mDAG == md


# -- finding 2: equality and hashing ------------------------------------------------------------------------------

def test_unlabelled_mdag_equality_is_an_equivalence_on_canonical_ids():
    g = Unlabelled_mDAG(DirectedStructure([(0, 1)], 3), Hypergraph([], 3))
    g2 = Unlabelled_mDAG(DirectedStructure([(2, 1)], 3), Hypergraph([], 3))
    h = Unlabelled_mDAG(DirectedStructure([(0, 1), (1, 2)], 3), Hypergraph([], 3))
    assert g == g and g == g2 and g2 == g and hash(g) == hash(g2)
    assert len({g, g2}) == 1 == len({g2, g}) and len({g, h}) == 2
    assert g != h and g != mDAG(DirectedStructure([(0, 1)], 3), Hypergraph([], 3))
    assert g != Unlabelled_mDAG(DirectedStructure([(0, 1)], 4), Hypergraph([], 4))


def test_mdag_and_directed_structure_equality_are_typed_and_node_count_aware():
    assert mDAG(DirectedStructure([], 2), Hypergraph([], 2)) != mDAG(DirectedStructure([], 3), Hypergraph([], 3))
    a, b = mDAG(DirectedStructure([(0, 1)], 3), Hypergraph([(1, 2)], 3)), mDAG(DirectedStructure([(0, 1)], 3), Hypergraph([(1, 2)], 3))
    assert a == b and hash(a) == hash(b) and len({a, b}) == 1
    assert DirectedStructure([], 3) != 0 and DirectedStructure([], 2) != DirectedStructure([], 3)
    assert DirectedStructure([(0, 1)], 3) == DirectedStructure([(0, 1)], 3)
    assert DirectedStructure([(0, 1)], 3) < DirectedStructure([(0, 1), (1, 2)], 3)     # the order still works
    assert mDAG(DirectedStructure([], 3), Hypergraph([], 3)) != 0


def test_directed_structure_order_agrees_with_node_count_aware_equality():
    small, large = DirectedStructure([], 2), DirectedStructure([], 3)
    assert small < large and large > small
    assert not large < small and not small > large
    assert sorted([large, small]) == [small, large]
    with pytest.raises(TypeError):
        small < 0


# -- finding 3: iterator inputs -----------------------------------------------------------------------------------

def test_named_constructors_read_an_iterator_once():
    from_list = LabelledHypergraph(('A', 'B', 'C'), [('A', 'B')])
    from_iter = LabelledHypergraph(('A', 'B', 'C'), iter([('A', 'B')]))
    assert from_iter.number_of_nonsingleton_latent == 1 and from_iter.as_tuples == from_list.as_tuples
    assert LabelledHypergraph(('A', 'B'), iter([('A', 'B', 'C')])).as_tuples == ((0, 1),)       # a subhypergraph
    assert LabelledDirectedStructure(('A', 'B', 'C'), iter([('A', 'B')])).edge_list == [(0, 1)]
    assert Hypergraph(iter([(0, 1)]), 3).as_tuples == ((0, 1),) and DirectedStructure(iter([(0, 1)]), 3).edge_list == [(0, 1)]
    assert LabelledHypergraph(('A', 'B', 'C'), [iter(('A', 'B'))]).as_tuples == ((0, 1),)     # the facets too


# -- boundary: node indices, tuple arguments ----------------------------------------------------------------------

@pytest.mark.parametrize("edges", [[(-1, 0)], [(0, 5)], [(1, 1)]])
def test_directed_structure_rejects_bad_node_indices(edges):
    with pytest.raises(ValueError):
        DirectedStructure(edges, 3)


@pytest.mark.parametrize("facets", [[(-1, 0)], [(0, 7)]])
def test_hypergraph_rejects_bad_node_indices(facets):
    with pytest.raises(ValueError):
        Hypergraph(facets, 3)


def test_closures_and_adjacency_accept_any_node_sequence():
    G = mDAG(DirectedStructure([(0, 1), (1, 2)], 3), Hypergraph([(0, 2)], 3))
    assert G.set_closure((0, 1)).tolist() == G.set_closure([0, 1]).tolist() == G.set_closure(np.array([0, 1])).tolist()
    assert G.set_closure(1).tolist() == G.set_closure([1]).tolist()
    A = AdjMat(DirectedStructure([(0, 1), (1, 2)], 3).as_bit_square_matrix)
    assert A.children_of((0, 1)).tolist() == A.children_of([0, 1]).tolist() == [1, 2]
    assert A.parents_of(frozenset({1, 2})).tolist() == [0, 1] and A.children_of(()).tolist() == []
    assert A.children_of(0).tolist() == [1]


def test_fractional_node_indices_are_rejected_instead_of_truncated():
    graph = mDAG(DirectedStructure([(0, 1), (1, 2)], 3), Hypergraph([(0, 2)], 3))
    for nodes in ([0.5], (0.5,), np.array([0.5])):
        with pytest.raises(IndexError):
            graph.directed_structure_instance.adjMat.children_of(nodes)
        with pytest.raises(IndexError):
            graph.set_closure(nodes)


# -- finding 9: the matrix-returning support wrapper --------------------------------------------------------------

def test_matrix_returning_support_wrapper_calls_the_matrix_api():
    triangle = mDAG(DirectedStructure([], 3), Hypergraph([(0, 1), (1, 2), (0, 2)], 3))
    found = triangle.infeasible_binary_supports_as_matrices_up_to(3)
    assert len(found) > 0 and all(m.dtype == int and m.shape in {(2, 3), (3, 3)} for m in found)
    expected = [m for n in (2, 3) for m in triangle.infeasible_binary_supports_n_events_as_matrices(n)]
    assert [tuple(map(tuple, m)) for m in found] == [tuple(map(tuple, m)) for m in expected]
    common_cause = mDAG(DirectedStructure([], 3), Hypergraph([(0, 1, 2)], 3))
    assert common_cause.infeasible_binary_supports_as_matrices_up_to(2) == []       # nothing is infeasible: empty
