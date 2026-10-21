"""Shared permutation results preserve exact encodings and coherent relabelling order."""
import itertools
import math

import pytest

import hypergraphs as H
from canonical_forms import CanonicalForms
from directed_structures import DirectedStructure
from metagraph_advanced import ClassicalMetagraph, Observable_unlabelled_mDAGs, QuantumMetagraph
from quantum_mDAG import QmDAG
from radix import bitarray_to_int


def reference_permutations(hypergraph):
    return tuple(bitarray_to_int(bits) for bits in hypergraph.bit_array_permutations)


@pytest.fixture
def isolated_cache(monkeypatch):
    cache = {}
    monkeypatch.setattr(H, '_INTEGER_PERMUTATIONS_CACHE', cache)
    return cache


def test_equal_parts_share_results_without_retaining_graph_objects(isolated_cache):
    first = H.Hypergraph([(0, 1), (1, 2)], 4)
    other = H.LabelledHypergraph(('A', 'B', 'C', 'D'), [('B', 'C'), ('B', 'A')])
    result = first.as_integer_permutations
    assert result is other.as_integer_permutations
    assert result == reference_permutations(first)
    assert len(result) == math.factorial(4)
    assert result is first.as_integer_permutations       # the per-object cache remains in use
    assert len(isolated_cache) == 1
    assert next(iter(isolated_cache)) == (4, ((0, 1), (1, 2)))


def test_node_counts_remain_distinct_even_for_equal_integer_encodings(isolated_cache):
    three, four = H.Hypergraph([], 3), H.Hypergraph([], 4)
    assert three.as_integer == four.as_integer == 0
    assert three.as_integer_permutations == (0,) * 6
    assert four.as_integer_permutations == (0,) * 24
    assert len(isolated_cache) == 2


@pytest.mark.parametrize('n', [3, 4])
def test_all_small_hypergraph_permutations_match_the_reference(n, isolated_cache):
    for graph in Observable_unlabelled_mDAGs(n, verbose=False).all_simplicial_complices:
        expected = reference_permutations(graph)
        assert graph.as_integer_permutations == expected
        copy = H.Hypergraph(reversed(graph.compressed_simplicial_complex), n)
        assert copy.as_integer_permutations is graph.as_integer_permutations


def test_permutations_preserve_arbitrary_precision(isolated_cache):
    graph = H.Hypergraph(list(itertools.combinations(range(6), 3)), 6)
    expected = reference_permutations(graph)
    assert max(map(int, expected)) > 2**64
    assert graph.as_integer_permutations == expected
    assert H.Hypergraph(graph.as_tuples, 6).as_integer_permutations is graph.as_integer_permutations


def test_quantum_id_uses_coherent_permutations_for_all_parts(isolated_cache):
    directed = DirectedStructure([(0, 1), (1, 3)], 4)
    classical, quantum = H.Hypergraph([(0, 2)], 4), H.Hypergraph([(1, 2), (2, 3)], 4)
    graph = QmDAG(directed, classical, quantum)
    expected = (4,) + min(zip(directed.as_integer_permutations,
                               reference_permutations(classical), reference_permutations(quantum)))
    assert graph.unique_unlabelled_id == expected
    for perm in itertools.permutations(range(4)):
        copy = QmDAG(DirectedStructure([(perm[a], perm[b]) for a, b in directed.edge_list], 4),
                     H.Hypergraph([tuple(perm[v] for v in f) for f in classical.as_tuples], 4),
                     H.Hypergraph([tuple(perm[v] for v in f) for f in quantum.as_tuples], 4))
        assert copy.unique_unlabelled_id == expected


def test_manual_clear_keeps_existing_objects_valid(isolated_cache):
    graph = H.Hypergraph([(0, 1), (1, 2)], 3)
    result = graph.as_integer_permutations
    H.clear_permutation_cache()
    assert not isolated_cache
    assert graph.as_integer_permutations is result
    copy = H.Hypergraph(graph.as_tuples, 3)
    assert copy.as_integer_permutations == result
    assert copy.as_integer_permutations is not result


@pytest.mark.parametrize('kind', [Observable_unlabelled_mDAGs, ClassicalMetagraph, QuantumMetagraph])
@pytest.mark.parametrize('n', [6, 7, 1000])
def test_unsupported_full_enumeration_fails_before_generating_families(kind, n, monkeypatch):
    def must_not_enumerate(self):
        pytest.fail('family generated before the size check')
    for name in ('all_directed_structures', 'all_simplicial_complices'):
        monkeypatch.setattr(Observable_unlabelled_mDAGs, name, property(must_not_enumerate))
    with pytest.raises(ValueError, match='at most 5 visible nodes'):
        kind(n, verbose=False).canonical_forms


def test_explicit_six_node_family_that_fits_is_still_supported():
    directed = DirectedStructure([(0, 1)], 6)
    hypergraph = H.Hypergraph([(1, 5)], 6)
    forms = CanonicalForms(6, [directed.as_integer], [directed.as_bit_square_matrix],
                           [hypergraph.as_integer], [hypergraph.as_tuples])
    expected = min(d + int(h) * 2**36 for d, h in
                   zip(directed.as_integer_permutations, reference_permutations(hypergraph)))
    assert forms.canonical(directed.as_integer, hypergraph.as_integer) == expected


def test_explicit_oversize_family_still_reports_the_table_limit():
    with pytest.raises(ValueError, match='20 facets x 6 nodes'):
        CanonicalForms(6, [], [], [], [list(itertools.combinations(range(6), 3))])
