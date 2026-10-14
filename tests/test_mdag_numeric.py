"""The numeric (bitmask) graph machinery of mDAG against networkx oracles: d-separation and e-separation sets,
automorphisms, predecessor-based edge and face tests, descendants, the supports.py parent convention, and the
relabelling-invariant keys."""
import itertools
import random

import networkx as nx
import numpy as np
import pytest

import semigraphoid as sg
from directed_structures import DirectedStructure
from hypergraphs import Hypergraph
from mDAG_advanced import mDAG
from metagraph_advanced import Observable_unlabelled_mDAGs


@pytest.fixture(scope="module")
def metagraph():
    return Observable_unlabelled_mDAGs(4, verbose=False)


@pytest.fixture(scope="module")
def sample(metagraph):
    """A reproducible sample of labelled 4-node mDAGs (directed structure x simplicial complex)."""
    rng = random.Random(2024)
    return rng.sample(metagraph.all_labelled_mDAGs, 250)


def nx_graph(md):
    """The DAG on visible and latent nodes in the numbering of parents_masks."""
    g = nx.DiGraph()
    g.add_nodes_from(range(md.number_of_nodes_including_singleton_latents))
    for v, mask in enumerate(md.parents_masks):
        for p in sg.bits_of(int(mask)):
            g.add_edge(p, v)
    return g


def nx_ci_and_esep(md):
    g = nx_graph(md)
    nodes = md.visible_nodes
    esep = set()
    for r in range(len(nodes) - 1):
        for deleted in itertools.combinations(nodes, r):
            h = g.copy()
            h.remove_nodes_from(deleted)
            rest = [v for v in nodes if v not in deleted]
            for x, y in itertools.combinations(rest, 2):
                others = [v for v in rest if v not in (x, y)]
                for k in range(len(others) + 1):
                    for Z in itertools.combinations(others, k):
                        if nx.is_d_separator(h, {x}, {y}, set(Z)):
                            esep.add(((x, y), Z, deleted))
    ci = {(pair, Z) for pair, Z, deleted in esep if not deleted}
    return ci, esep


def test_parents_masks_follow_the_supports_convention(sample):
    for md in sample:
        ext = md.as_extended_bit_array
        assert md.number_of_nodes_including_singleton_latents == ext.shape[0]
        for v in md.visible_nodes:
            assert sg.bits_of(int(md.parents_masks[v])) == md.parents_of_for_supports_analysis[v].tolist()
            assert md.parents_of[v] == md.parents_of_for_supports_analysis[v].tolist()
        for latent in range(md.number_of_visible, ext.shape[0]):
            assert md.parents_masks[latent] == 0
        singleton = [md.number_of_visible + r for r in range(ext.shape[0] - md.number_of_visible)
                     if ext[md.number_of_visible + r].sum() == 1]
        assert sg.bits_of(md.singleton_latent_mask) == singleton
        assert set(md.nonsingleton_latent_nodes) == set(range(md.number_of_visible, ext.shape[0])) - set(singleton)


def test_d_separation_and_e_separation_sets_match_networkx(sample):
    for md in sample[:120]:
        ci, esep = nx_ci_and_esep(md)
        assert md.all_CI_numeric == ci
        assert md.all_esep_numeric == esep
        assert md.CI_count == len(ci)
        assert md.d_separable_pairs == {pair for pair, Z in ci}
        assert md.no_dsep_relations == (not ci)


def test_automorphisms_match_the_graph_isomorphisms(sample):
    from networkx.algorithms.isomorphism import DiGraphMatcher
    for md in sample:
        g = nx_graph(md)
        expected = {tuple(mapping[v] for v in md.visible_nodes)
                    for mapping in DiGraphMatcher(g, g).isomorphisms_iter()}
        assert set(md.visible_automorphisms) == expected
        assert md.automorphic_order == len(expected)


def test_predecessor_tests_match_the_networkx_definitions(sample):
    for md in sample:
        g = md.as_graph   # visible and non-singleton latent nodes
        pa = {v: set(g.predecessors(v)) for v in g.nodes}
        droppable = [tuple(e) for e in md.directed_structure_instance.edge_list
                     if any(set(e) <= set(f) for f in md.simplicial_complex_instance.compressed_simplicial_complex)
                     and pa[e[0]] <= pa[e[1]]]
        assert md.droppable_edges == droppable
        addable = {e for e in set(md.skeleton_instance.as_edges) - set(md.directed_structure_instance.edge_list)
                   if pa[e[0]] <= pa[e[1]]}
        assert set(md.addable_edges) == addable
        pp = {v: pa[v] | {v} for v in md.visible_nodes}
        for facet in md.simplicial_complex_instance.compressed_simplicial_complex:
            for C, D in md._all_bipartitions(list(facet)):
                union = set().union(*(pp[c] for c in C))
                assert ((C, D) in md.splittable_faces) == all(union <= pp[d] for d in D)
                assert ((C, D) in md.eventually_splittable_faces) == all(union - C <= pp[d] - D for d in D)
        for v in md.visible_nodes:
            assert md.descendants(v) == nx.descendants(g, v)
            assert md.singleton_predecessors(v) == frozenset(pp[v])


def test_unlabelled_keys_are_relabelling_invariant(sample):
    for md in sample[:60]:
        perm = list(md.visible_nodes)
        random.Random(md.unique_id).shuffle(perm)
        ds = DirectedStructure([(perm[a], perm[b]) for a, b in md.directed_structure_instance.edge_list], 4)
        sc = Hypergraph([frozenset(perm[v] for v in f) for f in md.simplicial_complex_instance.compressed_simplicial_complex], 4)
        other = mDAG(ds, sc)
        assert other.unique_unlabelled_id == md.unique_unlabelled_id
        assert other.all_CI_unlabelled == md.all_CI_unlabelled
        assert other.all_esep_unlabelled == md.all_esep_unlabelled
        assert other.all_densely_connected_pairs_unlabelled == md.all_densely_connected_pairs_unlabelled
        # and the keys are relabellings of the labelled relations
        assert len(other.all_CI_unlabelled) == len(md.all_CI_numeric)
        assert len(other.all_esep_unlabelled) == len(md.all_esep_numeric)


def test_small_mdags_have_keys_too():
    for n, edges, facets in ((2, [(0, 1)], []), (2, [], [(0, 1)]), (3, [(0, 1)], [(1, 2)])):
        md = mDAG(DirectedStructure(edges, n), Hypergraph(facets, n))
        ci, esep = nx_ci_and_esep(md)
        assert md.all_CI_numeric == ci and md.all_esep_numeric == esep
        assert md.esep_model.shape == (1 << n, n, n, 1 << n)
        assert len(md.all_CI_unlabelled) == len(ci) and len(md.all_esep_unlabelled) == len(esep)
        assert md.all_densely_connected_pairs_unlabelled is not None


def test_metagraph_class_counts(metagraph):
    M = metagraph
    assert len(M.dict_uniqind_unlabelled_mDAGs) == 2809
    assert len(M.equivalence_classes_as_ids) == 396
    assert len(M.CI_classes) == 25
    assert len(M.esep_classes) == 55
    assert len(M.Dense_connectedness_and_esep) == 58
    # Only dominances between structures of equal skeleton and CI count feed the equivalence search.
    assert all(s.skeleton_nof_edges == w.skeleton_nof_edges and s.CI_count == w.CI_count
               for s, w in map(M.lookup_mDAG, M.boring_dominances))
