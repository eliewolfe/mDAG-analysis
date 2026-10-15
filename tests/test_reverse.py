"""The piggybacks in reverse (reverse_piggybacks, manuscript Appendix A): each inverse against the forward trick on
the Appendix A examples, the reverse closure against the forward elementary stage."""
import pytest

from directed_structures import DirectedStructure
from hypergraphs import Hypergraph
from quantum_mDAG import QmDAG
from equivalence_classes import ClassIndex
from qc_gap_search import Transition, default_stages, prove_gaps
import known_QC_gaps as K
import reverse_piggybacks as R


@pytest.fixture(scope="module")
def index():
    return ClassIndex((3, 4))


def Q(edges, n, C, Qf):
    return QmDAG(DirectedStructure(edges, n), Hypergraph(C, n), Hypergraph(Qf, n))


def same(a, b):
    return a.unique_unlabelled_id == b.unique_unlabelled_id


def candidates(inverse, c):
    return [(params, g) for trick, params, g in inverse(c)]


def test_interruption_of_the_instrumental_seeds_gives_the_bell_seeds(index):
    pairs = {'QG_Instrumental_Edge': 'QG_Bell_Edge_Edge', 'QG_Instrumental_EdgeC': 'QG_Bell_EdgeC_Edge',
             'QG_Instrumental_C': 'QG_Bell_C_Edge'}
    for seed, bell in pairs.items():
        c = K.SEEDS[seed]
        outs = candidates(R.inverse_node_stitching, c)
        assert len(outs) == 3                                  # one per node y (y childless gives an isolated node)
        assert any(same(g, K.SEEDS[bell]) for _, g in outs), seed
        for params, g in outs:                                 # every candidate stitches back to the seed exactly
            child = R.forward_child('node_stitching', params, g)
            assert child is not None and child.unique_id == c.unique_id
            assert index.key(child) == index.key(c)


def test_quantumification_is_confirmed_by_the_degradation(index):
    g3 = Q([(0, 1), (1, 2)], 4, [(0, 1, 3), (0, 1, 2)], [(0, 1), (0, 2)])     # G(3) of manuscript 0.4 (3\u2032 as 3)
    outs = candidates(R.inverse_degradation, g3)
    assert outs
    produced = {g.unique_id for _, g in outs}
    # the first structure of the 0.4 list: a quantum face added inside a classical facet (degrading it absorbs it)
    assert Q([(0, 1), (1, 2)], 4, [(0, 1, 3), (0, 1, 2)], [(0, 1), (0, 2), (1, 2)]).unique_id in produced
    # a quantum face strictly containing an existing quantum facet would absorb it, and the degradation could not
    # give G(3) back: not an option (the rest of the 0.4 list follows from the monotonicity of upgrading sources,
    # which the degradation piggyback as implemented does not realise in one step)
    assert not any(frozenset((0, 1, 3)) in R.q_facets_of(g) or frozenset((0, 1, 2)) in R.q_facets_of(g) for _, g in outs)
    for params, g in outs:
        child = R.forward_child('degradation', params, g)
        assert child is not None and child.unique_id == g3.unique_id
    tri = candidates(R.inverse_degradation, K.QG_Triangle)
    assert any(same(g, Q([], 3, [], [(0, 1), (1, 2), (0, 2)])) for _, g in tri)


def test_reverse_upward_reproduces_the_forward_lookup(index):
    C4 = index.classical[4]
    found = False
    for key in index.input_keys(4):
        rep = index.representative(key)
        for params, below in index.upward_steps(rep):
            below_key = index.key(below)
            triples = index.above_steps(below)
            assert any(index.key(m) == key and p == params for p, m, _ in triples)
            found = True
            break
        if found:
            break
    assert found


def test_param_translation_replays_on_the_relabelled_structure():
    c = K.SEEDS['QG_Instrumental_Edge']
    g = next(g for _, g in candidates(R.inverse_node_stitching, c) if same(g, K.SEEDS['QG_Bell_Edge_Edge']))
    other = K.SEEDS['QG_Bell_Edge_Edge']
    perm = R.permutation_to(g, other)
    assert perm is not None
    params = next(p for _, p, h in R.inverse_node_stitching(c) if h.unique_id == g.unique_id)
    translated = R.translate_params('node_stitching', params, perm)
    child = R.forward_child('node_stitching', translated, other)
    assert child is not None and same(child, c)
    assert R.translate_params('degradation', ((('classical', ((0, 1), (2, 3))),)), (3, 2, 1, 0)) == ((('classical', ((0, 1), (2, 3))),))


def test_closure_over_interruption_matches_the_forward_stage(index):
    closure = R.ReverseClosure(index, {'node_stitching': R.INVERSES['node_stitching']}, max_visible=4).run(K.SEEDS)
    assert closure.stats['node_stitching']['rejected'] == 0            # the inverse is exact
    inputs = index.inputs(4)
    forward = prove_gaps(inputs, K.SEEDS, stages=default_stages()[:1], index=index, verbose=False)
    with_reverse = prove_gaps(inputs, K.SEEDS, stages=default_stages()[:1], index=index, verbose=False, reverse=closure)
    assert set(with_reverse.proven) == set(forward.proven)
    # the main counts agree; the 'with X' attributions may shift towards the reversed trick, since a known key is
    # not expanded forward any more (its forward children are no longer recorded)
    for name in ('inputs', 'proven', 'remaining', 'structures', 'structures proven', 'structures remaining',
                 'classical classes of proven', 'with identity'):
        assert with_reverse.counts[name] == forward.counts[name], name
    assert with_reverse.counts['with node_stitching'] >= forward.counts['with node_stitching']
    for key in with_reverse.proven:
        assert with_reverse.seed_hit[key] in forward.seed_hit.values() or '+' in with_reverse.seed_hit[key]
    # every forward stitching edge into a known key has its reverse counterpart
    reverse_triples = {(t.trick, t.source, t.target) for ts in closure.edges.values() for t in ts}
    for ts in forward.explorer.edges.values():
        for t in ts:
            if t.trick == 'node_stitching' and t.target in closure.known:
                assert (t.trick, t.source, t.target) in reverse_triples
    # known keys are never expanded by the forward search
    assert all(t.trick in ('node_stitching', 'degradation', 'upward') for key in closure.known
               for t in with_reverse.explorer.edges.get(key, []))


def test_node_insertion_is_exact_and_contains_the_hybrid_example(index):
    c = K.SEEDS['QG_Instrumental_Edge']
    outs = list(R.inverse_PD(c))
    hybrid = Q([(0, 1), (1, 2)], 4, [(0, 3)], [(1, 2)])          # A.2: a node sharing a classical facet with the setting
    assert any(same(g, hybrid) for _, _, g in outs)
    assert any(same(g, Q([(0, 1), (1, 2)], 4, [], [(1, 2), (0, 3)])) for _, _, g in outs)   # its all-quantum version
    for trick, params, g in outs:
        assert g.number_of_visible == 4 and params == (('drop', 3),)
        child = R.forward_child(trick, params, g)
        assert child.unique_id == c.unique_id                           # exact: removing the node gives the seed back
    # without hybrids only all-quantum candidates are generated
    assert all(g.is_all_quantum for _, _, g in R.inverse_PD(c, hybrids=False))
    assert len(list(R.inverse_PD(c, hybrids=False))) < len(outs)


def test_un_marginalization_examples_are_confirmed(index):
    inst_C = K.SEEDS['QG_Instrumental_C']                                # 1->2; C{0,1}, Q{1,2}
    naive = list(R.inverse_marginalization(inst_C, False))
    tele = list(R.inverse_marginalization(inst_C, True))
    # A.3 (i): the naive marginalization of v in v->1->2; C{0,v}, Q{1,2}
    assert any(same(g, Q([(3, 1), (1, 2)], 4, [(0, 3)], [(1, 2)])) for _, _, g in naive)
    # A.3 (ii): the teleportation marginalization of 0 in G(3), with 3' as 3
    assert any(same(g, Q([(0, 1), (1, 2)], 4, [(0, 1, 3), (0, 1, 2)], [(0, 1), (0, 2)])) for _, _, g in tele)
    for outs in (naive, tele):
        for trick, params, g in outs:
            child = R.forward_child(trick, params, g)
            assert child is not None and index.key(child) == index.key(inst_C)
    # regression: a quantum 2-facet on a childless node leaves no quantum trace (singleton base)
    triangle = Q([], 3, [], [(0, 1), (1, 2), (0, 2)])
    assert any(same(g, Q([(2, 3)], 4, [], [(0, 1), (0, 2), (1, 2), (2, 3)]))
               for _, _, g in R.inverse_marginalization(triangle, False))
