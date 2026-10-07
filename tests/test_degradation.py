"""The degradation piggyback (quantum source to classical source) as a lookup, and the weakest-only seed list."""
from hypergraphs import Hypergraph
from directed_structures import DirectedStructure
from quantum_mDAG import QmDAG

import qc_gap_search as S
import known_QC_gaps as K


def Q(edges, n, C, Qf):
    return QmDAG(DirectedStructure(edges, n), Hypergraph(C, n), Hypergraph(Qf, n))


def test_degradations_make_quantum_facets_classical_and_absorb_them():
    g = Q([(0, 1)], 4, [(0, 1, 2)], [(0, 1), (2, 3)])
    outs = dict(g.degradations())
    assert set(outs) == {(('classical', ((0, 1),)),), (('classical', ((2, 3),)),)}   # both classical: no quantum facet left, skipped
    d = outs[(('classical', ((0, 1),)),)]
    # Q{0,1} made classical lies inside C{0,1,2} and is absorbed by it.
    assert d.C_simplicial_complex_instance.simplicial_complex_as_sets == {frozenset({0, 1, 2})}
    assert d.Q_simplicial_complex_instance.simplicial_complex_as_sets == {frozenset({2, 3})}


def test_seeds_are_weakest_and_one_per_relabelling_class():
    K.weakest_and_distinct(K.SEEDS)          # asserts both properties
    assert len(K.SEEDS) == 14 and len(K.SEEDS_3_NODES) == 5 and len(K.SEEDS_4_NODES) == 9
    for name in ('QG_Bell_Edge_Edge_SettingsEdge', 'QG_Bell_Edge_Edge_SettingsC', 'QG_Bell_Edge_Edge_SettingsEdgeC'):
        assert name in K.SEEDS
    # A relabelling or an upgrade of a seed is rejected by the check.
    import pytest
    upgraded = dict(K.SEEDS, extra=Q([(1, 2)], 3, [], [(0, 1), (1, 2)]))     # QG_Instrumental_C with C{0,1} quantum
    with pytest.raises(AssertionError):
        K.weakest_and_distinct(upgraded)
    relabelled = dict(K.SEEDS, extra=Q([(0, 2)], 4, [(1, 3)], [(2, 3)]))     # QG_Bell_C_Edge with the parties swapped
    with pytest.raises(AssertionError):
        K.weakest_and_distinct(relabelled)


def test_an_upgraded_seed_is_proven_by_one_degradation_lookup():
    instrumental_all_quantum = Q([(1, 2)], 3, [], [(0, 1), (1, 2)])             # upgrade of QG_Instrumental_C
    bell_all_quantum = Q([], 4, [], [(0, 2), (1, 3), (2, 3)])                   # upgrade of QG_Bell_C_C
    report = S.prove_gaps([instrumental_all_quantum, bell_all_quantum], K.SEEDS, verbose=False)
    for g, seed in ((instrumental_all_quantum, 'QG_Instrumental_C'), (bell_all_quantum, 'QG_Bell_C_C')):
        chain = report.proven[g.unique_unlabelled_id]
        assert [t.trick for t in chain] == ['degradation'] and report.seed_hit[g.unique_unlabelled_id] == seed
    # Lookup children are registered but never expanded.
    ex = report.explorer
    assert ex.lookup_only and all(gid not in ex.edges for gid in ex.lookup_only)
    assert 'degradation' in report.certificate(bell_all_quantum)


def test_lookup_applies_to_every_structure_the_search_reaches():
    # 0->1->2; Q{0,1,3}, Q{0,1,2} is all quantum; no all-quantum three-node seed is reachable, but PD on 0 (or
    # teleportation marginalization of 0) gives 1->2; Q{1,3}, Q{1,2}, whose degradation C{1,3}, Q{1,2} is the seed
    # QG_Instrumental_C: the lookup fires on a structure reached by an elementary step, not on the input.
    g = Q([(0, 1), (1, 2)], 4, [], [(0, 1, 3), (0, 1, 2)])
    report = S.prove_gaps([g], K.SEEDS_3_NODES, stages=S.default_stages()[:1], verbose=False)
    chain = report.proven[g.unique_unlabelled_id]
    assert len(chain) == 2 and chain[0].trick in ('PD', 'teleportation_marginalization') and chain[1].trick == 'degradation'
    assert report.seed_hit[g.unique_unlabelled_id] == 'QG_Instrumental_C'


def test_unified_fritz_records_the_certificate():
    tetra = Q([], 4, [], [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
    outs = dict(S.fritz_tricks(max_visible=5, use_lp=False)['Fritz'](tetra))
    params = (('predictors', (0,)), ('predicted', ((1, 'replace'),)), ('predictor_mode', 'drop'), ('certificate', 'dsep'))
    assert params in outs and outs[params].unique_unlabelled_id == K.QG_Triangle.unique_unlabelled_id
    assert all(dict(p)['certificate'] == 'dsep' for p in outs)


def test_unified_fritz_lp_certificate_is_named_entropic():
    import pytest
    pytest.importorskip("mosek")
    # The first LP example of the manuscript (output a relabelling of QG_Bell_C_Edge): predictor 3 kept, predicted 2, certified by the LP (relabel targets).
    g = Q([(0, 1), (1, 2)], 4, [], [(0, 2), (1, 3), (2, 3)])
    outs = dict(S.fritz_tricks(max_visible=5, predictor_mode='split', modes=('replace',))['Fritz'](g))
    params = (('predictors', (3,)), ('predicted', ((2, 'replace'),)), ('predictor_mode', 'split'), ('certificate', 'entropic'))
    assert params in outs
    assert outs[params].unique_unlabelled_id == K.QG_Bell_C_Edge.unique_unlabelled_id
    # Without the LP the same step is absent.
    outs_dsep = dict(S.fritz_tricks(max_visible=5, predictor_mode='split', modes=('replace',), use_lp=False)['Fritz'](g))
    assert params not in outs_dsep
