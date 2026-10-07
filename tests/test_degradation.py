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
    ids = [g.unique_unlabelled_id for g in K.SEEDS.values()]
    assert len(ids) == len(set(ids))
    known_ids = {g.unique_unlabelled_id for g in K.KNOWN.values()}
    for g in K.SEEDS.values():
        assert not any(d.unique_unlabelled_id in known_ids for _, d in g.degradations())
    # Every named gap is a seed or degrades to one (possibly in several steps, here always one).
    seed_ids = set(ids)
    for name, g in K.KNOWN.items():
        assert g.unique_unlabelled_id in seed_ids or any(d.unique_unlabelled_id in seed_ids for _, d in g.degradations()), name
    # The new Bell variants are seeds.
    for name in ('QG_Bell_SettingEdge', 'QG_Bell_SettingsC', 'QG_Bell_SettingEdgeC'):
        assert name in K.SEEDS


def test_an_upgraded_seed_is_proven_by_one_degradation_lookup():
    report = S.prove_gaps([K.QG_Instrumental3, K.QG_Bell6d], K.SEEDS, verbose=False)
    for g, seed in ((K.QG_Instrumental3, 'QG_Instrumental3b'), (K.QG_Bell6d, 'QG_Bell6')):
        chain = report.proven[g.unique_unlabelled_id]
        assert [t.trick for t in chain] == ['degradation'] and report.seed_hit[g.unique_unlabelled_id] == seed
    # Lookup children are registered but never expanded.
    ex = report.explorer
    assert ex.lookup_only and all(gid not in ex.edges for gid in ex.lookup_only)
    assert 'degradation' in report.certificate(K.QG_Bell6d)


def test_lookup_applies_to_every_structure_the_search_reaches():
    # 0->1->2; Q{0,1,3}, Q{0,1,2} is all quantum; no all-quantum three-node seed is reachable, but PD on 0 (or
    # teleportation marginalization of 0) gives 1->2; Q{1,3}, Q{1,2}, whose degradation C{1,3}, Q{1,2} is the seed
    # QG_Instrumental3b: the lookup fires on a structure reached by an elementary step, not on the input.
    g = Q([(0, 1), (1, 2)], 4, [], [(0, 1, 3), (0, 1, 2)])
    report = S.prove_gaps([g], K.SEEDS_3_NODES, stages=S.default_stages()[:1], verbose=False)
    chain = report.proven[g.unique_unlabelled_id]
    assert len(chain) == 2 and chain[0].trick in ('PD', 'teleportation_marginalization') and chain[1].trick == 'degradation'
    assert report.seed_hit[g.unique_unlabelled_id] == 'QG_Instrumental3b'


def test_unified_fritz_records_the_certificate():
    tetra = Q([], 4, [], [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
    outs = dict(S.fritz_tricks(max_visible=5, use_lp=False)['Fritz'](tetra))
    params = (('predictors', (0,)), ('predicted', ((1, 'replace'),)), ('predictor_mode', 'drop'), ('certificate', 'dsep'))
    assert params in outs and outs[params].unique_unlabelled_id == K.QG_Triangle3.unique_unlabelled_id
    assert all(dict(p)['certificate'] == 'dsep' for p in outs)


def test_unified_fritz_lp_certificate_is_named_entropic():
    import pytest
    pytest.importorskip("mosek")
    # The QG_Bell9 example of the manuscript: predictor 3 kept, predicted 2, certified by the LP (relabel targets).
    g = Q([(0, 1), (1, 2)], 4, [], [(0, 2), (1, 3), (2, 3)])
    outs = dict(S.fritz_tricks(max_visible=5, predictor_mode='split', modes=('replace',))['Fritz'](g))
    params = (('predictors', (3,)), ('predicted', ((2, 'replace'),)), ('predictor_mode', 'split'), ('certificate', 'entropic'))
    assert params in outs
    assert outs[params].unique_unlabelled_id == K.QG_Bell9.unique_unlabelled_id
    # Without the LP the same step is absent.
    outs_dsep = dict(S.fritz_tricks(max_visible=5, predictor_mode='split', modes=('replace',), use_lp=False)['Fritz'](g))
    assert params not in outs_dsep
