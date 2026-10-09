"""Tests for the entropic rung of the Fritz piggyback, with both certificate engines: the semigraphoid closure
(always available) and the LP (skipped without mosek)."""
import importlib.util

import pytest

from hypergraphs import Hypergraph
from directed_structures import DirectedStructure
from quantum_mDAG import QmDAG
from known_QC_gaps import QG_Bell_C_Edge


def Q(edges, n, C, Qf):
    return QmDAG(DirectedStructure(edges, n), Hypergraph(C, n), Hypergraph(Qf, n))


TRIANGLE = Q([], 3, [], [(0, 1), (1, 2), (0, 2)])
SQUARE = Q([], 4, [], [(2, 3), (1, 3), (0, 1), (0, 2)])
G2 = Q([(1, 2), (1, 3)], 4, [(0, 3), (0, 2)], [(0, 1), (1, 2)])
# Khanna-Pusey-Colbeck G1: 0=C, 1=D, 2=E, 3=F; latents A over {E,F}, B over {D,F}; edges C->D, C->E, D->E.
G1_KPC = Q([(0, 1), (0, 2), (1, 2)], 4, [], [(2, 3), (1, 3)])


HAVE_MOSEK = importlib.util.find_spec("mosek") is not None
ENGINES = [pytest.param('semigraphoid'), pytest.param('lp', marks=pytest.mark.skipif(not HAVE_MOSEK, reason="mosek not installed"))]


def certificate(g, X, kept, predicted, engine, try_markov=False):
    """_semigraphoid_certificate or _entropic_certificate, by engine."""
    if engine == 'lp':
        return g._entropic_certificate(frozenset(X), kept, predicted, try_markov=try_markov)
    return g._semigraphoid_certificate(frozenset(X), kept, predicted, try_markov=try_markov)


def kept_parents(g, s, K):
    """The LP's parent sets after restricting s to K (what fritz_certificate hands to _entropic_certificate)."""
    nodes, parents = g.lp_structure
    kept = dict(parents)
    kept[s] = frozenset(k for k in K if k in parents)
    return kept


def pairs(g, **kwargs):
    """{(target, predictor): certificate} over every emitted step."""
    return {(dict(p)['targets'][0], dict(p)['predictor']): dict(p)['certificate'] for p, _ in g.fritz_steps(**kwargs)
            if len(dict(p)['targets']) == 1}


BEYOND = ('semigraphoid', 'entropic')


def test_lp_structure_excludes_noise_nodes():
    nodes, parents = G1_KPC.lp_structure
    assert nodes == (0, 1, 2, 3, 4, 5)
    assert parents[2] == {0, 1, 5} and parents[3] == {4, 5}


@pytest.mark.parametrize('engine', ENGINES)
def test_d_separation_is_subsumed_by_the_engine(engine):
    for g, predictor in ((TRIANGLE, 2), (SQUARE, 3), (G1_KPC, 3)):
        for s in g.visible_nodes:
            if s == predictor or g.fritz_deletion(s, {predictor}) is None:
                continue
            K, _ = g.fritz_deletion(s, {predictor})
            if g.fritz_certificate(s, K, {predictor}, use_lp=False) == 'dsep':
                assert certificate(g, {predictor}, kept_parents(g, s, K), (s,), engine) is not None
    # The entropic pass never re-certifies what d-separation certified: the certificate names the cheaper rung.
    assert all(c == 'dsep' for c in pairs(TRIANGLE, use_lp=True, engine=engine).values())


@pytest.mark.parametrize('engine', ENGINES)
def test_kpc_example_is_rescued_by_the_relabelled_target_set(engine):
    # d-separation rejects E (F is d-connected to D through B); both engines accept it with the relabelled targets
    # and reject the plain Markov targets over G's own latents (a classical model may relabel latents).
    K, D = G1_KPC.fritz_deletion(2, {3})
    assert K == {5}
    assert G1_KPC.fritz_certificate(2, K, {3}, use_lp=False) is None
    assert G1_KPC.fritz_certificate(2, K, {3}, use_lp=True, engine=engine) == ('entropic' if engine == 'lp' else 'semigraphoid')
    kept = kept_parents(G1_KPC, 2, K)
    assert certificate(G1_KPC, {3}, kept, (2,), engine) == 'relabel'
    if engine == 'semigraphoid':
        import semigraphoid as sg
        E = G1_KPC._semigraphoid_model().copy()
        sg.add_functional_dependence(E, 2, 1 << 3)
        n = len(G1_KPC.lp_structure[0])
        cand = sg.dsep_all(n, sg.parents_to_masks(kept, n))
        E[:4, :4, :16] |= cand[:4, :4, :16]
        sg.close(E)
        assert not sg.contains(E, cand)
        return
    from entropic_lp import local_markov_rows
    nodes, _ = G1_KPC.lp_structure
    kept = kept_parents(G1_KPC, 2, K)
    lp = G1_KPC._entropic_lp()
    handle = lp.push_hypotheses(G1_KPC._entropic_hypotheses(frozenset({3}), kept, (2,)))
    try:
        assert not lp.implies_all(row for _, row in local_markov_rows(kept, nodes))
    finally:
        lp.pop_to(handle)


@pytest.mark.parametrize('engine', ENGINES)
def test_kpc_example_reaches_bell_in_one_kept_step(engine):
    outputs = {dict(p)['targets'][0]: (dict(p), c) for p, c in G1_KPC.fritz_steps(predictor_mode='split', engine=engine)
               if dict(p)['predictor'] == (3,)}
    info, out = outputs[2]
    assert out.unique_unlabelled_id == QG_Bell_C_Edge.unique_unlabelled_id
    assert info['certificate'] == ('entropic' if engine == 'lp' else 'semigraphoid') and info['deleted'] == ((0, 1),)
    assert outputs[1][0]['certificate'] == 'dsep'
    # Without the LP the step is absent.
    assert 2 not in {dict(p)['targets'][0] for p, _ in G1_KPC.fritz_steps(predictor_mode='split', use_lp=False)
                     if dict(p)['predictor'] == (3,)}


def test_entropic_steps_are_emitted_after_every_d_separation_step():
    certificates = [dict(p)['certificate'] for p, _ in G1_KPC.fritz_steps(predictor_mode='split')]
    assert 'semigraphoid' in certificates and 'dsep' in certificates
    assert certificates == sorted(certificates, key=lambda c: c in BEYOND)   # all 'dsep' first


def test_split_node_is_a_faithful_duplication():
    split = G1_KPC.split_node(3)
    assert split.number_of_visible == 5
    # The shared-noise facet {3, 4} is dominated by the quantum facets the pair now shares, so it is absorbed.
    assert split.C_simplicial_complex_instance.simplicial_complex_as_sets == set()
    assert split.Q_simplicial_complex_instance.simplicial_complex_as_sets == {frozenset({2, 3, 4}), frozenset({1, 3, 4})}
    assert split.directed_structure_instance.edge_list == G1_KPC.directed_structure_instance.edge_list
    chain = Q([(0, 1), (1, 2)], 3, [], [])
    split_chain = chain.split_node(1)
    assert split_chain.C_simplicial_complex_instance.simplicial_complex_as_sets == {frozenset({1, 3})}
    assert split_chain.directed_structure_instance.edge_list == [(0, 1), (0, 3), (1, 2), (3, 2)]


@pytest.mark.parametrize('engine', ENGINES)
def test_split_target_runs_the_engine_on_the_split_structure(engine):
    # Split target = split the target, then restrict the copy. For G1 the copy of E is NOT certified: the original E
    # keeps reading A, so neither target set can identify A with the copy (a classical model may encode A
    # differently for E and for the copy). The copy of D is certified by plain d-separation.
    by_target = {dict(p)['targets'][0]: dict(p)['certificate'] for p, _ in G1_KPC.fritz_steps(target_mode='split', predictor_mode='split', engine=engine)
                 if dict(p)['predictor'] == (3,)}
    assert by_target == {1: 'dsep'}
    split = G1_KPC.split_node(2)
    K, _ = split.fritz_deletion(4, {3})
    assert split.fritz_certificate(4, K, {3}, use_lp=True, engine=engine) is None


@pytest.mark.parametrize('engine', ENGINES)
def test_g2_is_proven_through_the_relabelled_certificate(engine):
    # The old d-separation code reached IV3b from G2 unsoundly (node 0 kept a latent its predictor could not see).
    # The entropic rung reaches IV3b soundly: predictor 3 predicts 0 restricted to their shared classical latent,
    # certified by the relabelled target set (0 itself plays the role of that latent); conditioning on 3 then gives
    # IV3b. Hand check: the lifted strategy keeps 3 = (3_out, copy of 0), so every observable d-separation of the
    # output holds for it, and a classical model of G2 with those independences and 0 = g(3) yields a model of the
    # output by setting the shared latent equal to 0.
    IV3b = Q([(1, 2)], 3, [(0, 1)], [(1, 2)])
    K, D = G2.fritz_deletion(0, {3})
    facet_03 = [idx for idx, (kind, members) in G2.effective_DAG_data[1].items() if kind == 'C' and members == {0, 3}]
    assert K == set(facet_03)
    assert G2.fritz_certificate(0, K, {3}, use_lp=False) is None
    assert certificate(G2, {3}, kept_parents(G2, 0, K), (0,), engine) == 'relabel'
    outputs = dict(G2.fritz_steps(predictor_mode='split', engine=engine))
    out = outputs[(('targets', (0,)), ('target_mode', 'unsplit'), ('deleted', (('C{0,2}', 'Q{0,1}'),)), ('predictor', (3,)),
                   ('predictor_mode', 'split'), ('certificate', 'entropic' if engine == 'lp' else 'semigraphoid'))]
    assert out.directed_structure_instance.edge_list == [(1, 2), (1, 3)]
    assert out.C_simplicial_complex_instance.simplicial_complex_as_sets == {frozenset({0, 3})}
    assert out.Q_simplicial_complex_instance.simplicial_complex_as_sets == {frozenset({1, 2})}
    assert out.condition(3).unique_unlabelled_id == IV3b.unique_unlabelled_id


@pytest.mark.parametrize('engine', ENGINES)
def test_descendant_predictor_is_certifiable_beyond_d_separation_only_when_allowed(engine):
    # 0 -> 1 -> 2 -> 3 with Q{0,2}, Q{1,3}: 3 is a descendant of 1 sharing a facet with it. d-separation fails for
    # every descendant (the noise of 1 reaches 3); the engine certifies the step. Off by default.
    g = Q([(0, 1), (1, 2), (2, 3)], 4, [], [(0, 2), (1, 3)])
    label = 'entropic' if engine == 'lp' else 'semigraphoid'
    assert g.fritz_pool(1) == [] and g.fritz_pool(1, allow_descendants=True) == [3]
    K, D = g.fritz_deletion(1, {3})
    assert g.fritz_certificate(1, K, {3}, use_lp=False) is None
    assert g.fritz_certificate(1, K, {3}, use_lp=True, engine=engine) == label
    assert (1, (3,)) not in pairs(g, predictor_mode='split', engine=engine)
    assert pairs(g, predictor_mode='split', allow_descendants=True, engine=engine)[(1, (3,))] == label


@pytest.mark.parametrize('engine', ENGINES)
def test_markov_and_relabel_certificates_are_logically_independent(engine):
    # relabel without markov: the KPC example (test above). markov without relabel: s=0 reads a visible parent 4 and a
    # facet L={0,1,2,3} shared with the predictor 3. d-separation (hence 'markov') certifies deleting 4->0, but the
    # relabelled targets need 2 ⊥ 1 | 0, which the model L=(L1,L2), 0=L1, 1=2=L2, 3=L violates (I(2:1|0) = 1 bit).
    g = Q([(4, 0)], 5, [], [(0, 1, 2, 3)])
    K, D = g.fritz_deletion(0, {3})
    assert g.fritz_certificate(0, K, {3}, use_lp=False) == 'dsep'
    kept = kept_parents(g, 0, K)
    assert certificate(g, {3}, kept, (0,), engine, try_markov=True) == 'markov'
    assert certificate(g, {3}, kept, (0,), engine, try_markov=False) is None
    if engine == 'semigraphoid':
        return
    from entropic_lp import local_markov_rows
    nodes, _ = g.lp_structure
    lam = min(kept[0])
    relabelled = {v: (ps - {lam}) | {0} if lam in ps else ps for v, ps in kept.items() if v != lam}
    relabelled[0] = frozenset()
    lp = g._entropic_lp()
    handle = lp.push_hypotheses(g._entropic_hypotheses(frozenset({3}), kept, (0,)))
    try:
        assert not lp.implies_all(row for _, row in local_markov_rows(relabelled, [v for v in nodes if v != lam]))
    finally:
        lp.pop_to(handle)


@pytest.mark.skipif(not HAVE_MOSEK, reason="mosek not installed")
def test_both_engines_agree_on_the_examples():
    import quantum_mDAG as QM
    before = len(QM.ENGINE_DISAGREEMENTS)
    for g in (TRIANGLE, SQUARE, G1_KPC, G2, Q([(0, 1), (1, 2), (2, 3)], 4, [], [(0, 2), (1, 3)])):
        for target_mode in ('unsplit', 'split'):
            for pm in ('unsplit', 'split'):
                both = {p[:-1] + (('certificate', 'x'),) if dict(p)['certificate'] in BEYOND else p: c.unique_unlabelled_id
                        for p, c in g.fritz_steps(target_mode=target_mode, predictor_mode=pm, engine='both', allow_descendants=True)}
                lp = {p[:-1] + (('certificate', 'x'),) if dict(p)['certificate'] in BEYOND else p: c.unique_unlabelled_id
                      for p, c in g.fritz_steps(target_mode=target_mode, predictor_mode=pm, engine='lp', allow_descendants=True)}
                assert both == lp
    assert len(QM.ENGINE_DISAGREEMENTS) == before
