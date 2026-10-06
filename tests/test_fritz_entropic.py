"""Tests for the entropic (LP-certified) Fritz piggyback."""
import pytest

pytest.importorskip("mosek")

from hypergraphs import Hypergraph
from directed_structures import DirectedStructure
from quantum_mDAG import QmDAG
from known_QC_gaps import SEEDS, QG_Bell5


def Q(edges, n, C, Qf):
    return QmDAG(DirectedStructure(edges, n), Hypergraph(C, n), Hypergraph(Qf, n))


TRIANGLE = Q([], 3, [], [(0, 1), (1, 2), (0, 2)])
SQUARE = Q([], 4, [], [(2, 3), (1, 3), (0, 1), (0, 2)])
G2 = Q([(1, 2), (1, 3)], 4, [(0, 3), (0, 2)], [(0, 1), (1, 2)])
# Khanna-Pusey-Colbeck G1: 0=C, 1=D, 2=E, 3=F; latents A over {E,F}, B over {D,F}; edges C->D, C->E, D->E.
G1_KPC = Q([(0, 1), (0, 2), (1, 2)], 4, [], [(2, 3), (1, 3)])
SEED_IDS = {g.unique_unlabelled_id: name for name, g in SEEDS.items()}


def test_lp_structure_excludes_noise_nodes():
    nodes, parents = G1_KPC.lp_structure
    assert nodes == (0, 1, 2, 3, 4, 5)
    assert parents[2] == {0, 1, 5} and parents[3] == {4, 5}


def test_d_separation_admissibility_is_subsumed_by_the_lp():
    for g, predictor in ((TRIANGLE, 2), (SQUARE, 3), (G1_KPC, 3)):
        by_dsep = g.fritz_admissible_targets((predictor,))
        for s, (common, others) in by_dsep.items():
            kept = g._fritz_kept_parents(by_dsep, {s: 'replace'})
            assert g._entropic_certificate(frozenset({predictor}), kept, (s,)) is not None
        entropic = g.fritz_entropic_admissible_targets((predictor,))
        assert set(by_dsep).issubset(entropic)
        for s in by_dsep:
            assert entropic[s][2] == 'dsep'


def test_kpc_example_is_rescued_by_the_relabelled_target_set():
    # d-separation rejects E (F is d-connected to D through B); the LP accepts it with the relabelled targets.
    assert 2 not in G1_KPC.fritz_admissible_targets((3,))
    admissible = G1_KPC.fritz_entropic_admissible_targets((3,))
    common, others, certificate = admissible[2]
    assert common == {5} and certificate == 'relabel'
    # The plain Markov targets over G's own latents are too strong here (a classical model may relabel latents).
    nodes, parents = G1_KPC.lp_structure
    kept = dict(parents)
    kept[2] = frozenset({5})
    from entropic_lp import local_markov_rows
    lp = G1_KPC._entropic_lp()
    handle = lp.push_hypotheses(G1_KPC._entropic_hypotheses(frozenset({3}), kept, (2,)))
    try:
        assert not lp.implies_all(row for _, row in local_markov_rows(kept, nodes))
    finally:
        lp.pop_to(handle)


def test_kpc_example_reaches_bell_in_one_split_step():
    outputs = G1_KPC.fritz_entropic_transitions((3,), predictor_modes=('split',), extra_deletions=False)
    by_params = {params[0]: (dict(params[1:]), out) for params, out in outputs}
    info, out = by_params[((2, 'replace'),)]
    assert out.unique_unlabelled_id == QG_Bell5.unique_unlabelled_id
    assert info['certificate'] == 'relabel'
    assert by_params[((1, 'replace'),)][0]['certificate'] == 'dsep'


def test_only_beyond_dsep_filters_plain_fritz_transitions():
    # Dropping the predictor with d-separation certificates is what the base Fritz trick already does; keeping it
    # ('split') is reserved for the final pass, so those outputs are emitted even when certified by d-separation.
    assert TRIANGLE.fritz_entropic_transitions((2,), predictor_modes=('drop',)) == []
    assert TRIANGLE.fritz_entropic_transitions((2,), predictor_modes=('drop',), only_beyond_dsep=False) != []
    split_outputs = TRIANGLE.fritz_entropic_transitions((2,), predictor_modes=('split',))
    assert split_outputs and all(dict(p[1:])['certificate'] == 'dsep' for p, _ in split_outputs)


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


def test_copy_mode_runs_the_lp_on_the_split_structure():
    # Copy mode = split the node, then replace mode on the copy. For G1 the copy of E is NOT certified: the
    # original E keeps reading A, so neither target set can identify A with the copy (a classical model may encode
    # A differently for E and for the copy). The copy of D is certified by plain d-separation.
    outputs = G1_KPC.fritz_entropic_transitions((3,), modes=('copy',), predictor_modes=('split',), extra_deletions=False)
    by_params = {params[0]: dict(params[1:]) for params, _ in outputs}
    assert set(by_params) == {((1, 'copy'),)}
    assert by_params[((1, 'copy'),)]['certificate'] == 'dsep'
    split = G1_KPC.split_node(2)
    assert set(split.fritz_entropic_admissible_targets((3,))) == {1}


def test_g2_is_proven_through_the_relabelled_certificate():
    # The old d-separation code reached IV3b from G2 unsoundly (node 0 kept a latent its predictor could not see).
    # The entropic trick reaches IV3b soundly: predictor 3 predicts 0 restricted to their shared classical latent,
    # certified by the relabelled target set (0 itself plays the role of that latent); conditioning on 3 then gives
    # IV3b. Hand check: the lifted strategy keeps 3 = (3_out, copy of 0), so every observable d-separation of the
    # output holds for it, and a classical model of G2 with those independences and 0 = g(3) yields a model of the
    # output by setting the shared latent equal to 0.
    IV3b = Q([(1, 2)], 3, [(0, 1)], [(1, 2)])
    admissible = G2.fritz_entropic_admissible_targets((3,))
    facet_03 = [idx for idx, (kind, members) in G2.effective_DAG_data[1].items() if kind == 'C' and members == {0, 3}]
    assert admissible[0][2] == 'relabel' and admissible[0][0] == set(facet_03)
    assert G2.fritz_admissible_targets((3,)) == {}
    outputs = dict(G2.fritz_entropic_transitions((3,), predictor_modes=('split',), extra_deletions=False))
    out = outputs[(((0, 'replace'),), ('predictor_mode', 'split'), ('certificate', 'relabel'), ('deleted', ()))]
    assert out.directed_structure_instance.edge_list == [(1, 2), (1, 3)]
    assert out.C_simplicial_complex_instance.simplicial_complex_as_sets == {frozenset({0, 3})}
    assert out.Q_simplicial_complex_instance.simplicial_complex_as_sets == {frozenset({1, 2})}
    assert out.condition(3).unique_unlabelled_id == IV3b.unique_unlabelled_id


def test_extra_deletions_keep_the_final_candidate_verified():
    outputs = G1_KPC.fritz_entropic_transitions((3,), predictor_modes=('split',), extra_deletions=True)
    for params, out in outputs:
        info = dict(params[1:])
        assert info['certificate'] in ('dsep', 'markov', 'relabel')
        assert out.number_of_visible >= 3


def test_extra_deletions_never_remove_the_last_shared_facet_of_a_predicted_node():
    # Predicted node 1, predictor 0 sharing the single facet {0,1}: deleting that facet would leave 1 an isolated
    # root while the hypotheses say 1 = f(0) and 1 ⊥ 0, a contradiction the LP would "certify" vacuously.
    g = Q([(1, 2)], 4, [], [(0, 1), (0, 3), (2, 3)])
    for params, out in g.fritz_entropic_transitions((0,), predictor_modes=('split',), extra_deletions=True):
        info = dict(params[1:])
        for p, t in info['deleted']:
            for s_node, mode in params[0]:
                predicted_label = s_node if mode == 'replace' else str(s_node) + '_copy'
                if t == predicted_label:
                    assert not (isinstance(p, str) and p.startswith('L{') and '0' in p), (params,)
