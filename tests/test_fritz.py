"""Tests for the edge-first Fritz piggyback (d-separation certificate) and the composition of piggybacks."""
import pytest

from hypergraphs import Hypergraph
from directed_structures import DirectedStructure
from quantum_mDAG import QmDAG
from known_QC_gaps import QG_Triangle, QG_Bell_C_Edge, QG_Instrumental_C


def Q(edges, n, C, Qf):
    return QmDAG(DirectedStructure(edges, n), Hypergraph(C, n), Hypergraph(Qf, n))


TRIANGLE = Q([], 3, [], [(0, 1), (1, 2), (0, 2)])
TETRAHEDRON = Q([], 4, [], [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
SQUARE = Q([], 4, [], [(2, 3), (1, 3), (0, 1), (0, 2)])
BELL6 = Q([], 4, [(1, 3), (0, 2)], [(2, 3)])
IV3 = Q([(1, 2)], 3, [], [(0, 1), (1, 2)])
IV3b = Q([(1, 2)], 3, [(0, 1)], [(1, 2)])
# Counterexample from the review of the first implementation: node 0 shares a classical latent with 3 but also a
# quantum latent with 1, and 1 -> 3 makes that quantum latent d-connected to 3. The old code produced IV3b from it.
G2 = Q([(1, 2), (1, 3)], 4, [(0, 3), (0, 2)], [(0, 1), (1, 2)])
# The lead's example: deleting the visible edge 0 -> 1 (manuscript 6.5, "deleting a visible edge").
EDGE_EXAMPLE = Q([(0, 1), (0, 3)], 4, [], [(1, 2), (2, 3)])


def latent_index(qmdag, kind, facet):
    g, latents = qmdag.effective_DAG_data
    matches = [idx for idx, (k, f) in latents.items() if k == kind and f == frozenset(facet)]
    assert len(matches) == 1
    return matches[0]


def steps(g, **kwargs):
    """{params: child} of fritz_steps (params are unique per child)."""
    out = dict(g.fritz_steps(use_lp=False, **kwargs))
    return out


def test_effective_DAG_has_one_noise_node_per_visible_node():
    g, latents = TRIANGLE.effective_DAG_data
    kinds = [kind for kind, _ in latents.values()]
    assert kinds.count('noise') == 3 and kinds.count('Q') == 3 and kinds.count('C') == 0
    assert g.number_of_nodes() == 9


# ---------------------------------------------------------------- the five layers

def test_pool_is_latent_siblings_then_visible_parents_ordered_by_shared_facets():
    g = Q([(0, 1), (3, 1)], 4, [], [(1, 2), (1, 3), (2, 3)])
    # Siblings of 1: 2 (one facet) and 3 (one facet); 3 is also a parent. 0 is a parent only.
    assert g.fritz_pool(1) == [2, 3]
    assert g.fritz_pool(1, pool='siblings+parents') == [2, 3, 0]
    g = Q([], 4, [], [(0, 1, 2), (0, 1, 3), (1, 2)])
    assert g.fritz_pool(1) == [0, 2, 3]            # 0 shares two facets with 1, 2 shares two, 3 shares one
    # A node that is neither a sibling nor a parent never appears.
    chain = Q([(0, 1), (1, 2)], 3, [], [])
    assert chain.fritz_pool(2, pool='siblings+parents') == [1]
    assert chain.fritz_pool(0, pool='siblings+parents') == []


def test_descendants_are_excluded_from_the_pool_by_default():
    g = Q([(0, 1), (1, 2)], 4, [], [(0, 2), (0, 3)])
    assert g.fritz_pool(0) == [3]
    assert g.fritz_pool(0, allow_descendants=True) == [2, 3]
    assert g.fritz_pool(2) == [0]
    # The d-separation test always fails for a descendant: the noise of 0 reaches 2 through 0 -> 1 -> 2.
    K, D = g.fritz_deletion(0, {2})
    assert K == {latent_index(g, 'Q', {0, 2})} and D == {latent_index(g, 'Q', {0, 3}), latent_index(g, 'noise', {0})}
    assert g.fritz_certificate(0, K, {2}, use_lp=False) is None


def test_deletion_is_maximal_and_requires_a_channel():
    K, D = TRIANGLE.fritz_deletion(0, {2})
    assert K == {latent_index(TRIANGLE, 'Q', {0, 2})}
    assert D == {latent_index(TRIANGLE, 'Q', {0, 1}), latent_index(TRIANGLE, 'noise', {0})}
    # No channel: 2 shares nothing with 0 and is not its parent.
    g = Q([(0, 1)], 3, [], [(1, 2)])
    assert g.fritz_deletion(0, {2}) is None
    # A joint predictor set sees the union of what its members see (in the triangle, {0, 2} sees every parent of
    # 1 but its noise: a noise-only deletion, None).
    assert TRIANGLE.fritz_deletion(1, {0, 2}) is None
    g = Q([], 4, [], [(0, 1), (1, 2), (1, 3)])
    K, D = g.fritz_deletion(1, {0, 2})
    assert K == {latent_index(g, 'Q', {0, 1}), latent_index(g, 'Q', {1, 2})}
    assert D == {latent_index(g, 'Q', {1, 3}), latent_index(g, 'noise', {1})}


def test_noise_only_deletions_are_not_emitted():
    g = Q([], 3, [], [(0, 1, 2)])
    assert g.fritz_deletion(0, {1}) is None
    assert steps(g) == {} and steps(g, target_mode='split') == {} and steps(g, predictor_mode='split') == {}


def test_parent_predictor_is_certified_vacuously():
    g = Q([(0, 1)], 3, [], [(1, 2)])
    K, D = g.fritz_deletion(1, {0})
    assert K == {0} and latent_index(g, 'Q', {1, 2}) in D
    assert g.fritz_certificate(1, K, {0}, use_lp=False) == 'dsep'


def test_private_noise_blocks_prediction_of_a_parent():
    # 0 shares a latent with 2 but also feeds 2 directly: 2 cannot predict 0 (private noise of 0 reaches 2).
    g = Q([(0, 2)], 3, [(0, 2)], [(0, 1)])
    K, D = g.fritz_deletion(0, {2})
    assert g.fritz_certificate(0, K, {2}, use_lp=False) is None


def test_restrict_target_keeps_the_facet_quantum_for_the_other_members():
    # Facet {0,1,2,3}; 0 restricted to it: 1, 2, 3 keep a quantum facet, 0 joins them classically.
    g = Q([], 4, [], [(0, 1, 2, 3)])
    out = g._restrict_target(0, frozenset({latent_index(g, 'Q', {0, 1, 2, 3})}))
    assert out.Q_simplicial_complex_instance.simplicial_complex_as_sets == {frozenset({1, 2, 3})}
    assert out.C_simplicial_complex_instance.simplicial_complex_as_sets == {frozenset({0, 1, 2, 3})}
    # Redundant sub-facets left by the restriction are cleaned (canonical ids).
    tetra_out = steps(TETRAHEDRON)[(('targets', (0,)), ('target_mode', 'unsplit'), ('deleted', (('Q{0,2,3}',),)), ('predictor', (1,)),
                                    ('predictor_mode', 'unsplit'), ('certificate', 'dsep'))]
    assert tetra_out.unique_unlabelled_id == QG_Triangle.unique_unlabelled_id


def test_realise_marginalizes_in_every_order_and_records_it():
    # Teleportation makes marginalization order-dependent; both orders are produced (found by brute force).
    g = Q([(0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)], 5, [(2, 3)],
          [(0, 2), (0, 3), (1, 2), (1, 4), (2, 4), (3, 4)])
    K, D = g.fritz_deletion(4, {0, 2})
    assert g.fritz_certificate(4, K, {0, 2}, use_lp=False) == 'dsep'
    results = g.fritz_realise({4: K}, {0, 2})
    assert len(results) == 2 and len({c.unique_unlabelled_id for _, c in results}) == 2
    assert {params for params, _ in results} == {(('order', (0, 2)),), (('order', (2, 0)),)}
    # A single removal carries no order parameter.
    K, D = TRIANGLE.fritz_deletion(0, {2})
    assert TRIANGLE.fritz_realise({0: K}, {2}) == [((), TRIANGLE.fritz_realise({0: K}, {2})[0][1])]


# ---------------------------------------------------------------- fritz_steps

def test_tetrahedron_reaches_the_triangle_by_dropping_one_facet():
    out = steps(TETRAHEDRON, max_targets=1)
    assert len(out) == 12 and all(c.unique_unlabelled_id == QG_Triangle.unique_unlabelled_id for c in out.values())
    assert all(dict(p)['certificate'] == 'dsep' and dict(p)['predictor_mode'] == 'unsplit' for p in out)


def test_fritz_steps_are_deterministic():
    first = list(TETRAHEDRON.fritz_steps(use_lp=False))
    second = list(TETRAHEDRON.fritz_steps(use_lp=False))
    assert [params for params, _ in first] == [params for params, _ in second]
    assert [g.unique_id for _, g in first] == [g.unique_id for _, g in second]


def test_deleting_a_visible_edge_is_proven_both_ways():
    # Route 1 (unsplit target, split predictor): 2 predicts 1 through Q{1,2} and deletes 0 -> 1; 2 is childless, so it
    # stays untouched; the output is 0 -> 3; C{1,2}, Q{2,3} = QG_Bell_C_Edge.
    kept = steps(EDGE_EXAMPLE, predictor_mode='split')
    params = (('targets', (1,)), ('target_mode', 'unsplit'), ('deleted', ((0,),)), ('predictor', (2,)), ('predictor_mode', 'split'),
              ('certificate', 'dsep'))
    assert kept[params].unique_unlabelled_id == QG_Bell_C_Edge.unique_unlabelled_id
    assert kept[params].directed_structure_instance.edge_list == [(0, 3)]
    # Route 2 (split target, unsplit predictor): 2 is split, 1 predicts the copy 2' through Q{1,2,2'} and deletes the
    # copy's share of Q{2,2',3}; 1 is childless and is marginalized: 0 -> 3; C{2,2'}, Q{2,3} (relabelled 0 -> 2;
    # C{1,3}, Q{1,2}), the same seed with the roles of the parties exchanged.
    copy = steps(EDGE_EXAMPLE, target_mode='split')
    params = (('targets', (2,)), ('target_mode', 'split'), ('deleted', (("Q{2,3,2'}",),)), ('predictor', (1,)), ('predictor_mode', 'unsplit'),
              ('certificate', 'dsep'))
    assert copy[params].unique_unlabelled_id == QG_Bell_C_Edge.unique_unlabelled_id
    assert copy[params].directed_structure_instance.edge_list == [(0, 2)]
    assert copy[params].C_simplicial_complex_instance.simplicial_complex_as_sets == {frozenset({1, 3})}
    assert copy[params].Q_simplicial_complex_instance.simplicial_complex_as_sets == {frozenset({1, 2})}


def test_parent_predictor_edge_is_re_supplied_by_the_kept_copy():
    # With the parents in the pool, 0 (parent of 1, no shared facet) predicts 1 vacuously: the copy 0' is 1's
    # twin parent, so 0 -> 1 and Q{1,2} are deleted and 1 keeps 0' alone; marginalizing 0' (childful) re-supplies
    # 0 -> 1 as a classical facet C{0,1,3} by teleportation. Nothing new is reached here, but the step is sound.
    out = steps(EDGE_EXAMPLE, predictor_mode='split', pool='siblings+parents')
    params = (('targets', (1,)), ('target_mode', 'unsplit'), ('deleted', ((0, 'Q{1,2}'),)), ('predictor', (0,)), ('predictor_mode', 'split'),
              ('certificate', 'dsep'))
    child = out[params]
    assert child.C_simplicial_complex_instance.simplicial_complex_as_sets == {frozenset({0, 1, 3})}
    assert child.Q_simplicial_complex_instance.simplicial_complex_as_sets == {frozenset({2, 3})}
    assert child.directed_structure_instance.edge_list == [(0, 3)]
    # Without the parents in the pool, 0 is not offered for 1 (it shares no facet with 1).
    assert all(dict(p)['predictor'] != (0,) or 1 not in dict(p)['targets'] for p in steps(EDGE_EXAMPLE, predictor_mode='split'))


def test_counterexample_G2_is_unreachable_by_d_separation():
    for pm in ('unsplit', 'split'):
        for target_mode in ('unsplit', 'split'):
            for params, child in steps(G2, target_mode=target_mode, predictor_mode=pm).items():
                assert child.unique_unlabelled_id not in (IV3.unique_unlabelled_id, IV3b.unique_unlabelled_id), params
    ids = G2.unique_unlabelled_ids_obtainable_by_Fritz_for_QC()
    assert IV3.unique_unlabelled_id not in ids and IV3b.unique_unlabelled_id not in ids


def test_split_target_keeps_the_children_of_the_target():
    # The copy is a faithful duplicate (split_node): it keeps the children of the target as well as its parents, and
    # the restriction then removes the parents the predictor cannot see from the copy only.
    g = Q([(0, 1)], 3, [], [(0, 1, 2)])
    out = steps(g, target_mode='split', predictor_mode='split')
    params = (('targets', (0,)), ('target_mode', 'split'), ('deleted', ((),)), ('predictor', (2,)), ('predictor_mode', 'split'), ('certificate', 'dsep'))
    assert params not in out       # the copy sees the whole facet: noise-only deletion, not emitted
    g = Q([(0, 1)], 3, [], [(0, 2), (1, 2), (0, 1)])
    out = steps(g, target_mode='split', predictor_mode='split')
    params = (('targets', (0,)), ('target_mode', 'split'), ('deleted', (("Q{0,1,0'}",),)), ('predictor', (2,)), ('predictor_mode', 'split'),
              ('certificate', 'dsep'))
    child = out[params]
    assert child.directed_structure_instance.edge_list == [(0, 1), (3, 1)]       # 0 and the copy 0' = 3 both feed 1
    assert child.C_simplicial_complex_instance.simplicial_complex_as_sets == {frozenset({0, 2, 3})}
    assert child.Q_simplicial_complex_instance.simplicial_complex_as_sets == {frozenset({0, 1}), frozenset({0, 2}), frozenset({1, 2})}


def test_split_predictors_are_split_and_marginalized_not_kept_untouched():
    # 0->1 with quantum facets {0,1},{0,2},{1,2} is saturated (latent-free equivalent), so no sound piggyback may
    # turn it into a known gap. Keeping the childful predictor 0 untouched while 0 predicts 2 would give the
    # instrumental gap QG_Instrumental_C: node 1 could read the prediction through the edge 0->1. The sound
    # realisation splits 0 into itself and a full copy, lets the copy predict, and marginalizes the copy, which
    # relays what 1 could learn: the instrument then shares a facet with the outcome as well, and there is no gap.
    g = Q([(0, 1)], 3, [], [(0, 1), (0, 2), (1, 2)])
    out = steps(g, predictor_mode='split')
    params = (('targets', (2,)), ('target_mode', 'unsplit'), ('deleted', (('Q{1,2}',),)), ('predictor', (0,)), ('predictor_mode', 'split'),
              ('certificate', 'dsep'))
    assert out[params].unique_unlabelled_id != QG_Instrumental_C.unique_unlabelled_id
    assert out[params].C_simplicial_complex_instance.simplicial_complex_as_sets == {frozenset({0, 1, 2})}
    assert out[params].Q_simplicial_complex_instance.simplicial_complex_as_sets == {frozenset({0, 1})}
    assert all(c.unique_unlabelled_id != QG_Instrumental_C.unique_unlabelled_id for c in out.values())
    # For a childless predictor, split means untouched: the output is the restricted structure itself.
    kpc = Q([(0, 1), (0, 2), (1, 2)], 4, [], [(2, 3), (1, 3)])
    out = steps(kpc, predictor_mode='split')
    params = (('targets', (1,)), ('target_mode', 'unsplit'), ('deleted', ((0,),)), ('predictor', (3,)), ('predictor_mode', 'split'),
              ('certificate', 'dsep'))
    K, _ = kpc.fritz_deletion(1, {3})
    assert out[params].unique_id == kpc._restrict_target(1, K).unique_id


def test_unsplit_childless_predictor_equals_deleting_it():
    for params, child in steps(SQUARE).items():
        T, X = dict(params)['targets'], dict(params)['predictor']
        kept = {s: SQUARE.fritz_deletion(s, X)[0] for s in T}
        direct = SQUARE._restrict_targets(kept).fix_to_point_distribution_QmDAG(X[0])
        assert child.unique_unlabelled_id == direct.unique_unlabelled_id


def test_joint_predictors_are_available_but_off_by_default():
    assert all(len(dict(p)['predictor']) == 1 for p in steps(TRIANGLE))
    assert all(len(dict(p)['targets']) == 1 for p in steps(TRIANGLE))   # with an unsplit target no two targets of one predictor
    joint = [p for p in steps(TRIANGLE, max_predictors=2) if len(dict(p)['predictor']) == 2]
    assert joint == []            # the joint set sees every facet of the target: noise-only deletion
    g = Q([(0, 1), (0, 4), (2, 4), (3, 4)], 5, [], [(0, 4), (1, 4), (3, 4)])
    joint = [p for p in steps(g, max_predictors=2, max_visible=5) if len(dict(p)['predictor']) == 2]
    assert joint and all(any(k == 'order' for k, _ in p) for p in joint)      # several removals: the order is recorded


# ---------------------------------------------------------------- composition

def test_closure_composes_in_both_directions():
    ids = SQUARE.unique_unlabelled_ids_obtainable_by_Fritz_for_QC()
    assert BELL6.unique_unlabelled_id in ids
    single_pass = {g.unique_unlabelled_id for target_mode in ('unsplit', 'split')
                   for _, g in SQUARE.fritz_steps(target_mode=target_mode, use_lp=False)}
    assert single_pass.issubset(ids)
    assert SQUARE.unique_unlabelled_id not in ids
    assert ids.issuperset(SQUARE.unique_unlabelled_ids_obtainable_by_reduction(districts_check=False,
                                                                                 apply_teleportation=True))


def test_triangle_reaches_bell_in_one_joint_step():
    # Fritz's original argument: 2 predicts copies of 0 and of 1 at once and is then dropped; the copies keep their
    # facet with 2 classically, which after 2 leaves is a classical pair facet each: the Bell structure with
    # classical settings. One joint step with split targets and an unsplit predictor.
    out = steps(TRIANGLE, target_mode='split')
    params = (('targets', (0, 1)), ('target_mode', 'split'), ('deleted', (("Q{0,1,0',1'}",), ("Q{0,1,0',1'}",))), ('predictor', (2,)),
              ('predictor_mode', 'unsplit'), ('certificate', 'dsep'))
    assert out[params].unique_unlabelled_id == BELL6.unique_unlabelled_id
    # Every single-target step of the triangle gives a three-node structure; only the joint steps reach Bell.
    assert all(c.number_of_visible == 3 for p, c in out.items() if len(dict(p)['targets']) == 1)
    assert {c.unique_unlabelled_id for p, c in out.items() if len(dict(p)['targets']) == 2} == {BELL6.unique_unlabelled_id}
    # Without joint target sets (max_targets=1) it would take two split-predictor steps and a point distribution.
    assert BELL6.unique_unlabelled_id not in {c.unique_unlabelled_id for _, c in TRIANGLE.fritz_steps(target_mode='split', use_lp=False, max_targets=1)}
    assert BELL6.unique_unlabelled_id in TRIANGLE.unique_unlabelled_ids_obtainable_by_Fritz_for_QC()


def test_joint_targets_require_every_member_certified_by_d_separation():
    # A target the closure alone certifies is never a member of a joint set: in KPC G1, 3 certifies 1 by
    # d-separation and 2 by the closure only, so the only step with two targets is absent.
    kpc = Q([(0, 1), (0, 2), (1, 2)], 4, [], [(2, 3), (1, 3)])
    assert all(len(dict(p)['targets']) == 1 for p, _ in kpc.fritz_steps(predictor_mode='split', use_lp=True))
    # Every d-separation step (single or joint) is emitted before any closure step, with the target split or not.
    certificates = [dict(p)['certificate'] for p, _ in kpc.fritz_steps(predictor_mode='split')]
    assert set(certificates) == {'dsep', 'semigraphoid'}
    assert certificates == sorted(certificates, key=lambda c: c != 'dsep')
    certificates = [dict(p)['certificate'] for p, _ in kpc.fritz_steps(target_mode='split', predictor_mode='split')]
    assert certificates and certificates == sorted(certificates, key=lambda c: c != 'dsep')


LOST_FOUR = [Q([(0, 2), (1, 2), (2, 3)], 4, [], [(0, 1), (0, 2), (1, 3)]),
             Q([(0, 1), (1, 2), (2, 3)], 4, [], [(0, 1), (0, 2), (1, 3)]),
             Q([(0, 2), (1, 2), (2, 3)], 4, [], [(0, 1), (0, 3), (1, 2)]),
             Q([(0, 1), (0, 2), (1, 2), (2, 3)], 4, [], [(0, 1), (0, 2), (1, 3)])]
IV2b = Q([(0, 1), (1, 2)], 3, [(0, 1)], [(1, 2)])


def test_childful_predictors_recover_the_four_graphs_lost_in_stage_2():
    for g in LOST_FOUR:
        assert IV2b.unique_unlabelled_id in g.unique_unlabelled_ids_obtainable_by_Fritz_for_QC()


def test_unique_id_has_no_prediction_component():
    assert len(TRIANGLE.unique_id) == 4
    with pytest.raises(TypeError):
        QmDAG(TRIANGLE.directed_structure_instance, TRIANGLE.C_simplicial_complex_instance,
              TRIANGLE.Q_simplicial_complex_instance, pp_restrictions=())
