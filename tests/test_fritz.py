"""Tests for the corrected Fritz piggyback and the composition of piggybacks."""
from hypergraphs import Hypergraph
from directed_structures import DirectedStructure
from quantum_mDAG import QmDAG


def Q(edges, n, C, Qf):
    return QmDAG(DirectedStructure(edges, n), Hypergraph(C, n), Hypergraph(Qf, n))


TRIANGLE = Q([], 3, [], [(0, 1), (1, 2), (0, 2)])
SQUARE = Q([], 4, [], [(2, 3), (1, 3), (0, 1), (0, 2)])
BELL6 = Q([], 4, [(1, 3), (0, 2)], [(2, 3)])
IV3 = Q([(1, 2)], 3, [], [(0, 1), (1, 2)])
IV3b = Q([(1, 2)], 3, [(0, 1)], [(1, 2)])
# Counterexample from the review: node 0 shares a classical latent with 3 but also a quantum latent with 1,
# and 1 -> 3 makes that quantum latent d-connected to 3. The old code produced IV3b from it.
G2 = Q([(1, 2), (1, 3)], 4, [(0, 3), (0, 2)], [(0, 1), (1, 2)])


def latent_index(qmdag, kind, facet):
    g, latents = qmdag.effective_DAG_data
    matches = [idx for idx, (k, f) in latents.items() if k == kind and f == frozenset(facet)]
    assert len(matches) == 1
    return matches[0]


def test_effective_DAG_has_one_noise_node_per_visible_node():
    g, latents = TRIANGLE.effective_DAG_data
    kinds = [kind for kind, _ in latents.values()]
    assert kinds.count('noise') == 3 and kinds.count('Q') == 3 and kinds.count('C') == 0
    assert g.number_of_nodes() == 9


def test_triangle_admissibility():
    admissible = TRIANGLE.fritz_admissible_targets((2,))
    assert set(admissible) == {0, 1}
    common, others = admissible[0]
    assert common == {latent_index(TRIANGLE, 'Q', {0, 2})}
    assert others == {latent_index(TRIANGLE, 'Q', {0, 1}), latent_index(TRIANGLE, 'noise', {0})}


def test_triangle_copy_mode_reaches_bell():
    outputs = dict(TRIANGLE.fritz_transitions((2,)))
    bell = outputs[((0, 'copy'), (1, 'copy'))]
    assert bell.unique_unlabelled_id == BELL6.unique_unlabelled_id
    assert bell.restricted_perfect_predictions_numeric == tuple()
    assert len(bell.unique_id) == 5  # (n, ds, C, Q, pp)
    assert not hasattr(bell, 'Fritz_trick_has_been_applied_already')


def test_fritz_outputs_are_deterministic():
    first = TRIANGLE.fritz_transitions((2,))
    second = TRIANGLE.fritz_transitions((2,))
    assert [params for params, _ in first] == [params for params, _ in second]
    assert [g.unique_id for _, g in first] == [g.unique_id for _, g in second]


def test_counterexample_G2_is_inadmissible_and_unreachable():
    for predictor in (2, 3):
        assert G2.fritz_admissible_targets((predictor,)) == {}
    ids = G2.unique_unlabelled_ids_obtainable_by_Fritz_for_QC()
    assert IV3.unique_unlabelled_id not in ids
    assert IV3b.unique_unlabelled_id not in ids


def test_childful_predictor_requires_opt_in():
    chain = Q([(0, 1)], 3, [], [(0, 2), (1, 2)])
    import pytest
    with pytest.raises(AssertionError):
        chain.fritz_admissible_targets((0,), allow_childful_predictors=False)
    with pytest.raises(AssertionError):
        chain.fritz_transitions((0,), allow_childful_predictors=False)
    # With the opt-in, 2 is predicted by 0 through their shared latent; 1 is downstream of 0 and never admissible.
    assert set(chain.fritz_admissible_targets((0,))) == {2}


def test_copy_feeds_only_children_that_still_see_the_original():
    # Facet {0,1,2}, edge 0->1, predictor 2: 1 in replace mode keeps only the facet, so neither 0 nor 0_copy feeds it.
    g = Q([(0, 1)], 3, [], [(0, 1, 2)])
    outputs = dict(g.fritz_transitions((2,)))
    out = outputs[((0, 'copy'), (1, 'replace'))]
    assert out.directed_structure_instance.edge_list == []
    out = outputs[((0, 'copy'), (1, 'copy'))]
    names = out.directed_structure_instance.variable_names
    edges = {(names[a], names[b]) for a, b in out.directed_structure_instance.edge_list}
    assert edges == {(0, 1), ('0_copy', 1)}


def test_private_noise_blocks_prediction_of_a_parent():
    # 0 shares a latent with 2 but also feeds 2 directly: 2 cannot predict 0 (private noise of 0 reaches 2).
    g = Q([(0, 2)], 3, [(0, 2)], [(0, 1)])
    assert 0 not in g.fritz_admissible_targets((2,))


def test_pp_intermediate_keeps_predictor_and_records_restrictions():
    intermediate = TRIANGLE.fritz_intermediate_with_pp((2,), {0: 'copy', 1: 'copy'})
    assert intermediate.number_of_visible == 5
    predicted = {i for i, _ in intermediate.restricted_perfect_predictions_numeric}
    assert len(predicted) == 2
    assert all(preds == (2,) for _, preds in intermediate.restricted_perfect_predictions_numeric)


def test_closure_composes_in_both_directions():
    ids = SQUARE.unique_unlabelled_ids_obtainable_by_Fritz_for_QC()
    assert BELL6.unique_unlabelled_id in ids
    single_pass = {g.unique_unlabelled_id for y in SQUARE.vis_nodes_with_no_children
                   for _, g in SQUARE.fritz_transitions((y,))}
    assert single_pass.issubset(ids)
    assert SQUARE.unique_unlabelled_id not in ids
    assert ids.issuperset(SQUARE.unique_unlabelled_ids_obtainable_by_reduction(districts_check=False,
                                                                                 apply_teleportation=True))


# ---------------------------------------------------------------- stage 3: extensions

LOST_FOUR = [Q([(0, 2), (1, 2), (2, 3)], 4, [], [(0, 1), (0, 2), (1, 3)]),
             Q([(0, 1), (1, 2), (2, 3)], 4, [], [(0, 1), (0, 2), (1, 3)]),
             Q([(0, 2), (1, 2), (2, 3)], 4, [], [(0, 1), (0, 3), (1, 2)]),
             Q([(0, 1), (0, 2), (1, 2), (2, 3)], 4, [], [(0, 1), (0, 2), (1, 3)])]
IV2b = Q([(0, 1), (1, 2)], 3, [(0, 1)], [(1, 2)])


def test_joint_predictors_keep_every_shared_latent():
    # The old code stripped both latents of node 1 here; joint prediction by {0, 2} keeps both.
    admissible = TRIANGLE.fritz_admissible_targets((0, 2))
    common, others = admissible[1]
    assert common == {latent_index(TRIANGLE, 'Q', {0, 1}), latent_index(TRIANGLE, 'Q', {1, 2})}
    assert others == {latent_index(TRIANGLE, 'noise', {1})}


def test_marginalizing_a_childless_predictor_equals_dropping_it():
    predictors = frozenset({3})
    admissible = SQUARE.fritz_admissible_targets(predictors)
    for params, direct in SQUARE.fritz_transitions(predictors):
        kept = SQUARE._fritz_kept_parents(admissible, dict(params))
        intermediate, to_nums = SQUARE._fritz_build(predictors, dict(params), kept, drop_predictors=False)
        to_original = {num: SQUARE._fritz_original_of(name) for name, num in to_nums.items()}
        marginalized, _ = SQUARE._marginalize_predictors(intermediate, to_original, (3,),
                                                         districts_check=False, apply_teleportation=True)
        assert marginalized.unique_unlabelled_id == direct.unique_unlabelled_id


def test_joint_childful_predictors_enumerate_every_removal_order():
    # Teleportation makes marginalization order-dependent; both orders must be produced (found by brute force).
    g = Q([(0, 2), (0, 3), (0, 4), (1, 3), (3, 4)], 5, [(2, 4)], [(0, 1), (1, 4), (0, 3)])
    outputs = [out.unique_unlabelled_id for params, out in g.fritz_transitions((0, 1), max_visible=5)
               if params == ((3, 'copy'),)]
    assert len(outputs) == 2 and len(set(outputs)) == 2


def test_childful_predictor_recovers_the_four_graphs_lost_in_stage_2():
    for g in LOST_FOUR:
        strict = g.unique_unlabelled_ids_obtainable_by_Fritz_for_QC(allow_childful_predictors=False)
        assert IV2b.unique_unlabelled_id not in strict
        assert IV2b.unique_unlabelled_id in g.unique_unlabelled_ids_obtainable_by_Fritz_for_QC()


def test_replace_mode_keeps_quantum_facet_on_other_children():
    # Facet {0,1,2,3}; predictor 3 predicts 0 (replace): 1 and 2 keep a quantum facet, 0 joins them classically.
    g = Q([], 4, [], [(0, 1, 2, 3)])
    outputs = dict(g.fritz_transitions((3,), keep_quantum_facets=True))
    out = outputs[((0, 'replace'),)]
    assert out.Q_simplicial_complex_instance.simplicial_complex_as_sets == {frozenset({1, 2})}
    assert out.C_simplicial_complex_instance.simplicial_complex_as_sets == {frozenset({0, 1, 2})}
    legacy = dict(g.fritz_transitions((3,), keep_quantum_facets=False))[((0, 'replace'),)]
    assert legacy.Q_simplicial_complex_instance.simplicial_complex_as_sets == set()
