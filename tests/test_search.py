"""Tests for the provenance-carrying piggyback search."""
from hypergraphs import Hypergraph
from directed_structures import DirectedStructure
from quantum_mDAG import QmDAG
import pytest

import qc_gap_search as S
from known_QC_gaps import SEEDS, QG_Bell_C_C


def Q(edges, n, C, Qf):
    return QmDAG(DirectedStructure(edges, n), Hypergraph(C, n), Hypergraph(Qf, n))


TRIANGLE = Q([], 3, [], [(0, 1), (1, 2), (0, 2)])
SQUARE = Q([], 4, [], [(2, 3), (1, 3), (0, 1), (0, 2)])
LOST = Q([(0, 2), (1, 2), (2, 3)], 4, [], [(0, 1), (0, 2), (1, 3)])


def test_explorer_expands_each_id_once_and_reachability_is_monotone():
    explorer = S.ClosureExplorer(S.default_tricks(max_visible=5), max_visible=5)
    reached = explorer.expand(SQUARE)
    assert (set(reached) - explorer.lookup_only).issubset(explorer.edges)  # everything reached was expanded (lookups aside)
    root = SQUARE.unique_unlabelled_id
    everything = explorer.reachable(root)
    for name, (group, _) in S.TRICK_GROUPS_FOR_REPORT.items():
        assert explorer.reachable(root, group).issubset(everything)
    assert explorer.reachable(root, frozenset()) == {root}
    n_edges_before = sum(map(len, explorer.edges.values()))
    explorer.expand(SQUARE)
    assert sum(map(len, explorer.edges.values())) == n_edges_before


EDGE_EXAMPLE = Q([(0, 1), (0, 3)], 4, [], [(1, 2), (2, 3)])   # 0 -> 1 is deleted by the prediction of 1 from 2


def test_edge_example_certificate_is_a_single_fritz_step_to_bell():
    seeds = {name: g for name, g in SEEDS.items() if g.number_of_visible == 4}
    report = S.prove_gaps([EDGE_EXAMPLE], seeds, max_visible=5, verbose=False)
    chain = report.proven[EDGE_EXAMPLE.unique_unlabelled_id]
    assert [t.trick for t in chain] == ['Fritz']
    assert report.seed_hit[EDGE_EXAMPLE.unique_unlabelled_id] == 'QG_Bell_C_Edge'
    assert dict(chain[0].params)['deleted'] == ((0,),) and dict(chain[0].params)['certificate'] == 'dsep'
    text = report.certificate(EDGE_EXAMPLE)
    assert 'Fritz' in text and '== known gap' in text
    # The cheapest stage that proves it is unsplit target, split predictor by d-separation (stage 4 of the cascade,
    # after the elementary and the parent-predictor stages).
    assert [count for _, count in report.stage_counts] == [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]


def test_lost_graph_needs_fritz_and_marginalization():
    report = S.prove_gaps([LOST], SEEDS, verbose=False)
    assert LOST.unique_unlabelled_id in report.proven
    tricks_used = [t.trick for t in report.proven[LOST.unique_unlabelled_id]]
    assert 'Fritz' in tricks_used
    assert report.provable_with['PD'] == 0
    # The expensive steps are assessed by the ladder, never by "provable alone".
    assert 'Fritz' not in ' '.join(report.provable_with)
    rungs = S.ladder(report)
    assert rungs[0] == ('elementary', 0, 0)
    assert rungs[-1][1] == 1


def test_certificates_chain_through_intermediate_structures():
    # With the node cap at 4, the five-node input (the square with an exogenous parent 4 of node 0) has no Fritz
    # output small enough: it reaches Bell only through the square, by conditioning on 4 first.
    g = Q([(4, 0)], 5, [], [(2, 3), (1, 3), (0, 1), (0, 2)])
    tricks = {name: trick for name, trick in S.default_tricks(max_visible=4, max_predictors=1).items()
              if name in ('conditioning', 'Fritz')}
    report = S.prove_gaps([g], {'QG_Bell_C_C': QG_Bell_C_C}, tricks=tricks, max_visible=4, verbose=False)
    chain = report.proven[g.unique_unlabelled_id]
    assert [t.trick for t in chain] == ['conditioning', 'Fritz']
    assert chain[0].target != g.unique_unlabelled_id and chain[1].source == chain[0].target
    assert report.seed_hit[g.unique_unlabelled_id] == 'QG_Bell_C_C'
    assert report.certificate(g).count('conditioning') == 1


def test_early_exit_changes_neither_the_stage_counts_nor_the_ladder():
    # Early exit records fewer alternative routes for a proven root; an unproven root still records everything, so
    # the cumulative counts and the ladder (computed from the recorded transitions) are the same.
    inputs = [LOST, EDGE_EXAMPLE, Q([(0, 1), (1, 2)], 4, [], [(0, 2), (1, 3), (2, 3)]), Q([(0, 1), (0, 2), (1, 2)], 4, [], [(2, 3), (1, 3)])]
    seeds = {name: g for name, g in SEEDS.items() if g.number_of_visible == 4}
    lazy = S.prove_gaps(inputs, seeds, verbose=False, with_entropic=False, early_exit=True)
    full = S.prove_gaps(inputs, seeds, verbose=False, with_entropic=False, early_exit=False)
    assert lazy.stage_counts == full.stage_counts and S.ladder(lazy) == S.ladder(full)
    assert set(lazy.proven) == set(full.proven) and lazy.seed_hit == full.seed_hit
    assert sum(map(len, lazy.explorer.edges.values())) <= sum(map(len, full.explorer.edges.values()))


def test_report_counts_are_consistent():
    report = S.prove_gaps([SQUARE, LOST, TRIANGLE], SEEDS, verbose=False)
    counts = report.counts
    assert counts['proven'] + counts['remaining'] == counts['inputs']
    for name in S.TRICK_GROUPS_FOR_REPORT:
        assert 0 <= counts['only via ' + name] <= counts['proven']
        assert counts['with ' + name] <= counts['proven']


def test_extend_and_add_stage_apply_extra_tricks_only_where_asked():
    base = {name: trick for name, trick in S.default_tricks(max_visible=4).items() if name in ('PD', 'conditioning')}
    calls = []

    def fake_expensive(g):
        calls.append(g.unique_unlabelled_id)
        if g.unique_unlabelled_id == SQUARE.unique_unlabelled_id:
            yield (('fake',),), TRIANGLE
    report = S.prove_gaps([LOST], {'QG_Bell_C_C': QG_Bell_C_C}, tricks=base, max_visible=4, verbose=False)
    assert LOST.unique_unlabelled_id not in report.proven
    rescued = S.add_stage(report, {'fake': fake_expensive}, trick_groups=S.TRICK_GROUPS_FOR_REPORT, verbose=False)
    explorer = rescued.explorer
    assert [name for name, _ in rescued.stage_counts] == ['base', 'fake'] and rescued.stage_counts[-1][1] == 0
    assert 'fake:fake' in explorer.applied[LOST.unique_unlabelled_id]   # roots-only applications are remembered per stage
    assert calls == [LOST.unique_unlabelled_id]          # roots only
    assert explorer.base_tricks == frozenset(base)       # extra tricks are not promoted to base tricks
    # A fresh root handed to extend is closed under the base tricks before the extra trick runs.
    fresh = S.ClosureExplorer(dict(base), max_visible=4)
    fresh.extend({'fake': fake_expensive}, [SQUARE])
    assert {'PD', 'conditioning', 'fake:fake'} <= fresh.applied[SQUARE.unique_unlabelled_id]
    assert TRIANGLE.unique_unlabelled_id in fresh.reachable(SQUARE.unique_unlabelled_id, frozenset({'fake'}))


def test_stage_params_are_stated_in_the_labels_of_the_stored_representative():
    # Two labellings of one structure: the first expanded becomes the representative of the shared id; a later stage
    # handed the other labelling must record params that replay on the representative (render_certificate prints it).
    pytest.importorskip("mosek")
    a = Q([(0, 2), (1, 2)], 4, [], [(0, 1), (1, 3), (2, 3)])
    b = Q([(0, 3), (1, 3)], 4, [], [(0, 1), (1, 2), (2, 3)])
    assert a.unique_unlabelled_id == b.unique_unlabelled_id
    extra = S.fritz_tricks(max_visible=5)   # with the LP; the base tricks have Fritz by d-separation only
    explorer = S.ClosureExplorer(S.default_tricks(max_visible=5), max_visible=5)
    explorer.expand(a)
    explorer.extend(extra, [b])
    stage_transitions = [t for t in explorer.edges[a.unique_unlabelled_id] if t.trick in extra]
    assert stage_transitions
    for t in stage_transitions:
        replay = {params: child.unique_unlabelled_id for params, child in extra[t.trick](explorer.representatives[t.source])}
        assert replay.get(t.params) == t.target


# --------------------------------------------------------------------------------------------------
# The search over quantum equivalence classes (equivalence_classes.ClassIndex)
# --------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def index():
    from equivalence_classes import ClassIndex
    return ClassIndex((3, 4))


def test_class_search_keys_identity_and_members(index):
    bell_settings = Q([(0, 1), (0, 3), (1, 2)], 4, [], [(0, 3), (2, 3)])   # G of the five-structure Bell-settings class
    explorer = S.ClosureExplorer(S.elementary_tricks(), max_visible=5, index=index)
    key = explorer.register(bell_settings)
    assert key == index.key(bell_settings) and len(explorer.members(key)) == 5
    # the seeds of that class share the key; the input is proven by identity with no transition
    report = S.prove_gaps([bell_settings], SEEDS, stages=S.default_stages()[:1], verbose=False, index=index)
    assert report.proven[key] == [] and report.seed_hit[key].startswith('QG_Bell_Edge_Edge_Settings')
    assert report.certificate(bell_settings).startswith('identity')
    assert report.counts['with identity'] == 1 and report.counts['structures'] == 5
    assert report.counts['classical classes of inputs'] == 1


def test_class_search_expands_every_member_and_records_the_member(index):
    # The class of the Bell seeds without a settings link has six members; PD on any member gives a three-node
    # structure, and the transition records which member it was applied to and what it produced.
    bell = Q([], 4, [], [(0, 2), (1, 3), (2, 3)])
    explorer = S.ClosureExplorer(S.elementary_tricks(), max_visible=5, index=index)
    key = explorer.register(bell)
    assert len(explorer.members(key)) == 6
    explorer.expand(bell)
    pd = [t for t in explorer.edges[key] if t.trick == 'PD']
    assert pd and {t.via for t in pd} == {m.unique_unlabelled_id for m in explorer.members(key)}
    for t in pd:
        assert t.produced in explorer.structures and explorer.key(explorer.structures[t.produced]) == t.target
    # degradations that lift back into the same class are not recorded as transitions
    assert all(t.target != key for t in explorer.edges[key])


def test_upward_rule_is_a_recorded_lookup(index):
    # A class structurally above a proven class inside the same classical class is proven through an 'upward'
    # transition. We fabricate the situation: the only seed is the lower class's representative.
    import networkx as nx
    C4, Q4 = index.classical[4], index.quantum[4]
    pair = None
    for key in index.input_keys(4):
        steps = index.upward_steps(index.representative(key))
        if steps:
            pair = (key, steps[0][1])
            break
    assert pair is not None
    above_key, below_rep = pair
    report = S.prove_gaps([index.representative(above_key)], {'lower': below_rep}, stages=S.default_stages()[:1],
                          verbose=False, index=index)
    chain = report.proven[above_key]
    assert [t.trick for t in chain] == ['upward'] and report.seed_hit[above_key] == 'lower'
    assert dict(chain[0].params).keys() == {'above', 'below'}
    # the step names the member it was applied to, in the metagraph's canonical labels
    C4 = index.classical[4]
    assert dict(chain[0].params)['above'] == str(C4.dict_uniqind_unlabelled_mDAGs[report.explorer.structures[chain[0].via].as_mDAG.unique_unlabelled_id])
    assert 'upward' in report.certificate(index.representative(above_key))


def test_reverse_fixpoint_equals_the_forward_fixpoint():
    """The provable sets are now computed by one reverse traversal from the seeds per filter; they must equal the
    former per-input forward computation, closed under implication, for every filter the report uses."""
    report = S.prove_gaps([SQUARE, LOST, TRIANGLE, EDGE_EXAMPLE], SEEDS, max_visible=5, verbose=False)
    explorer, input_ids, seed_ids = report.explorer, list(dict.fromkeys(report.input_ids)), set(report.seed_keys)

    def forward(tricks, keep=None):
        proven, unproven = set(seed_ids), set(input_ids)
        reach = {gid: explorer.reachable(gid, tricks, keep) for gid in input_ids}
        changed = True
        while changed:
            changed = False
            for gid in sorted(unproven):
                if not reach[gid].isdisjoint(proven):
                    proven.add(gid); unproven.discard(gid); changed = True
        return proven & set(input_ids)

    all_tricks = frozenset(dict(explorer.tricks)) | {t.trick for ts in explorer.edges.values() for t in ts}
    filters = [(all_tricks, None), (frozenset(), None)]
    filters += [(group, None) for group, _ in S.TRICK_GROUPS_FOR_REPORT.values()]
    filters += [(all_tricks - removed, None) for _, removed in S.TRICK_GROUPS_FOR_REPORT.values()]
    filters += [(None, explorer.stage_filter(k)) for k in range(len(explorer.stage_names))]
    filters += [(None, keep) for _, keep in S.LADDER]
    for tricks, keep in filters:
        assert S._fixpoint(explorer, input_ids, seed_ids, tricks, keep) == forward(tricks, keep)
    for _, keep in S.LADDER:
        assert S._fixpoint_if(report, keep) == forward(None, keep)
    assert len(report.proven) == len(forward(all_tricks)) == 4
