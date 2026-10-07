"""Tests for the provenance-carrying piggyback search."""
from hypergraphs import Hypergraph
from directed_structures import DirectedStructure
from quantum_mDAG import QmDAG
import pytest

import qc_gap_search as S
from known_QC_gaps import SEEDS, QG_Bell6


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


def test_triangle_certificate_is_a_single_fritz_step_to_bell():
    seeds = {name: g for name, g in SEEDS.items() if g.number_of_visible == 4}
    report = S.prove_gaps([TRIANGLE], seeds, max_visible=4, verbose=False)
    chain = report.proven[TRIANGLE.unique_unlabelled_id]
    assert [t.trick for t in chain] == ['Fritz']
    assert report.seed_hit[TRIANGLE.unique_unlabelled_id].startswith('QG_Bell')
    text = report.certificate(TRIANGLE)
    assert 'Fritz' in text and '== known gap' in text


def test_lost_graph_needs_fritz_and_marginalization():
    report = S.prove_gaps([LOST], SEEDS, verbose=False)
    assert LOST.unique_unlabelled_id in report.proven
    tricks_used = [t.trick for t in report.proven[LOST.unique_unlabelled_id]]
    assert 'Fritz' in tricks_used
    assert report.provable_with['PD'] == 0
    # The expensive steps are assessed by the ladder and the step categories, never by "provable alone".
    assert 'Fritz' not in ' '.join(report.provable_with)
    assert S.fritz_breakdown(report)['all Fritz steps'] == 1
    rungs = S.ladder(report)
    assert rungs[0] == ('elementary', 0, 0)
    assert rungs[-1][1] == 1


def test_certificates_chain_through_intermediate_structures():
    # With the node cap at 4, single predictors and no marginalization, the square reaches Bell6 only through the
    # triangle: conditioning on a node of the square gives the triangle, and one Fritz step gives Bell6.
    tricks = {name: trick for name, trick in S.default_tricks(max_visible=4, max_predictors=1).items()
              if name in ('PD', 'conditioning', 'Fritz')}
    report = S.prove_gaps([SQUARE], {'QG_Bell6': QG_Bell6}, tricks=tricks, max_visible=4, verbose=False)
    chain = report.proven[SQUARE.unique_unlabelled_id]
    assert [t.trick for t in chain] == ['conditioning', 'Fritz']
    assert report.seed_hit[SQUARE.unique_unlabelled_id] == 'QG_Bell6'
    assert report.certificate(SQUARE).count('conditioning') == 1


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
    report = S.prove_gaps([LOST], {'QG_Bell6': QG_Bell6}, tricks=base, max_visible=4, verbose=False)
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
