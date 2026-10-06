"""Tests for the provenance-carrying piggyback search."""
from hypergraphs import Hypergraph
from directed_structures import DirectedStructure
from quantum_mDAG import QmDAG
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
    assert set(reached).issubset(explorer.edges)  # everything reached was expanded
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
    assert report.provable_with['Fritz (+ marginalization)'] == 1
    assert report.provable_with['PD'] == 0
    assert report.only_via['Fritz (+ marginalization)'] == 1


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
