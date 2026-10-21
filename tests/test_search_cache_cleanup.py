"""Manual cleanup releases only the cache groups explicitly selected by the caller."""
import numpy as np

import hypergraphs as H
import quantum_mDAG as QM
import qc_gap_search as S
from directed_structures import DirectedStructure
from known_QC_gaps import SEEDS


class SolverTask:
    def __init__(self):
        self.closed = False

    def close(self):
        assert not self.closed
        self.closed = True


def isolate_search_caches(monkeypatch):
    for name in ('_UNLABELLED_ID_MEMO', '_PIGGYBACK_CHILDREN_MEMO', '_ENTROPIC_LP_CACHE', '_SEMIGRAPHOID_CACHE'):
        monkeypatch.setattr(QM, name, {})


def test_cleanup_is_selective_closes_tasks_and_keeps_permutations(monkeypatch):
    isolate_search_caches(monkeypatch)
    monkeypatch.setattr(H, '_INTEGER_PERMUTATIONS_CACHE', {})
    uid = (3, 0, 0, 0)
    child = object()
    model = np.zeros((2, 2), dtype=bool)
    tasks = [SolverTask(), SolverTask()]
    QM._UNLABELLED_ID_MEMO[uid] = uid
    QM._PIGGYBACK_CHILDREN_MEMO[uid] = [child]
    QM._SEMIGRAPHOID_CACHE[uid] = model
    QM._ENTROPIC_LP_CACHE.update(enumerate(tasks))
    parts = H.Hypergraph([(0, 1)], 3).as_integer_permutations
    QM.clear_search_caches(canonical_ids=False, piggyback_children=False)
    assert all(task.closed for task in tasks)
    assert not QM._ENTROPIC_LP_CACHE and not QM._SEMIGRAPHOID_CACHE
    assert QM._UNLABELLED_ID_MEMO == {uid: uid}
    assert QM._PIGGYBACK_CHILDREN_MEMO[uid][0] is child
    assert H.Hypergraph([(0, 1)], 3).as_integer_permutations is parts
    QM._SEMIGRAPHOID_CACHE[uid] = model
    QM.clear_search_caches(lp_models=False, semigraphoid_models=False)
    assert not QM._UNLABELLED_ID_MEMO and not QM._PIGGYBACK_CHILDREN_MEMO
    assert QM._SEMIGRAPHOID_CACHE[uid] is model
    QM.clear_search_caches()
    QM.clear_search_caches()     # idempotent, with no double-close


def test_completed_report_survives_manual_cleanup_and_new_search_agrees(monkeypatch):
    isolate_search_caches(monkeypatch)
    square = QM.QmDAG(DirectedStructure([], 4), H.Hypergraph([], 4),
                     H.Hypergraph([(0, 1), (0, 2), (1, 3), (2, 3)], 4))
    report = S.prove_gaps([square], SEEDS, tricks=S.elementary_tricks(), verbose=False)
    assert report.proven
    before = (report.counts, dict(report.proven), dict(report.seed_hit), report.certificate(square), S.ladder(report))
    QM.clear_search_caches()
    assert (report.counts, report.proven, report.seed_hit, report.certificate(square), S.ladder(report)) == before
    fresh = QM.QmDAG(DirectedStructure([], 4), H.Hypergraph([], 4), H.Hypergraph(square.Q_simplicial_complex_instance.as_tuples, 4))
    again = S.prove_gaps([fresh], SEEDS, tricks=S.elementary_tricks(), verbose=False)
    assert (again.counts, again.proven, again.seed_hit, again.certificate(fresh), S.ladder(again)) == before
