"""Tests for the on-disk cache of proven gaps and its version-based invalidation."""
import json

from hypergraphs import Hypergraph
from directed_structures import DirectedStructure
from quantum_mDAG import QmDAG

import qc_gap_search as S
import gap_cache as C
from known_QC_gaps import SEEDS


def Q(edges, n, Cf, Qf):
    return QmDAG(DirectedStructure(edges, n), Hypergraph(Cf, n), Hypergraph(Qf, n))


LOST = Q([(0, 2), (1, 2), (2, 3)], 4, [], [(0, 1), (0, 2), (1, 3)])
TRIANGLE = Q([], 3, [], [(0, 1), (1, 2), (0, 2)])          # degrades to the seed QG_Triangle (one lookup step)
SQUARE = Q([], 4, [], [(2, 3), (1, 3), (0, 1), (0, 2)])    # conditioning on a node gives the triangle
BELL_SEEDS = {k: g for k, g in SEEDS.items() if g.number_of_visible == 4}
EDGE_EXAMPLE = Q([(0, 1), (0, 3)], 4, [], [(1, 2), (2, 3)])            # one Fritz step (deleting 0 -> 1) to QG_Bell_C_Edge
EDGE_EXAMPLE_PLUS = Q([(0, 1), (0, 3), (4, 0)], 5, [], [(1, 2), (2, 3)])   # dropping or conditioning on 4 gives EDGE_EXAMPLE


def test_round_trip_and_invalidation_by_piggyback_version(tmp_path, monkeypatch):
    path = str(tmp_path / "known_gaps.json")
    report = S.prove_gaps([LOST, SQUARE, TRIANGLE], SEEDS, verbose=False)
    cache = C.GapCache(path)
    added = cache.record(report)
    assert added == 3 and len(cache) == 3
    tri_entry = cache.entries[C.id_key(TRIANGLE.unique_unlabelled_id)]
    assert [step['trick'] for step in tri_entry['chain']] == ['degradation'] and tri_entry['seed'] == 'QG_Triangle'
    cache.save()
    data = json.load(open(path))
    assert data['piggyback_versions'] == S.PIGGYBACK_VERSIONS
    entry = data['entries'][C.id_key(LOST.unique_unlabelled_id)]
    assert entry['id'] == list(LOST.unique_unlabelled_id)
    assert entry['seed'] in SEEDS
    assert entry['chain'] and all(set(step) == {'trick', 'params', 'source', 'target'} for step in entry['chain'])
    assert set(entry['versions']) == {step['trick'] for step in entry['chain']}
    assert 'Fritz' in entry['versions']
    assert entry['structure']['edges'] == [[0, 2], [1, 2], [2, 3]]
    assert '== known gap' in entry['certificate']

    # Reloading keeps both entries; they act as known gaps.
    loaded = C.GapCache.load(path)
    assert loaded.valid_ids() == {LOST.unique_unlabelled_id, SQUARE.unique_unlabelled_id, TRIANGLE.unique_unlabelled_id} and loaded.dropped == 0
    assert set(loaded.known().values()) == loaded.valid_ids()
    # Recording the same report again adds nothing.
    assert loaded.record(report) == 0

    # Bumping the version of a piggyback the proofs relied on drops exactly those entries.
    tricks_lost = {step['trick'] for step in entry['chain']}
    tricks_tri = {step['trick'] for step in data['entries'][C.id_key(SQUARE.unique_unlabelled_id)]['chain']}
    only_lost = tricks_lost - tricks_tri
    assert only_lost, "the test needs a trick used by LOST's proof but not by the square's"
    bumped = dict(S.PIGGYBACK_VERSIONS)
    bumped[next(iter(only_lost))] += 1
    monkeypatch.setattr(C, 'PIGGYBACK_VERSIONS', bumped)
    reloaded = C.GapCache.load(path)
    assert reloaded.valid_ids() == {SQUARE.unique_unlabelled_id, TRIANGLE.unique_unlabelled_id} and reloaded.dropped == 1
    # An unknown trick (removed from the versions table) also invalidates.
    for trick in tricks_tri | {'degradation'}:
        bumped.pop(trick, None)
    assert C.GapCache.load(path).valid_ids() == set()
    monkeypatch.setattr(C, 'PIGGYBACK_VERSIONS', S.PIGGYBACK_VERSIONS)
    # So does a seed that is no longer in the seed list.
    assert C.GapCache.load(path, seeds=SEEDS).valid_ids() == loaded.valid_ids()
    without = [name for name in SEEDS if name != entry['seed']]
    assert C.GapCache.load(path, seeds=without).valid_ids() == loaded.valid_ids() - {LOST.unique_unlabelled_id}


def test_cached_gaps_act_as_known_and_dependent_entries_inherit_versions(tmp_path):
    path = str(tmp_path / "known_gaps.json")
    # First run: prove the edge example from the Bell seeds (one Fritz step) and cache it.
    first = S.prove_gaps([EDGE_EXAMPLE], BELL_SEEDS, verbose=False)
    assert [t.trick for t in first.proven[EDGE_EXAMPLE.unique_unlabelled_id]] == ['Fritz']
    cache = C.GapCache(path)
    assert cache.record(first) == 1
    cache.save()
    # Second run: the five-node structure with no seeds at all and the cached edge example as the only known gap. Its
    # certificate ends at the cache entry, and the stored entry inherits the edge example's piggyback versions and
    # ultimate seed.
    loaded = C.GapCache.load(path)
    assert loaded.record(first) == 0
    second = S.prove_gaps([EDGE_EXAMPLE_PLUS], {}, verbose=False, known=loaded.known())
    assert EDGE_EXAMPLE_PLUS.unique_unlabelled_id in second.proven
    assert second.seed_hit[EDGE_EXAMPLE_PLUS.unique_unlabelled_id] == 'cache:' + C.id_key(EDGE_EXAMPLE.unique_unlabelled_id)
    assert loaded.record(second) == 1
    entry = loaded.entries[C.id_key(EDGE_EXAMPLE_PLUS.unique_unlabelled_id)]
    upstream = loaded.entries[C.id_key(EDGE_EXAMPLE.unique_unlabelled_id)]
    assert entry['via_cache'] == C.id_key(EDGE_EXAMPLE.unique_unlabelled_id)
    assert entry['seed'] == upstream['seed'] == 'QG_Bell_C_Edge'
    assert 'Fritz' in upstream['versions'] and set(upstream['versions']).issubset(entry['versions'])
    own_step = second.proven[EDGE_EXAMPLE_PLUS.unique_unlabelled_id][0].trick
    assert own_step in ('PD', 'conditioning') and own_step in entry['versions']
    loaded.save()
    # Dropping the upstream entry from the file drops the dependent one too.
    data = json.load(open(path))
    del data['entries'][C.id_key(EDGE_EXAMPLE.unique_unlabelled_id)]
    json.dump(data, open(path, 'w'))
    assert C.GapCache.load(path).valid_ids() == set()


def test_proven_structure_ids_include_cached_known_gaps():
    known = {'cache:x': TRIANGLE.unique_unlabelled_id}
    report = S.prove_gaps([SQUARE], {}, verbose=False, known=known)
    assert SQUARE.unique_unlabelled_id in report.proven
    assert {SQUARE.unique_unlabelled_id, TRIANGLE.unique_unlabelled_id}.issubset(report.proven_structure_ids())
