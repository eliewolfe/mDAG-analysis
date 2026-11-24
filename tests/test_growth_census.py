from types import SimpleNamespace

import pytest

from growth_census import BoundedPaths, Phase, advance
from qc_gap_search import Transition


def test_growth_budget_and_composed_implications():
    # A joint step takes four directly to six. A five-node horizontal edge after
    # the reduction distinguishes the two resume thresholds. Two routes arrive
    # at b with different budgets; only the one with unused growth can reach g.
    a, b, c, d = [(4, i, 0, 0) for i in range(4)]
    u, v, w = [(5, i, 0, 0) for i in range(3)]
    z = (6, 0, 0, 0)
    def edge(x, y, trick='Fritz'):
        return Transition(trick, (), x, y)
    edges = {a: [edge(a, z)], z: [edge(z, u, 'PD')],
             u: [edge(u, v)], v: [edge(v, b)], b: [edge(b, w)], w: [edge(w, c)]}
    paths = BoundedPaths(SimpleNamespace(edges=edges), 5)
    assert b in paths.forward(a) and c not in paths.forward(a)
    assert c in paths.forward(b)
    assert paths.forward(a, {z}) == [edges[a][0]]
    assert b not in BoundedPaths(SimpleNamespace(edges=edges), 4).forward(a)
    edges[a].append(edge(a, d))
    edges[d] = [edge(d, b)]
    paths = BoundedPaths(SimpleNamespace(edges=edges), 5)
    assert len(paths.forward(a, {c})) == 4
    routes = paths.to_goals({c})
    for k in edges:
        assert ((k, Phase.AVAILABLE) in routes) == (c in paths.forward(k))
    assert advance(Phase.MUST_REDUCE, edge(u, v), 5) is None
    assert advance(Phase.MUST_REDUCE, edge(u, v, 'degradation'), 5) == Phase.MUST_REDUCE


def test_output_bounds_preserve_joint_steps():
    from test_fritz import TRIANGLE
    options = dict(predictor_mode='unsplit', target_mode='split', use_lp=False, max_visible=4)
    full = {(p, g.unique_unlabelled_id) for p, g in TRIANGLE.fritz_steps(**options)}
    joint = {(p, gid) for p, gid in full if gid[0] == 4}
    assert joint
    assert joint == {(p, g.unique_unlabelled_id) for p, g in TRIANGLE.fritz_steps(**options, min_visible=4)}


def test_stage_added_after_bounded_census_recognizes_only_admissible_paths():
    import qc_gap_search as S
    from equivalence_classes import ClassIndex
    from growth_census import extend_census
    from test_fritz import Q

    root = Q([], 4, [], [])
    large_seed = Q([], 5, [], [])
    small_seed = Q([], 3, [], [])
    report = S.prove_gaps([root], {'large': large_seed, 'small': small_seed},
                          tricks={}, index=ClassIndex(()), verbose=False)
    report = extend_census(report, 5, 4, five_ids=None, screen=False, with_entropic=False,
                           with_split_predictors=False, allow_descendants=False, engine=None, verbose=False)
    assert report.remaining == [root]

    def later_step(g):
        yield (), large_seed   # only Fritz can grow, so this seed cannot justify early exit
        yield (), small_seed

    groups = {'later': ({'later'}, {'later'}), 'unrelated': (set(), {'unrelated'})}
    report = S.add_stage(report, {'later': later_step}, trick_groups=groups, verbose=False)
    key = report.key(root)
    assert report.seed_hit[key] == 'small'
    assert report.proven[key] == report.bounded.forward(key, set(report.seed_keys))
    assert report.stage_counts[-2:] == [('screened Fritz closure with one growth', 0), ('later', 1)]
    assert report.provable_with['later'] == report.only_via['later'] == 1
    assert report.only_via['unrelated'] == 0


def replay_gap_witnesses(report):
    """Re-run the surgeries named by all bounded input and intermediate certificates."""
    from functools import lru_cache
    import qc_gap_search as S
    explorer = report.explorer
    elementary = S.elementary_tricks()
    routes = report.bounded.to_goals(report.seed_keys)

    @lru_cache(None)
    def outputs(via, trick, certificate, tm, pm):
        g = explorer.structures[via]
        if trick == 'Fritz':
            children = g.fritz_steps(target_mode=tm, predictor_mode=pm, engine='semigraphoid',
                                     pool='parents' if certificate == 'parent' else 'siblings',
                                     max_visible=explorer.max_visible, use_lp=certificate == 'semigraphoid')
        else:
            children = {**elementary, **explorer.lookup_tricks}[trick](g)
        return {(params, child.unique_unlabelled_id) for params, child in children}

    verified = set()
    for key in report.proven_structure_ids():
        state = key, Phase.AVAILABLE
        while routes[state] is not None:
            edge, nxt = routes[state]
            assert nxt == (edge.target, advance(state[1], edge, report.bounded.resume_at))
            assert edge.source == state[0]
            if edge not in verified:
                opts = dict(edge.params)
                assert (edge.params, edge.produced) in outputs(edge.via, edge.trick, opts.get('certificate'),
                                                              opts.get('target_mode'), opts.get('predictor_mode'))
                assert explorer.key(explorer.structures[edge.produced]) == edge.target
                verified.add(edge)
            state = nxt
        assert state[0] in report.seed_keys
    return len(verified)


@pytest.mark.slow
def test_deeper_census_proofs_and_hierarchy(proving_QC_Gaps, tmp_path):
    """The production entry point, cold classical screen, then the same census with a warm cache."""
    from test_fritz import Q
    from growth_census import five_node_screen
    import qc_gap_search as S
    index = proving_QC_Gaps.build_index(verbose=False)
    path = tmp_path / 'known.json'
    report, _ = proving_QC_Gaps.run_search(index, cache_path=str(path), verbose=False)
    assert len(report.proven) == 380
    assert len(report.remaining) == 46
    assert report.expanded_states[6, 'MUST_REDUCE'] > 0 and report.expanded_states[5, 'SPENT'] > 0
    assert len(report.proven_structure_ids()) == 477
    assert replay_gap_witnesses(report) >= 370
    legacy, _ = proving_QC_Gaps.run_search(index, deep=False, use_cache=False, verbose=False)
    assert set(report.proven) == set(legacy.proven)
    assert legacy.proven_structure_ids() <= report.proven_structure_ids()
    additional = [
        Q([], 4, [(0, 2), (1, 2, 3)], [(0, 3)]),
        Q([], 4, [(0, 2, 3), (1, 2, 3)], [(0, 1)]),
        Q([], 4, [(0, 1, 3), (0, 2, 3)], [(1, 2, 3)]),
        Q([], 4, [(0, 1), (0, 2), (1, 2, 3)], [(1, 2)]),
        Q([], 4, [(0, 1, 3), (0, 2), (1, 2, 3)], [(1, 2)]),
        Q([(0, 1), (2, 0), (3, 1)], 4, [(0, 1, 3), (0, 2, 3)], [(0, 1)])]
    assert set(map(report.key, additional)) <= report.proven_structure_ids() - legacy.proven_structure_ids()
    assert len(five_node_screen(tmp_path)) == 708635

    warm, _ = proving_QC_Gaps.run_search(index, cache_path=str(path), verbose=False)
    numbers = report.original_open_numbers
    reached = {k: report.bounded.forward(k) for k in numbers}
    assert {k: warm.bounded.forward(k) for k in numbers} == reached
    targets = set().union(*reached.values())
    assert len(targets) == 15619
    candidates = {key for key in targets
                  if not report.screening.verdict(report.explorer.representatives[key]).startswith('gap-free')}
    assert len(candidates) == 1608
    from collections import Counter
    payoff = Counter(tuple(sorted(numbers[k] for k in numbers if target in reached[k]))
                     for target in candidates)
    assert {group: count for group, count in payoff.items() if len(group) > 1} == {
        (1, 2): 321, (1, 2, 3): 11, (1, 2, 6): 8, (1, 2, 7): 44, (1, 2, 3, 6, 7): 219, (4, 5): 6}
    assert warm.proven == report.proven
    assert {(numbers[k], numbers[t]) for k in numbers for t in (reached[k] & numbers.keys()) - {k}} == {
        (2, 1), (3, 1), (6, 1), (7, 1)}
    goal = next(k for k in numbers if numbers[k] == 1)
    hypothesis = S.build_report(report.explorer, report.inputs, report.seeds, {}, known={'class 1': goal})
    assert {numbers[k] for k in hypothesis.proven.keys() - report.proven.keys()} == {1, 2, 3, 6, 7}


def test_shared_source_saturation_against_classical_metagraphs():
    from growth_census import saturated_by_shared_source
    from equivalence_classes import ClassIndex
    from quantum_mDAG import upgrade_to_QmDAG
    from test_fritz import Q
    for graph in ClassIndex((3, 4)).classical.values():
        for uid, md in graph.dict_uniqind_unlabelled_mDAGs.items():
            if saturated_by_shared_source(upgrade_to_QmDAG(md)):
                assert uid in graph.latent_free_equivalent_ids
    broadcast = [(0, k) for k in range(1, 6)]
    assert saturated_by_shared_source(Q(broadcast, 6, [(1, 2, 3, 4, 5)], [(0, 2), (0, 3)]))
    assert not saturated_by_shared_source(Q(broadcast[:-1], 6, [(1, 2, 3, 4, 5)], [(0, 2), (0, 3)]))
