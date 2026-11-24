"""Reproduce the appendix comparison, starting each census from the named seeds.

Run with the repository's Python environment. An optional --screen-cache DIR
reuses the structural five-node screen across runs, never a database of gaps.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'Special Applications')]

from equivalence_classes import ClassIndex
from growth_census import Phase, five_node_screen
from open_classes_analysis import recorded_graph
from proving_QC_Gaps import run_search
from qc_gap_search import _reverse_reachable


class WithoutUpward(ClassIndex):
    """Remove exactly one rule before the explorer records any transitions."""
    def lookup_tricks(self):
        lookups = super().lookup_tricks()
        del lookups['upward']
        return lookups


def compare_hierarchy(report):
    record = recorded_graph(report)
    open_keys = set(record.open_keys)
    direct, composed, beneficiaries = set(), set(), defaultdict(set)
    bounded = getattr(report, 'bounded', None)
    for source in open_keys:
        reached = record.reachable_from(source)
        direct.update((goal, source) for goal in reached & (open_keys - {source}))
        composed.update((goal, source) for goal in record.composed_from(source) & (open_keys - {source}))
        if bounded:
            for goal in reached:
                if not report.screening.verdict(record.structure(goal)).startswith('gap-free'):
                    beneficiaries[goal].add(source)

    # Reverse the search once for each hypothetical new seed.
    for goal in open_keys:
        if bounded:
            reverse = {key for key, phase in bounded.to_goals({goal}) if phase == Phase.AVAILABLE}
        else:
            reverse = _reverse_reachable(report.explorer, {goal})
        assert reverse & (open_keys - {goal}) == {source for target, source in direct if target == goal}
    hierarchy = {(record.number[goal], record.number[source]) for goal, source in direct}
    assert hierarchy == {(1, 2), (1, 3), (1, 6), (1, 7)}
    assert direct == composed
    return direct, beneficiaries


def check_pair(cap, five_ids):
    reports = []
    for index_type in (ClassIndex, WithoutUpward):
        report, _ = run_search(index=index_type((3, 4)), use_cache=False, deep=cap is not None,
                               max_visible=cap, five_ids=five_ids, verbose=False)
        assert ('upward' in report.explorer.lookup_tricks) == (index_type is ClassIndex)
        reports.append(report)
    enabled, disabled = reports
    assert enabled.counts == disabled.counts
    assert enabled.stage_counts == disabled.stage_counts
    assert set(enabled.proven) == set(disabled.proven)
    assert enabled.proven_structure_ids() == disabled.proven_structure_ids()
    assert enabled.counts['structures proven'] == 935
    assert len(enabled.proven) == 380 and len(enabled.remaining) == 46
    database = len(enabled.proven_structure_ids())
    assert database == {None: 401, 5: 434, 6: 477}[cap]
    hierarchy, beneficiaries = compare_hierarchy(enabled)
    assert (hierarchy, beneficiaries) == compare_hierarchy(disabled)
    if cap is None:
        assert enabled.counts['with node_stitching'] == 12
        assert enabled.counts['only via node_stitching'] == 0
        selected = sum(any(t.trick == 'upward' for t in path) for path in enabled.proven.values())
        moves = {t for edges in enabled.explorer.edges.values() for t in edges if t.trick == 'upward'}
        assert selected == 45 and len(moves) == 156
        step = enabled.proven[4, 2, 0, 3255][0]
        assert step.trick == 'upward' and step.target == (4, 0, 0, 1755)
        print(f'Depth one: {database} database keys, {selected} selected proofs use one of '
              f'{len(moves)} recorded upward transitions.')
    else:
        shared = sum(len(sources) > 1 for sources in beneficiaries.values())
        assert (len(beneficiaries), shared) == {5: (1499, 541), 6: (1608, 609)}[cap]
        print(f'Cap {cap}: {database} database keys, {len(beneficiaries)} candidate targets, '
              f'{shared} shared. All target-to-input assignments agree.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen-cache', type=Path)
    args = parser.parse_args()
    check_pair(None, None)
    five_ids = five_node_screen(args.screen_cache)
    for cap in (5, 6):
        check_pair(cap, five_ids)
