"""
Worked examples for the manuscript (manuscript/piggybacks.md, Section 8): one certificate per category.

Runs the two-phase census of proving_QC_Gaps (cache disabled, so every certificate ends at a named seed) and prints,
for every elementary piggyback, inputs of phase 1 that are lost without it; for every category of expensive step
(STEP_CATEGORIES of qc_gap_search), inputs of phase 2 that are lost without it, with the shortest certificate each;
the proven-structure database by node count; and the still-unproven inputs.

Usage: python "Special Applications/census_breakdowns.py" [--no-entropic] [--only examples|lpclosure]
"""
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Set

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from known_QC_gaps import SEEDS  # noqa: E402
from qc_gap_search import (ClosureExplorer, Transition, GapReport, entropic_tricks, add_stage, render_certificate,  # noqa: E402
                           STEP_CATEGORIES, _fixpoint_if, is_fritz_type)
from proving_QC_Gaps import run_search, print_cheap_report, print_report  # noqa: E402

Predicate = Callable[[Transition], bool]


def _goals(report: GapReport) -> Dict:
    goals = {g.unique_unlabelled_id: name for name, g in report.seeds.items()}
    goals.update({gid: name for name, gid in report.known.items()})
    return goals


def _path(explorer: ClosureExplorer, root, goals: Set, keep: Predicate) -> Optional[List[Transition]]:
    """Shortest chain of transitions satisfying `keep` from root to a goal."""
    from collections import deque
    if root in goals:
        return []
    parent: Dict = dict()
    frontier = deque([root])
    seen = {root}
    while frontier:
        current = frontier.popleft()
        for t in explorer.edges.get(current, ()):
            if not keep(t) or t.target in seen:
                continue
            seen.add(t.target)
            parent[t.target] = t
            if t.target in goals:
                chain = []
                node = t.target
                while node != root:
                    chain.append(parent[node])
                    node = parent[node].source
                return list(reversed(chain))
            frontier.append(t.target)
    return None


def full_certificate(report: GapReport, gid, keep: Predicate, proven: Optional[Set] = None) -> Optional[List[Transition]]:
    """Shortest chain under `keep` to a seed or to another proven input, chained down to a seed. `proven` is the
    set of inputs provable under `keep` (computed if not given)."""
    goals = _goals(report)
    if proven is None:
        proven = _fixpoint_if(report, keep)

    def rec(node, excluded: Set):
        known = set(goals) | (proven - excluded)
        chain = _path(report.explorer, node, known, keep)
        if chain is None:
            return None
        last = chain[-1].target
        if last in goals:
            return chain
        rest = rec(last, excluded | {node})
        return None if rest is None else chain + rest
    return rec(gid, {gid})


def show_examples(report: GapReport, lost: Set, keep: Predicate, title: str, limit: int = 3) -> None:
    goals = _goals(report)
    print(f"\n==== {title}: {len(lost)} inputs lost")
    proven = _fixpoint_if(report, keep)
    chains = []
    for gid in sorted(lost):
        chain = full_certificate(report, gid, keep, proven)
        if chain is not None:
            chains.append((len(chain), report.explorer.representatives[gid].number_of_visible, gid, chain))
    chains.sort(key=lambda x: (x[0], x[1], x[2]))
    for _, _, gid, chain in chains[:limit]:
        print("--- input (unlabelled id %s):" % (gid,))
        print("    " + report.explorer.representatives[gid].as_string.replace("\n", "\n    ").rstrip())
        print(render_certificate(report.explorer, chain, goals[chain[-1].target]))


def examples(with_entropic: bool = True) -> None:
    t0 = time.time()
    cheap, report, _ = run_search(verbose=False, with_entropic=with_entropic, use_cache=False)
    print(f"[census in {time.time() - t0:.0f}s]")
    print_cheap_report(cheap)
    print_report(report)
    everything = lambda t: True  # noqa: E731

    # Phase 1: elementary piggybacks, three-node seeds. The explorer is shared with phase 2, so the Fritz-type
    # transitions recorded later have to be excluded here.
    elementary = lambda t: not is_fritz_type(t)  # noqa: E731
    proven_cheap = _fixpoint_if(cheap, elementary)
    assert len(proven_cheap) == cheap.counts['proven']
    for trick in ('PD', 'conditioning', 'interruption', 'naive_marginalization', 'teleportation_marginalization'):
        lost = proven_cheap - _fixpoint_if(cheap, lambda t, trick=trick: elementary(t) and t.trick != trick)
        assert len(lost) == cheap.only_via[trick]
        show_examples(cheap, lost, elementary, f"[phase 1] only via {trick}", limit=3)
    lost = proven_cheap - _fixpoint_if(cheap, lambda t: elementary(t) and t.trick not in ('naive_marginalization', 'teleportation_marginalization'))
    show_examples(cheap, lost, elementary, "[phase 1] only via marginalization (either kind)", limit=4)

    # Phase 2: categories of expensive step.
    proven_all = _fixpoint_if(report, everything)
    for name, pred in STEP_CATEGORIES.items():
        lost = proven_all - _fixpoint_if(report, lambda t, pred=pred: not pred(t))
        show_examples(report, lost, everything, f"[phase 2] lost without: {name}", limit=3)

    known = report.proven_structure_ids()
    reps_known = [report.explorer.representatives[i] for i in known if i in report.explorer.representatives]
    hybrid = [g for g in reps_known if g.C_simplicial_complex_instance.simplicial_complex_as_sets]
    by_n: Dict[int, int] = {}
    for g in reps_known:
        by_n[g.number_of_visible] = by_n.get(g.number_of_visible, 0) + 1
    print("\nproven structures (reach a seed), all sizes:", len(reps_known), "by visible-node count", dict(sorted(by_n.items())),
          "; with at least one classical facet:", len(hybrid))
    print("\nremaining unproven inputs:", len(report.remaining))
    for g in sorted(report.remaining, key=lambda g: g.unique_unlabelled_id):
        print("    " + g.as_string.replace("\n", "\n    ").rstrip())


def lp_closure() -> None:
    """Does applying the LP-certified trick to every reachable structure (not just to the unproven inputs) prove
    more? Every structure proven so far is already a known gap for the others (reachability is transitive), so
    this tests LP steps at depth greater than one."""
    t0 = time.time()
    _, report, _ = run_search(verbose=False, use_cache=False)
    print("staged search:", report.stage_counts, f"({time.time()-t0:.0f}s)")
    before = set(report.proven)
    report = add_stage(report, entropic_tricks(max_visible=5, max_predictors=1), roots_only=False, name='Fritz_entropic_closure', verbose=False)
    print("after LP closure over everything reachable:", report.stage_counts, f"({time.time()-t0:.0f}s); structures {len(report.explorer.edges)}")
    new = set(report.proven) - before
    print("newly proven inputs:", len(new))
    for gid in sorted(new):
        print("    " + report.explorer.representatives[gid].as_string.replace("\n", "\n    ").rstrip())
        print(report.certificate(report.explorer.representatives[gid]))


if __name__ == '__main__':
    with_entropic = '--no-entropic' not in sys.argv
    only = sys.argv[sys.argv.index('--only') + 1] if '--only' in sys.argv else None
    if only in (None, 'examples'):
        examples(with_entropic=with_entropic)
    if only == 'lpclosure':
        lp_closure()
