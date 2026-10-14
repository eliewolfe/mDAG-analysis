"""
Worked examples for the manuscript (manuscript/piggybacks.md, Sections 1 and 8): one certificate per row.

Runs the census of proving_QC_Gaps (cache disabled, so every certificate ends at a named seed) and prints, for every
elementary piggyback, input classes that are lost without it; for every rung of the ladder (LADDER of
qc_gap_search), the classes it adds, with the shortest certificate each; the proven-structure database by node
count; and the still-unproven classes.

Usage: python "Special Applications/census_breakdowns.py" [--no-entropic] [--engine semigraphoid|lp|both] [--only examples|lpclosure]
"""
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Set

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from known_QC_gaps import SEEDS  # noqa: E402
from qc_gap_search import (ClosureExplorer, Transition, GapReport, fritz_tricks, add_stage, render_certificate,  # noqa: E402
                           LADDER, _fixpoint_if, is_fritz_type)
from proving_QC_Gaps import run_search, print_report  # noqa: E402

Predicate = Callable[[Transition], bool]


def _goals(report: GapReport) -> Dict:
    return dict(report.seed_keys)


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


def examples(with_entropic: bool = True, engine=None) -> None:
    t0 = time.time()
    report, _ = run_search(verbose=False, with_entropic=with_entropic, use_cache=False, engine=engine)
    print(f"[census in {time.time() - t0:.0f}s]")
    print_report(report)

    # The elementary stage: what each elementary piggyback proves exclusively (the Fritz transitions excluded).
    elementary = lambda t: not is_fritz_type(t)  # noqa: E731
    proven_cheap = _fixpoint_if(report, elementary)
    assert len(proven_cheap) == report.stage_counts[0][1]
    for trick in ('PD', 'conditioning', 'node_stitching', 'naive_marginalization', 'teleportation_marginalization'):
        lost = proven_cheap - _fixpoint_if(report, lambda t, trick=trick: elementary(t) and t.trick != trick)
        show_examples(report, lost, elementary, f"[elementary] only via {trick}", limit=3)
    lost = proven_cheap - _fixpoint_if(report, lambda t: elementary(t) and t.trick not in ('naive_marginalization', 'teleportation_marginalization'))
    show_examples(report, lost, elementary, "[elementary] only via marginalization (either kind)", limit=4)

    # What each rung of the ladder adds (a certificate using that rung's transitions and cheaper ones).
    previous = _fixpoint_if(report, LADDER[0][1])
    for name, keep in LADDER[1:]:
        proven = _fixpoint_if(report, keep)
        show_examples(report, proven - previous, keep, f"new at rung {name}", limit=3)
        previous = proven

    known = report.proven_structure_ids()
    reps_known = [report.explorer.representatives[i] for i in known if i in report.explorer.representatives]
    hybrid = [g for g in reps_known if g.C_simplicial_complex_instance.simplicial_complex_as_sets]
    by_n: Dict[int, int] = {}
    for g in reps_known:
        by_n[g.number_of_visible] = by_n.get(g.number_of_visible, 0) + 1
    print("\nproven structures (reach a seed), all sizes:", len(reps_known), "by visible-node count", dict(sorted(by_n.items())),
          "; with at least one classical facet:", len(hybrid))
    print("\nremaining unproven classes:", len(report.remaining))
    for g in sorted(report.remaining, key=lambda g: report.key(g)):
        print("    " + g.as_string.replace("\n", "\n    ").rstrip())


def lp_closure() -> None:
    """Does applying the LP-certified trick to every reachable structure (not just to the unproven inputs) prove
    more? Every structure proven so far is already a known gap for the others (reachability is transitive), so
    this tests LP steps at depth greater than one."""
    t0 = time.time()
    report, _ = run_search(verbose=False, use_cache=False)
    print("staged search:", report.stage_counts, f"({time.time()-t0:.0f}s)")
    before = set(report.proven)
    report = add_stage(report, fritz_tricks(max_visible=5, max_predictors=1, predictor_mode='split'), roots_only=False, name='Fritz_closure', verbose=False)
    print("after LP closure over everything reachable:", report.stage_counts, f"({time.time()-t0:.0f}s); structures {len(report.explorer.edges)}")
    new = set(report.proven) - before
    print("newly proven inputs:", len(new))
    for gid in sorted(new):
        print("    " + report.explorer.representatives[gid].as_string.replace("\n", "\n    ").rstrip())
        print(report.certificate(report.explorer.representatives[gid]))


if __name__ == '__main__':
    with_entropic = '--no-entropic' not in sys.argv
    engine = sys.argv[sys.argv.index('--engine') + 1] if '--engine' in sys.argv and sys.argv.index('--engine') + 1 < len(sys.argv) else None
    only = sys.argv[sys.argv.index('--only') + 1] if '--only' in sys.argv else None
    if only in (None, 'examples'):
        examples(with_entropic=with_entropic, engine=engine)
    if only == 'lpclosure':
        lp_closure()
