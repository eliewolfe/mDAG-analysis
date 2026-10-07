"""
Finer breakdowns of the 4-node census for the manuscript (manuscript/piggybacks.md, Section 10).

Beyond the per-trick counts of `print_report`, this script splits the Fritz-type transitions by the parameters
recorded in their provenance (replace vs copy mode, single vs joint predictors, d-separation vs `markov` vs `relabel`
certificates, extra deletions) and prints, for every category, the inputs that are lost when that category of
transition is removed, with one certificate each. It also reruns the base closure with only the three-node seeds, to
show what the node-count-reducing piggybacks contribute when the Bell variants are not given as known gaps.

Usage: python "Special Applications/census_breakdowns.py" [--no-rescue] [--only headline|threeseeds|threeseeds-rescue|ladder|lpclosure]
"""
import os
import sys
import time
from collections import deque
from typing import Callable, Dict, List, Optional, Set

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from known_QC_gaps import SEEDS, SEEDS_3_NODES  # noqa: E402
from qc_gap_search import (ClosureExplorer, Transition, build_report, default_tricks, default_stages, entropic_tricks,  # noqa: E402
                           prove_gaps, render_certificate, add_stage, TRICK_GROUPS_FOR_REPORT, ENTROPIC_TRICK_GROUP)
from proving_QC_Gaps import four_node_representatives  # noqa: E402

Predicate = Callable[[Transition], bool]


# --------------------------------------------------------------------------------------------------
# Reading the provenance of Fritz-type transitions
# --------------------------------------------------------------------------------------------------

def predicted_modes(t: Transition):
    """((s, mode), ...) for Fritz-type transitions, () otherwise."""
    if t.trick in ('Fritz', 'Fritz_kept'):
        return dict(t.params)['predicted']
    if t.trick == 'Fritz_entropic':
        return t.params[1]
    return ()


def predictors_of(t: Transition):
    return dict(p for p in t.params if isinstance(p[0], str)).get('predictors', ())


def info_of(t: Transition) -> Dict:
    if t.trick == 'Fritz_entropic':
        return dict(t.params[2:])
    return {'certificate': 'dsep', 'deleted': ()}


def uses_copy(t: Transition) -> bool:
    return any(mode == 'copy' for _, mode in predicted_modes(t))


def is_fritz_type(t: Transition) -> bool:
    return t.trick in ('Fritz', 'Fritz_kept', 'Fritz_entropic')


# --------------------------------------------------------------------------------------------------
# Reachability under a predicate on transitions
# --------------------------------------------------------------------------------------------------

def reachable(explorer: ClosureExplorer, root, keep: Predicate) -> Set:
    seen = {root}
    frontier = deque([root])
    while frontier:
        current = frontier.popleft()
        for t in explorer.edges.get(current, ()):
            if keep(t) and t.target not in seen:
                seen.add(t.target)
                frontier.append(t.target)
    return seen


def path(explorer: ClosureExplorer, root, goals: Set, keep: Predicate) -> Optional[List[Transition]]:
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


def fixpoint(explorer: ClosureExplorer, input_ids: List, seed_ids: Set, keep: Predicate) -> Set:
    """Inputs provable (closed under implication among the inputs) using only transitions satisfying `keep`."""
    proven = set(seed_ids)
    reach = {gid: reachable(explorer, gid, keep) for gid in input_ids}
    unproven = set(input_ids)
    changed = True
    while changed:
        changed = False
        for gid in sorted(unproven):
            if not reach[gid].isdisjoint(proven):
                proven.add(gid)
                unproven.discard(gid)
                changed = True
    return proven.intersection(input_ids)


def full_certificate(explorer: ClosureExplorer, gid, seed_ids: Dict, proven: Set, keep: Predicate) -> Optional[List[Transition]]:
    """Shortest chain under `keep` to a seed or to another proven input, chained down to a seed."""
    known = set(seed_ids) | (proven - {gid})
    chain = path(explorer, gid, known, keep)
    if chain is None:
        return None
    last = chain[-1].target
    if last in seed_ids:
        return chain
    rest = full_certificate(explorer, last, seed_ids, proven - {gid}, keep)
    return None if rest is None else chain + rest


def seed_name_of(chain: List[Transition], seed_ids: Dict) -> str:
    return seed_ids[chain[-1].target]


def show_examples(explorer, lost: Set, seed_ids: Dict, proven: Set, keep: Predicate, title: str, limit: int = 3,
                  prefer: Optional[Callable[[List[Transition]], bool]] = None) -> None:
    print(f"\n==== {title}: {len(lost)} inputs lost")
    chains = []
    for gid in sorted(lost):
        chain = full_certificate(explorer, gid, seed_ids, proven, keep)
        if chain is not None:
            chains.append((len(chain), explorer.representatives[gid].number_of_visible, gid, chain))
    chains.sort(key=lambda x: (x[0], x[1], x[2]))
    if prefer is not None:
        preferred = [c for c in chains if prefer(c[3])]
        chains = preferred + [c for c in chains if not prefer(c[3])]
    for _, _, gid, chain in chains[:limit]:
        print("--- input (unlabelled id %s):" % (gid,))
        print("    " + explorer.representatives[gid].as_string.replace("\n", "\n    ").rstrip())
        print(render_certificate(explorer, chain, seed_name_of(chain, seed_ids)))


# --------------------------------------------------------------------------------------------------
# Headline census with rescue, split by transition parameters
# --------------------------------------------------------------------------------------------------

def headline(with_rescue: bool = True) -> None:
    reps = four_node_representatives()
    seed_ids = {g.unique_unlabelled_id: name for name, g in SEEDS.items()}
    inputs = [g for g in reps if g.unique_unlabelled_id not in seed_ids]
    t0 = time.time()
    report = prove_gaps(inputs, SEEDS, stages=default_stages(max_visible=5, with_entropic=with_rescue), max_visible=5, verbose=False)
    print(f"staged search: {report.counts['proven']} proven of {report.counts['inputs']} ({time.time()-t0:.0f}s)")
    print("stage counts:", report.stage_counts)
    ex = report.explorer
    input_ids = list(dict.fromkeys(g.unique_unlabelled_id for g in inputs))
    everything = lambda t: True  # noqa: E731
    proven_all = fixpoint(ex, input_ids, set(seed_ids), everything)
    print("proven (all transitions):", len(proven_all))

    categories: Dict[str, Predicate] = {
        'Fritz (dropped predictors) in copy mode': lambda t: not (t.trick == 'Fritz' and uses_copy(t)),
        'Fritz_kept in copy mode': lambda t: not (t.trick == 'Fritz_kept' and uses_copy(t)),
        'Fritz_entropic with kept predictors': lambda t: not (t.trick == 'Fritz_entropic' and dict(t.params[2:]).get('predictor_mode') == 'split'),
        'Fritz_entropic in copy mode': lambda t: not (t.trick == 'Fritz_entropic' and uses_copy(t)),
        'any copy-mode step': lambda t: not (is_fritz_type(t) and uses_copy(t)),
        'any joint-predictor step': lambda t: not (is_fritz_type(t) and len(predictors_of(t)) >= 2),
        'Fritz_entropic certified by relabel': lambda t: not (t.trick == 'Fritz_entropic' and info_of(t)['certificate'] == 'relabel'),
        'Fritz_entropic certified by markov': lambda t: not (t.trick == 'Fritz_entropic' and info_of(t)['certificate'] == 'markov'),
        'Fritz_entropic with extra deletions': lambda t: not (t.trick == 'Fritz_entropic' and info_of(t)['deleted']),
        'all Fritz_entropic steps': lambda t: t.trick != 'Fritz_entropic',
        'all Fritz_kept steps': lambda t: t.trick != 'Fritz_kept',
        'all kept-predictor steps (Fritz_kept and entropic split)': lambda t: not (t.trick == 'Fritz_kept' or (t.trick == 'Fritz_entropic' and dict(t.params[2:]).get('predictor_mode') == 'split')),
        'all Fritz (dropped predictors) steps': lambda t: t.trick != 'Fritz',
        'all Fritz-type steps': lambda t: not is_fritz_type(t),
    }
    print("\n== Inputs lost when a category of transition is removed (headline census, rescue included)")
    results = {}
    for name, keep in categories.items():
        kept = fixpoint(ex, input_ids, set(seed_ids), keep)
        results[name] = proven_all - kept
        print(f"{name:>70}: proven {len(kept)}, lost {len(results[name])}")

    print("\n== Proven using only Fritz-type transitions in replace mode (plus marginalization), no copy mode anywhere")
    replace_only = lambda t: (is_fritz_type(t) and not uses_copy(t)) or t.trick in ('naive_marginalization', 'teleportation_marginalization')  # noqa: E731
    fritz_replace = fixpoint(ex, input_ids, set(seed_ids), lambda t: t.trick in ('Fritz', 'Fritz_kept', 'naive_marginalization', 'teleportation_marginalization') and not uses_copy(t))
    fritz_any = fixpoint(ex, input_ids, set(seed_ids), lambda t: t.trick in ('Fritz', 'Fritz_kept', 'naive_marginalization', 'teleportation_marginalization'))
    print("Fritz (d-sep) + marginalization, replace mode only:", len(fritz_replace), "; with copy mode:", len(fritz_any))
    ent_replace = fixpoint(ex, input_ids, set(seed_ids), lambda t: replace_only(t))
    ent_any = fixpoint(ex, input_ids, set(seed_ids), lambda t: is_fritz_type(t) or t.trick in ('naive_marginalization', 'teleportation_marginalization'))
    print("Fritz + entropic + marginalization, replace mode only:", len(ent_replace), "; with copy mode:", len(ent_any))

    # Examples.
    non_fritz = fixpoint(ex, input_ids, set(seed_ids), lambda t: not is_fritz_type(t))
    show_examples(ex, results['all Fritz-type steps'] - results['all Fritz_entropic steps'], seed_ids, proven_all,
                  lambda t: t.trick != 'Fritz_entropic', "Needs a Fritz-type step, provable without the LP")
    replace_fritz_examples = fixpoint(ex, input_ids, set(seed_ids), lambda t: not (is_fritz_type(t) and uses_copy(t)) and t.trick != 'Fritz_entropic') - non_fritz
    show_examples(ex, replace_fritz_examples, seed_ids, proven_all, lambda t: not (is_fritz_type(t) and uses_copy(t)) and t.trick != 'Fritz_entropic',
                  "Provable with d-separation Fritz in replace mode (no copy mode, no LP) but not without Fritz", limit=4)
    show_examples(ex, results['all Fritz_kept steps'], seed_ids, proven_all, everything, "Lost when the kept-predictor d-separation steps are removed", limit=3)
    show_examples(ex, results['all kept-predictor steps (Fritz_kept and entropic split)'], seed_ids, proven_all, everything, "Lost when every kept-predictor step is removed", limit=3)
    show_examples(ex, results['any copy-mode step'], seed_ids, proven_all, everything, "Only via copy mode")
    show_examples(ex, results['any joint-predictor step'], seed_ids, proven_all, everything, "Only via joint predictors")
    only_entropic = results['all Fritz_entropic steps']
    for cert in ('relabel', 'markov', 'dsep'):
        show_examples(ex, only_entropic, seed_ids, proven_all, everything, f"Only via entropic Fritz; shortest certificates whose entropic step is '{cert}'", limit=3,
                      prefer=lambda chain, cert=cert: any(t.trick == 'Fritz_entropic' and info_of(t)['certificate'] == cert and not info_of(t)['deleted'] for t in chain))
    show_examples(ex, only_entropic, seed_ids, proven_all, everything, "Only via entropic Fritz; certificates with extra deletions", limit=3,
                  prefer=lambda chain: any(t.trick == 'Fritz_entropic' and info_of(t)['deleted'] for t in chain))
    show_examples(ex, results['Fritz_entropic certified by relabel'], seed_ids, proven_all, everything, "Lost when relabel-certified transitions are removed", limit=2)
    show_examples(ex, results['Fritz_entropic certified by markov'], seed_ids, proven_all, everything, "Lost when markov-certified transitions are removed", limit=2)
    show_examples(ex, results['Fritz_entropic with extra deletions'], seed_ids, proven_all, everything, "Lost when extra-deletion transitions are removed", limit=2)
    # Non-Fritz tricks: only-via examples.
    for trick in ('PD', 'conditioning', 'interruption', 'naive_marginalization', 'teleportation_marginalization'):
        lost = proven_all - fixpoint(ex, input_ids, set(seed_ids), lambda t, trick=trick: t.trick != trick)
        show_examples(ex, lost, seed_ids, proven_all, everything, f"Only via {trick}", limit=2)
    from quantum_mDAG import ENTROPIC_STATS
    print("\nENTROPIC_STATS:", dict(sorted(ENTROPIC_STATS.items())))
    known = report.proven_structure_ids()
    reps_known = [ex.representatives[i] for i in known if i in ex.representatives]
    hybrid = [g for g in reps_known if g.C_simplicial_complex_instance.simplicial_complex_as_sets]
    by_n = {}
    for g in reps_known:
        by_n[g.number_of_visible] = by_n.get(g.number_of_visible, 0) + 1
    print("proven structures (reach a seed), all sizes:", len(reps_known), "by visible-node count", dict(sorted(by_n.items())),
          "; with at least one classical facet:", len(hybrid))
    print("\nremaining unproven inputs:", len(input_ids) - len(proven_all))
    for gid in sorted(set(input_ids) - proven_all):
        print("    " + ex.representatives[gid].as_string.replace("\n", "\n    ").rstrip())


# --------------------------------------------------------------------------------------------------
# Three-node seeds only: what do the node-count-reducing piggybacks contribute?
# --------------------------------------------------------------------------------------------------

def three_node_seeds_only(with_rescue: bool = False) -> None:
    reps = four_node_representatives()
    seed_ids = {g.unique_unlabelled_id: name for name, g in SEEDS_3_NODES.items()}
    inputs = list(reps)
    report = prove_gaps(inputs, SEEDS_3_NODES, stages=default_stages(max_visible=5, with_entropic=with_rescue), max_visible=5, verbose=False)
    print("stage counts:", report.stage_counts)
    ex = report.explorer
    input_ids = list(dict.fromkeys(g.unique_unlabelled_id for g in inputs))
    everything = lambda t: True  # noqa: E731
    proven_all = fixpoint(ex, input_ids, set(seed_ids), everything)
    print(f"\n==== Three-node seeds only: {len(input_ids)} inputs (Bell variants included), proven {len(proven_all)}")
    for name, count in report.provable_with.items():
        print(f"    alone {name:>40}: {count}")
    for name, count in report.only_via.items():
        print(f"    only  {name:>40}: {count}")
    bell_ids = {g.unique_unlabelled_id: name for name, g in SEEDS.items() if g.number_of_visible == 4}
    print("Bell variants proven from three-node seeds:", len(set(bell_ids) & proven_all), "of", len(bell_ids))
    for gid in sorted(bell_ids):
        chain = full_certificate(ex, gid, seed_ids, proven_all, everything)
        print(f"    {bell_ids[gid]}: " + ("unproven" if chain is None else " -> ".join(t.trick for t in chain) + " -> " + seed_name_of(chain, seed_ids)))
    for trick in ('interruption', 'naive_marginalization', 'teleportation_marginalization', 'PD', 'conditioning'):
        lost = proven_all - fixpoint(ex, input_ids, set(seed_ids), lambda t, trick=trick: t.trick != trick)
        show_examples(ex, lost, seed_ids, proven_all, everything, f"[three-node seeds] only via {trick}", limit=3)



# --------------------------------------------------------------------------------------------------
# Cheap to expensive: a cumulative ladder of trick categories
# --------------------------------------------------------------------------------------------------

def ladder(with_rescue: bool = True) -> None:
    """Proven by the base tricks without Fritz; then additionally by Fritz without copy mode; then with copy mode;
    then additionally by the entropic trick without copy mode; then with copy mode."""
    reps = four_node_representatives()
    seed_ids = {g.unique_unlabelled_id: name for name, g in SEEDS.items()}
    inputs = [g for g in reps if g.unique_unlabelled_id not in seed_ids]
    report = prove_gaps(inputs, SEEDS, stages=default_stages(max_visible=5, with_entropic=with_rescue), max_visible=5, verbose=False)
    print("stage counts:", report.stage_counts)
    ex = report.explorer
    input_ids = list(dict.fromkeys(g.unique_unlabelled_id for g in inputs))
    rungs = [
        ('elementary reductions only', lambda t: not is_fritz_type(t)),
        ('+ Fritz, dropped predictors, replace mode', lambda t: not is_fritz_type(t) or (t.trick == 'Fritz' and not uses_copy(t))),
        ('+ Fritz, dropped predictors, copy mode', lambda t: not is_fritz_type(t) or t.trick == 'Fritz'),
        ('+ Fritz_kept, kept predictors, replace mode', lambda t: t.trick not in ('Fritz_kept', 'Fritz_entropic') or (t.trick == 'Fritz_kept' and not uses_copy(t))),
        ('+ Fritz_kept, kept predictors, copy mode', lambda t: t.trick != 'Fritz_entropic'),
        ('+ Fritz_entropic (LP), dropped predictors', lambda t: t.trick != 'Fritz_entropic' or dict(t.params[2:]).get('predictor_mode') == 'drop'),
        ('+ Fritz_entropic (LP), kept predictors', lambda t: True),
    ]
    print(f"\n==== Cumulative ladder (cheap to expensive); headline seeds; {len(input_ids)} inputs up to relabelling")
    previous = set()
    for name, keep in rungs:
        proven = fixpoint(ex, input_ids, set(seed_ids), keep)
        print(f"{name:>50}: proven {len(proven):4d}  (new {len(proven - previous):3d})")
        previous = proven
    print(f"{'remaining':>50}: {len(input_ids) - len(previous)}")


def lp_closure() -> None:
    """Does applying the LP-certified trick to every reachable structure (not just to the unproven inputs) prove
    more? Every structure proven so far is already a known gap for the others (reachability is transitive), so
    this tests LP steps at depth greater than one."""
    reps = four_node_representatives()
    seed_ids = {g.unique_unlabelled_id: name for name, g in SEEDS.items()}
    inputs = [g for g in reps if g.unique_unlabelled_id not in seed_ids]
    t0 = time.time()
    report = prove_gaps(inputs, SEEDS, stages=default_stages(max_visible=5), max_visible=5, verbose=False)
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
    with_rescue = '--no-rescue' not in sys.argv
    only = sys.argv[sys.argv.index('--only') + 1] if '--only' in sys.argv else None
    if only in (None, 'headline'):
        headline(with_rescue=with_rescue)
    if only in (None, 'threeseeds'):
        three_node_seeds_only(with_rescue=False)
    if only in (None, 'ladder'):
        ladder(with_rescue=with_rescue)
    if only == 'threeseeds-rescue':
        three_node_seeds_only(with_rescue=True)
    if only == 'lpclosure':
        lp_closure()
