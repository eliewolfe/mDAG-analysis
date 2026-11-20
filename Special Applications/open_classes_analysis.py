"""
The open classes of the four-node census: which would follow from which, which intermediate structures they share,
and which equivalences would settle them. Everything quoted in manuscript/open_classes.md is printed by this script.

    python "Special Applications/open_classes_analysis.py" --five --events 5

The hierarchy is computed from bounded forward paths and independently checked by
seeding each open class in the reverse search. The report also lists consequences
of composing recorded implications. Shared targets use the same screen as the census.
The final section evaluates classical equivalences and binary support invariants.
"""
from __future__ import absolute_import
import argparse
import collections
import sys
import time
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (str(PROJECT_ROOT), str(Path(__file__).resolve().parent)):
    if path not in sys.path:
        sys.path.insert(0, path)

from growth_census import GapFreeScreen, Phase, advance, five_node_screen, underlying_mdag
import qc_gap_search as S
import proving_QC_Gaps as census


# -- structures as text ------------------------------------------------------------------------------------------

def parts(g):
    return (g.number_of_visible, sorted(map(tuple, g.directed_structure_instance.edge_list)),
            sorted(map(tuple, g.C_simplicial_complex_instance.as_tuples)),
            sorted(map(tuple, g.Q_simplicial_complex_instance.as_tuples)))


def fmt(g):
    n, edges, C, Q = parts(g)
    e = ', '.join(f'{a}→{b}' for a, b in edges) or 'no edges'
    f = ', '.join(['C{' + ','.join(map(str, x)) + '}' for x in C] + ['Q{' + ','.join(map(str, x)) + '}' for x in Q])
    return f'{e}; {f or "no facets"}'




# -- the recorded transition graph ------------------------------------------------------------------------------

def recorded_graph(report):
    """The census record as the analysis uses it: the report, its explorer, the keys of the open classes in the
    order of the report (the numbering of manuscript Section 10) and the two lookups every part needs."""
    explorer = report.explorer
    open_keys = [report.key(g) for g in report.remaining]
    numbers = getattr(report, "original_open_numbers", {k: i for i, k in enumerate(open_keys, 1)})
    forward = report.bounded.forward if hasattr(report, 'bounded') else explorer.reachable
    return SimpleNamespace(report=report, explorer=explorer, open_keys=open_keys,
                           number={k: numbers[k] for k in open_keys},
                           reachable_from=lru_cache(len(open_keys))(forward),
                           composed_from=lru_cache(len(open_keys))(explorer.reachable),
                           structure=explorer.representatives.__getitem__)


def step_text(explorer, t):
    if t.trick == 'Fritz':
        p = dict(t.params)
        detail = (f"predicted {p['targets']}, predictor {p['predictor']}, {p['target_mode']} target, "
                  f"{p['predictor_mode']} predictor, certificate {p['certificate']}")
    else:
        detail = ', '.join(f'{k} {v}' for k, v in t.params) if t.params else ''
    return f"{t.trick} ({detail}): {fmt(explorer.structures[t.via])}  =>  {fmt(explorer.structures[t.produced])}"


def target_witnesses(rec, target):
    """Explain a candidate's payoff, showing exactly where a composed path uses another allowance."""
    print(f"Target {fmt(rec.structure(target))}")
    for source in rec.open_keys:
        if target not in rec.composed_from(source):
            continue
        path = rec.report.bounded.forward(source, {target})
        if path is None:
            path = rec.explorer.path(source, {target})
            kind = 'composed'
        else:
            kind = 'bounded'
        print(f"  class {rec.number[source]} ({kind})")
        phase = Phase.AVAILABLE
        for edge in path:
            nxt = advance(phase, edge, rec.report.bounded.resume_at)
            if nxt is None:
                print(f"     via the independently established implication from {fmt(rec.structure(edge.source))}")
                nxt = advance(Phase.AVAILABLE, edge, rec.report.bounded.resume_at)
                assert nxt is not None
            print("     ", step_text(rec.explorer, edge))
            phase = nxt


# -- Part 1 -----------------------------------------------------------------------------------------------------

def part_1(rec):
    print("=" * 110)
    print("PART 1. The proof hierarchy: declare one open class a known gap, see which others follow.")
    report, explorer = rec.report, rec.explorer
    index = explorer.index
    print(f"\n{len(rec.open_keys)} open classes (number, structures in the class, representative):")
    for key in rec.open_keys:
        print(f"  {rec.number[key]:>2}  [{index.size(key):>2}]  {fmt(rec.structure(key))}")
    # Independent seed-injection check: reverse traversal with each open key as
    # a goal must agree with forward reachability from each open class.
    bounded = getattr(report, "bounded", None)
    forward = {key: rec.reachable_from(key) for key in rec.open_keys}
    unrestricted = {key: rec.composed_from(key) for key in rec.open_keys}
    relations = {}
    seeds = set(report.seed_keys)
    composed_gaps = [key for key in rec.open_keys if unrestricted[key] & seeds]
    print("Additional input gaps by composition:", [rec.number[k] for k in composed_gaps])
    for key in composed_gaps:
        for edge in explorer.path(key, seeds):
            print("     ", step_text(explorer, edge))
    for goal in rec.open_keys:
        seeded = ({k for k, phase in bounded.to_goals({goal}) if phase == Phase.AVAILABLE}
                  if bounded else S._reverse_reachable(explorer, {goal}))
        sources = [k for k in rec.open_keys if k != goal and goal in forward[k]]
        assert set(sources) == {k for k in rec.open_keys if k != goal and k in seeded}
        if sources:
            relations[rec.number[goal]] = sorted(rec.number[k] for k in sources)
        for source in sources:
            print(f"  bounded: class {rec.number[source]} -> class {rec.number[goal]}")
            path = bounded.forward(source, {goal}) if bounded else explorer.path(source, {goal})
            for t in path:
                print("     ", step_text(explorer, t))
        for source in rec.open_keys:
            if goal in unrestricted[source] and goal not in forward[source]:
                print(f"  composed only: class {rec.number[source]} -> class {rec.number[goal]}")
                for t in explorer.path(source, {goal}):
                    print("     ", step_text(explorer, t))
    print("Bounded hierarchy (hypothesis -> classes proven):", relations)
    related = set(relations) | {j for lower in relations.values() for j in lower}
    print(f"\nOpen classes in no such relation: {len(rec.open_keys) - len(related)}.")
    return relations


# -- Part 2 -----------------------------------------------------------------------------------------------------

def part_2(rec, five):
    print("\n" + "=" * 110)
    print("PART 2. The structures the open classes reduce to, and the ones they share.")
    explorer = rec.explorer
    reach = {key: rec.reachable_from(key) for key in rec.open_keys}
    reached = set().union(*reach.values())
    payoff = collections.defaultdict(set)          # structure -> open classes that reduce to it
    for key in rec.open_keys:
        for x in reach[key]:
            payoff[x].add(rec.number[key])
    assert not (reached & set(rec.report.seed_keys)), "an open class reaches a named seed"
    screening = getattr(rec.report, 'screening', None)
    if screening is None:
        screening = GapFreeScreen(explorer.index, five_node_screen() if five else None)
    verdict = {x: screening.verdict(rec.structure(x)) for x in reached}
    table = collections.Counter((rec.structure(x).number_of_visible, verdict[x]) for x in reached)
    print(f"\n{len(reached)} structures are reachable from the open classes (none is a known gap); "
          f"{len(reached & explorer.lookup_only)} of them are degradations registered as lookups.")
    for (n, v), count in sorted(table.items()):
        print(f"  {n} visible nodes, {v}: {count}")
    candidates = {x for x in reached if not verdict[x].startswith('gap-free')}
    shared = collections.defaultdict(list)
    for x in candidates:
        if len(payoff[x]) >= 2:
            shared[tuple(sorted(payoff[x]))].append(x)
    maximal = [s for s in shared if not any(set(s) < set(t) for t in shared)]
    print(f"\nCandidates reached from two or more open classes: {sum(map(len, shared.values()))}, "
          f"in {len(shared)} distinct sets of classes; maximal sets: {sorted(maximal)}")
    for s in sorted(maximal, key=lambda s: (-len(s), s)):
        xs = sorted(shared[s], key=lambda x: (x in explorer.lookup_only, rec.structure(x).number_of_visible, fmt(rec.structure(x))))
        expanded = [x for x in xs if x not in explorer.lookup_only]
        print(f"\n  classes {list(s)}: {len(xs)} common candidates ({len(expanded)} of them not mere degradations)")
        for x in expanded[:20]:
            tag = 'open class ' + str(rec.number[x]) if x in rec.number else f"{rec.structure(x).number_of_visible}-node hybrid"
            print(f"     {tag}: {fmt(rec.structure(x))}  [key {','.join(map(str, x))}]")
    alone = [rec.number[k] for k in rec.open_keys if all(len(payoff[x]) == 1 for x in reach[k] if x in candidates)]
    print(f"\nOpen classes sharing no candidate with another: {len(alone)}: {alone}")
    small = [x for x in candidates if rec.structure(x).number_of_visible <= 4 and x not in explorer.lookup_only and x not in rec.number]
    print(f"\nCandidate hybrids of at most four nodes that are not mere degradations ({len(small)}), by the classes reaching them:")
    by_set = collections.defaultdict(list)
    for x in small:
        by_set[tuple(sorted(payoff[x]))].append(x)
    for s, xs in sorted(by_set.items()):
        print(f"  classes {list(s)}:")
        for x in sorted(xs, key=lambda x: (rec.structure(x).number_of_visible, fmt(rec.structure(x)))):
            print(f"     {fmt(rec.structure(x))}")
    print("Candidate counts by size and lookup-only status:",
          dict(sorted(collections.Counter((k[0], k in explorer.lookup_only) for k in candidates).items())))
    composed_extra = {rec.number[k]: len(rec.composed_from(k) - reach[k]) for k in rec.open_keys}
    print("Additional target keys reached by composition, by source:", {k: n for k, n in composed_extra.items() if n})
    composed_payoff = collections.defaultdict(set)
    for key in rec.open_keys:
        for target in rec.composed_from(key):
            if not screening.verdict(rec.structure(target)).startswith('gap-free'):
                composed_payoff[target].add(rec.number[key])
    for name, sets in [('bounded', payoff), ('composed', composed_payoff)]:
        groups = collections.Counter(tuple(sorted(s)) for k, s in sets.items()
                                     if len(s) > 1 and (name == 'composed' or k in candidates))
        print(f"Shared candidate counts by set of open classes ({name}):", dict(sorted(groups.items())))
    return reached, verdict, payoff, candidates


# -- Part 4 -----------------------------------------------------------------------------------------------------

def part_4(rec, candidates, payoff, max_events):
    print("\n" + "=" * 110)
    print("PART 4. Equivalences that would settle open classes.")
    explorer, index = rec.explorer, rec.explorer.index
    report = rec.report
    C4 = index.classical[4]
    proven = set(report.proven) | set(report.seed_keys)

    def status(n, uid):
        key = index._key_of_mdag[(n, uid)]
        if key in rec.number:
            return 'open', rec.number[key]
        if key in proven:
            return 'proven', key
        return ('latent-free' if index.is_latent_free_equivalent(key) else 'other'), key

    # (a) the classical class of each open class, and how the quantum classes split the classical ones
    print("\n(a) Classical classes.")
    split = collections.Counter()
    for members in C4.equivalence_classes_as_ids:
        keys = {index._key_of_mdag[(4, uid)] for uid in members}
        split[(len(keys), tuple(sorted({status(4, next(iter(index._mdag_ids_of[k])))[0] for k in keys})))] += 1
    for (count, statuses), classes in sorted(split.items()):
        print(f"  {classes:>3} classical classes hold {count} quantum class{'es' if count > 1 else ''}, {' and '.join(statuses)}")
    lonely = all(len({index._key_of_mdag[(4, uid)] for uid in C4.equivalence_classes_as_ids[index.classical_class(k)[1]]}) == 1
                 for k in rec.open_keys)
    print(f"  Every open class is the only quantum class of its classical class: {lonely}")

    # (b) hybrids whose all-quantum upgrade is a proven class
    print("\n(b) Candidate hybrids (at most four nodes) whose all-quantum upgrade is a proven class: proving that the "
          "classical facets can be upgraded for free would give the hybrid a gap.")
    lift = collections.defaultdict(list)
    for x in candidates:
        g = rec.structure(x)
        n = g.number_of_visible
        if n not in index.classical or x in rec.number:
            continue
        if status(n, underlying_mdag(g).unique_unlabelled_id)[0] == 'proven':
            lift[tuple(sorted(payoff[x]))].append(x)
    for s, xs in sorted(lift.items()):
        expanded = sorted((x for x in xs if x not in explorer.lookup_only), key=lambda x: (rec.structure(x).number_of_visible, fmt(rec.structure(x))))
        print(f"  classes {list(s)}: {len(xs)} hybrids ({len(expanded)} not mere degradations)")
        for x in expanded:
            print(f"     {fmt(rec.structure(x))}   [upgrade: {fmt(S_upgrade(underlying_mdag(rec.structure(x))))}]")

    # (c) structurally comparable pairs with unrefuted classical equivalence: the upward rule would apply
    print("\n(c) Pairs (open structure r, proven or open structure b structurally below r). If C(r) = C(b), a gap of b "
          "is a gap of r (the upward rule). The pairs that the invariants do not separate:")
    mdags = C4.dict_uniqind_unlabelled_mDAGs
    funnel = collections.Counter()
    survivors = []
    for key in rec.open_keys:
        for uid in sorted(index._mdag_ids_of[key]):
            upper = mdags[uid]
            for below in sorted(C4.structurally_below(uid)):
                kind, who = status(4, below)
                if kind not in ('proven', 'open') or (kind == 'open' and who == rec.number[key]):
                    continue
                lower = mdags[below]
                funnel[(kind, '1 structurally below an open structure')] += 1
                if upper.skeleton_nof_edges != lower.skeleton_nof_edges or upper.CI_count != lower.CI_count \
                        or upper.skeleton_unlabelled != lower.skeleton_unlabelled or upper.all_CI_unlabelled != lower.all_CI_unlabelled:
                    continue
                funnel[(kind, '2 same skeleton and d-separations')] += 1
                if upper.all_esep_unlabelled != lower.all_esep_unlabelled \
                        or upper.all_densely_connected_pairs_unlabelled != lower.all_densely_connected_pairs_unlabelled:
                    continue
                funnel[(kind, '3 same e-separations and densely connected pairs')] += 1
                survivors.append((rec.number[key], upper, lower, kind, who))
    alive = survivors
    for k in range(2, max_events + 1):
        t0 = time.time()
        alive = [s for s in alive if tuple(s[1].infeasible_binary_supports_n_events_beyond_esep_unlabelled(k))
                 == tuple(s[2].infeasible_binary_supports_n_events_beyond_esep_unlabelled(k))]
        for kind in ('proven', 'open'):
            funnel[(kind, f'4 same infeasible binary supports up to {k} events')] = sum(1 for s in alive if s[3] == kind)
        print(f"  [supports at {k} events: {time.time() - t0:.0f}s]")
    for kind in ('proven', 'open'):
        print(f"  lower structure {kind}:")
        for (k2, label), count in sorted(funnel.items()):
            if k2 == kind:
                print(f"     {label[2:]}: {count} pairs")
    over_proven = collections.defaultdict(set)
    for number, upper, lower, kind, who in alive:
        if kind == 'proven':
            over_proven[number].add((fmt(S_upgrade(upper)), fmt(S_upgrade(lower))))
    print(f"\n  Open classes with a PROVEN structure below them whose classical inequivalence is not established "
          f"(up to {max_events} events): {sorted(over_proven)}")
    for number in sorted(over_proven):
        for upper, lower in sorted(over_proven[number])[:6]:
            print(f"     class {number}:  {upper}   above   {lower}")
        if len(over_proven[number]) > 6:
            print(f"     class {number}:  ... {len(over_proven[number]) - 6} more pairs")
    over_open = sorted({(number, who) for number, upper, lower, kind, who in alive if kind == 'open'})
    print(f"\n  Pairs of open classes (upper, lower) not separated classically (up to {max_events} events): {len(over_open)}")
    print("    ", over_open)
    seven_hard_graphs(rec, over_open)
    return alive


def seven_hard_graphs(rec, over_open):
    """The seven mDAGs of arXiv:2308.02380 (Remark 47, Tables 2 and 3) among the open classes. No binary support is
    incompatible with any of them, so the binary test above says nothing about their pairs; the paper's support
    with a ternary node is incompatible with the four of its Table 2 and compatible with the three of its Table 3,
    which separates every pair across the two groups. The partition is taken from the paper; the stored support
    (the_7_hard_graphs.advanced_support) is run only to confirm that the class numbers are matched correctly."""
    from the_7_hard_graphs import the_7_hard_graphs, advanced_support
    index = rec.explorer.index
    numbers = [rec.number[index.key(S_upgrade(g))] for g in the_7_hard_graphs]
    four, three = set(numbers[:4]), set(numbers[4:])
    inside = [pair for pair in over_open if set(pair) <= four | three]
    across = [pair for pair in inside if (pair[0] in three) != (pair[1] in three)]
    remaining = [pair for pair in over_open if pair not in across]
    print(f"\n  The seven hard graphs (arXiv:2308.02380): Table 2 = classes {sorted(four)}, Table 3 = classes {sorted(three)}.")
    print(f"    {len(inside)} of the {len(over_open)} pairs lie inside this family; the paper's ternary support separates the "
          f"{len(across)} pairs across its two tables: {across}")
    print(f"    pairs left: {len(remaining)}; inside the family: {[pair for pair in remaining if pair in inside]}")
    upper = sorted({u for u, l in remaining if not (u == 1 and l == 7)})
    print(f"    upper classes that would follow from a lower one if every remaining equivalence held: {len(upper)}: {upper}")
    compatible = [g.support_testing_instance((3, 2, 2, 2), 7).feasibleQ_from_matrix(occurring_events=advanced_support)
                  for g in the_7_hard_graphs]
    print(f"    check of the matching: the stored support is compatible with classes "
          f"{sorted(n for n, ok in zip(numbers, compatible) if ok)} and incompatible with "
          f"{sorted(n for n, ok in zip(numbers, compatible) if not ok)}")
    assert {n for n, ok in zip(numbers, compatible) if ok} == three


def S_upgrade(md):
    """An mDAG as the all-quantum structure (for printing)."""
    from quantum_mDAG import upgrade_to_QmDAG
    return upgrade_to_QmDAG(md)


def run_open_classes_analysis():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--five', action='store_true', help=argparse.SUPPRESS)  # now always screened
    parser.add_argument('--max-visible', type=int, default=census.DEFAULT_CAP)
    parser.add_argument('--resume-at', type=int, choices=(4, 5))
    parser.add_argument('--target-key', help='show open-class witnesses to a recorded key N,D,C,Q')
    parser.add_argument('--no-cache', action='store_true', help='rebuild the five-node screening set')
    parser.add_argument('--events', type=int, default=4, help="largest number of events of the binary supports test")
    parser.add_argument('--skip-second-depth', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    t0 = time.time()
    report, _ = census.run_search(verbose=False, use_cache=False, max_visible=args.max_visible, resume_at=args.resume_at,
                                  screen_cache_dir=None if args.no_cache else Path(census.CACHE_PATH).parent)
    counts = report.counts
    print(f"[census in {time.time() - t0:.0f}s: {counts['inputs']} classes, {counts['proven']} proven, {counts['remaining']} open "
          f"({counts['structures remaining']} structures)]")
    rec = recorded_graph(report)
    if args.target_key is not None:
        try:
            target = tuple(map(int, args.target_key.split(',')))
        except ValueError:
            parser.error('--target-key requires four comma-separated integers')
        if target not in rec.explorer.representatives:
            parser.error('the requested key is absent from this census record')
        target_witnesses(rec, target)
        return
    start = time.monotonic()
    part_1(rec)
    reached, verdict, payoff, candidates = part_2(rec, args.five)
    print(f"Hierarchy and shared targets: {time.monotonic() - start:.1f}s")
    part_4(rec, candidates, payoff, args.events)


if __name__ == '__main__':
    run_open_classes_analysis()
