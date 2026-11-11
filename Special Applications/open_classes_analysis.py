"""
The open classes of the four-node census: which would follow from which, which intermediate structures they share,
and which equivalences would settle them. Everything quoted in manuscript/open_classes.md is printed by this script.

    python "Special Applications/open_classes_analysis.py" --five --events 5

Part 1 (the proof hierarchy). For every open class in turn, the class is declared a known gap and the report is
rebuilt on the recorded transition graph: the other open classes that become proven are the ones that piggyback
onto it. The census records every transition out of every member of an unproven class (elementary closure, one
Fritz step with elementary follow-up), so this is what a rerun with the class as an extra seed would prove.
Part 2 (shared intermediates). The structures the open classes reduce to, screened for provable gap-freeness, and
the sets of open classes that reach a common structure. `--five` builds the five-node classical metagraph (three
minutes, 3.2 GB) to screen the five-node intermediates; without it they are reported unscreened.
Part 3 (a second depth). The unscreened-out intermediates are given to the cascade as inputs.
Part 4 (equivalences). The classical class of each open class; hybrids whose all-quantum upgrade is proven; and
pairs (open structure, proven or open structure structurally below it) whose classical inequivalence the
invariants do not establish, tested with binary infeasible supports up to `--events` events.
"""
from __future__ import absolute_import
import argparse
import collections
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (str(PROJECT_ROOT), str(Path(__file__).resolve().parent)):
    if path not in sys.path:
        sys.path.insert(0, path)

from directed_structures import DirectedStructure
from hypergraphs import Hypergraph, hypergraph_full_cleanup
from known_QC_gaps import SEEDS
from mDAG_advanced import mDAG
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


def underlying_mdag(g):
    """The mDAG with the facets of both colours (dominated ones dropped): its classical model is that of g."""
    n, edges, C, Q = parts(g)
    facets = hypergraph_full_cleanup([frozenset(f) for f in C + Q])
    return mDAG(DirectedStructure(edges, n), Hypergraph([tuple(sorted(f)) for f in facets if len(f) > 1], n))


def quantum_part_is_free(g):
    """Every quantum facet has at most one member that is not premeasurable (no visible parent and exactly one
    facet of either colour). Such a facet is equivalent to a classical one (the pre-measurement argument,
    manuscript 0.4), so the structure has the quantum model of an all-classical structure: no gap."""
    n, edges, C, Q = parts(g)
    has_parent = {b for _, b in edges}
    count = collections.Counter(v for f in C + Q for v in f)
    premeasurable = {v for v in range(n) if v not in has_parent and count[v] == 1}
    return all(sum(1 for v in f if v not in premeasurable) <= 1 for f in Q)


# -- the recorded transition graph ------------------------------------------------------------------------------

def recorded_graph(report):
    """The census record as the analysis uses it: the report, its explorer, the keys of the open classes in the
    order of the report (the numbering of manuscript Section 10) and the two lookups every part needs."""
    explorer = report.explorer
    open_keys = [report.key(g) for g in report.remaining]
    return SimpleNamespace(report=report, explorer=explorer, open_keys=open_keys,
                           number={key: i + 1 for i, key in enumerate(open_keys)},
                           reachable_from=explorer.reachable,
                           structure=explorer.representatives.__getitem__)


def step_text(explorer, t):
    if t.trick == 'Fritz':
        p = dict(t.params)
        detail = (f"predicted {p['targets']}, predictor {p['predictor']}, {p['target_mode']} target, "
                  f"{p['predictor_mode']} predictor, certificate {p['certificate']}")
    else:
        detail = ', '.join(f'{k} {v}' for k, v in t.params) if t.params else ''
    return f"{t.trick} ({detail}): {fmt(explorer.structures[t.via])}  =>  {fmt(explorer.structures[t.produced])}"


# -- Part 1 -----------------------------------------------------------------------------------------------------

def part_1(rec):
    print("=" * 110)
    print("PART 1. The proof hierarchy: declare one open class a known gap, see which others follow.")
    report, explorer = rec.report, rec.explorer
    index = explorer.index
    print(f"\n{len(rec.open_keys)} open classes (number, structures in the class, representative):")
    for key in rec.open_keys:
        print(f"  {rec.number[key]:>2}  [{index.size(key):>2}]  {fmt(rec.structure(key))}")
    groups = S._trick_groups_for(explorer, None)
    relations = {}
    t0 = time.time()
    for key in rec.open_keys:
        hypothetical = S.build_report(explorer, report.inputs, report.seeds, groups,
                                      known={**report.known, 'hypothesis': key})
        newly = sorted(rec.number[k] for k in hypothetical.proven if k in rec.number and k != key)
        assert set(newly) == {rec.number[k] for k in rec.open_keys if k != key and key in rec.reachable_from(k)}
        if newly:
            relations[rec.number[key]] = newly
    print(f"\n[{len(rec.open_keys)} hypothetical reports in {time.time() - t0:.0f}s]")
    if not relations:
        print("No open class would follow from another.")
    for top, lower in relations.items():
        top_key = rec.open_keys[top - 1]
        structures = index.size(top_key) + sum(index.size(rec.open_keys[j - 1]) for j in lower)
        print(f"\nIf class {top} had a gap, classes {lower} would follow: {1 + len(lower)} classes, {structures} structures.")
        for j in lower:
            print(f"  class {j} -> class {top}:")
            for t in explorer.path(rec.open_keys[j - 1], {top_key}):
                print("     ", step_text(explorer, t))
    related = set(relations) | {j for lower in relations.values() for j in lower}
    print(f"\nOpen classes in no such relation: {len(rec.open_keys) - len(related)}.")
    return relations


# -- Part 2 -----------------------------------------------------------------------------------------------------

def five_node_latent_free_ids():
    from metagraph_advanced import ClassicalMetagraph, QuantumMetagraph
    t0 = time.time()
    classical = ClassicalMetagraph.from_quantum(QuantumMetagraph(5, verbose=False))
    ids = classical.latent_free_equivalent_ids
    print(f"[five-node classical metagraph in {time.time() - t0:.0f}s: {len(ids)} mDAGs classically equivalent to a latent-free DAG]")
    return ids


def part_2(rec, five):
    print("\n" + "=" * 110)
    print("PART 2. The structures the open classes reduce to, and the ones they share.")
    explorer, index = rec.explorer, rec.explorer.index
    reach = {key: rec.reachable_from(key) for key in rec.open_keys}
    reached = set().union(*reach.values())
    payoff = collections.defaultdict(set)          # structure -> open classes that reduce to it
    for key in rec.open_keys:
        for x in reach[key]:
            payoff[x].add(rec.number[key])
    known = S._reverse_reachable(explorer, set(rec.report.seed_keys))
    assert not (reached & known), "an open class reaches a known gap"
    lf5 = five_node_latent_free_ids() if five else None
    verdict = {}
    for x in reached:
        g = rec.structure(x)
        n = g.number_of_visible
        if not parts(g)[3]:
            verdict[x] = 'gap-free: no quantum facet'
            continue
        uid = underlying_mdag(g).unique_unlabelled_id
        latent_free = (uid in index.classical[n].latent_free_equivalent_ids) if n in index.classical \
            else (None if lf5 is None else uid in lf5)
        if latent_free:
            verdict[x] = 'gap-free: classically equivalent to a latent-free DAG'
        elif quantum_part_is_free(g):
            verdict[x] = 'gap-free: every quantum facet is equivalent to a classical one'
        else:
            verdict[x] = 'candidate' if latent_free is not None else 'unscreened (five nodes)'
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
            print(f"     {tag}: {fmt(rec.structure(x))}")
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
    return reached, verdict, payoff, candidates


# -- Part 3 -----------------------------------------------------------------------------------------------------

def part_3(rec, reached, verdict, payoff):
    print("\n" + "=" * 110)
    print("PART 3. A second depth: the intermediates as inputs of the cascade.")
    explorer, index = rec.explorer, rec.explorer.index
    known = S._reverse_reachable(explorer, set(rec.report.seed_keys))
    known_names = {f'known:{i}': k for i, k in enumerate(sorted(known - set(rec.report.seed_keys)))}
    inputs = [x for x in reached if x not in rec.number and x not in explorer.lookup_only
              and not verdict[x].startswith('gap-free')]
    by_n = collections.Counter(rec.structure(x).number_of_visible for x in inputs)
    t0 = time.time()
    report = S.prove_gaps([rec.structure(x) for x in inputs], SEEDS, stages=S.default_stages(max_visible=5),
                          max_visible=5, verbose=False, known=known_names, index=index)
    proven = [x for x in inputs if report.key(rec.structure(x)) in report.proven]
    print(f"\n{len(inputs)} intermediates that are not mere degradations and not provably gap-free "
          f"({dict(sorted(by_n.items()))} by node count) were given to the full cascade, with the seeds and the "
          f"{len(known_names)} structures the census proves as known gaps, outputs bounded at five visible nodes: "
          f"{len(proven)} proven in {time.time() - t0:.0f}s.")
    for x in proven:
        print(f"   PROVEN: {fmt(rec.structure(x))}  (open classes {sorted(payoff[x])})")
        print('     ' + report.certificate(rec.structure(x)).replace('\n', '\n     '))
    return proven


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
    return alive


def S_upgrade(md):
    """An mDAG as the all-quantum structure (for printing)."""
    from quantum_mDAG import upgrade_to_QmDAG
    return upgrade_to_QmDAG(md)


def run_open_classes_analysis():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--five', action='store_true', help="screen the five-node intermediates (three minutes, 3.2 GB)")
    parser.add_argument('--events', type=int, default=4, help="largest number of events of the binary supports test")
    parser.add_argument('--skip-second-depth', action='store_true')
    args = parser.parse_args()
    t0 = time.time()
    report, _ = census.run_search(verbose=False, use_cache=False)
    counts = report.counts
    print(f"[census in {time.time() - t0:.0f}s: {counts['inputs']} classes, {counts['proven']} proven, {counts['remaining']} open "
          f"({counts['structures remaining']} structures)]")
    rec = recorded_graph(report)
    part_1(rec)
    reached, verdict, payoff, candidates = part_2(rec, args.five)
    if not args.skip_second_depth:
        part_3(rec, reached, verdict, payoff)
    part_4(rec, candidates, payoff, args.events)


if __name__ == '__main__':
    run_open_classes_analysis()
