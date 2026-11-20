"""Four-node QC-gap census with any number of non-growing Fritz steps and one growth.

Run with python "Special Applications/fritz_growth_experiment.py". Only named seeds are
supplied. Degradation and upward remain lookups, just as in the production census.
A growth output can hit a known gap immediately. Otherwise it must shrink before
non-growing Fritz closure resumes. The search never expands a size-preserving move at five.
"""
import argparse
from collections import Counter, deque
from functools import partial
from time import perf_counter

import proving_QC_Gaps as census
import entropic_lp
import qc_gap_search as S
from known_QC_gaps import SEEDS
from open_classes_analysis import fmt
from growth_census import quantum_part_is_free, underlying_mdag


def fritz(g, kind):
    n = g.number_of_visible
    cap = {'non-growing': n, 'growth': 5, 'reducing': n - 1}[kind]
    if cap < 3:
        return
    # A strict decrease needs an unsplit predictor and unsplit targets (one predictor
    # in the census). A growth needs split targets, possibly several in a joint step.
    target_modes = ('split',) if kind == 'growth' else ('unsplit', 'split')
    predictor_modes = ('unsplit', 'split')
    if kind == 'reducing':
        target_modes = predictor_modes = ('unsplit',)
    for pool in ('parents', 'siblings'):
        for pm in predictor_modes:
            for tm in target_modes:
                for params, child in g.fritz_steps(pool=pool, predictor_mode=pm, target_mode=tm,
                                                   max_visible=cap, use_lp=pool == 'siblings',
                                                   engine='semigraphoid'):
                    if kind != 'growth' or child.number_of_visible > n:
                        yield params, child


def fritz_depth_counts(explorer, inputs, seeds, elementary, expected):
    """Apply one Fritz edge per iteration, with free elementary moves and lookups on both sides."""
    reverse = S._reverse_edges(explorer)
    free = frozenset(elementary) | S.LOOKUP_TRICKS
    reached = S._reverse_reachable(explorer, seeds, tricks=free, reverse=reverse)
    counts = [len(inputs & reached)]
    while not expected.issubset(reached):
        parents = {t.source for key in reached for t in reverse.get(key, ()) if t.trick == 'non-growing'}
        more = S._reverse_reachable(explorer, parents | reached, tricks=free, reverse=reverse)
        assert more != reached, 'non-growing depth check stalled'
        reached = more
        counts.append(len(inputs & reached))
    return counts


def run(index, screen=True):
    start = perf_counter()
    elementary = S.elementary_tricks()
    inputs = census.census_inputs(index)
    report = S.prove_gaps(inputs, SEEDS, tricks=elementary, index=index, verbose=False)
    explorer = report.explorer
    inputs = set(map(explorer.key, inputs))
    seeds = set(report.seed_keys)
    elementary_known = S._reverse_reachable(explorer, seeds)
    print(f'Elementary closure: {len(inputs & elementary_known)} proven, {perf_counter() - start:.2f}s', flush=True)

    explorer.tricks.update({kind: partial(fritz, kind=kind)
                            for kind in ('non-growing', 'growth', 'reducing')})
    screened = {}

    def gap_free(key):
        if not screen:
            return False
        if key not in screened:
            g = explorer.representatives[key]
            classical = index.classical.get(g.number_of_visible)
            screened[key] = (quantum_part_is_free(g) or (classical is not None and
                             underlying_mdag(g).unique_unlabelled_id in classical.latent_free_equivalent_ids))
        return screened[key]

    def expand(roots, names, known):
        """Follow real steps, stopping at gap-free structures or known gaps. Lookups
        link to keys independently reached in the census and never launch expansion."""
        seen = set(roots)
        queue = deque(sorted(seen))
        while queue:
            key = queue.popleft()
            if key in known or gap_free(key):
                continue
            for edge in explorer._expand_one(key, names):
                if edge.trick in names and edge.target not in seen:
                    seen.add(edge.target)
                    queue.append(edge.target)
        return seen

    low = frozenset(elementary) | {'non-growing'}
    low_and_lookups = low | S.LOOKUP_TRICKS
    before = expand(inputs - elementary_known, low, elementary_known)
    before_known = S._reverse_reachable(explorer, seeds, tricks=low_and_lookups)
    depths = fritz_depth_counts(explorer, inputs, seeds, elementary, before_known & inputs)
    print('Proven with at most 0, 1, ... non-growing Fritz steps:', depths)
    print(f'Non-growing closure: {len(inputs & before_known)} proven, {len(before)} keys reached, '
          f'{perf_counter() - start:.2f}s cumulative', flush=True)

    growth_outputs = set()
    for key in sorted(before - before_known):
        if not gap_free(key):
            growth_outputs.update(t.target for t in explorer._expand_one(key, {'growth'}) if t.trick == 'growth')
    print(f'Growth: {len(growth_outputs)} distinct outputs by size '
          f'{dict(sorted(Counter(k[0] for k in growth_outputs).items()))}, '
          f'{perf_counter() - start:.2f}s cumulative', flush=True)

    # 'reducing' includes Fritz itself, e.g. an unsplit predictor dropped at five.
    reductions = frozenset(elementary) | {'reducing'}
    reduced = set()
    for key in sorted(growth_outputs - before_known):
        if not gap_free(key):
            reduced.update(t.target for t in explorer._expand_one(key, reductions) if t.trick in reductions)
    after = expand(reduced, low, before_known)

    # Solve reachability backwards. Never feed a proof that already used growth back
    # into the post-growth goals. Horizontal cycles and lookups obey the same budget.
    reverse = S._reverse_edges(explorer)

    def back(goals, tricks):
        return S._reverse_reachable(explorer, goals, tricks=tricks, reverse=reverse)

    tail = back(seeds, low_and_lookups | reductions)
    # At the growth output, accept a gap already known before growth, or any
    # admissible reduction to a zero-growth proof. A lookup recognises an
    # independently established gap, including one found during the final closure.
    immediate = before_known | {t.source for key in tail for t in reverse.get(key, ())
                                if t.source[0] > t.target[0] or t.trick in S.LOOKUP_TRICKS}
    after_growth = back(immediate, S.LOOKUP_TRICKS)
    can_grow = {t.source for key in after_growth for t in reverse.get(key, ()) if t.trick == 'growth'}
    all_proven = back(tail | can_grow, low_and_lookups | reductions)
    proven = inputs & all_proven
    proven_keys = (all_proven & set(explorer.representatives)) - explorer.lookup_only
    elapsed = perf_counter() - start
    print(f'After reductions and non-growing closure: {len(after)} keys reached, '
          f'{len(proven)} proven, {len(inputs - proven)} open, {elapsed:.2f}s total')
    print('Keys expanded by size:', dict(sorted(Counter(key[0] for key in explorer.applied).items())))
    print('Expanded keys:', len(explorer.applied), 'registered keys including lookups:', len(explorer.representatives))
    return proven, inputs & before_known, proven_keys, explorer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-screen', action='store_true', help='also expand provably gap-free branches')
    parser.add_argument('--cold-cap', type=int, choices=(5, 6),
                        help='time the complete production census in a fresh process, including the classical screen')
    parser.add_argument('--resume-at', type=int, choices=(4, 5))
    args = parser.parse_args()
    if args.cold_cap:
        import resource
        import subprocess
        import sys
        from pathlib import Path
        command = [sys.executable, str(Path(census.__file__).resolve()), '--no-cache', '--max-visible', str(args.cold_cap)]
        if args.resume_at:
            command.extend(['--resume-at', str(args.resume_at)])
        if args.no_screen:
            command.append('--no-screen')
        start = perf_counter()
        subprocess.run(command, check=True)
        print(f'Cold census wall time: {perf_counter() - start:.2f}s')
        print('Peak child RSS (KiB on Linux):', resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
        sys.exit(0)
    index = census.build_index()
    proven, no_growth, proven_keys, explorer = run(index, screen=not args.no_screen)
    # The comparison runs after the experiment. Neither reads or writes the gap cache.
    start = perf_counter()
    baseline, _ = census.run_search(index, use_cache=False, deep=False, engine='semigraphoid', verbose=False)
    print(f'Baseline comparison: {len(baseline.proven)} proven, {perf_counter() - start:.2f}s')
    print('Baseline expanded keys:', len(baseline.explorer.applied),
          'registered keys including lookups:', len(baseline.explorer.representatives))
    print('Non-growing proven set equals baseline:', no_growth == set(baseline.proven))
    baseline_keys = baseline.proven_structure_ids()
    print('Proven input and intermediate keys:', len(proven_keys), 'versus', len(baseline_keys))
    print('Additional proven keys by size:', dict(sorted(Counter(k[0] for k in proven_keys - baseline_keys).items())))
    print('Baseline-only proven keys by size:', dict(sorted(Counter(k[0] for k in baseline_keys - proven_keys).items())))
    print('LP solves (both searches):', entropic_lp.SOLVES[0])
    added = [i for i, g in enumerate(baseline.remaining, 1) if baseline.key(g) in proven]
    print('Newly proven open classes (manuscript numbering):', added)
    print('Baseline classes lost:', sorted(set(baseline.proven) - proven))
    for key in sorted(proven_keys - baseline_keys):
        # Where a non-growing certificate exists, print its full route to a seed.
        chain = explorer.path(key, set(baseline.seed_keys), tricks=frozenset(explorer.tricks)
                              - {'growth'} | S.LOOKUP_TRICKS)
        print('Additional proven structure:', fmt(explorer.representatives[key]))
        if chain is not None:
            print(S.render_certificate(explorer, chain, baseline.seed_keys[chain[-1].target if chain else key]))

    start = perf_counter()
    capped, _ = census.run_search(index, max_visible=4, use_cache=False, deep=False, engine='semigraphoid', verbose=False)
    print(f'Default cascade with a four-visible-node cap: {len(capped.proven)} proven, '
          f'{perf_counter() - start:.2f}s; same input classes as baseline: {capped.proven.keys() == baseline.proven.keys()}')
