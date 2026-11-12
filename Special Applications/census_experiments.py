"""The experiments that the manuscript quotes beside the four-node census (manuscript/piggybacks.md 1.1, 7.9, 8.10,
9.3, 9.5, 10.2 and B.4). Each is a function that returns its figures; the command line prints them all.

    python "Special Applications/census_experiments.py"          # about two minutes
    python "Special Applications/census_experiments.py" --lp     # also counts LP solves (needs Mosek, five minutes more)

The quick ones are pinned by tests/test_baseline_slow.py (test_census_experiments).
"""
import argparse
import collections
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [p for p in (os.path.dirname(HERE), HERE) if p not in sys.path]

import quantum_mDAG as QM                                  # noqa: E402
import qc_gap_search as search                             # noqa: E402
import proving_QC_Gaps as census                           # noqa: E402
import entropic_lp                                         # noqa: E402
from hypergraphs import Hypergraph                         # noqa: E402
from known_QC_gaps import SEEDS                            # noqa: E402
from open_classes_analysis import fmt                      # noqa: E402
from quantum_mDAG import QmDAG                             # noqa: E402


def census_with_tally(index, **options):
    """Runs the census without the cache. Returns (report, tally), where the tally counts the candidate steps by how
    they were justified: 'parent', 'dsep', 'relabel', 'markov' or 'failed' (manuscript 9.3)."""
    QM.ENTROPIC_STATS.clear()
    QM.ENGINE_DISAGREEMENTS.clear()
    report, _ = census.run_search(index=index, verbose=False, use_cache=False, **options)
    tally = {key[1]: n for key, n in QM.ENTROPIC_STATS.items() if key[0] == 'certificate'}
    return report, tally


def experiment_closure_cost(index):
    """The closure certificate alone (manuscript 7.9): how often the census calls it, on how many of those calls the
    `relabel` target set applies (the target keeps exactly one parent, a facet), and the time spent inside it."""
    spent, calls, applicable = [0.0], [0], [0]
    original = QM.QmDAG._semigraphoid_certificate

    def timed_certificate(self, predictors, kept_parents, predicted, try_markov=True):
        start = time.perf_counter()
        try:
            return original(self, predictors, kept_parents, predicted, try_markov=try_markov)
        finally:
            spent[0] += time.perf_counter() - start
            calls[0] += 1
            if len(predicted) == 1:
                kept = kept_parents[predicted[0]]
                applicable[0] += len(kept) == 1 and min(kept) >= self.number_of_visible

    QM.QmDAG._semigraphoid_certificate = timed_certificate
    try:
        report, tally = census_with_tally(index, engine='semigraphoid')
    finally:
        QM.QmDAG._semigraphoid_certificate = original
    return {'proven': len(report.proven), 'tally': tally, 'closure calls': calls[0],
            'relabel applicable': applicable[0], 'seconds in the closure': spent[0]}


def experiment_descendants(index, baseline=None):
    """Descendants of the target admitted as predictors (manuscript 8.10): the number of checks beyond d-separation,
    and the steps that only a descendant certifies, with the classes they are on."""
    baseline = baseline or census_with_tally(index, engine='semigraphoid')
    report, tally = census_with_tally(index, engine='semigraphoid', allow_descendants=True)
    explorer = report.explorer
    steps = collections.OrderedDict()
    for key in dict.fromkeys(report.input_ids):
        for transition in explorer.edges.get(key, ()):
            if transition.trick != 'Fritz':
                continue
            params = dict(transition.params)
            if params['certificate'] not in ('semigraphoid', 'entropic'):
                continue
            g = explorer.structures[transition.via]
            s, x = params['targets'][0], params['predictor'][0]
            if x in g._descendants(s):
                steps[(fmt(g), s, x, params['target_mode'], params['predictor_mode'],
                       fmt(explorer.structures[transition.produced]))] = key
    classes = set(steps.values())
    return {'proven': len(report.proven), 'same proven classes': set(report.proven) == set(baseline[0].proven),
            'checks beyond d-separation': tally.get('relabel', 0) + tally.get('failed', 0),
            'checks without descendants': baseline[1]['relabel'] + baseline[1]['failed'],
            'steps only a descendant certifies': list(steps),
            'structures': len({step[0] for step in steps}), 'classes': len(classes),
            'class sizes': sorted(len(index.members(key)) for key in classes),
            'classes proven': sorted(key in report.proven for key in classes)}


def experiment_early_exit(index, baseline=None):
    """The closure stages without their early exit (manuscript 8.7, 9.1): the extra checks, and whether the proven
    classes and the database of proven structures change."""
    baseline = baseline or census_with_tally(index, engine='semigraphoid')
    report, tally = census_with_tally(index, engine='semigraphoid', early_exit=False)
    checks = lambda t: t.get('relabel', 0) + t.get('failed', 0)
    return {'checks with the exit': checks(baseline[1]), 'checks without': checks(tally),
            'certified with the exit': baseline[1]['relabel'], 'certified without': tally['relabel'],
            'same proven classes': set(report.proven) == set(baseline[0].proven),
            'same database': report.proven_structure_ids() == baseline[0].proven_structure_ids(),
            'database keys': len(report.proven_structure_ids())}


def experiment_markov_target_set(index):
    """The census with the `markov` target set switched on (manuscript 9.3): the tally of certificates."""
    QM.ENTROPIC_STATS.clear()
    stages = search.default_stages(max_visible=5, lp_markov_target=True, engine='semigraphoid')
    report = search.prove_gaps(census.census_inputs(index), SEEDS, stages=stages, max_visible=5, verbose=False,
                               index=index)
    tally = {key[1]: n for key, n in QM.ENTROPIC_STATS.items() if key[0] == 'certificate'}
    return {'proven': len(report.proven), 'tally': tally, 'certified by markov': tally.get('markov', 0)}


def input_structures(index):
    """Every member structure of every input class."""
    return [member for key in index.input_keys(4) for member in index.members(key)]


def experiment_sub_maximal_deletions(index):
    """How many (sibling predictor, target) pairs delete one, two, three or four parents other than the noise, and
    how many therefore admit a strictly smaller deletion (manuscript 10.2)."""
    by_size, structures, pairs = collections.Counter(), set(), 0
    members = input_structures(index)
    for g in members:
        pools = {s: g.fritz_pool(s) for s in g.visible_nodes}
        _, latent = g.effective_DAG_data
        for x in sorted({x for pool in pools.values() for x in pool}):
            for s in g.fritz_targets((x,), pools=pools):
                deletion = g.fritz_deletion(s, (x,))
                if deletion is None:
                    continue
                _, deleted = deletion
                size = sum(1 for v in deleted if v < g.number_of_visible or latent[v][0] != 'noise')
                pairs += 1
                by_size[size] += 1
                if size >= 2:
                    structures.add(g.unique_unlabelled_id)
    return {'structures': len(members), 'pairs': pairs, 'by parents deleted': dict(sorted(by_size.items())),
            'pairs with a strict sub-deletion': sum(n for size, n in by_size.items() if size >= 2),
            'structures with one': len(structures)}


def experiment_recolourings(index, baseline=None):
    """Every way of making some facets of an input structure classical, at least one staying quantum (manuscript
    9.5): how many distinct structures, how many the elementary tricks prove, and how many of those are not in
    the database of the census."""
    baseline = baseline or census_with_tally(index, engine='semigraphoid')
    distinct = {}
    for g in input_structures(index):
        facets = [tuple(sorted(f)) for f in g.Q_simplicial_complex_instance.as_tuples]
        n = g.number_of_visible
        for mask in range(1, 1 << len(facets)):                     # the facets that stay quantum
            quantum = [f for i, f in enumerate(facets) if mask >> i & 1]
            classical = [f for i, f in enumerate(facets) if not mask >> i & 1]
            h = QmDAG(g.directed_structure_instance, Hypergraph(classical, n), Hypergraph(quantum, n))
            distinct.setdefault(h.unique_unlabelled_id, h)
    structures = list(distinct.values())
    report = search.prove_gaps(structures, SEEDS, tricks=search.elementary_tricks(5), max_visible=5, verbose=False,
                               index=index)
    proven = [h for h in structures if report.key(h) in report.proven]
    database = baseline[0].proven_structure_ids()
    return {'recolourings': len(structures), 'proven by the elementary tricks': len(proven),
            'of these not in the database': sum(1 for h in proven if baseline[0].key(h) not in database),
            'database keys': len(database)}


def only_via_by_stage(report):
    """The "only via" column of manuscript 1.1, split: for each elementary trick (and for the two marginalizations
    together), the classes no longer proven when it is removed from the whole record, how many of those are proven
    through a Fritz step, and the classes lost when only the elementary stage is considered."""
    explorer, seeds = report.explorer, set(report.seed_keys)
    tricks = frozenset(t.trick for transitions in explorer.edges.values() for t in transitions)
    elementary = tricks - {'Fritz'}
    proven = search._fixpoint(explorer, report.input_ids, seeds, tricks)
    proven_elementary = search._fixpoint(explorer, report.input_ids, seeds, elementary)
    through_fritz = proven - proven_elementary
    groups = {'PD': {'PD'}, 'node_stitching': {'node_stitching'}, 'conditioning': {'conditioning'},
              'naive_marginalization': {'naive_marginalization'},
              'teleportation_marginalization': {'teleportation_marginalization'},
              'marginalization (either kind)': {'naive_marginalization', 'teleportation_marginalization'}}
    out = {}
    for name, removed in groups.items():
        lost = proven - search._fixpoint(explorer, report.input_ids, seeds, tricks - removed)
        lost_elementary = proven_elementary - search._fixpoint(explorer, report.input_ids, seeds, elementary - removed)
        out[name] = (len(lost), len(lost & through_fritz), len(lost_elementary))
    return out


def experiment_lp_solves(index, descendants=False):
    """The census with the LP as the engine (manuscript B.4): candidates sent to it, LP solves, time-outs, seconds."""
    entropic_lp.SOLVES[0] = 0
    start = time.time()
    report, tally = census_with_tally(index, engine='lp', allow_descendants=descendants)
    return {'proven': len(report.proven), 'candidates': tally.get('relabel', 0) + tally.get('failed', 0),
            'LP solves': entropic_lp.SOLVES[0], 'time-outs': entropic_lp.TIMEOUTS[0],
            'seconds': round(time.time() - start)}


def print_census_experiments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--lp', action='store_true', help='also run the census with the LP as the engine')
    args = parser.parse_args()
    index = census.build_index(verbose=False)
    baseline = census_with_tally(index, engine='semigraphoid')
    print(f'census: {len(baseline[0].proven)} classes proven; certificates {baseline[1]}')
    print('\n"only via" (1.1): whole record, of which proven through a Fritz step, elementary stage alone')
    for name, counts in only_via_by_stage(baseline[0]).items():
        print(f'  {name}: {counts}')
    for title, result in (
            ('closure cost (7.9)', experiment_closure_cost(index)),
            ('descendants as predictors (8.10)', experiment_descendants(index, baseline)),
            ('no early exit (8.7)', experiment_early_exit(index, baseline)),
            ('markov target set on (9.3)', experiment_markov_target_set(index)),
            ('sub-maximal deletions (10.2)', experiment_sub_maximal_deletions(index)),
            ('recolourings (9.5)', experiment_recolourings(index, baseline))):
        print(f'\n{title}')
        for name, value in result.items():
            if isinstance(value, list) and value and isinstance(value[0], tuple):
                print(f'  {name}:')
                for row in value:
                    print(f'    {row}')
            else:
                print(f'  {name}: {value}')
    if args.lp:
        print('\nLP as the engine (B.4):', experiment_lp_solves(index))
        print('LP as the engine, descendants admitted (8.10):', experiment_lp_solves(index, descendants=True))


if __name__ == '__main__':
    print_census_experiments()
