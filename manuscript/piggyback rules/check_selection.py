"""Reproduce the example-selection comparisons in the four-node census.

Run with the repository Python environment. --screen-cache DIR stores only the
classical five-node screen. Each comparison starts from the named gap seeds.
To compare the two stronger certificate engines on all candidates reached
by the standard census, run Special Applications/proving_QC_Gaps.py with
--engine both --no-cache. Add --depth-one --no-early-exit for the 706-candidate
comparison in the Fritz section.
"""
import argparse
from contextlib import ExitStack
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'Special Applications')]

import qc_gap_search as search
from growth_census import five_node_screen
from proving_QC_Gaps import build_index, run_search
from quantum_mDAG import QmDAG


# Expected total and illustrated inputs lost in each fresh search.
PARENT = (4, 328, 0, 3171)
TETRAHEDRON = (4, 0, 0, 60855)
KPC = (4, 20, 0, 3177)
POSTSELECTION = (4, 328, 0, 163)
EXCLUSIONS = {
    'degradation': (380, ()),
    'PD': (269, ((4, 328, 0, 107),)),
    'conditioning': (369, (POSTSELECTION,)),
    'node_stitching': (380, ()),
    'marginalizations': (380, ()),
    'split': (377, (PARENT, (4, 280, 0, 2661), (4, 344, 0, 3171))),
    'parent': (378, (PARENT, (4, 344, 0, 3171))),
    'dsep': (379, (TETRAHEDRON,)),
    'stronger certificates': (379, (KPC,)),
}


def without_graphical_test(graph, s, K, X, use_lp=True, lp_markov_target=False,
                           tally=True, engine=None):
    """Bypass the graphical sufficient condition but still try the standard closure."""
    if set(X) <= K:
        return 'parent'
    if use_lp:
        assert engine in (None, 'semigraphoid')
        _, parents = graph.lp_structure
        kept = dict(parents)
        kept[s] = frozenset(K).intersection(parents)
        if graph._semigraphoid_certificate(frozenset(X), kept, (s,), try_markov=lp_markov_target):
            return 'semigraphoid'
    return None


def remove_rule(rule, patches):
    if rule == 'stronger certificates':
        return {'with_entropic': False}
    if rule == 'dsep':
        patches.enter_context(patch.object(QmDAG, 'fritz_certificate', without_graphical_test))
    elif rule in ('split', 'parent'):
        generate = QmDAG.fritz_steps

        def allowed(graph, **options):
            if rule == 'parent' and options['pool'] == 'parents':
                return
            if rule == 'split' and options['target_mode'] == 'split':
                return
            for params, child in generate(graph, **options):
                if rule == 'split' and options['predictor_mode'] == 'split':
                    # A childless predictor in this mode stays in place without a copy.
                    if set(dict(params)['predictor']) - graph.vis_nodes_with_no_children:
                        continue
                yield params, child

        patches.enter_context(patch.object(QmDAG, 'fritz_steps', allowed))
    elif rule == 'degradation':
        patches.enter_context(patch.object(search, 'degradation', lambda graph: ()))
    else:
        removed = search.MARGINALIZATION_TRICKS if rule == 'marginalizations' else {rule}
        generate = search.elementary_tricks

        def allowed(*args, **kwargs):
            return {name: op for name, op in generate(*args, **kwargs).items() if name not in removed}

        patches.enter_context(patch.object(search, 'elementary_tricks', allowed))
    return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen-cache', type=Path)
    parser.add_argument('--without', choices=EXCLUSIONS, action='append',
                        help='check only the specified omission (repeatable)')
    args = parser.parse_args()
    index = build_index(verbose=False)
    five_ids = five_node_screen(args.screen_cache)

    def census(**options):
        report, _ = run_search(index=index, max_visible=6, resume_at=5, five_ids=five_ids,
                               use_cache=False, verbose=False, **options)
        return set(report.proven)

    baseline = census()
    assert len(baseline) == 380
    for rule in args.without or EXCLUSIONS:
        with ExitStack() as patches:
            proven = census(**remove_rule(rule, patches))
        lost = baseline - proven
        assert not proven - baseline
        count, examples = EXCLUSIONS[rule]
        assert len(proven) == count
        assert set(examples) <= lost
        print(f'Without {rule}: {len(proven)} classes proven. Lost {sorted(lost)}.', flush=True)


if __name__ == '__main__':
    main()
