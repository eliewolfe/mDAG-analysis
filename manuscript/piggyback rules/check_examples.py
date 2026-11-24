"""Replay the paper's examples and compare the labelled results with the actual TikZ panels.

Run with the repository Python environment. Add --mosek to check the alternate
KPC certificate through the census LP engine as well as the independent HiGHS solve.
The cold upward census and its ablation are reproduced by check_upward.py.
"""
import argparse
from collections import Counter
from itertools import product
from pathlib import Path
import re
import sys

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import vstack

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import entropic_lp as entropy
import semigraphoid as sg
from directed_structures import DirectedStructure
from equivalence_classes import ClassIndex
from gap_cache import GapCache, key_id
from hypergraphs import Hypergraph
from known_QC_gaps import SEEDS
from qc_gap_search import elementary_tricks, degradation, PIGGYBACK_VERSIONS
from quantum_mDAG import QmDAG


def graph(n, edges=(), classical=(), quantum=()):
    return QmDAG(DirectedStructure(list(edges), n), Hypergraph(list(classical), n),
                 Hypergraph(list(quantum), n))


def from_record(record):
    n = len(record['nodes'])
    assert record['nodes'] == list(map(str, range(n))), record
    return graph(n, record['edges'], record['classical_facets'], record['quantum_facets'])


def as_tuple(value):
    return tuple(map(as_tuple, value)) if isinstance(value, list) else value


def fritz_params(target, predictor, deleted, certificate, target_mode='unsplit', predictor_mode='unsplit'):
    return (('targets', (target,)), ('target_mode', target_mode), ('deleted', (deleted,)),
            ('predictor', (predictor,)), ('predictor_mode', predictor_mode), ('certificate', certificate))


# Each chain starts at the labelled member stored in "via". Each child retains its
# surviving original labels in index order, with the split copy appended.
CASES = {
    'deg': ('4,0,0,1755', 'QG_Triangle', [
        ('PD', (('drop', 2),)), ('degradation', (('classical', ((0, 1), (0, 2))),))]),
    'fix': ('4,328,0,107', 'QG_Instrumental_Edge', [('PD', (('drop', 0),))]),
    'naive': ('4,2,0,3163', 'QG_Instrumental_C', [('PD', (('drop', 1),))]),
    'tele': ('4,28,0,107', 'QG_Evans', [('teleportation_marginalization', (('marginalize', 0),))]),
    'stitch': ('4,2,0,3155', 'QG_Instrumental_C', [('PD', (('drop', 0),))]),
    'post': ('4,328,0,163', 'QG_Instrumental_Edge', [('conditioning', (('condition', 0),))]),
    'split': ('4,328,0,3171', 'QG_Instrumental_EdgeC', [
        ('Fritz', fritz_params(1, 0, ("Q{0,1,1'}", "Q{1,3,1'}"), 'parent', target_mode='split')),
        ('teleportation_marginalization', (('marginalize', 0),))]),
    'dsep': ('4,0,0,60855', 'QG_Triangle', [
        ('Fritz', fritz_params(1, 0, ('Q{1,2,3}',), 'dsep'))]),
    'kpc': ('4,20,0,3177', 'QG_Bell_C_Edge', [
        ('Fritz', fritz_params(3, 0, (1, 2), 'semigraphoid', predictor_mode='split'))]),
    # The stored proof fixes A directly. The cold proof follows the upward lookup first.
    'up': ('4,2,0,3255', 'QG_Triangle', [
        ('PD', (('drop', 2),)), ('degradation', (('classical', ((0, 1), (0, 2))),))]),
}


def replay(index, cache, case):
    key, seed, expected = CASES[case]
    entry = cache.entries[key]
    assert tuple(entry['id']) == key_id(key) == index.key(from_record(entry['structure']))
    assert entry['versions'] == {name: PIGGYBACK_VERSIONS[name] for name in ['metagraph', *(t for t, _ in expected)]}
    assert entry['via_cache'] is None and seed in entry['seed'].split('+')
    assert [(s['trick'], as_tuple(s['params'])) for s in entry['chain']] == expected, key
    first = entry['chain'][0]
    members = [from_record(r) for r in [entry['structure'], *entry['class_members']]]
    g, = [member for member in members if member.unique_unlabelled_id == tuple(first['via'])]
    path = [g]
    rules = {**elementary_tricks(), 'degradation': degradation}
    for step, (trick, params) in zip(entry['chain'], expected):
        assert g.unique_unlabelled_id == tuple(step['via']) and index.key(g) == tuple(step['source'])
        if trick == 'Fritz':
            p = dict(params)
            moves = g.fritz_steps(target_mode=p['target_mode'], predictor_mode=p['predictor_mode'],
                                  pool='parents' if p['certificate'] == 'parent' else 'siblings',
                                  engine='semigraphoid')
        else:
            moves = rules[trick](g)
        g, = [child for actual, child in moves if actual == params]
        assert g.unique_unlabelled_id == tuple(step['produced']) and index.key(g) == tuple(step['target'])
        path.append(g)
    assert index.key(g) == index.key(SEEDS[seed])
    assert entry['seed'] == index.seed_keys(SEEDS)[index.key(g)]
    return path


def kpc_checks(g, child, use_mosek):
    # F=0, C=1, D=2, E=3. The kept latent is shared by E,F; the other latent by D,F.
    F, C, D, E = range(4)
    kept, _ = g.fritz_deletion(E, {F})
    alpha, = kept
    nodes, parents = g.lp_structure
    beta, = parents[D] - {C}
    assert parents[E] == {C, D, alpha} and parents[F] == {alpha, beta}
    assert g.fritz_certificate(E, kept, {F}, use_lp=False) is None
    restricted = dict(parents)
    restricted[E] = kept
    assert g._semigraphoid_certificate(frozenset({F}), restricted, (E,), try_markov=False) == 'relabel'
    assert g._restrict_targets({E: kept}).unique_id == child.unique_id

    def holds(model, left, right, given=()):
        return sg.holds(model, sg.mask_of(left), sg.mask_of(right), sg.mask_of(given))

    source = g._semigraphoid_model()
    assert holds(source, [E], [beta], [C, D])
    assert holds(source, [F], [C, D], [beta])
    assert holds(source, [beta], [C])
    observables = sg.dsep_all(len(nodes), sg.parents_to_masks(restricted, len(nodes)), m=4)
    assert holds(observables, [E], [C, D])

    predicted = source.copy()
    sg.add_functional_dependence(predicted, E, 1 << F)
    sg.close(predicted)
    assert not holds(predicted, [E], [beta, C, D])
    unconditional = source.copy()
    unconditional[:4, :4, :16] |= observables[:4, :4, :16]
    sg.close(unconditional)
    assert not holds(unconditional, [F], [C, D], [beta, E])
    sg.add_triplet(predicted, 1 << E, sg.mask_of([C, D]), 0)
    sg.close(predicted)
    assert holds(predicted, [E], [beta, C, D])
    assert holds(predicted, [F], [C, D], [beta, E])

    # Recreate the LP using the paper's nodes and Shannon rows. A nonnegative
    # sum of target CMIs vanishes iff every target vanishes. Homogeneity lets
    # us ask whether a model with the sum >= 1 exists.
    shannon = entropy.elemental_inequalities(len(nodes))
    ncols = shannon.shape[1]
    hypotheses = [row for _, row in entropy.local_markov_rows(parents, nodes)]
    hypotheses.extend(g._entropic_hypotheses(frozenset({F}), restricted, (E,)))
    Aeq = entropy.rows_to_csr(hypotheses, ncols)
    def status(target_parents):
        rows = [row for _, row in entropy.local_markov_rows(target_parents, target_parents)]
        target = entropy.rows_to_csr([entropy.sum_rows(rows)], ncols)
        return linprog(np.zeros(ncols), A_ub=vstack([-shannon, -target]),
                       b_ub=np.r_[np.zeros(shannon.shape[0]), -1], A_eq=Aeq,
                       b_eq=np.zeros(Aeq.shape[0]), method='highs').status

    assert status(restricted) == 0              # A model with the old latent can violate the target.
    fresh = {F: {beta, E}, C: set(), D: {C, beta}, E: set(), beta: set()}
    assert status(fresh) == 2                   # Positive target impossible for the fresh-source DAG.
    if use_mosek:
        assert g._entropic_certificate(frozenset({F}), restricted, (E,), try_markov=False) == 'relabel'
        assert g.fritz_certificate(E, kept, {F}, engine='lp') == 'entropic'
        alternate = fritz_params(E, F, (C, D), 'entropic', predictor_mode='split')
        result, = [out for p, out in g.fritz_steps(predictor_mode='split', engine='lp') if p == alternate]
        assert result.unique_id == child.unique_id


def binary_sources():
    old, fresh = Counter(), Counter()
    for a, b, c in product(range(2), repeat=3):
        d = c ^ b
        old[c, d, a ^ c ^ d, (a, b)] += 1
        fresh[c, d, a, (a ^ b, b)] += 1
    assert old == fresh
    assert Counter((c, d, e) for c, d, e, _ in old) == Counter(product(range(2), repeat=3))
    assert all(e == (a ^ b) for _, _, e, (a, b) in old)
    for a in range(2):
        assert {e for _, _, e, f in old if f[0] == a} == {0, 1}


def split_checks(g, child, endpoint):
    split = g.split_node(1)
    kept, _ = split.fritz_deletion(4, {0})
    assert kept == {0} and split.fritz_certificate(4, kept, {0}, use_lp=False) == 'parent'
    restricted = split._restrict_targets({4: kept})
    assert restricted.marginalize(0).unique_id == child.unique_id
    # Here the output indices are C,D,B', in this order.
    assert endpoint.unique_id == graph(3, [(2, 0), (0, 1)], [(0, 2)], [(0, 1)]).unique_id
    return split, restricted


def joint_prediction_checks(index, tetrahedron):
    # First remove A after predicting the split fields B' and C'.
    def joint_move(g, predictor, targets, mode):
        children = []
        for params, child in g.fritz_steps(target_mode=mode, use_lp=False, max_visible=5):
            step = dict(params)
            if (step['predictor'], step['targets'], step['certificate']) == ((predictor,), targets, 'dsep'):
                children.append(child)
        result, = children
        return result

    # Direct Markov relations can also certify both copies jointly, outside the
    # standard census stages that test just one field with the stronger engines.
    split = tetrahedron.split_node(1).split_node(2)
    _, parents = split.lp_structure
    restricted = dict(parents)
    for target in (4, 5):
        restricted[target], _ = split.fritz_deletion(target, {0})
    assert split._semigraphoid_certificate(frozenset({0}), restricted, (4, 5), try_markov=True) == 'markov'

    work = joint_move(tetrahedron, 0, (1, 2), 'split')
    # Surviving order is B,C,D,B',C'. D now predicts the same two fields.
    assert work.unique_id == graph(5, classical=[(0, 1, 3, 4), (0, 2, 3), (1, 2, 4)],
                                   quantum=[(0, 1, 2)]).unique_id
    bell = joint_move(work, 2, (3, 4), 'unsplit')
    assert bell.unique_id == graph(4, classical=[(0, 2), (1, 3)], quantum=[(0, 1)]).unique_id
    assert index.key(bell) == index.key(SEEDS['QG_Bell_C_C'])


def upward_checks(index, upper, triangle):
    # C,S,A,B here correspond to labels 0,1,2,3 in the census.
    facets = [(0, 1), (0, 2, 3), (1, 2, 3)]
    assert upper.unique_id == graph(4, [(2, 3)], quantum=facets).unique_id
    lower = graph(4, quantum=facets)
    assert index.key(lower) == (4, 0, 0, 1755)
    assert index.classical_class(index.key(upper)) == index.classical_class(index.key(lower))
    params, = [dict(p) for p, rep in index.upward_steps(upper) if index.key(rep) == index.key(lower)]
    assert params == {'above': str(upper.as_mDAG), 'below': str(lower.as_mDAG)}
    assert lower.fix_to_point_distribution_QmDAG(2).unique_id == triangle.unique_id
    return lower


def panel_graphs():
    """Read the visible arrows and every latent-to-visible incidence actually drawn."""
    src = (HERE / 'pictures.tex').read_text()
    calls = re.findall(r'\\Diagram\s*\{([\w-]+)\}\s*\{([^{}]*)\}\s*\{([^{}]*)\}\s*\{([^{}]*)\}', src)
    def fields(group):
        return [tuple(item.strip().split('/')) for item in group.strip().split(',') if item.strip()]

    panels = {}
    for name, vis, lat, wires in calls:
        vv = [node for node, x, y in fields(vis)]
        ll = [node for node, x, y in fields(lat)]
        assert name not in panels and len(set(vv + ll)) == len(vv) + len(ll), name
        recipients = {node: set() for node in ll}
        edges = set()
        for a, b in fields(wires):
            assert b in vv and (a in recipients or a in vv), (name, a, b)
            if a in recipients:
                assert b not in recipients[a], (name, a, b)
                recipients[a].add(b)
            else:
                assert (a, b) not in edges, (name, a, b)
                edges.add((a, b))
        assert all(re.fullmatch(r'[QC]\d+', node) and len(r) >= 2 for node, r in recipients.items()), name
        facets = [Counter(frozenset(r) for node, r in recipients.items() if node.startswith(prefix))
                  for prefix in 'CQ']
        panels[name] = (set(vv), edges, *facets)
    return panels


def check_figure_pairs():
    """Check which panels the article actually uses, including the shared KPC pair."""
    prefixes = {'degradation': 'deg', 'fixing': 'fix', 'naive': 'naive', 'teleportation': 'tele',
                'stitching': 'stitch', 'postselection': 'post', 'splitting': 'split', 'parent': 'parent',
                'dsep': 'dsep', 'semigraphoid': 'kpc', 'entropic': 'kpc', 'upward': 'up'}
    # Read the first four arguments of each call, allowing nested braces in panel titles.
    src = (HERE / 'pictures.tex').read_text()
    found = {}
    for match in re.finditer(r'\\censusPair(?=\s*\{)', src):
        pos = match.end()
        args = []
        for _ in range(4):
            while src[pos].isspace():
                pos += 1
            assert src[pos] == '{'
            start = pos + 1
            depth = 1
            while depth:
                pos += 1
                depth += (src[pos] == '{') - (src[pos] == '}')
            args.append(src[start:pos])
            pos += 1
        name, before, _, after = args
        assert name not in found
        found[name] = (before, after)
    assert found == {k: (p + '-before', p + '-after') for k, p in prefixes.items()}

    main = (HERE / 'piggyback_rules.tex').read_text()
    body = main + ''.join((HERE / (name + '.tex')).read_text()
                          for name in re.findall(r'\\input\{([^}]+)\}', main))
    assert Counter(re.findall(r'\\censusfigure\{([a-z]+)\}', body)) == Counter(prefixes.keys())


def labelled(g, names):
    assert len(names) == len(set(names)) == g.number_of_visible
    edges = {(names[a], names[b]) for a, b in g.directed_structure_instance.edge_list}
    facets = [Counter(frozenset(names[i] for i in facet) for facet in hyp.simplicial_complex_as_sets)
              for hyp in (g.C_simplicial_complex_instance, g.Q_simplicial_complex_instance)]
    return (set(names), edges, *facets)


def check_unique_direct_rule(index, source, target, expected, use_mosek):
    """Try every one-step rule on every member, even when the census found a seed early."""
    from qc_gap_search import default_stages

    # Each Fritz stage contributes its own predictor pool, target and predictor modes,
    # and certification settings. Run the actual generators, without a root/stage cutoff.
    generators = list(index.lookup_tricks().items()) + [('degradation', degradation)]
    engine = 'both' if use_mosek else 'semigraphoid'
    for _, stage_rules, _, _, _ in default_stages(max_visible=6, engine=engine):
        generators.extend(stage_rules.items())
    target_key = index.key(target)
    members = index.members(index.key(source), fallback=source)
    hits = {name for member in members for name, generate in generators
            if any(index.key(child) == target_key for _, child in generate(member))}
    assert hits == {expected}, (index.key(source), target_key, expected, hits)


def main(use_mosek):
    index = ClassIndex((3, 4), verbose=False)
    cache = GapCache.load(seeds=SEEDS, keyed_by='class')
    paths = {case: replay(index, cache, case) for case in CASES}
    # The naive and stitching figures follow recorded alternatives to the cached proofs.
    stitch_input, _ = paths['stitch']
    stitch_params = ((('outcome', 1), ('source', 0)),)
    stitched = dict(stitch_input.node_stitching_steps())[stitch_params]
    assert stitched.unique_unlabelled_id == SEEDS['QG_Evans'].unique_unlabelled_id
    fix_input, _ = paths['fix']
    ordinary = fix_input.marginalize(0, apply_teleportation=False)
    assert ordinary.unique_id == graph(3, [(0, 1), (1, 2)], [(0, 1, 2)], [(1, 2)]).unique_id
    teleport_input, _ = paths['tele']
    ordinary = teleport_input.marginalize(0, apply_teleportation=False)
    assert ordinary.unique_id == graph(3, [(0, 1), (0, 2)], [(0, 1)], [(0, 2)]).unique_id
    post_input, _ = paths['post']
    assert post_input.fix_to_point_distribution_QmDAG(0).unique_id == graph(3, [(0, 1), (1, 2)]).unique_id
    naive_input, _ = paths['naive']
    naive = naive_input.marginalize(2, apply_teleportation=False)
    assert index.key(naive) == (3, 0, 3, 53)
    hybrid_seed = dict(naive.degradation_steps())[(('classical', ((1, 2),)),)]
    assert hybrid_seed.unique_unlabelled_id == SEEDS['QG_Triangle'].unique_unlabelled_id
    all_quantum_triangle = graph(3, quantum=[(0, 1), (0, 2), (1, 2)])
    assert naive_input.marginalize(2).unique_id == all_quantum_triangle.unique_id
    assert naive_input.fix_to_point_distribution_QmDAG(2).unique_id == graph(3, quantum=[(0, 1), (1, 2)]).unique_id
    split, restricted = split_checks(*paths['split'])
    joint_prediction_checks(index, paths['dsep'][0])
    kpc_checks(*paths['kpc'], use_mosek)
    binary_sources()
    upper, triangle, _ = paths['up']
    lower = upward_checks(index, upper, triangle)

    panels = panel_graphs()
    expected = {}
    def pair(prefix, before, before_labels, after, after_labels):
        expected[prefix + '-before'] = labelled(before, before_labels)
        expected[prefix + '-after'] = labelled(after, after_labels)

    recorded_lower, triangle, hybrid = paths['deg']
    assert recorded_lower.unique_id == lower.unique_id
    pair('deg', triangle, 'CSB', hybrid, 'CSB')
    pair('naive', naive_input, 'ABCD', naive, 'ABD')
    pair('stitch', stitch_input, 'ABCD', stitched, 'BCD')
    for key, survivors in [('fix', 'BCD'), ('tele', 'BCD'), ('post', 'BCD'), ('dsep', 'BCD')]:
        before, after = paths[key]
        pair(key, before, 'ABCD', after, survivors)
    before, child, _ = paths['split']
    copy_labels = ('A', 'B', 'C', 'D', 'Bp')
    pair('split', before, 'ABCD', split, copy_labels)
    pair('parent', restricted, copy_labels, child, copy_labels[1:])
    pair('kpc', paths['kpc'][0], 'FCDE', paths['kpc'][1], 'FCDE')
    pair('up', upper, 'CSAB', lower, 'CSAB')
    for rule, before, after in [
        ('degradation', triangle, hybrid),
        ('PD', *paths['fix']),
        ('naive_marginalization', naive_input, naive),
        ('teleportation_marginalization', *paths['tele']),
        ('node_stitching', stitch_input, stitched),
        ('conditioning', *paths['post']),
        ('upward', upper, lower),
    ]:
        check_unique_direct_rule(index, before, after, rule, use_mosek)
    if use_mosek:
        assert entropy.TIMEOUTS[0] == 0, 'An LP timed out in the direct-rule comparison'
    assert set(expected) == set(panels), (expected.keys() - panels.keys(), panels.keys() - expected.keys())
    for panel, spec in expected.items():
        assert spec == panels[panel], (panel, spec, panels[panel])
    check_figure_pairs()
    print(f'{len(paths)} cache records and {len(panels)} labelled panels across 12 figures verified. '
          'Parent, d-separation, semigraphoid and Shannon-LP checks passed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mosek', action='store_true', help='also run the census LP solver (requires a license)')
    main(parser.parse_args().mosek)
