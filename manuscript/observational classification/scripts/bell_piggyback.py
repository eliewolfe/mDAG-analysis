"""Reproduce the five displayed structures in the Bell piggyback figure and their reach from named seeds."""
from _repo import notation, save
from directed_structures import DirectedStructure
from equivalence_classes import ClassIndex
from growth_census import extend_census
from hypergraphs import Hypergraph
from known_QC_gaps import SEEDS
from only_via_equivalence import StructuresWithoutUpward, search
from quantum_mDAG import QmDAG

X, Y, A, B, U = 0, 1, 3, 2, 4
SEED = 'QG_Bell_Edge_Edge_SettingsC'


def displayed_members():
    """The common visible control U affects both outcomes. Only the links out of X vary."""
    def member(extra_edges, extra_sources):
        edges = [(U, A), (U, B), (Y, B)] + extra_edges
        sources = [(A, B)] + extra_sources
        return QmDAG(DirectedStructure(edges, 5), Hypergraph([], 5), Hypergraph(sources, 5))

    outcome_edge = [(X, A)]
    both_edges = outcome_edge + [(X, Y)]
    settings_source = [(X, Y)]
    local_source = [(X, A)]
    return [
        member(outcome_edge, settings_source),
        member(both_edges, settings_source),
        member(both_edges, []),
        member(both_edges, local_source),
        member([(X, Y)], local_source),
    ]


def main():
    graphs = displayed_members()
    assert len({g.unique_unlabelled_id for g in graphs}) == 5
    assert all(seed.number_of_visible < 5 for seed in SEEDS.values())
    g1, g2, g3, g4, g5 = [g.as_mDAG for g in graphs]
    assert g1.is_premeasurable(X) and (X, Y) in g1.quantum_addable_edges
    assert g1.unique_id in g2.weaker_mDAG_HLP('quantum')
    assert g3.unique_id in g2.generate_weaker_mDAGs_FaceSplitting('quantum')
    assert g3.unique_id in g4.generate_weaker_mDAGs_FaceSplitting('quantum')
    assert g4.is_premeasurable(X) and (X, A) in g4.quantum_droppable_edges
    assert g5.unique_id in g4.weaker_mDAG_HLP('quantum')

    # Check the actual labelled reduction before comparing unlabelled IDs with the repository seed.
    reduced = graphs[0].fix_to_point_distribution_QmDAG(U)
    seed_id = SEEDS[SEED].unique_unlabelled_id
    assert any(child.unique_unlabelled_id == seed_id and dict(params)['classical'] == ((X, Y),)
               for params, child in reduced.degradation_steps())
    out = {
        'node labels': dict(X=X, Y=Y, A=A, B=B, U=U),
        'G1 through G5': [notation(g.as_mDAG) for g in graphs],
        'G1 certificate': [f'fix U ({U})', f'make the settings source ({X},{Y}) classical', SEED],
        'equivalences in figure order': ['Rule A', 'face splitting', 'reverse face splitting', 'Rule A'],
        'search': {},
    }
    # Singleton keys, including for hybrids. No upward transfer, disk proofs or equivalence lookups.
    index = StructuresWithoutUpward(ClassIndex(()))
    for label, seeds, expected in [('common-cause seed', {SEED: SEEDS[SEED]}, [1]),
                                   ('all seeds', SEEDS, [1, 2, 3])]:
        baseline = search(graphs, seeds, index)
        bounded = extend_census(baseline, cap=6, resume_at=5, five_ids=None, screen=False,
                                with_entropic=True, with_split_predictors=True, allow_descendants=False,
                                engine=None, verbose=False)
        results = {}
        for depth, report in [('ten stages', baseline), ('bounded follow-up', bounded)]:
            reached = [i for i, g in enumerate(graphs, 1) if report.key(g) in report.proven]
            assert reached == expected, (label, depth, reached)
            results[depth] = reached
        # Each adjacent pair above is equivalent, so the G1 certificate applies throughout the chain.
        results['with the four displayed equivalences'] = list(range(1, len(graphs) + 1))
        out['search'][label] = results
    return out


if __name__ == '__main__':
    result = main()
    print(result['search'])
    print('saved', save('bell_piggyback.json', result))
