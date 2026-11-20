"""
Facts about the census of QC gaps that the document quotes and that no other script of this directory prints.

The census is the repository's ten-stage search over quantum equivalence classes (qc_gap_search.prove_gaps with
every seed of known_QC_gaps.SEEDS); the deeper search that is the repository's default proves the same classes.

  * the rows of the table of Section 5.2 for three and for four nodes;
  * the sizes of the open classes;
  * how many quantum classes the classical classes contain, whether those that contain several are entirely
    proven, and whether every open class is the only quantum class of its classical class;
  * for the structures that strong face splitting merges and the quantum rule keeps apart: the chain of steps that
    proves the gap of each (the tricks are named as in qc_gap_search).
"""
from collections import Counter, defaultdict

from _repo import notation, save
from class_counts import components
from equivalence_classes import ClassIndex
from known_QC_gaps import SEEDS
from qc_gap_search import default_stages, prove_gaps
from quantum_mDAG import upgrade_to_QmDAG


def main() -> dict:
    index = ClassIndex((3, 4))
    out = {}
    reports = {}
    for n in (3, 4):
        keys = index.input_keys(n)
        report = prove_gaps(index.inputs(n), SEEDS, stages=default_stages(max_visible=5), max_visible=5,
                            verbose=False, index=index)
        reports[n] = report
        proven = [key for key in keys if key in report.proven]
        open_keys = [key for key in keys if key not in report.proven]
        out[str(n)] = {'input classes': len(keys), 'structures in them': index.all_quantum_structures(keys),
                       'proven classes': len(proven), 'structures in proven classes': index.all_quantum_structures(proven),
                       'open classes': len(open_keys), 'structures in open classes': index.all_quantum_structures(open_keys),
                       'sizes of the open classes': dict(sorted(Counter(len(index.members(key)) for key in open_keys).items()))}
    # quantum classes inside classical classes, four nodes
    Q, C, report = index.quantum[4], index.classical[4], reports[4]
    mdags = Q.dict_uniqind_unlabelled_mDAGs
    key_of = {uid: index.key(upgrade_to_QmDAG(md)) for uid, md in mdags.items()}
    inputs = set(index.input_keys(4))
    inside = defaultdict(set)
    for uid, key in key_of.items():
        inside[int(C.class_index_of[uid])].add(key)
    several = [keys for keys in inside.values() if len(keys) > 1]
    open_keys = inputs - set(report.proven)
    out['classical classes'] = {
        'all': len(inside),
        'with one quantum class': sum(1 for keys in inside.values() if len(keys) == 1),
        'with one quantum class, by status': dict(sorted(Counter(
            ('proven' if key in report.proven else 'open' if key in inputs else 'no gap possible')
            for keys in inside.values() if len(keys) == 1 for key in keys).items())),
        'with several quantum classes': len(several),
        'quantum classes inside those': sum(map(len, several)),
        'sizes of those': dict(sorted(Counter(map(len, several)).items())),
        'of those, every quantum class is an input of the census and proven':
            sum(1 for keys in several if all(key in report.proven for key in keys)),
        'open classes that are the only quantum class of their classical class':
            sum(1 for keys in inside.values() if len(keys) == 1 and next(iter(keys)) in open_keys),
        'open classes': len(open_keys)}
    # the structures that the strong rule merges and the quantum rule keeps apart
    nodes = Q.metagraph_nodes.tolist()
    labels = {version: components(Q, Q._equivalence_edges(
        lambda md, v=version: md.generate_weaker_mDAGs_FaceSplitting(v), f'face splitting ({version})'))[1].tolist()
        for version in ('strong', 'quantum')}
    merged = defaultdict(lambda: defaultdict(list))
    for i, uid in enumerate(nodes):
        merged[labels['strong'][i]][labels['quantum'][i]].append(int(uid))
    groups = []
    for parts in merged.values():
        if len(parts) > 1:
            groups.append([{'structure': notation(mdags[uid]),
                            'proof': ' > '.join(step.trick for step in report.proven[key_of[uid]]),
                            'seed': report.seed_hit[key_of[uid]]}
                           for part in parts.values() for uid in part])
    out['separated by the quantum rule'] = {'groups': groups, 'structures': sum(map(len, groups)),
                                            'all proven': all(row['proof'] for group in groups for row in group)}
    return out


if __name__ == '__main__':
    result = main()
    for n in ('3', '4'):
        print(f'{n} nodes:', {k: v for k, v in result[n].items()})
    print('classical classes on four nodes:', result['classical classes'])
    for group in result['separated by the quantum rule']['groups']:
        print('group:')
        for row in group:
            print(f"   {row['structure']:55s} {row['proof']}  (seed: {row['seed']})")
    print('saved', save('census_facts.json', result))
