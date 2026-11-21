"""
Which four-node structures have a QC gap that the piggybacks reach only through quantum observational equivalence?

The census of the repository proves a gap for a quantum equivalence class as soon as one member reaches a seed, and
every member then has the gap. Here the same ten-stage search (qc_gap_search.default_stages; the elementary
reductions, then the Fritz stages) is run on every structure of the input classes separately, so that no step may
replace a structure by an equivalent one, and the result is compared with the census over classes.

Four baselines, from two choices:

* seeds: every seed of known_QC_gaps.SEEDS, or the standard seeds alone (the instrumental scenario, Evans, the
  triangle and the Bell scenario with independent settings; the Bell scenarios with linked settings are left out);
* equivalence: none at all (keys are structures, lookups: degradation), or the classical equivalence only, inside
  the upward rule (a gap passes to a structure that has more edges or sources and the same classical model set).

A structure is reported when its class is proven by the census over classes with the same seeds and the structure
is not proven in the baseline.

With every seed, the two baselines are run a second time with the deeper search that is the repository's default
(growth_census: after the ten stages, one Fritz step that may grow the structure up to six nodes, then reductions),
to check that the extra depth does not reach the reported structures either. Use --depth-one for just the
ten-stage comparison; its output is saved separately in only_via_equivalence_depth_one.json.
"""
import sys
import time
from collections import defaultdict

from _repo import notation, save
from equivalence_classes import ClassIndex
from known_QC_gaps import SEEDS
from qc_gap_search import default_stages, prove_gaps
from quantum_mDAG import upgrade_to_QmDAG

STANDARD_SEEDS = {name: g for name, g in SEEDS.items() if 'Settings' not in name}


class StructuresAsClasses(ClassIndex):
    """A class index in which every quantum class is one structure and a hybrid is never identified with its
    all-quantum version. The classical metagraph is kept, so the upward rule still applies."""
    def __init__(self, base: ClassIndex) -> None:
        self.quantum, self.classical = base.quantum, base.classical
        self._members, self._key_of_mdag, self._mdag_ids_of = {}, {}, {}
        self._upward, self._key_memo, self._representative = {}, {}, {}
        for n, Q in base.quantum.items():
            for uid, md in Q.dict_uniqind_unlabelled_mDAGs.items():
                g = upgrade_to_QmDAG(md)
                key = g.unique_unlabelled_id
                self._members[key] = [g]
                self._representative[key] = g
                self._mdag_ids_of[key] = {uid}
                self._key_of_mdag[(n, uid)] = key

    def key(self, g):
        return g.unique_unlabelled_id


class StructuresWithoutUpward(StructuresAsClasses):
    """The same without the upward rule: no equivalence of either kind is used."""
    def lookup_tricks(self):
        return {}


def search(inputs, seeds, index):
    return prove_gaps(inputs, seeds, stages=default_stages(max_visible=5), max_visible=5, verbose=False, index=index)


def main(deep: bool = True) -> dict:
    t0 = time.time()
    index = ClassIndex((3, 4))
    singles = StructuresAsClasses(index)
    class_keys = index.input_keys(4)
    structures = [g for key in class_keys for g in index.members(key)]
    assert sorted(g.unique_unlabelled_id for g in structures) == sorted(singles.input_keys(4))
    out = {'input classes': len(class_keys), 'structures': len(structures), 'baselines': {}}
    for seed_name, seeds in (('all seeds', SEEDS), ('standard seeds', STANDARD_SEEDS)):
        by_class = search(index.inputs(4), seeds, index)
        proven_classes = set(by_class.proven)
        in_proven_class = [g for g in structures if index.key(g) in proven_classes]
        for eq_name, baseline_index in (('no equivalence', None), ('classical equivalence in the upward rule', singles)):
            report = search(structures, seeds, baseline_index)
            proven = set(report.proven)
            missed = defaultdict(list)
            for g in in_proven_class:
                if g.unique_unlabelled_id not in proven:
                    missed[index.key(g)].append(g)
            groups = []
            for key, gs in sorted(missed.items()):
                members = index.members(key)
                groups.append({'class size': len(members),
                               'reached without equivalence': [notation(m.as_mDAG) for m in members
                                                               if m.unique_unlabelled_id in proven],
                               'reached only through equivalence': [notation(g.as_mDAG) for g in gs]})
            out['baselines'][f'{seed_name}; {eq_name}'] = {
                'classes proven by the census over classes': len(proven_classes),
                'structures in those classes': len(in_proven_class),
                'structures proven one by one': len(proven & {g.unique_unlabelled_id for g in structures}),
                'structures reached only through equivalence': sum(len(gs) for gs in missed.values()),
                'classes they lie in': len(missed),
                'groups': groups}
            print(f"[{time.time() - t0:5.0f}s] {seed_name}; {eq_name}: "
                  f"{out['baselines'][f'{seed_name}; {eq_name}']['structures reached only through equivalence']} "
                  f"structures in {len(missed)} classes", flush=True)
    if deep:
        import proving_QC_Gaps as census
        from _repo import ROOT
        by_class, _ = census.run_search(index, use_cache=False, verbose=False, screen_cache_dir=ROOT / 'cache')
        proven_classes = set(by_class.proven)
        in_proven_class = [g for g in structures if index.key(g) in proven_classes]
        out['deeper search'] = {}
        for eq_name, baseline_index in (('no equivalence', StructuresWithoutUpward(index)),
                                        ('classical equivalence in the upward rule', singles)):
            report, _ = census.run_search(baseline_index, use_cache=False, verbose=False,
                                          screen_cache_dir=ROOT / 'cache')
            proven = set(report.proven)
            missed = [g for g in in_proven_class if g.unique_unlabelled_id not in proven]
            out['deeper search'][f'all seeds; {eq_name}'] = {
                'classes proven by the census over classes': len(proven_classes),
                'structures proven one by one': len(proven & {g.unique_unlabelled_id for g in structures}),
                'reached only through equivalence': [notation(g.as_mDAG) for g in missed]}
            print(f"[{time.time() - t0:5.0f}s] deeper search, all seeds; {eq_name}: "
                  f"{[notation(g.as_mDAG) for g in missed]}", flush=True)
    return out


if __name__ == '__main__':
    depth_one = '--depth-one' in sys.argv
    result = main(deep=not depth_one)
    for name, data in result['baselines'].items():
        print('=' * 100)
        print(name)
        for k, v in data.items():
            if k != 'groups':
                print(f'   {k}: {v}')
        for group in data['groups']:
            print(f"   class of {group['class size']}:")
            for s in group['reached without equivalence']:
                print('        reached:     ', s)
            for s in group['reached only through equivalence']:
                print('        EQUIVALENCE: ', s)
    name = 'only_via_equivalence_depth_one.json' if depth_one else 'only_via_equivalence.json'
    print('saved', save(name, result))
