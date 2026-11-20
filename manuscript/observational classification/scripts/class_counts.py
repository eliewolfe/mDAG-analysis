"""
How many proven quantum equivalence classes each face-splitting rule gives, and what the quantum-safe rule separates.

For n = 3 and 4 (and 5 with --five; several minutes and a few GB) the script builds the quantum metagraph of the
repository (structural dominance, the HLP edge rule in its pre-measurement form, face splitting) and recomputes its
strongly connected components with the face-splitting edges of each version of the rule in turn:

  weak      Proposition 5 of arXiv:2502.07891 (Evans): no node of C in another source
  quantum   the families of sources C_i u D that no other source touches (mDAG.quantum_splittable_families):
            what the repository's quantum metagraph uses, and what the document proves
  moderate  Proposition 7: a node of C may lie in other sources that contain D
  strong    Proposition 8: the same for several C_i at once; what the classical metagraph uses

It also lists the groups of structures that the strong rule merges and the quantum rule keeps apart, with the
status of each structure in the census of QC gaps when --census is given.
"""
import sys
from collections import defaultdict

import numpy as np
from scipy import sparse

from _repo import notation, save
from metagraph_advanced import ClassicalMetagraph, QuantumMetagraph

VERSIONS = ('weak', 'quantum', 'moderate', 'strong')


def components(Q, face_splitting_edges):
    graph = Q._csr_from_edges(Q.structural_dominances, Q.hlp_edge_list, face_splitting_edges)
    return sparse.csgraph.connected_components(graph, directed=True, connection='strong')


def count(n: int, details: bool) -> dict:
    Q = QuantumMetagraph(n, verbose=False)
    C = ClassicalMetagraph.from_quantum(Q)
    nodes = Q.metagraph_nodes
    latent_free = C.latent_free_equivalent_ids
    out = {'mDAGs': int(len(nodes)), 'classical classes': len(C.equivalence_classes_as_ids),
           'classical classes with a latent-free DAG': sum(1 for cls in C.equivalence_classes_as_ids if cls & latent_free),
           'HLP edges (quantum, classical)': [int(len(Q.hlp_edge_list)), int(len(C.hlp_edge_list))],
           'versions': {}}
    labels = {}
    for version in VERSIONS:
        edges = Q._equivalence_edges(lambda md, v=version: md.generate_weaker_mDAGs_FaceSplitting(v),
                                     f'face splitting ({version})')
        number, labels[version] = components(Q, edges)
        groups = defaultdict(set)
        for i, label in enumerate(labels[version].tolist()):
            groups[label].add(int(nodes[i]))
        inputs = [g for g in groups.values() if not g & latent_free]
        out['versions'][version] = {'face-splitting edges': int(len(edges)), 'quantum classes': int(number),
                                    'input classes': len(inputs), 'structures in input classes': sum(map(len, inputs))}
    assert out['versions']['quantum']['quantum classes'] == len(Q.equivalence_classes_as_ids)
    assert np.array_equal(Q.face_splitting_edge_list,
                          Q._equivalence_edges(lambda md: md.generate_weaker_mDAGs_FaceSplitting('quantum'), 'check'))
    # how many quantum classes each classical class contains (the repository's rule), and the same for the classical
    # classes that hold the inputs of the census (no member classically equivalent to a latent-free DAG)
    per_classical = defaultdict(set)
    for i, uid in enumerate(nodes.tolist()):
        per_classical[int(C.class_index_of[int(uid)])].add(int(labels['quantum'][i]))
    histogram, histogram_inputs = defaultdict(int), defaultdict(int)
    for index, cls in enumerate(C.equivalence_classes_as_ids):
        histogram[len(per_classical[index])] += 1
        if not cls & latent_free:
            histogram_inputs[len(per_classical[index])] += 1
    out['classical classes by number of quantum classes inside'] = dict(sorted(histogram.items()))
    out['of which hold inputs of the census'] = dict(sorted(histogram_inputs.items()))
    if details:
        mdags = Q.dict_uniqind_unlabelled_mDAGs
        strong, quantum = labels['strong'].tolist(), labels['quantum'].tolist()
        merged = defaultdict(lambda: defaultdict(list))
        for i, uid in enumerate(nodes.tolist()):
            merged[strong[i]][quantum[i]].append(int(uid))
        out['separated groups'] = [
            {'classical class': int(C.class_index_of[next(iter(next(iter(parts.values()))))]),
             'quantum classes': [[notation(mdags[uid]) for uid in sorted(part)] for part in parts.values()]}
            for parts in merged.values() if len(parts) > 1]
    return out


def main(argv) -> dict:
    sizes = (3, 4, 5) if '--five' in argv else (3, 4)
    result = {str(n): count(n, details=(n == 4)) for n in sizes}
    return result


if __name__ == '__main__':
    result = main(sys.argv[1:])
    for n, data in result.items():
        print(f"{n} nodes: {data['mDAGs']} mDAGs, {data['classical classes']} classical classes, "
              f"HLP edges quantum/classical {data['HLP edges (quantum, classical)']}")
        for version, row in data['versions'].items():
            print(f"   {version:>9}: {row['face-splitting edges']:7d} edges, {row['quantum classes']:7d} quantum classes, "
                  f"{row['input classes']:7d} input classes with {row['structures in input classes']} structures")
        print('   classical classes by number of quantum classes inside:',
              data['classical classes by number of quantum classes inside'],
              '; holding inputs:', data['of which hold inputs of the census'])
        for group in data.get('separated groups', ()):
            print('   merged by the strong rule, apart under the quantum rule:')
            for part in group['quantum classes']:
                print('      ', ' | '.join(part))
    name = 'class_counts_with_five.json' if '5' in result else 'class_counts.json'
    print('saved', save(name, result))
