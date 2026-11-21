"""
How many proven quantum equivalence classes each face-splitting rule gives, and what the quantum-safe rule separates.

For n = 3 and 4 (and 5 with --five; several minutes and a few GB) the script builds the quantum metagraph of the
repository (structural dominance, the HLP edge rule in its pre-measurement form, face splitting) and recomputes its
strongly connected components with the face-splitting edges of each version of the rule in turn:

  weak      Proposition 5 of arXiv:2502.07891 (Evans): no node of C in another source
  quantum   the families of sources C_i u D such that no other source contains a node of a C_i
            (mDAG.quantum_splittable_families):
            what the repository's quantum metagraph uses, and what the document proves
  moderate  Proposition 7: a node of C may lie in other sources that contain D
  strong    Proposition 8: the same for several C_i at once; what the classical metagraph uses

It also lists the groups of structures that the strong rule merges and the quantum rule keeps apart (their status
in the census of QC gaps is printed by census_facts.py), the quantum classes that fall apart when only one
source may be split at a time (the weak rule), which are the places where sources must be split together, and what
becomes of the instances of the classical edge rule: for every edge that satisfies the classical hypothesis
(mDAG.droppable_edges), whether the wider Rule A or Rule B covers it (mDAG.quantum_droppable_edges), and if not, whether the
structure with the edge and the one without it are in the same quantum class anyway.
"""
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy import sparse

from _repo import notation, save
from directed_structures import DirectedStructure
from hypergraphs import Hypergraph
from mDAG_advanced import mDAG
from metagraph_advanced import ClassicalMetagraph, QuantumMetagraph

VERSIONS = ('weak', 'quantum', 'moderate', 'strong')


def components(Q, face_splitting_edges):
    graph = Q._csr_from_edges(Q.structural_dominances, Q.hlp_edge_list, face_splitting_edges)
    return sparse.csgraph.connected_components(graph, directed=True, connection='strong')


def edge_rule_instances(Q, n: int) -> dict:
    """The edges of n-node structures that satisfy the classical hypothesis of the edge rule, sorted by what the
    quantum rules make of them."""
    counts = {'instances': 0, 'covered by the wider Rule A or Rule B': 0, 'not covered, same quantum class by other rules': 0,
              'not covered, different quantum classes': 0}
    examples = {}
    edge_masks = {(a, b): DirectedStructure([(a, b)], n).as_integer
                  for a in range(n) for b in range(n) if a != b}
    for uid, md in Q.dict_uniqind_unlabelled_mDAGs.items():
        quantum = set(md.quantum_droppable_edges)
        for edge in md.droppable_edges:
            counts['instances'] += 1
            if edge in quantum:
                counts['covered by the wider Rule A or Rule B'] += 1
                continue
            # The edge is present, so clearing its bit gives the labelled directed part after the drop.
            weaker_directed = md.directed_structure_instance.as_integer ^ edge_masks[edge]
            other = Q.canonical_forms.canonical(weaker_directed, md.simplicial_complex_instance.as_integer)
            with_edge, without = Q.class_index_of[uid], Q.class_index_of[other]
            if with_edge == without:
                counts['not covered, same quantum class by other rules'] += 1
            else:
                counts['not covered, different quantum classes'] += 1
                pair = (min(int(with_edge), int(without)), max(int(with_edge), int(without)))
                if pair not in examples:
                    examples[pair] = f'{notation(md)}, edge {edge[0]}→{edge[1]}'
        if Q.compact:
            md.release_cache()
    # the pair of classes that the four-node theorem of the note proves different (nodes A, C, S, B = 0, 1, 2, 3)
    theorem = None
    if n == 4:
        sources = [(0, 1, 3), (0, 2, 3), (1, 2)]
        plus = mDAG(DirectedStructure([(0, 3)], 4), Hypergraph(sources, 4))
        minus = mDAG(DirectedStructure([], 4), Hypergraph(sources, 4))
        i, j = int(Q.class_index_of[plus.unique_unlabelled_id]), int(Q.class_index_of[minus.unique_unlabelled_id])
        theorem = (min(i, j), max(i, j)) in examples
    counts['pairs of quantum classes that such an edge would join'] = len(examples)
    counts['the pair of the four-node counterexample is one of them'] = theorem
    counts['one instance for each pair'] = sorted(examples.values())
    return counts


def count(n: int, details: bool) -> dict:
    Q = QuantumMetagraph(n, verbose=False)
    C = ClassicalMetagraph.from_quantum(Q)
    nodes = Q.metagraph_nodes
    classical = C.class_labels
    num_classical = int(classical.max()) + 1
    has_latent_free_dag = np.zeros(num_classical, dtype=bool)
    for dag in C.latent_free_DAGs_unlabelled:
        has_latent_free_dag[C.class_index_of[dag.unique_unlabelled_id]] = True
    latent_free_nodes = has_latent_free_dag[classical]
    out = {'mDAGs': int(len(nodes)), 'classical classes': num_classical,
           'classical classes with a latent-free DAG': int(has_latent_free_dag.sum()),
           'HLP edges (quantum, classical)': [int(len(Q.hlp_edge_list)), int(len(C.hlp_edge_list))],
           'versions': {}}
    labels = {}
    for version in VERSIONS:
        edges = Q._equivalence_edges(lambda md, v=version: md.generate_weaker_mDAGs_FaceSplitting(v),
                                     f'face splitting ({version})')
        number, labels[version] = components(Q, edges)
        if version == 'quantum':
            assert np.array_equal(Q.face_splitting_edge_list, edges)
        inputs = np.ones(number, dtype=bool)
        inputs[labels[version][latent_free_nodes]] = False
        out['versions'][version] = {'face-splitting edges': int(len(edges)), 'quantum classes': int(number),
                                    'input classes': int(inputs.sum()),
                                    'structures in input classes': int(inputs[labels[version]].sum())}
    assert out['versions']['quantum']['quantum classes'] == int(Q.class_labels.max()) + 1
    # how many quantum classes each classical class contains (the repository's rule), and the same for the classical
    # classes that hold the inputs of the census (no member classically equivalent to a latent-free DAG)
    _, representatives = np.unique(labels['quantum'], return_index=True)
    parent_class = classical[representatives]
    assert np.array_equal(parent_class[labels['quantum']], classical)
    quantum_per_classical = np.bincount(parent_class, minlength=num_classical)
    histogram = Counter(quantum_per_classical.tolist())
    histogram_inputs = Counter(quantum_per_classical[~has_latent_free_dag].tolist())
    out['classical classes by number of quantum classes inside'] = dict(sorted(histogram.items()))
    out['of which hold inputs of the census'] = dict(sorted(histogram_inputs.items()))
    out['classical edge rule'] = edge_rule_instances(Q, n)
    # what each classical rule merges when it alone replaces its quantum counterpart
    quantum_fs = Q.face_splitting_edge_list
    graph = Q._csr_from_edges(Q.structural_dominances, C.hlp_edge_list, quantum_fs)
    out['classes with the classical edge rule and the face splitting of the note'] = int(
        sparse.csgraph.connected_components(graph, directed=True, connection='strong')[0])
    out['classes with the wider Rule A, Rule B and strong face splitting'] = out['versions']['strong']['quantum classes']
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
        # the quantum classes that the weak rule (one source at a time) does not hold together
        weak = labels['weak'].tolist()
        inside = defaultdict(lambda: defaultdict(list))
        for i, uid in enumerate(nodes.tolist()):
            inside[quantum[i]][weak[i]].append(int(uid))
        out['quantum classes that need several sources split together'] = [
            {'structures': sum(map(len, parts.values())), 'classes under the weak rule': len(parts),
             'members': sorted(notation(mdags[uid]) for part in parts.values() for uid in part)}
            for parts in inside.values() if len(parts) > 1]
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
        print('   classical edge rule:', {k: v for k, v in data['classical edge rule'].items()
                                         if k != 'one instance for each pair'})
        print('   classes with the classical edge rule and the face splitting of the note:',
              data['classes with the classical edge rule and the face splitting of the note'],
              '; with the wider Rule A, Rule B and strong face splitting:',
              data['classes with the wider Rule A, Rule B and strong face splitting'])
        for group in data.get('separated groups', ()):
            print('   merged by the strong rule, apart under the quantum rule:')
            for part in group['quantum classes']:
                print('      ', ' | '.join(part))
        for group in data.get('quantum classes that need several sources split together', ()):
            print(f"   one quantum class of {group['structures']} structures, "
                  f"{group['classes under the weak rule']} classes under the weak rule:")
            for member in group['members']:
                print('      ', member)
    name = 'class_counts_with_five.json' if '5' in result else 'class_counts.json'
    print('saved', save(name, result))
