"""
Every example, warning and open case of the document, checked against the repository's code.

For each structure the script evaluates the code's own tests of the hypotheses (mDAG.weak_splittable_faces,
splittable_faces, quantum_splittable_families, droppable_edges, quantum_droppable_edges, quantum_addable_edges,
is_premeasurable) and, for structures on three or four nodes, whether two structures lie in the same quantum class
and in the same classical class of the metagraphs (equivalence_classes.ClassIndex). A structure with fewer than
three nodes is padded with isolated nodes. The script stops at the first claim of the document that the code does
not confirm.
"""
from _repo import save
from directed_structures import DirectedStructure
from equivalence_classes import ClassIndex
from hypergraphs import Hypergraph
from mDAG_advanced import mDAG

INDEX = ClassIndex((3, 4))
CHECKED = []


def S(edges, sources, n):
    return mDAG(DirectedStructure(edges, n), Hypergraph(sources, n))


def same_class(a, b, kind='quantum'):
    n = a.number_of_visible
    graph = INDEX.quantum[n] if kind == 'quantum' else INDEX.classical[n]
    return graph.class_index_of[a.unique_unlabelled_id] == graph.class_index_of[b.unique_unlabelled_id]


def splits(md, version):
    return set(md.generate_weaker_mDAGs_FaceSplitting(version))


def families(md):
    return {(tuple(sorted(D)), tuple(sorted(tuple(sorted(C)) for C in Cs))) for D, Cs in md.quantum_splittable_families}


def check(name, condition):
    assert condition, name
    CHECKED.append(name)


def valid_split(name, n, edges, before, after, D, parts):
    """An application of the face-splitting theorem: the code finds the family, emits the split, and the two
    structures are in one quantum class."""
    a, b = S(edges, before, n), S(edges, after, n)
    check(f'{name}: the family (D={D}, parts={parts}) is found',
          (tuple(D), tuple(sorted(parts))) in families(a))
    check(f'{name}: the quantum rule emits the split', b.unique_id in splits(a, 'quantum'))
    check(f'{name}: one quantum class', same_class(a, b))


# ---------------------------------------------------------------------------------------------------------------
# Section 2: face splitting
# ---------------------------------------------------------------------------------------------------------------
valid_split('Example 1', 3, [(0, 1)], [(0, 1)], [], (1,), [(0,)])
valid_split('Example 2', 3, [(0, 2), (1, 2)], [(0, 1, 2)], [(0, 1)], (2,), [(0, 1)])
valid_split('Example 3', 3, [(0, 1), (0, 2)], [(0, 1, 2)], [(1, 2)], (1, 2), [(0,)])
valid_split('Example 4', 3, [(0, 1), (0, 2), (1, 2)], [(1, 2)], [], (2,), [(1,)])
fan = [(0, 3), (1, 3), (2, 3)]
valid_split('Example 5', 4, fan, [(0, 1, 3), (0, 2, 3)], [(0, 1), (0, 2)], (3,), [(0, 1), (0, 2)])
check('Example 5: neither source can be split alone under Evans\' rule',
      S(fan, [(0, 1, 3), (0, 2, 3)], 4).weak_splittable_faces == [])
valid_split('Example 6', 4, [(0, 1), (0, 2)], [(0, 1, 2), (1, 2, 3)], [(1, 2, 3)], (1, 2), [(0,)])
bell_edges = [(0, 1), (0, 3), (1, 2)]
valid_split('Example 7', 4, bell_edges, [(0, 3), (2, 3)], [(2, 3)], (3,), [(0,)])

# Combination 1: nodes c, p, d, e = 0, 1, 2, 3
unsplit = S([(0, 2), (1, 2)], [(0, 1, 2), (0, 2, 3)], 4)
split = S([(0, 2), (1, 2)], [(0, 1), (0, 2, 3)], 4)
with_edge = S([(0, 2), (1, 2), (1, 0)], [(0, 1, 2), (0, 2, 3)], 4)
last = S([(0, 2), (1, 2), (1, 0)], [(0, 2, 3)], 4)
check('Combination 1: p is premeasurable', unsplit.is_premeasurable(1) and split.is_premeasurable(1))
split_with_edge = S([(0, 2), (1, 2), (1, 0)], [(0, 1), (0, 2, 3)], 4)
# an edge that Rule A may add is one that Rule A may drop from the structure that has it
check('Combination 1: Rule A adds p -> c',
      (1, 0) in with_edge.quantum_droppable_edges and (1, 0) in split_with_edge.quantum_droppable_edges)
check('Combination 1: the split structure with the edge leads to the same last structure',
      last.unique_id in splits(split_with_edge, 'quantum'))
check('Combination 1: then the theorem splits p off', last.unique_id in splits(with_edge, 'quantum'))
check('Combination 1: (S1) fails for the direct split', split.unique_id not in splits(unsplit, 'quantum')
      and split.unique_id in splits(unsplit, 'strong'))
check('Combination 1: all four structures are one quantum class',
      same_class(unsplit, split) and same_class(unsplit, with_edge) and same_class(unsplit, last))
# Combination 2: the sandwich inside Example 5
check('Combination 2: the half-split structure is in the class of Example 5',
      same_class(S(fan, [(0, 1), (0, 2, 3)], 4), S(fan, [(0, 1), (0, 2)], 4)))

# Warnings of Section 2
instrumental, chain = S([(0, 1), (1, 2)], [(1, 2)], 3), S([(0, 1), (1, 2)], [], 3)
check('Warning (S2): the instrumental source is not splittable', instrumental.splittable_faces == [])
check('Warning (S2): the two structures are not even classically equivalent (in the metagraph)',
      not same_class(instrumental, chain, 'classical'))
check('Warning: the source of the Bell scenario is not splittable',
      S([(0, 2), (1, 3)], [(2, 3)], 4).splittable_faces == [])

# ---------------------------------------------------------------------------------------------------------------
# Section 3: edge dropping
# ---------------------------------------------------------------------------------------------------------------
e8 = S([(0, 1)], [(0, 1)], 3)
check('Example 8: Rule A', e8.is_premeasurable(0) and e8.quantum_droppable_edges == [(0, 1)])
check('Example 8: edge with source, edge alone and source alone are one quantum class',
      same_class(e8, S([], [(0, 1)], 3)) and same_class(e8, S([(0, 1)], [], 3)))
check('Warning: with neither the edge nor the source the class is another one, classically too',
      not same_class(e8, S([], [], 3), 'classical'))
e9 = S([(0, 1), (3, 2)], [(0, 1, 2)], 4)
check('Example 9: Rule A with an arbitrary third member',
      e9.is_premeasurable(0) and not e9.is_premeasurable(2) and (0, 1) in e9.quantum_droppable_edges
      and same_class(e9, S([(3, 2)], [(0, 1, 2)], 4)))
H = S([(0, 1), (1, 2)], [(0, 3), (2, 3)], 4)
G = S(bell_edges, [(0, 3), (2, 3)], 4)
check('Example 10: Rule A adds 0 -> 3', H.is_premeasurable(0) and (0, 3) in G.quantum_droppable_edges and same_class(H, G))
e11 = S([(2, 0), (2, 1), (0, 1)], [(0, 1)], 3)
check('Example 11: Rule B, Rule A does not apply',
      not e11.is_premeasurable(0) and e11.quantum_droppable_edges == [(0, 1)]
      and same_class(e11, S([(2, 0), (2, 1)], [(0, 1)], 3)))
e12 = S([(0, 1), (0, 2), (1, 2)], [(1, 2, 3)], 4)
check('Example 12: Rule B with a premeasurable third member',
      e12.is_premeasurable(3) and (1, 2) in e12.quantum_droppable_edges
      and same_class(e12, S([(0, 1), (0, 2)], [(1, 2, 3)], 4)))
e12b = S([(0, 1), (0, 2), (1, 2), (0, 3)], [(1, 2, 3)], 4)
check('Example 12: with 0 -> 3 the third member is not premeasurable and the rule does not apply',
      (1, 2) in e12b.droppable_edges and (1, 2) not in e12b.quantum_droppable_edges)
e13 = S([(2, 0), (2, 1), (0, 1)], [(0, 1, 2), (0, 1, 3), (0, 1, 4)], 5)
check('Example 13: all three edges are droppable', set(e13.quantum_droppable_edges) == {(0, 1), (2, 0), (2, 1)})
open_edge = S([(0, 1), (0, 2), (0, 3), (2, 1), (2, 3)], [(0, 1, 2), (0, 3)], 4)
check('Open case of Section 3: classically droppable, not dropped by the quantum rule',
      (2, 1) in open_edge.droppable_edges and (2, 1) not in open_edge.quantum_droppable_edges)

# ---------------------------------------------------------------------------------------------------------------
# Section 4: the counterexamples
# ---------------------------------------------------------------------------------------------------------------
settings5 = S([(3, 0), (3, 1), (4, 2), (0, 1)], [(0, 1, 2)], 5)          # A, B, C, S, T = 0, 1, 2, 3, 4
check('Theorem (edge rule with settings): the classical rule drops A -> B, the quantum rules do not',
      (0, 1) in settings5.droppable_edges and (0, 1) not in settings5.quantum_droppable_edges)
plus = S([(0, 3)], [(0, 1, 3), (0, 2, 3), (1, 2)], 4)                    # A, B, C, D = 0, 1, 2, 3
minus = S([], [(0, 1, 3), (0, 2, 3), (1, 2)], 4)
check('Theorem (edge rule on four nodes): classically droppable, not quantum',
      (0, 3) in plus.droppable_edges and (0, 3) not in plus.quantum_droppable_edges)
check('Theorem (edge rule on four nodes): one classical class, two quantum classes',
      same_class(plus, minus, 'classical') and not same_class(plus, minus))
six_edges = [(4, 1), (4, 2), (5, 3), (0, 2), (1, 2)]                     # R, A, D, E, S, T = 0, 1, 2, 3, 4, 5
six, six_split = S(six_edges, [(0, 1, 2), (0, 2, 3)], 6), S(six_edges, [(0, 1), (0, 2, 3)], 6)
check('Theorem (six nodes): an instance of the moderate rule', six.splittable_faces == [(frozenset({0, 1}), frozenset({2}))]
      and six_split.unique_id in splits(six, 'moderate') and six_split.unique_id in splits(six, 'strong'))
check('Theorem (six nodes): not an instance of Evans\' rule, and not emitted by the quantum rule',
      six.weak_splittable_faces == [] and splits(six, 'quantum') == set())
check('Theorem (six nodes): neither A nor E is premeasurable, R has two sources',
      not six.is_premeasurable(1) and not six.is_premeasurable(3) and not six.is_premeasurable(0))
groups = [
    ([(0, 1), (0, 2), (0, 3), (2, 3)], [(0, 1, 3), (0, 2, 3)], [(0, 1, 3), (0, 2)]),
    ([(0, 1), (0, 2), (1, 2), (2, 3)], [(0, 1, 2), (0, 2, 3)], [(0, 1), (0, 2, 3)]),
    ([(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)], [(0, 1, 2), (0, 2, 3)], [(0, 1), (0, 2, 3)]),
    ([(1, 3), (2, 3)], [(0, 1, 3), (0, 2, 3), (1, 2, 3)], [(0, 1, 3), (0, 2, 3), (1, 2)]),
    ([(0, 2), (1, 2), (2, 3)], [(0, 1, 2), (0, 2, 3), (1, 2, 3)], [(0, 1), (0, 2, 3), (1, 2, 3)]),
]
for number, (edges, before, after) in enumerate(groups, 1):
    a, b = S(edges, before, 4), S(edges, after, 4)
    check(f'Open group {number}: the strong rule splits, the quantum rule does not',
          b.unique_id in splits(a, 'strong') and b.unique_id not in splits(a, 'quantum'))
    check(f'Open group {number}: one classical class, separate quantum classes',
          same_class(a, b, 'classical') and not same_class(a, b))

# ---------------------------------------------------------------------------------------------------------------
# Section 5: the class of the Bell scenario with linked settings
# ---------------------------------------------------------------------------------------------------------------
S0 = S(bell_edges, [(2, 3)], 4)
G1 = S(bell_edges, [(0, 1), (2, 3)], 4)
G2 = S([(0, 3), (1, 2)], [(0, 1), (2, 3)], 4)
members = [S0, G1, G2, G, H]
check('Bell class: five members, one quantum class',
      all(same_class(S0, m) for m in members)
      and len(INDEX.quantum[4].class_of(S0.unique_unlabelled_id)) == 5)
check('Bell class: S and G\' by the theorem (source on 0 -> 1)', S0.unique_id in splits(G1, 'quantum'))
check('Bell class: G\' and G\'\' by Rule A (edge 0 -> 1)', (0, 1) in G1.quantum_droppable_edges)
check('Bell class: S and G by the theorem (source on 0 -> 3)', S0.unique_id in splits(G, 'quantum'))
check('Bell class: G and H by Rule A (edge 0 -> 3)', (0, 3) in G.quantum_droppable_edges)

if __name__ == '__main__':
    for line in CHECKED:
        print('ok  ', line)
    print(f'{len(CHECKED)} claims confirmed')
    print('saved', save('check_examples.json', CHECKED))
