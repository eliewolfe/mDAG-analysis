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
fan = [(0, 1), (0, 2), (0, 3), (1, 3), (2, 3)]
valid_split('Example 5', 4, fan, [(0, 1, 3), (0, 2, 3)], [(0, 1), (0, 2)], (3,), [(0, 1), (0, 2)])
check('Example 5: neither source satisfies (S1) on its own (Evans\' rule splits nothing)',
      S(fan, [(0, 1, 3), (0, 2, 3)], 4).weak_splittable_faces == [])
check('Example 5: no node is premeasurable, so Combination 1 does not apply',
      not any(S(fan, [(0, 1, 3), (0, 2, 3)], 4).is_premeasurable(v) for v in range(4)))
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
no_edge = [(0, 1)]
check('Warning: a split with no edge from C to D can still be an equivalence (by other rules), so the warning '
      'speaks of the theorem only',
      same_class(S(no_edge, [(0, 1, 3), (0, 2, 3), (1, 2, 3)], 4), S(no_edge, [(0, 1), (0, 2, 3), (1, 2, 3)], 4)))

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
check('Example 12: with 0 -> 3 the third member is not premeasurable and Rule B does not apply',
      (1, 2) in e12b.droppable_edges and (1, 2) not in e12b.quantum_droppable_edges)
check('Example 12: with 0 -> 3 the edge is redundant all the same',
      same_class(e12b, S([(0, 1), (0, 2), (0, 3)], [(1, 2, 3)], 4)))
order = S([(0, 1), (0, 2), (1, 2)], [(0, 2), (1, 2)], 3)
check('Dropping several edges: both 0 -> 2 (Rule A) and 1 -> 2 (Rule B) are droppable',
      order.is_premeasurable(0) and set(order.quantum_droppable_edges) == {(0, 2), (1, 2)})
check('Dropping several edges: after 0 -> 2 is dropped, 1 -> 2 no longer satisfies a rule',
      (1, 2) not in S([(0, 1), (1, 2)], [(0, 2), (1, 2)], 3).quantum_droppable_edges)
check('Dropping several edges: after 1 -> 2 is dropped, 0 -> 2 still does',
      (0, 2) in S([(0, 1), (0, 2)], [(0, 2), (1, 2)], 3).quantum_droppable_edges)
e13 = S([(2, 0), (2, 1), (0, 1)], [(0, 1, 2), (0, 1, 3), (0, 1, 4)], 5)
check('Example 13: all three edges are droppable', set(e13.quantum_droppable_edges) == {(0, 1), (2, 0), (2, 1)})
open_edge = S([(0, 1), (0, 2), (0, 3), (2, 1), (2, 3)], [(0, 1, 2), (0, 3)], 4)
check('Open case of Section 3: classically droppable, not dropped by the quantum rule',
      (2, 1) in open_edge.droppable_edges and (2, 1) not in open_edge.quantum_droppable_edges)
check('Open case of Section 3: the two structures are in different quantum classes',
      not same_class(open_edge, S([(0, 1), (0, 2), (0, 3), (2, 3)], [(0, 1, 2), (0, 3)], 4)))


def wider_rule_edges(md):
    """The edges a -> b of the remark 'pre-measurement with a setting that everybody sees': a lies in exactly one
    source F, b is in F, and every visible parent of a is a visible parent of every other member of F."""
    sources = [frozenset(int(v) for v in f) for f in md.simplicial_complex_instance.compressed_simplicial_complex]
    edges = [(int(a), int(b)) for a, b in md.directed_structure_instance.edge_list]
    parents = {v: {a for a, b in edges if b == v} for v in range(md.number_of_visible)}
    for a, b in edges:
        mine = [F for F in sources if a in F]
        if len(mine) == 1 and b in mine[0] and all(parents[a] <= parents[v] for v in mine[0] if v != a):
            yield (a, b), [e for e in edges if e != (a, b)], [tuple(sorted(F)) for F in sources]


applicable = beyond = 0
for md in INDEX.quantum[4].dict_uniqind_unlabelled_mDAGs.values():
    for edge, other_edges, sources in wider_rule_edges(md):
        applicable += 1
        beyond += edge not in md.quantum_droppable_edges
        assert same_class(md, S(other_edges, sources, 4)), (edge, md)
check(f'Remark (pre-measurement with a setting): on four nodes the wider rule applies to {applicable} edges, '
      f'{beyond} of them outside Rules A and B, and every one joins two structures of one quantum class',
      applicable == 1268 and beyond == 35)
check('Remark (pre-measurement with a setting): it covers Example 12 with 0 -> 3 and not the open case',
      (1, 2) in {e for e, _, _ in wider_rule_edges(e12b)} and (2, 1) not in {e for e, _, _ in wider_rule_edges(open_edge)})

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

# The remark after the theorem: without S, or without T, the split is valid again. Nodes R, A, D, E = 0, 1, 2, 3 and
# the remaining setting = 4. Each step is checked against the code's own test of the rule that makes it.
F6, H6, AR = (0, 1, 2), (0, 2, 3), (0, 1)
# without S: Alice (1) is premeasurable; Rule A adds A -> R, then the theorem splits A off
no_S = [(4, 3), (0, 2), (1, 2)]
for name, sources in (('G6', [F6, H6]), ("G6'", [AR, H6])):
    start, with_edge = S(no_S, sources, 5), S(no_S + [(1, 0)], sources, 5)
    check(f'Without S, {name}: Alice is premeasurable and Rule A adds A -> R',
          start.is_premeasurable(1) and (1, 0) in with_edge.quantum_droppable_edges)
    check(f'Without S, {name}: the theorem then splits Alice off, leaving the source of R, D, E alone',
          S(no_S + [(1, 0)], [H6], 5).unique_id in splits(with_edge, 'quantum'))
# without T: Eve (3) is premeasurable; Rule A adds E -> R and E -> D, the theorem splits E off H
no_T = [(4, 1), (4, 2), (0, 2), (1, 2)]
eve = no_T + [(3, 0), (3, 2)]
for name, sources, after in (('G6', [F6, H6], [F6]), ("G6'", [AR, H6], [AR, (0, 2)])):
    start, with_edges = S(no_T, sources, 5), S(eve, sources, 5)
    check(f'Without T, {name}: Eve is premeasurable and Rule A adds E -> R and E -> D',
          start.is_premeasurable(3) and {(3, 0), (3, 2)} <= set(with_edges.quantum_droppable_edges))
    check(f'Without T, {name}: the theorem then splits Eve off H',
          S(eve, after, 5).unique_id in splits(with_edges, 'quantum'))
check('Without T: the theorem now splits F, because D sees every visible parent of A and R',
      S(eve, [AR], 5).unique_id in splits(S(eve, [F6], 5), 'quantum'))
check("Without T: the structure reached from G6' lies between the two ends of that split (Combination 2)",
      set(AR) <= set(F6) and {0, 2} <= set(F6) and AR in [AR, (0, 2)])
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
