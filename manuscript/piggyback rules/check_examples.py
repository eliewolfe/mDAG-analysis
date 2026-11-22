"""Check the labelled graphs and the hand calculations in the illustrated note."""
from collections import Counter
from itertools import product
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from directed_structures import DirectedStructure
from hypergraphs import Hypergraph
from known_QC_gaps import QG_Bell_C_C, QG_Bell_C_Edge, QG_Instrumental_C, QG_Instrumental_EdgeC
from quantum_mDAG import QmDAG
import semigraphoid as sg


def graph(n, edges=(), quantum=(), classical=()):
    return QmDAG(DirectedStructure(list(edges), n), Hypergraph(list(classical), n),
                 Hypergraph(list(quantum), n))


def same(actual, expected):
    """Node order is the surviving original order with copies appended."""
    assert actual.unique_id == expected.unique_id, (actual, expected)


def fritz(g, predictor, targets, certificate, **modes):
    for params, child in g.fritz_steps(engine="semigraphoid", **modes):
        p = dict(params)
        if (p["predictor"], p["targets"], p["certificate"]) == ((predictor,), targets, certificate):
            return child
    raise AssertionError((g, predictor, targets, certificate, modes))


def elementary():
    g = graph(4, [(0, 2), (0, 3)], [(0, 1, 2), (0, 1, 3), (2, 3)])
    same(g.fix_to_point_distribution_QmDAG(0), graph(3, quantum=[(0, 1), (0, 2), (1, 2)]))
    g = graph(4, [(0, 1), (1, 2)], [(0, 1, 3), (0, 2)])
    same(g.marginalize(0, apply_teleportation=False), graph(3, [(0, 1)], [(0, 2)], [(0, 1)]))
    same(g.marginalize(0), graph(3, [(0, 1)], [(0, 2), (0, 1)]))
    g = graph(4, [(2, 3)], [(0, 2), (0, 3), (1, 2)])
    assert g.conditioning_is_justified(0)
    same(g.condition(0), graph(3, [(1, 2)], [(0, 1), (1, 2)]))
    bad = graph(5, [(1, 2), (3, 2)], classical=[(0, 1), (3, 4)])
    assert not bad.conditioning_is_justified(2)
    bell = graph(4, [(0, 2), (1, 3)], [(2, 3)])
    same(bell.node_stitching(2, 1), graph(3, [(0, 1), (1, 2)], [(1, 2)]))


def predictions():
    triangle = graph(3, quantum=[(0, 1)], classical=[(0, 2), (1, 2)])
    bell = fritz(triangle, 2, (0, 1), "dsep", target_mode="split")
    assert bell.unique_unlabelled_id == QG_Bell_C_C.unique_unlabelled_id
    tetrahedron = graph(4, quantum=[(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
    same(fritz(tetrahedron, 0, (1,), "dsep"), graph(3, quantum=[(1, 2)], classical=[(0, 1), (0, 2)]))
    # F=0, C=1, D=2, E=3. Test each graphical independence used in the hand proof.
    kpc = graph(4, [(1, 2), (1, 3), (2, 3)], [(0, 3), (0, 2)])
    kept, _ = kpc.fritz_deletion(3, {0})
    assert kpc.fritz_certificate(3, kept, {0}, use_lp=False) is None
    bell = fritz(kpc, 0, (3,), "semigraphoid", predictor_mode="split")
    assert bell.unique_unlabelled_id == QG_Bell_C_Edge.unique_unlabelled_id
    nodes, parents = kpc.lp_structure
    b, = parents[2] - {1}
    statements = kpc._semigraphoid_model()
    def holds(model, left, right, given=()):
        return sg.holds(model, sg.mask_of(left), sg.mask_of(right), sg.mask_of(given))
    assert holds(statements, [3], [b], [1, 2])
    assert holds(statements, [0], [1, 2], [b])
    assert holds(statements, [b], [1])
    restricted = dict(parents)
    restricted[3] = kept
    visible = sg.dsep_all(len(nodes), sg.parents_to_masks(restricted, len(nodes)), m=4)
    assert holds(visible, [3], [1, 2])
    # P1 + P2 alone leave the fresh source's required independence unproved.
    predicted = statements.copy()
    sg.add_functional_dependence(predicted, 3, 1 << 0)
    sg.close(predicted)
    assert not holds(predicted, [3], [b, 1, 2])

    # Even all observable independences of the candidate cannot replace P2.
    unpredicted = statements.copy()
    unpredicted[:4, :4, :1 << 4] |= visible[:4, :4, :1 << 4]
    sg.close(unpredicted)
    assert not holds(unpredicted, [0], [1, 2], [b, 3])

    # With P2, the single displayed observable independence suffices.
    sg.add_triplet(predicted, 1 << 3, sg.mask_of([1, 2]), 0)
    sg.close(predicted)
    assert holds(predicted, [3], [b, 1, 2])
    assert holds(predicted, [0], [1, 2], [b, 3])


def chains():
    g = graph(4, [(0, 1), (1, 2), (2, 3)], [(0, 1), (0, 2), (1, 3)])
    child = fritz(g, 0, (1,), "parent", pool="parents", target_mode="split")
    same(child, graph(4, [(0, 1), (1, 2), (3, 1)], [(0, 1), (0, 2)], [(0, 1, 3)]))
    assert child.marginalize(0).unique_unlabelled_id == QG_Instrumental_EdgeC.unique_unlabelled_id

    g = graph(4, [(0, 1), (1, 2)], [(0, 2), (0, 3), (1, 3)])
    child = fritz(g, 0, (3,), "dsep", predictor_mode="split", target_mode="split")
    same(child, graph(5, [(0, 1), (1, 2)], [(0, 2), (0, 3), (1, 3)],
                      [(0, 1, 2), (0, 1, 3, 4)]))
    assert child.conditioning_is_justified(3)
    assert child.condition(3).marginalize(0).unique_unlabelled_id == QG_Instrumental_C.unique_unlabelled_id


def binary_sources():
    old, fresh = Counter(), Counter()
    for a, b, c in product(range(2), repeat=3):
        d = c ^ b
        old[c, d, a ^ c ^ d, (a, b)] += 1
        fresh[c, d, a, (a ^ b, b)] += 1
    assert old == fresh
    for c, d in product(range(2), repeat=2):
        for e in range(2):
            assert sum(n for (cc, dd, ee, _), n in old.items() if (cc, dd, ee) == (c, d, e)) == 1
    for (_, _, e, (a, b)) in old:
        assert e == a ^ b   # F determines E, even though A alone does not.
    for a in range(2):
        assert {e for _, _, e, f in old if f[0] == a} == {0, 1}


if __name__ == "__main__":
    elementary()
    predictions()
    chains()
    binary_sources()
    print("The illustrated graph changes, KPC deductions and binary distributions agree.")
