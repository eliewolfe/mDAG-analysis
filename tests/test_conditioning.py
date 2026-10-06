"""The conditioning piggyback is justified only when no grandparent (visible or latent) fails to be a parent."""
import itertools
import math

import networkx as nx

from hypergraphs import Hypergraph
from directed_structures import DirectedStructure
from quantum_mDAG import QmDAG


def Q(edges, n, C, Qf):
    return QmDAG(DirectedStructure(edges, n), Hypergraph(C, n), Hypergraph(Qf, n))


# X = 2 has visible parents 0 and 1; each parent shares a classical facet with an outside node (3, 4).
LATENT_GRANDPARENTS = Q([(0, 2), (1, 2)], 5, [(0, 3), (1, 4)], [])


def test_latent_grandparent_blocks_conditioning():
    assert not LATENT_GRANDPARENTS.has_grandparents_that_are_not_parents(2)
    assert LATENT_GRANDPARENTS.parents_have_external_latents(2)
    assert not LATENT_GRANDPARENTS.conditioning_is_justified(2)
    assert LATENT_GRANDPARENTS.conditioning_is_justified(2, strict_latents=False)
    conditionable = [v for v in range(5) if LATENT_GRANDPARENTS.conditioning_is_justified(v)]
    assert 2 not in conditionable
    first_level = [sub for sub in LATENT_GRANDPARENTS.subconditionals if sub.number_of_visible == 4]
    assert first_level and all(2 in sub.directed_structure_instance.variable_names for sub in first_level)


def test_counterexample_distribution_violates_the_output_structure():
    # Classical model of the structure above: lambda_i uniform, p_i = lambda_i with probability 3/4, o_i = lambda_i,
    # X = [p_1 == p_2]. Conditioning on X = 1 correlates o_1 and o_2, which the (loose) output d-separates.
    out = LATENT_GRANDPARENTS.condition(2)
    g, _ = out.effective_DAG_data
    names = out.directed_structure_instance.variable_names
    idx = {name: i for i, name in enumerate(names)}
    assert nx.is_d_separator(g, {idx[3]}, {idx[4]}, set())
    P = {}
    for l1, l2, f1, f2 in itertools.product([0, 1], repeat=4):
        pr = 0.25 * (0.75 if f1 == 0 else 0.25) * (0.75 if f2 == 0 else 0.25)
        if l1 ^ f1 == l2 ^ f2:
            P[(l1, l2)] = P.get((l1, l2), 0) + pr
    total = sum(P.values())
    P = {k: v / total for k, v in P.items()}
    m1 = {a: sum(v for (x, _), v in P.items() if x == a) for a in (0, 1)}
    m2 = {b: sum(v for (_, y), v in P.items() if y == b) for b in (0, 1)}
    mutual_information = sum(v * math.log2(v / (m1[a] * m2[b])) for (a, b), v in P.items())
    assert mutual_information > 0.04


def test_conditioning_still_applies_when_parents_latents_include_the_node():
    # Square: conditioning on node 1 (parents none, facets {0,1},{1,3} both contain 1) gives the triangle.
    square = Q([], 4, [], [(2, 3), (1, 3), (0, 1), (0, 2)])
    triangle = Q([], 3, [], [(0, 1), (1, 2), (0, 2)])
    assert square.conditioning_is_justified(1)
    assert triangle.unique_unlabelled_id in square.unique_unlabelled_ids_obtainable_by_conditioning
    # A parent that lies in a facet with the node: allowed.
    g = Q([(0, 2)], 3, [(0, 2)], [(1, 2)])
    assert g.conditioning_is_justified(2)


def test_facets_inside_the_parent_block_do_not_block_conditioning():
    # Node 2 has parents 0, 1, 3 and the facet {0,1} lies inside the parent block: post-selection only
    # redistributes the block, which the new common cause over the parents reproduces.
    g = Q([(0, 2), (1, 2), (3, 2)], 4, [(0, 1)], [])
    assert g.conditioning_is_justified(2)
