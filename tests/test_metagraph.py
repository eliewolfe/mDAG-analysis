"""The quantum metagraph (equivalence_classes.ClassIndex, metagraph_advanced.QuantumMetagraph): its classes, the
quantum-valid HLP rule, the lift of hybrids, the seed classes and the upward rule."""
import pytest

from hypergraphs import Hypergraph
from directed_structures import DirectedStructure
from mDAG_advanced import mDAG
from quantum_mDAG import QmDAG
from metagraph_advanced import ClassicalMetagraph, Observable_unlabelled_mDAGs, QuantumMetagraph
from equivalence_classes import ClassIndex
import known_QC_gaps as K


@pytest.fixture(scope="module")
def index():
    return ClassIndex((3, 4))


def Q(edges, n, C, Qf):
    return QmDAG(DirectedStructure(edges, n), Hypergraph(C, n), Hypergraph(Qf, n))


def test_class_counts_and_nesting(index):
    for n, (n_mdags, classical, quantum, latent_free, inputs, structures) in \
            {3: (46, 8, 8, 5, 3, 5), 4: (2809, 396, 445, 20, 424, 996)}.items():
        Qm, Cm = index.quantum[n], index.classical[n]
        assert len(Qm.dict_uniqind_unlabelled_mDAGs) == n_mdags
        assert len(Cm.equivalence_classes_as_ids) == classical and len(Qm.equivalence_classes_as_ids) == quantum
        assert sum(1 for cls in Cm.equivalence_classes_as_ids if cls & Cm.latent_free_equivalent_ids) == latent_free
        # every quantum class lies inside one classical class
        for cls in Qm.equivalence_classes_as_ids:
            assert len({Cm.class_index_of[uid] for uid in cls}) == 1
        keys = index.input_keys(n)
        assert len(keys) == inputs and index.all_quantum_structures(keys) == structures
        # the classical metagraph derived from the quantum one equals the standalone one
        standalone = Observable_unlabelled_mDAGs(n, verbose=False)
        assert sorted(map(sorted, Cm.equivalence_classes_as_ids)) == sorted(map(sorted, standalone.equivalence_classes_as_ids))
        assert Cm.structural_order is Qm.structural_order   # shared


def test_quantum_hlp_rule_fires_only_on_premeasurable_origins(index):
    Qm = index.quantum[4]
    assert len(Qm.hlp_edge_list) == 546 and len(index.classical[4].hlp_edge_list) == 648
    for md in Qm.dict_uniqind_unlabelled_mDAGs.values():
        for a, b in md.quantum_droppable_edges:
            assert md.is_premeasurable(a) and (a, b) in md.droppable_edges
        for a, b in md.droppable_edges:
            assert ((a, b) in md.quantum_droppable_edges) == md.is_premeasurable(a)
    # 0 -> 1 with Q{0,1}: 0 has no visible parent and one facet, so the edge is redundant classically and quantumly;
    # with a setting 2 of both 0 and 1 (Pa(0) = {2, Q01} is still inside Pa(1)) only the classical rule applies:
    # 0's measurement depends on 2, which the source of Q{0,1} does not know.
    one = mDAG(DirectedStructure([(0, 1)], 3), Hypergraph([(0, 1)], 3))
    two = mDAG(DirectedStructure([(2, 0), (2, 1), (0, 1)], 3), Hypergraph([(0, 1)], 3))
    assert one.quantum_droppable_edges == [(0, 1)] and one.droppable_edges == [(0, 1)]
    assert two.quantum_droppable_edges == [] and two.droppable_edges == [(0, 1)]
    assert not two.is_premeasurable(0) and one.is_premeasurable(0)


def test_the_bell_settings_class_collapses_under_quantum_rules(index):
    members = {
        'S': Q([(0, 1), (0, 3), (1, 2)], 4, [], [(2, 3)]),
        "G'": Q([(0, 1), (0, 3), (1, 2)], 4, [], [(0, 1), (2, 3)]),
        "G''": Q([(0, 3), (1, 2)], 4, [], [(0, 1), (2, 3)]),
        'G': Q([(0, 1), (0, 3), (1, 2)], 4, [], [(0, 3), (2, 3)]),
        'H': Q([(0, 1), (1, 2)], 4, [], [(0, 3), (2, 3)]),
    }
    keys = {index.key(g) for g in members.values()}
    assert len(keys) == 1
    key = next(iter(keys))
    assert index.size(key) == 5 and index.classical_class(key) is not None
    assert index.seed_keys(K.SEEDS)[key] == 'QG_Bell_Edge_Edge_SettingsC+QG_Bell_Edge_Edge_SettingsEdge+QG_Bell_Edge_Edge_SettingsEdgeC'


def test_lift_and_seed_classes(index):
    assert K.QG_Triangle.lift() is None
    for name, g in K.SEEDS.items():
        if name != 'QG_Triangle':
            lifted = g.lift()
            assert lifted is not None and lifted.is_all_quantum
            assert lifted.as_mDAG.unique_unlabelled_id == g.as_mDAG.unique_unlabelled_id
    seed_keys = index.seed_keys(K.SEEDS)
    assert len(seed_keys) == 5
    sizes = sorted((index.size(k), names.count('+') + 1) for k, names in seed_keys.items())
    assert sizes == [(1, 1), (1, 1), (3, 3), (5, 3), (6, 6)]   # triangle (hybrid), Evans, instrumental, Bell settings, Bell
    inputs4 = set(index.input_keys(4))
    assert sum(1 for k in seed_keys if k in inputs4) == 2
    # a hybrid that does not lift keeps its own id; a five-node structure too
    assert index.key(K.QG_Triangle) == K.QG_Triangle.unique_unlabelled_id
    five = Q([(4, 0)], 5, [], [(2, 3), (1, 3), (0, 1), (0, 2)])
    assert index.key(five) == five.unique_unlabelled_id and index.members(index.key(five), five) == [five]
    # a liftable hybrid is keyed by the class of its lift
    hybrid = Q([(0, 2), (1, 3)], 4, [(0, 2)], [(2, 3)])      # QG_Bell_EdgeC_Edge
    assert index.key(hybrid) == index.key(Q([(0, 2), (1, 3)], 4, [], [(0, 2), (2, 3)]))


def test_upward_steps_within_the_classical_class(index):
    # G = 0->1, 0->3, 1->2; Q{0,3}, Q{2,3} is structurally above S = 0->1, 0->3, 1->2; Q{2,3} (one facet more) and
    # classically equivalent to it; both are in the same quantum class here, so the upward step is a self-loop
    # and is not listed. A class structurally above another class of the same classical class is listed.
    G = Q([(0, 1), (0, 3), (1, 2)], 4, [], [(0, 3), (2, 3)])
    assert all(index.key(child) != index.key(G) for _, child in index.upward_steps(G))
    C4 = index.classical[4]
    found = 0
    for key in index.input_keys(4):
        rep = index.representative(key)
        for params, below in index.upward_steps(rep):
            found += 1
            below_key = index.key(below)
            assert below_key != key
            assert index.classical_class(below_key) == index.classical_class(key)
            # some member of the upper class is structurally above some member of the lower class
            lower_ids = index._mdag_ids_of[below_key]
            assert any(C4.structurally_below(uid) & lower_ids for uid in index._mdag_ids_of[key])
            # and the step names the pair in the structures' own words
            assert dict(params).keys() == {'above', 'below'}
    assert found > 0
    # the structural order is a sound containment: a structure below has fewer facets or edges
    uid = next(iter(index._mdag_ids_of[index.key(G)]))
    md = C4.dict_uniqind_unlabelled_mDAGs[uid]
    for lower in C4.structurally_below(uid):
        low = C4.dict_uniqind_unlabelled_mDAGs[lower]
        assert low.n_of_edges + low.n_of_latents < md.n_of_edges + md.n_of_latents or \
            low.simplicial_complex_instance.as_extended_bit_array.sum() < md.simplicial_complex_instance.as_extended_bit_array.sum()


# -- the orbit construction (canonical_forms.CanonicalForms) against the former labelled enumeration -------------

@pytest.mark.parametrize("n", [3, 4])
def test_orbit_construction_matches_the_labelled_enumeration(n):
    M = Observable_unlabelled_mDAGs(n, verbose=False)
    forms = M.canonical_forms
    # the node dict: same keys in the same order, same labelled representatives, keys are the members' ids
    legacy = {}
    for md in M.most_labelled_mDAGs:
        legacy[md.unique_unlabelled_id] = md
    assert list(M.dict_uniqind_unlabelled_mDAGs) == list(legacy)
    assert [md.unique_id for md in M.dict_uniqind_unlabelled_mDAGs.values()] == [md.unique_id for md in legacy.values()]
    for uid, md in M.dict_uniqind_unlabelled_mDAGs.items():
        assert md.unique_unlabelled_id == uid
    # canonical ids on demand equal mDAG.unique_unlabelled_id, through the view and the function
    sample = M.all_labelled_mDAGs if n == 3 else M.all_labelled_mDAGs[::13]
    for md in sample:
        ds_int, sc_int = md.directed_structure_instance.as_integer, md.simplicial_complex_instance.as_integer
        assert M.canonical_id(ds_int, sc_int) == md.unique_unlabelled_id
        assert M.dict_id_to_canonical_id[md.unique_id] == md.unique_unlabelled_id
        assert md.unique_id in M.dict_id_to_canonical_id
    assert M.dict_id_to_canonical_id.get(-1) is None
    # the lower-neighbour dominances are the former all-pairs dominances
    new, old = set(M.structural_dominances), set(M.legacy_structural_dominances)
    assert len(M.structural_dominances) == len(new) and new == old
    for strong, weak in new:
        s, w = M.dict_uniqind_unlabelled_mDAGs[strong], M.dict_uniqind_unlabelled_mDAGs[weak]
        assert (s.n_of_edges, len(s.simplicial_complex_instance.faces)) in \
            {(w.n_of_edges + 1, len(w.simplicial_complex_instance.faces)), (w.n_of_edges, len(w.simplicial_complex_instance.faces) + 1)}
    # the canonical forms object is shared by the classical companion
    Q = QuantumMetagraph(n)
    C = ClassicalMetagraph.from_quantum(Q)
    assert C.canonical_forms is Q.canonical_forms and C.structural_dominances is Q.structural_dominances


def test_facet_removals_and_automorphisms():
    M = Observable_unlabelled_mDAGs(4, verbose=False)
    forms = M.canonical_forms
    full = Hypergraph([(0, 1, 2)], 4)
    # removing the 3-face of {0,1,2} leaves its three edges; removing an edge leaves nothing
    assert set(forms.facet_removals(full.as_integer)) == {Hypergraph([(0, 1), (0, 2), (1, 2)], 4).as_integer}
    assert set(forms.facet_removals(Hypergraph([(0, 1)], 4).as_integer)) == {0}
    # a sub-face already covered by another facet is not added
    sc = Hypergraph([(0, 1, 2), (0, 1, 3)], 4)
    assert set(forms.facet_removals(sc.as_integer)) == {Hypergraph([(0, 1, 3), (0, 2), (1, 2)], 4).as_integer,
                                                        Hypergraph([(0, 1, 2), (0, 3), (1, 3)], 4).as_integer}
    # Aut of the empty directed structure is all of S_4; of a single edge 0 -> 1, the permutations fixing 0 and 1
    assert len(forms.ds_automorphisms(0)) == 24
    edge = DirectedStructure([(0, 1)], 4)
    assert len(forms.ds_automorphisms(edge.as_integer)) == 2
    assert len(forms.sc_orbits(0)) == len(M.all_unlabelled_simplicial_complices)


# -- the sparse metagraph (scipy) ---------------------------------------------------------------------------------

def test_sparse_metagraph_queries_and_compact_mode():
    C = Observable_unlabelled_mDAGs(4, verbose=False)
    N = C.num_metagraph_nodes
    assert C.meta_graph.shape == (N, N) and C.structural_order.shape == (N, N)
    # components partition the nodes; a dominance never joins two classes unless it lies on a cycle
    assert sum(map(len, C.equivalence_classes_as_ids)) == N
    assert len(set().union(*C.equivalence_classes_as_ids)) == N
    for strong, weak in C.structural_dominances:
        assert weak in C.structurally_below(strong) and strong in C.structurally_above(weak)
    # the reachability order of a subset is transitively reduced and loop-free
    subset = list(C.dict_uniqind_unlabelled_mDAGs.values())[100:112]
    order = C.partial_order_of_subset(subset)
    assert all(a is not b for a, b in order.edges())
    idx = C._dict_id_to_idx
    reach = {g: C._reachable(C.meta_graph, idx[g.unique_unlabelled_id]) for g in subset}
    for a, b in order.edges():
        assert idx[b.unique_unlabelled_id] in reach[a]
        assert not any(idx[c.unique_unlabelled_id] in reach[a] and idx[b.unique_unlabelled_id] in reach[c]
                       for c in subset if c is not a and c is not b)
    for a in subset:
        for b in subset:
            if a is not b and idx[b.unique_unlabelled_id] in reach[a] and not order.has_edge(a, b):
                assert any(idx[c.unique_unlabelled_id] in reach[a] and idx[b.unique_unlabelled_id] in reach[c]
                           for c in subset if c is not a and c is not b)
    # the HLP-in-reverse search finds the structures reaching a latent-free DAG
    boring = {m.unique_unlabelled_id for m in C.boring_by_virtue_of_HLP}
    assert {m.unique_unlabelled_id for m in C.latent_free_DAGs_unlabelled} <= boring <= C.latent_free_equivalent_ids
    # compact mode: unfiltered dominances and released caches give the same partition
    compact = Observable_unlabelled_mDAGs(4, verbose=False, compact=True)
    assert compact.compact and not C.compact
    assert sorted(map(sorted, compact.equivalence_classes_as_ids)) == sorted(map(sorted, C.equivalence_classes_as_ids))
    assert compact.meta_graph.nnz > C.meta_graph.nnz
    Qc = QuantumMetagraph(4, compact=True)
    assert ClassicalMetagraph.from_quantum(Qc).compact


def test_mdag_release_cache_and_derived_attributes():
    md = mDAG(DirectedStructure([(0, 1), (2, 1)], 4), Hypergraph([(1, 2, 3)], 4))
    assert md.visible_nodes == [0, 1, 2, 3] and md.latent_nodes == (4, 5) and md.nof_nonsingleton_latents == 1
    assert md.total_number_of_nodes == 5 and md.number_of_root_nodes == 2
    assert md.vis_nodes_with_no_children == {1, 3} and md.vis_nodes_with_no_parents == {0}
    before = (md.unique_unlabelled_id, md.droppable_edges, md.CI_count, list(md.weaker_mDAG_HLP('classical')))
    assert len(md.__dict__) > 4
    md.release_cache()
    assert set(md.__dict__) <= mDAG._CORE_ATTRIBUTES
    assert (md.unique_unlabelled_id, md.droppable_edges, md.CI_count, list(md.weaker_mDAG_HLP('classical'))) == before
