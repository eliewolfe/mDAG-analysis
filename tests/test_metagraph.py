"""The quantum metagraph (equivalence_classes.ClassIndex, metagraph_advanced.QuantumMetagraph): its classes, the
quantum-valid HLP rule, the lift of hybrids, the seed classes and the upward rule."""
import numpy as np
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
            {3: (46, 8, 8, 5, 3, 5), 4: (2809, 396, 438, 20, 418, 996)}.items():
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


def test_quantum_hlp_rule_is_the_pre_measurement_rule(index):
    Qm = index.quantum[4]
    assert len(Qm.hlp_edge_list) == 528 and len(index.classical[4].hlp_edge_list) == 648
    for md in Qm.dict_uniqind_unlabelled_mDAGs.values():
        facets = md.simplicial_complex_instance.simplicial_complex_as_sets
        for a, b in md.droppable_edges:
            others_ok = all(md.is_premeasurable(v) for F in facets if a in F for v in F if v not in (a, b))
            assert md.quantum_hlp_origin_ok(a, b) == (md.is_premeasurable(a) or others_ok)
            assert ((a, b) in md.quantum_droppable_edges) == md.quantum_hlp_origin_ok(a, b)
        assert set(md.quantum_droppable_edges) <= set(md.droppable_edges)
    # 0 -> 1 with Q{0,1}: 0 is premeasurable, so the edge is redundant classically and quantumly. With a setting 2 of
    # both 0 and 1 (Pa(0) = {2, Q01} inside Pa(1)) the facet {0,1} has no other member, so 1 holds every input of 0
    # and the pair's state is shared by the two alone: still quantum-redundant (second form of the rule).
    one = mDAG(DirectedStructure([(0, 1)], 3), Hypergraph([(0, 1)], 3))
    two = mDAG(DirectedStructure([(2, 0), (2, 1), (0, 1)], 3), Hypergraph([(0, 1)], 3))
    assert one.quantum_droppable_edges == [(0, 1)] and two.quantum_droppable_edges == [(0, 1)]
    assert not two.is_premeasurable(0) and one.is_premeasurable(0)
    # the other members of a's facets premeasurable: 2, 3, 4 have no visible parent and one facet each, so 0 -> 1 is
    # quantum-redundant although 0 has the setting 2 and three facets (2 -> 0 and 2 -> 1 are redundant by the
    # first form, 2 being premeasurable)
    five = mDAG(DirectedStructure([(2, 0), (2, 1), (0, 1)], 5), Hypergraph([(0, 1, 2), (0, 1, 3), (0, 1, 4)], 5))
    assert set(five.quantum_droppable_edges) == {(0, 1), (2, 0), (2, 1)}
    # a member with two facets blocks the argument: 0 holds shares of Q{0,1,2} and Q{0,3}
    blocked = mDAG(DirectedStructure([(0, 1), (0, 2), (0, 3), (2, 3), (2, 1)], 4), Hypergraph([(0, 1, 2), (0, 3)], 4))
    assert (2, 1) in blocked.droppable_edges and (2, 1) not in blocked.quantum_droppable_edges
    # a setting of a that the source does not know, with a third facet member that has a setting too: not droppable
    setting = mDAG(DirectedStructure([(0, 1), (0, 2), (1, 2), (0, 3)], 4), Hypergraph([(1, 2, 3)], 4))
    assert (1, 2) in setting.droppable_edges and (1, 2) not in setting.quantum_droppable_edges
    # the same with 3 premeasurable: the source of Q{1,2,3} measures 3's share, and 1 -> 2 becomes redundant
    assert (1, 2) in mDAG(DirectedStructure([(0, 1), (0, 2), (1, 2)], 4), Hypergraph([(1, 2, 3)], 4)).quantum_droppable_edges


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
    # a classical facet lifts when all its members but at most one are premeasurable: C{0,1,2} with 0 and 1 having
    # no visible parent and no other facet lifts even though 2 has a setting and another facet
    three = Q([(3, 2)], 4, [(0, 1, 2)], [(2, 3)])
    assert three.lift() is not None and three.lift().is_all_quantum
    assert Q([(3, 2)], 4, [(0, 1, 2)], [(1, 3), (2, 3)]).lift() is None    # two members with other facets
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
    new, old = set(M.edge_ids(M.structural_dominances)), set(M.legacy_structural_dominances)
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


def test_canonical_forms_preserve_nested_iterator_facets():
    from canonical_forms import CanonicalForms
    M = Observable_unlabelled_mDAGs(3, verbose=False)
    directed = list(M.all_directed_structures)
    complexes = list(M.all_simplicial_complices)
    forms = CanonicalForms(3, [d.as_integer for d in directed], [d.as_bit_square_matrix for d in directed],
                           [h.as_integer for h in complexes],
                           (iter([iter(facet) for facet in h.compressed_simplicial_complex]) for h in complexes))
    assert np.array_equal(forms.sc_table, M.canonical_forms.sc_table)
    for d in directed:
        for h in complexes:
            assert forms.canonical(d.as_integer, h.as_integer) == mDAG(d, h).unique_unlabelled_id


# -- the sparse metagraph (scipy) ---------------------------------------------------------------------------------

def test_sparse_metagraph_queries_and_compact_mode():
    C = Observable_unlabelled_mDAGs(4, verbose=False)
    N = C.num_metagraph_nodes
    assert C.meta_graph.shape == (N, N) and C.structural_order.shape == (N, N)
    # components partition the nodes; a dominance never joins two classes unless it lies on a cycle
    assert sum(map(len, C.equivalence_classes_as_ids)) == N
    assert C.structural_dominances.dtype == np.int32 and C.structural_dominances.shape[1] == 2
    assert all(C.class_index_of[uid] == k for k, cls in enumerate(C.equivalence_classes_as_ids) for uid in cls)
    assert len(C.class_index_of) == N and set(C.class_labels.tolist()) == set(range(len(C.equivalence_classes_as_ids)))
    assert len(set().union(*C.equivalence_classes_as_ids)) == N
    for strong, weak in C.edge_ids(C.structural_dominances):
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


# -- the October 2026 review: the HLP closure, class lookups by canonical id, singleton classification ----------------

def test_hlp_closure_is_the_reachability_of_the_hlp_graph():
    from scipy.sparse import csgraph
    M = Observable_unlabelled_mDAGs(3, verbose=False)
    closure = M.HLP_meta_adjmat_closure
    reachable = np.isfinite(csgraph.shortest_path(M.HLP_meta_adjmat.tocsr(), directed=True, unweighted=True))
    assert closure.dtype == bool and np.array_equal(closure.toarray(), reachable)
    assert closure.nnz > M.HLP_meta_adjmat.nnz            # paths of two or more edges are reached
    four = Observable_unlabelled_mDAGs(4, verbose=False).HLP_meta_adjmat_closure
    assert ((four.astype(np.int32) @ four.astype(np.int32)) != 0).nnz == four.nnz     # idempotent: a closure


def test_class_lookups_accept_any_labelling_and_partition_supplied_objects():
    M = Observable_unlabelled_mDAGs(3, verbose=False)
    copies = []
    for uid, md in M.dict_uniqind_unlabelled_mDAGs.items():
        perm = (2, 1, 0)
        copy = mDAG(DirectedStructure([(perm[a], perm[b]) for a, b in md.directed_structure_instance.edge_list], 3),
                    Hypergraph([tuple(perm[v] for v in f) for f in md.simplicial_complex_instance.compressed_simplicial_complex], 3))
        found = M.eqclass_of_mDAG(copy)
        assert {m.unique_unlabelled_id for m in found} == M.class_of(uid)
        copies.append(copy)
    blocks = M.eqclasses_of_subset(copies)
    assert sum(map(len, blocks)) == len(copies) and len(blocks) == len(M.equivalence_classes_as_ids)
    assert all(len({M.class_index_of[g.unique_unlabelled_id] for g in block}) == 1 for block in blocks)
    assert all(g in copies for block in blocks for g in block)        # the supplied objects, not the stored ones
    with pytest.raises(ValueError):
        M.eqclass_of_mDAG(mDAG(DirectedStructure([], 4), Hypergraph([], 4)))
    with pytest.raises(ValueError):
        M.eqclasses_of_subset([copies[0], mDAG(DirectedStructure([], 4), Hypergraph([], 4))])
    assert len(M.latent_free_DAG_ids_and_CI_labelled) == len(M.all_directed_structures)


def test_classification_always_returns_blocks():
    from metagraph_advanced import classify_by_attributes, further_classify_by_attributes
    import metagraph_temporally_ordered as T
    M = Observable_unlabelled_mDAGs(3, verbose=False)
    one, two = list(M.dict_uniqind_unlabelled_mDAGs.values())[:2]
    for classify in (classify_by_attributes, T.classify_by_attributes):
        assert classify([one], ['all_CI_unlabelled']) == ({one},)
        assert classify([], ['all_CI_unlabelled']) == ()
        assert all(isinstance(block, set) for block in classify([one, two], ['all_CI_unlabelled']))
    refined = further_classify_by_attributes([{one}, {one, two}], ['all_CI_unlabelled'], verbose=False)
    assert all(isinstance(block, set) for block in refined) and sum(map(len, refined)) == 3
    again = further_classify_by_attributes(refined, ['skeleton_unlabelled'], verbose=False)
    assert sum(map(len, again)) == 3
