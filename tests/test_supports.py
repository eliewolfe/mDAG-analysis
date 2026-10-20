"""The support-testing pipeline (supports.py, supports_beyond_esep.py, radix.py): the SAT feasibility test against a
brute-force oracle on small graphs, and one regression per finding of the October 2026 review (symmetry tables with
mixed cardinalities, coherent relabelling, the verdict of model-returning calls, stale selector ids, radix limits,
the causal-symmetry tester, one_off's definite events, solver options on the early-exit scan)."""
import itertools

import numpy as np
import pytest
from pysat.solvers import NoSuchSolverError

import radix
from supports import SupportTester, SupportTesting, one_off, verdict
from supports_beyond_esep import SmartSupportTesting

TRIANGLE = ([3, 4], [4, 5], [3, 5])          # visible 0, 1, 2; one latent per pair
SHARED = ([2], [2])                           # two visible nodes with one common latent
INSTRUMENTAL = ([2], [0, 2])                  # 0 -> 1 with a latent common to both


# -- the oracle --------------------------------------------------------------------------------------------------

def brute_force_feasible(parents_of, cardinalities, nof_events, support):
    """Enumerate every deterministic response-function assignment with latents of `nof_events` values. A support is
    feasible when, for some assignment, world k (every latent equal to k) produces event k of the support and no
    latent assignment produces an event outside the support: exactly what the SAT encoding asserts. Observed parents
    must precede their children in the node order."""
    n_obs = len(parents_of)
    n_lat = max(itertools.chain.from_iterable(parents_of)) + 1 - n_obs
    all_cards = list(cardinalities) + [nof_events] * n_lat
    support = [tuple(int(v) for v in row) for row in support]
    support_set = set(support)

    def functions(i):
        inputs = list(itertools.product(*[range(all_cards[p]) for p in parents_of[i]]))
        for outs in itertools.product(range(cardinalities[i]), repeat=len(inputs)):
            yield dict(zip(inputs, outs))

    latent_assignments = list(itertools.product(range(nof_events), repeat=n_lat))
    for fs in itertools.product(*[list(functions(i)) for i in range(n_obs)]):
        def event(lat):
            values = [None] * n_obs + list(lat)
            for i in range(n_obs):
                values[i] = fs[i][tuple(values[p] for p in parents_of[i])]
            return tuple(values[:n_obs])
        if [event((w,) * n_lat) for w in range(nof_events)] != support:
            continue
        if all(event(lat) in support_set for lat in latent_assignments):
            return True
    return False


@pytest.mark.parametrize("parents_of, cards, nof_events", [
    (SHARED, (2, 2), 2), (SHARED, (3, 2), 2), (SHARED, (3, 2), 3),
    (INSTRUMENTAL, (2, 2), 2), (INSTRUMENTAL, (2, 2), 3),
    (TRIANGLE, (2, 2, 2), 2),
])
def test_sat_feasibility_matches_the_brute_force_oracle(parents_of, cards, nof_events):
    st = SupportTester(parents_of, cards, nof_events)
    events = list(itertools.product(*map(range, cards)))
    checked = 0
    for rows in itertools.combinations(events, nof_events):
        support = np.array(rows)
        expected = brute_force_feasible(parents_of, cards, nof_events, support)
        assert st.feasibleQ_from_matrix(support) == expected
        with_model = st.feasibleQ_from_matrix(support, return_model=True)
        assert isinstance(with_model, tuple) and verdict(with_model) == expected
        checked += 1
    assert checked > 0


# -- finding 1: symmetry reduction with mixed cardinalities -----------------------------------------------------

def test_internal_symmetry_table_has_no_unwritten_rows():
    swap = ((0, 1), (1, 0))
    mixed = SupportTesting(SHARED, (3, 2), 2, visible_automorphisms=swap)
    # the swap does not preserve the cardinalities: only the identity remains, and it is written
    assert mixed.internal_party_relabelling_group.tolist() == [list(range(6))]
    assert not mixed.has_internal_party_symmetry
    plain = SupportTesting(SHARED, (3, 2), 2)
    assert np.array_equal(mixed.unique_candidate_supports_as_compressed_lists, plain.unique_candidate_supports_as_compressed_lists)
    for cand in mixed.unique_candidate_supports_as_compressed_lists:
        assert len(set(cand.tolist())) == len(cand)
    # binary: both rows written, the reduction is real and expanding the infeasible set undoes it exactly
    sym = SupportTesting(INSTRUMENTAL[:1] + ([2],), (2, 2), 2, visible_automorphisms=swap)   # 0 <- L -> 1
    idn = SupportTesting(INSTRUMENTAL[:1] + ([2],), (2, 2), 2)
    assert sym.internal_party_relabelling_group.shape == (2, 4)
    assert len(sym.unique_candidate_supports_as_compressed_lists) < len(idn.unique_candidate_supports_as_compressed_lists)
    assert set(sym.unique_infeasible_supports_as_expanded_integers().tolist()) == set(idn.unique_infeasible_supports_as_expanded_integers().tolist())
    # an empty automorphism argument means the identity
    assert SupportTesting(SHARED, (2, 2), 2).internal_party_relabelling_group.tolist() == [list(range(4))]


# -- finding 2: coherent relabelling ------------------------------------------------------------------------------

def _relabel(st, integers, perm):
    mats = st.from_integer_to_matrix(np.asarray(integers))
    lists = np.sort(st.from_matrix_to_list(mats[..., list(perm)]), axis=-1)
    return st.from_list_to_integer(lists)


@pytest.mark.parametrize("parents_of, automorphisms", [
    (TRIANGLE, tuple(itertools.permutations(range(3)))),       # fully symmetric
    (([3], [3], [3, 4]), ((0, 1, 2), (1, 0, 2))),              # one swap
    (([3], [0, 3], [1, 4]), ((0, 1, 2),)),                     # none
])
def test_coherent_canonical_form_is_relabelling_invariant(parents_of, automorphisms):
    st = SupportTesting(parents_of, (2, 2, 2), 3, visible_automorphisms=automorphisms)
    assert st.external_party_relabelling_group.shape == (6, 8)
    assert st.external_party_relabelling_group[0].tolist() == list(range(8))      # the identity comes first
    infeasible = st.unique_infeasible_supports_as_expanded_integers()
    if len(infeasible) == 0:         # any list will do for the invariance; take a few candidates
        infeasible = st.unique_candidate_supports_as_compressed_integers[:5]
    canonical = st.convert_integers_into_canonical_under_coherent_relabelling(infeasible)
    # every column is the orbit of its own integer, in the group's order, independently recomputed
    fresh = [SupportTesting(parents_of, (2, 2, 2), 3, visible_automorphisms=automorphisms).orbit_under_external_party_relabelling(int(n))
             for n in infeasible.flat]
    cols = np.stack(fresh, axis=1)
    cols.sort(axis=-1)
    assert np.array_equal(canonical, cols[np.lexsort(np.rot90(cols))[0]])
    # the outcome-canonical orbit of a second member is the same multiset, with permutation order preserved
    a, b = int(infeasible.flat[0]), int(st.orbit_under_external_party_relabelling(int(infeasible.flat[0]))[1])
    assert sorted(st.orbit_under_external_party_relabelling(a).tolist()) == sorted(st.orbit_under_external_party_relabelling(b).tolist())
    # relabelling the whole list by any permutation leaves the canonical form unchanged
    for perm in itertools.permutations(range(3)):
        relabelled = _relabel(st, infeasible, perm)
        assert np.array_equal(st.convert_integers_into_canonical_under_coherent_relabelling(relabelled), canonical)
    # the explicit example of the review: identity-only symmetry, inputs 10 and 12
    st2 = SupportTesting(([3], [3], [3]), (2, 2, 2), 3, visible_automorphisms=((0, 1, 2),))
    assert st2.convert_integers_into_canonical_under_coherent_relabelling(np.array([10, 12])).tolist() == [10, 12]


@pytest.mark.parametrize("n_nodes, edges, facets", [
    (3, [(0, 1), (1, 2)], []),
    (3, [], [(0, 1)]),
    (4, [(0, 3), (1, 2)], [(2, 3)]),
    (4, [(0, 1), (1, 2), (2, 3)], []),
    (4, [(0, 1)], [(1, 2), (2, 3)]),
    (4, [], [(0, 1), (1, 2), (2, 3)]),
])
def test_unlabelled_support_signatures_match_for_relabelled_graphs(n_nodes, edges, facets):
    from directed_structures import DirectedStructure
    from hypergraphs import Hypergraph
    from mDAG_advanced import mDAG

    def graph(perm):
        return mDAG(DirectedStructure([(perm[a], perm[b]) for a, b in edges], n_nodes),
                    Hypergraph([tuple(perm[a] for a in facet) for facet in facets], n_nodes))

    # Construct each graph anew, exercising enumeration and outcome canonicalisation
    # before comparing signatures. Relabelling one already-enumerated list misses this.
    expected = graph(tuple(range(n_nodes))).infeasible_binary_supports_n_events_unlabelled(3)
    for perm in itertools.permutations(range(n_nodes)):
        actual = graph(perm).infeasible_binary_supports_n_events_unlabelled(3)
        assert actual == expected, perm
    if n_nodes == 3 and edges == [(0, 1), (1, 2)]:
        # Previously the relabelled chain returned (10, 28, 46).
        assert expected == (10, 28, 29)


# -- finding 3: the verdict of model-returning calls ----------------------------------------------------------------

def test_conservative_and_bulk_checks_read_the_verdict_not_the_tuple():
    support = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    st = SupportTester(TRIANGLE, (2, 2, 2), 3)
    assert st.feasibleQ_from_matrix(support) is False
    assert st.feasibleQ_from_matrix_CONSERVATIVE(support, min_definite=3, max_definite=3) is False
    with_model = st.feasibleQ_from_matrix_CONSERVATIVE(support, min_definite=3, max_definite=3, return_model=True)
    assert isinstance(with_model, tuple) and with_model[0] is False and with_model[1]
    feasible = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0]])
    assert st.feasibleQ_from_matrix_CONSERVATIVE(feasible, min_definite=3, max_definite=3) is True
    assert st.feasibleQ_from_matrix_CONSERVATIVE(feasible, min_definite=3, max_definite=3, return_model=True) == (True, [])
    assert verdict(True) and not verdict(False) and not verdict((False, ['x'])) and verdict((True, []))
    # the bulk enumeration gives the same infeasible set with and without models
    stt = SupportTesting(TRIANGLE, (2, 2, 2), 3)
    plain = stt.unique_infeasible_supports_as_integers_among(stt.unique_candidate_supports_as_compressed_integers)
    with_models = stt.unique_infeasible_supports_as_integers_among(stt.unique_candidate_supports_as_compressed_integers, return_model=True)
    assert plain.tolist() == with_models.tolist() and len(plain) > 0


# -- finding 4: repeated model requests on one tester ---------------------------------------------------------------

def test_repeated_model_requests_do_not_decode_stale_selectors():
    st = SupportTester(TRIANGLE, (2, 2, 2), 3)
    infeasible, explanation = st.feasibleQ_from_matrix(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), return_model=True)
    assert infeasible is False and explanation
    feasible, model = st.feasibleQ_from_matrix(np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0]]), return_model=True)
    assert feasible is True
    assert model and all(name.startswith('v[') for name in model)
    # and again, in the other order, on the same tester
    assert st.feasibleQ_from_matrix(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), return_model=True)[0] is False
    assert st.feasibleQ_from_matrix(np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0]]), return_model=True)[0] is True
    pair = st.potentially_feasibleQ_from_matrix_pair(np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0]]),
                                                     np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 1, 1]]), return_model=True)
    assert pair[0] is True and all(name.startswith('v[') for name in pair[1])


# -- finding 5: radix ---------------------------------------------------------------------------------------------

def test_radix_round_trips_beyond_the_intp_and_uint64_limits():
    # 4 binary nodes, 16 events: the event base has product 2**64
    st = SupportTester(TRIANGLE + ([3],), (2, 2, 2, 2), 16)
    n = st.from_list_to_integer(np.arange(16))
    assert st.from_integer_to_list(n).tolist() == list(range(16))
    both = st.from_list_to_integer(np.array([np.arange(16), np.arange(16)[::-1]]))
    assert st.from_integer_to_list(both).tolist() == [list(range(16)), list(range(16))[::-1]]
    for nobs, nev in [(4, 15), (5, 12), (5, 13), (5, 20)]:
        base = (2 ** nobs,) * nev
        digits = np.array([[1] * nev, list(range(nev))]) % (2 ** nobs)
        assert np.array_equal(np.asarray(radix.to_digits(radix.from_digits(digits, base), base), dtype=np.int64), digits)
    # bits beyond 64, scalars and batches, Python integers as the reference
    for v in [0, 1, 2 ** 64, 2 ** 64 + 5, 3 ** 50]:
        bits = radix.to_bits(v, 100)
        assert bits.shape == (100,) and int(radix.from_bits(bits)) == v
    batch = np.array([[0, 1, 2 ** 70]], dtype=object)
    bits = radix.to_bits(batch, 71)
    assert bits.shape == (1, 3, 71) and radix.from_bits(bits).tolist() == batch.tolist()
    assert radix.to_bits(np.empty((0,), dtype=object), 70).shape == (0, 70)      # size-0 input on the >64-bit path
    assert radix.to_bits(np.array([3, 5]), 2, sanity_check=True).tolist() == [[0, 1, 1], [1, 0, 1]]
    assert radix.to_digits(np.array([5, 7]), (2, 2, 2), sanity_check=True).tolist() == [[1, 0, 1], [1, 1, 1]]
    with pytest.raises(AssertionError):
        radix.to_digits(np.array([8]), (2, 2, 2), sanity_check=True)
    # the sizes in use stay on the fast path
    small = SupportTester(TRIANGLE, (2, 2, 2), 3)
    assert small.from_list_to_integer(np.array([[0, 1, 2]])).dtype == small.int_dtype
    assert radix.to_digits(np.array([5]), (8,) * 4).tolist() == [[0, 0, 0, 5]]


# -- finding 6: the causal-symmetry tester ---------------------------------------------------------------------------

def test_causal_symmetry_tester_constructs_and_is_a_restriction():
    import supports_with_causal_sym as cs
    parents_of = ([3, 4], [4, 5], [5, 3])
    events = np.array([(0, 0, 0), (1, 1, 1), (2, 2, 2)])
    sym = cs.SupportTester_PartyCausalSymmetry(parents_of, (3, 3, 3), 3)
    full = cs.SupportTester_FullCausalSymmetry(parents_of, (3, 3, 3), 3)
    plain = SupportTester(parents_of, (3, 3, 3), 3)
    assert sym.var(0, 1, 2) == sym.vpool.id("A_[00,01]==2") and full.var(1, 0, 2) == full.var(0, 1, 2)
    r_plain, r_sym, r_full = plain.feasibleQ_from_matrix(events), sym.feasibleQ_from_matrix(events), full.feasibleQ_from_matrix(events)
    assert isinstance(r_sym, bool) and isinstance(r_full, bool)
    # a symmetric model is a model, so feasibility under symmetry implies feasibility without; the perfectly
    # correlated three-outcome support is infeasible in the classical triangle, with or without symmetry
    assert (not r_full or r_sym) and (not r_sym or r_plain)
    assert r_plain is False and r_sym is False and r_full is False
    assert sym.feasibleQ_from_matrix(events, return_model=True)[0] is False
    assert sym.reverse_var(-sym.var(0, 1, 2)) == "A_[00,01]!=2"       # the parent's decoder cannot parse these names


# -- finding 7: one_off --------------------------------------------------------------------------------------------

def test_one_off_relabels_definite_events_with_the_support():
    support = [(2, 2), (2, 3), (3, 2), (3, 3)]
    assert one_off(SHARED, support, definite_events=[(2, 2)]) == one_off(SHARED, [(0, 0), (0, 1), (1, 0), (1, 1)], definite_events=[(0, 0)])
    assert one_off(SHARED, support, definite_events=[(2, 2), (3, 3)]) is True
    with pytest.raises(ValueError):
        one_off(SHARED, support, definite_events=[(2, 4)])
    assert one_off(TRIANGLE, [(1, 0, 0), (0, 1, 0), (0, 0, 1)]) is False
    assert one_off(TRIANGLE, [(5, 0, 0), (0, 5, 0), (0, 0, 5)]) is False
    assert one_off(TRIANGLE, [(1, 0, 0), (0, 1, 0), (0, 0, 1)], stclass=SupportTesting) is False


# -- finding 8 and the small edge cases ------------------------------------------------------------------------------

def test_solver_options_reach_the_early_exit_scan_and_one_event_candidates_exist():
    st = SupportTesting(TRIANGLE, (2, 2, 2), 3)
    with pytest.raises(NoSuchSolverError):
        st.attempt_to_find_one_infeasible_support_among(st.unique_candidate_supports_as_compressed_lists, name='nosuchsolver')
    found = st.attempt_to_find_one_infeasible_support_among(st.unique_candidate_supports_as_compressed_lists, name='mgh', use_timer=False)
    assert found.shape == (3, 3) and st.feasibleQ_from_matrix(found) is False
    assert SupportTesting(TRIANGLE, (2, 2, 2), 1).unique_candidate_supports_as_compressed_lists.tolist() == [[0]]
    # the smart tester's wrappers, as mDAG calls them, accept the solver options
    smart = SmartSupportTesting(TRIANGLE, (2, 2, 2), 3, esep_relations=tuple())
    assert smart.no_infeasible_supports_beyond_esep(name='mgh', use_timer=False) is False
