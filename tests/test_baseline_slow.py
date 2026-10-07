"""Pins the 4-node QC-gap census (both phases, cache disabled). Takes a few minutes; run with `pytest -m slow`.
All counts are up to relabelling."""
import pytest

pytest.importorskip("mosek")

# Phase 1: the 2807 labelled 4-node mDAGs that respect the order 0 < 1 < 2 < 3 and are not provably algebraic, with
# every latent quantum (996 distinct up to relabelling, Bell variants included), elementary reductions only, three-node
# seeds only (manuscript/piggybacks.md, 8.1).
EXPECTED_CHEAP = {
    'inputs': 996,
    'proven': 917,
    'remaining': 79,
    'labelled_inputs': 2807,
    'with PD': 860,
    'with interruption': 10,
    'with conditioning': 289,
    'with naive_marginalization': 515,
    'with teleportation_marginalization': 540,
    'with marginalization (either kind)': 540,
    'only via PD': 246,
    'only via interruption': 3,
    'only via conditioning': 20,
    'only via naive_marginalization': 0,
    'only via teleportation_marginalization': 16,
    'only via marginalization (either kind)': 24,
}

# Phase 2: the Bell variants are seeds (990 inputs), the cascade Fritz -> Fritz_kept -> Fritz_entropic on what phase 1
# left (8.2, 8.3).
EXPECTED_STAGES = [('elementary', 914), ('Fritz', 918), ('Fritz_kept', 919), ('Fritz_entropic', 924)]
EXPECTED_LADDER = [914, 915, 918, 918, 919, 919, 924]
EXPECTED_LOST = {
    'Fritz (dropped predictors), copy mode': 3,
    'Fritz_kept, copy mode': 1,
    'Fritz_entropic, kept predictors': 5,
    'Fritz_entropic, dropped predictors': 0,
    'Fritz_entropic certified by relabel': 5,
    'Fritz_entropic certified by markov': 0,
    'any copy-mode step': 4,
    'all Fritz_kept steps': 1,
    'all kept-predictor steps (Fritz_kept and Fritz_entropic with kept predictors)': 6,
    'all Fritz (dropped predictors) steps': 4,
    'all Fritz_entropic steps': 5,
    'all Fritz-type steps': 10,
}


@pytest.mark.slow
def test_search_counts(proving_QC_Gaps):
    import entropic_lp
    from qc_gap_search import fritz_breakdown, ladder
    cheap, report, cache = proving_QC_Gaps.run_search(verbose=False, with_entropic=True, use_cache=False)
    assert cache is None
    assert cheap.counts == EXPECTED_CHEAP
    assert report.counts['inputs'] == 990 and report.counts['labelled_inputs'] == 2759
    assert report.counts['proven'] == 924 and report.counts['remaining'] == 66
    assert report.stage_counts == EXPECTED_STAGES
    assert [proven for _, proven, _ in ladder(report)] == EXPECTED_LADDER
    assert fritz_breakdown(report) == EXPECTED_LOST
    assert entropic_lp.TIMEOUTS[0] == 0
    # Every proven input has a certificate ending at a named seed.
    for gid in report.proven:
        assert report.seed_hit[gid] in report.seeds
