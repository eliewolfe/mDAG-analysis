"""Pins the 4-node QC-gap census (both phases, cache disabled). Takes about three minutes; run with `pytest -m slow`.
All counts are up to relabelling."""
import pytest

pytest.importorskip("mosek")

# Phase 1: the 2807 labelled 4-node mDAGs that respect the order 0 < 1 < 2 < 3 and are not provably algebraic, with
# every latent quantum (996 distinct up to relabelling, Bell variants included), elementary reductions only, three-node
# seeds only (manuscript/piggybacks.md, 1.1). The degradation lookup is part of every group.
EXPECTED_CHEAP = {
    'inputs': 996,
    'proven': 917,
    'remaining': 79,
    'labelled_inputs': 2807,
    'with PD': 860,
    'with node_stitching': 10,
    'with conditioning': 289,
    'with naive_marginalization': 515,
    'with teleportation_marginalization': 540,
    'with marginalization (either kind)': 540,
    'only via PD': 246,
    'only via node_stitching': 3,
    'only via conditioning': 20,
    'only via naive_marginalization': 0,
    'only via teleportation_marginalization': 16,
    'only via marginalization (either kind)': 24,
}

# Phase 2: the weakest Bell variants are seeds (994 inputs: the two all-quantum Bell variants are seeds, the other
# all-quantum Bell variants are inputs proven by degradation), the cascade of four Fritz stages on what phase 1 left
# (1.2). The cumulative stage counts (by the stage in which each transition was recorded) must agree with the LP rungs
# of the ladder (by the recorded parameters).
EXPECTED_STAGES = [
    ('elementary', 921),
    ('Fritz, replace mode, dropped predictors', 922),
    ('Fritz, replace mode, kept predictors', 928),
    ('Fritz, copy mode, dropped predictors', 931),
    ('Fritz, copy mode, kept predictors', 931),
]
EXPECTED_LADDER = [921, 922, 922, 922, 928, 931, 931, 931, 931]


@pytest.mark.slow
def test_search_counts(proving_QC_Gaps):
    import entropic_lp
    from qc_gap_search import ladder
    cheap, report, cache = proving_QC_Gaps.run_search(verbose=False, with_entropic=True, use_cache=False)
    assert cache is None
    assert cheap.counts == EXPECTED_CHEAP
    assert report.counts['inputs'] == 994 and report.counts['labelled_inputs'] == 2801
    assert report.counts['proven'] == 931 and report.counts['remaining'] == 63
    assert report.stage_counts == EXPECTED_STAGES
    rungs = [proven for _, proven, _ in ladder(report)]
    assert rungs == EXPECTED_LADDER
    assert rungs[0::2] == [count for _, count in EXPECTED_STAGES]
    assert entropic_lp.TIMEOUTS[0] == 0
    # Every proven input has a certificate ending at a named seed.
    for gid in report.proven:
        assert report.seed_hit[gid] in report.seeds
