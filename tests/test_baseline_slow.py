"""Pins the 4-node QC-gap census (both phases, cache disabled); run with `pytest -m slow`. The default engine
(semigraphoid closure) takes about half a minute; the cross-check against the LP needs mosek and about two minutes.
All counts are up to relabelling."""
import importlib.util

import pytest

HAVE_MOSEK = importlib.util.find_spec("mosek") is not None

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
    'with conditioning': 292,
    'with naive_marginalization': 515,
    'with teleportation_marginalization': 540,
    'with marginalization (either kind)': 540,
    'only via PD': 244,
    'only via node_stitching': 3,
    'only via conditioning': 20,
    'only via naive_marginalization': 0,
    'only via teleportation_marginalization': 16,
    'only via marginalization (either kind)': 24,
}

# Phase 2: the weakest Bell variants are seeds (994 inputs: the two all-quantum Bell variants are seeds, the other
# all-quantum Bell variants are inputs proven by degradation), the cascade of four d-separation Fritz stages and four
# LP Fritz stages on what phase 1 left (1.2). The cumulative stage counts (by the stage in which each transition was
# recorded) must agree with the rungs of the ladder (by the recorded parameters).
EXPECTED_STAGES = [
    ('elementary', 921),
    ('Fritz, unsplit target, unsplit predictor, d-separation', 922),
    ('Fritz, unsplit target, split predictor, d-separation', 922),
    ('Fritz, split target, unsplit predictor, d-separation', 925),
    ('Fritz, split target, split predictor, d-separation', 926),
    ('Fritz, unsplit target, unsplit predictor, semigraphoid closure', 926),
    ('Fritz, unsplit target, split predictor, semigraphoid closure', 931),
    ('Fritz, split target, unsplit predictor, semigraphoid closure', 931),
    ('Fritz, split target, split predictor, semigraphoid closure', 931),
]
EXPECTED_LADDER = [921, 922, 922, 925, 926, 926, 931, 931, 931]


EXPECTED_CERTIFICATES = {('certificate', 'vacuous'): 404, ('certificate', 'dsep'): 398, ('certificate', 'relabel'): 54,
                         ('certificate', 'failed'): 668}


def _check(cheap, report, cache, proving_QC_Gaps):
    from qc_gap_search import ladder
    assert cache is None
    assert cheap.counts == EXPECTED_CHEAP
    assert report.counts['inputs'] == 994 and report.counts['labelled_inputs'] == 2801
    assert report.counts['proven'] == 931 and report.counts['remaining'] == 63
    assert report.stage_counts == EXPECTED_STAGES
    rungs = [proven for _, proven, _ in ladder(report)]
    assert rungs == EXPECTED_LADDER
    assert rungs == [count for _, count in EXPECTED_STAGES]
    # Every proven input has a certificate ending at a named seed.
    for gid in report.proven:
        assert report.seed_hit[gid] in report.seeds


@pytest.mark.slow
def test_search_counts(proving_QC_Gaps):
    """The census with the default engine, the semigraphoid closure."""
    import quantum_mDAG as QM
    QM.ENTROPIC_STATS.clear()
    cheap, report, cache = proving_QC_Gaps.run_search(verbose=False, with_entropic=True, use_cache=False, engine='semigraphoid')
    _check(cheap, report, cache, proving_QC_Gaps)
    stats = {k: v for k, v in QM.ENTROPIC_STATS.items() if k[0] == 'certificate'}
    assert stats == EXPECTED_CERTIFICATES
    assert QM.ENTROPIC_STATS[('engine', 'semigraphoid')] == 54


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_MOSEK, reason="mosek not installed")
def test_search_counts_with_both_engines_agree(proving_QC_Gaps):
    """The closure and the LP certify exactly the same candidate steps on the whole census."""
    import entropic_lp
    import quantum_mDAG as QM
    QM.ENTROPIC_STATS.clear()
    QM.ENGINE_DISAGREEMENTS.clear()
    cheap, report, cache = proving_QC_Gaps.run_search(verbose=False, with_entropic=True, use_cache=False, engine='both')
    _check(cheap, report, cache, proving_QC_Gaps)
    assert QM.ENGINE_DISAGREEMENTS == []
    assert ('engine', 'entropic') not in QM.ENTROPIC_STATS     # the LP never certified what the closure had not
    assert entropic_lp.TIMEOUTS[0] == 0
