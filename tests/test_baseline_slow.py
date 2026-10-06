"""Pins the counts produced by the 4-node QC-gap pipeline. Takes several minutes; run with `pytest -m slow`."""
import pytest

# Counts observed before the Fritz overhaul (stage 1 must reproduce them exactly).
BASELINE_COUNTS = {
    'to_analyze': 2759,
    'already_known': 48,
    'PD': 2376,
    'interruption': 4,
    'naive_marginalization': 32,
    'teleportation_marginalization': 57,
    'conditioning': 65,
    'before_Fritz': 2534,
    'Fritz_without_splitting': 1,
    'Fritz_with_splitting': 4,
    'remaining': 220,
}


@pytest.mark.slow
def test_pipeline_counts(proving_QC_Gaps):
    counts = proving_QC_Gaps.run_pipeline()
    observed = {key: counts[key] for key in BASELINE_COUNTS}
    assert observed == BASELINE_COUNTS
