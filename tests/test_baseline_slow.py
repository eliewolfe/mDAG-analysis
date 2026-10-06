"""Pins the counts produced by the 4-node QC-gap pipeline. Takes several minutes; run with `pytest -m slow`."""
import pytest

# Counts after the corrected Fritz piggyback (stage 2). Before the correction the two Fritz passes found 1 + 4
# graphs and 220 remained; four of those five relied on steps the corrected formulation does not license.
BASELINE_COUNTS = {
    'to_analyze': 2759,
    'already_known': 48,
    'PD': 2376,
    'interruption': 4,
    'naive_marginalization': 32,
    'teleportation_marginalization': 57,
    'conditioning': 65,
    'before_Fritz': 2534,
    'Fritz': 1,
    'remaining': 224,
}


@pytest.mark.slow
def test_pipeline_counts(proving_QC_Gaps):
    counts = proving_QC_Gaps.run_pipeline()
    observed = {key: counts[key] for key in BASELINE_COUNTS}
    assert observed == BASELINE_COUNTS
