"""Pins the 4-node QC-gap search results. Takes several minutes; run with `pytest -m slow`."""
import pytest

# History of the hand-ordered pipeline this search replaced (inputs 2759): PD 2376, interruption 4, naive
# marginalization 32, teleportation marginalization 57, conditioning 65, Fritz 1 + 4 (original code; the same five
# with the corrected trick), 220 remaining.
EXPECTED = {
    'inputs': 2759,
    'proven': 2539,
    'remaining': 220,
    'proven_unique_ids': 920,
    'with PD': 2376,
    'with interruption': 36,
    'with conditioning': 1336,
    'with naive_marginalization': 1630,
    'with teleportation_marginalization': 1687,
    'with Fritz (+ marginalization)': 1804,
    'only via PD': 267,
    'only via interruption': 0,
    'only via conditioning': 62,
    'only via naive_marginalization': 0,
    'only via teleportation_marginalization': 0,
    'only via Fritz (+ marginalization)': 5,
}


@pytest.mark.slow
def test_search_counts(proving_QC_Gaps):
    report = proving_QC_Gaps.run_search(verbose=False)
    assert report.counts == EXPECTED
    # Every proven input has a certificate ending at a named seed.
    for g in report.inputs:
        if g.unique_unlabelled_id in report.proven:
            assert report.seed_hit[g.unique_unlabelled_id] in report.seeds
    assert len(proving_QC_Gaps.proven_through_fritz(report)) == 5
