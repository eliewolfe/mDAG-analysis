"""Pins the 4-node QC-gap search results. Takes several minutes; run with `pytest -m slow`."""
import pytest

# History of the hand-ordered pipeline this search replaced (inputs 2759): PD 2376, interruption 4, naive
# marginalization 32, teleportation marginalization 57, conditioning 65, Fritz 1 + 4 (original code; the same five
# with the corrected trick), 220 remaining. With the loose conditioning rule (visible grandparents only) the closure
# search proved 2539 (920 distinct); requiring latent grandparents to be parents as well (tests/test_conditioning.py)
# removes 8 of them (2 distinct up to relabelling).
EXPECTED = {
    'inputs': 2759,
    'proven': 2531,
    'remaining': 228,
    'proven_unique_ids': 918,
    'with PD': 2376,
    'with interruption': 36,
    'with conditioning': 973,
    'with naive_marginalization': 1630,
    'with teleportation_marginalization': 1687,
    'with Fritz (+ marginalization)': 1804,
    'only via PD': 394,
    'only via interruption': 0,
    'only via conditioning': 54,
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
