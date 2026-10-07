"""Pins the 4-node QC-gap census (the four default stages). Takes about an hour; run with `pytest -m slow`.
All counts are up to relabelling."""
import pytest

pytest.importorskip("mosek")

# Inputs: the 2759 labelled 4-node mDAGs that respect the order 0 < 1 < 2 < 3 and are not provably algebraic, with
# every latent quantum and the Bell seeds removed; 990 distinct up to relabelling.
# Stages: elementary reductions 914, Fritz with dropped predictors 918, Fritz with kept predictors 919, LP-certified
# steps 924 (manuscript/piggybacks.md, Section 8).
EXPECTED = {
    'inputs': 990,
    'proven': 924,
    'remaining': 66,
    'labelled_inputs': 2759,
    'with PD': 860,
    'with interruption': 7,
    'with conditioning': 289,
    'with naive_marginalization': 515,
    'with teleportation_marginalization': 540,
    'with Fritz (+ marginalization)': 575,
    'with Fritz_kept (+ Fritz, marginalization)': 575,
    'with Fritz_entropic (+ Fritz, marginalization)': 580,
    'only via PD': 216,
    'only via interruption': 0,
    'only via conditioning': 20,
    'only via naive_marginalization': 0,
    'only via teleportation_marginalization': 0,
    'only via Fritz (+ marginalization)': 4,
    'only via Fritz_kept (+ Fritz, marginalization)': 1,
    'only via Fritz_entropic (+ Fritz, marginalization)': 5,
}


@pytest.mark.slow
def test_search_counts(proving_QC_Gaps):
    report = proving_QC_Gaps.run_search(verbose=False, with_entropic=True)
    assert report.counts == EXPECTED
    assert [c for _, c in report.stage_counts] == [914, 918, 919, 924]
    # Every proven input has a certificate ending at a named seed.
    for gid in report.proven:
        assert report.seed_hit[gid] in report.seeds
