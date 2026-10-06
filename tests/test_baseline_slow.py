"""Pins the 4-node QC-gap census (base closure plus the entropic rescue). Takes about 20 minutes; run with
`pytest -m slow`. All counts are up to relabelling."""
import pytest

pytest.importorskip("mosek")

# Inputs: the 2759 labelled 4-node mDAGs that respect the order 0 < 1 < 2 < 3 and are not provably algebraic, with
# every latent quantum and the Bell seeds removed; 990 distinct up to relabelling.
# History: the hand-ordered pipeline this search replaced proved 2534 + 5 labelled structures (220 remaining).
# The closure search with the loose conditioning rule (visible grandparents only) proved 920 distinct inputs;
# requiring latent grandparents to be parents as well (tests/test_conditioning.py) leaves 918; the entropic rescue
# (Fritz_entropic on the 72 remaining inputs) proves 54 more.
EXPECTED = {
    'inputs': 990,
    'proven': 972,
    'remaining': 18,
    'labelled_inputs': 2759,
    'with PD': 860,
    'with interruption': 7,
    'with conditioning': 292,
    'with naive_marginalization': 515,
    'with teleportation_marginalization': 540,
    'with Fritz (+ marginalization)': 575,
    'with Fritz_entropic (+ Fritz, marginalization)': 615,
    'only via PD': 221,
    'only via interruption': 0,
    'only via conditioning': 23,
    'only via naive_marginalization': 0,
    'only via teleportation_marginalization': 0,
    'only via Fritz (+ marginalization)': 4,
    'only via Fritz_entropic (+ Fritz, marginalization)': 54,
}


@pytest.mark.slow
def test_search_counts(proving_QC_Gaps):
    report = proving_QC_Gaps.run_search(verbose=False, with_rescue=True)
    assert report.counts == EXPECTED
    # Every proven input has a certificate ending at a named seed.
    for gid in report.proven:
        assert report.seed_hit[gid] in report.seeds
