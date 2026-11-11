"""Pins the table of manuscript/kpc_figure4_vs_code.md: which readings of the 'Corollary 5' step of Figure 4 of the
KPC draft each certificate engine certifies (Special Applications/kpc_figure4_checks.py)."""
import importlib.util

import pytest

from conftest import load_special_application

HAVE_MOSEK = importlib.util.find_spec("mosek") is not None

# Row by row: the step as stated (given latent A) fails; with A relabelled it needs both directions of F_S = E while
# F_S is still a node (rows 3 and 10), and one direction once F_S is removed (rows 11 and 12).
EXPECTED = [False, False, True, False, False, False, False, False, False, True, True, True]


@pytest.fixture(scope="module")
def checks():
    return load_special_application("kpc_figure4_checks")


def test_closure_verdicts_and_agreement_with_the_lp(checks):
    results = checks.part_1()
    assert len(checks.ROWS) == len(EXPECTED) == 12
    assert [closure for _, closure in results] == EXPECTED
    if HAVE_MOSEK:
        assert [lp for lp, _ in results] == EXPECTED     # the two engines agree on every row


def test_the_three_bit_model_refutes_the_step_as_stated(checks):
    """A, B, C uniform bits, D = B xor C, E = A xor C xor D, F_S = A xor B: Markov to the split structure with
    F_S = E and E independent of (C, D), yet E is not a function of the given A."""
    import itertools
    C, D, E, FS, FO, A, B = checks.C, checks.D, checks.E, checks.FS, checks.FO, checks.A, checks.B
    for a, b, c in itertools.product((0, 1), repeat=3):
        d = b ^ c
        e = a ^ c ^ d
        assert e == a ^ b            # equals F_S, and depends on B once A is fixed
    # the closure row for E in the stated target is the one that fails
    model = checks.closure_model(checks.SPLIT, [checks.CONSTRAINT] + checks.BOTH_DIRECTIONS)
    sg = checks.sg
    assert not sg.holds(model, 1 << E, sg.mask_of([B]), sg.mask_of([A]))
    assert sg.holds(model, 1 << E, sg.mask_of([B, C, D]), 0)           # E is independent of B, C, D jointly
    assert sg.holds(model, 1 << FO, sg.mask_of([C, D, FS]), sg.mask_of([E, B]))
