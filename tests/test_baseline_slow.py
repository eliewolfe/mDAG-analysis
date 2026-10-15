"""Pins the 4-node QC-gap census (cache disabled); run with `pytest -m slow`. The default engine (semigraphoid
closure) takes about a minute; the cross-check against the LP needs mosek and about two minutes. All counts are over
quantum observational equivalence classes (equivalence_classes.ClassIndex)."""
import importlib.util

import pytest

HAVE_MOSEK = importlib.util.find_spec("mosek") is not None

# The inputs: the 424 quantum equivalence classes of the four-node mDAGs with every latent quantum that are not
# provably algebraic (996 structures up to relabelling, in 376 classical classes); every seed (three- and four-node)
# is known from the start and the two Bell seed classes are inputs proven by identity (manuscript/piggybacks.md,
# Section 1). The lookups (degradation, upward) are part of every group.
EXPECTED_COUNTS = {
    'inputs': 424,
    'proven': 378,
    'remaining': 46,
    'labelled_inputs': 424,
    'structures': 996,
    'structures proven': 935,
    'structures remaining': 61,
    'classical classes of inputs': 376,
    'classical classes of proven': 330,
    'classical classes of remaining': 46,
    'classical classes with proven and unproven': 0,
    'with identity': 2,
    'with PD': 345,
    'with node_stitching': 7,
    'with conditioning': 65,
    'with naive_marginalization': 172,
    'with teleportation_marginalization': 180,
    'with marginalization (either kind)': 180,
    'only via identity': 0,
    'only via PD': 166,
    'only via node_stitching': 0,
    'only via conditioning': 17,
    'only via naive_marginalization': 0,
    'only via teleportation_marginalization': 9,
    'only via marginalization (either kind)': 11,
}

# The cascade: the elementary stage (identity included), the parent piggyback, four d-separation Fritz stages and
# four closure Fritz stages. The cumulative stage counts (by the stage in which each transition was recorded) must
# agree with the rungs of the ladder (by the recorded parameters).
EXPECTED_STAGES = [
    ('elementary', 372),
    ('Fritz, parent predictor', 375),
    ('Fritz, unsplit target, unsplit predictor, d-separation', 376),
    ('Fritz, unsplit target, split predictor, d-separation', 376),
    ('Fritz, split target, unsplit predictor, d-separation', 376),
    ('Fritz, split target, split predictor, d-separation', 377),
    ('Fritz, unsplit target, unsplit predictor, semigraphoid closure', 377),
    ('Fritz, unsplit target, split predictor, semigraphoid closure', 378),
    ('Fritz, split target, unsplit predictor, semigraphoid closure', 378),
    ('Fritz, split target, split predictor, semigraphoid closure', 378),
]
EXPECTED_LADDER = [count for _, count in EXPECTED_STAGES]

EXPECTED_CERTIFICATES = {('certificate', 'parent'): 805, ('certificate', 'dsep'): 388, ('certificate', 'relabel'): 37,
                         ('certificate', 'failed'): 659}


def _expected_stages(engine):
    """EXPECTED_STAGES with the stage names of `engine` (the closure stages are named after it)."""
    from qc_gap_search import _RUNGS, _rung_name
    names = ['elementary'] + [_rung_name(*rung, engine=engine) for rung in _RUNGS]
    assert [name for name, _ in EXPECTED_STAGES] == ['elementary'] + [_rung_name(*rung) for rung in _RUNGS]
    return list(zip(names, [count for _, count in EXPECTED_STAGES]))


def _check(report, cache, proving_QC_Gaps, engine=None):
    from qc_gap_search import ladder
    assert cache is None
    assert report.counts == EXPECTED_COUNTS
    assert report.stage_counts == _expected_stages(engine)
    rungs = [proven for _, proven, _ in ladder(report)]
    assert rungs == EXPECTED_LADDER
    assert rungs == [count for _, count in EXPECTED_STAGES]
    # Every proven class has a certificate ending at a named seed class (a '+'-joined name when several seeds
    # share a class).
    seed_names = set(report.seeds)
    for gid in report.proven:
        assert all(part in seed_names for part in report.seed_hit[gid].split('+'))


@pytest.mark.slow
@pytest.mark.parametrize("reversed_tricks", ["default", ()], ids=["reverse-elementary", "forward-only"])
def test_search_counts(proving_QC_Gaps, reversed_tricks):
    """The census with the default engine, the semigraphoid closure: with the elementary piggybacks run in reverse
    (the default: the seeds closed under their inverses, the forward tricks applied to five-node structures only)
    and forward only; both give the same counts, stage counts and ladder."""
    import quantum_mDAG as QM
    QM.ENTROPIC_STATS.clear()
    kwargs = {} if reversed_tricks == "default" else {'reversed_tricks': reversed_tricks}
    report, cache = proving_QC_Gaps.run_search(verbose=False, with_entropic=True, use_cache=False, engine='semigraphoid', **kwargs)
    _check(report, cache, proving_QC_Gaps)
    if reversed_tricks == "default":
        closure = proving_QC_Gaps.LAST_CLOSURE
        assert closure is not None and set(closure.inverses) == set(proving_QC_Gaps.ELEMENTARY_REVERSED)
        assert len(closure.known) == 9814 and closure.known_by_nodes() == {3: (3, 2), 4: (372, 9437)}
        assert all(closure.stats[t]['rejected'] == 0 for t in closure.inverses)   # the forward check never rejects
    stats = {k: v for k, v in QM.ENTROPIC_STATS.items() if k[0] == 'certificate'}
    assert stats == EXPECTED_CERTIFICATES
    assert QM.ENTROPIC_STATS[('engine', 'semigraphoid')] == EXPECTED_CERTIFICATES[('certificate', 'relabel')]


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_MOSEK, reason="mosek not installed")
def test_search_counts_with_both_engines_agree(proving_QC_Gaps):
    """The closure and the LP certify exactly the same candidate steps on the whole census."""
    import entropic_lp
    import quantum_mDAG as QM
    QM.ENTROPIC_STATS.clear()
    QM.ENGINE_DISAGREEMENTS.clear()
    report, cache = proving_QC_Gaps.run_search(verbose=False, with_entropic=True, use_cache=False, engine='both')
    _check(report, cache, proving_QC_Gaps, engine='both')
    assert QM.ENGINE_DISAGREEMENTS == []
    assert ('engine', 'entropic') not in QM.ENTROPIC_STATS     # the LP never certified what the closure had not
    assert entropic_lp.TIMEOUTS[0] == 0
