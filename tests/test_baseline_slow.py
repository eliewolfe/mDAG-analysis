"""Pins the 4-node QC-gap census (cache disabled); run with `pytest -m slow`. The default engine (semigraphoid
closure) takes about a minute; the cross-check against the LP needs mosek and about two minutes. All counts are over
quantum observational equivalence classes (equivalence_classes.ClassIndex)."""
import importlib.util

import pytest

HAVE_MOSEK = importlib.util.find_spec("mosek") is not None

# The inputs: the 418 quantum equivalence classes of the four-node mDAGs with every latent quantum that are not
# provably algebraic (996 structures up to relabelling, in 376 classical classes); every seed (three- and four-node)
# is known from the start and the two Bell seed classes are inputs proven by identity (manuscript/piggybacks.md,
# Section 1). The lookups (degradation, upward) are part of every group.
EXPECTED_COUNTS = {
    'inputs': 418,
    'proven': 372,
    'remaining': 46,
    'labelled_inputs': 418,
    'structures': 996,
    'structures proven': 935,
    'structures remaining': 61,
    'classical classes of inputs': 376,
    'classical classes of proven': 330,
    'classical classes of remaining': 46,
    'classical classes with proven and unproven': 0,
    'with identity': 2,
    'with PD': 341,
    'with node_stitching': 7,
    'with conditioning': 63,
    'with naive_marginalization': 170,
    'with teleportation_marginalization': 178,
    'with marginalization (either kind)': 178,
    'only via identity': 0,
    'only via PD': 164,
    'only via node_stitching': 0,
    'only via conditioning': 15,
    'only via naive_marginalization': 0,
    'only via teleportation_marginalization': 9,
    'only via marginalization (either kind)': 11,
}

# The cascade: the elementary stage (identity included), the parent piggyback, four d-separation Fritz stages and
# four closure Fritz stages. The cumulative stage counts (by the stage in which each transition was recorded) must
# agree with the rungs of the ladder (by the recorded parameters).
EXPECTED_STAGES = [
    ('elementary', 366),
    ('Fritz, parent predictor', 369),
    ('Fritz, unsplit target, unsplit predictor, d-separation', 370),
    ('Fritz, unsplit target, split predictor, d-separation', 370),
    ('Fritz, split target, unsplit predictor, d-separation', 370),
    ('Fritz, split target, split predictor, d-separation', 371),
    ('Fritz, unsplit target, unsplit predictor, semigraphoid closure', 371),
    ('Fritz, unsplit target, split predictor, semigraphoid closure', 372),
    ('Fritz, split target, unsplit predictor, semigraphoid closure', 372),
    ('Fritz, split target, split predictor, semigraphoid closure', 372),
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
def test_search_counts(proving_QC_Gaps):
    """The census with the default engine, the semigraphoid closure."""
    import quantum_mDAG as QM
    QM.ENTROPIC_STATS.clear()
    report, cache = proving_QC_Gaps.run_search(verbose=False, with_entropic=True, use_cache=False, engine='semigraphoid')
    _check(report, cache, proving_QC_Gaps)
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

@pytest.mark.slow
def test_open_class_hierarchy(proving_QC_Gaps):
    """manuscript/open_classes.md, 1.2: with each open class declared a known gap in turn, the open classes that
    become proven. Only class 1 (the numbering of manuscript Section 10, the order of report.remaining) has any:
    classes 2, 3, 6 and 7. The census records every transition out of an unproven class, so this is reachability."""
    report, _ = proving_QC_Gaps.run_search(verbose=False, with_entropic=True, use_cache=False, engine='semigraphoid')
    open_keys = [report.key(g) for g in report.remaining]
    assert len(open_keys) == 46
    reach = {key: report.explorer.reachable(key) for key in open_keys}
    follows = {i + 1: sorted(j + 1 for j, other in enumerate(open_keys) if other != key and key in reach[other])
               for i, key in enumerate(open_keys)}
    assert {i: lower for i, lower in follows.items() if lower} == {1: [2, 3, 6, 7]}


@pytest.mark.slow
def test_census_is_closed_and_the_equality_adds_nothing(proving_QC_Gaps):
    """manuscript/piggybacks.md 9.1 ("any known gap, in any order"): rerunning the census with every proven class
    and structure declared known from the start leaves the same classes open, so nothing is gained by restarting
    as gaps become known. And manuscript/kpc_figure4_vs_code.md, Section 6: on the open classes, giving the closure
    the perfect correlation in both directions certifies exactly the steps the one-directional prediction does."""
    import qc_gap_search as S
    from conftest import load_special_application
    from known_QC_gaps import SEEDS
    report, _ = proving_QC_Gaps.run_search(verbose=False, with_entropic=True, use_cache=False, engine='semigraphoid')
    open_keys = [report.key(g) for g in report.remaining]
    index = report.explorer.index
    known = {f'known:{i}': key for i, key in enumerate(sorted(report.proven_structure_ids()))}
    again = S.prove_gaps(proving_QC_Gaps.census_inputs(index), SEEDS, stages=S.default_stages(max_visible=5),
                         max_visible=5, verbose=False, known=known, index=index)
    assert {again.key(g) for g in again.remaining} == set(open_keys)
    checks = load_special_application("kpc_figure4_checks")
    (tally,) = checks.part_4(index, (("the 46 open classes", open_keys),)).values()
    assert (tally['pairs'], tally['the census step certifies'], tally['both directions certify (either candidate)']) == (504, 184, 184)
    assert tally['both directions and the census step disagree'] == 0
