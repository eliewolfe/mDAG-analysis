"""
Which 4-node causal structures have a quantum-classical (QC) gap?

Inputs: every 4-node mDAG whose directed edges respect the node order 0 < 1 < 2 < 3 ("temporally ordered") and that
is not provably algebraic (not equivalent to a latent-free structure), with every latent quantum. The metagraph
yields them as members of equivalence classes, so many inputs are relabellings of one another; ALL COUNTS ARE UP TO
RELABELLING (distinct unlabelled ids).

The census has two phases.

Phase 1 (cheap): the node-count-reducing piggybacks (point distribution, node stitching, conditioning,
marginalization with and without teleportation), composed in any order over everything reachable, starting from the
THREE-node known gaps only. The 4-node Bell variants are inputs here, not seeds, so this phase shows what each
elementary piggyback contributes: how many inputs it proves alone and how many are lost without it.

Phase 2 (expensive): the inputs that phase 1 left unproven, minus the Bell variants (which are known gaps and are
now seeds), are attacked by the staged cascade of the Fritz piggyback: four d-separation stages (unsplit target and
unsplit predictor, unsplit target and split predictor, split target and unsplit predictor, split target and split
predictor; joint target sets included, every root recorded in full), then the same four modes with the semigraphoid
closure (single targets, each root stopping at its first success; the entropic LP with --engine lp). Each stage is applied once to the still-unproven
inputs with elementary follow-up. The report gives the cumulative counts after each stage, which are the rungs of
the cheap-to-expensive ladder.

Throughout, a structure counts as known as soon as one of its degradations (some quantum facets made classical) is
known: the degradation piggyback is applied as a lookup to every structure the search meets. The seeds are therefore
kept in their weakest form only (known_QC_gaps.SEEDS).

Within an LP stage, a root stops as soon as one of its outputs (or that output's elementary follow-up) is a known
gap (`--no-early-exit` records everything instead); the d-separation stages record everything.

Options (see qc_gap_search.default_stages): the LP tries the `relabel` target set only, because the `markov` target
set never decided an input (lp_markov_target=True turns it back on). The predictor pool of a target is its latent
siblings that are not its descendants; `--parents` adds the visible parents of the target to the pool and
`--descendants` keeps the descendants (the experiments of manuscript 1.2; neither proves a further input).
`--engine semigraphoid|lp|both` chooses the certificate engine of the entropic stages (manuscript 7.9): the
semigraphoid closure (default, fast), the entropic LP (needs mosek), or both with every disagreement reported.

Proven gaps are cached on disk (cache/known_gaps.json) with the version of every piggyback their proof relies on;
cached gaps count as known, so after the first run the expensive stages only touch inputs that are not yet proven.
Bumping a piggyback's version in qc_gap_search.PIGGYBACK_VERSIONS discards exactly the cached proofs that used it.
"""
from __future__ import absolute_import
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from itertools import chain
from typing import Dict, List, Optional, Tuple

from quantum_mDAG import upgrade_to_QmDAG, ENTROPIC_STATS, ENGINE_DISAGREEMENTS
import semigraphoid
from metagraph_temporally_ordered import Metagraph_temporally_ordered_mDAGs
from known_QC_gaps import SEEDS, SEEDS_3_NODES, SEEDS_4_NODES
from qc_gap_search import (prove_gaps, add_stage, build_report, default_stages, GapReport, FRITZ_TRICKS_ALL,
                           TRICK_GROUPS_FOR_REPORT, ladder)
from gap_cache import GapCache, CACHE_PATH
import entropic_lp


def four_node_representatives():
    """All temporally-ordered, not-provably-algebraic 4-node mDAGs (one per equivalence class member), as QmDAGs."""
    Metagraph4 = Metagraph_temporally_ordered_mDAGs(n=4, temporally_ordered=True)
    print("Number of temporally-ordered equivalence classes:", len(Metagraph4.equivalence_classes_as_ids))
    print("Number of temporally-ordered provably-algebraic equivalence classes:",
          len(Metagraph4.latent_free_equivalence_classes_as_ids))
    not_latent_free_classes = Metagraph4.NOT_latent_free_equivalence_classes_as_mDAGs
    print("Number of temporally-ordered not-provably-algebraic equivalence classes:", len(not_latent_free_classes))
    mDAGs4_representatives = list(chain.from_iterable(not_latent_free_classes))
    QmDAGs4_representatives = list(map(upgrade_to_QmDAG, mDAGs4_representatives))
    print("Number of temporally-ordered not-provably-algebraic mDAGs:", len(QmDAGs4_representatives))
    return QmDAGs4_representatives


STAGE_SECONDS: Dict[str, float] = {}   # wall-clock seconds of each expensive stage in the last expensive_run


def cheap_run(QmDAGs4_representatives, max_visible=5, verbose=True, strict_conditioning=True) -> GapReport:
    """Phase 1: the elementary reductions only, from the three-node seeds, over all inputs (Bell variants included)."""
    stages = default_stages(max_visible=max_visible, strict_conditioning=strict_conditioning)[:1]
    return prove_gaps(QmDAGs4_representatives, SEEDS_3_NODES, stages=stages, max_visible=max_visible, verbose=verbose)


def expensive_run(cheap: GapReport, max_visible=5, verbose=True, with_entropic=True, with_split_predictors=True,
                  strict_conditioning=True, cache: Optional[GapCache] = None, pool: str = 'siblings',
                  allow_descendants: bool = False, early_exit: bool = True, engine: Optional[str] = None) -> GapReport:
    """Phase 2: the staged cascade on the inputs phase 1 left unproven. The Bell variants become seeds (so they are
    no longer inputs), and the cached gaps count as known. Shares the explorer of `cheap`, so the elementary
    transitions are not recomputed."""
    bell_ids = {g.unique_unlabelled_id for g in SEEDS_4_NODES.values()}
    inputs = [g for g in cheap.inputs if g.unique_unlabelled_id not in bell_ids]
    known = cache.known() if cache is not None else {}
    # lp_markov_target=True would also try the `markov` LP target set after `relabel` fails; it is off because in the
    # four-node census it never decided an input.
    stages = default_stages(max_visible=max_visible, with_entropic=with_entropic, with_split_predictors=with_split_predictors,
                            strict_conditioning=strict_conditioning, pool=pool, allow_descendants=allow_descendants,
                            lp_markov_target=False, engine=engine)
    # Re-base the report on the phase-2 inputs, seeds and known gaps; then run the cascade on what is left.
    report = build_report(cheap.explorer, inputs, SEEDS, TRICK_GROUPS_FOR_REPORT, known=known)
    for name, extra, roots_only, followup, stage_exit in stages[1:]:
        t0 = time.time()
        report = add_stage(report, extra, verbose=verbose, roots_only=roots_only, name=name, followup=followup,
                           early_exit=early_exit and stage_exit)
        STAGE_SECONDS[name] = time.time() - t0
    return report


def run_search(QmDAGs4_representatives=None, max_visible=5, verbose=True, with_entropic=True, with_split_predictors=True,
               strict_conditioning=True, use_cache=True, cache_path: str = CACHE_PATH, pool: str = 'siblings',
               allow_descendants: bool = False, early_exit: bool = True,
               engine: Optional[str] = None) -> Tuple[GapReport, GapReport, Optional[GapCache]]:
    """Both phases. Returns (phase-1 report, phase-2 report, cache). The cache is loaded before phase 2 (entries whose
    piggyback versions are stale are dropped), updated with every newly proven input, and saved."""
    from quantum_mDAG import ENGINES
    if engine is not None and engine not in ENGINES:
        raise ValueError(f"unknown certificate engine {engine!r}; choose one of {ENGINES}")
    ENGINE_DISAGREEMENTS.clear()
    if QmDAGs4_representatives is None:
        QmDAGs4_representatives = four_node_representatives()
    distinct = len(set(g.unique_unlabelled_id for g in QmDAGs4_representatives))
    print(f"Total number of qmDAGs to analyze: {distinct} up to relabelling ({len(QmDAGs4_representatives)} labelled)")
    t0 = time.time()
    cheap = cheap_run(QmDAGs4_representatives, max_visible=max_visible, verbose=verbose, strict_conditioning=strict_conditioning)
    if verbose:
        print(f"[phase 1 done in {time.time() - t0:.0f}s]")
    cache = GapCache.load(cache_path, seeds=SEEDS) if use_cache else None
    if cache is not None and verbose:
        print(f"Cache: {len(cache)} valid entries loaded, {cache.dropped} dropped (stale piggyback version or seed)")
    report = expensive_run(cheap, max_visible=max_visible, verbose=verbose, with_entropic=with_entropic,
                           with_split_predictors=with_split_predictors, strict_conditioning=strict_conditioning, cache=cache, pool=pool,
                           allow_descendants=allow_descendants, early_exit=early_exit, engine=engine)
    if cache is not None:
        added = cache.record(report)
        cache.save()
        if verbose:
            print(f"Cache: {added} entries added in the final pass; {len(cache)} entries saved to {cache.path}")
    if verbose:
        print(f"[phase 2 done; total {time.time() - t0:.0f}s]")
    return cheap, report, cache


def proven_through_fritz(report: GapReport, tricks=FRITZ_TRICKS_ALL) -> List:
    """Inputs whose certificate uses one of the given Fritz-type tricks."""
    found, seen = [], set()
    for g in report.inputs:
        gid = g.unique_unlabelled_id
        chain_ = report.proven.get(gid)
        if gid not in seen and chain_ and any(t.trick in tricks for t in chain_):
            found.append(g)
            seen.add(gid)
    return found


def print_cheap_report(report: GapReport) -> None:
    counts = report.counts
    bell_ids = {g.unique_unlabelled_id: name for name, g in SEEDS_4_NODES.items()}
    print("=" * 20, "Phase 1: elementary piggybacks from the three-node seeds", "=" * 20)
    print(f"Inputs up to relabelling: {counts['inputs']} (from {counts['labelled_inputs']} labelled structures; "
          f"Bell variants included as inputs)")
    print("# of QC gaps proven: ", counts['proven'])
    print("# left for phase 2: ", counts['remaining'])
    print("Provable using only this piggyback (closed under implication among the inputs):")
    for name, count in report.provable_with.items():
        print(f"    via {name:>35}: {count}")
    print("Provable ONLY with this piggyback (lost when it is removed):")
    for name, count in report.only_via.items():
        print(f"    only via {name:>30}: {count}")
    input_ids = set(report.input_ids)
    bell_inputs = {gid: name for gid, name in bell_ids.items() if gid in input_ids}
    proven_bell = [name for gid, name in bell_inputs.items() if gid in report.proven]
    print(f"Bell variants that are census inputs (every facet quantum): {len(bell_inputs)} of {len(bell_ids)}; "
          f"proven from the three-node seeds: {len(proven_bell)}")
    for gid, name in sorted(bell_ids.items(), key=lambda kv: kv[1]):
        chain_ = report.proven.get(gid)
        print(f"    {name}: " + ("not a census input (classical facets)" if gid not in input_ids else
                                 "unproven" if chain_ is None else
                                 " -> ".join(t.trick for t in chain_) + " -> " + report.seed_hit[gid]))


def print_report(report: GapReport, certificates_for=()) -> None:
    counts = report.counts
    print("=" * 20, "Phase 2: the cascade on the remaining inputs", "=" * 20)
    print(f"Inputs up to relabelling: {counts['inputs']} (from {counts['labelled_inputs']} labelled structures; "
          f"Bell variants are seeds)")
    print("# of QC gaps proven: ", counts['proven'])
    print("# still to be assessed: ", counts['remaining'])
    if report.known:
        print("# known gaps supplied by the cache: ", len(report.known))
    print("Proven after each stage (cumulative, up to relabelling), and the stage's wall-clock time:")
    for name, count in report.stage_counts:
        seconds = STAGE_SECONDS.get(name)
        print(f"    {name:>50}: {count}" + (f"   ({seconds:.0f}s)" if seconds is not None else ""))
    print("Cheap to expensive, cumulatively (which transitions are allowed):")
    for name, proven, new in ladder(report):
        print(f"    {name:>50}: proven {proven:4d}  (new {new:3d})")
    print("Structures expanded by the search: ", len(report.explorer.edges))
    if ENTROPIC_STATS:
        print("Certificates (kind, outcome) -> count: ", dict(sorted(ENTROPIC_STATS.items())))
    print("Semigraphoid closures: ", int(semigraphoid.STATS['closures']), f"({semigraphoid.STATS['seconds']:.1f}s incl. d-separation models;"
          f" numba {'on' if semigraphoid.HAVE_NUMBA else 'off'})")
    print("LP solves: ", entropic_lp.SOLVES[0], "; hit the time limit: ", entropic_lp.TIMEOUTS[0])
    if ENGINE_DISAGREEMENTS:
        print("ENGINE DISAGREEMENTS (closure vs LP): ", len(ENGINE_DISAGREEMENTS))
        for item in ENGINE_DISAGREEMENTS:
            print("    ", item)
    for g in certificates_for:
        print("-" * 60)
        print(g.as_string.rstrip())
        print(report.certificate(g))
    print("Note that here, we ARE considering Evans as if it had a QC Gap (only if both latents go quantum).")


if __name__ == '__main__':
    with_entropic = '--no-entropic' not in sys.argv
    with_split_predictors = '--no-split-predictors' not in sys.argv
    use_cache = '--no-cache' not in sys.argv
    pool = 'siblings+parents' if '--parents' in sys.argv else 'siblings'
    allow_descendants = '--descendants' in sys.argv
    early_exit = '--no-early-exit' not in sys.argv
    engine = sys.argv[sys.argv.index('--engine') + 1] if '--engine' in sys.argv and sys.argv.index('--engine') + 1 < len(sys.argv) else None
    cheap, report, cache = run_search(with_entropic=with_entropic, with_split_predictors=with_split_predictors, use_cache=use_cache, pool=pool,
                                      allow_descendants=allow_descendants, early_exit=early_exit, engine=engine)
    print_cheap_report(cheap)
    print_report(report, certificates_for=proven_through_fritz(report))
