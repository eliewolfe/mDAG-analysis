"""
Which 4-node causal structures have a quantum-classical (QC) gap?

Inputs: the quantum observational equivalence classes of the 4-node mDAGs with every latent quantum that are not
provably algebraic (no member is classically equivalent to a latent-free structure). The classes come from the
quantum metagraph (metagraph_advanced.QuantumMetagraph, equivalence_classes.ClassIndex): its edges are structural
dominance, face splitting for the families of facets whose parts C_i no other facet touches
(mDAG.quantum_splittable_families) and the HLP edge rule in its pre-measurement form (mDAG.quantum_hlp_origin_ok).
For an HLP edge a -> b, either a is premeasurable or every member of a's facets except a and b is premeasurable.
Each strongly connected component is a proven quantum equivalence class. The classical metagraph uses strong face
splitting and the unrestricted HLP rule. ALL COUNTS ARE OVER CLASSES; the report also gives the number
of structures and of classical classes they comprise.

The seeds (known_QC_gaps.SEEDS, three- and four-node) are classes too: a seed whose classical facets each have
at most one member that is not premeasurable (QmDAG.lift) is the class of its all-quantum version, so the nine Bell seeds
are two classes and the three instrumental seeds one; the triangle seed with two classical facets does not lift and
stays a degradation target (its all-quantum class is proven through the degradation lookup). Seed classes that are
inputs are proven by the identity piggyback.

The primary search retains the ten-stage historical record, then expands the surviving branches with at most
one Fritz growth per path. The classical metagraphs screen provably gap-free branches through five nodes.
The named seeds anchor every bounded certificate, even if a disk gap cache exists. Use --max-visible and
--resume-at to select the output cap and the size for non-growing closure after a reduction. --depth-one
reproduces the original cascade alone; --no-screen disables gap-free pruning for comparison.

Options (see qc_gap_search.default_stages): the closure tries the `relabel` target set only (lp_markov_target=True
turns `markov` back on). The predictor pool of the certified stages is the latent siblings of the target that are
not its visible parents (those belong to the parent stage) nor its descendants; `--descendants` keeps the
descendants. `--engine semigraphoid|lp|both` chooses the certificate engine of the entropic stages.

Both searches record proven gaps in cache/known_gaps.json, keyed by class and versioned by the piggybacks used.
With --depth-one, valid cached gaps supply additional known goals. The primary search rebuilds its bounded
proofs from the named seeds on every run and can reuse the five-node gap-free screening table.
Changing a piggyback's version in qc_gap_search.PIGGYBACK_VERSIONS invalidates cached proofs that used it.
"""
from __future__ import absolute_import
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, List, Optional, Tuple

from quantum_mDAG import ENGINES, ENTROPIC_STATS, ENGINE_DISAGREEMENTS
import semigraphoid
from equivalence_classes import ClassIndex
from known_QC_gaps import SEEDS
from qc_gap_search import prove_gaps, default_stages, GapReport, FRITZ_TRICKS_ALL, ladder
from gap_cache import GapCache, CACHE_PATH
import entropic_lp


def build_index(verbose: bool = True) -> ClassIndex:
    """The quantum metagraphs (and their classical companions) for three and four nodes."""
    t0 = time.time()
    index = ClassIndex((3, 4))
    if verbose:
        for n in (3, 4):
            Q, C = index.quantum[n], index.classical[n]
            print(f"{n} nodes: {len(Q.dict_uniqind_unlabelled_mDAGs)} mDAGs up to relabelling; "
                  f"{len(C.equivalence_classes_as_ids)} classical and {len(Q.equivalence_classes_as_ids)} quantum "
                  f"equivalence classes; {len(index.input_keys(n))} input classes (not provably algebraic) with "
                  f"{index.all_quantum_structures(index.input_keys(n))} structures")
        print(f"[metagraphs built in {time.time() - t0:.1f}s]")
    return index


def census_inputs(index: ClassIndex) -> List:
    """One representative per four-node input class."""
    return index.inputs(4)


STAGE_SECONDS: Dict[str, float] = {}   # wall-clock seconds of each stage in the last run_search


DEFAULT_CAP = 6


def run_search(index: Optional[ClassIndex] = None, max_visible=None, verbose=True, with_entropic=True,
               with_split_predictors=True, strict_conditioning=True, use_cache=True, cache_path: str = CACHE_PATH,
               pool: str = 'siblings', allow_descendants: bool = False, early_exit: bool = True,
               engine: Optional[str] = None, *, deep=True, resume_at=None, screen=True, five_ids=None,
               screen_cache_dir=None) -> Tuple[GapReport, Optional[GapCache]]:
    """Return the full census record and optional on-disk gap cache. Bounded proofs start from named seeds.
    An explicit screen_cache_dir reuses just the five-node membership table during research queries."""
    if engine is not None and engine not in ENGINES:
        raise ValueError(f"unknown certificate engine {engine!r}; choose one of {ENGINES}")
    if max_visible is None:
        max_visible = DEFAULT_CAP if deep else 5
    resume_at = min(5, max(4, max_visible - 1)) if resume_at is None else resume_at
    if deep and (not 4 <= max_visible <= 6 or resume_at not in (4, 5)):
        raise ValueError('the deeper four-node census supports caps 4 through 6 and closure thresholds 4 or 5')
    ENGINE_DISAGREEMENTS.clear()
    STAGE_SECONDS.clear()
    if index is None:
        index = build_index(verbose=verbose)
    inputs = census_inputs(index)
    print(f"Census inputs: {len(inputs)} quantum equivalence classes "
          f"({index.all_quantum_structures(index.input_keys(4))} structures up to relabelling)")
    t0 = time.time()
    cache = GapCache.load(cache_path, seeds=SEEDS, keyed_by='class') if use_cache else None
    if cache is not None and verbose:
        print(f"Cache: {len(cache)} valid entries loaded, {cache.dropped} dropped (stale format or key kind, piggyback version or seed)")
    # A historical gap certificate may already have spent its growth allowance.
    # Regenerate bounded witnesses from named seeds, including on warm runs.
    known = cache.known() if cache is not None and not deep else {}
    baseline_cap = min(max_visible, 5) if deep else max_visible
    # lp_markov_target=True would also try the `markov` target set after `relabel` fails; it is off because in the
    # four-node census it never decided an input.
    stages = default_stages(max_visible=baseline_cap, with_entropic=with_entropic, with_split_predictors=with_split_predictors,
                            strict_conditioning=strict_conditioning, pool=pool, allow_descendants=allow_descendants,
                            lp_markov_target=False, engine=engine)
    report = prove_gaps(inputs, SEEDS, stages=stages, max_visible=baseline_cap, verbose=verbose, known=known,
                        early_exit=early_exit, index=index, on_stage=STAGE_SECONDS.__setitem__)
    if deep:
        from growth_census import extend_census
        if screen_cache_dir is None and use_cache:
            screen_cache_dir = Path(cache_path).parent
        start = time.monotonic()
        report = extend_census(report, max_visible, resume_at, five_ids=five_ids, screen=screen,
                               with_entropic=with_entropic, with_split_predictors=with_split_predictors,
                               allow_descendants=allow_descendants, engine=engine, verbose=verbose,
                               cache_dir=screen_cache_dir)
        STAGE_SECONDS[report.stage_counts[-1][0]] = time.monotonic() - start
    if cache is not None:
        added = cache.record(report)
        cache.save()
        if verbose:
            print(f"Cache: {added} entries added; {len(cache)} entries saved to {cache.path}")
    if verbose:
        print(f"[census done; total {time.time() - t0:.0f}s]")
    return report, cache


def proven_through_fritz(report: GapReport, tricks=FRITZ_TRICKS_ALL) -> List:
    """Inputs whose certificate uses one of the given Fritz-type tricks."""
    found, seen = [], set()
    for g in report.inputs:
        gid = report.key(g)
        chain_ = report.proven.get(gid)
        if gid not in seen and chain_ and any(t.trick in tricks for t in chain_):
            found.append(g)
            seen.add(gid)
    return found


def print_report(report: GapReport, certificates_for=()) -> None:
    counts = report.counts
    print("=" * 20, "The census over quantum equivalence classes", "=" * 20)
    print(f"Input classes: {counts['inputs']} ({counts.get('structures', counts['inputs'])} structures up to relabelling, "
          f"in {counts.get('classical classes of inputs', '?')} classical equivalence classes)")
    print(f"# of classes with a proven QC gap: {counts['proven']} ({counts.get('structures proven', '?')} structures; "
          f"{counts.get('classical classes of proven', '?')} classical classes)")
    print(f"# of classes still open: {counts['remaining']} ({counts.get('structures remaining', '?')} structures; "
          f"{counts.get('classical classes of remaining', '?')} classical classes, "
          f"{counts.get('classical classes with proven and unproven', '?')} of them also containing a proven class)")
    if report.known:
        print("# known gaps supplied by the cache: ", len(report.known))
    print("Seed classes (identity piggyback where they are inputs): ", ' | '.join(report.seed_class_names))
    print("Provable using only this piggyback (closed under implication among the inputs; lookups included):")
    for name, count in report.provable_with.items():
        print(f"    via {name:>35}: {count}")
    print("Provable ONLY with this piggyback (lost when it is removed):")
    for name, count in report.only_via.items():
        print(f"    only via {name:>30}: {count}")
    print("Proven after each stage (cumulative, over classes), and the stage's wall-clock time:")
    for name, count in report.stage_counts:
        seconds = STAGE_SECONDS.get(name)
        print(f"    {name:>50}: {count}" + (f"   ({seconds:.0f}s)" if seconds is not None else ""))
    if hasattr(report, "bounded"):
        print("Expanded search states by visible count and growth allowance:", dict(sorted(report.expanded_states.items())))
        print("Proven keys including intermediates:", len(report.proven_structure_ids()))
    else:
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
    print("Remaining classes (one representative each):")
    index = report.explorer.index
    for g in sorted(report.remaining, key=lambda g: report.key(g)):
        size = index.size(report.key(g)) if index is not None else 1
        print(f"    [{size} structure{'s' if size != 1 else ''}] " + g.as_string.replace("\n", " ").rstrip())


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Find proven quantum-classical gaps among four-node mDAGs.",
                                     allow_abbrev=False)
    if '--parents' in sys.argv:
        parser.error("--parents was removed: the parent piggyback is now the first Fritz stage of the cascade")
    parser.add_argument('--max-visible', type=int, help="maximum number of visible nodes in a search branch")
    parser.add_argument('--resume-at', type=int, choices=(4, 5), help="size at which non-growing closure resumes")
    parser.add_argument('--engine', choices=ENGINES)
    parser.add_argument('--depth-one', dest='deep', action='store_false', help="run the original cascade alone")
    parser.add_argument('--no-screen', dest='screen', action='store_false')
    parser.add_argument('--no-entropic', dest='with_entropic', action='store_false')
    parser.add_argument('--no-split-predictors', dest='with_split_predictors', action='store_false')
    parser.add_argument('--no-cache', dest='use_cache', action='store_false')
    parser.add_argument('--descendants', dest='allow_descendants', action='store_true')
    parser.add_argument('--no-early-exit', dest='early_exit', action='store_false')
    args = parser.parse_args()
    if args.deep and args.max_visible is not None and not 4 <= args.max_visible <= 6:
        parser.error("--max-visible must be between 4 and 6 for the deeper census")
    report, cache = run_search(**vars(args))
    print_report(report, certificates_for=proven_through_fritz(report))
