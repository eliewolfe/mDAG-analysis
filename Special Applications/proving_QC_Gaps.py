"""
Which 4-node causal structures have a quantum-classical (QC) gap?

Inputs: the quantum observational equivalence classes of the 4-node mDAGs with every latent quantum that are not
provably algebraic (no member is classically equivalent to a latent-free structure). The classes come from the
quantum metagraph (metagraph_advanced.QuantumMetagraph, equivalence_classes.ClassIndex): its edges are structural
dominance, face splitting and the HLP edge rule where the origin node has no visible parent and one latent parent,
all sound for quantum model sets, so each strongly connected component is a proven quantum equivalence class. ALL
COUNTS ARE OVER CLASSES; the report also gives the number of structures and of classical classes they comprise.

The seeds (known_QC_gaps.SEEDS, three- and four-node) are classes too: a seed with classical facets that are
2-facets with a premeasurable member (QmDAG.lift) is the class of its all-quantum version, so the nine Bell seeds
are two classes and the three instrumental seeds one; the triangle seed with two classical facets does not lift and
stays a degradation target (its all-quantum class is proven through the degradation lookup). Seed classes that are
inputs are proven by the identity piggyback.

One phase: the staged cascade of qc_gap_search.default_stages over every input class, cheapest first. (1) The
elementary reductions (point distribution, node stitching, conditioning, marginalization with and without
teleportation), closed over everything reachable, with the lookups: degradation (a structure is known when one of
its degradations is) and the upward rule (a class is known when a class structurally below it inside the same
classical class is: its quantum model contains the lower one's and its classical model equals it). (2) The parent
piggyback of the Fritz trick. (3-6) Four d-separation Fritz stages. (7-10) The same four modes with the
semigraphoid closure (the entropic LP with --engine lp). Each stage applies its trick to every member of each
still-unproven class, with elementary follow-up. The report gives the cumulative counts after each stage, the
rungs of the cheap-to-expensive ladder.

Within a closure stage, a root stops as soon as one of its outputs (or that output's elementary follow-up) is a
known gap (`--no-early-exit` records everything instead); the d-separation stages record everything.

Options (see qc_gap_search.default_stages): the closure tries the `relabel` target set only (lp_markov_target=True
turns `markov` back on). The predictor pool of the certified stages is the latent siblings of the target that are
not its visible parents (those belong to the parent stage) nor its descendants; `--descendants` keeps the
descendants. `--engine semigraphoid|lp|both` chooses the certificate engine of the entropic stages.

Proven gaps are cached on disk (cache/known_gaps.json, keyed by class) with the version of every piggyback their
proof relies on; cached gaps count as known, so after the first run the expensive stages only touch classes that
are not yet proven. Bumping a piggyback's version in qc_gap_search.PIGGYBACK_VERSIONS discards exactly the cached
proofs that used it.
"""
from __future__ import absolute_import
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, List, Optional, Tuple

from quantum_mDAG import ENTROPIC_STATS, ENGINE_DISAGREEMENTS
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


def run_search(index: Optional[ClassIndex] = None, max_visible=5, verbose=True, with_entropic=True,
               with_split_predictors=True, strict_conditioning=True, use_cache=True, cache_path: str = CACHE_PATH,
               pool: str = 'siblings', allow_descendants: bool = False, early_exit: bool = True,
               engine: Optional[str] = None) -> Tuple[GapReport, Optional[GapCache]]:
    """The census: one staged search over the four-node input classes with every seed known. Returns (report,
    cache). The cache is loaded first (entries whose piggyback versions are stale are dropped), updated with every
    newly proven class, and saved."""
    from quantum_mDAG import ENGINES
    if engine is not None and engine not in ENGINES:
        raise ValueError(f"unknown certificate engine {engine!r}; choose one of {ENGINES}")
    ENGINE_DISAGREEMENTS.clear()
    STAGE_SECONDS.clear()
    if index is None:
        index = build_index(verbose=verbose)
    inputs = census_inputs(index)
    print(f"Census inputs: {len(inputs)} quantum equivalence classes "
          f"({index.all_quantum_structures(index.input_keys(4))} structures up to relabelling)")
    t0 = time.time()
    cache = GapCache.load(cache_path, seeds=SEEDS) if use_cache else None
    if cache is not None and verbose:
        print(f"Cache: {len(cache)} valid entries loaded, {cache.dropped} dropped (stale format, piggyback version or seed)")
    known = cache.known() if cache is not None else {}
    # lp_markov_target=True would also try the `markov` target set after `relabel` fails; it is off because in the
    # four-node census it never decided an input.
    stages = default_stages(max_visible=max_visible, with_entropic=with_entropic, with_split_predictors=with_split_predictors,
                            strict_conditioning=strict_conditioning, pool=pool, allow_descendants=allow_descendants,
                            lp_markov_target=False, engine=engine)
    report = prove_gaps(inputs, SEEDS, stages=stages, max_visible=max_visible, verbose=verbose, known=known,
                        early_exit=early_exit, index=index, on_stage=STAGE_SECONDS.__setitem__)
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
    print("Remaining classes (one representative each):")
    index = report.explorer.index
    for g in sorted(report.remaining, key=lambda g: report.key(g)):
        size = index.size(report.key(g)) if index is not None else 1
        print(f"    [{size} structure{'s' if size != 1 else ''}] " + g.as_string.replace("\n", " ").rstrip())


if __name__ == '__main__':
    with_entropic = '--no-entropic' not in sys.argv
    with_split_predictors = '--no-split-predictors' not in sys.argv
    use_cache = '--no-cache' not in sys.argv
    if '--parents' in sys.argv:
        raise SystemExit("--parents was removed: the parent piggyback is now the first Fritz stage of the cascade")
    pool = 'siblings'
    allow_descendants = '--descendants' in sys.argv
    early_exit = '--no-early-exit' not in sys.argv
    engine = sys.argv[sys.argv.index('--engine') + 1] if '--engine' in sys.argv and sys.argv.index('--engine') + 1 < len(sys.argv) else None
    report, cache = run_search(with_entropic=with_entropic, with_split_predictors=with_split_predictors, use_cache=use_cache, pool=pool,
                               allow_descendants=allow_descendants, early_exit=early_exit, engine=engine)
    print_report(report, certificates_for=proven_through_fritz(report))
