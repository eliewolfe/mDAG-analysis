# mDAG-analysis

## Maintained branches

This checkout documents **`main`**. The repository maintains [`main`](https://github.com/eliewolfe/mDAG-analysis/tree/main) and [`reversed-piggybacks`](https://github.com/eliewolfe/mDAG-analysis/tree/reversed-piggybacks) as separate development branches. See [the branch guide](BRANCHES.md) for their search modes, HLP rules, results, and maintenance policy.

## Installation

The package is managed with poetry (`poetry install`). Two dependencies deserve a note:

* `numba` (optional group `jit`, `poetry install --with jit`) compiles the kernels of `semigraphoid.py`: the
  semigraphoid closure, the default certificate engine of the Fritz piggyback (manuscript Section 7), and the
  Bayes-ball enumerator of d-separations and e-separations on bitmasks that `mDAG` uses for its conditional-
  independence and e-separation sets, its equivalence-class keys and the metagraph edge tests. Without it the same
  code runs on numpy, about ten times slower, still well ahead of the LP.
* `mosek` (optional group `lp`, `poetry install --with lp`; a licence file is needed) enables the entropic LP
  (manuscript Appendix B) as an alternative engine: `python "Special Applications/proving_QC_Gaps.py" --engine lp`, or `--engine both` to
  cross-check the two engines on every candidate.

The four-node census: `python "Special Applications/proving_QC_Gaps.py"` (under two minutes; `--no-cache` to
recompute everything). It searches over quantum observational equivalence classes: `metagraph_advanced.QuantumMetagraph`
builds the metagraph whose edges are quantum-valid (structural dominance, face splitting, and the HLP edge rule in
its pre-measurement form: the origin node is premeasurable, or every other member of its facets is),
`ClassicalMetagraph.from_quantum` adds the remaining HLP edges for the classical classes, and `equivalence_classes.ClassIndex` keys every structure by its class. The nodes
of the metagraph are enumerated by orbits from permutation tables (`canonical_forms.py`: canonical ids on demand,
no dictionary over the labelled mDAGs) and held as scipy sparse matrices; both five-node metagraphs (1,718,596
unlabelled mDAGs, 490,800 classical and 583,674 quantum classes) build in about five minutes within 3.5 GB. Tests:
`pytest` (fast suite) and `pytest -m slow` (the census pins).

Full metagraph enumeration supports up to five visible nodes with the current 64-bit permutation tables. Larger
requests fail before generating directed structures or hypergraph families. The standalone `CanonicalForms` API
still accepts explicitly supplied larger-node families when their encodings fit its bit limits.

Hypergraph permutation integers are shared across equal numeric parts, in the original permutation order, and
remain cached on each object. Shared entries have no automatic eviction, so related searches retain their reuse.
After a completed batch, when its derived results are no longer useful, `quantum_mDAG.clear_search_caches()` releases
the canonical-ID, piggyback-child, LP and semigraphoid-model caches and closes cached solver tasks. Its keyword
switches allow retaining selected groups; for example, `canonical_ids=False, piggyback_children=False` keeps both
of those caches. Call it only between searches, with no LP task in use. It preserves recorded proofs and shared
hypergraph permutations. Clear those separately with `hypergraphs.clear_permutation_cache()` only when future
work will not reuse the same parts. Neither cleanup is invoked automatically by a search or report operation.

Compare the shared permutation results with the previous implementation using
`python benchmarks/benchmark_permutation_cache.py --census --repeat 3`; omit `--census` for only the part-level
benchmarks. The full comparison includes both fresh and related repeated searches and verifies their proofs.
