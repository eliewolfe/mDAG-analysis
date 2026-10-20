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
