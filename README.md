# mDAG-analysis

## Maintained branches

This checkout documents **`reversed-piggybacks`**. The repository maintains [`main`](https://github.com/eliewolfe/mDAG-analysis/tree/main) and [`reversed-piggybacks`](https://github.com/eliewolfe/mDAG-analysis/tree/reversed-piggybacks) as separate development branches. See [the branch guide](BRANCHES.md) for their search modes, HLP rules, results, and maintenance policy.

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

Run the four-node census with `python "Special Applications/proving_QC_Gaps.py"` or pass `--no-cache` to recompute
the proofs. `metagraph_advanced.QuantumMetagraph` groups structures by structural dominance, quantum face splitting
and the HLP edge rule with a premeasurable origin. A family of facets can split at a common part when every node
in that part sees the required outcomes and settings and no outside facet touches the parts being separated from it.
`ClassicalMetagraph.from_quantum` uses strong face splitting and the unrestricted HLP rule for classical classes.
`equivalence_classes.ClassIndex` keys every structure by its class. The elementary
piggybacks run in reverse first (`reverse_piggybacks.py`, manuscript Appendix A): the seeds are closed under their
inverses below five nodes and the forward search only reduces five-node structures; `--reverse none` is the forward-only
run, `--reverse all` adds the Fritz inverse, `--no-hybrids` restricts the closure to all-quantum structures, and
`python "Special Applications/verify_reverse.py"` checks a configuration against the forward census. The nodes
of the metagraph are enumerated by orbits from permutation tables (`canonical_forms.py`: canonical ids on demand,
no dictionary over the labelled mDAGs) and held as scipy sparse matrices. On five nodes, the 1,718,596
unlabelled mDAGs form 490,800 classical and 608,066 quantum classes. Tests:
`pytest` (fast suite) and `pytest -m slow` (the census pins).
