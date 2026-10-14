mDAG-analysis

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

The four-node census: `python "Special Applications/proving_QC_Gaps.py"` (about half a minute; `--no-cache` to
recompute everything). Tests: `pytest` (fast suite) and `pytest -m slow` (the census pins).
