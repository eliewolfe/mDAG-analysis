# Maintained branches

This repository maintains two independent development branches:

* [`main`](https://github.com/eliewolfe/mDAG-analysis/tree/main): the forward census,
  with the generalized pre-measurement HLP rule and hybrid lifting rule.
* [`reversed-piggybacks`](https://github.com/eliewolfe/mDAG-analysis/tree/reversed-piggybacks):
  the elementary inverse closure of the seeds, followed by the forward Fritz cascade.
  Its display name is **reversed piggybacks**; Git branch names cannot contain spaces.
  It retains the earlier HLP and lifting rules.

The reverse branch is the renamed `claude/pensive-cray-v991bo` branch from
[PR #16](https://github.com/eliewolfe/mDAG-analysis/pull/16). That work is maintained
separately and is not to be merged into `main`.

## Results belong to their branch

The HLP rules group the same structures into different proven quantum equivalence
classes. The different class counts below are intentional; they are not evidence
that reversing the elementary search proves additional all-quantum structures.

| Four-node census | `main` | `reversed-piggybacks` |
|---|---:|---:|
| Quantum equivalence classes | 438 | 445 |
| Input classes | 418 | 424 |
| Proven classes | 372 | 378 |
| Remaining classes | 46 | 46 |
| Input structures, up to relabelling | 996 | 996 |
| Proven structures | 935 | 935 |
| Remaining structures | 61 | 61 |

The five-node quantum metagraph has 583,674 classes on `main` and 595,074 on
`reversed-piggybacks`. Each branch's manuscript, cache, and test pins describe that
branch's rules. Read [the manuscript](manuscript/piggybacks.md) and
[notes for Marina](manuscript/comments_for_marina.md) from the branch being used.

## Run and validate

From the repository root, after installing the dependencies described in the
[README](README.md):

```sh
git switch main
python "Special Applications/proving_QC_Gaps.py" --no-cache
pytest
pytest -m slow -k test_search_counts

git switch reversed-piggybacks
python "Special Applications/proving_QC_Gaps.py" --no-cache
pytest
pytest -m slow -k test_search_counts
```

On `reversed-piggybacks`, `--reverse none` selects the forward-only search under
that branch's rules; it does not select `main`'s newer HLP rule. The default
elementary reverse closure and forward-only search have matching census pins.
`python "Special Applications/verify_reverse.py"` compares their proven sets,
counts, stages, ladder, and recorded forward edges. `--reverse all` also enables
the costly Fritz inverse (about fifty minutes in the recorded four-node run);
`--no-hybrids` restricts the reverse closure to all-quantum structures, retaining
the forward reductions. These reverse options and the verification script are
available only on `reversed-piggybacks`.

## Maintenance

Keep this guide and shared installation instructions consistent on both branches.
Apply shared fixes as selected commits to each branch, checking their diffs, rather
than merging either branch wholesale into the other. Keep algorithm descriptions,
results, test pins, and caches specific to their branch; regenerate results when
its rules change. The cache validates piggyback versions but does not encode the
branch name, so use `--no-cache` when comparing branches or after switching rules.
Refresh manuscript code links with `python manuscript/refresh_links.py` after
changing code or headings, and validate changes against the branch's own tests.
