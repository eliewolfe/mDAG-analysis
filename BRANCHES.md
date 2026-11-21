# Maintained branches

This repository maintains two independent development branches:

* [`main`](https://github.com/eliewolfe/mDAG-analysis/tree/main): the forward census,
  with the wider quantum HLP Rule A (one facet whose members all see the origin's settings),
  Rule B (premeasurable bystanders), and lifting for classical facets with at most one
  member that cannot be premeasured.
* [`reversed-piggybacks`](https://github.com/eliewolfe/mDAG-analysis/tree/reversed-piggybacks):
  the elementary inverse closure of the seeds, followed by the forward Fritz cascade.
  Its display name is **reversed piggybacks**; Git branch names cannot contain spaces.
  Its quantum HLP rule requires a premeasurable origin, and its lifting rule applies to
  two-member classical facets with a premeasurable member.

Both quantum metagraphs use face splitting for families of facets whose parts $C_i$ no
outside facet touches. Both classical metagraphs use strong face splitting.

The reverse branch is the renamed `claude/pensive-cray-v991bo` branch from
[PR #16](https://github.com/eliewolfe/mDAG-analysis/pull/16). That work is maintained
separately and is not to be merged into `main`.

## Results belong to their branch

The two HLP rules group the four-node structures into different proven quantum
equivalence classes. Both searches prove gaps for the same structures.

| Four-node census | `main` | `reversed-piggybacks` |
|---|---:|---:|
| Quantum equivalence classes | 446 | 453 |
| Input classes | 426 | 432 |
| Proven classes | 380 | 386 |
| Remaining classes | 46 | 46 |
| Input structures, up to relabelling | 996 | 996 |
| Proven structures | 935 | 935 |
| Remaining structures | 61 | 61 |

The five-node quantum metagraph has 595,871 classes on `main` and 608,066 on
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
its rules change. The branches use distinct metagraph cache
versions. Use `--no-cache` for an independent comparison of their results.
Refresh manuscript code links with `python manuscript/refresh_links.py` after
changing code or headings, and validate changes against the branch's own tests.
On `main`, check every edited Markdown file with `python manuscript/check_math.py`,
which checks the [rules for mathematics rendered by GitHub](https://github.com/eliewolfe/mDAG-analysis/blob/main/CLAUDE.md).
