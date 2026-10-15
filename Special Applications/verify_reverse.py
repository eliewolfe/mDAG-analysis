"""
Checks a reverse-piggyback configuration against the forward census (manuscript Appendix A; reverse_piggybacks).

    python "Special Applications/verify_reverse.py" --reverse node_stitching [PD ...] [--no-hybrids] [--cap]

Runs the forward census (no reversed tricks, no cache) and the reverse-enabled census, then reports: (1) the proven
class keys of both (the reverse run must prove every class the forward run proves; extras are listed); (2) the
report counts against the slow-test pins; (3) the stage counts and the ladder; (4) generator completeness: every
forward transition of a reversed trick from an input member into a known key has a reverse edge with the same
(trick, source, target); (5) wall times; (6) the reverse closure's database and redundancy table.
"""
from __future__ import absolute_import
import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from typing import Dict, Set, Tuple

import proving_QC_Gaps as census
from known_QC_gaps import SEEDS
from qc_gap_search import ladder
from reverse_piggybacks import INVERSES, ReverseClosure
from tests.test_baseline_slow import EXPECTED_COUNTS, EXPECTED_STAGES


def proven_keys(report) -> Set[Tuple[int, int, int, int]]:
    return set(report.proven)


def verify(reversed_tricks, hybrids=True, cap=None, verbose=False):
    index = census.build_index(verbose=False)
    t0 = time.time()
    forward, _ = census.run_search(index=index, use_cache=False, verbose=verbose, reversed_tricks=())
    t_forward = time.time() - t0
    forward_seconds = dict(census.STAGE_SECONDS)
    t0 = time.time()
    reverse, _ = census.run_search(index=index, use_cache=False, verbose=verbose, reversed_tricks=tuple(reversed_tricks),
                                   reverse_hybrids=hybrids, reverse_cap=cap)
    t_reverse = time.time() - t0
    reverse_seconds = dict(census.STAGE_SECONDS)
    closure = census.LAST_CLOSURE
    ok = True
    print("=" * 20, f"reverse {sorted(reversed_tricks)} hybrids={hybrids} cap={cap}", "=" * 20)
    # 1. proven classes
    pf, pr = proven_keys(forward), proven_keys(reverse)
    print(f"1. proven classes: forward {len(pf)}, reverse {len(pr)}; missing in reverse {len(pf - pr)}, extra {len(pr - pf)}")
    for key in sorted(pf - pr):
        ok = False
        print("   MISSING", key, str(reverse.explorer.representatives.get(key, index.representative(key))).replace('\n', ' | '))
    for key in sorted(pr - pf):
        print("   extra  ", key, str(reverse.explorer.representatives[key]).replace('\n', ' | '))
    # 2. counts
    diffs = {k: (EXPECTED_COUNTS.get(k), v) for k, v in reverse.counts.items() if EXPECTED_COUNTS.get(k) != v}
    print(f"2. report counts against the pins: {'identical' if not diffs else diffs}")
    if any(k in ('proven', 'structures proven', 'remaining') for k in diffs):
        ok = False
    # 3. stages and ladder
    exp = dict(EXPECTED_STAGES)
    stage_diffs = [(name, exp.get(name), n) for name, n in reverse.stage_counts if exp.get(name) != n]
    print(f"3. stage counts: {'identical to the pins' if not stage_diffs else stage_diffs}")
    lf, lr = ladder(forward), ladder(reverse)
    print(f"   ladder: {'identical to forward' if lf == lr else (lf, lr)}")
    # 4. generator completeness on the reversed tricks
    if closure is not None:
        # forward edges the closure must reproduce: from a structure within the cap (and all-quantum when hybrids
        # are off) into a known key
        structures = forward.explorer.structures
        ft = {(t.trick, t.source, t.target) for ts in forward.explorer.edges.values() for t in ts
              if t.trick in set(reversed_tricks) and t.source[0] <= closure.max_visible
              and (hybrids or structures[t.via].is_all_quantum)}
        known = set(closure.known) | set(reverse.explorer.known_keys)
        relevant = {trip for trip in ft if trip[2] in known}
        rt = {(t.trick, t.source, t.target) for ts in closure.edges.values() for t in ts}
        missing = relevant - rt
        print(f"4. forward edges of the reversed tricks into known keys: {len(relevant)}, reproduced {len(relevant & rt)}, missing {len(missing)}")
        for trick, src, tgt in sorted(missing)[:20]:
            ok = False
            print("   MISSING", trick, str(forward.explorer.representatives[src]).replace('\n', ' | '), '->',
                  str(forward.explorer.representatives[tgt]).replace('\n', ' | '))
    # 5. times
    print(f"5. wall time: forward census {t_forward:.1f} s (elementary stage {forward_seconds.get('elementary', 0):.1f} s); "
          f"reverse census {t_reverse:.1f} s (closure {reverse_seconds.get('reverse closure', 0):.1f} s, "
          f"elementary stage {reverse_seconds.get('elementary', 0):.1f} s)")
    # 6. database
    if closure is not None:
        print("6.", closure.summary())
    print("RESULT:", "OK" if ok else "FAILED")
    return ok, forward, reverse, closure


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--reverse', nargs='*', default=list(census.DEFAULT_REVERSED), help="tricks to run in reverse ('all' for every inverse)")
    ap.add_argument('--no-hybrids', action='store_true')
    ap.add_argument('--cap', action='store_true', help="apply the reversed tricks forward only above four nodes")
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    tricks = list(INVERSES) if args.reverse == ['all'] else args.reverse
    ok, *_ = verify(tricks, hybrids=not args.no_hybrids, cap=4 if args.cap else None, verbose=args.verbose)
    sys.exit(0 if ok else 1)
