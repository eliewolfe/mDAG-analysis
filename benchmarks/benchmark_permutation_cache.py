"""Compare shared hypergraph permutations with the previous per-object implementation.

Run from the repository root: python benchmarks/benchmark_permutation_cache.py --census --repeat 3
Each sample runs in a fresh process; modes alternate order. The census disables the disk proof cache and checks
all counts, stages, certificate chains, seed choices and recorded transitions by a deterministic digest. Time is
reported separately for a fresh census and a repeated related census in the same process. No cache is cleared
between those censuses. --output writes JSON evidence; omit --census for just the permutation workloads.
"""
from __future__ import annotations

import argparse
import contextlib
from functools import cached_property
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from radix import bitarray_to_int


def reference_permutations(self):
    return tuple(bitarray_to_int(bits) for bits in self.bit_array_permutations)


def worker(mode, census):
    import hypergraphs as H
    from metagraph_advanced import Observable_unlabelled_mDAGs

    if mode == 'reference':
        descriptor = cached_property(reference_permutations)
        descriptor.__set_name__(H.Hypergraph, 'as_integer_permutations')
        H.Hypergraph.as_integer_permutations = descriptor
    samples = list(Observable_unlabelled_mDAGs(4, verbose=False).all_simplicial_complices)

    def run_permutations(graphs):
        start = time.perf_counter()
        result = [graph.as_integer_permutations for graph in graphs]
        return time.perf_counter() - start, result

    # One object per distinct part with an empty shared cache, then repeated equal objects, then cached reads.
    H.clear_permutation_cache()
    unique_time, expected = run_permutations(samples)
    clones = [H.Hypergraph(graph.as_tuples, 4) for _ in range(20) for graph in samples]
    clones_time, actual = run_permutations(clones)
    assert actual == expected * 20
    cached_times = [run_permutations(clones)[0] for _ in range(20)]
    result = {'unique_parts_seconds': unique_time, 'equal_parts_seconds': clones_time,
              'cached_reads_seconds': statistics.median(cached_times), 'distinct_parts': len(samples),
              'equal_parts': len(clones)}
    if census:
        # Start with only the normal import-time cache state, independent of the microbenchmark above.
        # A second worker handles the census, so neither mode gets an artificial warm-up advantage.
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', mode, '--census-only']
        result.update(json.loads(subprocess.check_output(command, text=True, env=dict(os.environ, PYTHONHASHSEED='0'))))
    return result


def census_worker(mode):
    import hypergraphs as H
    if mode == 'reference':
        descriptor = cached_property(reference_permutations)
        descriptor.__set_name__(H.Hypergraph, 'as_integer_permutations')
        H.Hypergraph.as_integer_permutations = descriptor
    spec = importlib.util.spec_from_file_location('census', ROOT / 'Special Applications' / 'proving_QC_Gaps.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import qc_gap_search as search

    times, digests = [], []
    for _ in range(2):
        start = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            report, cache = module.run_search(verbose=False, use_cache=False)
        times.append(time.perf_counter() - start)
        assert cache is None
        assert [report.counts[k] for k in ('inputs', 'proven', 'remaining')] == [418, 372, 46]
        # repr preserves arbitrary-precision IDs and transition parameters without lossy integer conversion.
        payload = (sorted(report.counts.items()), report.stage_counts, search.ladder(report),
                   sorted(report.proven.items()), sorted(report.seed_hit.items()),
                   sorted((key, edges) for key, edges in report.explorer.edges.items()))
        digests.append(hashlib.sha256(repr(payload).encode()).hexdigest())
    assert digests[0] == digests[1], 'repeated census changed its proofs or recorded transitions'
    return {'fresh_census_seconds': times[0], 'related_census_seconds': times[1], 'census_digest': digests[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--census', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', choices=('reference', 'shared'), help=argparse.SUPPRESS)
    parser.add_argument('--census-only', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(census_worker(args.worker) if args.census_only else worker(args.worker, args.census)))
        return
    if args.repeat < 1:
        parser.error('--repeat must be positive')
    runs = {'reference': [], 'shared': []}
    for repeat in range(args.repeat):
        for mode in (('reference', 'shared') if repeat % 2 == 0 else ('shared', 'reference')):
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', mode]
            if args.census:
                command.append('--census')
            sample = json.loads(subprocess.check_output(command, text=True, env=dict(os.environ, PYTHONHASHSEED='0')))
            runs[mode].append(sample)
            print(f'{mode} sample {repeat + 1}: ' + ', '.join(f'{key}={value:.6f}' for key, value in sample.items()
                                                            if key.endswith('_seconds')), file=sys.stderr, flush=True)
    if args.census:
        assert len({sample['census_digest'] for samples in runs.values() for sample in samples}) == 1
    metrics = [key for key in runs['reference'][0] if key.endswith('_seconds')]
    result = {'samples': runs, 'median_seconds': {
        mode: {key: statistics.median(sample[key] for sample in samples) for key in metrics}
        for mode, samples in runs.items()}}
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
