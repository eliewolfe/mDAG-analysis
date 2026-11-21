"""Four-node gap census with a single growing Fritz move per witness path."""
from collections import Counter, defaultdict, deque
from enum import IntEnum
from pathlib import Path
import gzip
import os
import tempfile
from time import monotonic

import qc_gap_search as S
from metagraph_advanced import ClassicalMetagraph
from hypergraphs import Hypergraph, hypergraph_full_cleanup
from mDAG_advanced import mDAG


class Phase(IntEnum):
    AVAILABLE = 0
    MUST_REDUCE = 1
    SPENT = 2


def advance(phase, edge, resume_at):
    n, m = edge.source[0], edge.target[0]
    if edge.trick in S.LOOKUP_TRICKS:
        return phase
    if m > n:
        if phase == Phase.AVAILABLE and n <= 4 and edge.trick == 'Fritz':
            return Phase.MUST_REDUCE
        return None
    if m == n and (phase == Phase.MUST_REDUCE or n > resume_at):
        return None
    return Phase.SPENT if phase == Phase.MUST_REDUCE else phase


class BoundedPaths:
    """Search the product of the recorded graph and the remaining growth budget."""
    def __init__(self, explorer, resume_at):
        self.explorer = explorer
        self.resume_at = resume_at

    @property
    def reverse(self):
        return S._reverse_edges(self.explorer)

    def forward(self, root, goals=None):
        initial = (root, Phase.AVAILABLE)
        parents = {initial: None}
        queue = deque([initial])
        while queue:
            state = queue.popleft()
            key, phase = state
            if goals is not None and key in goals:
                path = []
                while parents[state] is not None:
                    state, edge = parents[state]
                    path.append(edge)
                return path[::-1]
            for edge in self.explorer.edges.get(key, ()):
                nxt = advance(phase, edge, self.resume_at)
                child = edge.target, nxt
                if nxt is not None and child not in parents:
                    parents[child] = state, edge
                    queue.append(child)
        if goals is None:
            return {key for key, phase in parents}
        return None

    def to_goals(self, goals, tricks=None, keep=None, *, reverse=None):
        """Shortest admissible suffix from every state, including zero-length seed hits."""
        if reverse is None:
            reverse = self.reverse
        routes = {(goal, phase): None for goal in sorted(goals) for phase in Phase}
        queue = deque(routes)
        while queue:
            key, phase = state = queue.popleft()
            for edge in reverse.get(key, ()):
                if (tricks is not None and edge.trick not in tricks) or (keep is not None and not keep(edge)):
                    continue
                for prev in Phase:
                    parent = (edge.source, prev)
                    if parent not in routes and advance(prev, edge, self.resume_at) == phase:
                        routes[parent] = edge, state
                        queue.append(parent)
        return routes


def underlying_mdag(g):
    facets = g.C_simplicial_complex_instance.simplicial_complex_as_sets | g.Q_simplicial_complex_instance.simplicial_complex_as_sets
    return mDAG(g.directed_structure_instance, Hypergraph(hypergraph_full_cleanup(facets), g.number_of_visible))


def quantum_part_is_free(g):
    classical = g.C_simplicial_complex_instance.as_tuples
    quantum = g.Q_simplicial_complex_instance.as_tuples
    incident = Counter(v for facets in (classical, quantum) for facet in facets for v in facet)
    has_parent = {b for _, b in g.directed_structure_instance.edge_list}
    return all(sum(v in has_parent or incident[v] > 1 for v in f) <= 1 for f in quantum)


def saturated_by_shared_source(g):
    """A clique of visible ancestors broadcasts to every member of one facet.

    The clique can sample an arbitrary joint law in topological order. All facet
    members know the clique values and can use one independent shared random
    variable to sample any joint conditional law (each reads its own component).
    This includes a single facet covering all nodes. Either source colour suffices
    in the classical model.
    """
    edges = set(map(tuple, g.directed_structure_instance.edge_list))
    nodes = set(g.visible_nodes)
    facets = g.C_simplicial_complex_instance.as_tuples + g.Q_simplicial_complex_instance.as_tuples
    for facet in facets:
        parents = nodes.difference(facet)
        if (all((p, v) in edges for p in parents for v in facet)
                and all((a, b) in edges or (b, a) in edges for a in parents for b in parents if a < b)):
            return True
    return False


def five_node_screen(cache_dir=None):
    """Cache just the membership set. IDs can exceed 64 bits; gzip verifies the payload CRC."""
    if cache_dir is None:
        return ClassicalMetagraph(5).latent_free_equivalent_ids
    version = S.PIGGYBACK_VERSIONS['metagraph']
    directory = Path(cache_dir)
    path = directory / f'latent-free-5-v{version}.txt.gz'
    try:
        with gzip.open(path, 'rt', encoding='ascii') as stream:
            return {int(line) for line in stream}
    except (OSError, EOFError, ValueError):
        ids = ClassicalMetagraph(5).latent_free_equivalent_ids
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as temporary:
        name = temporary.name
    try:
        with gzip.open(name, 'wt', encoding='ascii', compresslevel=1) as stream:
            stream.writelines(f'{uid}\n' for uid in sorted(ids))
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)
    return ids


class GapFreeScreen:
    def __init__(self, index, five_ids=None):
        self.tables = {n: graph.latent_free_equivalent_ids for n, graph in index.classical.items()}
        if five_ids is not None:
            self.tables[5] = five_ids
        self.classical = {}
        self.verdicts = {}

    def verdict(self, g):
        key = g.unique_unlabelled_id
        if key not in self.verdicts:
            n = g.number_of_visible
            if not g.Q_simplicial_complex_instance.as_tuples:
                reason = 'gap-free: no quantum facet'
            else:
                latent_free = False
                if n in self.tables:
                    md = underlying_mdag(g)
                    # Same labelled classical structure across different source colourings.
                    ck = (n, md.directed_structure_instance.as_integer, md.simplicial_complex_instance.as_integer)
                    if ck not in self.classical:
                        self.classical[ck] = md.unique_unlabelled_id in self.tables[n]
                    latent_free = self.classical[ck]
                if latent_free:
                    reason = 'gap-free: classically equivalent to a latent-free DAG'
                elif n not in self.tables and saturated_by_shared_source(g):
                    reason = 'gap-free: classically saturated by a shared source'
                elif quantum_part_is_free(g):
                    reason = 'gap-free: every quantum facet is equivalent to a classical one'
                else:
                    reason = 'candidate' if n in self.tables else f'unscreened ({n} nodes)'
            self.verdicts[key] = reason
        return self.verdicts[key]


def extend_census(report, cap, resume_at, *, five_ids, screen, with_entropic, with_split_predictors,
                  allow_descendants, engine, verbose, cache_dir=None):
    explorer = report.explorer
    explorer.max_visible = cap
    report.original_open_numbers = {report.key(g): i for i, g in enumerate(report.remaining, 1)}
    baseline_open = set(report.original_open_numbers)
    generated = dict.fromkeys(baseline_open, min(5, cap))
    elementary = frozenset(explorer.tricks)
    seeds = set(report.seed_keys)
    cheap = S._reverse_reachable(explorer, seeds, tricks=elementary | S.LOOKUP_TRICKS)
    explorer.begin_stage('screened Fritz closure with one growth', {'Fritz': None})
    if screen and cap >= 5 and five_ids is None:
        start = monotonic()
        five_ids = five_node_screen(cache_dir)
        if verbose:
            print(f'Five-node classical screen: {len(five_ids)} IDs in {monotonic()-start:.1f}s', flush=True)
    screening = GapFreeScreen(explorer.index, five_ids)
    modes = ('unsplit', 'split') if with_split_predictors else ('unsplit',)
    def generate(key, limit):
        old = generated.get(key, 2)
        if limit <= old:
            return
        for g in explorer.members(key):
            for pool in ('parents', 'siblings'):
                for pm in modes:
                    for tm in ('unsplit', 'split'):
                        single = key[0] + (tm == 'split') - (pm == 'unsplit')
                        if single > limit or (tm == 'unsplit' and single <= old):
                            continue
                        moves = g.fritz_steps(predictor_mode=pm, target_mode=tm, pool=pool,
                                              use_lp=with_entropic and pool == 'siblings', engine=engine,
                                              allow_descendants=allow_descendants, min_visible=old + 1, max_visible=limit)
                        explorer._record(key, [explorer._child_transition('Fritz', p, key, g, child) for p, child in moves])
        generated[key] = limit

    roots = set(report.input_ids) - cheap
    queue = deque((k, Phase.AVAILABLE) for k in sorted(roots))
    seen = set()
    expanded_states = Counter()
    # A lookup alone never launches expansion. If its target is independently
    # reached by a real move later, revisit every previously seen budget there.
    waiting = defaultdict(set)
    while queue:
        key, phase = state = queue.popleft()
        if state in seen or key in cheap:
            continue
        if key in explorer.lookup_only:
            waiting[key].add(phase)
            continue
        seen.add(state)
        if screen and screening.verdict(explorer.representatives[key]).startswith('gap-free'):
            continue
        n = key[0]
        expanded_states[n, phase.name] += 1
        limit = cap if phase == Phase.AVAILABLE and n <= 4 else n
        if phase == Phase.MUST_REDUCE or n > resume_at:
            limit = n - 1
        explorer._expand_one(key, elementary)
        generate(key, limit)
        for edge in explorer.edges.get(key, ()):
            nxt = advance(phase, edge, resume_at)
            if nxt is not None:
                queue.append((edge.target, nxt))
            if edge.trick not in S.LOOKUP_TRICKS:
                queue.extend((edge.target, other) for other in waiting.pop(edge.target, ()))

    paths = BoundedPaths(explorer, resume_at)
    explorer.bounded = paths
    paths.screening = screening
    paths.original_open_numbers = report.original_open_numbers
    paths.expanded_states = expanded_states
    return bounded_report(paths, report.inputs, report.seeds, S._trick_groups_for(explorer, None),
                          stage_counts=report.stage_counts)


def bounded_report(paths, inputs, named_seeds, groups, known=None, stage_counts=None):
    explorer = paths.explorer
    seed_ids = S._seed_keys(explorer, named_seeds, known)
    seeds = set(seed_ids)
    reverse = paths.reverse   # share one snapshot across all counts for this report
    routes = paths.to_goals(seeds, reverse=reverse)
    report = S.GapReport(inputs, named_seeds, {}, {}, [], {}, {}, explorer, [], known or {}, _seed_keys=seed_ids)
    report.bounded = paths
    report.screening = paths.screening
    report.original_open_numbers = paths.original_open_numbers
    report.expanded_states = paths.expanded_states
    report.bounded_keys = {k for k, phase in routes if phase == Phase.AVAILABLE}
    for k in report.input_ids:
        state = (k, Phase.AVAILABLE)
        if state in routes:
            chain = []
            while routes[state] is not None:
                edge, state = routes[state]
                chain.append(edge)
            report.proven[k] = chain
            report.seed_hit[k] = report.seed_keys[state[0]]
    report.remaining = [g for g in report.inputs if report.key(g) not in report.proven]
    inputs = set(report.input_ids)
    def count(tricks=None, keep=None):
        reach = paths.to_goals(seeds, tricks, keep, reverse=reverse)
        return sum((k, Phase.AVAILABLE) in reach for k in inputs)

    if stage_counts is None:
        stage_counts = [(name, count(keep=explorer.stage_filter(i)))
                        for i, name in enumerate(explorer.stage_names[:-1])]
    report.stage_counts = list(stage_counts) + [(explorer.stage_names[-1], len(report.proven))]
    for name, (allowed, excluded) in groups.items():
        report.provable_with[name] = count(allowed)
        report.only_via[name] = len(report.proven) - count(keep=lambda edge: edge.trick not in excluded)
    return report
