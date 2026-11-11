"""
The KPC structure step by step: Figure 4 of the Khanna-Pusey-Colbeck draft against the route the census records.
Everything quoted in manuscript/kpc_figure4_vs_code.md is printed by this script.

    python "Special Applications/kpc_figure4_checks.py"

Part 1 tests the "Corollary 5" step of Figure 4 (and its variants) as an implication between conditional-independence
statements over the visible nodes and the latents, with both certificate engines of the repository: the semigraphoid
closure (semigraphoid.py) and the entropic LP (entropic_lp.py, needs mosek; skipped without it), and prints a
four-variable implication that the LP derives and the closure does not. Part 2 verifies the
three-bit model that refutes the step as literally stated. Part 3 runs the repository's own Fritz step on the KPC
structure, and on the structure with F split by hand, and prints the certificate the census records.

    python "Special Applications/kpc_figure4_checks.py" --census

adds the two census-wide comparisons of "perfect prediction" with "perfect correlation" (five minutes). Part 4:
for every structure of the census and every (predictor, predicted node) pair the search tries, the predictor is
split into an explicit copy and the closure is given the equality in one direction and in both; the pairs certified
are compared with the pairs the census's own step certifies. Part 5: the surgery that restricts the predicting copy
as well as the predicted node (arrows dropped into both equal nodes, the shape of Corollary 5) before the copy is
marginalized; it is applied to structures that provably have no gap, and its outputs are tested for known gaps.
"""
from __future__ import absolute_import
import collections
import importlib.util
import itertools
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx

import semigraphoid as sg

HAVE_MOSEK = importlib.util.find_spec("mosek") is not None
if HAVE_MOSEK:
    import entropic_lp as L

# Variables of the split structure (Figure 4, second graph): observed C, D, E, F_S, F_O; latents A, B.
C, D, E, FS, FO, A, B = range(7)
NAMES = {C: 'C', D: 'D', E: 'E', FS: 'F_S', FO: 'F_O', A: 'A', B: 'B'}
N_VARIABLES = 7


def fs(*x):
    return frozenset(x)


# Parent maps (a node that is absent has no parents).
SPLIT = {D: fs(B, C), E: fs(A, C, D), FS: fs(A, B), FO: fs(A, B)}            # after the Fritz II split of F
FIG4_TARGET = {D: fs(B, C), E: fs(A), FS: fs(A), FO: fs(A, B)}                # Figure 4, after "Corollary 5"
FIG4_RELABELLED = {D: fs(B, C), FS: fs(E), FO: fs(E, B)}                      # the same with A := E (A dropped)
OURS_TARGET = {D: fs(B, C), E: fs(A), FS: fs(A, B), FO: fs(A, B)}             # the code's candidate: F_S keeps B
OURS_RELABELLED = {D: fs(B, C), FS: fs(E, B), FO: fs(E, B)}                   # the code's candidate with A := E
BELL = {D: fs(B, C), FO: fs(E, B)}                                            # E -> F_O <- B -> D <- C

ALL = list(range(N_VARIABLES))
NO_A = [v for v in ALL if v != A]
NO_A_FS = [v for v in NO_A if v != FS]

CONSTRAINT = ('ci', ([E], [C, D], []))        # the draft's C: P(E|CD) = P(E)
FS_PREDICTS_E = ('fd', (E, [FS]))             # H(E | F_S) = 0
E_PREDICTS_FS = ('fd', (FS, [E]))             # H(F_S | E) = 0


def local_markov_triplets(parents, nodes):
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    for child, pa in parents.items():
        g.add_edges_from((p, child) for p in pa)
    out = []
    for v in nodes:
        pa = set(parents.get(v, ()))
        nondesc = set(nodes) - nx.descendants(g, v) - {v} - pa
        if nondesc:
            out.append((v, sorted(nondesc), sorted(pa)))
    return out


def show(triplet):
    v, nondesc, pa = triplet
    text = f"{NAMES[v]} _||_ {{{','.join(NAMES[x] for x in nondesc)}}}"
    return text + (f" | {{{','.join(NAMES[x] for x in pa)}}}" if pa else '')


def closure_model(source, hypotheses):
    """Markov(source) as its d-separation model, the hypotheses written in, closed under the semigraphoid rules. A
    functional dependence H(s|X) = 0 enters as the sufficiency statement <s, rest | X> (manuscript 7.3)."""
    model = sg.dsep_all(N_VARIABLES, sg.parents_to_masks(source, N_VARIABLES)).copy()
    for kind, args in hypotheses:
        if kind == 'ci':
            a, b, c = args
            sg.add_triplet(model, sg.mask_of(a), sg.mask_of(b), sg.mask_of(c))
        else:
            s, X = args
            sg.add_functional_dependence(model, s, sg.mask_of(X))
    sg.close(model)
    return model


def lp_model(source, hypotheses):
    rows = [row for _, row in L.local_markov_rows(source, range(N_VARIABLES))]
    for kind, args in hypotheses:
        if kind == 'ci':
            rows.append(L.cmi_row(*args))
        else:
            s, X = args
            rows.append(L.cond_entropy_row([s], X))
    return L.EntropicLP(N_VARIABLES, rows)


def check(title, hypotheses, target, nodes, source=SPLIT):
    """Is every local Markov statement of `target` (over `nodes`) implied by Markov(source) and the hypotheses?"""
    model = closure_model(source, hypotheses)
    lp = lp_model(source, hypotheses) if HAVE_MOSEK else None
    rows = []
    for triplet in local_markov_triplets(target, nodes):
        v, nondesc, pa = triplet
        closure_ok = sg.holds(model, 1 << v, sg.mask_of(nondesc), sg.mask_of(pa))
        lp_ok = lp.implies(L.cmi_row([v], nondesc, pa)) if lp is not None else None
        rows.append((show(triplet), lp_ok, closure_ok))
    if lp is not None:
        lp.close()
    closure_all = all(r[2] for r in rows)
    lp_all = all(r[1] for r in rows) if HAVE_MOSEK else None
    word = {True: 'certified', False: 'NOT certified', None: 'skipped (no mosek)'}
    print(f"\n{title}\n    LP: {word[lp_all]};  semigraphoid closure: {word[closure_all]}")
    for text, lp_ok, closure_ok in rows:
        if not closure_ok or lp_ok is False:
            print(f"      fails on  {text}   (LP: {'ok' if lp_ok else ('fails' if lp_ok is False else 'n/a')}, "
                  f"closure: {'ok' if closure_ok else 'fails'})")
    return lp_all, closure_all


# The twelve readings of the step (the table of manuscript/kpc_figure4_vs_code.md, Section 4): title, hypotheses
# besides Markov(split structure), target parent map, variables of the target.
BOTH_DIRECTIONS = [FS_PREDICTS_E, E_PREDICTS_FS]
ROWS = [
    ("1. As stated in Proposition 6: Markov(G') with the given latent A; hypotheses E _||_ CD and F_S = E",
     [CONSTRAINT] + BOTH_DIRECTIONS, FIG4_TARGET, ALL),
    ("2. The same target without the equality (Corollary 3 alone)", [CONSTRAINT], FIG4_TARGET, ALL),
    ("3. A relabelled (A := E): Markov(G'[A := E]); E _||_ CD and F_S = E in both directions",
     [CONSTRAINT] + BOTH_DIRECTIONS, FIG4_RELABELLED, NO_A),
    ("4. As 3 with H(E|F_S) = 0 only (F_S predicts E)", [CONSTRAINT, FS_PREDICTS_E], FIG4_RELABELLED, NO_A),
    ("5. As 3 with H(F_S|E) = 0 only", [CONSTRAINT, E_PREDICTS_FS], FIG4_RELABELLED, NO_A),
    ("6. As 3 without E _||_ CD (Fritz I alone)", BOTH_DIRECTIONS, FIG4_RELABELLED, NO_A),
    ("7. As 3 without the equality (conditional independences only)", [CONSTRAINT], FIG4_RELABELLED, NO_A),
    ("8. The code's candidate (F_S keeps B), given latent A; E _||_ CD and H(E|F_S) = 0",
     [CONSTRAINT, FS_PREDICTS_E], OURS_TARGET, ALL),
    ("9. The code's candidate, A := E, F_S still a node; E _||_ CD and H(E|F_S) = 0",
     [CONSTRAINT, FS_PREDICTS_E], OURS_RELABELLED, NO_A),
    ("10. The code's candidate, A := E, F_S still a node; E _||_ CD and F_S = E in both directions",
     [CONSTRAINT] + BOTH_DIRECTIONS, OURS_RELABELLED, NO_A),
    ("11. End to end, F_S removed: Markov(Bell) over C, D, E, F_O, B; E _||_ CD and H(E|F_S) = 0",
     [CONSTRAINT, FS_PREDICTS_E], BELL, NO_A_FS),
    ("12. End to end, F_S removed; E _||_ CD and F_S = E in both directions", [CONSTRAINT] + BOTH_DIRECTIONS, BELL, NO_A_FS),
]


def part_1():
    """Returns one (LP verdict or None without mosek, closure verdict) pair per row of ROWS."""
    print("=" * 100)
    print("PART 1. The 'Corollary 5' step of Figure 4, formalised several ways.")
    print("Source: the split structure (observed C, D, E, F_S, F_O; A -> E, F_S, F_O; B -> F_S, F_O, D; C -> D, E; D -> E).")
    results = [check(title, hypotheses, target, nodes) for title, hypotheses, target, nodes in ROWS]
    if HAVE_MOSEK:
        disagreements = [i + 1 for i, (lp_ok, closure_ok) in enumerate(results) if lp_ok != closure_ok]
        print(f"\nRows on which the two engines disagree: {disagreements or 'none'}")
    return results


def four_variable_example():
    """The entropic test is in principle the stronger one: for four variables A, B, C, D the independences
    A _||_ B | CD, C _||_ D | A, C _||_ D | B and A _||_ B imply C _||_ D | AB, A _||_ B | C, A _||_ B | D and
    C _||_ D (the two sums of conditional mutual informations are equal as entropy expressions), and no chain of
    semigraphoid steps derives any of the four. Returns (closure verdicts, LP verdicts or None without mosek)."""
    a, b, c, d = range(4)
    given = [([a], [b], [c, d]), ([c], [d], [a]), ([c], [d], [b]), ([a], [b], [])]
    wanted = [([c], [d], [a, b]), ([a], [b], [c]), ([a], [b], [d]), ([c], [d], [])]
    model = sg.empty_model(4)
    for x, y, z in given:
        sg.add_triplet(model, sg.mask_of(x), sg.mask_of(y), sg.mask_of(z))
    sg.close(model)
    by_closure = [bool(sg.holds(model, sg.mask_of(x), sg.mask_of(y), sg.mask_of(z))) for x, y, z in wanted]
    by_lp = None
    if HAVE_MOSEK:
        lp = L.EntropicLP(4, [L.cmi_row(x, y, z) for x, y, z in given])
        by_lp = [bool(lp.implies(L.cmi_row(x, y, z))) for x, y, z in wanted]
        lp.close()
    print("\nThe four-variable example (the entropic test is the stronger one in principle): of the four implied "
          f"independences the semigraphoid closure derives {sum(by_closure)}, the LP "
          + ("was skipped (no mosek)" if by_lp is None else f"derives {sum(by_lp)}"))
    return by_closure, by_lp


def part_2():
    print("\n" + "=" * 100)
    print("PART 2. The three-bit model: A, B, C uniform independent bits; D = B xor C; E = A xor C xor D (= A xor B);")
    print("F_S = A xor B; F_O = (A, B). It is Markov to the split structure, has F_S = E and E _||_ CD.")
    names = ['C', 'D', 'E', 'F_S', 'F_O', 'A', 'B']
    index = {name: i for i, name in enumerate(names)}
    dist = collections.Counter()
    for a, b, c in itertools.product((0, 1), repeat=3):
        d = b ^ c
        dist[(c, d, a ^ c ^ d, a ^ b, (a, b), a, b)] += 1 / 8

    def H(vs):
        marginal = collections.Counter()
        for outcome, p in dist.items():
            marginal[tuple(outcome[index[v]] for v in vs)] += p
        return -sum(p * math.log2(p) for p in marginal.values() if p > 0)

    def I(x, y, z=()):
        x, y, z = list(x), list(y), list(z)
        return round(H(x + z) + H(y + z) - H(x + y + z) - H(z), 9)

    print("  hypotheses (bits):")
    print("    I(E : B,F_S,F_O | A,C,D) =", I(['E'], ['B', 'F_S', 'F_O'], ['A', 'C', 'D']),
          "  I(F_S : C,D,E,F_O | A,B) =", I(['F_S'], ['C', 'D', 'E', 'F_O'], ['A', 'B']),
          "  I(D : A,F_S,F_O | B,C) =", I(['D'], ['A', 'F_S', 'F_O'], ['B', 'C']))
    print("    I(A : B,C,D) =", I(['A'], ['B', 'C', 'D']), "  I(B : A,C) =", I(['B'], ['A', 'C']),
          "  I(C : A,B,F_S,F_O) =", I(['C'], ['A', 'B', 'F_S', 'F_O']))
    print("    I(E : C,D) =", I(['E'], ['C', 'D']), "  H(E | F_S) =", round(H(['E', 'F_S']) - H(['F_S']), 9),
          "  H(F_S | E) =", round(H(['E', 'F_S']) - H(['E']), 9))
    print("  Markov(G') with the given A (should vanish, do not):")
    print("    I(E : B,C,D,F_S,F_O | A) =", I(['E'], ['B', 'C', 'D', 'F_S', 'F_O'], ['A']),
          "  I(F_S : B,C,D,E,F_O | A) =", I(['F_S'], ['B', 'C', 'D', 'E', 'F_O'], ['A']),
          "  I(B : A,C,E,F_S) =", I(['B'], ['A', 'C', 'E', 'F_S']))
    print("  with A relabelled to E (vanish):")
    print("    I(E : B,C,D) =", I(['E'], ['B', 'C', 'D']), "  I(F_O : C,D,F_S | E,B) =", I(['F_O'], ['C', 'D', 'F_S'], ['E', 'B']),
          "  I(D : E,F_S,F_O | B,C) =", I(['D'], ['E', 'F_S', 'F_O'], ['B', 'C']))


def part_3():
    print("\n" + "=" * 100)
    print("PART 3. The repository's Fritz step on the KPC structure (nodes 0 = F, 1 = C, 2 = D, 3 = E).")
    from hypergraphs import Hypergraph
    from directed_structures import DirectedStructure
    from quantum_mDAG import QmDAG
    import quantum_mDAG as QM
    import qc_gap_search as S
    from known_QC_gaps import SEEDS
    from equivalence_classes import ClassIndex
    kpc = QmDAG(DirectedStructure([(1, 2), (1, 3), (2, 3)], 4), Hypergraph([], 4), Hypergraph([(0, 2), (0, 3)], 4))
    engines = ('semigraphoid', 'lp') if HAVE_MOSEK else ('semigraphoid',)

    def certificate(G, s, X, label):
        deletion = G.fritz_deletion(s, X)
        if deletion is None:
            print(f"  {label}: no deletion (no channel, or nothing but the noise to delete)")
            return
        kept, deleted = deletion
        verdicts = {engine: G.fritz_certificate(s, kept, X, use_lp=True, lp_markov_target=True, tally=False, engine=engine)
                    for engine in engines}
        print(f"  {label}: " + ', '.join(f"{engine}: {verdict}" for engine, verdict in verdicts.items()))

    certificate(kpc, 3, [0], "KPC, predictor F, predicted E (targets `relabel` then `markov`)")
    for predictor_mode in ('unsplit', 'split'):
        for params, child in kpc.fritz_steps(target_mode='unsplit', predictor_mode=predictor_mode, lp_only=True,
                                             engine='both' if HAVE_MOSEK else 'semigraphoid', max_visible=5):
            p = dict(params)
            print(f"    step beyond d-separation: predicted {p['targets']}, predictor {p['predictor']}, {predictor_mode} "
                  f"predictor  =>  " + child.as_string.strip().replace('\n', ' | '))
    print("  engine disagreements recorded:", QM.ENGINE_DISAGREEMENTS or 'none')

    split = kpc.split_node(0)       # node 4 is the copy of F
    print("\n  F split by hand (node 4 = the copy):", split.as_string.strip().replace('\n', ' | '))
    certificate(split, 3, [4], "hand-split structure, the copy predicts E")
    certificate(split, 3, [0], "hand-split structure, F predicts E")

    index = ClassIndex((3, 4))
    report = S.prove_gaps([kpc], SEEDS, max_visible=5, verbose=False, index=index)
    print("\n  The certificate the census records for the KPC class:")
    print('    ' + report.certificate(kpc).replace('\n', '\n    '))
    report = S.prove_gaps([split], SEEDS, max_visible=5, verbose=False, index=index)
    print("  The hand-split five-node structure given as an input with only the seeds known: "
          + ("proven" if report.proven else "not proven (one Fritz step does not suffice; point distribution on the "
             "copy returns the KPC structure, which the census proves)"))


# -- Parts 4 and 5: prediction against correlation, across the census ------------------------------------------------

def closure_with_dependences(work, dependences, kept, s):
    """QmDAG._semigraphoid_certificate with any set of functional dependences (node, determining set): 'relabel',
    'markov' or None for the candidate whose parent map is `kept`."""
    nodes, _ = work.lp_structure
    n, nv = len(nodes), work.number_of_visible
    model = work._semigraphoid_model().copy()
    for node, given in dependences:
        sg.add_functional_dependence(model, node, sg.mask_of(given))
    kept_masks = sg.parents_to_masks(kept, n)
    model[:nv, :nv, :1 << nv] |= sg.dsep_all(n, kept_masks, m=nv)        # the observable d-separations of the candidate
    sg.close(model)
    common = kept[s]
    if len(common) == 1 and min(common) >= nv:
        lam = min(common)
        relabelled = {v: (ps - {lam}) | {s} if lam in ps else ps for v, ps in kept.items() if v != lam}
        relabelled[s] = relabelled[lam] = frozenset()
        target = sg.restrict(sg.dsep_all(n, sg.parents_to_masks(relabelled, n)), ((1 << n) - 1) & ~(1 << lam))
        if sg.contains(model, target):
            return 'relabel'
    return 'markov' if sg.contains(model, sg.dsep_all(n, kept_masks)) else None


def fritz_pairs(g):
    """The (predictor, predicted node) pairs the search tries on g, latent siblings as predictors."""
    pools = {s: g.fritz_pool(s) for s in g.visible_nodes}
    for x in sorted({x for candidates in pools.values() for x in candidates}):
        for s in g.fritz_targets((x,), pools=pools):
            yield x, s


def explicit_copy(g, x, s, target_mode):
    """g with the predictor x split into x and an explicit copy (and the target split in split mode): the structure,
    the predicted node, the copy, the kept set K of the predicted node, and K as parents of the copy."""
    work = g.split_node(x)
    copy = work.number_of_visible - 1
    t = s
    if target_mode == 'split':
        work = work.split_node(s)
        t = work.number_of_visible - 1
    deletion = work.fritz_deletion(t, {copy})
    if deletion is None:
        return None
    K = deletion[0]
    effective, _ = work.effective_DAG_data
    return work, t, copy, K, frozenset(p for p in effective.predecessors(copy) if p in K)


def census_certifies(g, x, s, target_mode):
    """Does the census's own step certify (x predicts s)? The hypothesis is the one direction H(s | predictor) = 0.
    The predicting node is x itself (an unsplit predictor, or a childless split one, which is left as it stands)
    or, for a childful x, its copy (manuscript 6.2); the first certificate found is returned."""
    work = g if target_mode == 'unsplit' else g.split_node(s)
    t = s if target_mode == 'unsplit' else work.number_of_visible - 1
    attempts = [(work, t, x)]
    if x not in g.vis_nodes_with_no_children:
        made = explicit_copy(g, x, s, target_mode)
        if made is not None:
            attempts.append(made[:3])
    for work, t, predictor in attempts:
        deletion = work.fritz_deletion(t, {predictor})
        if deletion is not None:
            certificate = work.fritz_certificate(t, deletion[0], {predictor}, use_lp=True, engine='semigraphoid', tally=False)
            if certificate is not None:
                return certificate
    return None


def part_4(index, groups):
    print("\n" + "=" * 100)
    print("PART 4. Prediction or correlation: which (predictor, predicted node) pairs each hypothesis certifies.")
    print("The predictor is split into an explicit copy x'; s is the predicted node. 'One direction' is H(s|x') = 0,")
    print("'both' adds H(x'|s) = 0. Two candidates: x' keeps its parents; x' is restricted to K as well (Corollary 5).")
    tallies = {}
    for label, keys in groups:
        tally = tallies[label] = collections.Counter()
        for key in keys:
            for g in index.members(key):
                for x, s in fritz_pairs(g):
                    for target_mode in ('unsplit', 'split'):
                        made = explicit_copy(g, x, s, target_mode)
                        if made is None:
                            continue
                        work, t, copy, K, K_copy = made
                        _, parents = work.lp_structure
                        keeps = dict(parents)
                        keeps[t] = frozenset(k for k in K if k in parents)
                        restricted = dict(keeps)
                        restricted[copy] = frozenset(p for p in parents[copy] if p in keeps[t])
                        census = census_certifies(g, x, s, target_mode) is not None
                        verdict = {(name, shape): closure_with_dependences(work, dependences, kept, t) is not None
                                   for name, dependences in (('one', [(t, {copy})]), ('both', [(t, {copy}), (copy, {t})]))
                                   for shape, kept in (('keeps', keeps), ('restricted', restricted))}
                        both = verdict[('both', 'keeps')] or verdict[('both', 'restricted')]
                        tally['pairs'] += 1
                        tally['the census step certifies'] += census
                        tally['both directions certify (either candidate)'] += both
                        tally['both directions and the census step disagree'] += both != census
                        for (name, shape), ok in verdict.items():
                            tally[f"{name} direction{'s' if name == 'both' else ''}, copy {shape}"] += ok
        print(f"\n  {label}:")
        for name, count in tally.items():
            print(f"    {count:>6}  {name}")
        assert tally['both directions and the census step disagree'] == 0
    return tallies


def part_5(index, explorer, seed_keys, gap_free, open_keys):
    import qc_gap_search as S
    from hypergraphs import Hypergraph
    from directed_structures import DirectedStructure
    from quantum_mDAG import QmDAG
    from known_QC_gaps import SEEDS
    print("\n" + "=" * 100)
    print("PART 5. Restricting the predicting copy too, then marginalizing it: is that a piggyback?")
    g = QmDAG(DirectedStructure([(0, 3), (1, 2)], 4), Hypergraph([], 4), Hypergraph([(0, 1), (0, 3), (1, 3)], 4))
    work, t, copy, K, K_copy = explicit_copy(g, 1, 0, 'split')
    theirs = [child for _, child in work.fritz_realise({t: K, copy: K_copy}, {copy})]
    ours = [child for _, child in work.fritz_realise({t: K}, {copy})]
    print("  A structure with no gap:", text(g), "| classically equivalent to a latent-free DAG:",
          index.is_latent_free_equivalent(index.key(g)))
    print("  node 1 predicts a copy of node 0; certificate of the census step:", census_certifies(g, 1, 0, 'split'))
    for label, children in (("copy keeps its parents (the code)", ours), ("copy restricted too", theirs)):
        for child in children:
            explorer.expand(child)
            known = explorer.key(child) in S._reverse_reachable(explorer, seed_keys)
            print(f"    output, {label}: {text(child)}  | reaches a known gap: {known}")
    reduced = theirs[0].subgraph((0, 2, 3))
    print("  the second output with nodes 1 and 4 fixed (point distribution twice):", text(reduced),
          "| is the seed QG_Instrumental_C:", reduced.unique_unlabelled_id == SEEDS['QG_Instrumental_C'].unique_unlabelled_id)
    print("  Do the three tests accept the deletion of arrows into both nodes? (They test the classical direction.)")
    for name, verdict in tests_of_both_deletions(work, t, copy, K).items():
        print(f"    {name}: {verdict}")

    elementary = frozenset(S.elementary_tricks(5))
    for label, keys in (("classes with no gap (latent-free equivalent, four nodes)", gap_free), ("open classes", open_keys)):
        differing, outputs = 0, set()
        for key in keys:
            for g in index.members(key):
                for x, s in fritz_pairs(g):
                    for target_mode in ('unsplit', 'split'):
                        if census_certifies(g, x, s, target_mode) is None:
                            continue
                        work, t, copy, K, K_copy = explicit_copy(g, x, s, target_mode)
                        ours = {child.unique_unlabelled_id for _, child in work.fritz_realise({t: K}, {copy})}
                        new = [child for _, child in work.fritz_realise({t: K, copy: K_copy}, {copy})
                               if child.unique_unlabelled_id not in ours and 3 <= child.number_of_visible <= 5]
                        differing += bool(new)
                        for child in new:
                            explorer.expand(child, only=elementary)
                            outputs.add(explorer.key(child))
        known = S._reverse_reachable(explorer, seed_keys)
        print(f"  {label}: {differing} certified pairs give a different output when the copy is restricted too; "
              f"{len(outputs & known)} of the {len(outputs)} distinct new outputs reach a known gap")

    steps, outputs = 0, set()
    for key in gap_free:
        for g in index.members(key):
            for target_mode in ('unsplit', 'split'):
                for predictor_mode in ('unsplit', 'split'):
                    for pool in ('siblings', 'parents'):
                        for _, child in g.fritz_steps(target_mode=target_mode, predictor_mode=predictor_mode, pool=pool,
                                                      use_lp=(pool == 'siblings'), engine='semigraphoid'):
                            steps += 1
                            if 3 <= child.number_of_visible <= 5:
                                explorer.expand(child, only=elementary)
                                outputs.add(explorer.key(child))
    known = S._reverse_reachable(explorer, seed_keys)
    print(f"  The same test of the code's own step: {steps} Fritz steps on the classes with no gap, {len(outputs)} "
          f"distinct outputs, {len(outputs & known)} of which reach a known gap")


def tests_of_both_deletions(work, t, copy, K):
    """The candidate in which the predicted node t and the predicting copy are both restricted to K, put to the
    three tests of the classical direction: d-separation with each of the two nodes as the predictor of the other,
    the semigraphoid closure and the entropic LP (needs mosek), the last two with the equality in one direction
    and in both."""
    _, parents = work.lp_structure
    nodes = work.lp_structure[0]
    both_restricted = dict(parents)
    both_restricted[t] = frozenset(k for k in K if k in parents)
    both_restricted[copy] = frozenset(p for p in parents[copy] if p in both_restricted[t])
    verdicts = {"d-separation, the copy predicts the node": work.fritz_certificate(t, K, {copy}, use_lp=False, tally=False)}
    reverse = work.fritz_deletion(copy, {t})
    verdicts["d-separation, the node predicts the copy"] = \
        None if reverse is None else work.fritz_certificate(copy, reverse[0], {t}, use_lp=False, tally=False)
    for label, dependences in (("one direction", [(t, {copy})]), ("both directions", [(t, {copy}), (copy, {t})])):
        verdicts[f"closure, {label}"] = closure_with_dependences(work, dependences, both_restricted, t)
        if HAVE_MOSEK:
            lp = work._entropic_lp()
            handle = lp.push_hypotheses([L.cond_entropy_row([s], given) for s, given in dependences]
                                        + L.observable_dseparation_rows(both_restricted, nodes, work.visible_nodes))
            try:
                certified = lp.implies_all(row for _, row in L.local_markov_rows(both_restricted, nodes))
            finally:
                lp.pop_to(handle)
            verdicts[f"LP, {label}"] = 'markov' if certified else None
    return verdicts


def text(g):
    edges = ', '.join(f'{a}→{b}' for a, b in sorted(map(tuple, g.directed_structure_instance.edge_list))) or 'no edges'
    facets = ['C{' + ','.join(map(str, f)) + '}' for f in sorted(map(tuple, g.C_simplicial_complex_instance.as_tuples))] \
        + ['Q{' + ','.join(map(str, f)) + '}' for f in sorted(map(tuple, g.Q_simplicial_complex_instance.as_tuples))]
    return f"{edges}; {', '.join(facets)}"


def census_parts():
    import qc_gap_search as S
    from known_QC_gaps import SEEDS
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import proving_QC_Gaps as census
    report, _ = census.run_search(verbose=False, use_cache=False)
    explorer, index = report.explorer, report.explorer.index
    open_keys = [report.key(g) for g in report.remaining]
    inputs = list(dict.fromkeys(report.input_ids))
    gap_free = [key for key, members in index._members.items()
                if members and members[0].number_of_visible == 4 and index.is_latent_free_equivalent(key)]
    part_4(index, (("the 46 open classes", open_keys), (f"all {len(inputs)} classes of the census", inputs)))
    part_5(index, explorer, set(report.seed_keys), gap_free, open_keys)


if __name__ == '__main__':
    part_1()
    four_variable_example()
    part_2()
    part_3()
    if '--census' in sys.argv[1:]:
        census_parts()
