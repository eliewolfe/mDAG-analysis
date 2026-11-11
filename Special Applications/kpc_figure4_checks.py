"""
The KPC structure step by step: Figure 4 of the Khanna-Pusey-Colbeck draft against the route the census records.
Everything quoted in manuscript/kpc_figure4_vs_code.md is printed by this script.

    python "Special Applications/kpc_figure4_checks.py"

Part 1 tests the "Corollary 5" step of Figure 4 (and its variants) as an implication between conditional-independence
statements over the visible nodes and the latents, with both certificate engines of the repository: the semigraphoid
closure (semigraphoid.py) and the entropic LP (entropic_lp.py, needs mosek; skipped without it). Part 2 verifies the
three-bit model that refutes the step as literally stated. Part 3 runs the repository's own Fritz step on the KPC
structure, and on the structure with F split by hand, and prints the certificate the census records.
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


if __name__ == '__main__':
    part_1()
    part_2()
    part_3()
