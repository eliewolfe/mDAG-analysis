"""
Which 4-node causal structures have a quantum-classical (QC) gap?

Inputs: every 4-node mDAG whose directed edges respect the node order 0 < 1 < 2 < 3 ("temporally ordered") and that
is not provably algebraic (not equivalent to a latent-free structure), with every latent quantum. The metagraph
yields them as members of equivalence classes, so many inputs are relabellings of one another; ALL COUNTS ARE UP TO
RELABELLING (distinct unlabelled ids). Bell-scenario seeds are removed from the inputs.

Every input is expanded under all piggyback tricks (point distribution, interruption, conditioning, marginalization
with and without teleportation, Fritz) composed in any order, and is proven to have a QC gap when some reachable
structure is a known gap (a seed from known_QC_gaps.py, or an input already proven). The report lists how many
inputs each trick proves on its own, how many are provable only with it, and a certificate (the chain of tricks down
to a seed) for every proven input.
"""
from __future__ import absolute_import
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from itertools import chain

from quantum_mDAG import upgrade_to_QmDAG
from metagraph_temporally_ordered import Metagraph_temporally_ordered_mDAGs
from known_QC_gaps import SEEDS, SEEDS_4_NODES
from qc_gap_search import prove_gaps, rescue, entropic_tricks, default_tricks, GapReport, MARGINALIZATION_TRICKS
from quantum_mDAG import ENTROPIC_STATS


def four_node_representatives():
    """All temporally-ordered, not-provably-algebraic 4-node mDAGs (one per equivalence class member), as QmDAGs."""
    Metagraph4 = Metagraph_temporally_ordered_mDAGs(n=4, temporally_ordered=True)
    print("Number of temporally-ordered equivalence classes:", len(Metagraph4.equivalence_classes_as_ids))
    print("Number of temporally-ordered provably-algebraic equivalence classes:",
          len(Metagraph4.latent_free_equivalence_classes_as_ids))
    not_latent_free_classes = Metagraph4.NOT_latent_free_equivalence_classes_as_mDAGs
    print("Number of temporally-ordered not-provably-algebraic equivalence classes:", len(not_latent_free_classes))
    mDAGs4_representatives = list(chain.from_iterable(not_latent_free_classes))
    QmDAGs4_representatives = list(map(upgrade_to_QmDAG, mDAGs4_representatives))
    print("Number of temporally-ordered not-provably-algebraic mDAGs:", len(QmDAGs4_representatives))
    return QmDAGs4_representatives


def run_search(QmDAGs4_representatives=None, max_visible=5, verbose=True, with_rescue=False,
               strict_conditioning=True) -> GapReport:
    """Closure under the default piggybacks; with_rescue additionally applies the entropic (LP-certified) Fritz
    piggyback to whatever remains unproven."""
    if QmDAGs4_representatives is None:
        QmDAGs4_representatives = four_node_representatives()
    seed_ids = set(g.unique_unlabelled_id for g in SEEDS_4_NODES.values())
    inputs = [g for g in QmDAGs4_representatives if g.unique_unlabelled_id not in seed_ids]
    distinct = len(set(g.unique_unlabelled_id for g in inputs))
    print(f"Total number of qmDAGs to analyze: {distinct} up to relabelling ({len(inputs)} labelled)")
    print("Number of labelled representatives that are known Bell seeds: ", len(QmDAGs4_representatives) - len(inputs))
    report = prove_gaps(inputs, SEEDS, tricks=default_tricks(max_visible=max_visible, strict_conditioning=strict_conditioning),
                        max_visible=max_visible, verbose=verbose)
    if with_rescue:
        print("# still to be assessed before the entropic rescue: ", len(report.remaining))
        report = rescue(report, entropic_tricks(max_visible=max_visible), verbose=verbose)
    return report


FRITZ_TRICKS = ('Fritz', 'Fritz_entropic')


def proven_through_fritz(report: GapReport, tricks=FRITZ_TRICKS):
    """Inputs whose certificate uses one of the given Fritz-type tricks (and otherwise only marginalization)."""
    found = []
    for g in report.inputs:
        chain_ = report.proven.get(g.unique_unlabelled_id)
        if chain_ and any(t.trick in tricks for t in chain_) \
                and all(t.trick in tricks or t.trick in MARGINALIZATION_TRICKS for t in chain_):
            found.append(g)
    return found


def print_report(report: GapReport, certificates_for=()) -> None:
    counts = report.counts
    print(f"Inputs up to relabelling: {counts['inputs']} (from {counts['labelled_inputs']} labelled structures)")
    print("# of QC gaps proven: ", counts['proven'])
    print("# still to be assessed: ", counts['remaining'])
    print("Provable using only this trick (closed under implication among the inputs; up to relabelling):")
    for name, count in report.provable_with.items():
        print(f"    {name:>35}: {count}")
    print("Provable ONLY with this trick (lost when the trick is removed):")
    for name, count in report.only_via.items():
        print(f"    {name:>35}: {count}")
    print("Structures expanded by the search: ", len(report.explorer.edges))
    if ENTROPIC_STATS:
        print("Entropic certificates (kind, outcome) -> count: ", dict(sorted(ENTROPIC_STATS.items())))
    for g in certificates_for:
        print("-" * 60)
        print(g.as_string.rstrip())
        print(report.certificate(g))
    print("Note that here, we ARE considering Evans as if it had a QC Gap (only if both latents go quantum).")


if __name__ == '__main__':
    import sys
    with_rescue = '--no-rescue' not in sys.argv
    report = run_search(with_rescue=with_rescue)
    print_report(report, certificates_for=proven_through_fritz(report))
