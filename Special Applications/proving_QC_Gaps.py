"""
Which 4-node temporally-ordered causal structures have a quantum-classical (QC) gap?

Every not-provably-algebraic representative is expanded under all piggyback tricks (point distribution,
interruption, conditioning, marginalization with and without teleportation, Fritz) composed in any order, and is
proven to have a QC gap when some reachable structure is a known gap (a seed from known_QC_gaps.py, or an input
already proven). The report lists how many inputs each trick proves on its own, how many are provable only with it,
and a certificate (the chain of tricks down to a seed) for every proven input.
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
from qc_gap_search import prove_gaps, GapReport, MARGINALIZATION_TRICKS


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


def run_search(QmDAGs4_representatives=None, max_visible=5, verbose=True) -> GapReport:
    if QmDAGs4_representatives is None:
        QmDAGs4_representatives = four_node_representatives()
    seed_ids = set(g.unique_unlabelled_id for g in SEEDS_4_NODES.values())
    inputs = [g for g in QmDAGs4_representatives if g.unique_unlabelled_id not in seed_ids]
    print("Total number of qmDAGs to analyze: ", len(inputs))
    print("Number of representatives that are known QC Gaps: ", len(QmDAGs4_representatives) - len(inputs))
    return prove_gaps(inputs, SEEDS, max_visible=max_visible, verbose=verbose)


def proven_through_fritz(report: GapReport):
    """Inputs whose certificate uses the Fritz trick (and otherwise only marginalization)."""
    found = []
    for g in report.inputs:
        chain_ = report.proven.get(g.unique_unlabelled_id)
        if chain_ and any(t.trick == 'Fritz' for t in chain_) \
                and all(t.trick == 'Fritz' or t.trick in MARGINALIZATION_TRICKS for t in chain_):
            found.append(g)
    return found


def print_report(report: GapReport, certificates_for=()) -> None:
    counts = report.counts
    print("# of QC gaps proven: ", counts['proven'], f"({counts['proven_unique_ids']} distinct up to relabelling)")
    print("# still to be assessed: ", counts['remaining'])
    print("Provable using only this trick (closed under implication among the inputs):")
    for name, count in report.provable_with.items():
        print(f"    {name:>35}: {count}")
    print("Provable ONLY with this trick (lost when the trick is removed):")
    for name, count in report.only_via.items():
        print(f"    {name:>35}: {count}")
    print("Structures expanded by the search: ", len(report.explorer.edges))
    for g in certificates_for:
        print("-" * 60)
        print(g.as_string.rstrip())
        print(report.certificate(g))
    print("Note that here, we ARE considering Evans as if it had a QC Gap (only if both latents go quantum).")


if __name__ == '__main__':
    report = run_search()
    print_report(report, certificates_for=proven_through_fritz(report))
