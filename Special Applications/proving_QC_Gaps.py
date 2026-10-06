from __future__ import absolute_import
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypergraphs import Hypergraph
from directed_structures import DirectedStructure
from quantum_mDAG import QmDAG, upgrade_to_QmDAG
from metagraph_temporally_ordered import Metagraph_temporally_ordered_mDAGs
from itertools import chain


# ---------------------------------------------------------------------------
# Known QC gaps with 3 visible nodes
# ---------------------------------------------------------------------------
QG_Instrumental1 = QmDAG(DirectedStructure([(0, 1), (1, 2)], 3), Hypergraph([], 3), Hypergraph([(1, 2)], 3))
QG_Instrumental2 = QmDAG(DirectedStructure([(0, 1), (1, 2)], 3), Hypergraph([], 3), Hypergraph([(0, 1), (1, 2)], 3))
QG_Instrumental3 = QmDAG(DirectedStructure([(1, 2)], 3), Hypergraph([], 3), Hypergraph([(0, 1), (1, 2)], 3))
QG_Instrumental2b = QmDAG(DirectedStructure([(0, 1), (1, 2)], 3), Hypergraph([(0, 1)], 3), Hypergraph([(1, 2)], 3))
QG_Instrumental3b = QmDAG(DirectedStructure([(1, 2)], 3), Hypergraph([(0, 1)], 3), Hypergraph([(1, 2)], 3))

IV_ids = set(special_QmDAG.unique_unlabelled_id for special_QmDAG in {QG_Instrumental1, QG_Instrumental2, QG_Instrumental3,
                                                                       QG_Instrumental2b, QG_Instrumental3b})

QG_Triangle1 = QmDAG(DirectedStructure([], 3), Hypergraph([], 3), Hypergraph([(1, 2), (2, 0), (0, 1)], 3))
QG_Triangle2 = QmDAG(DirectedStructure([], 3), Hypergraph([(1, 2)], 3), Hypergraph([(2, 0), (0, 1)], 3))
QG_Triangle3 = QmDAG(DirectedStructure([], 3), Hypergraph([(1, 2), (2, 0)], 3), Hypergraph([(0, 1)], 3))

Triangle_ids = set(special_QmDAG.unique_unlabelled_id for special_QmDAG in {QG_Triangle1, QG_Triangle2, QG_Triangle3})

QG_Evans = QmDAG(DirectedStructure([(0, 1), (0, 2)], 3), Hypergraph([], 3), Hypergraph([(0, 1), (0, 2)], 3))
QG_Evansb = QmDAG(DirectedStructure([(0, 1), (0, 2)], 3), Hypergraph([(0, 1)], 3), Hypergraph([(0, 2)], 3))
# Evans is treated as a QC gap only if both latents are quantum.
Evans_ids = set(special_QmDAG.unique_unlabelled_id for special_QmDAG in {QG_Evans})

known_QC_Gaps_QmDAGs_small_ids = set().union(IV_ids, Triangle_ids, Evans_ids)

# ---------------------------------------------------------------------------
# Known QC gaps with 4 visible nodes (Bell scenario variants)
# ---------------------------------------------------------------------------
QG_Bell1 = QmDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([], 4), Hypergraph([(2, 3)], 4))

QG_Bell2 = QmDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([(0, 2)], 4), Hypergraph([(2, 3)], 4))
QG_Bell2b = QmDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([], 4), Hypergraph([(0, 2), (2, 3)], 4))

QG_Bell3 = QmDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([(1, 3)], 4), Hypergraph([(2, 3)], 4))
QG_Bell3b = QmDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([], 4), Hypergraph([(2, 3), (1, 3)], 4))

QG_Bell4 = QmDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([(0, 2), (1, 3)], 4), Hypergraph([(2, 3)], 4))
QG_Bell4b = QmDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([(0, 2)], 4), Hypergraph([(1, 3), (2, 3)], 4))
QG_Bell4c = QmDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([(1, 3)], 4), Hypergraph([(0, 2), (2, 3)], 4))
QG_Bell4d = QmDAG(DirectedStructure([(0, 2), (1, 3)], 4), Hypergraph([], 4), Hypergraph([(0, 2), (1, 3), (2, 3)], 4))

QG_Bell6 = QmDAG(DirectedStructure([], 4), Hypergraph([(1, 3), (0, 2)], 4), Hypergraph([(2, 3)], 4))
QG_Bell6b = QmDAG(DirectedStructure([], 4), Hypergraph([(1, 3)], 4), Hypergraph([(0, 2), (2, 3)], 4))
QG_Bell6c = QmDAG(DirectedStructure([], 4), Hypergraph([(0, 2)], 4), Hypergraph([(1, 3), (2, 3)], 4))
QG_Bell6d = QmDAG(DirectedStructure([], 4), Hypergraph([], 4), Hypergraph([(0, 2), (1, 3), (2, 3)], 4))

QG_Bell5 = QmDAG(DirectedStructure([(1, 3)], 4), Hypergraph([(0, 2)], 4), Hypergraph([(2, 3)], 4))
QG_Bell5b = QmDAG(DirectedStructure([(1, 3)], 4), Hypergraph([], 4), Hypergraph([(2, 3), (0, 2)], 4))

QG_Bell7 = QmDAG(DirectedStructure([(1, 3)], 4), Hypergraph([(0, 2), (1, 3)], 4), Hypergraph([(2, 3)], 4))
QG_Bell7b = QmDAG(DirectedStructure([(1, 3)], 4), Hypergraph([(1, 3)], 4), Hypergraph([(0, 2), (2, 3)], 4))
QG_Bell7c = QmDAG(DirectedStructure([(1, 3)], 4), Hypergraph([(0, 2)], 4), Hypergraph([(1, 3), (2, 3)], 4))
QG_Bell7d = QmDAG(DirectedStructure([(1, 3)], 4), Hypergraph([], 4), Hypergraph([(0, 2), (1, 3), (2, 3)], 4))

QG_Bell8 = QmDAG(DirectedStructure([(0, 2)], 4), Hypergraph([(0, 2), (1, 3)], 4), Hypergraph([(2, 3)], 4))
QG_Bell8b = QmDAG(DirectedStructure([(0, 2)], 4), Hypergraph([(1, 3)], 4), Hypergraph([(0, 2), (2, 3)], 4))
QG_Bell8c = QmDAG(DirectedStructure([(0, 2)], 4), Hypergraph([(0, 2)], 4), Hypergraph([(1, 3), (2, 3)], 4))
QG_Bell8d = QmDAG(DirectedStructure([(0, 2)], 4), Hypergraph([], 4), Hypergraph([(0, 2), (1, 3), (2, 3)], 4))

QG_Bell9 = QmDAG(DirectedStructure([(0, 2)], 4), Hypergraph([(1, 3)], 4), Hypergraph([(2, 3)], 4))
QG_Bell9b = QmDAG(DirectedStructure([(0, 2)], 4), Hypergraph([], 4), Hypergraph([(1, 3), (2, 3)], 4))

known_QC_Gaps_QmDAGs_big = {QG_Bell1, QG_Bell2, QG_Bell3, QG_Bell4, QG_Bell5, QG_Bell6, QG_Bell7, QG_Bell8, QG_Bell9,
                            QG_Bell2b, QG_Bell3b, QG_Bell4b, QG_Bell4c, QG_Bell4d, QG_Bell5b, QG_Bell6b, QG_Bell6c,
                            QG_Bell6d, QG_Bell7b, QG_Bell7c, QG_Bell7d, QG_Bell8b, QG_Bell8c, QG_Bell8d, QG_Bell9b}
known_QC_Gaps_QmDAGs_big_ids = set(special_QmDAG.unique_unlabelled_id for special_QmDAG in known_QC_Gaps_QmDAGs_big)

known_QC_Gaps_QmDAGs_ids = known_QC_Gaps_QmDAGs_small_ids.union(known_QC_Gaps_QmDAGs_big_ids)


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


def run_pipeline(QmDAGs4_representatives=None, max_visible=5):
    """Hand-ordered application of the piggyback tricks. Returns a dict of counts and the remaining QmDAGs."""
    if QmDAGs4_representatives is None:
        QmDAGs4_representatives = four_node_representatives()
    counts = dict()

    QC_remaining_representatives = set(
        qmDAG for qmDAG in QmDAGs4_representatives if qmDAG.unique_unlabelled_id not in known_QC_Gaps_QmDAGs_big_ids)
    things_we_got_rid_of = set(QmDAGs4_representatives).difference(QC_remaining_representatives)
    counts['to_analyze'] = len(QC_remaining_representatives)
    counts['already_known'] = len(things_we_got_rid_of)
    print("Total number of qmDAGs to analyze: ", counts['to_analyze'])
    print("Number of representatives that are known QC Gaps: ", counts['already_known'])

    def reduces_to_knownQCGap_by_PD_trick(qmDAG):
        return not known_QC_Gaps_QmDAGs_small_ids.isdisjoint(qmDAG.unique_unlabelled_ids_obtainable_by_PD_trick)

    QC_gap_by_PD_trick = list(filter(reduces_to_knownQCGap_by_PD_trick, QC_remaining_representatives))
    counts['PD'] = len(QC_gap_by_PD_trick)
    print("# of ADDITIONAL QC gaps seen via PD trick: ", counts['PD'])
    QC_remaining_representatives.difference_update(QC_gap_by_PD_trick)

    def reduces_to_knownQCGap_by_interruption(qmDAG):
        return not known_QC_Gaps_QmDAGs_small_ids.isdisjoint(qmDAG.unique_unlabelled_ids_obtainable_by_interruption)

    QC_gap_by_interruption = list(filter(reduces_to_knownQCGap_by_interruption, QC_remaining_representatives))
    counts['interruption'] = len(QC_gap_by_interruption)
    print("# of ADDITIONAL QC gaps seen via interruption: ", counts['interruption'])
    QC_remaining_representatives.difference_update(QC_gap_by_interruption)

    def reduces_to_knownQCGap_by_naive_marginalization(qmDAG):
        return not known_QC_Gaps_QmDAGs_small_ids.isdisjoint(
            qmDAG.unique_unlabelled_ids_obtainable_by_naive_marginalization(districts_check=False))

    QC_gap_by_naive_marginalization = list(filter(reduces_to_knownQCGap_by_naive_marginalization, QC_remaining_representatives))
    counts['naive_marginalization'] = len(QC_gap_by_naive_marginalization)
    print("# of ADDITIONAL QC gaps seen via naive marginalization: ", counts['naive_marginalization'])
    QC_remaining_representatives.difference_update(QC_gap_by_naive_marginalization)

    def reduces_to_knownQCGap_by_marginalization(qmDAG):
        return not known_QC_Gaps_QmDAGs_small_ids.isdisjoint(
            qmDAG.unique_unlabelled_ids_obtainable_by_marginalization(districts_check=False))

    QC_gap_by_marginalization = list(filter(reduces_to_knownQCGap_by_marginalization, QC_remaining_representatives))
    counts['teleportation_marginalization'] = len(QC_gap_by_marginalization)
    print("# of ADDITIONAL QC gaps seen via teleporation marginalization: ", counts['teleportation_marginalization'])
    QC_remaining_representatives.difference_update(QC_gap_by_marginalization)

    def reduces_to_knownQCGap_by_conditioning(qmDAG):
        return not known_QC_Gaps_QmDAGs_small_ids.isdisjoint(qmDAG.unique_unlabelled_ids_obtainable_by_conditioning)

    QC_gap_by_conditioning = list(filter(reduces_to_knownQCGap_by_conditioning, QC_remaining_representatives))
    counts['conditioning'] = len(QC_gap_by_conditioning)
    print("# of ADDITIONAL QC gaps seen via conditioning: ", counts['conditioning'])
    QC_remaining_representatives.difference_update(QC_gap_by_conditioning)

    updated_known_QC_Gaps_QmDAGs = set(QmDAGs4_representatives).difference(QC_remaining_representatives)
    updated_known_QC_Gaps_QmDAGs_ids = set(known_QmDAG.unique_unlabelled_id for known_QmDAG in updated_known_QC_Gaps_QmDAGs)
    print("Size of new database: ", len(updated_known_QC_Gaps_QmDAGs_ids))
    updated_known_QC_Gaps_QmDAGs_ids.update(known_QC_Gaps_QmDAGs_ids)
    print("Size of known database: ", len(updated_known_QC_Gaps_QmDAGs_ids))
    print("Knows about Bell etc: ", updated_known_QC_Gaps_QmDAGs_ids.issuperset(known_QC_Gaps_QmDAGs_ids))

    def reduces_to_knownQCGap_by_Fritz(qmDAG):
        # Closure of all piggybacks (Fritz included) composed in any order, with at most one extra visible node.
        return not updated_known_QC_Gaps_QmDAGs_ids.isdisjoint(
            qmDAG.unique_unlabelled_ids_obtainable_by_Fritz_for_QC(max_visible=max_visible))

    counts['before_Fritz'] = len(QC_gap_by_PD_trick + QC_gap_by_interruption + QC_gap_by_naive_marginalization
                                 + QC_gap_by_marginalization + QC_gap_by_conditioning)
    print("# of QC Gaps discovered so far: ", counts['before_Fritz'])
    print("# of QC Gaps still to be assessed: ", len(QC_remaining_representatives))

    QC_gap_by_Fritz = list(filter(reduces_to_knownQCGap_by_Fritz, QC_remaining_representatives))
    QC_remaining_representatives.difference_update(QC_gap_by_Fritz)
    counts['Fritz'] = len(QC_gap_by_Fritz)
    print("# of QC Gaps discovered via Fritz composed with the other piggybacks: ", counts['Fritz'])
    for found in sorted(QC_gap_by_Fritz):
        print(found)

    counts['remaining'] = len(QC_remaining_representatives)
    print("# of QC Gaps still to be assessed: ", counts['remaining'])
    print("Note that here, we ARE considering Evans as if it had a QC Gap (only if both latents go quantum).")
    counts['remaining_QmDAGs'] = QC_remaining_representatives
    return counts


if __name__ == '__main__':
    run_pipeline()
