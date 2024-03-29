from __future__ import absolute_import
from metagraph_advanced import Observable_unlabelled_mDAGs
from quantum_mDAG import as_classical_QmDAG
import gc
from supports import explore_candidates
# from the_7_hard_graphs import the_7_hard_graphs
# the_7_hard_ids = set(m.unique_unlabelled_id for m in the_7_hard_graphs)

believed_interesting_ids = set()

n=3
print(f"NOW CONSIDERING {n} NODES")
metagraph = Observable_unlabelled_mDAGs(n, fully_foundational=False, verbose=False)
unlabelled_mDAGs = set(metagraph.all_unlabelled_mDAGs)
print(f"Number of {n}-vis node unlabelled mDAGs: {len(unlabelled_mDAGs)}")
potentially_interesting_classes = unlabelled_mDAGs.difference(metagraph.boring_by_virtue_of_HLP)
potentially_interesting_after_HLP = potentially_interesting_classes.intersection(unlabelled_mDAGs)
believed_interesting_ids.update(
    set(as_classical_QmDAG(m).unique_unlabelled_id for m in potentially_interesting_classes))
print(f"Number of {n}-vis node unlabelled mDAGs not resolved by HLP: {len(potentially_interesting_after_HLP)}")
print("Conjecture sanity check: ", metagraph.boring_by_virtue_of_Evans.issubset(metagraph.boring_by_virtue_of_HLP))
print("Does Evans approach coincide with HLP? ", len(metagraph.boring_by_virtue_of_Evans) == len(metagraph.boring_by_virtue_of_HLP))

unresolved = potentially_interesting_after_HLP.copy()
unresolved = set([m for m in unresolved if not m.interesting_via_generalized_maximality])
print(f"Number unresolved after generalized maximality test:  {len(unresolved)}")
print(f"Sanity check: {len(set([m for m in unlabelled_mDAGs if m.interesting_via_generalized_maximality]))==len(potentially_interesting_after_HLP)}")
# unresolved = set([m for m in unresolved if not m.interesting_via_non_maximal])
# print(f"Number unresolved after normal maximality test:  {len(unresolved)}")
# unresolved = set([m for m in unresolved if not m.interesting_via_no_dsep_but_no_common_ancestor])
# print(f"Number unresolved after no-common-ancestor test:  {len(unresolved)}")
#
# uresolved_by_HLP_or_subgraph = set([m for m in potentially_interesting_after_HLP if
#     believed_interesting_ids.isdisjoint(as_classical_QmDAG(m).unique_unlabelled_ids_obtainable_by_PD_trick)
#                                     ])
# print(f"Number of {n}-vis node unlabelled mDAGs unresolved by HLP or subgraph: {len(uresolved_by_HLP_or_subgraph)}")
# print("Hereafter all subgraph-interesting mDAGs have been eliminated!")
# unresolved_by_HLP_or_nonmaximality = set([m for m in uresolved_by_HLP_or_subgraph if not m.interesting_via_non_maximal])
# unresolved_by_HLP_or_two_event_support = set([m for m in unresolved_by_HLP_or_nonmaximality if not m.interesting_via_no_dsep_but_no_common_ancestor])
# print(f"Number of {n}-vis node unlabelled mDAGs unresolved by HLP or 2-event-support: {len(unresolved_by_HLP_or_two_event_support)}")
# print("Hereafter all interesting via 2-event-support mDAGs have been eliminated!")
# unresolved_by_HLP_or_nonmaximality_or_dsep = set([m for m in unresolved_by_HLP_or_two_event_support if m.all_CI_unlabelled in metagraph.latent_free_CI_patterns])
# print(f"Number of {n}-vis node unlabelled mDAGs unresolved by HLP or dsep: {len(unresolved_by_HLP_or_nonmaximality_or_dsep)}")
# unresolved_by_HLP_or_esep = set([m for m in unresolved_by_HLP_or_nonmaximality_or_dsep if m.all_esep_unlabelled in metagraph.latent_free_esep_patterns])
# print(f"Number of {n}-vis node unlabelled mDAGs unresolved by HLP or esep: {len(unresolved_by_HLP_or_esep)}")
# still_remaining_after_supports_on_two_events = set([m for m in unresolved_by_HLP_or_esep if m.no_infeasible_binary_supports_beyond_dsep_up_to(2)])
# print(f"Remaining after testing with supports up to 2: {len(still_remaining_after_supports_on_two_events)}")
# still_remaining_after_supports_on_three_events = set([m for m in still_remaining_after_supports_on_two_events if m.no_infeasible_binary_supports_beyond_dsep_up_to(3)])
# print(f"Remaining after testing with supports up to 3: {len(still_remaining_after_supports_on_three_events)}")
# still_remaining_after_supports_on_four_events = set([m for m in still_remaining_after_supports_on_three_events if m.no_infeasible_binary_supports_beyond_dsep_up_to(4)])
# print(f"Remaining after testing with supports up to 4: {len(still_remaining_after_supports_on_four_events)}")

print("\n\n")
del metagraph
gc.collect()

n=4
print(f"NOW CONSIDERING {n} NODES")
metagraph = Observable_unlabelled_mDAGs(n, fully_foundational=False, verbose=True)
unlabelled_mDAGs = set(metagraph.all_unlabelled_mDAGs)
print(f"Number of {n}-vis node unlabelled mDAGs: {len(unlabelled_mDAGs)}")
potentially_interesting_classes = unlabelled_mDAGs.difference(metagraph.boring_by_virtue_of_HLP)
potentially_interesting_after_HLP = potentially_interesting_classes.intersection(unlabelled_mDAGs)
print(f"Number of {n}-vis node unlabelled mDAGs not resolved by HLP: {len(potentially_interesting_after_HLP)}")
print("Conjecture sanity check: ", metagraph.boring_by_virtue_of_Evans.issubset(metagraph.boring_by_virtue_of_HLP))
print("Does Evans approach coincide with HLP? ", len(metagraph.boring_by_virtue_of_Evans) == len(metagraph.boring_by_virtue_of_HLP))

unresolved = potentially_interesting_after_HLP.copy()
unresolved = set([m for m in unresolved if not m.interesting_via_non_maximal])
print(f"Number unresolved after maximality test:  {len(unresolved)}")
unresolved = set([m for m in unresolved if not m.interesting_via_generalized_maximality])
print(f"Number unresolved after generalized maximality test:  {len(unresolved)}")

unresolved_by_subgraph = set([m for m in unresolved if
    believed_interesting_ids.isdisjoint(as_classical_QmDAG(m).unique_unlabelled_ids_obtainable_by_PD_trick)
                                    ])
believed_interesting_ids.update(
    set(as_classical_QmDAG(m).unique_unlabelled_id for m in unresolved))
print(f"Number of {n}-vis node unlabelled mDAGs unresolved by subgraph: {len(unresolved_by_subgraph)}")
unresolved = set([m for m in unresolved if m.all_CI_unlabelled in metagraph.latent_free_CI_patterns])
print(f"Number of unresolved by dsep: {len(unresolved)}")
unresolved = set([m for m in unresolved if m.all_esep_unlabelled in metagraph.latent_free_esep_patterns])
print(f"Number of unresolved by esep: {len(unresolved)}")
# still_remaining_after_supports_on_two_events = set([m for m in unresolved_by_HLP_or_esep if m.no_infeasible_binary_supports_beyond_dsep_up_to(2)])
# print(f"Remaining after testing with supports up to 2: {len(still_remaining_after_supports_on_two_events)}")
# still_remaining_after_supports_on_three_events = set([m for m in still_remaining_after_supports_on_two_events if m.no_infeasible_binary_supports_beyond_dsep_up_to(3)])
# print(f"Remaining after testing with supports up to 3: {len(still_remaining_after_supports_on_three_events)}")
# still_remaining_after_supports_on_four_events = set([m for m in still_remaining_after_supports_on_three_events if m.no_infeasible_binary_supports_beyond_dsep_up_to(4)])
# print(f"Remaining after testing with supports up to 4: {len(still_remaining_after_supports_on_four_events)}")
print("RESTARTING COUNT!")
print(f"Number of {n}-vis node unlabelled mDAGs not resolved by HLP: {len(potentially_interesting_after_HLP)}")
unresolved_by_HLP_or_nonmaximality = set([m for m in potentially_interesting_after_HLP if not m.interesting_via_non_maximal])
print(f"Number of {n}-vis node unlabelled mDAGs unresolved by Nonmaximality: {len(unresolved_by_HLP_or_nonmaximality)}")
unresolved_by_HLP_or_nonmaximality_or_dsep = set([m for m in unresolved_by_HLP_or_nonmaximality if m.all_CI_unlabelled in metagraph.latent_free_CI_patterns])
print(f"Number of {n}-vis node unlabelled mDAGs unresolved by dsep: {len(unresolved_by_HLP_or_nonmaximality_or_dsep)}")
unresolved_by_HLP_or_esep = set([m for m in unresolved_by_HLP_or_nonmaximality_or_dsep if m.all_esep_unlabelled in metagraph.latent_free_esep_patterns])
print(f"Number of {n}-vis node unlabelled mDAGs unresolved by esep: {len(unresolved_by_HLP_or_esep)}")
still_remaining_after_supports_on_two_events = set([m for m in unresolved_by_HLP_or_esep if m.no_infeasible_binary_supports_beyond_dsep_up_to(2)])
print(f"Remaining after testing with supports up to 2: {len(still_remaining_after_supports_on_two_events)}")


print("\n\n")
del metagraph
gc.collect()

n=5
print(f"NOW CONSIDERING {n} NODES")
metagraph = Observable_unlabelled_mDAGs(n, fully_foundational=False, verbose=True)
unlabelled_mDAGs = set(metagraph.all_unlabelled_mDAGs)
print(f"Number of {n}-vis node unlabelled mDAGs: {len(unlabelled_mDAGs)}")
potentially_interesting_classes = unlabelled_mDAGs.difference(metagraph.boring_by_virtue_of_HLP)
potentially_interesting_after_HLP = potentially_interesting_classes.intersection(unlabelled_mDAGs)
print(f"Number of {n}-vis node unlabelled mDAGs not resolved by HLP: {len(potentially_interesting_after_HLP)}")
# print("Conjecture sanity check: ", metagraph.boring_by_virtue_of_Evans.issubset(metagraph.boring_by_virtue_of_HLP))
# print("Does Evans approach coincide with HLP? ", len(metagraph.boring_by_virtue_of_Evans) == len(metagraph.boring_by_virtue_of_HLP))
unresolved = potentially_interesting_after_HLP.copy()
unresolved = set([m for m in explore_candidates(unresolved,
                                                verbose=True,
                                                message="Applying maximality filter")
                  if not m.interesting_via_non_maximal])
print(f"Number unresolved after maximality test:  {len(unresolved)}")
unresolved = set([m for m in explore_candidates(unresolved,
                                                verbose=True,
                                                message="Applying generalized maximality filter")
                  if not m.interesting_via_generalized_maximality])
print(f"Number unresolved after generalized maximality test:  {len(unresolved)}")
unresolved = set([m for m in explore_candidates(unresolved,
                                                            verbose=True,
                                                            message="Applying subgraph filter")
                              if believed_interesting_ids.isdisjoint(
        as_classical_QmDAG(m).unique_unlabelled_ids_obtainable_by_PD_trick)
                              ])
print(f"Number of {n}-vis node unlabelled mDAGs unresolved by subgraph: {len(unresolved)}")
unresolved = set([m for m in explore_candidates(unresolved,
                                                verbose=True,
                                                message="Applying d-sep patterns filter")
                  if m.all_CI_unlabelled in metagraph.latent_free_CI_patterns])
print(f"Number of unresolved by dsep: {len(unresolved)}")
unresolved = set([m for m in explore_candidates(unresolved,
                                                verbose=True,
                                                message="Applying e-sep patterns filter")
                  if m.all_esep_unlabelled in metagraph.latent_free_esep_patterns])
print(f"Number of unresolved by esep: {len(unresolved)}")
unresolved = set([m for m in explore_candidates(unresolved,
                                               verbose=True,
                                               message="Applying 3-event-supports filter")
                 if m.no_infeasible_binary_supports_beyond_dsep(3)])
print(f"Remaining after testing with supports up to 3: {len(unresolved)}")
unresolved = set([m for m in explore_candidates(unresolved,
                                               verbose=True,
                                               message="Applying 4-event-supports filter")
                 if m.no_infeasible_binary_supports_beyond_dsep(4)])
print(f"Remaining after testing with supports up to 4: {len(unresolved)}")
print("RESTARTING COUNT!")
print(f"Number of {n}-vis node unlabelled mDAGs not resolved by HLP: {len(potentially_interesting_after_HLP)}")
unresolved_by_HLP_or_nonmaximality = set([m for m in potentially_interesting_after_HLP if not m.interesting_via_non_maximal])
print(f"Number of {n}-vis node unlabelled mDAGs unresolved by Nonmaximality: {len(unresolved_by_HLP_or_nonmaximality)}")
unresolved_by_HLP_or_nonmaximality_or_dsep = set([m for m in unresolved_by_HLP_or_nonmaximality if m.all_CI_unlabelled in metagraph.latent_free_CI_patterns])
print(f"Number of {n}-vis node unlabelled mDAGs unresolved by dsep: {len(unresolved_by_HLP_or_nonmaximality_or_dsep)}")
unresolved_by_HLP_or_esep = set([m for m in unresolved_by_HLP_or_nonmaximality_or_dsep if m.all_esep_unlabelled in metagraph.latent_free_esep_patterns])
print(f"Number of {n}-vis node unlabelled mDAGs unresolved by esep: {len(unresolved_by_HLP_or_esep)}")
# still_remaining_after_supports_on_two_events = set([m for m in unresolved_by_HLP_or_esep if m.no_infeasible_binary_supports_beyond_dsep_up_to(2)])
# print(f"Remaining after testing with supports up to 2: {len(still_remaining_after_supports_on_two_events)}")
