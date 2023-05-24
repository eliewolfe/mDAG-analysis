from __future__ import absolute_import
from metagraph_advanced import Observable_unlabelled_mDAGs
from quantum_mDAG import as_classical_QmDAG
# from the_7_hard_graphs import the_7_hard_graphs
# the_7_hard_ids = set(m.unique_unlabelled_id for m in the_7_hard_graphs)

potentially_interesting_after_HLP = set()

n=3
print(f"NOW CONSIDERING {n} NODES")
metagraph = Observable_unlabelled_mDAGs(n, fully_foundational=False, verbose=False)
unlabelled_mDAGs = set(metagraph.all_unlabelled_mDAGs)
print(f"Number of {n}-vis node unlabelled mDAGs: {len(unlabelled_mDAGs)}")
potentially_interesting_classes = unlabelled_mDAGs.difference(metagraph.boring_by_virtue_of_HLP)
potentially_interesting_after_HLP = potentially_interesting_classes.intersection(unlabelled_mDAGs)
print(f"Number of {n}-vis node unlabelled mDAGs not resolved by HLP: {len(potentially_interesting_after_HLP)}")
print("Conjecture sanity check: ", metagraph.boring_by_virtue_of_Evans.issubset(metagraph.boring_by_virtue_of_HLP))
print("Does Evans approach coincide with HLP? ", len(metagraph.boring_by_virtue_of_Evans) == len(metagraph.boring_by_virtue_of_HLP))
unresolved_by_CI_or_HLP = set([m for m in potentially_interesting_after_HLP if m.all_CI_unlabelled in metagraph.latent_free_CI_patterns])
print(f"Number of {n}-vis node unlabelled mDAGs unresolved by HLP or CI: {len(unresolved_by_CI_or_HLP)}")
unresolved_by_Shaashank_or_CI_or_HLP = set([m for m in unresolved_by_CI_or_HLP if not m.interesting_via_e_sep_theorem])
print(f"Number of {n}-vis node unlabelled mDAGs unresolved by HLP or CI or Shaashank's theorem: {len(unresolved_by_Shaashank_or_CI_or_HLP)}")

# unresolved_by_esep = set([m for m in unlabelled_mDAGs if m.all_esep_unlabelled in metagraph.latent_free_esep_patterns])
# print(f"Number of {n}-vis node unlabelled mDAGs not resolved by e-sep: {len(unresolved_by_esep)}")
# unresolved_by_esep_or_HLP = set(potentially_interesting_after_HLP).intersection(unresolved_by_esep)
# print(f"Number of {n}-vis node unlabelled mDAGs unresolved by e-sep or HLP: {len(unresolved_by_esep_or_HLP)}")

still_remaining_after_supports_on_two_events = set([m for m in unresolved_by_Shaashank_or_CI_or_HLP if m.no_infeasible_binary_supports_beyond_dsep_up_to(2)])
print(f"Remaining after testing with supports up to 2: {len(still_remaining_after_supports_on_two_events)}")
# still_remaining_after_supports_on_three_events = set([m for m in still_remaining_after_supports_on_two_events if m.no_infeasible_binary_supports_beyond_dsep_up_to(3)])
# print(f"Remaining after testing with supports up to 3: {len(still_remaining_after_supports_on_three_events)}")
# still_remaining_after_supports_on_four_events = set([m for m in still_remaining_after_supports_on_three_events if m.no_infeasible_binary_supports_beyond_dsep_up_to(4)])
# print(f"Remaining after testing with supports up to 4: {len(still_remaining_after_supports_on_four_events)}")
believed_interesting_ids = set(as_classical_QmDAG(m).unique_unlabelled_id for m in potentially_interesting_classes)

print("\n\n")

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
unresolved_by_CI_or_HLP = set(m for m in potentially_interesting_after_HLP if m.all_CI_unlabelled in metagraph.latent_free_CI_patterns)
print(f"Number of {n}-vis node unlabelled mDAGs unresolved by HLP or CI: {len(unresolved_by_CI_or_HLP)}")
unresolved_by_Shaashank_or_CI_or_HLP = set([m for m in unresolved_by_CI_or_HLP if not m.interesting_via_e_sep_theorem])
print(f"Number of {n}-vis node unlabelled mDAGs unresolved by HLP or CI or Shaashank's theorem: {len(unresolved_by_Shaashank_or_CI_or_HLP)}")
unresolved_by_reduction_or_Shaashank_or_CI_or_HLP = set(m for m in unresolved_by_Shaashank_or_CI_or_HLP if believed_interesting_ids.isdisjoint(as_classical_QmDAG(m).unique_unlabelled_ids_obtainable_by_PD_trick))
print(f"Number of {n}-vis node unlabelled mDAGs unresolved by HLP or CI or Shaashank's theorem or subgraph: {len(unresolved_by_reduction_or_Shaashank_or_CI_or_HLP)}")

# unresolved_by_esep = set([m for m in unlabelled_mDAGs if m.all_esep_unlabelled in metagraph.latent_free_esep_patterns])
# print(f"Number of {n}-vis node unlabelled mDAGs not resolved by e-sep: {len(unresolved_by_esep)}")
# unresolved_by_esep_or_HLP = set(potentially_interesting_after_HLP).intersection(unresolved_by_esep)
# print(f"Number of {n}-vis node unlabelled mDAGs unresolved by e-sep or HLP: {len(unresolved_by_esep_or_HLP)}")

still_remaining_after_supports_on_two_events = set([m for m in unresolved_by_reduction_or_Shaashank_or_CI_or_HLP if m.no_infeasible_binary_supports_beyond_dsep_up_to(2)])
print(f"Remaining after testing with supports up to 2: {len(still_remaining_after_supports_on_two_events)}")
still_remaining_after_supports_on_three_events = set([m for m in still_remaining_after_supports_on_two_events if m.no_infeasible_binary_supports_beyond_dsep_up_to(3)])
print(f"Remaining after testing with supports up to 3: {len(still_remaining_after_supports_on_three_events)}")
still_remaining_after_supports_on_four_events = set([m for m in still_remaining_after_supports_on_three_events if m.no_infeasible_binary_supports_beyond_dsep_up_to(4)])
print(f"Remaining after testing with supports up to 4: {len(still_remaining_after_supports_on_four_events)}")
believed_interesting_ids = set(as_classical_QmDAG(m).unique_unlabelled_id for m in potentially_interesting_classes)

print("\n\n")

n=5
print(f"NOW CONSIDERING {n} NODES")
metagraph = Observable_unlabelled_mDAGs(n, fully_foundational=False, verbose=True)
unlabelled_mDAGs = set(metagraph.all_unlabelled_mDAGs)
print(f"Number of {n}-vis node unlabelled mDAGs: {len(unlabelled_mDAGs)}")
potentially_interesting_classes = unlabelled_mDAGs.difference(metagraph.boring_by_virtue_of_HLP)
potentially_interesting_after_HLP = potentially_interesting_classes.intersection(unlabelled_mDAGs)
print(f"Number of {n}-vis node unlabelled mDAGs not resolved by HLP: {len(potentially_interesting_after_HLP)}")
print("Conjecture sanity check: ", metagraph.boring_by_virtue_of_Evans.issubset(metagraph.boring_by_virtue_of_HLP))
print("Does Evans approach coincide with HLP? ", len(metagraph.boring_by_virtue_of_Evans) == len(metagraph.boring_by_virtue_of_HLP))
unresolved_by_CI_or_HLP = set([m for m in potentially_interesting_after_HLP if m.all_CI_unlabelled in metagraph.latent_free_CI_patterns])
print(f"Number of {n}-vis node unlabelled mDAGs unresolved by HLP or CI: {len(unresolved_by_CI_or_HLP)}")
unresolved_by_Shaashank_or_CI_or_HLP = set([m for m in unresolved_by_CI_or_HLP if not m.interesting_via_e_sep_theorem])
print(f"Number of {n}-vis node unlabelled mDAGs unresolved by HLP or CI or Shaashank's theorem: {len(unresolved_by_Shaashank_or_CI_or_HLP)}")
unresolved_by_reduction_or_Shaashank_or_CI_or_HLP = set(m for m in unresolved_by_Shaashank_or_CI_or_HLP if believed_interesting_ids.isdisjoint(as_classical_QmDAG(m).unique_unlabelled_ids_obtainable_by_PD_trick))
print(f"Number of {n}-vis node unlabelled mDAGs unresolved by HLP or CI or Shaashank's theorem or subgraph: {len(unresolved_by_reduction_or_Shaashank_or_CI_or_HLP)}")

# unresolved_by_esep = set([m for m in unlabelled_mDAGs if m.all_esep_unlabelled in metagraph.latent_free_esep_patterns])
# print(f"Number of {n}-vis node unlabelled mDAGs not resolved by e-sep: {len(unresolved_by_esep)}")
# unresolved_by_esep_or_HLP = set(potentially_interesting_after_HLP).intersection(unresolved_by_esep)
# print(f"Number of {n}-vis node unlabelled mDAGs unresolved by e-sep or HLP: {len(unresolved_by_esep_or_HLP)}")

still_remaining_after_supports_on_two_events = set([m for m in unresolved_by_reduction_or_Shaashank_or_CI_or_HLP if m.no_infeasible_binary_supports_beyond_dsep_up_to(2)])
print(f"Remaining after testing with supports up to 2: {len(still_remaining_after_supports_on_two_events)}")
still_remaining_after_supports_on_three_events = set([m for m in still_remaining_after_supports_on_two_events if m.no_infeasible_binary_supports_beyond_dsep_up_to(3)])
print(f"Remaining after testing with supports up to 3: {len(still_remaining_after_supports_on_three_events)}")
still_remaining_after_supports_on_four_events = set([m for m in still_remaining_after_supports_on_three_events if m.no_infeasible_binary_supports_beyond_dsep_up_to(4)])
print(f"Remaining after testing with supports up to 4: {len(still_remaining_after_supports_on_four_events)}")