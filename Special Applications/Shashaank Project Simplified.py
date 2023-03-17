from __future__ import absolute_import
from metagraph_advanced import Observable_unlabelled_mDAGs

n=3
print(f"NOW CONSIDERING {n} NODES \n")
metagraph = Observable_unlabelled_mDAGs(n, fully_foundational=False, verbose=False)
unlabelled_mDAGs = list(metagraph.all_unlabelled_mDAGs_faster)
print(f"Number of {n}-vis node unlabelled mDAGs: {len(unlabelled_mDAGs)}")
fundamental_unlabelled = [m for m in unlabelled_mDAGs if m.fundamental_graphQ]
print(f"Number of {n}-vis node unfactorizable mDAGs: {len(fundamental_unlabelled)}")
print(f"**ALL FURTHER ANALYSIS PERFORMED ON UNFACTORIZABLE mDAGs**")
potentially_interesting_classes = metagraph.NOT_latent_free_eqclasses
potentially_interesting_after_HLP = set().union(*potentially_interesting_classes)
potentially_interesting_after_HLP = [m for m in potentially_interesting_after_HLP if m.fundamental_graphQ]
print(f"Number of {n}-vis node unlabelled mDAGs not resolved by HLP: {len(potentially_interesting_after_HLP)}")
unresolved_by_esep = [m for m in fundamental_unlabelled if m.all_esep_unlabelled in metagraph.latent_free_esep_patterns]
print(f"Number of {n}-vis node unlabelled mDAGs not resolved by e-sep: {len(unresolved_by_esep)}")
unresolved_by_esep_or_HLP = [m for m in potentially_interesting_after_HLP if m.all_esep_unlabelled in metagraph.latent_free_esep_patterns]
print(f"Number of {n}-vis node unlabelled mDAGs resolved by e-sep or HLP: {len(unresolved_by_esep_or_HLP)}")
print("As we proceed now to supports, we wonder if there are graphs with multiple CI relations that could be problematic.")
print("They are, if any: ", [m for m in unresolved_by_esep_or_HLP if len(m.all_CI)>1])
still_remaining_after_supports_on_two_events = [m for m in unresolved_by_esep_or_HLP if m.no_infeasible_binary_supports_beyond_dsep_up_to(2)]
print(f"Remaining after testing with supports up to 2: {len(still_remaining_after_supports_on_two_events)}")
still_remaining_after_supports_on_four_events = [m for m in still_remaining_after_supports_on_two_events if m.no_infeasible_binary_supports_beyond_dsep_up_to(4)]
print(f"Remaining after testing with supports up to 4: {len(still_remaining_after_supports_on_four_events)}")


print("\n\n")

n=4
print(f"NOW CONSIDERING {n} NODES \n")
metagraph = Observable_unlabelled_mDAGs(n, fully_foundational=False, verbose=False)
unlabelled_mDAGs = list(metagraph.all_unlabelled_mDAGs_faster)
print(f"Number of {n}-vis node unlabelled mDAGs: {len(unlabelled_mDAGs)}")
fundamental_unlabelled = [m for m in unlabelled_mDAGs if m.fundamental_graphQ]
print(f"Number of {n}-vis node unfactorizable mDAGs: {len(fundamental_unlabelled)}")
print(f"**ALL FURTHER ANALYSIS PERFORMED ON UNFACTORIZABLE mDAGs**")
potentially_interesting_classes = metagraph.NOT_latent_free_eqclasses
potentially_interesting_after_HLP = set().union(*potentially_interesting_classes)
potentially_interesting_after_HLP = [m for m in potentially_interesting_after_HLP if m.fundamental_graphQ]
print(f"Number of {n}-vis node unlabelled mDAGs not resolved by HLP: {len(potentially_interesting_after_HLP)}")
unresolved_by_esep = [m for m in fundamental_unlabelled if m.all_esep_unlabelled in metagraph.latent_free_esep_patterns]
print(f"Number of {n}-vis node unlabelled mDAGs not resolved by e-sep: {len(unresolved_by_esep)}")
unresolved_by_esep_or_HLP = [m for m in potentially_interesting_after_HLP if m.all_esep_unlabelled in metagraph.latent_free_esep_patterns]
print(f"Number of {n}-vis node unlabelled mDAGs resolved by e-sep or HLP: {len(unresolved_by_esep_or_HLP)}")
print("As we proceed now to supports, we wonder if there are graphs with multiple CI relations that could be problematic.")
print("They are, if any: ", [m for m in unresolved_by_esep_or_HLP if len(m.all_CI)>1])
still_remaining_after_supports_on_two_events = [m for m in unresolved_by_esep_or_HLP if m.no_infeasible_binary_supports_beyond_dsep_up_to(2)]
print(f"Remaining after testing with supports up to 2: {len(still_remaining_after_supports_on_two_events)}")
still_remaining_after_supports_on_four_events = [m for m in still_remaining_after_supports_on_two_events if m.no_infeasible_binary_supports_beyond_dsep_up_to(4)]
print(f"Remaining after testing with supports up to 4: {len(still_remaining_after_supports_on_four_events)}")
