from __future__ import absolute_import
from metagraph_advanced import Observable_unlabelled_mDAGs

n=3
print(f"NOW CONSIDERING {n} NODES")
metagraph = Observable_unlabelled_mDAGs(n, fully_foundational=False, verbose=False)
unlabelled_mDAGs = set(metagraph.all_unlabelled_mDAGs)
print(f"Number of {n}-vis node unlabelled mDAGs: {len(unlabelled_mDAGs)}")
# fundamental_unlabelled = set([m for m in unlabelled_mDAGs if m.fundamental_graphQ])
# print(f"Number of {n}-vis node unfactorizable mDAGs: {len(fundamental_unlabelled)}")
# print(f"**ALL FURTHER ANALYSIS PERFORMED ON UNFACTORIZABLE mDAGs**")
potentially_interesting_classes = metagraph.NOT_latent_free_eqclasses
# potentially_interesting_after_HLP = set().union(*potentially_interesting_classes).intersection(fundamental_unlabelled)
potentially_interesting_after_HLP = set().union(*potentially_interesting_classes).intersection(unlabelled_mDAGs)
print(f"Number of {n}-vis node unlabelled mDAGs not resolved by HLP: {len(potentially_interesting_after_HLP)}")
# potentially_interesting_after_Wolfe_bad_HLP_implementation = unlabelled_mDAGs.difference(metagraph.convert_to_unlabelled(metagraph.all_unlabelled_mDAGs_latent_free_equivalent))
# missed_by_Elie = potentially_interesting_after_Wolfe_bad_HLP_implementation.difference(potentially_interesting_after_HLP)
# print(f"Boring classes not detected by Elie's bad implementation of HLP: ({len(missed_by_Elie)} count)")
# for m in missed_by_Elie:
#     if m.fundamental_graphQ and m.has_no_eventually_splittable_face:
#         print(m)
unresolved_by_CI = set([m for m in unlabelled_mDAGs if m.all_CI_unlabelled in metagraph.latent_free_CI_patterns])
print(f"Number of {n}-vis node unlabelled mDAGs not resolved by CI: {len(unresolved_by_CI)}")
unresolved_by_CI_or_HLP = set(potentially_interesting_after_HLP).intersection(unresolved_by_CI)
print(f"Number of {n}-vis node unlabelled mDAGs unresolved by CI or HLP: {len(unresolved_by_CI_or_HLP)}")

unresolved_by_Shaashank = set([m for m in unlabelled_mDAGs if not m.interesting_via_e_sep_theorem])
print(f"Number of {n}-vis node unlabelled mDAGs not resolved by Shaashank's theorem: {len(unresolved_by_Shaashank)}")
unresolved_by_Shaashank_or_HLP = set(potentially_interesting_after_HLP).intersection(unresolved_by_Shaashank)
print(f"Number of {n}-vis node unlabelled mDAGs unresolved by Shaashank's theorem or HLP: {len(unresolved_by_Shaashank_or_HLP)}")

unresolved_by_esep = set([m for m in unlabelled_mDAGs if m.all_esep_unlabelled in metagraph.latent_free_esep_patterns])
print(f"Number of {n}-vis node unlabelled mDAGs not resolved by e-sep: {len(unresolved_by_esep)}")
unresolved_by_esep_or_HLP = set(potentially_interesting_after_HLP).intersection(unresolved_by_esep)
print(f"Number of {n}-vis node unlabelled mDAGs unresolved by e-sep or HLP: {len(unresolved_by_esep_or_HLP)}")

# print("As we proceed now to supports, we wonder if there are graphs with multiple CI relations that could be problematic.")
# print("They are, if any: ", [m for m in unresolved_by_esep_or_HLP if len(m.all_CI)>1])
still_remaining_after_supports_on_two_events = set([m for m in unresolved_by_esep_or_HLP if m.no_infeasible_binary_supports_beyond_dsep_up_to(2)])
print(f"Remaining after testing with supports up to 2: {len(still_remaining_after_supports_on_two_events)}")
still_remaining_after_supports_on_three_events = set([m for m in unresolved_by_esep_or_HLP if m.no_infeasible_binary_supports_beyond_dsep_up_to(3)])
print(f"Remaining after testing with supports up to 2: {len(still_remaining_after_supports_on_three_events)}")
still_remaining_after_supports_on_four_events = set([m for m in still_remaining_after_supports_on_three_events if m.no_infeasible_binary_supports_beyond_dsep_up_to(4)])
print(f"Remaining after testing with supports up to 4: {len(still_remaining_after_supports_on_four_events)}")


print("\n\n")

n=4
print(f"NOW CONSIDERING {n} NODES")
metagraph = Observable_unlabelled_mDAGs(n, fully_foundational=False, verbose=False)
unlabelled_mDAGs = set(metagraph.all_unlabelled_mDAGs)
print(f"Number of {n}-vis node unlabelled mDAGs: {len(unlabelled_mDAGs)}")
# fundamental_unlabelled = set([m for m in unlabelled_mDAGs if m.fundamental_graphQ])
# print(f"Number of {n}-vis node unfactorizable mDAGs: {len(fundamental_unlabelled)}")
# print(f"**ALL FURTHER ANALYSIS PERFORMED ON UNFACTORIZABLE mDAGs**")
potentially_interesting_classes = metagraph.NOT_latent_free_eqclasses
# potentially_interesting_after_HLP = set().union(*potentially_interesting_classes).intersection(fundamental_unlabelled)
potentially_interesting_after_HLP = set().union(*potentially_interesting_classes).intersection(unlabelled_mDAGs)
print(f"Number of {n}-vis node unlabelled mDAGs not resolved by HLP: {len(potentially_interesting_after_HLP)}")
# potentially_interesting_after_Wolfe_bad_HLP_implementation = unlabelled_mDAGs.difference(metagraph.convert_to_unlabelled(metagraph.all_unlabelled_mDAGs_latent_free_equivalent))
# missed_by_Elie = potentially_interesting_after_Wolfe_bad_HLP_implementation.difference(potentially_interesting_after_HLP)
# print(f"Boring classes not detected by Elie's bad implementation of HLP: ({len(missed_by_Elie)} count)")
# for m in missed_by_Elie:
#     if m.fundamental_graphQ and m.has_no_eventually_splittable_face:
#         print(m)
unresolved_by_CI = set([m for m in unlabelled_mDAGs if m.all_CI_unlabelled in metagraph.latent_free_CI_patterns])
print(f"Number of {n}-vis node unlabelled mDAGs not resolved by CI: {len(unresolved_by_CI)}")
unresolved_by_CI_or_HLP = set(potentially_interesting_after_HLP).intersection(unresolved_by_CI)
print(f"Number of {n}-vis node unlabelled mDAGs unresolved by CI or HLP: {len(unresolved_by_CI_or_HLP)}")

unresolved_by_Shaashank = set([m for m in unlabelled_mDAGs if not m.interesting_via_e_sep_theorem])
print(f"Number of {n}-vis node unlabelled mDAGs not resolved by Shaashank's theorem: {len(unresolved_by_Shaashank)}")
unresolved_by_Shaashank_or_HLP = set(potentially_interesting_after_HLP).intersection(unresolved_by_Shaashank)
print(f"Number of {n}-vis node unlabelled mDAGs unresolved by Shaashank's theorem or HLP: {len(unresolved_by_Shaashank_or_HLP)}")

unresolved_by_esep = set([m for m in unlabelled_mDAGs if m.all_esep_unlabelled in metagraph.latent_free_esep_patterns])
print(f"Number of {n}-vis node unlabelled mDAGs not resolved by e-sep: {len(unresolved_by_esep)}")
unresolved_by_esep_or_HLP = set(potentially_interesting_after_HLP).intersection(unresolved_by_esep)
print(f"Number of {n}-vis node unlabelled mDAGs unresolved by e-sep or HLP: {len(unresolved_by_esep_or_HLP)}")

# print("As we proceed now to supports, we wonder if there are graphs with multiple CI relations that could be problematic.")
# print("They are, if any: ", [m for m in unresolved_by_esep_or_HLP if len(m.all_CI)>1])
still_remaining_after_supports_on_two_events = set([m for m in unresolved_by_esep_or_HLP if m.no_infeasible_binary_supports_beyond_dsep_up_to(2)])
print(f"Remaining after testing with supports up to 2: {len(still_remaining_after_supports_on_two_events)}")
still_remaining_after_supports_on_three_events = set([m for m in unresolved_by_esep_or_HLP if m.no_infeasible_binary_supports_beyond_dsep_up_to(3)])
print(f"Remaining after testing with supports up to 2: {len(still_remaining_after_supports_on_three_events)}")
still_remaining_after_supports_on_four_events = set([m for m in still_remaining_after_supports_on_three_events if m.no_infeasible_binary_supports_beyond_dsep_up_to(4)])
print(f"Remaining after testing with supports up to 4: {len(still_remaining_after_supports_on_four_events)}")