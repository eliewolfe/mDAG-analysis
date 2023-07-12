from __future__ import absolute_import
from metagraph_advanced import Observable_unlabelled_mDAGs


n=3
metagraph = Observable_unlabelled_mDAGs(n, fully_foundational=False, verbose=False)
labelled_mDAGs = metagraph.all_labelled_mDAGs
print(f"Number of {n}-vis node labelled mDAGs: {len(labelled_mDAGs)}")
unlabelled_mDAGs = list(metagraph.all_unlabelled_mDAGs_faster)
print(f"Number of {n}-vis node unlabelled mDAGs: {len(unlabelled_mDAGs)}")
unresolved_by_esep = [m for m in unlabelled_mDAGs if m.all_esep_unlabelled in metagraph.latent_free_esep_patterns]
print(f"Number of {n}-vis node unlabelled mDAGs with potentially-boring e-sep patterns: {len(unresolved_by_esep)}")
provably_boring = metagraph.all_unlabelled_mDAGs_latent_free_equivalent
provably_boring_unique_unlabelled_ids = set(m.unique_unlabelled_id for m in provably_boring)
print(f"Number of {n}-vis node unlabelled boring mDAGs: {len(provably_boring)}")
# potentially_interesting = [m for m in unlabelled_mDAGs if m.unique_unlabelled_id not in provably_boring_unique_unlabelled_ids]
potentially_interesting = [m for m in unresolved_by_esep if m.unique_unlabelled_id not in provably_boring_unique_unlabelled_ids]
print(f"Number of {n}-vis node unlabelled potentially interesting mDAGs: {len(potentially_interesting)}")
# print(potentially_interesting)
print(f"**ALL FURTHER REDUCTIONS ARE PERFORMED ON THE SET OF POTENTIALLY INTERESTING mDAGs**")
fundamental_unlabelled = [m for m in potentially_interesting if m.fundamental_graphQ]
print(f"Number of {n}-vis node unlabelled fundamental mDAGs: {len(fundamental_unlabelled)}")
canonical_unlabelled = [m for m in potentially_interesting if m.has_no_eventually_splittable_face]
print(f"Number of {n}-vis node unlabelled canonical mDAGs: {len(canonical_unlabelled)}")
canonical_fundamental_unlabelled = [m for m in fundamental_unlabelled if m.has_no_eventually_splittable_face]
print(f"Number of {n}-vis node unlabelled fundamental canonical mDAGs: {len(canonical_fundamental_unlabelled)}")
canonical_fundamental_unlabelled_after_esep_v2 = [m for m in canonical_fundamental_unlabelled if not m.interesting_via_non_maximal]
# print(canonical_fundamental_unlabelled_after_esep_v2)
# print(canonical_fundamental_unlabelled_after_esep_v2[1].eventually_splittable_faces)
# print(canonical_fundamental_unlabelled_after_esep_v2[1].splittable_faces)
print(f"Remaining after applying Shashaank e-sep vs d-sep test: {len(canonical_fundamental_unlabelled_after_esep_v2)}")
canonical_fundamental_unlabelled_after_esep_and_binary_supports_2_events = [
    m for m in canonical_fundamental_unlabelled_after_esep_v2 if m.no_infeasible_binary_supports_beyond_dsep_up_to(2)]
print(f"Remaining after testing with supports on 2 events: {len(canonical_fundamental_unlabelled_after_esep_and_binary_supports_2_events)}")


print("\n\n NOW CONSIDERING 4 NODES \n")

n=4
metagraph = Observable_unlabelled_mDAGs(n, fully_foundational=False, verbose=False)

# labelled_mDAGs = metagraph.all_labelled_mDAGs
# print(f"Number of {n}-vis node labelled mDAGs: {len(labelled_mDAGs)}")
unlabelled_mDAGs = list(metagraph.all_unlabelled_mDAGs_faster)
print(f"Number of {n}-vis node unlabelled mDAGs: {len(unlabelled_mDAGs)}")
# provably_boring = metagraph.all_unlabelled_mDAGs_latent_free_equivalent
potentially_interesting_classes = metagraph.NOT_latent_free_eqclasses
potentially_interesting_after_HLP = set().union(*potentially_interesting_classes)
print(f"Number of {n}-vis node unlabelled mDAGs not resolved by HLP: {len(potentially_interesting_after_HLP)}")
unresolved_by_esep = [m for m in unlabelled_mDAGs if m.all_esep_unlabelled in metagraph.latent_free_esep_patterns]
print(f"Number of {n}-vis node unlabelled mDAGs not resolved by e-sep: {len(unresolved_by_esep)}")
unresolved_by_esep_or_HLP = [m for m in potentially_interesting_after_HLP if m.all_esep_unlabelled in metagraph.latent_free_esep_patterns]
print(f"Number of {n}-vis node unlabelled mDAGs resolved by e-sep or HLP: {len(unresolved_by_esep_or_HLP)}")
# provably_boring_unique_unlabelled_ids = set(m.unique_unlabelled_id for m in provably_boring)
# print(f"Number of {n}-vis node unlabelled boring mDAGs: {len(provably_boring)}")
# potentially_interesting = [m for m in unlabelled_mDAGs if m.unique_unlabelled_id not in provably_boring_unique_unlabelled_ids]
# print(f"Number of {n}-vis node unlabelled potentially interesting mDAGs: {len(potentially_interesting)}")
print(f"**ALL FURTHER REDUCTIONS ARE PERFORMED ON THE SET OF POTENTIALLY INTERESTING mDAGs**")
fundamental_unlabelled = [m for m in unresolved_by_esep_or_HLP if m.fundamental_graphQ]
# print(f"Number of {n}-vis node unlabelled fundamental mDAGs: {len(fundamental_unlabelled)}")
# canonical_unlabelled = [m for m in unresolved_by_esep_or_HLP if m.has_no_eventually_splittable_face]
# print(f"Number of {n}-vis node unlabelled canonical mDAGs: {len(canonical_unlabelled)}")
canonical_fundamental_unlabelled = [m for m in fundamental_unlabelled if m.has_no_eventually_splittable_face]
print(f"Number of {n}-vis node unlabelled fundamental canonical mDAGs: {len(canonical_fundamental_unlabelled)}")
canonical_fundamental_unlabelled_after_esep_v2 = [m for m in canonical_fundamental_unlabelled if not m.interesting_via_non_maximal]
# print(canonical_fundamental_unlabelled_after_esep_v2)
# print(canonical_fundamental_unlabelled_after_esep_v2[1].eventually_splittable_faces)
# print(canonical_fundamental_unlabelled_after_esep_v2[1].splittable_faces)
print(f"Remaining after applying Shashaank e-sep vs d-sep test: {len(canonical_fundamental_unlabelled_after_esep_v2)}")
canonical_fundamental_unlabelled_after_esep_and_binary_supports_2_events = [
    m for m in canonical_fundamental_unlabelled_after_esep_v2 if m.no_infeasible_binary_supports_beyond_dsep_up_to(2)]
print(f"Remaining after testing with supports up to 2: {len(canonical_fundamental_unlabelled_after_esep_and_binary_supports_2_events)}")
# print(canonical_fundamental_unlabelled_after_esep_and_binary_supports_2_events)
canonical_fundamental_unlabelled_after_esep_and_binary_supports_4_events = [
    m for m in canonical_fundamental_unlabelled_after_esep_and_binary_supports_2_events if m.no_infeasible_binary_supports_beyond_dsep_up_to(4)]
print(f"Remaining after testing with supports up to 4: {len(canonical_fundamental_unlabelled_after_esep_and_binary_supports_4_events)}")
# print(canonical_fundamental_unlabelled_after_esep_and_binary_supports_4_events)
from the_7_hard_graphs import the_7_hard_graphs
print("If everything works correctly, we should have just the 7 hard graphs and the Bell DAG leftover.")
print(set(canonical_fundamental_unlabelled_after_esep_and_binary_supports_4_events).difference(the_7_hard_graphs))
print("Yup!")


#
# n=4
# metagraph = Observable_unlabelled_mDAGs(n)
# labelled_mDAGs = metagraph.all_labelled_mDAGs
# print(f"Number of {n}-vis node labelled mDAGs: {len(labelled_mDAGs)}")
# unlabelled_mDAGs = list(metagraph.all_unlabelled_mDAGs_faster)
# print(f"Number of {n}-vis node unlabelled mDAGs: {len(unlabelled_mDAGs)}")
# fundamental_unlabelled = [m for m in unlabelled_mDAGs if m.fundamental_graphQ]
# print(f"Number of {n}-vis node unlabelled fundamental mDAGs: {len(fundamental_unlabelled)}")
# canonical_unlabelled = [m for m in unlabelled_mDAGs if m.has_no_eventually_splittable_face]
# print(f"Number of {n}-vis node unlabelled canonical mDAGs: {len(canonical_unlabelled)}")
# canonical_fundamental_unlabelled = [m for m in fundamental_unlabelled if m.has_no_eventually_splittable_face]
# print(f"Number of {n}-vis node unlabelled fundamental canonical mDAGs: {len(canonical_fundamental_unlabelled)}")
# canonical_fundamental_unlabelled_after_esep_v2 = [m for m in canonical_fundamental_unlabelled if not m.interesting_via_e_sep_theorem]
# print(f"Remaining after applying Shashaank e-sep vs d-sep test: {len(canonical_fundamental_unlabelled_after_esep_v2)}")
# # canonical_fundamental_unlabelled_after_esep = [m for m in canonical_fundamental_unlabelled if m.no_esep_beyond_dsep(2)]
# # print(f"Remaining after applying Wolfe e-sep vs d-sep test: {len(canonical_fundamental_unlabelled_after_esep)}")
# # canonical_fundamental_unlabelled_after_esep_and_binary_supports_4_events = [
# #     m for m in canonical_fundamental_unlabelled_after_esep_v2 if m.no_infeasible_binary_supports_beyond_dsep_up_to(4)]
# # print(f"Remaining after testing with supports [expect 7 nontrivial undetected]: {len(canonical_fundamental_unlabelled_after_esep_and_binary_supports_4_events)}")

