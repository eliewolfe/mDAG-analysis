from __future__ import absolute_import
from metagraph_advanced import Observable_unlabelled_mDAGs

n=4
metagraph = Observable_unlabelled_mDAGs(n)
labelled_mDAGs = metagraph.all_labelled_mDAGs
print(f"Number of {n}-vis node labelled mDAGs: {len(labelled_mDAGs)}")
unlabelled_mDAGs = list(metagraph.all_unlabelled_mDAGs_faster)
print(f"Number of {n}-vis node unlabelled mDAGs: {len(unlabelled_mDAGs)}")
fundamental_unlabelled = [m for m in unlabelled_mDAGs if m.fundamental_graphQ]
print(f"Number of {n}-vis node unlabelled fundamental mDAGs: {len(fundamental_unlabelled)}")
canonical_unlabelled = [m for m in unlabelled_mDAGs if m.has_no_eventually_splittable_face]
print(f"Number of {n}-vis node unlabelled canonical mDAGs: {len(canonical_unlabelled)}")
canonical_fundamental_unlabelled = [m for m in fundamental_unlabelled if m.has_no_eventually_splittable_face]
print(f"Number of {n}-vis node unlabelled fundamental canonical mDAGs: {len(canonical_fundamental_unlabelled)}")
canonical_fundamental_unlabelled_after_esep_v2 = [m for m in canonical_fundamental_unlabelled if not m.interesting_via_e_sep_theorem]
print(f"Remaining after applying Shashaank e-sep vs d-sep test: {len(canonical_fundamental_unlabelled_after_esep_v2)}")
# canonical_fundamental_unlabelled_after_esep = [m for m in canonical_fundamental_unlabelled if not m.no_esep_beyond_dsep(2)]
# print(f"Remaining after applying Wolfe e-sep vs d-sep test: {len(canonical_fundamental_unlabelled_after_esep)}")
# canonical_fundamental_unlabelled_after_esep_and_binary_supports_4_events = [
#     m for m in canonical_fundamental_unlabelled_after_esep_v2 if m.no_infeasible_binary_supports_beyond_dsep_up_to(4)]
# print(f"Remaining after testing with supports [expect 7 nontrivial undetected]: {len(canonical_fundamental_unlabelled_after_esep_and_binary_supports_4_events)}")

