from __future__ import absolute_import
from mDAG_advanced import mDAG
from metagraph_advanced import Observable_mDAGs_Analysis


Observable_mDAGs3 = Observable_mDAGs_Analysis(nof_observed_variables=3, max_nof_events_for_supports=3)

print(Observable_mDAGs3.representative_mDAGs_list[0])

print(Observable_mDAGs3.representative_mDAGs_list[0].skeleton_instance)

print(Observable_mDAGs3.representative_mDAGs_list[0].skeleton)

print(Observable_mDAGs3.representative_mDAGs_list[-1].skeleton_instance)

print(Observable_mDAGs3.representative_mDAGs_list[-1].skeleton)




