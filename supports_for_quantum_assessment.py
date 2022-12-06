from __future__ import absolute_import
from supports import explore_candidates
from metagraph_advanced import Observable_mDAGs_Analysis
import json

# if __name__ == '__main__':
Observable_mDAGs4 = Observable_mDAGs_Analysis(nof_observed_variables=4, max_nof_events_for_supports=0)
to_analyze = Observable_mDAGs4.representative_mDAGs_list
all_mDAGs_to_study = {ICmDAG: ICmDAG.infeasible_binary_supports_n_events_beyond_esep_as_matrices(4) for ICmDAG in explore_candidates(
    to_analyze, verbose=True)}
good_mDAGs_to_study = {key:val for key, val in all_mDAGs_to_study.items() if len(val)>0}
challenges = sorted(list(filter(lambda kv: len(kv[1]) <= 4, good_mDAGs_to_study.items())), key=lambda kv: len(kv[1]))
f = open('mDAGs_to_study.json', 'w')
print(json.dumps({key.as_string: val.tolist() for key, val in challenges}), file=f)
f.close()