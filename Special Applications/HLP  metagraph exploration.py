from metagraph_advanced import Observable_unlabelled_mDAGs
my_class = Observable_unlabelled_mDAGs(4, fully_foundational=False, verbose=True)
HLP_graph = my_class.HLP_meta_graph
ids_to_dominate = [m.unique_unlabelled_id for m in my_class.latent_free_DAGs_unlabelled]
from collections import defaultdict
d = defaultdict(list)
for m in my_class.latent_free_DAGs_unlabelled:
    d[m.all_CI_unlabelled].append(m.unique_unlabelled_id)
subids_to_dominate = [v[0] for v in d.values()]
import networkx as nx
dominate_latent_free = [nx.ancestors(HLP_graph, s) for s in ids_to_dominate]
provably_boring = set(ids_to_dominate).union(*dominate_latent_free)
print("Provably boring: ", len(provably_boring))
print("Potentially interesting: ", HLP_graph.number_of_nodes()-len(provably_boring))