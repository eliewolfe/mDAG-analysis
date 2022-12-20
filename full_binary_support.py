from __future__ import absolute_import
from hypergraphs import Hypergraph, LabelledHypergraph
from directed_structures import DirectedStructure, LabelledDirectedStructure
from mDAG_advanced import mDAG
from metagraph_advanced import further_classify_by_attributes
import itertools
from more_itertools import ilen


G_a=mDAG(DirectedStructure([(0,1),(1,2),(2,3),(1,3)],4),Hypergraph([(0,2),(1,2),(0,3)],4))
G_b=mDAG(DirectedStructure([(0,1),(1,2),(2,3),(1,3)],4),Hypergraph([(0,2),(1,2),(2,3),(0,3)],4))
G_c=mDAG(DirectedStructure([(0,1),(1,2),(2,3),(1,3)],4),Hypergraph([(0,2),(1,3),(0,3)],4))
G_d=mDAG(DirectedStructure([(0,1),(1,2),(2,3),(1,3)],4),Hypergraph([(0,2),(2,3),(1,3),(0,3)],4))
G_e=mDAG(DirectedStructure([(0,1),(1,2),(2,3),(1,3)],4),Hypergraph([(0,2),(1,3),(1,2),(0,3)],4))
G_f=mDAG(DirectedStructure([(0,1),(1,2),(2,3),(1,3)],4),Hypergraph([(0,2),(1,3),(2,3),(1,2),(0,3)],4))
G_g=mDAG(DirectedStructure([(0,1),(1,2),(2,3),(1,3)],4),Hypergraph([(0,2),(1,2,3),(0,3)],4))



smart_supports_dict = dict()
singletons_dict={1:[]}
non_singletons_dict={1:[(G_a,G_b,G_c,G_d)]}
for k in range(2, 8):
    print("[Working on nof_events={}]".format(k))
    smart_supports_dict[k] = further_classify_by_attributes([(G_a,G_b,G_c,G_d)],
                                                    [('infeasible_4222_supports_n_events',
                                                      k)], verbose=True)
    singletons_dict[k] = list(itertools.chain.from_iterable(
        filter(lambda eqclass: (len(eqclass) == 1),
               smart_supports_dict[k]))) + singletons_dict[k - 1]
    non_singletons_dict[k] = sorted(
        filter(lambda eqclass: (len(eqclass) > 1),
               smart_supports_dict[k]), key=len)
    print("# of singleton classes from also considering Supports Up To {}: ".format(k), len(singletons_dict[k]))
    print("# of non-singleton classes from also considering Supports Up To {}: ".format(k), len(non_singletons_dict[k]),", comprising {} total graph patterns".format(ilen(itertools.chain.from_iterable(non_singletons_dict[k]))))  