from __future__ import absolute_import
from hypergraphs import Hypergraph
from directed_structures import DirectedStructure
from mDAG_advanced import mDAG
from supports_beyond_esep import SmartSupportTesting


list_shashaank=[[['', '', '', 'AC', 'AB'], 'BCDE'],
[['', '', '', 'AC', 'BC', 'AB'], 'CDEF'],    
[['', '', '', 'AC', 'BCD', 'AB'], 'CDEF'],
[['', '', '', 'AB', 'AC', 'BE', 'CDF'], 'DEFG'],
[['', '', '', 'AC', 'BD', 'ABE', 'CEF'], 'DEFG'],
[['', '', '', 'AC', 'BD', 'AE', 'BCF'], 'DEFG'],
[['', '', '', 'AC', 'BD', 'AE', 'BCEF'], 'DEFG'],
[['', '', '', '', 'AC', 'BCD', 'ABDF'], 'DEFG'],
[['', '', '', '', 'AB', 'ACD', 'BDF', 'CEG'], 'EFGH'],
[['', '', '', '', 'ABD', 'ACE', 'BF', 'CDG'], 'EFGH'],
[['', '', '', '', 'AD', 'CE', 'ABC', 'BDFG'], 'EFGH'],
[['', '', '', '', 'AB', 'AC', 'BD', 'CD'], 'EFGH'],
[['', '', '', '', 'AB', 'AC', 'BDF', 'CD'], 'EFGH'],
[['', '', '', '', 'AB', 'AC', 'BDF', 'CDE'], 'EFGH'],
[['', '', '', '', 'AB', 'ACE', 'BDF', 'CD'], 'EFGH'],
[['', '', '', '', 'AB', 'AC', 'BD', 'CDEG'], 'EFGH'],
[['', '', '', '', 'AD', 'BCE', 'ACF', 'BDG'], 'EFGH'],
[['', '', '', '', 'AB', 'AC', 'BDF', 'CDEG'], 'EFGH'],
[['', '', '', '', 'AB', 'ACE', 'BDEF', 'CD'], 'EFGH'],
[['', '', '', '', 'AD', 'BCE', 'ACF', 'BDFG'], 'EFGH'],
[['', '', '', '', 'AD', 'CE', 'ABCF', 'BDFG'], 'EFGH'],
[['', '', '', '', 'AD', 'BE', 'ACF', 'BCDG'], 'EFGH'],
[['', '', '', '', 'AD', 'BE', 'ACF', 'BCDFG'], 'EFGH'],
[['', '', '', '', '', 'CD', 'BE', 'ABD', 'ACEH'], 'FGHI'],[['', '', '', '', '', 'CD', 'BEF', 'ABD', 'ACEH'], 'FGHI'],
[['', '', '', '', '', 'BCD', 'ACEF', 'ADG', 'BEH'], 'FGHI'],
[['', '', '', '', '', 'CD', 'BCE', 'ABDG', 'AEFH'], 'FGHI'],
[['', '', '', '', '', 'CDE', 'BDF', 'ABE', 'ACGH'], 'FGHI'],
[['', '', '', '', '', 'CD', 'BE', 'ABD', 'ACEGH'], 'FGHI'],
[['', '', '', '', '', 'BCD', 'CEF', 'ADG', 'ABEH'], 'FGHI'],
[['', '', '', '', '', 'CD', 'BEF', 'ABD', 'ACEGH'], 'FGHI'],
[['', '', '', '', '', 'CD', 'BE', 'ABDG', 'ACEH'], 'FGHI'],
[['', '', '', '', '', 'CD', 'BE', 'ABDG', 'ACEGH'], 'FGHI'],
[['', '', '', '', '', 'CD', 'BEF', 'ABDG', 'ACEH'], 'FGHI'],
[['', '', '', '', '', 'CD', 'BEF', 'ABDG', 'ACEGH'], 'FGHI'],
[['', '', '', '', '', '', 'CDE', 'BDF', 'ABEH', 'ACFG'], 'GHIJ'],
[['', '', '', '', '', '', 'CDE', 'BDFG', 'ABEH', 'ACFI'], 'GHIJ'],
[['', '', '', '', '', '', 'CDE', 'BDF', 'ABEH', 'ACFGI'], 'GHIJ'],
[['', '', '', 'BC', 'A', 'AB', 'ACF'], 'DEFG'],
[['', '', '', 'BC', 'A', 'AB', 'ACEF'], 'DEFG'],
[['', '', '', 'BC', 'A', 'ABE', 'ACF'], 'DEFG'],
[['', '', '', 'BC', 'A', 'ABE', 'ACEF'], 'DEFG'],
[['', '', '', 'BC', 'AD', 'AB', 'ACF'], 'DEFG'],
[['', '', '', 'BC', 'AB', 'ACE', 'ADF'], 'DEFG'],
[['', '', '', 'BC', 'AD', 'AB', 'ACEF'], 'DEFG'],
[['', '', '', 'BC', 'AD', 'ABE', 'ACF'], 'DEFG'],
[['', '', '', 'BC', 'AD', 'ABE', 'ACEF'], 'DEFG'],
[['', '', '', '', 'BCD', 'AB', 'ACF', 'ADE'], 'EFGH'],
[['', '', '', '', 'BCD', 'AB', 'ACF', 'ADEG'], 'EFGH'],
[['', '', '', '', 'BCD', 'AB', 'ACE', 'ADFG'], 'EFGH'],
[['', '', '', '', 'BCD', 'ABE', 'ACF', 'ADG'], 'EFGH'],
[['', '', '', 'BC', 'AB', 'ACE', 'ACD'], 'DEFG'],
[['', '', '', 'BC', 'AB', 'ACE', 'ACDF'], 'DEFG'],
[['', '', '', '', 'BCD', 'ABD', 'ACD', 'ABC'], 'EFGH']]


len(list_shashaank)

list_unresolved_mDAGs=[]
for m in list_shashaank:
    mDAGm=[list(m[1])]
    p=[]
    for i in range(-4,0):
        p.append(list(m[0][i]))
    mDAGm.append(p)
    list_unresolved_mDAGs.append(mDAGm)


def back_to_mDAG(l):
    edges=[]
    latents=[]
    facets_dict={}
    for i in range(4):
        for parent in l[1][i]:
            if parent in l[0]:
                edges.append((l[0].index(parent),i))
            else:
                if parent not in facets_dict.keys():
                    facets_dict[parent]=[i]
                else:
                    facet=facets_dict[parent].copy()
                    facet.append(i)
                    facets_dict[parent]=facet
    facets=[tuple(facet) for facet in facets_dict.values()]
    return mDAG(DirectedStructure(edges,4),Hypergraph(facets,4))

list_unresolved_mDAGs_my_notation=[]
for mdag in list_unresolved_mDAGs:
    list_unresolved_mDAGs_my_notation.append(back_to_mDAG(mdag))
set_unresolved_mDAGs_my_notation=set(list_unresolved_mDAGs_my_notation)

len(list_unresolved_mDAGs_my_notation)

print(list(set_unresolved_mDAGs_my_notation)[0].no_infeasible_binary_supports_beyond_dsep_up_to(4))
    
print(list(set_unresolved_mDAGs_my_notation)[1].no_infeasible_binary_supports_beyond_dsep_up_to(4))


print("# of unresolved mDAGs that Shashaank sent me:",len(set_unresolved_mDAGs_my_notation))
provably_interesting_via_binary_supports4 = [ICmDAG for ICmDAG in set_unresolved_mDAGs_my_notation if not ICmDAG.no_infeasible_binary_supports_beyond_dsep_up_to(4)]
set_unresolved_mDAGs_my_notation.difference_update(provably_interesting_via_binary_supports4)
print("# of IC Gaps discovered via TC's Algorithm (binary) up to 4 events: ", len(provably_interesting_via_binary_supports4))


# =============================================================================
# eight_mDAG=list(set_unresolved_mDAGs_my_notation)[0] 
# if not eight_mDAG.no_infeasible_binary_supports_beyond_dsep_up_to(8):
#     provably_interesting_via_binary_supports8 = [eight_mDAG]
#     set_unresolved_mDAGs_my_notation.difference_update(provably_interesting_via_binary_supports8)
#     print("Extra IC Gap discovered via TC's Algorithm (binary) up to 8 events: ", eight_mDAG)
#     print("# of IC Gaps still to be assessed: ", len(set_unresolved_mDAGs_my_notation))
#     print("These are the 7 mDAGs that have full support when all variables are binary. Analyzing each one for cardinality 4222:")
#     
# 
# provably_interesting_via_4222_supports=[]
# for mdag in set_unresolved_mDAGs_my_notation:
#     print(mdag)
#     if not mdag.no_infeasible_4222_supports_beyond_dsep_up_to(7):
#         print("is shown interesting by TC's algorithm (cardinality 4222) up to 7 events.")
#         provably_interesting_via_4222_supports.append(mdag)
#     else: 
#         print("is still not shown interesting by TC's algorithm (cardinality 4222) up to 7 events.")
# set_unresolved_mDAGs_my_notation.difference_update(provably_interesting_via_4222_supports)
# 
# print("# of IC Gaps still to be assessed: ", len(set_unresolved_mDAGs_my_notation))
# =============================================================================
