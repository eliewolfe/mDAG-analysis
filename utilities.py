from __future__ import absolute_import
import networkx as nx
import numpy as np
from operator import itemgetter
from radix import bitarray_to_int

def partsextractor(thing_to_take_parts_of, indices):
    if len(indices) == 0:
        return tuple()
    elif len(indices) == 1:
        return (itemgetter(*indices)(thing_to_take_parts_of),)
    else:
        return itemgetter(*indices)(thing_to_take_parts_of)


def nx_to_tuples(g):
    return tuple(sorted(g.edges()))

def nx_to_bitarray(g):
    return nx.to_numpy_array(g, nodelist=sorted(g.nodes()), dtype=bool)

def nx_to_int(g):
    # return nx_to_tuples(g)
    return bitarray_to_int(nx_to_bitarray(g)).tolist()


def hypergraph_to_canonical_tuples(hypergraph):
    return tuple(sorted(map(lambda s: tuple(sorted(s)), hypergraph)))

# def hypergraph_to_bitarray(integers_hypergraph):
#     nof_nodes = np.hstack(integers_hypergraph).max()+1
#     r = np.zeros((len(integers_hypergraph), nof_nodes), dtype=bool)
#     for i, lp in enumerate(integers_hypergraph):
#         r[i, list(lp)] = True
#     return r[np.lexsort(r.T)]

def hypergraph_to_bitarray(integers_hypergraph):
    nof_nodes = np.hstack(integers_hypergraph).max()+1
    r = np.zeros((len(integers_hypergraph), nof_nodes), dtype=bool)
    for i, lp in enumerate(hypergraph_to_canonical_tuples(integers_hypergraph)):
        r[i, list(lp)] = True
    return r

def hypergraph_to_int(integers_hypergraph):
    # return hypergraph_to_canonical_tuples(integers_hypergraph)
    return bitarray_to_int(hypergraph_to_bitarray(integers_hypergraph)).tolist()



if __name__ == '__main__':
    print(partsextractor({'a': 1, 'b': 2}, ['a', 'b']))