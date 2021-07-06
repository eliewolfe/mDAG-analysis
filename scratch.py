import networkx as nx
import numpy as np
def CanonicalQ(G):
    possibleorderings = list(nx.all_topological_sorts(G))
    print(possibleorderings)
    possibleadjmats = np.array([nx.adjacency_matrix(G, order).toarray() for order in possibleorderings])
    flattedarrays = np.vstack(
        tuple(np.hstack(tuple(np.diag(ar, i) for i in np.arange(G.number_of_nodes()))) for ar in possibleadjmats))
    triangle_number = 2 * (flattedarrays.shape[1])
    signatures = np.dot(flattedarrays, 2 ** np.flip(np.arange(triangle_number).reshape((2, -1))[0]))
    # canonical_signature = np.amin(signatures)
    # print(signatures)
    minpos = np.argmin(signatures)
    return np.array_equal(possibleadjmats[minpos],nx.adjacency_matrix(G, sorted(G.nodes())).toarray())
    #print(np.transpose([[",".join(map(str,possibleorderings[i])) for i in np.where(signatures==signatures[minpos])[0]]]))
    # return [possibleorderings[i] for i in np.where(signatures==signatures[minpos])[0]]
    # minpos=np.argmin(signatures)
    # return signatures[minpos],possibleadjmats[minpos],np.transpose([[",".join(map(str,possibleorderings[i])) for i in np.where(signatures==signatures[minpos])[0]]])
