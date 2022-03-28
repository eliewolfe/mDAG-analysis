from __future__ import absolute_import
import numpy as np
import itertools
import networkx as nx
from hypergraphs import Hypergraph, LabelledHypergraph, hypergraph_full_cleanup
from directed_structures import DirectedStructure, LabelledDirectedStructure
# from radix import bitarray_to_int
from mDAG_advanced import mDAG
from sys import hexversion

if hexversion >= 0x3080000:
    from functools import cached_property
elif hexversion >= 0x3060000:
    from backports.cached_property import cached_property
else:
    cached_property = property


def C_facets_not_dominated_by_Q(c_facets, q_facets):
    c_facets_copy = c_facets.copy()
    for Q_facet in q_facets:
        dominated_by_quantum = set(filter(Q_facet.issuperset, c_facets_copy))
        c_facets_copy.difference_update(dominated_by_quantum)
    return c_facets_copy


def upgrade_to_QmDAG(mDAG):
    return QmDAG(
        mDAG.directed_structure_instance,
        Hypergraph([], mDAG.number_of_visible),
        mDAG.simplicial_complex_instance)


# This class does NOT represent every possible quantum causal structure. It only represents the causal structures where every quantum latent is exogenized. This is the case, for example, of the known QC Gaps.
class QmDAG:
    def __init__(self, directed_structure_instance, C_simplicial_complex_instance, Q_simplicial_complex_instance):
        self.directed_structure_instance = directed_structure_instance
        self.number_of_visible = self.directed_structure_instance.number_of_visible
        assert directed_structure_instance.number_of_visible == C_simplicial_complex_instance.number_of_visible, 'Different number of nodes in directed structure vs classical simplicial complex.'
        assert directed_structure_instance.number_of_visible == Q_simplicial_complex_instance.number_of_visible, 'Different number of nodes in directed structure vs quantum simplicial complex.'

        self.Q_simplicial_complex_instance = Q_simplicial_complex_instance
        self.C_simplicial_complex_instance = Hypergraph(C_facets_not_dominated_by_Q(
            C_simplicial_complex_instance.simplicial_complex_as_sets,
            Q_simplicial_complex_instance.simplicial_complex_as_sets
        ), self.number_of_visible)
        if hasattr(self.directed_structure_instance, 'variable_names'):
            self.variable_names = self.directed_structure_instance.variable_names
            if hasattr(self.C_simplicial_complex_instance, 'variable_names'):
                assert frozenset(self.variable_names) == frozenset(
                    self.C_simplicial_complex_instance.variable_names), 'Error: Inconsistent node names.'
                if not tuple(self.variable_names) == tuple(self.C_simplicial_complex_instance.variable_names):
                    print('Warning: Inconsistent node ordering. Following ordering of directed structure!')
        if hasattr(self.directed_structure_instance, 'variable_names'):
            self.variable_names = self.directed_structure_instance.variable_names
            if hasattr(self.Q_simplicial_complex_instance, 'variable_names'):
                assert frozenset(self.variable_names) == frozenset(
                    self.Q_simplicial_complex_instance.variable_names), 'Error: Inconsistent node names.'
                if not tuple(self.variable_names) == tuple(self.Q_simplicial_complex_instance.variable_names):
                    print('Warning: Inconsistent node ordering. Following ordering of directed structure!')
        self.visible_nodes = self.directed_structure_instance.visible_nodes
        self.classical_latent_nodes = tuple(
            range(self.number_of_visible, self.C_simplicial_complex_instance.number_of_visible_plus_latent))
        self.nonsingleton_classical_latent_nodes = tuple(range(self.number_of_visible,
                                                               self.C_simplicial_complex_instance.number_of_visible_plus_nonsingleton_latent))
        # it is not necessary to talk about quantum singletons in the first place:
        self.quantum_latent_nodes = tuple(range(self.C_simplicial_complex_instance.number_of_visible_plus_latent,
                                                self.Q_simplicial_complex_instance.number_of_visible_plus_nonsingleton_latent + self.C_simplicial_complex_instance.number_of_visible_plus_latent - self.number_of_visible))

    @cached_property
    def as_string(self):
        return 'Children'.ljust(10) + ': ' + self.directed_structure_instance.as_string \
               + '\nClassical'.ljust(11) + ': ' + self.C_simplicial_complex_instance.as_string \
               + '\nQuantum'.ljust(11) + ': ' + self.Q_simplicial_complex_instance.as_string + '\n'

    def __str__(self):
        return self.as_string

    def __repr__(self):
        return self.as_string

    @cached_property
    def unique_id(self):
        # Returns a unique identification tuple.
        return (
            self.number_of_visible,
            self.directed_structure_instance.as_integer,
            self.C_simplicial_complex_instance.as_integer,
            self.Q_simplicial_complex_instance.as_integer,
        )

    def __hash__(self):
        return hash(self.unique_id)

    def __eq__(self, other):
        return self.unique_id == other.unique_id

    @cached_property
    def unique_unlabelled_id(self):
        # Returns a unique identification tuple up to relabelling.
        return (self.number_of_visible,) + min(zip(
            self.directed_structure_instance.as_integer_permutations,
            self.C_simplicial_complex_instance.as_integer_permutations,
            self.Q_simplicial_complex_instance.as_integer_permutations,
        ))

    # def clean_C_simplicial_complex(self):  #remove classical facets that are redundant to quantum facets
    #     new_C_simplicial_complex=self.C_simplicial_complex_instance.simplicial_complex_as_sets.copy()
    #     for Q_facet in self.Q_simplicial_complex_instance.simplicial_complex_as_sets:
    #         dominated_by_quantum = set(filter(Q_facet.issuperset, new_C_simplicial_complex))
    #         new_C_simplicial_complex.difference_update(dominated_by_quantum)
    #     return Hypergraph(new_C_simplicial_complex, self.number_of_visible)

    #ON THE POINT DISTRIBUTION TRICK

    def subgraph(self, list_of_nodes):
        return QmDAG(
            LabelledDirectedStructure(list_of_nodes, self.directed_structure_instance.edge_list),
            LabelledHypergraph(list_of_nodes, self.C_simplicial_complex_instance.simplicial_complex_as_sets),
            LabelledHypergraph(list_of_nodes, self.Q_simplicial_complex_instance.simplicial_complex_as_sets),
        )

    def fix_to_point_distribution_QmDAG(self, node):  # returns a smaller QmDAG
        return self.subgraph(self.visible_nodes[:node] + self.visible_nodes[(node + 1):])

    def _subgraphs_generator(self):
        for r in range(3, self.number_of_visible):
            for to_keep in itertools.combinations(self.visible_nodes, r):
                yield self.subgraph(to_keep)

    @cached_property
    def unique_unlabelled_ids_obtainable_by_PD_trick(self):
        return [subQmDAG.unique_unlabelled_id for subQmDAG in self._subgraphs_generator()]

    # ON THE MARGINALIZATION TRICK

    def classical_sibling_sets_of(self, node):
        return [facet.difference({node}) for facet in self.C_simplicial_complex_instance.simplicial_complex_as_sets if
                node in facet]

    def quantum_sibling_sets_of(self, node):
        return [facet.difference({node}) for facet in self.Q_simplicial_complex_instance.simplicial_complex_as_sets if
                node in facet]

    def quantum_siblings_of(self, node):
        return set(itertools.chain.from_iterable(self.quantum_sibling_sets_of(node)))

    def marginalize(self, node, apply_teleportation=False):  # returns a smaller QmDAG
        remaining_nodes = self.visible_nodes[:node] + self.visible_nodes[(node + 1):]
        # Pass visible children on to visible children
        # Pass latent children on to visible children **classically**
        # Apply teleportation
        visible_children = set(np.flatnonzero(self.directed_structure_instance.as_bit_square_matrix[node]))
        visible_parents = set(np.flatnonzero(self.directed_structure_instance.as_bit_square_matrix[:, node]))
        # print("Visible Children: ", visible_children)
        # print("Visible Parents: ", visible_parents)
        new_directed_edges = set(self.directed_structure_instance.edge_list)
        for parent in visible_parents:
            for child in visible_children:
                new_directed_edges.add((parent, child))

        new_C_facets = self.C_simplicial_complex_instance.simplicial_complex_as_sets.copy()
        C_facets_to_grow = self.classical_sibling_sets_of(node)
        for C_facet_to_grow in C_facets_to_grow:
            new_C_facets.add(C_facet_to_grow.union(visible_children))


        Q_facets_to_grow = self.quantum_sibling_sets_of(node)
        for Q_facet_to_grow in Q_facets_to_grow:
            new_C_facets.add(Q_facet_to_grow.union(visible_children))
        new_C_facets = hypergraph_full_cleanup(new_C_facets)

        if not apply_teleportation:
            return QmDAG(
                LabelledDirectedStructure(remaining_nodes, list(new_directed_edges)),
                LabelledHypergraph(remaining_nodes, new_C_facets),
                LabelledHypergraph(remaining_nodes, self.Q_simplicial_complex_instance.simplicial_complex_as_sets),
                )

        else:
            # teleportation_children_possibilities = list(filter(visible_children.issuperset, Q_facets_to_grow))
            teleportable_children = self.quantum_siblings_of(node).intersection(visible_children)
            # for teleportable_children in teleportation_children_possibilities:
            new_Q_facets = self.Q_simplicial_complex_instance.simplicial_complex_as_sets.copy()
            for Q_facet_to_grow in Q_facets_to_grow:
                new_Q_facets.add(Q_facet_to_grow.union(teleportable_children))
            new_Q_facets = hypergraph_full_cleanup(new_Q_facets)
            # if not frozenset(new_Q_facets) == frozenset(self.Q_simplicial_complex_instance.simplicial_complex_as_sets):
            return QmDAG(
                LabelledDirectedStructure(remaining_nodes, list(new_directed_edges)),
                LabelledHypergraph(remaining_nodes, new_C_facets),
                LabelledHypergraph(remaining_nodes, new_Q_facets),
                )



    @cached_property
    def as_mDAG(self):
        return mDAG(
            self.directed_structure_instance,
            Hypergraph(C_facets_not_dominated_by_Q(
                self.C_simplicial_complex_instance.simplicial_complex_as_sets,
                self.Q_simplicial_complex_instance.simplicial_complex_as_sets
            ).union(
                self.Q_simplicial_complex_instance.simplicial_complex_as_sets
            ), self.number_of_visible)
        )


    def _unique_unlabelled_ids_obtainable_by_marginalization(self, **kwargs):
        for node in self.visible_nodes:
            # for sub_QmDAG in self.marginalize(node, **kwargs):
            #     yield sub_QmDAG.unique_unlabelled_id
            yield self.marginalize(node, **kwargs).unique_unlabelled_id

    @cached_property
    def unique_unlabelled_ids_obtainable_by_naive_marginalization(self):
        return set(self._unique_unlabelled_ids_obtainable_by_marginalization(apply_teleportation=False))

    @cached_property
    def unique_unlabelled_ids_obtainable_by_marginalization(self):
        return set(self._unique_unlabelled_ids_obtainable_by_marginalization(apply_teleportation=True))

    #Iterate over sets of parents that we want to disconnect from the target
    def Fritz(self, target,set_of_visible_parents_to_delete,set_of_C_facets_to_delete, set_of_Q_facets_to_delete):
        if not set_of_visible_parents_to_delete.issubset(set(np.flatnonzero(self.directed_structure_instance.as_bit_square_matrix[:, target]))):
            return "Visible parents are not correct"
        if not set_of_C_facets_to_delete.issubset(self.C_simplicial_complex_instance.simplicial_complex_as_sets) or not set_of_Q_facets_to_delete.issubset(self.Q_simplicial_complex_instance.simplicial_complex_as_sets):
            return "Latent parents are not correct"
        dict_latents={i:children for i, children in zip(self.as_mDAG.nonsingleton_latent_nodes, self.as_mDAG.simplicial_complex_instance.compressed_simplicial_complex)}
        visible_parents = set(np.flatnonzero(self.directed_structure_instance.as_bit_square_matrix[:, target]))
        latent_parents=set(i for i in dict_latents.keys() if target in dict_latents[i])
        facets_with_target=set(facet for facet in self.C_simplicial_complex_instance.simplicial_complex_as_sets.union(self.Q_simplicial_complex_instance.simplicial_complex_as_sets) if target in facet)
        siblings_by_latent=set(node for facet in facets_with_target for node in facet if node!=target)
        set_of_latent_parents_to_delete=set(i for i in dict_latents.keys() if frozenset(dict_latents[i]) in set_of_C_facets_to_delete.union(set_of_Q_facets_to_delete))
        parents_not_to_delete=visible_parents.difference(set_of_visible_parents_to_delete).union(latent_parents.difference(set_of_latent_parents_to_delete))
        s=visible_parents.difference(set_of_visible_parents_to_delete).union(siblings_by_latent)
        # Y, the set that can perfectly predict the target, is a set of visible parents or siblings by latent that does not overlap with the parents that we will disconnect from the target in the end. It is a subset of s:
        for Y in [list(subset) for i in range(0, len(s) + 1) for subset in itertools.combinations(s, i)]:
            Y_satisfies_condition1=True
            Y_satisfies_condition2=True
            for element in Y:
                # it should be d-separated from any of the nodes to be disconnected from target:
                for parent_to_disconnect in set_of_visible_parents_to_delete.union(set_of_latent_parents_to_delete):
                    if not nx.d_separated(self.as_mDAG.as_graph, {element}, {parent_to_disconnect}, set()):  
                        Y_satisfies_condition1=False
                        break
            if Y_satisfies_condition1:
                other_parents=parents_not_to_delete.difference(Y)
                for W in other_parents:
                    W_not_d_separated_from_Y=False
                    for element in Y:
                        if not nx.d_separated(self.as_mDAG.as_graph, {element}, {W}, set()):
                            W_not_d_separated_from_Y=True
                            break
                    if not W_not_d_separated_from_Y:
                        Y_satisfies_condition2=False
                        break
                if Y_satisfies_condition2:
                    return True
        return False
                    
    def _unique_unlabelled_ids_obtainable_by_Fritz_without_node_splitting(self):
        for target in self.visible_nodes:
            visible_parents=set(np.flatnonzero(self.directed_structure_instance.as_bit_square_matrix[:, target]))
            C_facets_with_target=set(facet for facet in self.C_simplicial_complex_instance.simplicial_complex_as_sets if target in facet)
            Q_facets_with_target=set(facet for facet in self.Q_simplicial_complex_instance.simplicial_complex_as_sets if target in facet)
            for set_of_visible_parents_to_delete in [set(subset) for i in range(0, len(visible_parents) + 1) for subset in itertools.combinations(visible_parents, i)]:
                for set_of_C_facets_to_delete in [set(subset) for i in range(0, len(C_facets_with_target) + 1) for subset in itertools.combinations(C_facets_with_target, i)]:
                    for set_of_Q_facets_to_delete in [set(subset) for i in range(0, len(Q_facets_with_target) + 1) for subset in itertools.combinations(Q_facets_with_target, i)]:
                        if self.Fritz(target,set_of_visible_parents_to_delete,set_of_C_facets_to_delete,set_of_Q_facets_to_delete):
                            new_directed_structure=self.directed_structure_instance.as_tuples
                            for p in set_of_visible_parents_to_delete:
                                new_directed_structure=tuple(t for t in new_directed_structure if t!=(p,target))
                            new_C_simplicial_complex=self.C_simplicial_complex_instance.simplicial_complex_as_sets.copy()
                            for C_facet in set_of_C_facets_to_delete:
                                new_C_simplicial_complex.remove(C_facet)
                                new_C_facet=frozenset(i for i in C_facet if i!=target)
                                if len(new_C_facet)>1:
                                    new_C_simplicial_complex.add(new_C_facet)
                            new_Q_simplicial_complex=self.Q_simplicial_complex_instance.simplicial_complex_as_sets.copy()
                            for Q_facet in set_of_Q_facets_to_delete:
                                new_Q_simplicial_complex.remove(Q_facet)
                                new_Q_facet=frozenset(i for i in Q_facet if i!=target)
                                if len(new_Q_facet)>1:
                                    new_Q_simplicial_complex.add(new_Q_facet)
                            yield QmDAG(DirectedStructure(new_directed_structure, self.number_of_visible),Hypergraph(new_C_simplicial_complex, self.number_of_visible), Hypergraph(new_Q_simplicial_complex, self.number_of_visible)).unique_unlabelled_id

    @cached_property
    def unique_unlabelled_ids_obtainable_by_Fritz_without_node_splitting(self):
        return set(self._unique_unlabelled_ids_obtainable_by_Fritz_without_node_splitting())

if __name__ == '__main__': 
    QG_Ghost = QmDAG(DirectedStructure([(0,1),(0,3)], 4), Hypergraph([(1,2)], 4), Hypergraph([(2,3)], 4))
    print(QG_Ghost.Fritz(1,{0},set(),set()))
    print(QG_Ghost.Fritz(1,set(),{frozenset((1,2))},set()))
    print(QG_Ghost.unique_unlabelled_ids_obtainable_by_Fritz_without_node_splitting)
    # QG = QmDAG(DirectedStructure([(1, 2), (2, 3)], 4), Hypergraph([(0, 2), (1, 2), (2, 3)], 4),
    #            Hypergraph([(1, 2, 3)], 4))
    # print(QG)
    # print(QG.unique_id)
    # print(QG.unique_unlabelled_id)
    #
    # print(DirectedStructure([(1, 2), (2, 3)], 4).as_bit_square_matrix.astype(int))

# =============================================================================
#     example_for_marginalization = QmDAG(
#         DirectedStructure([(0, 1), (1, 2), (2, 3)], 4),
#         Hypergraph([], 4),
#         Hypergraph([(0, 1), (1, 3), (2, 3)], 4)
#     )
#     marginalized = example_for_marginalization.marginalize(2, apply_teleportation=False)
#     print(marginalized)
#     print(marginalized.unique_unlabelled_id)
#     QG_Instrumental2 = QmDAG(DirectedStructure([(0, 1), (1, 2)], 3), Hypergraph([], 3), Hypergraph([(0, 1), (1, 2)], 3))
#     print(QG_Instrumental2.unique_unlabelled_id)
# 
#     example_for_teleportation = QmDAG(
#         DirectedStructure([(0, 1), (1, 3), (2, 3)], 4),
#         Hypergraph([], 4),
#         Hypergraph([(0, 2), (2, 3)], 4)
#     )
# 
#     print(example_for_teleportation.marginalize(2, apply_teleportation=True))
# 
# =============================================================================
