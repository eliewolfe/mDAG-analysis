from __future__ import absolute_import
import numpy as np
import itertools
# import networkx as nx
from hypergraphs import Hypergraph, LabelledHypergraph, hypergraph_full_cleanup, hypergraph_canonicalize_with_deduplication
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

def as_classical_QmDAG(mDAG):
    return QmDAG(
        mDAG.directed_structure_instance,
        mDAG.simplicial_complex_instance,
        Hypergraph([], mDAG.number_of_visible))


# This class does NOT represent every possible quantum causal structure. It only represents the causal structures where every quantum latent is exogenized. This is the case, for example, of the known QC Gaps.
class QmDAG:
    def __init__(self, directed_structure_instance, C_simplicial_complex_instance, Q_simplicial_complex_instance):
        self.directed_structure_instance = directed_structure_instance
        self.number_of_visible = self.directed_structure_instance.number_of_visible
        assert directed_structure_instance.number_of_visible == C_simplicial_complex_instance.number_of_visible, 'Different number of nodes in directed structure vs classical simplicial complex.'
        assert directed_structure_instance.number_of_visible == Q_simplicial_complex_instance.number_of_visible, 'Different number of nodes in directed structure vs quantum simplicial complex.'

        self.Q_simplicial_complex_instance = Q_simplicial_complex_instance
        # print("Raw C_simp_complex:", C_simplicial_complex_instance.simplicial_complex_as_sets)
        if hasattr(C_simplicial_complex_instance, 'variable_names'):
            self.C_simplicial_complex_instance = LabelledHypergraph(
                C_simplicial_complex_instance.variable_names,
                C_facets_not_dominated_by_Q(
                C_simplicial_complex_instance.translated_simplicial_complex,
                Q_simplicial_complex_instance.translated_simplicial_complex
            ))
        else:
            self.C_simplicial_complex_instance = Hypergraph(C_facets_not_dominated_by_Q(
                C_simplicial_complex_instance.simplicial_complex_as_sets,
                Q_simplicial_complex_instance.simplicial_complex_as_sets
            ), self.number_of_visible)
        # print("Utilized C_simp_complex:", self.C_simplicial_complex_instance.simplicial_complex_as_sets)
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
    def subgraphs(self):
        return set(self._subgraphs_generator())

    @cached_property
    def unique_unlabelled_ids_obtainable_by_PD_trick(self):
        return set(subQmDAG.unique_unlabelled_id for subQmDAG in self.subgraphs)

    # ON THE MARGINALIZATION TRICK

    def classical_sibling_sets_of(self, node):
        return set(facet.difference({node}) for facet in self.C_simplicial_complex_instance.simplicial_complex_as_sets if
                node in facet)

    def quantum_sibling_sets_of(self, node):
        return set(facet.difference({node}) for facet in self.Q_simplicial_complex_instance.simplicial_complex_as_sets if
                node in facet)

    def quantum_siblings_of(self, node):
        return set(itertools.chain.from_iterable(self.quantum_sibling_sets_of(node)))

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

    def latent_sibling_sets_of(self, node):
        return set(facet.difference({node}) for facet in self.as_mDAG.simplicial_complex_instance.simplicial_complex_as_sets if
                node in facet)

    def latent_siblings_of(self, node):
        return set(itertools.chain.from_iterable(self.latent_sibling_sets_of(node)))
    
    def has_grandparents_that_are_not_parents(self, node):
        # visible_parents_bit_vec = self.directed_structure_instance.as_bit_square_matrix[:, node]
        # visible_grandparents_bit_vec = np.bitwise_or.reduce(
        #     self.directed_structure_instance.as_bit_square_matrix[:, visible_parents_bit_vec],
        #     axis=0)
        # return not np.array_equal(visible_parents_bit_vec, visible_grandparents_bit_vec)
        visible_parents = set(np.flatnonzero(self.directed_structure_instance.as_bit_square_matrix[:, node]))
        for parent in visible_parents:
            grandparents=set(np.flatnonzero(self.directed_structure_instance.as_bit_square_matrix[:, parent]))
            if not grandparents.issubset(visible_parents):
                return True
            # for granny in grandparents:
            #     if not granny in visible_parents:
            #         return True
        return False
    
    def condition(self, node):
        #assume we already checked that it doesn't have grandparents that are not parents
        remaining_nodes = self.visible_nodes[:node] + self.visible_nodes[(node + 1):]
        # new_directed_edges = set(self.directed_structure_instance.edge_list)
        visible_parents = set(np.flatnonzero(self.directed_structure_instance.as_bit_square_matrix[:, node]))
        new_C_facets=self.C_simplicial_complex_instance.simplicial_complex_as_sets.copy()
        new_C_facets.add(frozenset(visible_parents.union(self.latent_siblings_of(node))))
        new_Q_facets=self.Q_simplicial_complex_instance.simplicial_complex_as_sets.copy()
        new_Q_facets.add(frozenset(self.quantum_siblings_of(node)))
        return QmDAG(
                LabelledDirectedStructure(remaining_nodes, self.directed_structure_instance.edge_list),
                LabelledHypergraph(remaining_nodes, new_C_facets),
                LabelledHypergraph(remaining_nodes, new_Q_facets),
                )

    def _subconditionals(self):
        if self.number_of_visible > 3:
            for node in self.visible_nodes:
                if not self.has_grandparents_that_are_not_parents(node):
                    conditional_QM = self.condition(node)
                    yield conditional_QM
                    for new_QmDAG in conditional_QM.subconditionals:
                        yield new_QmDAG

    @cached_property
    def subconditionals(self):
        return set(self._subconditionals())
    
    @cached_property
    def unique_unlabelled_ids_obtainable_by_conditioning(self):
        return set(new_QmDAG.unique_unlabelled_id for new_QmDAG in self.subconditionals)

    def marginalize(self, node, districts_check=False, apply_teleportation=True):  # returns a smaller QmDAG
        remaining_nodes = self.visible_nodes[:node] + self.visible_nodes[(node + 1):]
        fake_qmDAG = QmDAG(
                DirectedStructure([], 1),
                Hypergraph([], 1),
                Hypergraph([], 1),
                )
        # Pass visible children on to visible children
        # Pass latent children on to visible children **classically**
        # Apply teleportation
        visible_children = set(np.flatnonzero(self.directed_structure_instance.as_bit_square_matrix[node]))
        visible_parents = set(np.flatnonzero(self.directed_structure_instance.as_bit_square_matrix[:, node]))
        new_directed_edges = set(self.directed_structure_instance.edge_list)
        for parent in visible_parents:
            for child in visible_children:
                new_directed_edges.add((parent, child))
        # print("Visible Children: ", visible_children)
        # print("Visible Parents: ", visible_parents)
        new_C_facets = self.C_simplicial_complex_instance.simplicial_complex_as_sets.copy()
        # print(new_C_facets)
        # C_facets_to_grow = self.classical_sibling_sets_of(node)
        # Q_facets_to_grow = self.quantum_sibling_sets_of(node)
        # C_facets_to_grow = C_facets_to_grow + Q_facets_to_grow
        C_facets_to_grow = self.latent_sibling_sets_of(node)
        if len(C_facets_to_grow)>0:
            for C_facet_to_grow in C_facets_to_grow:
                new_C_facets.add(C_facet_to_grow.union(visible_children))
        else:
            new_C_facets.add(frozenset(visible_children))
        # Q_facets_to_grow = self.quantum_sibling_sets_of(node)
        # for Q_facet_to_grow in Q_facets_to_grow:
        #     new_C_facets.add(Q_facet_to_grow.union(visible_children))
        new_C_facets = hypergraph_full_cleanup(new_C_facets)
        new_C_simplicial_complex = LabelledHypergraph(remaining_nodes, new_C_facets)
        if not districts_check:
            ok_to_proceed = True
        else:
            previous_districts = [district.difference({node}) for district in self.as_mDAG.numerical_districts]
            new_districts = new_C_simplicial_complex.translated_districts
            ok_to_proceed = frozenset(map(frozenset, previous_districts)) == frozenset(map(frozenset, new_districts))
            # all(
            #     any(
            #         previous_district == new_district for new_district in new_districts)
            #     for previous_district in previous_districts)
        if not ok_to_proceed:
            return fake_qmDAG
        else:
        # if districts_check:
        #     if not [district.difference({node}) for district in self.as_mDAG.numerical_districts if len(district.difference({node}))>0]==LabelledHypergraph(remaining_nodes, new_C_facets).translated_districts:
        #         return QmDAG(
        #         LabelledDirectedStructure(remaining_nodes, []),
        #         LabelledHypergraph(remaining_nodes, []),
        #         LabelledHypergraph(remaining_nodes, []),
        #         )  #in this case, we give up on the marginalization trick if it does not keep the districts.
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
                Q_facets_to_grow = self.quantum_sibling_sets_of(node)
                for Q_facet_to_grow in Q_facets_to_grow:
                    new_Q_facets.add(Q_facet_to_grow.union(teleportable_children))
                new_Q_facets = hypergraph_full_cleanup(new_Q_facets)
                # if not frozenset(new_Q_facets) == frozenset(self.Q_simplicial_complex_instance.simplicial_complex_as_sets):
                return QmDAG(
                    LabelledDirectedStructure(remaining_nodes, list(new_directed_edges)),
                    LabelledHypergraph(remaining_nodes, new_C_facets),
                    LabelledHypergraph(remaining_nodes, new_Q_facets),
                    )


    def labelled_multi_marginalize(self,
                                   nodes_to_marginalize,
                                   all_nodes,
                                   directed_structure_list,
                                   C_simplicial_complex_instance_as_sets,
                                   Q_simplicial_complex_instance_as_sets):  # returns a smaller QmDAG
        new_directed_structure_set = set(directed_structure_list).copy()
        new_C_simplicial_complex_instance_as_sets = set(C_simplicial_complex_instance_as_sets).copy()
        new_Q_simplicial_complex_instance_as_sets = set(Q_simplicial_complex_instance_as_sets).copy()
        remaining_nodes = set(all_nodes)

        for node in set(nodes_to_marginalize):
            remaining_nodes.discard(node)
            visible_children = set()
            visible_parents = set()
            for i in remaining_nodes:
                if (node, i) in new_directed_structure_set:
                    visible_children.add(i)
                    new_directed_structure_set.remove( (node, i) )
                if (i, node) in new_directed_structure_set:
                    visible_parents.add(i)
                    new_directed_structure_set.remove( (i, node))
            for parent in visible_parents:
                for child in visible_children:
                    new_directed_structure_set.add((parent, child))

            new_C_simplicial_complex_instance_as_sets.add(frozenset(visible_children))
            facets_to_kill = set()
            facets_to_add = set()
            for facet in new_C_simplicial_complex_instance_as_sets:
                if node in facet:
                    marginalized_facet = facet.difference({node}).union(visible_children)
                    facets_to_kill.add(facet)
                    facets_to_add.add(marginalized_facet)
            new_C_simplicial_complex_instance_as_sets.update(facets_to_add)
            new_C_simplicial_complex_instance_as_sets.difference_update(facets_to_kill)
            teleportable_children = set()
            facets_to_expand_by_teleportation = set()
            facets_to_kill = set()
            for facet in new_Q_simplicial_complex_instance_as_sets:
                if node in facet:
                    sub_qfacet = facet.difference({node})
                    facets_to_expand_by_teleportation.add(sub_qfacet)
                    classical_marginalized_facet = sub_qfacet.union(visible_children)
                    teleportable_children.update(sub_qfacet.intersection(visible_children))
                    facets_to_kill = new_Q_simplicial_complex_instance_as_sets.add(facet)
                    new_C_simplicial_complex_instance_as_sets.add(classical_marginalized_facet)
            new_Q_simplicial_complex_instance_as_sets.difference_update(facets_to_kill)
            for facet in facets_to_expand_by_teleportation:
                new_Q_simplicial_complex_instance_as_sets.add(facet.union(teleportable_children))
        remaining_nodes = tuple(remaining_nodes)
        return QmDAG(
                LabelledDirectedStructure(remaining_nodes, list(new_directed_structure_set)),
                LabelledHypergraph(remaining_nodes, hypergraph_full_cleanup(new_C_simplicial_complex_instance_as_sets)),
                LabelledHypergraph(remaining_nodes, hypergraph_full_cleanup(new_Q_simplicial_complex_instance_as_sets)),
            )

    def _submarginals(self, **kwargs):
        if self.number_of_visible > 3:
            for node in self.visible_nodes:
                marginalized_QM = self.marginalize(node, **kwargs)
                yield marginalized_QM
                for new_QmDAG in marginalized_QM.submarginals(**kwargs):
                    yield new_QmDAG

    def submarginals(self, **kwargs):
        return set(self._submarginals(**kwargs))

    def _unique_unlabelled_ids_obtainable_by_marginalization(self, **kwargs):
        return set(new_QmDAG.unique_unlabelled_id for new_QmDAG in self.submarginals(**kwargs))

    def unique_unlabelled_ids_obtainable_by_naive_marginalization(self, **kwargs):
        new_kwargs = kwargs.copy()
        new_kwargs['apply_teleportation'] = False
        return set(self._unique_unlabelled_ids_obtainable_by_marginalization(**new_kwargs))

    def unique_unlabelled_ids_obtainable_by_marginalization(self, **kwargs):
        new_kwargs = kwargs.copy()
        new_kwargs['apply_teleportation'] = True
        return set(self._unique_unlabelled_ids_obtainable_by_marginalization(**new_kwargs))

    def unique_unlabelled_ids_obtainable_by_reduction(self, **kwargs):
        subgraph_unlabelled_ids = set(self.unique_unlabelled_ids_obtainable_by_PD_trick)
        subgraph_unlabelled_ids.update(self.unique_unlabelled_ids_obtainable_by_conditioning)
        subgraph_unlabelled_ids.update(self.unique_unlabelled_ids_obtainable_by_marginalization(**kwargs))
        return subgraph_unlabelled_ids

    def assess_Fritz_Wolfe_style(self, target, set_of_visible_parents_to_delete, set_of_C_facets_to_delete, set_of_Q_facets_to_delete):
        visible_parents = set(np.flatnonzero(self.directed_structure_instance.as_bit_square_matrix[:, target]))
        set_of_C_facets_to_delete_drop_target = set(facet.difference({target}) for facet in set_of_C_facets_to_delete)
        set_of_Q_facets_to_delete_drop_target = set(facet.difference({target}) for facet in set_of_Q_facets_to_delete)
        set_of_visible_parents_to_keep = visible_parents.difference(set_of_visible_parents_to_delete)
        set_of_C_facets_to_keep = self.classical_sibling_sets_of(target).difference(set_of_C_facets_to_delete_drop_target)
        set_of_Q_facets_to_keep = self.quantum_sibling_sets_of(target).difference(set_of_Q_facets_to_delete_drop_target)
        candidate_Y = visible_parents.union(self.latent_siblings_of(target))
        candidate_Y.difference_update(self.directed_structure_instance.adjMat.descendantsplus_of(target))
        #CONDITION 1: Y must not be correlated with any dropped edges
        for facet in set_of_C_facets_to_delete_drop_target:
            candidate_Y.difference_update(
                self.directed_structure_instance.adjMat.descendantsplus_of(list(facet)))
            if len(candidate_Y) == 0:
                return [False, candidate_Y]
        for facet in set_of_Q_facets_to_delete_drop_target:
            candidate_Y.difference_update(
                self.directed_structure_instance.adjMat.descendantsplus_of(list(facet)))
            if len(candidate_Y)==0:
                return [False, candidate_Y]
        for common_cause_connected_set in self.as_mDAG.common_cause_connected_sets:
            if not common_cause_connected_set.isdisjoint(set_of_visible_parents_to_delete):
                candidate_Y.difference_update(common_cause_connected_set)
                if len(candidate_Y) == 0:
                    return [False, candidate_Y]
        #CONDITION 2: Y must be correlated with every kept edge
        for facet in set_of_C_facets_to_keep:
            if candidate_Y.isdisjoint(self.directed_structure_instance.adjMat.descendantsplus_of(list(facet))):
                return [False, candidate_Y]
        for facet in set_of_Q_facets_to_keep:
            if candidate_Y.isdisjoint(self.directed_structure_instance.adjMat.descendantsplus_of(list(facet))):
                return [False, candidate_Y]
        for visible_parent in set_of_visible_parents_to_keep:
            nodes_potentially_correlated_with_this_parent = set()
            for common_cause_connected_set in self.as_mDAG.common_cause_connected_sets:
                if visible_parent in common_cause_connected_set:
                    nodes_potentially_correlated_with_this_parent.update(common_cause_connected_set)
            if candidate_Y.isdisjoint(nodes_potentially_correlated_with_this_parent):
                return [False, candidate_Y]
        #If everything has worked as planned...
        return [True, candidate_Y]

    def apply_Fritz_trick(self, node_decomposition=True, safe_for_inference=True):
        new_directed_structure = [tuple(map(str, edge)) for edge in self.directed_structure_instance.edge_list]
        new_C_simplicial_complex = set(frozenset(map(str, h)) for h in self.C_simplicial_complex_instance.simplicial_complex_as_sets)
        new_Q_simplicial_complex = set(frozenset(map(str, h)) for h in self.Q_simplicial_complex_instance.simplicial_complex_as_sets)
        new_node_names = list(map(str, self.visible_nodes))
        allnode_name_variants = [{str(target)} for target in self.visible_nodes]
        specialcases_dict = dict()
        for target in self.visible_nodes:
            str_target = str(target)
            visible_parents = set(np.flatnonzero(self.directed_structure_instance.as_bit_square_matrix[:, target]))
            C_facets_with_target = set(
                facet for facet in self.C_simplicial_complex_instance.simplicial_complex_as_sets if target in facet)
            Q_facets_with_target = set(
                facet for facet in self.Q_simplicial_complex_instance.simplicial_complex_as_sets if target in facet)
            all_facets_with_target = C_facets_with_target.union(Q_facets_with_target)
            specialcases_dict[str_target] = (visible_parents,
                                             C_facets_with_target,
                                             Q_facets_with_target,
                                             all_facets_with_target,
                                             set())


        for target in self.visible_nodes:
            str_target = str(target)
            (visible_parents,
             C_facets_with_target,
             Q_facets_with_target,
             all_facets_with_target,
             emptyset) = specialcases_dict[str_target]
            # C_facets_with_target = hypergraph_canonicalize_with_deduplication(extended_C_facets_with_target)
            # Q_facets_with_target = hypergraph_canonicalize_with_deduplication(raw_Q_facets_with_target)

            visible_children = set(np.flatnonzero(self.directed_structure_instance.as_bit_square_matrix[target, :]))
            for set_of_visible_parents_to_delete in [set(subset) for i in range(0, len(visible_parents) + 1) for
                                                     subset in itertools.combinations(visible_parents, i)]:
                for set_of_C_facets_to_delete in [set(subset) for i in range(0, len(C_facets_with_target) + 1) for
                                                  subset in itertools.combinations(C_facets_with_target, i)]:
                    for set_of_Q_facets_to_delete in [set(subset) for i in range(0, len(Q_facets_with_target) + 1)
                                                      for subset in
                                                      itertools.combinations(Q_facets_with_target, i)]:
                        if not ((len(set_of_visible_parents_to_delete)==len(visible_parents)
                        ) and (len(set_of_C_facets_to_delete)==len(C_facets_with_target)
                        ) and (len(set_of_Q_facets_to_delete)==len(Q_facets_with_target)
                        )) and not ((len(set_of_visible_parents_to_delete)==0
                        ) and (len(set_of_C_facets_to_delete)==0
                        ) and (len(set_of_Q_facets_to_delete)==0
                        )):

                            Fritz_assessment = self.assess_Fritz_Wolfe_style(target, set_of_visible_parents_to_delete,
                                                          set_of_C_facets_to_delete, set_of_Q_facets_to_delete)
                            if Fritz_assessment[0]:

                                Y = Fritz_assessment[1]
                                # print(target, set_of_Q_facets_to_delete, Q_facets_with_target, Y)
                                sub_target = str_target + "_" + str(Y)
                                # new_directed_structure.append((sub_target, str_target))
                                set_of_C_facets_not_to_delete = C_facets_with_target.difference(set_of_C_facets_to_delete)
                                set_of_Q_facets_not_to_delete = Q_facets_with_target.difference(
                                    set_of_Q_facets_to_delete)
                                set_of_all_facets_not_to_delete = set_of_C_facets_not_to_delete.union(set_of_Q_facets_not_to_delete)
                                # set_of_C_facets_not_to_delete.update(set_of_Q_facets_not_to_delete)
                                visible_parents_not_to_delete = set(visible_parents).difference(
                                    set_of_visible_parents_to_delete)
                                specialcases_dict[sub_target] = (visible_parents_not_to_delete,
                                                                set_of_C_facets_not_to_delete,
                                                                set_of_Q_facets_to_delete,
                                                                set_of_all_facets_not_to_delete,
                                                                set(map(str, Y)))
                                # new_C_simplicial_complex.discard(frozenset(allnode_name_variants[target]))
                                allnode_name_variants[target].add(sub_target)
                                new_node_names.append(sub_target)
                                # new_C_simplicial_complex.add(frozenset(allnode_name_variants[target]))
                                for facet in set_of_all_facets_not_to_delete:
                                    new_cfacet = set(key
                                                     for key, val in
                                                     specialcases_dict.items() if
                                                     facet in val[3])
                                    new_cfacet.discard(sub_target)
                                    new_C_simplicial_complex.discard(frozenset(new_cfacet))
                                    new_cfacet.add(sub_target)
                                    new_C_simplicial_complex.add(frozenset(new_cfacet))

                                # for qfacet in raw_set_of_Q_facets_not_to_delete:
                                #     new_qfacet = set(key
                                #                      for key, val in
                                #                      specialcases_dict.items() if
                                #                      any(subqfacet.issuperset(qfacet) for
                                #                          subqfacet in val[2]))
                                #     new_qfacet.discard(sub_target)
                                #     new_Q_simplicial_complex.discard(frozenset(new_qfacet))
                                #     new_qfacet.add(sub_target)
                                #     new_Q_simplicial_complex.add(frozenset(new_qfacet))

                                for p in visible_parents_not_to_delete:
                                    for str_p in allnode_name_variants[p]:
                                        new_directed_structure.append((str_p, sub_target))
                                for c in visible_children:
                                    for str_c in allnode_name_variants[c]:
                                        if target in specialcases_dict[str_c][0]:
                                            new_directed_structure.append((sub_target, str_c))
            # allnode_name_variants.append(target_name_variants)
        new_C_simplicial_complex = hypergraph_full_cleanup(new_C_simplicial_complex)
        new_Q_simplicial_complex = hypergraph_full_cleanup(new_Q_simplicial_complex)
        core_nodes = tuple(map(str, self.visible_nodes))
        if not node_decomposition:
            for choice_of_nodes in itertools.product(*allnode_name_variants):
                if safe_for_inference:
                    nodes_to_get_rid_of = [specialcases_dict[name][4] for name in choice_of_nodes]
                    nodes_to_get_rid_of = set(itertools.chain.from_iterable(nodes_to_get_rid_of))
                    if nodes_to_get_rid_of.issubset(choice_of_nodes):
                        yield self.labelled_multi_marginalize(
                                                   nodes_to_get_rid_of,
                                                   choice_of_nodes,
                                                   new_directed_structure,
                                                   new_C_simplicial_complex,
                                                   new_Q_simplicial_complex)
                # choice_of_bonus_nodes = set(choice_of_nodes).difference(core_nodes)
                else:
                    coreQmDAG = QmDAG(
                        LabelledDirectedStructure(choice_of_nodes, new_directed_structure),
                        LabelledHypergraph(choice_of_nodes, new_C_simplicial_complex),
                        LabelledHypergraph(choice_of_nodes, new_Q_simplicial_complex))
                    yield coreQmDAG
                    # for to_fix_to_PD in itertools.chain.from_iterable(
                    #         itertools.combinations(choice_of_bonus_nodes, r) for r in range(1, self.number_of_visible-2)):
                    #     remaining_nodes = set(choice_of_nodes).difference(to_fix_to_PD)
                    #     if len(remaining_nodes)>=3:
                    #         yield coreQmDAG.subgraph(remaining_nodes)
        else:
            bonus_node_variants = [name_variants.difference(core_nodes) for name_variants in allnode_name_variants if
                                   len(name_variants) >= 2]
            bonus_node_variants = [name_variants.union({'-1'}) for name_variants in bonus_node_variants]
            # all_bonus_nodes = list(itertools.chain.from_iterable(bonus_node_variants))
            for bonus_nodes in itertools.product(*bonus_node_variants):
                actual_bonus_nodes = set(bonus_nodes).difference({'-1'})
                new_nodes = tuple(core_nodes) + tuple(actual_bonus_nodes)
                if len(actual_bonus_nodes) > 0:
                    if safe_for_inference:
                        nodes_to_get_rid_of = [specialcases_dict[name][4] for name in actual_bonus_nodes]
                        nodes_to_get_rid_of = set(itertools.chain.from_iterable(nodes_to_get_rid_of))
                        yield self.labelled_multi_marginalize(
                            nodes_to_get_rid_of,
                            new_nodes,
                            new_directed_structure,
                            new_C_simplicial_complex,
                            new_Q_simplicial_complex)
                    else:

                        yield QmDAG(
                            LabelledDirectedStructure(new_nodes, new_directed_structure),
                            LabelledHypergraph(new_nodes, new_C_simplicial_complex),
                            LabelledHypergraph(new_nodes, new_Q_simplicial_complex))

    def _unique_unlabelled_ids_obtainable_by_Fritz_for_QC(self, **kwargs):
        for new_QmDAG in self.apply_Fritz_trick(**kwargs):
            #if new_QmDAG.as_mDAG.fundamental_graphQ: #Elie: meant to preserve interestingness
            yield new_QmDAG.unique_unlabelled_id
            # for unlabelled_id in new_QmDAG.unique_unlabelled_ids_obtainable_by_reduction(districts_check=False, apply_teleportation=True):
            for unlabelled_id in new_QmDAG.unique_unlabelled_ids_obtainable_by_reduction(districts_check=False, apply_teleportation=True):
                yield unlabelled_id
                    
    def unique_unlabelled_ids_obtainable_by_Fritz_for_QC(self, **kwargs):
        return set(self._unique_unlabelled_ids_obtainable_by_Fritz_for_QC(**kwargs))

    def _unique_unlabelled_ids_obtainable_by_Fritz_for_IC(self, **kwargs):
        for new_QmDAG in self.apply_Fritz_trick(**kwargs):
            #if new_QmDAG.as_mDAG.fundamental_graphQ: #Elie: meant to preserve interestingness
            yield new_QmDAG.unique_unlabelled_id
            for unlabelled_id in new_QmDAG.unique_unlabelled_ids_obtainable_by_reduction(districts_check=True, apply_teleportation=False):
                yield unlabelled_id
                    
    def unique_unlabelled_ids_obtainable_by_Fritz_for_IC(self, **kwargs):
        return set(self._unique_unlabelled_ids_obtainable_by_Fritz_for_IC(**kwargs))
    




    #
    # def _unique_unlabelled_ids_obtainable_by_Fritz_without_node_splitting(self):
    #     for target in self.visible_nodes:
    #         visible_parents=set(np.flatnonzero(self.directed_structure_instance.as_bit_square_matrix[:, target]))
    #         C_facets_with_target = set(facet for facet in self.C_simplicial_complex_instance.simplicial_complex_as_sets if target in facet)
    #         Q_facets_with_target = set(facet for facet in self.Q_simplicial_complex_instance.simplicial_complex_as_sets if target in facet)
    #         for set_of_visible_parents_to_delete in [set(subset) for i in range(0, len(visible_parents) + 1) for subset in itertools.combinations(visible_parents, i)]:
    #             for set_of_C_facets_to_delete in [set(subset) for i in range(0, len(C_facets_with_target) + 1) for subset in itertools.combinations(C_facets_with_target, i)]:
    #                 for set_of_Q_facets_to_delete in [set(subset) for i in range(0, len(Q_facets_with_target) + 1) for subset in itertools.combinations(Q_facets_with_target, i)]:
    #                     Fritz_assessment = self.assess_Fritz_Wolfe_style(target,set_of_visible_parents_to_delete, set_of_C_facets_to_delete, set_of_Q_facets_to_delete)
    #                     if Fritz_assessment[0]:
    #                         Y = Fritz_assessment[1]
    #                         new_directed_structure = self.directed_structure_instance.as_set_of_tuples.copy()
    #                         new_directed_structure.difference_update((p, target) for p in set_of_visible_parents_to_delete)
    #
    #                         set_of_C_facets_to_delete_drop_target = set(facet.difference({target}) for facet in
    #                                                                  set_of_C_facets_to_delete if len(facet)>2)
    #                         set_of_Q_facets_to_delete_drop_target = set(facet.difference({target}) for facet in
    #                                                                  set_of_Q_facets_to_delete if len(facet)>2)
    #
    #                         #If the target is perfectly predicted, then it cannot utilize quantumness.
    #                         set_of_Q_facets_which_turn_classical = Q_facets_with_target.difference(set_of_Q_facets_to_delete)
    #                         remnants_of_Q_facets_which_turn_classical = set(facet.difference({target}) for facet in
    #                                                                  set_of_Q_facets_which_turn_classical if len(facet)>2)
    #
    #
    #                         new_C_simplicial_complex = self.C_simplicial_complex_instance.simplicial_complex_as_sets.copy()
    #                         new_C_simplicial_complex.difference_update(set_of_C_facets_to_delete)
    #                         new_C_simplicial_complex.update(set_of_C_facets_to_delete_drop_target)
    #                         new_C_simplicial_complex.update(set_of_Q_facets_which_turn_classical)
    #                         new_C_simplicial_complex = hypergraph_full_cleanup(new_C_simplicial_complex)
    #
    #                         new_Q_simplicial_complex = self.Q_simplicial_complex_instance.simplicial_complex_as_sets.copy()
    #                         new_Q_simplicial_complex.difference_update(set_of_Q_facets_to_delete)
    #                         new_Q_simplicial_complex.difference_update(set_of_Q_facets_which_turn_classical)
    #                         new_Q_simplicial_complex.update(set_of_Q_facets_to_delete_drop_target)
    #                         new_Q_simplicial_complex.update(remnants_of_Q_facets_which_turn_classical)
    #                         new_Q_simplicial_complex = hypergraph_full_cleanup(new_Q_simplicial_complex)
    #
    #                         new_QmDAG = QmDAG(
    #                                    DirectedStructure(new_directed_structure, self.number_of_visible),
    #                                    Hypergraph(new_C_simplicial_complex, self.number_of_visible),
    #                                    Hypergraph(new_Q_simplicial_complex, self.number_of_visible)
    #                                )
    #
    #                         yield (target,
    #                                frozenset(Y),
    #                                frozenset(set_of_visible_parents_to_delete),
    #                                frozenset(set_of_Q_facets_to_delete),
    #                                new_QmDAG.unique_unlabelled_id)
    #
    #                         subgraph_unlabelled_ids = set(new_QmDAG.unique_unlabelled_ids_obtainable_by_PD_trick)
    #                         subgraph_unlabelled_ids.update(new_QmDAG.unique_unlabelled_ids_obtainable_by_marginalization)
    #                         for unlabelled_id in subgraph_unlabelled_ids:
    #                             yield (target,
    #                                    frozenset(Y),
    #                                    frozenset(set_of_visible_parents_to_delete),
    #                                    frozenset(set_of_Q_facets_to_delete),
    #                                    unlabelled_id)
    #
    #
    # @cached_property
    # def unique_unlabelled_ids_obtainable_by_Fritz_without_node_splitting(self):
    #     return set(self._unique_unlabelled_ids_obtainable_by_Fritz_without_node_splitting())

if __name__ == '__main__': 
    boring_QmDAG = QmDAG(
        DirectedStructure([(0,1), (1,2), (1,3), (2,3)],4),
        Hypergraph([], 4),
        Hypergraph([(0,2),(0,3),(1,2),(1,3)],4)
    )

    assess = boring_QmDAG.assess_Fritz_Wolfe_style(target=3,
                                          set_of_visible_parents_to_delete={},
                                          set_of_C_facets_to_delete={},
                                          set_of_Q_facets_to_delete={frozenset({2,3})})
    print(assess)
    pass

# =============================================================================
#     QG = QmDAG(DirectedStructure([(0,3), (1,2)],4),Hypergraph([], 4),Hypergraph([(0,1),(1,3),(3,2),(2,0)],4))
#     for (n,dag) in QG.unique_unlabelled_ids_obtainable_by_Fritz_for_QC(node_decomposition=False):
#         if n in known_QC_Gaps_QmDAGs_ids:
#             print(n,dag)
#             break
#     dag.condition(0).unique_unlabelled_id
# =============================================================================
