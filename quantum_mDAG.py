from __future__ import absolute_import
import itertools
import warnings
import numpy as np
import numpy.typing as npt
from hypergraphs import Hypergraph, LabelledHypergraph, hypergraph_full_cleanup
from directed_structures import DirectedStructure, LabelledDirectedStructure
from mDAG_advanced import mDAG
from sys import version_info
assert version_info >= (3, 8), "Python 3.8+ is required for cached_property support."
from functools import total_ordering
from utilities import stringify_in_set as stringify
from typing import Optional, Any, Dict, Iterable, List, Set, Tuple
try:
    import networkx as nx
except ImportError:
    print("Functions which depend on networkx are not available.")

from functools import cached_property
from methodtools import lru_cache

BoolMatrix = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int_]


def C_facets_not_dominated_by_Q(c_facets: Set[frozenset], q_facets: Set[frozenset]) -> Set[frozenset]:
    c_facets_copy = c_facets.copy()
    for Q_facet in q_facets:
        dominated_by_quantum = set(filter(Q_facet.issuperset, c_facets_copy))
        c_facets_copy.difference_update(dominated_by_quantum)
    return c_facets_copy


def upgrade_to_QmDAG(mdag: mDAG) -> "QmDAG":
    return QmDAG(
        mdag.directed_structure_instance,
        Hypergraph([], mdag.number_of_visible),
        mdag.simplicial_complex_instance)

def as_classical_QmDAG(mdag: mDAG) -> "QmDAG":
    return QmDAG(
        mdag.directed_structure_instance,
        mdag.simplicial_complex_instance,
        Hypergraph([], mdag.number_of_visible))


_UNLABELLED_ID_MEMO: Dict[Tuple, Tuple[int, int, int, int]] = dict()
# Children of a structure under the piggybacks, keyed by its unlabelled id and the search options. All piggybacks
# are label-equivariant, so the children of a relabelled structure are relabellings of these.
_PIGGYBACK_CHILDREN_MEMO: Dict[Tuple, List["QmDAG"]] = dict()
# Entropic LPs (Shannon cone + local Markov equalities) per labelled structure; each holds a solver task.
_ENTROPIC_LP_CACHE: Dict[Tuple, Any] = dict()
_ENTROPIC_LP_CACHE_SIZE = 64
# Elementary d-separation models (semigraphoid.dsep_all of lp_structure) per labelled structure.
_SEMIGRAPHOID_CACHE: Dict[Tuple, Any] = dict()
_SEMIGRAPHOID_CACHE_SIZE = 64
# The certificate engine for Fritz steps that d-separation does not certify: 'semigraphoid' (the closure of
# semigraphoid.py, fast and always available), 'lp' (the entropic LP, needs mosek) or 'both' (closure first, LP
# where it fails, disagreements recorded in ENGINE_DISAGREEMENTS). Manuscript 7.9.
DEFAULT_ENGINE = 'semigraphoid'
ENGINE_DISAGREEMENTS: List[Tuple] = []
# Outcome tally of the Fritz certificates (QmDAG.fritz_certificate), keyed by ('certificate', outcome) with outcome
# 'vacuous' (every predictor a parent of the target), 'dsep', 'relabel', 'markov' or 'failed' (the engine certified
# nothing); ('engine', name) counts which engine certified, ('disagreement', kind) the engine='both' disagreements.
ENTROPIC_STATS: Dict[Tuple[str, str], int] = dict()


def _tally(kind: str, outcome: str) -> None:
    ENTROPIC_STATS[(kind, outcome)] = ENTROPIC_STATS.get((kind, outcome), 0) + 1


# This class does NOT represent every possible quantum causal structure. It only represents the causal structures where every quantum latent is exogenized. This is the case, for example, of the known QC Gaps.
@total_ordering
class QmDAG:
    def __init__(self, directed_structure_instance: DirectedStructure, C_simplicial_complex_instance: Hypergraph,
                 Q_simplicial_complex_instance: Hypergraph) -> None:
        self.directed_structure_instance = directed_structure_instance
        self.number_of_visible = self.directed_structure_instance.number_of_visible
        assert directed_structure_instance.number_of_visible == C_simplicial_complex_instance.number_of_visible, 'Different number of nodes in directed structure vs classical simplicial complex.'
        assert directed_structure_instance.number_of_visible == Q_simplicial_complex_instance.number_of_visible, 'Different number of nodes in directed structure vs quantum simplicial complex.'

        self.Q_simplicial_complex_instance = Q_simplicial_complex_instance
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
                                                self.Q_simplicial_complex_instance.number_of_visible_plus_nonsingleton_latent
                                                + self.C_simplicial_complex_instance.number_of_visible_plus_latent
                                                - self.number_of_visible))
        self.vis_nodes_with_no_children = set(self.directed_structure_instance.nodes_with_no_children)

    @cached_property
    def exogenous_visible_nodes(self) -> Set[int]:
        """Visible nodes with no visible parents and no (nonsingleton) latent parents, classical or quantum."""
        return set(self.directed_structure_instance.nodes_with_no_parents).intersection(
            self.C_simplicial_complex_instance.vis_nodes_with_singleton_latent_parents,
            self.Q_simplicial_complex_instance.vis_nodes_with_singleton_latent_parents)

    @cached_property
    def as_string(self) -> str:
        return 'Children'.ljust(10) + ': ' + self.directed_structure_instance.as_string \
               + '\nClassical'.ljust(11) + ': ' + self.C_simplicial_complex_instance.as_string \
               + '\nQuantum'.ljust(11) + ': ' + self.Q_simplicial_complex_instance.as_string + '\n'

    def __str__(self) -> str:
        return self.as_string

    def __repr__(self) -> str:
        return self.as_string

    #@cached_property
    @property
    def unique_id(self) -> Tuple[int, int, int, int]:
        # Returns a unique identification tuple.
        return (
            self.number_of_visible,
            self.directed_structure_instance.as_integer,
            self.C_simplicial_complex_instance.as_integer,
            self.Q_simplicial_complex_instance.as_integer)
    def __hash__(self) -> int:
        return hash(self.unique_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QmDAG):
            return False
        return self.unique_id == other.unique_id

    def __lt__(self, other: "QmDAG") -> bool:
        return self.unique_id < other.unique_id

    @cached_property
    def unique_unlabelled_id(self) -> Tuple[int, int, int, int]:
        # Returns a unique identification tuple up to relabelling.
        # Memoized across equal-but-distinct objects, since the search constructs many copies.
        key = self.unique_id
        try:
            return _UNLABELLED_ID_MEMO[key]
        except KeyError:
            unlabelled_id = (self.number_of_visible,) + min(zip(
                self.directed_structure_instance.as_integer_permutations,
                self.C_simplicial_complex_instance.as_integer_permutations,
                self.Q_simplicial_complex_instance.as_integer_permutations
            ))
            _UNLABELLED_ID_MEMO[key] = unlabelled_id
            return unlabelled_id

    #ON THE POINT DISTRIBUTION TRICK

    def subgraph(self, list_of_nodes: Tuple[int, ...]) -> "QmDAG":
        return QmDAG(
            LabelledDirectedStructure(list_of_nodes, self.directed_structure_instance.edge_list),
            LabelledHypergraph(list_of_nodes, self.C_simplicial_complex_instance.simplicial_complex_as_sets),
            LabelledHypergraph(list_of_nodes, self.Q_simplicial_complex_instance.simplicial_complex_as_sets),
        )

    def fix_to_point_distribution_QmDAG(self, node: int) -> "QmDAG":  # returns a smaller QmDAG
        return self.subgraph(self.visible_nodes[:node] + self.visible_nodes[(node + 1):])

    def _subgraphs_generator(self) -> Iterable["QmDAG"]:
        for r in range(3, self.number_of_visible):
            for to_keep in itertools.combinations(self.visible_nodes, r):
                yield self.subgraph(to_keep)

    @cached_property
    def subgraphs(self) -> Set["QmDAG"]:
        return {self.subgraph(to_keep) for to_keep in itertools.combinations(self.visible_nodes, self.number_of_visible-1)}
        # return set(self._subgraphs_generator())

    @cached_property
    def unique_unlabelled_ids_obtainable_by_PD_trick(self) -> Set[Tuple[int, int, int, int]]:
        return set(subQmDAG.unique_unlabelled_id for subQmDAG in self.subgraphs)

    # ON THE MARGINALIZATION TRICK

    def classical_sibling_sets_of(self, node: int) -> Set[frozenset]:
        return set(facet.difference({node}) for facet in self.C_simplicial_complex_instance.simplicial_complex_as_sets if
                node in facet)

    def quantum_sibling_sets_of(self, node: int) -> Set[frozenset]:
        return set(facet.difference({node}) for facet in self.Q_simplicial_complex_instance.simplicial_complex_as_sets if
                node in facet)

    def quantum_siblings_of(self, node: int) -> Set[Any]:
        return set(itertools.chain.from_iterable(self.quantum_sibling_sets_of(node)))

    @cached_property
    def as_mDAG(self) -> mDAG:
        return mDAG(
            self.directed_structure_instance,
            Hypergraph(C_facets_not_dominated_by_Q(
                self.C_simplicial_complex_instance.simplicial_complex_as_sets,
                self.Q_simplicial_complex_instance.simplicial_complex_as_sets
            ).union(
                self.Q_simplicial_complex_instance.simplicial_complex_as_sets
            ), self.number_of_visible)
        )

    @cached_property
    def as_graph(self):
        return self.as_mDAG.as_graph

    def latent_sibling_sets_of(self, node: int) -> Set[frozenset]:
        return set(facet.difference({node}) for facet in self.as_mDAG.simplicial_complex_instance.simplicial_complex_as_sets if
                node in facet)

    def latent_siblings_of(self, node: int) -> Set[Any]:
        return set(itertools.chain.from_iterable(self.latent_sibling_sets_of(node)))
    
    def has_grandparents_that_are_not_parents(self, node: int) -> bool:
        visible_parents = set(np.flatnonzero(self.directed_structure_instance.as_bit_square_matrix[:, node]))
        for parent in visible_parents:
            grandparents=set(np.flatnonzero(self.directed_structure_instance.as_bit_square_matrix[:, parent]))
            if not grandparents.issubset(visible_parents):
                return True
        return False
    
    def parents_have_external_latents(self, node: int) -> bool:
        """True if some visible parent of `node` lies in a latent facet (classical or quantum) that neither contains
        `node` nor lies inside the set of visible parents of `node`. Conditioning on `node` then correlates that
        facet's outside children in a way the output structure cannot express, so the conditioning piggyback is
        not justified. (A facet contained in the parent block only redistributes the block and is harmless.)"""
        visible_parents = set(np.flatnonzero(self.directed_structure_instance.as_bit_square_matrix[:, node]))
        for facet in self.as_mDAG.simplicial_complex_instance.simplicial_complex_as_sets:
            if node not in facet and not facet.isdisjoint(visible_parents) and not facet.issubset(visible_parents):
                return True
        return False

    def conditioning_is_justified(self, node: int, strict_latents: bool = True) -> bool:
        """Three conditions. (1) Every visible grandparent of `node` is a parent of `node`. (2, strict_latents)
        every facet containing a visible parent of `node` contains `node` or lies within the parents. (3) A visible
        parent that shares no facet with `node` has to receive the post-selected common cause through its own
        output (it guesses it, and `node` post-selects on the guess); its output is then fine-grained, which is
        only harmless if every visible child of that parent is `node`, another parent, or a latent sibling of
        `node` (nodes that can read the common cause from the new facet). Then the block of parents is closed
        under all its inputs and post-selecting on `node` can be absorbed into one common cause of parents and
        latent siblings."""
        if self.has_grandparents_that_are_not_parents(node):
            return False
        if strict_latents and self.parents_have_external_latents(node):
            return False
        parents = set(np.flatnonzero(self.directed_structure_instance.as_bit_square_matrix[:, node]))
        siblings = set(self.latent_siblings_of(node))
        allowed_children = parents | siblings | {node}
        for p in parents:
            if p in siblings:
                continue
            children = set(np.flatnonzero(self.directed_structure_instance.as_bit_square_matrix[p]))
            if not children.issubset(allowed_children):
                return False
        return True

    def condition(self, node: int) -> "QmDAG":
        #assume we already checked that conditioning is justified (conditioning_is_justified)
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

    def _subconditionals(self) -> Iterable["QmDAG"]:
        if self.number_of_visible > 3:
            for node in self.visible_nodes:
                if self.conditioning_is_justified(node):
                    conditional_QM = self.condition(node)
                    yield conditional_QM
                    for new_QmDAG in conditional_QM.subconditionals:
                        yield new_QmDAG

    @cached_property
    def subconditionals(self) -> Set["QmDAG"]:
        return set(self._subconditionals())
    
    @cached_property
    def unique_unlabelled_ids_obtainable_by_conditioning(self) -> Set[Tuple[int, int, int, int]]:
        return set(new_QmDAG.unique_unlabelled_id for new_QmDAG in self.subconditionals)


    def node_stitching(self, node_with_no_children: int, node_with_no_parents: int) -> "QmDAG":
        """We make the children of the node with no parents into the children of the node that previously had no children, and then we remove the node with no parents."""
        remaining_nodes = self.visible_nodes[:node_with_no_parents] + self.visible_nodes[(node_with_no_parents + 1):]
        new_directed_edges = set(self.directed_structure_instance.edge_list)
        for edge in self.directed_structure_instance.edge_list:
            if edge[0] == node_with_no_parents:
                new_directed_edges.add((node_with_no_children, edge[1]))
                new_directed_edges.remove(edge)
        return QmDAG(
                LabelledDirectedStructure(remaining_nodes, list(new_directed_edges)),
                LabelledHypergraph(remaining_nodes, self.C_simplicial_complex_instance.simplicial_complex_as_sets),
                LabelledHypergraph(remaining_nodes, self.Q_simplicial_complex_instance.simplicial_complex_as_sets),
                )


    # The forward map stitches the exogenous node onto the sink; its inverse interrupts a node. Old name kept.
    interruption_creation = node_stitching

    def _subinterruptions(self) -> Set["QmDAG"]:
        for node_with_no_children in self.vis_nodes_with_no_children:
            for node_with_no_parents in self.exogenous_visible_nodes:
                if node_with_no_children in self.directed_structure_instance.adjMat.descendantsplus_of(node_with_no_parents):
                    continue  # we don't want to create a cycle
                yield self.interruption_creation(node_with_no_children, node_with_no_parents)
    
    @cached_property
    def subinterruptions(self) -> Set["QmDAG"]:
        return set(self._subinterruptions())

    @cached_property
    def unique_unlabelled_ids_obtainable_by_interruption(self) -> Set[Tuple[int, int, int, int]]:
        return set(new_QmDAG.unique_unlabelled_id for new_QmDAG in self.subinterruptions)




    def marginalize(self, node: int, districts_check: bool = False, apply_teleportation: bool = True) -> "QmDAG":  # returns a smaller QmDAG
        remaining_nodes = self.visible_nodes[:node] + self.visible_nodes[(node + 1):]
        # Pass visible children on to visible children
        # Pass latent children on to visible children **classically**
        # Apply teleportation
        visible_children = set(np.flatnonzero(self.directed_structure_instance.as_bit_square_matrix[node]))
        visible_parents = set(np.flatnonzero(self.directed_structure_instance.as_bit_square_matrix[:, node]))
        new_directed_edges = set(self.directed_structure_instance.edge_list)
        for parent in visible_parents:
            for child in visible_children:
                new_directed_edges.add((parent, child))
        new_C_facets = self.C_simplicial_complex_instance.simplicial_complex_as_sets.copy()
        C_facets_to_grow = self.latent_sibling_sets_of(node)
        if len(C_facets_to_grow)>0:
            for C_facet_to_grow in C_facets_to_grow:
                new_C_facets.add(C_facet_to_grow.union(visible_children))
        else:
            new_C_facets.add(frozenset(visible_children))
        new_C_facets = hypergraph_full_cleanup(new_C_facets)
        new_C_simplicial_complex = LabelledHypergraph(remaining_nodes, new_C_facets)
        if not districts_check:
            ok_to_proceed = True
        else:
            previous_districts = [district.difference({node}) for district in self.as_mDAG.numerical_districts]
            new_districts = new_C_simplicial_complex.translated_districts
            ok_to_proceed = frozenset(map(frozenset, previous_districts)) == frozenset(map(frozenset, new_districts))
        if not ok_to_proceed:
            return None  # the marginalization trick does not apply when districts are not preserved
        else:
            if not apply_teleportation:
                return QmDAG(
                    LabelledDirectedStructure(remaining_nodes, list(new_directed_edges)),
                    LabelledHypergraph(remaining_nodes, new_C_facets),
                    LabelledHypergraph(remaining_nodes, self.Q_simplicial_complex_instance.simplicial_complex_as_sets),
                    )
            else:
                teleportable_children = self.quantum_siblings_of(node).intersection(visible_children)
                new_Q_facets = self.Q_simplicial_complex_instance.simplicial_complex_as_sets.copy()
                Q_facets_to_grow = self.quantum_sibling_sets_of(node)
                for Q_facet_to_grow in Q_facets_to_grow:
                    new_Q_facets.add(Q_facet_to_grow.union(teleportable_children))
                new_Q_facets = hypergraph_full_cleanup(new_Q_facets)
                return QmDAG(
                    LabelledDirectedStructure(remaining_nodes, list(new_directed_edges)),
                    LabelledHypergraph(remaining_nodes, new_C_facets),
                    LabelledHypergraph(remaining_nodes, new_Q_facets),
                    )


    def _submarginals(self, **kwargs):
        if self.number_of_visible > 3:
            for node in self.visible_nodes:
                marginalized_QM = self.marginalize(node, **kwargs)
                if marginalized_QM is None:
                    continue
                yield marginalized_QM
                for new_QmDAG in marginalized_QM.submarginals(**kwargs):
                    yield new_QmDAG

    def submarginals(self, **kwargs):
        return set(self._submarginals(**kwargs))

    def _unique_unlabelled_ids_obtainable_by_marginalization(self, **kwargs):
        return set(new_QmDAG.unique_unlabelled_id for new_QmDAG in self.submarginals(**kwargs))

    @lru_cache(maxsize=None)
    def unique_unlabelled_ids_obtainable_by_naive_marginalization(self, **kwargs):
        new_kwargs = kwargs.copy()
        new_kwargs['apply_teleportation'] = False
        return set(self._unique_unlabelled_ids_obtainable_by_marginalization(**new_kwargs))

    @lru_cache(maxsize=None)
    def unique_unlabelled_ids_obtainable_by_marginalization(self, **kwargs):
        new_kwargs = kwargs.copy()
        new_kwargs['apply_teleportation'] = True
        return set(self._unique_unlabelled_ids_obtainable_by_marginalization(**new_kwargs))

    def unique_unlabelled_ids_obtainable_by_reduction(self, **kwargs):
        subgraph_unlabelled_ids = set(self.unique_unlabelled_ids_obtainable_by_PD_trick)
        subgraph_unlabelled_ids.update(self.unique_unlabelled_ids_obtainable_by_conditioning)
        subgraph_unlabelled_ids.update(self.unique_unlabelled_ids_obtainable_by_marginalization(**kwargs))
        subgraph_unlabelled_ids.update(self.unique_unlabelled_ids_obtainable_by_interruption)
        return subgraph_unlabelled_ids

    # ------------------------------------------------------------------
    # THE FRITZ PIGGYBACK
    #
    # Let X1 be a set of visible nodes (the predictors, jointly) and s a visible node sharing a latent with some
    # member of X1 (a candidate predicted node). In the effective DAG (visible nodes, one node per latent facet, one
    # private-noise node per visible node) split the parents of s into common(s), those also seen by X1 (parents of
    # some predictor, or predictors themselves), and others(s). If X1 is d-separated from others(s) given common(s),
    # then:
    #   * classically, any model in which a subvariable of X1 is perfectly correlated with s (so that X1 perfectly
    #     predicts s, H(s | X1) = 0) can be rewritten so that s depends on common(s) only;
    #   * quantumly, any strategy for the reduced structure in which s is a deterministic function of its (classical)
    #     parents extends to the original structure with X1 outputting a copy of s.
    # Hence the structure G' obtained by deleting X1 and restricting s to common(s) satisfies: a QC gap in G' implies a
    # QC gap in G, with no caveat about perfect correlations in G'. Quantum facets read by s become classical for s
    # (the other children of the facet keep its quantum part).
    # In "copy" mode s is left untouched and a fresh node s_copy carrying the common (classical) part is added; this is
    # the node-splitting version of the trick (e.g. triangle -> Bell).
    # Predictors with children cannot simply be deleted: they are removed by the marginalization piggyback instead
    # (relaying their parents to their children, teleporting quantum shares when allowed) while still outputting the
    # copies of the predicted nodes, so the composition is sound whenever marginalization is. Since teleportation makes
    # marginalization order-dependent, every order of removing the predictors is enumerated.
    # ------------------------------------------------------------------

    @cached_property
    def effective_DAG_data(self) -> Tuple["nx.DiGraph", Dict[int, Tuple[str, frozenset]]]:
        """Directed graph on visible nodes plus one node per classical facet, quantum facet and private noise source.
        Returns the graph and a dict latent_index -> (kind, frozenset of visible children)."""
        n = self.number_of_visible
        edges = set(self.directed_structure_instance.as_set_of_tuples)
        latent_nodes: Dict[int, Tuple[str, frozenset]] = dict()
        idx = n
        for kind, hypergraph in (('C', self.C_simplicial_complex_instance), ('Q', self.Q_simplicial_complex_instance)):
            for facet in sorted(map(tuple, map(sorted, hypergraph.simplicial_complex_as_sets))):
                latent_nodes[idx] = (kind, frozenset(facet))
                edges.update((idx, v) for v in facet)
                idx += 1
        for v in range(n):
            latent_nodes[idx] = ('noise', frozenset({v}))
            edges.add((idx, v))
            idx += 1
        g = nx.DiGraph()
        g.add_nodes_from(range(idx))
        g.add_edges_from(edges)
        return g, latent_nodes

    # ------------------------------------------------------------------
    # THE ENTROPIC CERTIFICATE (Khanna, Pusey and Colbeck; manuscript Section 7)
    #
    # Used by fritz_certificate where d-separation fails. Hypotheses: Shannon inequalities over all nodes of G
    # (latent facets as variables, no explicit noise), the local Markov equalities of G, perfect prediction
    # H(s | X) = 0 (the one direction of the perfect correlation between a subvariable of X and s that the lift
    # arranges), and the elementary conditional independences among visible nodes that hold by d-separation in the
    # candidate (they hold for free in the quantum lift, since the lifted distribution is Markov to the candidate,
    # yet are genuine extra hypotheses classically). Two sound target sets:
    #   'relabel' : when s keeps a single latent facet L, the Markov equalities of G'' = candidate with L deleted
    #               and s made a parent of L's other children (a G''-model gives a candidate model by L := s);
    #   'markov'  : the local Markov equalities of the candidate over G's own latents (off by default: it never
    #               decided a census input).
    # ------------------------------------------------------------------

    @cached_property
    def lp_structure(self) -> Tuple[Tuple[int, ...], Dict[int, frozenset]]:
        """Nodes (visible nodes, then latent facets; effective-DAG indices, noise sources excluded) and their
        parent sets."""
        g, latent_nodes = self.effective_DAG_data
        nodes = tuple(v for v in sorted(g.nodes) if v < self.number_of_visible or latent_nodes[v][0] != 'noise')
        node_set = set(nodes)
        parents = {v: frozenset(p for p in g.predecessors(v) if p in node_set) for v in nodes}
        return nodes, parents

    def _entropic_lp(self):
        """The Shannon cone plus the local Markov equalities of this structure (cached per labelled id)."""
        from entropic_lp import EntropicLP, local_markov_rows
        key = self.unique_id
        lp = _ENTROPIC_LP_CACHE.get(key)
        if lp is None:
            nodes, parents = self.lp_structure
            lp = EntropicLP(len(nodes), [row for _, row in local_markov_rows(parents, nodes)])
            if len(_ENTROPIC_LP_CACHE) >= _ENTROPIC_LP_CACHE_SIZE:
                _ENTROPIC_LP_CACHE.pop(next(iter(_ENTROPIC_LP_CACHE))).close()
            _ENTROPIC_LP_CACHE[key] = lp
        return lp

    def _entropic_hypotheses(self, predictors: frozenset, kept_parents: Dict[int, frozenset],
                             predicted: Iterable[int]) -> list:
        from entropic_lp import cond_entropy_row, observable_dseparation_rows
        nodes, _ = self.lp_structure
        rows = [cond_entropy_row([s], predictors) for s in predicted]
        rows += observable_dseparation_rows(kept_parents, nodes, self.visible_nodes)
        return rows

    def _entropic_certificate(self, predictors: frozenset, kept_parents: Dict[int, frozenset],
                              predicted: Tuple[int, ...], try_markov: bool = True):
        """Returns 'relabel', 'markov' or None: whether the LP certifies that a classical model of G in which the
        predictors perfectly predict the predicted nodes (H(s | X) = 0, the one direction of the perfect correlation
        the lift arranges) yields a classical model of the candidate. Each target set
        is one LP (the sum of its Markov rows, see EntropicLP.implies_all). With try_markov=False only the
        `relabel` target set is tried (the census does this: `markov` never decided an input)."""
        from entropic_lp import local_markov_rows
        nodes, parents = self.lp_structure
        lp = self._entropic_lp()
        handle = lp.push_hypotheses(self._entropic_hypotheses(predictors, kept_parents, predicted))
        try:
            # 'relabel' is tried first (it is the certificate that reaches beyond d-separation in practice), 'markov'
            # only if 'relabel' is inapplicable or fails.
            if len(predicted) == 1:
                s = predicted[0]
                common = kept_parents[s]
                if len(common) == 1 and min(common) >= self.number_of_visible:
                    lam = min(common)
                    relabelled = {v: (ps - {lam}) | {s} if lam in ps else ps
                                  for v, ps in kept_parents.items() if v != lam}
                    relabelled[s] = frozenset()
                    nodes_c = [v for v in nodes if v != lam]
                    if lp.implies_all(row for _, row in local_markov_rows(relabelled, nodes_c)):
                        return 'relabel'
            if try_markov and lp.implies_all(row for _, row in local_markov_rows(kept_parents, nodes)):
                return 'markov'
            return None
        finally:
            lp.pop_to(handle)

    def _semigraphoid_model(self):
        """The elementary d-separation model of lp_structure (noise excluded), cached per labelled structure."""
        import semigraphoid as sg
        key = self.unique_id
        model = _SEMIGRAPHOID_CACHE.get(key)
        if model is None:
            nodes, parents = self.lp_structure
            n = len(nodes)
            assert nodes == tuple(range(n)), "lp_structure indices are 0..n-1"
            model = sg.dsep_all(n, sg.parents_to_masks(parents, n))
            if len(_SEMIGRAPHOID_CACHE) >= _SEMIGRAPHOID_CACHE_SIZE:
                _SEMIGRAPHOID_CACHE.pop(next(iter(_SEMIGRAPHOID_CACHE)))
            _SEMIGRAPHOID_CACHE[key] = model
        return model

    def _semigraphoid_certificate(self, predictors: frozenset, kept_parents: Dict[int, frozenset],
                                  predicted: Tuple[int, ...], try_markov: bool = True):
        """The semigraphoid counterpart of _entropic_certificate (manuscript 7.9): the same hypotheses (the
        d-separations of G, the perfect predictions H(s|X) = 0 as elementary triplets, the observable d-separations
        of the candidate) closed under the exchange rule, and the same target sets, each checked as the containment
        of a d-separation model: `relabel` is the d-separation model of the relabelled candidate (the kept facet
        replaced by s) restricted to the remaining variables, `markov` that of the candidate itself. Returns
        'relabel', 'markov' or None. Sound because every semigraphoid step is a Shannon-type identity."""
        import semigraphoid as sg
        nodes, parents = self.lp_structure
        n = len(nodes)
        nv = self.number_of_visible
        E = self._semigraphoid_model().copy()
        X = sg.mask_of(predictors)
        for s in predicted:
            sg.add_functional_dependence(E, s, X)
        candidate = sg.dsep_all(n, sg.parents_to_masks(kept_parents, n))
        block = 1 << nv   # masks below 2^nv are exactly the subsets of the visible nodes
        E[:nv, :nv, :block] |= candidate[:nv, :nv, :block]
        sg.close(E)
        if len(predicted) == 1:
            s = predicted[0]
            common = kept_parents[s]
            if len(common) == 1 and min(common) >= nv:
                lam = min(common)
                relabelled = {v: (ps - {lam}) | {s} if lam in ps else ps
                              for v, ps in kept_parents.items() if v != lam}
                relabelled[s] = frozenset()
                relabelled[lam] = frozenset()   # lam is isolated and then ignored
                target = sg.restrict(sg.dsep_all(n, sg.parents_to_masks(relabelled, n)), ((1 << n) - 1) & ~(1 << lam))
                if sg.contains(E, target):
                    return 'relabel'
        if try_markov and sg.contains(E, candidate):
            return 'markov'
        return None

    def split_node(self, node: int) -> "QmDAG":
        """Node splitting piggyback: `node` is replaced by itself and a copy (appended as the last visible node)
        with the same parents and children and a shared classical two-party latent (their common private noise).
        A QC gap in the split structure implies one in the original (the original node outputs the pair)."""
        n = self.number_of_visible
        copy = n
        edges = set(self.directed_structure_instance.as_set_of_tuples)
        for (a, b) in self.directed_structure_instance.as_set_of_tuples:
            if a == node:
                edges.add((copy, b))
            if b == node:
                edges.add((a, copy))
        C_facets = {f | {copy} if node in f else f for f in self.C_simplicial_complex_instance.simplicial_complex_as_sets}
        C_facets.add(frozenset({node, copy}))
        Q_facets = {f | {copy} if node in f else f for f in self.Q_simplicial_complex_instance.simplicial_complex_as_sets}
        return QmDAG(DirectedStructure(sorted(edges), n + 1), Hypergraph(hypergraph_full_cleanup(C_facets), n + 1),
                     Hypergraph(hypergraph_full_cleanup(Q_facets), n + 1))

    # ------------------------------------------------------------------
    # PIGGYBACKS AS THE SEARCH APPLIES THEM
    #
    # Every transformation of the search is a generator of (params, child) pairs on this class, so that a piggyback
    # can be read, and debugged, in one place. The elementary reductions wrap the primitives above; the Fritz
    # piggyback is written predictor-first in six short methods (fritz_pool, fritz_targets, fritz_deletion,
    # fritz_certificate, fritz_realise, fritz_steps): pick a predictor set X, try its candidate targets s in order;
    # at each target X dictates the deletion D (the parents of s that X cannot see), which is justified by
    # d-separation or by the entropic LP, then realised by surgery, singly or for several targets of X at once
    # (manuscript Section 8).
    # Indices are effective-DAG indices (effective_DAG_data): visible nodes, then facets, then noise sources.
    # ------------------------------------------------------------------

    def pd_steps(self) -> Iterable[Tuple[Tuple, "QmDAG"]]:
        """Point distribution: fix one visible node and drop it (Section 2 of the manuscript)."""
        if self.number_of_visible <= 3:
            return
        for node in self.visible_nodes:
            yield (('drop', node),), self.fix_to_point_distribution_QmDAG(node)

    def node_stitching_steps(self) -> Iterable[Tuple[Tuple, "QmDAG"]]:
        """Stitch an exogenous node onto a childless sink by post-selecting on their equality (Section 5); the
        inverse map interrupts a node, hence the old name 'interruption'."""
        if self.number_of_visible <= 3:
            return
        for sink in sorted(self.vis_nodes_with_no_children):
            for source in sorted(self.exogenous_visible_nodes):
                if sink in self.directed_structure_instance.adjMat.descendantsplus_of(source):
                    continue
                yield ((('sink', int(sink)), ('source', int(source))),), self.node_stitching(sink, source)

    def conditioning_steps(self, strict_latents: bool = True) -> Iterable[Tuple[Tuple, "QmDAG"]]:
        """Condition on one visible node where conditioning_is_justified (Section 4)."""
        if self.number_of_visible <= 3:
            return
        for node in self.visible_nodes:
            if self.conditioning_is_justified(node, strict_latents=strict_latents):
                yield (('condition', node),), self.condition(node)

    def marginalization_steps(self, apply_teleportation: bool = True,
                              districts_check: bool = False) -> Iterable[Tuple[Tuple, "QmDAG"]]:
        """Marginalize one visible node, naively or with teleportation (Section 3)."""
        if self.number_of_visible <= 3:
            return
        for node in self.visible_nodes:
            child = self.marginalize(node, districts_check=districts_check, apply_teleportation=apply_teleportation)
            if child is not None:
                yield (('marginalize', node),), child

    def degradation_steps(self) -> List[Tuple[Tuple, "QmDAG"]]:
        """Quantum source to classical source (manuscript 0.4): every structure obtained by making a nonempty set
        of quantum facets classical. A gap in any of them is a gap here, since the classical model sets coincide
        and the quantum set only shrinks; a quantum facet made classical inside a classical facet is absorbed. The
        search applies this as a lookup: the results are registered but never expanded. params:
        (('classical', (facet, ...)),)."""
        Q_facets = sorted(sorted(f) for f in self.Q_simplicial_complex_instance.simplicial_complex_as_sets)
        C_facets = [tuple(sorted(f)) for f in self.C_simplicial_complex_instance.simplicial_complex_as_sets]
        n = self.number_of_visible
        results = []
        for r in range(1, len(Q_facets) + 1):
            for chosen in itertools.combinations(Q_facets, r):
                chosen = tuple(tuple(f) for f in chosen)
                remaining_Q = [f for f in Q_facets if tuple(f) not in chosen]
                if not remaining_Q:
                    continue   # a structure without quantum facets has no QC gap: useless as a lookup target
                new_C = Hypergraph(hypergraph_full_cleanup(set(map(frozenset, C_facets + list(chosen)))), n)
                results.append(((('classical', chosen),), QmDAG(self.directed_structure_instance, new_C, Hypergraph(remaining_Q, n))))
        return results

    # -- the Fritz piggyback, predictor-first (manuscript Sections 6 to 8) ---------------------------------------

    def fritz_pool(self, s: int, pool: str = 'siblings', allow_descendants: bool = False) -> List[int]:
        """Candidate predictors of the target s, in the order the search tries them. The lift must hand s its private
        randomness through a channel the predictor also sees: a facet shared with s (latent sibling) or the edge
        X -> s (visible parent, pool='siblings+parents'); nothing else is sound. Descendants of s are removed unless
        allow_descendants (the d-separation test always fails for them, since s's noise reaches them through s; the LP
        is sound either way). Within each group, nodes sharing more facets with s come first."""
        assert pool in ('siblings', 'siblings+parents'), pool
        g, latent_nodes = self.effective_DAG_data
        n = self.number_of_visible
        descendants = set(nx.descendants(g, s))
        facets_with_s = [members for idx, (kind, members) in latent_nodes.items() if kind != 'noise' and s in members]

        def shared(x: int) -> int:
            return sum(1 for members in facets_with_s if x in members)

        def ordered(nodes: Iterable[int]) -> List[int]:
            return sorted((x for x in nodes if x != s and (allow_descendants or x not in descendants)),
                          key=lambda x: (-shared(x), x))
        siblings = ordered(self.latent_siblings_of(s))
        if pool == 'siblings':
            return siblings
        parents = ordered(p for p in g.predecessors(s) if p < n and p not in siblings)
        return siblings + parents

    def fritz_deletion(self, s: int, X: Iterable[int]) -> Optional[Tuple[frozenset, frozenset]]:
        """The deletion the predictor set X dictates at the target s: K = Pa(s) ∩ seen(X) (what X sees: the
        predictors themselves and their parents), D = Pa(s) minus K. Returns (K, D), or None when the pair is useless:
        K holds no channel (a facet containing a predictor, or a predictor itself), or D contains nothing but s's own
        noise. There is one deletion per pair: with a smaller D the kept set would contain a parent X cannot see, and
        the lifted X could not compute s (manuscript 8.2). Whether the deletion is justified is fritz_certificate's
        question, asked about this same D."""
        g, latent_nodes = self.effective_DAG_data
        X = frozenset(X)
        seen = set(X)
        for x in X:
            seen.update(g.predecessors(x))
        parents = set(g.predecessors(s))
        K = frozenset(parents & seen)
        D = frozenset(parents - K)
        channel = any(k in X for k in K) or any(latent_nodes[k][0] != 'noise' and latent_nodes[k][1] & X
                                                for k in K if k >= self.number_of_visible)
        noise_only = all(k >= self.number_of_visible and latent_nodes[k][0] == 'noise' for k in D)
        if not channel or noise_only:
            return None
        return K, D

    def fritz_certificate(self, s: int, K: frozenset, X: Iterable[int], use_lp: bool = True,
                          lp_markov_target: bool = False, tally: bool = True,
                          engine: Optional[str] = None) -> Optional[str]:
        """Is restricting s to K justified by the predictors X? 'dsep' when every predictor is a parent of s (then
        s = g(X) is a function of K outright, manuscript 8.5) or when X minus K is d-separated from Pa(s) minus K given
        K in the effective DAG (Theorem 6.3); otherwise, with use_lp, the engine ('semigraphoid', 'lp' or 'both', default
        DEFAULT_ENGINE) is asked to certify the same deletion with the `relabel` target set (and `markov` too if
        lp_markov_target): 'semigraphoid' when the closure certifies it, 'entropic' when the LP does (Theorem 7.5);
        None otherwise (manuscript 8.3, 7.9)."""
        g, latent_nodes = self.effective_DAG_data
        X = frozenset(X)
        count = _tally if tally else (lambda kind, outcome: None)   # joint re-checks are not counted (ENTROPIC_STATS)
        if X <= K:
            count('certificate', 'vacuous')
            return 'dsep'
        others = set(g.predecessors(s)) - K
        if nx.is_d_separator(g, X - K, others, set(K)):
            count('certificate', 'dsep')
            return 'dsep'
        if use_lp:
            engine = engine or DEFAULT_ENGINE
            assert engine in ('semigraphoid', 'lp', 'both'), engine
            nodes, parents = self.lp_structure
            kept = dict(parents)
            kept[s] = frozenset(k for k in K if k in parents)
            closure = lp = None
            if engine in ('semigraphoid', 'both'):
                closure = self._semigraphoid_certificate(X, kept, (s,), try_markov=lp_markov_target)
            if engine == 'lp' or (engine == 'both' and closure is None):
                lp = self._entropic_certificate(X, kept, (s,), try_markov=lp_markov_target)
            if engine == 'both' and (closure is None) != (lp is None) and closure is None:
                count('disagreement', 'lp_only')
                ENGINE_DISAGREEMENTS.append((self.unique_id, tuple(sorted(X)), s, tuple(sorted(kept[s])), lp, closure))
            outcome = closure or lp
            count('certificate', outcome or 'failed')
            if closure is not None:
                count('engine', 'semigraphoid')
                return 'semigraphoid'
            if lp is not None:
                count('engine', 'entropic')
                return 'entropic'
        return None

    def fritz_targets(self, X: Iterable[int], pool: str = 'siblings', allow_descendants: bool = False,
                      pools: Optional[Dict[int, List[int]]] = None) -> List[int]:
        """Candidate targets of the predictor set X, in the order the search tries them: the nodes s whose pool
        (fritz_pool) contains every member of X, so that each member has its own channel to s. Nodes sharing more
        facets with X come first, then by index. `pools` may hold precomputed fritz_pool lists per node."""
        X = frozenset(X)
        _, latent_nodes = self.effective_DAG_data
        facets = [members for kind, members in latent_nodes.values() if kind != 'noise']
        if pools is None:
            pools = {s: self.fritz_pool(s, pool=pool, allow_descendants=allow_descendants) for s in self.visible_nodes}

        def shared(s: int) -> int:
            return sum(1 for members in facets if s in members and members & X)
        out = [s for s in self.visible_nodes if s not in X and X <= set(pools[s])]
        return sorted(out, key=lambda s: (-shared(s), s))

    def _restrict_targets(self, kept: Dict[int, frozenset]) -> "QmDAG":
        """The candidate structure: every target s in `kept` keeps exactly the parents kept[s] (visible nodes and
        facets, effective-DAG indices of self); a facet kept by a target stays quantum for its other members and
        becomes classical for the target (a classical facet over all its members is added); every other node is
        untouched. Plain integer labels."""
        g, latent_nodes = self.effective_DAG_data
        n = self.number_of_visible
        edges = sorted((a, b) for (a, b) in self.directed_structure_instance.as_set_of_tuples
                       if b not in kept or a in kept[b])
        C_facets, Q_facets = set(), set()
        for idx, (kind, members) in latent_nodes.items():
            if kind == 'noise':
                continue
            losers = {s for s in members if s in kept and idx not in kept[s]}
            keepers = {s for s in members if s in kept and idx in kept[s]}
            rest = members - losers
            if keepers:
                C_facets.add(rest)
                if kind == 'Q':
                    Q_facets.add(rest - keepers)
            else:
                (C_facets if kind == 'C' else Q_facets).add(rest)
        C_facets = {f for f in C_facets if len(f) >= 2}
        Q_facets = {f for f in Q_facets if len(f) >= 2}
        return QmDAG(DirectedStructure(edges, n), Hypergraph(hypergraph_full_cleanup(C_facets), n),
                     Hypergraph(hypergraph_full_cleanup(Q_facets), n))

    def _restrict_target(self, s: int, K: frozenset) -> "QmDAG":
        """_restrict_targets for a single target."""
        return self._restrict_targets({s: frozenset(K)})

    def _marginalize_nodes(self, order: Iterable[int]) -> Optional["QmDAG"]:
        """Marginalizes the given nodes (indices of self) in the given order, with teleportation; indices shift
        after each removal, which is tracked. A childless node is thereby simply deleted."""
        labels = list(range(self.number_of_visible))
        work = self
        for y in order:
            idx = labels.index(y)
            work = work.marginalize(idx, districts_check=False, apply_teleportation=True)
            if work is None:
                return None
            labels.pop(idx)
        return work

    def fritz_realise(self, kept: Dict[int, frozenset], remove: Iterable[int]) -> List[Tuple[Tuple, "QmDAG"]]:
        """The surgery of one Fritz step, with no certificate logic: restrict every target in `kept` to its kept
        parents, then remove the nodes in `remove` (the dropped predictors, or the copies of kept predictors) by
        marginalization in every order (teleportation is order-dependent). Returns (order params, child) pairs; the
        params are empty when there is one order."""
        candidate = self._restrict_targets(kept)
        remove = sorted(remove)
        if not remove:
            return [((), candidate)]
        results, seen = [], set()
        for order in itertools.permutations(remove):
            child = candidate._marginalize_nodes(order)
            if child is None or child.unique_id in seen:
                continue
            seen.add(child.unique_id)
            results.append(((('order', order),) if len(remove) > 1 else (), child))
        return results

    def fritz_steps(self, mode: str = 'replace', predictor_mode: str = 'dropped', use_lp: bool = True,
                    pool: str = 'siblings', allow_descendants: bool = False, max_predictors: int = 1,
                    max_targets: Optional[int] = None, lp_markov_target: bool = False,
                    max_visible: Optional[int] = None, lp_only: bool = False,
                    engine: Optional[str] = None) -> Iterable[Tuple[Tuple, "QmDAG"]]:
        """The Fritz piggyback as the search applies it (manuscript Section 8): for every predictor set X, every
        candidate target s of X and the deletion X dictates at s, the realised output, certified by d-separation
        first and by the LP only where d-separation fails. Besides single targets, every set of two or more targets
        of the same X that d-separation certifies (on the structure carrying all their splits) is emitted as one
        joint step, up to max_targets members (all by default): Fritz's derivation of Bell from the triangle is one
        such step. All d-separation steps, single and joint, are emitted before any LP step, so a search that stops at
        the first success never pays for an LP it does not need; the LP certifies single targets only. With lp_only
        the d-separation steps are not emitted at all (the census records them in its d-separation stages first,
        manuscript 9.2) and no joint sets are formed: only the single targets the LP certifies come out.
        mode 'copy' splits each target first and restricts the copy (6.2); predictor_mode 'kept' splits each childful
        predictor and removes the copy, 'dropped' removes the predictors themselves. Justification and surgery run
        on the structure that carries the splits; copies are the indices >= self.number_of_visible, in splitting
        order (predictor copies first), and params are stated in self's indices with a prime for a copy.
        `engine` names the beyond-d-separation certificate engine (DEFAULT_ENGINE: 'semigraphoid', 'lp' or 'both').
        params: (('targets', (s, ...)), ('mode', m), ('deleted', (labels of D_s, ...)), ('predictor', X),
        ('predictor_mode', pm), ('certificate', 'dsep' | 'semigraphoid' | 'entropic')) [+ ('order', ...) when several
        predictors are marginalized]."""
        assert mode in ('replace', 'copy') and predictor_mode in ('dropped', 'kept')
        if lp_only:
            if not use_lp:
                return   # an LP stage without the LP (mosek missing) has nothing to emit
            max_targets = 1
        n = self.number_of_visible
        if max_visible is None:
            max_visible = n + 1
        pools = {s: self.fritz_pool(s, pool=pool, allow_descendants=allow_descendants) for s in self.visible_nodes}
        predictor_candidates = sorted({x for candidates in pools.values() for x in candidates})
        deferred = []

        def labels_of(work: "QmDAG", D: frozenset, copy_of: Dict[int, int]) -> Tuple:
            _, latent_nodes = work.effective_DAG_data
            m = work.number_of_visible

            def node_label(v: int):
                return f"{copy_of[v]}'" if v >= n else v
            out = []
            for d in sorted(D):
                if d < m:
                    out.append(node_label(d))
                else:
                    kind, members = latent_nodes[d]
                    if kind != 'noise':
                        out.append(kind + '{' + ','.join(str(node_label(v)) for v in sorted(members)) + '}')
            return tuple(out)

        def split_predictors(X: Tuple[int, ...]):
            """The structure with the childful predictors split (kept mode), the predictors' effective indices, the
            nodes to remove and the copy map; computed once per predictor set."""
            work, copy_of, X_eff = self, {}, list(X)
            if predictor_mode == 'kept':
                for i, x in enumerate(X):
                    if x not in self.vis_nodes_with_no_children:
                        work = work.split_node(x)
                        X_eff[i] = work.number_of_visible - 1
                        copy_of[X_eff[i]] = x
                remove = frozenset(x for x in X_eff if x >= n)
            else:
                remove = frozenset(X_eff)
            return work, frozenset(X_eff), remove, copy_of

        def prepare(base, T: Tuple[int, ...]):
            """The structure carrying all the splits for the targets T on top of the predictor splits `base`, with
            the effective indices of the predictors and targets, the nodes to remove and the copy map; None when the
            output would be too small or too large."""
            work, X_eff, remove, copy_of = base
            copy_of, targets = dict(copy_of), list(T)
            if mode == 'copy':
                for i, s in enumerate(T):
                    work = work.split_node(s)
                    targets[i] = work.number_of_visible - 1
                    copy_of[targets[i]] = s
            if not (3 <= work.number_of_visible - len(remove) <= max_visible):
                return None
            return work, X_eff, tuple(targets), remove, copy_of

        for r in range(1, min(max_predictors, len(predictor_candidates)) + 1):
            for X in itertools.combinations(predictor_candidates, r):
                targets = self.fritz_targets(X, pools=pools)
                if not targets:
                    continue
                base = split_predictors(X)
                certified = []   # targets certified by d-separation on their own
                for s in targets:
                    prepared = prepare(base, (s,))
                    if prepared is None:
                        continue
                    work, X_eff, (target,), remove, copy_of = prepared
                    deletion = work.fritz_deletion(target, X_eff)
                    if deletion is None:
                        continue
                    K, D = deletion
                    item = (X, (s,), work, {target: K}, {target: D}, X_eff, remove, copy_of)
                    certificate = work.fritz_certificate(target, K, X_eff, use_lp=False, tally=not lp_only)
                    if certificate is None:
                        if use_lp:
                            deferred.append(item)
                        continue
                    if lp_only:
                        continue   # recorded by the d-separation stage
                    certified.append(s)
                    yield from self._fritz_emit(item, mode, predictor_mode, certificate, labels_of)
                top = len(certified) if max_targets is None else min(max_targets, len(certified))
                for k in range(2, top + 1):
                    for T in itertools.combinations(certified, k):
                        prepared = prepare(base, T)
                        if prepared is None:
                            continue
                        work, X_eff, eff_targets, remove, copy_of = prepared
                        kept, deleted = {}, {}
                        for target in eff_targets:
                            deletion = work.fritz_deletion(target, X_eff)
                            if deletion is None or work.fritz_certificate(target, deletion[0], X_eff, use_lp=False, tally=False) is None:
                                kept = None
                                break
                            kept[target], deleted[target] = deletion
                        if kept is None:
                            continue
                        item = (X, T, work, kept, deleted, X_eff, remove, copy_of)
                        yield from self._fritz_emit(item, mode, predictor_mode, 'dsep', labels_of)
        for item in deferred:
            X, T, work, kept, deleted, X_eff, remove, copy_of = item
            (target,) = kept
            certificate = work.fritz_certificate(target, kept[target], X_eff, use_lp=True, lp_markov_target=lp_markov_target,
                                                 engine=engine)
            if certificate is not None:
                yield from self._fritz_emit(item, mode, predictor_mode, certificate, labels_of)

    @staticmethod
    def _fritz_emit(item, mode: str, predictor_mode: str, certificate: str, labels_of) -> Iterable[Tuple[Tuple, "QmDAG"]]:
        X, T, work, kept, deleted, X_eff, remove, copy_of = item
        base = (('targets', tuple(T)), ('mode', mode),
                ('deleted', tuple(labels_of(work, deleted[t], copy_of) for t in kept)), ('predictor', tuple(X)),
                ('predictor_mode', predictor_mode), ('certificate', certificate))
        for order_params, child in work.fritz_realise(kept, remove):
            yield base + order_params, child

    # ------------------------------------------------------------------
    # COMPOSITION OF PIGGYBACKS
    # ------------------------------------------------------------------

    def piggyback_children(self, max_visible: int, min_visible: int = 3, districts_check: bool = False,
                           apply_teleportation: bool = True, include_Fritz: bool = True, max_predictors: int = 2,
                           predictor_modes: Tuple[str, ...] = ('dropped', 'kept'),
                           strict_conditioning: bool = True) -> Iterable["QmDAG"]:
        """One application of every piggyback (PD, conditioning, marginalization, node stitching, Fritz by
        d-separation in both predicted-node modes and the given predictor modes): the old composition API behind
        unique_unlabelled_ids_obtainable_by_*, a closure of any depth (unlike the census cascade)."""
        n = self.number_of_visible
        if n > min_visible:
            yield from self.subgraphs
            for node in self.visible_nodes:
                if self.conditioning_is_justified(node, strict_latents=strict_conditioning):
                    yield self.condition(node)
                marginalized = self.marginalize(node, districts_check=districts_check,
                                                apply_teleportation=apply_teleportation)
                if marginalized is not None:
                    yield marginalized
            yield from self.subinterruptions
        if include_Fritz:
            for mode in ('replace', 'copy'):
                for predictor_mode in predictor_modes:
                    for _, child in self.fritz_steps(mode=mode, predictor_mode=predictor_mode, use_lp=False,
                                                     max_predictors=max_predictors, max_visible=max_visible):
                        yield child

    def piggyback_closure(self, max_visible: int = None, min_visible: int = 3, max_states: int = 50000,
                          **kwargs) -> Dict[Tuple[int, int, int, int], "QmDAG"]:
        """Every structure reachable by composing piggybacks in any order, keyed by unlabelled id.
        Each unlabelled id is expanded once (all tricks are label-equivariant)."""
        if max_visible is None:
            max_visible = self.number_of_visible + 1
        reached = {self.unique_unlabelled_id: self}
        frontier = [self]
        options = (max_visible, min_visible, tuple(sorted(kwargs.items())))
        while frontier:
            current = frontier.pop()
            memo_key = (current.unique_unlabelled_id,) + options
            try:
                children = _PIGGYBACK_CHILDREN_MEMO[memo_key]
            except KeyError:
                children = list({child.unique_unlabelled_id: child for child in
                                 current.piggyback_children(max_visible=max_visible, min_visible=min_visible, **kwargs)
                                 if min_visible <= child.number_of_visible <= max_visible}.values())
                _PIGGYBACK_CHILDREN_MEMO[memo_key] = children
            for child in children:
                child_id = child.unique_unlabelled_id
                if child_id not in reached:
                    reached[child_id] = child
                    frontier.append(child)
                    if len(reached) > max_states:
                        warnings.warn("Piggyback closure exceeded max_states; the result is incomplete. "
                                      "Raise max_states or lower max_visible.")
                        return reached
        return reached

    @lru_cache(maxsize=None)
    def unique_unlabelled_ids_obtainable_by_Fritz_for_QC(self, max_visible: int = None,
                                                         max_predictors: int = 2) -> Set[Tuple[int, int, int, int]]:
        """Unlabelled ids reachable from self by any composition of the piggybacks including Fritz (self excluded)."""
        reached = self.piggyback_closure(max_visible=max_visible, districts_check=False, apply_teleportation=True,
                                         include_Fritz=True, max_predictors=max_predictors)
        return set(reached).difference({self.unique_unlabelled_id})

    @lru_cache(maxsize=None)
    def unique_unlabelled_ids_obtainable_by_Fritz_for_IC(self, max_visible: int = None,
                                                         max_predictors: int = 2) -> Set[Tuple[int, int, int, int]]:
        reached = self.piggyback_closure(max_visible=max_visible, districts_check=True, apply_teleportation=False,
                                         include_Fritz=True, max_predictors=max_predictors)
        return set(reached).difference({self.unique_unlabelled_id})


if __name__ == '__main__':
    ghost = QmDAG(DirectedStructure([(1, 2), (1, 3)], 4), Hypergraph([], 4), Hypergraph([(0, 2), (0, 3)], 4))
    print("All graphs obtainable from the Ghost by node stitching (should be Evans)")
    print(ghost.subinterruptions)
    print("Now assessing the Fritz piggyback on the triangle (should reach Bell):")
    triangle = QmDAG(DirectedStructure([], 3), Hypergraph([], 3), Hypergraph([(0, 1), (1, 2), (0, 2)], 3))
    for params, post_Fritz in triangle.fritz_steps(mode='copy', use_lp=False):
        print(params)
        print(post_Fritz)
