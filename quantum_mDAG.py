from __future__ import absolute_import
import numpy as np
import itertools
from hypergraphs import Hypergraph, LabelledHypergraph
from directed_structures import DirectedStructure, LabelledDirectedStructure
from radix import bitarray_to_int
from mDAG_advanced import mDAG
from sys import hexversion
if hexversion >= 0x3080000:
    from functools import cached_property
elif hexversion >= 0x3060000:
    from backports.cached_property import cached_property
else:
    cached_property = property


#This class does NOT represent every possible quantum causal structure. It only represents the causal structures where every quantum latent is exogenized. This is the case, for example, of the known QC Gaps.
class QmDAG:
    def __init__(self, directed_structure_instance, C_simplicial_complex_instance,Q_simplicial_complex_instance):
        self.directed_structure_instance = directed_structure_instance
        self.number_of_visible = self.directed_structure_instance.number_of_visible
        self.C_simplicial_complex_instance = C_simplicial_complex_instance
        self.Q_simplicial_complex_instance = Q_simplicial_complex_instance
        assert directed_structure_instance.number_of_visible == C_simplicial_complex_instance.number_of_visible, 'Different number of nodes in directed structure vs classical simplicial complex.'
        assert directed_structure_instance.number_of_visible == Q_simplicial_complex_instance.number_of_visible, 'Different number of nodes in directed structure vs quantum simplicial complex.'
        if hasattr(self.directed_structure_instance, 'variable_names'):
            self.variable_names = self.directed_structure_instance.variable_names
            if hasattr(self.C_simplicial_complex_instance, 'variable_names'):
                assert frozenset(self.variable_names) == frozenset(self.C_simplicial_complex_instance.variable_names), 'Error: Inconsistent node names.'
                if not tuple(self.variable_names) == tuple(self.C_simplicial_complex_instance.variable_names):
                    print('Warning: Inconsistent node ordering. Following ordering of directed structure!')
        if hasattr(self.directed_structure_instance, 'variable_names'):
            self.variable_names = self.directed_structure_instance.variable_names
            if hasattr(self.Q_simplicial_complex_instance, 'variable_names'):
                assert frozenset(self.variable_names) == frozenset(self.Q_simplicial_complex_instance.variable_names), 'Error: Inconsistent node names.'
                if not tuple(self.variable_names) == tuple(self.Q_simplicial_complex_instance.variable_names):
                    print('Warning: Inconsistent node ordering. Following ordering of directed structure!')
        self.visible_nodes = self.directed_structure_instance.visible_nodes
        self.classical_latent_nodes = tuple(range(self.number_of_visible, self.C_simplicial_complex_instance.number_of_visible_plus_latent))
        self.nonsingleton_classical_latent_nodes = tuple(range(self.number_of_visible, self.C_simplicial_complex_instance.number_of_visible_plus_nonsingleton_latent))
        #it is not necessary to talk about quantum singletons in the first place:
        self.quantum_latent_nodes = tuple(range(self.C_simplicial_complex_instance.number_of_visible_plus_latent, self.Q_simplicial_complex_instance.number_of_visible_plus_nonsingleton_latent+self.C_simplicial_complex_instance.number_of_visible_plus_latent-self.number_of_visible))

    @cached_property
    def as_string(self):
        return self.directed_structure_instance.as_string + '|' + self.C_simplicial_complex_instance.as_string+ '|' + self.Q_simplicial_complex_instance.as_string
    
    def __str__(self):
        return self.as_string
    
    def __repr__(self):
        return self.as_string


    def quantum_parents(self, node):
        return [quantum_facet for  quantum_facet in self.quantum_simplicial_complex_instance.simplicial_complex_as_sets if node in quantum_facet]
   
    def clean_C_simplicial_complex(self):  #remove classical facets that are redundant to quantum facets
        new_C_simplicial_complex=self.C_simplicial_complex_instance.simplicial_complex_as_sets.copy()
        for C_facet in self.C_simplicial_complex_instance.simplicial_complex_as_sets:
            for Q_facet in self.Q_simplicial_complex_instance.simplicial_complex_as_sets:
                if set(C_facet).issubset(Q_facet):
                    new_C_simplicial_complex.remove(C_facet)
                    break
        return Hypergraph(new_C_simplicial_complex,self.number_of_visible)
         
    
    # When finding the unique Quantum idetification tuple, remember to do it for the CLEAN classical simplicial complex

    
    def fix_to_point_distribution(self,node):  #returns a smaller QmDAG
        ds=self.directed_structure_instance.edge_list.copy()
        Csc=self.C_simplicial_complex_instance.simplicial_complex_as_sets.copy()
        Qsc=self.Q_simplicial_complex_instance.simplicial_complex_as_sets.copy()
        for edge in self.directed_structure_instance.edge_list:
            if node in edge:
                ds.remove(edge)
        for C_facet in self.C_simplicial_complex_instance.simplicial_complex_as_sets:
            if node in C_facet:
                Csc.remove(C_facet)
                if len(set(C_facet)-set((node)))>1:
                    Csc.append(set(C_facet)-set((node)))
        for Q_facet in self.Q_simplicial_complex_instance.simplicial_complex_as_sets:
            if node in Q_facet:
                Qsc.remove(Q_facet)
                if len(set(Q_facet)-set((node)))>1:
                    Qsc.append(set(Q_facet)-set((node)))
        not_clean_QmDAG=QmDAG(DirectedStructure(ds,self.number_of_visible-1),Hypergraph(Csc,self.number_of_visible-1),Hypergraph(Qsc,self.number_of_visible-1)) 
        clean_QmDAG=QmDAG(not_clean_QmDAG.directed_structure_instance,not_clean_QmDAG.clean_C_simplicial_complex,not_clean_QmDAG.Q_simplicial_complex_instance)
        return clean_QmDAG
    
    
   #can you simulate one of the simple ones (Instrumental with 1 quantum, 1 classical latent)
   # def can_teleport(self, node):  #node is a classical visible variable that we may marginalize over (turning it into a classical latent node)
   #get the number that corresponds to mDAG
            

QG= QmDAG(DirectedStructure([(1,2),(2,3)],4),Hypergraph([(0,2),(1,2),(2,3)],4),Hypergraph([(1,2,3)],4))    
QGc=QG.clean_C_simplicial_complex()


DirectedStructure([(1,2),(2,3)],4).as_bit_square_matrix

#Teleportation reduces classical latent nodes:
      