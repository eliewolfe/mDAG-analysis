"""
Quantum observational equivalence classes as the unit of the QC-gap search.

A `ClassIndex` holds the quantum metagraph (metagraph_advanced.QuantumMetagraph) and its classical companion for
every node count it covers (3 and 4 in the census), and turns a QmDAG into the *key* of its class:

* an all-quantum structure with a covered node count is keyed by its quantum class, represented by the smallest
  unlabelled id among the class's all-quantum members (so a key is always the unlabelled id of some structure);
* a hybrid structure (some classical facets) that lifts (QmDAG.lift: every classical facet is a 2-facet one of whose
  members has no visible parent and no other facet, so the facet is classical and quantum at once) is keyed by the
  class of its lift;
* anything else (a hybrid that does not lift, a five-node structure) is keyed by its own unlabelled id.

`members(key)` lists the all-quantum structures of the class (one labelled representative each): the search applies
the piggybacks to every member, since different members have different children. `inputs(n)` are the classes with
no member classically equivalent to a latent-free DAG (for those Q(G) = C(G): no gap). `upward_steps(g)` yields,
for an input class, the classes structurally below it inside the same classical class: a gap of any of those is a
gap of g's class (its quantum model contains theirs, its classical model equals theirs), which the search records
as 'upward' lookups.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set, Tuple

from metagraph_advanced import ClassicalMetagraph, QuantumMetagraph
from quantum_mDAG import QmDAG, upgrade_to_QmDAG

UnlabelledId = Tuple[int, int, int, int]


class ClassIndex:
    def __init__(self, node_counts: Iterable[int] = (3, 4), verbose: bool = False) -> None:
        self.quantum: Dict[int, QuantumMetagraph] = {}
        self.classical: Dict[int, ClassicalMetagraph] = {}
        self._members: Dict[UnlabelledId, List[QmDAG]] = {}          # key -> all-quantum members
        self._key_of_mdag: Dict[Tuple[int, int], UnlabelledId] = {}   # (n, mDAG unlabelled id) -> key
        self._mdag_ids_of: Dict[UnlabelledId, Set[int]] = {}         # key -> member mDAG unlabelled ids
        self._upward: Dict[UnlabelledId, List[Tuple[Tuple, QmDAG]]] = {}   # key -> upward_steps, computed once
        for n in node_counts:
            Q = QuantumMetagraph(n, verbose=verbose)
            C = ClassicalMetagraph.from_quantum(Q)
            self.quantum[n], self.classical[n] = Q, C
            for cls in Q.equivalence_classes_as_ids:
                members = [upgrade_to_QmDAG(Q.dict_uniqind_unlabelled_mDAGs[uid]) for uid in sorted(cls)]
                key = min(m.unique_unlabelled_id for m in members)
                self._members[key] = members
                self._mdag_ids_of[key] = set(cls)
                for uid in cls:
                    self._key_of_mdag[(n, uid)] = key

    # -- keys ---------------------------------------------------------------------------------------------

    def covers(self, g: QmDAG) -> bool:
        return g.number_of_visible in self.quantum

    def key(self, g: QmDAG) -> UnlabelledId:
        """The class key of a structure (see the module docstring)."""
        lifted = g.lift() if self.covers(g) else None
        if lifted is None:
            return g.unique_unlabelled_id
        return self._key_of_mdag[(lifted.number_of_visible, lifted.as_mDAG.unique_unlabelled_id)]

    def is_class_key(self, key: UnlabelledId) -> bool:
        return key in self._members

    def members(self, key: UnlabelledId, fallback: Optional[QmDAG] = None) -> List[QmDAG]:
        """The all-quantum members of a class (labelled representatives); a non-class key has the one structure
        the caller holds."""
        if key in self._members:
            return self._members[key]
        return [fallback] if fallback is not None else []

    def representative(self, key: UnlabelledId) -> Optional[QmDAG]:
        members = self._members.get(key)
        if not members:
            return None
        return min(members, key=lambda m: m.unique_unlabelled_id)

    def size(self, key: UnlabelledId) -> int:
        return len(self._members.get(key, ())) or 1

    def classical_class(self, key: UnlabelledId) -> Optional[Tuple[int, int]]:
        """(n, index) of the classical equivalence class containing the quantum class `key`; None for a key that is
        not a class (a hybrid that does not lift, a structure of an uncovered node count)."""
        members = self._members.get(key)
        if not members:
            return None
        n = members[0].number_of_visible
        uid = next(iter(self._mdag_ids_of[key]))
        return (n, self.classical[n].class_index_of[uid])

    def classical_classes(self, keys: Iterable[UnlabelledId]) -> int:
        """How many classical equivalence classes the given quantum classes fall into."""
        return len({self.classical_class(k) for k in keys if self.classical_class(k) is not None})

    # -- inputs -------------------------------------------------------------------------------------------

    def is_latent_free_equivalent(self, key: UnlabelledId) -> bool:
        """Does the class contain a structure classically equivalent to a latent-free DAG?"""
        members = self._members.get(key)
        if not members:
            return False
        n = members[0].number_of_visible
        return bool(self._mdag_ids_of[key] & self.classical[n].latent_free_equivalent_ids)

    def input_keys(self, n: int) -> List[UnlabelledId]:
        """Keys of the quantum classes of n-node all-quantum structures that are not provably algebraic."""
        return sorted(key for key, members in self._members.items()
                      if members[0].number_of_visible == n and not self.is_latent_free_equivalent(key))

    def inputs(self, n: int) -> List[QmDAG]:
        """One representative per input class."""
        return [self.representative(key) for key in self.input_keys(n)]

    def all_quantum_structures(self, keys: Iterable[UnlabelledId]) -> int:
        return sum(self.size(key) for key in keys)

    # -- the upward rule ----------------------------------------------------------------------------------

    def upward_steps(self, g: QmDAG) -> List[Tuple[Tuple, QmDAG]]:
        """For a structure of a covered class: the classes structurally below its class inside the same classical
        class, as (params, representative) pairs. A gap of such a class is a gap of g's class: g's quantum model
        contains the lower structure's (it has more edges or facets) while its classical model equals it (same
        classical class). params: (('above', <member of g's class>), ('below', <structure below it>))."""
        key = self.key(g)
        members = self._members.get(key)
        if not members:
            return []
        if key in self._upward:
            return self._upward[key]
        n = members[0].number_of_visible
        C = self.classical[n]
        out: Dict[UnlabelledId, Tuple] = {}
        for uid in sorted(self._mdag_ids_of[key]):
            for below in sorted(C.within_class_below(uid)):
                below_key = self._key_of_mdag[(n, below)]
                if below_key == key or below_key in out:
                    continue
                above_structure = str(C.dict_uniqind_unlabelled_mDAGs[uid])
                below_structure = str(C.dict_uniqind_unlabelled_mDAGs[below])
                out[below_key] = (('above', above_structure), ('below', below_structure))
        self._upward[key] = [(params, self.representative(below_key)) for below_key, params in out.items()]
        return self._upward[key]

    # -- seeds --------------------------------------------------------------------------------------------

    def seed_keys(self, seeds: Dict[str, QmDAG]) -> Dict[UnlabelledId, str]:
        """key -> '+'-joined seed names. A seed that lifts is the class of its lift; a seed that does not (a hybrid
        strictly weaker than its all-quantum version) keeps its own id, and the class of its all-quantum version is
        proven through the degradation lookup."""
        by_key: Dict[UnlabelledId, List[str]] = {}
        for name, g in seeds.items():
            by_key.setdefault(self.key(g), []).append(name)
        return {key: '+'.join(sorted(names)) for key, names in by_key.items()}
