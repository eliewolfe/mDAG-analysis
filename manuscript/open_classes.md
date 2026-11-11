# The 46 open classes: what would follow from what, and which equivalences would help

*A working record for directing effort at the four-node classes the census leaves open. Class numbers are those of the list in Section 10 of [`piggybacks.md`](piggybacks.md) (reprinted in 1.2 below); section numbers such as 6.3 refer to that manuscript. Every count is printed by [`Special Applications/open_classes_analysis.py`](../Special%20Applications/open_classes_analysis.py); see Section 4. Structures are written as edges, then sources: `Q{a,b}` is a quantum source shared by nodes a and b, `C{...}` a classical one.*

## Summary

The question (Roger Colbeck): tell the code that one open class has a gap, see which other open classes it then proves, repeat for all 46. Is there a hierarchy, and a smaller set of key structures?

1. **There is a hierarchy, and it is small.** Exactly one open class has others hanging on it. If **class 1** had a gap, **classes 2, 3, 6 and 7** would follow: 5 classes, 16 of the 61 open structures. Class 1 is the KPC structure with one more source, shared by $D$ and $E$. No other open class would prove any other.
2. **Below the level of classes the picture barely changes.** The open classes reduce, by the piggybacks, to 15,057 structures. All but 1,474 of them provably have no gap. Among the rest, structures reached from more than one class occur in two groups only: the five classes above, and classes 4 and 5, which meet at a hybrid structure. The other 39 classes share nothing with anyone.
3. **A second Fritz step does not help.** Given to the cascade as inputs, none of the 94 intermediates that are neither degradations nor provably gap-free is proven.
4. **Equivalences.** No classical equivalence the metagraph knows can shrink the open set if proven quantum-valid, because every open class is already alone in its classical class. What could help is new *classical* equivalences, through the upward rule. Against the proven classes this is ruled out: every candidate pair is classically inequivalent. Among the open classes, 24 pairs are not separated by any test we ran. If those equivalences hold, 13 more classes become consequences of others, leaving 29 independent problems.

| Target | Settles | Status |
|---|---|---|
| a gap for **class 1** (KPC plus a source on $D,E$) | classes 1, 2, 3, 6, 7 (16 structures) | open; the KPC argument fails on it (1.3) |
| a gap for the hybrid `0→1, 1→2, 1→3; C{0,1,2}, C{0,1,3}, Q{0,2}, Q{0,3}` | classes 4 and 5 | open (1.4) |
| a gap for the Evans structure with **one** source classical | class 5 | open in the literature as far as the seed list records (2.4) |
| classical equivalence of the "seven hard graphs" (classes 39, 40, 41, 43, 44, 45, 46) | makes 41, 43, 44, 45, 46 consequences of 39 or 40 | not refuted through 7 events (2.3) |
| nine further classical equivalences, each removing a two-party source that sits on an edge | makes classes 10, 17, 21, 24, 29, 31, 35, 37 consequences of 8, 15, 18, 23, 25, 28, 30, 34; ties class 1 to class 7 | not refuted through 7 events (2.3) |

## 1. The proof hierarchy

### 1.1 The experiment, and why the recorded graph answers it

The census records every transition it finds: a labelled edge from a structure (or class) to the structure a piggyback produces, meaning *a gap of the target is a gap of the source*. For a class that stays unproven, nothing is cut short: every elementary reduction of every member, closed over everything reachable, and every Fritz step of every member in all nine stages, each followed by the elementary reductions (9.2). So "which classes would be proven if class $i$ were known" is a question about this graph, and it has the same answer as rerunning the census with class $i$ as an extra seed, since a rerun explores a subset of what is already recorded.

The script does the experiment literally: for each of the 46 classes it rebuilds the report with that class added to the known gaps and lists the open classes that become proven (46 reports, 15 seconds), and checks that this agrees with reachability in the graph.

Two limits carry over from the census. The Fritz step is applied once per input ("depth one"); Section 1.5 tests what a second depth adds. And a hypothesis "class $i$ has a gap" is about the all-quantum structures of the class; it says nothing about their degradations, which are weaker.

### 1.2 Result

```mermaid
flowchart BT
    c2["class 2"] --> c1["class 1 (12 structures)"]
    c3["class 3"] --> c1
    c6["class 6"] --> c1
    c7["class 7"] --> c1
```

An arrow $x\to y$ reads "a gap of $y$ would prove $x$". These four are the only relations among the 46.

| From | Steps to class 1 |
|---|---|
| class 2 | one Fritz step by d-separation (node 2 predicted by node 0, split target, unsplit predictor); the output `1→3, 2→3; C{1,2}, Q{0,2}, Q{0,3}, Q{2,3}` lifts into class 1 |
| class 3 | parent-predictor Fritz step (2 predicted by its parent 1, split target, split predictor), then point distribution on the copy; lands on `1→2, 2→3; Q{0,2}, Q{0,3}, Q{1,2,3}` |
| class 6 | parent-predictor Fritz step (3 predicted by its parent 2), then point distribution; lands on `1→3, 2→3; Q{0,2}, Q{0,3}, Q{1,2,3}` |
| class 7 | parent-predictor Fritz step (2 predicted by its parent 1), then point distribution; lands on `1→2, 1→3, 2→3; Q{0,2}, Q{0,3}, Q{1,2,3}` |

<details><summary>The 46 open classes (number, structures in the class, representative)</summary>

| | structures | representative |
|---|---|---|
| 1 | 12 | 2→3; Q{0,2}, Q{0,3}, Q{1,2,3} |
| 2 | 1 | 2→3; Q{0,2}, Q{0,3}, Q{1,2}, Q{1,3}, Q{2,3} |
| 3 | 1 | 1→2, 2→3; Q{0,2}, Q{0,3}, Q{1,2}, Q{1,3} |
| 4 | 1 | 0→1, 1→2, 1→3; Q{0,2}, Q{0,3} |
| 5 | 1 | 0→1, 1→2, 1→3; Q{0,2}, Q{0,3}, Q{1,3} |
| 6 | 1 | 1→3, 2→3; Q{0,1}, Q{0,3}, Q{1,2}, Q{2,3} |
| 7 | 1 | 1→2, 1→3, 2→3; Q{0,2}, Q{0,3}, Q{1,2}, Q{1,3} |
| 8 | 1 | 0→3, 1→2; Q{0,1}, Q{0,2}, Q{1,2,3} |
| 9 | 1 | 0→3, 1→2; Q{0,1}, Q{0,2,3}, Q{1,2,3} |
| 10 | 1 | 0→3, 1→2; Q{0,1}, Q{0,2}, Q{0,3}, Q{1,2,3} |
| 11 | 1 | 0→3, 1→2; Q{0,1}, Q{0,2}, Q{1,3}, Q{2,3} |
| 12 | 1 | 0→3, 1→2; Q{0,1}, Q{0,2}, Q{1,2}, Q{1,3}, Q{2,3} |
| 13 | 1 | 0→3, 1→2; Q{0,1}, Q{0,2}, Q{0,3}, Q{1,2}, Q{1,3}, Q{2,3} |
| 14 | 5 | 0→2, 1→2, 2→3; Q{0,1}, Q{1,3} |
| 15 | 1 | 0→1, 1→2, 2→3; Q{0,2}, Q{0,3}, Q{1,2,3} |
| 16 | 1 | 0→1, 1→2, 2→3; Q{0,2}, Q{0,3}, Q{1,3} |
| 17 | 1 | 0→1, 1→2, 2→3; Q{0,1}, Q{0,2}, Q{0,3}, Q{1,2,3} |
| 18 | 1 | 0→1, 1→2, 2→3; Q{0,2}, Q{0,3}, Q{1,2}, Q{1,3} |
| 19 | 1 | 0→1, 1→2, 2→3; Q{0,1}, Q{0,2}, Q{0,3}, Q{1,3} |
| 20 | 1 | 0→1, 1→2, 2→3; Q{0,2}, Q{0,3}, Q{1,3}, Q{2,3} |
| 21 | 1 | 0→1, 1→2, 2→3; Q{0,1}, Q{0,2}, Q{0,3}, Q{1,2}, Q{1,3} |
| 22 | 1 | 0→1, 1→2, 2→3; Q{0,1}, Q{0,2}, Q{0,3}, Q{1,3}, Q{2,3} |
| 23 | 1 | 0→1, 1→2, 2→3; Q{0,2}, Q{0,3}, Q{1,2}, Q{1,3}, Q{2,3} |
| 24 | 1 | 0→1, 1→2, 2→3; Q{0,1}, Q{0,2}, Q{0,3}, Q{1,2}, Q{1,3}, Q{2,3} |
| 25 | 1 | 0→3, 1→2, 2→3; Q{0,1}, Q{0,2}, Q{1,2,3} |
| 26 | 1 | 0→3, 1→2, 2→3; Q{0,1}, Q{0,2,3}, Q{1,2,3} |
| 27 | 1 | 0→3, 1→2, 2→3; Q{0,1}, Q{0,2}, Q{1,3} |
| 28 | 1 | 0→3, 1→2, 2→3; Q{0,1}, Q{0,2,3}, Q{1,3} |
| 29 | 1 | 0→3, 1→2, 2→3; Q{0,1}, Q{0,2}, Q{0,3}, Q{1,2,3} |
| 30 | 1 | 0→3, 1→2, 2→3; Q{0,1}, Q{0,2}, Q{1,2}, Q{1,3} |
| 31 | 1 | 0→3, 1→2, 2→3; Q{0,1}, Q{0,2,3}, Q{1,2}, Q{1,3} |
| 32 | 1 | 0→3, 1→2, 2→3; Q{0,1}, Q{0,2}, Q{1,3}, Q{2,3} |
| 33 | 1 | 0→3, 1→2, 2→3; Q{0,1}, Q{0,2}, Q{0,3}, Q{1,3} |
| 34 | 1 | 0→3, 1→2, 2→3; Q{0,1}, Q{0,2}, Q{1,2}, Q{1,3}, Q{2,3} |
| 35 | 1 | 0→3, 1→2, 2→3; Q{0,1}, Q{0,2}, Q{0,3}, Q{1,2}, Q{1,3} |
| 36 | 1 | 0→3, 1→2, 2→3; Q{0,1}, Q{0,2}, Q{0,3}, Q{1,3}, Q{2,3} |
| 37 | 1 | 0→3, 1→2, 2→3; Q{0,1}, Q{0,2}, Q{0,3}, Q{1,2}, Q{1,3}, Q{2,3} |
| 38 | 1 | 0→1, 1→2, 1→3, 2→3; Q{0,2}, Q{0,3} |
| 39 | 1 | 0→1, 1→2, 1→3, 2→3; Q{0,2}, Q{0,3}, Q{1,2} |
| 40 | 1 | 0→1, 1→2, 1→3, 2→3; Q{0,2}, Q{0,3}, Q{1,3} |
| 41 | 1 | 0→1, 1→2, 1→3, 2→3; Q{0,2}, Q{0,3}, Q{1,2,3} |
| 42 | 1 | 0→1, 1→2, 1→3, 2→3; Q{0,2}, Q{0,3}, Q{2,3} |
| 43 | 1 | 0→1, 1→2, 1→3, 2→3; Q{0,2}, Q{0,3}, Q{1,2}, Q{1,3} |
| 44 | 1 | 0→1, 1→2, 1→3, 2→3; Q{0,2}, Q{0,3}, Q{1,2}, Q{2,3} |
| 45 | 1 | 0→1, 1→2, 1→3, 2→3; Q{0,2}, Q{0,3}, Q{1,3}, Q{2,3} |
| 46 | 1 | 0→1, 1→2, 1→3, 2→3; Q{0,2}, Q{0,3}, Q{1,2}, Q{1,3}, Q{2,3} |

</details>

### 1.3 The hub: KPC with one more source

In the labels of the KPC structure ($F=0$, $C=1$, $D=2$, $E=3$; sources $A=\lbrace E,F\rbrace$, $B=\lbrace D,F\rbrace$; edges $C\to D$, $C\to E$, $D\to E$) one of the twelve members of class 1 is

`1→2, 1→3, 2→3; Q{0,2}, Q{0,3}, Q{2,3}`,

the KPC structure with an additional source shared by $D$ and $E$. The other eleven are its quantum-equivalent variants: the extra source may include $C$, and then some or all of $C$'s edges may be dropped. The four classes hanging on class 1 are neighbours of it: class 7 is KPC with sources on $\lbrace C,D\rbrace$ and $\lbrace C,E\rbrace$ instead, and classes 2, 3 and 6 replace some of the edges among $C$, $D$, $E$ by two-party sources.

```mermaid
flowchart BT
    A((A)) --> E[E]
    A --> F[F]
    B((B)) --> F
    B --> D[D]
    L(("Λ")) --> D
    L --> E
    C[C] --> D
    C --> E
    D --> E
    classDef latent fill:#f8b195,stroke:#555,color:#000
    classDef extra fill:#fbd46d,stroke:#555,color:#000
    class A,B latent
    class L extra
```

The new source is $\Lambda$ (yellow). The KPC argument stops here because Lemma 1 of the paper uses the d-separation $E\perp B\mid CD$. With the new source the path $E\leftarrow\Lambda\to D\leftarrow B$ is opened by conditioning on $D$, and the d-separation fails. The census tries the corresponding step ($E$ predicted by $F$, which now has to delete the new source too) and neither the semigraphoid closure nor the LP certifies it. The steps it does certify on this structure lead nowhere known.

So the single most valuable new result would be a gap for KPC with a $D$–$E$ source: it decides 16 of the 61 open structures, and it is one latent away from a structure whose gap is understood.

### 1.4 Below the class level: shared intermediates

A class may fail to reach another class and still reach a structure that another class reaches too. Such a structure, if it had a gap, would settle both. The open classes reach 15,057 structures; none is a known gap, by definition of open. Most cannot be targets, because they provably have no gap:

| visible nodes | reached | no quantum source | classically equivalent to a latent-free DAG | every quantum source equivalent to a classical one | **not provably gap-free** |
|---|---|---|---|---|---|
| 3 | 93 | 17 | 74 | 1 | **1** |
| 4 | 13,628 | 20 | 12,541 | 10 | **1,057** |
| 5 | 1,336 | 0 | 915 | 5 | **416** |

The second criterion uses the classical metagraphs, the five-node one included (a structure classically equivalent to a latent-free DAG has $\mathcal Q=\mathcal C$, 1.0). The third is the pre-measurement argument of 0.4 read backwards: a quantum source all of whose members but one have no parent and no other source is the same as a classical source.

The screening matters. Before it, five-node Fritz outputs look like promising common targets: 775 of them are reached from two or more classes, some from nine (classes 38 to 46). Outside the two groups below, every one of them (398 structures) is classically equivalent to a latent-free DAG, so it has no gap and proves nothing.

Of the 1,474 survivors, 1,334 are degradations that the search registers as lookups: structures weaker than something else on the list, never easier to prove. The other 140 are the 46 classes themselves and 94 of their outputs. Survivors reached from **two or more** open classes form exactly two groups:

* **Classes 1, 2, 3, 6, 7.** 523 common structures. The strongest is class 1 itself; the rest are outputs of its members and their degradations, such as `1→2, 1→3, 2→3; C{1,2,3}, Q{0,2}, Q{0,3}`. Nothing here improves on "prove class 1".
* **Classes 4 and 5.** Six common structures, three of them not degradations:
  * `1→2, 1→3; C{0,1,2}, C{0,1,3}, Q{0,2}, Q{0,3}` (from each class by a parent-predictor Fritz step, node 1 predicted by its parent 0, unsplit target, split predictor);
  * `0→1, 1→2, 1→3; C{0,1,2}, C{0,1,3}, Q{0,2}, Q{0,3}` and its five-node parent `0→1, 1→2, 1→3, 4→2, 4→3; C{0,1,2,4}, C{0,1,3,4}, Q{0,2}, Q{0,3}` (the same step with a split target).

  Class 5 is class 4 with a source added on $\lbrace 1,3\rbrace$; the two do not piggyback onto each other, but a gap in either hybrid settles both.

The remaining **39 classes share no candidate with any other class**. For each of them the only structures it reduces to, apart from provably gap-free ones, are its own Fritz outputs and degradations.

Three classes have small hybrid targets of their own that are worth knowing:

* **Class 5** reduces, by marginalizing node 0, to `0→1, 0→2; C{0,1}, Q{0,2}`: the Evans structure with one source classical (2.4).
* **Class 14** (five structures) has fifteen four-node hybrid outputs, for instance `0→2, 1→2, 2→3; C{0,1,2}, Q{1,3}`. Its three-node output `0→1, 1→2; C{1,2}, Q{0,1}` looks like a target and is not one: it is gap-free by the third criterion.
* **Class 2** has one: `2→3; C{0,2,3}, Q{0,2}, Q{0,3}, Q{1,2}, Q{1,3}, Q{2,3}`.

### 1.5 A second Fritz depth

The census applies the Fritz step once per input. To test whether that is what keeps the open classes open, the 94 intermediates that are neither degradations nor provably gap-free (one with three nodes, 34 with four, 59 with five) were given to the full cascade as inputs, with the seeds and the 388 further structures the census proves all counting as known, and outputs bounded at five visible nodes. **None is proven** (17 seconds). Within the piggybacks we have, the open classes are open at depth two as well.

### 1.6 The isolated classes, by family

Grouped by their directed edges, the open classes are a handful of families that differ only in which sources sit on a fixed skeleton:

| edges of the representative | classes |
|---|---|
| 2→3 | 1, 2 |
| 1→2, 2→3 | 3 |
| 0→1, 1→2, 1→3 | 4, 5 |
| 1→3, 2→3 | 6 |
| 1→2, 1→3, 2→3 | 7 |
| 0→3, 1→2 | 8 to 13 |
| 0→2, 1→2, 2→3 | 14 |
| 0→1, 1→2, 2→3 | 15 to 24 |
| 0→3, 1→2, 2→3 | 25 to 37 |
| 0→1, 1→2, 1→3, 2→3 | 38 to 46 |

Within a family no member piggybacks onto another. The relations that do exist inside families are the conditional ones of Section 2.3.

## 2. Equivalences

### 2.1 What an equivalence can buy

Three facts turn equivalences into gap transfers.

* **Quantum equivalence** ($\mathcal Q$ and $\mathcal C$ both equal) merges two classes into one problem. The census already counts by proven quantum classes (1.0).
* **The upward rule** (1.0): if $r$ is structurally above $b$ (more edges or sources) then $\mathcal Q(r)\supseteq\mathcal Q(b)$ for free; if in addition $\mathcal C(r)=\mathcal C(b)$, a gap of $b$ is a gap of $r$. Only the *classical* equivalence has to be proven.
* **Upgrading a source for free**: if a hybrid $H$ has the same quantum model as a structure $U$ obtained by making some of its classical sources quantum, then a gap of $U$ is a gap of $H$ ($\mathcal C(H)=\mathcal C(U)$ always). The lift of 0.4 is the proven instance.

### 2.2 Classical equivalences the metagraph already has: nothing to gain

One might hope that a classical equivalence which is not known to be quantum-valid, such as an uncertified HLP edge (1.0: 648 classical HLP edges against 528 certified quantumly), would merge an open class into a proven one if it were shown to hold quantumly. It cannot. Of the 396 classical classes on four nodes:

| quantum classes inside | classical classes | status of every quantum class inside |
|---|---|---|
| 1 | 20 | latent-free equivalent |
| 1 | 308 | proven |
| 1 | **46** | **open** |
| 2 | 12 | proven |
| 3 | 5 | proven |
| 4 | 2 | proven |
| 5 | 2 | proven |
| 7 | 1 | proven |

Every open class is the only quantum class of its classical class. The 42 splits the quantum rules introduce all lie inside classical classes that are entirely proven. Showing more HLP edges quantum-valid would tidy the proven part of the census and leave the 46 untouched. Any equivalence that helps has to be one the classical metagraph does not have.

### 2.3 New classical equivalences, through the upward rule

The classical classes of the metagraph are *proven* equivalences; two different classes may still be equal in truth. Take every pair ($r$ an open structure, $b$ a structure structurally below it) and ask whether $\mathcal C(r)=\mathcal C(b)$ can be excluded. Since $r$ is above $b$, $\mathcal C(r)\supseteq\mathcal C(b)$, and a single distribution, or support, compatible with $r$ and not with $b$ excludes it. The tests, in order: skeleton and d-separations, e-separations and densely connected pairs, then the infeasible supports of binary variables with 2 to 7 events (a support infeasible for $b$ and feasible for $r$ separates them).

| pairs | $b$ in a proven class | $b$ in another open class |
|---|---|---|
| structurally comparable | 3,129 | 392 |
| same skeleton and d-separations | 259 | 230 |
| same e-separations and densely connected pairs | 259 | 229 |
| same infeasible supports, 2 events | 42 | 229 |
| 3 events | 41 | 149 |
| 4 events | 31 | 70 |
| 5 events | 31 | 40 |
| 6 events | 1 | 25 |
| 7 events | **0** | **25** |

**Against the proven classes: ruled out.** The pairs that survive longest are the natural ones: classes 1, 3, 6 and 7 sit above members of the KPC class, which is proven, and class 2 above another proven class, and through 5 events nothing separates them. Six events do (117 infeasible supports for the class 1 structures against 147 for KPC), and seven events separate the last pair, class 6 above the KPC-class member `1→3, 2→3; Q{0,1}, Q{0,3}, Q{1,2}` (222 against 230). So the extra sources of these classes are not classically redundant, and no open class can be proven by showing it classically equal to a proven structure below it.

**Among the open classes: 24 pairs of classes survive every test** (25 pairs of structures; class 1 meets class 7 through two of its members). For each, if the classical equivalence holds, a gap of the lower class is a gap of the upper one. If in addition the quantum models are equal, the two classes are one problem. These are the cases the question "which equivalences hold classically and perhaps quantumly" comes down to on four nodes.

*The seven hard graphs.* Fifteen of the 24 pairs are inside the family `0→1, 1→2, 1→3, 2→3; Q{0,2}, Q{0,3}` plus sources among nodes 1, 2, 3: every comparable pair among classes 39, 40, 41, 43, 44, 45, 46 is unseparated. These seven are the "seven hard graphs" of the repository's earlier work on inequality constraints ([`Special Applications/the_7_hard_graphs.py`](../Special%20Applications/the_7_hard_graphs.py)). Their two family members without a source touching node 1, classes 38 and 42, *are* separated from them.

```mermaid
flowchart TB
    c41["41: + Q{1,2,3}"] -.-> c46["46: + Q{1,2}, Q{1,3}, Q{2,3}"]
    c46 -.-> c43["43: + Q{1,2}, Q{1,3}"]
    c46 -.-> c44["44: + Q{1,2}, Q{2,3}"]
    c46 -.-> c45["45: + Q{1,3}, Q{2,3}"]
    c43 -.-> c39["39: + Q{1,2}"]
    c43 -.-> c40["40: + Q{1,3}"]
    c44 -.-> c39
    c45 -.-> c40
```

A dashed arrow $x\to y$ reads "if $\mathcal C(x)=\mathcal C(y)$, a gap of $y$ proves $x$". If all hold, a gap for class 39 gives 43, 44, 46 and 41, and a gap for class 40 gives 43, 45, 46 and 41.

*A two-party source on an edge.* In the other nine pairs the upper class is the lower one plus a single two-party source placed on an existing visible edge:

| upper | lower | the extra source |
|---|---|---|
| 1 | 7 | Q{2,3} on 2→3 (in the member `1→2, 1→3, 2→3; Q{0,2}, Q{0,3}, Q{1,2}, Q{1,3}, Q{2,3}`) |
| 10 | 8 | Q{0,3} on 0→3 |
| 17 | 15 | Q{0,1} on 0→1 |
| 21 | 18 | Q{0,1} on 0→1 |
| 24 | 23 | Q{0,1} on 0→1 |
| 29 | 25 | Q{0,3} on 0→3 |
| 31 | 28 | Q{1,2} on 1→2 |
| 35 | 30 | Q{0,3} on 0→3 |
| 37 | 34 | Q{0,3} on 0→3 |

The metagraph removes a source $\lbrace a,b\rbrace$ on an edge $a\to b$ only when $a$ has no other parent (1.0). Here $a$ has one, so the rule does not apply, yet no support through 7 events tells the two structures apart. A proof that such a source is classically redundant in these nine cases would make eight classes consequences of others. The pair (1, 7) is special: class 7 already follows from class 1 by piggyback (1.2), so the classical equivalence would make them equivalent as problems.

*Counting.* With the four piggyback relations of 1.2, 42 problems are independent. If all 24 classical equivalences hold, the thirteen upper classes 10, 17, 21, 24, 29, 31, 35, 37, 41, 43, 44, 45 and 46 follow from lower ones, leaving **29**.

A caution on the evidence. "Not separated" means binary outcomes and at most 7 events. In the earlier work binary supports said nothing about the seven hard graphs and larger alphabets had to be tried, so these are open questions, not conjectures we have reason to believe. Their value is that each is a purely classical statement about two explicit structures.

### 2.4 Upgrading a classical source for free

A hybrid is a candidate for this when its all-quantum upgrade is a proven class. Among the structures the open classes reach, that happens for 39 hybrids, reached from classes 4, 5 and 14 only. Showing that the classical sources of one of them can be made quantum without enlarging the quantum model, an extension of the lift, would give it a gap.

| reached from | hybrid | its all-quantum upgrade (proven) |
|---|---|---|
| 4 and 5 | `0→1, 1→2, 1→3; C{0,1,2}, C{0,1,3}, Q{0,2}, Q{0,3}` | `0→1, 1→2, 1→3; Q{0,1,2}, Q{0,1,3}` |
| 4 and 5 | `1→2, 1→3; C{0,1,2}, C{0,1,3}, Q{0,2}, Q{0,3}` | `1→2, 1→3; Q{0,1,2}, Q{0,1,3}` |
| 5 | `0→1, 0→2; C{0,1}, Q{0,2}` | `0→1, 0→2; Q{0,1}, Q{0,2}` (Evans) |
| 5 | `0→1, 0→2, 3→1, 3→2; C{0,1,3}, C{0,2,3}, Q{0,2}` | `0→1, 0→2, 3→1, 3→2; Q{0,1,3}, Q{0,2,3}` |
| 14 | fifteen hybrids, e.g. `0→2, 1→2, 2→3; C{0,1,2}, Q{1,3}` | e.g. `0→2, 1→2, 2→3; Q{0,1,2}, Q{1,3}` |

The third row is the cleanest target in this document. The seed list takes the Evans structure as a gap only with both sources quantum ([`known_QC_gaps.py`](../known_QC_gaps.py): "with one source classical it is not known to have a gap and is not a seed"). A gap for Evans with one classical source, by any argument, proves class 5 at once, through a single marginalization.

For the rows of classes 4 and 5 the upgrade enlarges a source as well as recolouring it (the quantum part of `C{0,1,2}` is `Q{0,2}`), so equality of the quantum models is a strong statement and a direct gap proof for the hybrid may be the easier route.

The converse situation, a hybrid that is structurally above a proven structure with the same classical model, would be an immediate proof by the upward rule. There is none among the structures the open classes reach.

## 3. Where to put effort

1. **KPC with a source on $D$ and $E$ (class 1).** Five classes, 16 structures, and a known proof one source away. What has to be replaced is the d-separation $E\perp B\mid CD$.
2. **Evans with one classical source.** Three visible nodes, a question of independent interest, and class 5 follows by one marginalization. If the answer is no, class 5 needs another route, and that is worth knowing too.
3. **The hybrid shared by classes 4 and 5.** Two classes for one proof.
4. **The classical equivalences of 2.3.** Purely classical questions. The nine "source on an edge" cases look like one lemma; the seven hard graphs are a known hard family. Each proven equivalence removes one open problem without any quantum argument.
5. **The rest.** 26 to 39 classes, depending on item 4, each standing alone: for these the piggybacks reduce nothing to anything shared, and each needs either a new piggyback or its own construction.

## 4. Reproducing

```
python "Special Applications/open_classes_analysis.py" --five --events 7
```

The census takes 20 seconds, the hierarchy 15, the five-node metagraph behind the screening of 1.4 three minutes and 3.2 GB (`--five`; without it the five-node intermediates are reported unscreened), the second depth 20 seconds, and the supports test of 2.3 about 8 minutes through 6 events and about 12 more for the seventh (`--events`, default 4). Parts 1 to 4 of the output correspond to Sections 1.2, 1.4, 1.5 and 2.
