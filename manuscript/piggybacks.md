# Piggybacks for quantum-classical gaps

> Branch: **`main`**. This document describes this branch's rules and results. See the [branch guide](../BRANCHES.md) for the differences between the two maintained branches.

*What each structural transformation in this repository proves, why it is sound, what it contributes to the four-node census, and where it lives in the code.*

**Where to start.**

* For the results, read Section [1](#1-results-the-four-node-census): which classes of four-node structures are proven to have a quantum-classical gap, and which transformation proves what.
* For the vocabulary, read Section [0](#0-setting): structures, piggybacks, the two directions of every proof, and the names the code uses. Its last subsection, [0.8](#08-glossary), is a glossary.
* Sections 2 to 5 present the elementary transformations, one per section, each with its statement, its proof and its place in the code.
* Sections 6 to 8 present the Fritz piggyback: the mechanism, the theorem and worked examples (6), the stronger test by conditional independences (7), and how the search applies it (8).
* Section [9](#9-search-certificates-and-the-four-node-census-qc_gap_searchpy-special-applicationsproving_qc_gapspy) describes how the census is run, how to reproduce it and how to read its record. Section [10](#10-open-questions) lists what remains open.
* [Appendix A](#appendix-a-the-piggybacks-in-reverse) discusses the transformations in reverse. [Appendix B](#appendix-b-the-entropic-lp-as-an-alternative-certificate-engine-entropic_lppy-qmdag_entropic_certificate) presents the entropic linear program, the alternative to the test of Section [7](#7-the-semigraphoid-closure-certificate-for-the-fritz-piggyback-semigraphoidpy-qmdag_semigraphoid_certificate). [Appendix C](#appendix-c-five-nodes-the-two-metagraphs-and-their-counts) gives the counts of structures and classes on five nodes.

Two companion notes: [open_classes.md](open_classes.md) on how the 46 open classes relate, and [kpc_figure4_vs_code.md](kpc_figure4_vs_code.md) on the KPC structure and on why the search uses prediction and not perfect correlation.

<details><summary><b>Contents</b></summary>

<!-- contents -->
- [0. Setting](#0-setting)
  - [0.1 Structures and model sets](#01-structures-and-model-sets)
  - [0.2 Piggybacks](#02-piggybacks)
  - [0.3 Two general facts](#03-two-general-facts)
  - [0.4 The degradation piggyback (quantum source to classical source)](#04-the-degradation-piggyback-quantum-source-to-classical-source)
  - [0.5 When a classical source is as good as a quantum one](#05-when-a-classical-source-is-as-good-as-a-quantum-one)
  - [0.6 Quantum observational equivalence and the two metagraphs](#06-quantum-observational-equivalence-and-the-two-metagraphs)
  - [0.7 The census, the seeds and the names of the tricks](#07-the-census-the-seeds-and-the-names-of-the-tricks)
  - [0.8 Glossary](#08-glossary)
- [1. Results: the four-node census](#1-results-the-four-node-census)
  - [1.0 The four-node counts, and why results are counted by class](#10-the-four-node-counts-and-why-results-are-counted-by-class)
  - [1.1 The census over quantum classes](#11-the-census-over-quantum-classes)
- [2. Point distribution (`PD`)](#2-point-distribution-pd)
- [3. Marginalization, naive and with teleportation (`naive_marginalization`, `teleportation_marginalization`)](#3-marginalization-naive-and-with-teleportation-naive_marginalization-teleportation_marginalization)
- [4. Postselection (`conditioning`)](#4-postselection-conditioning)
- [5. Node stitching (`node_stitching`)](#5-node-stitching-node_stitching)
- [6. The Fritz piggyback (`Fritz`)](#6-the-fritz-piggyback-fritz)
  - [6.1 The mechanism](#61-the-mechanism)
  - [6.2 The unit: a predictor and its targets](#62-the-unit-a-predictor-and-its-targets)
  - [6.3 Example: the triangle, Fritz's original argument](#63-example-the-triangle-fritzs-original-argument)
  - [6.4 Node splitting; split and unsplit predictors, split and unsplit targets (`QmDAG.split_node`)](#64-node-splitting-split-and-unsplit-predictors-split-and-unsplit-targets-qmdagsplit_node)
  - [6.5 The parent predictor: deletion without a certificate](#65-the-parent-predictor-deletion-without-a-certificate)
  - [6.6 Theorem (d-separation certificate)](#66-theorem-d-separation-certificate)
  - [6.7 Example: the tetrahedron, unsplit target and unsplit predictor](#67-example-the-tetrahedron-unsplit-target-and-unsplit-predictor)
  - [6.8 Example: deleting a visible edge, three routes](#68-example-deleting-a-visible-edge-three-routes)
  - [6.9 Example: a split target with a childful unsplit predictor](#69-example-a-split-target-with-a-childful-unsplit-predictor)
  - [6.10 Example: a split predictor and a split target, worked out in full](#610-example-a-split-predictor-and-a-split-target-worked-out-in-full)
- [7. The semigraphoid closure certificate for the Fritz piggyback (`semigraphoid.py`, `QmDAG._semigraphoid_certificate`)](#7-the-semigraphoid-closure-certificate-for-the-fritz-piggyback-semigraphoidpy-qmdag_semigraphoid_certificate)
  - [7.1 Why the same mechanism admits a stronger certificate](#71-why-the-same-mechanism-admits-a-stronger-certificate)
  - [7.2 Conditional independence as the proof system: semigraphoids and the exchange rule (`semigraphoid.py`)](#72-conditional-independence-as-the-proof-system-semigraphoids-and-the-exchange-rule-semigraphoidpy)
  - [7.3 The hypotheses, and how perfect prediction enters (`QmDAG._semigraphoid_model`, `semigraphoid.add_functional_dependence`)](#73-the-hypotheses-and-how-perfect-prediction-enters-qmdag_semigraphoid_model-semigraphoidadd_functional_dependence)
  - [7.4 The two target sets, as containment of d-separation models (`QmDAG._semigraphoid_certificate`)](#74-the-two-target-sets-as-containment-of-d-separation-models-qmdag_semigraphoid_certificate)
  - [7.5 Worked example: the KPC structure](#75-worked-example-the-kpc-structure)
  - [7.6 Why `relabel` is justified](#76-why-relabel-is-justified)
  - [7.7 Theorem (semigraphoid closure certificate)](#77-theorem-semigraphoid-closure-certificate)
  - [7.8 Further examples from the census](#78-further-examples-from-the-census)
  - [7.9 Strength, agreement with the LP, and cost](#79-strength-agreement-with-the-lp-and-cost)
- [8. The Fritz search procedure (`QmDAG.fritz_steps`, `qc_gap_search.fritz_tricks`)](#8-the-fritz-search-procedure-qmdagfritz_steps-qc_gap_searchfritz_tricks)
  - [8.1 The pool and its order (`QmDAG.fritz_pool`, `QmDAG.fritz_targets`)](#81-the-pool-and-its-order-qmdagfritz_pool-qmdagfritz_targets)
  - [8.2 What a predictor sees, and the deletion it dictates (`QmDAG.fritz_deletion`)](#82-what-a-predictor-sees-and-the-deletion-it-dictates-qmdagfritz_deletion)
  - [8.3 Justifying the deletion: one set, two certificates (`QmDAG.fritz_certificate`)](#83-justifying-the-deletion-one-set-two-certificates-qmdagfritz_certificate)
  - [8.4 Parent predictors: the free piggyback as its own stage](#84-parent-predictors-the-free-piggyback-as-its-own-stage)
  - [8.5 Joint steps, and why not a second Fritz depth](#85-joint-steps-and-why-not-a-second-fritz-depth)
  - [8.6 Modes and the surgery (`QmDAG._restrict_targets`, `QmDAG.fritz_realise`)](#86-modes-and-the-surgery-qmdag_restrict_targets-qmdagfritz_realise)
  - [8.7 Two kinds of stage: d-separation breadth-first, then the semigraphoid closure with early exit](#87-two-kinds-of-stage-d-separation-breadth-first-then-the-semigraphoid-closure-with-early-exit)
  - [8.8 What the procedure decides, and what it does not](#88-what-the-procedure-decides-and-what-it-does-not)
  - [8.9 The six classes that need a Fritz step](#89-the-six-classes-that-need-a-fritz-step)
  - [8.10 Experiment: descendants as predictors](#810-experiment-descendants-as-predictors)
- [9. Search, certificates and the four-node census (`qc_gap_search.py`, `Special Applications/proving_QC_Gaps.py`)](#9-search-certificates-and-the-four-node-census-qc_gap_searchpy-special-applicationsproving_qc_gapspy)
  - [9.1 The ten stages](#91-the-ten-stages)
  - [9.2 Reading the tables, and why the seeds matter](#92-reading-the-tables-and-why-the-seeds-matter)
  - [9.3 Certificates beyond d-separation attempted](#93-certificates-beyond-d-separation-attempted)
  - [9.4 Why node stitching and naive marginalization show few exclusive proofs](#94-why-node-stitching-and-naive-marginalization-show-few-exclusive-proofs)
  - [9.5 The known-gap database the search establishes](#95-the-known-gap-database-the-search-establishes)
  - [9.6 Bookkeeping: the explorer, the cache and the record](#96-bookkeeping-the-explorer-the-cache-and-the-record)
  - [9.7 Reproducing the results](#97-reproducing-the-results)
- [10. Open questions](#10-open-questions)
  - [10.1 The 46 open classes](#101-the-46-open-classes)
  - [10.2 Open questions](#102-open-questions)
  - [10.3 Variants tried, settled or left out](#103-variants-tried-settled-or-left-out)
- [Appendix A. The piggybacks in reverse](#appendix-a-the-piggybacks-in-reverse)
  - [A.0 Why the search runs forward](#a0-why-the-search-runs-forward)
  - [A.1 Degradation in reverse: quantumification](#a1-degradation-in-reverse-quantumification)
  - [A.2 Point distribution in reverse: node insertion](#a2-point-distribution-in-reverse-node-insertion)
  - [A.3 Marginalization in reverse: node insertion along edges and facets](#a3-marginalization-in-reverse-node-insertion-along-edges-and-facets)
  - [A.4 Postselection in reverse: un-swapping entanglement](#a4-postselection-in-reverse-un-swapping-entanglement)
  - [A.5 Node stitching in reverse: interruption](#a5-node-stitching-in-reverse-interruption)
  - [A.6 The Fritz piggyback in reverse](#a6-the-fritz-piggyback-in-reverse)
  - [A.7 Node splitting in reverse: merging](#a7-node-splitting-in-reverse-merging)
  - [A.8 What a reverse closure would compute](#a8-what-a-reverse-closure-would-compute)
- [Appendix B. The entropic LP as an alternative certificate engine (`entropic_lp.py`, `QmDAG._entropic_certificate`)](#appendix-b-the-entropic-lp-as-an-alternative-certificate-engine-entropic_lppy-qmdag_entropic_certificate)
  - [B.0 Why the LP is not the default engine](#b0-why-the-lp-is-not-the-default-engine)
  - [B.1 Entropy vectors as the proof system (`entropic_lp.py`)](#b1-entropy-vectors-as-the-proof-system-entropic_lppy)
  - [B.2 The hypotheses and the target sets as LP rows (`QmDAG._entropic_certificate`)](#b2-the-hypotheses-and-the-target-sets-as-lp-rows-qmdag_entropic_certificate)
  - [B.3 Theorem (entropic certificate)](#b3-theorem-entropic-certificate)
  - [B.4 LP size and time](#b4-lp-size-and-time)
  - [B.5 The KPC example and the "decisive" diagnostic](#b5-the-kpc-example-and-the-decisive-diagnostic)
  - [B.6 Relation to the closure, and what is specific to the LP](#b6-relation-to-the-closure-and-what-is-specific-to-the-lp)
- [Appendix C. Five nodes: the two metagraphs and their counts](#appendix-c-five-nodes-the-two-metagraphs-and-their-counts)
- [References](#references)
<!-- /contents -->

</details>

Section numbers and backticked code names are links: a section number leads to the section, a code name to the line of the repository where it is defined. `python manuscript/refresh_links.py` rebuilds both after the headings or the code change.

---

## 0. Setting

This section fixes the vocabulary. [0.1](#01-structures-and-model-sets) to [0.3](#03-two-general-facts) define the structures, the piggybacks and two facts used throughout. [0.4](#04-the-degradation-piggyback-quantum-source-to-classical-source) presents the simplest piggyback, degradation, and [0.5](#05-when-a-classical-source-is-as-good-as-a-quantum-one) the cases in which a degradation changes nothing. [0.6](#06-quantum-observational-equivalence-and-the-two-metagraphs) defines the equivalence classes that the census counts. [0.7](#07-the-census-the-seeds-and-the-names-of-the-tricks) defines the census, the seeds and the names of the tricks. [0.8](#08-glossary) is a glossary.

### 0.1 Structures and model sets

A **QmDAG** ([`quantum_mDAG.QmDAG`](../quantum_mDAG.py#L101)) is a causal structure with visible nodes $V=\lbrace 0,\dots,n-1\rbrace,$ a directed structure on $V,$ and two simplicial complexes of **latent facets**: classical facets $\mathcal C$ and quantum facets $\mathcal Q.$ A facet $F\subseteq V$ stands for one exogenous latent, a classical random variable or a quantum state, whose children are exactly the nodes of $F.$ Every visible node additionally has implicit private randomness. A classical facet contained in a quantum facet is redundant and is dropped by the constructor. A **hybrid** is a QmDAG with at least one classical facet. An **mDAG** is the classical object: a directed structure with a single complex of facets and no distinction between classical and quantum. Structures are written as their edges, then their facets, Q for quantum and C for classical: 0→1→2; Q{0,2}, C{1,3}.

For a structure $G$ write $\mathcal C(G)$ for the set of distributions over $V$ realisable with all latents classical (a Bayesian network with latent variables), and $\mathcal Q(G)$ for the set realisable when the quantum facets carry quantum states on which their children perform measurements, classical facets and visible edges remaining classical channels. Always $\mathcal C(G)\subseteq\mathcal Q(G).$ The structure $G$ has a **QC gap** when $\mathcal Q(G)\not\subseteq\mathcal C(G).$

The **effective DAG** of a QmDAG ([`QmDAG.effective_DAG_data`](../quantum_mDAG.py#L485)) is the ordinary DAG on the visible nodes, one node per facet and one private-noise node per visible node; [`QmDAG.lp_structure`](../quantum_mDAG.py#L522) is the same without the noise nodes. A "parent" of a visible node below is a parent in the effective DAG: a visible parent or a facet containing the node.

### 0.2 Piggybacks

A **piggyback** is a map $G\mapsto G'$ together with a proof of

```math
\mathcal Q(G')\not\subseteq\mathcal C(G')\;\Longrightarrow\;\mathcal Q(G)\not\subseteq\mathcal C(G).
```

Every proof in this document has the same two halves.

* **The quantum direction.** A quantum model of $G$ can produce every quantum distribution of $G'$ in disguise. Given $P'\in\mathcal Q(G')\setminus\mathcal C(G'),$ build $P\in\mathcal Q(G)$ from which $P'$ is recovered by a fixed operation $\pi$ (a marginal, a conditional, a coarse-graining, or the identity): $\pi(P)=P'.$ The construction may give $P$ extra observable properties, such as one node perfectly predicting another.
* **The classical direction.** If $P$ were classical in $G,$ then $P'$ would be classical in $G'.$ Show that every classical model of $G$ for a distribution with those observable properties yields a classical model of $G'$ for $\pi(P).$

Then $P\in\mathcal C(G)$ would give $P'=\pi(P)\in\mathcal C(G'),$ a contradiction, so $P\in\mathcal Q(G)\setminus\mathcal C(G).$ Below, the *constructed distribution* is a $P$ that the quantum direction produces.

All piggybacks here are **caveat-free**: the hypothesis is a QC gap of $G'$ as a structure, with no side condition on the gap-witnessing distribution. This is what lets them be composed in any order and searched as a closure (Section [9](#9-search-certificates-and-the-four-node-census-qc_gap_searchpy-special-applicationsproving_qc_gapspy)). Structures are identified up to relabelling by [`unique_unlabelled_id`](../quantum_mDAG.py#L177), and every count in this document is a count of distinct structures up to relabelling.

### 0.3 Two general facts

* **(F1) Edge monotonicity.** If $G'$ is obtained from $G$ only by deleting visible edges and removing nodes from facets, then $\mathcal Q(G')\subseteq\mathcal Q(G)$ and $\mathcal C(G')\subseteq\mathcal C(G)$: run the same strategy and ignore the extra inputs. For an edge-deleting piggyback the quantum direction is therefore immediate and the whole content is the classical direction.
* **(F2) d-separation.** Among the *visible* variables, every d-separation of $G$ implies the corresponding conditional independence, in classical and in quantum models alike: observable d-separation is theory-independent (Henson, Lal and Pusey). Conversely no other conditional independence among visible variables holds in every classical model, hence none holds in every quantum model either, since $\mathcal C(G)\subseteq\mathcal Q(G).$ Conditional independences that *involve latents* are statements about a joint distribution of visible and latent variables. Such a joint exists for a classical model, where every node of the effective DAG is a random variable, but not in general for a quantum one: a quantum facet is a state, not a variable, and only coexisting sets of nodes have a joint distribution (Chaves, Majenz and Gross). The proofs of the classical direction below reason about classical models only, so they may use latent-involving d-separations and conditional independences of the effective DAG freely (and its entropies, [Appendix B](#appendix-b-the-entropic-lp-as-an-alternative-certificate-engine-entropic_lppy-qmdag_entropic_certificate)).

### 0.4 The degradation piggyback (quantum source to classical source)

**Map.** $G'$ is $G$ with some quantum facets made classical. Equivalently, $G$ is $G'$ with some classical facets upgraded to quantum, or with quantum facets added inside existing classical facets (a quantum facet contained in a classical one is a special case of upgrading part of that classical facet).

**The classical direction.** $\mathcal C(G)=\mathcal C(G')$: in a classical model every facet, quantum or classical, is one latent random variable with the facet as its children, and a latent whose facet lies inside another facet's is redundant, so the two structures have the same classical models.

**The quantum direction.** $\mathcal Q(G')\subseteq\mathcal Q(G)$: a classical source is a quantum state that happens to be diagonal.

Hence $\mathcal Q(G')\not\subseteq\mathcal C(G')$ implies $\mathcal Q(G)\not\subseteq\mathcal C(G).$ $\square$

**What the piggyback says.** This is the facet counterpart of (F1) and the only piggyback that changes no edge and no node. It says that a QC gap is **monotone under upgrading sources**: the structures worth recording are the *weakest* ones, with as many facets classical as the gap allows, and every structure above them in the upgrade order is a gap for free.

**Example.** Take G(3) of [6.10](#610-example-a-split-predictor-and-a-split-target-worked-out-in-full), written with the copy 3′ relabelled 3: 0→1→2; Q{0,1}, Q{0,2}, C{0,1,3}, C{0,1,2}. It has a gap (one teleportation marginalization to [`QG_Instrumental_C`](../known_QC_gaps.py#L41)), hence so do

* 0→1→2; Q{0,1}, Q{0,2}, Q{1,2}, C{0,1,3}, C{0,1,2} (a quantum facet added inside C{0,1,2});
* 0→1→2; Q{0,1}, Q{0,2}, Q{1,2}, Q{0,3}, C{0,1,3}, C{0,1,2}, and likewise with Q{1,3}, or with both;
* 0→1→2; Q{0,2}, Q{1,2}, Q{0,1,3}, C{0,1,2} (C{0,1,3} upgraded; it absorbs Q{0,1});
* 0→1→2; C{0,1,3}, Q{0,1,2} (C{0,1,2} upgraded; it absorbs Q{0,1} and Q{0,2});
* and so on up to 0→1→2; Q{0,1,3}, Q{0,1,2}, every source quantum.

**How the search applies it: as a lookup.** A **degradation** of a structure is that structure with some of its quantum facets made classical (the $G'$ of the map). The search applies the piggyback as a **lookup**, not as an expansion ([`degradation`](../qc_gap_search.py#L77), [`ClosureExplorer.lookup_tricks`](../qc_gap_search.py#L222)): every structure the search meets, whether an input of the census ([0.7](#07-the-census-the-seeds-and-the-names-of-the-tricks)) or an intermediate structure, has its degradations registered as transitions to structures that are never expanded. A structure is therefore known as soon as any of its degradations is, at the cost of one id computation per degradation and no search.

Expanding the degradations would not help. It would multiply the structures to explore by up to $2^k$ for a structure with $k$ quantum facets (one degradation for each choice of the facets made classical), and it could prove nothing new, since a degradation is harder to prove than the structure it comes from.

**The seeds are kept in weakest form.** The list of seeds, the structures with an established QC gap ([0.7](#07-the-census-the-seeds-and-the-names-of-the-tricks)), is kept in **weakest form only** ([`known_QC_gaps.SEEDS`](../known_QC_gaps.py#L108): a named gap is dropped when another named gap is a degradation of it, and one representative per relabelling class is kept), so [`QG_Triangle`](../known_QC_gaps.py#L42) stands for all three triangle colourings. Certificates end with the degradation step that reaches the weakest form.

### 0.5 When a classical source is as good as a quantum one

Some degradations change nothing: the classical facet gives the same model sets as the quantum facet, so the classical source can be replaced by a quantum one for free. The code name of the rule that recognises these facets is [`QmDAG.lift`](../quantum_mDAG.py#L235). The notion it rests on, a premeasurable node, is also what the edge rule for quantum models in [0.6](#06-quantum-observational-equivalence-and-the-two-metagraphs) rests on.

**Premeasurable nodes.** Call a node *premeasurable* when it has no visible parent and exactly one facet ([`mDAG.is_premeasurable`](../mDAG_advanced.py#L592)): its measurement is then fixed, and the facet's source can perform it in the node's place, since measuring one share before distribution changes no statistics.

**The rule.** A classical facet all of whose members but at most one are premeasurable is *the same* as the quantum facet:

1. the source pre-measures those members' shares and hands them their outcomes;
2. the remaining member's share is then a classical mixture labelled by outcomes the source can tell it as well;
3. a measurement on such a share, correlated with classical values the member knows, adds nothing a classical common cause does not.

Making such a facet classical shrinks neither model set, and the two structures are quantum-equivalent as well as classically equivalent (they have the same quantum and the same classical model set, [0.6](#06-quantum-observational-equivalence-and-the-two-metagraphs)). The 2-facet $`\{a,b\}`$ with $a$ premeasurable is the simplest case.

**In the code.** [`QmDAG.lift`](../quantum_mDAG.py#L235) returns the all-quantum structure of a hybrid whose classical facets are all of this kind, and the census identifies the two ([0.6](#06-quantum-observational-equivalence-and-the-two-metagraphs)).

**Which seeds the rule covers.** The rule applies to every Bell seed and every instrumental seed: the setting facets C{0,2}, C{1,3} and C{0,1} of the Bell family sit on settings, which have no parents and no other facet, and so does C{0,1} of the instrumental variants. These seeds therefore *are* the quantum classes of their all-quantum versions. The rule does not apply to [`QG_Triangle`](../known_QC_gaps.py#L42), since every node of the triangle holds two facets: this seed remains a degradation target (a structure that others reach by the degradation lookup of [0.4](#04-the-degradation-piggyback-quantum-source-to-classical-source)), strictly weaker than the all-quantum triangle.

### 0.6 Quantum observational equivalence and the two metagraphs

The census ([0.7](#07-the-census-the-seeds-and-the-names-of-the-tricks)) counts classes of equivalent structures, not single structures. This subsection defines the two equivalences, states the rules that prove them, and says how the code turns the rules into two graphs, the metagraphs.

**The two equivalences.** Two structures are **classically observationally equivalent** when they have the same classical model set, **quantum observationally equivalent** when they have the same quantum model set.

**The metagraph and its three rules.** The repository's **metagraph** ([`metagraph_advanced.py`](../metagraph_advanced.py)) proves equivalences of the first kind by sound containments of model sets: its nodes are the mDAGs up to relabelling, its edges three kinds of relation, and every strongly connected component is a proven equivalence class (an upper bound on the true number of classes: two components may still be equivalent by an argument the rules do not capture). The three rules are:

* **Structural dominance.** Adding an edge, adding a facet or enlarging a facet can only enlarge a model set, classical or quantum. These edges point from the stronger to the weaker structure, and they are the only edges that are *not* equivalences; a cycle through them is one.
* **Face splitting** (Evans). A facet $C\cup D$ may be split into the facets $C$ and $D$ when every $d\in D$ sees everything $C$ sees: $\mathrm{Pa}^+(C)\subseteq\mathrm{Pa}^+(d)$ for all $d\in D$ (parents, latent and visible, plus the nodes of $C$ themselves). The split structure is contained in the unsplit one trivially; the converse is Evans' lemma classically, and holds quantumly as well: the source of $D$ distributes, for every value of what the members of $D$ see of $C$ (the outputs of $C$ and their settings), the conditional state the joint source would have left on $D$'s shares given that value, and each $d$ uses the component indexed by what it sees, while the source of $C$ distributes the reduced state of $C$'s shares; the joint statistics are unchanged. When $C$ or $D$ is a single node whose facet is thereby reduced to a singleton, the facet disappears: a facet $`\{a,b\}`$ on an edge $a\to b$ with $a$ having no other parent is redundant.
* **The HLP edge rule** (Henson, Lal and Pusey; Evans). An edge $a\to b$ between two members of a facet with $\mathrm{Pa}(a)\subseteq\mathrm{Pa}(b)$ may be added or dropped: classically $b$ sees everything $a$ is a function of, and the latent can carry $a$'s noise, so $b$ can compute $a.$ For quantum models the rule needs more, as the next paragraphs explain.

**The edge rule for quantum models: the pre-measurement argument.** Quantumly the HLP edge rule holds when $b$ can obtain $a$'s value without the edge, which the **pre-measurement argument** gives in two forms ([`mDAG.quantum_hlp_origin_ok`](../mDAG_advanced.py#L598)):

1. **First form.** The node $a$ is premeasurable (no visible parent, one facet, [0.5](#05-when-a-classical-source-is-as-good-as-a-quantum-one)): its measurement is fixed, so the facet's source performs it and hands the outcome to $a$ and to $b.$
2. **Second form.** Every member of every facet of $a$ other than $a$ and $b$ is premeasurable: each source pre-measures those members and tells $a$ and $b$ their outcomes, after which what is left of $a$'s quantum resources is shared with $b$ alone, and $b,$ which holds every input of $a$ (its visible parents, its facets and those outcomes), produces the pair's joint conditional from shared classical randomness and so computes $a$'s output itself. This form includes the facet $`\{a,b\}`$ with no other member whatever visible parents $a$ has (2→0, 2→1, 0→1; Q{0,1} is such a case).

**Where the argument fails.** Outside these two forms the argument fails: a setting of $a$ that the source does not know, or a member of $a$'s facet that the source cannot pre-measure (a visible parent of its own, or a second facet it can measure jointly with this one), leaves $a$'s outcome correlated with a share $b$ does not hold. Two counterexamples:

* **With a setting.** Q{A,B,C}, S→A, S→B, T→C, where Pa(A) ⊆ Pa(B). With the edge A→B, B outputs a copy of A's outcome while A and C violate CHSH maximally. No strategy without the edge reproduces this, since B's share is then uncorrelated with A's outcome by monogamy.
* **With a second facet.** Take the triangle Q{A,B}, Q{B,C}, Q{A,C}, which has a gap (Fritz's distribution embeds a maximal CHSH violation between A and B, [6.3](#63-example-the-triangle-fritzs-original-argument)), and enlarge two facets to Q{A,B,D}, Q{A,C,D}. Then $\mathrm{Pa}(A)=\mathrm{Pa}(D),$ so the classical rule adds the edge A→D for free. With the edge, D outputs a copy of A's outcome while A and B produce Fritz's distribution. Without it, D would have to predict A's outcome for both of A's settings from its shares, and the monogamy of nonlocality with B as the common party (Toner and Verstraete: $`\mathrm{CHSH}_{BA}^2+\mathrm{CHSH}_{BD}^2\le 8,`$ and perfect prediction makes the two equal) caps the A–B violation at the classical bound. So the quantum set grows when the edge is added, although the classical set does not.

**The open question.** Whether the rule extends to an $a$ whose facet has a member with two facets but no visible parent (for instance 2→1 in 0→1, 0→2, 0→3, 2→3; Q{0,1,2}, Q{0,3}, where 0 holds shares of both facets) is open: the pre-measurement argument does not apply, since 0 can measure its two shares jointly, and no counterexample is known.

**The two metagraphs.** The **quantum metagraph** ([`QuantumMetagraph`](../metagraph_advanced.py#L1228)) uses structural dominance, face splitting and the HLP rule in its pre-measurement form; its components are proven *quantum* equivalence classes, each inside one classical class. The **classical metagraph** ([`ClassicalMetagraph.from_quantum`](../metagraph_advanced.py#L1241)) is the same graph with the unrestricted HLP edges added. The two are built together, the classical one for the price of its extra edges. Their sizes on three and on four nodes are in [1.0](#10-the-four-node-counts-and-why-results-are-counted-by-class).

The HLP edges generated under the two rules are not nested sets, since the rule drops every droppable edge at once and only when nothing is addable (an edge is droppable or addable when the rule allows dropping or adding it). But every quantum edge is a classical equivalence, and [`tests/test_metagraph.py`](../tests/test_metagraph.py) checks that every quantum class lies inside one classical class.

**The upward rule.** One more rule uses both kinds of class; it is about gaps, not about equivalence. Let $G$ have a QC gap. If $\mathcal C(G')=\mathcal C(G)$ and $G'$ is structurally above $G$ (more edges or facets), then $\mathcal Q(G')\supseteq\mathcal Q(G)\not\subseteq\mathcal C(G)=\mathcal C(G'),$ so $G'$ has a gap too. The search applies the rule as a lookup (`upward`): a class is known as soon as a class structurally below it inside the same classical class is. Whether a gap also passes downward inside a classical class is open ([1.0](#10-the-four-node-counts-and-why-results-are-counted-by-class)).

**Provably algebraic classes.** The census ([0.7](#07-the-census-the-seeds-and-the-names-of-the-tricks)) leaves out the classes that are *provably algebraic*: those with a member classically equivalent to a latent-free DAG, a DAG with no latent facet. Such a class is excluded from the inputs because its members have $\mathcal Q(G)=\mathcal C(G)$: the quantum model set always satisfies the observable d-separations of $G,$ which coincide with those of the equivalent DAG, and the DAG's classical model is everything that satisfies them.

**Structures outside the metagraphs.** Hybrid structures (some facets classical) are not nodes of the metagraph. A hybrid to which the rule [`QmDAG.lift`](../quantum_mDAG.py#L235) applies ([0.5](#05-when-a-classical-source-is-as-good-as-a-quantum-one)) is identified with the all-quantum structure that the rule returns, which has the same quantum model set; any other hybrid is treated by its own id and the degradation lookup ([`equivalence_classes.ClassIndex`](../equivalence_classes.py#L31)). Structures of five visible nodes, which the Fritz trick produces, are not looked up in a metagraph and are likewise treated by id: the class index covers three and four nodes, and the five-node metagraphs of [Appendix C](#appendix-c-five-nodes-the-two-metagraphs-and-their-counts) are built for their counts only.

### 0.7 The census, the seeds and the names of the tricks

This subsection fixes four terms used from Section [1](#1-results-the-four-node-census) on: the census, the seeds, what "proven" means, and the names of the tricks.

* **The census** (Sections [1](#1-results-the-four-node-census) and [9](#9-search-certificates-and-the-four-node-census-qc_gap_searchpy-special-applicationsproving_qc_gapspy)) is the set of **quantum observational equivalence classes** of four-node mDAGs with every latent facet quantum that are not provably algebraic (no member is classically equivalent to a latent-free DAG; [0.6](#06-quantum-observational-equivalence-and-the-two-metagraphs) defines both terms). These classes are the **inputs**. A class is proven as soon as any of its members is, since all members have the same quantum and the same classical model set.
* **The seeds** ([`known_QC_gaps.py`](../known_QC_gaps.py)) are structures with an established QC gap: the instrumental, triangle and Evans variants with three visible nodes, and the Bell variants with four. A seed is understood as the quantum class of the all-quantum structure that [`QmDAG.lift`](../quantum_mDAG.py#L235) returns for it ([0.5](#05-when-a-classical-source-is-as-good-as-a-quantum-one)): the three instrumental variants are one class, the nine Bell variants two (with and without a link between the settings). The rule [`QmDAG.lift`](../quantum_mDAG.py#L235) does not apply to the triangle seed, which stays a degradation target. Every seed is known from the start, and the two Bell classes are inputs, proven by the **identity** piggyback. The table below the list collects the seeds.
* **Proven.** An input is **proven** when a chain of piggybacks leads from any of its members to a seed or to an input already proven; a proven class is a **known gap**. "Provable only through $T$" means that the input is no longer proven when every transition of trick $T$ is removed from the record and everything else is kept ([9.2](#92-reading-the-tables-and-why-the-seeds-matter)).
* **The tricks** are the piggybacks as the search applies them. Their code names:
  * `identity`: the input is a seed class.
  * The **elementary reductions**, which lower the number of visible nodes: `PD` (point distribution, Section [2](#2-point-distribution-pd)), `naive_marginalization` and `teleportation_marginalization` (the two marginalizations, Section [3](#3-marginalization-naive-and-with-teleportation-naive_marginalization-teleportation_marginalization)), `conditioning` (postselection, Section [4](#4-postselection-conditioning)) and [`node_stitching`](../quantum_mDAG.py#L341) (node stitching, Section [5](#5-node-stitching-node_stitching)).
  * The **lookups**: [`degradation`](../qc_gap_search.py#L77) ([0.4](#04-the-degradation-piggyback-quantum-source-to-classical-source)) and `upward` ([0.6](#06-quantum-observational-equivalence-and-the-two-metagraphs): a class is a gap when a class structurally below it inside the same classical class is).
  * The single trick `Fritz` (Sections [6](#6-the-fritz-piggyback-fritz) and [7](#7-the-semigraphoid-closure-certificate-for-the-fritz-piggyback-semigraphoidpy-qmdag_semigraphoid_certificate)). It rests on one node, the **predictor**, perfectly predicting another, the **predicted node**. The predicted node is also called the **target**: the two names mean the same node. The transitions of this trick record the target mode ([`split`](../canonical_forms.py#L206) or `unsplit`), the predictor mode ([`split`](../canonical_forms.py#L206) or `unsplit`) and the certificate: `parent` for the parent predictor of [6.5](#65-the-parent-predictor-deletion-without-a-certificate) (no certificate needed), `dsep` for d-separation and `semigraphoid` for the semigraphoid closure (`entropic` when the LP of [Appendix B](#appendix-b-the-entropic-lp-as-an-alternative-certificate-engine-entropic_lppy-qmdag_entropic_certificate) is the engine). The search runs the Fritz trick in **stages** named after these parameters (Section [9](#9-search-certificates-and-the-four-node-census-qc_gap_searchpy-special-applicationsproving_qc_gapspy)).

The seeds, as named in [`known_QC_gaps.py`](../known_QC_gaps.py). In the instrumental seeds node 0 is the instrument (the setting), 1 the treatment and 2 the outcome; in the Bell seeds 0 and 1 are the settings and 2 and 3 the outcomes.

| seed class | named structures (edges; facets) | remarks |
|---|---|---|
| instrumental | [`QG_Instrumental_Edge`](../known_QC_gaps.py#L39): 0→1, 1→2; Q{1,2} | one quantum class of three structures ([1.1](#11-the-census-over-quantum-classes)): the rule [`QmDAG.lift`](../quantum_mDAG.py#L235) applies to the classical facet C{0,1}, which sits on the setting |
| | [`QG_Instrumental_EdgeC`](../known_QC_gaps.py#L40): 0→1, 1→2; C{0,1}, Q{1,2} | |
| | [`QG_Instrumental_C`](../known_QC_gaps.py#L41): 1→2; C{0,1}, Q{1,2} | |
| triangle | [`QG_Triangle`](../known_QC_gaps.py#L42): no edges; C{0,2}, C{1,2}, Q{0,1} | the weakest form, standing for all three triangle colourings ([0.4](#04-the-degradation-piggyback-quantum-source-to-classical-source)); the rule does not apply (every node holds two facets), so this seed is a hybrid, strictly weaker than the all-quantum triangle, and is reached by the degradation lookup |
| Evans | [`QG_Evans`](../known_QC_gaps.py#L43): 0→1, 0→2; Q{0,1}, Q{0,2} | all quantum |
| Bell, no link between the settings | [`QG_Bell_C_C`](../known_QC_gaps.py#L78): no edges; C{0,2}, C{1,3}, Q{2,3} | one quantum class, an input, proven by `identity`; each setting reaches its outcome by an edge, by a classical facet or by both |
| | [`QG_Bell_C_EdgeC`](../known_QC_gaps.py#L79): 1→3; C{0,2}, C{1,3}, Q{2,3} | |
| | [`QG_Bell_C_Edge`](../known_QC_gaps.py#L80): 1→3; C{0,2}, Q{2,3} | |
| | [`QG_Bell_EdgeC_EdgeC`](../known_QC_gaps.py#L81): 0→2, 1→3; C{0,2}, C{1,3}, Q{2,3} | |
| | [`QG_Bell_EdgeC_Edge`](../known_QC_gaps.py#L82): 0→2, 1→3; C{0,2}, Q{2,3} | |
| | [`QG_Bell_Edge_Edge`](../known_QC_gaps.py#L83): 0→2, 1→3; Q{2,3} | |
| Bell, settings linked (correlated settings) | [`QG_Bell_Edge_Edge_SettingsEdge`](../known_QC_gaps.py#L84): 0→1, 0→2, 1→3; Q{2,3} | one quantum class, an input, proven by `identity`; the settings are linked by an edge, by a classical facet or by both (the example of [1.0](#10-the-four-node-counts-and-why-results-are-counted-by-class) lists the all-quantum members) |
| | [`QG_Bell_Edge_Edge_SettingsC`](../known_QC_gaps.py#L85): 0→2, 1→3; C{0,1}, Q{2,3} | |
| | [`QG_Bell_Edge_Edge_SettingsEdgeC`](../known_QC_gaps.py#L86): 0→1, 0→2, 1→3; C{0,1}, Q{2,3} | |

**The order of the search.** The census runs in one phase ([9.1](#91-the-ten-stages)): the elementary reductions first, closed over everything reachable from every input class, then the Fritz piggybacks in nine stages of increasing cost, each applied to whatever the earlier stages left.

**Siblings.** Two nodes are **latent siblings** when they share a facet, **quantum siblings** when they share a quantum facet.

### 0.8 Glossary

Every term below is defined where it is first used; this subsection collects them, with the place of the definition. Four groups of words are easily confused, and the first table sets them apart.

**Words that are easily confused.**

| words | how they differ |
|---|---|
| **target**, **target set** | The *target* is a node: the predicted node of a Fritz step, whose unseen parents are deleted (Section [6](#6-the-fritz-piggyback-fritz)). A *target set* is a set of conditional independences that a certificate beyond d-separation has to derive: `relabel` or `markov` ([7.4](#74-the-two-target-sets-as-containment-of-d-separation-models-qmdag_semigraphoid_certificate)). |
| **certificate** (three meanings) | Of a Fritz step: the test that justifies its deletion, recorded as `parent`, `dsep`, `semigraphoid` or `entropic` ([6.5](#65-the-parent-predictor-deletion-without-a-certificate), [6.6](#66-theorem-d-separation-certificate), Section [7](#7-the-semigraphoid-closure-certificate-for-the-fritz-piggyback-semigraphoidpy-qmdag_semigraphoid_certificate), [Appendix B](#appendix-b-the-entropic-lp-as-an-alternative-certificate-engine-entropic_lppy-qmdag_entropic_certificate)). Of a proven class: the printed chain of recorded steps from the class down to a seed ([9.1](#91-the-ten-stages)). Of one LP: the Farkas certificate, a dual vector that proves one implication ([B.1](#b1-entropy-vectors-as-the-proof-system-entropic_lppy)). |
| **structure**, **class**, **input** | A *structure* is one QmDAG up to relabelling. A *class* is a set of structures proven to have the same quantum model set ([0.6](#06-quantum-observational-equivalence-and-the-two-metagraphs)). An *input* is a class that the census tries to prove ([0.7](#07-the-census-the-seeds-and-the-names-of-the-tricks)); a step is applied to a member structure, and a gap of one member is a gap of the class. |
| **piggyback**, **trick**, **reduction** | A *piggyback* is a map from $G$ to $G'$ with a proof that a gap of $G'$ implies a gap of $G$ ([0.2](#02-piggybacks)). A *trick* is a piggyback as the search applies it, under its code name ([0.7](#07-the-census-the-seeds-and-the-names-of-the-tricks)). An *elementary reduction* is a trick that lowers the number of visible nodes: point distribution, the two marginalizations, postselection, node stitching ([0.7](#07-the-census-the-seeds-and-the-names-of-the-tricks)). |
| **stage**, **cascade**, **ladder**, **rung** | A *stage* is one pass of the search with one set of tricks; the census has ten, the elementary stage and nine Fritz stages ([9.1](#91-the-ten-stages)). The *cascade* is the nine Fritz stages in order of cost. The *ladder* is the list of ten cumulative counts of proven classes, computed from the recorded parameters, and a *rung* is one entry of it ([9.1](#91-the-ten-stages)). |

**Structures and classes.**

| term | meaning |
|---|---|
| QmDAG, structure | visible nodes, edges among them, and classical and quantum facets ([0.1](#01-structures-and-model-sets)) |
| facet | one latent source, named by the nodes it feeds: Q{…} quantum, C{…} classical ([0.1](#01-structures-and-model-sets)) |
| hybrid | a structure with at least one classical facet ([0.1](#01-structures-and-model-sets)) |
| QC gap | a quantum distribution of the structure that no classical model of it produces ([0.1](#01-structures-and-model-sets)) |
| effective DAG, parent | the DAG with one node per facet and one noise node per visible node; a parent is a visible parent, a facet or the noise ([0.1](#01-structures-and-model-sets)) |
| latent siblings, quantum siblings | nodes that share a facet, a quantum facet ([0.7](#07-the-census-the-seeds-and-the-names-of-the-tricks)) |
| degradation | a structure with some of its quantum facets made classical ([0.4](#04-the-degradation-piggyback-quantum-source-to-classical-source)) |
| premeasurable node | a node with no visible parent and exactly one facet ([0.5](#05-when-a-classical-source-is-as-good-as-a-quantum-one)) |
| the rule [`QmDAG.lift`](../quantum_mDAG.py#L235) | a classical facet with at most one member that is not premeasurable can be replaced by a quantum facet without changing either model set ([0.5](#05-when-a-classical-source-is-as-good-as-a-quantum-one)) |
| classical, quantum equivalence class | structures proven to have the same classical, quantum model set ([0.6](#06-quantum-observational-equivalence-and-the-two-metagraphs)) |
| metagraph | the graph of proven containments of model sets whose components are those classes ([0.6](#06-quantum-observational-equivalence-and-the-two-metagraphs)) |
| upward rule | a gap passes to a structurally higher structure with the same classical model set ([0.6](#06-quantum-observational-equivalence-and-the-two-metagraphs)) |
| provably algebraic | said of a class with a member classically equivalent to a latent-free DAG; such a class has no gap ([0.6](#06-quantum-observational-equivalence-and-the-two-metagraphs)) |

**The census.**

| term | meaning |
|---|---|
| census | the search over every four-node input class ([0.7](#07-the-census-the-seeds-and-the-names-of-the-tricks), Section [9](#9-search-certificates-and-the-four-node-census-qc_gap_searchpy-special-applicationsproving_qc_gapspy)) |
| seed | a class whose gap is established outside this note; known from the start ([0.7](#07-the-census-the-seeds-and-the-names-of-the-tricks)) |
| proven, known gap | a class from which a chain of piggybacks leads to a seed ([0.7](#07-the-census-the-seeds-and-the-names-of-the-tricks)) |
| open class | an input that the census does not prove ([10.1](#101-the-46-open-classes)) |
| lookup | [`degradation`](../qc_gap_search.py#L77) and `upward`: transitions registered without expanding their ends ([0.4](#04-the-degradation-piggyback-quantum-source-to-classical-source), [0.6](#06-quantum-observational-equivalence-and-the-two-metagraphs), [9.1](#91-the-ten-stages)) |
| `identity` | the trick that proves an input which is itself a seed class ([0.7](#07-the-census-the-seeds-and-the-names-of-the-tricks)) |
| explorer, transition | the object that applies the tricks and records every step; one recorded step ([9.6](#96-bookkeeping-the-explorer-the-cache-and-the-record)) |
| "via", "only via" | provable with one elementary trick alone; lost when that trick alone is removed ([9.2](#92-reading-the-tables-and-why-the-seeds-matter)) |
| depth one | a Fritz step is applied once, and its outputs are reduced by elementary tricks only ([9.1](#91-the-ten-stages)) |
| root of the search | a structure that a stage is applied to ([8.7](#87-two-kinds-of-stage-d-separation-breadth-first-then-the-semigraphoid-closure-with-early-exit)) |
| early exit | a closure stage stops working on a class at the first step that reaches a known gap ([8.7](#87-two-kinds-of-stage-d-separation-breadth-first-then-the-semigraphoid-closure-with-early-exit)) |
| database | every class or structure the search met that reaches a seed, hybrids included ([9.5](#95-the-known-gap-database-the-search-establishes)) |

**The proofs.**

| term | meaning |
|---|---|
| the quantum direction | a quantum model of $G$ produces every quantum distribution of $G',$ with the observable properties the step relies on ([0.2](#02-piggybacks)) |
| the classical direction | if a distribution with those properties were classical in $G,$ the corresponding one would be classical in $G'$ ([0.2](#02-piggybacks)) |
| constructed distribution | a distribution that the quantum direction produces ([0.2](#02-piggybacks)) |
| (F1), (F2) | edge monotonicity; observable d-separation holds in classical and quantum models alike ([0.3](#03-two-general-facts)) |

**The Fritz piggyback.**

| term | meaning |
|---|---|
| predictor, predictor set $\mathbf X$ | the node or nodes whose output determines the target (Section [6](#6-the-fritz-piggyback-fritz)) |
| predicted node, target $s$ | the node whose parents unseen by the predictor are deleted (Section [6](#6-the-fritz-piggyback-fritz)) |
| sees | a predictor sees itself and its parents ([8.2](#82-what-a-predictor-sees-and-the-deletion-it-dictates-qmdagfritz_deletion)) |
| $\mathrm{common}(s)$ or $K;$ $\mathrm{others}(s)$ or $D$ | the parents of $s$ that the predictors see, which $s$ keeps; the rest, which is deleted (Section [6](#6-the-fritz-piggyback-fritz), [8.2](#82-what-a-predictor-sees-and-the-deletion-it-dictates-qmdagfritz_deletion)) |
| channel | a facet shared by a predictor and the target, or an edge from a predictor to the target ([6.1](#61-the-mechanism)) |
| parent predictor, sibling predictor | a predictor that is a visible parent of the target; one that only shares a facet with it ([6.5](#65-the-parent-predictor-deletion-without-a-certificate), [8.1](#81-the-pool-and-its-order-qmdagfritz_pool-qmdagfritz_targets)) |
| perfect prediction | the target is a function of the predictors' outputs; the only hypothesis a step uses ([6.1](#61-the-mechanism), [6.4](#64-node-splitting-split-and-unsplit-predictors-split-and-unsplit-targets-qmdagsplit_node)) |
| perfect correlation | a part of the predictors' output equals the target; what the quantum direction arranges ([6.1](#61-the-mechanism)) |
| candidate $G_1$ | the structure with the target restricted to what it keeps, the predictors still present ([6.1](#61-the-mechanism)); a *candidate step* is one pair of predictor and target with its deletion ([8.2](#82-what-a-predictor-sees-and-the-deletion-it-dictates-qmdagfritz_deletion)) |
| output $G'$ | the candidate with the predictors removed ([6.1](#61-the-mechanism)) |
| childless, childful | said of a predictor without, with visible children ([6.1](#61-the-mechanism)) |
| split, unsplit (target mode, predictor mode) | whether the target, the predictor, is first split into itself and a copy ([6.4](#64-node-splitting-split-and-unsplit-predictors-split-and-unsplit-targets-qmdagsplit_node)) |
| joint step | one predictor set restricting several targets at once ([6.2](#62-the-unit-a-predictor-and-its-targets), [8.5](#85-joint-steps-and-why-not-a-second-fritz-depth)) |
| sufficiency, determinism | the two halves of perfect prediction as conditional independences; the closure uses the first ([7.3](#73-the-hypotheses-and-how-perfect-prediction-enters-qmdag_semigraphoid_model-semigraphoidadd_functional_dependence)) |

---

## 1. Results: the four-node census

The results come first; everything after this section explains them. Where to find what the results rest on:

* what a quantum equivalence class is and how the two metagraphs are built: [0.6](#06-quantum-observational-equivalence-and-the-two-metagraphs);
* which classes are inputs, what a seed is and what the tricks are called: [0.7](#07-the-census-the-seeds-and-the-names-of-the-tricks);
* the piggybacks themselves: Sections 2 to 7;
* how the Fritz search chooses its targets and predictors, and why the search order decides which certificate an input gets: Section [8](#8-the-fritz-search-procedure-qmdagfritz_steps-qc_gap_searchfritz_tricks);
* the search infrastructure and the reading of the tables: Section [9](#9-search-certificates-and-the-four-node-census-qc_gap_searchpy-special-applicationsproving_qc_gapspy).

All counts are over quantum equivalence classes and are pinned by [`tests/test_baseline_slow.py`](../tests/test_baseline_slow.py) and [`tests/test_metagraph.py`](../tests/test_metagraph.py).

### 1.0 The four-node counts, and why results are counted by class

The two metagraphs of [0.6](#06-quantum-observational-equivalence-and-the-two-metagraphs) have one node for every mDAG up to relabelling, 2809 on four nodes, and their components are the proven classical and quantum equivalence classes. Their counts on three and on four nodes:

| | three nodes | four nodes |
|---|---|---|
| mDAGs up to relabelling | 46 | 2809 |
| classical equivalence classes (containing a latent-free DAG) | 8 (5) | 396 (20) |
| quantum equivalence classes (containing a latent-free DAG) | 8 (5) | 438 (20) |
| HLP edges, classical rule / pre-measurement rule | 12 / 12 | 648 / 528 |
| input classes (not provably algebraic), structures they contain | 3, 5 | 418, 996 |
| classical classes the input classes fall into | 3 | 376 |

In the table, the input classes are the quantum classes that are not provably algebraic ([0.6](#06-quantum-observational-equivalence-and-the-two-metagraphs)); they are the inputs of the census ([0.7](#07-the-census-the-seeds-and-the-names-of-the-tricks)). The HLP edges generated under the two rules are not nested sets ([0.6](#06-quantum-observational-equivalence-and-the-two-metagraphs)). How the nodes of the metagraphs are enumerated, and the corresponding counts on five nodes, are in [Appendix C](#appendix-c-five-nodes-the-two-metagraphs-and-their-counts).


**Why count by class, and what the two counts are.** The inequality-constraint results of this repository are counted over classical observational equivalence classes, because an inequality constraint is a property of the classical model set. A QC gap is not: it compares the quantum and the classical model set of one structure, and the conjecture that if $G$ has a QC gap then so does every $G'$ classically equivalent to $G$ is open. Its proven half is the upward rule ([0.6](#06-quantum-observational-equivalence-and-the-two-metagraphs)). What remains open is the downward direction, and on four nodes it is never needed: of the 376 classical classes the inputs fall into, none contains both a proven and an unproven quantum class after the census ([1.1](#11-the-census-over-quantum-classes)), so counting by quantum class or by classical class gives the same open set.

**Example.** Take the class of the Bell scenario with correlated settings. It has five structures:

* 0→1, 0→3, 1→2; Q{2,3} (settings 0 and 1, outcomes 3 and 2, the settings correlated by 0→1);
* its two versions with the redundant facet Q{0,1} or Q{0,3} added;
* its version with the edge 0→1 dropped in favour of Q{0,1};
* its version with the edge 0→3 dropped in favour of Q{0,3}.

The five are one classical class by face splitting and the HLP rule, and one quantum class because every facet or edge involved sits on a premeasurable node. In an earlier census, which counted structure by structure and had only the Bell scenario without a settings link as a four-node seed, three of the five were proven and two were left among the open structures, although their gaps follow from the Bell argument at once.

### 1.1 The census over quantum classes

The inputs are the 418 quantum equivalence classes of [1.0](#10-the-four-node-counts-and-why-results-are-counted-by-class) (996 structures up to relabelling, in 376 classical classes). Every seed is known from the start: the three-node seed classes (instrumental, 3 structures; Evans; the triangle through its hybrid seed) and the two Bell classes, which are inputs. The census runs in one phase, cheapest stage first ([9.1](#91-the-ten-stages)); [`tests/test_baseline_slow.py`](../tests/test_baseline_slow.py) pins every number below and [`Special Applications/census_breakdowns.py`](../Special%20Applications/census_breakdowns.py) prints one certificate per row of the tables.

**The elementary stage.** The elementary reductions act on every member of every input class, closed over everything reachable, with `identity` and the two lookups, degradation and upward ([9.1](#91-the-ten-stages)). "Via" is the number of input classes provable using only that trick (lookups included), closed under implication among the inputs; "only via" is the number no longer provable when that trick alone is removed from the recorded transitions, everything else kept.

| elementary piggyback | via | only via |
|---|---|---|
| identity (the input is a seed class) | 2 | 0 |
| point distribution | 341 | 164 |
| node stitching | 7 | 0 |
| postselection | 63 | 15 |
| naive marginalization | 170 | 0 |
| teleportation marginalization | 178 | 9 |
| marginalization, either kind (both removed at once) | 178 | 11 |
| all elementary piggybacks | 366 | |

Of the 418 input classes, 366 are proven by the elementary stage and 52 are left for the Fritz stages. Node stitching, whose characteristic product is a Bell variant derived from an instrumental variant, has no exclusive proof: the Bell classes it reaches are seeds ([9.4](#94-why-node-stitching-and-naive-marginalization-show-few-exclusive-proofs)). Removing the naive marginalization alone loses nothing, since the two marginalizations coincide wherever the removed node has no quantum facet; removing the teleportation one loses 9 classes, those where the relayed entanglement is what the proof needs; removing both loses 11.

The "only via" column is counted over the whole record, the Fritz stages included. It therefore also counts a class that is proven through a Fritz step when the chain after that step needs the trick ([8.9](#89-the-six-classes-that-need-a-fritz-step)): one of the 15 classes for postselection, four of the 9 for teleportation marginalization and four of the 11 for marginalization of either kind. Within the elementary stage alone these three counts are 14, 5 and 7, and point distribution keeps its 164.

**The four Fritz piggybacks** are then applied to the 52 classes that are left, cheapest first: the parent predictor ([6.5](#65-the-parent-predictor-deletion-without-a-certificate)), the d-separation certificate ([6.6](#66-theorem-d-separation-certificate)), the semigraphoid closure (Section [7](#7-the-semigraphoid-closure-certificate-for-the-fritz-piggyback-semigraphoidpy-qmdag_semigraphoid_certificate)) and, as the alternative engine of the same hypotheses, the entropic LP ([Appendix B](#appendix-b-the-entropic-lp-as-an-alternative-certificate-engine-entropic_lppy-qmdag_entropic_certificate)). Six classes are proven, 46 remain. The first table says what each piggyback proved after the cheaper ones had failed, and through which of the splittings of [6.4](#64-node-splitting-split-and-unsplit-predictors-split-and-unsplit-targets-qmdagsplit_node).

| Fritz piggyback | new proofs (classes) | no splitting | split target only | split predictor only | target and predictor split |
|---|---|---|---|---|---|
| parent predictor ([6.5](#65-the-parent-predictor-deletion-without-a-certificate)) | 3 | 0 | 2 | 0 | 1 |
| d-separation certificate ([6.6](#66-theorem-d-separation-certificate)) | 2 | 1 | 0 | 0 | 1 |
| semigraphoid closure (Section [7](#7-the-semigraphoid-closure-certificate-for-the-fritz-piggyback-semigraphoidpy-qmdag_semigraphoid_certificate)) | 1 | 0 | 0 | 1 | 0 |
| entropic LP ([Appendix B](#appendix-b-the-entropic-lp-as-an-alternative-certificate-engine-entropic_lppy-qmdag_entropic_certificate)) | 0 beyond the closure: it certifies the same steps | | | | |

The second table names the six classes explicitly, one row per class with every member structure written out, and the version of the Fritz piggyback that witnesses each (the stage of [9.1](#91-the-ten-stages) that proved it; every stage not listed proved nothing). Three are witnessed by the **parent predictor** (the free piggyback of [6.5](#65-the-parent-predictor-deletion-without-a-certificate): the predictor is a visible parent of the predicted node, no certificate), two by the **d-separation certificate** ([6.6](#66-theorem-d-separation-certificate)) and one by the **semigraphoid closure** (Section [7](#7-the-semigraphoid-closure-certificate-for-the-fritz-piggyback-semigraphoidpy-qmdag_semigraphoid_certificate)). In every row the predictor is node 0.

| class (every member; edges, quantum facets) | Fritz version | predicted node, predictor; splitting | stage | seed class reached |
|---|---|---|---|---|
| 0→1→2→3; Q{0,1}, Q{0,2}, Q{1,3} (1 structure) | parent predictor | 1 ← 0 (0 is a parent of 1 and shares Q{0,1} with it); split target, unsplit predictor | 2 | instrumental |
| 0→1, 0→2, 1→2, 2→3; Q{0,1}, Q{0,2}, Q{1,3} (1 structure) | parent predictor | 1 ← 0 (as above); split target, unsplit predictor | 2 | instrumental |
| 0→2, 1→2, 2→3; Q{0,1}, Q{0,2}, Q{1,3} (1 structure) | parent predictor | 2 ← 0 (0 is a parent of 2 and shares Q{0,2} with it); split target, split predictor (0 has the child 2) | 2 | instrumental |
| no edges; Q{0,1,2}, Q{0,1,3}, Q{0,2,3}, Q{1,2,3} (the tetrahedron, [6.7](#67-example-the-tetrahedron-unsplit-target-and-unsplit-predictor); 1 structure) | d-separation certificate | 1 ← 0 (latent siblings through Q{0,1,2} and Q{0,1,3}); unsplit target, unsplit predictor | 3 | triangle |
| 1→2→3; Q{0,1}, Q{0,2}, Q{1,3} ([6.10](#610-example-a-split-predictor-and-a-split-target-worked-out-in-full); 1 structure) | d-separation certificate | 1 ← 0 (latent siblings through Q{0,1}); split target, split predictor (0 is childless, so untouched) | 6 | instrumental |
| 1→2, 1→3, 2→3; Q{0,2}, Q{0,3} (KPC, [7.5](#75-worked-example-the-kpc-structure)) and its quantum-equivalent relatives 1→3, 2→3; Q{0,2}, Q{0,3}, Q{1,2}; 1→2, 1→3, 2→3; Q{0,2}, Q{0,3}, Q{1,2}; 1→2, 2→3; Q{0,2}, Q{0,3}, Q{1,3}; 1→2, 1→3, 2→3; Q{0,2}, Q{0,3}, Q{1,3} (5 structures) | semigraphoid closure | 3 ← 0 (latent siblings through Q{0,3}); unsplit target, split predictor (0 childless, untouched) | 8 | Bell (no settings link) |

[8.9](#89-the-six-classes-that-need-a-fritz-step) gives each step in full (predictor, target, deletion, follow-up); the cumulative counts after each of the ten stages, with their times, are in [9.1](#91-the-ten-stages). The five structures of the last row are one quantum class: they differ by a facet or an edge on a premeasurable node, so one closure certificate serves all of them.

**In terms of structures.** The 372 proven classes contain 935 of the 996 structures; the 46 open classes contain 61 structures and lie in 46 distinct classical classes, none of which contains a proven class. Of the 46 open classes, 44 consist of a single structure; the largest open class has 12 (no edges but 2→3; Q{0,2}, Q{0,3}, Q{1,2,3}, with the facet Q{1,2,3} redundant for several of its members). Section [10](#10-open-questions) lists all 46, and [open_classes.md](open_classes.md) works out how they relate: four of them would follow from a gap in the largest one, no other open class from any other.

**The parent piggyback** was applied to the members of the 52 classes left by the elementary stage with both target modes and both predictor modes: 805 candidate steps, every one justified for free, three classes proven: 0→1→2→3; Q{0,1}, Q{0,2}, Q{1,3}, the same with the edge 0→2 added, and 0→2, 1→2, 2→3; Q{0,1}, Q{0,2}, Q{1,3} (the family of [6.9](#69-example-a-split-target-with-a-childful-unsplit-predictor)). In all three the predictor 0 is a parent of the predicted node (1, 1 and 2 respectively) that also shares a quantum facet with it: the copy of the predicted node keeps the edge from the predictor only, and the teleportation marginalization that follows carries the predictor's quantum share to the node that survives into the seed. [8.9](#89-the-six-classes-that-need-a-fritz-step) walks through each.

**The d-separation certificate** proves two classes: the tetrahedron with no splitting ([6.7](#67-example-the-tetrahedron-unsplit-target-and-unsplit-predictor)), and the class of [6.10](#610-example-a-split-predictor-and-a-split-target-worked-out-in-full), proven with a split target and a split predictor. Two such routes are recorded at that stage: the one in the table above, whose predictor is childless and so left untouched, and the one worked out in [6.10](#610-example-a-split-predictor-and-a-split-target-worked-out-in-full), whose predictor has a child and is split. That last class also has a closure route with an unsplit target ([7.8](#78-further-examples-from-the-census)); with the d-separation stages run first the census records the d-separation route. Joint steps ([8.5](#85-joint-steps-and-why-not-a-second-fritz-depth)), in which one predictor restricts several targets at once, are enumerated in all four d-separation stages and prove no four-node class that single targets do not.

**The semigraphoid closure** proves one more class, with a childless predictor split (hence untouched, [6.4](#64-node-splitting-split-and-unsplit-predictors-split-and-unsplit-targets-qmdagsplit_node)), an unsplit target and the `relabel` target set ([7.8](#78-further-examples-from-the-census)). With unsplit predictors the closure adds nothing beyond d-separation in four nodes, and with split targets nothing at all. It costs about a millisecond per candidate: the 696 candidates that d-separation rejects (37 certified, 659 not) take under a second in all.

**The entropic LP** (`--engine lp`) proves the same class by the same step: on every candidate the closure and the LP agree ([7.9](#79-strength-agreement-with-the-lp-and-cost), [9.3](#93-certificates-beyond-d-separation-attempted)).

**Two experiments**, both measured on this census (Section [8](#8-the-fritz-search-procedure-qmdagfritz_steps-qc_gap_searchfritz_tricks)).

* *Descendants as predictors.* Admitting the descendants of the target to the certified pools (`--descendants`) nearly doubles the checks beyond d-separation, 1332 against 696, since every descendant candidate fails d-separation. It certifies six more candidate steps and proves nothing new ([8.10](#810-experiment-descendants-as-predictors) works them out).
* *No early exit.* A closure stage stops working on a class at the first step that reaches a known gap ([8.7](#87-two-kinds-of-stage-d-separation-breadth-first-then-the-semigraphoid-closure-with-early-exit)). Switching that off (`--no-early-exit`) checks ten more candidates, 706 against 696, all of them in the one class that the closure proves; nine of the ten are certified, and the proven classes and the database of [9.5](#95-the-known-gap-database-the-search-establishes) are the same.

---

## 2. Point distribution (`PD`)

**Map.** Remove one visible node $v$ together with all its edges and its membership in facets; write $G-v$ for the result, so $G'=G-v$ ([`QmDAG.fix_to_point_distribution_QmDAG`](../quantum_mDAG.py#L201); trick `PD`, [`QmDAG.pd_steps`](../quantum_mDAG.py#L681)). $\pi$ is the marginal over $v.$

**The quantum direction.** Given $P'\in\mathcal Q(G'),$ let $v$ output a constant and every other node run its $G'$ strategy; call the resulting distribution $P.$ Its marginal over $v$ is $P',$ and $P\in\mathcal Q(G)$ because the children of $v$ may ignore a constant input.

**The classical direction.** In a classical model of $G$ for a distribution with $v$ constant, substitute the constant into the kernels of $v$'s children; the result is a classical model of $G-v$ for the marginal. $\square$

**Example.** Input (provable only through PD; 164 such classes, [1.1](#11-the-census-over-quantum-classes)): 0→2, 0→3; Q{0,1,2}, Q{0,1,3}, Q{2,3}. Fixing 0 to a point distribution deletes its two edges and removes it from the two triple facets, leaving Q{1,2}, Q{1,3}, Q{2,3}: the all-quantum triangle, which degrades to [`QG_Triangle`](../known_QC_gaps.py#L42). Most PD proofs look like this: a seed with one more node attached to it.

---

## 3. Marginalization, naive and with teleportation (`naive_marginalization`, `teleportation_marginalization`)

**Map.** Remove the visible node $v$ ([`QmDAG.marginalize(node, apply_teleportation)`](../quantum_mDAG.py#L377)) and change the rest of the structure as follows:

* every visible parent of $v$ becomes a parent of every visible child of $v;$
* every facet $F$ containing $v$ loses $v$ (a quantum facet stays quantum on $F\setminus v$), and in addition a *classical* facet over $(F\setminus v)$ and all visible children of $v$ is added;
* a new classical facet over the visible children of $v$ is added as well (the private randomness of $v$);
* with teleportation, additionally every quantum facet $F\ni v$ is enlarged to a quantum facet on $(F\setminus v)\cup T,$ where $T$ is the set of visible children of $v$ that share some quantum facet with $v.$

Without the last change the map is the naive marginalization; with it, the marginalization with teleportation. $\pi$ is the marginal over $v.$

**The quantum direction.** Given $P'\in\mathcal Q(G'),$ build the quantum model of $G$ as follows:

* $v$ receives its visible parents and forwards them to its children (the new visible edges);
* $v$ forwards the classical values its facets carry (a quantum facet may carry the classical string that the new classical facet over $(F\setminus v)$ and the children uses in $G'$);
* $v$ broadcasts a private random string (the new classical facet over the children);
* for a quantum facet $F\ni v$ that in $G'$ extends to the children in $T,$ $v$ holds a composite share of the $G'$-state, one subsystem per child in $T;$ $v$ and each such child $c$ share some quantum facet, which may carry an extra maximally entangled pair, and $v$ teleports the subsystem destined for $c$ through that pair and the visible edge $v\to c.$

The children then hold exactly the resources of $G'.$ $\square$

**The classical direction.** Without teleportation $G'$ is the latent projection of $G$ over $v$ (Evans), and the marginal of a classical model of $G$ is a classical model of the projection: the kernel of $v$ is absorbed into its children, its latent parents are relayed to its children, and its private randomness becomes a common cause of its children. With teleportation $G'$ has more facets than the projection, and $\mathcal C$ only grows with facets. $\square$

**Example.** Input (provable only through marginalization; one of the 7 such classes of the elementary stage, [1.1](#11-the-census-over-quantum-classes)): 0→1→2→3; Q{1,3}. Node 2 relays the outcome of 1 to 3 and has no latent of its own. Marginalizing it relays the edge, 0→1→3; Q{1,3}, which is [`QG_Instrumental_Edge`](../known_QC_gaps.py#L39). Nothing else works: PD on 2 leaves 3 without an input; postselecting on 2 is inadmissible because the grandparent 0 is not a parent of 2, and after PD on 0 it is still inadmissible by condition 2 of Section [4](#4-postselection-conditioning), since the facet Q{1,3} contains the parent 1 but not 2. Both marginalizations give the same output here, since 2 has no quantum facet.

The same move, marginalizing a relay node that holds no facet, proves other inputs as well, for instance 0→1→2; Q{0,2}, Q{0,3}, where marginalizing 1 relays the setting 0 to 2 and gives 0→2; Q{0,2}, Q{0,3}, which degrades to [`QG_Instrumental_C`](../known_QC_gaps.py#L41).

**Order dependence.** Teleportation is not symmetric, so marginalizing several nodes in different orders can give different structures; wherever the code removes several nodes it enumerates every order ([`QmDAG.fritz_realise`](../quantum_mDAG.py#L902), through [`QmDAG._marginalize_nodes`](../quantum_mDAG.py#L889)).

---

## 4. Postselection (`conditioning`)

**Map.** Write $B=\mathrm{Pa}(X)\cap V$ for the visible parents of $X$ and $S$ for its latent siblings, the visible nodes sharing some facet with $X.$ Remove $X;$ add a classical facet over $B\cup S;$ add a quantum facet over the quantum siblings of $X,$ the nodes sharing a quantum facet with it ([`QmDAG.condition(node)`](../quantum_mDAG.py#L308)). The operation that recovers $`G'`$'s distribution is postselection: $\pi(P)=P(\cdot\mid X=x_0)$ for a suitable value $x_0.$

The idea is the following. Postselecting on $X=x_0$ correlates everything $X$ depended on: its latents, its visible parents, and whatever the inputs of those parents reach. In $G'$ one new classical latent $\mu,$ shared by the parents and the latent siblings of $X,$ must account for all of these correlations, and the quantum siblings may in addition share a state, because postselecting on a joint measurement by $X$ can leave them entangled (entanglement swapping). The admissibility conditions make sure that nothing the postselection correlates lies beyond the reach of the new facet.

The map is admissible ([`QmDAG.conditioning_is_justified`](../quantum_mDAG.py#L293)) when:

1. every visible grandparent of $X$ is a visible parent of $X$ ($B$ is ancestrally closed among visible nodes);
2. every facet containing a node of $B$ either contains $X$ or is contained in $B.$

Together the two conditions say that every input of the parent block $B$ is inside the block, or a latent of $X,$ or a latent whose children all lie in $B,$ so that the postselected block can be re-sampled from the new facet alone. Both halves of the proof use exactly these two conditions. No condition on the *children* of the parents is needed; the last remark of this section says why.

**The quantum direction.** Given $P'\in\mathcal Q(G'),$ with the new classical facet carrying the string $\mu$ and the new quantum facet carrying the state $\rho$ over the quantum siblings $S_Q$ of $X,$ build $P\in\mathcal Q(G)$ with $P(\cdot\mid X=x_0)=P'.$ The device, due to Marina Maciel Ansanelli, is that **every visible parent of $X$ outputs a uniformly random value, and $X$ recomputes what all of them should have output and postselects on agreement**. In $G$:

* every facet $F\ni X$ carries an independent uniform copy $\mu_F$ of the string $\mu,$ the state that $G'$ assigns to the retained facet $F\setminus X$ (with the shares of the members of $B$ held by $X$ instead, see below), and, if $F$ is quantum, one maximally entangled pair between $X$ and each quantum sibling in $F$ outside $B;$
* the latent siblings of $X$ outside $B$ read $\mu_F$ from a facet they share with $X$ and run their $G'$ strategy;
* $X$ prepares $\rho$ locally and teleports the share of each quantum sibling outside $B$ to it through the corresponding pair; keeping only the identity outcome of each Bell measurement (postselected teleportation) leaves those siblings holding their shares of $\rho$ exactly, while $X$ keeps the shares of the quantum siblings in $B;$
* every visible parent $p\in B$ ignores its inputs and outputs a uniformly random value;
* $X$ draws one joint sample of what the parents would have output under their $G'$ strategies, and outputs $x_0$ iff that sample equals the parents' actual outputs, all the $\mu_F$ coincide, and every Bell outcome is the identity.

$X$ can draw that sample because, by conditions 1 and 2, every input a parent uses in $G'$ is in $X$'s hands:

* the parents' visible parents lie in $B,$ so $X$ sees their realised outputs;
* $\mu$ is the common value of the copies $\mu_F$ (when $X$ lies in no facet, so that the new facet covers $B$ alone, $X$ draws $\mu$ itself as part of its sample);
* a classical facet inside $B$ is absorbed into the new facet of $G',$ so it is part of $\mu;$
* a quantum facet inside $B$ is used only by members of $B,$ so $X$ prepares its state itself and measures every share;
* the share a parent would hold of a retained facet $F\setminus X$ or of $\rho$ is held by $X$ instead, since $X$ is a member of $F$ and prepares $\rho.$

Conditioned on $X=x_0$ the block $B,$ the latent $\mu$ and the siblings are jointly distributed as in $P'$: the uniform outputs contribute a constant factor, and the agreement check selects exactly the joint sample. Every other node keeps its $G'$ kernel and reads the realised values of its parents. A child of a parent that lies outside $B\cup\lbrace X\rbrace$ is not an ancestor of $X$ (an ancestor among the children of a parent would be a grandparent, hence by condition 1 a parent, hence recomputed), so the postselection event does not depend on it and its kernel is unaffected. Hence $P(\cdot\mid x_0)=P'.$ $\square$

**The classical direction.** Let $\Lambda_X$ be the latents of $X$ and $\Lambda_B$ the latents whose children all lie in $B.$ By conditions 1 and 2 every input of a node in $B$ lies in $B,$ in $\Lambda_X$ or in $\Lambda_B;$ the latents of $\Lambda_B$ influence nothing outside $B;$ and every other latent is independent of $(B,X,\Lambda_X,\Lambda_B).$ In a classical model of $G$ the conditional distribution therefore factorises as

```math
P(\text{all}\mid x_0)\ =\ P(\Lambda_X,\Lambda_B\mid x_0)\cdot P(B\mid \Lambda_X,\Lambda_B,x_0)\cdot\Big[\prod_{\text{other nodes}}\text{kernels}\Big]\cdot P(\text{other latents}),
```

where the block distribution $P(B\mid\Lambda_X,\Lambda_B,x_0)$ is proportional to $\prod_{p\in B}P(p\mid\mathrm{pa}(p)\cap B,\Lambda_X,\Lambda_B)\cdot P(x_0\mid B,\Lambda_X)$: the product of the block's kernels, reweighted by the probability that $X$ takes the value $x_0.$

Define the new latent $\mu=(\Lambda_X,\Lambda_B,R),$ where $(\Lambda_X,\Lambda_B)$ is drawn from the posterior $P(\Lambda_X,\Lambda_B\mid x_0)$ and $R$ is fresh uniform randomness. The classical model of $G'$ is then the following.

* The latent siblings of $X$ read $\Lambda_X$ through $\mu;$ this is legitimate because every facet containing $X$ lies within $S\cup\lbrace X\rbrace,$ so nobody else reads $\Lambda_X$ (the facets $F\setminus X$ that $G'$ retains are set trivial).
* Visible children of $X,$ if any, lose the edge from $X$ and read the constant $x_0$ instead.
* The block $B$ is sampled jointly, as deterministic functions of $\mu$ alone, from the reweighted block distribution: all inputs of the block are available in $\mu,$ so the shared randomness $R$ lets each $p$ compute its coordinate of one joint sample.
* Every other node keeps its kernel.

The result is a classical model of $G'$ for $P(\cdot\mid x_0).$ $\square$

**Example.** Input (provable only through postselection; 15 such classes, [1.1](#11-the-census-over-quantum-classes)): 2→3; Q{0,2}, Q{0,3}, Q{1,2}. Node 0 has no parents, so both conditions hold vacuously. Postselecting on 0 removes it and adds a quantum facet over its quantum siblings 2 and 3 (the classical facet over the same pair is absorbed), giving 2→3; Q{1,2}, Q{2,3}, the all-quantum version of [`QG_Instrumental_C`](../known_QC_gaps.py#L41). Operationally the quantum construction is entanglement swapping: 0 performs a Bell measurement on its two shares and the postselected outcome leaves 2 and 3 entangled. Among the reductions only postselection and teleportation marginalization create a facet between nodes that did not share one, and teleportation needs a visible edge from 0 to a quantum sibling, which is absent here.

**Why condition 2 cannot be dropped.** Without it some parent $p\in B$ shares a latent with an outside node $o;$ the reweighted block of the classical direction then depends on an input that $o$ reads as well, and the argument breaks. So does its conclusion: take $X$ with parents $p_1,p_2,$ each $p_i$ sharing a classical facet $\lambda_i$ with an outside node $o_i,$ and the classical model $p_i=\lambda_i\oplus\varepsilon_i$ with biased noise, $o_i=\lambda_i,$ $X=[p_1=p_2].$ Postselecting on $X=1$ makes $o_1$ and $o_2$ dependent ($I(o_1{:}o_2)\approx0.046$ bits), while $G'$ d-separates them. The postselected distribution of a classical model of $G$ is then not a classical model of $G',$ which is exactly what the argument has to deliver ([`tests/test_conditioning.py`](../tests/test_conditioning.py)). Whether the implication "gap in $G'$ implies gap in $G$" itself can fail without condition 2 is not settled; only this proof of it fails.

**Why no condition on the parents' children is needed.** There is an alternative construction, in which a parent $p$ sharing no facet with $X$ *guesses* $\mu$ and announces the guess in its output, $(p,\hat\mu_p),$ for $X$ to check. That fine-grains $p$'s output, and coarse-graining it back is only safe when every child of $p$ reads $\mu$ anyway, so the alternative needs a third admissibility condition (every visible child of such a parent among the parents, the siblings and $X$). Recomputing the parents instead of letting them announce removes the fine-graining and with it the condition. On the four-node census, dropping the third condition admits 36 more postselection transitions and proves no further input ([1.1](#11-the-census-over-quantum-classes)).

---

## 5. Node stitching (`node_stitching`)

**Map.** Let $x$ be an exogenous node (no visible parents, no non-singleton facet: [`QmDAG.exogenous_visible_nodes`](../quantum_mDAG.py#L138)) and $y$ a node that is not a descendant of $x,$ so that $G'$ is acyclic. Remove $x;$ its visible children become children of $y$ ([`QmDAG.node_stitching(node_with_no_children=y, node_with_no_parents=x)`](../quantum_mDAG.py#L341)). The search additionally restricts $y$ to childless nodes; the proof does not need this. $\pi$ is the renormalised diagonal $x=y$ of $P$:

```math
\pi(P)(y,\text{rest})=\frac{P(x{=}y,\;y,\;\text{rest})}{P_x(y)},
```

where $P_x$ is the marginal of $x$ (uniform for every constructed distribution, so the denominator is then a constant).

**The quantum direction.** Given $P'\in\mathcal Q(G'),$ in $G$ let $x$ be uniform, let the former children of $x$ treat the value of $x$ as they treated $y$ in $G',$ and let every other node run its $G'$ strategy. Then $`P(x,y,\text{rest})=P_x(x)\,P'^{\,do(y:=x)}(y,\text{rest}),`$ where the right factor is the distribution of the $G'$ strategy with the input that the former children of $x$ now take from $y$ fixed to $x$ (an edge intervention). Fixing that input to the value $y$ actually takes changes nothing, so $`P'^{\,do(y:=y)}=P'`$ and $\pi(P)=P'.$

**The classical direction.** Let $M$ be a classical model of $G$ for a distribution with $x$ uniform; $x$ is independent of all latents of $M.$ Define a model of $G'$ in which every node keeps its kernel, except that the former children of $x$ receive $y$ in place of $x.$ Since $y\notin\mathrm{desc}(x),$ the value $Y(\lambda)$ does not depend on $x$ and the new model is acyclic; its distribution is $`\sum_\lambda p(\lambda)[y=Y(\lambda)]\,[\text{rest}=f(\lambda,x{:=}Y(\lambda))],`$ which is the diagonal $x=Y(\lambda)$ of $M$'s distribution divided by $P_x(y),$ that is $\pi(P).$ Other children of $y$ read $y$ in both structures. $\square$

**Example.** [`QG_Bell_Edge_Edge`](../known_QC_gaps.py#L83) is 0→2, 1→3; Q{2,3}. Stitching with $x=1$ (exogenous) and $y=2$ (childless, not a descendant of 1) removes 1 and makes 3 a child of 2: 0→2→3; Q{2,3}, which is [`QG_Instrumental_Edge`](../known_QC_gaps.py#L39). The postselection $x=y$ identifies Bob's setting with Alice's outcome. Read as a piggyback: a gap in the instrumental scenario implies a gap in the Bell scenario. Likewise 0→2, 1→3; Q{1,3}, Q{2,3} maps with $x=0,$ $y=3$ to 1→3→2; Q{1,3}, Q{2,3}, which degrades to [`QG_Instrumental_EdgeC`](../known_QC_gaps.py#L40), and [`QG_Bell_C_Edge`](../known_QC_gaps.py#L80) (0→2; C{1,3}, Q{2,3}) maps to [`QG_Instrumental_C`](../known_QC_gaps.py#L41).

**Naming.** The map *stitches* the exogenous node $x$ onto the sink $y$ by postselecting $x=y.$ Its inverse, which cuts a node into an exogenous source and a sink and distributes the children ([Appendix A](#appendix-a-the-piggybacks-in-reverse)), is the operation named interruption.

---

## 6. The Fritz piggyback (`Fritz`)

The Fritz piggyback deletes arrows into a node. One node, the **predictor**, perfectly predicts another, the **predicted node** or **target**. The arrows into the target from parents the predictor cannot see are then idle in every classical model, and may be deleted.

The section is organised as follows.

* [6.1](#61-the-mechanism) gives the mechanism, and says how the predictor leaves the structure at the end of a step.
* [6.2](#62-the-unit-a-predictor-and-its-targets) defines the unit of a step: a predictor set and its targets.
* [6.3](#63-example-the-triangle-fritzs-original-argument) walks through Fritz's original argument, the triangle, first in words.
* [6.4](#64-node-splitting-split-and-unsplit-predictors-split-and-unsplit-targets-qmdagsplit_node) presents node splitting and the split and unsplit variants of a step.
* [6.5](#65-the-parent-predictor-deletion-without-a-certificate) and [6.6](#66-theorem-d-separation-certificate) are the two theorems.
* [6.7](#67-example-the-tetrahedron-unsplit-target-and-unsplit-predictor) to [6.10](#610-example-a-split-predictor-and-a-split-target-worked-out-in-full) are worked examples.

Every example, [6.3](#63-example-the-triangle-fritzs-original-argument) included, has the same layout: **Input**, **Predictor, target, modes**, **Test** (which certificate justifies the deletion, and why it holds), **Steps**, **Structure reached**.

**Notation.** Parents are taken in the effective DAG: $\mathrm{Pa}(s)$ consists of the visible parents of $s,$ the facets containing $s$ and the private noise of $s.$ The predictor is in general a **set** of visible nodes, written $\mathbf X;$ its members are the predictors. $\mathrm{Pa}(\mathbf X)=\bigcup_{x\in\mathbf X}\mathrm{Pa}(x).$ The nodes **seen** by $\mathbf X$ are $\mathrm{Pa}(\mathbf X)\cup\mathbf X.$ For a predicted node $s,$

```math
\mathrm{common}(s)=\mathrm{Pa}(s)\cap\big(\mathrm{Pa}(\mathbf X)\cup\mathbf X\big),\qquad
\mathrm{others}(s)=\mathrm{Pa}(s)\setminus\mathrm{common}(s),
```

the latter always containing the private noise of $s.$ Structures in examples are written as a list of visible edges and facets, with Q for quantum and C for classical facets, e.g. "1→3, 2→3; Q{0,2}, Q{0,3}, Q{1,2}".

### 6.1 The mechanism

The Fritz piggyback comes in four versions, which differ only in how the deletion is justified: the parent predictor of [6.5](#65-the-parent-predictor-deletion-without-a-certificate), the d-separation certificate of [6.6](#66-theorem-d-separation-certificate), the semigraphoid closure of Section [7](#7-the-semigraphoid-closure-certificate-for-the-fritz-piggyback-semigraphoidpy-qmdag_semigraphoid_certificate), and the entropic LP of [Appendix B](#appendix-b-the-entropic-lp-as-an-alternative-certificate-engine-entropic_lppy-qmdag_entropic_certificate). All are instances of one scheme:

> choose a set of edges of $G$ to delete, giving the candidate $G_1$ defined below, and justify the deletion by a **perfect correlation** that the quantum direction can arrange, between a subvariable of a predictor (a function of its output) and the predicted node, and that forces the deleted dependences to be idle in every classical model.

**Predictor, target and channel.** Fix a predictor set $\mathbf X$ and a predicted node $s\notin\mathbf X$ with a **channel** from a predictor to $s$: a facet shared by $s$ and a predictor, or an edge $X\to s$ from a predictor that is a visible parent of $s.$ The two kinds of channel give two kinds of piggyback: a visible parent needs no certificate ([6.5](#65-the-parent-predictor-deletion-without-a-certificate)), a latent sibling (a predictor that shares a facet with $s,$ also called a sibling predictor) does ([6.6](#66-theorem-d-separation-certificate) and Section [7](#7-the-semigraphoid-closure-certificate-for-the-fritz-piggyback-semigraphoidpy-qmdag_semigraphoid_certificate)); the search treats them as separate pools ([8.1](#81-the-pool-and-its-order-qmdagfritz_pool-qmdagfritz_targets)).

**How the quantum direction arranges the correlation.** In the quantum direction, $s$ is a deterministic function $f$ of $\mathrm{common}(s),$ with its private randomness drawn from a facet shared with a predictor (or computed by a predictor that is a parent of $s$ and sent down the edge), and each predictor outputs the part of $f(\mathrm{common}(s))$ computable from what it sees, so that $\mathbf X$ jointly carries a subvariable $X_S$ equal to $s$: the perfect correlation. This is why $s$ needs a channel from a predictor (noise absorption) and why only $\mathrm{common}(s)$ may be kept: the predictors must be able to compute $s.$

**Correlation and prediction.** The words *predictor* and *predicted*, which are the code's, name the roles; the relation itself is symmetric, following KPC: the quantum direction gives $\mathbf X$ a subvariable $X_S$ with $X_S=s,$ a **perfect correlation**, and nothing flows from $\mathbf X$ to $s,$ both being functions of shared inputs. The certificates consume only one direction of it, $H(s\mid\mathbf X)=0,$ which is called **perfect prediction** below; [6.4](#64-node-splitting-split-and-unsplit-predictors-split-and-unsplit-targets-qmdagsplit_node) (**Prediction, not perfect correlation**) says why a step assumes this and nothing stronger.

**The candidate.** The **candidate** $G_1$ restricts $s$ to the parents it keeps, $\mathrm{common}(s)$ for a sibling predictor and $\mathbf X$ itself for a parent predictor ([6.5](#65-the-parent-predictor-deletion-without-a-certificate)), and leaves everything else in place, predictors included: visible co-parents are kept; facets of $\mathrm{common}(s)$ are kept but become classical *for* $s$ (a quantum facet $F$ keeps its quantum part on $F\setminus\lbrace s\rbrace$ and a classical facet over $F$ is added, which is the state $\rho\otimes\sigma$ with $\sigma$ classical).

**What has to be proven.** Apart from the added classical facet, $G_1\subseteq G,$ so by (F1) the quantum direction from $G_1$ to $G$ only has to construct the correlation, and the entire content is the classical direction: a classical model of $G$ in which the prediction holds, and in which the other observable properties of the constructed distribution hold, is also a classical model of $G_1.$ Throughout, "Markov($H$)" for a DAG $H$ means the local Markov property, each node independent of its non-descendants given its parents, which for a DAG is equivalent to the factorisation of the joint into one kernel per node.

**Removing the predictor.** The **output** $G'$ of the piggyback is $G_1$ with the predictors removed: a predictor without children (*childless*) is deleted, a predictor with children (*childful*) is marginalized (Section [3](#3-marginalization-naive-and-with-teleportation-naive_marginalization-teleportation_marginalization)). When the predictors are *split* ([6.4](#64-node-splitting-split-and-unsplit-predictors-split-and-unsplit-targets-qmdagsplit_node)), their copies are removed instead. The removal is sound in both directions. *The quantum direction:* a distribution $P'\in\mathcal Q(G')$ is realised in $G_1,$ where a deleted predictor outputs a constant (Section [2](#2-point-distribution-pd)) and a marginalized one acts as in the quantum direction of Section [3](#3-marginalization-naive-and-with-teleportation-naive_marginalization-teleportation_marginalization) (it forwards its visible parents, broadcasts the classical strings and teleports its quantum shares to its children); this gives $P_1\in\mathcal Q(G_1)$ with $P'$ as its marginal over the predictors. *The classical direction:* marginalizing a node of a classical model gives, for the marginal, a classical model of the latent projection, which is $G'$ or, with teleportation, is contained in $G'$ (Section [3](#3-marginalization-naive-and-with-teleportation-naive_marginalization-teleportation_marginalization)), and a childless predictor is simply deleted; so a classical model of $G_1$ gives a classical model of $G'$ for the marginal over the predictors. [6.4](#64-node-splitting-split-and-unsplit-predictors-split-and-unsplit-targets-qmdagsplit_node), [6.5](#65-the-parent-predictor-deletion-without-a-certificate) and [6.6](#66-theorem-d-separation-certificate) refer to this paragraph.

### 6.2 The unit: a predictor and its targets

The unit of the piggyback is a predictor set $\mathbf X$ together with a set $T$ of predicted nodes, its **targets**. Each target's deletion is read off the predictor: $s$ keeps $\mathrm{common}(s),$ what $\mathbf X$ sees, and loses $\mathrm{others}(s),$ everything $\mathbf X$ cannot see, all of it or nothing (Section [8](#8-the-fritz-search-procedure-qmdagfritz_steps-qc_gap_searchfritz_tricks) states the procedure). Deleting a single parent $p\to s$ while $s$ keeps another parent $q$ that $\mathbf X$ cannot see is not a candidate: the predictors cannot predict $s,$ and no other piggyback justifies the lone deletion.

Example: 1→3, 2→3; Q{0,2}, Q{0,3}, Q{1,2}. Predictor 0 deletes 1→3 and 2→3 together and reaches C{0,3}, Q{0,2}, Q{1,2}, which degrades to [`QG_Bell_C_C`](../known_QC_gaps.py#L78) (a semigraphoid-closure step, example 2 of [7.8](#78-further-examples-from-the-census)), whereas neither structure with only one of the two edges deleted is reachable from the input by any trick of the toolkit, out of 760 reachable structures. Across the four-node census, of the single-target Fritz steps that delete two or more non-noise parents, more than half have no reachable single-edge intermediate.

**Several targets at once.** One predictor may predict several nodes in the same step, each restricted to what the predictor sees of it; such a step is called a **joint step**. Theorem [6.6](#66-theorem-d-separation-certificate) is stated for a set $T,$ and its classical direction replaces the kernels one at a time. This is Fritz's original argument: in the triangle, node 2 predicts copies of 0 and of 1 together, and when 2 is then removed what is left is the Bell structure with classical settings ([6.3](#63-example-the-triangle-fritzs-original-argument)).

**Joint steps and sequences of single steps.** Whether a joint step equals its single-target steps taken one after the other depends on what becomes of the predictor. A predictor is either *unsplit*, which means that it is removed at the end of the step ([6.1](#61-the-mechanism)), or *split*, which means that it stays in the structure ([6.4](#64-node-splitting-split-and-unsplit-predictors-split-and-unsplit-targets-qmdagsplit_node)); a childless split predictor is left untouched. With the predictor *split* and childless, a joint step is literally the sequence of its single-target steps, since the restrictions commute and the predictor is untouched. With the predictor *unsplit*, it is not: after the first single-target step the predictor is gone and nothing predicts the second target, so the sequential route needs two split-predictor steps followed by a point distribution, three steps in all, two of them Fritz steps.

**Several predictors.** Joint predictor *sets*, on the other hand, are not reducible to single predictors: a set sees more than any member.

Which joint steps the search tries, and why it tries them, is the subject of [8.5](#85-joint-steps-and-why-not-a-second-fritz-depth).

### 6.3 Example: the triangle, Fritz's original argument

One joint step ([6.2](#62-the-unit-a-predictor-and-its-targets)); the paradigm of the piggyback. The argument is given twice: first in words, with the notions of [6.1](#61-the-mechanism) and [6.2](#62-the-unit-a-predictor-and-its-targets), then in the layout of the later examples. Two notions are used informally here and made precise later: splitting a node into itself and a copy ([6.4](#64-node-splitting-split-and-unsplit-predictors-split-and-unsplit-targets-qmdagsplit_node)), and the d-separation test that certifies the deletion ([6.6](#66-theorem-d-separation-certificate)).

**The argument in words.** In the triangle each of the three pairs of nodes shares a quantum source. Node 2 is the predictor, and it predicts 0 and 1 in one step. The targets are not 0 and 1 themselves but a copy of each: 0 becomes 0 and 0′, 1 becomes 1 and 1′, and each copy sits in the same sources as its original. The originals are not restricted, so they keep the source that 0 and 1 share. The step has three parts.

* *The quantum direction.* Node 2 holds a share of each pair state; 0′ and 1′ are deterministic functions of their shares, which 2 announces alongside its own output: the perfect prediction of [6.1](#61-the-mechanism) (the announced subvariable equals the copies; 2 itself, unsplit, is not a function of them).
* *The deletion.* Node 2 sees the source it shares with 0 and the source it shares with 1. Each copy keeps the one source it shares with 2, as a classical source, and loses what 2 cannot see: the source between 0 and 1, and its own noise. The test of [6.6](#66-theorem-d-separation-certificate) certifies that these lost parents are idle in every classical model in which 2 predicts the copies.
* *Removing the predictor.* Node 2 has no children, so it is deleted ([6.1](#61-the-mechanism)) and its shares vanish.

What is left is the Bell structure with classical settings: the copies are the settings and the originals are the measurements. Read backwards, this is Fritz's observation: a Bell experiment whose settings are chosen by a third party sharing randomness with each measurement is a triangle.

**Input.** No edges; Q{0,1}, Q{0,2}, Q{1,2}.

**Predictor, target, modes.** Predictor 2, unsplit; predicted nodes 0 and 1, both split, in one joint step. The split-target mode splits both targets first: 0 becomes 0 and 0′, 1 becomes 1 and 1′, and the facets follow the copies, Q{0,0′,1,1′}, Q{0,0′,2}, Q{1,1′,2}. Node 2 sees Q{0,0′,2} and Q{1,1′,2}, so the copy 0′ keeps Q{0,0′,2} and loses Q{0,0′,1,1′} and its noise, and 1′ likewise keeps Q{1,1′,2}.

**Test.** d-separation ([6.6](#66-theorem-d-separation-certificate)). The test for 0′ asks whether 2 is separated from Q{0,0′,1,1′} and the noise of 0′ given Q{0,0′,2}: every path from 2 to that facet runs through 0 or 1 as a collider (2 ← Q{0,0′,2} → 0 ← Q{0,0′,1,1′}), so it holds, and the same for 1′.

**Steps.**

* `Fritz`, predictor 2, split targets 0′ and 1′, predictor 2 unsplit. 0′ keeps Q{0,0′,2}, classical for it (C{0,0′,2}); 1′ keeps C{1,1′,2}; the originals keep Q{0,1}; the childless 2 is deleted and its shares vanish.

**Structure reached.** C{0,0′}, C{1,1′}, Q{0,1}, which is [`QG_Bell_C_C`](../known_QC_gaps.py#L78): 0′ and 1′ are the settings, each shared classically with its measurement, and the two measurements share the entangled pair.

Both copies must be predicted in one step: with 2 removed after predicting 0′ alone, nothing is left to predict 1′ ([6.2](#62-the-unit-a-predictor-and-its-targets)).

### 6.4 Node splitting; split and unsplit predictors, split and unsplit targets (`QmDAG.split_node`)

Node splitting is a piggyback of its own. This subsection states it, then describes the two roles in which it enters a Fritz step, the predictor mode and the target mode, and finally states the one hypothesis that every mode uses: prediction, not perfect correlation.

**Map.** Replace $s$ by two nodes $s,s'$ with the same parents and the same children, sharing a classical two-party facet $\lbrace s,s'\rbrace$ (their common private randomness; absorbed when $s$ already lies in a facet). $\pi$ merges $(s,s')$ into one node.

**The quantum direction.** $s$ computes both outputs from its inputs and private randomness.

**The classical direction.** A classical kernel for the pair $(s,s')$ splits into two kernels sharing the randomness carried by the new facet. $\square$ When the absorbing facet is quantum the pair's shared randomness is carried by that quantum facet, which is legitimate because the classical direction concerns classical models only.

Node splitting is never run on its own: the split structure has one more visible node and no new independences, so it is only a stepping stone. It enters the Fritz piggyback in two different roles, as a **predictor mode** and as a **target mode**, each either [`split`](../canonical_forms.py#L206) or `unsplit`, and the difference between the roles matters.

**The predictor: unsplit or split.** In the quantum direction of a Fritz step, each predictor $x\in\mathbf X$ outputs, alongside its own value, the part of $f(\mathrm{common}(s))$ it can compute: the predictor's output is *fine-grained*, and the extra component is the subvariable $X_S$ perfectly correlated with $s.$ The classical direction first gives a classical model of the candidate $G_1$ ([6.1](#61-the-mechanism)), in which $x$ still has the fine-grained output, and the proof must get rid of the extra component. The two predictor modes are the two ways of doing so.

* **Unsplit predictor** (`predictor_mode='unsplit'`). $x$ itself is removed ([6.1](#61-the-mechanism), **Removing the predictor**); the component is integrated out with it and nothing more is needed.
* **Split predictor** (`predictor_mode='split'`). $x$ is to stay in $G',$ so the component has to be coarse-grained away, and **coarse-graining the output of a node with children is unsound**: a classical model of $G_1$ for the fine-grained distribution may let the children of $x$ read the correlated subvariable $X_S$ through the visible edge, and no model for the coarsened distribution need exist. The sound way to keep a predictor is node splitting with a *full* copy: $x$ is split into $x$ and $x'$ with the same parents and the same children, $x'$ is the predictor, and $x'$ is removed as an unsplit predictor would be.

For a childless $x$ the split mode coincides with keeping $x$ untouched (the childless copy is deleted). For a childful $x$ the copy is marginalized, which relays to $x$'s children everything the correlated subvariable could have told them. ([`QmDAG.fritz_steps`](../quantum_mDAG.py#L920) splits each childful predictor before justifying and realising the step.)

Example of the unsound shortcut: 0→1; Q{0,1}, Q{0,2}, Q{1,2} is saturated (latent-free equivalent, so it has no gap), yet "0 predicts 2, 0 kept untouched" would output 0→1; C{0,2}, Q{0,1}, which is [`QG_Instrumental_C`](../known_QC_gaps.py#L41): node 1 learns the instrument 2 from 0's fine-grained output, which is exactly what the Bonet inequality forbids the binary treatment to transmit. With the full copy marginalized, the step gives on the same example 0→1; C{0,1,2}, Q{0,1}, where the instrument also shares randomness with the outcome, no gap.

**The target: unsplit or split.**

* **Unsplit target** (`target_mode='unsplit'`): $s$ itself is restricted to $\mathrm{common}(s).$
* **Split target** (`target_mode='split'`): $s$ is split and the deletion is applied to the copy $s',$ so the original keeps all of its parents while $s'$ keeps $\mathrm{common}(s).$

The copy must keep the children of $s$: the classical direction turns a classical model in which $s$ outputs a pair $(s_1,s_2)$ into a model with $s:=s_1$ and $s':=s_2,$ and the children of $s$ may depend on $s_2.$ Removing an edge $s'\to c$ would be an unjustified deletion; those edges can only go later, by a piggyback that justifies it. [`fritz_steps`](../quantum_mDAG.py#L920) literally calls [`split_node`](../quantum_mDAG.py#L649) first and then runs the unsplit-target step on the copy; the pair's shared facet is absorbed by the facet the copy shares with the predictors.

Both target modes and both predictor modes are available to both certificates. How the search records the modes, and at which stages it applies them, is described in Sections [8](#8-the-fritz-search-procedure-qmdagfritz_steps-qc_gap_searchfritz_tricks) and [9](#9-search-certificates-and-the-four-node-census-qc_gap_searchpy-special-applicationsproving_qc_gapspy).

**Prediction, not perfect correlation.** In every mode, a Fritz step assumes one property of the distribution that the quantum direction constructs: the perfect prediction $H(s\mid\mathbf X)=0$ of [6.1](#61-the-mechanism). It does not assume that a predictor is perfectly correlated with the target, and it treats the copy of a split predictor as a predictor that keeps all of its parents, not as a twin of the target. This is stated here once, in four points; [6.1](#61-the-mechanism) and [6.10](#610-example-a-split-predictor-and-a-split-target-worked-out-in-full) refer to it.

*Prediction and the correlation of [6.1](#61-the-mechanism).* The perfect correlation of [6.1](#61-the-mechanism) is between $s$ and a subvariable of the predictors. It and the perfect prediction are interconvertible (given $H(s\mid\mathbf X)=0,$ $X_S:=g(\mathbf X)$ is a subvariable of the predictors' fine-grained output equal to $s$), so nothing is lost by stating the hypothesis in the weaker form.

*The unsplit predictor.* The unsplit predictor is *not* perfectly correlated with the target: its output is the pair (own value, prediction), of which only the second component equals $s,$ so $H(s\mid x)=0$ while $H(x\mid s)>0$ in general. The certificates use exactly this one direction.

*The copy of a split predictor.* In the quantum construction the copy carries nothing but the prediction: $x'$ outputs exactly $f(\mathrm{common}(s))=s,$ so the construction can make **the split-off copy perfectly correlated with the target**, $H(s\mid x')=0$ *and* $H(x'\mid s)=0,$ while $x$ keeps its original output. The piggyback, however, does not assume this. Its hypothesis is the one-directional perfect prediction $H(s\mid x')=0,$ and $x'$ keeps every parent it has, those $s$ cannot see included: as far as the piggyback is concerned $x'$ perfectly predicts $s$ but is not perfectly correlated with it, since a classical model may let $x'$ depend on parents $s$ does not share.

Accordingly the certificates of [6.6](#66-theorem-d-separation-certificate) and Section [7](#7-the-semigraphoid-closure-certificate-for-the-fritz-piggyback-semigraphoidpy-qmdag_semigraphoid_certificate) use only $H(s\mid x')=0,$ the same hypothesis they use for an unsplit predictor (in Section [7](#7-the-semigraphoid-closure-certificate-for-the-fritz-piggyback-semigraphoidpy-qmdag_semigraphoid_certificate) in the form "$`x'`$ is a sufficient statistic for $s$", [7.3](#73-the-hypotheses-and-how-perfect-prediction-enters-qmdag_semigraphoid_model-semigraphoidadd_functional_dependence)), and they treat $x'$ as a predictor, not as a second instance of the target. That is deliberate: it is what lets Theorem [6.6](#66-theorem-d-separation-certificate) cover the removal of the copy by marginalization, with all of the copy's parents relayed to $x$'s children, and nothing in the proof depends on the copy being more than a predictor.

*The unused direction.* The other direction, $H(x'\mid s)=0,$ which the construction could supply, is left unused, and it has to be. Restricting the copy itself to what it shares with $s$ before it is marginalized, so that fewer of its parents are relayed, is not a sound piggyback: Section [10](#10-open-questions) gives a structure without a gap that it would send to the instrumental scenario. And with the copy left alone, the second direction certifies no step that the first does not. [6.10](#610-example-a-split-predictor-and-a-split-target-worked-out-in-full) works through a census input where the distinction is visible.

### 6.5 The parent predictor: deletion without a certificate

The simplest Fritz piggyback is the one in which the predictor is a **visible parent** of the target. Then the perfect prediction is itself a legal kernel for $s,$ and nothing has to be certified.

**Claim.** Let $\mathbf X$ be a set of visible parents of $s.$ Let $G_1$ be $G$ with $s$ restricted to $\mathbf X$: every other parent of $s$ is deleted, the facets containing $s$ included, and no facet is made classical. Let $G'$ be $G_1$ with the predictors removed as in [6.1](#61-the-mechanism) (**Removing the predictor**), or, for split predictors ([6.4](#64-node-splitting-split-and-unsplit-predictors-split-and-unsplit-targets-qmdagsplit_node)), with their copies removed. Then a QC gap in $G'$ implies a QC gap in $G.$

**Proof.** *The quantum direction:* given $P'\in\mathcal Q(G'),$ realise it in $G_1$ through the removal of the predictors ([6.1](#61-the-mechanism), **Removing the predictor**); in $G_1$ the node $s$ reads $\mathbf X$ and nothing else, so it computes $s=h(\mathbf X,\varepsilon)$ with private noise $\varepsilon.$ Run the same strategies in $G,$ and let one predictor $x_0$ additionally output a copy of the random seed $\varepsilon,$ which $s$ then uses in place of its own noise (the edge $x_0\to s$ carries it). The result $P\in\mathcal Q(G)$ has $P'$ as its marginal over the predictors and satisfies $H(s\mid\mathbf X)=0,$ since $s=h(\mathbf X,\varepsilon)$ is a function of the predictors' fine-grained outputs.

*The classical direction:* let $M$ be a classical model of $G$ for $P.$ Then $s=g(\mathbf X)$ almost surely, and replacing the kernel of $s$ by $g(\mathbf X)$ leaves the joint distribution unchanged. Since every member of $\mathbf X$ is a parent of $s,$ the new kernel is a legal kernel of $G_1$: the model is Markov to $G_1.$ Removing the predictors as in [6.1](#61-the-mechanism) gives $P'\in\mathcal C(G').$ $\square$

Compared with the certified piggybacks of [6.6](#66-theorem-d-separation-certificate) and Section [7](#7-the-semigraphoid-closure-certificate-for-the-fritz-piggyback-semigraphoidpy-qmdag_semigraphoid_certificate), two things are different.

* The deletion is **maximal**: $s$ keeps the edges from the predictors only, so a facet shared by $s$ and a predictor is deleted as well, where a sibling predictor would keep it (classical for $s$). Nothing is lost by this: the predictor has the child $s,$ so it (or its copy) is always marginalized, and the marginalization re-creates the classical facet over the facet's other members and the predictor's children, $s$ included, that the sub-maximal deletion would have kept; on the census the two deletions give identical output structures for every pair.
* There is **no certificate**: the hypothesis the certified piggybacks must establish, that the function $g(\mathbf X)$ factors through parents $s$ keeps, is immediate because $\mathbf X$ consists of such parents.

In the code the pool `'parents'` lists the visible parents of a target ([`QmDAG.fritz_pool`](../quantum_mDAG.py#L739)), [`QmDAG.fritz_deletion`](../quantum_mDAG.py#L769) performs the maximal deletion for them, and [`QmDAG.fritz_certificate`](../quantum_mDAG.py#L796) returns the label `parent` without a test; the census runs this piggyback as its own stage, before every certified one ([9.1](#91-the-ten-stages)). A predictor that is both a visible parent and a latent sibling of $s$ belongs here, not to the certified pools.

**What the step achieves.** What the step does to $s$ is a real reduction: the parents of $s$ other than the predictors are gone for good, and no elementary piggyback deletes them. What the predictor's departure adds back is a classical version of the predictor's own inputs: unsplit, the predictor is marginalized (it has the child $s$) and relays its parents to $s$ and to its other children as edges and a classical facet; split, the copy $x'$ is a twin parent of $s,$ $s$ keeps the edge from $x'$ only, and marginalizing $x'$ relays $\mathrm{Pa}(x)$ to $s$ together with a classical facet over $x$'s children. The third route of [6.8](#68-example-deleting-a-visible-edge-three-routes) is the smallest example: 1 loses 0→1 and Q{1,2} and gains C{0,1,3}. After the step $s$ has no quantum parent at all.

**In the census.** In the four-node census the parent piggyback, applied first to every input the elementary piggybacks left, proves three of them ([1.1](#11-the-census-over-quantum-classes)): the family of [6.9](#69-example-a-split-target-with-a-childful-unsplit-predictor), where the predictor 0 is a parent of the target as well as its latent sibling, so that the step belongs here. Every one of the three needs a split target; an unsplit target would strip the quantum facet that survives into the seed.

### 6.6 Theorem (d-separation certificate)

**Claim.** Let $\mathbf X$ be a predictor set and $T$ a set of predicted nodes, each as in [6.1](#61-the-mechanism), and suppose that for every $s\in T,$ in the effective DAG of $G$ (private noise nodes included),

```math
\mathbf X\setminus\mathrm{common}(s)\ \perp_d\ \mathrm{others}(s)\ \big|\ \mathrm{common}(s).
```

(A predictor that is itself a parent of $s$ lies in $\mathrm{common}(s)$ and is conditioned on rather than tested; if every predictor is a parent of $s$ the condition is vacuous.) Let $G'$ be the output of [6.1](#61-the-mechanism), every $s\in T$ restricted to $\mathrm{common}(s),$ with the predictors removed as described there (**Removing the predictor**), in any order. Then a QC gap in $G'$ implies a QC gap in $G.$ The same holds for split predictors ([6.4](#64-node-splitting-split-and-unsplit-predictors-split-and-unsplit-targets-qmdagsplit_node): a childless predictor untouched; a childful one split, the copy being the predictor that is marginalized), by applying the statement to the split structure and composing with the node-splitting piggyback.

**Proof.** *The quantum direction:* given $P'\in\mathcal Q(G'),$ first realise it in the candidate $G_1$ through the removal of the predictors ([6.1](#61-the-mechanism), **Removing the predictor**). This gives $P_1\in\mathcal Q(G_1)$ with $P'$ as its marginal over the predictors. In $P_1$ the parents of each $s\in T$ are classical and $s$ has a channel from a predictor, so $s$ can be realised as a deterministic function $f_s(\mathrm{common}(s))$ with its private randomness drawn from that channel. Now run $P_1$ in $G$ and let each predictor additionally output what it sees of every $f_s(\mathrm{common}(s)).$ The result $P$ lies in $\mathcal Q(G),$ satisfies $H(s\mid\mathbf X)=0$ for every $s\in T,$ and has $P'$ as the marginal over the predictors.

*The classical direction:* let $M$ be a classical model of $G$ for $P.$ Take $s\in T$ and write $s=h(\mathrm{common},\mathrm{others})$ with $\mathrm{others}$ including the private noise, and $s=g(\mathbf X).$ Condition on $\mathrm{common}(s).$ By the d-separation hypothesis and (F2) for classical models, $\mathrm{others}(s)$ and $\mathbf X$ are independent given $\mathrm{common}(s).$ A random variable that is a function of each of two independent variables is almost surely constant, so $s=\varphi_s(\mathrm{common}(s))$ a.s. Replace the kernel of $s$ in $M$ by $\varphi_s;$ the joint distribution is unchanged.

Do this for each $s\in T$ in turn: every replacement leaves the joint distribution, hence the independences the next one uses, untouched, and the d-separation hypothesis of each $s$ is a statement about $G,$ not about the structure left by the earlier replacements. The resulting model is Markov to $G_1,$ the predictors still present with their fine-grained outputs. Removing the predictors as in [6.1](#61-the-mechanism) gives a classical model of $G'$ for the marginal of $P,$ which is $P'.$ Hence $P'\in\mathcal C(G').$ $\square$

**Remarks on the test.** The set test is equivalent to the conjunction of the per-edge tests $\mathbf X\perp_d v\mid\mathrm{Pa}(s)\setminus\lbrace v\rbrace$ over $v\in\mathrm{others}(s),$ by the weak-union and intersection properties of d-separation. Since $\mathrm{others}(s)$ always contains the private noise of $s,$ the d-separation test fails for every predictor that is a descendant of $s$ (the semigraphoid closure of Section [7](#7-the-semigraphoid-closure-certificate-for-the-fritz-piggyback-semigraphoidpy-qmdag_semigraphoid_certificate) does not, [8.10](#810-experiment-descendants-as-predictors)).

**In the code.** The implementation is [`QmDAG.fritz_certificate`](../quantum_mDAG.py#L796), which hands the parent case $\mathbf X\subseteq\mathrm{Pa}(s)$ to [6.5](#65-the-parent-predictor-deletion-without-a-certificate) (label `parent`) and otherwise tests the set at once with networkx's `is_d_separator`; the candidate is built by [`QmDAG._restrict_target`](../quantum_mDAG.py#L885) from the kept set, and the predictors are removed by [`QmDAG.fritz_realise`](../quantum_mDAG.py#L902).

### 6.7 Example: the tetrahedron, unsplit target and unsplit predictor

The one census input that needs a `Fritz` step of the cheapest kind and nothing more.

**Input.** No edges; Q{0,1,2}, Q{0,1,3}, Q{0,2,3}, Q{1,2,3}, every triple entangled.

**Predictor, target, modes.** Predictor 0, unsplit; predicted node 1, unsplit. The parents of 1 are the three facets containing it; 0 sees Q{0,1,2} and Q{0,1,3}, so $\mathrm{common}(1)$ is those two facets and $\mathrm{others}(1)$ is Q{1,2,3} plus the noise of 1.

**Test.** d-separation ([6.6](#66-theorem-d-separation-certificate)). The test asks whether 0 is separated from Q{1,2,3} given the two common facets: every path from 0 to Q{1,2,3} runs through 2 or 3 as a collider (0 ← Q{0,2,3} → 2 ← Q{1,2,3}), so it holds. In the quantum direction, 1 is a function of its two shares and 0, which holds the same shares, announces that function; classically, any model in which 0 predicts 1 perfectly cannot let 1 depend on Q{1,2,3}, which 0 never sees.

**Steps.**

* `Fritz`, predictor 0, unsplit target 1, predictor 0 unsplit. Node 1 keeps Q{0,1,2} and Q{0,1,3}, which become classical for it, and loses Q{1,2,3}; the childless 0 is deleted, so its shares vanish.

**Structure reached.** C{1,2}, C{1,3}, Q{2,3}, which is [`QG_Triangle`](../known_QC_gaps.py#L42).

### 6.8 Example: deleting a visible edge, three routes

Nothing in the mechanism distinguishes a visible parent of $s$ from a facet: $\mathrm{Pa}(s)$ contains both, and a parent the predictor cannot see is deleted either way.

**Input.** 0→1, 0→3; Q{1,2}, Q{2,3} (a census input, though one that node stitching already proves: stitching the sink 2 onto the source 0 gives [`QG_Evans`](../known_QC_gaps.py#L43)).

Three routes delete the edge 0→1. Routes 1 and 2 need a certificate and reach a seed; both are recorded by the search ([`tests/test_fritz.py`](../tests/test_fritz.py)). Route 3 is the parent piggyback of [6.5](#65-the-parent-predictor-deletion-without-a-certificate), which needs none, and it reaches a structure without a gap.

**Route 1: unsplit target, split predictor.**

**Predictor, target, modes.** Predictor 2, split; predicted node 1, unsplit. $\mathrm{Pa}(1)$ is the visible parent 0, the facet Q{1,2} and the noise of 1; 2 sees Q{1,2} and Q{2,3}, so $\mathrm{common}(1)$ is Q{1,2} and $\mathrm{others}(1)$ is the edge 0→1 and the noise.

**Test.** d-separation ([6.6](#66-theorem-d-separation-certificate)). The test asks whether 2 is separated from 0 given Q{1,2}: the only path, 2 ← Q{1,2} → 1 ← 0, is blocked at the collider 1. In the quantum direction, 1 is a function of its share of Q{1,2}, and 2 announces that function. In the classical direction, a model in which 2 predicts 1 perfectly cannot let 1 depend on 0, which 2 never sees.

**Steps.**

* `Fritz`, predictor 2, unsplit target 1, predictor 2 split. Node 1 keeps Q{1,2}, classical for it, and loses 0→1; 2 is childless, so split means untouched.

**Structure reached.** 0→3; C{1,2}, Q{2,3}, which is [`QG_Bell_C_Edge`](../known_QC_gaps.py#L80) (1 and 3 are the measurements, 2 and 0 their settings).

**Route 2: split target, unsplit predictor.**

**Predictor, target, modes.** Predictor 1, unsplit; predicted node 2, split: 2 is split into 2 and 2′ (Q{1,2,2′}, Q{2,2′,3}), and 1 sees Q{1,2,2′} only, so 2′ keeps that facet, classical for it, and loses its share of Q{2,2′,3}.

**Test.** d-separation ([6.6](#66-theorem-d-separation-certificate)), the certificate with which the search records this step.

**Steps.**

* `Fritz`, predictor 1, split target 2, predictor 1 unsplit. The copy 2′ is restricted as just described; the childless 1 is deleted. Here it is the *predictor*, not the target, that loses the edge: 1 is deleted and 0→1 with it.

**Structure reached.** 0→3; C{2,2′}, Q{2,3}, which is again [`QG_Bell_C_Edge`](../known_QC_gaps.py#L80) with the roles of the parties exchanged.

**Route 3: the parent piggyback of [6.5](#65-the-parent-predictor-deletion-without-a-certificate), split predictor.**

**Predictor, target, modes.** Predictor 0, a visible parent of 1, split; predicted node 1, unsplit.

**Test.** None: 0 is a parent of 1, so 0 predicts 1 with no certificate ([6.5](#65-the-parent-predictor-deletion-without-a-certificate)).

**Steps.**

* `Fritz`, predictor 0, unsplit target 1, predictor 0 split. Node 1 keeps the edge from 0 alone and loses Q{1,2}; with the predictor split, the copy 0′ that did the predicting is marginalized, which re-supplies the deleted edge as the classical facet C{0,1,3}.

**Structure reached.** 0→3; C{0,1,3}, Q{2,3}, a structure without a gap. A parent predictor deletes what it cannot see and hands back a classical version of what it can see, so it rarely leads anywhere; in the census the parent stage proves three inputs ([1.1](#11-the-census-over-quantum-classes)).

### 6.9 Example: a split target with a childful unsplit predictor

One of three census inputs that need a split-target `Fritz` step (four need a split target in some trick); the census records all three through the parent piggyback ([6.5](#65-the-parent-predictor-deletion-without-a-certificate), [1.1](#11-the-census-over-quantum-classes)), and this subsection gives the d-separation route of the first, which is recorded as well.

**Input.** 0→2, 1→2, 2→3; Q{0,1}, Q{0,2}, Q{1,3}.

**Predictor, target, modes.** Predictor 0 (childful: it feeds 2), unsplit; predicted node 1, split. $\mathrm{common}(1)$ is Q{0,1}; $\mathrm{others}(1)$ is Q{1,3} plus noise. The copy 1′ keeps Q{0,1}, classically, and the child 2 of 1; the original 1 keeps Q{1,3}. Unsplit, 0 is marginalized with teleportation (its share of Q{0,1} is teleported to its child 2 through Q{0,2}, and its facets become classical for 2). An unsplit target cannot do this: it would strip Q{1,3} from 1, and that facet is the one that survives into the seed.

**Test.** d-separation ([6.6](#66-theorem-d-separation-certificate)): 0 is separated from Q{1,3} given Q{0,1} because the paths through 1 and through 2 both end in colliders.

**Steps.**

1. `Fritz`, predictor 0, split target 1, predictor 0 unsplit. The copy 1′ keeps Q{0,1} (classical for it, as C{0,1,1′}) and the child 2; the original 1 keeps Q{1,3} and 2. Then 0 is marginalized with teleportation: its share of Q{0,1} is teleported to 2 through Q{0,2}, giving Q{1,2}; its classical facet is relayed to 2: 1→2→3, 1′→2; Q{1,2}, Q{1,3}, C{1,1′,2}.
2. Teleportation marginalization of 1: its share of Q{1,3} is teleported to 2 through Q{1,2}.

**Structure reached.** 1′→2→3; Q{2,3}, C{1′,2}, which is [`QG_Instrumental_EdgeC`](../known_QC_gaps.py#L40) (instrument 1′, treatment 2, outcome 3).

**The other two inputs.** The other two inputs that need a split-target `Fritz` step are 0→1→2→3; Q{0,1}, Q{0,2}, Q{1,3} and 0→1, 0→2, 1→2, 2→3; Q{0,1}, Q{0,2}, Q{1,3}. In both, 0 is a visible parent of 1, so the step is the parent piggyback of [6.5](#65-the-parent-predictor-deletion-without-a-certificate): the copy 1′ keeps the edge from 0 only, losing Q{0,1} as well as Q{1,3}, and 0 is then marginalized with teleportation. The output is the same 1→2→3, 1′→2; Q{1,2}, Q{1,3}, C{1,1′,2} in all three cases, because the marginalization of 0 relays its facets to its children as the classical facet C{1,1′,2} whether or not 1′ kept Q{0,1} classically; the second step is the same teleportation marginalization of 1. In the input of this subsection 0 is a parent of 2, not of 1, and the parent stage proves it through the pair (0, 2) instead, by a longer route ([8.9](#89-the-six-classes-that-need-a-fritz-step)).

### 6.10 Example: a split predictor and a split target, worked out in full

The one census input first proven with a split predictor by d-separation, with a split target.

**Input.** 0→1→2; Q{0,2}, Q{0,3}, Q{1,3}.

**Predictor, target, modes.** Predictor 0, which feeds 1, split; predicted node 3, split. Since 0 has a child, it is split into 0 and a full copy 0′ (both feeding 1); the copy predicts and is marginalized. $\mathrm{common}(3)$ is Q{0,3}, $\mathrm{others}(3)$ is Q{1,3} and noise.

**Test.** d-separation ([6.6](#66-theorem-d-separation-certificate)): $\mathrm{others}(3)$ is separated from 0 given Q{0,3} because 0→1←Q{1,3} is blocked at 1.

**Steps.** Worked out, with $s=3$ and $X=0$:

1. **Node splitting of the predicted node** (the split-target preprocessing, a piggyback of its own, [6.4](#64-node-splitting-split-and-unsplit-predictors-split-and-unsplit-targets-qmdagsplit_node)): 3 becomes 3 and 3′, same parents, no children: 0→1→2; Q{0,2}, Q{0,3,3′}, Q{1,3,3′}.
2. **One Fritz step**, predictor 0, target 3′ (the copy; no further split), predictor 0 split. The three operations below are bundled into this single step, and the structures after the first two are *not* QmDAGs in the sense of [0.1](#01-structures-and-model-sets): the one after the second operation carries the observational restriction that 0′ perfectly predicts 3′, $H(3'\mid 0')=0,$ so its model sets are not the unrestricted $\mathcal Q$ and $\mathcal C$ of its graph and no theorem of this note applies to it. Only the output of the third operation, with the predicting copy marginalized out, is a structure whose gap the piggyback asserts; pausing earlier would be unsound.

   As [6.4](#64-node-splitting-split-and-unsplit-predictors-split-and-unsplit-targets-qmdagsplit_node) states (**Prediction, not perfect correlation**), the step assumes only this prediction, and 0′ is a predictor that keeps all of its parents, not a twin of 3′: it is not pulled out of Q{0,0′,2}, which 3′ does not share, and its removal in the third operation is an ordinary marginalization covered by Theorem [6.6](#66-theorem-d-separation-certificate) (Section [10](#10-open-questions) records what assuming the correlation as well might buy).

   1. Split the predictor 0 into 0 and a full copy 0′ (same facets, same child 1): 0→1→2, 0′→1; Q{0,0′,2}, Q{0,0′,3,3′}, Q{1,3,3′}. The copy 0′ is the node that does the predicting; its output in the quantum construction is the value of 3′.
   2. Restrict 3′ to $\mathrm{common}(3')$: 3′ loses Q{1,3,3′}, which stays Q{1,3}, and keeps the facet shared with 0′, classical for it: 0→1→2, 0′→1; Q{0,0′,2}, Q{0,0′,3}, C{0,0′,3,3′}, Q{1,3}, with 0′ perfectly predicting 3′. The d-separation of [6.6](#66-theorem-d-separation-certificate) holds since 0′→1←Q{1,3} is blocked at 1.
   3. Marginalize 0′ (no teleportation: 0′ and 1 share no quantum facet), relaying its facets to 1 classically: 0→1→2; Q{0,2}, Q{0,3}, Q{1,3}, C{0,1,3,3′}, C{0,1,2}. The copy 3′ now has a single classical parent and no quantum one. This is the output of the Fritz step, and the first structure after the input whose gap follows.
3. **Postselection** on 3 (no visible parents; every facet of 3 lies within its latent siblings 0, 1, 3′): 3 is removed, Q{0,3} and Q{1,3} swap entanglement onto 0 and 1: 0→1→2; Q{0,1}, Q{0,2}, C{0,1,3′}, C{0,1,2}.
4. **Teleportation marginalization** of 0: its share of Q{0,2} is teleported to 1 through Q{0,1}, its classical facets lose 0.

**Structure reached.** 1→2; Q{1,2}, C{1,3′}, which is [`QG_Instrumental_C`](../known_QC_gaps.py#L41).

**Other routes for this input.** The census records this route at the fourth d-separation stage ([1.1](#11-the-census-over-quantum-classes)). At the same stage it records a second route, in which the predictor has no children: 3 predicts a copy of 0. That second route is the one its printed certificate shows, and the one in the table of [1.1](#11-the-census-over-quantum-classes). The input also has a closure route with an unsplit target: predictor 3 (childless, so split means untouched) predicts 1, which loses 0→1 and keeps Q{1,3} classically, giving 1→2; C{1,3}, Q{0,2}, Q{0,3}; marginalizing 1 (no quantum facet left, so plainly) gives C{2,3}, Q{0,2}, Q{0,3}, which degrades to [`QG_Triangle`](../known_QC_gaps.py#L42). The census does not record it, because every d-separation stage runs before every closure stage and the input is already proven.

**The order of steps 3 and 4.** The order of steps 3 and 4 cannot be swapped. Before the postselection, 0 and 1 share no quantum facet, so marginalizing 0 first cannot teleport its share of Q{0,2}; that share is relayed to 1 as a classical common cause only, leaving 1→2; Q{1,3}, C{1,2}, C{1,3,3′}, and postselecting on 3 then swaps nothing, since 3 has a single quantum sibling left, and leaves 1→2; C{1,2}, C{1,3′}, a structure with no quantum facet and hence no gap. The entanglement between 0 and 1 that the teleportation needs exists only after the postselection has swapped it there. With 0 unsplit instead of split, 0's facets land on 1 classically from the start and the same quantum link is lost.

**Why the chain of steps is sound.** The soundness of such a composition does not require the quantum directions of the steps to combine into one protocol: each step is the implication "gap in the output implies gap in the input", and the chain is a chain of implications.

In the quantum direction the steps are applied from the seed upwards, each to the distribution the previous one produced: for postselection, 3 performs a Bell measurement on its two shares; for the Fritz step, only 3′ and 0′ are re-implemented and the strategy of 3 is left untouched; 3′ depends on a classical facet alone, so it is a deterministic function of it with randomness from that facet, and 0′ announces that function. For the node splitting, 3 and 3′ are merged into one node of the input whose output is the pair, Bell outcome and predicted value; 0 predicts the second component only.

In the classical direction, which is the proof of the gap, each step takes an arbitrary classical model of its input and produces one of its output, using nothing about how the quantum directions were built.

## 7. The semigraphoid closure certificate for the Fritz piggyback (`semigraphoid.py`, `QmDAG._semigraphoid_certificate`)

Section [6](#6-the-fritz-piggyback-fritz) justified a deletion by d-separation. This section gives a stronger test of the same deletion. The hypotheses are conditional independences; they are closed under the semigraphoid rules; and the deletion is certified when the Markov conditions of the reduced structure lie in the closure.

The subsections, in reading order:

* [7.1](#71-why-the-same-mechanism-admits-a-stronger-certificate) says why the mechanism of Section [6](#6-the-fritz-piggyback-fritz) admits a stronger test.
* [7.2](#72-conditional-independence-as-the-proof-system-semigraphoids-and-the-exchange-rule-semigraphoidpy) sets up the proof system: conditional independence statements, the semigraphoid rules and the closure.
* [7.3](#73-the-hypotheses-and-how-perfect-prediction-enters-qmdag_semigraphoid_model-semigraphoidadd_functional_dependence) lists the hypotheses and says how perfect prediction enters them.
* [7.4](#74-the-two-target-sets-as-containment-of-d-separation-models-qmdag_semigraphoid_certificate) says what has to be derived: the two target sets, `relabel` and `markov`.
* [7.5](#75-worked-example-the-kpc-structure) works one example through, the structure of Khanna, Pusey and Colbeck (KPC). It comes before the proofs so that they can refer to its nodes.
* [7.6](#76-why-relabel-is-justified) explains why deriving the `relabel` target set justifies the deletion.
* [7.7](#77-theorem-semigraphoid-closure-certificate) states and proves the theorem.
* [7.8](#78-further-examples-from-the-census) gives further examples, from the census.
* [7.9](#79-strength-agreement-with-the-lp-and-cost) compares the closure with the entropic LP of [Appendix B](#appendix-b-the-entropic-lp-as-an-alternative-certificate-engine-entropic_lppy-qmdag_entropic_certificate): what the closure cannot derive, where the two agree, and what each costs.

### 7.1 Why the same mechanism admits a stronger certificate

In the classical direction of [6.6](#66-theorem-d-separation-certificate) the only facts about the classical model that were used were the Markov property of $G$ and $H(s\mid\mathbf X)=0.$ But the constructed distribution $P$ has more structure: it lies in $\mathcal Q(G_1),$ so by (F2) **every observable d-separation of $G_1$ holds for $P$**, and therefore holds in any classical model of $G$ for $P.$ These conditional independences are properties of the *new* graph, obtained for free, and there is no reason not to use them as hypotheses when deciding whether a deletion is justified in the *original* graph. The d-separation test of [6.6](#66-theorem-d-separation-certificate) ignores them; it is a shortcut that trades proof power for speed. Khanna, Pusey and Colbeck's six-node example ([7.5](#75-worked-example-the-kpc-structure)) is a case where the shortcut fails and the extra independences are exactly what is needed.

The stronger certificate is a *proof system for conditional independence*: given the independences the hypotheses provide, derive the independences the candidate needs. Two engines implement it.

* The **semigraphoid closure** is the default and the subject of this section. KPC also use it for their Lemma 1. It reasons about conditional independence statements directly, with the four semigraphoid rules, and costs about a millisecond per candidate.
* The **entropic LP** of [Appendix B](#appendix-b-the-entropic-lp-as-an-alternative-certificate-engine-entropic_lppy-qmdag_entropic_certificate) is the alternative. It reasons about entropy vectors with the Shannon inequalities; it is stronger in principle, needs a commercial solver and a tenth of a second per candidate, and on the four-node census proves exactly the same steps ([7.9](#79-strength-agreement-with-the-lp-and-cost)).

The two have the same hypotheses and the same target sets (the statements to be derived, [7.4](#74-the-two-target-sets-as-containment-of-d-separation-models-qmdag_semigraphoid_certificate)); only the inference engine differs.

### 7.2 Conditional independence as the proof system: semigraphoids and the exchange rule (`semigraphoid.py`)

This subsection fixes the statements the proof system works with, its rules, the compact form in which the engine stores a set of statements, and the d-separation models that serve as hypotheses and as target sets.

**Variables and notation.** The variables are those of a classical model of $G$: the visible nodes and the facet latents ([`QmDAG.lp_structure`](../quantum_mDAG.py#L522); quantum facets are ordinary latent variables here, since only classical models are analysed, F2), $n$ of them, 5 to 12 in the census, one more per split. Write $\langle A,B\mid C\rangle$ for "$`A`$ is independent of $B$ given $C$", for disjoint sets of variables. Such a statement is called a triplet.

**Semigraphoids.** The independences that hold in a given distribution form a **semigraphoid** (Pearl and Paz; Studený): a set of triplets closed under four rules:

* symmetry
* decomposition: $\langle A,BD\mid C\rangle$ ⇒ $\langle A,B\mid C\rangle$
* weak union: $\langle A,BD\mid C\rangle$ ⇒ $\langle A,B\mid CD\rangle$
* contraction: $\langle A,B\mid C\rangle\wedge\langle A,D\mid BC\rangle$ ⇒ $\langle A,BD\mid C\rangle$

**The proof system.** The certificate's proof system is exactly these rules:

* the hypotheses are triplets that hold in every classical model of $G$ for the constructed distribution;
* the **semigraphoid closure** of the hypotheses is the set of triplets derivable from them by the rules;
* a target triplet, that is, a statement to be derived, is certified when it lies in the closure.

Every derivation is valid for every distribution, with or without full support: the rules do not include intersection, which needs positivity.

**Elementary triplets.** A semigraphoid is determined by its **elementary triplets** $\langle i,j\mid K\rangle,$ single variables $i,j$ and a conditioning set $K$: a general triplet $\langle A,B\mid C\rangle$ belongs to it iff $\langle i,j\mid K\rangle$ does for every $i\in A,$ $j\in B$ and every $K$ with $C\subseteq K\subseteq(A\cup B\cup C)\setminus\lbrace i,j\rbrace$ (the chain rule $I(A{:}B\mid C)=\sum I(i{:}j\mid K)$ along any ordering, every term nonnegative).

**How the engine stores them.** The engine stores a set of elementary triplets, called a **model** below (not to be confused with a classical model of a structure), as a boolean array indexed by $i,$ $j$ and the bitmask of $K,$ 1.4 MB at $n=13;$ a hypothesis is added by writing all its components ([`semigraphoid.add_triplet`](../semigraphoid.py#L201)) and a target triplet is checked by reading them ([`semigraphoid.holds`](../semigraphoid.py#L227), [`semigraphoid.contains`](../semigraphoid.py#L241)).

**The exchange rule.** A set of elementary triplets is the elementary part of a semigraphoid iff it is closed under one rule, the **exchange rule**, for distinct $i,j,k$ and $K$ disjoint from them (Matúš; Studený, *Probabilistic Conditional Independence Structures*, Lemma 2.2):

```math
\langle i,j\mid K\rangle\ \wedge\ \langle i,k\mid K\cup j\rangle
\quad\Longleftrightarrow\quad
\langle i,k\mid K\rangle\ \wedge\ \langle i,j\mid K\cup k\rangle ,
```

the elementary form of contraction, weak union and decomposition at once (it is the identity $I(i{:}j\mid K)+I(i{:}k\mid Kj)=I(i{:}k\mid K)+I(i{:}j\mid Kk)$ between nonnegative terms, so both sides vanish together).

**Computing the closure.** Sweeping the rule over all cells until nothing changes computes the semigraphoid closure ([`semigraphoid.close`](../semigraphoid.py#L445)): a sweep costs $n\binom{n-1}{2}2^{n-3}$ cell checks (0.9 million at $n=13$), the fixpoint is reached in two to four sweeps, and the compiled kernel (`numba`) takes about [0.2](#02-piggybacks) ms at $n=10$ and 5 ms at $n=13$ per closure. A pure-numpy implementation of the same rule, ten times slower, is the fallback without `numba` and a cell-by-cell check of the compiled one in the tests ([`tests/test_semigraphoid.py`](../tests/test_semigraphoid.py)).

**d-separation models.** For a DAG $H$ on the variables, the triplets $\langle A,B\mid C\rangle$ with $A\perp_d B\mid C$ form a semigraphoid, the **d-separation model** of $H,$ and it is the semigraphoid closure of the local Markov triplets $\langle v,\ \mathrm{nondesc}(v)\setminus\mathrm{pa}(v)\mid\mathrm{pa}(v)\rangle$ (Verma and Pearl; Geiger, Verma and Pearl): every d-separation follows from the Markov statements by the four rules, and nothing beyond the d-separations does, since classical models realise every non-d-separated dependence (F2).

The engine computes d-separation models directly rather than by closure, with a bit-parallel Bayes-ball walk, one walk per source and conditioning set ([`semigraphoid.dsep_all`](../semigraphoid.py#L507)); the same enumerator serves the whole code, in the LP's hypotheses and in [`mDAG`](../mDAG_advanced.py#L64)'s own conditional-independence and e-separation enumerators. Both facts, closure of the Markov triplets equals the d-separation model and the walk agrees with networkx cell by cell, are tested on random DAGs.

### 7.3 The hypotheses, and how perfect prediction enters (`QmDAG._semigraphoid_model`, `semigraphoid.add_functional_dependence`)

Here $s$ is the predicted node and $\mathbf X$ the predicting set (for a split predictor, [6.4](#64-node-splitting-split-and-unsplit-predictors-split-and-unsplit-targets-qmdagsplit_node), the copy $x'$). Three kinds of fact hold for the joint distribution of *every* classical model of $G$ for the constructed distribution, and the certificate starts from all three:

1. **Markov($G$).** The d-separation model of $G$ over the visible nodes and the facets, with the private noises integrated out: a d-separation among these nodes in the effective DAG is one in [`lp_structure`](../quantum_mDAG.py#L522), and (F2) for classical models makes it an independence of the joint. It is computed once per structure and cached ([`QmDAG._semigraphoid_model`](../quantum_mDAG.py#L601)).

2. **Perfect prediction, $H(s\mid\mathbf X)=0.$** A vanishing conditional entropy is not a conditional independence, and a semigraphoid has no cell for it. What the closure takes from it is the statement that **$\mathbf X$ is a sufficient statistic for $s$: $s$ is conditionally independent of every other variable given $\mathbf X$**,

   $$\big\langle s,\ V\setminus(\mathbf X\cup s)\ \big|\ \mathbf X\big\rangle ,$$

   entered as the elementary triplets $\langle s,j\mid K\rangle$ for every $K\supseteq\mathbf X$ with $s\notin K$ and every $j\notin K\cup\lbrace s\rbrace$ ([`semigraphoid.add_functional_dependence`](../semigraphoid.py#L213)): by the expansion into elementary triplets ([7.2](#72-conditional-independence-as-the-proof-system-semigraphoids-and-the-exchange-rule-semigraphoidpy)) these components are exactly the general triplet together with all its weak-union consequences.

   *Why this hypothesis is sound.* The sufficiency statement is implied by $H(s\mid\mathbf X)=0,$ since $I(s{:}j\mid K)\le H(s\mid K)\le H(s\mid\mathbf X)$ for $K\supseteq\mathbf X,$ so the hypothesis the closure uses is weaker than the one the quantum direction provides, and soundness is unaffected.

   *What is left out.* The sufficiency statement leaves out the other half of $H(s\mid\mathbf X)=I(s{:}\mathrm{rest}\mid\mathbf X)+H(s\mid\mathbf X,\mathrm{rest})$: the vanishing of $H(s\mid\text{everything else}),$ that $s$ carries no randomness of its own beyond the other variables. That half is not a conditional independence either (in the extended formalisms that let the two sides of a triplet overlap, it is the degenerate triplet $\langle s,s\mid V\setminus s\rangle$), and the engine does not represent it; the LP of [Appendix B](#appendix-b-the-entropic-lp-as-an-alternative-certificate-engine-entropic_lppy-qmdag_entropic_certificate), which works with entropies, uses $H(s\mid\mathbf X)=0$ in full, and [7.9](#79-strength-agreement-with-the-lp-and-cost) says what the difference amounts to on the census (nothing).

   *No separate determinism rules.* The "determinism rules" one might be tempted to add on top of the sufficiency triplets, that a function of the conditioning set may be added to or removed from it ($\langle i,j\mid K\rangle$ ⇔ $\langle i,j\mid Ks\rangle$ when $\mathbf X\subseteq K$) or moved into a separated side ($\langle i,j\mid K\rangle$ ⇔ $\langle is,j\mid K\rangle$ when $\mathbf X\subseteq Ki$), are instances of the exchange rule with one of these triplets as a premise, so they come out of the closure by themselves and must not be implemented separately.

   *One statement for both predictor modes.* The same sufficiency statement serves both predictor modes of [6.4](#64-node-splitting-split-and-unsplit-predictors-split-and-unsplit-targets-qmdagsplit_node).

   * For an unsplit predictor it is literally what the quantum direction provides: $x$ outputs its own value and the prediction, so $H(s\mid x)=0$ while $x$ is not a function of $s.$
   * For a split predictor the quantum construction can make the copy $x'$ perfectly correlated with $s,$ $H(s\mid x')=0$ *and* $H(x'\mid s)=0,$ but the piggyback hypothesises only the first direction, in the same sufficiency form, and treats $x'$ as a predictor that keeps all of its parents; the second direction is deliberately left unused ([6.4](#64-node-splitting-split-and-unsplit-predictors-split-and-unsplit-targets-qmdagsplit_node), Section [10](#10-open-questions)).

   *Several predicted nodes.* Several predicted nodes of one $\mathbf X$ would each contribute their own triplets (the engine accepts a set); beyond d-separation the search certifies one predicted node at a time ([8.5](#85-joint-steps-and-why-not-a-second-fritz-depth)).

3. **The observable d-separations of the candidate, $\mathcal I(G_1).$** The d-separation model of $G_1$ restricted to the visible nodes ([`semigraphoid.dsep_all`](../semigraphoid.py#L507) with the kept parent sets, visible block only). Every one of these triplets holds for the constructed distribution by (F2), hence for the joint of any classical model of $G$ producing it. Note that they are facts about $G_1,$ the graph *after* the deletion; nothing in the graph of $G$ provides them.

The closure of the union of the three is computed once per candidate (one call of [`semigraphoid.close`](../semigraphoid.py#L445) on the copied model of $G$ with the triplets of 2 and 3 written in) and read against the target sets of [7.4](#74-the-two-target-sets-as-containment-of-d-separation-models-qmdag_semigraphoid_certificate).

### 7.4 The two target sets, as containment of d-separation models (`QmDAG._semigraphoid_certificate`)

**Target set and target.** The two terms name different things.

* The **target** of a Fritz step is a node: the predicted node $s$ (Section [6](#6-the-fritz-piggyback-fritz)).
* A **target set** is a set of conditional independence statements, the **target triplets**, that are to be derived from the hypotheses of [7.3](#73-the-hypotheses-and-how-perfect-prediction-enters-qmdag_semigraphoid_model-semigraphoidadd_functional_dependence). The candidate is certified when the closure contains every one of them.

So a target set is not a set of targets. Several predicted nodes restricted in one step (a joint step, [6.2](#62-the-unit-a-predictor-and-its-targets) and [8.5](#85-joint-steps-and-why-not-a-second-fritz-depth)) are called "a set of predicted nodes" in this section.

**The two target sets.** Two target sets are implemented, `relabel` and `markov`. Each target set is the local Markov property of a DAG over $G$'s variables, and a closed model contains a DAG's local Markov triplets iff it contains its whole d-separation model ([7.2](#72-conditional-independence-as-the-proof-system-semigraphoids-and-the-exchange-rule-semigraphoidpy)), so each check is the containment of one d-separation model in the closure, one call of [`semigraphoid.contains`](../semigraphoid.py#L241).

* **`relabel`**: applicable when there is a single predicted node $s$ and it keeps exactly one parent in $G_1,$ a facet $L$ (the code tests `len(predicted) == 1`, `len(common) == 1` and that the element is a facet index). Let $G''$ be $G_1$ with $L$ deleted and $s$ made a parent of every other child of $L.$ Target set: the d-separation model of $G''$ over $G$'s variables other than $L$ ([`semigraphoid.restrict`](../semigraphoid.py#L246)).
* **`markov`**: the d-separation model of $G_1$ itself over $G$'s own variables. If it is contained, the joint of any classical model of $G$ satisfying the hypotheses is Markov to $G_1,$ hence $P\in\mathcal C(G_1).$

Each target set is one d-separation model, so it is also called a target model, for instance the `relabel` model. For `markov` the item above already says why containment justifies the deletion. For `relabel` that is the subject of [7.6](#76-why-relabel-is-justified), after the example of [7.5](#75-worked-example-the-kpc-structure).

**Which one runs.** The census runs only `relabel`, the one that reaches beyond d-separation in practice ([1.1](#11-the-census-over-quantum-classes)). `markov` is an option that is off, since it never decided an input ([9.3](#93-certificates-beyond-d-separation-attempted)); [`default_stages(lp_markov_target=True)`](../qc_gap_search.py#L145) turns it on, and it is then tried only where `relabel` is inapplicable or fails.

**Place in the search.** In the search ([9.1](#91-the-ten-stages)) the d-separation test of [6.6](#66-theorem-d-separation-certificate) is run first on every candidate and the closure only where it fails, so a transition's certificate, `dsep` or `semigraphoid`, says which justification the step needs (`entropic` when the LP of [Appendix B](#appendix-b-the-entropic-lp-as-an-alternative-certificate-engine-entropic_lppy-qmdag_entropic_certificate) is the engine, [B.2](#b2-the-hypotheses-and-the-target-sets-as-lp-rows-qmdag_entropic_certificate)). Every step beyond d-separation predicts a single node ([8.5](#85-joint-steps-and-why-not-a-second-fritz-depth)).

### 7.5 Worked example: the KPC structure

This is the six-node example of Khanna, Pusey and Colbeck announced in [7.1](#71-why-the-same-mechanism-admits-a-stronger-certificate): four visible nodes and two latent ones. It shows the closure at work on a structure where the d-separation test fails. That the containment found here justifies the deletion is proven in [7.6](#76-why-relabel-is-justified), which uses this structure as its example, and stated as a theorem in [7.7](#77-theorem-semigraphoid-closure-certificate).

**The structure.** $G$ has the visible nodes $C,D,E,F;$ the latent facets $A=\lbrace E,F\rbrace$ and $B=\lbrace D,F\rbrace;$ and the edges $C\to D,$ $C\to E,$ $D\to E.$

**The code's labels.** Where a structure is written in the code's notation (visible edges, then facets, with Q for a quantum and C for a classical facet, Section [6](#6-the-fritz-piggyback-fritz)), the nodes are numbered $F=0,$ $C=1,$ $D=2,$ $E=3.$ In that notation $G$ is 1→2, 1→3, 2→3; Q{0,2}, Q{0,3}, where $B$ is Q{0,2} and $A$ is Q{0,3}. The digits are needed because the letter C is also the prefix of a classical facet in that notation: C{0,3} is a classical facet on the nodes $E$ and $F$ (its C is not the node $C$ of this example).

**The step.** $F$ is the predictor and $E$ the predicted node. In the notation of Section [6](#6-the-fritz-piggyback-fritz), $E$ keeps $\mathrm{common}(E)=\lbrace A\rbrace,$ its parents that $F$ sees, and loses $\mathrm{others}(E)=\lbrace C,D,\text{noise}\rbrace.$ The d-separation test fails, since $F\leftarrow B\to D$ is open. The candidate $G_1$ has $E$ with parent $A$ only and d-separates $E$ from $\lbrace C,D\rbrace.$ With the hypothesis $E\perp CD$ the closure contains the `relabel` model ($A:=E$). This reproduces KPC's Lemma 1, which they too establish by a semigraphoid closure, without the equality $F_S=E$: their split node $F_S$ is exactly the perfectly correlated copy of [6.4](#64-node-splitting-split-and-unsplit-predictors-split-and-unsplit-targets-qmdagsplit_node) that a split childless predictor $F$ implicitly deletes, so splitting $F$ is sound and leaves $F$ untouched.

**Split or unsplit predictor.** With $F$ split the output is 1→2; Q{0,2}, C{0,3}, that is, the edge $C\to D$ with the facet $B$ still quantum and the facet $A$ now classical: the Bell variant [`QG_Bell_C_Edge`](../known_QC_gaps.py#L80), in one step ([`tests/test_fritz_entropic.py`](../tests/test_fritz_entropic.py)). With $F$ unsplit, $E$ is left with a private facet only and nothing is learnt: this is a case where split predictors are essential.

How this single step relates to the four-step derivation in Figure 4 of the KPC draft, and what each engine certifies of that derivation, is set out in [kpc_figure4_vs_code.md](kpc_figure4_vs_code.md).

**The step worked out**, with $s=E$ and $X=F$:

* Candidate $G_1$: $E$ keeps $A$ only (classical for it) and loses $C\to E,$ $D\to E$ and its noise: 1→2; Q{0,2}, C{0,3}, with $C$ and $D$ as before. Observable d-separations of $G_1$: $E\perp CD$ is the one that matters; example 3 of [7.8](#78-further-examples-from-the-census) lists all of them.
* Closure: from Markov($G$), the sufficiency of $F$ for $E$ ($\langle E,\ ABCD\mid F\rangle$ and its components) and $E\perp CD,$ the `relabel` model, the d-separations of $G''$ with $E$ a root and a parent of $F,$ is contained in the closure. Output, $F$ untouched because it is split and childless: 1→2; Q{0,2}, C{0,3}, which is [`QG_Bell_C_Edge`](../known_QC_gaps.py#L80).

### 7.6 Why `relabel` is justified

The claim to be proven is not about one classical model. It is: *every* classical model of $G$ whose observed statistics satisfy the perfect prediction and the independences $\mathcal I(G_1)$ can be exchanged for a classical model of $G_1$ with the same observed statistics. The argument has three steps.

1. **Fix one model.** Any classical model $M$ of $G$ producing $P$ defines one joint distribution $Q$ over the visible nodes and the facet latents (noises integrated out). The hypotheses of [7.3](#73-the-hypotheses-and-how-perfect-prediction-enters-qmdag_semigraphoid_model-semigraphoidadd_functional_dependence) are facts about $Q$: $Q$ is Markov to $G;$ $\mathbf X$ is sufficient for $s$ (a consequence of $H(s\mid\mathbf X)=0$); the independences of $\mathcal I(G_1)$ hold among the visible nodes. The independences of $Q$ form a semigraphoid, so it contains the hypotheses and with them their closure.
2. **What the closure proves.** Every triplet in the closure therefore holds for $Q,$ in particular every target triplet. Since $M$ was arbitrary, the target triplets hold for every model satisfying the hypotheses.
3. **From Markov($G''$) to a model of $G_1.$** The target triplets are the local Markov conditions of $G''$ over $W=V\cup(\text{facets}\setminus\lbrace L\rbrace).$ For a DAG, the local Markov conditions are equivalent to the factorisation of $Q(W)$ into one kernel per node given its parents in $G'',$ so the marginal of $Q$ over $W$ *is* a classical model of $G'',$ one in which $s$ is a root.

   Now rename:

   * introduce a fresh latent $\lambda'$ carrying the value of $s,$ with the root distribution $Q(s);$
   * let $s$ read $\lambda'$ deterministically (its private noise in $G_1$ may be trivial);
   * let every other former child $c$ of $L$ keep its $G''$ kernel with $s$ replaced by $\lambda',$ which is legal because the parents of $c$ in $G_1$ are its parents in $G''$ with $s$ replaced by $L;$
   * the kept quantum part of the facet over $F\setminus\lbrace s\rbrace$ is left unused (here $F$ is the node set of the facet $L$ as in [6.1](#61-the-mechanism), not the node $F$ of [7.5](#75-worked-example-the-kpc-structure));
   * everything else is unchanged.

   This is a classical model of $G_1$ with observed marginal $Q(V)=P.$ The original latent $\lambda_L$ of $M$ is simply discarded: in the new model the common cause of $s$ and its siblings is played by the value of $s$ itself.

In the KPC example ([7.5](#75-worked-example-the-kpc-structure)) this reads as follows, with $s=E$ and $L=A$ and with $A'$ written for the new latent $\lambda'$: every model in which $E$ depends on $C$ and $D,$ with $F$ predicting $E$ and $E$ independent of $(C,D),$ is replaced by a model in which $E$ depends on a single latent $A'$ alone, where $A'$ is the old value of $E.$

**Why `markov` cannot do this.** The model $E=A\oplus B,$ $D=C\oplus B,$ $F=(A,B)$ satisfies every hypothesis of the KPC candidate, yet $E$ is not a function of the given $A.$ No certificate that keeps $A$ as the witness can succeed; `relabel` succeeds because it discards $A.$

**Why the single-parent condition.** If $s$ kept another parent $v$ in $G_1,$ then $s$ would not be a root of $G'',$ and $\lambda':=s$ would be correlated with $v,$ violating the independence of roots in $G_1.$

**Neither target set implies the other.**

* `relabel` without `markov` is the KPC example.
* `markov` without `relabel`: let $s$ read a visible parent $v$ and one facet $L=\lbrace s,c,d,x\rbrace$ shared with the predictor $x.$ d-separation certifies deleting $v\to s,$ so `markov` holds, but the `relabel` target set contains $d\perp c\mid s,$ which the model $L=(L_1,L_2),$ $s=L_1,$ $c=d=L_2,$ $x=L$ violates with $I(d{:}c\mid s)=1$ bit while satisfying every hypothesis ([`tests/test_fritz_entropic.py`](../tests/test_fritz_entropic.py)).

**One scheme behind both.** Both target sets are instances of one scheme, replace each latent of $G_1$ by a set of $G$'s variables and ask for Markov($G_1$) with the substitution: `markov` substitutes $\lbrace L\rbrace,$ `relabel` substitutes $\lbrace s\rbrace.$ Further substitutions are an open direction (Section [10](#10-open-questions)).

**Relation to the d-separation certificate.** The classical direction of [6.6](#66-theorem-d-separation-certificate) derives $s\perp\mathrm{others}\mid\mathrm{common}$ and keeps the given latents, so Theorem [6.6](#66-theorem-d-separation-certificate) always produces a `markov`-type witness and never a relabelling.

The closure subsumes it for the row of $s.$ Write $K=\mathrm{common}(s),$ $D=\mathrm{others}(s)$ and $N$ for the non-descendants of $s$ outside $\mathrm{Pa}(s)$ (here $D$ is a set of parents, not the node $D$ of [7.5](#75-worked-example-the-kpc-structure)). If $\mathbf X\setminus K\perp_d D\mid K$ then:

1. Markov($G$) contains $\langle D,\mathbf X\setminus K\mid K\rangle;$
2. sufficiency gives $\langle D,s\mid K\cup\mathbf X\rangle;$
3. contraction and decomposition give $\langle D,s\mid K\rangle;$
4. Markov($G$) also contains $\langle s,N\mid K\cup D\rangle,$ and contraction once more gives $\langle s,D\cup N\mid K\rangle,$ the local Markov triplet of $s$ in $G_1.$

The `markov` rows of the deleted parents, whose non-descendant sets grow in $G_1,$ were checked empirically (every d-separation-admissible pair in the test structures is admissible for the closure); the search never relies on this, since d-separation-certified candidates are admitted by Theorem [6.6](#66-theorem-d-separation-certificate) without a closure.

`relabel` has no d-separation analogue in $G$: its hypotheses $\mathcal I(G_1)$ are facts about $G_1,$ not about the graph of $G.$

### 7.7 Theorem (semigraphoid closure certificate)

**Claim.** Let $G_1$ be a candidate as in [6.1](#61-the-mechanism), possibly with further parents deleted from any non-predictor node (the extra deletions of Section [10](#10-open-questions)), and $G'$ the output with the predictors, or the copies of split predictors, removed as in [6.6](#66-theorem-d-separation-certificate). If the semigraphoid closure of the hypotheses of [7.3](#73-the-hypotheses-and-how-perfect-prediction-enters-qmdag_semigraphoid_model-semigraphoidadd_functional_dependence) contains the `markov` target model, or the single-parent condition holds and it contains the `relabel` target model, then a QC gap in $G'$ implies a QC gap in $G.$

**Proof.** *The quantum direction* is as in [6.6](#66-theorem-d-separation-certificate): $P\in\mathcal Q(G)$ with $H(s\mid\mathbf X)=0;$ since the constructed strategy is a strategy for $G_1$ with fine-grained predictor outputs, (F2) gives that $P$ satisfies every observable d-separation of $G_1.$

*The classical direction.* Let $M$ be a classical model of $G$ for $P$ and $Q$ its joint over the visible nodes and the facets. $Q$ is Markov to $G,$ so it satisfies the d-separation model of $G$ (F2 for classical models); it satisfies the sufficiency triplets, since $I(s{:}j\mid K)\le H(s\mid\mathbf X)=0$ for $K\supseteq\mathbf X;$ and it satisfies $\mathcal I(G_1).$ The independences of $Q$ form a semigraphoid, so they include the closure of these hypotheses and with it every target triplet: $Q$ is Markov to $G_1$ (or to $G'',$ and then [7.6](#76-why-relabel-is-justified) gives a model of $G_1$ with the same observed marginal). Removing the predictors as in [6.6](#66-theorem-d-separation-certificate), by deletion or marginalization, gives $P'\in\mathcal C(G').$ $\square$

The same argument read for a set of predicted nodes, one sufficiency hypothesis per node, one candidate $G_1$ restricting all of them and the `markov` target set, is sound as well (the `relabel` substitution is per predicted node, and the relabelled certificates of two predicted nodes do not obviously compose). The code certifies sets of predicted nodes by d-separation only (the joint steps of [6.2](#62-the-unit-a-predictor-and-its-targets) and [8.5](#85-joint-steps-and-why-not-a-second-fritz-depth)); the joint closure certificate was never decisive in four nodes, and it is listed in Section [10](#10-open-questions).

### 7.8 Further examples from the census

**Which structures.** Six structures among the inputs have a semigraphoid-closure certificate with a split predictor and an unsplit target (the LP of [Appendix B](#appendix-b-the-entropic-lp-as-an-alternative-certificate-engine-entropic_lppy-qmdag_entropic_certificate) certifies the same steps), and no input has one at any other closure stage (a closure stage runs the closure where d-separation fails, [8.7](#87-two-kinds-of-stage-d-separation-breadth-first-then-the-semigraphoid-closure-with-early-exit)). Of the six:

* five are the five members of the KPC class ([1.1](#11-the-census-over-quantum-classes)), which is proven only this way, at the second closure stage of the cascade (the cascade is the fixed sequence of search stages, [9.1](#91-the-ten-stages));
* the sixth, the split-predictor example of [6.10](#610-example-a-split-predictor-and-a-split-target-worked-out-in-full), also has a d-separation route with a split target, which the census records since every d-separation stage runs first.

**What the six share.** All six use a childless predictor $X$ that is split, and the `relabel` target set for the predicted node $s.$ Examples 1 to 5 are written in the labels of [7.5](#75-worked-example-the-kpc-structure), so that $X=0$ and $s=3$ (example 3 is the KPC structure itself); example 6 is written in the labels of [6.10](#610-example-a-split-predictor-and-a-split-target-worked-out-in-full), with $X=3$ and $s=1.$

Splitting a predictor always means the split-and-marginalize of [6.4](#64-node-splitting-split-and-unsplit-predictors-split-and-unsplit-targets-qmdagsplit_node): $X$ is split into $X$ and a full copy $X',$ the copy does the predicting, and the copy is removed, by deletion when childless and by marginalization otherwise. In every example below $X$ is childless, so the copy is simply deleted and the operation leaves no trace in the output; the examples look as if $X$ were kept untouched, which is exactly the childless case of [6.4](#64-node-splitting-split-and-unsplit-predictors-split-and-unsplit-targets-qmdagsplit_node).

**How to read an example.** Each example has four parts:

* **Input.** The structure $G$ to which the step is applied.
* **Output.** The structure that the step produces, and how it reaches a seed (a structure with an established QC gap, [0.7](#07-the-census-the-seeds-and-the-names-of-the-tricks)).
* **Facts.** What the closure is given beyond Markov($G$): the perfect prediction that the construction in the quantum direction arranges, in the one-directional sufficiency form of [7.3](#73-the-hypotheses-and-how-perfect-prediction-enters-qmdag_semigraphoid_model-semigraphoidadd_functional_dependence), written $H(s\mid X)=0,$ and *all* observable conditional independences of the candidate $G_1,$ read off $G_1$ by d-separation.
* **Step.** What the step deletes, which facet becomes classical for $s$ and why the d-separation test fails.

The proof uses all the facts: every one holds for the constructed distribution, so there is nothing to gain by omitting any, and a minimal subset plays no role in the certificate. The independence marked "decisive" is a diagnostic, found afterwards by re-running the certificate with the prediction and one independence at a time, and is there only to show which fact carries the inference; the others are then consequences, since the target triplets imply every observable independence of the output.

1. **Input.** 1→2→3; Q{0,2}, Q{0,3}, Q{1,3}.

   **Output.** 1→2; C{0,3}, Q{0,2}, which is a relabelling of the seed [`QG_Bell_C_Edge`](../known_QC_gaps.py#L80) (party 2 with setting 1, party 0 whose "setting" 3 is a classical copy correlated with it).

   **Facts.** $H(3\mid0)=0;$ $G$ itself has only $0\perp1;$ the candidate $G_1$ adds $3\perp1,$ $3\perp1\mid2,$ $3\perp1\mid0,$ $3\perp2,$ $3\perp2\mid1$ and $0\perp1\mid3,$ of which $3\perp2\mid1$ alone, with the prediction, implies the `relabel` target set; the prediction alone does not.

   **Step.** $s=3$ loses the edge 2→3 and the facet Q{1,3}, which shrinks to nothing on 1; Q{0,3} stays quantum on 0 and becomes classical for 3, C{0,3}; d-separation fails because 0←Q{0,2}→2→3 is open.

2. **Input.** 1→3, 2→3; Q{0,2}, Q{0,3}, Q{1,2}.

   **Output.** C{0,3}, Q{0,2}, Q{1,2}, which degrades (Q{1,2} made classical) to the seed [`QG_Bell_C_C`](../known_QC_gaps.py#L78).

   **Facts.** $H(3\mid0)=0;$ $G$ has only $0\perp1;$ $G_1$ adds the same six independences as in example 1; $3\perp2\mid1$ is decisive.

   **Step.** $s=3$ loses both visible parents 1→3 and 2→3 at once ([6.2](#62-the-unit-a-predictor-and-its-targets)); Q{0,3} becomes classical for 3; no facet shrinks; d-separation fails through 0←Q{0,2}→2→3.

3. **Input.** 1→2, 1→3, 2→3; Q{0,2}, Q{0,3}, the KPC structure of [7.5](#75-worked-example-the-kpc-structure).

   **Output.** 1→2; C{0,3}, Q{0,2}, which is [`QG_Bell_C_Edge`](../known_QC_gaps.py#L80).

   **Facts.** $H(3\mid0)=0;$ $G$ has only $0\perp1;$ $G_1$ adds the same six independences as in example 1; $3\perp2\mid1$ is decisive.

   **Step.** $s=3$ loses 1→3 and 2→3; Q{0,3} becomes classical for 3; no facet shrinks; d-separation fails through 0←Q{0,2}→2→3.

4. **Input.** 1→2, 1→3, 2→3; Q{0,2}, Q{0,3}, Q{1,2}.

   **Output.** 1→2; C{0,3}, Q{0,2}, Q{1,2}, which degrades (Q{1,2} made classical) to a relabelling of the seed [`QG_Bell_C_EdgeC`](../known_QC_gaps.py#L79).

   **Facts.** $H(3\mid0)=0;$ $G$ has only $0\perp1;$ $G_1$ adds the same six independences as in example 1; $3\perp2\mid1$ is decisive.

   **Step.** $s=3$ loses 1→3 and 2→3; Q{0,3} becomes classical for 3; no facet shrinks; d-separation fails through 0←Q{0,2}→2→3.

5. **Input.** 1→2, 1→3, 2→3; Q{0,2}, Q{0,3}, Q{1,3}.

   **Output.** 1→2; C{0,3}, Q{0,2}, which is [`QG_Bell_C_Edge`](../known_QC_gaps.py#L80).

   **Facts.** $H(3\mid0)=0;$ $G$ has only $0\perp1;$ $G_1$ adds the same six independences as in example 1; $3\perp2\mid1$ is decisive.

   **Step.** $s=3$ loses 1→3, 2→3 and the facet Q{1,3}, which shrinks to nothing on 1; Q{0,3} becomes classical for 3; d-separation fails through 0←Q{0,2}→2→3.

6. **Input.** 0→1→2; Q{0,2}, Q{0,3}, Q{1,3}. This is the split-predictor example of [6.10](#610-example-a-split-predictor-and-a-split-target-worked-out-in-full), whose d-separation proof needs a split target and a childful predictor; the closure proves it with the childless predictor 3 instead.

   **Output.** 1→2; C{1,3}, Q{0,2}, Q{0,3}; marginalizing 1 (no quantum facet left on it) gives C{2,3}, Q{0,2}, Q{0,3}, which degrades to the seed [`QG_Triangle`](../known_QC_gaps.py#L42).

   **Facts.** $H(1\mid3)=0;$ $G$ has no observable independence at all; $G_1$ adds $1\perp0$ and $2\perp3\mid1,$ of which $1\perp0$ alone, with the prediction, implies the `relabel` target set; $2\perp3\mid1$ plays no role.

   **Step.** $s=1$ loses the edge 0→1; Q{1,3} becomes classical for 1; no facet shrinks; d-separation fails because 3←Q{0,3}→0→1 is open.

Leaving the predictor unsplit instead would delete it and with it the only facet $s$ keeps, proving nothing, so these inputs are exactly where split predictors matter beyond d-separation ([1.1](#11-the-census-over-quantum-classes)).

Three variants of the certificate were tried, certified many candidate steps and were never decisive in four nodes: the `markov` target set, the joint closure certificate for a set of predicted nodes ([7.7](#77-theorem-semigraphoid-closure-certificate)) and the extra deletions ([7.7](#77-theorem-semigraphoid-closure-certificate)). The code keeps only `markov`, as an option that is off ([9.3](#93-certificates-beyond-d-separation-attempted)); the other two are recorded as open questions (Section [10](#10-open-questions)).

### 7.9 Strength, agreement with the LP, and cost

**What the closure cannot do.** Soundness is Theorem [7.7](#77-theorem-semigraphoid-closure-certificate); completeness fails in two ways. First, the semigraphoid rules are not complete for implication among conditional independences of probability distributions: already for four variables there are implications valid for every distribution that the four rules do not derive (Studený), and the Shannon inequalities derive some of them. Second, the closure uses only the sufficiency half of the perfect prediction ([7.3](#73-the-hypotheses-and-how-perfect-prediction-enters-qmdag_semigraphoid_model-semigraphoidadd_functional_dependence)). The entropic LP of [Appendix B](#appendix-b-the-entropic-lp-as-an-alternative-certificate-engine-entropic_lppy-qmdag_entropic_certificate), which reasons with the Shannon inequalities and uses $H(s\mid\mathbf X)=0$ in full, is therefore stronger in principle: every instance of the exchange rule is a Shannon-type identity and every sufficiency triplet is Shannon-implied by $H(s\mid\mathbf X)=0,$ so whatever the closure derives the LP derives too, and an input the LP proves and the closure does not is possible a priori.

**The smallest example of what the closure misses.** For four variables $A,B,C,D$ the two sums below are equal as expressions in the entropies:

```math
I(A:B\mid CD)+I(C:D\mid A)+I(C:D\mid B)+I(A:B) \;=\; I(C:D\mid AB)+I(A:B\mid C)+I(A:B\mid D)+I(C:D).
```

Every term is nonnegative, so if the four on the left vanish, the four on the right vanish too. As an implication among conditional independences:

| hypotheses | conclusions |
|---|---|
| $\langle A,B\mid CD\rangle$ | $\langle C,D\mid AB\rangle$ |
| $\langle C,D\mid A\rangle$ | $\langle A,B\mid C\rangle$ |
| $\langle C,D\mid B\rangle$ | $\langle A,B\mid D\rangle$ |
| $\langle A,B\mid\emptyset\rangle$ | $\langle C,D\mid\emptyset\rangle$ |

The LP derives all four conclusions. The closure derives none: the four hypotheses are already closed under the semigraphoid rules, so the closure adds nothing to them. The reason is the size of the identity. The exchange rule of [7.2](#72-conditional-independence-as-the-proof-system-semigraphoids-and-the-exchange-rule-semigraphoidpy) trades two triplets for two, and each instance of it is an entropy identity with two terms on a side. No chain of such trades reaches an identity with four terms on a side.

This shows what a piggyback would have to look like for the LP to justify it and the closure to miss it: a deletion whose Markov statements follow from the hypotheses only through an identity with more than two terms on a side. It is a possibility in principle. No candidate step of the four-node census is of that kind, as the next paragraph records. ([`four_variable_example`](../Special%20Applications/kpc_figure4_checks.py#L176) in [`Special Applications/kpc_figure4_checks.py`](../Special%20Applications/kpc_figure4_checks.py) runs both engines on the example, and [`tests/test_kpc_figure4.py`](../tests/test_kpc_figure4.py) pins the outcome.)

**In four nodes there is none.** Run with both engines on every candidate (`--engine both`, which records every disagreement in either direction, [`ENGINE_DISAGREEMENTS`](../quantum_mDAG.py#L64)), the 696 candidate steps that d-separation rejects give 37 `relabel` certificates from each engine, the same 37, and 659 failures from both ([1.1](#11-the-census-over-quantum-classes), [9.3](#93-certificates-beyond-d-separation-attempted); [`tests/test_baseline_slow.py`](../tests/test_baseline_slow.py) pins the agreement). So on the census the dropped half of the perfect prediction and the non-semigraphoid Shannon inferences never matter, and the closure is the engine of record; a five-node candidate where they do matter would be worth having (Section [10](#10-open-questions)).

**Cost.** The census's 696 closures and the d-separation models they need take under a second in all, against about a minute and a half of LP solves for the same candidates with the LP as the engine, so the closure stages cost no more than the surgery and bookkeeping of the search ([9.1](#91-the-ten-stages)). The order of the search is unchanged by the engine: one check per (predictor, target) pair whatever the loop order, and the per-structure model is cached, so the closure stages stay predictor-first ([8.7](#87-two-kinds-of-stage-d-separation-breadth-first-then-the-semigraphoid-closure-with-early-exit)). The closure certifies by containment and keeps no derivation; a trace of the rule instances that set each target cell, the analogue of the LP's Farkas certificate ([B.1](#b1-entropy-vectors-as-the-proof-system-entropic_lppy)), is a possible addition (Section [10](#10-open-questions)).

---

## 8. The Fritz search procedure (`QmDAG.fritz_steps`, `qc_gap_search.fritz_tricks`)

This section is self-contained. Throughout, $\mathbf X$ is the predictor set (a single node $X$ in the census), $s$ a predicted node, called a **target** in the code, and parents are parents in the effective DAG: the visible parents of a node, the facets containing it, and its private noise. The procedure is written in a few short methods of [`QmDAG`](../quantum_mDAG.py#L101) that are read top to bottom ([`fritz_pool`](../quantum_mDAG.py#L739), [`fritz_targets`](../quantum_mDAG.py#L842), [`fritz_deletion`](../quantum_mDAG.py#L769), [`fritz_certificate`](../quantum_mDAG.py#L796), [`fritz_realise`](../quantum_mDAG.py#L902), [`fritz_steps`](../quantum_mDAG.py#L920)), so that any Fritz step in a certificate can be re-derived without leaving the class.

**The unit of the search** is a predictor set $\mathbf X$ together with the targets it predicts ([6.2](#62-the-unit-a-predictor-and-its-targets) gives the definition). The search does not choose edges: it picks a predictor set, runs through its candidate targets in a fixed order, and at each target *reads off* the predictor which parents to delete, so that the only question left is whether that deletion is justified. Nothing is enumerated over subsets of parents: for one $(\mathbf X, s)$ there is exactly one candidate deletion, and if it is not justified the search moves to the next target, not to a smaller deletion.

The subsections follow the procedure as it runs:

1. [8.1](#81-the-pool-and-its-order-qmdagfritz_pool-qmdagfritz_targets): which nodes may predict a target (the pool), and the order in which predictors and their targets are tried.
2. [8.2](#82-what-a-predictor-sees-and-the-deletion-it-dictates-qmdagfritz_deletion): the deletion that a predictor dictates at a target.
3. [8.3](#83-justifying-the-deletion-one-set-two-certificates-qmdagfritz_certificate): whether that deletion is justified, by d-separation or by the semigraphoid closure.
4. [8.4](#84-parent-predictors-the-free-piggyback-as-its-own-stage): predictors that are visible parents of their target, which need no certificate and have a stage of their own.
5. [8.5](#85-joint-steps-and-why-not-a-second-fritz-depth): joint steps. Once the targets of $\mathbf X$ are settled singly, every set of two or more of those that d-separation certified is emitted as one joint step, which is how Fritz's triangle reaches Bell.
6. [8.6](#86-modes-and-the-surgery-qmdag_restrict_targets-qmdagfritz_realise): the splitting modes, and the surgery that carries a step out.
7. [8.7](#87-two-kinds-of-stage-d-separation-breadth-first-then-the-semigraphoid-closure-with-early-exit): the two kinds of stage in which the search runs the procedure.
8. [8.8](#88-what-the-procedure-decides-and-what-it-does-not): which of these choices decide the certificate an input gets, and what none of them changes.

Two subsections stand beside the procedure: [8.9](#89-the-six-classes-that-need-a-fritz-step) works through the six classes of the census whose proof needs a Fritz step, and [8.10](#810-experiment-descendants-as-predictors) reports an experiment that admits descendants of the target as predictors.

### 8.1 The pool and its order (`QmDAG.fritz_pool`, `QmDAG.fritz_targets`)

**Who may predict whom.** The construction in the quantum direction must hand $s$ its private randomness through a **channel** that the predictor also sees ([6.1](#61-the-mechanism); a node sees itself and its own parents, [8.2](#82-what-a-predictor-sees-and-the-deletion-it-dictates-qmdagfritz_deletion)). There are two kinds of channel:

* a facet shared with $s$ (the predictor is a **latent sibling** of the target);
* the edge $X\to s$ (the predictor is a **visible parent** of the target).

A node with neither channel cannot predict $s$ for an arbitrary $P',$ since it would have to reproduce randomness it has no access to, so the pool of sound predictors is exactly *latent siblings $`\cup`$ visible parents*, and nothing wider. This is narrower than the Markov blanket of $s,$ which also holds the children of $s$ and the other parents of those children and of the siblings: a co-parent of a child of $s$ sees nothing of $s$'s inputs and is unsound as a predictor, not merely unhelpful.

**Two pools.** The two kinds are two piggybacks: a visible parent needs no certificate ([6.5](#65-the-parent-predictor-deletion-without-a-certificate)), a latent sibling does ([6.6](#66-theorem-d-separation-certificate), Section [7](#7-the-semigraphoid-closure-certificate-for-the-fritz-piggyback-semigraphoidpy-qmdag_semigraphoid_certificate)). The census therefore uses two pools:

* `pool='parents'`: the visible parents of the target. A predictor from this pool is called a **parent predictor**, whether or not it also shares a facet with the target. This is the pool of the **parent stage** ([8.4](#84-parent-predictors-the-free-piggyback-as-its-own-stage)).
* `pool='siblings'`: the latent siblings that are *not* visible parents. A predictor from this pool is called a **sibling predictor**. This is the pool of the **certified stages**: the d-separation stages and the closure stages of [8.7](#87-two-kinds-of-stage-d-separation-breadth-first-then-the-semigraphoid-closure-with-early-exit), whose steps need a certificate.

A node that is both a parent and a sibling is thus handled once, by the parent stage, with the maximal deletion (`pool='siblings+parents'` lists both). [`fritz_pool(s)`](../quantum_mDAG.py#L739) lists the predictors of a target with this bound; [`fritz_targets(X)`](../quantum_mDAG.py#L842) inverts it, listing the nodes whose pool contains the predictor, and is what the predictor-first loop iterates over ([8.7](#87-two-kinds-of-stage-d-separation-breadth-first-then-the-semigraphoid-closure-with-early-exit)).

**Order.**

* Predictors are tried by index.
* The targets of a predictor: nodes sharing more facets with it first, then by index.
* Joint predictor sets, which see the union of what their members see, are enumerated up to `max_predictors` (1 in the census); a set is tried at a target only when every member lies in the target's pool, each with its own channel.

**Descendants are left out.** A descendant $x$ of $s$ fails the d-separation test of [8.3](#83-justifying-the-deletion-one-set-two-certificates-qmdagfritz_certificate) whatever the target keeps: the noise of $s$ is always among the parents to delete, and the path noise → $s$ → … → $x$ is open given any subset of $\mathrm{Pa}(s).$ The search therefore removes descendants from the pool by default (`allow_descendants=False`). They are nevertheless *sound* predictors when they have a channel; [8.10](#810-experiment-descendants-as-predictors) gives the argument and the census experiment that admits them.

### 8.2 What a predictor sees, and the deletion it dictates (`QmDAG.fritz_deletion`)

This subsection and the next concern sibling predictors, the pool of the certified stages ([8.1](#81-the-pool-and-its-order-qmdagfritz_pool-qmdagfritz_targets)). For a parent predictor the kept set is $\mathbf X$ itself and the deletion is everything else ([6.5](#65-the-parent-predictor-deletion-without-a-certificate), [8.4](#84-parent-predictors-the-free-piggyback-as-its-own-stage)).

A predictor $x$ **sees** itself and its parents: its visible parents, the facets it belongs to, and its own noise. A set sees the union, $\mathrm{seen}(\mathbf X)=\mathbf X\cup\mathrm{Pa}(\mathbf X).$ The parents of $s$ split accordingly:

```math
K=\mathrm{common}(s)=\mathrm{Pa}(s)\cap\mathrm{seen}(\mathbf X),\qquad D=\mathrm{others}(s)=\mathrm{Pa}(s)\setminus K .
```

$K$ is what $s$ keeps, $D$ is what is deleted, and $D$ always contains the noise of $s,$ which no other node sees. Both sets are fixed by $(\mathbf X, s)$ alone, before any test is run; they are the `common` and `others` defined at the start of Section [6](#6-the-fritz-piggyback-fritz).

Two degenerate outcomes end the pair at once:

* **No channel.** $K$ holds no channel from a predictor to $s$ (no facet containing both, no edge $X\to s$). Then the construction in the quantum direction cannot give $s$ its randomness through $\mathbf X,$ and the pair is unsound as a predictor-target pair.
* **Nothing to delete.** $D$ is the noise of $s$ and nothing else. Then the step would restrict $s$ to all of its parents, a no-op whose only effect, removing the predictors, is what marginalization and point distribution already do.

Everything else is a **candidate**: $s$ restricted to $K,$ with $D$ gone.

Three readings of the same rule follow, in the examples used below (Q{…} names the facet node of the effective DAG). The KPC structure of [7.5](#75-worked-example-the-kpc-structure) is written in the code's labels: its nodes $F,$ $C,$ $D,$ $E$ are the nodes 0, 1, 2, 3 (the letter $D$ of [7.5](#75-worked-example-the-kpc-structure) is node 2, not the deleted set).

* **Tetrahedron**, no edges; Q{0,1,2}, Q{0,1,3}, Q{0,2,3}, Q{1,2,3}; target 1, predictor 0. $\mathrm{seen}(0)$ is 0 and its facets Q{0,1,2}, Q{0,1,3}, Q{0,2,3}. $\mathrm{Pa}(1)$ is Q{0,1,2}, Q{0,1,3}, Q{1,2,3} and the noise of 1. So $K$ = {Q{0,1,2}, Q{0,1,3}} and $D$ = {Q{1,2,3}, noise}.
* **KPC** ([7.5](#75-worked-example-the-kpc-structure)), 1→2, 1→3, 2→3; Q{0,2}, Q{0,3}; target 3, predictor 0. $\mathrm{seen}(0)$ is 0, Q{0,2}, Q{0,3}. $\mathrm{Pa}(3)$ is the visible parents 1 and 2, the facet Q{0,3} and the noise. So $K$ = {Q{0,3}} and $D$ = {1→3, 2→3, noise}: two visible edges go, because 0 sees neither 1 nor 2.
* **A pair that fails**, 1→3; Q{0,1,3}, Q{1,2}, Q{2,3} (a census input still unproven, Section [10](#10-open-questions)); target 2, predictor 3. $\mathrm{seen}(3)$ is 3, its visible parent 1, and Q{0,1,3}, Q{2,3}. $\mathrm{Pa}(2)$ is Q{1,2}, Q{2,3} and the noise. So $K$ = {Q{2,3}} and $D$ = {Q{1,2}, noise}.

In all three the deletion is settled. Whether it is *justified* is the next question ([8.3](#83-justifying-the-deletion-one-set-two-certificates-qmdagfritz_certificate)), and it is a different question.

**Split targets.** A split target ([6.4](#64-node-splitting-split-and-unsplit-predictors-split-and-unsplit-targets-qmdagsplit_node)) changes nothing in the rule: the target is then the copy $s',$ which has the parents of $s,$ and the predictor sees the facets with the copy added, so for the triangle with 0 and 1 split, 2 sees Q{0,0′,2} and Q{1,1′,2} and the copy 0′ keeps Q{0,0′,2} ([6.3](#63-example-the-triangle-fritzs-original-argument)).

### 8.3 Justifying the deletion: one set, two certificates (`QmDAG.fritz_certificate`)

The candidate is sound iff every classical model of $G$ in which $\mathbf X$ predicts $s$ perfectly ($H(s\mid\mathbf X)=0$) and which satisfies the observable independences of the candidate is a classical model of the candidate ([6.6](#66-theorem-d-separation-certificate) and [7.7](#77-theorem-semigraphoid-closure-certificate)). A parent predictor needs no argument ([6.5](#65-the-parent-predictor-deletion-without-a-certificate), [8.4](#84-parent-predictors-the-free-piggyback-as-its-own-stage)). For a latent sibling two arguments can establish it, and both are asked about the *same* $D$:

1. **d-separation**: $\mathbf X\setminus K\ \perp_d\ D\ \big|\ K$ in the effective DAG, noise included. When it holds the classical direction is the elementary argument of [6.6](#66-theorem-d-separation-certificate): conditioned on $K,$ $s$ is a function of $\mathbf X$ and a function of $D,$ which are independent, hence constant. Certificate `dsep`.
2. **the semigraphoid closure** of Section [7](#7-the-semigraphoid-closure-certificate-for-the-fritz-piggyback-semigraphoidpy-qmdag_semigraphoid_certificate), run only when d-separation fails, with the same kept set $K$ as its candidate (certificate `semigraphoid`); the entropic LP of [Appendix B](#appendix-b-the-entropic-lp-as-an-alternative-certificate-engine-entropic_lppy-qmdag_entropic_certificate) is the alternative engine (certificate `entropic`), and on the census the two certify the same steps.

The point that is easy to misread: a failed d-separation test does **not** mean that the set to delete is empty. The set is $D,$ fixed in [8.2](#82-what-a-predictor-sees-and-the-deletion-it-dictates-qmdagfritz_deletion); the failure means that the elementary argument cannot justify deleting it, and the closure is asked to justify deleting exactly it. A pair that neither argument justifies gives no step, never a step with a smaller deletion.

The three examples of [8.2](#82-what-a-predictor-sees-and-the-deletion-it-dictates-qmdagfritz_deletion), continued:

* **Tetrahedron**: is 0 separated from {Q{1,2,3}, noise of 1} given {Q{0,1,2}, Q{0,1,3}}? Every path from 0 to Q{1,2,3} passes through 2 or 3 as a collider (0 ← Q{0,2,3} → 2 ← Q{1,2,3}), and the noise of 1 reaches 0 only through 1, a collider on that path too. Yes: `dsep`. The output is [`QG_Triangle`](../known_QC_gaps.py#L42).
* **KPC**: is 0 separated from {1, 2, noise of 3} given Q{0,3}? No: 0 ← Q{0,2} → 2 is open. d-separation cannot justify deleting 1→3 and 2→3. The closure, given the sufficiency of 0 for 3 (the form $H(3\mid 0)=0$ takes in it, [7.3](#73-the-hypotheses-and-how-perfect-prediction-enters-qmdag_semigraphoid_model-semigraphoidadd_functional_dependence)), the Markov conditions of $G$ and the observable independences of the candidate (3 ⊥ {1,2}), contains the `relabel` target model ([7.6](#76-why-relabel-is-justified)), and the LP agrees. The deletion is the same {1→3, 2→3}; only the argument changed. The output with 0 kept is [`QG_Bell_C_Edge`](../known_QC_gaps.py#L80).
* **The failing pair**: is 3 separated from {Q{1,2}, noise of 2} given Q{2,3}? No: 3 ← 1 ← Q{1,2} is open, 3 reads 1 and 1 reads the facet. The closure fails as well (and so does the LP), and here the failure is a true negative: the model $L_{12}$ uniform, $1=L_{12},$ $2=L_{12},$ $3=1$ is a classical model of $G$ in which 3 predicts 2 perfectly, yet 2 depends on Q{1,2} and not on Q{2,3}, so it is not a model of the candidate, and no certificate can exist. The search moves on to the next predictor of 2.

### 8.4 Parent predictors: the free piggyback as its own stage

**Why no certificate, and why the deletion is larger.** Let $X$ be a visible parent of $s$ (with or without a shared facet). The soundness statement of [8.3](#83-justifying-the-deletion-one-set-two-certificates-qmdagfritz_certificate) asks that in every classical model with $s=g(X),$ $s$ be a function of the parents it keeps. If $s$ keeps $X,$ that is already true, with nothing to show: this is the parent piggyback of [6.5](#65-the-parent-predictor-deletion-without-a-certificate), and [`fritz_certificate`](../quantum_mDAG.py#L796) returns `parent` without a test. For the same reason the deletion is larger than for a sibling: [`fritz_deletion`](../quantum_mDAG.py#L769) lets $s$ keep the edge from $X$ alone, a facet shared by $X$ and $s$ included in the deletion, where a sibling predictor keeps the shared facets ([8.2](#82-what-a-predictor-sees-and-the-deletion-it-dictates-qmdagfritz_deletion)).

**The parent stage.** The parent piggyback is run as the first Fritz stage of the cascade ([9.1](#91-the-ten-stages)), over the pool of visible parents, with both target modes and both predictor modes ([6.4](#64-node-splitting-split-and-unsplit-predictors-split-and-unsplit-targets-qmdagsplit_node)). Everything that each root of the search (a structure the stage is applied to) produces is recorded ([9.1](#91-the-ten-stages)). The certified stages never see a parent predictor, since their pool is the latent siblings that are not visible parents ([8.1](#81-the-pool-and-its-order-qmdagfritz_pool-qmdagfritz_targets)). (The `relabel` target set would not even apply to a parent predictor, since the kept set is not a single facet.)

**What the step does for the target** is a genuine deletion followed by a partial refund. The parents of $s$ other than $X$ are gone for good, and no elementary piggyback removes them. Then $X$ must leave, in one of two ways:

* **Unsplit**, it is marginalized (it has the child $s$), which relays its own parents to $s$ and to its other children as edges and a classical facet.
* **Split**, the copy $X'$ predicts, $s$ keeps the edge from $X'$ only (the edge $X\to s$ is deleted too), and $X'$ is marginalized and relays $\mathrm{Pa}(X)$ to $s$ with a classical facet over $X,$ $s$ and $X$'s other children.

The third route of [6.8](#68-example-deleting-a-visible-edge-three-routes) shows the net effect: 1 loses 0→1 and Q{1,2} and gains C{0,1,3}. After the step $s$ has no quantum parent at all.

**In the census.** In the four-node census the stage produces 265 structures and proves three inputs, the family of [6.9](#69-example-a-split-target-with-a-childful-unsplit-predictor), in each of which the parent predictor also shares a facet with the target ([1.1](#11-the-census-over-quantum-classes)); it takes 8 seconds ([9.1](#91-the-ten-stages)).

### 8.5 Joint steps, and why not a second Fritz depth

**What is emitted.** For one predictor set and one pair of modes (a target mode and a predictor mode, [6.4](#64-node-splitting-split-and-unsplit-predictors-split-and-unsplit-targets-qmdagsplit_node)), the targets that d-separation certifies singly are known once the single steps are done. Every subset of two or more of them is then emitted as a **joint step**:

1. the structure carrying all the splits is built (with split targets every member is split);
2. each member's deletion is read off again on that structure and its d-separation test is re-run there;
3. every member is restricted at once;
4. the predictors leave as usual.

**Soundness.** Theorem [6.6](#66-theorem-d-separation-certificate) is stated for a set of targets, and its classical direction replaces the kernels one at a time, so a joint step is as sound as its members.

**Only d-separation members are admitted**: the `relabel` certificate substitutes the target for one of its latents, and two such substitutions for two targets do not obviously compose into a model of the joint candidate (a joint closure over the joint candidate with the `markov` target set would; it is sound but not implemented, and when it was tried it decided no input, Section [10](#10-open-questions)).

**Why joint steps rather than a deeper search.** With the predictor *unsplit*, the joint step of the triangle ([6.3](#63-example-the-triangle-fritzs-original-argument)) has no sequential counterpart at all, since after one copy is predicted the predictor is gone; the sequential route needs two *split*-predictor steps and a point distribution. The cascade of [9.1](#91-the-ten-stages) applies the Fritz trick once per input ("depth one"), so without joint steps the search would miss the paradigmatic derivation of Bell from the triangle, and would in general miss every output that needs a predictor to serve two targets.

A second Fritz depth would recover them at the cost of re-expanding every Fritz output under the trick, which was measured to cost far more than everything else combined; the joint steps recover them for one extra surgery per subset of already-certified targets and no closure at all. In four nodes a predictor has at most three certified targets, so at most four joint subsets are added per predictor and modes.

**Where joint steps run, and what they cost.** Joint steps belong to the d-separation stages of the cascade only ([8.7](#87-two-kinds-of-stage-d-separation-breadth-first-then-the-semigraphoid-closure-with-early-exit)); `max_targets` bounds the subsets (all by default; 1 switches joint steps off), and the census shows them to be cheap: the four d-separation stages take about a second each or less, joint steps included, and they prove no four-node class that single steps did not.

### 8.6 Modes and the surgery (`QmDAG._restrict_targets`, `QmDAG.fritz_realise`)

**Modes first.** The target mode and the predictor mode of [6.4](#64-node-splitting-split-and-unsplit-predictors-split-and-unsplit-targets-qmdagsplit_node) are applied before the deletion is read off, so that [8.2](#82-what-a-predictor-sees-and-the-deletion-it-dictates-qmdagfritz_deletion) and [8.3](#83-justifying-the-deletion-one-set-two-certificates-qmdagfritz_certificate) run on the structure that carries the splits:

* with split targets each target is split first and the copies are the targets;
* with split predictors each childful predictor (a predictor that has children) is split and the copy predicts.

**What the modes cost.** Both target modes and both predictor modes are available to both certificates. Split predictors reach strictly more (the KPC construction of [7.5](#75-worked-example-the-kpc-structure) keeps its childless predictor), but an output with a split predictor has as many visible nodes as its source, so an exhaustive closure under split predictors is far larger than under unsplit ones; the search applies the split-predictor stages at depth one, once to each input the cheaper stages leave unproven (Section [9](#9-search-certificates-and-the-four-node-census-qc_gap_searchpy-special-applicationsproving_qc_gapspy)). The cascade of [1.1](#11-the-census-over-quantum-classes) separates the contributions of the predictor mode, the target mode and the certificate, and says where split predictors prove something unsplit predictors cannot.

**The surgery** itself has no certificate logic. It has two parts:

1. **The targets are restricted.** Each target keeps exactly its $K$: a kept facet stays quantum for its other members and becomes classical for $s;$ every other node is untouched.
2. **The predictors leave.** Unsplit predictors are deleted when childless and marginalized with teleportation otherwise; split predictors leave their copies to the same fate, a childless split predictor staying untouched. When several nodes are marginalized every order is realised and recorded, since teleportation is order-dependent (Section [3](#3-marginalization-naive-and-with-teleportation-naive_marginalization-teleportation_marginalization)).

**Labels and the record.** Outputs carry plain integer labels; a copy made by splitting is the next free index and is printed with a prime in the recorded parameters. Each recorded step states:

* the targets and the target mode;
* the deleted parents of each target (visible parents by index, facets as `Q{…}` or `C{…}`, the noise omitted);
* the predictor set and the predictor mode;
* the certificate: `parent`, `dsep` when d-separation suffices, `semigraphoid` when the closure of Section [7](#7-the-semigraphoid-closure-certificate-for-the-fritz-piggyback-semigraphoidpy-qmdag_semigraphoid_certificate) was needed (`entropic` when the LP of [Appendix B](#appendix-b-the-entropic-lp-as-an-alternative-certificate-engine-entropic_lppy-qmdag_entropic_certificate) is the engine);
* when several nodes were marginalized, the order.

### 8.7 Two kinds of stage: d-separation breadth-first, then the semigraphoid closure with early exit

The cascade of [9.1](#91-the-ten-stages) runs the Fritz trick in the parent stage ([8.4](#84-parent-predictors-the-free-piggyback-as-its-own-stage)) and then in two groups of four stages, one stage per pair of modes in each group. A structure that a stage is applied to is a **root of the search**. The parent stage behaves like a d-separation stage: every root of the search is recorded in full.

**The d-separation stages.** [`fritz_steps`](../quantum_mDAG.py#L920) runs with the closure off (`use_lp=False`; the flag is named after the LP of [Appendix B](#appendix-b-the-entropic-lp-as-an-alternative-certificate-engine-entropic_lppy-qmdag_entropic_certificate), the alternative engine). For every predictor set it

1. visits every target, reads off the deletion and runs the d-separation test;
2. emits the single steps and then the joint steps of that predictor set.

The explorer records everything the root of the search produces, breadth-first, with no early exit (it never stops before the root of the search is recorded in full). The test is cheap, so there is nothing to save (the four d-separation stages take about a second each or less in the census, [9.1](#91-the-ten-stages)), and recording every d-separation route keeps what is recorded, and with it the database of proven structures ([9.5](#95-the-known-gap-database-the-search-establishes)), independent of the order in which the predictors of a root of the search happen to be tried. (The joint steps themselves would survive early exit either way: [`fritz_steps`](../quantum_mDAG.py#L920) collects a predictor's certified targets before emitting any of its joint steps.)

**The closure stages.** [`fritz_steps`](../quantum_mDAG.py#L920) runs with `lp_only` (a flag likewise named after the LP). The loop is the same, with four differences:

* a pair that d-separation certifies is skipped (its step is already recorded);
* a pair that d-separation rejects goes to the semigraphoid closure (Section [7](#7-the-semigraphoid-closure-certificate-for-the-fritz-piggyback-semigraphoidpy-qmdag_semigraphoid_certificate); the LP of [Appendix B](#appendix-b-the-entropic-lp-as-an-alternative-certificate-engine-entropic_lppy-qmdag_entropic_certificate) with `--engine lp`);
* no joint sets are formed;
* the explorer stops working on a root of the search at the first child structure whose elementary reduction is a known gap (the **early exit**).

The early exit is there for the LP, where every candidate costs a solve; with the closure a candidate costs about a millisecond and the exit is harmless.

**Loop order.** The loop order is predictor-first in both groups. In the d-separation stages it has to be, for the joint steps. In the closure stages it is kept for consistency, and nothing is lost. Either engine costs one check per (predictor, target) pair whatever the order. Its per-structure part, for the structure that carries the splits, is built once per structure and cached: the d-separation model for the closure, the Shannon cone with the Markov rows for the LP. Grouping the pairs by target, which would group them by split structure when targets are split, would therefore only matter if that cache were smaller than the number of split structures of one input (at most a handful in four nodes, cache size 64).

### 8.8 What the procedure decides, and what it does not

Three choices of the procedure influence *which* certificate an input gets, and none influences *whether* it gets one.

* **Stage order** ([9.1](#91-the-ten-stages)): the parent stage first, every d-separation stage before every closure stage, and within each group unsplit targets before split targets, unsplit predictors before split ones; each stage is applied only to the inputs the earlier stages left unproven. An input proven at stage $k$ never sees stage $k+1,$ so a closure proof is never recorded for an input that any d-separation route settles. The split-predictor example of [6.10](#610-example-a-split-predictor-and-a-split-target-worked-out-in-full) is the case in point: it has a d-separation route (split target, split predictor) and a closure route (unsplit target, split predictor); the census records the former.
* **Early exit** ([8.7](#87-two-kinds-of-stage-d-separation-breadth-first-then-the-semigraphoid-closure-with-early-exit)): within a closure stage, the first child structure whose elementary reduction reaches a known gap ends the input, so later alternatives of the same stage are not recorded; the d-separation stages record everything.
* **Shortest chain, earliest stage** ([`build_report`](../qc_gap_search.py#L699)): among the recorded routes, the certificate is the shortest chain of transitions using the transitions of the earliest stage that has one, ending at a seed (a class whose gap is taken as known, [0.7](#07-the-census-the-seeds-and-the-names-of-the-tricks)) or at an input already certified.

**What is invariant under all three:**

* the set of proven inputs;
* the cumulative count after each stage;
* the ladder of [1.1](#11-the-census-over-quantum-classes).

The ladder is computed from the recorded transitions by the rung each one belongs to (its certificate, target mode and predictor mode, [9.1](#91-the-ten-stages)), and an input proven at some rung has recorded the step that proves it there in every variant of the procedure (an unproven input records everything; a proven input is proven at the rung of the step that proved it).

**What varies:**

* the certificate text;
* the proven-structure database of [9.5](#95-the-known-gap-database-the-search-establishes), since alternative routes visit other intermediates.

The variants were measured against each other on the census, and the counts agree ([1.1](#11-the-census-over-quantum-classes)). The early exit in particular changes neither the proven classes nor the database: without it ten more candidates are checked, all in the one class that the closure proves, and the structures they produce are in the database already. The six classes that the census proves through a Fritz step are worked through in [8.9](#89-the-six-classes-that-need-a-fritz-step).

### 8.9 The six classes that need a Fritz step

Six classes of the census are proven through a Fritz step. Each proof is one Fritz step followed by at most two elementary steps, as recorded by the census (the member a step was applied to is the one written; a follow-up step is applied to the member of its class that the search met). The table gives the overview, the list below it gives each derivation in full, and a closing list reads the six against the procedure.

**Notation.** Throughout, $s$ is the predicted node, $X$ the predictor, and a prime marks the copy made by node splitting ([6.4](#64-node-splitting-split-and-unsplit-predictors-split-and-unsplit-targets-qmdagsplit_node)). "Kept" is $K,$ the parents of $s$ (or of its copy) that $X$ sees; "deleted" is $D$ ([8.2](#82-what-a-predictor-sees-and-the-deletion-it-dictates-qmdagfritz_deletion)); a kept quantum facet becomes classical for $s$ and stays quantum for its other members.

| no. | class (representative) | version of the piggyback | predicted node $s$ and predictor $X$ | stage | seed reached |
|---|---|---|---|---|---|
| 1 | 0→1→2→3; Q{0,1}, Q{0,2}, Q{1,3} | parent predictor | $s=1$ split into 1 and 1′; $X=0$ unsplit | parent stage | [`QG_Instrumental_EdgeC`](../known_QC_gaps.py#L40) |
| 2 | 0→1, 0→2, 1→2, 2→3; Q{0,1}, Q{0,2}, Q{1,3} | parent predictor | $s=1$ split; $X=0$ unsplit | parent stage | [`QG_Instrumental_EdgeC`](../known_QC_gaps.py#L40) |
| 3 | 0→2, 1→2, 2→3; Q{0,1}, Q{0,2}, Q{1,3} | parent predictor | $s=2$ split into 2 and 2′; $X=0$ split into 0 and 0′, the copy 0′ predicts | parent stage | [`QG_Instrumental_Edge`](../known_QC_gaps.py#L39) |
| 4 | tetrahedron: no edges; Q{0,1,2}, Q{0,1,3}, Q{0,2,3}, Q{1,2,3} | d-separation | $s=1,$ $X=0,$ both unsplit | first d-separation stage | [`QG_Triangle`](../known_QC_gaps.py#L42) |
| 5 | 1→2→3; Q{0,1}, Q{0,2}, Q{1,3} | d-separation | $s=1$ split into 1 and 1′; $X=0$ in split-predictor mode (childless, so left as it is) | fourth d-separation stage | [`QG_Instrumental_EdgeC`](../known_QC_gaps.py#L40) |
| 6 | 1→2, 1→3, 2→3; Q{0,2}, Q{0,3} (KPC; the class has 5 members, [1.1](#11-the-census-over-quantum-classes)) | semigraphoid closure | $s=3,$ $X=0$ in split-predictor mode (childless, untouched) | second closure stage | [`QG_Bell_C_Edge`](../known_QC_gaps.py#L80) |

The derivations, class by class:

1. **Class 1**, 0→1→2→3; Q{0,1}, Q{0,2}, Q{1,3}.
   * **Fritz step (parent predictor).** $s=1$ split into 1 and 1′, $X=0$ unsplit. 0 is a parent of 1′ (the copy has 1's parents), so 1′ keeps the edge 0→1′ alone; deleted: Q{0,1,1′} and Q{1,3,1′} for 1′ (the facets of 1 with the copy added), no certificate. Then 0, which has the children 1 and 1′, is marginalized with teleportation: it relays nothing visible (no parents), turns Q{0,1} and Q{0,2} into the classical facet C{1,2,1′} over its facet partners and its children, and teleports its share of Q{0,2} to 1 (its child in Q{0,1}), giving Q{1,2}.
   * **Output of the step.** 1→2→3, 1′→2; C{1,2,1′}, Q{1,2}, Q{1,3}.
   * **Follow-up.** Teleportation marginalization of 1: its parent 1′ is relayed to its child 2, its facets become C{1′,2} (from C{1,2,1′}) and C{2,3}, and its share of Q{1,3} is teleported to 2, giving Q{2,3}.
   * **Seed.** 1′→2→3; C{1′,2}, Q{2,3}, which is [`QG_Instrumental_EdgeC`](../known_QC_gaps.py#L40) with setting 1′.
2. **Class 2**, 0→1, 0→2, 1→2, 2→3; Q{0,1}, Q{0,2}, Q{1,3}.
   * **Fritz step (parent predictor).** The same step: $s=1$ split, $X=0$ unsplit, 1′ keeps 0→1′, deleted Q{0,1,1′} and Q{1,3,1′}; 0 (children 1, 2, 1′) marginalized with teleportation: its shares of Q{0,1} and Q{0,2} are both teleported to 1 and 2, giving Q{1,2}, and the classical relay is again C{1,2,1′}.
   * **Output of the step.** The same structure 1→2→3, 1′→2; C{1,2,1′}, Q{1,2}, Q{1,3}.
   * **Follow-up.** The same.
   * **Seed.** [`QG_Instrumental_EdgeC`](../known_QC_gaps.py#L40).
3. **Class 3**, 0→2, 1→2, 2→3; Q{0,1}, Q{0,2}, Q{1,3}.
   * **Fritz step (parent predictor).** $s=2$ split into 2 and 2′; $X=0$ has the child 2, so in split-predictor mode it is split into 0 and 0′ and the copy 0′ predicts. 0′ is a parent of 2′, so 2′ keeps the edge 0′→2′ alone; deleted: 0→2′, 1→2′ and Q{0,2,0′,2′}, no certificate. Then 0′ (children 2 and 2′) is marginalized with teleportation: its share of Q{0,1,0′} is teleported to 2, giving Q{0,1,2}; the classical relay over its facet partners and children is C{0,1,2,2′}.
   * **Output of the step.** 0→2, 1→2, 2→3, 2′→3; C{0,1,2,2′}, Q{0,1,2}, Q{1,3}.
   * **Follow-up.** Point distribution on 2′ gives 0→2, 1→2, 2→3; Q{0,1,2}, Q{1,3} (C{0,1,2} is absorbed); this class also contains 0→1, 0→3, 1→2, 3→1; Q{1,3}, Q{2,3}, where the teleportation marginalization of 3 (parent 0 relayed to its child 1, its share of Q{2,3} teleported to 1) gives 0→1→2; Q{1,2}.
   * **Seed.** [`QG_Instrumental_Edge`](../known_QC_gaps.py#L39).
4. **Class 4**, the tetrahedron: no edges; Q{0,1,2}, Q{0,1,3}, Q{0,2,3}, Q{1,2,3}.
   * **Fritz step (d-separation).** $s=1,$ $X=0,$ both unsplit. 0 sees its facets Q{0,1,2}, Q{0,1,3}, Q{0,2,3}; kept for 1: Q{0,1,2} and Q{0,1,3}, which become classical for 1; deleted: Q{1,2,3}. Certificate: 0 is d-separated from Q{1,2,3} given the kept facets ([6.7](#67-example-the-tetrahedron-unsplit-target-and-unsplit-predictor)). 0 is childless and is simply removed.
   * **Output of the step.** C{1,2}, C{1,3}, Q{2,3} on the nodes 1, 2, 3.
   * **Follow-up.** None.
   * **Seed.** [`QG_Triangle`](../known_QC_gaps.py#L42).
5. **Class 5**, 1→2→3; Q{0,1}, Q{0,2}, Q{1,3}.
   * **Fritz step (d-separation).** $s=1$ split into 1 and 1′, $X=0$ in split-predictor mode (childless, so left as it is). 0 sees Q{0,1,1′} and Q{0,2}; kept for 1′: Q{0,1,1′}, which becomes C{0,1,1′}; deleted: Q{1,3,1′}. Certificate: 0 is d-separated from Q{1,3,1′} given Q{0,1,1′} ([6.10](#610-example-a-split-predictor-and-a-split-target-worked-out-in-full)). Nothing is removed (five nodes).
   * **Output of the step.** 1→2→3, 1′→2; C{0,1,1′}, Q{0,1}, Q{0,2}, Q{1,3}.
   * **Follow-up.** Postselection on 0 (no parents; latent siblings 1, 1′, 2; quantum siblings 1, 2): C{1,2,1′} and Q{1,2} are added and 0 leaves, giving 1→2→3, 1′→2; C{1,2,1′}, Q{1,2}, Q{1,3}; then the teleportation marginalization of 1 as in class 1.
   * **Seed.** 1′→2→3; C{1′,2}, Q{2,3} = [`QG_Instrumental_EdgeC`](../known_QC_gaps.py#L40).
6. **Class 6**, 1→2, 1→3, 2→3; Q{0,2}, Q{0,3} (KPC; the class has 5 members, [1.1](#11-the-census-over-quantum-classes)).
   * **Fritz step (semigraphoid closure).** $s=3,$ $X=0$ in split-predictor mode (childless, untouched). 0 sees Q{0,2} and Q{0,3}; kept for 3: Q{0,3}, which becomes C{0,3}; deleted: the visible edges 1→3 and 2→3, which 0 does not see. d-separation fails (0 and 2 are connected through their shared facet Q{0,2}, which is not conditioned on); the closure certifies the `relabel` target set ([7.5](#75-worked-example-the-kpc-structure)). Nothing is removed.
   * **Output of the step.** 1→2; C{0,3}, Q{0,2}.
   * **Follow-up.** None: this is a member of the Bell class (settings 1 and 3, outcomes 2 and 0).
   * **Seed.** [`QG_Bell_C_Edge`](../known_QC_gaps.py#L80).

Reading the six classes against the procedure:

* **Classes 1 to 3 are the parent stage, which runs first.** In each the predictor 0 is a visible parent of the predicted node and also its latent sibling, so the pair belongs to the parent pool, the copy of the predicted node keeps the edge from the predictor (or from the predictor's copy) only, and the teleportation marginalization that removes the predictor carries its quantum share to the node that survives into the seed. An unsplit target would strip the facet that survives into the seed, which is why all three need the split target. In class 3 the predictor has a child, so split-predictor mode splits it too; this input also has a d-separation route, through the sibling pair 0, 1, which is recorded as well; the parent stage comes first.
* **Class 4, the tetrahedron,** is the one class settled at the first d-separation stage.
* **Class 5, the class of [6.10](#610-example-a-split-predictor-and-a-split-target-worked-out-in-full),** is settled at the fourth d-separation stage. It also has a closure route with an unsplit target ([7.8](#78-further-examples-from-the-census)), which the census never records since the d-separation stages run first.
* **Class 6, the KPC class,** is the only one that needs the closure, at the second closure stage, and its five members are proven at once since one certificate serves the class.

### 8.10 Experiment: descendants as predictors

By default the search removes the descendants of a target from its pool of predictors, because none of them can pass the d-separation test ([8.1](#81-the-pool-and-its-order-qmdagfritz_pool-qmdagfritz_targets)). This subsection reports what the census finds when they are admitted.

**Why a descendant may be admitted.** Descendants are nevertheless *sound* predictors when they have a channel: the quantum direction of [6.6](#66-theorem-d-separation-certificate) applies verbatim, and the classical direction of [7.7](#77-theorem-semigraphoid-closure-certificate) is a statement about every classical model satisfying the hypotheses, in which the noise plays no role.

**What the census finds.** The census experiment with `--descendants` ([1.1](#11-the-census-over-quantum-classes)) finds six candidate steps that only a descendant certifies. They are on three structures, all members of one open class of five structures, and all have the same target and predictor:

| structure | target | predictor | deleted | output with the predictor kept |
|---|---|---|---|---|
| 0→1→2→3; Q{0,2}, Q{1,3} | 1 | 3 | 0→1 | 1→2→3; C{1,3}, Q{0,2} |
| 0→1, 0→2, 1→2, 2→3; Q{0,2}, Q{1,3} | 1 | 3 | 0→1 | 0→2, 1→2, 2→3; C{1,3}, Q{0,2} |
| 0→1, 0→2, 1→2, 2→3; Q{1,3} | 1 | 3 | 0→1 | 0→2, 1→2, 2→3; C{1,3} |

The other three are the same steps with 3 dropped, which deletes 3 and leaves three-node outputs.

**Why d-separation fails and the closure succeeds.** In each, 3 is a descendant of 1 through 1→2→3, $\mathrm{seen}(3)$ is 3, its parent 2 and Q{1,3}, so $K$ = {Q{1,3}} and $D$ = {0→1, noise of 1}. d-separation fails because the noise of 1 reaches 3 along 1→2→3. The closure does not model that path as a leak. Its hypotheses are:

* the sufficiency of 3 for 1 ($H(1\mid 3)=0$);
* the Markov conditions of $G;$
* the observable independence 1 ⊥ 0 of the candidate (in the candidate, 1 reads Q{1,3} only).

From these it derives the `relabel` targets, Markov($G_1$) with 1 in the role of Q{1,3}; the LP agrees. Informally, d-separation insists that the predictor's knowledge of $s$ arrive through the shared facet, while the closure only uses *that* 3 knows 1, together with the independences, not *how*; here 3 learns 1 through the visible path and that is enough.

**What the experiment adds, and what it costs.** These are six more certified implications, not new gaps: none of the three outputs is a known gap as far as the seeds know (none reaches a seed under the elementary tricks), so their class stays unproven, and the ladder is unchanged. The cost, with the LP as the engine, is real: every descendant candidate goes to the LP, 1280 solves against 688 (176 seconds against 107); with the closure it is a fraction of a second either way.

---

## 9. Search, certificates and the four-node census (`qc_gap_search.py`, `Special Applications/proving_QC_Gaps.py`)

This section describes the census as a computation: the search over every four-node input class ([0.7](#07-the-census-the-seeds-and-the-names-of-the-tricks)) that produces the results of Section [1](#1-results-the-four-node-census). The results come first and the bookkeeping last:

* [9.1](#91-the-ten-stages): the ten stages of the search, and how many classes each of them proves;
* [9.2](#92-reading-the-tables-and-why-the-seeds-matter): how to read the tables of [1.1](#11-the-census-over-quantum-classes), why only the elementary tricks are assessed one at a time, and why the Bell classes are seeds;
* [9.3](#93-certificates-beyond-d-separation-attempted): how many candidate Fritz steps the search tests, and how each is justified;
* [9.4](#94-why-node-stitching-and-naive-marginalization-show-few-exclusive-proofs): why node stitching and naive marginalization show few exclusive proofs, with worked examples;
* [9.5](#95-the-known-gap-database-the-search-establishes): the database of known gaps that the search establishes, structures with classical sources included;
* [9.6](#96-bookkeeping-the-explorer-the-cache-and-the-record): the bookkeeping, that is, how the explorer identifies classes and records steps, what a run returns and what the cache stores;
* [9.7](#97-reproducing-the-results): how to reproduce the results.

The word "certificate" has two uses in this section. The certificate of a Fritz step is the test that justifies its deletion: d-separation ([6.6](#66-theorem-d-separation-certificate)) or the semigraphoid closure (Section [7](#7-the-semigraphoid-closure-certificate-for-the-fritz-piggyback-semigraphoidpy-qmdag_semigraphoid_certificate)), and no test at all for a parent predictor ([6.5](#65-the-parent-predictor-deletion-without-a-certificate)). The certificate of a proven class is its printed proof: the chain of recorded steps from the class down to a seed.

### 9.1 The ten stages

The census is one run of one explorer, in ten stages: the elementary stage, then nine Fritz stages of increasing cost. The explorer ([`ClosureExplorer`](../qc_gap_search.py#L222), [9.6](#96-bookkeeping-the-explorer-the-cache-and-the-record)) is the object that applies the tricks to the classes and records every step. Every seed class (a class whose gap is taken as known, [0.7](#07-the-census-the-seeds-and-the-names-of-the-tricks)) is known from the start.

The table gives the counts: the number of classes proven after each stage, with what each stage added, how many classes it was applied to and its wall-clock time on one core (the whole run takes about 17 seconds). The first row is the elementary stage. Each row below it adds one Fritz stage, named by its target mode and predictor mode (split or unsplit, [6.4](#64-node-splitting-split-and-unsplit-predictors-split-and-unsplit-targets-qmdagsplit_node)) and by its certificate. The stages themselves are described after the table.

| stage | proven (cumulative) | new | applied to | seconds |
|---|---|---|---|---|
| elementary reductions (with `identity` and the two lookups, degradation and upward) | 366 | 366 | 418 | 3 |
| + `Fritz`, parent predictor | 369 | 3 | 52 | 8 |
| + `Fritz`, unsplit target, unsplit predictor, d-separation | 370 | 1 | 49 | 0.5 |
| + `Fritz`, unsplit target, split predictor, d-separation | 370 | 0 | 48 | 0.5 |
| + `Fritz`, split target, unsplit predictor, d-separation | 370 | 0 | 48 | 0.6 |
| + `Fritz`, split target, split predictor, d-separation | 371 | 1 | 48 | 1.1 |
| + `Fritz`, unsplit target, unsplit predictor, semigraphoid closure | 371 | 0 | 47 | 0.6 |
| + `Fritz`, unsplit target, split predictor, semigraphoid closure | 372 | 1 | 47 | 0.8 |
| + `Fritz`, split target, unsplit predictor, semigraphoid closure | 372 | 0 | 46 | 0.8 |
| + `Fritz`, split target, split predictor, semigraphoid closure | 372 | 0 | 46 | 1.1 |
| remaining | | 46 | | |

**How the counts are obtained.** The explorer records the stage in which each transition (a recorded step) was found, so the cumulative count of classes proven after each stage is a graph query ([`GapReport.stage_counts`](../qc_gap_search.py#L505)). The **ladder** ([`ladder`](../qc_gap_search.py#L851)) computes the same ten counts from the recorded parameters (certificate, target mode, predictor mode) instead of the recorded stage, and the slow test checks that the two agree. A stage identified in this way by its parameters is a **rung** of the ladder.

**The elementary stage.** The elementary reductions act on every member of every input class, closed over everything reachable. Two more things belong to this stage:

* `identity`: an input class that is itself a seed class is proven as it stands;
* the two lookups, which register transitions without expanding their targets: every structure met has its degradations registered ([`degradation`](../qc_gap_search.py#L77), [0.4](#04-the-degradation-piggyback-quantum-source-to-classical-source)), and every input class has the classes structurally below it inside its classical class registered (`upward`, [0.6](#06-quantum-observational-equivalence-and-the-two-metagraphs)).

Reachability is transitive within this stage, so the set of classes it proves does not depend on the order in which the tricks are applied, and [`_fixpoint`](../qc_gap_search.py#L602) iterates the implication closure among the inputs until nothing changes. What each elementary trick proves on its own is read off the same record ([9.2](#92-reading-the-tables-and-why-the-seeds-matter)).

**The Fritz stages.** The classes the elementary stage left unproven are attacked by a cascade of **nine Fritz stages of increasing cost** ([`default_stages`](../qc_gap_search.py#L145), [`CASCADE`](../qc_gap_search.py#L840)), each applied once to every member of each class the previous stages left unproven. The first is the parent piggyback ([6.5](#65-the-parent-predictor-deletion-without-a-certificate)), over the visible parents of each target, with both target modes and both predictor modes and no certificate. The next four certify by d-separation over the latent siblings, in the order

1. unsplit targets, unsplit predictors;
2. unsplit targets, split predictors (childless untouched, childful split and the copy marginalized, [6.4](#64-node-splitting-split-and-unsplit-predictors-split-and-unsplit-targets-qmdagsplit_node));
3. split targets (the predicted node is split, [6.4](#64-node-splitting-split-and-unsplit-predictors-split-and-unsplit-targets-qmdagsplit_node)), unsplit predictors;
4. split targets, split predictors;

and the last four repeat the same modes with the semigraphoid closure of Section [7](#7-the-semigraphoid-closure-certificate-for-the-fritz-piggyback-semigraphoidpy-qmdag_semigraphoid_certificate) (the LP of [Appendix B](#appendix-b-the-entropic-lp-as-an-alternative-certificate-engine-entropic_lppy-qmdag_entropic_certificate) with `--engine lp`; `lp_only`: the pairs d-separation certified are already recorded, only the pairs it rejected go to the engine).

**Why this order.** The stages run from the cheapest to the most expensive:

* The parent stage comes first because it has no certificate at all.
* Every d-separation stage precedes every closure stage because the certificate beyond d-separation is, with the LP as the engine, the only expensive thing in the search.
* Within each group, unsplit targets precede split targets because a split target adds a node and so works on a larger structure.
* Unsplit predictors precede split ones because a split predictor leaves the node count unchanged and, when childful, adds a split and a marginalization.

**Breadth-first, then early exit** ([8.7](#87-two-kinds-of-stage-d-separation-breadth-first-then-the-semigraphoid-closure-with-early-exit)). In the parent and the d-separation stages each input is recorded in full: the test is cheap, and recording every route keeps the record independent of the order in which predictors are tried. In the closure stages each input is handled lazily ([`ClosureExplorer.extend`](../qc_gap_search.py#L376) with `stop_when_known`): each emitted child is recorded and reduced with the elementary tricks at once, and the input stops as soon as a child or its reduction is a seed, a cached gap or an input proven in an earlier stage.

Every recorded transition is a sound step, so the partial record is sound. The cumulative counts and the ladder do not depend on where an input stops, since an unproven input records everything and a proven input is proven at the stage of the step that proved it either way. In the four-node census early exit saves ten of 706 checks, all in the one class that the closure proves ([1.1](#11-the-census-over-quantum-classes)), which is nothing measurable. Early exit is there for larger inputs with the LP as the engine, where a stage has more candidates per root (a structure the stage is applied to) than four-node structures offer.

**Depth one.** Each stage applies the trick once to each member of each class still unproven, and the outputs are then reduced with the elementary tricks only, never with another Fritz step ("depth one"). The cascade thus loses proofs that would need two Fritz steps, which is accepted: a Fritz output has at least as many visible nodes as its source, and re-expanding every such output under the Fritz trick was found to be far slower than everything else combined. The counts of [1.1](#11-the-census-over-quantum-classes) are therefore lower bounds for what the tricks can prove, exact for the elementary stage.

**What the order decides.** The order decides cost (the closure runs only where the cheap stages failed) and which tricks a certificate prefers ([`build_report`](../qc_gap_search.py#L699) takes the shortest chain that uses only transitions recorded up to some stage, for the earliest stage at which such a chain exists). [`prove_gaps`](../qc_gap_search.py#L631) closes the proven set under implication from the seeds and attaches to every proven input a certificate, the shortest chain of transitions down to a named seed.

**Any known gap, in any order.** A Fritz step is not aimed at the Bell scenario, nor at any structure. It outputs whatever the deletion leaves, and that output, with everything the elementary tricks reduce it to, is compared with every gap known: the five seed classes and every class proven so far. Of the six classes whose proof needs a Fritz step, one ends at the Bell class, four at the instrumental scenario and one at the triangle. Of the 366 proven without one, 184 end at the instrumental scenario, 96 at the Evans structure and 84 at the triangle, and two are the Bell classes themselves.

Nor does it matter in which order gaps become known, and nothing is restarted when a class is proven. Every transition is recorded whether or not its target is known at the time, and after each stage the proven set is recomputed as the set of inputs from which a seed can be reached in the whole recorded graph ([`_fixpoint`](../qc_gap_search.py#L602)). A class that is still open has had every one of its steps recorded in full. If one of its outputs lies in a class that is proven later, by a later stage or by hand, the chain is already in the record; this is how [open_classes.md](open_classes.md) reads off that classes 2, 3, 6 and 7 would follow from class 1 (numbered as in the list of open classes in Section [10](#10-open-questions)).

Elementary tricks before a Fritz step are covered in the same way: an all-quantum four-node structure they reach is itself an input, with its own Fritz stages. As a check, rerunning the census from scratch with all 372 proven classes declared known from the start, or with the whole database of [9.5](#95-the-known-gap-database-the-search-establishes), leaves the same 46 classes open.

What the census does not try is a Fritz step on an intermediate that is not an input, that is, on a hybrid (a structure with some facets classical) or on a five-node Fritz output. This is the second depth of the paragraph "Depth one" above; for the open classes it was measured separately and proves none of their 94 intermediates ([open_classes.md](open_classes.md), 1.6).

### 9.2 Reading the tables, and why the seeds matter

**The counts are queries on the record.** Reachability restricted to any subset of tricks, or to any predicate on the recorded parameters, is a graph query over the recorded transitions. The tables of [1.1](#11-the-census-over-quantum-classes) hold two kinds of count:

* For each elementary trick, "via" counts the input classes provable with that one elementary trick alone (the two lookups, degradation and upward, always allowed) and "only via" the classes lost when that trick is removed and everything else kept.
* For the Fritz stages, the counts are cumulative over the rungs of the ladder ([9.1](#91-the-ten-stages)).

**Why only the elementary tricks are assessed alone.** Every elementary trick is cheap and the elementary stage is exhaustive, so it is the one stage where "what does this trick prove alone, and what is lost without it" is a meaningful and affordable question, and [1.1](#11-the-census-over-quantum-classes) reports exactly that, per elementary trick. The search never asks how many inputs an expensive piggyback proves *alone*: that would mean closing the search under that piggyback over everything reachable, which is exactly the cost the cascade avoids, and it would answer a question nobody needs answered. The expensive steps are assessed by the cumulative ladder of [1.1](#11-the-census-over-quantum-classes) alone: what each successive stage adds to everything cheaper.

**What a rung does and does not say.** Since each stage of the cascade ran only on classes the earlier stages left unproven, a class counted at an early rung is proven by that cheaper combination of modes and certificate. This does not mean that the later, stronger combinations would fail on it: split predictors subsume unsplit ones and the closure subsumes d-separation ([6.4](#64-node-splitting-split-and-unsplit-predictors-split-and-unsplit-targets-qmdagsplit_node), [7.6](#76-why-relabel-is-justified)).

**Why the Bell classes are seeds.** Every piggyback except Fritz reduces the number of visible nodes or leaves the structure's nodes alone (degradation, upward), so for a four-node class a four-node seed class can only be reached by a Fritz step, a degradation or the identity. The Bell classes have to be seeds, on the strength of the direct Bell argument: deleting a facet is never a piggyback, so the Bell scenario without a settings link is not reached from the three-node seeds, and the one with correlated settings is not reached from any other class. Making the seeds inputs costs nothing and makes the counting uniform: the census starts from 418 classes and the identity piggyback proves two of them.

### 9.3 Certificates beyond d-separation attempted

The census counts every candidate step with a single target (a predictor set, a target and the deletion read off) over the nine Fritz stages, split structures included; the re-checks made for joint steps are not counted ([`ENTROPIC_STATS`](../quantum_mDAG.py#L69)). The two target sets named below, `relabel` and `markov`, are those of [7.4](#74-the-two-target-sets-as-containment-of-d-separation-models-qmdag_semigraphoid_certificate).

| candidate steps | how justified | count |
|---|---|---|
| parent stage | no certificate needed (`parent`, [6.5](#65-the-parent-predictor-deletion-without-a-certificate)) | 805 |
| certified stages | d-separation | 388 |
| certified stages, d-separation fails | the `relabel` target set | 37 |
| certified stages, d-separation fails | nothing: the step is not taken | 659 |

The last two rows are the same with the semigraphoid closure (Section [7](#7-the-semigraphoid-closure-certificate-for-the-fritz-piggyback-semigraphoidpy-qmdag_semigraphoid_certificate)) and with the LP of [Appendix B](#appendix-b-the-entropic-lp-as-an-alternative-certificate-engine-entropic_lppy-qmdag_entropic_certificate): run together (`--engine both`) the two disagree on no candidate, and no LP solve hits the time limit.

The `markov` target set is off ([7.4](#74-the-two-target-sets-as-containment-of-d-separation-models-qmdag_semigraphoid_certificate)). Switched on in the present code, it certifies none of the 659 candidates that `relabel` leaves. An earlier version of the code also had joint steps certified beyond d-separation and extra deletions ([10.3](#103-variants-tried-settled-or-left-out)); in a run of that version with all three on, `markov` certified 91 candidates after `relabel` had failed, the joint steps 27 and the extra deletions hundreds, and none of these decided an input. Those two options no longer exist, so these three figures cannot be measured again.

Among candidates that d-separation rejects, the closure succeeds for roughly one in nineteen. Success is far from universal, and a failure is not a proof that the implication is false ([7.6](#76-why-relabel-is-justified)); [8.3](#83-justifying-the-deletion-one-set-two-certificates-qmdagfritz_certificate) shows a failure that is a true negative. Almost none of these certified steps is decisive ([1.1](#11-the-census-over-quantum-classes)): their outputs are structures the cheaper tricks reach as well, or structures that are not gaps as far as the seeds know. Without the closure stages' early exit ten more candidates are checked, all in the one class that the closure proves, and nine of them are certified (706 checks, 46 certified).

### 9.4 Why node stitching and naive marginalization show few exclusive proofs

**Node stitching.** Its characteristic product is a Bell variant derived from an instrumental variant. In an earlier census, which started from the three-node seeds alone, it was the sole route for exactly the three Bell variants that could be proven at all. Now that the Bell classes are seeds, every class it proves (seven in all) is also reachable by PD or postselection ([1.1](#11-the-census-over-quantum-classes)). Node stitching is the only reduction that re-uses an outcome as a setting; how much it contributes is bounded by how many Bell-type structures the input set contains that the seed list does not already hold.

**The two marginalizations.** They coincide at a node without a quantum facet, so naive marginalization alone is never exclusive: wherever it applies, the teleportation version makes the same move. The 9 classes lost without teleportation marginalization are where the removed node holds a quantum share and relaying it to the node's children is what the proof needs. Postselection covers part of the same ground, since postselecting on a node $v$ without visible parents adds a quantum facet over all quantum siblings of $v$ (entanglement swapping), the same connection teleportation relays to the children of $v,$ only not restricted to children; what postselection cannot provide is the relayed visible input of $v,$ and where that input matters, or where postselection is blocked by a grandparent, teleportation marginalization is the route. An earlier census, which closed the unsplit-predictor Fritz trick over everything reachable, found alternative routes for all of these; the elementary stage alone does not.

**Example** (only via teleportation marginalization). Input: 0→1→2; Q{0,1,3}, Q{0,2}. Node 0 is the setting of 1, holds a share of a tripartite state with 1 and 3, and a bipartite state with 2. Marginalizing 0 with teleportation relays its share of Q{0,1,3} and of Q{0,2} to its child 1: 1→2; Q{1,3}, Q{1,2}, whose degradation C{1,3}, Q{1,2} is [`QG_Instrumental_C`](../known_QC_gaps.py#L41). Naive marginalization of 0 relays only a classical common cause to 1 and 2 and leaves 1→2; C{1,2}, Q{1,3}, not a known gap; postselecting on 0 gives 1→2; Q{1,2,3}, PD on 0 gives 1→2; Q{1,3}, neither a known gap; node stitching needs an exogenous node, and 0 holds facets; and no reduction at 1, 2 or 3 reaches a seed either, since each either destroys the setting or isolates 3. A second example of the same shape, 0→1, 1→2, 1→3 with Q{0,1,2}, Q{0,3}, relays the shares of 0 to 1 and gives [`QG_Evans`](../known_QC_gaps.py#L43).

**Example** (only via marginalization, either kind). Input: 2→3; Q{0,1}, Q{0,3}, Q{1,2}. Node 2 holds one share, of Q{1,2}, and feeds 3. Marginalizing 2 relays that share to 3 as a classical facet: Q{0,1}, Q{0,3}, C{1,3}, which degrades (Q{0,3} made classical) to [`QG_Triangle`](../known_QC_gaps.py#L42). Teleportation has nothing to add here, because 3 shares no quantum facet with 2 and so there is no entangled pair to teleport through (Section [3](#3-marginalization-naive-and-with-teleportation-naive_marginalization-teleportation_marginalization)); both marginalizations give the same structure, and the input counts in the "either kind" row but in neither single row. PD on 2 or postselecting on 2 leaves Q{0,1}, Q{0,3} without the third side of the triangle.

### 9.5 The known-gap database the search establishes

Every class or structure the search touches, inputs, intermediates and the hybrid structures with classical facets that the Fritz steps create, is a proven QC gap as soon as it reaches a seed, and [`GapReport.proven_structure_ids`](../qc_gap_search.py#L555) returns that set of keys. This is the database to keep: the census inputs have every latent quantum, but a structure with some facets classical is a *weaker* structure, its gap is not implied by the gap of the all-quantum version (making a facet classical shrinks the quantum set and leaves the classical set), and it is a gap only if a chain of piggybacks from it reaches a seed. Fritz steps are indifferent to whether the shared facet is classical or quantum, so chains often transfer; postselection and teleportation are not, since they create quantum facets only among quantum siblings.

**The database is not the set of four-node structures the cheap tricks can prove.** The inputs are the all-quantum structures, so a hybrid structure enters the database only when some chain from an all-quantum input passes through it. The intermediate G(3) of the split-predictor example of [6.10](#610-example-a-split-predictor-and-a-split-target-worked-out-in-full), 0→1→2; Q{0,1}, Q{0,2}, C{0,1,3′}, C{0,1,2}, is proven by one teleportation marginalization to [`QG_Instrumental_C`](../known_QC_gaps.py#L41); it is in the database only because the chain the census records for that input passes through it. A hybrid structure that no recorded chain happens to visit is absent even when a single elementary step proves it.

The size of the omission: recolouring each facet of each all-quantum input classical or quantum gives 8756 distinct four-node structures with at least one quantum facet, and the elementary closure over all of them from the seeds (18 seconds) proves 5082, of which 3829 are not in the census database of 393 keys. G(3) is not even among these recolourings: its quantum facets Q{0,1} and Q{0,2} lie strictly inside its classical facets, a shape that only arises when a facet loses a quantum member, so the hybrid input space is larger still. Hybrid structures as *inputs* are therefore a cheap and worthwhile extension of the census (Section [10](#10-open-questions)), and the right place to apply the elementary tricks exhaustively; with the degradation piggyback of [0.4](#04-the-degradation-piggyback-quantum-source-to-classical-source) as a trick, only the weakest colouring of each proven structure would need storing.

### 9.6 Bookkeeping: the explorer, the cache and the record

This subsection is for readers who want to use the code or inspect a proof: how the explorer identifies classes and records steps, what a run returns (the record), and what is kept between runs (the cache).

**The explorer.** [`ClosureExplorer`](../qc_gap_search.py#L222) expands every reachable **class**, exactly once under a set of tricks, and records each step as a [`Transition`](../qc_gap_search.py#L59). A class is expanded by applying every trick to every one of its members, since different members have different children, and it is proven as soon as any member reaches a known gap.

**Class keys.** Each class is identified by a key. The keys come from [`equivalence_classes.ClassIndex`](../equivalence_classes.py#L31) ([0.6](#06-quantum-observational-equivalence-and-the-two-metagraphs)):

* an all-quantum structure of three or four nodes is keyed by its quantum equivalence class (the smallest unlabelled id among the class's members);
* a hybrid to which the rule [`QmDAG.lift`](../quantum_mDAG.py#L235) applies (every one of its classical sources can be replaced by a quantum one for free, [0.5](#05-when-a-classical-source-is-as-good-as-a-quantum-one)) is keyed by the class of the all-quantum structure that the rule returns;
* anything else (a hybrid to which the rule does not apply, a five-node structure) is keyed by its own unlabelled id, so that without a class index the explorer is a structure-by-structure search.

Keying on unlabelled ids within a class is legitimate because every piggyback is label-equivariant.

**The fields of a transition.** A recorded step is a [`Transition(trick, params, source, target, via, produced)`](../qc_gap_search.py#L59):

| field | what it holds |
|---|---|
| [`trick`](../qc_gap_search.py#L60) | the piggyback's name as in [`PIGGYBACK_VERSIONS`](../qc_gap_search.py#L32), or `upward`, [`degradation`](../qc_gap_search.py#L77) |
| [`params`](../qc_gap_search.py#L61) | the step's parameters, stated in the labels of the structure the step was applied to |
| [`source`](../qc_gap_search.py#L62), [`target`](../qc_gap_search.py#L63) | the class keys at both ends of the step |
| [`via`](../qc_gap_search.py#L64) | the unlabelled id of the member of the class [`source`](../qc_gap_search.py#L62) that the step was applied to |
| [`produced`](../qc_gap_search.py#L65) | the unlabelled id of the structure the step made, a member of the class [`target`](../qc_gap_search.py#L63) |

The [`params`](../qc_gap_search.py#L61) read, trick by trick:

* point distribution: `(('drop', v),)`;
* both marginalizations: `(('marginalize', v),)`;
* postselection: `(('condition', v),)`;
* node stitching: `((('sink', y), ('source', x)),)`;
* a degradation: `(('classical', (facet, ...)),)`;
* the upward rule: `(('above', G), ('below', G'))`;
* a Fritz step: `targets`, `target_mode`, `deleted` (the labels of the deleted set $D$ of [8.2](#82-what-a-predictor-sees-and-the-deletion-it-dictates-qmdagfritz_deletion): node indices, a prime for a copy, `Q{…}` or `C{…}` for a facet), `predictor`, `predictor_mode`, [`certificate`](../tests/test_fritz_entropic.py#L28) (`parent`, `dsep`, `semigraphoid` or `entropic`) and, when several predictors are marginalized, `order`.

The order dependence of Section [3](#3-marginalization-naive-and-with-teleportation-naive_marginalization-teleportation_marginalization) is enumerated rather than fixed because different removal orders give different structures and none is canonical.

**Indices, copies and what a certificate prints.** Every transformation returns a plainly indexed structure. A copy made by node splitting is the next free index, and the recorded parameters print it with a prime (0′ is the copy of 0). Certificates print the member a step was applied to above the step and, when the structure the last step produced is not the representative of the known gap's class, the representative beside it.

**The record of a run.** [`run_search`](../Special%20Applications/proving_QC_Gaps.py#L83) returns a [`GapReport`](../qc_gap_search.py#L493) (and the cache). Its explorer holds everything the search did:

* `report.explorer.edges[key]` is the list of transitions recorded from the class `key`;
* `report.explorer.structures[uid]` is every labelled structure met, by its own unlabelled id;
* `report.explorer.representatives[key]` is one structure per class.

The report itself holds the proofs:

* `report.proven[key]` is the certificate chain of a proven class (empty for a seed class);
* `report.seed_hit[key]` is the seed class the chain ends at;
* `report.certificate(g)` renders the chain as the text printed in [8.9](#89-the-six-classes-that-need-a-fritz-step): each step with the structure it was applied to, then the known gap reached.

**The cache** ([`gap_cache.py`](../gap_cache.py), [`cache/known_gaps.json`](../cache/known_gaps.json)). The cache stores the same chain per class as JSON. Every proven input class is stored with:

* its key;
* the structure of its representative and of its other members, as node, edge and facet lists;
* the seed class its certificate ends at (`seed`);
* the chain of transitions (`chain`), each with the six fields of the table above;
* the rendered certificate ([`certificate`](../tests/test_fritz_entropic.py#L28));
* the **version** of every piggyback the chain uses (`versions`; the versions are [`PIGGYBACK_VERSIONS`](../qc_gap_search.py#L32) in [`qc_gap_search.py`](../qc_gap_search.py)).

The file carries a format version that changes when the meaning of the keys does. A certificate that ends at a cached gap inherits that entry's versions, so dependence is transitive. When a piggyback is corrected its version is bumped, and loading the cache drops exactly the entries whose proof relied on it; everything else stays known, and the expensive stages run only on the classes that are neither cached nor proven by the elementary stage.

### 9.7 Reproducing the results

**Running the census.** Everything in Section [1](#1-results-the-four-node-census) comes from one command, run from the repository root:

```
python "Special Applications/proving_QC_Gaps.py" --no-cache
```

It builds the two metagraphs, runs the cascade of [9.1](#91-the-ten-stages), and prints the counts of [1.1](#11-the-census-over-quantum-classes), the stage table, the ladder and one certificate per class proven through a Fritz step. Without `--no-cache` the classes stored in [`cache/known_gaps.json`](../cache/known_gaps.json) count as known and only the rest is searched ([9.6](#96-bookkeeping-the-explorer-the-cache-and-the-record)).

The options:

* `--engine lp` or `--engine both` chooses the certificate engine beyond d-separation ([Appendix B](#appendix-b-the-entropic-lp-as-an-alternative-certificate-engine-entropic_lppy-qmdag_entropic_certificate));
* `--no-split-predictors`, `--no-entropic`, `--descendants` and `--no-early-exit` switch the corresponding stages or options off and on.

Two more entry points:

* `python "Special Applications/census_breakdowns.py"` prints, for every elementary piggyback, classes that are lost without it and, for every rung of the ladder, the classes it adds with a certificate each (the rows of the tables of [1.1](#11-the-census-over-quantum-classes) and [8.9](#89-the-six-classes-that-need-a-fritz-step) were read off its output);
* `pytest -m slow` pins every number of [1.1](#11-the-census-over-quantum-classes) and [9.1](#91-the-ten-stages) in [`tests/test_baseline_slow.py`](../tests/test_baseline_slow.py).
* `python "Special Applications/census_experiments.py"` prints the figures of the experiments quoted beside the census: the split of the "only via" counts ([1.1](#11-the-census-over-quantum-classes)), the cost of the closure ([7.9](#79-strength-agreement-with-the-lp-and-cost)), descendants as predictors ([8.10](#810-experiment-descendants-as-predictors)), the early exit and the `markov` target set ([9.3](#93-certificates-beyond-d-separation-attempted)), the recolourings ([9.5](#95-the-known-gap-database-the-search-establishes)) and the sub-maximal deletions ([10.2](#102-open-questions)); with `--lp` it also counts the LP solves ([B.4](#b4-lp-size-and-time)). The slow tests pin all of them but the LP counts.

**Which classes a given set of piggybacks proves.** The counts "via" and "only via" of [1.1](#11-the-census-over-quantum-classes) are reachability over the recorded transitions, closed under implication among the inputs ([`_fixpoint`](../qc_gap_search.py#L602)). The two lookups ([`LOOKUP_TRICKS`](../qc_gap_search.py#L202): degradation and the upward rule) are always allowed, since they are part of what "known" means. The same function answers any other combination, for instance the classes provable with point distribution and teleportation marginalization together, and those among them that need the teleportation:

```python
import proving_QC_Gaps as census                       # from "Special Applications"
from qc_gap_search import _fixpoint, LOOKUP_TRICKS
report, cache = census.run_search(use_cache=False, verbose=False)
ex, seeds = report.explorer, set(report.seed_keys)
with_both = _fixpoint(ex, report.input_ids, seeds, frozenset({'PD', 'teleportation_marginalization'}) | LOOKUP_TRICKS)
only_tele = with_both - _fixpoint(ex, report.input_ids, seeds, frozenset({'PD'}) | LOOKUP_TRICKS)
```

(356 and 11 classes on the census.)

**Further queries on the same report.**

* *Certificates.* `report.certificate(ex.representatives[key])` prints the recorded certificate of a class, which is the shortest chain among the transitions of the earliest stages that suffice ([9.1](#91-the-ten-stages)); a certificate restricted to a chosen set is [`census_breakdowns.full_certificate(report, key, keep)`](../Special%20Applications/census_breakdowns.py#L58) with `keep` a predicate on transitions, e.g. `lambda t: t.trick in tricks`.
* *The structures behind a count of classes.* They are [`index.members(key)`](../qc_gap_search.py#L236) for `index = ex.index` (the class's all-quantum members) and [`index.size(key)`](../qc_gap_search.py#L236); `report.proven_structure_ids()` is the set of every structure the search touched that reaches a seed, hybrids included ([9.5](#95-the-known-gap-database-the-search-establishes)).
* *The rows the census prints.* To change the "via" and "only via" rows, edit [`TRICK_GROUPS_FOR_REPORT`](../qc_gap_search.py#L206) in [`qc_gap_search.py`](../qc_gap_search.py): each entry names a group of tricks to count (the lookups are added) and the tricks whose removal defines "only via".
* *The Fritz modes.* To assess them, [`ladder`](../qc_gap_search.py#L851) classifies the Fritz transitions by their recorded [`certificate`](../tests/test_fritz_entropic.py#L28), `target_mode` and `predictor_mode`, and `census.proven_through_fritz(report)` lists the classes whose certificate uses a Fritz step.

---

## 10. Open questions

The census ([0.7](#07-the-census-the-seeds-and-the-names-of-the-tricks), Section [1](#1-results-the-four-node-census)) takes as inputs the classes of four-node structures with every latent facet quantum, except those classically equivalent to a latent-free DAG, and tries to prove a QC gap for each. This section records what remains after it: the classes it leaves open ([10.1](#101-the-46-open-classes)), the questions that are open ([10.2](#102-open-questions)), and variants that were tried, settled or left out of the search ([10.3](#103-variants-tried-settled-or-left-out)).

### 10.1 The 46 open classes

The census leaves 46 quantum equivalence classes unproven. A quantum equivalence class is a set of structures that the rules of [0.6](#06-quantum-observational-equivalence-and-the-two-metagraphs) prove to have the same quantum model set. A class is proven when a chain of piggybacks leads from one of its members to a seed, a structure with an established QC gap ([0.7](#07-the-census-the-seeds-and-the-names-of-the-tricks)).

The table below lists the 46 classes, one representative each with the number of structures in the class (61 in all, in 46 distinct classical classes; a classical class is a set of structures proven to have the same classical model set, [0.6](#06-quantum-observational-equivalence-and-the-two-metagraphs)). Every representative contains a visible edge out of a node that also holds quantum facets, and most contain a chain of two or three visible edges. The labellings are those of the representatives the class index stores ([9.6](#96-bookkeeping-the-explorer-the-cache-and-the-record)), not necessarily the temporally ordered ones.

A separate record, [open_classes.md](open_classes.md), says where effort on them is best spent:

* which open classes would follow from a gap in which (classes 2, 3, 6 and 7 from class 1, and nothing else);
* which hybrid structures (structures with at least one classical facet, [0.1](#01-structures-and-model-sets)) would settle several classes at once;
* what a second Fritz depth adds, that is, a Fritz step applied to an intermediate structure that is not an input ([9.1](#91-the-ten-stages)): nothing;
* and, as a separate question, which classical equivalences would shorten the list if proven.

<details><summary>The 46 remaining classes (edges; quantum facets; structures in the class)</summary>

| | edges | quantum facets | structures |
|---|---|---|---|
| 1 | 2→3 | {0,2}, {0,3}, {1,2,3} | 12 |
| 2 | 2→3 | {0,2}, {0,3}, {1,2}, {1,3}, {2,3} | 1 |
| 3 | 1→2, 2→3 | {0,2}, {0,3}, {1,2}, {1,3} | 1 |
| 4 | 0→1, 1→2, 1→3 | {0,2}, {0,3} | 1 |
| 5 | 0→1, 1→2, 1→3 | {0,2}, {0,3}, {1,3} | 1 |
| 6 | 1→3, 2→3 | {0,1}, {0,3}, {1,2}, {2,3} | 1 |
| 7 | 1→2, 1→3, 2→3 | {0,2}, {0,3}, {1,2}, {1,3} | 1 |
| 8 | 0→3, 1→2 | {0,1}, {0,2}, {1,2,3} | 1 |
| 9 | 0→3, 1→2 | {0,1}, {0,2,3}, {1,2,3} | 1 |
| 10 | 0→3, 1→2 | {0,1}, {0,2}, {0,3}, {1,2,3} | 1 |
| 11 | 0→3, 1→2 | {0,1}, {0,2}, {1,3}, {2,3} | 1 |
| 12 | 0→3, 1→2 | {0,1}, {0,2}, {1,2}, {1,3}, {2,3} | 1 |
| 13 | 0→3, 1→2 | {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3} | 1 |
| 14 | 0→2, 1→2, 2→3 | {0,1}, {1,3} | 5 |
| 15 | 0→1, 1→2, 2→3 | {0,2}, {0,3}, {1,2,3} | 1 |
| 16 | 0→1, 1→2, 2→3 | {0,2}, {0,3}, {1,3} | 1 |
| 17 | 0→1, 1→2, 2→3 | {0,1}, {0,2}, {0,3}, {1,2,3} | 1 |
| 18 | 0→1, 1→2, 2→3 | {0,2}, {0,3}, {1,2}, {1,3} | 1 |
| 19 | 0→1, 1→2, 2→3 | {0,1}, {0,2}, {0,3}, {1,3} | 1 |
| 20 | 0→1, 1→2, 2→3 | {0,2}, {0,3}, {1,3}, {2,3} | 1 |
| 21 | 0→1, 1→2, 2→3 | {0,1}, {0,2}, {0,3}, {1,2}, {1,3} | 1 |
| 22 | 0→1, 1→2, 2→3 | {0,1}, {0,2}, {0,3}, {1,3}, {2,3} | 1 |
| 23 | 0→1, 1→2, 2→3 | {0,2}, {0,3}, {1,2}, {1,3}, {2,3} | 1 |
| 24 | 0→1, 1→2, 2→3 | {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3} | 1 |
| 25 | 0→3, 1→2, 2→3 | {0,1}, {0,2}, {1,2,3} | 1 |
| 26 | 0→3, 1→2, 2→3 | {0,1}, {0,2,3}, {1,2,3} | 1 |
| 27 | 0→3, 1→2, 2→3 | {0,1}, {0,2}, {1,3} | 1 |
| 28 | 0→3, 1→2, 2→3 | {0,1}, {0,2,3}, {1,3} | 1 |
| 29 | 0→3, 1→2, 2→3 | {0,1}, {0,2}, {0,3}, {1,2,3} | 1 |
| 30 | 0→3, 1→2, 2→3 | {0,1}, {0,2}, {1,2}, {1,3} | 1 |
| 31 | 0→3, 1→2, 2→3 | {0,1}, {0,2,3}, {1,2}, {1,3} | 1 |
| 32 | 0→3, 1→2, 2→3 | {0,1}, {0,2}, {1,3}, {2,3} | 1 |
| 33 | 0→3, 1→2, 2→3 | {0,1}, {0,2}, {0,3}, {1,3} | 1 |
| 34 | 0→3, 1→2, 2→3 | {0,1}, {0,2}, {1,2}, {1,3}, {2,3} | 1 |
| 35 | 0→3, 1→2, 2→3 | {0,1}, {0,2}, {0,3}, {1,2}, {1,3} | 1 |
| 36 | 0→3, 1→2, 2→3 | {0,1}, {0,2}, {0,3}, {1,3}, {2,3} | 1 |
| 37 | 0→3, 1→2, 2→3 | {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3} | 1 |
| 38 | 0→1, 1→2, 1→3, 2→3 | {0,2}, {0,3} | 1 |
| 39 | 0→1, 1→2, 1→3, 2→3 | {0,2}, {0,3}, {1,2} | 1 |
| 40 | 0→1, 1→2, 1→3, 2→3 | {0,2}, {0,3}, {1,3} | 1 |
| 41 | 0→1, 1→2, 1→3, 2→3 | {0,2}, {0,3}, {1,2,3} | 1 |
| 42 | 0→1, 1→2, 1→3, 2→3 | {0,2}, {0,3}, {2,3} | 1 |
| 43 | 0→1, 1→2, 1→3, 2→3 | {0,2}, {0,3}, {1,2}, {1,3} | 1 |
| 44 | 0→1, 1→2, 1→3, 2→3 | {0,2}, {0,3}, {1,2}, {2,3} | 1 |
| 45 | 0→1, 1→2, 1→3, 2→3 | {0,2}, {0,3}, {1,3}, {2,3} | 1 |
| 46 | 0→1, 1→2, 1→3, 2→3 | {0,2}, {0,3}, {1,2}, {1,3}, {2,3} | 1 |

</details>

Bell scenarios with extra structure among the settings are not among the open classes. The Bell scenario with correlated settings is a seed class ([0.7](#07-the-census-the-seeds-and-the-names-of-the-tricks), [9.2](#92-reading-the-tables-and-why-the-seeds-matter)), and that class contains both 0→1, 2→3; Q{0,2}, Q{1,3}, the Bell scenario with entangled settings (0 and 2 are the settings of 1 and 3, Q{1,3} the shared state), and 0→1, 0→2, 2→3; Q{1,3}, where one party's setting influences the other's. Both have a QC gap by the usual argument, because the latents of the settings are independent of the latent of the outcomes, so any classical model is local for $P(a,b\mid x,y).$ No piggyback reaches them from the Bell scenario without a settings link, since deleting a facet or an edge between settings is not a piggyback; this is why the class is a seed.

### 10.2 Open questions

The first three questions concern the Fritz piggyback and its certificates, the fourth the set of inputs, the fifth the record that a closure leaves, and the last the counting by class.

* **Completeness of the certificate beyond d-separation.** The certificate beyond d-separation is the semigraphoid closure of Section [7](#7-the-semigraphoid-closure-certificate-for-the-fritz-piggyback-semigraphoidpy-qmdag_semigraphoid_certificate), or the LP of [Appendix B](#appendix-b-the-entropic-lp-as-an-alternative-certificate-engine-entropic_lppy-qmdag_entropic_certificate), and [`ENTROPIC_STATS`](../quantum_mDAG.py#L69) tallies its attempts ([9.3](#93-certificates-beyond-d-separation-attempted)). The failures recorded there are failures of the closure (and of the LP), not necessarily of the piggyback ([7.6](#76-why-relabel-is-justified)).

  The two target sets of [7.4](#74-the-two-target-sets-as-containment-of-d-separation-models-qmdag_semigraphoid_certificate) are instances of one scheme ([7.6](#76-why-relabel-is-justified)): each latent is replaced by a set of variables of the input, and the Markov property of the candidate is asked for with the substitution. With $s$ the predicted node and $L$ the single facet it keeps, `markov` substitutes the facet $L$ for itself and `relabel` substitutes the node $s$ for it. Substituting other sets of variables for the latents of $G',$ beyond $\lbrace L\rbrace$ and $\lbrace s\rbrace,$ is the natural next level: each substitution is the d-separation model of a relabelled DAG, checked by the same containment, and the soundness proof is that of [7.6](#76-why-relabel-is-justified).

* **Sub-maximal deletions.** In a Fritz step a predictor set $\mathbf X$ predicts a node $s,$ and $s$ loses parents (Section [8](#8-the-fritz-search-procedure-qmdagfritz_steps-qc_gap_searchfritz_tricks)). A predictor sees itself and its parents; $\mathrm{seen}(\mathbf X)$ is everything the predictors see, and $\mathrm{common}(s)$ is the set of parents of $s$ among it ([8.2](#82-what-a-predictor-sees-and-the-deletion-it-dictates-qmdagfritz_deletion)). The deletion of a Fritz step is all of $\mathrm{Pa}(s)\setminus\mathrm{seen}(\mathbf X)$ or nothing: a predictor that cannot see a kept parent cannot announce $f(K).$ Here $K$ is the set of parents that $s$ keeps and $f$ the function of them that $s$ computes.

  There is a sound way around it. Let $U=K\setminus\mathrm{seen}(\mathbf X)$ be the kept parents the predictors cannot see. In the quantum direction, $s$ computes $f(\mathrm{common}(s),U)$ and each predictor announces the *table* $u\mapsto f(\mathrm{common}(s),u),$ a subvariable from which $s$ is recovered given $U;$ the hypothesis becomes $H(s\mid\mathbf X,U)=0,$ and the classical direction has to show that a classical model of $G$ satisfying it is a model of the candidate in which $s$ keeps $K.$ Neither Theorem [6.6](#66-theorem-d-separation-certificate) nor Theorem [7.7](#77-theorem-semigraphoid-closure-certificate) covers this hypothesis as stated, so it is a new theorem to prove, not an option to switch on in the code.

  Sub-maximal deletions are available for many pairs. Of the 2241 (sibling predictor, target) pairs the certified stages consider in the four-node census (up to relabelling), 1276 delete one non-noise parent, 790 two, 162 three and 13 four, so 965 pairs, on 507 of the 996 structures of the input classes, admit a strict sub-deletion. (A sibling predictor shares a facet with its target, [6.1](#61-the-mechanism); the certified stages are the Fritz stages that need a certificate, [9.1](#91-the-ten-stages).) Whether any of them proves one of the open classes has not been tried.

* **Semigraphoid against Shannon, and the dropped half of the perfect prediction.** The closure of Section [7](#7-the-semigraphoid-closure-certificate-for-the-fritz-piggyback-semigraphoidpy-qmdag_semigraphoid_certificate) certifies a subset of what the LP of [Appendix B](#appendix-b-the-entropic-lp-as-an-alternative-certificate-engine-entropic_lppy-qmdag_entropic_certificate) certifies, for two reasons ([7.9](#79-strength-agreement-with-the-lp-and-cost)):

  * the semigraphoid rules derive fewer implications than the Shannon inequalities;
  * the closure uses only the sufficiency half of $H(s\mid\mathbf X)=0$ ([7.3](#73-the-hypotheses-and-how-perfect-prediction-enters-qmdag_semigraphoid_model-semigraphoidadd_functional_dependence)). The sufficiency half says that $s$ is independent of every other variable given the predictors; the determinism half, which the closure drops, says that $s$ carries no randomness of its own beyond the other variables.

  In four nodes the subset is everything. An input (or even a candidate step) that the LP certifies and the closure does not would be a concrete inference beyond the semigraphoid rules arising in a causal structure, worth having; five-node inputs are the place to look (`--engine both` reports every disagreement).

  A candidate that the LP certifies only through the determinism half of the prediction would also say whether the engine should represent degenerate triplets, the triplets whose two sides overlap ([7.3](#73-the-hypotheses-and-how-perfect-prediction-enters-qmdag_semigraphoid_model-semigraphoidadd_functional_dependence)).

* **Hybrid inputs.** A hybrid is a structure with at least one classical facet ([0.1](#01-structures-and-model-sets)). The census inputs have every facet quantum, and [9.5](#95-the-known-gap-database-the-search-establishes) shows that this leaves out thousands of four-node structures with classical facets that the elementary tricks prove in seconds. Among them are structures whose quantum facets lie inside classical ones, which no recolouring of an all-quantum input produces (a recolouring makes each facet of the input classical or quantum, [9.5](#95-the-known-gap-database-the-search-establishes)).

  There are two fixes. The direct one is to enumerate all pairs (classical complex, quantum complex) on four nodes and to run the elementary stage of the census ([9.1](#91-the-ten-stages)) on them. The other is to run the piggybacks in reverse from the seeds ([Appendix A](#appendix-a-the-piggybacks-in-reverse)).

* **Derivation traces.** The closure certifies by containment, that is, by checking that the closed set of independences contains the targets ([7.4](#74-the-two-target-sets-as-containment-of-d-separation-models-qmdag_semigraphoid_certificate)), and records no derivation. A trace of the rule instances that set each target cell would make every closure step a human-checkable proof, the analogue of the LP's Farkas certificate ([B.6](#b6-relation-to-the-closure-and-what-is-specific-to-the-lp), which also records what an exact version of that certificate would take).

* **The conjecture behind counting by class.** Does a QC gap of $G$ imply a QC gap of every structure classically equivalent to $G$? The upward half is proven ([0.6](#06-quantum-observational-equivalence-and-the-two-metagraphs)): a classically equivalent structure with more edges or facets has the larger quantum set. The downward half is open, and no four-node example tests it: after the census no classical class contains both a proven and an unproven quantum class. A proof would let the gap census be counted over classical classes, as the inequality-constraint results are; a counterexample would be a classically equivalent pair of which exactly one has a gap.

  The quantum equivalence classes of the census are themselves only an upper bound on the true number of quantum classes ([0.6](#06-quantum-observational-equivalence-and-the-two-metagraphs)), and refining them is the natural next step. The classes are the strongly connected components of the quantum metagraph, a graph whose edges are proven containments of model sets ([0.6](#06-quantum-observational-equivalence-and-the-two-metagraphs)): a quantum-valid rule the metagraph lacks would merge open classes, and merging an open class into a proven one would prove it.

### 10.3 Variants tried, settled or left out

This subsection lists four variants of the Fritz piggyback or of its certificate that the search does not use, each with what is known about it.

* **Extra deletions certified by the same hypotheses.** Let $s$ be the predicted node and $\mathbf X$ the predicting set. The hypotheses of [7.3](#73-the-hypotheses-and-how-perfect-prediction-enters-qmdag_semigraphoid_model-semigraphoidadd_functional_dependence) do not mention which node a deletion concerns, so the same closure can certify deleting a parent $p$ of any non-predictor node $t$ through the per-edge target $\langle t,p\mid\mathrm{Pa}'(t)\setminus p\rangle,$ where $\mathrm{Pa}'(t)$ are the parents of $t$ in the current candidate.

  Deleting an edge enlarges $\mathcal I(G_1),$ the set of observable d-separations of the candidate $G_1$ ([7.3](#73-the-hypotheses-and-how-perfect-prediction-enters-qmdag_semigraphoid_model-semigraphoidadd_functional_dependence)). The hypotheses would therefore be recomputed after each deletion and the loop run to a fixed point, and the final candidate verified as a whole with [7.4](#74-the-two-target-sets-as-containment-of-d-separation-models-qmdag_semigraphoid_certificate), whose soundness proof (Theorem [7.7](#77-theorem-semigraphoid-closure-certificate)) already allows further parents deleted from non-predictor nodes.

  The loop was tried, with the LP as the engine: in a census run, extra deletions certified hundreds of candidate steps and decided no input. The code therefore does not include it.

  Two constraints any implementation must respect:

  * deletions never touch a predictor (it must keep seeing $`\mathrm{common}(s)`$);
  * deletions never remove the last channel from the predictors to the predicted node (a channel is a facet shared with a predictor or an edge from one, [6.1](#61-the-mechanism)). Without it the hypotheses $H(s\mid\mathbf X)=0$ and $s\perp\mathbf X$ (an observable d-separation of the resulting $G_1$) are jointly satisfiable only by a constant $s,$ the certificate would certify a vacuous statement, and the quantum direction could not be carried out.

* **Using both directions of the split predictor's correlation: settled, in the negative.** A split predictor $x$ stays in the output of a Fritz step: it is split into $x$ and a copy $x'$ with the same parents and the same children, and the copy is the predictor, removed as any predictor is ([6.4](#64-node-splitting-split-and-unsplit-predictors-split-and-unsplit-targets-qmdagsplit_node)). The copy $x'$ of a split predictor can be made perfectly correlated with the target in the quantum construction ([6.4](#64-node-splitting-split-and-unsplit-predictors-split-and-unsplit-targets-qmdagsplit_node)), $H(s\mid x')=0$ and $H(x'\mid s)=0,$ but the piggyback assumes only the first of the two (the copy determines the target) and treats $x'$ as a predictor that keeps all of its parents.

  The second equality suggests a stronger surgery: restrict $x'$ to $K=\mathrm{common}(s)$ as well before it is marginalized, so that fewer of its parents are passed on to the children of $x.$ That is deleting arrows into both of two equal nodes, the shape of Corollary 5 in the KPC draft. The classical direction of this stronger step is true, and every test accepts it: d-separation with each of the two nodes in turn as the predictor of the other, and the closure and the LP once they are given both equalities (in the census the closure certifies the candidate with both nodes restricted for all 2,491 certified pairs of predictor and target).

  But the step is **not a piggyback**, because its quantum direction fails. The structure 0→3, 1→2; Q{0,1}, Q{0,3}, Q{1,3} is classically equivalent to a latent-free DAG, so it has no gap. With node 1 predicting a copy of node 0, the stronger surgery outputs 0→3, 1→2, 4→3; C{0,1,2,4}, Q{0,1}, Q{0,3}, Q{1,3}, and fixing nodes 1 and 4 there leaves 0→3; C{0,2}, Q{0,3}, which is the seed [`QG_Instrumental_C`](../known_QC_gaps.py#L41).

  The gap of the output is witnessed by a distribution in which node 2, the child of the predictor, reads the source it inherited while the copy of the target is constant. No quantum model of the input produces that distribution with $x'$ equal to $s$: to serve node 2, $x'$ would have to pass the source on, and then it is not equal to a constant. The equality can be arranged for a strategy one chooses, not for an arbitrary gap of the output.

  The code's surgery outputs the same structure with the further facet C{1,2,3}, which reaches no known gap, and over the 20 four-node classes without a gap its 42,962 Fritz steps produce 4,023 distinct outputs of which none reaches one.

  Nor would anything be gained. With the copy left alone, the second equality certifies exactly the pairs the first does (2,491 of 4,482), and from the open classes the stronger surgery yields 80 new outputs, none of which reaches a known gap. For a childless predictor the question does not arise: the copy is deleted and the two surgeries coincide. The example is worked through with diagrams in the appendix of [kpc_figure4_vs_code.md](kpc_figure4_vs_code.md); the numbers are printed by [`Special Applications/kpc_figure4_checks.py`](../Special%20Applications/kpc_figure4_checks.py) with `--census`.

* **Joint steps certified beyond d-separation.** A joint step restricts several targets of one predictor set at once, and joint steps are admitted on d-separation certificates only ([8.5](#85-joint-steps-and-why-not-a-second-fritz-depth)). A joint closure (one candidate restricting every target of the set, one sufficiency hypothesis per target and the `markov` target set) is sound by Theorem [7.7](#77-theorem-semigraphoid-closure-certificate) read for a set of targets. It was tried, as a joint LP, and never decided a four-node input; the code does not run it.

  Whether the `relabel` certificates of two targets can be composed, so that a set with closure-certified members could be admitted without the joint certificate, is open.

* **The prediction-free edge-deletion piggyback** (KPC, Corollary 3): the same closure without the perfect-prediction hypothesis certifies deleting edges justified by the new graph's independences alone. The code can check it ([`QmDAG._semigraphoid_certificate(..., predicted=())`](../quantum_mDAG.py#L615)), but it is not exposed as a trick, that is, as a piggyback the search applies ([0.7](#07-the-census-the-seeds-and-the-names-of-the-tricks)).

---

## Appendix A. The piggybacks in reverse

### A.0 Why the search runs forward

Every piggyback is an implication, "a QC gap in $G'$ implies a QC gap in $G$", and the search of Section [9](#9-search-certificates-and-the-four-node-census-qc_gap_searchpy-special-applicationsproving_qc_gapspy) applies it forward: it starts from a structure $G$ to be tested, maps it down through piggybacks until a known gap is reached, and so proves exactly the structures it is given. Run in reverse, the same implications generate gaps. The **inverse image** of a known gap $G'$ under a piggyback is the set of structures that the piggyback maps to it. Every structure in the inverse image is a gap, and the closure of the seeds under all inverse images is the set of every structure the toolkit can prove. The two directions answer different questions.

The question this work asks is: *which causal structures with quantum latents admit correlations that no classical model of the same structure admits?* Its natural input set is the list of all-quantum structures on $n$ visible nodes, and the forward search answers it input by input, with a certificate for each. In that setting the reverse closure buys nothing: the all-quantum four-node structures are a finite list of 996, each is tested directly, and a reverse closure restricted to four all-quantum nodes would reproduce the elementary proofs by a longer route.

The other question, *which hybrid structures, with some sources classical and some quantum, admit a QC gap?*, has a much larger input set ([9.5](#95-the-known-gap-database-the-search-establishes): already 8756 recolourings of the all-quantum inputs, and more structures whose quantum facets lie inside classical ones), most of it structures nobody asks about. There the reverse direction is the efficient one: the inverse images of the seeds produce exactly the hybrid structures that are gaps, classical facets arising naturally where a reversed marginalization or degradation puts them, without listing the inputs first.

The code is designed for the first question; this appendix records how each piggyback inverts, so that the second can be attacked when wanted. [A.1](#a1-degradation-in-reverse-quantumification) to [A.7](#a7-node-splitting-in-reverse-merging) take the piggybacks one at a time, each with its forward map, its inverse and examples; [A.8](#a8-what-a-reverse-closure-would-compute) says what the closure under all the inverses would compute.

Throughout, $G'$ is the known gap and $G$ a structure in the inverse image, so that $G\mapsto G'$ is the forward map of the section cited. Inverse images are finite once the number of visible nodes of $G$ is bounded, and every member is a gap by the forward theorem, with no new proof needed.

### A.1 Degradation in reverse: quantumification

Forward ([0.4](#04-the-degradation-piggyback-quantum-source-to-classical-source)): make some quantum facets of $G$ classical.

Inverse: upgrade any set of classical facets of $G'$ to quantum, or add a quantum facet inside an existing classical facet (which is the same as upgrading part of it). The inverse image of $G'$ with $c$ classical facets of sizes $m_1,\dots,m_c$ has at most $\prod_i 2^{2^{m_i}}$ members, in practice a few dozen for four nodes.

Example ([0.4](#04-the-degradation-piggyback-quantum-source-to-classical-source), with 3′ relabelled 3): from G(3) = 0→1→2; Q{0,1}, Q{0,2}, C{0,1,3}, C{0,1,2}, an intermediate structure of the example of [6.10](#610-example-a-split-predictor-and-a-split-target-worked-out-in-full), the inverse image contains all of

* 0→1→2; Q{0,1}, Q{0,2}, Q{1,2}, C{0,1,3}, C{0,1,2};
* 0→1→2; Q{0,2}, Q{1,2}, Q{0,1,3}, C{0,1,2};
* 0→1→2; C{0,1,3}, Q{0,1,2};
* up to 0→1→2; Q{0,1,3}, Q{0,1,2}.

Every member of the inverse image is a gap. This is the one inverse that never changes a node or an edge, and the reason the seeds are stored in weakest form (only the structures with as many facets classical as the gap allows are kept, [0.4](#04-the-degradation-piggyback-quantum-source-to-classical-source)).

### A.2 Point distribution in reverse: node insertion

Forward (Section [2](#2-point-distribution-pd)): remove a node $x$ with its edges and its facet memberships.

Inverse: insert a node $x$ into $G'$ with any set of visible parents, any set of visible children not creating a cycle, and membership in any facets, new or existing, classical or quantum. The inverse image is large (every placement of one node) but finite, and every member is a gap because fixing $x$ to a point distribution gives $G'$ back.

Strictly hybrid example: insert into [`QG_Instrumental_Edge`](../known_QC_gaps.py#L39) = 0→1→2; Q{1,2} a node 3 that shares a classical facet with the treatment and the outcome, 0→1→2; Q{1,2}, C{1,2,3}. This is a gap. It is not all quantum, it is not a recolouring of an all-quantum structure (its quantum facet lies inside its classical one), and the rule [`QmDAG.lift`](../quantum_mDAG.py#L235) does not apply to it, since two members of C{1,2,3} have a visible parent ([0.5](#05-when-a-classical-source-is-as-good-as-a-quantum-one)). The reverse search would list it from the seed; the forward census never meets it.

### A.3 Marginalization in reverse: node insertion along edges and facets

Forward (Section [3](#3-marginalization-naive-and-with-teleportation-naive_marginalization-teleportation_marginalization)): remove $v,$ relay its visible parents to its children, turn each facet $F\ni v$ into $F\setminus v$ plus a classical facet over $(F\setminus v)$ and the children of $v,$ add a classical facet over the children (the private randomness of $v$), and, with teleportation, extend each quantum facet $F\ni v$ to $(F\setminus v)\cup T$ for the children $T$ that share a quantum facet with $v.$

Inverse: choose in $G'$

* a set of nodes to become the children of $v;$
* a set of their parents to become the parents of $v$ instead (the edges $p\to c$ are then explained as $p\to v\to c$ and may be removed);
* for each classical facet $C$ of $G'$ that contains the children, an explanation as the relay of a facet $F\ni v$ with $F\setminus v=C\setminus T.$

A quantum facet of $G'$ over $(F\setminus v)\cup T$ may be explained as the teleportation of a facet $F\ni v,$ which requires $v$ to share a quantum facet with each member of $T.$ Each explanation is one member of the inverse image.

Strictly hybrid examples.

* The reverse of the first step of the Section [3](#3-marginalization-naive-and-with-teleportation-naive_marginalization-teleportation_marginalization) example: [`QG_Instrumental_C`](../known_QC_gaps.py#L41) = 1→2; C{0,1}, Q{1,2} is the naive marginalization of $v$ in $v$→1→2; C{0,$`v`$}, Q{1,2}: the classical facet C{0,1} of the seed is the relay of C{0,$`v`$} to $v$'s child 1.
* The reverse of the last step of the split-predictor example of [6.10](#610-example-a-split-predictor-and-a-split-target-worked-out-in-full): [`QG_Instrumental_C`](../known_QC_gaps.py#L41) written as 1→2; Q{1,2}, C{1,3′} is the teleportation marginalization of 0 in G(3) = 0→1→2; Q{0,1}, Q{0,2}, C{0,1,3′}, C{0,1,2}: the quantum facet Q{1,2} of the seed is explained as Q{0,2} teleported to 1 through Q{0,1}, and the classical facets as relays of C{0,1,3′} and of 0's private randomness.

G(3) has classical facets that strictly contain quantum ones, a shape no recolouring of an all-quantum input has; the forward census meets it only as an intermediate of the chain recorded for one input ([9.5](#95-the-known-gap-database-the-search-establishes)), while the reverse search finds it directly from the seed.

### A.4 Postselection in reverse: un-swapping entanglement

Forward (Section [4](#4-postselection-conditioning)): remove $x,$ whose facets all lie within its latent siblings $S$ plus $x$ (condition 2); every facet containing $x$ loses $x;$ the quantum siblings of $x$ gain a joint quantum facet; the visible children of $x$ lose their edge from $x;$ the visible parents of $x$ are handled by conditions 1 and 2. (Latent siblings are nodes that share a facet, quantum siblings nodes that share a quantum facet, [0.7](#07-the-census-the-seeds-and-the-names-of-the-tricks).)

Inverse: choose in $G'$ a quantum facet $Q\lbrace S_q\rbrace$ to be explained as the swap, and insert $x$ with

* quantum facets $F_i\ni x$ such that $\bigcup_i (F_i\setminus x)=S_q$ (each $F_i\setminus x$ then also appears in $G',$ possibly absorbed into $Q\lbrace S_q\rbrace$);
* optionally, classical facets $F\ni x$ whose $F\setminus x$ appear in $G';$
* optionally, edges from $x$ to nodes of $G';$
* visible parents of $x$ only as conditions 1 and 2 allow.

Strictly hybrid example: [`QG_Instrumental_C`](../known_QC_gaps.py#L41) = 1→2; C{0,1}, Q{1,2} is the postselecting on $x$ of 1→2; C{0,1}, Q{1,$`x`$}, Q{$`x`$,2}: the quantum link between treatment and outcome is explained as entanglement swapped through $x,$ while the classical facet with the instrument stays as it is.

The reverse of the postselection step in the split-predictor example of [6.10](#610-example-a-split-predictor-and-a-split-target-worked-out-in-full), G(2) → G(3), is another: G(2) = 0→1→2; Q{0,2}, Q{0,3}, Q{1,3}, C{0,1,3,3′}, C{0,1,2} explains the Q{0,1} of G(3) as a swap through 3 and the C{0,1,3′} as the relic of C{0,1,3,3′}.

### A.5 Node stitching in reverse: interruption

Forward (Section [5](#5-node-stitching-node_stitching)): remove the exogenous node $x$ and give its visible children to the sink $y.$

Inverse, which is the operation that deserves the name *interruption*: choose a node $y$ of $G'$ and a subset of its visible children, cut those children off from $y$ and give them to a new exogenous node $x$ (no visible parents, no facets), with $y$ not a descendant of $x,$ which is automatic. The inverse image is one member per node and subset of its children, so it is tiny and exact.

Examples:

* [`QG_Instrumental_Edge`](../known_QC_gaps.py#L39) = 0→1→2; Q{1,2}, interrupted at $y=1$ with the child 2 given to $x,$ is 0→1, $x$→2; Q{1,2}, which is [`QG_Bell_Edge_Edge`](../known_QC_gaps.py#L83);
* the same interruption of [`QG_Instrumental_EdgeC`](../known_QC_gaps.py#L40) = 0→1→2; C{0,1}, Q{1,2} gives 0→1, $x$→2; C{0,1}, Q{1,2}, which is [`QG_Bell_EdgeC_Edge`](../known_QC_gaps.py#L82);
* and of [`QG_Instrumental_C`](../known_QC_gaps.py#L41) = 1→2; C{0,1}, Q{1,2} gives $x$→2; C{0,1}, Q{1,2}, which is [`QG_Bell_C_Edge`](../known_QC_gaps.py#L80), a strictly hybrid structure.

The three Bell variants that node stitching reaches ([1.1](#11-the-census-over-quantum-classes)) are exactly the interruptions of the three instrumental seeds, found forward because they happen to be inputs; the reverse search would find them, and the hybrid ones, directly.

### A.6 The Fritz piggyback in reverse

Forward (Section [6](#6-the-fritz-piggyback-fritz)): for a predictor set $\mathbf X$ and a predicted node $s,$ restrict $s$ to $\mathrm{common}(s),$ whose facets become classical for $s,$ and remove the predictors (unsplit) or keep them (split: childless untouched, childful split and the copy marginalized). Here $\mathrm{common}(s)$ is the set of parents of $s$ that the predictors see ([8.2](#82-what-a-predictor-sees-and-the-deletion-it-dictates-qmdagfritz_deletion)).

Inverse, unsplit target, in three choices and a check:

1. Choose in $G'$ a node $s$ all of whose latent parents are classical facets (the former $\mathrm{common}(s),$ which in $G$ may be quantum again, with the classical copy removed).
2. Choose the predictor set, either present in $G'$ (split) or to be inserted (unsplit: a new node $X$ that shares with $s$ at least one of those facets and may have any other parents and facets, the facets of $X$ that were relayed to its children by the marginalization being un-relayed as in [A.3](#a3-marginalization-in-reverse-node-insertion-along-edges-and-facets)).
3. Choose a set $D$ of further parents to give $s$: visible parents or facets, any that the predictor cannot see, and then *all* of those, since the forward deletion is maximal (Section [8](#8-the-fritz-search-procedure-qmdagfritz_steps-qc_gap_searchfritz_tricks)): every parent of $s$ in the reconstructed $G$ that the predictor cannot see must lie in $D.$
4. Check the certificate on the reconstructed $G,$ by d-separation ([6.6](#66-theorem-d-separation-certificate)) or by the semigraphoid closure ([7.7](#77-theorem-semigraphoid-closure-certificate)).

Unlike the inverses above, this one does not come for free, because the certificate is a property of $G$ as a whole, of how $\mathbf X,$ $\mathrm{common}(s)$ and $D$ relate. The members of the inverse image are the reconstructions that pass.

Example ([7.8](#78-further-examples-from-the-census) reversed): [`QG_Bell_C_Edge`](../known_QC_gaps.py#L80) written as 1→2; C{0,3}, Q{0,2} has $s=3$ with the single classical facet C{0,3} and the childless predictor $X=0$ present. Two reconstructions:

* giving 3 the parents 2→3 and Q{1,3} and making C{0,3} quantum again reconstructs 1→2→3; Q{0,2}, Q{0,3}, Q{1,3} (example 1 of [7.8](#78-further-examples-from-the-census)), and the closure certifies it;
* giving 3 the parents 1→3 and 2→3 instead reconstructs 1→2, 1→3, 2→3; Q{0,2}, Q{0,3}, the KPC structure itself (example 3), also certified.

Both are members of the KPC class, which the census proves forward, and both are found from the seed in reverse.

A split target inverts in two moves. The forward step splits $s$ into $s$ and $s'$ and restricts $s',$ so $G'$ has one more visible node than the structure whose gap is at stake, and the inverse must first reconstruct a structure $G_+$ with $n+1$ visible nodes in which $s'$ has its restricted parents and $s$ its full ones, then merge $s$ and $s'$ ([A.7](#a7-node-splitting-in-reverse-merging)). The reverse search for four-node gaps therefore passes through five-node structures, exactly as the forward search does.

### A.7 Node splitting in reverse: merging

Forward ([6.4](#64-node-splitting-split-and-unsplit-predictors-split-and-unsplit-targets-qmdagsplit_node)): replace $s$ by two nodes with the same parents, the same children and a shared classical pair facet.

Inverse: merge two nodes of $G'$ that have the same parents and the same children into one (their shared pair facet, if explicit, disappears).

Node splitting is never run on its own forward and would not be in reverse either; merging is the second move of the reversed split target ([A.6](#a6-the-fritz-piggyback-in-reverse)) and nothing else.

### A.8 What a reverse closure would compute

Starting from the weakest seeds and closing under [A.1](#a1-degradation-in-reverse-quantumification) to [A.7](#a7-node-splitting-in-reverse-merging) with the number of visible nodes bounded by $n$ (and $n+1$ for the intermediate of [A.6](#a6-the-fritz-piggyback-in-reverse)) gives every structure on at most $n$ nodes that the toolkit can prove, hybrid structures included, each with a certificate read off the inverse steps. The inverse images of [A.1](#a1-degradation-in-reverse-quantumification), [A.2](#a2-point-distribution-in-reverse-node-insertion) and [A.3](#a3-marginalization-in-reverse-node-insertion-along-edges-and-facets) are the large ones; [A.4](#a4-postselection-in-reverse-un-swapping-entanglement) to [A.7](#a7-node-splitting-in-reverse-merging) are small. Because the degradation inverse ([A.1](#a1-degradation-in-reverse-quantumification)) is cheap and closes upward, the closure can again be stored in weakest form only.

Two things the reverse closure does not change, as long as it is restricted to the steps the forward census takes, with the Fritz piggyback at depth one ([9.1](#91-the-ten-stages)): the set of all-quantum four-node gaps proven, which is the 372 classes of [1.1](#11-the-census-over-quantum-classes) by construction, and the 46 classes left open. The reason is that forward and reverse use the same implications: an input is proven forward iff it lies in the reverse closure under the same steps.

This agrees with the statement of [9.1](#91-the-ten-stages) that the counts of [1.1](#11-the-census-over-quantum-classes) are lower bounds for what the tricks can prove. That statement is about chains with more Fritz steps than the census tries; such chains could add proofs, forward and in reverse alike.

---

## Appendix B. The entropic LP as an alternative certificate engine (`entropic_lp.py`, `QmDAG._entropic_certificate`)

### B.0 Why the LP is not the default engine

The certificate beyond d-separation of Section [7](#7-the-semigraphoid-closure-certificate-for-the-fritz-piggyback-semigraphoidpy-qmdag_semigraphoid_certificate) has two engines. The default is the semigraphoid closure. The other, the subject of this appendix, follows KPC's entropic reading of their Lemma 1: a linear program (LP) over entropy vectors (`--engine lp`; `--engine both` runs both and reports every disagreement).

The LP is stronger in principle than the semigraphoid closure ([7.9](#79-strength-agreement-with-the-lp-and-cost)). It is not the default engine for three reasons:

* it needs the commercial solver Mosek;
* it costs about a tenth of a second per candidate against a millisecond;
* on the four-node census it proves exactly the same steps.

The search therefore runs the closure by default, and everything that has to do with entropies is collected here: the LP as a proof system ([B.1](#b1-entropy-vectors-as-the-proof-system-entropic_lppy)), its form of the hypotheses and targets ([B.2](#b2-the-hypotheses-and-the-target-sets-as-lp-rows-qmdag_entropic_certificate)), its theorem ([B.3](#b3-theorem-entropic-certificate)), its size and time ([B.4](#b4-lp-size-and-time)), the KPC example ([B.5](#b5-the-kpc-example-and-the-decisive-diagnostic)), and its relation to the closure ([B.6](#b6-relation-to-the-closure-and-what-is-specific-to-the-lp)).

Transitions certified by this engine carry the certificate label `entropic`. [`ENTROPIC_STATS`](../quantum_mDAG.py#L69), which tallies the attempts of either engine, and the option names `use_lp`, `lp_only` and `lp_markov_target` are named after the LP, although they serve whichever engine runs.

### B.1 Entropy vectors as the proof system (`entropic_lp.py`)

For $n$ random variables the entropy vector $h\in\mathbb R^{2^n-1}$ lists $H(S)$ for every nonempty $S.$ Every entropy vector satisfies the **elemental Shannon inequalities** $H(i\mid[n]\setminus i)\ge0$ and $I(i{:}j\mid K)\ge0$ for $i\lt j$ and $K\subseteq[n]\setminus\lbrace i,j\rbrace,$ which number $n+\binom n2 2^{n-2}$ ([`elemental_inequalities(n)`](../entropic_lp.py#L100); the $n=3$ matrix is checked against the Mathematica notebook of KPC, the row count for $n\le8$). The vectors that satisfy them form the **Shannon cone**.

A conditional independence is the vanishing of a conditional mutual information, a functional dependence the vanishing of a conditional entropy; both are nonnegative on the cone, so each hypothesis is one inequality "$`\le0`$".

The local Markov property of a DAG is one conditional mutual information per node, $I(v{:}\mathrm{nondesc}(v)\setminus\mathrm{pa}(v)\mid\mathrm{pa}(v))=0$ ([`local_markov_rows`](../entropic_lp.py#L141)). The Shannon cone with these hypotheses implies exactly the d-separations of the DAG: every d-separation follows from the local Markov statements by the semigraphoid rules ([7.2](#72-conditional-independence-as-the-proof-system-semigraphoids-and-the-exchange-rule-semigraphoidpy)), which are Shannon-derivable, and nothing beyond the d-separations can follow because classical models realise every non-d-separated dependence (F2). This is tested on random DAGs.

A target functional $t$ is **implied** when the LP $\lbrace\text{Shannon}\ge0,\ \text{hypotheses}\le0,\ t\ge1\rbrace$ is infeasible (the cone is scale invariant). Geometrically the question is **cone inclusion**: the hypotheses cut out a polyhedral cone $K,$ each target is a halfspace, and the certificate asks whether $K$ lies in the intersection of the target halfspaces, that is, inside the cone of entropy vectors Markov to the target graph. Both cones are given by inequalities, and inclusion of one inequality description in another needs no conversion to extreme rays (as cdd, lrs or PANDA would do; that is exponential and unnecessary).

The inclusion needs one LP per target set, not one per target: every target is a conditional mutual information, nonnegative on the Shannon cone, and a sum of nonnegative quantities vanishes iff each term does, so all targets are implied iff their **sum** is implied ([`EntropicLP.implies_all`](../entropic_lp.py#L379), [`sum_rows`](../entropic_lp.py#L74)). The Farkas certificate of the summed row certifies every target at once. Solving one LP in place of $n$ per target set is where the time of an LP stage goes, so this is roughly an $n$-fold saving.

Infeasibility is decided with the Mosek Optimizer API ([`EntropicLP`](../entropic_lp.py#L216), interior point); the dual ray is a Farkas certificate, a nonnegative combination of elemental inequalities and hypotheses that reproduces $t,$ available through [`EntropicLP.farkas_certificate`](../entropic_lp.py#L320). Implications proven this way are valid for every distribution, with or without full support, because they use only Shannon inequalities. The method is incomplete in the other direction: a feasible LP does not exhibit a distribution, only a vector in the Shannon cone.

### B.2 The hypotheses and the target sets as LP rows (`QmDAG._entropic_certificate`)

The variables, hypotheses and target sets are those of [7.3](#73-the-hypotheses-and-how-perfect-prediction-enters-qmdag_semigraphoid_model-semigraphoidadd_functional_dependence) and [7.4](#74-the-two-target-sets-as-containment-of-d-separation-models-qmdag_semigraphoid_certificate), each written as a row. As there, $s$ is the predicted node, $\mathbf X$ the predictor set, $G_1$ the candidate (the input with $s$ restricted to the parents it keeps, [6.1](#61-the-mechanism)), $L$ the single facet that $s$ keeps when it keeps one parent only, and $G''$ the graph $G_1$ with $L$ deleted and $s$ made a parent of every other child of $L$ ([7.4](#74-the-two-target-sets-as-containment-of-d-separation-models-qmdag_semigraphoid_certificate)).

Hypotheses:

* Shannon;
* the local Markov rows of $G$ (one per node, [B.1](#b1-entropy-vectors-as-the-proof-system-entropic_lppy); their closure is the d-separation model of [7.3](#73-the-hypotheses-and-how-perfect-prediction-enters-qmdag_semigraphoid_model-semigraphoidadd_functional_dependence));
* the perfect prediction as the single row $H(s\mid\mathbf X)\le0,$ in full, both the sufficiency half that the closure uses and the determinism half it drops ([7.3](#73-the-hypotheses-and-how-perfect-prediction-enters-qmdag_semigraphoid_model-semigraphoidadd_functional_dependence));
* $\mathcal I(G_1),$ the elementary observable d-separations $I(x{:}y\mid Z)=0$ of the candidate over its visible nodes ([`observable_dseparation_rows`](../entropic_lp.py#L159), built from the same bit-parallel enumerator as the closure's models).

Target sets:

* `relabel` is the local Markov rows of $G''$ over $G$'s variables other than $L;$
* `markov` is the local Markov rows of $G_1$ over $G$'s own latents.

Each is tried as one LP on its summed row ([B.1](#b1-entropy-vectors-as-the-proof-system-entropic_lppy)), `relabel` first and `markov` only when `relabel` is inapplicable or fails, exactly as in [7.4](#74-the-two-target-sets-as-containment-of-d-separation-models-qmdag_semigraphoid_certificate).

The justification of `relabel` ([7.6](#76-why-relabel-is-justified)) is unchanged, with "the closure contains the target" read as "the LP implies the target": step 2 becomes *the LP shows that every entropy vector satisfying the hypotheses has each target conditional mutual information equal to zero, and a vanishing conditional mutual information is a conditional independence*.

Where [7.6](#76-why-relabel-is-justified) derives the `markov` row of $s$ from a d-separation certificate by contraction, the LP derives it entropically: with $I(\mathbf X{:}\mathrm{others}\mid\mathrm{common})\le0$ from Markov($G$), $H(s\mid\mathbf X)\le0$ and the Markov row of $s,$ $H(s\mid\mathrm{common})\le I(s{:}\mathbf X\mid\mathrm{common})\le I(\mathrm{others}{:}\mathbf X\mid\mathrm{common})+I(s{:}\mathbf X\mid\mathrm{common},\mathrm{others})=0.$

### B.3 Theorem (entropic certificate)

**Claim.** Let $G_1$ be a candidate as in [6.1](#61-the-mechanism), possibly with further parents deleted from any non-predictor node, and $G'$ the output with the predictors removed as in [6.6](#66-theorem-d-separation-certificate). If the LP of [B.2](#b2-the-hypotheses-and-the-target-sets-as-lp-rows-qmdag_entropic_certificate) implies the `markov` targets, or the single-parent condition holds and it implies the `relabel` targets, then a QC gap in $G'$ implies a QC gap in $G.$

**Proof.** *The quantum direction* is as in [6.6](#66-theorem-d-separation-certificate): $P\in\mathcal Q(G)$ with $H(s\mid\mathbf X)=0;$ since the constructed strategy is a strategy for $G_1$ with fine-grained predictor outputs, (F2) gives that $P$ satisfies every observable d-separation of $G_1.$

*The classical direction.* Let $M$ be a classical model of $G$ for $P.$ Its joint entropy vector lies in the Shannon cone, satisfies the local Markov equalities of $G$ and the hypotheses $H(s\mid\mathbf X)=0$ and $\mathcal I(G_1).$ The LP implication forces the target quantities to vanish, so the joint of $M$ is Markov to the restricted structure with the predictors present (or to its relabelled version, and then [7.6](#76-why-relabel-is-justified) gives such a model). Removing the predictors as in [6.6](#66-theorem-d-separation-certificate), by deletion or marginalization, gives $P'\in\mathcal C(G').$ $\square$

Several predicted nodes for the same $\mathbf X$ could be handled jointly, with one hypothesis $H(s_i\mid\mathbf X)\le0$ per node, one candidate $G_1$ restricting all of them, and only the `markov` target set. This joint LP was tried and was never decisive in four nodes; the code does not run it ([10.3](#103-variants-tried-settled-or-left-out)).

### B.4 LP size and time

An LP over $n$ variables (visible nodes plus facets, after any splitting) has $2^n-1$ columns and $n+\binom n2 2^{n-2}$ elemental rows; measured solve times on one core are about 0.2, 0.5, 2 and 9 seconds for $n=10,11,12,13,$ and memory stays below a gigabyte up to $n=13.$

The number of variables is not capped. Instead every solve carries a Mosek time limit of 60 seconds ([`EntropicLP(max_time=60.0)`](../entropic_lp.py#L216), `optimizer_max_time`); a solve that hits it is counted ([`entropic_lp.TIMEOUTS`](../entropic_lp.py#L85)) and treated as undecided, which is the conservative answer: the implication is then not claimed. The census report states how many solves timed out (none, [9.3](#93-certificates-beyond-d-separation-attempted)).

On the census the LP stages take 7, 16, 20 and 51 seconds (696 candidates and 688 LP solves, one per candidate: for eight candidates the `relabel` target set does not apply and nothing is solved; a split target adds a variable and costs more per solve) and the whole census 104 seconds, against 17 with the closure ([9.1](#91-the-ten-stages)). The per-structure part of the LP, the Shannon cone with the Markov rows of the structure that carries the splits, is built once per structure and cached, like the closure's d-separation model. The descendants experiment ([1.1](#11-the-census-over-quantum-classes), [8.10](#810-experiment-descendants-as-predictors)) was timed with this engine, where its cost shows: 1280 solves and 176 seconds, against 688 and 107.

### B.5 The KPC example and the "decisive" diagnostic

For the KPC structure of [7.5](#75-worked-example-the-kpc-structure) the LP reads: from Markov($G$), $H(E\mid F)=0$ and $E\perp CD,$ the `relabel` targets, Markov($G''$) with $E$ a root and a parent of $F,$ are implied, and the Farkas multipliers of the summed target row are the certificate.

The independences marked "decisive" in the census examples of [7.8](#78-further-examples-from-the-census) were found with this engine, by re-solving with the prediction and one observable independence of the candidate at a time; the closure gives the same answers, as it must on these candidates ([7.9](#79-strength-agreement-with-the-lp-and-cost)).

The model $E=A\oplus B,$ $D=C\oplus B,$ $F=(A,B)$ of [7.6](#76-why-relabel-is-justified) shows that a feasible LP for the `markov` targets is a true negative there.

### B.6 Relation to the closure, and what is specific to the LP

The two engines share the hypotheses, the target sets and Theorems [7.7](#77-theorem-semigraphoid-closure-certificate) and [B.3](#b3-theorem-entropic-certificate), and differ in the inference: the four semigraphoid rules against the full Shannon cone, and the sufficiency half of the perfect prediction against all of $H(s\mid\mathbf X)=0$ ([7.3](#73-the-hypotheses-and-how-perfect-prediction-enters-qmdag_semigraphoid_model-semigraphoidadd_functional_dependence)). Every semigraphoid derivation is a Shannon derivation, so the LP certifies everything the closure does; the converse can fail ([7.9](#79-strength-agreement-with-the-lp-and-cost)) and on the census does not: zero disagreements on 696 candidates (`--engine both`). What only the LP offers: a human-checkable certificate, the Farkas multipliers of [B.1](#b1-entropy-vectors-as-the-proof-system-entropic_lppy), which are floating point; rationalising them and re-verifying the combination exactly is cheap and would make every entropic step an exact proof. Adding non-Shannon inequalities (Zhang and Yeung, and their successors) to the cone would make the LP stronger still, at a cost in rows that grows quickly with $n;$ nothing in the census asks for it.

---

## Appendix C. Five nodes: the two metagraphs and their counts

**How the nodes are enumerated** ([`canonical_forms.py`](../canonical_forms.py)). An mDAG's unlabelled id is the minimum of its id over the $n!$ relabellings; the complex occupies the high bits, so the minimum is the canonical complex first and then the smallest directed structure over the relabellings that fix it. Two tables hold the integer of every labelled directed structure and of every labelled complex under every permutation (on five nodes 1024 × 120 and 6894 × 120 entries, compiled with numba), so a canonical id is two table lookups, computed on demand with a memo and never stored for all labelled mDAGs. The nodes are enumerated by orbits: for each directed structure up to relabelling, the complexes up to its automorphism group (read off its table row), one representative per orbit, so no non-canonical mDAG is ever built. The structural dominances are generated from each node's lower neighbours (drop one edge; drop one maximal face, whose uncovered sub-faces of two or more nodes become facets) and canonicalised, which gives exactly the former all-pairs edge set. The metagraph is held as sparse boolean adjacency matrices (`scipy.sparse`, one row per node): the strongly connected components, the reachability searches and the topological order of the structural dominances all run on the compressed-row arrays, which on four nodes finds the components four to six times faster than the graph library used before and on five nodes is what makes the graph fit in memory at all. An mDAG object holds only its two parts and derives everything else on demand, and in [`compact`](../metagraph_advanced.py#L161) mode (the default from five nodes) the metagraph releases each node's cached attributes once its edges are read off and skips the CI filter on the classical dominances, a pruning that never changes the components. On five nodes there are 1,718,596 mDAGs up to relabelling (302 directed structures, 6894 complexes; Burnside's count over the 29,281 labelled DAGs confirms the orbit count) with 18,311,253 structural dominances, and both five-node metagraphs are built in about five minutes within 3.5 GB (the edge collections are int32 index arrays, the component labels an int32 array):

| five nodes | |
|---|---|
| mDAGs up to relabelling | 1,718,596 |
| structural dominances / face-splitting edges / HLP edges, pre-measurement rule / classical rule | 18,311,253 / 197,592 / 139,144 / 301,487 |
| classical equivalence classes (containing a latent-free DAG, and the mDAGs in them) | 490,800 (142 classes, 708,635 mDAGs) |
| quantum equivalence classes (containing a latent-free DAG, and the mDAGs in them) | 583,674 (142 classes, 707,970 mDAGs) |
| classical classes not provably algebraic, and the mDAGs in them | 490,658 classes, 1,009,961 mDAGs |
| quantum classes with no member classically equivalent to a latent-free DAG (the inputs of a five-node census) | 583,531 |
| singleton classical classes / the largest (the class of the complete DAG) | 413,698 / 562,155 |

The last count but one is 583,674 minus 143, not minus 142. The 708,635 mDAGs classically equivalent to a latent-free DAG form 143 quantum classes: the 142 that contain a latent-free DAG (707,970 mDAGs) and one more, of 665 mDAGs, that contains none. On four nodes the two counts coincide (20 and 20).

The 1,009,961 structures not provably algebraic are the candidates for inequality constraints counted in the repository's earlier five-node work; the 490,658 classical classes they form are proven upper bounds on the number of classes, not the classes themselves.

---

## References

* T. Fritz, *Beyond Bell's theorem: correlation scenarios*, New J. Phys. 14, 103001 (2012); *Beyond Bell's theorem II: scenarios with arbitrary causal structure*, Commun. Math. Phys. 341, 391 (2016). The triangle argument and the perfect-prediction mechanism of Section [6](#6-the-fritz-piggyback-fritz).
* R. J. Evans, *Graphs for margins of Bayesian networks*, Scand. J. Stat. 43, 625 (2016). Latent projection (Section [3](#3-marginalization-naive-and-with-teleportation-naive_marginalization-teleportation_marginalization)) and mDAGs.
* J. Henson, R. Lal and M. F. Pusey, *Theory-independent limits on correlations from generalized Bayesian networks*, New J. Phys. 16, 113043 (2014). Observable d-separation holds in every generalised probabilistic theory (F2).
* R. Chaves, C. Majenz and D. Gross, *Information-theoretic implications of quantum causal structures*, Nat. Commun. 6, 5766 (2015). Coexisting sets and the entropic treatment of quantum causal structures (F2, [Appendix B](#appendix-b-the-entropic-lp-as-an-alternative-certificate-engine-entropic_lppy-qmdag_entropic_certificate)).
* T. Verma and J. Pearl, *Causal networks: semantics and expressiveness*, Proc. UAI (1988). Local Markov property, d-separation and the semigraphoid axioms ([7.2](#72-conditional-independence-as-the-proof-system-semigraphoids-and-the-exchange-rule-semigraphoidpy)).
* D. Geiger, T. Verma and J. Pearl, *Identifying independence in Bayesian networks*, Networks 20, 507 (1990). The d-separation model is the semigraphoid closure of the local Markov statements ([7.2](#72-conditional-independence-as-the-proof-system-semigraphoids-and-the-exchange-rule-semigraphoidpy)).
* M. Studený, *Probabilistic Conditional Independence Structures*, Springer (2005). Semigraphoids, elementary triplets and the exchange rule; the incompleteness of the semigraphoid rules ([7.2](#72-conditional-independence-as-the-proof-system-semigraphoids-and-the-exchange-rule-semigraphoidpy), [7.9](#79-strength-agreement-with-the-lp-and-cost)).
* F. Matúš, *Ascending and descending conditional independence relations*, Trans. 11th Prague Conf. (1992). The elementary characterisation of semigraphoids ([7.2](#72-conditional-independence-as-the-proof-system-semigraphoids-and-the-exchange-rule-semigraphoidpy)).
* B. Bonet, *Instrumentality tests revisited*, Proc. UAI (2001). The instrumental inequality (Section [6](#6-the-fritz-piggyback-fritz).4 example).
* R. Khanna, M. F. Pusey and R. Colbeck, draft manuscript and Mathematica notebook `piggyback_check.nb` (KPC). Lemma 1 and its proof by semigraphoid closure, the six-node example of [7.5](#75-worked-example-the-kpc-structure) and the prediction-free variant of Section [10](#10-open-questions).
