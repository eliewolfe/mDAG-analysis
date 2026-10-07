# Piggybacks for quantum-classical gaps

*What each structural transformation in this repository proves, why it is sound, what it contributes to the four-node census, and where it lives in the code.*

**Contents.** 0 Setting · 1 Point distribution · 2 Marginalization · 3 Conditioning · 4 Interruption · 5 Node splitting · 6 The Fritz piggyback · 7 The entropic Fritz piggyback · 8 Search and certificates · 9 Results on the four-node census · 10 Open questions

---

## 0. Setting

### 0.1 Structures and model sets

A **QmDAG** (`quantum_mDAG.QmDAG`) is a causal structure with visible nodes $V=\lbrace 0,\dots,n-1\rbrace$, a directed structure on $V$, and two simplicial complexes of **latent facets**: classical facets $\mathcal C$ and quantum facets $\mathcal Q$. A facet $F\subseteq V$ stands for one exogenous latent, a classical random variable or a quantum state, whose children are exactly the nodes of $F$. Every visible node additionally has implicit private randomness. A classical facet contained in a quantum facet is redundant and is dropped by the constructor.

For a structure $G$ write $\mathcal C(G)$ for the set of distributions over $V$ realisable with all latents classical (a Bayesian network with latent variables), and $\mathcal Q(G)$ for the set realisable when the quantum facets carry quantum states on which their children perform measurements, classical facets and visible edges remaining classical channels. Always $\mathcal C(G)\subseteq\mathcal Q(G)$. The structure $G$ has a **QC gap** when $\mathcal Q(G)\not\subseteq\mathcal C(G)$.

The **effective DAG** of a QmDAG (`QmDAG.effective_DAG_data`) is the ordinary DAG on the visible nodes, one node per facet and one private-noise node per visible node; `QmDAG.lp_structure` is the same without the noise nodes. A "parent" of a visible node below is a parent in the effective DAG: a visible parent or a facet containing the node.

### 0.2 Piggybacks

A **piggyback** is a map $G\mapsto G'$ together with a proof of

```math
\mathcal Q(G')\not\subseteq\mathcal C(G')\;\Longrightarrow\;\mathcal Q(G)\not\subseteq\mathcal C(G).
```

Every proof in this document has the same two halves.

* **Quantum lift.** Given $P'\in\mathcal Q(G')\setminus\mathcal C(G')$, construct $P\in\mathcal Q(G)$ from which $P'$ is recovered by a fixed operation $\pi$ (a marginal, a conditional, a coarse-graining, or the identity): $\pi(P)=P'$.
* **Classical pull-back.** Show that every classical model of $G$ for a distribution of the kind the lift produces yields a classical model of $G'$ for $\pi(P)$.

Then $P\in\mathcal C(G)$ would give $P'=\pi(P)\in\mathcal C(G')$, a contradiction, so $P\in\mathcal Q(G)\setminus\mathcal C(G)$.

All piggybacks here are **caveat-free**: the hypothesis is a QC gap of $G'$ as a structure, with no side condition on the gap-witnessing distribution. This is what lets them be composed in any order and searched as a closure (Section 8). Structures are identified up to relabelling by `unique_unlabelled_id`, and every count in this document is a count of distinct structures up to relabelling.

### 0.3 Two general facts

* **(F1) Edge monotonicity.** If $G'$ is obtained from $G$ only by deleting visible edges and removing nodes from facets, then $\mathcal Q(G')\subseteq\mathcal Q(G)$ and $\mathcal C(G')\subseteq\mathcal C(G)$: run the same strategy and ignore the extra inputs. For an edge-deleting piggyback the lift is therefore immediate and the whole content is the pull-back.
* **(F2) d-separation.** Among the *visible* variables, every d-separation of $G$ implies the corresponding conditional independence, in classical and in quantum models alike: observable d-separation is theory-independent (Henson, Lal and Pusey). Conversely no other conditional independence among visible variables holds in every classical model, hence none holds in every quantum model either, since $\mathcal C(G)\subseteq\mathcal Q(G)$. Conditional independences that *involve latents* are statements about a joint distribution of visible and latent variables. Such a joint exists for a classical model, where every node of the effective DAG is a random variable, but not in general for a quantum one: a quantum facet is a state, not a variable, and only coexisting sets of nodes have a joint distribution (Chaves, Majenz and Gross). The pull-backs below reason about classical models only, so they may use latent-involving d-separations and the entropies of the effective DAG freely.

### 0.4 Notation for the Fritz piggybacks

The predictor is in general a **set** of visible nodes, written $\mathbf X$; its members are the predictors. $\mathrm{Pa}(\mathbf X)=\bigcup_{x\in\mathbf X}\mathrm{Pa}(x)$. The nodes **seen** by $\mathbf X$ are $\mathrm{Pa}(\mathbf X)\cup\mathbf X$. For a predicted node $s$,

```math
\mathrm{common}(s)=\mathrm{Pa}(s)\cap\big(\mathrm{Pa}(\mathbf X)\cup\mathbf X\big),\qquad
\mathrm{others}(s)=\mathrm{Pa}(s)\setminus\mathrm{common}(s),
```

the latter always containing the private noise of $s$. Structures in examples are written as a list of visible edges and facets, with Q for quantum and C for classical facets, e.g. "0→2, 1→2; Q{0,1}, Q{1,3}, Q{2,3}".

---

## 1. Point distribution (`PD`)

**Map.** Remove one visible node $v$ together with all its edges and its membership in facets: $G'=G-v$ (`QmDAG.fix_to_point_distribution_QmDAG`; trick `pd_trick` in `qc_gap_search.py`).

**Lift.** Given $P'\in\mathcal Q(G')$, let $v$ output a constant and every other node run its $G'$ strategy; $\pi$ is the marginal over $v$. $P\in\mathcal Q(G)$ because the children of $v$ may ignore a constant input.

**Pull-back.** In a classical model of $G$ for a distribution with $v$ constant, substitute the constant into the kernels of $v$'s children; the result is a classical model of $G-v$ for the marginal. $\square$

**Example** (provable only through PD; 221 such inputs). Input: 0→2, 0→3; Q{0,1,2}, Q{0,1,3}, Q{2,3}. Fixing 0 to a point distribution deletes its two edges and removes it from the two triple facets, leaving Q{1,2}, Q{1,3}, Q{2,3}: the triangle `QG_Triangle1`. Most PD proofs look like this: a seed with one more node attached to it.

---

## 2. Marginalization, naive and with teleportation (`naive_marginalization`, `teleportation_marginalization`)

**Map** (`QmDAG.marginalize(node, apply_teleportation)`). Remove $v$; every visible parent of $v$ becomes a parent of every visible child of $v$; every facet containing $v$ loses $v$ and gains all visible children of $v$, as a *classical* facet; a new classical facet over the visible children of $v$ is added (the private randomness of $v$). With teleportation, additionally every quantum facet $F\ni v$ is kept quantum on $(F\setminus v)\cup T$, where $T$ is the set of visible children of $v$ that share some quantum facet with $v$. $\pi$ is the marginal over $v$.

**Pull-back.** Without teleportation $G'$ is the latent projection of $G$ over $v$ (Evans), and the marginal of a classical model of $G$ is a classical model of the projection: the kernel of $v$ is absorbed into its children, its latent parents are relayed to its children, and its private randomness becomes a common cause of its children. With teleportation $G'$ has more facets than the projection, and $\mathcal C$ only grows with facets. $\square$

**Lift.** Given $P'\in\mathcal Q(G')$: $v$ receives its visible parents and forwards them to its children (the new visible edges), forwards the values of its classical facets and broadcasts a private random string (the new classical facets). For a quantum facet $F\ni v$ that in $G'$ extends to a child $c\in T$: $v$ and $c$ share some quantum facet, which may carry an extra maximally entangled pair; $v$ teleports its share of $F$'s state to $c$ through that pair and the visible edge $v\to c$. The children then hold exactly the resources of $G'$. $\square$

*Order dependence.* Teleportation is not symmetric, so marginalizing several nodes in different orders can give different structures; wherever the code removes several nodes it enumerates every order (`QmDAG._marginalize_predictors`).

**Example** (provable only through marginalization; five such inputs, 9.3). Input: 0→1→2→3; Q{1,3}. Node 2 relays the outcome of 1 to 3 and has no latent of its own. Marginalizing it relays the edge, 0→1→3; Q{1,3}, which is `QG_Instrumental1`. Nothing else works: PD on 2 leaves 3 without an input, conditioning on 2 is inadmissible because the grandparent 0 is not a parent of 2, and after PD on 0 conditioning on 2 would turn 1's edge into a facet and lose the setting. Both marginalizations give the same output here, since 2 has no quantum facet. The same pattern, a chain through a latent-free relay node, accounts for all five inputs; another is 0→1→2; Q{0,2}, Q{0,3}, where marginalizing 1 relays the setting 0 to 2 and gives `QG_Instrumental3`.

**Why teleportation never decides a case in four nodes.** Removing only the marginalization steps whose node holds a quantum facet, the steps where teleportation actually acts, loses no input (9.3). Conditioning explains most of this: conditioning on a node $v$ without visible parents adds a quantum facet over all quantum siblings of $v$, which is entanglement swapping, and that is the same connection teleportation marginalization would relay to the children of $v$, only not restricted to children. What conditioning cannot provide is the relayed visible input of $v$. But a node that holds a quantum share and has a visible input is itself a candidate party of a Bell-type seed, so in the structures where the relayed input matters, the structure with $v$ in place is usually provable with $v$ as the party. Where conditioning on $v$ is blocked by a grandparent, PD on that grandparent typically unblocks it. None of this is a theorem; it is why no four-node example separates the two marginalizations, and larger structures may yet do so (the lift of Section 2 is sound regardless).

---

## 3. Conditioning (`conditioning`)

**Map** (`QmDAG.condition(node)`). Remove $X$; add a classical facet over $\mathrm{Pa}(X)\cap V$ together with the latent siblings of $X$ (the visible nodes sharing some facet with $X$); add a quantum facet over the nodes sharing a quantum facet with $X$. $\pi(P)=P(\cdot\mid X=x_0)$ for a suitable value $x_0$.

**Admissibility** (`QmDAG.conditioning_is_justified`). Writing $B=\mathrm{Pa}(X)\cap V$ for the visible parents:

1. every visible grandparent of $X$ is a visible parent of $X$, so $B$ is ancestrally closed among visible nodes;
2. every facet containing a node of $B$ either contains $X$ or is contained in $B$.

In words: no grandparent of $X$, visible or latent, may fail to be a parent of $X$, except for latents whose children all lie in $B$, which only redistribute the parent block. Condition 2 is necessary. Take $X$ with parents $p_1,p_2$, each $p_i$ sharing a classical facet with an outside node $o_i$, and the classical model $p_i=\lambda_i\oplus\varepsilon_i$ with biased noise, $o_i=\lambda_i$, $X=[p_1=p_2]$. Conditioning on $X=1$ makes $o_1$ and $o_2$ dependent ($I(o_1{:}o_2)\approx0.046$ bits), while $G'$ d-separates them; so $\pi(\mathcal C(G))\not\subseteq\mathcal C(G')$ and the inference would be unsound (`tests/test_conditioning.py`).

**Pull-back.** Let $\Lambda_X$ be the latents of $X$ and $\Lambda_B$ the latents whose children all lie in $B$. By the two conditions every input of a node in $B$ lies in $B$, in $\Lambda_X$ or in $\Lambda_B$, and the latents of $\Lambda_B$ influence nothing outside $B$. In a classical model of $G$,

```math
P(\text{all}\mid x_0)\ \propto\ \Big[\prod_{\text{other nodes}}\text{kernels}\Big]\cdot\Big[\prod_{p\in B}P\big(p\mid \mathrm{pa}(p)\cap B,\Lambda_X,\Lambda_B\big)\Big]\cdot P(x_0\mid B,\Lambda_X)\cdot P(\Lambda).
```

Define the new latent $\mu=(\Lambda_X,\Lambda_B,R)$ with $R$ fresh uniform randomness. The latent siblings of $X$ read $\Lambda_X$ through $\mu$ (the facets $F\setminus X$ that $G'$ retains are set trivial). Visible children of $X$, if any, lose the edge from $X$ and read the constant $x_0$ instead. The block $B$ is sampled jointly, as deterministic functions of $\mu$ alone, from the weighted distribution proportional to $\prod_{p\in B}P(p\mid\cdot)\,P(x_0\mid B,\Lambda_X)$: all inputs of the block are available in $\mu$, so the shared seed $R$ lets each $p$ compute its coordinate of one joint sample. Every other node keeps its kernel. The result is a classical model of $G'$ for $P(\cdot\mid x_0)$. Without condition 2 some $p$ has a latent shared with outsiders, the weighted joint depends on inputs the other block members cannot see, and the counterexample above is exactly this failure. $\square$

**Lift.** Given $P'\in\mathcal Q(G')$ with the new classical facet $\mu$ and the new quantum state $\rho$ over the quantum siblings $S_Q$ of $X$. In $G$: every facet $F\ni X$ carries an independent uniform copy $\mu_F$ of the string $\mu$; every quantum facet $F\ni X$ additionally carries one maximally entangled pair between $X$ and each quantum sibling in $F$. Latent siblings read $\mu_F$ from their facet. $X$ prepares $\rho$ locally and teleports each subsystem to its sibling through the corresponding pair; post-selected teleportation, keeping only the identity outcome of each Bell measurement, leaves $S_Q$ holding $\rho$ exactly. Each visible parent $p$ draws a private guess $\hat\mu_p$, runs its $G'$ strategy with $\mu:=\hat\mu_p$, and outputs $(p,\hat\mu_p)$; its children ignore the second component. $X$ sets $X=x_0$ iff all $\mu_F$ and all $\hat\mu_p$ coincide and every Bell outcome is the identity. Conditioned on $X=x_0$ the strings are a single uniform $\mu$ shared by $B$ and the siblings, $S_Q$ shares $\rho$, and every node has run its $G'$ strategy; the parents' outputs are fine-grainings $(p,\hat\mu_p)$ of their $G'$ outputs. A fine-graining of $P'$ that lies in $\mathcal C(G')$ coarse-grains to $P'\in\mathcal C(G')$, so the contradiction argument goes through unchanged. $\square$

**Example** (provable only through conditioning; 23 such inputs). Input: 2→3; Q{0,2}, Q{0,3}, Q{1,2}. Node 0 has no parents, so both conditions hold vacuously. Conditioning on 0 removes it and adds a quantum facet over its quantum siblings 2 and 3 (the classical facet over the same pair is absorbed), giving 2→3; Q{1,2}, Q{2,3}: `QG_Instrumental3`. Operationally the lift is entanglement swapping: 0 performs a Bell measurement on its two shares and the post-selected outcome leaves 2 and 3 entangled. Among the reductions only conditioning and teleportation marginalization create a facet between nodes that did not share one, and teleportation needs a visible edge from 0 to a quantum sibling, which is absent here.

---

## 4. Interruption (`interruption`)

**Map** (`QmDAG.interruption_creation(node_with_no_children=y, node_with_no_parents=x)`). $x$ is exogenous (no visible parents, no non-singleton facet: `QmDAG.exogenous_visible_nodes`) and $y$ is not a descendant of $x$, so that $G'$ is acyclic. Remove $x$; its visible children become children of $y$. The search additionally restricts $y$ to childless nodes; the proof does not need this. $\pi$ is the renormalised diagonal $x=y$ of $P$:

```math
\pi(P)(y,\text{rest})=\frac{P(x{=}y,\;y,\;\text{rest})}{P_x(y)},
```

where $P_x$ is the marginal of $x$ (uniform for every lifted distribution, so the denominator is then a constant).

**Lift.** Given $P'\in\mathcal Q(G')$, in $G$ let $x$ be uniform, let the former children of $x$ treat the value of $x$ as they treated $y$ in $G'$, and let every other node run its $G'$ strategy. Then $P(x,y,\text{rest})=P_x(x)\,P'^{\,do(y:=x)}(y,\text{rest})$, where the right factor is the distribution of the $G'$ strategy with the input of $y$'s former children fixed to $x$ (an edge intervention). Fixing that input to the value $y$ actually takes changes nothing, so $P'^{\,do(y:=y)}=P'$ and $\pi(P)=P'$.

**Pull-back.** Let $M$ be a classical model of $G$ for a distribution with $x$ uniform; $x$ is independent of all latents of $M$. Define a model of $G'$ in which every node keeps its kernel, except that the former children of $x$ receive $y$ in place of $x$. Since $y\notin\mathrm{desc}(x)$, the value $Y(\lambda)$ does not depend on $x$ and the new model is acyclic; its distribution is $\sum_\lambda p(\lambda)[y=Y(\lambda)]\,[\text{rest}=f(\lambda,x{:=}Y(\lambda))]$, which is the diagonal $x=Y(\lambda)$ of $M$'s distribution divided by $P_x(y)$, that is $\pi(P)$. Other children of $y$ read $y$ in both structures. $\square$

**Example.** `QG_Bell1` is 0→2, 1→3; Q{2,3}. Interruption with $x=1$ (exogenous) and $y=2$ (childless, not a descendant of 1) removes 1 and makes 3 a child of 2: 0→2→3; Q{2,3}, which is `QG_Instrumental1`. The post-selection $x=y$ identifies Bob's setting with Alice's outcome. Read as a piggyback: a gap in the instrumental scenario implies a gap in the Bell scenario. Likewise `QG_Bell3b` (0→2, 1→3; Q{1,3}, Q{2,3}) maps to `QG_Instrumental2` with $x=0$, $y=3$, and `QG_Bell9`, `QG_Bell9b` map to `QG_Instrumental3b`, `QG_Instrumental3`.

**Why 9.2 shows nothing exclusive for interruption.** Its characteristic product is a Bell variant derived from an instrumental variant, and the Bell variants are seeds of the headline census, removed from the inputs. In the run with three-node seeds only (9.4) interruption is the sole route for exactly those Bell variants that are census inputs. Every other input it proves (seven in the headline census) is also reachable by PD or conditioning. Interruption is the only reduction that re-uses an outcome as a setting; how much it contributes is bounded by how many Bell-type seeds the seed list already contains.

---

## 5. Node splitting, a preprocessing step for the Fritz piggybacks (`QmDAG.split_node`)

**Map.** Replace $s$ by two nodes $s,s'$ with the same parents and the same children, sharing a classical two-party facet $\lbrace s,s'\rbrace$ (their common private randomness; absorbed when $s$ already lies in a facet). $\pi$ merges $(s,s')$ into one node.

**Lift.** $s$ computes both outputs from its inputs and private randomness. **Pull-back.** A classical kernel for the pair $(s,s')$ splits into two kernels sharing the randomness carried by the new facet. $\square$ When the absorbing facet is quantum the pair's shared randomness is carried by that quantum facet, which is legitimate because the pull-back concerns classical models only.

Node splitting is never run on its own: the split structure has one more visible node and no new independences, so it is only a stepping stone. It enters the Fritz piggybacks in two different roles, and the difference matters.

* **Splitting a predictor.** The Fritz lift makes each predictor $x\in\mathbf X$ output, alongside its own value, the part of $f(\mathrm{common}(s))$ it can compute. Formally this is a split of $x$ into $x$ and a copy $x'$ carrying the prediction. The copy may be taken *childless* without loss: it exists only to be removed again, and a childless node is removed by deleting it. So the code never materialises $x'$. Keeping $x$ untouched in $G'$ is the `split` predictor mode; removing $x$ as well (deleting it when childless, marginalizing it otherwise, Section 2) is the `drop` predictor mode, which is the composition of `split` with a marginalization. Every Fritz step treats its predictors this way; the two modes differ only in whether the predictors survive into $G'$.
* **Splitting the predicted node** (`copy` mode). Here $s$ is split and the deletion is applied to the copy $s'$, so the original keeps all of its parents while $s'$ keeps $\mathrm{common}(s)$. The copy must keep the children of $s$: the pull-back turns a classical model in which $s$ outputs a pair $(s_1,s_2)$ into a model with $s:=s_1$ and $s':=s_2$, and the children of $s$ may depend on $s_2$. Removing an edge $s'\to c$ would be an unjustified deletion; those edges can only go later, by a piggyback that justifies it. `fritz_transitions` builds the copy directly in `_fritz_build`; `fritz_entropic_transitions` literally calls `split_node` first and then runs replace mode on the copy. The two constructions coincide because the pair's shared facet is absorbed by the facet the copy shares with the predictors.

Both predicted-node modes (`replace`, `copy`) are available to both Fritz-type tricks. The predictor mode and the certificate are separate choices. The two tricks of the search (Section 8) differ in the certificate only: `Fritz` certifies by d-separation, `Fritz_entropic` emits only steps whose justification needs the LP (Section 7); both drop their predictors. Keeping the predictors is the primitive and reaches strictly more (the KPC construction of 7.7 keeps its predictor), but every kept-predictor output has as many visible nodes as its source and is expanded again, so an exhaustive closure under kept predictors is far larger than under dropped ones. Kept predictors are therefore an option of the search (`default_stages(with_kept=True)`, `predictor_modes=('split',)`) that the census does not run; Section 10 lists it as the next step. Section 9.3 separates the contributions of the predicted-node mode and the certificate.

---

## 6. The Fritz piggyback (`Fritz`)

### 6.1 The mechanism

Both Fritz piggybacks are instances of one scheme:

> choose a set of edges of $G$ to delete, giving $G'\subseteq G$, and justify the deletion by a **perfect prediction** that the quantum lift can arrange and that forces the deleted dependences to be idle in every classical model.

Fix a predictor set $\mathbf X$ and a predicted node $s\notin\mathbf X$ sharing at least one facet with some predictor. The candidate $G'$ restricts $s$ to $\mathrm{common}(s)$: visible co-parents are kept; facets of $\mathrm{common}(s)$ are kept but become classical *for $s$* (a quantum facet $F$ keeps its quantum part on $F\setminus\lbrace s\rbrace$ and a classical facet over $F$ is added, which is the state $\rho\otimes\sigma$ with $\sigma$ classical). Apart from that added classical facet $G'\subseteq G$, so by (F1) the lift is the construction of the prediction, and the entire content is the pull-back: *a classical model of $G$ in which the prediction holds, and in which the lifted distribution's other observable properties hold, is also a classical model of $G'$.*

The lift lets $s$ be a deterministic function $f$ of $\mathrm{common}(s)$, with its private randomness drawn from a facet shared with a predictor, and lets each predictor output the part of $f(\mathrm{common}(s))$ computable from what it sees, so that $\mathbf X$ jointly determines $s$. This is why $s$ must share a facet with a predictor (noise absorption) and why only $\mathrm{common}(s)$ may be kept: the predictors must be able to compute $s$.

### 6.2 Theorem (d-separation certificate)

**Claim.** Let $\mathbf X$ and $s$ be as above and suppose, in the effective DAG of $G$ (private noise nodes included),

```math
\mathbf X\setminus\mathrm{common}(s)\ \perp_d\ \mathrm{others}(s)\ \big|\ \mathrm{common}(s).
```

(A predictor that is itself a parent of $s$ lies in $\mathrm{common}(s)$ and is conditioned on rather than tested; if every predictor is a parent of $s$ the condition is vacuous.) Let $G'$ be $G$ with $s$ restricted to $\mathrm{common}(s)$ as in 6.1, predictors kept. Then a QC gap in $G'$ implies a QC gap in $G$. The same holds after removing the predictors as in Section 5 (deletion of childless predictors, marginalization of the others, in every order).

**Proof.** *Lift:* given $P'\in\mathcal Q(G')$, run it in $G$ with $s:=f(\mathrm{common}(s))$ as in $P'$, legitimate because in $G'$ the parents of $s$ are classical and $s$ shares a facet with a predictor that carries its randomness, and let each predictor additionally output what it sees of $f(\mathrm{common}(s))$. The result $P$ lies in $\mathcal Q(G)$, indeed in $\mathcal Q(G')$ since the extra outputs are functions of the predictors' parents, coarse-grains to $P'$, and satisfies $H(s\mid\mathbf X)=0$.
*Pull-back:* let $M$ be a classical model of $G$ for $P$. Write $s=h(\mathrm{common},\mathrm{others})$ with $\mathrm{others}$ including the private noise, and $s=g(\mathbf X)$. Condition on $\mathrm{common}(s)$. By the d-separation hypothesis and (F2) for classical models, $\mathrm{others}(s)$ and $\mathbf X$ are independent given $\mathrm{common}(s)$. A random variable that is a function of each of two independent variables is almost surely constant, so $s=\varphi(\mathrm{common}(s))$ a.s. Replace the kernel of $s$ in $M$ by $\varphi$; the joint distribution is unchanged and the new model is Markov to $G'$. Hence $P\in\mathcal C(G')$ and, coarse-graining the predictors, $P'\in\mathcal C(G')$. $\square$

The set test is equivalent to the conjunction of the per-edge tests $\mathbf X\perp_d v\mid\mathrm{Pa}(s)\setminus\lbrace v\rbrace$ over $v\in\mathrm{others}(s)$, by the weak-union and intersection properties of d-separation. The implementation is `QmDAG.fritz_admissible_targets`, which tests the set at once with networkx's `is_d_separator`; the structure is built by `QmDAG._fritz_build` from an explicit map of kept parents.

### 6.3 The unit of deletion

The piggyback deletes all of $\mathrm{others}(s)$ or nothing. Deleting a single parent $p\to s$ while $s$ keeps another parent $q$ that $\mathbf X$ cannot see is not a candidate: the predictors cannot predict $s$, and no other piggyback justifies the lone deletion. Example: 0→2, 1→2; Q{0,1}, Q{1,3}, Q{2,3}. Predictor 3 deletes 0→2 and 1→2 together and reaches `QG_Bell6c` (an entropic step, Section 7.8), whereas neither structure with only one of the two edges deleted is reachable from the input by any trick of the toolkit, out of 760 reachable structures. Across the four-node census, of the single-target Fritz steps that delete two or more non-noise parents, more than half have no reachable single-edge intermediate.

What *does* decompose is the choice of several predicted nodes for one predictor set. With the predictors kept, the predicted nodes can be restricted one at a time and the predictors removed afterwards by Section 2, so the subset enumeration in `fritz_transitions` is a convenience of the `drop` mode, not a logical necessity. Joint predictor *sets*, on the other hand, are not reducible to single predictors: a set sees more than any member.

### 6.4 Examples

**Replace mode** (the one census input that needs a `Fritz` step in replace mode and nothing more). Input: no edges; Q{0,1,2}, Q{0,1,3}, Q{0,2,3}, Q{1,2,3}, every triple entangled. Predictor 0, predicted 1. The parents of 1 are the three facets containing it; 0 sees Q{0,1,2} and Q{0,1,3}, so $\mathrm{common}(1)$ is those two facets and $\mathrm{others}(1)$ is Q{1,2,3} plus the noise of 1. The d-separation test asks whether 0 is separated from Q{1,2,3} given the two common facets: every path from 0 to Q{1,2,3} runs through 2 or 3 as a collider (0 ← Q{0,2,3} → 2 ← Q{1,2,3}), so it holds. Node 1 is restricted to its two common facets, which become classical for it, and the childless predictor 0 is deleted: the output is C{1,2}, C{1,3}, Q{2,3}, the triangle with one quantum and two classical sources, `QG_Triangle3`. In the lift, 1 is a function of its two shares and 0, which holds the same shares, announces that function; classically, any model in which 0 predicts 1 perfectly cannot let 1 depend on Q{1,2,3}, which 0 never sees.

**Copy mode** (one of TODO-FRITZ-COPY-COUNT census inputs that need a copy-mode `Fritz` step). Input: 0→2, 1→2, 2→3; Q{0,1}, Q{0,2}, Q{1,3}. Predictor 0 (childful: it feeds 2), predicted 1 in copy mode. $\mathrm{common}(1)$ is Q{0,1}; $\mathrm{others}(1)$ is Q{1,3} plus noise; 0 is separated from Q{1,3} given Q{0,1} because the paths through 1 and through 2 both end in colliders. The copy 1' keeps Q{0,1}, classically, and the child 2 of 1; the original 1 keeps Q{1,3}. Dropping 0 marginalizes it with teleportation (its share of Q{0,2} goes to 2, its facets become classical for 2). Output: 1→2, 2→3, 1'→2; C{1,1',2}, Q{1,2}, Q{1,3}. Marginalizing 1 next, with teleportation of its share of Q{1,3} to 2, gives 1'→2→3; C{1',2}, Q{2,3}: `QG_Instrumental2b`. Replace mode cannot do this: it would strip Q{1,3} from 1, and that facet is the one that survives into the seed.

**Joint predictor sets.** The code supports a predictor *set* that predicts jointly: $\mathrm{common}(s)$ collects everything any member sees. In the four-node census such steps are never essential (9.3). Fritz's own construction for the tetrahedron (no edges; all four triples as quantum facets) is different: each of two nodes alone predicts a copy, so the copy keeps only what *both* see, the intersection. In the search this is two sequential single-predictor steps: predictor 2 makes copies 0' and 1' and keeps Q{0,1,2}, Q{0,2,3} for 0' and Q{0,1,2}, Q{1,2,3} for 1', then predictor 3 restricts 0' to Q{0,2,3} and 1' to Q{1,2,3}; dropping both predictors leaves Q{0,1}, C{0,0'}, C{1,1'}, which is `QG_Bell6`. The tetrahedron is also proven in one step by a single predictor (the replace-mode example above), so neither route is essential for it.

---

## 7. The entropic Fritz piggyback (`Fritz_entropic`)

### 7.1 Why the same mechanism admits a stronger certificate

In the pull-back of 6.2 the only facts about the classical model that were used were the Markov property of $G$ and $H(s\mid\mathbf X)=0$. But the lifted distribution $P$ has more structure: it lies in $\mathcal Q(G')$, so by (F2) **every observable d-separation of $G'$ holds for $P$**, and therefore holds in any classical model of $G$ for $P$. These conditional independences are properties of the *new* graph, obtained for free, and there is no reason not to use them as hypotheses when deciding whether a deletion is justified in the *original* graph. The d-separation test of 6.2 ignores them; it is a shortcut that trades proof power for speed. Khanna, Pusey and Colbeck's six-node example (7.7) is a case where the shortcut fails and the extra independences are exactly what is needed.

### 7.2 Entropy vectors as the proof system (`entropic_lp.py`)

For $n$ random variables the entropy vector $h\in\mathbb R^{2^n-1}$ lists $H(S)$ for every nonempty $S$. Every entropy vector satisfies the **elemental Shannon inequalities** $H(i\mid[n]\setminus i)\ge0$ and $I(i{:}j\mid K)\ge0$ for $i\lt j$ and $K\subseteq[n]\setminus\lbrace i,j\rbrace$, which number $n+\binom n2 2^{n-2}$ (`elemental_inequalities(n)`; the $n=3$ matrix is checked against the authors' Mathematica notebook, the row count for $n\le8$). A conditional independence is the vanishing of a conditional mutual information, a functional dependence the vanishing of a conditional entropy; both are nonnegative on the cone, so each hypothesis is one inequality "$\le0$". The local Markov property of a DAG is one conditional mutual information per node, $I(v{:}\mathrm{nondesc}(v)\setminus\mathrm{pa}(v)\mid\mathrm{pa}(v))=0$ (`local_markov_rows`). The Shannon cone with these hypotheses implies exactly the d-separations of the DAG: every d-separation follows from the local Markov statements by the semigraphoid axioms (Verma and Pearl), which are Shannon-derivable, and nothing beyond the d-separations can follow because classical models realise every non-d-separated dependence (F2). This is tested on random DAGs.

A target functional $t$ is **implied** when the LP $\lbrace\text{Shannon}\ge0,\ \text{hypotheses}\le0,\ t\ge1\rbrace$ is infeasible (the cone is scale invariant). Geometrically the question is **cone inclusion**: the hypotheses cut out a polyhedral cone $K$, each target is a halfspace, and the certificate asks whether $K$ lies in the intersection of the target halfspaces, that is, inside the cone of entropy vectors Markov to the target graph. Both cones are given by inequalities, and inclusion of one inequality description in another is decided by one LP per inequality of the inner one; converting to extreme rays (as cdd, lrs or PANDA would) is exponential and unnecessary. Infeasibility is decided with the Mosek Optimizer API (`EntropicLP`, interior point); the dual ray is a Farkas certificate, a nonnegative combination of elemental inequalities and hypotheses that reproduces $t$, available through `EntropicLP.farkas_certificate`. Implications proven this way are valid for every distribution, with or without full support, because they use only Shannon inequalities. The method is incomplete in the other direction: a feasible LP does not exhibit a distribution, only a vector in the Shannon cone.

### 7.3 The certificate and its two target sets (`QmDAG._entropic_certificate`)

Variables: all nodes of `lp_structure`, the visible nodes and the facets of $G$; quantum facets are ordinary latent variables here, since only classical models are analysed (F2). Hypotheses: Shannon; local Markov of $G$; $H(s\mid\mathbf X)\le0$; and $\mathcal C(G')$, the elementary observable d-separations $I(x{:}y\mid Z)$ of the candidate $G'$ over its visible nodes (`observable_dseparation_rows`). Two target sets are tried, in this order.

* **`markov`**: the local Markov equalities of $G'$ over $G$'s own latents. If all are implied, the joint of any classical model of $G$ satisfying the hypotheses is Markov to $G'$, hence $P\in\mathcal C(G')$.
* **`relabel`**: applicable when $s$ keeps exactly one parent in $G'$ and that parent is a facet $L$ (the code tests `len(common) == 1` and that the element is a facet index). Let $G''$ be $G'$ with $L$ deleted and $s$ made a parent of every other child of $L$. Targets: the local Markov equalities of $G''$ over $G$'s variables other than $L$.

### 7.4 Why `relabel` is justified

The claim to be proven is not about one classical model. It is: *every* classical model of $G$ whose observed statistics satisfy the perfect prediction and the independences $\mathcal C(G')$ can be exchanged for a classical model of $G'$ with the same observed statistics. The argument has three steps.

1. **Fix one model.** Any classical model $M$ of $G$ producing $P$ defines one joint distribution $Q$ over the visible nodes and the facet latents (noises integrated out). The hypotheses are facts about $Q$: $Q$ is Markov to $G$; $H(s\mid\mathbf X)=0$; the independences of $\mathcal C(G')$ hold among the visible nodes. Shannon's inequalities hold for $Q$, so the entropy vector of $Q$ satisfies every LP hypothesis.
2. **What infeasibility proves.** The LP shows that every entropy vector satisfying the hypotheses has each target conditional mutual information equal to zero. A vanishing conditional mutual information is a conditional independence. So the targets hold for $Q$, and since $M$ was arbitrary they hold for every model satisfying the hypotheses.
3. **From Markov($G''$) to a model of $G'$.** The targets are the local Markov conditions of $G''$ over $W=V\cup(\text{facets}\setminus\lbrace L\rbrace)$. For a DAG, the local Markov conditions are equivalent to the factorisation of $Q(W)$ into one kernel per node given its parents in $G''$, so the marginal of $Q$ over $W$ *is* a classical model of $G''$, one in which $s$ is a root. Now rename. Introduce a fresh latent $\lambda'$ carrying the value of $s$, with the root distribution $Q(s)$; let $s$ read $\lambda'$ deterministically (its private noise in $G'$ may be trivial); let every other former child $c$ of $L$ keep its $G''$ kernel with $s$ replaced by $\lambda'$, which is legal because the parents of $c$ in $G'$ are its parents in $G''$ with $s$ replaced by $L$; the kept quantum part of the facet over $F\setminus\lbrace s\rbrace$ is left unused; everything else is unchanged. This is a classical model of $G'$ with observed marginal $Q(V)=P$. The original latent $\lambda_L$ of $M$ is simply discarded: in the new model the common cause of $s$ and its siblings is played by the value of $s$ itself.

In the KPC example (7.7) this reads: every model in which $E$ depends on $C$ and $D$, with $F$ predicting $E$ and $E$ independent of $(C,D)$, is replaced by a model in which $E$ depends on a single latent $A'$ alone, where $A'$ is the old value of $E$.

**Why `markov` cannot do this.** The model $E=A\oplus B$, $D=C\oplus B$, $F=(A,B)$ satisfies every hypothesis of the KPC candidate, yet $E$ is not a function of the given $A$. No certificate that keeps $A$ as the witness can succeed; `relabel` succeeds because it discards $A$.

**Why the single-parent condition.** If $s$ kept another parent $v$ in $G'$, then $s$ would not be a root of $G''$, and $\lambda':=s$ would be correlated with $v$, violating the independence of roots in $G'$.

**Neither target set implies the other.** `relabel` without `markov` is the KPC example. `markov` without `relabel`: let $s$ read a visible parent $v$ and one facet $L=\lbrace s,c,d,x\rbrace$ shared with the predictor $x$. d-separation certifies deleting $v\to s$, so `markov` holds, but the `relabel` targets contain $d\perp c\mid s$, which the model $L=(L_1,L_2)$, $s=L_1$, $c=d=L_2$, $x=L$ violates with $I(d{:}c\mid s)=1$ bit while satisfying every hypothesis (`tests/test_fritz_entropic.py`). Both target sets are instances of one scheme, replace each latent of $G'$ by a set of $G$'s variables and ask for Markov($G'$) with the substitution: `markov` substitutes $\lbrace L\rbrace$, `relabel` substitutes $\lbrace s\rbrace$. Further substitutions are an open direction (Section 10).

**Relation to the d-separation certificate.** The pull-back of 6.2 derives $s\perp\mathrm{others}\mid\mathrm{common}$ and keeps the given latents, so Theorem 6.2 always produces a `markov`-type witness and never a relabelling. The LP subsumes it at least for the row of $s$: if $\mathbf X\perp_d\mathrm{others}\mid\mathrm{common}$ then the Shannon cone with the Markov hypotheses derives $I(\mathbf X{:}\mathrm{others}\mid\mathrm{common})\le0$, and with $H(s\mid\mathbf X)\le0$ and the Markov row of $s$ it derives $H(s\mid\mathrm{common})\le I(s{:}\mathbf X\mid\mathrm{common})\le I(\mathrm{others}{:}\mathbf X\mid\mathrm{common})+I(s{:}\mathbf X\mid\mathrm{common},\mathrm{others})=0$, which implies the `markov` row of $s$. The `markov` rows of other nodes, whose non-descendant sets grow in $G'$, were checked empirically (every d-separation-admissible pair in the test structures is LP-admissible); the search never relies on this, since d-separation-certified candidates are admitted by Theorem 6.2 without an LP. `relabel` has no d-separation analogue in $G$: its hypotheses $\mathcal C(G')$ are facts about $G'$, not about the graph of $G$.

### 7.5 Theorem (entropic certificate)

**Claim.** Let $G'$ be a candidate as in 6.1, possibly with further parents deleted from any non-predictor node (7.6). If the LP of 7.3 implies the `markov` targets, or the single-parent condition holds and it implies the `relabel` targets, then a QC gap in $G'$ implies a QC gap in $G$.

**Proof.** Lift as in 6.2: $P\in\mathcal Q(G')\subseteq\mathcal Q(G)$, $H(s\mid\mathbf X)=0$, and by (F2) $P$ satisfies every observable d-separation of $G'$. Let $M$ be a classical model of $G$ for $P$. Its joint entropy vector lies in the Shannon cone, satisfies the local Markov equalities of $G$ and the hypotheses $H(s\mid\mathbf X)=0$ and $\mathcal C(G')$. The LP implication forces the target quantities to vanish, so the joint of $M$ is Markov to $G'$ (or to $G''$, and then 7.4 gives a model of $G'$). Hence $P\in\mathcal C(G')$ and, coarse-graining the predictors, $P'\in\mathcal C(G')$. $\square$

Several predicted nodes for the same $\mathbf X$ are handled jointly: one hypothesis $H(s_i\mid\mathbf X)\le0$ per node, one candidate $G'$ restricting all of them.

### 7.6 The search procedure and extra deletions (`QmDAG.fritz_entropic_transitions`, `QmDAG._entropic_extra_deletions`)

Stated edge-first, the unit of search is a **candidate deletion**: a visible node $s$ and a set $D\subseteq\mathrm{Pa}(s)$ of its parents in the effective DAG to delete, a parent being a visible parent or a facet containing $s$. Deleting $D$ dictates everything else.

1. **Who may predict.** After the deletion $s$ depends only on $K=\mathrm{Pa}(s)\setminus D$, so a predictor set $\mathbf X$ is usable iff $\mathbf X$ sees every element of $K$ (each kept facet contains a predictor, each kept visible parent is a parent of a predictor or is itself a predictor) and $K$ contains at least one facet shared with a predictor, which carries the private randomness of $s$ in the lift.
2. **Which conditional independences.** $G'$ is $G$ with $D$ deleted, predictors kept, so the hypothesis set $\mathcal C(G')$ is fixed by $D$.
3. **What has to be proven.** The `markov` or `relabel` targets from Shannon, Markov($G$), $H(s\mid\mathbf X)=0$ and $\mathcal C(G')$.

The implementation runs this in the opposite order: for each predictor set (single nodes by default, pairs optionally) and each latent sibling $s$ of a predictor, it takes the maximal deletion $D_0=\mathrm{Pa}(s)\setminus\mathrm{common}(s)$, everything the predictors cannot see. The d-separation test of 6.2 is run first and the LP only where it fails. Then a **greedy loop** tries further deletions: the hypotheses do not mention which node a deletion concerns, so the same LP can certify deleting a parent $p$ of any non-predictor node $t$ through the per-edge target $I(t{:}p\mid\mathrm{Pa}'(t)\setminus p)\le0$, where $\mathrm{Pa}'(t)$ are the parents of $t$ in the current candidate. Deleting an edge enlarges $\mathcal C(G')$, so the hypotheses are recomputed from the current candidate after each deletion and the loop runs to a fixed point, budgeted (`max_lps`) and order-dependent, not exhaustive. Because deleting edges also enlarges non-descendant sets, the final candidate is verified as a whole with 7.3; if that fails, the candidate without extra deletions is kept. Soundness is Theorem 7.5 for the final candidate. Deletions never touch a predictor (it must keep seeing $\mathrm{common}(s)$), and never remove the last facet a predicted node shares with its predictors: without it the hypotheses $H(s\mid\mathbf X)=0$ and $s\perp\mathbf X$ (an observable d-separation of the resulting $G'$) are jointly satisfiable only by a constant $s$, the LP would certify a vacuous statement, and the lift would no longer exist.

For every candidate the code records the predictor set, the predicted nodes and their modes, the predictor mode, which certificate closed it and the extra deletions, so each entropic step in a certificate can be re-derived by hand. `Fritz_entropic` emits only LP-reliant steps: those certified by `markov` or `relabel`, or carrying extra deletions. A candidate that plain d-separation certifies is left to `Fritz` and `Fritz_kept`, so the three trick names partition the Fritz-type steps by the justification they need.

### 7.7 Worked example: the KPC structure

$G_1$: visible $C,D,E,F$; facets $A=\lbrace E,F\rbrace$, $B=\lbrace D,F\rbrace$; edges $C\to D$, $C\to E$, $D\to E$. Predictor $F$, predicted $E$: $\mathrm{common}(E)=\lbrace A\rbrace$, $\mathrm{others}(E)=\lbrace C,D,\text{noise}\rbrace$. The d-separation test fails, since $F\leftarrow B\to D$ is open. The candidate $G'$ has $E$ with parent $A$ only and d-separates $E$ from $\lbrace C,D\rbrace$. With the hypothesis $E\perp CD$ the LP certifies the `relabel` targets ($A:=E$), reproducing the authors' Lemma 1 without the equality $F_S=E$ or an explicit split. Their construction keeps $F$ (the predictor is a childless copy $F_S$, Section 5); with the predictor kept the output is $C\to D$, $D$–$F$ quantum, $E$–$F$ classical: the Bell variant `QG_Bell5`, in one step (`tests/test_fritz_entropic.py`, with `predictor_modes=('split',)`). With $F$ dropped, as the census runs, $E$ is left with a private facet only and nothing is learnt, so this example is reproduced by the unit test and not by the census.

### 7.8 Further examples from the census

TODO-ENTROPIC-COUNT inputs are provable only with an LP-certified step (9.2). Three, with different certificates:

**`relabel`, one step.** Input: 0→1→2; Q{0,2}, Q{1,3}, Q{2,3}. Predictor 3, predicted 2: $\mathrm{common}(2)$ is Q{2,3}; $\mathrm{others}(2)$ is the visible parent 1, the facet Q{0,2} and noise. d-separation fails because 3 ← Q{1,3} → 1 is open. The candidate $G'$ has 2 reading Q{2,3} alone, and in $G'$ node 2 is separated from 0 and 1 (every path leaves 2 through Q{2,3} to 3, where it meets a collider), so $I(2{:}01)=0$ is a hypothesis. With it the LP derives the `relabel` targets: the joint is Markov to $G''$ in which 2 is a root and a parent of 3. Output, predictor kept: 0→1; C{2,3}, Q{1,3}, which is `QG_Bell9` in one step. The same input is the Bell6c example of 6.3 after relabelling.

**`relabel` where `markov` fails** (the one census input that is lost without `relabel`-certified steps). Input: 0→3, 1→2; Q{0,1}, Q{0,2}, Q{1,3}, Q{2,3}. Predictor 0, predicted 2: $\mathrm{common}(2)$ is Q{0,2}; $\mathrm{others}(2)$ is 1, Q{2,3} and noise; d-separation fails through 0 ← Q{0,1} → 1. In $G'$, 2 is separated from 1 (the paths 2 ← Q{0,2} → 0 ← Q{0,1} → 1 and 2 ← Q{0,2} → 0 → 3 ← Q{1,3} → 1 both contain colliders), giving the hypothesis $I(2{:}1)=0$. Output: 0→3; C{0,2}, Q{0,1}, Q{1,3}. Node 1 now has no visible parents, so conditioning on it is admissible and swaps entanglement onto its quantum siblings 0 and 3: 0→3; C{0,2}, Q{0,3}, which is `QG_Instrumental3b`.

**`markov` with an extra deletion** (one of six inputs lost without extra deletions). Input: 0→2, 1→2, 2→3; Q{0,1}, Q{1,3}. Predictor 1, predicted 0: $\mathrm{common}(0)$ is Q{0,1} and $\mathrm{others}(0)$ is only the noise of 0, so the first step is trivial (d-separation). The greedy loop then deletes the edge 0→2: with $H(0\mid1)=0$ among the hypotheses, $I(2{:}0\mid1)\le I(0{:}2,3,\ldots\mid1)\le H(0\mid1)=0$, so 0 is redundant as a parent of 2 wherever 1 is also a parent. The final candidate is verified as a whole with the `markov` targets. Output: 1→2→3; C{0,1}, Q{1,3}. Marginalizing the middle node 2 relays 1→3: 1→3; C{0,1}, Q{1,3}, which is `QG_Instrumental3b`. This mechanism, a predicted node becoming redundant as a parent next to its predictor, is what every extra deletion in the census does.

---

## 8. Search and certificates (`qc_gap_search.py`)

`ClosureExplorer` expands every reachable structure, up to relabelling, exactly once under a set of tricks and records each `Transition(trick, params, source, target)`. Every transformation builds a `LabelledDirectedStructure`/`LabelledHypergraph` over named nodes and re-indexes them; copies are named `"<s>_copy"` during construction. Each unlabelled id has one stored representative, every trick is applied to that representative, and the recorded params are stated in its labels; certificates print the representative above each step. Keying on unlabelled ids is legitimate because every piggyback is label-equivariant, which is also why the order dependence of Section 2 is enumerated rather than fixed.

**Stages.** `prove_gaps` runs the search in stages, cheapest first (`default_stages`). Every structure proven in a stage is a known gap for the next.

1. `elementary`: PD, interruption, conditioning and both marginalizations, closed over everything reachable from every input.
2. `Fritz`: the d-separation Fritz piggyback, closed over everything reachable.
3. `Fritz_entropic`: the LP-certified steps, applied to the inputs still unproven; their children are expanded with the tricks of stages 1 and 2.

Reachability is transitive, so the set of inputs proven does not depend on the order in which tricks are applied: an input that reaches a structure which reaches a seed reaches the seed, and `_fixpoint` iterates the implication closure among the inputs until nothing changes. Re-running a stage on the same roots therefore adds nothing; what stage 3 leaves out is LP steps at intermediate structures (depth greater than one), and 9.3 reports the experiment of closing under the LP trick everywhere. What the order decides is cost (the LP runs only where the cheap stages failed) and which tricks a certificate prefers (`build_report` takes the shortest chain within the earliest stage that has one). The seeds (`known_QC_gaps.py`) are the instrumental, triangle and Evans variants with three visible nodes and the Bell variants with four; `prove_gaps` closes the proven set under implication from them and attaches to every proven input a certificate, the shortest chain of transitions down to a named seed.

**The known-gap database the search establishes.** Every structure the search touches, inputs, intermediates and the hybrid structures with classical facets that the Fritz steps create, is a proven QC gap as soon as it reaches a seed, and `GapReport.proven_structure_ids` returns that set. This is the database to keep: the census inputs have every latent quantum, but a structure with some facets classical is a *weaker* structure, its gap is not implied by the gap of the all-quantum version (making a facet classical shrinks the quantum set and leaves the classical set), and it is a gap only if a chain of piggybacks from it reaches a seed. Fritz steps are indifferent to whether the shared facet is classical or quantum, so chains often transfer; conditioning and teleportation are not, since they create quantum facets only among quantum siblings. 9.1 reports how many hybrid structures the census proves in passing. Hybrid four-node structures as *inputs* are not in the census yet.

**Reading the tables.** Reachability restricted to any subset of tricks, or to any predicate on the recorded parameters, is a graph query over the recorded transitions. "Alone" counts the inputs provable with one trick or group; "only" counts the inputs lost when one trick is removed and everything else kept; the stage counts of 9.1 are cumulative. Since `Fritz_kept` and `Fritz_entropic` were applied only to inputs the earlier stages left unproven, an input is "only via `Fritz`" when it needs a Fritz-type step that d-separation with dropped predictors supplies, not because the later tricks could not supply it.

**The seeds matter.** Every piggyback except the Fritz type reduces the number of visible nodes, so for four-node inputs a four-node seed can only be reached by a Fritz-type step. Listing the Bell variants as seeds is what lets the Fritz steps conclude in one move; it also means the reductions get no credit for the Bell variants themselves, which is why 9.4 also reports a run with the three-node seeds only. The Bell variants that are not derivable from the three-node seeds by any piggyback (9.4) have to be seeds, on the strength of the direct Bell argument.

---

## 9. Results on the four-node census

The inputs are the four-node mDAGs whose edges respect the order $0\lt1\lt2\lt3$ and that are not provably algebraic, with every latent quantum and the Bell seeds removed. All counts are of distinct structures up to relabelling. `tests/test_baseline_slow.py` pins the headline values; `Special Applications/census_breakdowns.py` produces the finer tables.

### 9.1 Headline: proven after each stage

| stage | proven (cumulative) | new |
|---|---|---|
TODO-STAGE-TABLE

Labelled input structures 2759, distinct 990, remaining TODO-REMAINING-COUNT. TODO-WALLTIME

### 9.2 Per trick

"Alone" is the number of inputs provable using only that trick, closed under implication among the inputs; the Fritz-type tricks are counted together with the marginalizations (their outputs often need reducing), `Fritz_kept` together with `Fritz`, and `Fritz_entropic` together with both. "Only" is the number of inputs no longer provable when that single trick is removed from the recorded transitions, everything else kept.

| trick | alone | only |
|---|---|---|
TODO-PER-TRICK-TABLE

Three remarks on reading the "only" column.

* The inputs lost without `Fritz` are of course also within reach of `Fritz_kept` and `Fritz_entropic`, which subsume it (Section 5, 7.4). They count as `Fritz`-only because the later stages ran only on inputs the earlier stages had not proven (Section 8). One of them is the replace-mode example of 6.4; the others need copy mode.
* The zeros for interruption and the marginalizations are not statements of uselessness. Interruption's exclusive product is pre-empted by the Bell seeds (Section 4). The two marginalizations coincide whenever the removed node has no quantum facet, so removing one of them alone loses nothing; removing both loses five inputs (Section 2 and 9.3).
* "Only" counts are not additive. The `Fritz_entropic`-only inputs are exactly those the LP stage adds; the PD-only and conditioning-only sets are disjoint from each other.

### 9.3 Fritz-type steps by mode and certificate

The provenance of each Fritz-type transition records its predictor set, the mode of each predicted node, the predictor mode and the certificate, so the counts can be refined to categories of steps. "Lost" is the number of inputs no longer provable when every transition of that category is removed, all other transitions kept.

| category of step | lost when removed |
|---|---|
TODO-CATEGORY-TABLE

Cheap to expensive, cumulatively:

| tricks allowed | proven | new |
|---|---|---|
TODO-LADDER-TABLE

TODO-MODE-SUMMARY

### 9.4 Three-node seeds only

Rerunning the staged search with the three-node seeds only (instrumental, triangle and Evans variants), the Bell variants being ordinary inputs:

| quantity | value |
|---|---|
TODO-THREESEEDS-TABLE

TODO-THREESEEDS-TEXT

### 9.5 Entropic certificates attempted

TODO-ENTROPIC-STATS

---

## 10. Open questions

* **Completeness of the entropic certificate.** The failures recorded in `ENTROPIC_STATS` are failures of the LP, not necessarily of the piggyback (7.4). Substituting other sets of variables for the latents of $G'$, beyond $\lbrace L\rbrace$ and $\lbrace s\rbrace$, is the natural next level: each substitution is expressible because the LP indexes joint entropies of sets, and the soundness proof is that of 7.4.
* **The prediction-free edge-deletion piggyback** (KPC, Corollary 3): the same LP without the perfect-prediction hypothesis certifies deleting edges justified by the new graph's independences alone. The code can check it (`QmDAG._entropic_certificate(..., predicted=())`), but it is not exposed as a trick.
* **Exact certificates.** Farkas multipliers are floating point; rationalising them and re-verifying the combination exactly is cheap and would make every entropic step a checkable proof.
* **The remaining structures.** The 18 unproven inputs all contain the edge 0→1 with 0 entangled with later nodes, and most contain the chain 0→1→2→3:

  | | edges | quantum facets |
  |---|---|---|
  | 1 | 0→1, 1→2, 1→3 | {0,2}, {0,3} |
  | 2 | 0→1, 2→3 | {0,2}, {1,3} |
  | 3 | 0→1, 0→2, 2→3 | {1,3} |
  | 4 | 0→1, 0→2, 2→3 | {0,2}, {1,3} |
  | 5 | 0→1, 1→2, 2→3 | {0,2}, {1,3} |
  | 6 | 0→1, 1→2, 2→3 | {0,2}, {0,3}, {1,2,3} |
  | 7 | 0→1, 1→2, 2→3 | {0,2}, {0,3}, {1,3} |
  | 8 | 0→1, 1→2, 2→3 | {0,2}, {0,3}, {1,2}, {1,3} |
  | 9 | 0→1, 1→2, 2→3 | {0,1}, {0,2}, {0,3}, {1,3} |
  | 10 | 0→1, 1→2, 2→3 | {0,2}, {0,3}, {1,3}, {2,3} |
  | 11 | 0→1, 1→2, 2→3 | {0,2}, {0,3}, {1,2}, {1,3}, {2,3} |
  | 12 | 0→1, 0→2, 1→2, 2→3 | {1,3} |
  | 13 | 0→1, 0→2, 1→2, 2→3 | {0,2}, {1,3} |
  | 14 | 0→1, 1→2, 1→3, 2→3 | {0,2}, {0,3} |
  | 15 | 0→1, 1→2, 1→3, 2→3 | {0,2}, {0,3}, {1,3} |
  | 16 | 0→1, 1→2, 1→3, 2→3 | {0,2}, {0,3}, {1,2,3} |
  | 17 | 0→1, 1→2, 1→3, 2→3 | {0,2}, {0,3}, {2,3} |
  | 18 | 0→1, 1→2, 1→3, 2→3 | {0,2}, {0,3}, {1,3}, {2,3} |

  Several are Bell scenarios with extra structure among the settings. Number 2 is the Bell scenario with entangled settings (0 and 2 are the settings of 1 and 3, Q{1,3} the shared state); number 3 lets Alice's setting influence Bob's. Both have a QC gap by the usual argument, because the latents of the settings are independent of the latent of the outcomes, so any classical model is local for $P(a,b\mid x,y)$. No piggyback can reach them from the Bell seeds, since deleting a facet or an edge between settings is not a piggyback. Extending the seed list with such Bell variants, after checking each by the direct argument, is the cheapest next step.
