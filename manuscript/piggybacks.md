# Piggybacks for quantum-classical gaps

*What each structural transformation in this repository proves, why it is sound, what it contributes to the four-node census, and where it lives in the code.*

**Contents.** 0 Setting · 1 Point distribution · 2 Marginalization · 3 Conditioning · 4 Interruption · 5 The Fritz piggyback (mechanism, node splitting and predictor modes, theorem, examples) · 6 The entropic Fritz piggyback · 7 Search and certificates · 8 Results on the four-node census · 9 Open questions · 10 References

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

All piggybacks here are **caveat-free**: the hypothesis is a QC gap of $G'$ as a structure, with no side condition on the gap-witnessing distribution. This is what lets them be composed in any order and searched as a closure (Section 7). Structures are identified up to relabelling by `unique_unlabelled_id`, and every count in this document is a count of distinct structures up to relabelling.

### 0.3 Two general facts

* **(F1) Edge monotonicity.** If $G'$ is obtained from $G$ only by deleting visible edges and removing nodes from facets, then $\mathcal Q(G')\subseteq\mathcal Q(G)$ and $\mathcal C(G')\subseteq\mathcal C(G)$: run the same strategy and ignore the extra inputs. For an edge-deleting piggyback the lift is therefore immediate and the whole content is the pull-back.
* **(F2) d-separation.** Among the *visible* variables, every d-separation of $G$ implies the corresponding conditional independence, in classical and in quantum models alike: observable d-separation is theory-independent (Henson, Lal and Pusey). Conversely no other conditional independence among visible variables holds in every classical model, hence none holds in every quantum model either, since $\mathcal C(G)\subseteq\mathcal Q(G)$. Conditional independences that *involve latents* are statements about a joint distribution of visible and latent variables. Such a joint exists for a classical model, where every node of the effective DAG is a random variable, but not in general for a quantum one: a quantum facet is a state, not a variable, and only coexisting sets of nodes have a joint distribution (Chaves, Majenz and Gross). The pull-backs below reason about classical models only, so they may use latent-involving d-separations and the entropies of the effective DAG freely.

### 0.4 The census, the seeds and the names of the tricks

The **census** (Section 7) is the set of four-node mDAGs whose directed edges respect the order $0\lt1\lt2\lt3$ and that are not provably algebraic, with every latent facet quantum; these are the **inputs**. The **seeds** (`known_QC_gaps.py`) are structures with an established QC gap: the instrumental, triangle and Evans variants with three visible nodes, and the Bell variants with four. The census runs in two phases (Section 7): a cheap phase in which only the elementary reductions act and only the three-node seeds are known, so that the Bell variants are inputs, and an expensive phase in which the Bell variants are seeds and the Fritz-type tricks are applied to whatever the cheap phase left. An input is **proven** when a chain of piggybacks leads from it to a seed or to an input already proven; a proven structure is a **known gap**. "Provable only through $T$" means that the input is no longer proven when every transition of trick $T$ is removed from the record and everything else is kept (Section 7).

The **tricks** are the piggybacks as the search applies them: `PD`, `interruption`, `conditioning`, `naive_marginalization` and `teleportation_marginalization` (the **elementary reductions**, which lower the number of visible nodes), and the Fritz-type tricks `Fritz`, `Fritz_kept` and `Fritz_entropic` (Sections 5 and 6). The search runs them in **stages** named after the tricks (Section 7). Two nodes are **latent siblings** when they share a facet, **quantum siblings** when they share a quantum facet.

---

## 1. Point distribution (`PD`)

**Map.** Remove one visible node $v$ together with all its edges and its membership in facets; write $G-v$ for the result, so $G'=G-v$ (`QmDAG.fix_to_point_distribution_QmDAG`; trick `pd_trick` in `qc_gap_search.py`).

**Lift.** Given $P'\in\mathcal Q(G')$, let $v$ output a constant and every other node run its $G'$ strategy; $\pi$ is the marginal over $v$. $P\in\mathcal Q(G)$ because the children of $v$ may ignore a constant input.

**Pull-back.** In a classical model of $G$ for a distribution with $v$ constant, substitute the constant into the kernels of $v$'s children; the result is a classical model of $G-v$ for the marginal. $\square$

**Example** (provable only through PD; 221 such inputs). Input: 0→2, 0→3; Q{0,1,2}, Q{0,1,3}, Q{2,3}. Fixing 0 to a point distribution deletes its two edges and removes it from the two triple facets, leaving Q{1,2}, Q{1,3}, Q{2,3}: the triangle `QG_Triangle1`. Most PD proofs look like this: a seed with one more node attached to it.

---

## 2. Marginalization, naive and with teleportation (`naive_marginalization`, `teleportation_marginalization`)

**Map** (`QmDAG.marginalize(node, apply_teleportation)`). Remove $v$; every visible parent of $v$ becomes a parent of every visible child of $v$; every facet containing $v$ loses $v$ (a quantum facet stays quantum on $F\setminus v$) and in addition a *classical* facet over $(F\setminus v)$ and all visible children of $v$ is added; a new classical facet over the visible children of $v$ is added as well (the private randomness of $v$). With teleportation, additionally every quantum facet $F\ni v$ is enlarged to a quantum facet on $(F\setminus v)\cup T$, where $T$ is the set of visible children of $v$ that share some quantum facet with $v$. $\pi$ is the marginal over $v$.

**Pull-back.** Without teleportation $G'$ is the latent projection of $G$ over $v$ (Evans), and the marginal of a classical model of $G$ is a classical model of the projection: the kernel of $v$ is absorbed into its children, its latent parents are relayed to its children, and its private randomness becomes a common cause of its children. With teleportation $G'$ has more facets than the projection, and $\mathcal C$ only grows with facets. $\square$

**Lift.** Given $P'\in\mathcal Q(G')$: $v$ receives its visible parents and forwards them to its children (the new visible edges), forwards the classical values its facets carry (a quantum facet may carry the classical string that the new classical facet over $(F\setminus v)$ and the children uses in $G'$) and broadcasts a private random string (the new classical facet over the children). For a quantum facet $F\ni v$ that in $G'$ extends to the children in $T$: $v$ holds a composite share of the $G'$-state, one subsystem per child in $T$; $v$ and each such child $c$ share some quantum facet, which may carry an extra maximally entangled pair, and $v$ teleports the subsystem destined for $c$ through that pair and the visible edge $v\to c$. The children then hold exactly the resources of $G'$. $\square$

*Order dependence.* Teleportation is not symmetric, so marginalizing several nodes in different orders can give different structures; wherever the code removes several nodes it enumerates every order (`QmDAG._marginalize_predictors`).

**Example** (provable only through marginalization; one of the 24 such inputs of the cheap phase, 7.7). Input: 0→1→2→3; Q{1,3}. Node 2 relays the outcome of 1 to 3 and has no latent of its own. Marginalizing it relays the edge, 0→1→3; Q{1,3}, which is `QG_Instrumental1`. Nothing else works: PD on 2 leaves 3 without an input; conditioning on 2 is inadmissible because the grandparent 0 is not a parent of 2, and after PD on 0 it is still inadmissible by condition 2 of Section 3, since the facet Q{1,3} contains the parent 1 but not 2. Both marginalizations give the same output here, since 2 has no quantum facet. The same pattern, a chain through a latent-free relay node, accounts for all three inputs; another is 0→1→2; Q{0,2}, Q{0,3}, where marginalizing 1 relays the setting 0 to 2 and gives `QG_Instrumental3`.

---

## 3. Conditioning (`conditioning`)

**Map** (`QmDAG.condition(node)`). Remove $X$; add a classical facet over $\mathrm{Pa}(X)\cap V$ together with the latent siblings of $X$ (the visible nodes sharing some facet with $X$); add a quantum facet over the nodes sharing a quantum facet with $X$. $\pi(P)=P(\cdot\mid X=x_0)$ for a suitable value $x_0$.

**Admissibility** (`QmDAG.conditioning_is_justified`). Writing $B=\mathrm{Pa}(X)\cap V$ for the visible parents and $S$ for the latent siblings of $X$:

1. every visible grandparent of $X$ is a visible parent of $X$, so $B$ is ancestrally closed among visible nodes;
2. every facet containing a node of $B$ either contains $X$ or is contained in $B$;
3. every visible child of a parent $p\in B\setminus S$ (a parent sharing no facet with $X$) lies in $B\cup S\cup\lbrace X\rbrace$.

Conditions 1 and 2 say that no grandparent of $X$, visible or latent, may fail to be a parent of $X$, except for latents whose children all lie in $B$, which only redistribute the parent block. Condition 3 comes from the lift below: a parent that shares no facet with $X$ can only learn the post-selected common cause by guessing it and announcing the guess in its output, and that fine-grained output must not be readable by nodes outside the new facet. Condition 2 is needed for the pull-back as stated; whether the implication itself can fail without it is not settled, but the pull-back lemma does fail. Take $X$ with parents $p_1,p_2$, each $p_i$ sharing a classical facet with an outside node $o_i$, and the classical model $p_i=\lambda_i\oplus\varepsilon_i$ with biased noise, $o_i=\lambda_i$, $X=[p_1=p_2]$. Conditioning on $X=1$ makes $o_1$ and $o_2$ dependent ($I(o_1{:}o_2)\approx0.046$ bits), while $G'$ d-separates them; so $\pi(\mathcal C(G))\not\subseteq\mathcal C(G')$, which is what the pull-back needs (`tests/test_conditioning.py`).

**Pull-back.** Let $\Lambda_X$ be the latents of $X$ and $\Lambda_B$ the latents whose children all lie in $B$. By conditions 1 and 2 every input of a node in $B$ lies in $B$, in $\Lambda_X$ or in $\Lambda_B$; the latents of $\Lambda_B$ influence nothing outside $B$; and every other latent is independent of $(B,X,\Lambda_X,\Lambda_B)$. In a classical model of $G$ the conditional distribution therefore factorises as

```math
P(\text{all}\mid x_0)\ =\ P(\Lambda_X,\Lambda_B\mid x_0)\cdot P(B\mid \Lambda_X,\Lambda_B,x_0)\cdot\Big[\prod_{\text{other nodes}}\text{kernels}\Big]\cdot P(\text{other latents}),
```

where the block distribution $P(B\mid\Lambda_X,\Lambda_B,x_0)$ is proportional to $\prod_{p\in B}P(p\mid\mathrm{pa}(p)\cap B,\Lambda_X,\Lambda_B)\cdot P(x_0\mid B,\Lambda_X)$: the product of the block's kernels, reweighted by the probability that $X$ takes the value $x_0$. Define the new latent $\mu=(\Lambda_X,\Lambda_B,R)$, where $(\Lambda_X,\Lambda_B)$ is drawn from the posterior $P(\Lambda_X,\Lambda_B\mid x_0)$ and $R$ is fresh uniform randomness. The latent siblings of $X$ read $\Lambda_X$ through $\mu$; this is legitimate because every facet containing $X$ lies within $S\cup\lbrace X\rbrace$, so nobody else reads $\Lambda_X$ (the facets $F\setminus X$ that $G'$ retains are set trivial). Visible children of $X$, if any, lose the edge from $X$ and read the constant $x_0$ instead. The block $B$ is sampled jointly, as deterministic functions of $\mu$ alone, from the reweighted block distribution: all inputs of the block are available in $\mu$, so the shared seed $R$ lets each $p$ compute its coordinate of one joint sample. Every other node keeps its kernel. The result is a classical model of $G'$ for $P(\cdot\mid x_0)$. Without condition 2 some $p$ has a latent shared with outsiders, the reweighted block depends on inputs the other block members cannot see, and the counterexample above is exactly this failure. $\square$

**Lift.** Given $P'\in\mathcal Q(G')$ with the new classical facet $\mu$ and the new quantum state $\rho$ over the quantum siblings $S_Q$ of $X$. In $G$: every facet $F\ni X$ carries an independent uniform copy $\mu_F$ of the string $\mu$; every quantum facet $F\ni X$ additionally carries one maximally entangled pair between $X$ and each quantum sibling in $F$. Latent siblings, and parents in $B\cap S$, read $\mu_F$ from a facet they share with $X$. $X$ prepares $\rho$ locally and teleports each subsystem to its sibling through the corresponding pair; post-selected teleportation, keeping only the identity outcome of each Bell measurement, leaves $S_Q$ holding $\rho$ exactly. A parent $p\in B\setminus S$ has no facet with $X$: it draws a private guess $\hat\mu_p$, runs its $G'$ strategy with $\mu:=\hat\mu_p$, and outputs $(p,\hat\mu_p)$ so that $X$ can check the guess. $X$ sets $X=x_0$ iff all $\mu_F$ and all $\hat\mu_p$ coincide and every Bell outcome is the identity. Conditioned on $X=x_0$ the strings are a single uniform $\mu$ shared by $B$ and the siblings, $S_Q$ shares $\rho$, and every node has run its $G'$ strategy.

The guessing parents' outputs are fine-grainings $(p,\hat\mu_p)$ of their $G'$ outputs, and this is where condition 3 is needed. Coarse-graining a node's output is not in general an operation under which $\mathcal C(G')$ is closed: a classical model for the fine-grained distribution may let the children of $p$ read $\hat\mu_p$ through the visible edge, and no model for the coarsened distribution need exist (Section 5.2 gives a structure where exactly this fails). Under condition 3 the children of a guessing parent are $X$, other parents or latent siblings, all of which read $\mu$ from the new facet of $G'$, so in the classical model of $G'$ that the pull-back constructs, where $\hat\mu_p=\mu$ is a function of the new facet, every use of the second component of $p$'s output can be rerouted through the facet, and the coarsened distribution $P'$ is in $\mathcal C(G')$ whenever the fine-grained one is. $\square$

**Example** (provable only through conditioning; 23 such inputs). Input: 2→3; Q{0,2}, Q{0,3}, Q{1,2}. Node 0 has no parents, so all three conditions hold vacuously. Conditioning on 0 removes it and adds a quantum facet over its quantum siblings 2 and 3 (the classical facet over the same pair is absorbed), giving 2→3; Q{1,2}, Q{2,3}: `QG_Instrumental3`. Operationally the lift is entanglement swapping: 0 performs a Bell measurement on its two shares and the post-selected outcome leaves 2 and 3 entangled. Among the reductions only conditioning and teleportation marginalization create a facet between nodes that did not share one, and teleportation needs a visible edge from 0 to a quantum sibling, which is absent here.

---

## 4. Interruption (`interruption`)

**Map** (`QmDAG.interruption_creation(node_with_no_children=y, node_with_no_parents=x)`). $x$ is exogenous (no visible parents, no non-singleton facet: `QmDAG.exogenous_visible_nodes`) and $y$ is not a descendant of $x$, so that $G'$ is acyclic. Remove $x$; its visible children become children of $y$. The search additionally restricts $y$ to childless nodes; the proof does not need this. $\pi$ is the renormalised diagonal $x=y$ of $P$:

```math
\pi(P)(y,\text{rest})=\frac{P(x{=}y,\;y,\;\text{rest})}{P_x(y)},
```

where $P_x$ is the marginal of $x$ (uniform for every lifted distribution, so the denominator is then a constant).

**Lift.** Given $P'\in\mathcal Q(G')$, in $G$ let $x$ be uniform, let the former children of $x$ treat the value of $x$ as they treated $y$ in $G'$, and let every other node run its $G'$ strategy. Then $P(x,y,\text{rest})=P_x(x)\,P'^{\,do(y:=x)}(y,\text{rest})$, where the right factor is the distribution of the $G'$ strategy with the input that the former children of $x$ now take from $y$ fixed to $x$ (an edge intervention). Fixing that input to the value $y$ actually takes changes nothing, so $P'^{\,do(y:=y)}=P'$ and $\pi(P)=P'$.

**Pull-back.** Let $M$ be a classical model of $G$ for a distribution with $x$ uniform; $x$ is independent of all latents of $M$. Define a model of $G'$ in which every node keeps its kernel, except that the former children of $x$ receive $y$ in place of $x$. Since $y\notin\mathrm{desc}(x)$, the value $Y(\lambda)$ does not depend on $x$ and the new model is acyclic; its distribution is $\sum_\lambda p(\lambda)[y=Y(\lambda)]\,[\text{rest}=f(\lambda,x{:=}Y(\lambda))]$, which is the diagonal $x=Y(\lambda)$ of $M$'s distribution divided by $P_x(y)$, that is $\pi(P)$. Other children of $y$ read $y$ in both structures. $\square$

**Example.** `QG_Bell1` is 0→2, 1→3; Q{2,3}. Interruption with $x=1$ (exogenous) and $y=2$ (childless, not a descendant of 1) removes 1 and makes 3 a child of 2: 0→2→3; Q{2,3}, which is `QG_Instrumental1`. The post-selection $x=y$ identifies Bob's setting with Alice's outcome. Read as a piggyback: a gap in the instrumental scenario implies a gap in the Bell scenario. Likewise `QG_Bell3b` (0→2, 1→3; Q{1,3}, Q{2,3}) maps to `QG_Instrumental2` with $x=0$, $y=3$, and `QG_Bell9`, `QG_Bell9b` map to `QG_Instrumental3b`, `QG_Instrumental3`.

---

## 5. The Fritz piggyback (`Fritz`)

**Notation.** The predictor is in general a **set** of visible nodes, written $\mathbf X$; its members are the predictors. $\mathrm{Pa}(\mathbf X)=\bigcup_{x\in\mathbf X}\mathrm{Pa}(x)$. The nodes **seen** by $\mathbf X$ are $\mathrm{Pa}(\mathbf X)\cup\mathbf X$. For a predicted node $s$,

```math
\mathrm{common}(s)=\mathrm{Pa}(s)\cap\big(\mathrm{Pa}(\mathbf X)\cup\mathbf X\big),\qquad
\mathrm{others}(s)=\mathrm{Pa}(s)\setminus\mathrm{common}(s),
```

the latter always containing the private noise of $s$. Structures in examples are written as a list of visible edges and facets, with Q for quantum and C for classical facets, e.g. "0→2, 1→2; Q{0,1}, Q{1,3}, Q{2,3}".

### 5.1 The mechanism

Both Fritz piggybacks are instances of one scheme:

> choose a set of edges of $G$ to delete, giving $G'\subseteq G$, and justify the deletion by a **perfect prediction** that the quantum lift can arrange and that forces the deleted dependences to be idle in every classical model.

Fix a predictor set $\mathbf X$ and a predicted node $s\notin\mathbf X$ sharing at least one facet with some predictor. Since $\mathrm{others}(s)$ always contains the private noise of $s$, no predictor may be a descendant of $s$: the d-separation test below fails for such a predictor. The **candidate** $G_1$ restricts $s$ to $\mathrm{common}(s)$ and leaves everything else in place, predictors included: visible co-parents are kept; facets of $\mathrm{common}(s)$ are kept but become classical *for $s$* (a quantum facet $F$ keeps its quantum part on $F\setminus\lbrace s\rbrace$ and a classical facet over $F$ is added, which is the state $\rho\otimes\sigma$ with $\sigma$ classical). The **output** $G'$ of the piggyback is $G_1$ with the predictors removed, deleted if childless and marginalized otherwise (Section 2), or kept in the split form of Section 5. Apart from the added classical facet $G_1\subseteq G$, so by (F1) the lift from $G_1$ to $G$ is the construction of the prediction, and the entire content is the pull-back: a classical model of $G$ in which the prediction holds, and in which the other observable properties of the lifted distribution hold, is also a classical model of $G_1$. Throughout, "Markov($H$)" for a DAG $H$ means the local Markov property, each node independent of its non-descendants given its parents, which for a DAG is equivalent to the factorisation of the joint into one kernel per node.

The lift lets $s$ be a deterministic function $f$ of $\mathrm{common}(s)$, with its private randomness drawn from a facet shared with a predictor, and lets each predictor output the part of $f(\mathrm{common}(s))$ computable from what it sees, so that $\mathbf X$ jointly determines $s$. This is why $s$ must share a facet with a predictor (noise absorption) and why only $\mathrm{common}(s)$ may be kept: the predictors must be able to compute $s$.

### 5.2 Node splitting and the two predictor modes (`QmDAG.split_node`)

**Map.** Replace $s$ by two nodes $s,s'$ with the same parents and the same children, sharing a classical two-party facet $\lbrace s,s'\rbrace$ (their common private randomness; absorbed when $s$ already lies in a facet). $\pi$ merges $(s,s')$ into one node.

**Lift.** $s$ computes both outputs from its inputs and private randomness. **Pull-back.** A classical kernel for the pair $(s,s')$ splits into two kernels sharing the randomness carried by the new facet. $\square$ When the absorbing facet is quantum the pair's shared randomness is carried by that quantum facet, which is legitimate because the pull-back concerns classical models only.

Node splitting is never run on its own: the split structure has one more visible node and no new independences, so it is only a stepping stone. It enters the Fritz piggybacks in two different roles, and the difference matters.

* **Splitting a predictor.** The Fritz lift makes each predictor $x\in\mathbf X$ output, alongside its own value, the part of $f(\mathrm{common}(s))$ it can compute: the predictor's output is *fine-grained*. The pull-back ends with a classical model of $G'$ in which $x$ still has the fine-grained output, and the proof must get rid of the extra component. If $x$ is dropped from $G'$, deleted when childless or marginalized otherwise, the component is integrated out and nothing more is needed (`drop` predictor mode). If $x$ is to stay in $G'$, the component has to be coarse-grained away, and **coarse-graining the output of a node with children is unsound**: a classical model of $G'$ for the fine-grained distribution may let the children of $x$ read the prediction through the visible edge, and no model for the coarsened distribution need exist. 

  Example: 0→1; Q{0,1}, Q{0,2}, Q{1,2} is saturated (latent-free equivalent, so it has no gap), yet "0 predicts 2, 0 kept untouched" would output 0→1; C{0,2}, Q{0,1}, which is `QG_Instrumental3b`: node 1 learns the instrument 2 from 0's fine-grained output, which is exactly what the Bonet inequality forbids the binary treatment to transmit. 

  The sound way to keep a predictor is node splitting with a *full* copy: $x$ is split into $x$ and $x'$ with the same parents and the same children, $x'$ is the predictor, and $x'$ is dropped as above. For a childless $x$ this coincides with keeping $x$ untouched (the childless copy is deleted). For a childful $x$ the copy is marginalized, which relays to $x$'s children everything the prediction could have told them; on the example it gives 0→1; C{0,1,2}, Q{0,1}, where the instrument also shares randomness with the outcome, no gap. This is the `split` predictor mode of the code, called the *kept* predictor mode below (`QmDAG._split_predictors`, trick `Fritz_kept`); `drop` is the *dropped* mode.
* **Splitting the predicted node** (`copy` mode). Here $s$ is split and the deletion is applied to the copy $s'$, so the original keeps all of its parents while $s'$ keeps $\mathrm{common}(s)$. The copy must keep the children of $s$: the pull-back turns a classical model in which $s$ outputs a pair $(s_1,s_2)$ into a model with $s:=s_1$ and $s':=s_2$, and the children of $s$ may depend on $s_2$. Removing an edge $s'\to c$ would be an unjustified deletion; those edges can only go later, by a piggyback that justifies it. `fritz_transitions` builds the copy directly in `_fritz_build`; `fritz_entropic_transitions` literally calls `split_node` first and then runs replace mode on the copy. The two constructions coincide because the pair's shared facet is absorbed by the facet the copy shares with the predictors.

Both predicted-node modes (`replace`, `copy`) and both predictor modes are available to both certificates. The search names its tricks by predictor mode and certificate (Section 7): `Fritz` drops the predictors and certifies by d-separation; `Fritz_kept` keeps them as above and certifies by d-separation; `Fritz_entropic` emits only steps whose justification needs the LP (Section 6), in either predictor mode. Kept predictors reach strictly more (the KPC construction of 6.7 keeps its childless predictor), but a kept-predictor output has as many visible nodes as its source, so an exhaustive closure under kept predictors is far larger than under dropped ones; the search applies the kept-predictor tricks at depth one, once to each input the cheaper stages leave unproven (Section 7). The cascade of 7.8 separates the contributions of the predictor mode, the predicted-node mode and the certificate, and says where kept predictors prove something dropped predictors cannot.

### 5.3 Theorem (d-separation certificate)

**Claim.** Let $\mathbf X$ and $s$ be as above and suppose, in the effective DAG of $G$ (private noise nodes included),

```math
\mathbf X\setminus\mathrm{common}(s)\ \perp_d\ \mathrm{others}(s)\ \big|\ \mathrm{common}(s).
```

(A predictor that is itself a parent of $s$ lies in $\mathrm{common}(s)$ and is conditioned on rather than tested; if every predictor is a parent of $s$ the condition is vacuous.) Let $G'$ be the output of 5.1 with the predictors removed, deleted if childless and marginalized otherwise, in any order. Then a QC gap in $G'$ implies a QC gap in $G$. The same holds for the kept form of Section 5.2 (a childless predictor untouched; a childful one split, the copy being the predictor that is marginalized), by applying the statement to the split structure and composing with the node-splitting piggyback.

**Proof.** *Lift:* given $P'\in\mathcal Q(G')$, first lift it to the candidate $G_1$ through the removal of the predictors: a deleted predictor outputs a constant (Section 1), a marginalized one runs the lift of Section 2 (it forwards its visible parents, broadcasts the classical strings and teleports its quantum shares to its children). This gives $P_1\in\mathcal Q(G_1)$ with $P'$ as its marginal over the predictors. In $P_1$ the parents of $s$ are classical and $s$ shares a facet with a predictor, so $s$ can be realised as a deterministic function $f(\mathrm{common}(s))$ with its private randomness drawn from that facet. Now run $P_1$ in $G$ and let each predictor additionally output what it sees of $f(\mathrm{common}(s))$. The result $P$ lies in $\mathcal Q(G)$, satisfies $H(s\mid\mathbf X)=0$, and has $P'$ as the marginal over the predictors.
*Pull-back:* let $M$ be a classical model of $G$ for $P$. Write $s=h(\mathrm{common},\mathrm{others})$ with $\mathrm{others}$ including the private noise, and $s=g(\mathbf X)$. Condition on $\mathrm{common}(s)$. By the d-separation hypothesis and (F2) for classical models, $\mathrm{others}(s)$ and $\mathbf X$ are independent given $\mathrm{common}(s)$. A random variable that is a function of each of two independent variables is almost surely constant, so $s=\varphi(\mathrm{common}(s))$ a.s. Replace the kernel of $s$ in $M$ by $\varphi$; the joint distribution is unchanged and the new model is Markov to $G_1$, the predictors still present with their fine-grained outputs. Now remove the predictors: marginalizing a node of a classical model gives a classical model of the latent projection (Section 2), which is $G'$, for the marginal of $P$, which is $P'$; a childless predictor is simply deleted. Hence $P'\in\mathcal C(G')$. $\square$

The set test is equivalent to the conjunction of the per-edge tests $\mathbf X\perp_d v\mid\mathrm{Pa}(s)\setminus\lbrace v\rbrace$ over $v\in\mathrm{others}(s)$, by the weak-union and intersection properties of d-separation. The implementation is `QmDAG.fritz_admissible_targets`, which tests the set at once with networkx's `is_d_separator`; the structure is built by `QmDAG._fritz_build` from an explicit map of kept parents.

### 5.4 The unit of deletion

The piggyback deletes all of $\mathrm{others}(s)$ or nothing. Deleting a single parent $p\to s$ while $s$ keeps another parent $q$ that $\mathbf X$ cannot see is not a candidate: the predictors cannot predict $s$, and no other piggyback justifies the lone deletion. Example: 0→2, 1→2; Q{0,1}, Q{1,3}, Q{2,3}. Predictor 3 deletes 0→2 and 1→2 together and reaches `QG_Bell6c` (an entropic step, Section 6.8), whereas neither structure with only one of the two edges deleted is reachable from the input by any trick of the toolkit, out of 760 reachable structures. Across the four-node census, of the single-target Fritz steps that delete two or more non-noise parents, more than half have no reachable single-edge intermediate.

What *does* decompose is the choice of several predicted nodes for one *childless* predictor set. With such predictors kept, the predicted nodes can be restricted one at a time and the predictors deleted afterwards, so for childless predictors the subset enumeration in `fritz_transitions` is a convenience of the dropped mode, not a logical necessity. For a childful predictor the kept form marginalizes a split copy, which adds classical facets, so the sequential route gives a different structure from the joint dropped-mode step. Joint predictor *sets*, on the other hand, are not reducible to single predictors: a set sees more than any member.

### 5.5 Examples

**Replace mode** (the one census input that needs a `Fritz` step in replace mode and nothing more). Input: no edges; Q{0,1,2}, Q{0,1,3}, Q{0,2,3}, Q{1,2,3}, every triple entangled. Predictor 0, predicted 1. The parents of 1 are the three facets containing it; 0 sees Q{0,1,2} and Q{0,1,3}, so $\mathrm{common}(1)$ is those two facets and $\mathrm{others}(1)$ is Q{1,2,3} plus the noise of 1. The d-separation test asks whether 0 is separated from Q{1,2,3} given the two common facets: every path from 0 to Q{1,2,3} runs through 2 or 3 as a collider (0 ← Q{0,2,3} → 2 ← Q{1,2,3}), so it holds. Node 1 is restricted to its two common facets, which become classical for it, and the childless predictor 0 is deleted: the output is C{1,2}, C{1,3}, Q{2,3}, the triangle with one quantum and two classical sources, `QG_Triangle3`. In the lift, 1 is a function of its two shares and 0, which holds the same shares, announces that function; classically, any model in which 0 predicts 1 perfectly cannot let 1 depend on Q{1,2,3}, which 0 never sees.

**Copy mode** (one of three census inputs that need a copy-mode `Fritz` step; four need copy mode in some trick). Input: 0→2, 1→2, 2→3; Q{0,1}, Q{0,2}, Q{1,3}. Predictor 0 (childful: it feeds 2), predicted 1 in copy mode. $\mathrm{common}(1)$ is Q{0,1}; $\mathrm{others}(1)$ is Q{1,3} plus noise; 0 is separated from Q{1,3} given Q{0,1} because the paths through 1 and through 2 both end in colliders. The copy 1' keeps Q{0,1}, classically, and the child 2 of 1; the original 1 keeps Q{1,3}. Dropping 0 marginalizes it with teleportation (its share of Q{0,2} goes to 2, its facets become classical for 2). Output: 1→2, 2→3, 1'→2; C{1,1',2}, Q{1,2}, Q{1,3}. Marginalizing 1 next, with teleportation of its share of Q{1,3} to 2, gives 1'→2→3; C{1',2}, Q{2,3}: `QG_Instrumental2b`. Replace mode cannot do this: it would strip Q{1,3} from 1, and that facet is the one that survives into the seed.

**Kept predictor, d-separation** (`Fritz_kept`; the one census input that needs it). Input: 0→1→2; Q{0,2}, Q{0,3}, Q{1,3}. Predictor 0, which feeds 1, predicts 3 in copy mode: $\mathrm{common}(3)$ is Q{0,3}, $\mathrm{others}(3)$ is Q{1,3} and noise, separated from 0 given Q{0,3} because 0→1←Q{1,3} is blocked at 1. Since 0 has a child, it is split into 0 and a full copy 0′ (both feeding 1); the copy predicts and is marginalized, which relays its facets to 1: the copy 3′ of 3 reads a classical facet shared by 0, 1, 3 and 3′, and 1 joins a classical facet with 0 and 2. Conditioning on 3 (no visible parents, so admissible) swaps entanglement onto 0 and 1, and marginalizing 0 with teleportation gives 1→2; C{1,3′}, Q{1,2}: `QG_Instrumental3b`. With 0 dropped instead of split, 0's facets land on 1 classically and the quantum link needed for the instrumental shape is lost.


### 5.6 The search procedure (`QmDAG.fritz_transitions`, `QmDAG.fritz_entropic_transitions`, `qc_gap_search.fritz_tricks`)

Throughout, $s$ is the predicted node and $\mathbf X$ the predicting set (a single node $X$ in the census). Stated edge-first, the unit of search is a **candidate deletion**: a visible node $s$ and a set $D\subseteq\mathrm{Pa}(s)$ of its parents in the effective DAG to delete, a parent being a visible parent or a facet containing $s$. Deleting $D$ dictates everything else.

1. **Who may predict.** After the deletion $s$ depends only on $K=\mathrm{Pa}(s)\setminus D$, so a predictor set $\mathbf X$ is usable iff $\mathbf X$ sees every element of $K$ (each kept facet contains a predictor, each kept visible parent is a parent of a predictor or is itself a predictor) and $K$ contains at least one facet shared with a predictor, which carries the private randomness of $s$ in the lift.
2. **Which conditional independences.** The candidate $G_1$ is $G$ with $D$ deleted and the predictors kept, so the observable d-separations of $G_1$, which the entropic certificate uses as hypotheses (Section 6), are fixed by $D$.
3. **What has to be proven.** That every classical model of $G$ with the prediction is a model of $G_1$: by the d-separation test of 5.3, or by the LP of Section 6.

The implementation runs this in the opposite order: for each predictor $X$ (every visible node with a latent sibling; joint predictor sets are supported by `max_predictors` but off, see below) and each latent sibling $s$ of $X$, it takes the maximal deletion $D_0=\mathrm{Pa}(s)\setminus\mathrm{common}(s)$, everything the predictor cannot see, in replace mode and in copy mode (5.2), with the predictor dropped or kept. The d-separation test of 5.3 is run first and the LP of Section 6 only where it fails. For every candidate the code records the predictor set, the predicted node and its mode, the predictor mode and which certificate closed it, so each Fritz-type step in a certificate can be re-derived by hand; the search of Section 7 applies these steps in a fixed order of increasing cost.

**Joint predictor sets.** The code supports a predictor *set* that predicts jointly (`max_predictors`): $\mathrm{common}(s)$ collects everything any member sees. In an earlier four-node census run with pairs enabled such steps were never essential, and the census now uses single predictors only. Fritz's own construction for the tetrahedron (no edges; all four triples as quantum facets) is different: each of two nodes alone predicts a copy, so the copy keeps only what *both* see, the intersection. In the search this is two sequential single-predictor steps: predictor 2 makes copies 0' and 1' and keeps Q{0,1,2}, Q{0,2,3} for 0' and Q{0,1,2}, Q{1,2,3} for 1', then predictor 3 restricts 0' to Q{0,2,3} and 1' to Q{1,2,3}; dropping both predictors leaves Q{0,1}, C{0,0'}, C{1,1'}, which is `QG_Bell6`. The tetrahedron is also proven in one step by a single predictor (the replace-mode example of 5.5), so neither route is essential for it, and in the search the two-step route is in any case beyond the depth-one cascade of Section 7 and is never the one recorded, since `build_report` prefers the shortest chain and the tetrahedron is captured by its one-step derivation of `QG_Triangle3`.

**Predictors that are parents of the predicted node.** The candidate set of every Fritz-type trick is the set of latent siblings of the predictor $X$: $s$ qualifies because a facet $L$ shared with $X$ lets the lifted $X$ learn the value $s$ will compute from $\mathrm{common}(s)$. A visible *child* $s$ of $X$ qualifies for the same reason with the edge $X\to s$ in the role of $L$: in the lift $X$ computes the restricted $s$ from what it sees and sends it down the edge, and $s$ outputs it, so $H(s\mid X)=0$ holds and $s$ depends on $\mathrm{common}(s)$ alone; the pull-back is the same as in 5.3 and 6.5, since neither the d-separation test nor the LP hypotheses care whether the channel from $X$'s inputs to $s$ is a facet or an edge, and the rule that a predicted node must keep a facet shared with its predictors (5.1) becomes that it must keep the edge or such a facet. In the output the edge is removed with $X$ when $X$ is dropped, and relayed by marginalization when $X$ is split. This would enlarge the candidate pool of every childful node; it is not implemented, and the 66 remaining inputs (Section 8), all of which have a visible edge out of a node holding quantum facets, are where it would be tried first.

---

## 6. The entropic Fritz piggyback (`Fritz_entropic`)

### 6.1 Why the same mechanism admits a stronger certificate

In the pull-back of 5.3 the only facts about the classical model that were used were the Markov property of $G$ and $H(s\mid\mathbf X)=0$. But the lifted distribution $P$ has more structure: it lies in $\mathcal Q(G_1)$, so by (F2) **every observable d-separation of $G_1$ holds for $P$**, and therefore holds in any classical model of $G$ for $P$. These conditional independences are properties of the *new* graph, obtained for free, and there is no reason not to use them as hypotheses when deciding whether a deletion is justified in the *original* graph. The d-separation test of 5.3 ignores them; it is a shortcut that trades proof power for speed. Khanna, Pusey and Colbeck's six-node example (6.7) is a case where the shortcut fails and the extra independences are exactly what is needed.

### 6.2 Entropy vectors as the proof system (`entropic_lp.py`)

For $n$ random variables the entropy vector $h\in\mathbb R^{2^n-1}$ lists $H(S)$ for every nonempty $S$. Every entropy vector satisfies the **elemental Shannon inequalities** $H(i\mid[n]\setminus i)\ge0$ and $I(i{:}j\mid K)\ge0$ for $i\lt j$ and $K\subseteq[n]\setminus\lbrace i,j\rbrace$, which number $n+\binom n2 2^{n-2}$ (`elemental_inequalities(n)`; the $n=3$ matrix is checked against the Mathematica notebook of Khanna, Pusey and Colbeck (KPC), the row count for $n\le8$). A conditional independence is the vanishing of a conditional mutual information, a functional dependence the vanishing of a conditional entropy; both are nonnegative on the cone, so each hypothesis is one inequality "$\le0$". The local Markov property of a DAG is one conditional mutual information per node, $I(v{:}\mathrm{nondesc}(v)\setminus\mathrm{pa}(v)\mid\mathrm{pa}(v))=0$ (`local_markov_rows`). The Shannon cone with these hypotheses implies exactly the d-separations of the DAG: every d-separation follows from the local Markov statements by the semigraphoid axioms (Verma and Pearl), which are Shannon-derivable, and nothing beyond the d-separations can follow because classical models realise every non-d-separated dependence (F2). This is tested on random DAGs.

A target functional $t$ is **implied** when the LP $\lbrace\text{Shannon}\ge0,\ \text{hypotheses}\le0,\ t\ge1\rbrace$ is infeasible (the cone is scale invariant). Geometrically the question is **cone inclusion**: the hypotheses cut out a polyhedral cone $K$, each target is a halfspace, and the certificate asks whether $K$ lies in the intersection of the target halfspaces, that is, inside the cone of entropy vectors Markov to the target graph. Both cones are given by inequalities, and inclusion of one inequality description in another needs no conversion to extreme rays (as cdd, lrs or PANDA would do; that is exponential and unnecessary). It needs one LP per target set, not one per target: every target is a conditional mutual information, nonnegative on the Shannon cone, and a sum of nonnegative quantities vanishes iff each term does, so all targets are implied iff their **sum** is implied (`EntropicLP.implies_all`, `sum_rows`). The Farkas certificate of the summed row certifies every target at once. Solving one LP in place of $n$ per target set is where the time of the entropic stage goes, so this is roughly an $n$-fold saving. Infeasibility is decided with the Mosek Optimizer API (`EntropicLP`, interior point); the dual ray is a Farkas certificate, a nonnegative combination of elemental inequalities and hypotheses that reproduces $t$, available through `EntropicLP.farkas_certificate`. Implications proven this way are valid for every distribution, with or without full support, because they use only Shannon inequalities. The method is incomplete in the other direction: a feasible LP does not exhibit a distribution, only a vector in the Shannon cone.

### 6.3 The certificate and its two target sets (`QmDAG._entropic_certificate`)

Here $s$ is the predicted node and $\mathbf X$ the predicting set. Variables: all nodes of `lp_structure`, the visible nodes and the facets of $G$; quantum facets are ordinary latent variables here, since only classical models are analysed (F2). Hypotheses: Shannon; local Markov of $G$; $H(s\mid\mathbf X)\le0$; and $\mathcal I(G_1)$, the set of elementary observable d-separations $I(x{:}y\mid Z)=0$ of the candidate $G_1$ over its visible nodes (`observable_dseparation_rows`). Two target sets are tried, each as one LP on its summed row (6.2), in this order: `relabel` first, since it is the certificate that reaches beyond d-separation in practice (7.8), and `markov` only when `relabel` is inapplicable or fails.

* **`relabel`**: applicable when there is a single predicted node $s$ and it keeps exactly one parent in $G_1$, a facet $L$ (the code tests `len(predicted) == 1`, `len(common) == 1` and that the element is a facet index). Let $G''$ be $G_1$ with $L$ deleted and $s$ made a parent of every other child of $L$. Targets: the local Markov equalities of $G''$ over $G$'s variables other than $L$.
* **`markov`**: the local Markov equalities of $G_1$ over $G$'s own latents. If all are implied, the joint of any classical model of $G$ satisfying the hypotheses is Markov to $G_1$, hence $P\in\mathcal C(G_1)$.

`Fritz_entropic` emits only LP-reliant steps, those certified by `markov` or `relabel`; a candidate that plain d-separation certifies is left to `Fritz` and `Fritz_kept`, so the three trick names partition the Fritz-type steps by the justification they need.

### 6.4 Why `relabel` is justified

The claim to be proven is not about one classical model. It is: *every* classical model of $G$ whose observed statistics satisfy the perfect prediction and the independences $\mathcal I(G_1)$ can be exchanged for a classical model of $G_1$ with the same observed statistics. The argument has three steps.

1. **Fix one model.** Any classical model $M$ of $G$ producing $P$ defines one joint distribution $Q$ over the visible nodes and the facet latents (noises integrated out). The hypotheses are facts about $Q$: $Q$ is Markov to $G$; $H(s\mid\mathbf X)=0$; the independences of $\mathcal I(G_1)$ hold among the visible nodes. Shannon's inequalities hold for $Q$, so the entropy vector of $Q$ satisfies every LP hypothesis.
2. **What infeasibility proves.** The LP shows that every entropy vector satisfying the hypotheses has each target conditional mutual information equal to zero. A vanishing conditional mutual information is a conditional independence. So the targets hold for $Q$, and since $M$ was arbitrary they hold for every model satisfying the hypotheses.
3. **From Markov($G''$) to a model of $G_1$.** The targets are the local Markov conditions of $G''$ over $W=V\cup(\text{facets}\setminus\lbrace L\rbrace)$. For a DAG, the local Markov conditions are equivalent to the factorisation of $Q(W)$ into one kernel per node given its parents in $G''$, so the marginal of $Q$ over $W$ *is* a classical model of $G''$, one in which $s$ is a root. Now rename. Introduce a fresh latent $\lambda'$ carrying the value of $s$, with the root distribution $Q(s)$; let $s$ read $\lambda'$ deterministically (its private noise in $G_1$ may be trivial); let every other former child $c$ of $L$ keep its $G''$ kernel with $s$ replaced by $\lambda'$, which is legal because the parents of $c$ in $G_1$ are its parents in $G''$ with $s$ replaced by $L$; the kept quantum part of the facet over $F\setminus\lbrace s\rbrace$ is left unused; everything else is unchanged. This is a classical model of $G_1$ with observed marginal $Q(V)=P$. The original latent $\lambda_L$ of $M$ is simply discarded: in the new model the common cause of $s$ and its siblings is played by the value of $s$ itself.

In the KPC example (6.7) this reads: every model in which $E$ depends on $C$ and $D$, with $F$ predicting $E$ and $E$ independent of $(C,D)$, is replaced by a model in which $E$ depends on a single latent $A'$ alone, where $A'$ is the old value of $E$.

**Why `markov` cannot do this.** The model $E=A\oplus B$, $D=C\oplus B$, $F=(A,B)$ satisfies every hypothesis of the KPC candidate, yet $E$ is not a function of the given $A$. No certificate that keeps $A$ as the witness can succeed; `relabel` succeeds because it discards $A$.

**Why the single-parent condition.** If $s$ kept another parent $v$ in $G_1$, then $s$ would not be a root of $G''$, and $\lambda':=s$ would be correlated with $v$, violating the independence of roots in $G_1$.

**Neither target set implies the other.** `relabel` without `markov` is the KPC example. `markov` without `relabel`: let $s$ read a visible parent $v$ and one facet $L=\lbrace s,c,d,x\rbrace$ shared with the predictor $x$. d-separation certifies deleting $v\to s$, so `markov` holds, but the `relabel` targets contain $d\perp c\mid s$, which the model $L=(L_1,L_2)$, $s=L_1$, $c=d=L_2$, $x=L$ violates with $I(d{:}c\mid s)=1$ bit while satisfying every hypothesis (`tests/test_fritz_entropic.py`). Both target sets are instances of one scheme, replace each latent of $G_1$ by a set of $G$'s variables and ask for Markov($G_1$) with the substitution: `markov` substitutes $\lbrace L\rbrace$, `relabel` substitutes $\lbrace s\rbrace$. Further substitutions are an open direction (Section 8).

**Relation to the d-separation certificate.** The pull-back of 5.3 derives $s\perp\mathrm{others}\mid\mathrm{common}$ and keeps the given latents, so Theorem 5.3 always produces a `markov`-type witness and never a relabelling. The LP subsumes it at least for the row of $s$: if $\mathbf X\perp_d\mathrm{others}\mid\mathrm{common}$ then the Shannon cone with the Markov hypotheses derives $I(\mathbf X{:}\mathrm{others}\mid\mathrm{common})\le0$, and with $H(s\mid\mathbf X)\le0$ and the Markov row of $s$ it derives $H(s\mid\mathrm{common})\le I(s{:}\mathbf X\mid\mathrm{common})\le I(\mathrm{others}{:}\mathbf X\mid\mathrm{common})+I(s{:}\mathbf X\mid\mathrm{common},\mathrm{others})=0$, which implies the `markov` row of $s$. The `markov` rows of other nodes, whose non-descendant sets grow in $G_1$, were checked empirically (every d-separation-admissible pair in the test structures is LP-admissible); the search never relies on this, since d-separation-certified candidates are admitted by Theorem 5.3 without an LP. `relabel` has no d-separation analogue in $G$: its hypotheses $\mathcal I(G_1)$ are facts about $G_1$, not about the graph of $G$.

### 6.5 Theorem (entropic certificate)

**Claim.** Let $G_1$ be a candidate as in 5.1, possibly with further parents deleted from any non-predictor node (the extra deletions of Section 8), and $G'$ the output with the predictors removed as in 5.3. If the LP of 6.3 implies the `markov` targets, or the single-parent condition holds and it implies the `relabel` targets, then a QC gap in $G'$ implies a QC gap in $G$.

**Proof.** Lift as in 5.3: $P\in\mathcal Q(G)$ with $H(s\mid\mathbf X)=0$; since the lifted strategy is a strategy for $G_1$ with fine-grained predictor outputs, (F2) gives that $P$ satisfies every observable d-separation of $G_1$. Let $M$ be a classical model of $G$ for $P$. Its joint entropy vector lies in the Shannon cone, satisfies the local Markov equalities of $G$ and the hypotheses $H(s\mid\mathbf X)=0$ and $\mathcal I(G_1)$. The LP implication forces the target quantities to vanish, so the joint of $M$ is Markov to the restricted structure with the predictors present (or to its relabelled version, and then 6.4 gives such a model). Removing the predictors as in 5.3, by deletion or marginalization, gives $P'\in\mathcal C(G')$. $\square$

Several predicted nodes for the same $\mathbf X$ can be handled jointly (the "joint targets" of `ENTROPIC_STATS`, 7.9): one hypothesis $H(s_i\mid\mathbf X)\le0$ per node, one candidate $G_1$ restricting all of them, and only the `markov` target set.

### 6.6 LP size and time

Here $s$ is the predicted node and $X$ the predictor, as in 5.6. An LP over $n$ variables (visible nodes plus facets, after any splitting) has $2^n-1$ columns and $n+\binom n2 2^{n-2}$ elemental rows; measured solve times on one core are about 0.2, 0.5, 2 and 9 seconds for $n=10,11,12,13$, and memory stays below a gigabyte up to $n=13$. The number of variables is not capped. Instead every solve carries a Mosek time limit of 60 seconds (`EntropicLP(max_time=60.0)`, `optimizer_max_time`); a solve that hits it is counted (`entropic_lp.TIMEOUTS`) and treated as undecided, which is the conservative answer: the implication is then not claimed. The census report states how many solves timed out (7.9).

### 6.7 Worked example: the KPC structure

$G_1$: visible $C,D,E,F$; facets $A=\lbrace E,F\rbrace$, $B=\lbrace D,F\rbrace$; edges $C\to D$, $C\to E$, $D\to E$. Predictor $F$, predicted $E$: $\mathrm{common}(E)=\lbrace A\rbrace$, $\mathrm{others}(E)=\lbrace C,D,\text{noise}\rbrace$. The d-separation test fails, since $F\leftarrow B\to D$ is open. The candidate $G_1$ has $E$ with parent $A$ only and d-separates $E$ from $\lbrace C,D\rbrace$. With the hypothesis $E\perp CD$ the LP certifies the `relabel` targets ($A:=E$), reproducing KPC's Lemma 1 without the equality $F_S=E$: their split node $F_S$ is exactly the childless copy that keeping the childless predictor $F$ implicitly deletes (Section 5.2), so keeping $F$ is sound. With $F$ kept the output is C→D; Q{D,F}, C{E,F}: the Bell variant `QG_Bell5`, in one step (`tests/test_fritz_entropic.py`). With $F$ dropped, $E$ is left with a private facet only and nothing is learnt: this is a case where kept predictors are essential.

### 6.8 Further examples from the census

Five inputs are proven only at the LP rungs of the cascade (7.8), all at the rung with kept predictors in replace mode. All five use a childless predictor $X$ that is kept and the `relabel` target set for the predicted node $s$, and each reaches a Bell seed in one step. Two of them:

**One step to `QG_Bell9`.** Input: 0→1→2; Q{0,2}, Q{1,3}, Q{2,3}. Predictor 3 (childless, kept), predicted 2: $\mathrm{common}(2)$ is Q{2,3}; $\mathrm{others}(2)$ is the visible parent 1, the facet Q{0,2} and noise. d-separation fails because 3←Q{1,3}→1 is open. In the candidate $G_1$, where 2 reads Q{2,3} alone, node 2 is separated from 0 and 1 (every path leaves 2 through Q{2,3} to 3, where it meets a collider), so $I(2{:}01)=0$ is a hypothesis. With it the LP derives the `relabel` targets: the joint is Markov to $G''$ in which 2 is a root and a parent of 3. Output: 0→1; C{2,3}, Q{1,3}, which is `QG_Bell9` (party 1 with setting 0, party 3 whose "setting" 2 is a classical copy correlated with it).

**One step to `QG_Bell6c`.** Input: 0→2, 1→2; Q{0,1}, Q{1,3}, Q{2,3}. Predictor 3 (childless, kept), predicted 2: both visible parents 0 and 1 are deleted at once (5.4), Q{2,3} is kept and becomes classical for 2. d-separation fails through 3←Q{1,3}→1→2. In $G_1$ node 2 is separated from 0 and 1, and the LP certifies the `relabel` targets. Output: Q{0,1}, C{2,3}, Q{1,3}, which is `QG_Bell6c`.

Dropping the predictor instead would delete 3 and with it the only facet 2 keeps, proving nothing, so these five inputs are exactly where kept predictors matter for the LP trick (7.8). The `markov` target set, which certifies many candidate steps during the search (7.9), is never decisive in four nodes; neither were the extra deletions in the earlier run that had them on (Section 8).

---

## 7. Search, certificates and the four-node census (`qc_gap_search.py`, `Special Applications/proving_QC_Gaps.py`)

### 7.1 The explorer and its certificates

`ClosureExplorer` expands every reachable structure, up to relabelling, exactly once under a set of tricks and records each `Transition(trick, params, source, target)`. Every transformation builds a `LabelledDirectedStructure`/`LabelledHypergraph` over named nodes and re-indexes them; copies are named `"<s>_copy"` during construction. Each unlabelled id has one stored representative, every trick is applied to that representative, and the recorded params are stated in its labels; certificates print the representative above each step. Keying on unlabelled ids is legitimate because every piggyback is label-equivariant. The order dependence of Section 2 is enumerated rather than fixed because different removal orders give different structures and none is canonical.

### 7.2 Two phases, nine stages

The census runs in two phases over one shared explorer. Throughout, $s$ is the predicted node and $X$ the predictor of a Fritz-type step.

*Phase 1 (cheap).* Only the elementary reductions act, closed over everything reachable from every input, and only the three-node seeds are known, so the Bell variants are inputs like any other. Reachability is transitive within this phase, so the set of inputs it proves does not depend on the order in which the tricks are applied, and `_fixpoint` iterates the implication closure among the inputs until nothing changes. Because every elementary trick is cheap and the phase is exhaustive, it is the one phase where "what does this trick prove alone, and what is lost without it" is a meaningful and affordable question, and 7.7 reports exactly that, per elementary trick.

*Phase 2 (expensive).* The Bell variants join the seeds (every Bell variant has a gap by the direct argument, and those not reachable from the three-node seeds, such as `QG_Bell6d`, can only be seeds), the gaps of phase 1 and of the cache (7.4) are known, and the inputs the cheap phase left unproven are attacked by a cascade of **eight stages of increasing cost** (`default_stages`, `CASCADE`), each applied once to each input the previous stages left unproven. The stages are the Fritz-type steps split by trick, predictor mode and predicted-node mode, cheapest first:

1. `Fritz`, dropped predictors, replace mode (d-separation certificate);
2. `Fritz`, dropped predictors, copy mode (the predicted node $s$ is split, 5.2);
3. `Fritz_kept`, kept predictors (childless untouched, childful split and the copy marginalized), replace mode;
4. `Fritz_kept`, copy mode;
5. `Fritz_entropic`, dropped predictors, replace mode (LP-certified steps only);
6. `Fritz_entropic`, dropped predictors, copy mode;
7. `Fritz_entropic`, kept predictors, replace mode;
8. `Fritz_entropic`, kept predictors, copy mode.

Replace mode precedes copy mode because copy mode splits $s$ and so works on a larger structure; dropped predictors precede kept ones because a kept predictor leaves the node count unchanged and, when childful, adds a split and a marginalization; the LP stages come last. Each stage applies its trick once to each input still unproven, and the outputs are then reduced with the elementary tricks only, never with another Fritz-type step ("depth one"). Every transition records the trick name and its parameters (predictor, predicted node and mode, predictor mode, certificate), and the explorer records the stage in which it was found, so the cumulative count of inputs proven after each stage is a graph query (`GapReport.stage_counts`), and it coincides with the ladder computed from the parameters alone (`ladder`), which the slow test checks.

The cascade loses proofs that would need two Fritz-type steps, which is accepted: a Fritz-type output has at least as many visible nodes as its source, and re-expanding every such output under the Fritz tricks was found to be far slower than everything else combined. The counts of 7.8 are therefore lower bounds for what the tricks can prove, exact for phase 1. What the order decides is cost (the LP runs only where the cheap stages failed) and which tricks a certificate prefers (`build_report` takes the shortest chain within the earliest stage that has one). `prove_gaps` closes the proven set under implication from the seeds and attaches to every proven input a certificate, the shortest chain of transitions down to a named seed.

### 7.3 What is never computed

The search never asks how many inputs an expensive piggyback proves *alone*: that would mean closing the search under that piggyback over everything reachable, which is exactly the cost the cascade avoids, and it would answer a question nobody needs answered. The expensive steps are assessed by the cumulative ladder of 7.8 alone: what each successive stage adds to everything cheaper.

### 7.4 The cache

(`gap_cache.py`, `cache/known_gaps.json`.) Every proven input is stored with its unlabelled id, its structure, the seed its certificate ends at, the chain of transitions (trick, parameters, source and target ids), the rendered certificate, and the **version** of every piggyback the chain uses (`PIGGYBACK_VERSIONS` in `qc_gap_search.py`). A certificate that ends at a cached gap inherits that entry's versions, so dependence is transitive. When a piggyback is corrected its version is bumped, and loading the cache drops exactly the entries whose proof relied on it; everything else stays known, and the expensive stages run only on the inputs that are neither cached nor proven by phase 1. The entries are up to relabelling, like everything else.

### 7.5 The known-gap database the search establishes

Every structure the search touches, inputs, intermediates and the hybrid structures with classical facets that the Fritz steps create, is a proven QC gap as soon as it reaches a seed, and `GapReport.proven_structure_ids` returns that set. This is the database to keep: the census inputs have every latent quantum, but a structure with some facets classical is a *weaker* structure, its gap is not implied by the gap of the all-quantum version (making a facet classical shrinks the quantum set and leaves the classical set), and it is a gap only if a chain of piggybacks from it reaches a seed. Fritz steps are indifferent to whether the shared facet is classical or quantum, so chains often transfer; conditioning and teleportation are not, since they create quantum facets only among quantum siblings. 7.8 reports how many hybrid structures the census proves in passing. Hybrid four-node structures as *inputs* are not in the census yet.

### 7.6 Reading the tables, and why the seeds matter

Reachability restricted to any subset of tricks, or to any predicate on the recorded parameters, is a graph query over the recorded transitions. In 7.7, "via" counts the inputs provable with one elementary trick alone and "only via" the inputs lost when that trick is removed and everything else kept; both are computed in phase 1 with the three-node seeds. In 7.8 the counts are cumulative over the stages. Since each stage of the cascade ran only on inputs the earlier stages left unproven, an input counted at the `Fritz` rungs needs a Fritz-type step that d-separation with dropped predictors supplies, not one that the later tricks could not supply: `Fritz_kept` and `Fritz_entropic` subsume `Fritz` (Sections 5.2, 6.4).

Every piggyback except the Fritz type reduces the number of visible nodes, so for four-node inputs a four-node seed can only be reached by a Fritz-type step. Listing the Bell variants as seeds in phase 2 is what lets the Fritz steps conclude in one move; keeping them out of phase 1 is what lets the reductions get credit for the Bell variants they do reach (7.7). The Bell variants that are not derivable from the three-node seeds by any piggyback have to be seeds, on the strength of the direct Bell argument.

### 7.7 Phase 1 results: the elementary piggybacks, from the three-node seeds

The inputs are the four-node mDAGs whose edges respect the order $0\lt1\lt2\lt3$ and that are not provably algebraic, with every latent quantum: 2807 labelled structures, 996 distinct up to relabelling, of which 6 are Bell variants (the Bell variants with a classical facet are not census inputs). All counts are of distinct structures up to relabelling. `tests/test_baseline_slow.py` pins every number below; `Special Applications/census_breakdowns.py` prints one certificate per row of the tables.

Only the elementary reductions act and only the three-node seeds (instrumental, triangle, Evans variants) are known, so the Bell variants are inputs. "Via" is the number of inputs provable using only that trick, closed under implication among the inputs; "only via" is the number no longer provable when that trick alone is removed from the recorded transitions, everything else kept. This phase takes two seconds.

| elementary piggyback | via | only via |
|---|---|---|
| point distribution | 860 | 246 |
| interruption | 10 | 3 |
| conditioning | 289 | 20 |
| naive marginalization | 515 | 0 |
| teleportation marginalization | 540 | 16 |
| marginalization, either kind (both removed at once) | 540 | 24 |
| all elementary piggybacks | 917 | |

Of the 996 inputs, 917 are proven and 79 are left for phase 2. Of the 6 Bell variants that are inputs, 3 are proven, all by interruption from an instrumental variant (Section 4): `QG_Bell1` to `QG_Instrumental1`, `QG_Bell3b` to `QG_Instrumental2`, `QG_Bell9b` to `QG_Instrumental3`. The other 3 (`QG_Bell6d` among them) are not reached from the three-node seeds by any piggyback: deleting a facet is never a piggyback, so a structure such as `QG_Bell6d` (no edges; Q{0,2}, Q{1,3}, Q{2,3}) has no route to a three-node seed and has to be a seed itself, on the strength of the direct Bell argument. Those 3 are the difference between the 917 of phase 1 and the 914 inputs that phase 2 counts as proven by the elementary stage (7.8): in phase 2 they are seeds, not inputs.

Reading the "only via" column. The two marginalizations coincide whenever the removed node has no quantum facet, so removing the naive one alone loses nothing; removing the teleportation one alone loses 16 inputs, those where the removed node holds a quantum facet and the relayed entanglement is what the proof needs; removing both loses 24. Interruption is the sole route for 3 inputs, the 3 Bell variants above: its characteristic product is a Bell variant derived from an instrumental variant, and every other input it proves is also reachable by PD or conditioning (7.10).

### 7.8 Phase 2 results: the cascade

The Bell variants are seeds, so there are 990 inputs (2759 labelled), of which the elementary stage proves 914. The cascade runs on the 76 that are left; each row gives the inputs proven after that stage, what the stage added, and how many inputs it was applied to.

| stage | proven (cumulative) | new | applied to |
|---|---|---|---|
| elementary reductions | 914 | 914 | 990 |
| + `Fritz`, dropped predictors, replace mode | 915 | 1 | 76 |
| + `Fritz`, dropped predictors, copy mode | 918 | 3 | 75 |
| + `Fritz_kept`, replace mode | 918 | 0 | 72 |
| + `Fritz_kept`, copy mode | 919 | 1 | 72 |
| + `Fritz_entropic`, dropped predictors, replace mode | 919 | 0 | 71 |
| + `Fritz_entropic`, dropped predictors, copy mode | 919 | 0 | 71 |
| + `Fritz_entropic`, kept predictors, replace mode | 924 | 5 | 71 |
| + `Fritz_entropic`, kept predictors, copy mode | 924 | 0 | 66 |
| remaining | | 66 | |

Remaining 66; 1763 structures expanded. The whole census, both phases, takes under five minutes on one core (284 seconds in the pinned run), against about 45 minutes before the Fritz stages were made depth-one and the LP was reduced to one solve per target set; no LP solve reached the 60-second time limit.

The search also established, in passing, the gap of every structure it touched that reaches a seed: 960 structures in all (9 with three visible nodes, 939 with four, 12 with five), 25 of them with at least one classical facet (`GapReport.proven_structure_ids`, Section 7). All 924 proven inputs are stored in `cache/known_gaps.json` with their certificates and piggyback versions; a second run loads them and finds nothing left for the expensive stages to do.

The four inputs first proven at the `Fritz` rungs are of course also within reach of `Fritz_kept` and `Fritz_entropic`, which subsume it (Sections 5.2, 6.4); they appear at the `Fritz` rungs because those ran first. One of them is the replace-mode example of 5.5; the other three need copy mode.

**Where kept predictors prove something dropped predictors cannot.** Six inputs: one by d-separation (the `Fritz_kept` example of 5.5, where the childful predictor is split and its copy marginalized) and five by the LP (6.8), all five with a childless predictor kept and the `relabel` target set. The LP trick with dropped predictors adds nothing beyond d-separation in four nodes, and the `markov` targets are never decisive, although they certify many candidate steps (7.9) whose outputs are also reached otherwise or are not gaps as far as the seeds know. Copy mode of the predicted node is decisive for four inputs, three at the `Fritz` rung and one at the `Fritz_kept` rung; at the LP rungs copy mode adds nothing.

### 7.9 Entropic certificates attempted

Per predictor–target candidate over the four LP stages, split structures included (`ENTROPIC_STATS`): admissible by d-separation 2640; beyond d-separation, `relabel` 228, `markov` 91 (tried only where `relabel` was inapplicable or failed), failed 1752; joint targets (several predicted nodes at once, `markov` only) certified 27, failed 34. No solve hit the time limit. Among candidates that d-separation rejects, the LP certifies roughly one in six. Success is common but far from universal, and a failure of the LP is not a proof that the implication is false (6.4). Almost none of these certified steps is decisive (7.8): their outputs are structures the cheaper tricks reach as well, or structures that are not gaps as far as the seeds know.

### 7.10 Why interruption and naive marginalization show few exclusive proofs

**Interruption.** Its characteristic product is a Bell variant derived from an instrumental variant. In phase 1 it is the sole route for exactly the three Bell variants that are proven at all (7.7); every other input it proves (ten in all) is also reachable by PD or conditioning. Interruption is the only reduction that re-uses an outcome as a setting; how much it contributes is bounded by how many Bell-type structures the input set contains that the seed list does not already hold.

**The two marginalizations.** They coincide at a node without a quantum facet, so naive marginalization alone is never exclusive: wherever it applies, the teleportation version makes the same move. The 16 inputs lost without teleportation marginalization are where the removed node holds a quantum share and relaying it to the node's children is what the proof needs. Conditioning covers part of the same ground, since conditioning on a node $v$ without visible parents adds a quantum facet over all quantum siblings of $v$ (entanglement swapping), the same connection teleportation relays to the children of $v$, only not restricted to children; what conditioning cannot provide is the relayed visible input of $v$, and where that input matters, or where conditioning is blocked by a grandparent, teleportation marginalization is the route. The earlier census, which closed the dropped-predictor Fritz trick over everything reachable and had the Bell variants as seeds, found alternative routes for all of these; the cheap phase, with the elementary tricks alone and the three-node seeds, does not.

**Example** (only via teleportation marginalization). Input: 0→1→2; Q{0,1,3}, Q{0,2}. Node 0 is the setting of 1, holds a share of a tripartite state with 1 and 3, and a bipartite state with 2. Marginalizing 0 with teleportation relays its share of Q{0,1,3} and of Q{0,2} to its child 1: 1→2; Q{1,3}, Q{1,2}, which is `QG_Instrumental3`. Naive marginalization of 0 relays only a classical common cause to 1 and 2 and leaves 1→2; C{1,2}, Q{1,3}, not a known gap; conditioning on 0 gives 1→2; Q{1,2,3}, PD on 0 gives 1→2; Q{1,3}, neither a known gap; interruption needs an exogenous node, and 0 holds facets; and no reduction at 1, 2 or 3 reaches a seed either, since each either destroys the setting or isolates 3. A second example of the same shape, 0→1, 1→2, 1→3 with Q{0,1,2}, Q{0,3}, relays the shares of 0 to 1 and gives `QG_Evans`.

**Example** (only via marginalization, either kind). Input: 2→3; Q{0,1}, Q{0,3}, Q{1,2}. Node 2 has no latent of its own: it relays nothing but its outcome to 3. Marginalizing 2 gives Q{0,1}, Q{0,3}, C{1,3} up to relabelling, which is `QG_Triangle2`; here teleportation has nothing to relay, so both marginalizations coincide, and the input counts in the "either kind" row but in neither single row.

---

## 8. Open questions

* **Completeness of the entropic certificate.** The failures recorded in `ENTROPIC_STATS` are failures of the LP, not necessarily of the piggyback (6.4). Substituting other sets of variables for the latents of $G'$, beyond $\lbrace L\rbrace$ and $\lbrace s\rbrace$, is the natural next level: each substitution is expressible because the LP indexes joint entropies of sets, and the soundness proof is that of 6.4.
* **Extra deletions certified by the same LP** (`QmDAG._entropic_extra_deletions`, `extra_deletions=True`; off in the census). With $s$ the predicted node and $\mathbf X$ the predicting set, the hypotheses of 6.3 do not mention which node a deletion concerns, so the same LP can certify deleting a parent $p$ of any non-predictor node $t$ through the per-edge target $I(t{:}p\mid\mathrm{Pa}'(t)\setminus p)\le0$, where $\mathrm{Pa}'(t)$ are the parents of $t$ in the current candidate. Deleting an edge enlarges $\mathcal I(G_1)$, so the hypotheses are recomputed from the current candidate after each deletion and the loop runs to a fixed point, order-dependent and budgeted at `max_lps=60` LP solves per candidate, not exhaustive. Because deleting edges also enlarges non-descendant sets, the final candidate is verified as a whole with 6.3; if that fails, the candidate without extra deletions is kept. Soundness is Theorem 6.5 for the final candidate, whose statement already allows further parents deleted from non-predictor nodes. In an earlier census run with the loop on, extra deletions certified hundreds of candidate steps and decided no input, which is why the census leaves them off. Deletions never touch a predictor (it must keep seeing $\mathrm{common}(s)$), and never remove the last facet a predicted node shares with its predictors: without it the hypotheses $H(s\mid\mathbf X)=0$ and $s\perp\mathbf X$ (an observable d-separation of the resulting $G_1$) are jointly satisfiable only by a constant $s$, the LP would certify a vacuous statement, and the lift would no longer exist.
* **The prediction-free edge-deletion piggyback** (KPC, Corollary 3): the same LP without the perfect-prediction hypothesis certifies deleting edges justified by the new graph's independences alone. The code can check it (`QmDAG._entropic_certificate(..., predicted=())`), but it is not exposed as a trick.
* **Exact certificates.** Farkas multipliers are floating point; rationalising them and re-verifying the combination exactly is cheap and would make every entropic step a checkable proof.
* **The remaining structures.** The 66 unproven inputs are listed below. Every one of them contains a visible edge out of a node that also holds quantum facets with later nodes, and most contain a chain of two or three visible edges.

  <details><summary>The 66 remaining inputs (edges; quantum facets)</summary>

  | | edges | quantum facets |
  |---|---|---|
  | 1 | 0→1 | {0,1,2}, {0,3}, {1,3} |
  | 2 | 0→1 | {0,1}, {0,2}, {0,3}, {1,2}, {1,3} |
  | 3 | 1→2, 2→3 | {0,1}, {0,3} |
  | 4 | 0→1, 1→2 | {0,1,2}, {1,3}, {2,3} |
  | 5 | 0→1, 1→2 | {0,1}, {0,2}, {1,3}, {2,3} |
  | 6 | 0→1, 1→2 | {0,2}, {1,2}, {1,3}, {2,3} |
  | 7 | 0→1, 1→2 | {0,1}, {0,2}, {1,2}, {1,3}, {2,3} |
  | 8 | 0→1, 1→2, 1→3 | {0,2}, {0,3} |
  | 9 | 0→1, 1→2, 1→3 | {0,2}, {0,3}, {1,2} |
  | 10 | 0→2, 1→2 | {0,1,2}, {1,3}, {2,3} |
  | 11 | 0→2, 1→2 | {0,1}, {0,2}, {1,3}, {2,3} |
  | 12 | 0→2, 1→2 | {0,1}, {1,2}, {1,3}, {2,3} |
  | 13 | 0→2, 1→2 | {0,1}, {0,2}, {1,2}, {1,3}, {2,3} |
  | 14 | 0→1, 0→2, 1→2 | {1,2}, {1,3}, {2,3} |
  | 15 | 0→1, 0→2, 1→2 | {0,1,2}, {1,3}, {2,3} |
  | 16 | 0→1, 0→2, 1→2 | {0,1}, {0,2}, {1,3}, {2,3} |
  | 17 | 0→1, 0→2, 1→2 | {0,1}, {1,2}, {1,3}, {2,3} |
  | 18 | 0→1, 0→2, 1→2 | {0,2}, {1,2}, {1,3}, {2,3} |
  | 19 | 0→1, 0→2, 1→2 | {0,1}, {0,2}, {1,2}, {1,3}, {2,3} |
  | 20 | 0→1, 2→3 | {0,2}, {1,3} |
  | 21 | 0→3, 1→2 | {0,1}, {0,2,3}, {1,3} |
  | 22 | 0→1, 2→3 | {0,1,3}, {0,2}, {1,2,3} |
  | 23 | 0→3, 1→2 | {0,1}, {0,2,3}, {1,2}, {1,3} |
  | 24 | 0→3, 1→2 | {0,1}, {0,2}, {1,3}, {2,3} |
  | 25 | 0→3, 1→2 | {0,1}, {0,2}, {1,2}, {1,3}, {2,3} |
  | 26 | 0→3, 1→2 | {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3} |
  | 27 | 0→1, 0→2, 2→3 | {1,3} |
  | 28 | 0→1, 0→2, 2→3 | {0,2}, {1,3} |
  | 29 | 0→1, 0→2, 2→3 | {0,1}, {1,3} |
  | 30 | 0→2, 1→2, 2→3 | {0,1}, {1,3} |
  | 31 | 0→1, 1→2, 2→3 | {0,2}, {1,3} |
  | 32 | 0→1, 1→2, 2→3 | {0,2}, {0,3}, {1,2,3} |
  | 33 | 0→1, 1→2, 2→3 | {0,2}, {0,3}, {1,3} |
  | 34 | 0→1, 1→2, 2→3 | {0,1}, {0,2}, {0,3}, {1,2,3} |
  | 35 | 0→1, 1→2, 2→3 | {0,2}, {0,3}, {1,2}, {1,3} |
  | 36 | 0→1, 1→2, 2→3 | {0,1}, {0,2}, {0,3}, {1,3} |
  | 37 | 0→1, 1→2, 2→3 | {0,2}, {0,3}, {1,3}, {2,3} |
  | 38 | 0→1, 1→2, 2→3 | {0,1}, {0,2}, {0,3}, {1,2}, {1,3} |
  | 39 | 0→1, 1→2, 2→3 | {0,1}, {0,2}, {0,3}, {1,3}, {2,3} |
  | 40 | 0→1, 1→2, 2→3 | {0,2}, {0,3}, {1,2}, {1,3}, {2,3} |
  | 41 | 0→1, 1→2, 2→3 | {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3} |
  | 42 | 0→1, 0→2, 1→2, 2→3 | {1,3} |
  | 43 | 0→1, 0→2, 1→2, 2→3 | {0,2}, {1,3} |
  | 44 | 0→1, 0→2, 1→2, 2→3 | {0,1}, {1,3} |
  | 45 | 0→2, 1→3, 2→3 | {0,1}, {0,2,3}, {1,2} |
  | 46 | 0→3, 1→2, 2→3 | {0,1}, {0,2,3}, {1,2,3} |
  | 47 | 0→2, 1→3, 2→3 | {0,1}, {0,3}, {1,2} |
  | 48 | 0→3, 1→2, 2→3 | {0,1}, {0,2,3}, {1,3} |
  | 49 | 0→2, 1→3, 2→3 | {0,1}, {0,2,3}, {1,2}, {1,3} |
  | 50 | 0→2, 1→3, 2→3 | {0,1}, {0,2}, {0,3}, {1,2} |
  | 51 | 0→1, 1→3, 2→3 | {0,1}, {0,2}, {0,3}, {1,2,3} |
  | 52 | 0→1, 1→3, 2→3 | {0,2}, {0,3}, {1,2}, {1,3} |
  | 53 | 0→3, 1→2, 2→3 | {0,1}, {0,2}, {0,3}, {1,3} |
  | 54 | 0→1, 1→3, 2→3 | {0,1}, {0,2}, {0,3}, {1,2}, {1,3} |
  | 55 | 0→2, 1→3, 2→3 | {0,1}, {0,2}, {0,3}, {1,2}, {1,3} |
  | 56 | 0→3, 1→2, 2→3 | {0,1}, {0,2}, {0,3}, {1,3}, {2,3} |
  | 57 | 0→2, 1→3, 2→3 | {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3} |
  | 58 | 0→1, 1→2, 1→3, 2→3 | {0,2}, {0,3} |
  | 59 | 0→1, 1→2, 1→3, 2→3 | {0,2}, {0,3}, {1,2} |
  | 60 | 0→1, 1→2, 1→3, 2→3 | {0,2}, {0,3}, {1,3} |
  | 61 | 0→1, 1→2, 1→3, 2→3 | {0,2}, {0,3}, {1,2,3} |
  | 62 | 0→1, 1→2, 1→3, 2→3 | {0,2}, {0,3}, {2,3} |
  | 63 | 0→1, 1→2, 1→3, 2→3 | {0,2}, {0,3}, {1,2}, {1,3} |
  | 64 | 0→1, 1→2, 1→3, 2→3 | {0,2}, {0,3}, {1,2}, {2,3} |
  | 65 | 0→1, 1→2, 1→3, 2→3 | {0,2}, {0,3}, {1,3}, {2,3} |
  | 66 | 0→1, 1→2, 1→3, 2→3 | {0,2}, {0,3}, {1,2}, {1,3}, {2,3} |

  </details>

  Several are Bell scenarios with extra structure among the settings. The input 0→1, 2→3; Q{0,2}, Q{1,3} is the Bell scenario with entangled settings (0 and 2 are the settings of 1 and 3, Q{1,3} the shared state); 0→1, 0→2, 2→3; Q{1,3} lets Alice's setting influence Bob's. Both have a QC gap by the usual argument, because the latents of the settings are independent of the latent of the outcomes, so any classical model is local for $P(a,b\mid x,y)$. No piggyback can reach them from the Bell seeds, since deleting a facet or an edge between settings is not a piggyback. Extending the seed list with such Bell variants, after checking each by the direct argument, is the cheapest next step.

---

## 9. References

* T. Fritz, *Beyond Bell's theorem: correlation scenarios*, New J. Phys. 14, 103001 (2012); *Beyond Bell's theorem II: scenarios with arbitrary causal structure*, Commun. Math. Phys. 341, 391 (2016). The triangle argument and the perfect-prediction mechanism of Section 5.
* R. J. Evans, *Graphs for margins of Bayesian networks*, Scand. J. Stat. 43, 625 (2016). Latent projection (Section 2) and mDAGs.
* J. Henson, R. Lal and M. F. Pusey, *Theory-independent limits on correlations from generalized Bayesian networks*, New J. Phys. 16, 113043 (2014). Observable d-separation holds in every generalised probabilistic theory (F2).
* R. Chaves, C. Majenz and D. Gross, *Information-theoretic implications of quantum causal structures*, Nat. Commun. 6, 5766 (2015). Coexisting sets and the entropic treatment of quantum causal structures (F2, Section 6).
* T. Verma and J. Pearl, *Causal networks: semantics and expressiveness*, Proc. UAI (1988). Local Markov property, d-separation and the semigraphoid axioms (6.2).
* B. Bonet, *Instrumentality tests revisited*, Proc. UAI (2001). The instrumental inequality (Section 5.2 example).
* R. Khanna, M. F. Pusey and R. Colbeck, draft manuscript and Mathematica notebook `piggyback_check.nb` (KPC). The entropic certificate, the six-node example of 6.7 and the prediction-free variant of Section 8.
