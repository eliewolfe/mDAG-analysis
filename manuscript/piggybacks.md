# Piggybacks for quantum-classical gaps

*How each structural transformation in this repository is justified, and where it lives in the code.*

## 0. Setting

A **QmDAG** (`quantum_mDAG.QmDAG`) is a causal structure with visible nodes $V=\{0,\dots,n-1\}$, a directed structure on $V$, and two simplicial complexes of **latent facets**: classical facets $\mathcal C$ and quantum facets $\mathcal Q$. A facet $F\subseteq V$ stands for one exogenous latent (a classical random variable, or a quantum state) whose children are exactly the nodes of $F$. Every visible node additionally has implicit private randomness. Classical facets contained in a quantum facet are redundant and dropped by the constructor.

For a structure $G$ write $\mathcal C(G)$ for the set of distributions over $V$ realisable with all latents classical (a classical Bayesian network with latents), and $\mathcal Q(G)$ for the set realisable when the quantum facets carry quantum states on which their children perform measurements (classical facets and visible edges remain classical channels). Always $\mathcal C(G)\subseteq\mathcal Q(G)$. $G$ has a **QC gap** when $\mathcal Q(G)\not\subseteq\mathcal C(G)$.

A **piggyback** is a map $G\mapsto G'$ together with a proof of

$$\mathcal Q(G')\not\subseteq\mathcal C(G')\;\Longrightarrow\;\mathcal Q(G)\not\subseteq\mathcal C(G).$$

Every proof below has the same two halves.

* **Quantum lift.** Given $P'\in\mathcal Q(G')\setminus\mathcal C(G')$, construct $P\in\mathcal Q(G)$ from which $P'$ is recovered by a fixed operation $\pi$ (a marginal, a conditional, a coarse-graining, or the identity): $\pi(P)=P'$.
* **Classical pull-back.** Show $\pi(\mathcal C(G))\subseteq\mathcal C(G')$, i.e. every classical model of $G$ for a distribution of the kind produced by the lift yields a classical model of $G'$ for $\pi(P)$.

Then $P\in\mathcal C(G)$ would give $P'=\pi(P)\in\mathcal C(G')$, a contradiction, so $P\in\mathcal Q(G)\setminus\mathcal C(G)$.

All piggybacks are **caveat-free**: the hypothesis is a QC gap of $G'$ as a structure, with no side condition on the gap-witnessing distribution. This is what lets them be composed in any order and searched as a closure (Section 9). Structures are identified up to relabelling by `unique_unlabelled_id`.

Two general facts are used repeatedly.

* **(F1) Edge monotonicity of the lift.** If $G'$ is obtained from $G$ only by deleting visible edges and removing nodes from facets, then $\mathcal Q(G')\subseteq\mathcal Q(G)$ (run the same strategy and ignore the extra inputs) and likewise $\mathcal C(G')\subseteq\mathcal C(G)$. The nontrivial half of an edge-deleting piggyback is therefore always the pull-back.
* **(F2) d-separation.** In any classical model of $G$ (latents included as nodes), and in any quantum model of $G$ for the visible variables, a d-separation of $G$ implies the corresponding conditional independence. Classical models are also *complete* for d-separation: the only conditional independences holding in every classical model are the d-separations.

The **effective DAG** of a QmDAG (`QmDAG.effective_DAG_data`) is the DAG on visible nodes, one node per facet and one private-noise node per visible node; `QmDAG.lp_structure` is the same without noise nodes.

---

## 1. Point distribution (`PD`)

**Map.** Remove one visible node $v$: $G'=G-v$ (`QmDAG.fix_to_point_distribution_QmDAG`, `QmDAG.subgraph`; trick `pd_trick` in `qc_gap_search.py`).

**Lift.** Given $P'\in\mathcal Q(G')$, let $v$ output a constant and every other node run its $G'$ strategy; $\pi$ = marginal over $v$ (trivial, $v$ is constant). $P\in\mathcal Q(G)$ because the children of $v$ may ignore a constant input.

**Pull-back.** In a classical model of $G$ with $v$ constant, substitute the constant into the kernels of $v$'s children; the result is a classical model of $G-v$ for the marginal. $\square$

## 2. Marginalization, naive and with teleportation (`naive_marginalization`, `teleportation_marginalization`)

**Map** (`QmDAG.marginalize(node, apply_teleportation)`). Remove $v$; every visible parent of $v$ becomes a parent of every visible child of $v$; every facet containing $v$ loses $v$ and gains all visible children of $v$ as a *classical* facet; a new classical facet over the visible children of $v$ is added (the private randomness of $v$). With teleportation, additionally every quantum facet $F\ni v$ is kept quantum on $(F\setminus v)\cup T$ where $T$ is the set of visible children of $v$ that share some quantum facet with $v$. $\pi$ = marginal over $v$.

**Pull-back.** Without teleportation $G'$ is the latent projection of $G$ over $v$ (Evans), and the marginal of a classical model of $G$ is a classical model of the projection: $v$'s kernel is absorbed into its children, its latent parents are relayed to its children, and its private randomness becomes a common cause of its children. With teleportation $G'$ has more facets than the projection, so the pull-back is the same inclusion followed by (F1) in reverse ($\mathcal C$ grows with facets).

**Lift.** Given $P'\in\mathcal Q(G')$: $v$ receives its visible parents and forwards them to its children (the new visible edges), forwards the classical values of its classical facets and broadcasts a private random string (the new classical facets). For a quantum facet $F\ni v$ that in $G'$ extends to a child $c\in T$: $v$ and $c$ share some quantum facet, which may carry an extra maximally entangled pair; $v$ teleports its share of $F$'s state to $c$ through that pair and the visible edge $v\to c$. The children then hold exactly the resources of $G'$. $\square$

*Order dependence.* Because teleportation is not symmetric, marginalizing several nodes in different orders can give different structures; wherever the code removes several nodes it enumerates every order (`QmDAG._marginalize_predictors`).

## 3. Conditioning (`conditioning`)

**Map** (`QmDAG.condition(node)`). Remove $X$; add a classical facet over $\mathrm{Pa}_V(X)\cup S$ where $S$ is the set of visible nodes sharing some facet with $X$ (its latent siblings); add a quantum facet over the nodes sharing a quantum facet with $X$. $\pi(P)=P(\,\cdot\mid X=x_0)$ for a suitable value $x_0$.

**Admissibility** (`QmDAG.conditioning_is_justified`). Two conditions on $X$:

1. every visible grandparent of $X$ is a visible parent of $X$ (so $\mathrm{Pa}_V(X)$ is ancestrally closed among visible nodes);
2. every facet containing a visible parent of $X$ either contains $X$ or is contained in $\mathrm{Pa}_V(X)$.

In words: no grandparent of $X$, visible or latent, may fail to be a parent of $X$, except for latents whose children all lie among the parents (those only redistribute the parent block). Condition 2 was added in this revision. Before, only condition 1 was checked, and the pull-back below fails without condition 2: take $X$ with parents $p_1,p_2$, each $p_i$ sharing a classical facet with an outside node $o_i$, and the classical model $p_i=\lambda_i\oplus\varepsilon_i$ (biased noise), $o_i=\lambda_i$, $X=[p_1=p_2]$. Conditioning on $X=1$ makes $o_1$ and $o_2$ dependent ($I(o_1{:}o_2)\approx0.046$ bits), while $G'$ d-separates them. Hence $\pi(\mathcal C(G))\not\subseteq\mathcal C(G')$ and the inference is not justified. The effect of condition 2 on the 4-node census is reported in Section 10.

**Pull-back** (with conditions 1 and 2). Let $B=\mathrm{Pa}_V(X)$, $\Lambda_X$ the latents of $X$, and $\Lambda_B$ the latents whose children all lie in $B$. By the two conditions every input of a node in $B$ is in $B$, in $\Lambda_X$ or in $\Lambda_B$; the latents in $\Lambda_B$ influence nothing outside $B$ and are absorbed into $\mu$ below together with $\Lambda_X$. In a classical model of $G$,
$$P(\text{all}\mid x_0)\ \propto\ \Big[\prod_{\text{other nodes}}\text{kernels}\Big]\cdot\Big[\prod_{p\in B}P(p\mid \mathrm{pa}(p)\cap B,\Lambda_X,\Lambda_B)\Big]\cdot P(x_0\mid B,\Lambda_X)\cdot P(\Lambda).$$
Define the new latent $\mu=(\Lambda_X,\Lambda_B,R)$ with $R$ fresh uniform randomness. The nodes of $S$ read $\Lambda_X$ through $\mu$ (the facets $F\setminus X$ that $G'$ retains are set trivial). Visible children of $X$, if any, lose the edge from $X$ and read the constant $x_0$ instead. The block $B$ is sampled jointly, as deterministic functions of $(\mu)$ only, from the weighted distribution $\propto\prod_{p\in B}P(p\mid\cdot)\,P(x_0\mid B,\Lambda_X)$: this is possible because all of the block's inputs are available in $\mu$, so a shared random seed $R$ lets each $p$ compute its coordinate of one joint sample. Every other node keeps its kernel. The result is a classical model of $G'$ for $P(\cdot\mid x_0)$. (Without condition 2 some $p$ has a private latent shared with outsiders; the weighted joint then depends on inputs the other block members cannot see, which is exactly what the counterexample exploits.)

**Lift.** Given $P'\in\mathcal Q(G')$ with the new classical facet $\mu$ and the new quantum state $\rho$ over the quantum siblings $S_Q$. In $G$: every facet $F\ni X$ carries an independent uniform copy $\mu_F$ of the string $\mu$; every quantum facet $F\ni X$ additionally carries one maximally entangled pair between $X$ and each quantum sibling in $F$. Latent siblings read $\mu_F$ from their facet. $X$ prepares $\rho$ locally and teleports each subsystem to its sibling through the corresponding pair (a Bell measurement per subsystem); post-selected teleportation, i.e. keeping only the identity outcome, leaves $S_Q$ holding $\rho$ exactly. Each visible parent $p$ draws a private guess $\hat\mu_p$, runs its $G'$ strategy with $\mu:=\hat\mu_p$, and outputs $(p,\hat\mu_p)$; its children ignore the second component. $X$ sets $X=x_0$ iff all $\mu_F$ and all $\hat\mu_p$ coincide and every Bell outcome is the identity. Conditioned on $X=x_0$, the strings are a single uniform $\mu$ shared by $B\cup S$, $S_Q$ shares $\rho$, and every node has run its $G'$ strategy; the parents' outputs are fine-grainings $(p,\hat\mu_p)$ of their $G'$ outputs. A fine-graining of $P'$ that is in $\mathcal C(G')$ coarse-grains to $P'\in\mathcal C(G')$, so the contradiction argument goes through unchanged. $\square$

## 4. Interruption (`interruption`)

**Map** (`QmDAG.interruption_creation(node_with_no_children=y, node_with_no_parents=x)`). $x$ is exogenous (no visible parents, no nonsingleton facet: `QmDAG.exogenous_visible_nodes`) and $y$ is not a descendant of $x$ (so $G'$ is acyclic). Remove $x$; its visible children become children of $y$. The search additionally restricts $y$ to childless nodes (`vis_nodes_with_no_children`); the proof does not need this. $\pi$ is the renormalised diagonal $x=y$ of $P$:
$$\pi(P)(y,\text{rest})=\frac{P(x{=}y,\;y,\;\text{rest})}{P_x(y)},$$
where $P_x$ is the marginal of $x$ (uniform for every lifted distribution, so the denominator is then the constant $1/|X|$).

**Lift.** Given $P'\in\mathcal Q(G')$, in $G$ let $x$ be uniform, let the former children of $x$ treat the value of $x$ as they treated $y$ in $G'$, and let every other node run its $G'$ strategy. Then $P(x,y,\text{rest})=P_x(x)\,P'^{\,do(y:=x)}(y,\text{rest})$, where $P'^{do(y:=x)}$ is the distribution of the $G'$ strategy with the input of $y$'s former children fixed to $x$ (an edge intervention). By consistency of such interventions (fixing the children's input to the value $y$ actually takes changes nothing), $P'^{do(y:=y)}(y,\text{rest})=P'(y,\text{rest})$, so $\pi(P)=P'$.

**Pull-back.** Let $M$ be a classical model of $G$ for a distribution $P$ with $x$ uniform. $x$ is independent of all latents of $M$. Define a model of $G'$: every node keeps its kernel, except that the former children of $x$ receive $y$ in place of $x$. Since $y\notin\mathrm{desc}(x)$, $Y(\lambda)$ does not depend on $x$ and the new model is acyclic; its distribution is $\sum_\lambda p(\lambda)[y=Y(\lambda)]\,[\text{rest}=f(\lambda,x{:=}Y(\lambda))]$, which is the diagonal $x=Y(\lambda)$ of $M$'s distribution divided by $P_x(y)$, i.e. exactly $\pi(P)$. Other children of $y$ keep reading $y$ in both structures. $\square$

## 5. Node splitting (`QmDAG.split_node`)

**Map.** Replace $s$ by two nodes $s,s'$ with the same parents and children and a shared classical two-party facet $\{s,s'\}$ (their common private randomness; it is absorbed when $s$ already lies in a facet). $\pi$ = merge $(s,s')$ into one node.

**Lift.** $s$ computes both outputs from its inputs and private randomness.
**Pull-back.** A classical kernel for the pair $(s,s')$ splits into two kernels sharing the randomness carried by the new facet. $\square$

When the absorbing facet is quantum, the pair's shared randomness is carried by that quantum facet, which is legitimate because the pull-back concerns classical models only. Node splitting is used as a building block of the Fritz piggybacks (copy mode, split predictors); it is not run on its own.

---

## 6. The Fritz piggyback (`Fritz`)

### 6.1 The mechanism

Both Fritz piggybacks are instances of one scheme:

> choose a set of edges of $G$ to delete, giving $G'\subseteq G$, and justify the deletion by a **perfect prediction** that the quantum lift can arrange and that forces the deleted dependences to be idle in every classical model.

Apart from the classical facet added for $s$ (the $\rho\otimes\sigma$ remark below), $G'\subseteq G$, so the lift is (F1) plus the construction of the predictor; the entire content is the pull-back: *a classical model of $G$ in which the prediction holds (and in which the lifted distribution's other observable properties hold) is also a classical model of $G'$*.

Concretely, fix a **predictor** $X_1$ (a visible node) and a **predicted node** $s$ sharing at least one facet with $X_1$. Partition the parents of $s$ in the effective DAG into
$$\mathrm{common}(s)=\mathrm{Pa}(s)\cap\big(\mathrm{Pa}(X_1)\cup\{X_1\}\big),\qquad \mathrm{others}(s)=\mathrm{Pa}(s)\setminus\mathrm{common}(s),$$
the latter always containing the private noise of $s$. The candidate $G'$ restricts $s$ to $\mathrm{common}(s)$ (visible co-parents are kept; facets of $\mathrm{common}(s)$ are kept but become classical *for $s$*: a quantum facet $F$ keeps its quantum part on $F\setminus\{s\}$ and a classical facet over $F$ is added, which is the state $\rho\otimes\sigma$ with $\sigma$ classical). The lift then lets $s$ be a deterministic function $f$ of $\mathrm{common}(s)$, with its private randomness drawn from the facet shared with $X_1$, and lets $X_1$ output the same value $f(\mathrm{common}(s))$ alongside its own output. This is why $s$ must share a facet with $X_1$ (noise absorption) and why only $\mathrm{common}(s)$ may be kept: $X_1$ must be able to compute $s$.

### 6.2 Predictor modes and node modes (`QmDAG.fritz_transitions`)

* **Dropped predictor** (`predictor_mode='drop'`, the default of the cheap base pass). $X_1$ itself is removed: deleted if childless, otherwise by the marginalization piggyback of Section 2 (all removal orders), the lift being "forward your parents and output the copy of $s$".
* **Split predictor** (`predictor_mode='split'`, used in the expensive final pass). The predictor that is "removed" is a childless split-off copy $X_1'$ of $X_1$ (Section 5 with no children), so $X_1$ itself stays in $G'$ untouched. Formally the lift makes $X_1$ output $(X_1,\;f(\mathrm{common}(s)))$. $X_1$ may later be removed by the ordinary reductions (marginalization with teleportation, point distribution) inside the closure. Dropping is equivalent to splitting followed by marginalizing $X_1$, but keeping every predictor in the base closure multiplies the number of reachable structures and was found to be far too slow for the first pass.
* **Replace / copy** for the predicted node. In *copy* mode $s$ is split (Section 5) and the piggyback is applied to the copy, so the original keeps all of its parents and quantum facets while the copy carries the classical common part (the node-splitting form of Fritz's triangle argument: $A\mapsto(A,X)$ with $X$ a copy of the setting). `fritz_transitions` builds the copy directly in `_fritz_build`; `fritz_entropic_transitions` literally calls `split_node` first. The two constructions coincide because the pair's shared facet is absorbed by the facet the copy shares with $X_1$.
* **Joint predictors.** A set $X_1$ of predictors predicts $s$ jointly: $\mathrm{common}(s)$ collects the parents seen by any member, each member outputs the part it sees, and the pull-back below conditions on the set.

### 6.3 Theorem (d-separation certificate)

**Claim.** Let $X_1$ and $s$ be as above and suppose, in the effective DAG of $G$ (private noise nodes included),
$$X_1\setminus\mathrm{common}(s)\ \perp_d\ \mathrm{others}(s)\ \big|\ \mathrm{common}(s)$$
(a predictor that is itself a parent of $s$ lies in $\mathrm{common}(s)$ and is conditioned on rather than tested; if every predictor is a parent of $s$ the condition is vacuous).
Let $G'$ be $G$ with $s$ restricted to $\mathrm{common}(s)$ as in 6.1 (predictor kept). Then a QC gap in $G'$ implies a QC gap in $G$. The same holds after removing a childless $X_1'$ or marginalizing $X_1$ (Section 2).

**Proof.** *Lift:* given $P'\in\mathcal Q(G')$, run it in $G$ with $s:=f(\mathrm{common}(s))$ as in $P'$, which is legitimate because in $G'$ the parents of $s$ are classical and $s$ shares a facet with $X_1$ that carries its randomness, and let $X_1$ additionally output $f(\mathrm{common}(s))$. The result $P$ lies in $\mathcal Q(G)$ (indeed in $\mathcal Q(G')$: $X_1$'s extra output is a function of its parents), coarse-grains to $P'$, and satisfies $H(s\mid X_1)=0$.
*Pull-back:* let $M$ be a classical model of $G$ for $P$. Write $s=h(\mathrm{common},\mathrm{others})$ with $\mathrm{others}$ including the private noise, and $s=g(X_1)$. Condition on $\mathrm{common}(s)$. By the d-separation hypothesis and (F2), $\mathrm{others}(s)$ and $X_1$ are independent given $\mathrm{common}(s)$. A random variable that is a function of each of two independent variables is almost surely constant, so $s=\varphi(\mathrm{common}(s))$ a.s. Replace the kernel of $s$ in $M$ by $\varphi$; the joint distribution is unchanged and the new model is Markov to $G'$. Hence $P\in\mathcal C(G')$ and, coarse-graining $X_1$, $P'\in\mathcal C(G')$. $\square$

The piggyback deletes all of $\mathrm{others}(s)$ or nothing: the lift requires every parent the predictor cannot see to go. The set test above is equivalent to the conjunction of the per-edge tests $X_1\perp_d v\mid\mathrm{Pa}(s)\setminus\{v\}$ over $v\in\mathrm{others}(s)$, by the weak-union and intersection properties of d-separation. The implementation is `QmDAG.fritz_admissible_targets`, which tests the set at once with networkx's `is_d_separator`; the structure is built by `QmDAG._fritz_build` from an explicit map of kept parents.

### 6.4 What the earlier implementation got wrong

The version of the trick before this revision had two unsound steps, both now covered by tests: joint predictor sets took the *union* of the parents each predictor could remove (the correct test conditions on the whole set), and the predictor was marginalized while the predicted node kept parents the predictor could not see, so the lift did not exist. In the 4-node census the corrected trick re-derived (soundly) every structure the old code had attributed to the Fritz trick.

---

## 7. The entropic Fritz piggyback (`Fritz_entropic`)

### 7.1 Why the same mechanism admits a stronger certificate

In the pull-back of 6.3 the only facts about the classical model that were used were (a) the Markov property of $G$ and (b) $H(s\mid X_1)=0$. But the lifted distribution $P$ has more structure: it lies in $\mathcal Q(G')$, so by (F2) **every observable d-separation of $G'$ holds for $P$**, and therefore holds in any classical model of $G$ for $P$. These conditional independences are properties of the *new* graph, obtained for free, and there is no reason not to use them as hypotheses when deciding whether a deletion is justified in the *original* graph. The d-separation test of 6.3 ignores them; it is a shortcut that trades proof power for speed, and Khanna, Pusey and Colbeck's six-node example is a case where the shortcut fails and the extra independences are exactly what is needed.

### 7.2 Entropy vectors as the proof system (`entropic_lp.py`)

For $n$ random variables the entropy vector $h\in\mathbb R^{2^n-1}$ lists $H(S)$ for every nonempty $S$. Every entropy vector satisfies the **elemental Shannon inequalities** $H(i\mid[n]\setminus i)\ge0$ and $I(i{:}j\mid K)\ge0$ ($i<j$, $K\subseteq[n]\setminus\{i,j\}$), $n+\binom n2 2^{n-2}$ of them (`elemental_inequalities(n)`; the $n=3$ matrix is checked against the authors' Mathematica notebook, the row count for $n\le8$). A conditional independence is the vanishing of a CMI, a functional dependence the vanishing of a conditional entropy; both are nonnegative on the cone, so each hypothesis is one inequality "$\le0$". The local Markov property of a DAG is one CMI per node, $I(v{:}\mathrm{nondesc}(v)\setminus\mathrm{pa}(v)\mid\mathrm{pa}(v))=0$ (`local_markov_rows`). The Shannon cone with these hypotheses implies exactly the d-separations: every d-separation follows from the local Markov statements by the semigraphoid axioms (Verma and Pearl), which are Shannon-derivable, and nothing beyond the d-separations can follow because classical models are complete for d-separation (F2). This is tested on random DAGs.

A target functional $t$ is **implied** when the LP $\{\text{Shannon}\ \ge0,\ \text{hypotheses}\le0,\ t\ge1\}$ is infeasible (the cone is scale invariant). Infeasibility is decided with the Mosek Optimizer API (`EntropicLP`, interior point); the dual ray is a Farkas certificate, a nonnegative combination of elemental inequalities and hypotheses that reproduces $t$, available through `EntropicLP.farkas_certificate`. Implications proven this way are valid for every distribution, with or without full support, because they use only Shannon inequalities. The method is incomplete in the other direction: a feasible LP does not exhibit a distribution, only a vector in the Shannon cone.

### 7.3 The certificate and its two target sets (`QmDAG._entropic_certificate`)

Variables: all nodes of `lp_structure` (visible nodes and facets of $G$; quantum facets are ordinary latents, since only classical models are analysed). Hypotheses: Shannon; local Markov of $G$; $H(s\mid X_1)\le0$ (jointly $H(s\mid X_1\text{ set})$ for joint predictors); and $\mathcal C(G')$, the elementary observable d-separations $I(x{:}y\mid Z)$ of the candidate $G'$ over its visible nodes (`observable_dseparation_rows`). Two target sets are tried.

* **`markov`**: the local Markov equalities of $G'$ over $G$'s own latents. If all are implied, the classical joint of any model of $G$ satisfying the hypotheses is Markov to $G'$, hence $P\in\mathcal C(G')$.
* **`relabel`**: when $s$ keeps exactly one parent in $G'$ and it is a facet $L$ (so no visible parents; the code tests `len(common) == 1` and that the element is a facet index), let $G''$ be $G'$ with $L$ deleted and $s$ made a parent of every other child of $L$. Targets: the local Markov equalities of $G''$ over $G$'s variables other than $L$. If implied, the classical joint is Markov to $G''$; in particular $s$ is a root of $G''$, so its row says $s$ is jointly independent of every remaining latent. A $G''$-model is then a $G'$-model with $L:=s$: $L$ is an independent latent, $s$ is the deterministic function "read $L$" (its private noise in $G'$ is allowed to be trivial), and the kept quantum facet over $F\setminus\{s\}$ is left unused. Hence $P\in\mathcal C(G')$.

Both are sound; neither is complete. The `markov` set is too strong for the KPC example: the classical model $E=A\oplus B$, $D=C\oplus B$, $F=(A,B)$ of $G_1$ satisfies every hypothesis yet $E$ is not a function of $A$ alone, although the observed distribution is in $\mathcal C(G')$ with the relabelled latent $A'=E$. This is a general limitation: the LP speaks about the latents of the given model, while $\mathcal C(G')$ allows any latents, and the only re-encodings a linear certificate can express are the substitutions above. Copy mode is correspondingly weak: when the original $s$ keeps reading the facet, neither target set can identify the facet with the copy, so copy-mode certificates beyond d-separation are rare.

The d-separation certificate of 6.3 is subsumed, at least for the row of $s$: if $X_1\perp_d\mathrm{others}\mid\mathrm{common}$ then the Shannon cone with the Markov hypotheses derives $I(X_1{:}\mathrm{others}\mid\mathrm{common})\le0$, and with $H(s\mid X_1)\le0$ and the Markov row of $s$ (which gives $I(s{:}X_1\mid\mathrm{Pa}(s))\le0$, since private noise is not an LP variable) it derives $H(s\mid\mathrm{common})\le I(s{:}X_1\mid\mathrm{common})\le I(\mathrm{others}{:}X_1\mid\mathrm{common})+I(s{:}X_1\mid\mathrm{common},\mathrm{others})=0$, which implies the `markov` row of $s$. The `markov` set also contains rows for the former parents of $s$ and the descendants of $s$, whose non-descendant sets grow in $G'$; their Shannon-derivability is not argued here and was only checked empirically (every d-separation-admissible pair in the test structures is LP-admissible). The search never relies on it: d-separation-certified candidates are admitted by Theorem 6.3 without an LP, and `Fritz_entropic` emits, in predictor modes the base pass already covers, only transitions whose certificate is not plain d-separation or that delete extra edges (in the split predictor mode of the rescue pass every certified candidate is emitted, since the base pass never keeps predictors).

### 7.4 Theorem (entropic certificate)

**Claim.** Let $G'$ be a candidate as in 6.1 (possibly with further parents deleted from any non-predictor node, Section 7.5). If the LP of 7.3 implies the `markov` targets, or the single-facet condition holds and it implies the `relabel` targets, then a QC gap in $G'$ implies a QC gap in $G$.

**Proof.** Lift as in 6.3: $P\in\mathcal Q(G')\subseteq\mathcal Q(G)$, $H(s\mid X_1)=0$, and by (F2) $P$ satisfies every observable d-separation of $G'$. Let $M$ be a classical model of $G$ for $P$. Its joint entropy vector lies in the Shannon cone, satisfies the local Markov equalities of $G$ and the hypotheses $H(s\mid X_1)=0$ and $\mathcal C(G')$. The LP implication therefore forces the target CMIs to vanish, so $M$'s joint is Markov to $G'$ (or to $G''$, and then the substitution $L:=s$ gives a model of $G'$). Hence $P\in\mathcal C(G')$ and, coarse-graining $X_1$, $P'\in\mathcal C(G')$. $\square$

### 7.5 The search procedure, stated edge-first

The unit of search is a **candidate deletion**: a visible node $s$ and a set $D\subseteq\mathrm{Pa}(s)$ of its parents in the effective DAG to delete, where a "parent" is either a visible parent ($v\to s$) or a latent facet containing $s$ (a latent-to-observed edge). Deleting $D$ dictates everything else; nothing further is chosen.

1. **Who may predict.** After the deletion $s$ depends only on $K=\mathrm{Pa}(s)\setminus D$, so a predictor set $X_1$ is usable iff $X_1$ *sees* every element of $K$ (each kept facet contains a predictor, each kept visible parent is a parent of a predictor or is itself a predictor) and $K$ contains at least one facet shared with a predictor (it carries the private randomness of $s$ in the lift). The usable predictor sets are the covers of $K$ by such nodes.
2. **Which conditional independences.** $G'$ is $G$ with $D$ deleted (predictor retained), so the hypothesis set $\mathcal C(G')$, the elementary observable d-separations of $G'$, is fixed by $D$.
3. **What has to be proven.** The local Markov equalities of $G'$ (or of the relabelled $G''$) from Shannon, Markov($G$), $H(s\mid X_1)=0$ and $\mathcal C(G')$.

The implementation runs the search in the opposite order: for each predictor set $X_1$ (single nodes by default, pairs optionally) and each latent sibling $s$ of $X_1$, it takes the maximal deletion $D_0=\mathrm{Pa}(s)\setminus\mathrm{common}(s)$, everything $X_1$ cannot see. Every minimal candidate $(s,D_0)$ is enumerated this way, from each of its covers. Larger deletions $D\supsetneq D_0$ (parents $X_1$ could have kept) and deletions at nodes other than $s$ are explored by the greedy fixed-point loop of 7.6, which is order-dependent and budgeted, not exhaustive. The d-separation test of Section 6.3 is the special case of step 3 that ignores $\mathcal C(G')$ and needs no LP; it is run first and the LP only where it fails. For every candidate the code records $D$ (`deleted`), the predictor set, the target mode and which certificate closed it, so each entropic step in a certificate can be re-derived by hand.

### 7.6 Extra deletions (`QmDAG._entropic_extra_deletions`)

The hypotheses do not mention which node the deletion concerns, so the same LP can certify deleting a parent $p$ of any non-predictor node $t$: the per-edge target $I(t{:}p\mid\mathrm{Pa}_{G'}(t)\setminus p)\le0$. Deleting an edge enlarges $\mathcal C(G')$ (more d-separations), so the hypotheses are recomputed from the current candidate after each deletion and the loop is run to a fixed point. Because deleting edges also enlarges non-descendant sets, the final candidate is verified as a whole with 7.3 (both target sets); if that fails the candidate without extra deletions is kept. Soundness is then exactly Theorem 7.4 for the final candidate. Deletions never touch a predictor (it must keep seeing $\mathrm{common}(s)$), and never remove the last facet a predicted node shares with its predictor: without it the hypotheses $H(s\mid X_1)=0$ and $s\perp X_1$ (an observable d-separation of the resulting $G'$) are jointly satisfiable only by a constant $s$, and the LP would certify a vacuous statement while the lift no longer exists.

### 7.7 Worked example: the KPC structure

$G_1$: visible $C,D,E,F$; facets $A=\{E,F\}$, $B=\{D,F\}$; edges $C\to D$, $C\to E$, $D\to E$. Predictor $F$, predicted $E$: $\mathrm{common}(E)=\{A\}$, $\mathrm{others}(E)=\{C,D,\text{noise}\}$. The d-separation test fails ($F\leftarrow B\to D$ is open). The candidate $G'$ has $E$ with parent $A$ only and d-separates $E$ from $\{C,D\}$. With the hypothesis $E\perp CD$ the LP certifies the `relabel` targets ($A:=E$), reproducing the authors' Lemma 1 without the equality $F_S=E$ or the explicit split. In split-predictor mode the output is $C\to D$, $D$–$F$ quantum, $E$–$F$ classical: the Bell variant `QG_Bell5`, in one step (`tests/test_fritz_entropic.py`).

---

## 8. Node bookkeeping

Every transformation builds a `LabelledDirectedStructure`/`LabelledHypergraph` over named nodes and re-indexes them to $0..m-1$; copies are named `"<s>_copy"` only during construction. The children memo of the closure is keyed by unlabelled id, which is legitimate because every piggyback is label-equivariant (this is why order dependence in Section 2 must be enumerated rather than fixed).

## 9. Search and certificates (`qc_gap_search.py`)

`ClosureExplorer` expands every reachable structure (up to relabelling) exactly once under all tricks and records each `Transition(trick, params, source, target)`. Reachability restricted to any subset of tricks is a graph query over the recorded transitions, which gives, for each trick, the inputs provable with it alone and the inputs provable only with it. `prove_gaps` closes the set of proven inputs under implication from the seeds (`known_QC_gaps.py`) and attaches to every proven input a certificate: the shortest chain of transitions down to a named seed, preferring tricks of the base search. The entropic trick is expensive (one LP per target per candidate) and is applied in a **rescue phase** (`rescue`) to the inputs the base closure leaves unproven; its children are then expanded with the base tricks. `ENTROPIC_STATS` in `quantum_mDAG.py` tallies how often each certificate succeeds or fails.

## 10. Results on the 4-node census

All counts are up to relabelling (distinct unlabelled ids). The inputs are the 4-node mDAGs whose edges respect the order $0<1<2<3$ and that are not provably algebraic, with every latent quantum and the Bell seeds removed. (`tests/test_baseline_slow.py` pins these values.)

| quantity | value |
|---|---|
| inputs up to relabelling (2759 labelled structures) | 990 |
| proven with the base tricks, loose conditioning (visible grandparents only) | 920 |
| proven with the base tricks, strict conditioning (Section 3, condition 2) | 918 (72 remaining) |
| additionally proven by the entropic rescue | 54 (972 proven, 18 remaining) |
| structures expanded by the search | 9046 |
| wall time (base closure plus rescue) | 19 min |

Per trick (strict conditioning, rescue included): "alone" is the number of inputs provable using only that trick (Fritz together with marginalization, the entropic trick together with Fritz and marginalization); "only" is the number no longer provable when that single trick is removed.

| trick | alone | only |
|---|---|---|
| point distribution | 860 | 221 |
| interruption | 7 | 0 |
| conditioning | 292 | 23 |
| naive marginalization | 515 | 0 |
| teleportation marginalization | 540 | 0 |
| Fritz (d-separation) | 575 | 4 |
| entropic Fritz | 615 | 54 |

Entropic certificates attempted during the rescue (per predictor–target candidate, split structures included): admissible by d-separation 754; beyond d-separation, `markov` 58, `relabel` 70, failed 549; joint targets certified 33 / failed 54; extra-deletion candidates verified 155, none failed. Among candidates that d-separation rejects, the LP therefore certifies roughly one in five; success is common but far from universal, and (Section 7.3) a failure of the LP is not a proof that the implication is false.

## 11. Open questions

* **Completeness of the entropic certificate.** The failures recorded in `ENTROPIC_STATS` are failures of the LP, not necessarily of the piggyback (Section 7.3). A certificate that can re-encode latents, or a theorem that the `markov`/`relabel` pair is complete for some class of candidates, would remove the need for per-instance certificates.
* **The pp-free edge-deletion piggyback** (KPC Corollary 3): the same LP without the perfect-prediction hypothesis certifies deleting edges justified by the new graph's independences alone. The code can already check it (`QmDAG._entropic_certificate(..., predicted=())`), but it is not exposed as a trick.
* **Exact certificates.** Farkas multipliers are floating point; rationalising them and re-verifying the combination exactly is cheap and would make every entropic step a checkable proof.
