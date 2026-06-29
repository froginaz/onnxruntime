# Formalizing GAIA-Diff: a verified parity oracle and proved boundary contracts

This turns Pillar 2 ([`gaia_diff_design.md`](gaia_diff_design.md)) from empirical
differential testing into **verified differential testing**: the equivalence
oracle is *proved sound*, so any divergence it reports beyond the certified
tolerance is — by theorem — a real defect, not floating-point noise. We do **not**
attempt to verify a whole transformer end-to-end (infeasible). We make the small,
load-bearing parts formal: the **tolerance bound** (Idea 1), a set of
**metamorphic invariants** (Idea 3), and the **resource/ABI protocol** (Idea 5).

> Framing. The oracle becomes **sound but intentionally incomplete**: no false
> positives from legitimate rounding (proved), but it may miss a bug whose effect
> stays within the certified `ε`. Test generation/fuzzing
> ([`gaia_diff_design.md`](gaia_diff_design.md) §5) covers the incompleteness.
> "Proved-sound oracle + fuzzing" mirrors the verified-compiler + fuzzing line
> (CompCert [9] + Csmith [10]).

---

## 0. Foundation: a numeric-parameterized semantics of the boundary

Give the canonical container `nnc` two interpretations over the same op DAG `g`:

- **Ideal semantics** `⟦g⟧_ℝ : ℝⁿ → ℝᵐ` — exact real arithmetic.
- **Machine semantics** `⟦g⟧_{𝔽,δ} : 𝔽ⁿ → 𝔽ᵐ` — IEEE-754 [1] under a dtype
  assignment `δ` (per-tensor element type: fp32/fp16/bf16/Q4_K/…).

"Parity" is only definable once these exist. Each front-end's run is
`y_f = ⟦L_f(m)⟧_{𝔽,δ_f}(x)`; the reference is `y_ref = ⟦nnc⟧_{𝔽,δ_ref}` (or `⟦⟧_ℝ`
when affordable). All theorems below are statements about these two semantics.

---

## 1. Idea 1 (core): a certified floating-point tolerance

### 1.1 IEEE-754 model and notation
We use the standard model [2, Higham §2]: for `op ∈ {+,−,×,÷}` and `√`,
```
fl(a op b) = (a op b)(1 + δ),   |δ| ≤ u,
```
with unit roundoff `u = 2^{-24}` (fp32), `2^{-11}` (fp16), `2^{-8}` (bf16). Define
`γ_n = nu / (1 − nu)` for `nu < 1`. (With a fused multiply-add the constants
roughly halve.)

### 1.2 Per-op certified bounds (the building blocks)

**Dot product / MatMul.** For `x,y ∈ 𝔽^k`, recursive-sum dot product satisfies
[2, Thm 3.1]:
```
| fl(xᵀy) − xᵀy | ≤ γ_k · |x|ᵀ|y|.
```
Hence for `C = AB`, `A∈𝔽^{m×k}`, `B∈𝔽^{k×n}`, componentwise:
```
| fl(C)_{ij} − C_{ij} | ≤ γ_k · (|A||B|)_{ij}.          (MatMul bound)
```
The bound depends only on the **contraction length k** and the operand
magnitudes — exactly the data the Tap already captures.

**Sum / Add / Mul / bias.** `|fl(Σ_{i=1}^n x_i) − Σ x_i| ≤ γ_{n−1} Σ|x_i|`;
elementwise `±,×` contribute a single `(1+δ)`.

**Softmax (with max-subtraction).** Computing
`σ(x)_i = e^{x_i−m}/Σ_j e^{x_j−m}`, `m = max x`, is well-conditioned and
overflow-free; Blanchard–Higham–Mary [3] give a rigorous bound: the computed
softmax `σ̂` satisfies, for each i,
```
| σ̂_i − σ_i | ≤ ( (n+2)u + ε_exp + O(u²) ) · σ_i,
```
where `ε_exp` is the library's certified relative error of `exp` (a few ulp). We
import `ε_exp` as a checked parameter (e.g. from a Gappa-verified polynomial).

**RMSNorm / LayerNorm.** `y = x / √(mean(x²)+ε) · γ`. Composition of a
length-`d` sum-of-squares (`γ_{d}` bound), one `rsqrt` (relative bound `≤ c·u`
for a correctly-rounded/certified routine), and an elementwise scale. Bounds
compose multiplicatively (§1.3).

**RoPE / elementwise transcendental.** Each `sin/cos` is treated as a certified
relative-error primitive `|fl(φ(x))−φ(x)| ≤ ε_φ|φ(x)|`, `ε_φ` from a Gappa/FPTaylor
proof of the implementation polynomial.

### 1.3 Structural composition (DAG → whole-subgraph bound)
Errors propagate along `g` by abstract interpretation. For a node `f` with inputs
carrying absolute errors `e_in` over an input **enclosure** `I` (interval/affine):
```
e_out ≤ sup_{ξ∈I} ‖∇f(ξ)‖ · e_in   +   localrounding(f).
```
For linear ops `∇f` is exact (no enclosure needed); for nonlinear ops we bound
`‖∇f‖` over `I` with **affine arithmetic** (à la FPTaylor [5] / Daisy [6]). The
result is a certified per-output bound `ε(g, δ, I)`.

### 1.4 The Sound-Oracle theorem

> **Theorem 1 (Sound oracle).** Let `g` be an op DAG, `δ` a dtype assignment,
> `I ⊇ X` an input enclosure. Let `ε(g,δ,I)` be the bound from §1.2–1.3. Then
> ```
> ∀ x ∈ X :  ‖ ⟦g⟧_{𝔽,δ}(x) − ⟦g⟧_ℝ(x) ‖ ≤ ε(g,δ,I).
> ```
> **Corollary (no false positives).** If two correct implementations of `g`
> agree with `⟦g⟧_ℝ` then `‖y_f − y_g‖ ≤ 2ε`. Hence the differential check
> `‖y_f − y_ref‖ > ε_f + ε_ref ⇒ a genuine defect` raises **no alarm from
> rounding alone**.

**Proof obligations (mechanized):**
- **PO1** per-op local bounds (MatMul/sum/scale): **Flocq/VCFloat** [4, 7]
  (the inductive `γ_n` lemmas) + **Gappa** [8] for concrete unrolled kernels.
- **PO2** nonlinear primitives (`exp,rsqrt,sin`): certified relative bounds via
  **Gappa/FPTaylor** on the implementation polynomial.
- **PO3** DAG composition soundness: affine-arithmetic propagation with a
  checkable certificate (**Daisy/FPTaylor** export, or a Coq composition lemma).
- **PO4** input enclosure `I ⊇ X`: from model/static range analysis (documented
  assumption; e.g. post-softmax ∈ [0,1], normalized activations bounded).

### 1.5 Quantized weights (block formats)
Block dequant gives, per weight element, `|w − ŵ| ≤ Δ_blk` (half the block step +
scale rounding) — a *bounded perturbation*, even though dequant is nonlinear.

> **Theorem 2 (quant sensitivity).** For weights perturbed by `‖W − Ŵ‖ ≤ Δ`,
> `‖⟦g⟧(Ŵ,x) − ⟦g⟧(W,x)‖ ≤ L_W(g,I)·Δ`, where `L_W` is the certified
> weight-sensitivity (condition number) of `g` over `I`. Total bound:
> `ε_total = ε(g,δ,I) + L_W·Δ`.

So quantized parity uses a **format-derived** tolerance, not a guessed `rtol`.

### 1.6 Worked tooling examples

**Gappa** — a concrete unrolled kernel (illustrative; Gappa discharges the exact
constant):
```
# binary32, round-to-nearest
@rnd = float<ieee_32, ne>;
p0 rnd= a0 * b0;  p1 rnd= a1 * b1;
s  rnd= p0 + p1;                  # computed length-2 dot product (no FMA)
exact = a0*b0 + a1*b1;            # ideal
{ |a0|<=1 /\ |b0|<=1 /\ |a1|<=1 /\ |b1|<=1 ->
  |s - exact| in ? }              # Gappa proves a tight bound ~ gamma_2 * 2
```
Gappa verifies the *concrete* constant; the **parameterized** `γ_k` MatMul bound
is proved once by induction in **Flocq/VCFloat**:
```coq
(* sketch *)
Lemma dotprod_error : forall k (x y : list (ftype Tsingle)),
  length x = k -> length y = k ->
  Rabs (FT2R (dotprod x y) - dotprodR x y)
    <= gamma k * dotprodR (map Rabs x) (map Rabs y).
```
Then a Coq tactic composes per-op lemmas along the captured DAG to emit
`ε(g,δ,I)` with a proof term — the certificate the oracle ships.

### 1.7 Honest limits
- Bounds are **loose for long reductions** (`γ_k` grows with k) and
  ill-conditioned softmax pre-max-subtraction → use affine arithmetic and report
  bound tightness as a metric; tightening is itself a research sub-result.
- **Assumptions are explicit**: no overflow/NaN on `I`, correctly-rounded or
  certified transcendental routines, input range `I ⊇ X`. The paper states them.
- The reference `⟦⟧_ℝ` is approximated by extended precision where exact reals are
  infeasible; the gap is bounded and folded into `ε`.

---

## 2. Idea 3: metamorphic invariants as proved theorems

Over the §0 semantics, the GAIA-Diff metamorphic relations become *theorems*; a
violation is then provably a bug (not a heuristic).

> **Theorem 3 (Residency invariance).** If `deliver(W, COPY_TO_DEVICE)` and
> `deliver(W, REFERENCE_HOST)` yield byte-equal device buffers, then
> `⟦plan⟧` is identical. *Proof:* data movement is the identity on bytes ⇒ the
> downstream pure function is applied to equal inputs. (Directly validates the
> `npu_blob_t` residency design and risk R3.)

> **Theorem 4 (Partition compositionality).** If `g = g₁ ∘ g₂` at a cut, and each
> sub-implementation refines its `nnc` semantics within `ε_i` over the relevant
> enclosure, then the composed pipeline refines `g` within `ε₁ + L·ε₂` (Lipschitz
> `L` of `g₁` over the enclosure). Justifies CPU-fallback / partial-offload parity.

> **Theorem 5 (Layout / dtype-cast invariance).** Semantics-preserving layout
> permutations (NCHW↔NHWC) and exact-width casts preserve `⟦·⟧_ℝ`; lossy casts
> incur a bounded, composed error (§1).

These are small Coq/Isabelle proofs; the hard numeric content is reused from §1.

---

## 3. Idea 5: the resource/ABI contract as a checked protocol

The lifetime/ownership/versioning contract is a **protocol**, model-checked rather
than ASAN-tested.

- **TLA+ / TLC.** Specify init/shutdown reference counting, `model load/free`
  pairing, and the residency lifetime rule (`REFERENCE_HOST` blob must outlive the
  model). Model-check invariants: *no use-after-free*, *no double-free*,
  *refcount balanced at quiescence* (risks R3, R6). The conformance suite is then
  **generated from the spec** (model-based testing), so `U-MODEL`/`C-WMX` in
  [`test_plan.md`](test_plan.md) are spec-derived, not ad hoc.
- **Separation logic / typestate (Iris, VeriFast).** Encode `owns_weights` and
  the blob-lifetime obligation as ownership; statically reject use-after-free in
  adapter code.
- **ABI compatibility.** Define a binary-compatibility refinement and prove the
  `struct_size`/`api_version` negotiation accepts compatible and rejects
  incompatible host/plugin pairs (risk R4).

> **Theorem 6 (Lifetime safety).** Under the TLA+ protocol, every reachable state
> satisfies: no model dereferences a freed weight blob, and the device-handle
> refcount returns to zero at shutdown. *Established by TLC over the bounded model;
> the conformance tests instantiate the counterexample-free traces.*

---

## 4. What each contributes to the paper

| Idea | Formal artifact | Strengthens | Tools |
|---|---|---|---|
| 1 | certified `ε(g,δ,I)` + Sound-Oracle theorem | RQ2 oracle soundness (no false positives) | Flocq/VCFloat, Gappa, FPTaylor, Daisy |
| 3 | MR theorems (residency/partition/layout) | divergence = provable bug; R3 | Coq/Isabelle |
| 5 | TLA+ protocol + lifetime theorem | R3/R4/R6; spec-derived conformance | TLA+/TLC, Iris/VeriFast |

**Minimal formal core (do these first):** Idea 1 for MatMul + Softmax + RMSNorm
(covers a transformer block's numeric spine), Idea 3 Theorem 3 (cheap, validates
the residency design), Idea 5 TLA+ lifetime model. This already yields a
*verified* oracle and *proved* resource safety — enough to claim the
"verified differential testing" result and open PLDI/CAV/OOPSLA in addition to
MLSys/ASPLOS.

## 5. Mapping to the repo
- `ε` inputs (contraction length, operand magnitudes, dtype, op DAG) = the Tap's
  captured Trace; the Reference Executor extends `MatMulF32` into the `⟦·⟧`
  interpreter for both `𝔽` and extended-precision `ℝ`.
- Theorem 3 is exactly the `COPY_TO_DEVICE`/`REFERENCE_HOST` contract in
  `npu_blob_t`/`npu_model.h`.
- The TLA+ protocol formalizes `Initialize/Shutdown`, `npu_model_load/free`,
  `npu_model_owns_weights`, and `npu_weight_residency_t` already in the C ABI.

---

## References
[1] IEEE Std 754-2019, *Standard for Floating-Point Arithmetic*.
[2] N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002.
[3] P. Blanchard, N. J. Higham, T. Mary, "Accurately Computing the Log-Sum-Exp and Softmax Functions," *IMA J. Numerical Analysis*, 2021.
[4] S. Boldo, G. Melquiond, *Computer Arithmetic and Formal Proofs* (Flocq), 2017.
[5] A. Solovyev et al., "Rigorous Estimation of Floating-Point Round-off Errors with Symbolic Taylor Expansions (FPTaylor)," *ACM TOPLAS*, 2018.
[6] E. Darulova, V. Kuncak et al., "Daisy — Framework for Analysis and Optimization of Numerical Programs," *TACAS*, 2018 (and "Sound Compilation of Reals," *POPL*, 2014).
[7] A. W. Appel, A. Kellison et al., "VCFloat2: Floating-Point Error Analysis in Coq," *CPP*, 2023.
[8] M. Daumas, G. Melquiond, "Certification of Bounds on Expressions Involving Rounded Operators (Gappa)," *ACM TOMS*, 2010.
[9] X. Leroy, "Formal Verification of a Realistic Compiler (CompCert)," *CACM*, 2009; G. Necula, "Translation Validation," *PLDI*, 2000.
[10] X. Yang et al., "Finding and Understanding Bugs in C Compilers (Csmith)," *PLDI*, 2011.
[11] L. Lamport, *Specifying Systems: The TLA+ Language and Tools*, 2002.
[12] R. Jung et al., "Iris from the Ground Up," *JFP*, 2018; J. C. Reynolds, "Separation Logic," *LICS*, 2002.
[13] A. Zeller, R. Hildebrandt, "Simplifying and Isolating Failure-Inducing Input," *IEEE TSE*, 2002.
