# Paper redesign — competitive submission plan

**Working title.** *One Backend, Many Frameworks: Quantifying Integration Cost
and Verifying Cross-Framework Parity for NPU-Accelerated LLM Inference.*

**Target venues.** MLSys or EuroSys (primary); ASPLOS becomes viable because
Pillar 2 is a verification/testing mechanism (cf. NNSmith @ ASPLOS'23); ISSTA/ICSE
as a fallback for the verification contribution alone.

> Honest framing. The original "Adapter+Factory / stable C API / header-only
> shim" framing is **not publishable as novelty** — those are textbook patterns
> (GoF [13]) already used verbatim by ONNX Runtime EPs, ggml-backend, TVM BYOC
> [5], OpenXLA PJRT, and ExecuTorch delegates. We **demote** them to the
> *substrate* (an engineering artifact, Section 3) and build two *measured*
> research contributions on top of it. Each pillar is a falsifiable claim with a
> baseline and a metric.

---

## 1. Thesis and research questions

A single accelerator software stack is only valuable if many ML frameworks can
reach it; naïve pairwise integration scales O(N×M) and the dominant cost is
*recurring maintenance* under independent upstream churn. A common "narrow-waist"
backend boundary is widely *assumed* to fix this, and to be *safe* to share — but
**neither assumption has been measured.** We ask:

- **RQ1 (cost).** How much does a shared backend boundary actually reduce
  integration and *maintenance* cost versus pairwise integration, on real release
  histories and in a controlled multi-framework port? Is the reduction O(N×M)→
  O(N+M) in practice, or do escape hatches erode it?
- **RQ2 (correctness).** When one backend serves heterogeneous front-end ABIs
  (graph-compile EP vs op-dispatch ggml vs AOT-delegate), do the front-ends
  actually produce *equivalent* results — and can divergences be found and
  *localized* automatically across the front-end / boundary / kernel layers?

The two pillars are complementary: RQ1 establishes *why* the shared waist is worth
it; RQ2 delivers the *mechanism* that makes sharing trustworthy (and is the main
technical novelty).

**Contributions.**
1. **A cost model and the first empirical study** of NPU/accelerator-to-framework
   integration cost, mining the integration history of production backends
   (ONNX Runtime EPs, ggml backends, TVM BYOC, ExecuTorch delegates) plus a
   controlled N=1 accelerator × K=4 framework port measured both ways. (RQ1)
2. **GAIA-Diff**, a cross-framework **parity oracle + differential/metamorphic
   testing** system for a shared accelerator boundary, with **layer-localizing**
   divergence attribution (front-end lowering vs API/weight-delivery vs kernel)
   and **quantization-aware tolerance**. (RQ2)
3. A **conformance suite** for the common backend API and an **open artifact**
   (the three real adapters + harness) enabling reproducible cross-framework
   testing.

---

## 2. Background & gap (positioning)

Per-framework backend hooks already exist (ONNX Runtime EP, ggml-backend, TVM
BYOC [5], PJRT, `TfLiteDelegate`, ExecuTorch delegate). Compiler IRs (MLIR [6],
StableHLO, TVM Relay [11], Glow [7]) standardize *representation*. **Neither line
of work measures integration/maintenance cost, nor verifies equivalence of the
*same accelerator* reached through *different front-end ABIs*.** Differential and
metamorphic testing of DL systems exists — CRADLE [8], LEMON [9], EAGLE [10],
NNSmith [11a], Tzer [12a] — but targets a *single* framework or compiler stack.
The unaddressed gap, and our delta:

| Prior art | What it tests | What it does **not** |
|---|---|---|
| CRADLE/LEMON/EAGLE | one framework across libraries/backends | not across heterogeneous serving frameworks; no shared-boundary conformance |
| NNSmith/Tzer | one DL compiler's codegen | not cross-framework parity; no front-end-ABI localization |
| PJRT/BYOC/EP | a plug interface | no cost measurement; no parity guarantee |

---

## 3. Substrate (engineering artifact, not a claim)

A minimal narrow-waist backend API — a stable C ABI for model loading
(`npu_model`) plus a thin device/memory façade — implemented once and reached by
three real adapters (ONNX Runtime EP, ggml-backend, ExecuTorch delegate) over one
NPU stack. We describe it in ~1.5 pages and **explicitly cite it as conventional**
(GoF [13], narrow-waist [14,15], information hiding [16]). It exists to *enable
the measurements*, not as novelty. This honesty preempts the "re-describing known
patterns" rejection.

---

## 4. Pillar 1 — Quantifying integration cost (RQ1)

### 4.1 Cost model
Define total cost `C = Σ_build + Σ_maint`, where the *maintenance* term dominates
and is driven by upstream change events. For F frameworks and A accelerators:
- Direct: `C_direct = Σ_{f,a} ( build(f,a) + Σ_t churn(f,a,t) )` over release
  timeline `t` (≈ O(F·A)).
- Mediated: `C_api = Σ_f adapt(f) + Σ_a impl(a) + Σ_t churn_to_API(·,t)` (≈
  O(F+A)), **plus** an *escape-hatch tax* `E` for features the waist cannot
  express. We test whether `E` stays small.

### 4.2 Measurement method
- **Historical mining.** For production backends with public histories (ONNX
  Runtime EPs: CUDA/TensorRT/OpenVINO/QNN/CoreML; ggml: CUDA/Vulkan/Metal/SYCL;
  ExecuTorch delegates; TVM BYOC), extract per-integration **adapter LOC**, **code
  churn over time**, **breakage commits** (commits that fix a build/behavior break
  caused by an upstream framework/driver release), and **ABI-break events**. This
  yields real `churn(f,a,t)` curves.
- **Controlled port (internal validity).** Integrate one NPU into **K=4
  frameworks two ways**: (i) pairwise/direct and (ii) via the common API. Measure
  adapter LOC, developer time, defects, and — crucially — **resilience under
  injected upstream change**: replay N synthetic but realistic API/ABI changes
  (rename, struct field add, signature change, dtype addition) and count how many
  break each design and the fix effort.

### 4.3 Baselines & metrics
Baseline = pairwise integration (status quo). Metrics: adapter LOC, churn/year,
breakages per upstream release, mean-time-to-fix, and the realized exponent in
`C ∝ (F·A)^α` vs `(F+A)^β`. **Falsifiable:** if `E` (escape-hatch tax) or
`churn_to_API` is large, the waist does *not* pay off — we report that outcome
either way.

---

## 5. Pillar 2 — GAIA-Diff: cross-framework parity verification (RQ2)

> Full system design: [`gaia_diff_design.md`](gaia_diff_design.md) (+ diagrams in
> `gaia_diff_design.puml`).

### 5.1 Problem
One backend, three front-end ABIs. A bug can live in (a) a front-end's lowering
to the common container, (b) the API/weight-delivery boundary, or (c) the shared
kernel. We want to **find** equivalence violations and **localize** the
responsible layer automatically.

### 5.2 Mechanism (the novelty)
1. **Tri-source differential testing.** For a model `m` and input `x`, obtain
   `y_f = run(framework_f, m, x)` for each `f`, plus a deterministic
   `y_ref = reference(m, x)` (CPU fp32 / fixed-seed). Assert pairwise equivalence
   under a tolerance model.
2. **Layer-localizing attribution.** Decompose each run into front-end-lowered
   container `nnc_f`, the bound weights, and the kernel outputs. By **swapping
   layers** (same `nnc`, different front-end loader; same kernel, different
   front-end activation binding) we attribute a divergence to lowering vs
   weight-delivery vs kernel — a capability single-framework testers lack.
3. **Quantization-aware tolerance.** Equivalence for block-quantized weights
   (Q4_K/Q8_0) uses a format-derived tolerance and metamorphic relations
   (dequant-then-compute ≡ compute path), not naive bit-equality.
4. **Test generation.** Combine NNSmith-style random valid graphs [11a] (op/dtype
   coverage) with real LLM subgraphs; **delta-minimize** failing cases to a
   minimal divergent subgraph + dtype.
5. **Boundary conformance suite.** Encode the API contract (versioning,
   residency/ownership, weight-delivery source kinds) as executable conformance
   tests any accelerator implementation must pass.

### 5.3 Why it's new vs CRADLE/LEMON/NNSmith/EAGLE
Those test *within* one framework/compiler. GAIA-Diff treats **equivalence across
heterogeneous serving-framework ABIs through one shared accelerator boundary** as
the property, and adds **front-end-ABI-aware divergence localization** and a
**conformance suite for the boundary itself** — neither exists in prior work.

---

## 6. Evaluation plan

| RQ | Setup | Metrics | Baseline | Expected result |
|---|---|---|---|---|
| RQ1 | history mining (≥12 real integrations) + controlled K=4 port | LOC, churn/yr, breakages/release, MTTF, fitted α/β | pairwise | near-additive maintenance; small escape-hatch tax `E` (report if not) |
| RQ2 | ≥2 accelerators × 3 frameworks (ORT, llama.cpp, ExecuTorch) + CPU ref; random + real LLMs | #divergences found, localization accuracy, op/dtype coverage, time-to-find, overhead | single-framework differential testing (CRADLE/NNSmith-style) | finds cross-framework-only bugs; localizes layer with high accuracy |

**Generalization requirement (be honest in the paper):** ≥2 distinct accelerators
and ≥3 frameworks; a single vendor NPU + stub is a case study, not a result. The
mock-NPU + reference-kernel mode lets RQ2 run without silicon, but the paper needs
at least one real accelerator for credibility.

---

## 7. Threats to validity
- **External:** results may not generalize beyond the studied frameworks/
  accelerators → mitigate with breadth (≥3 frameworks, ≥2 accelerators) and by
  releasing the harness.
- **Construct:** "integration cost" via LOC/churn is imperfect → triangulate with
  breakage counts and the controlled injected-change experiment.
- **Internal:** historical breakages may have confounds (refactors) → manual
  classification with inter-rater agreement.
- **Parity tolerance:** wrong tolerance hides/forges bugs → derive tolerances from
  numeric formats and validate against known-good references.

---

## 8. Related work (one paragraph each)
Accelerator plug interfaces (ONNX Runtime EP, ggml-backend, BYOC [5], PJRT,
ExecuTorch delegate); ML compilers/IRs (TVM [11], MLIR [6], Glow [7], OpenXLA);
DL differential/metamorphic testing (CRADLE [8], LEMON [9], EAGLE [10], NNSmith
[11a], Tzer [12a]); classic compiler differential testing (Csmith [17], EMI
[18]); systems-layering theory (hourglass [14,15], information hiding [16]);
DSAs & benchmarking (Golden Age [1], TPU [2], MLPerf [12]).

---

## 9. Why this version clears the bar
- **Novelty** moves from "known patterns" to (a) a **measured** cost study nobody
  has published, and (b) a **new verification mechanism** (cross-ABI parity +
  layer localization + boundary conformance).
- **Evaluation** is quantified, with baselines and falsifiable hypotheses.
- **Artifact** (three real adapters + GAIA-Diff + conformance suite) supports
  reproducibility — increasingly required at MLSys/EuroSys/ASPLOS.
- **Risk**: still needs ≥1 real accelerator and ≥3 live frameworks; the
  cost-study mining must be done rigorously (manual breakage classification). Both
  are effort, not novelty risk.

---

## 10. Minimal path to a submission
1. Land the three real adapters end-to-end on one real NPU (correctness first).
2. Build GAIA-Diff (tri-source diff + localization + minimizer) — **lead with
   this**; it is the defensible technical core.
3. Run the controlled K=4 injected-change experiment (cheap, high-signal for RQ1).
4. Mine ≥12 real integrations for the historical cost curves.
5. Add a 2nd accelerator for generalization.

---

## References
[1] Hennessy & Patterson, "A New Golden Age for Computer Architecture," CACM 2019.
[2] Jouppi et al., "In-Datacenter Performance Analysis of a TPU," ISCA 2017.
[5] Chen et al., "Bring Your Own Codegen to Deep Learning Compiler," 2021.
[6] Lattner et al., "MLIR," CGO 2021.
[7] Rotem et al., "Glow," arXiv:1805.00907, 2018.
[8] Pham et al., "CRADLE: Cross-Backend Validation of DL Libraries," ICSE 2019.
[9] Wang et al., "LEMON: Deep Learning Library Testing via Guided Mutation," ESEC/FSE 2020.
[10] Wang et al., "EAGLE: Creating Equivalent Graphs to Test DL Libraries," ICSE 2022.
[11] Chen et al., "TVM," OSDI 2018.
[11a] Liu et al., "NNSmith: Generating Diverse and Valid Test Cases for DL Compilers," ASPLOS 2023.
[12] Reddi et al., "MLPerf Inference Benchmark," ISCA 2020.
[12a] Liu et al., "Coverage-Guided Tensor Compiler Fuzzing with Joint IR-Pass Mutation (Tzer)," OOPSLA 2022.
[13] Gamma, Helm, Johnson, Vlissides, "Design Patterns," 1994.
[14] Akhshabi & Dovrolis, "The Evolution of Layered Protocol Stacks Leads to an Hourglass-Shaped Architecture," SIGCOMM 2011.
[15] Beck, "On the Hourglass Model," CACM 2019.
[16] Parnas, "On the Criteria to Be Used in Decomposing Systems into Modules," CACM 1972.
[17] Yang et al., "Finding and Understanding Bugs in C Compilers (Csmith)," PLDI 2011.
[18] Le et al., "Compiler Validation via Equivalence Modulo Inputs (EMI)," PLDI 2014.
