# GAIA-Diff: a cross-framework parity verification system for a shared accelerator boundary

This is the detailed system design for Pillar 2 of the paper
([`paper_proposal.md`](paper_proposal.md)). GAIA-Diff verifies that the **same
model + weights** produce **equivalent results** when reached through
heterogeneous front-end ABIs (ONNX Runtime EP, ggml-backend, ExecuTorch
delegate) over **one shared accelerator boundary**, and — its core novelty —
**localizes** any divergence to the responsible layer (front-end lowering,
weight delivery, I/O binding, or kernel).

> Diagrams: PlantUML in [`gaia_diff_design.puml`](gaia_diff_design.puml).

---

## 1. Problem model

### 1.1 The shared pipeline, factored into stages
For a framework `f ∈ F`, a model `m`, and input `x`, inference factors through the
common boundary into four stages:

```
m ──L_f──▶ (nnc, W, iomap) ──D──▶ W_dev ──P──▶ plan ──E──▶ y
   lowering   canonical IR    weight        compile   execute
   (per f)    at the waist    delivery       (shared)  (shared kernels)
```

- **L_f (Lowering, per-framework):** the adapter converts `m` (`.onnx` graph /
  ggml cgraph / ExecuTorch edge program) into the NPU-native container:
  topology `nnc`, named weights `W`, and an I/O map `iomap` (which framework
  tensor ↔ which nnc input/output, with dtype + layout).
- **D (Weight delivery, shared API):** `npu_model_load` ingests `W` via a
  `npu_blob_t` source (memory / file / fd / callback) with a residency mode
  (copy vs reference-host) → device-resident weights `W_dev`.
- **P (Compile, shared):** the NPU stack compiles `nnc` → an execution `plan`.
- **E (Execute, shared kernels):** the plan runs on `x` → output `y`.

**Key observation that the whole design exploits:** `L_f` is the *only*
per-framework stage; `D`, `P`, `E` are **shared**. Therefore a divergence among
front-ends is *either* a lowering/I-O-binding bug in some `L_f`, *or* a
non-determinism/bug in the shared stages that manifests differently — and because
all front-ends funnel through the **same canonical artifacts at the waist**, we
can *capture and substitute* those artifacts to pinpoint the layer. Single-
framework testers (CRADLE, NNSmith, …) have no shared waist to tap and so cannot
do this.

### 1.2 Equivalence property
Let `≈_τ` be approximate equality under a tolerance model `τ` (Section 4). The
verification target is:

> **P-PARITY:** ∀ m∈M, x∈X, ∀ f,g ∈ F: `y_f(m,x) ≈_τ y_g(m,x)` and
> `y_f(m,x) ≈_τ y_ref(m,x)`,

where `y_ref` is a deterministic framework-independent reference. A violation is a
tuple `(m, x, f)` with a **layer attribution** `ℓ ∈ {LOWER, IOBIND, DELIVER,
KERNEL, DTYPE}`.

---

## 2. Architecture

Seven components (see `gaia_diff_design.puml`, diagram 1):

1. **Boundary Tap** — a transparent recording shim inserted at the `npu_model` /
   `npu_api` C ABI. For each run it captures a canonical **Trace**: the `nnc`
   bytes, the weight manifest + per-tensor fingerprints, residency, the bound I/O
   descriptors, and (in *record mode*) per-op intermediate tensors. Zero framework
   changes — it is a wrapper library `libnpu_model` is loaded through.
2. **Reference Executor (oracle)** — a deterministic, framework-independent
   interpreter of `nnc` over `W`, computing in fp32 (and a quantized reference)
   with a **fixed reduction order**. Doubles as the *stage-substitution provider*
   for the Localizer.
3. **Differential Engine** — drives `m` through each `f`, collects `(y_f, Trace_f)`
   and `(y_ref)`, applies `≈_τ`, records violations.
4. **Tolerance Model** — op/dtype-aware bounds; fp via forward-error analysis over
   the op DAG, block-quant via format-defined bounds + metamorphic identities.
5. **Localizer** — attributes a violation to a layer via **stage substitution**
   (Section 3) over the captured canonical artifacts.
6. **Generator + Minimizer** — NNSmith-style valid-graph generation across the op
   set supported by ≥2 frameworks, plus real LLM subgraph extraction; failing
   cases minimized with `ddmin` over nodes + a dtype-demotion lattice.
7. **Conformance Suite** — executable contract tests for the API itself
   (versioning, residency/ownership, source-kind equivalence, manifest
   validation) that any accelerator implementation must pass.

```
 ORT EP ─┐
 ggml   ─┤── Boundary Tap ──▶ Trace_f ──▶ Differential Engine ──▶ violations
 ExecuT ─┘        │                              │
                  ├──▶ Reference Executor ◀───────┤ (oracle + substitution)
                  └──▶ Localizer ◀── stage substitution ──▶ attribution ℓ
   Generator/Minimizer feeds m,x        Tolerance Model feeds τ
   Conformance Suite checks the API contract
```

---

## 3. The core mechanism: stage-substitution localization

A violation `(m,x,f)` is localized by replacing one pipeline stage at a time with
a **known-good (reference) stage** and observing when the divergence vanishes.
Because stages communicate through the canonical waist artifacts, substitution is
well-defined.

### 3.1 Canonical checkpoints
The Tap exposes four comparable artifacts per run:
- `A1 = nnc_f` (topology bytes, canonicalized — Section 3.3).
- `A2 = W_f` (named weights; compared by per-tensor fingerprint, e.g. order-
  independent hash + numeric summary).
- `A3 = iomap_f` (input/output name, dtype, layout, order).
- `A4 = {op_out}` (per-op intermediate tensors, record mode).

### 3.2 Localization algorithm
```
Localize(m, x, f):
  # 1. LOWERING: did f produce a different graph/weights than the reference lowering?
  if A1_f ≇ A1_ref:                      return LOWER        # graph topology differs
  if any tensor t: A2_f[t] ≉ A2_ref[t]:  return LOWER        # weight conversion differs
  # 2. IO BINDING: same graph+weights, but inputs/outputs mapped differently?
  if A3_f ≠ A3_ref (name/dtype/layout/order):
      if rerun(nnc_ref, W_ref, remap(x, A3_f)) ≈_τ y_ref:  return IOBIND
  # 3. DELIVERY: same A1..A3, but device weights differ by source kind/residency?
  Wdev_copy   = deliver(W_f, COPY_TO_DEVICE)
  Wdev_ref    = deliver(W_f, REFERENCE_HOST)
  if fingerprint(Wdev_copy) ≠ fingerprint(Wdev_ref):       return DELIVER
  # 4. DTYPE MAPPING: per-op, does f's dtype assignment differ from canonical?
  if dtype_map(A1_f) ≠ dtype_map(A1_ref):                  return DTYPE
  # 5. KERNEL: artifacts agree, yet output diverges → shared kernel/op bug.
  k = first op where A4_f[k] ≉ A4_ref[k]   (record-mode replay)
  return KERNEL @ op k
```
This is **delta-debugging over the *stage* dimension** rather than the input
dimension — the contribution single-framework differential testers cannot make,
because they have neither a shared canonical IR across front-ends nor a tap at a
common boundary.

### 3.3 Graph canonicalization (`≅`)
`A1_f ≅ A1_ref` is decided by canonicalizing each `nnc` to a normal form:
topological sort with a deterministic tie-break (op type, then sorted input
fingerprints), constant-folding of shape ops, and a whitelist of
semantics-preserving rewrites (e.g. `reshape∘reshape`, transpose cancellation).
Equality is then a hash compare; near-misses fall back to a typed graph-edit
distance to produce a human-readable diff for the report.

### 3.4 Why substitution is sound
Substituting stage `s` with the reference implementation `s_ref` yields a hybrid
pipeline whose output equals the original iff stages `< s` agreed. By the
monotonic structure of the factorization, the **first** stage whose substitution
restores parity is the culprit (standard delta-debugging soundness under the
"interference-free stage" assumption, which holds because stages communicate only
through the typed waist artifacts).

---

## 4. Tolerance model `τ`

Naive bit-equality is wrong (legitimate fp reordering, quantization). A single
global `rtol/atol` is also wrong (hides bugs in long reductions, false-positives
in well-conditioned ops). GAIA-Diff derives a **per-output bound**:

- **Floating point.** For an output produced by an op DAG, bound the error by
  forward-error analysis: `τ(y) ≈ γ_K · κ(op) · u`, where `u` is unit roundoff,
  `K` the accumulation length (e.g. the contraction dim of a matmul), `γ_K =
  Ku/(1-Ku)`, and `κ` an op-specific condition factor. Bounds compose along the
  DAG. This tightens for short reductions (catches more bugs) and loosens for long
  ones (avoids false alarms).
- **Block-quantized weights (Q4_K/Q8_0/…).** Equivalence is checked with a
  **format-derived** bound (max representable error per block) plus **metamorphic
  relations**: `MR-DQ` (dequantize-then-compute ≡ the backend's fused path within
  block tolerance) and `MR-BLK` (per-block scale/zero-point round-trip identity).
- **End-to-end LLM outputs.** For generation, compare **logit distributions**
  (top-k overlap, KL) and **greedy token sequences** at temperature 0, not raw
  activations, to absorb benign last-bit noise while still catching real drift.

The tolerance derivation is itself validated against known-good references and a
mutation oracle (Section 6) to confirm it neither hides injected bugs nor flags
correct reorderings.

---

## 5. Test generation & minimization

- **Structured generation.** Extend NNSmith-style valid-graph synthesis to emit
  graphs in the *intersection* of ops supported by ≥2 front-ends, parameterized
  over shapes, dtypes (incl. quant formats), and layouts — maximizing the chance
  of exercising a divergent `L_f`.
- **Real workloads.** Extract subgraphs from production LLMs (attention blocks,
  MLP, RMSNorm, RoPE) exported to both `.onnx` and `.gguf`/`.pte`, so the *same
  logical model* enters every front-end.
- **Metamorphic seeds.** Generate paired inputs satisfying MRs (layout swap,
  partition boundary move, residency swap) to surface bugs without a reference.
- **Minimization.** On a failure, run `ddmin` over (i) graph nodes and (ii) a
  **dtype-demotion lattice** (quant → fp16 → fp32) to yield a minimal divergent
  subgraph + dtype, with the layer attribution attached.

---

## 6. Validation of the verifier (meta-evaluation)

A verifier that finds nothing is useless; one that cries wolf is worse. We
validate GAIA-Diff with **mutation testing**:
- Inject known faults into each layer — a transposed weight in `L_ORT`, a wrong
  dtype map in `L_ggml`, a residency bug in `D`, an off-by-one in a kernel — and
  measure **detection rate** and **localization accuracy** (does it blame the
  injected layer?).
- Report **false-positive rate** on a corpus of *correct* runs (benign fp
  reorderings, quant) to validate `τ`.

Metrics: detections, localization top-1 accuracy, FPR, time-to-first-failure,
op/dtype coverage, and per-run overhead of the Tap (record vs replay mode).

---

## 7. Implementation & integration with this repo

- **Boundary Tap** = a thin wrapper over the existing `npu_model.h` / `npu_api.h`
  C ABI; frameworks load `libnpu_model` through it unchanged. The canonical
  artifacts map directly to existing types: `nnc`/`weights` = `npu_blob_t`,
  weight manifest = `npu_weight_manifest_t`, I/O = `npu_model_get_input/output`,
  residency = `npu_weight_residency_t`.
- **Reference Executor** generalizes the current CPU reference (`MatMulF32`) into
  a full `nnc` interpreter with fixed reduction order.
- **MR-DELIVER** (weight-delivery invariance) is exactly the `COPY_TO_DEVICE` vs
  `REFERENCE_HOST` contract already designed into `npu_blob_t`; the conformance
  suite reuses the `C-WMX` weight-delivery matrix and `U-MODEL` cases from
  [`test_plan.md`](test_plan.md).
- **Three real front-ends** already exist as adapters (`ort_ep/`,
  `ggml_backend/`, `executorch/`), giving heterogeneous ABIs for the differential
  engine for free.

---

## 8. Complexity, overhead, limitations

- **Cost.** Differential run = `|F|+1` executions per `(m,x)`. Localization adds at
  most a constant number of hybrid runs (one per stage) — `O(1)` extra, not `O(#ops)`,
  except the optional record-mode op-replay which is `O(#ops)` and used only after
  a kernel-stage verdict.
- **Tap overhead.** Replay mode ≈ a pointer pass-through (near-zero); record mode
  serializes intermediates (amortized, used selectively).
- **Limitations / threats.** (i) The Reference Executor is itself trusted —
  mitigated by mutation testing and by cross-checking against framework CPU
  backends. (ii) Stage-substitution soundness assumes interference-free stages;
  stateful kernels (KV-cache) need explicit state checkpoints in the Trace.
  (iii) Coverage is bounded by the op intersection across front-ends; ops unique
  to one front-end are tested only against the reference, not cross-framework.

---

## 9. Novelty restated (for the related-work table)
- **Tap at a shared waist across heterogeneous serving-framework ABIs** —
  CRADLE/LEMON/EAGLE/NNSmith/Tzer have no such shared boundary.
- **Stage-substitution localization** attributing divergence to LOWER / IOBIND /
  DELIVER / DTYPE / KERNEL — prior DL testers report *that* a bug exists, not
  *which layer across the front-end/boundary/kernel stack*.
- **A conformance suite for the boundary contract** (versioning, residency,
  source-kind, manifest) — a reusable artifact other accelerator vendors run.
