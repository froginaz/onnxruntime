# A Common Backend API for Heterogeneous ML Accelerators: Background, Necessity, and Significance

*A position paper on the GAIA API — a single, stable public interface through
which multiple machine-learning frameworks target one hardware-accelerator
software stack.*

In this repository the GAIA API is realized concretely as the `myaccel::npu`
façade (`core/include/myaccel/npu_api.h`) plus the model-loading C ABI
(`npu_model.h`); the per-framework adapters for onnxruntime (Execution Provider),
llama.cpp (ggml-backend), and ExecuTorch (BackendInterface) are thin translators
above it. This document argues, in academic terms, *why* such an interface is
necessary and important.

---

## Abstract

The end of Dennard scaling and the deceleration of Moore's law have pushed
machine-learning systems toward **domain-specific architectures (DSAs)** —
NPUs, TPUs, and other accelerators [1, 2]. Simultaneously, the application layer
has fragmented into many ML frameworks (PyTorch/ExecuTorch, TensorFlow,
ONNX Runtime, llama.cpp, TVM, JAX/XLA, …). A hardware vendor who naïvely
integrates *each* framework with *each* accelerator faces an **N×M
integration matrix** whose engineering and maintenance cost grows
multiplicatively. We argue that the principled remedy is a **narrow-waist
("hourglass") common backend API** that decouples frameworks from the
accelerator software stack, transforming an O(N×M) problem into O(N+M). We
ground this claim in classical software-engineering theory (information hiding,
stable interfaces), in the systems literature on layering and the hourglass
model, and in the recent practice of ML compilers and backend-plugin systems.
We then state the design principles that make such an API — here, the GAIA API —
sustainable: separation of concerns, ABI stability with explicit versioning,
capability negotiation, and an abstract data/weight-delivery contract.

---

## 1. Background

### 1.1 A new golden age of computer architecture, and its cost

Hennessy and Patterson, in their 2018 Turing Lecture, observe that
general-purpose performance gains have largely stalled and that the path forward
is **domain-specific architectures** co-designed with domain-specific languages
and compilers [1]. The TPU [2] is the canonical datacenter example; at the edge,
a proliferation of mobile/embedded NPUs follows the same logic. Each such
accelerator ships its own software stack: a driver, a runtime, a compiler, and a
device-native program/container format.

The benefit — order-of-magnitude efficiency for tensor workloads — comes with a
structural cost: **the accelerator is only as useful as the set of software
front-ends that can target it.** Performance that cannot be reached by the
frameworks practitioners actually use is, economically, stranded.

### 1.2 Framework proliferation

The ML application layer is deliberately plural. TensorFlow [3] and PyTorch [4]
dominate training; for inference and deployment the landscape widens further:
ONNX Runtime as a cross-framework inference engine, llama.cpp/ggml for
on-device LLMs, ExecuTorch for edge PyTorch, Apache TVM, JAX/XLA, TensorRT,
OpenVINO, TFLite, and others. Each exposes a *different* extension mechanism for
third-party hardware:

| Framework | Backend extension point |
|---|---|
| ONNX Runtime | Execution Provider (plugin EP) |
| llama.cpp | ggml-backend registry |
| ExecuTorch | `BackendInterface` delegate |
| Apache TVM | BYOC (Bring-Your-Own-Codegen) [5] |
| TFLite | `TfLiteDelegate` |
| XLA/OpenXLA | PJRT plugin / custom call |

These extension points differ in ABI, in granularity (whole-graph vs. operator
vs. fused-subgraph), in memory ownership, and in lifecycle. There is **no single
interface** that satisfies them all.

### 1.3 The interoperability gap

Interchange formats (ONNX) and compiler IRs (MLIR [6], StableHLO, TVM Relay,
Glow [7]) standardize the *representation* of a model. They do **not**, by
themselves, standardize the *runtime boundary* between a host framework and a
vendor's accelerator stack — i.e., how a compiled subgraph and its weights are
handed to the device, executed, and synchronized. That runtime boundary is
exactly where the N×M cost concentrates, and exactly what a common backend API
addresses.

---

## 2. Problem formulation: the N×M integration matrix

Let `N` be the number of ML frameworks a vendor wishes to support and `M` the
number of accelerator stacks (or stack versions) the vendor ships. Direct,
pairwise integration requires up to **N × M** distinct adapters, each of which
must be independently developed, tested, and *maintained across the
independent release cadences of both sides*. Because frameworks and drivers
evolve continuously, the dominant term is not the one-time build cost but the
**recurring maintenance cost**, which also scales with N × M.

Introducing a common backend API `A` inserts a stable indirection:

```
direct:     framework_i  ⟷  stack_j           (N × M edges)
mediated:   framework_i  ⟶  A  ⟵  stack_j      (N + M edges)
```

Each framework is integrated to `A` **once** (N adapters); each accelerator
stack implements `A` **once** (M implementations). The integration surface
collapses from multiplicative to additive, O(N×M) → O(N+M). This is the same
economic argument that justifies device-driver models in operating systems and
the byte-stream socket in networking; it is restated here for ML accelerators.

---

## 3. Necessity

### 3.1 The hourglass / narrow-waist principle

Large, long-lived systems that must connect many things above to many things
below tend to evolve a **thin, stable "waist"** — a single spanning layer that
everything agrees on — with diversity allowed above and below it. Akhshabi and
Dovrolis show this hourglass shape emerges evolutionarily in layered protocol
stacks [8]; Beck formalizes *why* a deliberately minimal spanning layer
maximizes the number of supportable applications and implementations [9]. The
Internet's IP waist is the archetype.

A common ML-accelerator backend API is the waist for this domain: above it,
arbitrarily many frameworks; below it, arbitrarily many accelerator stacks;
the waist itself kept **deliberately small and slow-changing**. The necessity is
not incidental convenience — it is the only known structure that keeps an
many-to-many ecosystem economically and evolutionarily viable.

### 3.2 Information hiding and module decomposition

Parnas's criterion for decomposing systems into modules is to **hide the
decisions most likely to change behind stable interfaces** [10]. The accelerator
stack — driver versions, kernel implementations, the device-native program
container, memory topology — is precisely the volatile part. A common API that
hides it means a new silicon revision, a new compiler, or a re-tuned kernel
library is absorbed *below the waist* without touching any framework adapter.
Conversely, a framework's ABI churn is absorbed in its single adapter without
touching the accelerator stack. The two volatile surfaces are decoupled.

### 3.3 ABI stability and versioned evolution

For a *binary* boundary — a plugin loaded by a pre-built host — source-level
agreement is insufficient; the contract must be **ABI-stable**. This motivates
the engineering invariants the literature and practice converge on: a pure-C
boundary (no name mangling), opaque handles, and explicit version negotiation
(`struct_size` + `api_version`) so a host built against an older header can
safely load a newer plugin and vice versa. Without these, the N+M benefit is
lost the moment either side updates independently.

### 3.4 Performance portability and the abstraction-vs-efficiency tension

A recurring objection is that abstraction costs performance. The ML-compiler
literature answers this by **separating the portable boundary from the
optimizing compiler**: TVM [11], XLA, IREE, and Glow [7] show that a stable
front-end boundary can coexist with aggressive, hardware-specific code
generation *below* it. The common API therefore should standardize **what
crosses the boundary** (a compiled program + its weights + an execution/IO
contract) while leaving **how the device executes** entirely to the vendor —
preserving the DSA efficiency that motivated the accelerator in the first place.
Empirically, fair cross-stack comparison is itself only meaningful given a
common harness, which is the motivation behind MLPerf [12].

---

## 4. Significance

1. **Ecosystem reach (vendor view).** A single conformant implementation of the
   common API instantly exposes the accelerator to *every* framework already
   integrated with it. Time-to-market for "supported by framework X" drops from
   a bespoke project to a configuration.
2. **Sustainability (maintenance view).** Recurring cost scales as N+M, not
   N×M. This is the difference between a backend program that a small team can
   maintain and one that is perpetually behind the frameworks' releases.
3. **Decoupled innovation.** Silicon, compiler, and kernels can evolve below the
   waist on their own cadence; frameworks can evolve above it on theirs. Neither
   forces a synchronized rewrite of the other.
4. **Correctness and trust (parity).** A single backend implementation shared by
   all front-ends makes **cross-framework numerical parity** a testable property
   (same weights ⇒ same result through any framework), rather than an accident
   of two independent ports. This is a first-class verification target.
5. **Portability as a strategic asset.** For the *framework/user* side, the same
   model artifact can be retargeted across accelerators that implement the API,
   reducing vendor lock-in and de-risking hardware choices — a public-good
   property analogous to that of POSIX or the C ABI.

---

## 5. Design principles of the GAIA API

From the above, the common API must embody:

- **Separation of concerns / narrow waist.** It standardizes the runtime
  boundary only; it knows nothing of `.onnx`, `.gguf`, or any framework type.
  (Here: adapters convert their container to the device-native `{model.nnc,
  weight.bin}` and call the same API.)
- **Information hiding.** Device, memory, and kernel internals sit *below* the
  public façade and never appear in its surface. (Here: `npu_core`/`npu_memory`
  are internal; only `myaccel::npu` + the model C ABI are public.)
- **ABI stability.** Pure-C boundary, opaque handles, `struct_size`/
  `api_version` negotiation for forward/backward compatibility.
- **Capability negotiation, graceful fallback.** Frameworks query what the
  accelerator supports and place the remainder on the host — never crashing when
  an op/dtype is unsupported. (Here: ORT `GetCapability`, ggml `supports_op`.)
- **An abstract data/weight-delivery contract.** One descriptor spans in-memory,
  file, mmap/zero-copy, and streamed weights, with explicit residency/ownership
  semantics, so each framework uses its natural storage without the device
  caring. (Here: `npu_blob_t` + residency in `npu_model.h`.)
- **Verifiable parity.** The shared implementation makes equality across
  front-ends a property that can be asserted in CI.

These are not stylistic preferences; each maps directly to one of the
necessity/significance arguments above.

---

## 6. Related work and positioning

- **ML compilers / IRs** — TVM [11], MLIR [6], XLA/OpenXLA, Glow [7], IREE
  standardize representation and code generation. The common backend API is
  *complementary*: it standardizes the runtime hand-off, not the IR.
- **Backend-plugin mechanisms** — TVM BYOC [5], ONNX Runtime EPs, ExecuTorch
  delegates, `TfLiteDelegate`, PJRT. Each is a *per-framework* waist; the common
  API is a *cross-framework* waist that those plugins translate down to.
- **Heterogeneous-compute abstractions** — SYCL, oneAPI Level Zero, OpenCL
  provide vendor-neutral *device* programming models. The common ML backend API
  operates one level up, at the granularity of *compiled models and subgraphs*,
  and can be implemented on top of such device layers.
- **Interchange formats** — ONNX standardizes the model; it is the *input* a
  framework converts before crossing the API, not a substitute for the runtime
  boundary.
- **Benchmarking** — MLPerf [12] presumes a common harness across stacks; a
  common backend API is what makes such cross-stack measurement and parity
  practical.

The contribution of the GAIA approach is therefore not a new IR or a new device
language, but the explicit articulation — and a working reference — of the
**single stable runtime waist** that lets one accelerator stack serve many ML
frameworks at O(N+M) cost.

---

## 7. Limitations and future work

A common waist trades some peak expressiveness for breadth: features unique to
one framework's extension point (e.g., framework-specific graph-capture or
profiling hooks) must be either generalized into the API or handled in the
adapter. Determining the *minimal sufficient* waist — small enough to stay
stable, rich enough to avoid per-framework escape hatches — is an empirical,
ongoing design question, mirroring debates over the Internet waist's ossification
[9]. Future work includes formalizing the capability-negotiation contract,
quantifying the maintenance-cost reduction on real release histories, and
standardizing the parity-verification methodology.

---

## 8. Conclusion

Domain-specific accelerators are now necessary for efficient ML, but their value
is gated by software reach. Framework plurality makes naïve, pairwise
integration scale as O(N×M) in both build and — decisively — maintenance cost. A
**common, stable, narrow public backend API** is the principled structure that
collapses this to O(N+M), grounded in the hourglass model of layered systems
[8, 9] and in classical information hiding [10], while the ML-compiler tradition
[5–7, 11] shows that such a boundary need not sacrifice hardware-specific
performance. The GAIA API is this waist: it lets a single in-house accelerator
software stack serve onnxruntime, llama.cpp, ExecuTorch, and future frameworks
through thin adapters, turning stranded silicon performance into broadly
reachable capability, with cross-framework parity as a verifiable guarantee.

---

## References

[1] J. L. Hennessy and D. A. Patterson, "A New Golden Age for Computer
Architecture," *Communications of the ACM*, vol. 62, no. 2, pp. 48–60, 2019.

[2] N. P. Jouppi et al., "In-Datacenter Performance Analysis of a Tensor
Processing Unit," *ISCA*, 2017.

[3] M. Abadi et al., "TensorFlow: A System for Large-Scale Machine Learning,"
*OSDI*, 2016.

[4] A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep
Learning Library," *NeurIPS*, 2019.

[5] Z. Chen et al., "Bring Your Own Codegen to Deep Learning Compiler," arXiv:
2105.03215, 2021.

[6] C. Lattner et al., "MLIR: Scaling Compiler Infrastructure for Domain
Specific Computation," *CGO*, 2021.

[7] N. Rotem et al., "Glow: Graph Lowering Compiler Techniques for Neural
Networks," arXiv:1805.00907, 2018.

[8] S. Akhshabi and C. Dovrolis, "The Evolution of Layered Protocol Stacks Leads
to an Hourglass-Shaped Architecture," *ACM SIGCOMM*, 2011.

[9] M. Beck, "On the Hourglass Model," *Communications of the ACM*, vol. 62,
no. 7, pp. 48–57, 2019.

[10] D. L. Parnas, "On the Criteria to Be Used in Decomposing Systems into
Modules," *Communications of the ACM*, vol. 15, no. 12, pp. 1053–1058, 1972.

[11] T. Chen et al., "TVM: An Automated End-to-End Optimizing Compiler for Deep
Learning," *OSDI*, 2018.

[12] V. J. Reddi et al., "MLPerf Inference Benchmark," *ISCA*, 2020.
