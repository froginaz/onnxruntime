# In-house NPU SDK — software-ecosystem strategy

Goal restated: build a vertically-integrated NPU software stack that (a) runs
models from **PyTorch, TensorFlow, ONNX, and llama.cpp**, (b) maximizes
**developer efficiency** (don't reinvent compilers/op libraries), and (c) offers a
**CUDA-compatible front-end programming model** that targets **both SIMD and SIMT**
execution. This document explains what those requirements *mean* concretely and
the workstreams/decisions/roadmap to achieve them.

> This builds on the common-backend work in this repo (the `npu_api`/`npu_model`
> boundary and the ONNX/ggml/ExecuTorch adapters) — that is the *framework
> integration layer* (L4 below), already prototyped.

---

## 1. Decoding the requirements

### 1.1 "Support PyTorch / TensorFlow / ONNX / llama.cpp"
These four hit the device through **different extension points**, and at two very
different *depths*:

| Framework | Integration point | Depth |
|---|---|---|
| ONNX Runtime | Execution Provider (EP) | inference |
| llama.cpp | ggml-backend | inference (LLM) |
| PyTorch | `torch.compile`/Inductor custom backend, **ExecuTorch delegate**, or `PrivateUse1` device | inference (eager training = much bigger) |
| TensorFlow | **PJRT plugin** (via XLA/OpenXLA) or PluggableDevice | inference/training |

**Strategic implication:** do **inference-first**. Full eager *training* for
PyTorch/TF means autograd + thousands of ops + dynamic shapes — a multi-year lift.
Enter via **graph/AOT paths** (ONNX, ExecuTorch, `torch.compile`, XLA/PJRT) with a
curated op set; defer eager training.

### 1.2 "CUDA-compatible front-end supporting SIMD and SIMT" — what it really means
Two independent things bundled in one sentence:

- **CUDA-compatible front-end = source-level portability of kernel code.** Kernels
  written in CUDA C++ syntax (`__global__`, `<<<grid,block>>>`, `threadIdx`,
  `__shared__`, warp shuffles) **compile and run on your NPU** — so existing CUDA
  kernels, libraries, and developer skills transfer. This is exactly what **AMD
  HIP** does for CUDA, and conceptually what SYCL/oneAPI offers as an open
  alternative. **Not** binary/PTX compatibility (that is a legal + technical
  minefield — avoid; do *source* compatibility).
- **SIMD *and* SIMT = the kernel must run on two hardware execution models.**
  - *SIMT* (GPU-like): many independent threads, hardware groups them into
    warps; divergence handled by the HW.
  - *SIMD* (DSP/vector-like): one instruction stream over vector lanes; "threads"
    must be **mapped to lanes** by the compiler, with **predication** for
    divergence (the "SPMD-on-SIMD" technique, cf. Intel **ispc**, OpenAI
    **Triton**, GPU implementations of SIMT-on-SIMD).

  So the *programming model* is one SPMD/CUDA-like source; the **compiler chooses
  the lowering** per target: thread-block→SIMT-core, or thread→SIMD-lane +
  predication. The CUDA front-end is the *language*; SIMD/SIMT is the *backend
  mapping*.

### 1.3 "Developer efficiency"
The lever is **reuse, not rewrite**: adopt **LLVM/MLIR/Clang** as the compiler
spine, a **single kernel-authoring path**, **compiler-generated op coverage**
(don't hand-write 2000 ops), and the **common backend API** to avoid N×M framework
integration. Plus automated **cross-framework parity** verification (GAIA-Diff in
this repo) so correctness scales.

---

## 2. Target stack (layers) — mirror CUDA's stack, reuse where possible

```
 L5  Tools         profiler · debugger · parity verifier (GAIA-Diff) · pkg/distro · docs
 L4  Frameworks    ONNX EP · ggml · ExecuTorch · torch.compile/Inductor · TF PJRT   ← this repo
 L3  Graph compiler / inference runtime   nnc container · partitioner · fusion · quantization (≈ TensorRT)
 L2  Kernel/op libs   GEMM · attention · norm · conv · quant  (hand-tuned + autogen)  (≈ cuBLAS/cuDNN)
 L1  Programming model + compiler   CUDA-compatible source → MLIR/LLVM → NPU ISA ; SIMT/SIMD lowering  (≈ nvcc/PTX)
 L0  Runtime + driver   device mgmt · memory · queues/streams · sync   (≈ CUDA driver/UMD/KMD)
```

Reference ecosystems to study/borrow from: CUDA (cuBLAS/cuDNN/TensorRT/Triton),
**AMD ROCm/HIP** (CUDA source portability), **Intel oneAPI/SYCL/Level Zero**,
**OpenXLA/PJRT**, **Apache TVM**, **OpenAI Triton** (productive kernel DSL),
**MLIR** (compiler infrastructure).

---

## 3. Build-vs-adopt decisions (the efficiency core)

| Component | Decision | Rationale |
|---|---|---|
| Compiler spine | **Adopt MLIR + LLVM** | building a compiler from scratch is the classic fatal mistake; MLIR gives dialects (`gpu`, `linalg`, `vector`, `affine`) and reuse |
| CUDA front-end | **Adopt Clang CUDA/HIP front-end**, retarget codegen to your ISA | source-portable CUDA without writing a C++ parser; provide a `hipify`-like shim for headers/intrinsics |
| SIMT↔SIMD mapping | **Build** two MLIR lowering pipelines (gpu→SIMT-core; gpu→`vector` SPMD-on-SIMD w/ predication), selected by a **target description** | this is your real IP; ispc/Triton are references |
| Op coverage | **Compiler-generate** from `linalg`/graph + **hand-tune a small hot core** (GEMM/attention/quant) | avoid hand-writing thousands of ops |
| Kernel DSL (optional) | **Offer a Triton-style Python DSL** in addition to CUDA C++ | huge productivity for kernel authors; compiles through the same MLIR spine |
| Framework integration | **Reuse the common backend API** (this repo) + per-framework adapters | O(N+M) not O(N×M) |
| Quantization | **Adopt** existing schemes (GGUF block formats, ONNX QDQ) + your kernels | interop with llama.cpp/ONNX |
| Runtime/driver | **Build** (HW-specific) but keep the public boundary = `npu_model`/`npu_api` | stable ABI already designed |
| Correctness | **Reuse GAIA-Diff** parity + conformance | scales testing across frameworks |

---

## 4. Workstreams (what to actually do)

- **WS1 — Compiler spine (L1).** Stand up MLIR/LLVM; define a **target
  description** (ISA, register/lane width, memory hierarchy, sync primitives);
  build an LLVM backend (or MLIR→ISA) for your NPU. Deliver: compile a trivial
  kernel to device ISA.
- **WS2 — CUDA-compatible front-end + SIMT/SIMD lowering (L1).** Wire Clang's CUDA
  path → MLIR `gpu` dialect; implement two lowerings: (i) SIMT-core mapping,
  (ii) SPMD-on-SIMD (thread→lane + predicated divergence) via the `vector`
  dialect. Provide CUDA header/intrinsic shims (`threadIdx`, `__shared__`,
  `__syncthreads`, warp ops). Deliver: a CUDA `__global__` GEMM runs both ways.
- **WS3 — Runtime + driver + the public boundary (L0).** Device/memory/queues/
  streams/events; expose them under the existing `npu_api` façade; the
  `npu_model` C ABI for model loading. Deliver: load + run an nnc on real HW.
- **WS4 — Kernel/op library (L2).** Hand-tune GEMM (fp16/bf16/int8 + block-quant),
  fused attention (flash-style), RMSNorm/LayerNorm, RoPE, softmax, conv. Authored
  in the CUDA dialect or Triton-style DSL. Deliver: a transformer block at target
  perf.
- **WS5 — Graph/inference compiler (L3).** The `nnc` container + partitioner +
  operator fusion + layout/precision selection + quantization import. Deliver:
  ONNX/torch graph → fused nnc → kernels.
- **WS6 — Framework adapters (L4).** ONNX EP, ggml, ExecuTorch (have prototypes);
  add **`torch.compile`/Inductor custom backend** and **TF PJRT plugin**. All
  funnel through `npu_model`/`npu_api`. Deliver: same model runs from all four.
- **WS7 — Tooling (L5).** Profiler (timeline, occupancy), debugger
  (source-level over the CUDA dialect), **GAIA-Diff** parity + conformance, CI,
  packaging (a CUDA-toolkit-like distributable), versioned ABI.
- **WS8 — Docs/samples/enablement.** Porting guide (CUDA→your dialect), op-library
  reference, model zoo, perf cookbook.

---

## 5. Phased roadmap

- **Phase 0 — Foundations (0–3 mo).** MLIR/LLVM bring-up + target description
  (WS1); runtime skeleton + `npu_api`/`npu_model` on mock NPU (WS3); **ONNX EP**
  end-to-end with a reference GEMM (WS6); parity harness vs CPU reference (WS7).
  *Milestone:* one framework runs one model on the mock/real device.
- **Phase 1 — CUDA front-end + LLM (3–9 mo).** CUDA-dialect MVP with **SIMT**
  lowering (WS2); hot kernels GEMM/attention/quant (WS4); **llama.cpp + ExecuTorch**
  adapters (WS6); quantization import (WS5). *Milestone:* a quantized LLM runs via
  llama.cpp on real HW; CUDA kernels compile.
- **Phase 2 — SIMD path + PyTorch/TF + perf (9–18 mo).** **SPMD-on-SIMD** lowering
  (WS2); **`torch.compile` backend + TF PJRT** (WS6); compiler-generated op
  coverage via `linalg` (WS5); profiler/debugger (WS7). *Milestone:* all four
  frameworks; competitive perf on a transformer block.
- **Phase 3 — Ecosystem + correctness (18 mo+).** Triton-style DSL, training
  subset, package/distribution, **verified parity oracle** (GAIA-Diff + formal),
  conformance program for partners. *Milestone:* a self-serve SDK others build on.

---

## 6. Risks & anti-patterns (cold)

- **Writing a compiler from scratch** → use MLIR/LLVM. Single biggest failure mode.
- **Chasing 100% CUDA / PTX binary compatibility** → impossible & legally fraught;
  target a **source-portable subset** (HIP-style). Document unsupported features.
- **SIMT-on-SIMD divergence cost** → predication is expensive on divergent code;
  expose lane-width/occupancy to the kernel author and the autotuner.
- **Eager training as a near-term goal** → it is a trap; inference-first via
  graph/AOT paths.
- **N×M framework integration** → funnel through the common backend API (this
  repo); each framework = one thin adapter.
- **Correctness sprawl across frameworks** → GAIA-Diff parity + conformance from
  day one, not as an afterthought.
- **Op-coverage by hand** → compiler-generate the long tail; hand-tune only the
  hot core.

---

## 7. How this repo fits
The existing `npu_api`/`npu_model` boundary and the ONNX/ggml/ExecuTorch adapters
**are L4 + the public face of L0/L3**. The SDK strategy extends *downward*
(L0–L2 compiler/runtime/kernels) and *sideways* (PyTorch/TF adapters), while the
**parity/verification** assets (`gaia_diff_design.md`, `formal_methods.md`,
`test_plan.md`) become the SDK's correctness program. The narrow-waist boundary is
what keeps the whole ecosystem at O(N+M).
