# WS2 design — CUDA-compatible front-end with SIMT *and* SIMD lowering

Detailed technical design for the kernel programming model of the NPU SDK
([`sdk_strategy.md`](sdk_strategy.md) WS2): one CUDA-style single source compiled
to **either** a SIMT core **or** SIMD vector lanes, chosen by a target
description. The hard part is not parsing CUDA — it is **mapping the SIMT thread
model onto SIMD hardware** (SPMD-on-SIMD) with correct, efficient divergence
handling.

> Diagrams: [`cuda_frontend_design.puml`](cuda_frontend_design.puml).

---

## 1. Scope and the two meanings

- **Source compatibility, not binary.** Support a **HIP-style portable subset** of
  CUDA C++ (`__global__`/`__device__`, `<<<grid,block,shmem,stream>>>`,
  `threadIdx/blockIdx/blockDim/gridDim`, `__shared__`, `__syncthreads`,
  `__shfl_*_sync`, `atomic*`, `wmma`/`mma` for tensor units). **No PTX/SASS binary
  compatibility** (legal + technical minefield). Unsupported features are reported
  at compile time.
- **One SPMD source → two execution models.** The same kernel lowers to:
  - **SIMT**: threads/warps map to hardware threads/lanes; HW handles divergence.
  - **SIMD**: threads map to **vector lanes**; the *compiler* handles divergence by
    **predication/masking** (whole-function vectorization, à la ispc [5], WFV
    [3], PoCL [6]).

---

## 2. Compilation pipeline

```
CUDA C++ (subset)
  │  Clang CUDA front-end (custom target triple)  +  cuda_compat.h shim
  ▼
Clang→MLIR bridge (Polygeist/cgeist [7] or ClangIR)        # C++ AST → MLIR
  ▼
SPMD convergence IR  =  MLIR `gpu` dialect (+ `memref`,`scf`,`arith`,`vector`)
  │   - gpu.func / gpu.launch_func, gpu.thread_id, gpu.block_id, gpu.barrier
  │   - gpu.shuffle (warp), memory spaces: global / workgroup(shared) / private
  ▼  ── analysis: uniformity(divergence) + address-affinity + barrier-convergence
  ├─────────────── SIMT pipeline ──────────────┐     ┌──── SIMD pipeline (SPMD-on-SIMD) ────┐
  │ map block→core, thread→HW lane;            │     │ thread-vectorize axis (width W);      │
  │ barrier→HW sync; shared→scratchpad;        │     │ varying→vector<WxT>, uniform→scalar;  │
  │ shuffle→HW warp op                         │     │ if-convert/linearize → masked ops;    │
  │                                            │     │ barrier→loop fission; shuffle→lane perm│
  └──────────────────┬─────────────────────────┘     └──────────────────┬───────────────────┘
                     ▼  lower to `llvm` dialect + NPU intrinsics          ▼
                          LLVM IR  →  NPU LLVM backend  →  device ISA  →  kernel binary
```

The **`gpu`-dialect SPMD IR is the convergence point**: both pipelines start
there, so the front-end and all analyses are shared; only the *backend mapping*
differs, selected by the target description (§6).

---

## 3. The SPMD convergence IR

We reuse MLIR's `gpu` dialect as the target-independent kernel IR:

- **Index ops:** `gpu.thread_id {x|y|z}`, `gpu.block_id`, `gpu.block_dim`,
  `gpu.grid_dim`.
- **Sync:** `gpu.barrier` (must be *convergent* per CUDA semantics — exploited in
  §5.4).
- **Warp ops:** `gpu.shuffle`, `gpu.all_reduce`, `gpu.subgroup_mma` (tensor units).
- **Memory spaces:** `memref` with address space = global / `workgroup` (shared) /
  `private`.
- **Control flow:** `scf.if/for/while` (structured) preferred; unstructured CFG
  handled by linearization (§5.3).

Authoring alternatives that emit the *same* IR (developer efficiency):
- **CUDA C++** (primary, for portability of existing kernels);
- a **Triton-style Python DSL** [8] (tile-level, autotuned) → same `gpu`/`linalg`
  IR — recommended for new kernels.

---

## 4. SIMT lowering (the easy direction)

When the target is SIMT, the mapping is direct:
- `gpu.launch` grid/block → hardware grid/cores/warps.
- `thread_id` → hardware lane/thread id intrinsic.
- `gpu.barrier` → hardware block barrier.
- `__shared__`/`workgroup` memref → on-chip scratchpad allocation.
- `gpu.shuffle`/`mma` → warp-shuffle / tensor-core intrinsics.
- Divergence is left to hardware (HW reconvergence stack / independent thread
  scheduling).

Lower to `llvm` dialect with NPU intrinsics → LLVM backend → ISA. This path is
close to standard GPU codegen and reuses existing MLIR `gpu`→`llvm` infrastructure.

---

## 5. SIMD lowering (SPMD-on-SIMD) — the core IP

Goal: execute a thread block on a single instruction stream over `W` vector lanes
(`W` = native SIMD width from the target description). This is **whole-function
vectorization of an SPMD kernel** [3, 5].

### 5.1 Thread vectorization
1. Pick the **vectorization axis** (default `threadIdx.x`). Tile the block:
   block of `B` threads → `ceil(B/W)` vector iterations of width `W`; remaining
   thread axes (`y`,`z`) become enclosing scalar loops.
2. **Scalarize-then-vectorize:** values that vary along the axis become
   `vector<WxT>`; values uniform along the axis stay scalar (§5.2). `thread_id.x`
   becomes the lane-index vector `base + iota(W)`.

### 5.2 Uniformity (divergence) analysis
Classify every SSA value and branch as **uniform** (identical across the
vectorized lanes) or **varying** [4, Sampaio et al.]:
- Seeds: `thread_id.x` is varying; `block_id`, kernel scalar args, constants are
  uniform; loads from thread-invariant addresses are uniform.
- Propagate: an op is varying iff any operand is varying. A branch is varying iff
  its condition is varying.
- **Use:** uniform branches keep scalar control flow (cheap); only **varying**
  control flow is predicated (§5.3). This "partial" treatment (vs linearizing
  everything) is the efficiency lever [2, Moll & Hack].

### 5.3 Divergence handling = if-conversion + partial linearization
- **Varying `if/else`:** if-convert into a **mask**; execute both sides under
  complementary masks; merge with `vector.select`. Memory writes are masked
  (masked store / scatter).
- **Unstructured varying CFG:** apply **partial control-flow linearization** [2]
  — linearize only the divergent sub-region, preserving uniform branches; insert
  masks; reconverge at the immediate post-dominator.
- **Varying loops:** execute until **all** lanes exit; per-lane "active" mask;
  lanes that finished are disabled (predicated) — a masked while-loop.
- **Nested divergence:** masks compose (AND of enclosing masks).

### 5.4 Barriers under vectorization (the subtle part)
CUDA requires `__syncthreads()` to be **convergent** (all threads of the block
reach it). We exploit this:
- If the whole block fits in **one** vector iteration (`B ≤ W`), a barrier across
  the vectorized axis is **lane-synchronous already → no-op**.
- If `B > W` (multiple vector iterations) or there are enclosing `y/z` loops, a
  barrier forces **loop fission**: split the kernel at the barrier into
  sub-kernels, each fully iterating all lanes/iterations, with shared-memory live
  values spilled to a scratch array between phases (classic SPMD-on-SIMD barrier
  handling [6, PoCL]).
- Barriers inside **varying** control flow are illegal in CUDA → diagnosed.

### 5.5 Memory: coalescing vs gather/scatter
- **Address-affinity analysis:** if a thread-varying global address is **affine**
  in `thread_id.x` with unit stride, emit a **contiguous `vector.load/store`**
  (coalesced). Otherwise emit **gather/scatter** (masked).
- `__shared__` → a scratch `memref`; bank-conflict-aware layout from the target
  description.

### 5.6 Warp/shuffle and tensor units
- `gpu.shuffle`/warp reductions → **lane permute/reduce** vector intrinsics (when
  `warp == W`) or emulated across iterations.
- `wmma`/`mma` → map to the NPU **tensor/MMA unit** op if present (target
  description capability), else lower to a vectorized GEMM microkernel.

---

## 6. Target description schema (selects SIMT vs SIMD + parameters)

A declarative spec (TableGen or JSON) the lowering reads. Example:

```json
{
  "target": "myaccel-v1",
  "execution_model": "SIMD",            // "SIMT" | "SIMD" | "HYBRID"
  "vector_width": 32,                    // W lanes for SIMD vectorization
  "warp_size": 32,                       // for SIMT / shuffle width
  "cores": 16,
  "memory_spaces": {
    "global":   {"size_mb": 8192, "align": 256},
    "shared":   {"size_kb": 192, "banks": 32},   // scratchpad
    "private":  {"size_kb": 16}
  },
  "sync": {"block_barrier": true, "lane_sync": "implicit"},
  "dtypes": ["fp32","fp16","bf16","int8","q4_k"],
  "special_units": {"mma": {"shapes": [[16,16,16]], "dtypes": ["fp16","int8"]}},
  "intrinsics": {"thread_id":"npu.lane_id","barrier":"npu.barrier","shuffle":"npu.perm"}
}
```

The pipeline picks the SIMT or SIMD path from `execution_model`, parameterizes
vectorization by `vector_width`, allocates `shared` from the scratchpad spec, and
maps CUDA intrinsics via `intrinsics`. A "HYBRID" target (SIMT cores each with
SIMD lanes) runs both mappings at two levels.

---

## 7. Front-end and intrinsic shim

- **Clang CUDA front-end** with a custom target triple parses the CUDA subset;
  a **`cuda_compat.h`** shim (HIP-`hipify` style) maps `threadIdx`,
  `__syncthreads`, `__shfl_sync`, `atomicAdd`, `__shared__`, `wmma::*` to builtins.
- **Clang→MLIR**: Polygeist/cgeist [7] (or ClangIR) lifts the C++ AST into MLIR so
  the `gpu`/`scf`/`memref` analyses above apply. (Alternative: Clang→LLVM-IR with
  NVVM-like intrinsics, then lift IR→`gpu` dialect; more brittle.)
- **Diagnostics:** unsupported CUDA features (e.g., dynamic parallelism,
  unsupported intrinsics) produce clear compile errors, not silent miscompiles.

---

## 8. Launch & runtime integration

- `kernel<<<grid, block, shmem, stream>>>(args)` lowers to a **host-side runtime
  call** into the NPU runtime (L0), enqueuing the compiled kernel binary with
  grid/block dims, dynamic shared-mem size, and a stream/queue handle — exposed
  through the existing `npu_api` device/stream surface.
- Kernel binaries are packaged in a fat-binary-like container keyed by target id;
  the runtime selects the binary matching the device's target description.

---

## 9. Worked example (divergent ReLU-bias)

```cuda
__global__ void relu_bias(const float* x, const float* b, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;   // varying
  if (i < n) {                                      // varying branch (tail)
    float v = x[i] + b[i];                          // coalesced loads
    y[i] = v > 0.f ? v : 0.f;                       // varying select
  }
}
```
- **SIMT:** `i`→lane id; `if (i<n)` handled by HW; `x[i]` coalesced; `?:`→select.
- **SIMD (W=32):** `i = base + iota(32)`; `mask = i < n` (tail predicate);
  `v = load_coalesced(x+base, mask) + load_coalesced(b+base, mask)`;
  `y = select(v>0, v, 0)`; `store_masked(y+base, y, mask)`. No barrier ⇒ single
  vector iteration per 32 threads, no fission.

---

## 10. Build-vs-adopt, testing, risks

- **Adopt:** Clang CUDA front-end, Polygeist/ClangIR, MLIR `gpu`/`vector`/`llvm`
  dialects, LLVM backend infra, divergence/WFV literature.
- **Build (your IP):** the SIMD lowering (vectorization + partial linearization +
  barrier fission + coalescing), the target description, and the NPU LLVM backend.
- **Testing:** every kernel is checked **SIMT vs SIMD vs CPU reference** with
  **GAIA-Diff** ([`gaia_diff_design.md`](gaia_diff_design.md)) — the two lowerings
  of one source must be numerically equivalent (a built-in parity oracle for the
  compiler). Add divergence-stress kernels (nested varying branches, barriers
  after divergence-reconvergence, gather/scatter).
- **Risks:** (i) divergence/predication overhead on branchy kernels — expose
  lane width + occupancy to an autotuner; (ii) barrier fission cost/complexity —
  restrict patterns or spill efficiently; (iii) Clang→MLIR maturity (Polygeist/
  ClangIR evolving) — keep the LLVM-IR fallback path; (iv) gather/scatter
  performance — rely on address-affinity analysis to maximize coalesced loads.

---

## References
[1] C. Lattner et al., "MLIR," CGO 2021.
[2] S. Moll, S. Hack, "Partial Control-Flow Linearization," PLDI 2018.
[3] R. Karrenberg, S. Hack, "Whole-Function Vectorization," CGO 2011.
[4] D. Sampaio et al., "Divergence Analysis," ACM TOPLAS 2013.
[5] M. Pharr, W. Mark, "ispc: A SPMD Compiler for High-Performance CPU Programming," InPar 2012.
[6] P. Jääskeläinen et al., "pocl: A Performance-Portable OpenCL Implementation," Int. J. Parallel Programming, 2015.
[7] W. Moses et al., "Polygeist: Raising C to Polyhedral MLIR," PACT 2021.
[8] P. Tillet et al., "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations," MAPL 2019.
