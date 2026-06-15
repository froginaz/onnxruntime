# Test plan — MyAccel shared NPU backend

Scope: the 3-layer architecture in `shared_backend/` — the framework-agnostic
**core** (`npu_api.h` façade + internal `npu_core`/`npu_memory`, `npu_model.h`), the **onnxruntime plugin-EP** adapter,
and the **llama.cpp ggml** adapter — plus the cross-cutting guarantees that
make "one NPU, two runtimes" actually hold.

```
            ┌── numerical parity ──┐         ┌── ABI / version compat ──┐
 onnxruntime ─EP─▶ myaccel_core ◀─ggml─ llama.cpp     weight delivery matrix
                       │
                  NPU SW stack (real HW) / mock NPU (CI)
```

## 1. Objectives & risk-driven focus

| # | Architectural risk | Why it matters here | Primary test level |
|---|---|---|---|
| R1 | **Front-end divergence** — same weights, different result via EP vs ggml | The whole value prop is one backend for both | Cross-runtime parity (§6) |
| R2 | **Weight delivery variants** — memory/file/fd/callback × copy/reference | ORT in-memory vs gguf mmap differ fundamentally | Component (§4.2) |
| R3 | **Zero-copy lifetime** — REFERENCE_HOST blob freed too early | Use-after-free / silent corruption | Robustness (§8) |
| R4 | **ABI / version skew** — struct_size, api_version, ggml struct order, dynamic symbol name | Plugin loaded by a prebuilt host | ABI/compat (§7) |
| R5 | **dtype mapping** — ONNX TensorProto ↔ ggml_type ↔ npu_dtype, incl. quant | Wrong map = garbage or crash | Unit + parity (§3, §6) |
| R6 | **Resource lifecycle** — Initialize/Shutdown refcount, model load/free, device handles | Leaks/double-free across many sessions | Unit + stress (§8) |
| R7 | **Graceful fallback** — unsupported op/model rejected, host runs it | Must never crash the host runtime | Component (§4) |

## 2. Test levels & tooling

| Level | Target | Tool | Where it runs |
|---|---|---|---|
| Unit | `npu_api` façade + `npu_model` C/C++ functions | GoogleTest | host, mock NPU |
| Component | each adapter against a fake host | GoogleTest + ORT C API harness; ggml `test-backend-ops` | host, mock NPU |
| Integration | adapter + real ORT / real llama.cpp | ORT `onnxruntime_test_all`, `llama-cli`, pytest | host |
| System / E2E | full inference of a real model | pytest (ORT) / `llama-cli` prompt | host + HW |
| Parity | EP result == ggml result == CPU reference | custom harness (§6) | host + HW |
| Performance | latency/throughput/mem | benchmark harness (§9) | HW |

**Oracle:** the current core stub computes a CPU fp32 reference (`MatMulF32`,
naive). Keep a "reference mode" build so every NPU result can be diffed against
a deterministic CPU answer even before silicon exists.

**Mock NPU:** a build of `myaccel_core` whose SDK calls hit an in-process fake
(host memory + reference kernels). Lets all of §3–§7 run in CI with no hardware.

## 3. Unit tests — core

### 3.1 Device / memory (`npu_api.h` façade)
- U-CORE-01 `Initialize`/`Shutdown` are refcounted: N inits need N shutdowns; real teardown only on last.
- U-CORE-02 `GetDeviceCount` ≥ 0; `GetDeviceInfo` rejects out-of-range id (`kInvalidArgument`).
- U-CORE-03 `OpenDevice`/`CloseDevice` round-trip; open invalid id → nullptr.
- U-CORE-04 `Alloc`/`Free`: zero-size alloc valid; `DevicePtr` non-null; double-free guarded (ASAN).
- U-CORE-05 `Copy` H2D/D2H/D2D correctness; null args → `kInvalidArgument`.
- U-CORE-06 `MatMulF32` correctness vs hand-computed small matrices; null args rejected; non-square / M≠N shapes.

### 3.2 Model API (`npu_model.h`)
- U-MODEL-01 `npu_model_load` rejects `struct_size`/`api_version` mismatch → `NPU_ERR_VERSION_MISMATCH` (R4).
- U-MODEL-02 null `info`/`out_model` → `NPU_ERR_INVALID_ARGUMENT`.
- U-MODEL-03 each `npu_source_kind_t` (MEMORY, CALLBACK; FILE/FD when impl) reads identical bytes (R2).
- U-MODEL-04 CALLBACK partial reads (read() returns < requested) are looped correctly; read()<0 → `NPU_ERR_IO`.
- U-MODEL-05 residency contract: `COPY_TO_DEVICE` ⇒ `npu_model_owns_weights()==1`; `REFERENCE_HOST`+MEMORY ⇒ `==0` (R3).
- U-MODEL-06 `npu_model_compile` then `bind_weights` == one-shot `npu_model_load` (same state).
- U-MODEL-07 `bind_weights` twice (rebind) supported for weight-sharing path.
- U-MODEL-08 introspection bounds: `get_input/get_output` out-of-range → error; counts match.
- U-MODEL-09 `npu_status_str` returns non-null for every enum incl. unknown; `npu_last_error` populated on failure.
- U-MODEL-10 manifest validation: dtype/shape/offset+size beyond blob → `NPU_ERR_WEIGHT_MISMATCH` (R5).
- U-MODEL-11 leak check: 10k load/free cycles flat under ASAN/LeakSanitizer (R6).
- U-MODEL-12 header parses as **pure C** (`gcc -std=c11 -fsyntax-only`) — guards the C ABI (R4).

## 4. Component tests — adapters (mock NPU)

### 4.1 onnxruntime EP adapter
- C-ORT-01 DLL exports exactly `CreateEpFactories` + `ReleaseEpFactory` (dumpbin/nm on the built lib).
- C-ORT-02 `CreateEpFactories` with `max_factories==0` → `ORT_INVALID_ARGUMENT`; with ≥1 → one factory, `ort_version_supported==ORT_API_VERSION`.
- C-ORT-03 Factory metadata: name/vendor/vendor_id/version match expected.
- C-ORT-04 `GetSupportedDevices` selects only `OrtHardwareDeviceType_NPU` with matching vendor id; ignores CPU/GPU (R7).
- C-ORT-05 `RegisterExecutionProviderLibrary` → session creation succeeds; device appears in `GetEpDevices`.
- C-ORT-06 `GetCapability` returns empty ⇒ all nodes on CPU EP, inference still correct (current safe state, R7).
- C-ORT-07 (after Compile impl) fused subgraph calls `npu_model_load`; result matches CPU EP within tol (R1).
- C-ORT-08 Release path: `ReleaseEp`/`ReleaseEpFactory` no leak; `Shutdown` refcount balanced (R6).
- C-ORT-09 dtype mapping table ONNX→npu_dtype unit-tested for all supported elem types (R5).

### 4.2 Weight-delivery matrix (both adapters drive this)
Cross-product, each must load + infer correctly:

| source ↓ / residency → | COPY_TO_DEVICE | REFERENCE_HOST |
|---|---|---|
| MEMORY | C-WMX-01 | C-WMX-02 |
| FILE | C-WMX-03 | C-WMX-04 |
| FD (mmap) | C-WMX-05 | C-WMX-06 |
| CALLBACK | C-WMX-07 | C-WMX-08 (n/a → expect rejection) |

Plus C-WMX-09: with manifest vs without (nnc self-describing offsets) → identical result.

### 4.3 llama.cpp ggml adapter
- C-GGML-01 `ggml_backend_myaccel_reg()` returns reg with `api_version==GGML_BACKEND_API_VERSION` (R4).
- C-GGML-02 dynamic load: `ggml_backend_load_all()` discovers `ggml-myaccel.dll`; confirm the expected init symbol name for the pinned ggml version (R4).
- C-GGML-03 device props (name/type/memory) populated; `supports_op` ⟺ ops handled in `graph_compute` are consistent (R7).
- C-GGML-04 buffer type alloc/free; `set_tensor`/`get_tensor` round-trip equals host data (R2).
- C-GGML-05 `graph_compute` MUL_MAT matches CPU backend; unsupported op → `GGML_STATUS_FAILED` and ggml falls back (R7).
- C-GGML-06 `test-backend-ops` (ggml's own op test suite) run against the MyAccel backend for the supported op set.

## 5. Integration tests (real runtimes, mock or real NPU)

- I-ORT-01 Real ORT session on a small `.onnx` (MatMul/MLP), EP selected, output == CPU EP (rtol 1e-3 fp32).
- I-ORT-02 Mixed graph: some nodes NPU, some CPU (partitioning) → correct end-to-end.
- I-ORT-03 EPContext / compiled-model cache path (if used) loads and infers.
- I-GGML-01 `llama-cli` runs a tiny GGUF model with `-dev MyAccel`; produces same tokens as CPU backend (greedy, temp 0).
- I-GGML-02 `llama-bench` selects the backend; offload of supported layers works, rest fall back.
- I-X-01 Same logical model converted to both `.onnx` and `.gguf` → both load via the same `npu_model_load` (exercises the shared boundary).

## 6. Cross-runtime parity (the headline guarantee, R1)

Goal: prove the backend is truly shared, not two implementations.

- P-01 Pack identical weights into one `weight.bin`; load via the ORT path and the llama path; assert bit-identical (or within 1 ULP) **device outputs for the same input** on the supported op set.
- P-02 Parity across residency modes: COPY vs REFERENCE_HOST produce identical results (R3).
- P-03 Parity across source kinds: MEMORY vs FILE vs FD vs CALLBACK all yield identical bytes loaded (hash the resolved weight set).
- P-04 Quantized parity: a Q4_K/Q8_0 weight dequantized on-NPU matches the gguf CPU dequant reference within format tolerance (R5).
- P-05 Three-way diff harness: `EP_result`, `ggml_result`, `cpu_reference` — fail if any pair diverges beyond tol; report which leg differs.

## 7. ABI & version-compatibility tests (R4)

- A-01 Host built against API v1 loads a plugin built against v1 — OK; v(N+1) host + vN plugin and vice-versa handled via `struct_size`/`api_version` (graceful reject or forward-compat, never UB).
- A-02 `onnxruntime_ep_c_api.h` lacks an include guard — regression test that `ep_common.h` includes it exactly once (compile guard).
- A-03 ggml struct field-order pin: a static_assert / golden test that `ggml_backend_i`, `ggml_backend_device_i`, `ggml_backend_reg_i` field layout matches the vendored ggml header version.
- A-04 Symbol surface: export tables locked — only the documented symbols are visible (`.def`/`.lds` enforced; checked via dumpbin `/EXPORTS`, `nm -D`).
- A-05 Build-from-anywhere: configure `shared_backend/` from a path outside the ORT tree with only `ORT_HOME`/`LLAMA_CPP_DIR` set — both DLLs build (relocatability regression).

## 8. Robustness, concurrency, resource (R3, R6)

- S-01 Lifetime: free a REFERENCE_HOST weight blob while the model is live → detected (debug build poisons / asserts), documented as caller error.
- S-02 Concurrency: N ORT sessions in N threads sharing one factory; no data race (TSAN).
- S-03 Concurrency: ggml multi-device / multi-stream parallel `graph_compute` (TSAN).
- S-04 Soak: 24h load/infer/free loop, memory flat (no growth), no handle leak.
- S-05 Fault injection: NPU alloc fails / DMA fails / device lost mid-run → API returns proper `NPU_ERR_*`, host runtime stays alive and can fall back.
- S-06 Large weights: 8–70 GB weight.bin via mmap/FD — loads without 2× host copy; verify RSS stays near mmap size (R2).
- S-07 Corrupt inputs: truncated nnc, mismatched manifest, wrong dtype → clean error, no crash.

## 9. Performance (HW required)

- PF-01 Model-load latency by source kind; assert FD/mmap zero-copy ≪ MEMORY+COPY for large weights.
- PF-02 First-token / single-inference latency and steady-state throughput vs CPU and (if available) GPU EP.
- PF-03 Host memory overhead: REFERENCE_HOST shows no device→host duplication.
- PF-04 No regression gate: parity + latency tracked per commit in CI (HW lane).

## 10. CI matrix & gates

| Lane | Platform | NPU | Suites | Blocking? |
|---|---|---|---|---|
| L1 fast | Windows x64 (MSVC), Linux (clang) | mock | Unit §3, Component §4, ABI §7 | yes (PR gate) |
| L2 sanitizers | Linux clang | mock | §3,§4,§8 under ASAN/UBSAN/TSAN/LSAN | yes |
| L3 integration | Windows + Linux | mock | §5,§6 parity | yes |
| L4 hardware | Windows x64 | real NPU | §5 (real), §8 soak subset, §9 perf | nightly, non-PR-blocking |

Build matrix per lane: `{ORT EP only, ggml only, both}` × `{core static, core shared}` × `{reference-kernel, real-SDK}`.

## 11. Entry / exit criteria

**Entry:** core + adapters compile on all L1 platforms; mock NPU available; reference kernels produce known-good output.

**Exit (per release):**
- All L1–L3 suites green; zero sanitizer findings in L2.
- Parity §6 within tolerance for the full supported op/dtype set.
- Weight-delivery matrix §4.2 fully covered.
- ABI tests §7 green; export surface locked.
- L4 soak S-04 passes; no perf regression beyond agreed threshold.
- Known-failure list documents every unsupported op with its fallback path verified.

## 12. Test artifacts & structure (proposed)

```
shared_backend/tests/
├─ unit/        gtest: core, npu_model            (mock NPU, no HW)
├─ component/   gtest: ort_ep harness, ggml harness
├─ parity/      three-way EP/ggml/CPU diff harness + fixtures
├─ integration/ pytest (ORT) + llama-cli scripts
├─ fixtures/    tiny .onnx, tiny .gguf, packed weight.bin, golden outputs
└─ mock_npu/    fake SDK backing myaccel_core for CI
```
Gate everything reproducibly via `ctest` (C++), `pytest` (ORT), and a thin
`run_parity.py`.
