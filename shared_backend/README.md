# MyAccel — shared NPU backend for onnxruntime, llama.cpp & ExecuTorch

A single hardware-accelerator (in-house NPU) core that plugs into **multiple ML
runtimes** through each framework's native loadable-backend ABI. There is no
single common ABI across these projects, so we use a 3-layer design: one shared
core + one thin adapter per runtime (onnxruntime, llama.cpp, ExecuTorch).

> Architecture rationale, class/sequence diagrams, and how to port a new ML
> runtime: [`docs/design.md`](docs/design.md). Why a common backend API at all
> (background, necessity, significance — academic):
> [`docs/rationale_common_backend_api.md`](docs/rationale_common_backend_api.md).

```
                  ┌────────────────────────────────────┐
                  │  myaccel_core (static / shared)     │  framework-agnostic
                  │  device / memory / kernels          │  → your NPU SDK
                  └──┬────────────┬─────────────┬───────┘
   plugin-EP ABI     │  ggml ABI  │             │  ExecuTorch BackendInterface
      ┌──────────────▼─┐  ┌───────▼──────┐  ┌───▼──────────────────┐
      │ myaccel_ort_ep │  │ ggml-myaccel │  │ myaccel_backend      │
      │ .dll           │  │ .dll         │  │ .dll                 │
      │ CreateEp-      │  │ ggml_backend │  │ register_backend /   │
      │ Factories      │  │ _init        │  │ myaccel_backend_     │
      │                │  │              │  │ register             │
      └───────▲────────┘  └──────▲───────┘  └─────────▲────────────┘
       onnxruntime loads   llama.cpp loads    ExecuTorch loads
```

## Layout

| Path | Role |
|------|------|
| `core/` | NPU SDK abstraction. The only shared code. **No framework types.** |
| `core/include/myaccel/npu_api.h` | **Public façade** (`myaccel::npu`) — the adapters' sole NPU access point. |
| `core/include/myaccel/npu_model.h` | **Model-loading C ABI** the runtimes call — see [`docs/npu_model_api.md`](docs/npu_model_api.md). |
| `core/internal/myaccel/` | Internal `npu_core.h` / `npu_memory.h` (PRIVATE; wrapped by the façade). |
| `ort_ep/` | onnxruntime plugin Execution Provider adapter (-> `myaccel_ort_ep.dll`). |
| `ggml_backend/` | llama.cpp ggml backend adapter (-> `ggml-myaccel.dll`) — using it from llama.cpp: [`docs/llama_integration.md`](docs/llama_integration.md). |
| `executorch/` | ExecuTorch backend adapter (-> `myaccel_backend.dll`), references `myaccel_core`. Guide: [`docs/executorch_integration.md`](docs/executorch_integration.md). Opt-in: `-DMYACCEL_BUILD_EXECUTORCH=ON -DEXECUTORCH_DIR=...`. |

Public NPU access — the adapters include only these:
- `npu_api.h` — **public façade** (namespace `myaccel::npu`) for device/memory/kernels. The single NPU access point for every adapter.
- `npu_model.h` — **pure C ABI** for loading `model.nnc` + `weight.bin` onto the NPU. The stable runtime↔NPU boundary; it knows nothing about onnx/gguf.

Internal (under `core/internal/`, a PRIVATE include not on the adapters' path; wrapped by `npu_api.h` and hidden from the DLL exports):
- `npu_core.h` — device, compute, streams, kernels (raw SDK calls).
- `npu_memory.h` — device allocation and host↔device copies.

## Build on Windows (CMake)

> On Windows you don't hand-write a Makefile — CMake **generates** the build
> (`.sln`/`.vcxproj`, Ninja, or NMake Makefiles) from `CMakeLists.txt`.

From an **x64 Native Tools Command Prompt for VS 2022**:

```bat
:: Visual Studio generator (produces a .sln)
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 ^
      -DORT_HOME=C:\path\to\onnxruntime ^
      -DLLAMA_CPP_DIR=C:\path\to\llama.cpp
cmake --build build --config Release

:: ...or Ninja (faster, build.ninja)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DORT_HOME=... -DLLAMA_CPP_DIR=...
cmake --build build

:: ...or a real NMake Makefile
cmake -S . -B build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DORT_HOME=... -DLLAMA_CPP_DIR=...
cmake --build build
```

Outputs: `build\ort_ep\Release\myaccel_ort_ep.dll`,
`build\ggml_backend\Release\ggml-myaccel.dll`, and (when enabled)
`build\executorch\Release\myaccel_backend.dll`.

The ExecuTorch adapter is opt-in (it needs an ExecuTorch checkout):

```bat
cmake -S . -B build -DMYACCEL_BUILD_EXECUTORCH=ON -DEXECUTORCH_DIR=C:\path\to\executorch ...
```

### Configure options

| Option | Default | Meaning |
|--------|---------|---------|
| `ORT_HOME` | — | onnxruntime install or source root (needs `include/`). Required for the EP. |
| `LLAMA_CPP_DIR` | — | llama.cpp checkout (needs `ggml/include` + `ggml/src`). Required for ggml. |
| `EXECUTORCH_DIR` | — | ExecuTorch root/install (headers + runtime libs). Required for the ExecuTorch adapter. |
| `MYACCEL_BUILD_ORT_EP` | `ON` | Build the onnxruntime adapter. |
| `MYACCEL_BUILD_GGML` | `ON` | Build the llama.cpp adapter. |
| `MYACCEL_BUILD_EXECUTORCH` | `OFF` | Build the ExecuTorch adapter (`myaccel_backend`). |
| `MYACCEL_CORE_SHARED` | `OFF` | Build the core as its own DLL instead of static. |

Build just one adapter by turning the others `OFF` (e.g. `-DMYACCEL_BUILD_GGML=OFF`).

## Using the built DLLs

**onnxruntime** (external plugin EP — no ORT rebuild needed):
```cpp
Ort::Env env;
env.RegisterExecutionProviderLibrary("myaccel", L"myaccel_ort_ep.dll");
// then select the "myaccel" EP via the device/EP selection API
```

**llama.cpp**: place `ggml-myaccel.dll` next to the executable (or on the ggml
backend search path); `ggml_backend_load_all()` discovers it at startup. Details:
[`docs/llama_integration.md`](docs/llama_integration.md).

**ExecuTorch**: link/load `myaccel_backend.dll` so its `register_backend("MyAccelBackend")`
runs (static init on load, or call the exported `myaccel_backend_register()`).
At export time, partition the NPU subgraphs to the `MyAccelBackend` so they are
delegated in the `.pte`. Full AOT + runtime walkthrough:
[`docs/executorch_integration.md`](docs/executorch_integration.md).

> If the core is built as a separate DLL (`-DMYACCEL_CORE_SHARED=ON`), ship
> `myaccel_core.dll` next to each adapter DLL — they all reference it.

## What is implemented vs. TODO

This is a **compilable skeleton**, not a finished backend:

- ✅ Build system, the two export surfaces, device enumeration, memory/copy
  plumbing, and a reference fp32 MatMul so you can validate end-to-end on CPU.
- 🔧 `core/src/npu_core.cpp` — every body is a host-memory placeholder; replace
  with your NPU SDK calls.
- 🔧 `ort_ep/ep.cpp` — `GetCapability` currently fuses **no** nodes (a safe,
  crash-free "device discovered" state). Grow `GetCapability`/`Compile` to
  offload subgraphs. Reference: `onnxruntime/test/autoep/library/example_plugin_ep`.
- 🔧 `ggml_backend/ggml-myaccel.cpp` — only `MUL_MAT` is dispatched; extend
  `backend_graph_compute` + `device_supports_op` together. Confirm the dynamic
  entry symbol (`ggml_backend_init`) against your ggml version.
- 🔧 `executorch/executorch_backend.cpp` — `BackendInterface` skeleton: `init`
  loads the delegate blob via `npu_model_load`; `execute` still needs `EValue`↔
  NPU I/O binding. Confirm the ExecuTorch namespace/headers
  (`executorch::runtime`) against your version.
