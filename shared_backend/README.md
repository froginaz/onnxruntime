# MyAccel — shared NPU backend for onnxruntime + llama.cpp

A single hardware-accelerator (in-house NPU) core that plugs into **both**
onnxruntime and llama.cpp through each framework's native loadable-backend ABI.
There is no single common ABI between the two projects, so we use a 3-layer
design: one shared core + two thin adapter DLLs.

```
                         ┌──────────────────────────────┐
                         │  myaccel_core (static/shared) │  framework-agnostic
                         │  device / memory / kernels    │  → your NPU SDK
                         └───────┬───────────────┬──────┘
            plugin-EP ABI        │               │      ggml-backend ABI
   ┌─────────────────────────────▼──┐   ┌────────▼──────────────────────────┐
   │ myaccel_ort_ep.dll              │   │ ggml-myaccel.dll                   │
   │ exports CreateEpFactories /     │   │ exports ggml_backend_myaccel_reg / │
   │         ReleaseEpFactory        │   │         ggml_backend_init          │
   └──────────────▲──────────────────┘   └──────────────▲────────────────────┘
       onnxruntime loads it                 llama.cpp loads it
```

## Layout

| Path | Role |
|------|------|
| `core/` | NPU SDK abstraction. The only shared code. **No framework types.** |
| `core/include/myaccel/npu_model.h` | **Model-loading C ABI** both runtimes call — see [`docs/npu_model_api.md`](docs/npu_model_api.md). |
| `ort_ep/` | onnxruntime plugin Execution Provider adapter. |
| `ggml_backend/` | llama.cpp ggml backend adapter. |

Two complementary core APIs:
- `npu_core.h` — C++ convenience layer for device/memory/kernels (used by the adapters internally).
- `npu_model.h` — **pure C ABI** for loading `model.nnc` + `weight.bin` onto the NPU. This is the stable runtime↔NPU boundary; it knows nothing about onnx/gguf.

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
`build\ggml_backend\Release\ggml-myaccel.dll`.

### Configure options

| Option | Default | Meaning |
|--------|---------|---------|
| `ORT_HOME` | — | onnxruntime install or source root (needs `include/`). Required for the EP. |
| `LLAMA_CPP_DIR` | — | llama.cpp checkout (needs `ggml/include` + `ggml/src`). Required for ggml. |
| `MYACCEL_BUILD_ORT_EP` | `ON` | Build the onnxruntime adapter. |
| `MYACCEL_BUILD_GGML` | `ON` | Build the llama.cpp adapter. |
| `MYACCEL_CORE_SHARED` | `OFF` | Build the core as its own DLL instead of static. |

Build just one side by turning the other `OFF` (e.g. `-DMYACCEL_BUILD_GGML=OFF`).

## Using the built DLLs

**onnxruntime** (external plugin EP — no ORT rebuild needed):
```cpp
Ort::Env env;
env.RegisterExecutionProviderLibrary("myaccel", L"myaccel_ort_ep.dll");
// then select the "myaccel" EP via the device/EP selection API
```

**llama.cpp**: place `ggml-myaccel.dll` next to the executable (or on the ggml
backend search path); `ggml_backend_load_all()` discovers it at startup.

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
