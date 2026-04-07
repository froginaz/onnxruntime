# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ONNX Runtime is a cross-platform inference and training engine for machine learning models in the ONNX format. Version 1.26.0. C++20 codebase with bindings for Python, C#, Java, JavaScript/WebAssembly, Objective-C, and Rust.

## Build Commands

The CMakeLists.txt is at `cmake/CMakeLists.txt` (not the repo root). The primary build orchestrator is `tools/ci_build/build.py`.

```bash
# Basic build (Linux/Mac)
./build.sh --build_dir build/Debug --config Debug --parallel

# Build with Python bindings
./build.sh --build_dir build/Debug --config Debug --build_wheel --parallel

# Build with specific execution provider
./build.sh --build_dir build/Debug --config Debug --use_cuda --parallel

# Direct CMake (from a build directory)
cmake ../cmake -DCMAKE_BUILD_TYPE=Debug
cmake --build . --config Debug --parallel

# Build via Python script with more control
python3 tools/ci_build/build.py --build_dir build/Debug --config Debug --build_shared_lib --parallel
```

Key build flags: `--use_cuda`, `--use_tensorrt`, `--use_openvino`, `--use_coreml`, `--use_dml`, `--use_qnn`, `--build_wheel`, `--enable_training`.

## Testing

```bash
# Build and run all tests
./build.sh --build_dir build/Debug --config Debug --parallel

# Build without running tests
./build.sh --build_dir build/Debug --config Debug --parallel --skip_tests

# Run C++ tests via ctest (from build directory)
cd build/Debug
ctest --build-config Debug
ctest -R <test_name_pattern>    # Run specific tests

# Python tests (unittest-based, in onnxruntime/test/python/)
python -m pytest onnxruntime/test/python/<test_file>.py
```

CMake test option: `onnxruntime_BUILD_UNIT_TESTS` (ON by default).

## Linting and Formatting

Uses [lintrunner](https://github.com/suo/lintrunner) with three linters: RUFF (Python style), RUFF-FORMAT (Python formatting), CLANGFORMAT (C/C++).

```bash
# Setup
pip install -r requirements-lintrunner.txt
lintrunner init

# Format local changes
lintrunner -a

# Format all files
lintrunner -a --all-files
```

Python: Ruff configured in `pyproject.toml` (target Python 3.10, line length 120). Use `pyright` for static type checking.

C++: `.clang-format` in repo root (Google style, max line 120). `.clang-tidy` for static analysis.

## Architecture

### Core Engine (`onnxruntime/core/`)

- **`session/`** - `InferenceSession` implementation; manages model loading, graph optimization, and kernel execution
- **`graph/`** - ONNX model graph representation, parsing, and manipulation
- **`framework/`** - Core abstractions: `Tensor`, `OrtValue`, execution frames, memory allocation
- **`optimizer/`** - Graph optimization passes (constant folding, operator fusion, layout transformation, QDQ transforms)
- **`providers/`** - Execution Provider (EP) implementations (~27): CPU, CUDA, TensorRT, CoreML, OpenVINO, DML, WebGPU, etc.
- **`mlas/`** - Math Library for Accelerated Scenarios: hand-optimized math kernels for CPU
- **`platform/`** - OS abstraction layer (Windows, Linux, macOS, Android, iOS)
- **`flatbuffers/`** - ORT format model serialization

### Key Abstractions

- **Execution Providers (EPs)**: Pluggable backends that execute operators on different hardware. Each EP in `core/providers/<name>/` implements kernel selection and operator dispatch. CPU EP is the universal fallback.
- **Graph Optimizers**: Transform the ONNX graph before execution. Located in `core/optimizer/` with sub-modules for specific optimization categories.
- **Contrib Ops** (`onnxruntime/contrib_ops/`): Non-standard operators with per-EP implementations (cpu/, cuda/, js/, webgpu/).

### Public API Surface

- **C API** (`include/onnxruntime/core/session/onnxruntime_c_api.h`): Stable ABI, primary API
- **C++ API** (`include/onnxruntime/core/session/onnxruntime_cxx_api.h`): Header-only C++ wrapper
- **EP Plugin API** (`include/onnxruntime/ep/api.h`): Interface for building execution providers
- **Python API**: `onnxruntime.InferenceSession` is the main entry point; bindings in `onnxruntime/python/onnxruntime_pybind_state.cc`

### Training (`orttraining/`)

Separate module for training functionality with its own graph transformations, Python API, and tests.

### Language Bindings

- **Python**: `onnxruntime/python/` (pybind11-based)
- **C#**: `csharp/`
- **Java**: `java/`
- **JavaScript/WASM**: `js/`
- **Objective-C**: `objectivec/`
- **Rust**: `rust/`

## C++ Coding Conventions

- Google C++ style with modifications: max line 120, exceptions allowed for fatal errors, non-const references allowed
- Use `gsl::span<const T>` instead of `const std::vector<T>&` for function parameters
- Use `std::string_view` instead of `const std::string&`
- Required container typedefs to minimize allocations:
  - `TensorShapeVector` (from `core/framework/tensor_shape.h`)
  - `InlinedVector<T>` (from `core/common/inlined_containers_fwd.h`) instead of `std::vector`
  - `InlinedHashSet<T>`, `InlinedHashMap<T>` (from `core/common/inlined_containers.h`) instead of `std::unordered_set/map`
  - `NodeHashSet`, `NodeHashMap` when pointer stability is needed
- Do not use `absl` namespace directly; use the ORT typedefs above
- Use `SafeInt<size_t>` (from `core/common/safeint.h`) for memory allocation size calculations
- Use `ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE` on new classes by default
- Use `std::optional` over `std::unique_ptr` for delayed/optional construction
- Don't use `else` after `return`; avoid `long` type; prefer `std::make_unique()`
- Use `reserve()` not `resize()` on vectors
