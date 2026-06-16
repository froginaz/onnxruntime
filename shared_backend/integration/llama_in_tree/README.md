# In-tree ggml backend that references the prebuilt shared_backend NPU core

This wires an **in-tree** ggml backend (`llama.cpp/ggml/src/ggml-myaccel/`) that
links the **prebuilt** MyAccel NPU core (`myaccel_npu.lib`/`.dll`, a.k.a.
`myaccel_core`) from the sibling `shared_backend` checkout.

Assumed layout (siblings):

```
./
├─ llama.cpp/
│  └─ ggml/src/ggml-myaccel/        ← you create this (copy of the template)
└─ shared_backend/
   └─ build/.../myaccel_npu.{lib,dll}   ← prebuilt NPU core
```

The backend **reuses** `shared_backend/ggml_backend/ggml-myaccel.cpp` (no source
copy) and links the prebuilt core, so there is a single source of truth.

---

## Step 1 — build the NPU core as a shared library

```bat
cmake -S shared_backend -B shared_backend\build ^
      -DMYACCEL_CORE_SHARED=ON -DMYACCEL_BUILD_GGML=OFF -DMYACCEL_BUILD_ORT_EP=OFF
cmake --build shared_backend\build --config Release
:: -> shared_backend\build\Release\myaccel_core.dll (+ .lib)
```

> If you want the artifact named `myaccel_npu` instead of `myaccel_core`, set
> `set_target_properties(myaccel_core PROPERTIES OUTPUT_NAME myaccel_npu)` in
> `shared_backend/CMakeLists.txt`, or just point `-DMYACCEL_NPU_LIB`/`-DMYACCEL_NPU_DLL`
> at the actual files. The template's `find_library` already tries both names.

## Step 2 — copy this template into the llama.cpp tree

```bat
xcopy /E /I shared_backend\integration\llama_in_tree\ggml-myaccel ^
            llama.cpp\ggml\src\ggml-myaccel
```

(The folder name **must** be `ggml-myaccel`, lowercase.)

## Step 3 — register the backend in ggml's CMake

In `llama.cpp/ggml/src/CMakeLists.txt`, next to the other `ggml_add_backend(...)`
calls, add:

```cmake
option(GGML_MYACCEL "ggml: use MyAccel NPU backend" OFF)
ggml_add_backend(MyAccel)
```

## Step 4 — register the backend at runtime

In `llama.cpp/ggml/src/ggml-backend-reg.cpp`, mirror an existing backend (e.g.
how `GGML_USE_VULKAN` is handled):

```cpp
#ifdef GGML_USE_MYACCEL
extern "C" ggml_backend_reg_t ggml_backend_myaccel_reg(void);
#endif
// ... inside the registry constructor, next to the other register_backend() calls:
#ifdef GGML_USE_MYACCEL
    register_backend(ggml_backend_myaccel_reg());
#endif
```

`ggml_add_backend_library` defines `GGML_USE_MYACCEL` for the build when the
backend is enabled. (Exact details vary by ggml version — copy what `ggml-vulkan`
/ `ggml-cuda` do in your checkout.)

## Step 5 — configure & build llama.cpp

```bat
cmake -S llama.cpp -B llama.cpp\build -DGGML_MYACCEL=ON
:: optional explicit overrides (otherwise auto-located under shared_backend\build):
::   -DSHARED_BACKEND_DIR=C:\path\to\shared_backend
::   -DMYACCEL_NPU_LIB=C:\...\shared_backend\build\Release\myaccel_core.lib
::   -DMYACCEL_NPU_DLL=C:\...\shared_backend\build\Release\myaccel_core.dll
cmake --build llama.cpp\build --config Release
```

After the build, the POST_BUILD step copies `myaccel_npu.dll`/`myaccel_core.dll`
next to the produced binaries (`llama.cpp\build\bin\Release\`), so `llama-cli.exe`
finds it at runtime.

## Step 6 — run

```bat
cd llama.cpp\build\bin\Release
llama-cli.exe --list-devices            :: should show myaccel_npu
llama-cli.exe -m model.gguf --device myaccel_npu -ngl 99 -p "Hello"
```

---

## Notes

- **`.lib` vs `.dll`.** The `.lib` (import library) is consumed at **link time**
  by this CMakeLists (`target_link_libraries`). The `.dll` is needed at
  **runtime** next to `llama-cli.exe`; the POST_BUILD step handles the copy. You
  do not manually drop the `.lib` anywhere.
- **ABI match.** Building in-tree guarantees the adapter compiles against the
  *same* ggml headers as the host, avoiding the version-skew pitfalls of the
  dynamic-plugin path.
- **Linux.** Replace `myaccel_npu.lib`/`.dll` with `libmyaccel_core.so`; the
  `find_library` call already matches it. The POST_BUILD copy is Windows-only —
  on Linux set an `RPATH` (e.g. `-DCMAKE_BUILD_RPATH=.../shared_backend/build`)
  or install the `.so` on the loader path instead.
- **Alternative (no in-tree build).** If you'd rather not modify the llama.cpp
  tree, use the dynamic-plugin path instead (`docs/llama_integration.md` §A):
  build `ggml-myaccel.dll` standalone and load it via `GGML_BACKEND_PATH`.
