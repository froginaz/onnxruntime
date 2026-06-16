# In-tree ggml backend that links the prebuilt shared_backend NPU core

This builds an **in-tree** ggml backend at `llama.cpp/ggml/src/ggml-myaccel/`.
The adapter source (`ggml-myaccel.cpp`) lives **inside the llama.cpp tree** and is
compiled there; the only things taken from the sibling `shared_backend` are the
**public NPU headers** and the **prebuilt core** `npu_core.lib`/`.dll`.

Assumed layout (siblings under the repo root):

```
./
├─ llama.cpp/
│  └─ ggml/src/ggml-myaccel/          ← copy of this folder (CMakeLists.txt + ggml-myaccel.cpp)
└─ shared_backend/
   └─ build/.../npu_core.{lib,dll}    ← prebuilt NPU core (headers in core/include)
```

What this folder contains:
- `ggml-myaccel.cpp` — the adapter source, compiled **locally** in the llama tree.
- `CMakeLists.txt` — builds it and references `shared_backend` only for headers +
  the `npu_core` import lib, then copies `npu_core.dll` next to the binaries.

---

## Step 1 — build the NPU core as a shared library

```bat
cmake -S shared_backend -B shared_backend\build ^
      -DMYACCEL_CORE_SHARED=ON -DMYACCEL_CORE_OUTPUT_NAME=npu_core ^
      -DMYACCEL_BUILD_GGML=OFF -DMYACCEL_BUILD_ORT_EP=OFF
cmake --build shared_backend\build --config Release
:: -> shared_backend\build\Release\npu_core.dll (+ npu_core.lib)
```

(`MYACCEL_CORE_OUTPUT_NAME` is optional — the template's `find_library` also
matches `myaccel_core`/`myaccel_npu`. Or point `-DNPU_CORE_LIB`/`-DNPU_CORE_DLL`
at the exact files.)

## Step 2 — copy this folder into the llama.cpp tree

```bat
xcopy /E /I shared_backend\integration\llama_in_tree\ggml-myaccel ^
            llama.cpp\ggml\src\ggml-myaccel
```

Now `llama.cpp\ggml\src\ggml-myaccel\` contains `ggml-myaccel.cpp` +
`CMakeLists.txt`. (Folder name **must** be `ggml-myaccel`, lowercase.)

## Step 3 — register the backend in ggml's CMake

In `llama.cpp/ggml/src/CMakeLists.txt`, next to the other `ggml_add_backend(...)`
calls:

```cmake
option(GGML_MYACCEL "ggml: use MyAccel NPU backend" OFF)
ggml_add_backend(MyAccel)
```

## Step 4 — register the backend at runtime

In `llama.cpp/ggml/src/ggml-backend-reg.cpp`, mirror an existing backend (e.g.
`GGML_USE_VULKAN`):

```cpp
#ifdef GGML_USE_MYACCEL
extern "C" ggml_backend_reg_t ggml_backend_myaccel_reg(void);
#endif
// ... inside the registry constructor, next to the other register_backend() calls:
#ifdef GGML_USE_MYACCEL
    register_backend(ggml_backend_myaccel_reg());
#endif
```

## Step 5 — configure & build llama.cpp

```bat
cmake -S llama.cpp -B llama.cpp\build -DGGML_MYACCEL=ON
:: optional explicit overrides (otherwise auto-located under shared_backend\build):
::   -DSHARED_BACKEND_DIR=C:\path\to\shared_backend
::   -DNPU_CORE_LIB=C:\...\shared_backend\build\Release\npu_core.lib
::   -DNPU_CORE_DLL=C:\...\shared_backend\build\Release\npu_core.dll
cmake --build llama.cpp\build --config Release
```

The POST_BUILD step copies `npu_core.dll` next to the produced binaries
(`llama.cpp\build\bin\Release\`), so `llama-cli.exe` finds it at runtime.

## Step 6 — run

```bat
cd llama.cpp\build\bin\Release
llama-cli.exe --list-devices            :: should show myaccel_npu
llama-cli.exe -m model.gguf --device myaccel_npu -ngl 99 -p "Hello"
```

---

## Notes

- **`.lib` vs `.dll`.** The `npu_core.lib` (import library) is consumed at
  **link time** by this CMakeLists (`target_link_libraries`). The `npu_core.dll`
  is needed at **runtime** next to `llama-cli.exe`; the POST_BUILD step copies it.
  You do not manually drop the `.lib` anywhere.
- **ABI match.** Building in-tree compiles `ggml-myaccel.cpp` against the *same*
  ggml headers as the host, avoiding the version-skew pitfalls of the
  dynamic-plugin path.
- **Header path.** The adapter includes `myaccel/npu_api.h`; the CMakeLists adds
  `shared_backend/core/include` to the include path. No shared_backend `.cpp` is
  compiled here — only the prebuilt core is linked.
- **Linux.** Replace `npu_core.lib`/`.dll` with `libnpu_core.so`; `find_library`
  already matches it. The POST_BUILD copy is Windows-only — on Linux set an
  `RPATH` (e.g. `-DCMAKE_BUILD_RPATH=.../shared_backend/build`) or install the
  `.so` on the loader path.
- **Keeping the source in sync.** This `ggml-myaccel.cpp` is the single canonical
  adapter in the repo (the standalone `shared_backend/ggml_backend/` build also
  compiles this same file). Edit it here; re-copy into the llama tree after
  changes.
- **Alternative (no in-tree build).** Prefer not to modify the llama.cpp tree?
  Use the dynamic-plugin path (`docs/llama_integration.md` §A): build
  `ggml-myaccel.dll` standalone and load it via `GGML_BACKEND_PATH`.

## Troubleshooting: `NPU_CORE_LIB not found`

The CMakeLists resolves the core import lib in this order: explicit
`-DNPU_CORE_LIB=...`, else a **recursive** search under
`shared_backend/build` and `shared_backend/out`. Common causes when it still
fails:

1. **shared_backend not built as a shared lib.** You must build it with
   `-DMYACCEL_CORE_SHARED=ON` (Step 1). Without it there is no import `.lib`.
2. **Built into a different directory.** VS "Open Folder"/presets emit to
   `out/build/<preset>/...`, not `build/`. Find the real path:
   `dir /s /b shared_backend\*.lib` and pass `-DNPU_CORE_LIB=<that path>`.
3. **Wrong/empty `SHARED_BACKEND_DIR`.** If the headers were found (no header
   FATAL_ERROR) the path is correct; otherwise pass `-DSHARED_BACKEND_DIR=...`.
4. **Stale CMake cache (most common after a fix).** If you configured *before*
   building the lib, a `find_library` result may be cached as `-NOTFOUND` in the
   llama build dir. Delete `llama.cpp\build\CMakeCache.txt` (or the whole
   `llama.cpp\build` dir) and reconfigure. (This template avoids the cache trap
   by re-globbing every configure, but a cache from an older template version
   can still bite.)
5. **Name mismatch.** The lib must be `npu_core.lib` / `myaccel_core.lib`
   (or `libnpu_core.so`). If you used a different `OUTPUT_NAME`, pass the exact
   path via `-DNPU_CORE_LIB`.

Quick unblock: always works regardless of layout —
```bat
cmake -S llama.cpp -B llama.cpp\build -DGGML_MYACCEL=ON ^
      -DNPU_CORE_LIB=C:\...\shared_backend\build\Release\npu_core.lib ^
      -DNPU_CORE_DLL=C:\...\shared_backend\build\Release\npu_core.dll
```
