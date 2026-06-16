# Library file extensions — Windows vs Linux (vs macOS)

DLL (Windows) and `.so` (Linux) are the **same concept** — a dynamic / shared
library — but **different binary formats** (PE vs ELF) and **not interchangeable**.
A `.dll` cannot be loaded on Linux and a `.so` cannot be loaded on Windows.

## Extension cheat-sheet

| Kind | Windows | Linux | macOS |
|---|---|---|---|
| Dynamic / shared library | `.dll` | `.so` | `.dylib` |
| Import library (for linking) | `.lib` (MinGW: `.dll.a`) | — (link the `.so` directly) | — (link the `.dylib`) |
| Static library | `.lib` | `.a` | `.a` |
| Executable | `.exe` | (none) | (none) |
| Object file | `.obj` | `.o` | `.o` |

## Key differences

1. **Windows splits a DLL into two files.**
   - `.dll` — the actual code, loaded at runtime.
   - `.lib` — the **import library**: just a table of which symbols live in which
     DLL, used by the linker at build time. It lands in the *archive* output
     directory, **not** next to the `.dll` (the runtime output) — which is why an
     import `.lib` often seems "missing" when you only look beside the `.dll`.
   - ⚠️ On Windows a `.lib` is ambiguous: it can be a **static** library **or** an
     **import** library. On Linux these are distinct (`.a` vs `.so`).
   - A Linux `.so` serves **both** roles (link-time and run-time) in one file.

2. **Naming convention.**
   - Linux requires a `lib` prefix: `libmyaccel_core.so` (link with `-lmyaccel_core`).
   - Windows has no prefix: `myaccel_core.dll` + `myaccel_core.lib`.

3. **Symbol export defaults.**
   - Windows: symbols are **hidden by default**; export with
     `__declspec(dllexport)` or a `.def` file. If a DLL exports nothing, no
     import `.lib` is generated.
   - Linux: symbols are **visible by default**; hide them with
     `-fvisibility=hidden` + `__attribute__((visibility("default")))` on the
     public API (this is what `myaccel_core` does for its shared build).

4. **Runtime search path.**
   - Windows: same folder as the `.exe`, then `PATH`.
   - Linux: `RPATH`/`RUNPATH` in the binary, then `LD_LIBRARY_PATH`, then the
     `ld.so` cache (`/etc/ld.so.conf`).
   - macOS: `@rpath` / `@loader_path` plus each dylib's `install_name`.

## This project's artifacts

| Output | Windows | Linux |
|---|---|---|
| NPU core | `myaccel_core.dll` (+ `myaccel_core.lib`) | `libmyaccel_core.so` |
| onnxruntime adapter | `myaccel_ort_ep.dll` | `libmyaccel_ort_ep.so` |
| llama.cpp adapter | `ggml-myaccel.dll` | `libggml-myaccel.so` |
| ExecuTorch adapter | `myaccel_backend.dll` | `libmyaccel_backend.so` |

> CMake abstracts most of this away: `add_library(... SHARED)` produces the right
> format, prefix, and import library per platform automatically.
