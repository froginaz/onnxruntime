# Function-level profiling (Perfetto / Chrome trace) — incl. Windows

Yes — **Perfetto works on Windows** for app tracing. The full system daemon
(`traced`) is Linux/Android-oriented, but the **Perfetto Tracing SDK** (in-process
`TRACE_EVENT` "track events") is cross-platform (Windows/MSVC, Linux, macOS) and
is exactly what you want for **per-function elapsed time**: each instrumented
function becomes a timed *slice*; the result opens in
[ui.perfetto.dev](https://ui.perfetto.dev).

`shared_backend` ships an optional, **zero-overhead-when-off** tracing layer with
three backends, selected at build time:

| `-DMYACCEL_TRACE=` | Dependency | Output | Notes |
|---|---|---|---|
| `off` (default) | none | — | macros compile to nothing; no cost |
| `chrome` | **none** (built-in) | `*.json` (Chrome Trace Event) | opens in `chrome://tracing` **and** ui.perfetto.dev. Easiest. |
| `perfetto` | Perfetto SDK | `*.perfetto-trace` | full Perfetto features; fetched via CMake |

Both `chrome` and `perfetto` work on Windows. `chrome` needs **no dependency**,
so it is the simplest way to get function timings; `perfetto` adds the richer
Perfetto ecosystem (counters, flows, the SDK).

---

## 1. Build with tracing

```bat
:: built-in JSON profiler (no dependency) — recommended to start
cmake -S shared_backend -B build -DMYACCEL_TRACE=chrome ...
cmake --build build --config Release

:: Perfetto SDK (CMake fetches it; or -DPERFETTO_SDK_DIR=<sdk>)
cmake -S shared_backend -B build -DMYACCEL_TRACE=perfetto ...
```

## 2. Capture a trace (host code)

C ABI:
```c
npu_trace_start("run.json");   /* ".perfetto-trace" for the perfetto backend */
/* ... run inference (npu_model_load, kernels, ...) ... */
npu_trace_stop();              /* writes the trace file */
```
C++ (RAII, from `npu.hpp`):
```cpp
{
  myaccel::npu::TraceSession ts("run.json");
  // ... inference ...
}   // stops + flushes here
```

## 3. View it
- `chrome` backend `.json` → open in **https://ui.perfetto.dev** (drag & drop) or
  `chrome://tracing`.
- `perfetto` backend `.perfetto-trace` → **https://ui.perfetto.dev**.

Each instrumented function shows as a slice with its **duration**; the UI
aggregates by name (self/total time), groups by thread, and lets you zoom.

Example captured by the built-in `chrome` backend (durations in µs):
```json
[
  {"name":"npu_model_load","ph":"X","pid":2336,"tid":...,"ts":...,"dur":12.4,"cat":"npu"},
  {"name":"npu_alloc","ph":"X",...,"dur":0.23,"cat":"npu"},
  {"name":"npu_matmul_f32","ph":"X",...,"dur":0.23,"cat":"npu"}
]
```

## 4. What is instrumented, and how to add more

Core entry points (`npu_initialize`, `npu_open_device`, `npu_alloc`, `npu_copy`,
`npu_matmul_f32`, `npu_model_load`/`compile`/`bind_weights`, ...) carry
`MYACCEL_TRACE_FUNC()`. To time any scope, include the core-internal `trace.h`
and add:

```cpp
npu_status_t my_kernel(...) {
  MYACCEL_TRACE_FUNC();              // whole-function slice
  { MYACCEL_TRACE_SCOPE("dma:h2d"); /* ... */ }   // sub-scope slice
  ...
}
```
When `MYACCEL_TRACE=off`, these macros expand to nothing — **no runtime cost and
no dependency**, so production builds are unaffected.

## 5. Notes
- Overhead: the `chrome` backend uses a mutex-guarded buffer and
  `std::chrono::steady_clock`; fine for function-level profiling. For very hot,
  fine-grained loops prefer the `perfetto` backend (lock-free per-thread buffers)
  or instrument coarser scopes.
- Thread/async: both backends record the calling thread; slices from different
  threads appear on separate tracks in the UI.
- The instrumentation lives in the core; adapters (ORT/ggml/ExecuTorch) get the
  core's slices for free. To trace adapter-side code too, add the same
  `MYACCEL_TRACE_*` macros there (and the `MYACCEL_TRACE_*` compile definition).
