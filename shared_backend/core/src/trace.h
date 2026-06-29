// Copyright (c) 2026.
// Core-internal instrumentation macros for function-level tracing.
//
// Backend selected at build time (CMake -DMYACCEL_TRACE=off|chrome|perfetto):
//   off      -> macros expand to nothing (zero overhead, no dependency)
//   chrome   -> built-in Chrome Trace Event JSON (no dependency); opens in
//               chrome://tracing and https://ui.perfetto.dev
//   perfetto -> Perfetto Tracing SDK (TRACE_EVENT); .perfetto-trace
//
// Usage inside core .cpp:
//   npu_status_t npu_foo(...) { MYACCEL_TRACE_FUNC(); ... }      // whole function
//   { MYACCEL_TRACE_SCOPE("phase:dma"); ... }                    // a sub-scope

#ifndef MYACCEL_CORE_TRACE_H_
#define MYACCEL_CORE_TRACE_H_

#define MYACCEL_TRACE_CONCAT2(a, b) a##b
#define MYACCEL_TRACE_CONCAT(a, b)  MYACCEL_TRACE_CONCAT2(a, b)

#if defined(MYACCEL_TRACE_PERFETTO)

#include <perfetto.h>
PERFETTO_DEFINE_CATEGORIES(perfetto::Category("npu").SetDescription("MyAccel NPU core"));
#define MYACCEL_TRACE_SCOPE(name) TRACE_EVENT("npu", name)
#define MYACCEL_TRACE_FUNC()      TRACE_EVENT("npu", __func__)

#elif defined(MYACCEL_TRACE_CHROME)

namespace myaccel {
namespace trace {
// RAII slice: records a "complete" event {name, ts, dur} on destruction.
class ScopedSlice {
 public:
  explicit ScopedSlice(const char* name);
  ~ScopedSlice();
  ScopedSlice(const ScopedSlice&) = delete;
  ScopedSlice& operator=(const ScopedSlice&) = delete;
 private:
  const char* name_;
  int64_t     begin_ns_;
};
}  // namespace trace
}  // namespace myaccel

#define MYACCEL_TRACE_SCOPE(name) \
  ::myaccel::trace::ScopedSlice MYACCEL_TRACE_CONCAT(_myaccel_slice_, __LINE__)(name)
#define MYACCEL_TRACE_FUNC() MYACCEL_TRACE_SCOPE(__func__)

#else  // tracing off

#define MYACCEL_TRACE_SCOPE(name) ((void)0)
#define MYACCEL_TRACE_FUNC()      ((void)0)

#endif

#endif  // MYACCEL_CORE_TRACE_H_
