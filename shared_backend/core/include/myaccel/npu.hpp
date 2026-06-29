// Copyright (c) 2026.
// MyAccel NPU — optional header-only C++ wrapper over the unified C ABI (npu.h).
//
// Ergonomics (namespaces, enum class, typed handle aliases) with ZERO ABI cost:
// every function below is `inline` and forwards 1:1 to the C ABI, so it is NOT
// part of the exported ABI. Adapters include this header and call
// `myaccel::npu::*`; nothing crosses a binary boundary as a C++ type.
//
// Prefer this in C++ code. Use the raw C ABI (npu.h) for FFI / other languages /
// other compilers.

#ifndef MYACCEL_NPU_HPP_
#define MYACCEL_NPU_HPP_

#include "myaccel/npu.h"

namespace myaccel {
namespace npu {

// --- handle aliases (the same opaque C types) ------------------------------
using Device     = ::npu_device_t;
using Buffer     = ::npu_buffer_t;
using Stream     = ::npu_stream_t;
using Model      = ::npu_model_t;
using DeviceInfo = ::npu_device_info_t;
using Status     = ::npu_status_t;

// --- identity --------------------------------------------------------------
inline constexpr uint32_t    kVendorId   = NPU_VENDOR_ID;
inline constexpr const char* kVendorName = NPU_VENDOR_NAME;
inline constexpr const char* kBackendName = NPU_BACKEND_NAME;
inline constexpr const char* kVersion    = NPU_VERSION_STRING;

// --- copy direction (typed) ------------------------------------------------
enum class CopyKind : int32_t {
  kHostToDevice   = NPU_COPY_HOST_TO_DEVICE,
  kDeviceToHost   = NPU_COPY_DEVICE_TO_HOST,
  kDeviceToDevice = NPU_COPY_DEVICE_TO_DEVICE,
};

// --- lifetime / device -----------------------------------------------------
inline Status Initialize() { return npu_initialize(); }
inline void   Shutdown()   { npu_shutdown(); }
inline int32_t GetDeviceCount() { return npu_get_device_count(); }
inline Status GetDeviceInfo(int32_t id, DeviceInfo* out) { return npu_get_device_info(id, out); }
inline Device* OpenDevice(int32_t id) { return npu_open_device(id); }
inline void   CloseDevice(Device* d)  { npu_close_device(d); }

// --- memory ----------------------------------------------------------------
inline Buffer* Alloc(Device* d, size_t bytes) { return npu_alloc(d, bytes); }
inline void    Free(Buffer* b)                { npu_free(b); }
inline void*   DevicePtr(Buffer* b)           { return npu_device_ptr(b); }
inline Status  Copy(Device* d, void* dst, const void* src, size_t bytes, CopyKind k) {
  return npu_copy(d, dst, src, bytes, static_cast<npu_copy_kind_t>(k));
}

// --- streams / kernels -----------------------------------------------------
inline Stream* CreateStream(Device* d) { return npu_create_stream(d); }
inline void    DestroyStream(Stream* s){ npu_destroy_stream(s); }
inline Status  Synchronize(Stream* s)  { return npu_synchronize(s); }
inline Status  MatMulF32(Device* d, Stream* s, const void* a, const void* b, void* out,
                         int64_t m, int64_t k, int64_t n) {
  return npu_matmul_f32(d, s, a, b, out, m, k, n);
}

// --- tracing / profiling ---------------------------------------------------
inline void TraceStart(const char* output_path) { npu_trace_start(output_path); }
inline void TraceStop()                          { npu_trace_stop(); }

// RAII trace session: start on construction, stop+flush on destruction.
//   { myaccel::npu::TraceSession ts("run.json"); /* inference */ }
struct TraceSession {
  explicit TraceSession(const char* output_path) { npu_trace_start(output_path); }
  ~TraceSession() { npu_trace_stop(); }
  TraceSession(const TraceSession&) = delete;
  TraceSession& operator=(const TraceSession&) = delete;
};

}  // namespace npu
}  // namespace myaccel

#endif  // MYACCEL_NPU_HPP_
