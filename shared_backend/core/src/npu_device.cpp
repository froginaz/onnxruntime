// Copyright (c) 2026.
// Reference / stub implementation of the device / memory / kernel part of the
// unified C ABI (npu.h). Host-memory placeholders so the tree builds and the
// adapters can be validated on CPU. Replace each TODO with your NPU SDK call.

#include "myaccel/npu.h"
#include "trace.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <new>

// Opaque handle definitions (declared in npu.h). In a real backend these wrap
// SDK handles; here they wrap host memory.
struct npu_device { int32_t id = 0; };
struct npu_buffer { void* host_ptr = nullptr; size_t bytes = 0; };
struct npu_stream { npu_device_t* device = nullptr; };

namespace {
std::atomic<int> g_init_refcount{0};
}  // namespace

extern "C" {

const char* npu_vendor_name(void)    { return NPU_VENDOR_NAME; }
const char* npu_backend_name(void)   { return NPU_BACKEND_NAME; }
const char* npu_version_string(void) { return NPU_VERSION_STRING; }

// --- lifetime --------------------------------------------------------------
npu_status_t npu_initialize(void) {
  MYACCEL_TRACE_FUNC();
  g_init_refcount.fetch_add(1, std::memory_order_acq_rel);
  // TODO: npuInit() / driver open.
  return NPU_OK;
}

void npu_shutdown(void) {
  if (g_init_refcount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    // TODO: npuShutdown().
  }
}

// --- device enumeration ----------------------------------------------------
int32_t npu_get_device_count(void) {
  // TODO: query the SDK. The stub advertises a single device.
  return 1;
}

npu_status_t npu_get_device_info(int32_t device_id, npu_device_info_t* out_info) {
  MYACCEL_TRACE_FUNC();
  if (out_info == nullptr || device_id < 0 || device_id >= npu_get_device_count()) {
    return NPU_ERR_INVALID_ARGUMENT;
  }
  out_info->device_id = device_id;
  std::snprintf(out_info->name, sizeof(out_info->name), "%s NPU %d", NPU_VENDOR_NAME, device_id);
  out_info->total_memory = static_cast<uint64_t>(8) * 1024 * 1024 * 1024;  // TODO: real value
  out_info->compute_units = 1;                                            // TODO: real value
  return NPU_OK;
}

npu_device_t* npu_open_device(int32_t device_id) {
  MYACCEL_TRACE_FUNC();
  if (device_id < 0 || device_id >= npu_get_device_count()) return nullptr;
  auto* d = new (std::nothrow) npu_device();
  if (d != nullptr) d->id = device_id;
  // TODO: npuOpenDevice(device_id).
  return d;
}

void npu_close_device(npu_device_t* device) {
  MYACCEL_TRACE_FUNC();
  // TODO: npuCloseDevice(device).
  delete device;
}

// --- memory ----------------------------------------------------------------
npu_buffer_t* npu_alloc(npu_device_t* /*device*/, size_t bytes) {
  MYACCEL_TRACE_FUNC();
  auto* b = new (std::nothrow) npu_buffer();
  if (b == nullptr) return nullptr;
  // TODO: npuMalloc(device, bytes). Stub uses host memory.
  b->host_ptr = std::malloc(bytes ? bytes : 1);
  b->bytes = bytes;
  if (b->host_ptr == nullptr) {
    delete b;
    return nullptr;
  }
  return b;
}

void npu_free(npu_buffer_t* buffer) {
  MYACCEL_TRACE_FUNC();
  if (buffer == nullptr) return;
  // TODO: npuFree(buffer).
  std::free(buffer->host_ptr);
  delete buffer;
}

void* npu_device_ptr(npu_buffer_t* buffer) { return buffer ? buffer->host_ptr : nullptr; }

npu_status_t npu_copy(npu_device_t* /*device*/, void* dst, const void* src, size_t bytes,
                      npu_copy_kind_t /*kind*/) {
  MYACCEL_TRACE_FUNC();
  if (dst == nullptr || src == nullptr) return NPU_ERR_INVALID_ARGUMENT;
  // TODO: npuMemcpy with the right direction. Stub is a host memcpy for all kinds.
  std::memcpy(dst, src, bytes);
  return NPU_OK;
}

// --- streams ---------------------------------------------------------------
npu_stream_t* npu_create_stream(npu_device_t* device) {
  auto* s = new (std::nothrow) npu_stream();
  if (s != nullptr) s->device = device;
  // TODO: npuCreateStream(device).
  return s;
}

void npu_destroy_stream(npu_stream_t* stream) {
  // TODO: npuDestroyStream(stream).
  delete stream;
}

npu_status_t npu_synchronize(npu_stream_t* /*stream*/) {
  MYACCEL_TRACE_FUNC();
  // TODO: npuStreamSynchronize / npuDeviceSynchronize.
  return NPU_OK;
}

// --- kernels ---------------------------------------------------------------
npu_status_t npu_matmul_f32(npu_device_t* /*device*/, npu_stream_t* /*stream*/,
                            const void* a, const void* b, void* out,
                            int64_t m, int64_t k, int64_t n) {
  MYACCEL_TRACE_FUNC();
  if (a == nullptr || b == nullptr || out == nullptr) return NPU_ERR_INVALID_ARGUMENT;
  // TODO: dispatch to the NPU GEMM. Stub is a naive CPU reference so the adapters
  // can be validated for correctness before the kernel exists.
  const auto* A = static_cast<const float*>(a);
  const auto* B = static_cast<const float*>(b);
  auto* C = static_cast<float*>(out);
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (int64_t p = 0; p < k; ++p) acc += A[i * k + p] * B[p * n + j];
      C[i * n + j] = acc;
    }
  }
  return NPU_OK;
}

}  // extern "C"
