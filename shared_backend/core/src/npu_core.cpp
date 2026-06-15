// Copyright (c) 2026.
// Reference / stub implementation of the MyAccel NPU core (npu_core + npu_memory).
//
// Every body below is a HOST-MEMORY placeholder so the whole tree compiles and
// links and you can smoke-test the adapters end to end on CPU. Replace each
// TODO with the matching call into your NPU SDK.

#include "myaccel/npu_core.h"    // internal: myaccel::core
#include "myaccel/npu_memory.h"  // internal: myaccel::memory

#include <atomic>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <new>

namespace myaccel {

// Definitions completing the opaque handles forward-declared in npu_api.h.
// In a real backend these wrap SDK handles; here they wrap host memory.
namespace npu {
struct Device {
  int32_t id = 0;
};
struct Buffer {
  void* host_ptr = nullptr;
  size_t bytes = 0;
};
struct Stream {
  Device* device = nullptr;
};
}  // namespace npu

namespace {
std::atomic<int> g_init_refcount{0};
}  // namespace

// ===========================================================================
// npu_core — device lifetime, enumeration, streams, kernels
// ===========================================================================
namespace core {

npu::Status Initialize() {
  g_init_refcount.fetch_add(1, std::memory_order_acq_rel);
  // TODO: npuInit() / driver open.
  return npu::Status::kOk;
}

void Shutdown() {
  if (g_init_refcount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    // TODO: npuShutdown().
  }
}

int32_t GetDeviceCount() {
  // TODO: query the SDK. The stub advertises a single device.
  return 1;
}

npu::Status GetDeviceInfo(int32_t device_id, npu::DeviceInfo* out_info) {
  if (out_info == nullptr || device_id < 0 || device_id >= GetDeviceCount()) {
    return npu::Status::kInvalidArgument;
  }
  out_info->device_id = device_id;
  std::snprintf(out_info->name, sizeof(out_info->name), "%s NPU %d", npu::kVendorName, device_id);
  out_info->total_memory = static_cast<uint64_t>(8) * 1024 * 1024 * 1024;  // TODO: real value
  out_info->compute_units = 1;                                            // TODO: real value
  return npu::Status::kOk;
}

npu::Device* OpenDevice(int32_t device_id) {
  if (device_id < 0 || device_id >= GetDeviceCount()) return nullptr;
  auto* d = new (std::nothrow) npu::Device();
  if (d != nullptr) d->id = device_id;
  // TODO: npuOpenDevice(device_id).
  return d;
}

void CloseDevice(npu::Device* device) {
  // TODO: npuCloseDevice(device).
  delete device;
}

npu::Stream* CreateStream(npu::Device* device) {
  auto* s = new (std::nothrow) npu::Stream();
  if (s != nullptr) s->device = device;
  // TODO: npuCreateStream(device).
  return s;
}

void DestroyStream(npu::Stream* stream) {
  // TODO: npuDestroyStream(stream).
  delete stream;
}

npu::Status Synchronize(npu::Stream* /*stream*/) {
  // TODO: npuStreamSynchronize / npuDeviceSynchronize.
  return npu::Status::kOk;
}

npu::Status MatMulF32(npu::Device* /*device*/, npu::Stream* /*stream*/,
                      const void* a, const void* b, void* out,
                      int64_t m, int64_t k, int64_t n) {
  if (a == nullptr || b == nullptr || out == nullptr) return npu::Status::kInvalidArgument;
  // TODO: dispatch to the NPU GEMM kernel. Stub is a naive CPU reference so the
  // adapters can be validated for correctness before the kernel exists.
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
  return npu::Status::kOk;
}

}  // namespace core

// ===========================================================================
// npu_memory — allocation and host<->device copies
// ===========================================================================
namespace memory {

npu::Buffer* Alloc(npu::Device* /*device*/, size_t bytes) {
  auto* b = new (std::nothrow) npu::Buffer();
  if (b == nullptr) return nullptr;
  // TODO: npuMalloc(device, bytes) -> device pointer. Stub uses host memory.
  b->host_ptr = std::malloc(bytes ? bytes : 1);
  b->bytes = bytes;
  if (b->host_ptr == nullptr) {
    delete b;
    return nullptr;
  }
  return b;
}

void Free(npu::Buffer* buffer) {
  if (buffer == nullptr) return;
  // TODO: npuFree(buffer).
  std::free(buffer->host_ptr);
  delete buffer;
}

void* DevicePtr(npu::Buffer* buffer) { return buffer ? buffer->host_ptr : nullptr; }

npu::Status Copy(npu::Device* /*device*/, void* dst, const void* src, size_t bytes,
                 npu::CopyKind /*kind*/) {
  if (dst == nullptr || src == nullptr) return npu::Status::kInvalidArgument;
  // TODO: npuMemcpy with the right direction. Stub is a host memcpy for all kinds.
  std::memcpy(dst, src, bytes);
  return npu::Status::kOk;
}

}  // namespace memory
}  // namespace myaccel
