// Copyright (c) 2026.
// Reference / stub implementation of the MyAccel NPU core.
//
// Every body below is a HOST-MEMORY placeholder so the whole tree compiles and
// links and you can smoke-test the adapters end to end on CPU. Replace each
// TODO with the matching call into your NPU SDK.

#include "myaccel/npu_core.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <new>

namespace myaccel {
namespace {

std::atomic<int> g_init_refcount{0};

// In a real backend these opaque types wrap SDK handles. Here they wrap host
// memory so the reference build is self-contained.
struct DeviceImpl {
  int32_t id = 0;
};
struct BufferImpl {
  void* host_ptr = nullptr;
  size_t bytes = 0;
};

}  // namespace

struct Device : DeviceImpl {};
struct Buffer : BufferImpl {};
struct Stream {
  Device* device = nullptr;
};

Status Initialize() {
  g_init_refcount.fetch_add(1, std::memory_order_acq_rel);
  // TODO: npuInit() / driver open.
  return Status::kOk;
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

Status GetDeviceInfo(int32_t device_id, DeviceInfo* out_info) {
  if (out_info == nullptr || device_id < 0 || device_id >= GetDeviceCount()) {
    return Status::kInvalidArgument;
  }
  out_info->device_id = device_id;
  std::snprintf(out_info->name, sizeof(out_info->name), "%s NPU %d", kVendorName, device_id);
  out_info->total_memory = static_cast<uint64_t>(8) * 1024 * 1024 * 1024;  // TODO: real value
  out_info->compute_units = 1;                                            // TODO: real value
  return Status::kOk;
}

Device* OpenDevice(int32_t device_id) {
  if (device_id < 0 || device_id >= GetDeviceCount()) return nullptr;
  auto* d = new (std::nothrow) Device();
  if (d != nullptr) d->id = device_id;
  // TODO: npuOpenDevice(device_id).
  return d;
}

void CloseDevice(Device* device) {
  // TODO: npuCloseDevice(device).
  delete device;
}

Buffer* Alloc(Device* /*device*/, size_t bytes) {
  auto* b = new (std::nothrow) Buffer();
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

void Free(Buffer* buffer) {
  if (buffer == nullptr) return;
  // TODO: npuFree(buffer).
  std::free(buffer->host_ptr);
  delete buffer;
}

void* DevicePtr(Buffer* buffer) { return buffer ? buffer->host_ptr : nullptr; }

Status Copy(Device* /*device*/, void* dst, const void* src, size_t bytes, CopyKind /*kind*/) {
  if (dst == nullptr || src == nullptr) return Status::kInvalidArgument;
  // TODO: npuMemcpy with the right direction. Stub is a host memcpy for all kinds.
  std::memcpy(dst, src, bytes);
  return Status::kOk;
}

Stream* CreateStream(Device* device) {
  auto* s = new (std::nothrow) Stream();
  if (s != nullptr) s->device = device;
  // TODO: npuCreateStream(device).
  return s;
}

void DestroyStream(Stream* stream) {
  // TODO: npuDestroyStream(stream).
  delete stream;
}

Status Synchronize(Stream* /*stream*/) {
  // TODO: npuStreamSynchronize / npuDeviceSynchronize.
  return Status::kOk;
}

Status MatMulF32(Device* /*device*/, Stream* /*stream*/,
                 const void* a, const void* b, void* out,
                 int64_t m, int64_t k, int64_t n) {
  if (a == nullptr || b == nullptr || out == nullptr) return Status::kInvalidArgument;
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
  return Status::kOk;
}

}  // namespace myaccel
