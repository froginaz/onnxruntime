// Copyright (c) 2026.
// Implementation of the public façade myaccel::npu — forwards to the internal
// npu_core (device/compute) and npu_memory (allocation) modules. This is the
// single boundary the framework adapters link against.

#include "myaccel/npu_api.h"

#include "myaccel/npu_core.h"    // internal: myaccel::core
#include "myaccel/npu_memory.h"  // internal: myaccel::memory

namespace myaccel {
namespace npu {

// --- lifetime / device (npu_core) -----------------------------------------
Status Initialize() { return core::Initialize(); }
void Shutdown() { core::Shutdown(); }
int32_t GetDeviceCount() { return core::GetDeviceCount(); }
Status GetDeviceInfo(int32_t device_id, DeviceInfo* out_info) {
  return core::GetDeviceInfo(device_id, out_info);
}
Device* OpenDevice(int32_t device_id) { return core::OpenDevice(device_id); }
void CloseDevice(Device* device) { core::CloseDevice(device); }

// --- memory (npu_memory) --------------------------------------------------
Buffer* Alloc(Device* device, size_t bytes) { return memory::Alloc(device, bytes); }
void Free(Buffer* buffer) { memory::Free(buffer); }
void* DevicePtr(Buffer* buffer) { return memory::DevicePtr(buffer); }
Status Copy(Device* device, void* dst, const void* src, size_t bytes, CopyKind kind) {
  return memory::Copy(device, dst, src, bytes, kind);
}

// --- streams / kernels (npu_core) -----------------------------------------
Stream* CreateStream(Device* device) { return core::CreateStream(device); }
void DestroyStream(Stream* stream) { core::DestroyStream(stream); }
Status Synchronize(Stream* stream) { return core::Synchronize(stream); }
Status MatMulF32(Device* device, Stream* stream, const void* a, const void* b, void* out,
                 int64_t m, int64_t k, int64_t n) {
  return core::MatMulF32(device, stream, a, b, out, m, k, n);
}

}  // namespace npu
}  // namespace myaccel
