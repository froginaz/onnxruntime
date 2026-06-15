// Copyright (c) 2026.
// Internal: npu_core — device lifetime, enumeration, streams, compute kernels.
//
// PRIVATE to the myaccel_core target (core/internal is not on the adapters'
// include path). Adapters must go through the public façade myaccel::npu in
// <myaccel/npu_api.h>; they cannot include this header.

#ifndef MYACCEL_INTERNAL_NPU_CORE_H_
#define MYACCEL_INTERNAL_NPU_CORE_H_

#include "myaccel/npu_api.h"  // shared types: npu::Status / Device / Stream / DeviceInfo

namespace myaccel {
namespace core {

// Low-level device + compute, implemented against the real NPU SDK (stubbed).
// No MYACCEL_API: these symbols stay internal to the core DLL.
npu::Status Initialize();
void Shutdown();

int32_t GetDeviceCount();
npu::Status GetDeviceInfo(int32_t device_id, npu::DeviceInfo* out_info);
npu::Device* OpenDevice(int32_t device_id);
void CloseDevice(npu::Device* device);

npu::Stream* CreateStream(npu::Device* device);
void DestroyStream(npu::Stream* stream);
npu::Status Synchronize(npu::Stream* stream);

npu::Status MatMulF32(npu::Device* device, npu::Stream* stream,
                      const void* a, const void* b, void* out,
                      int64_t m, int64_t k, int64_t n);

}  // namespace core
}  // namespace myaccel

#endif  // MYACCEL_INTERNAL_NPU_CORE_H_
