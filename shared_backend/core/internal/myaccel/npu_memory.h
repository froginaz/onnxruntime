// Copyright (c) 2026.
// Internal: npu_memory — device allocation and host<->device copies.
//
// PRIVATE to the myaccel_core target. Adapters reach memory through the public
// façade myaccel::npu in <myaccel/npu_api.h>, never this header.

#ifndef MYACCEL_INTERNAL_NPU_MEMORY_H_
#define MYACCEL_INTERNAL_NPU_MEMORY_H_

#include "myaccel/npu_api.h"  // shared types: npu::Buffer / Device / Status / CopyKind

namespace myaccel {
namespace memory {

// Low-level memory ops (stubbed over host memory). No MYACCEL_API: internal.
npu::Buffer* Alloc(npu::Device* device, size_t bytes);
void Free(npu::Buffer* buffer);
void* DevicePtr(npu::Buffer* buffer);
npu::Status Copy(npu::Device* device, void* dst, const void* src, size_t bytes, npu::CopyKind kind);

}  // namespace memory
}  // namespace myaccel

#endif  // MYACCEL_INTERNAL_NPU_MEMORY_H_
