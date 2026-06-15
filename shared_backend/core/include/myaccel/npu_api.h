// Copyright (c) 2026.
// MyAccel NPU — common public API (façade).
//
// This is the ONE header the framework adapters (ggml-myaccel, myaccel_ort_ep)
// include to reach the NPU. It wraps the internal `npu_core` (device + compute)
// and `npu_memory` (allocation) modules behind a single namespace, and also
// pulls in the model-loading C ABI. The low-level headers live in core/internal
// and are NOT on the adapters' include path, so the adapters can only touch the
// NPU through `myaccel::npu::*` declared here.
//
//   adapters ──> myaccel::npu (this header) ──> npu_core / npu_memory (internal)

#ifndef MYACCEL_NPU_API_H_
#define MYACCEL_NPU_API_H_

#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#if defined(MYACCEL_CORE_BUILD_DLL)
#define MYACCEL_API __declspec(dllexport)
#elif defined(MYACCEL_CORE_USE_DLL)
#define MYACCEL_API __declspec(dllimport)
#else
#define MYACCEL_API
#endif
#else
#define MYACCEL_API __attribute__((visibility("default")))
#endif

namespace myaccel {
namespace npu {

// Stable identity the adapters report to their framework. Keep kVendorId in sync
// with the value registered with each framework (ORT OrtEpFactory::GetVendorId,
// ggml device vendor, ...).
constexpr uint32_t kVendorId = 0x1ACC;       // TODO: your PCI/SoC vendor id
constexpr const char* kVendorName = "MyAccel";
constexpr const char* kBackendName = "myaccel_npu";
constexpr const char* kVersion = "0.1.0";

enum class Status : int32_t {
  kOk = 0,
  kInvalidArgument = 1,
  kOutOfMemory = 2,
  kDeviceError = 3,
  kNotImplemented = 4,
};

// Opaque handles owned by the core. Adapters never dereference these.
struct Device;       // one physical NPU
struct Buffer;       // a device allocation
struct Stream;       // an async execution queue (optional)

struct DeviceInfo {
  int32_t device_id = 0;
  char name[128] = {0};
  uint64_t total_memory = 0;
  uint32_t compute_units = 0;
};

enum class CopyKind : int32_t { kHostToDevice, kDeviceToHost, kDeviceToDevice };

// --- lifetime (npu_core) --------------------------------------------------
MYACCEL_API Status Initialize();
MYACCEL_API void Shutdown();

// --- device enumeration (npu_core) ----------------------------------------
MYACCEL_API int32_t GetDeviceCount();
MYACCEL_API Status GetDeviceInfo(int32_t device_id, DeviceInfo* out_info);
MYACCEL_API Device* OpenDevice(int32_t device_id);
MYACCEL_API void CloseDevice(Device* device);

// --- memory (npu_memory) --------------------------------------------------
MYACCEL_API Buffer* Alloc(Device* device, size_t bytes);
MYACCEL_API void Free(Buffer* buffer);
MYACCEL_API void* DevicePtr(Buffer* buffer);  // raw device pointer for tensors
MYACCEL_API Status Copy(Device* device, void* dst, const void* src, size_t bytes, CopyKind kind);

// --- streams (npu_core, optional async) -----------------------------------
MYACCEL_API Stream* CreateStream(Device* device);
MYACCEL_API void DestroyStream(Stream* stream);
MYACCEL_API Status Synchronize(Stream* stream);  // nullptr => sync whole device

// --- kernels (npu_core) ---------------------------------------------------
// out[M,N] = a[M,K] * b[K,N]   (fp32 reference signature). Grow as needed.
MYACCEL_API Status MatMulF32(Device* device, Stream* stream,
                             const void* a, const void* b, void* out,
                             int64_t m, int64_t k, int64_t n);

}  // namespace npu
}  // namespace myaccel

// Model-loading C ABI is part of the same public surface (load model.nnc +
// weight.bin). Adapters reach it through this umbrella too.
#include "myaccel/npu_model.h"

#endif  // MYACCEL_NPU_API_H_
