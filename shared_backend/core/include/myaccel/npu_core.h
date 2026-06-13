// Copyright (c) 2026.
// Framework-agnostic abstraction over the in-house NPU SDK.
//
// This header is the ONLY thing the onnxruntime and llama.cpp adapters depend
// on. It contains no onnxruntime and no ggml types on purpose: the two adapters
// translate their framework's ABI down to these calls, and this layer owns all
// real NPU SDK interaction (device enumeration, memory, kernels).
//
// Replace the stub bodies in npu_core.cpp with calls into your NPU SDK.

#ifndef MYACCEL_NPU_CORE_H_
#define MYACCEL_NPU_CORE_H_

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

// Stable identity used by both adapters when they report the device to their
// framework. Keep VENDOR_ID in sync with the value you register with each
// framework (ORT OrtEpFactory::GetVendorId, ggml device vendor string, etc.).
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

// Opaque handles owned by the core. The adapters never dereference these.
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

// --- lifetime -------------------------------------------------------------
// Idempotent; reference-counted internally so both adapters can call it.
MYACCEL_API Status Initialize();
MYACCEL_API void Shutdown();

// --- device enumeration ---------------------------------------------------
MYACCEL_API int32_t GetDeviceCount();
MYACCEL_API Status GetDeviceInfo(int32_t device_id, DeviceInfo* out_info);
MYACCEL_API Device* OpenDevice(int32_t device_id);
MYACCEL_API void CloseDevice(Device* device);

// --- memory ---------------------------------------------------------------
MYACCEL_API Buffer* Alloc(Device* device, size_t bytes);
MYACCEL_API void Free(Buffer* buffer);
// Raw device pointer (for frameworks that need to embed it in a tensor).
MYACCEL_API void* DevicePtr(Buffer* buffer);
MYACCEL_API Status Copy(Device* device, void* dst, const void* src, size_t bytes, CopyKind kind);

// --- streams (optional async support) -------------------------------------
MYACCEL_API Stream* CreateStream(Device* device);
MYACCEL_API void DestroyStream(Stream* stream);
MYACCEL_API Status Synchronize(Stream* stream);  // pass nullptr to sync whole device

// --- kernels --------------------------------------------------------------
// The two adapters lower a fused subgraph / a ggml op onto these. Start with a
// couple of primitives and grow. All shapes are row-major.
//
// out[M,N] = a[M,K] * b[K,N]   (fp32 reference signature)
MYACCEL_API Status MatMulF32(Device* device, Stream* stream,
                             const void* a, const void* b, void* out,
                             int64_t m, int64_t k, int64_t n);

}  // namespace myaccel

#endif  // MYACCEL_NPU_CORE_H_
