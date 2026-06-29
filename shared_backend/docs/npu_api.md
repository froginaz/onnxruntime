# NPU public façade API (`myaccel/npu_api.h`)

The C++ façade (`namespace myaccel::npu`) that the framework adapters call for
**device, memory, and kernel** access. It is the **single NPU access point**:
adapters include only this header, and it wraps the internal `npu_core`
(device/compute) and `npu_memory` (allocation) modules. It also umbrella-includes
the model-loading C ABI (`npu_model.h`).

```
ort_ep / ggml-myaccel / myaccel_backend
        │  include <myaccel/npu_api.h>  (only)
        ▼
   myaccel::npu  ──wraps──▶ npu_core (device/compute) + npu_memory (alloc)   [internal]
```

> Why C++ here, but C for `npu_model.h`? The façade's only callers are this
> repo's adapters, built with the **same toolchain** as the core, passing opaque
> pointers + scalars — so C++ (type safety, `enum class`, namespaces) is the
> better fit. Model loading is the external/versioned boundary, hence C. See
> [`design.md`](design.md#2-layered-structure) and [`npu_model_api.md`](npu_model_api.md).

## Surface

| Group | Functions | Notes |
|---|---|---|
| Identity | `kVendorId`, `kVendorName`, `kBackendName`, `kVersion` | `constexpr`; reported to each framework |
| Lifetime | `Initialize()`, `Shutdown()` | reference-counted; N inits need N shutdowns |
| Device | `GetDeviceCount()`, `GetDeviceInfo()`, `OpenDevice()`, `CloseDevice()` | `OpenDevice` returns `nullptr` on bad id |
| Memory | `Alloc()`, `Free()`, `DevicePtr()`, `Copy()` | `Copy` takes a `CopyKind` direction |
| Streams | `CreateStream()`, `DestroyStream()`, `Synchronize()` | optional async; `Synchronize(nullptr)` syncs the whole device |
| Kernels | `MatMulF32()` | grow this set as the NPU gains ops |

Opaque handles (`Device`, `Buffer`, `Stream`) are never dereferenced by adapters;
`Status` / `CopyKind` are `enum class`, `DeviceInfo` is a POD.

## llama.cpp ggml adapter (alloc / copy / kernel)

```cpp
#include "myaccel/npu_api.h"
using namespace myaccel;

// device discovery (ggml reg.get_device_count / device props)
npu::Initialize();
int n = npu::GetDeviceCount();
npu::DeviceInfo info{};
npu::GetDeviceInfo(0, &info);                 // -> info.name, info.total_memory
npu::Device* dev = npu::OpenDevice(0);

// buffer-type alloc (ggml buft_alloc_buffer)
npu::Buffer* buf = npu::Alloc(dev, nbytes);
void* base = npu::DevicePtr(buf);             // embed in the ggml tensor

// set_tensor / get_tensor
npu::Copy(dev, dst, src, size, npu::CopyKind::kHostToDevice);
npu::Copy(dev, host, dev_ptr, size, npu::CopyKind::kDeviceToHost);

// graph_compute -> MUL_MAT
npu::MatMulF32(dev, /*stream*/ nullptr, a, b, out, m, k, n);

// teardown
npu::Free(buf);
npu::CloseDevice(dev);
npu::Shutdown();
```

## onnxruntime EP adapter (lifecycle hooks)

```cpp
#include "myaccel/npu_api.h"
using namespace myaccel;

// CreateEpFactories -> factory ctor
npu::Initialize();

// CreateEp -> open the matched device
npu::Device* dev = npu::OpenDevice(0);        // store in the OrtEp instance
if (dev == nullptr) { /* return ORT_EP_FAIL */ }

// OrtEp::Sync
npu::Synchronize(nullptr);                    // whole-device sync

// ~MyAccelEp / ReleaseEpFactory
npu::CloseDevice(dev);
npu::Shutdown();
```

For actual model offload, the adapter combines the façade with the model C ABI:
extract `model.nnc` + `weight.bin`, call `npu_model_load(...)` (see
[`npu_model_api.md`](npu_model_api.md)), then bind I/O and dispatch kernels.

## Lifetime & ownership rules

- **`Initialize` / `Shutdown` are reference-counted.** Every adapter that calls
  `Initialize()` must call `Shutdown()` exactly once; the real teardown happens
  on the last `Shutdown`. (ORT: factory ctor ↔ `ReleaseEpFactory`; ggml: reg
  device-count ↔ process teardown.)
- **`OpenDevice` ↔ `CloseDevice`** must be balanced; `OpenDevice` returns
  `nullptr` for an out-of-range id.
- **`Alloc` ↔ `Free`** must be balanced. `DevicePtr(buf)` is the raw device
  pointer for tensor embedding; do not free it directly.
- **`Copy`** validates non-null `dst`/`src` and returns `Status::kInvalidArgument`
  otherwise; pick the `CopyKind` matching the transfer direction.
- **Streams are optional.** Pass `nullptr` as the stream for synchronous
  behavior; `Synchronize(nullptr)` waits on the whole device.

## Extending the kernel set

`MatMulF32` is the seed. To offload more of a model, add kernels here (e.g.
`SoftmaxF32`, `RMSNormF32`, attention) and call them from the adapters' dispatch
points — ggml `backend_graph_compute` / the ORT `OrtNodeComputeInfo::Compute`.
Keep the façade's signatures framework-agnostic (raw pointers + shapes); each
adapter maps its tensors onto them.

> The façade only declares the public surface; the real work lives in the
> internal `npu_core` / `npu_memory` implementation (stub today, your NPU SDK
> later). Adapters never see those internals.
