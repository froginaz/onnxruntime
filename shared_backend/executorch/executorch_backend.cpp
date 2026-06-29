// Copyright (c) 2026.
// ExecuTorch backend (delegate) adapter -> myaccel_backend.dll / libmyaccel_backend.so
//
// Implements the ExecuTorch BackendInterface (init / execute / destroy) on top
// of the framework-agnostic MyAccel NPU façade. Like the ggml/ort adapters it
// only translates the framework's ABI down to myaccel::npu + the npu_model C
// ABI, so this DLL references myaccel_core (and ships beside it).
//
// NOTE on ExecuTorch API/namespace: ExecuTorch has moved its public symbols
// across versions (torch::executor -> executorch::runtime) and header paths
// change between releases. This is a structural skeleton; confirm the includes
// and namespace against your ExecuTorch checkout
// (executorch/runtime/backend/interface.h) and adjust the aliases below.

#include <executorch/runtime/backend/interface.h>

#include "myaccel/npu.hpp"  // sole NPU access point (myaccel::npu + npu_model C ABI)

namespace {

namespace et = executorch::runtime;

// State kept per delegated subgraph between init() and execute()/destroy().
struct MyAccelDelegateHandle {
  npu_model_t* model = nullptr;
};

// The backend: AOT-compiled NPU programs (model.nnc + weight.bin) are embedded
// in the ExecuTorch .pte as the delegate "processed" blob by the partitioner.
class MyAccelBackend final : public et::BackendInterface {
 public:
  bool is_available() const override { return myaccel::npu::GetDeviceCount() > 0; }

  et::Result<et::DelegateHandle*> init(et::BackendInitContext& context,
                                       et::FreeableBuffer* processed,
                                       et::ArrayRef<et::CompileSpec> /*compile_specs*/) const override {
    myaccel::npu::Initialize();

    // The processed blob is the NPU-native program produced at export time.
    npu_model_load_info_t info = {};
    info.struct_size = sizeof(info);
    info.api_version = NPU_API_VERSION;
    info.nnc.kind = NPU_SOURCE_MEMORY;
    info.nnc.u.memory.data = processed->data();
    info.nnc.u.memory.size = processed->size();
    // Skeleton: weights packed inside the same blob. A real backend would split
    // nnc vs weight.bin, or carry weights via CompileSpec.
    info.weights = info.nnc;
    info.residency = NPU_WEIGHTS_COPY_TO_DEVICE;

    npu_model_t* model = nullptr;
    // TODO: thread the real npu_device_t through; the stub ignores it.
    if (npu_model_load(/*device*/ nullptr, &info, &model) != NPU_OK) {
      return et::Error::Internal;
    }
    processed->Free();  // COPY_TO_DEVICE: the AOT blob is no longer needed

    auto* handle =
        context.get_runtime_allocator()->allocateInstance<MyAccelDelegateHandle>();
    if (handle == nullptr) {
      npu_model_free(model);
      return et::Error::MemoryAllocationFailed;
    }
    handle->model = model;
    return handle;
  }

  et::Error execute(et::BackendExecutionContext& /*context*/, et::DelegateHandle* handle,
                    et::EValue** /*args*/) const override {
    auto* h = static_cast<MyAccelDelegateHandle*>(handle);
    if (h == nullptr || h->model == nullptr) return et::Error::InvalidArgument;
    // TODO: bind args (EValue tensors) to npu_model inputs/outputs via
    // npu_model_get_input/get_output, run on the NPU, write results back.
    return et::Error::Ok;
  }

  void destroy(et::DelegateHandle* handle) const override {
    auto* h = static_cast<MyAccelDelegateHandle*>(handle);
    if (h != nullptr) {
      npu_model_free(h->model);
      h->model = nullptr;
    }
    myaccel::npu::Shutdown();
  }
};

// Static registration: runs on library load (dlopen / process start). Backends
// linked statically may need --whole-archive so this object is retained.
MyAccelBackend g_myaccel_backend;
et::Backend g_backend_descriptor{"MyAccelBackend", &g_myaccel_backend};
[[maybe_unused]] const et::Error g_registered = et::register_backend(g_backend_descriptor);

}  // namespace

// Explicit registration entry for hosts that load this as a plugin and prefer
// to register on demand instead of relying on static init.
#if defined(_WIN32)
#define MYACCEL_BACKEND_EXPORT __declspec(dllexport)
#else
#define MYACCEL_BACKEND_EXPORT __attribute__((visibility("default")))
#endif

extern "C" MYACCEL_BACKEND_EXPORT int myaccel_backend_register(void) {
  return static_cast<int>(et::register_backend(g_backend_descriptor));
}
