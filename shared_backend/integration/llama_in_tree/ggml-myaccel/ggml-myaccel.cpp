// Copyright (c) 2026.
// llama.cpp / ggml backend adapter -> ggml-myaccel
//
// This file is meant to live IN the llama.cpp tree at
//   llama.cpp/ggml/src/ggml-myaccel/ggml-myaccel.cpp
// and is built in-tree (see the sibling CMakeLists.txt). It implements the
// ggml-backend vtables (reg -> device -> buffer-type -> buffer -> backend) on top
// of the MyAccel NPU core, which it links from the prebuilt shared_backend
// npu_core.lib/.dll. It is a structural skeleton: device/buffer plumbing is
// wired, op dispatch in graph_compute is a stub you extend op-by-op.
//
// NOTE on the dynamic-load entry symbol: ggml has changed the exact symbol it
// looks up for out-of-tree backends across versions (e.g. ggml_backend_init).
// For an in-tree build the stable reg function ggml_backend_myaccel_reg() is
// what ggml-backend-reg.cpp calls; the ggml_backend_init alias is only needed
// for the dynamic-plugin path.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <cstring>
#include <vector>

#include "myaccel/npu.hpp"  // sole NPU access point (myaccel::npu); from shared_backend headers

// ---------------------------------------------------------------------------
// Buffer
// ---------------------------------------------------------------------------
namespace {

struct MyAccelBufferCtx {
  myaccel::npu::Device* device = nullptr;
  myaccel::npu::Buffer* buffer = nullptr;
};

void buffer_free(ggml_backend_buffer_t buffer) {
  auto* ctx = static_cast<MyAccelBufferCtx*>(buffer->context);
  myaccel::npu::Free(ctx->buffer);
  delete ctx;
}

void* buffer_get_base(ggml_backend_buffer_t buffer) {
  auto* ctx = static_cast<MyAccelBufferCtx*>(buffer->context);
  return myaccel::npu::DevicePtr(ctx->buffer);
}

void buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor* tensor,
                       const void* data, size_t offset, size_t size) {
  auto* ctx = static_cast<MyAccelBufferCtx*>(buffer->context);
  myaccel::npu::Copy(ctx->device, static_cast<char*>(tensor->data) + offset, data, size,
                myaccel::npu::CopyKind::kHostToDevice);
}

void buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor* tensor,
                       void* data, size_t offset, size_t size) {
  auto* ctx = static_cast<MyAccelBufferCtx*>(buffer->context);
  myaccel::npu::Copy(ctx->device, data, static_cast<const char*>(tensor->data) + offset, size,
                myaccel::npu::CopyKind::kDeviceToHost);
}

void buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
  auto* ctx = static_cast<MyAccelBufferCtx*>(buffer->context);
  void* base = myaccel::npu::DevicePtr(ctx->buffer);
  std::memset(base, value, ggml_backend_buffer_get_size(buffer));  // stub: host-visible memory
}

constexpr ggml_backend_buffer_i kBufferIface = {
    /* .free_buffer  = */ buffer_free,
    /* .get_base     = */ buffer_get_base,
    /* .init_tensor  = */ nullptr,
    /* .memset_tensor= */ nullptr,
    /* .set_tensor   = */ buffer_set_tensor,
    /* .get_tensor   = */ buffer_get_tensor,
    /* .cpy_tensor   = */ nullptr,
    /* .clear        = */ buffer_clear,
    /* .reset        = */ nullptr,
};

// ---------------------------------------------------------------------------
// Buffer type
// ---------------------------------------------------------------------------
const char* buft_get_name(ggml_backend_buffer_type_t /*buft*/) { return myaccel::npu::kBackendName; }

ggml_backend_buffer_t buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
  auto* device = static_cast<myaccel::npu::Device*>(buft->device->context);
  auto* ctx = new MyAccelBufferCtx{device, myaccel::npu::Alloc(device, size)};
  if (ctx->buffer == nullptr) {
    delete ctx;
    return nullptr;
  }
  return ggml_backend_buffer_init(buft, kBufferIface, ctx, size);
}

size_t buft_get_alignment(ggml_backend_buffer_type_t /*buft*/) { return 256; }  // TODO: NPU alignment

// ---------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------
const char* backend_get_name(ggml_backend_t /*backend*/) { return myaccel::npu::kBackendName; }

void backend_free(ggml_backend_t backend) { delete backend; }

ggml_status backend_graph_compute(ggml_backend_t backend, ggml_cgraph* cgraph) {
  auto* device = static_cast<myaccel::npu::Device*>(backend->device->context);
  for (int i = 0; i < cgraph->n_nodes; ++i) {
    ggml_tensor* node = cgraph->nodes[i];
    switch (node->op) {
      case GGML_OP_NONE:
      case GGML_OP_RESHAPE:
      case GGML_OP_VIEW:
      case GGML_OP_PERMUTE:
      case GGML_OP_TRANSPOSE:
        break;  // no-ops / metadata only
      case GGML_OP_MUL_MAT: {
        // ggml MUL_MAT: result = src0^T-style matmul; this maps M,K,N from the
        // operand shapes. Wire to your NPU GEMM. Stub uses the fp32 reference.
        const ggml_tensor* src0 = node->src[0];
        const ggml_tensor* src1 = node->src[1];
        const int64_t k = src0->ne[0];
        const int64_t m = src1->ne[1];
        const int64_t n = src0->ne[1];
        myaccel::npu::MatMulF32(device, nullptr, src1->data, src0->data, node->data, m, k, n);
        break;
      }
      default:
        // TODO: implement remaining ops, or return GGML_STATUS_FAILED so ggml
        // knows this backend cannot run the graph (it will fall back).
        return GGML_STATUS_FAILED;
    }
  }
  return GGML_STATUS_SUCCESS;
}

constexpr ggml_backend_i kBackendIface = {
    /* .get_name           = */ backend_get_name,
    /* .free               = */ backend_free,
    /* .set_tensor_async   = */ nullptr,
    /* .get_tensor_async   = */ nullptr,
    /* .cpy_tensor_async   = */ nullptr,
    /* .synchronize        = */ nullptr,
    /* .graph_plan_create  = */ nullptr,
    /* .graph_plan_free    = */ nullptr,
    /* .graph_plan_update  = */ nullptr,
    /* .graph_plan_compute = */ nullptr,
    /* .graph_compute      = */ backend_graph_compute,
    /* .event_record       = */ nullptr,
    /* .event_wait         = */ nullptr,
};

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------
const char* device_get_name(ggml_backend_dev_t /*dev*/) { return myaccel::npu::kBackendName; }
const char* device_get_description(ggml_backend_dev_t /*dev*/) { return "MyAccel NPU"; }

void device_get_memory(ggml_backend_dev_t /*dev*/, size_t* free, size_t* total) {
  myaccel::npu::DeviceInfo info{};
  myaccel::npu::GetDeviceInfo(0, &info);
  *total = static_cast<size_t>(info.total_memory);
  *free = *total;  // TODO: query live free memory
}

ggml_backend_dev_type device_get_type(ggml_backend_dev_t /*dev*/) { return GGML_BACKEND_DEVICE_TYPE_GPU; }

void device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props* props) {
  props->name = device_get_name(dev);
  props->description = device_get_description(dev);
  props->type = device_get_type(dev);
  device_get_memory(dev, &props->memory_free, &props->memory_total);
  props->caps = {/*async*/ false, /*host_buffer*/ false, /*buffer_from_host_ptr*/ false, /*events*/ false};
}

ggml_backend_buffer_type_t device_get_buffer_type(ggml_backend_dev_t dev) {
  static ggml_backend_buffer_type buft = {
      /* .iface = */ {buft_get_name, buft_alloc_buffer, buft_get_alignment,
                      /*get_max_size*/ nullptr, /*get_alloc_size*/ nullptr, /*is_host*/ nullptr},
      /* .device  = */ dev,
      /* .context = */ nullptr,
  };
  return &buft;
}

ggml_backend_t device_init_backend(ggml_backend_dev_t dev, const char* /*params*/) {
  auto* backend = new ggml_backend{
      /* .guid    = */ nullptr,
      /* .iface   = */ kBackendIface,
      /* .device  = */ dev,
      /* .context = */ nullptr,
  };
  return backend;
}

bool device_supports_op(ggml_backend_dev_t /*dev*/, const ggml_tensor* op) {
  // Keep in sync with backend_graph_compute. Returning the supported set lets
  // ggml's scheduler place ops correctly and fall back for the rest.
  switch (op->op) {
    case GGML_OP_MUL_MAT:
    case GGML_OP_NONE:
      return true;
    default:
      return false;
  }
}

bool device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
  return buft->device == dev;
}

constexpr ggml_backend_device_i kDeviceIface = {
    /* .get_name         = */ device_get_name,
    /* .get_description  = */ device_get_description,
    /* .get_memory       = */ device_get_memory,
    /* .get_type         = */ device_get_type,
    /* .get_props        = */ device_get_props,
    /* .init_backend     = */ device_init_backend,
    /* .get_buffer_type  = */ device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op      = */ device_supports_op,
    /* .supports_buft    = */ device_supports_buft,
    /* .offload_op       = */ nullptr,
    /* .event_new        = */ nullptr,
    /* .event_free       = */ nullptr,
    /* .event_synchronize= */ nullptr,
};

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------
const char* reg_get_name(ggml_backend_reg_t /*reg*/) { return myaccel::npu::kBackendName; }

size_t reg_get_device_count(ggml_backend_reg_t /*reg*/) {
  myaccel::npu::Initialize();
  return static_cast<size_t>(myaccel::npu::GetDeviceCount());
}

ggml_backend_dev_t reg_get_device(ggml_backend_reg_t reg, size_t index) {
  static std::vector<ggml_backend_device> devices;
  static std::vector<myaccel::npu::Device*> handles;
  if (devices.empty()) {
    const int count = myaccel::npu::GetDeviceCount();
    devices.resize(count);
    handles.resize(count);
    for (int i = 0; i < count; ++i) {
      handles[i] = myaccel::npu::OpenDevice(i);
      devices[i] = ggml_backend_device{/* .iface = */ kDeviceIface, /* .reg = */ reg,
                                       /* .context = */ handles[i]};
    }
  }
  return &devices[index];
}

constexpr ggml_backend_reg_i kRegIface = {
    /* .get_name         = */ reg_get_name,
    /* .get_device_count = */ reg_get_device_count,
    /* .get_device       = */ reg_get_device,
    /* .get_proc_address = */ nullptr,
};

}  // namespace

// Stable, link-time entry point. ggml-backend-reg.cpp calls this for an in-tree
// backend (register_backend(ggml_backend_myaccel_reg())).
extern "C" GGML_BACKEND_API ggml_backend_reg_t ggml_backend_myaccel_reg(void) {
  static ggml_backend_reg reg = {
      /* .api_version = */ GGML_BACKEND_API_VERSION,
      /* .iface       = */ kRegIface,
      /* .context     = */ nullptr,
  };
  return &reg;
}

// Dynamic-load entry point used by ggml_backend_load_all() (plugin path only).
// Confirm the exact expected symbol name for your ggml version and adjust.
extern "C" GGML_BACKEND_API ggml_backend_reg_t ggml_backend_init(void) {
  return ggml_backend_myaccel_reg();
}
