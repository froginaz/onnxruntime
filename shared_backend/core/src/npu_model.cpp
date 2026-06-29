// Copyright (c) 2026.
// Reference / stub implementation of the NPU model-loading C ABI.
//
// It validates the contract (versioning, required fields, weight residency)
// and reads the blobs through every source kind, but does NOT actually compile
// to the NPU. Replace the marked sections with your NPU SW stack calls.

#include "myaccel/npu.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>  // header-only, vendored; core-internal use only

namespace {

thread_local std::string g_last_error;

void set_error(const char* msg) { g_last_error = msg ? msg : ""; }

// Optional per-load configuration is passed as a JSON string under the option
// key "options_json" (e.g. {"precision":"fp16","max_batch":8}). Parse + validate
// it here so the NPU compiler gets structured options. Returns false on bad JSON.
bool parse_options_json(uint32_t n_options, const char* const* keys,
                        const char* const* values, nlohmann::json& out) {
  out = nlohmann::json::object();
  for (uint32_t i = 0; i < n_options; ++i) {
    if (keys == nullptr || values == nullptr || keys[i] == nullptr) continue;
    if (std::strcmp(keys[i], "options_json") != 0) continue;
    out = nlohmann::json::parse(values[i] ? values[i] : "", /*cb*/ nullptr,
                                /*allow_exceptions*/ false);
    if (out.is_discarded()) return false;  // malformed JSON
  }
  return true;
}

// Materialize a blob into a contiguous host buffer regardless of source kind.
// A real stack would mmap FILE/FD and stream CALLBACK lazily; the stub copies
// so the rest of the loader is source-agnostic.
npu_status_t read_blob(const npu_blob_t& blob, std::vector<uint8_t>& out) {
  switch (blob.kind) {
    case NPU_SOURCE_MEMORY: {
      if (blob.u.memory.data == nullptr && blob.u.memory.size != 0) return NPU_ERR_INVALID_ARGUMENT;
      const auto* p = static_cast<const uint8_t*>(blob.u.memory.data);
      out.assign(p, p + blob.u.memory.size);
      return NPU_OK;
    }
    case NPU_SOURCE_CALLBACK: {
      const npu_read_callback_t& cb = blob.u.callback;
      if (cb.read == nullptr || cb.total_size == nullptr) return NPU_ERR_INVALID_ARGUMENT;
      const uint64_t total = cb.total_size(cb.user);
      out.resize(static_cast<size_t>(total));
      uint64_t off = 0;
      while (off < total) {
        int64_t n = cb.read(cb.user, off, out.data() + off, total - off);
        if (n <= 0) { set_error("weight read callback failed"); return NPU_ERR_IO; }
        off += static_cast<uint64_t>(n);
      }
      return NPU_OK;
    }
    case NPU_SOURCE_FILE:
    case NPU_SOURCE_FD:
      // TODO: mmap the path/handle (zero-copy when residency==REFERENCE_HOST).
      set_error("FILE/FD blob sources require the real NPU stack (stub does memory/callback only)");
      return NPU_ERR_IO;
    default:
      return NPU_ERR_INVALID_ARGUMENT;
  }
}

}  // namespace

struct npu_model {
  std::vector<uint8_t> nnc_bytes;
  std::vector<uint8_t> weight_bytes;      // empty when referencing host blob
  bool owns_weights = true;
  std::vector<npu_tensor_info_t> inputs;
  std::vector<npu_tensor_info_t> outputs;
  // TODO: real handle from the NPU SW stack (compiled plan + device buffers).
};

extern "C" {

const char* npu_status_str(npu_status_t status) {
  switch (status) {
    case NPU_OK: return "NPU_OK";
    case NPU_ERR_INVALID_ARGUMENT: return "NPU_ERR_INVALID_ARGUMENT";
    case NPU_ERR_VERSION_MISMATCH: return "NPU_ERR_VERSION_MISMATCH";
    case NPU_ERR_UNSUPPORTED_MODEL: return "NPU_ERR_UNSUPPORTED_MODEL";
    case NPU_ERR_WEIGHT_MISMATCH: return "NPU_ERR_WEIGHT_MISMATCH";
    case NPU_ERR_OUT_OF_MEMORY: return "NPU_ERR_OUT_OF_MEMORY";
    case NPU_ERR_IO: return "NPU_ERR_IO";
    case NPU_ERR_DEVICE: return "NPU_ERR_DEVICE";
    case NPU_ERR_NOT_IMPLEMENTED: return "NPU_ERR_NOT_IMPLEMENTED";
  }
  return "NPU_ERR_UNKNOWN";
}

const char* npu_last_error(void) { return g_last_error.c_str(); }

npu_status_t npu_model_compile(npu_device_t* /*device*/, const npu_blob_t* nnc,
                               uint32_t n_options, const char* const* keys,
                               const char* const* values, npu_model_t** out_model) {
  if (nnc == nullptr || out_model == nullptr) return NPU_ERR_INVALID_ARGUMENT;

  nlohmann::json options;
  if (!parse_options_json(n_options, keys, values, options)) {
    set_error("'options_json' option is not valid JSON");
    return NPU_ERR_INVALID_ARGUMENT;
  }
  // TODO: apply structured options (e.g. options.value("precision", "fp32"),
  // options.value("max_batch", 1)) to the NPU compiler.

  auto model = std::unique_ptr<npu_model>(new npu_model());
  if (npu_status_t st = read_blob(*nnc, model->nnc_bytes)) return st;

  // TODO: hand model->nnc_bytes to the NPU compiler; populate inputs/outputs
  // from the compiled plan. The stub reports an empty I/O signature.

  *out_model = model.release();
  return NPU_OK;
}

npu_status_t npu_model_bind_weights(npu_model_t* model, const npu_blob_t* weights,
                                    const npu_weight_manifest_t* /*manifest*/,
                                    npu_weight_residency_t residency) {
  if (model == nullptr || weights == nullptr) return NPU_ERR_INVALID_ARGUMENT;

  if (residency == NPU_WEIGHTS_REFERENCE_HOST && weights->kind == NPU_SOURCE_MEMORY) {
    // Zero-copy: keep referencing the caller's buffer, do not copy.
    model->owns_weights = false;
    // TODO: program the NPU to DMA directly from weights->u.memory.data.
    return NPU_OK;
  }

  if (npu_status_t st = read_blob(*weights, model->weight_bytes)) return st;
  model->owns_weights = true;
  // TODO: resolve manifest entries -> device allocations -> DMA weight regions.
  return NPU_OK;
}

npu_status_t npu_model_load(npu_device_t* device, const npu_model_load_info_t* info,
                            npu_model_t** out_model) {
  if (info == nullptr || out_model == nullptr) return NPU_ERR_INVALID_ARGUMENT;
  if (info->struct_size != sizeof(npu_model_load_info_t) || info->api_version != NPU_API_VERSION) {
    set_error("npu_model_load_info_t struct_size/api_version mismatch");
    return NPU_ERR_VERSION_MISMATCH;
  }

  npu_model_t* model = nullptr;
  if (npu_status_t st = npu_model_compile(device, &info->nnc, info->n_options,
                                          info->option_keys, info->option_values, &model)) {
    return st;
  }
  if (npu_status_t st = npu_model_bind_weights(model, &info->weights, info->manifest,
                                               info->residency)) {
    npu_model_free(model);
    return st;
  }
  *out_model = model;
  return NPU_OK;
}

void npu_model_free(npu_model_t* model) { delete model; }

int npu_model_owns_weights(const npu_model_t* model) {
  return (model != nullptr && model->owns_weights) ? 1 : 0;
}

uint32_t npu_model_num_inputs(const npu_model_t* model) {
  return model ? static_cast<uint32_t>(model->inputs.size()) : 0;
}
uint32_t npu_model_num_outputs(const npu_model_t* model) {
  return model ? static_cast<uint32_t>(model->outputs.size()) : 0;
}
npu_status_t npu_model_get_input(const npu_model_t* model, uint32_t index, npu_tensor_info_t* out) {
  if (model == nullptr || out == nullptr || index >= model->inputs.size()) return NPU_ERR_INVALID_ARGUMENT;
  *out = model->inputs[index];
  return NPU_OK;
}
npu_status_t npu_model_get_output(const npu_model_t* model, uint32_t index, npu_tensor_info_t* out) {
  if (model == nullptr || out == nullptr || index >= model->outputs.size()) return NPU_ERR_INVALID_ARGUMENT;
  *out = model->outputs[index];
  return NPU_OK;
}

}  // extern "C"
