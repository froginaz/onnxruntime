/* Copyright (c) 2026.
 *
 * MyAccel NPU model-loading API — the stable C ABI boundary between a host
 * runtime (onnxruntime, llama.cpp, ...) and the NPU software stack.
 *
 * Contract
 * --------
 * The NPU consumes models in its OWN container format:
 *
 *   model.nnc   - topology / compiled execution plan (small)
 *   weight.bin  - the packed weight blob (potentially many GB)
 *
 * Each runtime is responsible for extracting/converting its container
 * (.onnx, .gguf, ...) into {model.nnc, weight.bin} and handing them to this
 * API. THIS API KNOWS NOTHING ABOUT onnx OR gguf. That separation is the whole
 * point: one NPU stack, many front-ends.
 *
 *   .onnx  --[ORT EP adapter]------>  model.nnc + weight.bin --\
 *                                                               >-- npu_model_load()
 *   .gguf  --[llama.cpp adapter]--->  model.nnc + weight.bin --/
 *
 * ABI stability
 * -------------
 * Pure C, opaque handles, status codes. Input descriptor structs carry
 * `struct_size` + `api_version` so the stack can accept binaries built against
 * older headers. Never reorder/remove existing struct fields; only append.
 */

#ifndef MYACCEL_NPU_MODEL_H_
#define MYACCEL_NPU_MODEL_H_

#include <stdint.h>
#include <stddef.h>

#if defined(_WIN32)
#if defined(MYACCEL_CORE_BUILD_DLL)
#define NPU_API __declspec(dllexport)
#elif defined(MYACCEL_CORE_USE_DLL)
#define NPU_API __declspec(dllimport)
#else
#define NPU_API
#endif
#else
#define NPU_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define NPU_API_VERSION 1
#define NPU_MAX_DIMS 8

/* --------------------------------------------------------------------------
 * Status / errors
 * ------------------------------------------------------------------------ */
typedef enum npu_status {
  NPU_OK = 0,
  NPU_ERR_INVALID_ARGUMENT,
  NPU_ERR_VERSION_MISMATCH,    /* api_version / struct_size not understood   */
  NPU_ERR_UNSUPPORTED_MODEL,   /* nnc references ops/dtypes the NPU lacks    */
  NPU_ERR_WEIGHT_MISMATCH,     /* manifest/weight.bin inconsistent with nnc  */
  NPU_ERR_OUT_OF_MEMORY,
  NPU_ERR_IO,                  /* file/fd/callback read failure              */
  NPU_ERR_DEVICE,              /* driver / hardware error                    */
} npu_status_t;

/* Static string for a status code (never NULL). */
NPU_API const char* npu_status_str(npu_status_t status);

/* Thread-local, human-readable detail for the most recent failure on the
 * calling thread. Valid until the next API call on the same thread. */
NPU_API const char* npu_last_error(void);

/* --------------------------------------------------------------------------
 * Opaque handles
 * ------------------------------------------------------------------------ */
typedef struct npu_device npu_device_t;  /* a NPU device (from device mgmt)  */
typedef struct npu_model  npu_model_t;   /* a loaded / compiled model         */

/* --------------------------------------------------------------------------
 * Data types
 * ------------------------------------------------------------------------ */
typedef enum npu_dtype {
  NPU_DTYPE_UNKNOWN = 0,
  NPU_DTYPE_F32,
  NPU_DTYPE_F16,
  NPU_DTYPE_BF16,
  NPU_DTYPE_I64,
  NPU_DTYPE_I32,
  NPU_DTYPE_I8,
  NPU_DTYPE_U8,
  /* Block-quantized weight formats (gguf-style); extend as the NPU supports.
   * The NPU stack interprets the raw bytes per dtype, so dequant stays on the
   * accelerator and weight.bin can be passed through unchanged. */
  NPU_DTYPE_Q4_0,
  NPU_DTYPE_Q4_K,
  NPU_DTYPE_Q8_0,
} npu_dtype_t;

/* --------------------------------------------------------------------------
 * Blob source — uniform way to hand both `model.nnc` and `weight.bin` to the
 * NPU regardless of where the runtime keeps the bytes.
 *
 *   MEMORY   : host pointer the runtime already holds (ORT in-memory weights)
 *   FILE     : a path the NPU opens/mmaps itself (external-data / .gguf file)
 *   FD       : an already-open OS handle for zero-copy mmap (gguf mmap reuse)
 *   CALLBACK : pull-based read for streaming / encrypted / sharded storage
 * ------------------------------------------------------------------------ */
typedef enum npu_source_kind {
  NPU_SOURCE_MEMORY = 0,
  NPU_SOURCE_FILE,
  NPU_SOURCE_FD,
  NPU_SOURCE_CALLBACK,
} npu_source_kind_t;

/* Pull-based reader. `read` copies `size` bytes at `offset` into `dst` and
 * returns bytes read, or a negative value on error. The NPU stack calls this
 * only for the regions it actually needs, in whatever order it prefers. */
typedef struct npu_read_callback {
  int64_t  (*read)(void* user, uint64_t offset, void* dst, uint64_t size);
  uint64_t (*total_size)(void* user);
  void*    user;
} npu_read_callback_t;

typedef struct npu_blob {
  npu_source_kind_t kind;
  union {
    struct { const void* data; uint64_t size; }                 memory;
    struct { const char* path; uint64_t offset; uint64_t size; } file;   /* size==0 => to EOF */
    struct { intptr_t handle;  uint64_t offset; uint64_t size; } fd;
    npu_read_callback_t                                          callback;
  } u;
} npu_blob_t;

/* --------------------------------------------------------------------------
 * Optional weight manifest.
 *
 * Omit (set manifest=NULL) when the nnc already encodes byte offsets into
 * weight.bin — then the NPU resolves weights by (base + offset) directly.
 *
 * Provide it when the nnc references weights BY NAME, when weights come from a
 * CALLBACK/sharded source, or to let the NPU validate dtype/shape against the
 * compiled graph (recommended for safety).
 * ------------------------------------------------------------------------ */
typedef struct npu_weight_entry {
  const char*    name;     /* tensor name as referenced by the nnc      */
  uint64_t       offset;   /* byte offset within the weight blob        */
  uint64_t       size;     /* byte size of this tensor's data           */
  npu_dtype_t    dtype;
  uint32_t       n_dims;
  const int64_t* dims;     /* shape, n_dims entries (row-major)         */
} npu_weight_entry_t;

typedef struct npu_weight_manifest {
  uint32_t                  count;
  const npu_weight_entry_t* entries;
} npu_weight_manifest_t;

/* --------------------------------------------------------------------------
 * Weight residency — who keeps the bytes alive after load.
 * ------------------------------------------------------------------------ */
typedef enum npu_weight_residency {
  /* NPU DMAs weights into device memory during load; the host weight blob may
   * be freed/unmapped as soon as npu_model_load returns. Default. */
  NPU_WEIGHTS_COPY_TO_DEVICE = 0,
  /* NPU references the host blob in place (e.g. mmap'd weight.bin). The blob
   * MUST stay valid for the lifetime of the model. Enables zero-copy. */
  NPU_WEIGHTS_REFERENCE_HOST,
} npu_weight_residency_t;

/* --------------------------------------------------------------------------
 * Load descriptor (versioned).
 * ------------------------------------------------------------------------ */
typedef struct npu_model_load_info {
  uint32_t struct_size;   /* set to sizeof(npu_model_load_info_t)       */
  uint32_t api_version;   /* set to NPU_API_VERSION                     */

  npu_blob_t nnc;         /* the model.nnc container                    */
  npu_blob_t weights;     /* the weight.bin blob                        */
  const npu_weight_manifest_t* manifest;  /* optional, may be NULL      */

  npu_weight_residency_t residency;

  /* Free-form compile/runtime options (precision mode, max batch, tiling,
   * power profile, ...). Parallel arrays of length n_options. */
  uint32_t           n_options;
  const char* const* option_keys;
  const char* const* option_values;
} npu_model_load_info_t;

/* --------------------------------------------------------------------------
 * One-shot load: compile nnc + bind weights in a single call. This is the
 * primary entry point both runtimes use.
 * ------------------------------------------------------------------------ */
NPU_API npu_status_t npu_model_load(npu_device_t* device,
                                    const npu_model_load_info_t* info,
                                    npu_model_t** out_model);

NPU_API void npu_model_free(npu_model_t* model);

/* Returns non-zero if the model holds its own device-side copy of the weights
 * (i.e. residency resolved to COPY_TO_DEVICE) and the host blob is now free to
 * release; zero if the model still references the host weight blob. */
NPU_API int npu_model_owns_weights(const npu_model_t* model);

/* --------------------------------------------------------------------------
 * Optional two-phase path: compile the graph once, then (re)bind weights.
 * Use for weight-sharing (one nnc, several weight sets) or to overlap compile
 * with weight streaming. npu_model_load == compile + bind_weights.
 * ------------------------------------------------------------------------ */
NPU_API npu_status_t npu_model_compile(npu_device_t* device,
                                       const npu_blob_t* nnc,
                                       uint32_t n_options,
                                       const char* const* option_keys,
                                       const char* const* option_values,
                                       npu_model_t** out_model);

NPU_API npu_status_t npu_model_bind_weights(npu_model_t* model,
                                            const npu_blob_t* weights,
                                            const npu_weight_manifest_t* manifest,
                                            npu_weight_residency_t residency);

/* --------------------------------------------------------------------------
 * Introspection — let the runtime bind activation I/O after load.
 * ------------------------------------------------------------------------ */
typedef struct npu_tensor_info {
  const char* name;
  npu_dtype_t dtype;
  uint32_t    n_dims;
  int64_t     dims[NPU_MAX_DIMS];   /* -1 marks a dynamic dimension */
} npu_tensor_info_t;

NPU_API uint32_t npu_model_num_inputs(const npu_model_t* model);
NPU_API uint32_t npu_model_num_outputs(const npu_model_t* model);
NPU_API npu_status_t npu_model_get_input(const npu_model_t* model, uint32_t index,
                                         npu_tensor_info_t* out_info);
NPU_API npu_status_t npu_model_get_output(const npu_model_t* model, uint32_t index,
                                          npu_tensor_info_t* out_info);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* MYACCEL_NPU_MODEL_H_ */
