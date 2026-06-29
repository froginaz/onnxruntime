/* Copyright (c) 2026.
 *
 * MyAccel NPU — unified public C ABI.
 *
 * One stable C ABI for the whole boundary between a host runtime
 * (onnxruntime, llama.cpp, ExecuTorch, ...) and the NPU software stack:
 *   - identity / lifetime / device enumeration
 *   - memory (alloc / free / copy) and streams
 *   - kernels
 *   - model loading (model.nnc + weight.bin)
 *
 * Why pure C: language-neutral (any FFI), ABI-stable across compilers/STL
 * versions, opaque handles + struct_size/api_version negotiation. An optional
 * header-only C++ wrapper (npu.hpp) adds ergonomics without touching the ABI.
 * See docs/design.md and docs/examples/abi/.
 */

#ifndef MYACCEL_NPU_H_
#define MYACCEL_NPU_H_

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
#define NPU_MAX_DIMS    8

/* Stable identity reported to each framework. */
#define NPU_VENDOR_ID       0x1ACCu        /* TODO: your PCI/SoC vendor id */
#define NPU_VENDOR_NAME     "MyAccel"
#define NPU_BACKEND_NAME    "myaccel_npu"
#define NPU_VERSION_STRING  "0.1.0"

/* Same strings as functions, for non-C/C++ FFI consumers. */
NPU_API const char* npu_vendor_name(void);
NPU_API const char* npu_backend_name(void);
NPU_API const char* npu_version_string(void);

/* ==========================================================================
 * Status / errors
 * ======================================================================== */
typedef enum npu_status {
  NPU_OK = 0,
  NPU_ERR_INVALID_ARGUMENT,
  NPU_ERR_VERSION_MISMATCH,    /* api_version / struct_size not understood   */
  NPU_ERR_UNSUPPORTED_MODEL,   /* nnc references ops/dtypes the NPU lacks    */
  NPU_ERR_WEIGHT_MISMATCH,     /* manifest/weight.bin inconsistent with nnc  */
  NPU_ERR_OUT_OF_MEMORY,
  NPU_ERR_IO,                  /* file/fd/callback read failure              */
  NPU_ERR_DEVICE,              /* driver / hardware error                    */
  NPU_ERR_NOT_IMPLEMENTED,
} npu_status_t;

NPU_API const char* npu_status_str(npu_status_t status);  /* never NULL */
NPU_API const char* npu_last_error(void);                 /* thread-local detail */

/* ==========================================================================
 * Opaque handles (owned by the core; never dereferenced by callers)
 * ======================================================================== */
typedef struct npu_device npu_device_t;  /* one physical NPU                 */
typedef struct npu_buffer npu_buffer_t;  /* a device allocation              */
typedef struct npu_stream npu_stream_t;  /* an async execution queue         */
typedef struct npu_model  npu_model_t;   /* a loaded / compiled model        */

/* ==========================================================================
 * Device — identity, enumeration, lifetime
 * ======================================================================== */
typedef struct npu_device_info {
  int32_t  device_id;
  char     name[128];
  uint64_t total_memory;
  uint32_t compute_units;
} npu_device_info_t;

/* Lifetime is reference-counted: N inits need N shutdowns. */
NPU_API npu_status_t npu_initialize(void);
NPU_API void         npu_shutdown(void);

NPU_API int32_t      npu_get_device_count(void);
NPU_API npu_status_t npu_get_device_info(int32_t device_id, npu_device_info_t* out_info);
NPU_API npu_device_t* npu_open_device(int32_t device_id);   /* NULL on bad id */
NPU_API void         npu_close_device(npu_device_t* device);

/* ==========================================================================
 * Memory
 * ======================================================================== */
typedef enum npu_copy_kind {
  NPU_COPY_HOST_TO_DEVICE = 0,
  NPU_COPY_DEVICE_TO_HOST,
  NPU_COPY_DEVICE_TO_DEVICE,
} npu_copy_kind_t;

NPU_API npu_buffer_t* npu_alloc(npu_device_t* device, size_t bytes);
NPU_API void          npu_free(npu_buffer_t* buffer);
NPU_API void*         npu_device_ptr(npu_buffer_t* buffer);  /* raw device pointer */
NPU_API npu_status_t  npu_copy(npu_device_t* device, void* dst, const void* src,
                               size_t bytes, npu_copy_kind_t kind);

/* ==========================================================================
 * Streams (optional async)
 * ======================================================================== */
NPU_API npu_stream_t* npu_create_stream(npu_device_t* device);
NPU_API void          npu_destroy_stream(npu_stream_t* stream);
NPU_API npu_status_t  npu_synchronize(npu_stream_t* stream);  /* NULL => whole device */

/* ==========================================================================
 * Kernels (grow as the NPU gains ops). out[M,N] = a[M,K] * b[K,N] (fp32 ref)
 * ======================================================================== */
NPU_API npu_status_t npu_matmul_f32(npu_device_t* device, npu_stream_t* stream,
                                    const void* a, const void* b, void* out,
                                    int64_t m, int64_t k, int64_t n);

/* ==========================================================================
 * Data types (for model I/O and weights)
 * ======================================================================== */
typedef enum npu_dtype {
  NPU_DTYPE_UNKNOWN = 0,
  NPU_DTYPE_F32, NPU_DTYPE_F16, NPU_DTYPE_BF16,
  NPU_DTYPE_I64, NPU_DTYPE_I32, NPU_DTYPE_I8, NPU_DTYPE_U8,
  /* block-quantized weight formats (gguf-style); extend as supported */
  NPU_DTYPE_Q4_0, NPU_DTYPE_Q4_K, NPU_DTYPE_Q8_0,
} npu_dtype_t;

/* ==========================================================================
 * Model loading: blob source (memory/file/fd/callback) + residency
 * ======================================================================== */
typedef enum npu_source_kind {
  NPU_SOURCE_MEMORY = 0, NPU_SOURCE_FILE, NPU_SOURCE_FD, NPU_SOURCE_CALLBACK,
} npu_source_kind_t;

typedef struct npu_read_callback {
  int64_t  (*read)(void* user, uint64_t offset, void* dst, uint64_t size);
  uint64_t (*total_size)(void* user);
  void*    user;
} npu_read_callback_t;

typedef struct npu_blob {
  npu_source_kind_t kind;
  union {
    struct { const void* data; uint64_t size; }                 memory;
    struct { const char* path; uint64_t offset; uint64_t size; } file;   /* size==0 => EOF */
    struct { intptr_t handle;  uint64_t offset; uint64_t size; } fd;
    npu_read_callback_t                                          callback;
  } u;
} npu_blob_t;

typedef struct npu_weight_entry {
  const char*    name;
  uint64_t       offset;
  uint64_t       size;
  npu_dtype_t    dtype;
  uint32_t       n_dims;
  const int64_t* dims;
} npu_weight_entry_t;

typedef struct npu_weight_manifest {
  uint32_t                  count;
  const npu_weight_entry_t* entries;
} npu_weight_manifest_t;

typedef enum npu_weight_residency {
  NPU_WEIGHTS_COPY_TO_DEVICE = 0,   /* NPU copies to device; host blob freeable */
  NPU_WEIGHTS_REFERENCE_HOST,       /* NPU references host blob; must outlive model */
} npu_weight_residency_t;

typedef struct npu_model_load_info {
  uint32_t struct_size;   /* = sizeof(npu_model_load_info_t) */
  uint32_t api_version;   /* = NPU_API_VERSION               */

  npu_blob_t nnc;
  npu_blob_t weights;
  const npu_weight_manifest_t* manifest;  /* optional, may be NULL */

  npu_weight_residency_t residency;

  uint32_t           n_options;
  const char* const* option_keys;
  const char* const* option_values;
} npu_model_load_info_t;

NPU_API npu_status_t npu_model_load(npu_device_t* device,
                                    const npu_model_load_info_t* info,
                                    npu_model_t** out_model);
NPU_API void         npu_model_free(npu_model_t* model);
NPU_API int          npu_model_owns_weights(const npu_model_t* model);

/* Optional two-phase path (weight sharing / streamed weights). */
NPU_API npu_status_t npu_model_compile(npu_device_t* device, const npu_blob_t* nnc,
                                       uint32_t n_options, const char* const* option_keys,
                                       const char* const* option_values,
                                       npu_model_t** out_model);
NPU_API npu_status_t npu_model_bind_weights(npu_model_t* model, const npu_blob_t* weights,
                                            const npu_weight_manifest_t* manifest,
                                            npu_weight_residency_t residency);

/* Introspection — bind activation I/O after load. */
typedef struct npu_tensor_info {
  const char* name;
  npu_dtype_t dtype;
  uint32_t    n_dims;
  int64_t     dims[NPU_MAX_DIMS];   /* -1 marks a dynamic dimension */
} npu_tensor_info_t;

NPU_API uint32_t     npu_model_num_inputs(const npu_model_t* model);
NPU_API uint32_t     npu_model_num_outputs(const npu_model_t* model);
NPU_API npu_status_t npu_model_get_input(const npu_model_t* model, uint32_t index,
                                         npu_tensor_info_t* out_info);
NPU_API npu_status_t npu_model_get_output(const npu_model_t* model, uint32_t index,
                                          npu_tensor_info_t* out_info);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* MYACCEL_NPU_H_ */
