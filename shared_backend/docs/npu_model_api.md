# NPU model-loading API (`myaccel/npu_model.h`)

The stable C ABI both runtimes call to load a model onto the NPU. The NPU only
ever sees its own container — `model.nnc` (topology) + `weight.bin` (weights).
Converting `.onnx` / `.gguf` into that pair is each runtime adapter's job.

```
.onnx  ─[ORT EP adapter]──────▶ model.nnc + weight.bin ─┐
                                                         ├─▶ npu_model_load()
.gguf  ─[llama.cpp adapter]───▶ model.nnc + weight.bin ─┘
```

## Why a `npu_blob_t` for both nnc and weights

The two runtimes hold weights very differently, and one descriptor absorbs all
of it so the NPU stack has a single code path:

| Runtime source | `npu_source_kind_t` | `residency` | Result |
|---|---|---|---|
| `.gguf` is `mmap`'d by llama.cpp | `NPU_SOURCE_FD` / `MEMORY` | `REFERENCE_HOST` | zero-copy, NPU DMAs from the mapping |
| ORT in-memory initializers | `NPU_SOURCE_MEMORY` | `COPY_TO_DEVICE` | NPU copies to device, host buffer freeable |
| ORT external-data file | `NPU_SOURCE_FILE` | `COPY_TO_DEVICE` | NPU opens/mmaps + copies |
| Encrypted / sharded store | `NPU_SOURCE_CALLBACK` | either | NPU pulls only the regions it needs |

## llama.cpp adapter (zero-copy from a mmap'd gguf)

```c
// gguf is already mmap'd by llama.cpp; the runtime has extracted/converted the
// NPU's nnc and knows where the weight region sits in the mapping.
npu_model_load_info_t info = {0};
info.struct_size = sizeof(info);
info.api_version = NPU_API_VERSION;

info.nnc.kind            = NPU_SOURCE_MEMORY;
info.nnc.u.memory.data   = nnc_ptr;     // converted topology
info.nnc.u.memory.size   = nnc_size;

info.weights.kind          = NPU_SOURCE_MEMORY;
info.weights.u.memory.data = mmap_weight_base;   // points into the gguf mapping
info.weights.u.memory.size = weight_size;
info.residency             = NPU_WEIGHTS_REFERENCE_HOST;  // keep the mapping alive

npu_model_t* model = NULL;
npu_status_t st = npu_model_load(device, &info, &model);
if (st != NPU_OK) { fprintf(stderr, "%s: %s\n", npu_status_str(st), npu_last_error()); }
```

## onnxruntime EP adapter (copy in-memory initializers, named manifest)

```c
// Build a manifest so the NPU validates each initializer against the compiled
// graph and resolves weights by name.
npu_weight_entry_t entries[N];
for (size_t i = 0; i < N; ++i) {
    entries[i].name   = init_name[i];
    entries[i].offset = init_offset[i];   // offset within weight.bin you packed
    entries[i].size   = init_size[i];
    entries[i].dtype  = to_npu_dtype(ort_elem_type[i]);
    entries[i].n_dims = init_rank[i];
    entries[i].dims   = init_dims[i];
}
npu_weight_manifest_t manifest = { N, entries };

npu_model_load_info_t info = {0};
info.struct_size = sizeof(info);
info.api_version = NPU_API_VERSION;
info.nnc.kind            = NPU_SOURCE_MEMORY;
info.nnc.u.memory.data   = nnc_ptr;
info.nnc.u.memory.size   = nnc_size;
info.weights.kind          = NPU_SOURCE_MEMORY;
info.weights.u.memory.data = weight_bin_ptr;
info.weights.u.memory.size = weight_bin_size;
info.manifest              = &manifest;
info.residency             = NPU_WEIGHTS_COPY_TO_DEVICE;  // ORT may free initializers after

npu_model_t* model = NULL;
npu_model_load(device, &info, &model);

if (npu_model_owns_weights(model)) { /* safe to release host weight buffers */ }
```

Call this from `OrtEp::Compile` once per fused subgraph; cache the returned
`npu_model_t*` in the `OrtNodeComputeInfo` so each inference run reuses it.

## Lifetime rules

- `COPY_TO_DEVICE`: after `npu_model_load` returns and `npu_model_owns_weights()`
  is non-zero, the host weight blob can be freed/unmapped.
- `REFERENCE_HOST`: the blob (e.g. the gguf mapping) **must outlive** the model.
- Always pair every successful load with `npu_model_free`.

## Advanced: weight sharing / streamed weights

Split the one-shot call when one compiled graph is reused with several weight
sets, or to overlap compile with weight transfer:

```c
npu_model_compile(device, &nnc_blob, 0, NULL, NULL, &model);   // topology only
npu_model_bind_weights(model, &weights_blob, &manifest, NPU_WEIGHTS_COPY_TO_DEVICE);
```
