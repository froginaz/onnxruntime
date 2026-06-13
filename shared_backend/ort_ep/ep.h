// Copyright (c) 2026.
// The OrtEp instance: one per inference session. It answers "which nodes can I
// run?" (GetCapability) and turns the accepted subgraphs into callable compute
// functions (Compile). This scaffold reports no supported nodes yet, which is a
// valid, crash-free starting point; grow GetCapability/Compile to offload ops
// to the MyAccel NPU core.

#ifndef MYACCEL_ORT_EP_EP_H_
#define MYACCEL_ORT_EP_EP_H_

#include <string>

#include "ep_common.h"
#include "myaccel/npu_core.h"

class ExampleEpFactory;

class MyAccelEp : public OrtEp, public ApiPtrs {
 public:
  MyAccelEp(const std::string& name, ApiPtrs apis, myaccel::Device* device, const OrtLogger& logger);
  ~MyAccelEp();

  const std::string& name() const { return name_; }

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEp* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* graph,
                                                   OrtEpGraphSupportInfo* graph_support_info) noexcept;

  static OrtStatus* ORT_API_CALL CompileImpl(OrtEp* this_ptr, const OrtGraph** graphs,
                                             const OrtNode** fused_nodes, size_t count,
                                             OrtNodeComputeInfo** node_compute_infos,
                                             OrtNode** ep_context_nodes) noexcept;

  static void ORT_API_CALL ReleaseNodeComputeInfosImpl(OrtEp* this_ptr,
                                                       OrtNodeComputeInfo** node_compute_infos,
                                                       size_t num_node_compute_infos) noexcept;

  static OrtStatus* ORT_API_CALL SyncImpl(OrtEp* this_ptr) noexcept;

  std::string name_;
  myaccel::Device* device_ = nullptr;
  const OrtLogger& logger_;
};

#endif  // MYACCEL_ORT_EP_EP_H_
