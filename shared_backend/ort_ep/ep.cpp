// Copyright (c) 2026.

#include "ep.h"

MyAccelEp::MyAccelEp(const std::string& name, ApiPtrs apis, myaccel::Device* device, const OrtLogger& logger)
    : OrtEp{}, ApiPtrs(apis), name_(name), device_(device), logger_(logger) {
  ort_version_supported = ORT_API_VERSION;  // compiled-against ORT version

  GetName = GetNameImpl;
  GetCapability = GetCapabilityImpl;
  Compile = CompileImpl;
  ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;
  Sync = SyncImpl;
  // All other OrtEp function pointers are optional and intentionally left null
  // until the corresponding feature is implemented (data layout, streams,
  // graph capture, profiler, kernel registry, ...). See ORT's example at
  // onnxruntime/test/autoep/library/example_plugin_ep for how to wire them.
}

MyAccelEp::~MyAccelEp() {
  if (device_ != nullptr) myaccel::CloseDevice(device_);
}

const char* ORT_API_CALL MyAccelEp::GetNameImpl(const OrtEp* this_ptr) noexcept {
  return static_cast<const MyAccelEp*>(this_ptr)->name_.c_str();
}

OrtStatus* ORT_API_CALL MyAccelEp::GetCapabilityImpl(OrtEp* /*this_ptr*/, const OrtGraph* /*graph*/,
                                                     OrtEpGraphSupportInfo* /*graph_support_info*/) noexcept {
  // TODO: walk the graph, and for each node your NPU supports call
  //   ep_api.EpGraphSupportInfo_AddNodesToFuse(...)  // for compiled subgraphs
  // Returning without adding anything means "I support no nodes" -> ORT keeps
  // everything on the CPU EP. This is the safe initial state.
  return nullptr;
}

OrtStatus* ORT_API_CALL MyAccelEp::CompileImpl(OrtEp* /*this_ptr*/, const OrtGraph** /*graphs*/,
                                               const OrtNode** /*fused_nodes*/, size_t /*count*/,
                                               OrtNodeComputeInfo** /*node_compute_infos*/,
                                               OrtNode** /*ep_context_nodes*/) noexcept {
  // Never reached while GetCapability fuses nothing. Once you fuse nodes, build
  // an OrtNodeComputeInfo per fused node whose Compute() reads inputs, calls
  // myaccel::MatMulF32 / your kernels, and writes outputs.
  return nullptr;
}

void ORT_API_CALL MyAccelEp::ReleaseNodeComputeInfosImpl(OrtEp* /*this_ptr*/,
                                                         OrtNodeComputeInfo** /*node_compute_infos*/,
                                                         size_t /*num*/) noexcept {
  // Free whatever CompileImpl allocated.
}

OrtStatus* ORT_API_CALL MyAccelEp::SyncImpl(OrtEp* this_ptr) noexcept {
  auto* ep = static_cast<MyAccelEp*>(this_ptr);
  myaccel::Synchronize(nullptr);  // sync the whole device for the stub
  (void)ep;
  return nullptr;
}
