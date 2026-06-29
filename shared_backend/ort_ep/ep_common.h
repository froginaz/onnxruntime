// Copyright (c) 2026.
// Small helpers shared by the onnxruntime plugin-EP adapter.

#ifndef MYACCEL_ORT_EP_COMMON_H_
#define MYACCEL_ORT_EP_COMMON_H_

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"  // pulls in onnxruntime_c_api.h + onnxruntime_ep_c_api.h
#undef ORT_API_MANUAL_INIT

// Bundle of API tables handed to us in CreateEpFactories so every object can
// reach ORT without a global. Mirrors the pattern in ORT's example_plugin_ep.
struct ApiPtrs {
  const OrtApi& ort_api;
  const OrtEpApi& ep_api;
};

#endif  // MYACCEL_ORT_EP_COMMON_H_
