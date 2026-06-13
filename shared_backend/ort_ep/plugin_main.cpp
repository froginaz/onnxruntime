// Copyright (c) 2026.
// The two C symbols ORT looks up after RegisterExecutionProviderLibrary().
// These are the ENTIRE public surface of the plugin DLL; everything else is
// hidden via exports.def / the version script.

#include "ep_factory.h"
#include "myaccel/npu_core.h"

#ifdef __APPLE__
#define EXPORT_SYMBOL __attribute__((visibility("default")))
#else
#define EXPORT_SYMBOL
#endif

extern "C" {

EXPORT_SYMBOL OrtStatus* CreateEpFactories(const char* registration_name, const OrtApiBase* ort_api_base,
                                           const OrtLogger* default_logger,
                                           OrtEpFactory** factories, size_t max_factories, size_t* num_factories) {
  const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);
  const OrtEpApi* ep_api = ort_api->GetEpApi();
  Ort::InitApi(ort_api);  // enable the C++ wrappers inside the adapter

  if (max_factories < 1) {
    return ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                 "MyAccel: need room for at least one EP factory");
  }

  factories[0] = new MyAccelEpFactory(registration_name, ApiPtrs{*ort_api, *ep_api}, *default_logger);
  *num_factories = 1;
  return nullptr;
}

EXPORT_SYMBOL OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  delete static_cast<MyAccelEpFactory*>(factory);
  myaccel::Shutdown();
  return nullptr;
}

}  // extern "C"
