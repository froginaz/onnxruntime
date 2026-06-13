// Copyright (c) 2026.
// OrtEpFactory: the object ORT gets from CreateEpFactories. It advertises which
// hardware devices this EP can drive (GetSupportedDevices) and creates one
// MyAccelEp per session (CreateEp).

#ifndef MYACCEL_ORT_EP_EP_FACTORY_H_
#define MYACCEL_ORT_EP_EP_FACTORY_H_

#include <string>

#include "ep_common.h"

class MyAccelEpFactory : public OrtEpFactory, public ApiPtrs {
 public:
  MyAccelEpFactory(const char* registration_name, ApiPtrs apis, const OrtLogger& default_logger);

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory* this_ptr) noexcept;
  static const char* ORT_API_CALL GetVendorImpl(const OrtEpFactory* this_ptr) noexcept;
  static uint32_t ORT_API_CALL GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept;
  static const char* ORT_API_CALL GetVersionImpl(const OrtEpFactory* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                                         const OrtHardwareDevice* const* devices,
                                                         size_t num_devices,
                                                         OrtEpDevice** ep_devices,
                                                         size_t max_ep_devices,
                                                         size_t* num_ep_devices) noexcept;

  static OrtStatus* ORT_API_CALL CreateEpImpl(OrtEpFactory* this_ptr,
                                              const OrtHardwareDevice* const* devices,
                                              const OrtKeyValuePairs* const* ep_metadata,
                                              size_t num_devices,
                                              const OrtSessionOptions* session_options,
                                              const OrtLogger* logger,
                                              OrtEp** ep) noexcept;

  static void ORT_API_CALL ReleaseEpImpl(OrtEpFactory* this_ptr, OrtEp* ep) noexcept;

  const std::string ep_name_;
  const std::string vendor_{myaccel_vendor_name()};
  const uint32_t vendor_id_{0x1ACC};
  const std::string version_{"0.1.0"};
  const OrtLogger& default_logger_;

  static const char* myaccel_vendor_name();
};

#endif  // MYACCEL_ORT_EP_EP_FACTORY_H_
