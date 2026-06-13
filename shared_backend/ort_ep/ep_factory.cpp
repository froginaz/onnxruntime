// Copyright (c) 2026.

#include "ep_factory.h"

#include "ep.h"
#include "myaccel/npu_core.h"

const char* MyAccelEpFactory::myaccel_vendor_name() { return myaccel::kVendorName; }

MyAccelEpFactory::MyAccelEpFactory(const char* registration_name, ApiPtrs apis, const OrtLogger& default_logger)
    : OrtEpFactory{}, ApiPtrs(apis), ep_name_(registration_name), default_logger_(default_logger) {
  ort_version_supported = ORT_API_VERSION;

  GetName = GetNameImpl;
  GetVendor = GetVendorImpl;
  GetVendorId = GetVendorIdImpl;
  GetVersion = GetVersionImpl;
  GetSupportedDevices = GetSupportedDevicesImpl;
  CreateEp = CreateEpImpl;
  ReleaseEp = ReleaseEpImpl;
  // Allocator / data-transfer / stream / custom-op hooks are optional and left
  // null until GetCapability starts fusing nodes (which is when ORT needs to
  // move tensors to the device). Fill them in following ORT's example EP.

  myaccel::Initialize();
}

const char* ORT_API_CALL MyAccelEpFactory::GetNameImpl(const OrtEpFactory* this_ptr) noexcept {
  return static_cast<const MyAccelEpFactory*>(this_ptr)->ep_name_.c_str();
}
const char* ORT_API_CALL MyAccelEpFactory::GetVendorImpl(const OrtEpFactory* this_ptr) noexcept {
  return static_cast<const MyAccelEpFactory*>(this_ptr)->vendor_.c_str();
}
uint32_t ORT_API_CALL MyAccelEpFactory::GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept {
  return static_cast<const MyAccelEpFactory*>(this_ptr)->vendor_id_;
}
const char* ORT_API_CALL MyAccelEpFactory::GetVersionImpl(const OrtEpFactory* this_ptr) noexcept {
  return static_cast<const MyAccelEpFactory*>(this_ptr)->version_.c_str();
}

OrtStatus* ORT_API_CALL MyAccelEpFactory::GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                                                 const OrtHardwareDevice* const* devices,
                                                                 size_t num_devices,
                                                                 OrtEpDevice** ep_devices,
                                                                 size_t max_ep_devices,
                                                                 size_t* num_ep_devices) noexcept {
  auto* factory = static_cast<MyAccelEpFactory*>(this_ptr);
  const OrtApi& ort_api = factory->ort_api;
  const OrtEpApi& ep_api = factory->ep_api;

  size_t count = 0;
  for (size_t i = 0; i < num_devices && count < max_ep_devices; ++i) {
    const OrtHardwareDevice* hw = devices[i];
    // Claim NPU-class devices that report our vendor id. Adjust the predicate
    // to however your driver exposes the part (vendor id, device id, name).
    if (ort_api.HardwareDevice_Type(hw) != OrtHardwareDeviceType_NPU) continue;
    if (ort_api.HardwareDevice_VendorId(hw) != factory->vendor_id_) continue;

    OrtEpDevice* ep_device = nullptr;
    if (OrtStatus* st = ep_api.CreateEpDevice(factory, hw, /*metadata*/ nullptr,
                                              /*options*/ nullptr, &ep_device)) {
      return st;
    }
    ep_devices[count++] = ep_device;
  }
  *num_ep_devices = count;
  return nullptr;
}

OrtStatus* ORT_API_CALL MyAccelEpFactory::CreateEpImpl(OrtEpFactory* this_ptr,
                                                       const OrtHardwareDevice* const* /*devices*/,
                                                       const OrtKeyValuePairs* const* /*ep_metadata*/,
                                                       size_t /*num_devices*/,
                                                       const OrtSessionOptions* /*session_options*/,
                                                       const OrtLogger* logger,
                                                       OrtEp** ep) noexcept {
  auto* factory = static_cast<MyAccelEpFactory*>(this_ptr);

  // TODO: pick the device index from the matched OrtHardwareDevice. Stub uses 0.
  myaccel::Device* device = myaccel::OpenDevice(0);
  if (device == nullptr) {
    return factory->ort_api.CreateStatus(ORT_EP_FAIL, "MyAccel: failed to open NPU device");
  }

  *ep = new MyAccelEp(factory->ep_name_, ApiPtrs{factory->ort_api, factory->ep_api}, device, *logger);
  return nullptr;
}

void ORT_API_CALL MyAccelEpFactory::ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep) noexcept {
  delete static_cast<MyAccelEp*>(ep);
}
