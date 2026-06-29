# onnxruntime plugin-EP adapter -> myaccel_ort_ep.dll
#
# Requires ORT's public C/C++ headers. Point ORT_HOME at either an installed
# onnxruntime package (has include/ + lib/) or a source checkout.

add_library(myaccel_ort_ep SHARED
    plugin_main.cpp
    ep_factory.cpp
    ep.cpp)

target_link_libraries(myaccel_ort_ep PRIVATE myaccel_core)

if(NOT DEFINED ORT_HOME)
    message(FATAL_ERROR "Set -DORT_HOME=<onnxruntime install or source root> to build the ORT EP")
endif()

# Header search: support both an installed package layout and a source tree.
target_include_directories(myaccel_ort_ep PRIVATE
    "${ORT_HOME}/include"
    "${ORT_HOME}/include/onnxruntime"
    "${ORT_HOME}/include/onnxruntime/core/session")

# Restrict the exported surface to the two plugin entry points.
if(MSVC)
    set_property(TARGET myaccel_ort_ep APPEND_STRING PROPERTY
                 LINK_FLAGS " /DEF:\"${CMAKE_CURRENT_SOURCE_DIR}/exports.def\"")
elseif(APPLE)
    set_property(TARGET myaccel_ort_ep APPEND_STRING PROPERTY
                 LINK_FLAGS " -Wl,-exported_symbol,_CreateEpFactories -Wl,-exported_symbol,_ReleaseEpFactory")
else()
    target_link_options(myaccel_ort_ep PRIVATE
                        "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.lds")
endif()
