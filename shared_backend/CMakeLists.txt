# MyAccel shared HW-accelerator backend for onnxruntime + llama.cpp.
#
#   onnxruntime  --(plugin EP ABI)----> myaccel_ort_ep.dll  --+
#   llama.cpp    --(ggml backend ABI)-> ggml-myaccel.dll    --+--> myaccel_core (NPU SDK)
#   ExecuTorch   --(BackendInterface)-> myaccel_backend.dll --+
#
# Configure examples (Windows, x64 Native Tools prompt):
#   cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DORT_HOME=C:/ort -DLLAMA_CPP_DIR=C:/llama.cpp
#   cmake --build build --config Release
#
# Build only one adapter:
#   cmake -S . -B build -DMYACCEL_BUILD_GGML=OFF -DORT_HOME=C:/ort
#
# Separate the core into its own DLL (myaccel_core.dll), imported by both
# adapters instead of static-linked into each:
#   cmake -S . -B build -DMYACCEL_CORE_SHARED=ON -DORT_HOME=C:/ort -DLLAMA_CPP_DIR=C:/llama.cpp
#   => myaccel_core.dll + ggml-myaccel.dll + myaccel_ort_ep.dll
#      (ship myaccel_core.dll next to the adapter DLLs / the host exe)

cmake_minimum_required(VERSION 3.26)
project(myaccel_shared_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(MYACCEL_BUILD_ORT_EP "Build the onnxruntime plugin-EP adapter" ON)
option(MYACCEL_BUILD_GGML   "Build the llama.cpp ggml backend adapter" ON)
option(MYACCEL_BUILD_EXECUTORCH "Build the ExecuTorch backend adapter" OFF)
option(MYACCEL_CORE_SHARED  "Build the NPU core as a shared library"   OFF)

# --- shared, framework-agnostic core --------------------------------------
set(MYACCEL_CORE_SOURCES
    core/src/npu_core.cpp     # internal: device / memory / kernels (stub SDK)
    core/src/npu_api.cpp      # public façade myaccel::npu (npu_api.h)
    core/src/npu_model.cpp)   # model-loading C ABI (npu_model.h)

if(MYACCEL_CORE_SHARED)
    add_library(myaccel_core SHARED ${MYACCEL_CORE_SOURCES})
    # MYACCEL_API / NPU_API -> dllexport while building the core, dllimport for
    # consumers (the adapters), so the symbols cross the DLL boundary cleanly.
    target_compile_definitions(myaccel_core PRIVATE MYACCEL_CORE_BUILD_DLL)
    target_compile_definitions(myaccel_core PUBLIC MYACCEL_CORE_USE_DLL)
    # Export only the symbols we mark (MYACCEL_API / NPU_API). On Windows that is
    # automatic (dllexport); on Linux, hidden visibility keeps vendored header-only
    # deps like nlohmann/json out of the DLL's symbol table.
    set_target_properties(myaccel_core PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS OFF
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON)
    message(STATUS "myaccel_core: SHARED -> myaccel_core.dll (ship it beside the adapters)")
else()
    add_library(myaccel_core STATIC ${MYACCEL_CORE_SOURCES})
    message(STATUS "myaccel_core: STATIC -> linked into each adapter DLL")
endif()

# Optional: rename the produced library file (stem). The CMake target stays
# `myaccel_core`, only the output file changes — e.g. set to "myaccel_npu" to get
# myaccel_npu.dll / myaccel_npu.lib for the in-tree llama.cpp integration.
set(MYACCEL_CORE_OUTPUT_NAME "myaccel_core" CACHE STRING "Output file name (stem) of the NPU core library")
set_target_properties(myaccel_core PROPERTIES OUTPUT_NAME "${MYACCEL_CORE_OUTPUT_NAME}")
# PUBLIC: the façade (npu_api.h) + model C ABI (npu_model.h) only.
# PRIVATE: the internal npu_core / npu_memory headers — adapters that link
# myaccel_core inherit PUBLIC but NOT PRIVATE, so they can reach the NPU only
# through myaccel::npu in npu_api.h.
target_include_directories(myaccel_core PUBLIC  core/include)
target_include_directories(myaccel_core PRIVATE core/internal)

# nlohmann/json (header-only, vendored single header in third_party/).
# PRIVATE: json is an implementation detail of the core .cpp files only — it is
# never exposed through npu_model.h / npu_core.h, so it does not leak into the
# adapters or the DLL export surface. Header-only => compiled into myaccel_core.
target_include_directories(myaccel_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# Link your NPU SDK here, e.g.:
#   target_include_directories(myaccel_core PRIVATE "${MYNPU_SDK}/include")
#   target_link_libraries(myaccel_core PRIVATE "${MYNPU_SDK}/lib/npu_runtime.lib")

# --- framework adapters ----------------------------------------------------
if(MYACCEL_BUILD_ORT_EP)
    add_subdirectory(ort_ep)
endif()
if(MYACCEL_BUILD_GGML)
    add_subdirectory(ggml_backend)
endif()
if(MYACCEL_BUILD_EXECUTORCH)
    add_subdirectory(executorch)  # -> myaccel_backend.dll, references myaccel_core
endif()
