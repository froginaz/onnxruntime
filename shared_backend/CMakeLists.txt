# MyAccel shared HW-accelerator backend for onnxruntime + llama.cpp.
#
#   onnxruntime  --(plugin EP ABI)-->  myaccel_ort_ep.dll  --+
#                                                            +--> myaccel_core (NPU SDK)
#   llama.cpp    --(ggml backend ABI)-> ggml-myaccel.dll   --+
#
# Configure examples (Windows, x64 Native Tools prompt):
#   cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DORT_HOME=C:/ort -DLLAMA_CPP_DIR=C:/llama.cpp
#   cmake --build build --config Release
#
# Build only one adapter:
#   cmake -S . -B build -DMYACCEL_BUILD_GGML=OFF -DORT_HOME=C:/ort

cmake_minimum_required(VERSION 3.26)
project(myaccel_shared_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(MYACCEL_BUILD_ORT_EP "Build the onnxruntime plugin-EP adapter" ON)
option(MYACCEL_BUILD_GGML   "Build the llama.cpp ggml backend adapter" ON)
option(MYACCEL_CORE_SHARED  "Build the NPU core as a shared library"   OFF)

# --- shared, framework-agnostic core --------------------------------------
if(MYACCEL_CORE_SHARED)
    add_library(myaccel_core SHARED core/src/npu_core.cpp)
    target_compile_definitions(myaccel_core PRIVATE MYACCEL_CORE_BUILD_DLL)
    target_compile_definitions(myaccel_core PUBLIC MYACCEL_CORE_USE_DLL)
else()
    add_library(myaccel_core STATIC core/src/npu_core.cpp)
endif()
target_include_directories(myaccel_core PUBLIC core/include)

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
