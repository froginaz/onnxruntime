# In-tree ggml backend: ggml-myaccel
#
# COPY this folder to  llama.cpp/ggml/src/ggml-myaccel/  (folder name must be
# lowercase "ggml-myaccel" so ggml_add_backend(MyAccel) finds it).
#
# It does NOT duplicate the adapter source — it compiles the canonical
# shared_backend/ggml_backend/ggml-myaccel.cpp and links the prebuilt MyAccel NPU
# core (myaccel_npu.lib/.dll, a.k.a. myaccel_core) from the sibling shared_backend
# checkout. Layout assumed:
#
#   ./
#   ├─ llama.cpp/ggml/src/ggml-myaccel/   (this CMakeLists)
#   └─ shared_backend/                    (myaccel_npu.lib/.dll live under build/)

# --- locate the sibling shared_backend ------------------------------------
# ggml-myaccel -> src -> ggml -> llama.cpp -> repo root -> shared_backend
set(SHARED_BACKEND_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../shared_backend"
    CACHE PATH "Path to the shared_backend checkout (sibling of llama.cpp)")
get_filename_component(SHARED_BACKEND_DIR "${SHARED_BACKEND_DIR}" ABSOLUTE)

if(NOT EXISTS "${SHARED_BACKEND_DIR}/core/include/myaccel/npu_api.h")
    message(FATAL_ERROR
        "shared_backend not found at '${SHARED_BACKEND_DIR}'. "
        "Pass -DSHARED_BACKEND_DIR=<path to shared_backend>.")
endif()

# --- the backend library (reuses the canonical adapter source) ------------
ggml_add_backend_library(ggml-myaccel
    "${SHARED_BACKEND_DIR}/ggml_backend/ggml-myaccel.cpp")

# public NPU headers: myaccel::npu façade (npu_api.h) + model C ABI (npu_model.h)
target_include_directories(ggml-myaccel PRIVATE
    "${SHARED_BACKEND_DIR}/core/include")

# --- link the prebuilt NPU core import library ----------------------------
# Build shared_backend first, e.g.:
#   cmake -S shared_backend -B shared_backend/build -DMYACCEL_CORE_SHARED=ON -DMYACCEL_BUILD_GGML=OFF -DMYACCEL_BUILD_ORT_EP=OFF
#   cmake --build shared_backend/build --config Release
set(MYACCEL_NPU_LIB "" CACHE FILEPATH "Path to myaccel_npu.lib / libmyaccel_core.so")
if(NOT MYACCEL_NPU_LIB)
    find_library(MYACCEL_NPU_LIB
        NAMES myaccel_npu myaccel_core
        HINTS "${SHARED_BACKEND_DIR}/build" "${SHARED_BACKEND_DIR}/build/Release"
              "${SHARED_BACKEND_DIR}/build/Debug"
        PATH_SUFFIXES "" Release Debug lib lib/Release lib/Debug
        NO_DEFAULT_PATH)
endif()
if(NOT MYACCEL_NPU_LIB)
    message(FATAL_ERROR
        "myaccel_npu/myaccel_core import library not found under "
        "'${SHARED_BACKEND_DIR}/build'.\n"
        "  1) Build shared_backend with -DMYACCEL_CORE_SHARED=ON\n"
        "  2) Re-run with -DMYACCEL_NPU_LIB=<full path to myaccel_npu.lib / libmyaccel_core.so>")
endif()
message(STATUS "ggml-myaccel: linking NPU core -> ${MYACCEL_NPU_LIB}")
target_link_libraries(ggml-myaccel PRIVATE "${MYACCEL_NPU_LIB}")

# --- copy the runtime DLL next to the built binaries (Windows) -------------
# So llama-cli.exe (in the same bin/ dir) can load myaccel_npu.dll at runtime.
if(WIN32)
    set(MYACCEL_NPU_DLL "" CACHE FILEPATH "Path to myaccel_npu.dll / myaccel_core.dll")
    if(NOT MYACCEL_NPU_DLL)
        file(GLOB _myaccel_dlls
            "${SHARED_BACKEND_DIR}/build/Release/myaccel_npu.dll"
            "${SHARED_BACKEND_DIR}/build/Debug/myaccel_npu.dll"
            "${SHARED_BACKEND_DIR}/build/myaccel_npu.dll"
            "${SHARED_BACKEND_DIR}/build/Release/myaccel_core.dll"
            "${SHARED_BACKEND_DIR}/build/Debug/myaccel_core.dll"
            "${SHARED_BACKEND_DIR}/build/myaccel_core.dll")
        if(_myaccel_dlls)
            list(GET _myaccel_dlls 0 MYACCEL_NPU_DLL)
        endif()
    endif()
    if(MYACCEL_NPU_DLL)
        add_custom_command(TARGET ggml-myaccel POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${MYACCEL_NPU_DLL}" "$<TARGET_FILE_DIR:ggml-myaccel>"
            COMMENT "ggml-myaccel: copying ${MYACCEL_NPU_DLL} -> $<TARGET_FILE_DIR:ggml-myaccel>")
    else()
        message(WARNING "myaccel_npu.dll not found to auto-copy; copy it next to llama-cli.exe manually")
    endif()
endif()
