# In-tree ggml backend: ggml-myaccel
#
# COPY this whole folder to  llama.cpp/ggml/src/ggml-myaccel/  (folder name must
# be lowercase "ggml-myaccel" so ggml_add_backend(MyAccel) finds it).
#
# The adapter source `ggml-myaccel.cpp` lives HERE (in the llama.cpp tree) and is
# compiled locally. The only thing pulled from the sibling shared_backend is:
#   - the public NPU headers (myaccel/npu_api.h, npu_model.h)   [include dir]
#   - the prebuilt NPU core import library  npu_core.lib/.dll   [link + runtime]
#
# Assumed layout (siblings under the repo root):
#   ./
#   ├─ llama.cpp/ggml/src/ggml-myaccel/   (this folder + ggml-myaccel.cpp)
#   └─ shared_backend/                    (npu_core.lib/.dll under build/)

# --- locate the sibling shared_backend (for headers + the core lib/dll) ----
# ggml-myaccel -> src -> ggml -> llama.cpp -> repo root -> shared_backend
set(SHARED_BACKEND_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../shared_backend"
    CACHE PATH "Path to the shared_backend checkout (sibling of llama.cpp)")
get_filename_component(SHARED_BACKEND_DIR "${SHARED_BACKEND_DIR}" ABSOLUTE)

if(NOT EXISTS "${SHARED_BACKEND_DIR}/core/include/myaccel/npu_api.h")
    message(FATAL_ERROR
        "shared_backend headers not found at '${SHARED_BACKEND_DIR}'. "
        "Pass -DSHARED_BACKEND_DIR=<path to shared_backend>.")
endif()

# --- the backend library: compile the LOCAL adapter source -----------------
ggml_add_backend_library(ggml-myaccel
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml-myaccel.cpp)

# public NPU headers only (no shared_backend source is compiled here)
target_include_directories(ggml-myaccel PRIVATE
    "${SHARED_BACKEND_DIR}/core/include")

# --- link the prebuilt NPU core import library (npu_core.lib / .so) ---------
# Build shared_backend first, e.g.:
#   cmake -S shared_backend -B shared_backend/build -DMYACCEL_CORE_SHARED=ON ^
#         -DMYACCEL_CORE_OUTPUT_NAME=npu_core -DMYACCEL_BUILD_GGML=OFF -DMYACCEL_BUILD_ORT_EP=OFF
#   cmake --build shared_backend/build --config Release
set(NPU_CORE_LIB "" CACHE FILEPATH "Path to npu_core.lib / libnpu_core.so (or myaccel_core)")
if(NOT NPU_CORE_LIB)
    find_library(NPU_CORE_LIB
        NAMES npu_core myaccel_core myaccel_npu
        HINTS "${SHARED_BACKEND_DIR}/build" "${SHARED_BACKEND_DIR}/build/Release"
              "${SHARED_BACKEND_DIR}/build/Debug"
        PATH_SUFFIXES "" Release Debug lib lib/Release lib/Debug
        NO_DEFAULT_PATH)
endif()
if(NOT NPU_CORE_LIB)
    message(FATAL_ERROR
        "npu_core import library not found under '${SHARED_BACKEND_DIR}/build'.\n"
        "  1) Build shared_backend with -DMYACCEL_CORE_SHARED=ON (optionally\n"
        "     -DMYACCEL_CORE_OUTPUT_NAME=npu_core)\n"
        "  2) Re-run with -DNPU_CORE_LIB=<full path to npu_core.lib / libnpu_core.so>")
endif()
message(STATUS "ggml-myaccel: linking NPU core -> ${NPU_CORE_LIB}")
target_link_libraries(ggml-myaccel PRIVATE "${NPU_CORE_LIB}")

# --- copy the runtime core DLL next to the built binaries (Windows) --------
# So llama-cli.exe (same bin/ dir) can load npu_core.dll at runtime.
if(WIN32)
    set(NPU_CORE_DLL "" CACHE FILEPATH "Path to npu_core.dll (or myaccel_core.dll)")
    if(NOT NPU_CORE_DLL)
        file(GLOB _npu_core_dlls
            "${SHARED_BACKEND_DIR}/build/Release/npu_core.dll"
            "${SHARED_BACKEND_DIR}/build/Debug/npu_core.dll"
            "${SHARED_BACKEND_DIR}/build/npu_core.dll"
            "${SHARED_BACKEND_DIR}/build/Release/myaccel_core.dll"
            "${SHARED_BACKEND_DIR}/build/Debug/myaccel_core.dll"
            "${SHARED_BACKEND_DIR}/build/myaccel_core.dll")
        if(_npu_core_dlls)
            list(GET _npu_core_dlls 0 NPU_CORE_DLL)
        endif()
    endif()
    if(NPU_CORE_DLL)
        add_custom_command(TARGET ggml-myaccel POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${NPU_CORE_DLL}" "$<TARGET_FILE_DIR:ggml-myaccel>"
            COMMENT "ggml-myaccel: copying ${NPU_CORE_DLL} -> $<TARGET_FILE_DIR:ggml-myaccel>")
    else()
        message(WARNING "npu_core.dll not found to auto-copy; copy it next to llama-cli.exe manually")
    endif()
endif()
