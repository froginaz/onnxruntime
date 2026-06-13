# llama.cpp / ggml backend adapter -> ggml-myaccel.dll
#
# The adapter CALLS ggml functions (ggml_backend_buffer_init,
# ggml_backend_buffer_get_size, ...) whose definitions live in the ggml-base
# library. Headers alone let it compile but NOT link -> LNK2019. You must point
# the linker at a prebuilt ggml-base (build llama.cpp first), or build this
# adapter from inside the llama.cpp tree so the 'ggml-base' target exists.
#
#   -DLLAMA_CPP_DIR=<llama.cpp root>          (headers; required)
#   -DLLAMA_CPP_BUILD_DIR=<llama.cpp build>   (where ggml-base.lib/.so was built)

if(NOT DEFINED LLAMA_CPP_DIR)
    message(FATAL_ERROR "Set -DLLAMA_CPP_DIR=<llama.cpp root> to build the ggml backend")
endif()

add_library(ggml-myaccel SHARED ggml-myaccel.cpp)

target_link_libraries(ggml-myaccel PRIVATE myaccel_core)

target_include_directories(ggml-myaccel PRIVATE
    "${LLAMA_CPP_DIR}/ggml/include"
    "${LLAMA_CPP_DIR}/ggml/src")  # ggml-backend-impl.h lives in src

# Preferred: the ggml-base CMake target already exists (we are built via
# add_subdirectory() from inside the llama.cpp tree). CMake then resolves the
# import lib + ABI automatically.
if(TARGET ggml-base)
    target_link_libraries(ggml-myaccel PRIVATE ggml-base)
else()
    # Standalone build: locate the prebuilt ggml-base import/shared library.
    # On the VS (multi-config) generator the lib lands in a Debug/ or Release/
    # subfolder, which is why a naive hint misses it.
    if(NOT DEFINED LLAMA_CPP_BUILD_DIR)
        set(LLAMA_CPP_BUILD_DIR "${LLAMA_CPP_DIR}/build")
    endif()

    find_library(GGML_BASE_LIB
        NAMES ggml-base
        HINTS "${LLAMA_CPP_BUILD_DIR}"
        PATH_SUFFIXES src src/Debug src/Release
                      ggml/src ggml/src/Debug ggml/src/Release
                      bin bin/Debug bin/Release lib lib/Debug lib/Release
        NO_DEFAULT_PATH)

    if(NOT GGML_BASE_LIB)
        message(FATAL_ERROR
            "ggml-base library not found under '${LLAMA_CPP_BUILD_DIR}'.\n"
            "  1) Build llama.cpp first:  cmake -S ${LLAMA_CPP_DIR} -B <build> && cmake --build <build> --config Debug\n"
            "  2) Re-run with -DLLAMA_CPP_BUILD_DIR=<build>\n"
            "Linking without it causes LNK2019 on ggml_backend_* symbols.")
    endif()

    message(STATUS "ggml-myaccel: linking ggml-base -> ${GGML_BASE_LIB}")
    target_link_libraries(ggml-myaccel PRIVATE "${GGML_BASE_LIB}")
endif()
