# llama.cpp / ggml backend adapter -> ggml-myaccel.dll
#
# Requires the ggml headers from a llama.cpp checkout (ggml/include) and links
# against ggml-base. Set -DLLAMA_CPP_DIR=<llama.cpp root>.

if(NOT DEFINED LLAMA_CPP_DIR)
    message(FATAL_ERROR "Set -DLLAMA_CPP_DIR=<llama.cpp root> to build the ggml backend")
endif()

add_library(ggml-myaccel SHARED ggml-myaccel.cpp)

target_link_libraries(ggml-myaccel PRIVATE myaccel_core)

target_include_directories(ggml-myaccel PRIVATE
    "${LLAMA_CPP_DIR}/ggml/include"
    "${LLAMA_CPP_DIR}/ggml/src")  # ggml-backend-impl.h lives in src

# Link ggml-base for ggml_backend_buffer_init / ggml helpers. If you build this
# from inside the llama.cpp tree, the target 'ggml-base' already exists; standalone
# builds need to point the linker at the prebuilt import library.
if(TARGET ggml-base)
    target_link_libraries(ggml-myaccel PRIVATE ggml-base)
else()
    find_library(GGML_BASE_LIB ggml-base HINTS "${LLAMA_CPP_DIR}/build/ggml/src" "${LLAMA_CPP_DIR}/build/bin")
    if(GGML_BASE_LIB)
        target_link_libraries(ggml-myaccel PRIVATE "${GGML_BASE_LIB}")
    else()
        message(WARNING "ggml-base library not found; link it manually for a standalone build")
    endif()
endif()
