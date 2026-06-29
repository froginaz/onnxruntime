# ExecuTorch backend adapter -> myaccel_backend.dll / libmyaccel_backend.so
#
# References the MyAccel NPU core (myaccel_core) and the ExecuTorch runtime.
# Requires an ExecuTorch checkout/install for headers + the runtime/backend libs.
#   -DEXECUTORCH_DIR=<executorch root or install prefix>

if(NOT DEFINED EXECUTORCH_DIR)
    message(FATAL_ERROR "Set -DEXECUTORCH_DIR=<executorch root/install> to build the ExecuTorch adapter")
endif()

add_library(myaccel_backend SHARED executorch_backend.cpp)

# References myaccel_core (static -> absorbed, or shared -> imports myaccel_core.dll).
target_link_libraries(myaccel_backend PRIVATE myaccel_core)

target_include_directories(myaccel_backend PRIVATE
    "${EXECUTORCH_DIR}"
    "${EXECUTORCH_DIR}/include")

# Link the ExecuTorch runtime + backend-interface library. Names/paths vary by
# ExecuTorch version and build; adjust to your install layout.
if(TARGET executorch)
    target_link_libraries(myaccel_backend PRIVATE executorch)
else()
    find_library(EXECUTORCH_LIB NAMES executorch
        HINTS "${EXECUTORCH_DIR}" "${EXECUTORCH_DIR}/lib"
        PATH_SUFFIXES lib cmake-out cmake-out/lib)
    if(EXECUTORCH_LIB)
        target_link_libraries(myaccel_backend PRIVATE "${EXECUTORCH_LIB}")
    else()
        message(WARNING "executorch runtime library not found under EXECUTORCH_DIR; link it manually")
    endif()
endif()
