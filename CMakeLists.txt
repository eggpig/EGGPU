cmake_minimum_required(VERSION 3.23)

project(easygraph)

option(EASYGRAPH_ENABLE_OPENMP "Enable OpenMP acceleration (auto-detect)" ON)

option(EASYGRAPH_ENABLE_GPU "EASYGRAPH_ENABLE_GPU" OFF)

option(EGMGPU_ENABLE_NCCL "Build EGMGPU with NCCL collectives" OFF)

if (EGMGPU_ENABLE_NCCL AND NOT EASYGRAPH_ENABLE_GPU)
    message(FATAL_ERROR "EGMGPU_ENABLE_NCCL=ON requires EASYGRAPH_ENABLE_GPU=ON")
endif()

if (EGMGPU_ENABLE_NCCL)
    find_path(EGMGPU_NCCL_INCLUDE_DIR nccl.h)
    find_library(EGMGPU_NCCL_LIBRARY nccl)
    if (NOT EGMGPU_NCCL_INCLUDE_DIR OR NOT EGMGPU_NCCL_LIBRARY)
        message(FATAL_ERROR
            "EGMGPU_ENABLE_NCCL=ON requires both nccl.h and libnccl")
    endif()
    add_library(EGMGPU::NCCL UNKNOWN IMPORTED GLOBAL)
    set_target_properties(EGMGPU::NCCL PROPERTIES
        IMPORTED_LOCATION "${EGMGPU_NCCL_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${EGMGPU_NCCL_INCLUDE_DIR}"
    )
endif()


add_subdirectory(cpp_easygraph)

if (EASYGRAPH_ENABLE_GPU)
    message("easygraph gpu module is enabled")
    add_subdirectory(gpu_easygraph)
    target_include_directories(cpp_easygraph
        PRIVATE gpu_easygraph
    )
else()
    message("easygraph gpu module is disabled")
endif()
