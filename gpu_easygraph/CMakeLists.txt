cmake_minimum_required(VERSION 3.23)
project(gpu_easygraph LANGUAGES CXX CUDA)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

file(GLOB SOURCES
    functions/*/*.c
    functions/*/*.cpp
    functions/*/*.cu
    common/*.c
    common/*.cpp
    common/*.cu
    runtime/*.cpp
    runtime/*.cu
)

find_package(CUDAToolkit REQUIRED)
add_library(gpu_easygraph OBJECT ${SOURCES})
target_link_libraries(gpu_easygraph PRIVATE CUDA::cudart CUDA::cublas CUDA::cusparse)
# Conda CUDA 13 packages CCCL under include/cccl; toolkit installations may
# already provide thrust/cub directly in include. Accept both layouts.
find_path(EGMGPU_CCCL_INCLUDE_DIR thrust/version.h
    HINTS ${CUDAToolkit_INCLUDE_DIRS}
    PATH_SUFFIXES cccl)
if (EGMGPU_CCCL_INCLUDE_DIR)
    target_include_directories(gpu_easygraph PRIVATE ${EGMGPU_CCCL_INCLUDE_DIR})
    target_include_directories(cpp_easygraph PRIVATE ${EGMGPU_CCCL_INCLUDE_DIR})
endif()

target_include_directories(gpu_easygraph
    PRIVATE common
    PRIVATE functions
    PRIVATE runtime
)

if (EGMGPU_ENABLE_NCCL)
    target_compile_definitions(gpu_easygraph PRIVATE EGMGPU_HAS_NCCL=1)
    target_link_libraries(gpu_easygraph PRIVATE EGMGPU::NCCL)
endif()


set_target_properties(gpu_easygraph  PROPERTIES
    LINK_SEARCH_START_STATIC ON
    LINK_SEARCH_END_STATIC ON
)
