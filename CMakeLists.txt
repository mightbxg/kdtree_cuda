cmake_minimum_required(VERSION 3.10)

project(cuda_kdtree)

#set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(BUILD_SAMPLES "build samples (need OpenCV)" ON)
option(BUILD_TESTS "build tests (need GTest)" ON)
option(BUILD_BENCHMARKS "build benchmarks (need Google Benchmark)" ON)

# common dependencies
add_library(common_depends INTERFACE)
target_include_directories(common_depends INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(common_depends INTERFACE Eigen3::Eigen)

# CUDA kdtree
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};--std=c++14")
cuda_add_library(kdtree_cuda kdtree_cuda.cu)

# samples
if (BUILD_SAMPLES)
    add_subdirectory(samples)
endif ()

# tests
if (BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()

# benchmark
if (BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif ()
