project (PuassonGradientCuda)
cmake_minimum_required(VERSION 3.1)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")

find_package(CUDA REQUIRED)
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
include(FindCUDA)

set(CUDA_SOURCES main_par.cu mesh.h mesh.cpp mat.h cuda_mesh.cuh cuda_mesh.cu iter.cuh iter.cu)

cuda_add_executable(parallel_cuda ${CUDA_SOURCES})
target_link_libraries(parallel_cuda ${MPI_CXX_LIBRARIES})
