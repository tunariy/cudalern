cmake_minimum_required(VERSION 3.21)
project(cudalern VERSION 1.0.0 LANGUAGES CXX CUDA)

# CXX settings
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)

# CUDA settings
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89" CACHE STRING "CUDA architectures")

enable_testing()

add_subdirectory(src)