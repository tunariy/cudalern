cmake_minimum_required(VERSION 3.21)
project(cudalern VERSION 1.0.0 LANGUAGES CXX CUDA)

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA settings
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")
find_package(CUDAToolkit REQUIRED)

# Optional: set default CUDA architectures (adjust to your hardware)
set(CMAKE_CUDA_ARCHITECTURES "75;80;86" CACHE STRING "CUDA architectures")

# Add the src subdirectory (contains both library and test)
add_subdirectory(src)