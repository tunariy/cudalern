include(FetchContent)

FetchContent_Declare(
    Benchtools
    GIT_REPOSITORY https://github.com/tunariy/benchmark.git
    GIT_TAG v1.0
)

FetchContent_MakeAvailable(Benchtools)

add_library(cudalern)

target_compile_features(cudalern PRIVATE cxx_std_20)

# Note for include directories:
#   PUBLIC: ${CMAKE_SOURCE_DIR}/include -> so users can #include <ABI/...> and <cudalern/...>
#   PRIVATE: ${CMAKE_CURRENT_SOURCE_DIR} -> for internal headers
target_include_directories(cudalern
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(cudalern PUBLIC 
    CUDA::cudart
    Benchtools::Timers
)

target_sources(cudalern
    PRIVATE
        ABI/kernel.cu
        Core/device.cpp
        ABI/stream.cpp
        # add other .cu or .cpp files here if any
)

set_target_properties(cudalern PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

# Export compile commands (for clangd)
set_property(TARGET cudalern PROPERTY EXPORT_COMPILE_COMMANDS ON)

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
option(BUILD_TESTING "Build tests" ON)
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()