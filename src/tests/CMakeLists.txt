include(FetchContent)

# Fetch gtest
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

add_executable(ndarray_test test_ndarray.cpp)
set_source_files_properties(test_ndarray.cpp PROPERTIES LANGUAGE CUDA)


target_link_libraries(ndarray_test PRIVATE cudalern gtest_main)
set_target_properties(ndarray_test PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

# Register the test with CTest
add_test(NAME NdArrayTest COMMAND ndarray_test)