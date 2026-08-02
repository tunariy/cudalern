option(TESTING OFF)

if(TESTING)
    # Fetch gtest
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    FetchContent_MakeAvailable(googletest)

    # --- ndarray test ---
    add_executable(ndarray_test test_ndarray.cpp)
    set_source_files_properties(test_ndarray.cpp PROPERTIES LANGUAGE CUDA)
    target_link_libraries(ndarray_test PRIVATE cudalern gtest_main)
    set_target_properties(ndarray_test PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    add_test(NAME NdArrayTest COMMAND ndarray_test)

    # --- memory test ---
    add_executable(memory_test test_memory.cpp)   
    set_source_files_properties(test_memory.cpp PROPERTIES LANGUAGE CUDA)
    target_link_libraries(memory_test PRIVATE cudalern gtest_main)
    set_target_properties(memory_test PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    add_test(NAME MemoryTest COMMAND memory_test)
else()
    add_executable(main main.cpp)
    set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
    target_link_libraries(main PRIVATE cudalern)
    set_target_properties(main PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
endif()