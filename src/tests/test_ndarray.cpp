#include <gtest/gtest.h>
#include <cudalern/Containers/NdArray.hpp>
#include <cudalern/Core/device.hpp>

#include <array>
#include <vector>
#include <cmath>

// -----------------------------------------------------------------------------
// Test fixture – initializes CUDA context once for all tests
// -----------------------------------------------------------------------------
class NdArrayTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        cudalern::InitializeContext(0);
    }
};

// -----------------------------------------------------------------------------
// Factory method tests
// -----------------------------------------------------------------------------

TEST_F(NdArrayTest, Zeros) {
    std::array<std::size_t, 2> dims = {2, 3};
    auto arr = cudalern::NdArray<int, 2>::zeros(dims);
    EXPECT_EQ(arr.size(), 6);
    EXPECT_EQ(arr.nbytes(), 24);
    for (std::size_t i = 0; i < dims[0]; ++i)
        for (std::size_t j = 0; j < dims[1]; ++j)
            EXPECT_EQ(arr(i, j), 0);
}

TEST_F(NdArrayTest, Ones) {
    std::array<std::size_t, 3> dims = {2, 2, 2};
    auto arr = cudalern::NdArray<int, 3>::ones(dims);
    EXPECT_EQ(arr.size(), 8);
    for (std::size_t i = 0; i < dims[0]; ++i)
        for (std::size_t j = 0; j < dims[1]; ++j)
            for (std::size_t k = 0; k < dims[2]; ++k)
                EXPECT_EQ(arr(i, j, k), 1);
}

TEST_F(NdArrayTest, Full) {
    std::array<std::size_t, 2> dims = {3, 3};
    auto arr = cudalern::NdArray<int, 2>::full(dims, 42);
    for (std::size_t i = 0; i < dims[0]; ++i)
        for (std::size_t j = 0; j < dims[1]; ++j)
            EXPECT_EQ(arr(i, j), 42);
}

TEST_F(NdArrayTest, Arange) {
    auto arr = cudalern::NdArray<int, 1>::arange(0, 10, 1);
    EXPECT_EQ(arr.size(), 10);
    for (std::size_t i = 0; i < arr.size(); ++i)
        EXPECT_EQ(arr(i), static_cast<int>(i));
}

TEST_F(NdArrayTest, Eye) {
    auto arr = cudalern::NdArray<int, 2>::eye(4);
    EXPECT_EQ(arr.size(), 16);
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            int expected = (i == j) ? 1 : 0;
            EXPECT_EQ(arr(i, j), expected);
        }
    }
}

TEST_F(NdArrayTest, RandomUniformRange) {
    std::array<std::size_t, 2> dims = {100, 100};
    auto arr = cudalern::NdArray<float, 2>::random_uniform(dims, 0.0f, 1.0f);
    auto data = arr.data();
    for (float v : data) {
        EXPECT_GE(v, 0.0f);
        EXPECT_LT(v, 1.0f);
    }
}

TEST_F(NdArrayTest, RandomNormalStats) {
    std::array<std::size_t, 2> dims = {100, 100};
    auto arr = cudalern::NdArray<float, 2>::random_normal(dims, 0.0f, 1.0f);
    auto data = arr.data();
    // Basic sanity: no NaNs, values within 5 sigma
    for (float v : data) {
        EXPECT_FALSE(std::isnan(v));
        EXPECT_GT(v, -5.0f);
        EXPECT_LT(v, 5.0f);
    }
}

// -----------------------------------------------------------------------------
// Construction from nested sequences
// -----------------------------------------------------------------------------

TEST_F(NdArrayTest, From2DVector) {
    std::vector<std::vector<int>> data = {{1, 2, 3}, {4, 5, 6}};
    auto arr = cudalern::NdArray<int, 2>(data);
    EXPECT_EQ(arr(0, 0), 1);
    EXPECT_EQ(arr(0, 2), 3);
    EXPECT_EQ(arr(1, 1), 5);
}

TEST_F(NdArrayTest, From3DVector) {
    std::vector<std::vector<std::vector<int>>> data = {
        {{1, 2}, {3, 4}},
        {{5, 6}, {7, 8}}
    };
    auto arr = cudalern::NdArray<int, 3>(data);
    EXPECT_EQ(arr(0, 0, 0), 1);
    EXPECT_EQ(arr(0, 1, 1), 4);
    EXPECT_EQ(arr(1, 0, 1), 6);
}

// -----------------------------------------------------------------------------
// Copy and move semantics
// -----------------------------------------------------------------------------

TEST_F(NdArrayTest, CopyConstructor) {
    std::array<std::size_t, 2> dims = {2, 2};
    auto original = cudalern::NdArray<int, 2>::full(dims, 99);
    auto copy = original;
    EXPECT_EQ(copy(0, 0), 99);
    EXPECT_EQ(copy(1, 1), 99);
    // Original unchanged
    EXPECT_EQ(original(0, 0), 99);
}

TEST_F(NdArrayTest, CopyAssignment) {
    std::array<std::size_t, 2> dims = {2, 2};
    auto original = cudalern::NdArray<int, 2>::full(dims, 77);
    auto copy = cudalern::NdArray<int, 2>::zeros(dims);
    copy = original;
    EXPECT_EQ(copy(0, 0), 77);
    EXPECT_EQ(copy(1, 1), 77);
}

TEST_F(NdArrayTest, MoveConstructor) {
    std::array<std::size_t, 2> dims = {2, 2};
    auto original = cudalern::NdArray<int, 2>::full(dims, 88);
    auto moved = std::move(original);
    EXPECT_EQ(moved(0, 0), 88);
    // After move, original should be empty
    EXPECT_TRUE(original.empty());
    EXPECT_EQ(original.size(), 0);
}

TEST_F(NdArrayTest, MoveAssignment) {
    std::array<std::size_t, 2> dims = {2, 2};
    auto original = cudalern::NdArray<int, 2>::full(dims, 66);
    auto moved = cudalern::NdArray<int, 2>::zeros(dims);
    moved = std::move(original);
    EXPECT_EQ(moved(0, 0), 66);
    EXPECT_TRUE(original.empty());
}

// -----------------------------------------------------------------------------
// Data access and extraction
// -----------------------------------------------------------------------------

TEST_F(NdArrayTest, ReadAndOperator) {
    std::array<std::size_t, 3> dims = {2, 2, 2};
    auto arr = cudalern::NdArray<int, 3>::ones(dims);
    EXPECT_EQ(arr(1, 1, 1), 1);
    EXPECT_EQ(arr.read(0, 0, 0), 1);
}

TEST_F(NdArrayTest, DataCopy) {
    std::array<std::size_t, 1> dims = {5};
    auto arr = cudalern::NdArray<int, 1>::arange(10, 15, 1);
    auto vec = arr.data();
    std::vector<int> expected = {10, 11, 12, 13, 14};
    EXPECT_EQ(vec, expected);
}

TEST_F(NdArrayTest, ToHost) {
    std::array<std::size_t, 1> dims = {5};
    auto arr = cudalern::NdArray<int, 1>::arange(20, 25, 1);
    std::vector<int> host;
    host.reserve(arr.size());
    arr.to_host(host, cudalern::Stream());
    std::vector<int> expected = {20, 21, 22, 23, 24};
    EXPECT_EQ(host, expected);
}

// -----------------------------------------------------------------------------
// Pinned and Managed memory
// -----------------------------------------------------------------------------

TEST_F(NdArrayTest, Pinned) {
    std::array<std::size_t, 2> dims = {2, 2};
    auto arr = cudalern::NdArray<int, 2>::pinned(dims);
    EXPECT_FALSE(arr.empty());
    // Can we write and read?
    auto full = cudalern::NdArray<int, 2>::full(dims, 7);
    EXPECT_EQ(full(0, 0), 7);
}

TEST_F(NdArrayTest, Managed) {
    std::array<std::size_t, 2> dims = {2, 2};
    auto arr = cudalern::NdArray<int, 2>::managed(dims);
    EXPECT_FALSE(arr.empty());
    auto full = cudalern::NdArray<int, 2>::full(dims, 8);
    EXPECT_EQ(full(0, 0), 8);
}

// -----------------------------------------------------------------------------
// from_host
// -----------------------------------------------------------------------------

TEST_F(NdArrayTest, FromHost) {
    std::vector<int> host_data = {10, 20, 30, 40, 50};
    std::array<std::size_t, 1> dims = {5};
    auto arr = cudalern::NdArray<int, 1>::from_host(host_data, dims, cudalern::Stream());
    auto vec = arr.data();
    EXPECT_EQ(vec, host_data);
}

// -----------------------------------------------------------------------------
// release() and getUnderlying()
// -----------------------------------------------------------------------------

TEST_F(NdArrayTest, Release) {
    std::array<std::size_t, 1> dims = {3};
    auto arr = cudalern::NdArray<int, 1>::full(dims, 100);
    auto ptr = arr.release();
    EXPECT_NE(ptr, nullptr);
    EXPECT_TRUE(arr.empty());  // after release, array is empty
    // Memory stays alive – we can still access it via ptr
    // In a real test we might copy the value, but we don't have a device->host copy here.
    // We'll just check it's not null.
    // Cleanup: we need to free it ourselves, but we don't know the original allocator.
    // For testing, we can just let it leak (or use cudaFree if we know it's device).
    // We'll skip freeing to keep it simple.
}

TEST_F(NdArrayTest, GetUnderlying) {
    std::array<std::size_t, 1> dims = {1};
    auto arr = cudalern::NdArray<int, 1>::full(dims, 42);
    auto ptr = arr.getUnderlying();
    EXPECT_NE(ptr, nullptr);
    // ptr is a shared_ptr, so it's safe.
}

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------

TEST_F(NdArrayTest, EmptyAndNbytes) {
    std::array<std::size_t, 2> dims = {2, 3};
    auto arr = cudalern::NdArray<int, 2>::zeros(dims);
    arr.synchronize();
    EXPECT_FALSE(arr.empty());
    EXPECT_EQ(arr.nbytes(), 6 * sizeof(int));
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}