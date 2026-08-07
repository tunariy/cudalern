#include <cudalern/ABI/memory/allocator.hpp>
#include <cudalern/ABI/memory/memory.hpp>
#include <cudalern/ABI/memory/stream.hpp>
#include <cudalern/Core/Device/device.hpp>

#include <gtest/gtest.h>

#include <cstring>
#include <vector>

// -----------------------------------------------------------------------------
// Test fixture – initializes CUDA context once for all tests
// -----------------------------------------------------------------------------
class MemoryTest : public ::testing::Test {
  protected:
    static void SetUpTestSuite() {
        // Initialize CUDA context on device 0
        cudalern::internal::InitializeContext(0);
    }
};

// -----------------------------------------------------------------------------
// Stream tests
// -----------------------------------------------------------------------------

TEST_F(MemoryTest, StreamDefaultConstruction) {
    cudalern::Stream stream;
    EXPECT_TRUE(stream.valid());
    EXPECT_NE(stream.get(), nullptr);
    EXPECT_EQ(stream.synchronize(), cudaSuccess);
}

TEST_F(MemoryTest, StreamCopyAndAssignment) {
    cudalern::Stream s1;
    cudalern::Stream s2(s1);  // copy constructor
    EXPECT_TRUE(s2.valid());
    EXPECT_EQ(s1.get(), s2.get());  // shared pointer, same underlying stream

    cudalern::Stream s3;
    s3 = s1;  // copy assignment
    EXPECT_TRUE(s3.valid());
    EXPECT_EQ(s1.get(), s3.get());
}

TEST_F(MemoryTest, StreamMoveSemantics) {
    cudalern::Stream s1;
    cudaStream_t raw = s1.get();

    cudalern::Stream s2(std::move(s1));
    EXPECT_TRUE(s2.valid());
    EXPECT_EQ(s2.get(), raw);
    EXPECT_FALSE(s1.valid());  // moved-from is invalid

    cudalern::Stream s3;
    s3 = std::move(s2);
    EXPECT_TRUE(s3.valid());
    EXPECT_EQ(s3.get(), raw);
    EXPECT_FALSE(s2.valid());
}

TEST_F(MemoryTest, StreamTake) {
    cudalern::Stream stream;
    cudaStream_t raw = stream.get();
    EXPECT_NE(raw, nullptr);

    // Create a new raw stream to demonstrate take()
    cudaStream_t new_raw;
    cudaStreamCreate(&new_raw);
    stream.take(new_raw);  // now stream owns new_raw
    EXPECT_TRUE(stream.valid());
    EXPECT_EQ(stream.get(), new_raw);
}

TEST_F(MemoryTest, StreamSynchronize) {
    cudalern::Stream stream;
    EXPECT_EQ(stream.synchronize(), cudaSuccess);
}

// -----------------------------------------------------------------------------
// Allocator tests
// -----------------------------------------------------------------------------

TEST_F(MemoryTest, AllocatorStaticDeviceAllocateDeallocate) {
    const std::size_t n = 100;
    int* ptr = cudalern::allocator<int>::allocate<cudalern::allocatorPolicy::Device>(n);
    ASSERT_NE(ptr, nullptr);

    cudalern::Stream stream;
    [[maybe_unused]] auto err1 =
        cudaMemsetAsync(ptr, 0xAA, n * sizeof(int), stream.get());
    [[maybe_unused]] auto err2 = stream.synchronize();

    cudalern::allocator<int>::deallocate<cudalern::allocatorPolicy::Device>(ptr);
}

TEST_F(MemoryTest, AllocatorStaticPinnedAllocateDeallocate) {
    const std::size_t n = 100;
    int* ptr = cudalern::allocator<int>::allocate<cudalern::allocatorPolicy::Pinned>(n);
    ASSERT_NE(ptr, nullptr);
    for (std::size_t i = 0; i < n; ++i)
        ptr[i] = static_cast<int>(i);
    for (std::size_t i = 0; i < n; ++i)
        EXPECT_EQ(ptr[i], static_cast<int>(i));

    cudalern::allocator<int>::deallocate<cudalern::allocatorPolicy::Pinned>(ptr);
}

TEST_F(MemoryTest, AllocatorStaticManagedAllocateDeallocate) {
    const std::size_t n = 100;
    float* ptr =
        cudalern::allocator<float>::allocate<cudalern::allocatorPolicy::Managed>(n);
    ASSERT_NE(ptr, nullptr);
    for (std::size_t i = 0; i < n; ++i)
        ptr[i] = static_cast<float>(i) * 1.5f;
    cudalern::Stream stream;
    [[maybe_unused]] auto err = stream.synchronize();
    for (std::size_t i = 0; i < n; ++i)
        EXPECT_FLOAT_EQ(ptr[i], static_cast<float>(i) * 1.5f);

    cudalern::allocator<float>::deallocate<cudalern::allocatorPolicy::Managed>(ptr);
}

TEST_F(MemoryTest, AllocatorInstanceAllocate) {
    const std::size_t n = 50;
    cudalern::allocator<double> alloc;
    double* ptr = alloc.allocate(n, cudalern::allocatorPolicy::Device);
    ASSERT_NE(ptr, nullptr);

    cudalern::Stream stream;
    [[maybe_unused]] auto err1 =
        cudaMemsetAsync(ptr, 0, n * sizeof(double), stream.get());
    [[maybe_unused]] auto err2 = stream.synchronize();

    alloc.deallocate<cudalern::allocatorPolicy::Device>(ptr, stream);
}

TEST_F(MemoryTest, AllocatorSizeZero) {
    int* ptr = cudalern::allocator<int>::allocate<cudalern::allocatorPolicy::Device>(0);
    cudalern::allocator<int>::deallocate<cudalern::allocatorPolicy::Device>(ptr);
}

TEST_F(MemoryTest, AllocatorDeleters) {
    {
        int* ptr =
            cudalern::allocator<int>::allocate<cudalern::allocatorPolicy::Device>(10);
        ASSERT_NE(ptr, nullptr);
        auto str = cudalern::Stream{};
        cudalern::DeviceDeleter<int> deleter{str};
        deleter(ptr);
    }
    {
        float* ptr =
            cudalern::allocator<float>::allocate<cudalern::allocatorPolicy::Pinned>(10);
        ASSERT_NE(ptr, nullptr);
        cudalern::PinnedDeleter<float> deleter;
        deleter(ptr);
    }
    {
        double* ptr =
            cudalern::allocator<double>::allocate<cudalern::allocatorPolicy::Managed>(10);
        ASSERT_NE(ptr, nullptr);
        cudalern::ManagedDeleter<double> deleter;
        deleter(ptr);
    }
    {
        auto str = cudalern::Stream{};
        cudalern::DeviceDeleter<int> deleter{str};
        deleter(nullptr);
    }
}

// -----------------------------------------------------------------------------
// Direct allocation functions tests
// -----------------------------------------------------------------------------

TEST_F(MemoryTest, AllocateDeviceDirect) {
    const std::size_t n = 200;
    float* ptr = cudalern::allocateDevice<float>(n);
    ASSERT_NE(ptr, nullptr);
    cudalern::deallocateDevice(ptr);
}

TEST_F(MemoryTest, AllocatePinnedDirect) {
    const std::size_t n = 200;
    float* ptr = cudalern::allocatePinned<float>(n);
    ASSERT_NE(ptr, nullptr);
    cudalern::deallocatePinned(ptr);
}

TEST_F(MemoryTest, AllocateManagedDirect) {
    const std::size_t n = 200;
    float* ptr = cudalern::allocateManaged<float>(n);
    ASSERT_NE(ptr, nullptr);
    cudalern::deallocateManaged(ptr);
}

// -----------------------------------------------------------------------------
// memcpy tests
// -----------------------------------------------------------------------------

TEST_F(MemoryTest, MemcpyHostToDeviceAndBack) {
    const std::size_t n = 1024;
    std::vector<int> host_src(n);
    for (std::size_t i = 0; i < n; ++i)
        host_src[i] = static_cast<int>(i % 100);

    int* dev_dst = cudalern::allocateDevice<int>(n);
    ASSERT_NE(dev_dst, nullptr);

    cudalern::Stream stream;
    cudalern::cudalernErr err = cudalern::memcpy(
        dev_dst, host_src.data(), n, cudalern::memcpyKind::HostToDevice, stream.get());
    EXPECT_EQ(err, cudaSuccess);
    [[maybe_unused]] auto sync1 = stream.synchronize();

    std::vector<int> host_dst(n);
    err = cudalern::memcpy(host_dst.data(), dev_dst, n,
                           cudalern::memcpyKind::DeviceToHost, stream.get());
    EXPECT_EQ(err, cudaSuccess);
    [[maybe_unused]] auto sync2 = stream.synchronize();

    EXPECT_EQ(host_src, host_dst);
    cudalern::deallocateDevice(dev_dst);
}

TEST_F(MemoryTest, MemcpyDeviceToDevice) {
    const std::size_t n = 512;
    float* dev_src = cudalern::allocateDevice<float>(n);
    float* dev_dst = cudalern::allocateDevice<float>(n);
    ASSERT_NE(dev_src, nullptr);
    ASSERT_NE(dev_dst, nullptr);

    std::vector<float> host_src(n, 3.14f);
    cudalern::Stream stream;

    // Capture each nodiscard call
    [[maybe_unused]] auto err1 = cudalern::memcpy(
        dev_src, host_src.data(), n, cudalern::memcpyKind::HostToDevice, stream.get());
    [[maybe_unused]] auto sync1 = stream.synchronize();

    [[maybe_unused]] auto err2 = cudalern::memcpy(
        dev_dst, dev_src, n, cudalern::memcpyKind::DeviceToDevice, stream.get());
    [[maybe_unused]] auto sync2 = stream.synchronize();

    std::vector<float> host_dst(n);
    [[maybe_unused]] auto err3 = cudalern::memcpy(
        host_dst.data(), dev_dst, n, cudalern::memcpyKind::DeviceToHost, stream.get());
    [[maybe_unused]] auto sync3 = stream.synchronize();

    for (std::size_t i = 0; i < n; ++i)
        EXPECT_FLOAT_EQ(host_dst[i], 3.14f);

    cudalern::deallocateDevice(dev_src);
    cudalern::deallocateDevice(dev_dst);
}

TEST_F(MemoryTest, MemcpyHostToHost) {
    const std::size_t n = 100;
    std::vector<double> src(n, 2.718);
    std::vector<double> dst(n, 0.0);

    cudalern::Stream stream;
    cudalern::cudalernErr err = cudalern::memcpy(
        dst.data(), src.data(), n, cudalern::memcpyKind::HostToHost, stream.get());
    EXPECT_EQ(err, cudaSuccess);
    EXPECT_EQ(src, dst);
}

TEST_F(MemoryTest, MemcpyZeroSize) {
    int* dev_ptr = cudalern::allocateDevice<int>(1);
    ASSERT_NE(dev_ptr, nullptr);
    int host_val = 42;
    cudalern::Stream stream;
    cudalern::cudalernErr err = cudalern::memcpy(
        dev_ptr, &host_val, 0, cudalern::memcpyKind::HostToDevice, stream.get());
    EXPECT_EQ(err, cudaSuccess);
    cudalern::deallocateDevice(dev_ptr);
}

// -----------------------------------------------------------------------------
// memset tests
// -----------------------------------------------------------------------------

TEST_F(MemoryTest, MemsetDevice) {
    const std::size_t n = 256;
    unsigned char* dev_ptr = cudalern::allocateDevice<unsigned char>(n);
    ASSERT_NE(dev_ptr, nullptr);

    cudalern::Stream stream;
    unsigned char pattern = 0xAB;
    cudalern::cudalernErr err = cudalern::memset(dev_ptr, pattern, n, stream.get());
    EXPECT_EQ(err, cudaSuccess);
    [[maybe_unused]] auto sync = stream.synchronize();

    std::vector<unsigned char> host_data(n);
    [[maybe_unused]] auto err2 = cudalern::memcpy(
        host_data.data(), dev_ptr, n, cudalern::memcpyKind::DeviceToHost, stream.get());
    [[maybe_unused]] auto sync2 = stream.synchronize();

    for (std::size_t i = 0; i < n; ++i)
        EXPECT_EQ(host_data[i], pattern);

    cudalern::deallocateDevice(dev_ptr);
}

TEST_F(MemoryTest, MemsetZeroCount) {
    int* dev_ptr = cudalern::allocateDevice<int>(1);
    ASSERT_NE(dev_ptr, nullptr);
    cudalern::Stream stream;
    cudalern::cudalernErr err = cudalern::memset(dev_ptr, 0, 0, stream.get());
    EXPECT_EQ(err, cudaSuccess);
    cudalern::deallocateDevice(dev_ptr);
}

// -----------------------------------------------------------------------------
// Edge cases and mixed usage
// -----------------------------------------------------------------------------

TEST_F(MemoryTest, MultipleStreamsAndCopies) {
    const std::size_t n = 1000;
    std::vector<int> host_src(n);
    for (std::size_t i = 0; i < n; ++i)
        host_src[i] = static_cast<int>(i);

    int* dev = cudalern::allocateDevice<int>(n);
    ASSERT_NE(dev, nullptr);

    cudalern::Stream s1, s2;

    [[maybe_unused]] auto err1 = cudalern::memcpy(
        dev, host_src.data(), n, cudalern::memcpyKind::HostToDevice, s1.get());
    [[maybe_unused]] auto sync1 = s1.synchronize();

    [[maybe_unused]] auto err2 = cudalern::memset(dev, 0x00, n, s2.get());
    [[maybe_unused]] auto sync2 = s2.synchronize();

    std::vector<int> host_dst(n);
    [[maybe_unused]] auto err3 = cudalern::memcpy(
        host_dst.data(), dev, n, cudalern::memcpyKind::DeviceToHost, s1.get());
    [[maybe_unused]] auto sync3 = s1.synchronize();

    for (int v : host_dst)
        EXPECT_EQ(v, 0);

    cudalern::deallocateDevice(dev);
}

TEST_F(MemoryTest, AllocateAndDeallocateWithStream) {
    const std::size_t n = 100;
    int* ptr = cudalern::allocateDevice<int>(n);
    ASSERT_NE(ptr, nullptr);
    cudalern::Stream stream;
    cudalern::deallocateDevice(ptr, stream.get());
    // No crash
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}