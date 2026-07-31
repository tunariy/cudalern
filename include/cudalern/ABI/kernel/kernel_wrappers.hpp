#pragma once

#include <cmath>
#include <cudalern/ABI/memory/stream.hpp>
#include <cudalern/Core/Device/device.hpp>
#include <cudalern/Core/core.hpp>

#include <cuda_runtime.h>
#include <driver_types.h>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace cudalern {

/**
 * @brief Generic launcher for kernels
 *
 * @param Kernel kernel function
 * @param dim3 grid count
 * @param dim3 block size (thread count)
 * @param std::size_t sharedMem
 */
template <typename Kernel, class... Args>
    requires(std::is_invocable_v<Kernel, Args...>)
[[nodiscard]] cudalernErr launchKernel(Kernel function, dim3 gridDim, dim3 blockDim,
                                       std::size_t sharedMem, Stream stream,
                                       Args... args) {
    void* params[] = {(void*)&args...};
    return cudaLaunchKernel(function, gridDim, blockDim, params, sharedMem, stream);
}

template <typename Kernel, class... Args>
    requires(std::is_invocable_v<Kernel, Args...>)
[[nodiscard]] cudalernErr launchKernel1D(Kernel function, std::size_t sharedMem,
                                         Stream stream, std::size_t N, Args... args) {
    auto& props = gDeviceHandler.getDeviceProps();
    auto maxThreadsPerBlock = props.getMaxThreadsPerBlock();

    auto threadsPerBlock = maxThreadsPerBlock;
    auto numBlocks = (N + maxThreadsPerBlock - 1) / maxThreadsPerBlock;

    return launchKernel(function, dim3(numBlocks), dim3(threadsPerBlock), sharedMem,
                        stream, std::forward(args)...);
}

template <typename Kernel, class... Args>
    requires(std::is_invocable_v<Kernel, Args...>)
[[nodiscard]] cudalernErr launchKernel2D(Kernel function, std::size_t width,
                                         std::size_t height, std::size_t sharedMem,
                                         Stream stream, Args... args) {
    auto& props = DeviceProperties::getDeviceProps();
    int maxThreads = props.getMaxThreadsPerBlock();
    auto maxDim = props.getMaxThreadsDim();

    // x, y = {sqrt(maxThreads), sqrt(maxThreads)}
    // essentially splitting the threads equally across dimensions
    int blockDimX = std::sqrt(static_cast<double>(maxThreads));
    int blockDimY{blockDimX};

    // ensure we don't exceed per‑axis maxima
    blockDimX = std::min(blockDimX, maxDim[0]);
    blockDimY = std::min(blockDimY, maxDim[1]);

    // Ensure at least 1
    blockDimX = std::max(blockDimX, 1);
    blockDimY = std::max(blockDimY, 1);

    dim3 block(static_cast<unsigned int>(blockDimX), static_cast<unsigned int>(blockDimY),
               1);

    // Compute grid dimensions
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);

    return launchKernel(function, grid, block, sharedMem, stream,
                        std::forward<Args>(args)...);
}

// 3D convenience wrapper – computes block dims automatically
template <typename Kernel, class... Args>
    requires(std::is_invocable_v<Kernel, Args...>)
[[nodiscard]] cudalernErr
launchKernel3D(Kernel function, std::size_t width, std::size_t height, std::size_t depth,
               std::size_t sharedMem, Stream stream, Args... args) {
    auto& props = DeviceProperties::getDeviceProps();
    int maxThreads = props.getMaxThreadsPerBlock();
    auto maxDim = props.getMaxThreadsDim();

    int blockDim = static_cast<int>(std::cbrt(static_cast<double>(maxThreads)));

    // ensure we don't exceed per‑axis maxima
    int blockDimX = std::min(blockDim, maxDim[0]);
    int blockDimY = std::min(blockDim, maxDim[1]);
    int blockDimZ = std::min(blockDim, maxDim[2]);

    // Ensure at least 1
    blockDimX = std::max(blockDimX, 1);
    blockDimY = std::max(blockDimY, 1);
    blockDimZ = std::max(blockDimZ, 1);

    dim3 block(static_cast<unsigned int>(blockDimX), static_cast<unsigned int>(blockDimY),
               static_cast<unsigned int>(blockDimZ));

    // Compute grid dimensions
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y,
              (depth + block.z - 1) / block.z);

    return launchKernel(function, grid, block, sharedMem, stream,
                        std::forward<Args>(args)...);
}
}  // namespace cudalern