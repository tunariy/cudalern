#pragma once

#include "cudalern/Core/concepts.hpp"
#include <cudalern/Core/err.hpp>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <benchtools/Loggers/Logger.hpp>

#include <cstddef>
#include <cstdint>

namespace cudalern {

enum class memcpyKind : uint8_t {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
    HostToHost,
};

[[nodiscard]] inline auto format(const memcpyKind copyKind) -> std::string {
    switch (copyKind) {
    case memcpyKind::HostToDevice: {
        return "HostToDevice";
    }
    case memcpyKind::DeviceToHost: {
        return "DeviceToHost";
    }
    case memcpyKind::DeviceToDevice: {
        return "DeviceToDevice";
    }
    case memcpyKind::HostToHost: {
        return "HostToHost";
    }
    }
    assert(false && "Invalid memcpyKind!");
    return "";
}

template <class T>
    requires(CudaCompatible<T>)
[[nodiscard]] T* allocateDevice(std::size_t size,
                                cudaStream_t stream = nullptr) noexcept {
    T* temp;
    auto err = cudaMalloc(&temp, size * sizeof(T));
    if (err) BENCHTOOLS_CRITICAL("Failed to allocate at device! " + CUDALERN_ERROR(err));
    return (!err ? temp : nullptr);
}

template <class T>
    requires(CudaCompatible<T>)
[[nodiscard]] T* allocatePinned(std::size_t size,
                                cudaStream_t stream = nullptr) noexcept {
    T* temp;
    auto err = cudaMallocHost(&temp, size * sizeof(T));
    if (err) BENCHTOOLS_CRITICAL("Failed to allocate at host! " + CUDALERN_ERROR(err));
    return (!err ? temp : nullptr);
}

template <class T>
    requires(CudaCompatible<T>)
[[nodiscard]] T* allocateManaged(std::size_t size,
                                 cudaStream_t stream = nullptr) noexcept {
    T* temp;
    auto err = cudaMallocManaged(&temp, size * sizeof(T));
    if (err) BENCHTOOLS_CRITICAL("Failed to allocate pinned! " + CUDALERN_ERROR(err));
    return (!err ? temp : nullptr);
}

template <class T>
    requires(CudaCompatible<T>)
void deallocateDevice(const T* ptr, cudaStream_t stream = nullptr) {
    auto err = cudaFreeAsync((void*)ptr, stream);
    if (err)
        BENCHTOOLS_CRITICAL("Failed to free at device address!" + CUDALERN_ERROR(err));
}

template <class T>
    requires(CudaCompatible<T>)
void deallocatePinned(const T* ptr) {
    auto err = cudaFreeHost((void*)ptr);
    if (err) BENCHTOOLS_CRITICAL("Failed to free at host address!" + CUDALERN_ERROR(err));
}

template <class T>
    requires(CudaCompatible<T>)
void deallocateManaged(const T* ptr) {
    auto err = cudaFree((void*)ptr);
    if (err) BENCHTOOLS_CRITICAL("Failed to free at host address!" + CUDALERN_ERROR(err));
}

template <class T>
    requires(CudaCompatible<T>)
[[nodiscard]] auto memcpy(const T* dst, const T* src, const std::size_t n,
                          memcpyKind copyKind, cudaStream_t stream = nullptr) noexcept
    -> error_t {
    error_t err;
    switch (copyKind) {
    case memcpyKind::HostToDevice: {
        err = cudaMemcpyAsync((void*)dst, (void*)src, n * sizeof(T),
                              cudaMemcpyKind::cudaMemcpyHostToDevice);
        break;
    }
    case memcpyKind::DeviceToHost: {
        err = cudaMemcpyAsync((void*)dst, (void*)src, n * sizeof(T),
                              cudaMemcpyKind::cudaMemcpyDeviceToHost);
        break;
    }
    case memcpyKind::DeviceToDevice: {
        err = cudaMemcpyAsync((void*)dst, (void*)src, n * sizeof(T),
                              cudaMemcpyKind::cudaMemcpyDeviceToDevice);
        break;
    }
    case memcpyKind::HostToHost: {
        err = cudaMemcpyAsync((void*)dst, (void*)src, n * sizeof(T),
                              cudaMemcpyKind::cudaMemcpyHostToHost);
        break;
    }
    }
    if (err)
        BENCHTOOLS_CRITICAL("Memcpy failed for: " + format(copyKind) + " " +
                            CUDALERN_ERROR(err));
    return err;
}

template <class T>
    requires(CudaCompatible<T>)
[[nodiscard]] auto memset(const T* ptr, T val, std::size_t count,
                          cudaStream_t stream = nullptr) noexcept -> error_t {
    return cudaMemsetAsync(ptr, val, count, stream);
}

}  // namespace cudalern
