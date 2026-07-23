#pragma once

#include <cudalern/Core/err.hpp>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <benchtools/Loggers/Logger.hpp>
#include <benchtools/Loggers/Logger.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

namespace cudalern {

enum class memcpyKind : uint8_t {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
    HostToHost,
};

[[nodiscard]] inline std::string format(const memcpyKind copyKind) {
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
}

template <class T>
    requires(std::is_integral_v<T> || std::is_floating_point_v<T>)
[[nodiscard]] T* allocateDevice(std::size_t size,
                                cudaStream_t stream = nullptr) noexcept {
    T* temp;
    auto err = cudaMalloc(&temp, size * sizeof(T));
    if (err) BENCHTOOLS_CRITICAL("Failed to allocate at device! " + CUDALERN_ERROR(err));
    return (!err ? temp : nullptr);
}

template <class T>
    requires(std::is_integral_v<T> || std::is_floating_point_v<T>)
[[nodiscard]] T* allocateHost(std::size_t size, cudaStream_t stream = nullptr) noexcept {
    T* temp;
    auto err = cudaMallocHost(&temp, size * sizeof(T));
    if (err) BENCHTOOLS_CRITICAL("Failed to allocate at host! " + CUDALERN_ERROR(err));
    return (!err ? temp : nullptr);
}

template <class T>
    requires(std::is_integral_v<T> || std::is_floating_point_v<T>)
[[nodiscard]] T* allocatePinned(std::size_t size,
                                cudaStream_t stream = nullptr) noexcept {
    T* temp;
    auto err = cudaMallocManaged(&temp, size * sizeof(T));
    if (err) BENCHTOOLS_CRITICAL("Failed to allocate pinned! " + CUDALERN_ERROR(err));
    return (!err ? temp : nullptr);
}

template <class T>
    requires(std::is_integral_v<T> || std::is_floating_point_v<T>)
void deallocateDevice(const T* ptr, cudaStream_t stream = nullptr) {
    auto err = cudaFreeAsync((void*)ptr, stream);
    if (err)
        BENCHTOOLS_CRITICAL("Failed to free at device address!" + CUDALERN_ERROR(err));
}

template <class T>
    requires(std::is_integral_v<T> || std::is_floating_point_v<T>)
void deallocateHost(const T* ptr) {
    auto err = cudaFreeHost((void*)ptr);
    if (err) BENCHTOOLS_CRITICAL("Failed to free at host address!" + CUDALERN_ERROR(err));
}

template <class T>
    requires(std::is_integral_v<T> || std::is_floating_point_v<T>)
void deallocatePinned(const T* ptr) {
    auto err = cudaFree((void*)ptr);
    if (err) BENCHTOOLS_CRITICAL("Failed to free at host address!" + CUDALERN_ERROR(err));
}

template <class T>
    requires(std::is_integral_v<T> || std::is_floating_point_v<T>)
[[nodiscard]] cudaError_t memcpy(const T* dst, const T* src, const std::size_t n,
                                 memcpyKind copyKind, cudaStream_t stream = nullptr) {
    cudaError_t err;
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

}  // namespace cudalern
