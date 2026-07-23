#pragma once

#include <cuda_runtime_api.h>
#include <cudalern/ABI/memory.hpp>
#include <cudalern/ABI/stream.hpp>

#include <driver_types.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cudalern {

enum class allocatorPolicy : uint8_t { Host, Device, Pinned };

template <class T>
    requires(std::is_integral_v<T> || std::is_floating_point_v<T>)
class allocator {
  public:
    allocator() = default;

    allocator(const allocator&) = delete;
    allocator& operator=(const allocator&) = delete;
    allocator(allocator&&) noexcept = delete;
    allocator& operator=(allocator&&) noexcept = delete;

    template <class U>
    allocator(allocator<U> other) = delete;

    // Static allocation with policy and stream
    template <allocatorPolicy policy>
    [[nodiscard]] static T* allocate(std::size_t n,
                                     const Stream& stream = Stream()) noexcept {
        if constexpr (policy == allocatorPolicy::Device)
            return allocateDevice<T>(n, stream.get());
        else if constexpr (policy == allocatorPolicy::Host)
            return allocateHost<T>(n, stream.get());
        else if constexpr (policy == allocatorPolicy::Pinned)
            return allocatePinned<T>(n, stream.get());
    }

    // Instance allocation with policy and stream
    [[nodiscard]] T* allocate(std::size_t n, allocatorPolicy policy,
                              const Stream& stream = Stream()) noexcept {
        if (policy == allocatorPolicy::Device)
            return allocateDevice<T>(n, stream.get());
        else if (policy == allocatorPolicy::Host)
            return allocateHost<T>(n, stream.get());
        else if (policy == allocatorPolicy::Pinned)
            return allocatePinned<T>(n, stream.get());
    }

    // Static deallocation with policy and stream
    template <allocatorPolicy policy>
    static void deallocate(T* ptr, const Stream& stream = Stream()) noexcept {
        if constexpr (policy == allocatorPolicy::Device)
            deallocateDevice<T>(ptr, stream.get());
        else if constexpr (policy == allocatorPolicy::Host)
            deallocateHost<T>(ptr);
        else if constexpr (policy == allocatorPolicy::Pinned)
            deallocatePinned<T>(ptr);
    }
};

template <typename T>
struct DeviceDeleter {
    void operator()(T* ptr) const noexcept {
        if (ptr) allocator<T>::template deallocate<allocatorPolicy::Device>(ptr);
    }
};

template <typename T>
struct HostDeleter {
    void operator()(T* ptr) const noexcept {
        if (ptr) {
            auto err = allocator<T>::template deallocate<allocatorPolicy::Host>(ptr);
            if (err != cudaSuccess)
                BENCHTOOLS_CRITICAL("HostDeleter: deallocation failed "s +
                                    CUDALERN_ERROR(err));
        }
    }
};

template <typename T>
struct PinnedDeleter {
    void operator()(T* ptr) const noexcept {
        if (ptr) {
            auto err = allocator<T>::template deallocate<allocatorPolicy::Pinned>(ptr);
            if (err != cudaSuccess)
                BENCHTOOLS_CRITICAL("PinnedDeleter: deallocation failed "s +
                                    CUDALERN_ERROR(err));
        }
    }
};

}  // namespace cudalern