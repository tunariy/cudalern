#pragma once

#include "cudalern/Containers/NdArray.hpp"
#include <cudalern/Core/core.cuh>

#include <driver_types.h>

#include <type_traits>

namespace cudalern {

template <class T, std::size_t Rank, class KernelFunc>
    requires(std::is_function_v<KernelFunc>)
NdArray<T, Rank> arrayOp(NdArray<T, Rank> in1, NdArray<T, Rank> in2,
                         std::size_t size) noexcept {
    auto ptr1 = in1.get().lock(), ptr2 = in2.get().lock();

    

    return cudaSuccess;
}

template <class T, std::size_t Rank, class KernelFunc>
    requires(std::is_function_v<KernelFunc>)
NdArray<T, Rank> scalarOp(NdArray<T, Rank> in1, T in2, std::size_t size) noexcept {
    return cudaSuccess;
}
}  // namespace cudalern