#pragma once

#include "cudalern/ABI/memory/memory.hpp"
#include "cudalern/Core/err.hpp"
#include <cstring>
#include <cudalern/ABI/kernel/kernel.cuh>
#include <cudalern/ABI/kernel/kernel_wrappers.hpp>
#include <cudalern/ABI/memory/allocator.hpp>
#include <cudalern/ABI/memory/stream.hpp>

#include <cudalern/Core/Device/device.hpp>
#include <cudalern/Core/core.hpp>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace cudalern {

template <class T, std::size_t Rank>
class NdView {
    std::weak_ptr<T> m_Data;
    Stream m_Stream;
    std::size_t m_Offset;
    std::array<std::size_t, Rank> m_Dimensions;
    std::array<std::size_t, Rank> m_Strides;

  public:
    NdView() = default;

    explicit NdView(std::shared_ptr<T> data, std::size_t offset,
                    const std::array<std::size_t, Rank>& dims,
                    const std::array<std::size_t, Rank>& strides)
        : m_Data(data), m_Offset(offset), m_Dimensions(dims), m_Strides(strides) {}

    [[nodiscard]] NdView<T, Rank - 1> operator[](std::size_t idx) const noexcept {
        static_assert(Rank > 0);
        if (idx >= m_Dimensions[0]) return {};

        std::size_t newOffset = m_Offset + idx * m_Strides[0];
        std::array<std::size_t, Rank - 1> newDims;
        std::array<std::size_t, Rank - 1> newStrides;
        for (std::size_t i = 1; i < Rank; ++i) {
            newDims[i - 1] = m_Dimensions[i];
            newStrides[i - 1] = m_Strides[i];
        }
        return NdView<T, Rank - 1>(m_Data.lock(), newOffset, newDims, newStrides);
    }

    [[nodiscard]] const auto& dims() const noexcept { return m_Dimensions; }
    [[nodiscard]] const auto& strides() const noexcept { return m_Strides; }
    [[nodiscard]] std::size_t offset() const noexcept { return m_Offset; }
    [[nodiscard]] std::shared_ptr<T> getShared() const { return m_Data.lock(); }
};

template <class T>
class NdView<T, 0> {
    std::weak_ptr<T> m_Data;
    Stream m_Stream;
    std::size_t m_Offset;

  public:
    NdView() = default;

    explicit NdView(std::shared_ptr<T> data, std::size_t offset, ...)
        : m_Data(data), m_Offset(offset) {}

    T get() const {
        auto ptr = m_Data.lock();
        if (!ptr) return T{};
        T val;
        auto err =
            memcpy(&val, ptr.get() + m_Offset, 1, memcpyKind::DeviceToHost, m_Stream);
        if (err) return T{};
        err = m_Stream.synchronize();
        return val;
    }

    void set(const T& val) const {
        auto ptr = m_Data.lock();
        if (!ptr) return;
        auto err =
            memcpy(ptr.get() + m_Offset, &val, 1, memcpyKind::HostToDevice, m_Stream);
        if (!err) err = m_Stream.synchronize();
    }

    operator T() const { return get(); }

    NdView& operator=(const T& val) {
        set(val);
        return *this;
    }
};

template <class T, std::size_t Rank>
    requires(CudaCompatible<T>)
class NdArray {
    static_assert(Rank > 0, "Use NdArray<T, 0> specialization for scalars!");

    std::shared_ptr<T> m_Data{};
    Stream m_Stream{};
    std::array<std::size_t, Rank> m_Dimensions{};
    std::array<std::size_t, Rank> m_Strides{};
    std::size_t m_Size{0};

  public:
    NdArray() = default;

    ~NdArray() noexcept { cleanup(); }

    /**
     * Construct from dimensions.
     * @param dims Dimensions array.
     */
    explicit NdArray(const std::array<std::size_t, Rank>& dims) noexcept
        : m_Dimensions(dims) {
        if (m_Size != 0) cleanup();

        m_Size = 1;
        for (const auto d : m_Dimensions)
            m_Size *= d;

        this->computeStrides();

        m_Data = std::shared_ptr<T>(
            allocator<T>::template allocate<allocatorPolicy::Device>(m_Size, m_Stream),
            DeviceDeleter<T>(m_Stream));
        this->synchronize();
    }

    /**
     * Construct from existing shared data.
     * @param data Shared pointer to device data.
     * @param dims Dimensions array.
     */
    explicit NdArray(std::shared_ptr<T> data,
                     const std::array<std::size_t, Rank>& dims) noexcept
        : m_Data(std::move(data)), m_Dimensions(dims) {
        if (m_Size != 0) cleanup();

        m_Size = 1;
        for (const auto d : m_Dimensions)
            m_Size *= d;

        this->computeStrides();
    }

    /**
     * Deep-copy from an NdView.
     * @param view Source view to copy.
     */
    explicit NdArray(const NdView<T, Rank>& view) noexcept {
        auto srcData = view.getShared();
        if (!srcData) {
            CUDALERN_CRITICAL(
                "Cannot construct NdArray from NdView: source data has expired");
            return;
        }

        if (m_Size) this->cleanup();

        m_Dimensions = view.dims();
        m_Stream = view.stream();

        m_Size = 1;
        for (const auto d : m_Dimensions)
            m_Size *= d;

        this->computeStrides();

        m_Data = std::shared_ptr<T>(
            allocator<T>::template allocate<allocatorPolicy::Device>(m_Size, m_Stream),
            DeviceDeleter<T>(m_Stream));

        [[maybe_unused]] auto err = memcpy(m_Data.get(), srcData.get() + view.offset(),
                                           m_Size, memcpyKind::DeviceToDevice, m_Stream);
        this->synchronize();
    }

    /**
     * Construct from nested initializer sequences.
     * @tparam Sequence_t Pack of iterable sequences.
     */
    template <class... Sequence_t>
        requires((sizeof...(Sequence_t) <= Rank && ElementIterable<Sequence_t>) && ...)
    explicit NdArray(Sequence_t... args) noexcept {
        static_assert(Rank > 0, "NdArray must have at least one dimension");

        if (m_Size != 0) this->cleanup();

        if constexpr (sizeof...(Sequence_t) == 1) {
            this->deduceDimensions<0>(std::get<0>(std::tuple(args...)), m_Dimensions);
        } else {
            m_Dimensions[0] = sizeof...(Sequence_t);

            auto first_arg = std::get<0>(std::tuple(args...));
            this->deduceDimensions<1>(first_arg, m_Dimensions);
        }

        this->computeStrides();

        m_Size = 1;
        for (std::size_t d : m_Dimensions)
            m_Size *= d;

        m_Data = std::shared_ptr<T>(
            allocator<T>::template allocate<allocatorPolicy::Device>(m_Size, m_Stream),
            DeviceDeleter<T>(m_Stream));

        std::vector<T> hostData(m_Size);
        std::size_t offset = 0;

        if constexpr (sizeof...(Sequence_t) == 1) {
            auto tup = std::tuple(args...);
            decomposeRanges<0>(std::get<0>(tup), hostData, offset);
        } else {
            if (sizeof...(Sequence_t) != m_Dimensions[0])
                CUDALERN_CRITICAL("Number of arguments does not match first dimension");
            ([&](const auto& arg) { decomposeRanges<1>(arg, hostData, offset); }(args),
             ...);
        }

        if (offset != m_Size) CUDALERN_CRITICAL("Element count mismatch!");

        [[maybe_unused]] auto err = memcpy(m_Data.get(), hostData.data(), m_Size,
                                           memcpyKind::HostToDevice, m_Stream);
        this->synchronize();
    }

    NdArray(const NdArray<T, Rank>& rhs) {
        if (this == &rhs) return;
        if (!rhs.m_Size) return;

        if (m_Size) this->cleanup();

        m_Size = rhs.m_Size;
        m_Stream = rhs.m_Stream;
        m_Dimensions = rhs.m_Dimensions;
        m_Strides = rhs.m_Strides;

        m_Data = std::shared_ptr<T>(
            allocator<T>::template allocate<allocatorPolicy::Device>(m_Size, m_Stream),
            DeviceDeleter<T>(m_Stream));

        [[maybe_unused]] auto err = memcpy(m_Data.get(), rhs.m_Data.get(), m_Size,
                                           memcpyKind::DeviceToDevice, m_Stream);
        this->synchronize();
    };

    NdArray& operator=(const NdArray<T, Rank>& rhs) {
        if (this == &rhs) return *this;
        if (!rhs.m_Size) return *this;

        if (m_Size) cleanup();

        m_Size = rhs.m_Size;
        m_Stream = rhs.m_Stream;
        m_Dimensions = rhs.m_Dimensions;
        m_Strides = rhs.m_Strides;

        m_Data = std::shared_ptr<T>(
            allocator<T>::template allocate<allocatorPolicy::Device>(m_Size, m_Stream),
            DeviceDeleter<T>(m_Stream));

        [[maybe_unused]] auto err = memcpy(m_Data.get(), rhs.m_Data.get(), m_Size,
                                           memcpyKind::DeviceToDevice, m_Stream);
        this->synchronize();

        return *this;
    }

    NdArray(NdArray<T, Rank>&& rhs) noexcept {
        if (!rhs.m_Size) return;

        m_Size = std::move(rhs.m_Size);
        m_Stream = std::move(rhs.m_Stream);
        m_Dimensions = std::move(rhs.m_Dimensions);
        m_Strides = std::move(rhs.m_Strides);

        m_Data = std::move(rhs.m_Data);

        rhs.m_Data.reset();
        rhs.m_Size = 0;
        rhs.m_Dimensions.fill(0);

        this->synchronize();
    }

    NdArray& operator=(NdArray<T, Rank>&& rhs) noexcept {
        if (!rhs.m_Size) return *this;

        m_Size = std::move(rhs.m_Size);
        m_Stream = std::move(rhs.m_Stream);
        m_Dimensions = std::move(rhs.m_Dimensions);
        m_Strides = std::move(rhs.m_Strides);

        m_Data = std::move(rhs.m_Data);

        rhs.m_Data.reset();
        rhs.m_Size = 0;
        rhs.m_Dimensions.fill(0);

        this->synchronize();
        return *this;
    }

  public:
    /**
     * Batched matrix multiplication.
     * @tparam diffRank Must equal Rank and be at least 2.
     */
    template <std::size_t diffRank>
        requires(Rank == diffRank && Rank >= 2)
    NdArray<T, Rank> operator*(const NdArray<T, diffRank>& rhs) noexcept {
        static_assert(Rank >= 2);

        if (!m_Data || !rhs.m_Data) return NdArray<T, Rank>();

        if constexpr (Rank > 2) {
            for (std::size_t d = 0; d < Rank - 2; ++d)
                if (m_Dimensions[d] != rhs.m_Dimensions[d]) return NdArray<T, Rank>();
        }

        size_t M = m_Dimensions[Rank - 2];
        size_t K = m_Dimensions[Rank - 1];
        size_t N = rhs.m_Dimensions[Rank - 1];
        if (K != rhs.m_Dimensions[Rank - 2]) return NdArray<T, Rank>();

        std::array<size_t, Rank> result_dims = m_Dimensions;
        result_dims[Rank - 2] = M;
        result_dims[Rank - 1] = N;

        std::array<size_t, Rank> result_strides;
        size_t stride = 1;
        for (int d = static_cast<int>(Rank) - 1; d >= 0; --d) {
            result_strides[d] = stride;
            stride *= result_dims[d];
        }

        size_t batch_count = 1;
        if constexpr (Rank > 2) {
            for (size_t d = 0; d < static_cast<int>(Rank) - 2; ++d)
                batch_count *= m_Dimensions[d];
        }

        size_t totalElems = batch_count * M * N;
        if (totalElems == 0) return NdArray<T, Rank>();

        size_t* d_dimsA{};
        size_t* d_dimsB{};
        size_t* d_dimsC{};
        size_t* d_stridesA{};
        size_t* d_stridesB{};
        size_t* d_stridesC{};

        auto cleanupDeviceDim = [this, &d_dimsA, &d_dimsB, &d_dimsC, &d_stridesA,
                                 &d_stridesB, &d_stridesC]() {
            allocator<size_t>::template deallocate<allocatorPolicy::Device>(d_dimsA,
                                                                            m_Stream);
            allocator<size_t>::template deallocate<allocatorPolicy::Device>(d_dimsB,
                                                                            m_Stream);
            allocator<size_t>::template deallocate<allocatorPolicy::Device>(d_dimsC,
                                                                            m_Stream);
            allocator<size_t>::template deallocate<allocatorPolicy::Device>(d_stridesA,
                                                                            m_Stream);
            allocator<size_t>::template deallocate<allocatorPolicy::Device>(d_stridesB,
                                                                            m_Stream);
            allocator<size_t>::template deallocate<allocatorPolicy::Device>(d_stridesC,
                                                                            m_Stream);
        };

        this->synchronize();

        auto allocateCopy = [&](size_t*& dst,
                                const std::array<size_t, Rank>& src) -> cudalernErr {
            dst = allocator<size_t>::template allocate<allocatorPolicy::Device>(Rank,
                                                                                m_Stream);
            if (!dst) return cudaErrorUnknown;

            if (memcpy(dst, src.data(), Rank, memcpyKind::HostToDevice, m_Stream)) {
                allocator<size_t>::template deallocate<allocatorPolicy::Device>(dst,
                                                                                m_Stream);
                dst = nullptr;
                return cudaErrorUnknown;
            }
            return cudaSuccess;
        };

        allocateCopy(d_dimsA, m_Dimensions);
        allocateCopy(d_dimsB, rhs.m_Dimensions);
        allocateCopy(d_dimsC, result_dims);
        allocateCopy(d_stridesA, m_Strides);
        allocateCopy(d_stridesB, rhs.m_Strides);
        allocateCopy(d_stridesC, result_strides);

        this->synchronize();

        T* d_result = allocator<T>::template allocate<allocatorPolicy::Device>(totalElems,
                                                                               m_Stream);
        if (!d_result) {
            cleanupDeviceDim();
            return NdArray<T, Rank>();
        }

        std::shared_ptr<T> resultPtr(d_result, DeviceDeleter<T>(m_Stream));
        NdArray<T, Rank> result(resultPtr, result_dims);

        cudalernErr launchErr = launchKernel1D(
            kernel::batched_matmul_kernel<T, Rank>, 0, m_Stream, totalElems, m_Data.get(),
            rhs.m_Data.get(), resultPtr.get(), d_dimsA, d_dimsB, d_dimsC, d_stridesA,
            d_stridesB, d_stridesC, M, K, N, totalElems);

        this->synchronize();

        cleanupDeviceDim();

        this->synchronize();
        result.synchronize();

        if (launchErr != cudaSuccess) return NdArray<T, Rank>();

        return result;
    }

    /**
     * Element-wise negation.
     * @return Result array.
     */
    NdArray<T, Rank> operator-() const noexcept {
        if (!m_Data || m_Size == 0) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err = launchKernel1D(kernel::negate_kernel<T>, 0, m_Stream, m_Size,
                                         result.m_Data.get(), m_Data.get(),
                                         static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();

        this->synchronize();
        result.synchronize();

        return result;
    }

    /**
     * Element-wise addition with scalar.
     * @param val Scalar value.
     * @return Result array.
     */
    NdArray<T, Rank> operator+(const T& val) const noexcept {
        if (!m_Data || m_Size == 0) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err = launchKernel1D(
            (void (*)(T*, const T*, T, uint32_t))kernel::add<T>, 0, m_Stream, m_Size,
            result.m_Data.get(), m_Data.get(), val, static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();

        this->synchronize();
        return result;
    }

    /**
     * Element-wise subtraction with scalar.
     * @param val Scalar value.
     * @return Result array.
     */
    NdArray<T, Rank> operator-(const T& val) const noexcept {
        if (!m_Data || m_Size == 0) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err = launchKernel1D(
            (void (*)(T*, const T*, T, uint32_t))kernel::sub<T>, 0, m_Stream, m_Size,
            result.m_Data.get(), m_Data.get(), val, static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();

        this->synchronize();
        result.synchronize();

        return result;
    }

    /**
     * Element-wise multiplication with scalar.
     * @param val Scalar value.
     * @return Result array.
     */
    NdArray<T, Rank> operator*(const T& val) const noexcept {
        if (!m_Data || m_Size == 0) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err = launchKernel1D(
            (void (*)(T*, const T*, T, uint32_t))kernel::mul<T>, 0, m_Stream, m_Size,
            result.m_Data.get(), m_Data.get(), val, static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();

        this->synchronize();
        result.synchronize();

        return result;
    }

    /**
     * Element-wise division with scalar.
     * @param val Scalar value.
     * @return Result array.
     */
    NdArray<T, Rank> operator/(const T& val) const noexcept {
        if (!m_Data || m_Size == 0) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err = launchKernel1D(
            (void (*)(T*, const T*, T, uint32_t))kernel::div<T>, 0, m_Stream, m_Size,
            result.m_Data.get(), m_Data.get(), val, static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();

        this->synchronize();
        result.synchronize();

        return result;
    }

    /**
     * Element-wise addition.
     * @param rhs Right-hand array.
     * @return Result array.
     */
    NdArray<T, Rank> operator+(const NdArray<T, Rank>& rhs) const noexcept {
        if (!m_Data || !rhs.m_Data || m_Size != rhs.m_Size) return NdArray<T, Rank>();
        for (size_t d = 0; d < Rank; ++d)
            if (m_Dimensions[d] != rhs.m_Dimensions[d]) return NdArray<T, Rank>();

        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err =
            launchKernel1D((void (*)(T*, const T*, const T*, uint32_t))kernel::add<T>, 0,
                           m_Stream, m_Size, result.m_Data.get(), m_Data.get(),
                           rhs.m_Data.get(), static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();
        return result;
    }

    /**
     * Element-wise subtraction.
     * @param rhs Right-hand array.
     * @return Result array.
     */
    NdArray<T, Rank> operator-(const NdArray<T, Rank>& rhs) const noexcept {
        if (!m_Data || !rhs.m_Data || m_Size != rhs.m_Size) return NdArray<T, Rank>();
        for (size_t d = 0; d < Rank; ++d)
            if (m_Dimensions[d] != rhs.m_Dimensions[d]) return NdArray<T, Rank>();

        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err =
            launchKernel1D((void (*)(T*, const T*, const T*, uint32_t))kernel::sub<T>, 0,
                           m_Stream, m_Size, result.m_Data.get(), m_Data.get(),
                           rhs.m_Data.get(), static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();

        this->synchronize();
        result.synchronize();

        return result;
    }

    /**
     * Element-wise multiplication.
     * @param rhs Right-hand array.
     * @return Result array.
     */
    NdArray<T, Rank> operator%(const NdArray<T, Rank>& rhs) const noexcept {
        if (!m_Data || !rhs.m_Data || m_Size != rhs.m_Size) return NdArray<T, Rank>();
        for (size_t d = 0; d < Rank; ++d)
            if (m_Dimensions[d] != rhs.m_Dimensions[d]) return NdArray<T, Rank>();

        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err =
            launchKernel1D((void (*)(T*, const T*, const T*, uint32_t))kernel::mul<T>, 0,
                           m_Stream, m_Size, result.m_Data.get(), m_Data.get(),
                           rhs.m_Data.get(), static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();

        this->synchronize();
        result.synchronize();

        return result;
    }

    /**
     * Vector element-wise multiplication (Rank 1).
     * @tparam diffRank Must equal Rank and be 1.
     */
    template <std::size_t diffRank>
        requires(Rank == diffRank && Rank == 1)
    NdArray<T, Rank> operator*(const NdArray<T, diffRank>& rhs) const noexcept {
        return (*this) % rhs;
    }

    /**
     * Element-wise division.
     * @param rhs Right-hand array.
     * @return Result array.
     */
    NdArray<T, Rank> operator/(const NdArray<T, Rank>& rhs) const noexcept {
        if (!m_Data || !rhs.m_Data || m_Size != rhs.m_Size) return NdArray<T, Rank>();
        for (size_t d = 0; d < Rank; ++d)
            if (m_Dimensions[d] != rhs.m_Dimensions[d]) return NdArray<T, Rank>();

        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err =
            launchKernel1D((void (*)(T*, const T*, const T*, uint32_t))kernel::div<T>, 0,
                           m_Stream, m_Size, result.m_Data.get(), m_Data.get(),
                           rhs.m_Data.get(), static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();

        this->synchronize();
        result.synchronize();

        return result;
    }

    /**
     * In-place scalar addition.
     * @param val Scalar value.
     */
    void operator+=(const T& val) const noexcept {
        if (!m_Data || m_Size == 0) return;
        cudalernErr err = launchKernel1D(
            (void (*)(T*, const T*, T, uint32_t))kernel::add<T>, 0, m_Stream, m_Size,
            m_Data.get(), m_Data.get(), val, static_cast<uint32_t>(m_Size));

        if (err) CUDALERN_ERR("In-place scalar addition failed");
        this->synchronize();
    }

    /**
     * In-place scalar subtraction.
     * @param val Scalar value.
     */
    void operator-=(const T& val) const noexcept {
        if (!m_Data || m_Size == 0) return;
        cudalernErr err = launchKernel1D(
            (void (*)(T*, const T*, T, uint32_t))kernel::sub<T>, 0, m_Stream, m_Size,
            m_Data.get(), m_Data.get(), val, static_cast<uint32_t>(m_Size));
        if (err) CUDALERN_ERR("In-place scalar subtraction failed");
        this->synchronize();
    }

    /**
     * In-place scalar multiplication.
     * @param val Scalar value.
     */
    void operator*=(const T& val) const noexcept {
        if (!m_Data || m_Size == 0) return;
        cudalernErr err = launchKernel1D(
            (void (*)(T*, const T*, T, uint32_t))kernel::mul<T>, 0, m_Stream, m_Size,
            m_Data.get(), m_Data.get(), val, static_cast<uint32_t>(m_Size));
        if (err) CUDALERN_ERR("In-place scalar multiplication failed");
        this->synchronize();
    }

    /**
     * In-place scalar division.
     * @param val Scalar value.
     */
    void operator/=(const T& val) const noexcept {
        if (!m_Data || m_Size == 0) return;
        cudalernErr err = launchKernel1D(
            (void (*)(T*, const T*, T, uint32_t))kernel::div<T>, 0, m_Stream, m_Size,
            m_Data.get(), m_Data.get(), val, static_cast<uint32_t>(m_Size));

        if (err) CUDALERN_ERR("In-place scalar division failed");
        this->synchronize();
    }

    /**
     * In-place element-wise addition.
     * @param rhs Right-hand array.
     */
    void operator+=(const NdArray<T, Rank>& rhs) const noexcept {
        if (!m_Data || !rhs.m_Data || m_Size != rhs.m_Size) {
            CUDALERN_CRITICAL("Shape mismatch for in-place add");
            return;
        }
        for (size_t d = 0; d < Rank; ++d)
            if (m_Dimensions[d] != rhs.m_Dimensions[d]) {
                CUDALERN_CRITICAL("Dimension mismatch for in-place add");
                return;
            }
        cudalernErr err =
            launchKernel1D((void (*)(T*, const T*, const T*, uint32_t))kernel::add<T>, 0,
                           m_Stream, m_Size, m_Data.get(), m_Data.get(), rhs.m_Data.get(),
                           static_cast<uint32_t>(m_Size));

        if (err) CUDALERN_ERR("In-place array addition failed");
        this->synchronize();
    }

    /**
     * In-place element-wise subtraction.
     * @param rhs Right-hand array.
     */
    void operator-=(const NdArray<T, Rank>& rhs) const noexcept {
        if (!m_Data || !rhs.m_Data || m_Size != rhs.m_Size) {
            CUDALERN_CRITICAL("Shape mismatch for in-place sub");
            return;
        }
        for (size_t d = 0; d < Rank; ++d)
            if (m_Dimensions[d] != rhs.m_Dimensions[d]) {
                CUDALERN_CRITICAL("Dimension mismatch for in-place sub");
                return;
            }
        cudalernErr err =
            launchKernel1D((void (*)(T*, const T*, const T*, uint32_t))kernel::sub<T>, 0,
                           m_Stream, m_Size, m_Data.get(), m_Data.get(), rhs.m_Data.get(),
                           static_cast<uint32_t>(m_Size));

        if (err) CUDALERN_ERR("In-place array subtraction failed");
        this->synchronize();
    }

    /**
     * In-place element-wise multiplication.
     * @param rhs Right-hand array.
     */
    void operator%=(const NdArray<T, Rank>& rhs) const noexcept {
        if (!m_Data || !rhs.m_Data || m_Size != rhs.m_Size) {
            CUDALERN_CRITICAL("Shape mismatch for in-place mul");
            return;
        }
        for (size_t d = 0; d < Rank; ++d)
            if (m_Dimensions[d] != rhs.m_Dimensions[d]) {
                CUDALERN_CRITICAL("Dimension mismatch for in-place mul");
                return;
            }
        cudalernErr err =
            launchKernel1D((void (*)(T*, const T*, const T*, uint32_t))kernel::mul<T>, 0,
                           m_Stream, m_Size, m_Data.get(), m_Data.get(), rhs.m_Data.get(),
                           static_cast<uint32_t>(m_Size));

        if (err) CUDALERN_ERR("In-place array multiplication failed");
        this->synchronize();
    }

    /**
     * In-place element-wise multiplication (alias).
     * @param rhs Right-hand array.
     */
    void operator*=(const NdArray<T, Rank>& rhs) const noexcept { (*this) %= rhs; }

    /**
     * In-place element-wise division.
     * @param rhs Right-hand array.
     */
    void operator/=(const NdArray<T, Rank>& rhs) const noexcept {
        if (!m_Data || !rhs.m_Data || m_Size != rhs.m_Size) {
            CUDALERN_CRITICAL("Shape mismatch for in-place div");
            return;
        }
        for (size_t d = 0; d < Rank; ++d)
            if (m_Dimensions[d] != rhs.m_Dimensions[d]) {
                CUDALERN_CRITICAL("Dimension mismatch for in-place div");
                return;
            }
        cudalernErr err =
            launchKernel1D((void (*)(T*, const T*, const T*, uint32_t))kernel::div<T>, 0,
                           m_Stream, m_Size, m_Data.get(), m_Data.get(), rhs.m_Data.get(),
                           static_cast<uint32_t>(m_Size));
        if (err) CUDALERN_ERR("In-place array division failed");
    }

    /**
     * Element-wise equality comparison.
     * @param rhs Right-hand array.
     * @return Result array.
     */
    NdArray<T, Rank> operator==(const NdArray<T, Rank>& rhs) const noexcept {
        if (!m_Data || !rhs.m_Data || m_Size != rhs.m_Size) return NdArray<T, Rank>();
        for (size_t d = 0; d < Rank; ++d)
            if (m_Dimensions[d] != rhs.m_Dimensions[d]) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err =
            launchKernel1D(kernel::eq_kernel<T>, 0, m_Stream, m_Size, result.m_Data.get(),
                           m_Data.get(), rhs.m_Data.get(), static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();

        this->synchronize();
        result.synchronize();

        return result;
    }

    /**
     * Element-wise inequality comparison.
     * @param rhs Right-hand array.
     * @return Result array.
     */
    NdArray<T, Rank> operator!=(const NdArray<T, Rank>& rhs) const noexcept {
        if (!m_Data || !rhs.m_Data || m_Size != rhs.m_Size) return NdArray<T, Rank>();
        for (size_t d = 0; d < Rank; ++d)
            if (m_Dimensions[d] != rhs.m_Dimensions[d]) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err =
            launchKernel1D(kernel::ne_kernel<T>, 0, m_Stream, m_Size, result.m_Data.get(),
                           m_Data.get(), rhs.m_Data.get(), static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();

        this->synchronize();
        result.synchronize();

        return result;
    }

    /**
     * Element-wise less-than comparison.
     * @param rhs Right-hand array.
     * @return Result array.
     */
    NdArray<T, Rank> operator<(const NdArray<T, Rank>& rhs) const noexcept {
        if (!m_Data || !rhs.m_Data || m_Size != rhs.m_Size) return NdArray<T, Rank>();
        for (size_t d = 0; d < Rank; ++d)
            if (m_Dimensions[d] != rhs.m_Dimensions[d]) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err =
            launchKernel1D(kernel::lt_kernel<T>, 0, m_Stream, m_Size, result.m_Data.get(),
                           m_Data.get(), rhs.m_Data.get(), static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();

        this->synchronize();
        result.synchronize();

        return result;
    }

    /**
     * Element-wise greater-than comparison.
     * @param rhs Right-hand array.
     * @return Result array.
     */
    NdArray<T, Rank> operator>(const NdArray<T, Rank>& rhs) const noexcept {
        if (!m_Data || !rhs.m_Data || m_Size != rhs.m_Size) return NdArray<T, Rank>();
        for (size_t d = 0; d < Rank; ++d)
            if (m_Dimensions[d] != rhs.m_Dimensions[d]) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err =
            launchKernel1D(kernel::gt_kernel<T>, 0, m_Stream, m_Size, result.m_Data.get(),
                           m_Data.get(), rhs.m_Data.get(), static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();

        this->synchronize();
        result.synchronize();

        return result;
    }

    /**
     * Element-wise less-than-or-equal comparison.
     * @param rhs Right-hand array.
     * @return Result array.
     */
    NdArray<T, Rank> operator<=(const NdArray<T, Rank>& rhs) const noexcept {
        if (!m_Data || !rhs.m_Data || m_Size != rhs.m_Size) return NdArray<T, Rank>();
        for (size_t d = 0; d < Rank; ++d)
            if (m_Dimensions[d] != rhs.m_Dimensions[d]) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err =
            launchKernel1D(kernel::le_kernel<T>, 0, m_Stream, m_Size, result.m_Data.get(),
                           m_Data.get(), rhs.m_Data.get(), static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();

        this->synchronize();
        result.synchronize();

        return result;
    }

    /**
     * Element-wise greater-than-or-equal comparison.
     * @param rhs Right-hand array.
     * @return Result array.
     */
    NdArray<T, Rank> operator>=(const NdArray<T, Rank>& rhs) const noexcept {
        if (!m_Data || !rhs.m_Data || m_Size != rhs.m_Size) return NdArray<T, Rank>();
        for (size_t d = 0; d < Rank; ++d)
            if (m_Dimensions[d] != rhs.m_Dimensions[d]) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err =
            launchKernel1D(kernel::ge_kernel<T>, 0, m_Stream, m_Size, result.m_Data.get(),
                           m_Data.get(), rhs.m_Data.get(), static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();

        this->synchronize();
        result.synchronize();

        return result;
    }

    /**
     * Element-wise power.
     * @param exponent Exponent value.
     * @return Result array.
     */
    NdArray<T, Rank> pow(T exponent) const noexcept {
        if (!m_Data || m_Size == 0) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err = launchKernel1D(kernel::pow_kernel<T>, 0, m_Stream, m_Size,
                                         result.m_Data.get(), m_Data.get(), exponent,
                                         static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();

        this->synchronize();
        result.synchronize();

        return result;
    }

    /**
     * Element-wise exponential.
     * @return Result array.
     */
    NdArray<T, Rank> exp() const noexcept {
        if (!m_Data || m_Size == 0) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err = launchKernel1D(kernel::exp_kernel<T>, 0, m_Stream, m_Size,
                                         result.m_Data.get(), m_Data.get(),
                                         static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();

        this->synchronize();
        result.synchronize();

        return result;
    }

    /**
     * Element-wise natural logarithm.
     * @return Result array.
     */
    NdArray<T, Rank> log() const noexcept {
        if (!m_Data || m_Size == 0) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err = launchKernel1D(kernel::log_kernel<T>, 0, m_Stream, m_Size,
                                         result.m_Data.get(), m_Data.get(),
                                         static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();

        this->synchronize();
        result.synchronize();

        return result;
    }

    /**
     * Element-wise square root.
     * @return Result array.
     */
    NdArray<T, Rank> sqrt() const noexcept {
        if (!m_Data || m_Size == 0) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err = launchKernel1D(kernel::sqrt_kernel<T>, 0, m_Stream, m_Size,
                                         result.m_Data.get(), m_Data.get(),
                                         static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();

        this->synchronize();
        result.synchronize();

        return result;
    }

    /**
     * Element-wise absolute value.
     * @return Result array.
     */
    NdArray<T, Rank> abs() const noexcept {
        if (!m_Data || m_Size == 0) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err = launchKernel1D(kernel::abs_kernel<T>, 0, m_Stream, m_Size,
                                         result.m_Data.get(), m_Data.get(),
                                         static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();
        return result;
    }

    /**
     * Element-wise ReLU activation.
     * @return Result array.
     */
    NdArray<T, Rank> relu() const noexcept {
        if (!m_Data || m_Size == 0) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err =
            launchKernel1D(kernel::relu<T>, 0, m_Stream, m_Size, result.m_Data.get(),
                           m_Data.get(), static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();

        this->synchronize();
        result.synchronize();

        return result;
    }

    /**
     * Element-wise leaky ReLU activation.
     * @param alpha Negative slope.
     * @return Result array.
     */
    NdArray<T, Rank> leaky_relu(T alpha = static_cast<T>(0.01)) const noexcept {
        if (!m_Data || m_Size == 0) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err = launchKernel1D(kernel::leaky_relu<T>, 0, m_Stream, m_Size,
                                         result.m_Data.get(), m_Data.get(), alpha,
                                         static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();

        this->synchronize();
        result.synchronize();

        return result;
    }

    /**
     * Element-wise sigmoid activation.
     * @return Result array.
     */
    NdArray<T, Rank> sigmoid() const noexcept {
        if (!m_Data || m_Size == 0) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err =
            launchKernel1D(kernel::sigmoid<T>, 0, m_Stream, m_Size, result.m_Data.get(),
                           m_Data.get(), static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();

        this->synchronize();
        result.synchronize();

        return result;
    }

    /**
     * Element-wise hyperbolic tangent.
     * @return Result array.
     */
    NdArray<T, Rank> tanh() const noexcept {
        if (!m_Data || m_Size == 0) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err = launchKernel1D(kernel::tanh_kernel<T>, 0, m_Stream, m_Size,
                                         result.m_Data.get(), m_Data.get(),
                                         static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();
        return result;
    }

  public:
    /**
     * Read element at indices.
     * @tparam Args Index types.
     * @return Element value.
     */
    template <class... Args>
        requires(sizeof...(Args) == Rank)
    T operator()(Args... dims) const {
        return read(
            std::array<std::size_t, sizeof...(Args)>{static_cast<std::size_t>(dims)...});
    }

    /**
     * Index into the first dimension.
     * @param idx Index.
     * @return Sub-dimensional view.
     */
    NdView<T, Rank - 1> operator[](std::size_t idx) {
        std::array<std::size_t, Rank - 1> newDims;
        std::array<std::size_t, Rank - 1> newStrides;
        for (std::size_t i = 1; i < Rank; ++i) {
            newDims[i - 1] = m_Dimensions[i];
            newStrides[i - 1] = m_Strides[i];
        }

        std::size_t newOffset = idx * m_Strides[0];
        return NdView<T, Rank - 1>(m_Data, newOffset, newDims, newStrides);
    }

    /**
     * Read element at indices.
     * @tparam Args Index types.
     * @return Element value.
     */
    template <class... Args>
        requires(sizeof...(Args) == Rank)
    [[nodiscard]] T read(Args... dims) const {
        return read(
            std::array<std::size_t, sizeof...(Args)>{static_cast<std::size_t>(dims)...});
    }

    /**
     * Read element at coordinate array.
     * @param dims Coordinate indices.
     * @return Element value.
     */
    [[nodiscard]] T read(std::array<std::size_t, Rank> dims) const {
        std::size_t offset{};
        for (std::size_t i = 0; i < Rank; ++i) {
            offset += m_Strides[i] * dims[i];
        }

        T temp;
        [[maybe_unused]] auto err =
            memcpy(&temp, m_Data.get() + offset, 1, memcpyKind::DeviceToHost, m_Stream);

        this->synchronize();

        return temp;
    }

    /**
     * Copy all data to host vector.
     * @return Host vector copy.
     */
    [[nodiscard]] std::vector<T> data() const {
        std::unique_ptr<T[]> temp{new T[m_Size]};
        [[maybe_unused]] auto err =
            memcpy(temp.get(), m_Data.get(), m_Size, memcpyKind::DeviceToHost, m_Stream);

        this->synchronize();

        return std::vector<T>{temp.get(), temp.get() + m_Size};
    }

    /**
     * Copy all data to host vector.
     * @param out Output host vector.
     */
    void to_host(std::vector<T>& out) const noexcept { out = data(); }

    /**
     * Release ownership of device pointer.
     * @return Raw device pointer.
     */
    [[nodiscard]] T* release() noexcept {
        auto temp{m_Data.get()};
        m_Data = std::shared_ptr<T>{nullptr, [](T* ptr) {}};
        m_Size = {};

        return temp;
    }

    /**
     * Synchronize the associated stream.
     */
    void synchronize() const noexcept {
        [[maybe_unused]]
        auto err = m_Stream.synchronize();
        if (err) CUDALERN_ERROR_MESSAGE("Error synchronizing the stream! ", err);
    }

    /**
     * Check if array is empty.
     * @return True if empty.
     */
    [[nodiscard]] bool empty() const noexcept { return m_Size == 0; }

    /**
     * Check if storage is contiguous.
     * @return True if contiguous.
     */
    [[nodiscard]] bool is_contiguous() const {
        std::size_t expected = 1;
        for (int i{Rank - 1}; i >= 0; --i) {
            if (m_Strides[i] != expected) return false;
            expected *= m_Dimensions[i];
        }
        return true;
    }

    /**
     * Total bytes allocated.
     * @return Size in bytes.
     */
    [[nodiscard]] size_t nbytes() const noexcept { return m_Size * sizeof(T); }

    /**
     * Diagnostic string with shape and memory info.
     * @return Info string.
     */
    [[nodiscard]] std::string info() const noexcept {
        std::stringstream ss;

        ss << "Size: " << m_Size << "\n";
        ss << "Taking up " << nbytes() << " bytes of space\n";

        ss << "Dimensions:\n";
        auto iteration{0};
        for (const auto& dim : m_Dimensions) {
            ss << dim << (iteration != m_Dimensions.size() ? "x" : "\n");
            iteration++;
        }

        ss << "Strides:\n";
        iteration = 0;
        for (const auto& dim : m_Strides) {
            ss << dim << (iteration != m_Dimensions.size() ? "x" : "\n");
            iteration++;
        }

        ss << "Memory location: " << "TODO!\n";

        return ss.str();
    }

    [[nodiscard]] std::string print() const {
        if (!m_Size) return "[]";

        T* temp = new T[m_Size];
        auto err = memcpy(temp, m_Data.get(), m_Size, memcpyKind::DeviceToHost, m_Stream);
        if (err) CUDALERN_ERROR_MESSAGE("Failed to read!", err);

        struct Deleter {
            void operator()(T* ptr) { delete[] ptr; }
        };
        std::unique_ptr<T, Deleter> hostData{temp};

        std::ostringstream oss;
        std::size_t flatIdx = 0;

        if constexpr (Rank == 0) {
            oss << hostData.get()[0];
        } else {
            auto printDim = [&](auto& self, std::size_t dim) -> void {
                oss << '[';
                for (std::size_t i = 0; i < m_Dimensions[dim]; ++i) {
                    if (i != 0) {
                        oss << ", ";
                    }
                    if (dim + 1 < Rank) {
                        self(self, dim + 1);
                    } else {
                        oss << hostData.get()[flatIdx++];
                    }
                }
                oss << ']';
            };
            printDim(printDim, 0);
        }

        return oss.str();
    }

  public:
    /**
     * Create array filled with zeros.
     * @param dims Dimensions.
     * @return New array.
     */
    static NdArray zeros(const std::array<size_t, Rank>& dims) noexcept {
        NdArray arr(dims);
        arr.fill(static_cast<T>(0));

        arr.synchronize();
        return arr;
    }

    /**
     * Create array filled with ones.
     * @param dims Dimensions.
     * @return New array.
     */
    static NdArray ones(const std::array<size_t, Rank>& dims) noexcept {
        NdArray arr(dims);
        arr.fill(static_cast<T>(1));

        arr.synchronize();
        return arr;
    }

    /**
     * Create array filled with a constant value.
     * @param dims Dimensions.
     * @param value Fill value.
     * @return New array.
     */
    static NdArray full(const std::array<size_t, Rank>& dims, T value) noexcept {
        NdArray arr(dims);
        arr.fill(value);

        arr.synchronize();
        return arr;
    }

    /**
     * Create 1D range array.
     * @param start Start value.
     * @param stop Stop value.
     * @param step Step size.
     * @return New array.
     */
    static NdArray arange(T start, T stop, T step = 1) noexcept {
        static_assert(Rank == 1, "arange only valid for Rank=1");
        size_t n = static_cast<size_t>((stop - start) / step);
        std::array<size_t, 1> dims = {n};
        NdArray arr(dims);

        [[maybe_unused]] auto err =
            launchKernel1D(kernel::arange<T>, 0, arr.m_Stream, arr.m_Size,
                           arr.m_Data.get(), arr.m_Size, start, step);
        if (err) CUDALERN_ERROR_MESSAGE("Failed to generate arange NdArray", err);
        arr.synchronize();
        return arr;
    }

    /**
     * Create 2D identity matrix.
     * @param n Matrix size.
     * @return New array.
     */
    static NdArray eye(size_t n) noexcept {
        static_assert(Rank == 2, "eye only valid for Rank=2");
        std::array<size_t, 2> dims = {n, n};
        NdArray arr(dims);

        [[maybe_unused]] auto err = launchKernel1D(kernel::eye<T>, 0, arr.m_Stream,
                                                   arr.size(), arr.m_Data.get(), n, n);
        if (err)
            CUDALERN_ERROR_MESSAGE(
                "Failed to construct the NdArray during the kernel stage", err);

        arr.synchronize();
        return arr;
    }

    /**
     * Create array with uniform random values.
     * @param dims Dimensions.
     * @param low Minimum value.
     * @param high Maximum value.
     * @return New array.
     */
    static NdArray random_uniform(const std::array<size_t, Rank>& dims, T low = 0,
                                  T high = 1) noexcept {
        NdArray arr(dims);

        static unsigned int seed_counter = 0;
        unsigned int seed =
            static_cast<unsigned int>(std::time(nullptr)) + seed_counter++;

        [[maybe_unused]] auto err =
            launchKernel1D(kernel::random_uniform<T>, 0, arr.m_Stream, arr.m_Size,
                           arr.m_Data.get(), arr.m_Size, low, high, seed);
        if (err) CUDALERN_ERROR_MESSAGE("Failed to generate random_uniform NdArray", err);

        arr.synchronize();
        return arr;
    }

    /**
     * Create array with normal-distributed random values.
     * @param dims Dimensions.
     * @param mean Distribution mean.
     * @param std Distribution standard deviation.
     * @return New array.
     */
    static NdArray random_normal(const std::array<size_t, Rank>& dims, T mean = 0,
                                 T std = 1) noexcept {
        NdArray arr(dims);

        static unsigned int seed_counter = 0;
        unsigned int seed =
            static_cast<unsigned int>(std::time(nullptr)) + seed_counter++;

        [[maybe_unused]] auto err =
            launchKernel1D(kernel::random_normal<T>, 0, arr.m_Stream, arr.m_Size,
                           arr.m_Data.get(), arr.m_Size, mean, std, seed);
        if (err) CUDALERN_ERROR_MESSAGE("Failed to generate random_normal NdArray", err);

        arr.synchronize();
        return arr;
    }

    /**
     * Create array from host data.
     * @param data Host vector.
     * @param dims Dimensions.
     * @return New array.
     */
    static NdArray fromHost(const std::vector<T>& data,
                            const std::array<size_t, Rank>& dims) noexcept {
        NdArray arr(dims);
        if (data.size() != arr.size())
            CUDALERN_CRITICAL("Data size does not match array size");
        [[maybe_unused]] auto err = memcpy(arr.m_Data.get(), data.data(), data.size(),
                                           memcpyKind::HostToDevice, arr.stream());
        arr.synchronize();
        return arr;
    };

    /**
     * Create pinned-host array.
     * @param dims Dimensions.
     * @return New array.
     */
    static NdArray pinned(const std::array<size_t, Rank>& dims) noexcept {
        std::size_t total = 1;
        for (auto d : dims)
            total *= d;
        std::shared_ptr<T> data(
            allocator<T>::template allocate<allocatorPolicy::Pinned>(total),
            PinnedDeleter<T>(Stream{}));

        auto result = NdArray(std::move(data), dims);
        result.synchronize();

        return result;
    };

    /**
     * Create managed-memory array.
     * @param dims Dimensions.
     * @return New array.
     */
    static NdArray managed(const std::array<size_t, Rank>& dims) noexcept {
        std::size_t total = 1;
        for (auto d : dims)
            total *= d;

        std::shared_ptr<T> data(
            allocator<T>::template allocate<allocatorPolicy::Managed>(total),
            ManagedDeleter<T>());

        auto result = NdArray(std::move(data), dims);

        return result;
    };

  public:
    /**
     * Number of dimensions.
     * @return Rank constant.
     */
    [[nodiscard]] constexpr auto rank() const noexcept { return Rank; }

    /**
     * Total element count.
     * @return Element count.
     */
    [[nodiscard]] auto size() const noexcept { return m_Size; }

    /**
     * Associated CUDA stream.
     * @return Stream object.
     */
    [[nodiscard]] auto stream() const noexcept { return *m_Stream; }

    /**
     * Dimension sizes.
     * @return Dimensions array.
     */
    [[nodiscard]] const auto& dims() const noexcept { return m_Dimensions; }

    /**
     * Stride values.
     * @return Strides array.
     */
    [[nodiscard]] auto strides() const noexcept { return m_Strides; }

  private:
    /**
     * Recursively decompose nested sequences into flat host data.
     * @tparam I Dimension index.
     * @param range Input sequence.
     * @param host_data Output flat vector.
     * @param offset Current write offset.
     */
    template <std::size_t I>
    void decomposeRanges(const auto& range, std::vector<T>& host_data,
                         std::size_t& offset) const {
        constexpr std::size_t NumDims = Rank;
        static_assert(I < NumDims, "Dimension index out of bounds");

        auto size = range.size();
        if (size != m_Dimensions[I])
            CUDALERN_CRITICAL("Dimension size mismatch at level " + std::to_string(I));

        if constexpr (I == NumDims - 1)
            for (const auto& elem : range)
                host_data[offset++] = static_cast<T>(elem);
        else
            for (const auto& sub : range)
                decomposeRanges<I + 1>(sub, host_data, offset);
    }

    /**
     * Recursively deduce dimensions from nested sequences.
     * @tparam I Dimension index.
     * @param range Input sequence.
     * @param dims Output dimensions array.
     */
    template <std::size_t I>
    void deduceDimensions(const auto& range, std::array<std::size_t, Rank>& dims) const {
        dims[I] = std::ranges::size(range);

        if constexpr (I + 1 < Rank) {
            if (dims[I] == 0)
                CUDALERN_CRITICAL("Dimension " + std::to_string(I) + " has zero size!");
            const auto& first_sub = *std::ranges::begin(range);
            deduceDimensions<I + 1>(first_sub, dims);
        }
    }

    /**
     * Fill array with a value.
     * @param value Fill value.
     */
    void fill(const T& value) noexcept {
        if (m_Size == 0 || !m_Data) return;

        if constexpr (std::is_integral_v<T> && sizeof(T) == 1) {
            memset(m_Data.get(), static_cast<int>(value), m_Size * sizeof(T),
                   m_Stream.get());
        } else {
            [[maybe_unused]] auto err = launchKernel1D(
                kernel::fill<T>, 0, m_Stream, m_Size, m_Data.get(), m_Size, value);
            if (err)
                CUDALERN_ERROR_MESSAGE("Failed to fill NdArray during kernel launch",
                                       err);
        }
    }

    /**
     * Compute row-major strides from dimensions.
     */
    void computeStrides() {
        m_Strides[Rank - 1] = 1;
        for (std::size_t i = Rank - 1; i > 0; --i)
            m_Strides[i - 1] = m_Strides[i] * m_Dimensions[i];
    }

    /**
     * Release resources and reset state.
     */
    void cleanup() noexcept {
        m_Data.reset();
        m_Size = 0;
        m_Dimensions.fill(0);
    }
};

/**
 * Scalar NdArray specialization (placeholder).
 * @tparam T Element type.
 */
template <class T>
class NdArray<T, 0> {};

}  // namespace cudalern