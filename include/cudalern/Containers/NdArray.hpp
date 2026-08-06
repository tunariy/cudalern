#pragma once

#include "cudalern/ABI/memory/memory.hpp"
#include "driver_types.h"
#include <cmath>
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
#include <string>
#include <type_traits>
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
    [[nodiscard]] auto stream() const noexcept { return m_Stream; }
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

    explicit NdArray(const std::array<std::size_t, Rank>& dims) noexcept
        : m_Dimensions(dims) {
        if (m_Size != 0) cleanup();

        m_Size = 1;
        for (const auto d : m_Dimensions)
            m_Size *= d;

        computeStrides();

        m_Data = std::shared_ptr<T>(
            allocator<T>::template allocate<allocatorPolicy::Device>(m_Size, m_Stream),
            DeviceDeleter<T>());
    }

    /**
     * @brief Construct a new Nd Array object
     * With std::shared_ptr the type of the deleter is erased
     * so regardless of the memory type we can pass in any kind of nd array without type
     * checking since as simply the deleter handles the freeing
     *
     * @param data
     * @param dims
     */
    explicit NdArray(std::shared_ptr<T> data,
                     const std::array<std::size_t, Rank>& dims) noexcept
        : m_Data(std::move(data)), m_Dimensions(dims) {
        if (m_Size != 0) cleanup();

        m_Size = 1;
        for (const auto d : m_Dimensions)
            m_Size *= d;

        computeStrides();
    }

    /**
     * @brief Construct a new NdArray by deep-copying the region of device memory
     * referenced by an NdView into freshly allocated storage of its own.
     *
     * @param view the view to copy data from
     */
    explicit NdArray(const NdView<T, Rank>& view) noexcept {
        auto srcData = view.getShared();
        if (!srcData) {
            CUDALERN_CRITICAL(
                "Cannot construct NdArray from NdView: source data has expired");
            return;
        }

        if (this->m_Size) cleanup();

        m_Dimensions = view.dims();
        m_Stream = view.stream();

        m_Size = 1;
        for (const auto d : m_Dimensions)
            m_Size *= d;

        computeStrides();

        m_Data = std::shared_ptr<T>(
            allocator<T>::template allocate<allocatorPolicy::Device>(m_Size, m_Stream),
            DeviceDeleter<T>());

        [[maybe_unused]] auto err = memcpy(m_Data.get(), srcData.get() + view.offset(),
                                           m_Size, memcpyKind::DeviceToDevice, m_Stream);
        synchronize();
    }

    /**
     * @brief Takes in a pack of sequences and constructs a n dimensional matrix
     *
     * @tparam Sequence pack of sequences
     */
    template <class... Sequence_t>
        requires((sizeof...(Sequence_t) <= Rank && ElementIterable<Sequence_t>) && ...)
    explicit NdArray(Sequence_t... args) noexcept {
        static_assert(Rank > 0, "NdArray must have at least one dimension");

        if (m_Size != 0) cleanup();

        // Deduce the dimensions using the param pack
        if constexpr (sizeof...(Sequence_t) == 1) {
            this->deduceDimensions<0>(std::get<0>(std::tuple(args...)), m_Dimensions);
        } else {
            m_Dimensions[0] = sizeof...(Sequence_t);

            auto first_arg = std::get<0>(std::tuple(args...));
            this->deduceDimensions<1>(first_arg, m_Dimensions);
        }

        this->computeStrides();

        // Calculating the total size
        m_Size = 1;
        for (std::size_t d : m_Dimensions)
            m_Size *= d;

        m_Data = std::shared_ptr<T>(
            allocator<T>::template allocate<allocatorPolicy::Device>(m_Size, m_Stream),
            DeviceDeleter<T>());

        std::vector<T> hostData(m_Size);
        std::size_t offset = 0;

        if constexpr (sizeof...(Sequence_t) == 1) {
            // if we have a single sequence
            // decompose that
            auto tup = std::tuple(args...);
            decomposeRanges<0>(std::get<0>(tup), hostData, offset);
        } else {
            // ???
            if (sizeof...(Sequence_t) != m_Dimensions[0])
                CUDALERN_CRITICAL("Number of arguments does not match first dimension");
            ([&](const auto& arg) { decomposeRanges<1>(arg, hostData, offset); }(args),
             ...);
        }

        // if the offset does not match the expected size
        // not enough or more than enough elements were supplied
        if (offset != m_Size) CUDALERN_CRITICAL("Element count mismatch!");

        [[maybe_unused]] auto err = memcpy(m_Data.get(), hostData.data(), m_Size,
                                           memcpyKind::HostToDevice, m_Stream);
        this->synchronize();
    }

    NdArray(const NdArray<T, Rank>& rhs) {
        if (this == &rhs) return;
        if (!rhs.m_Size) return;

        if (this->m_Size)  // if we have anything allocated
            cleanup();

        m_Size = rhs.m_Size;
        m_Stream = rhs.m_Stream;
        m_Dimensions = rhs.m_Dimensions;
        m_Strides = rhs.m_Strides;

        m_Data = std::shared_ptr<T>(
            allocator<T>::template allocate<allocatorPolicy::Device>(m_Size, m_Stream),
            DeviceDeleter<T>());

        [[maybe_unused]] auto err = memcpy(m_Data.get(), rhs.m_Data.get(), m_Size,
                                           memcpyKind::DeviceToDevice, m_Stream);
        synchronize();
    };

    NdArray& operator=(const NdArray<T, Rank>& rhs) {
        if (this == &rhs) return *this;
        if (!rhs.m_Size) return *this;

        if (this->m_Size)  // if we have anything allocated
            cleanup();

        m_Size = rhs.m_Size;
        m_Stream = rhs.m_Stream;
        m_Dimensions = rhs.m_Dimensions;
        m_Strides = rhs.m_Strides;

        m_Data = std::shared_ptr<T>(
            allocator<T>::template allocate<allocatorPolicy::Device>(m_Size, m_Stream),
            DeviceDeleter<T>());

        [[maybe_unused]] auto err = memcpy(m_Data.get(), rhs.m_Data.get(), m_Size,
                                           memcpyKind::DeviceToDevice, m_Stream);
        synchronize();

        return *this;
    }

    NdArray(NdArray<T, Rank>&& rhs) noexcept {
        if (!rhs.m_Size) return;  // discard move if rhs nd is empty

        m_Size = std::move(rhs.m_Size);
        m_Stream = std::move(rhs.m_Stream);
        m_Dimensions = std::move(rhs.m_Dimensions);
        m_Strides = std::move(rhs.m_Strides);

        m_Data = std::move(rhs.m_Data);

        rhs.m_Data.reset();
        rhs.m_Size = 0;
        rhs.m_Dimensions.fill(0);
    }

    NdArray& operator=(NdArray<T, Rank>&& rhs) noexcept {
        if (!rhs.m_Size) return *this;  // discard move if rhs nd is empty

        m_Size = std::move(rhs.m_Size);
        m_Stream = std::move(rhs.m_Stream);
        m_Dimensions = std::move(rhs.m_Dimensions);
        m_Strides = std::move(rhs.m_Strides);

        m_Data = std::move(rhs.m_Data);

        rhs.m_Data.reset();
        rhs.m_Size = 0;
        rhs.m_Dimensions.fill(0);

        return *this;
    }

  public:
    /**
     * @briefa
     *
     * @tparam diffRank
     */
    template <std::size_t diffRank>
        requires(Rank == diffRank && Rank >= 2)
    NdArray<T, Rank> operator*(const NdArray<T, diffRank>& rhs) const noexcept {
        static_assert(Rank >= 2);

        if (!m_Data || !rhs.m_Data) return NdArray<T, Rank>();

        // Check batch dimensions
        for (std::size_t d = 0; d < Rank - 2; ++d)
            if (m_Dimensions[d] != rhs.m_Dimensions[d]) return NdArray<T, Rank>();

        size_t M = m_Dimensions[Rank - 2];
        size_t K = m_Dimensions[Rank - 1];
        size_t N = rhs.m_Dimensions[Rank - 1];
        if (K != rhs.m_Dimensions[Rank - 2]) return NdArray<T, Rank>();

        // Result dimensions
        std::array<size_t, Rank> result_dims = m_Dimensions;
        result_dims[Rank - 2] = M;
        result_dims[Rank - 1] = N;

        // Result strides (row‑major contiguous)
        std::array<size_t, Rank> result_strides;
        size_t stride = 1;
        for (int d = static_cast<int>(Rank) - 1; d >= 0; --d) {
            result_strides[d] = stride;
            stride *= result_dims[d];
        }

        // Total number of output elements
        size_t batch_count = 1;
        for (size_t d = 0; d < Rank - 2; ++d)
            batch_count *= m_Dimensions[d];

        size_t totalElems = batch_count * M * N;
        if (totalElems == 0) return NdArray<T, Rank>();

        size_t* d_dimsA{};
        size_t* d_dimsB{};
        size_t* d_dimsC{};
        size_t* d_stridesA{};
        size_t* d_stridesB{};
        size_t* d_stridesC{};

        auto cleanupDeviceDim = [&d_dimsA, &d_dimsB, &d_dimsC, &d_stridesA, &d_stridesB,
                                 &d_stridesC]() {
            allocator<size_t>::template deallocate<allocatorPolicy::Device>(d_dimsA);
            allocator<size_t>::template deallocate<allocatorPolicy::Device>(d_dimsB);
            allocator<size_t>::template deallocate<allocatorPolicy::Device>(d_dimsC);
            allocator<size_t>::template deallocate<allocatorPolicy::Device>(d_stridesA);
            allocator<size_t>::template deallocate<allocatorPolicy::Device>(d_stridesB);
            allocator<size_t>::template deallocate<allocatorPolicy::Device>(d_stridesC);
        };

        auto allocateCopy = [&](size_t*& dst, const std::array<size_t, Rank>& src,
                                const Stream& stream) -> cudalernErr {
            dst = allocator<size_t>::template allocate<allocatorPolicy::Device>(Rank);
            if (!dst) return cudaErrorUnknown;

            if (memcpy(dst, src.data(), Rank, memcpyKind::HostToDevice, stream)) {
                allocator<size_t>::template deallocate<allocatorPolicy::Device>(dst);
                dst = nullptr;
                return cudaErrorUnknown;
            }
            return cudaSuccess;
        };

        allocateCopy(d_dimsA, m_Dimensions, m_Stream);
        allocateCopy(d_dimsB, rhs.m_Dimensions, m_Stream);
        allocateCopy(d_dimsC, result_dims, m_Stream);
        allocateCopy(d_stridesA, m_Strides, m_Stream);
        allocateCopy(d_stridesB, rhs.m_Strides, m_Stream);
        allocateCopy(d_stridesC, result_strides, m_Stream);

        // --- Allocate result memory using the class allocator ---
        T* d_result = allocator<T>::template allocate<allocatorPolicy::Device>(totalElems,
                                                                               m_Stream);
        if (!d_result) {
            cleanupDeviceDim();
            return NdArray<T, Rank>();
        }

        std::shared_ptr<T> resultPtr(d_result, DeviceDeleter<T>());
        NdArray<T, Rank> result(resultPtr, result_dims);

        // --- Launch kernel ---
        cudalernErr launchErr = launchKernel1D(
            kernel::batched_matmul_kernel<T, Rank>,
            0,           // shared memory size
            m_Stream,    // stream
            totalElems,  // total elements (1D grid)
            m_Data.get(), rhs.m_Data.get(), resultPtr.get(), d_dimsA, d_dimsB, d_dimsC,
            d_stridesA, d_stridesB, d_stridesC, M, K, N, totalElems);

        synchronize();
        cleanupDeviceDim();

        if (launchErr != cudaSuccess) return NdArray<T, Rank>();

        return result;
    }

    // -------------------------------------------------------------------------
    // Unary negation
    // -------------------------------------------------------------------------
    /** Returns a new array with all elements negated. */
    NdArray<T, Rank> operator-() const noexcept {
        if (!m_Data || m_Size == 0) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err = launchKernel1D(kernel::negate_kernel<T>, 0, m_Stream, m_Size,
                                         result.m_Data.get(), m_Data.get(),
                                         static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();
        return result;
    }

    // -------------------------------------------------------------------------
    // Scalar arithmetic (return new array)
    // -------------------------------------------------------------------------
    /** Element-wise addition with scalar, returns new array. */
    NdArray<T, Rank> operator+(const T& val) const noexcept {
        if (!m_Data || m_Size == 0) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err = launchKernel1D(
            (void (*)(T*, const T*, T, uint32_t))kernel::add<T>, 0, m_Stream, m_Size,
            result.m_Data.get(), m_Data.get(), val, static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();
        return result;
    }

    /** Element-wise subtraction with scalar, returns new array. */
    NdArray<T, Rank> operator-(const T& val) const noexcept {
        if (!m_Data || m_Size == 0) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err = launchKernel1D(
            (void (*)(T*, const T*, T, uint32_t))kernel::sub<T>, 0, m_Stream, m_Size,
            result.m_Data.get(), m_Data.get(), val, static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();
        return result;
    }

    /** Element-wise multiplication with scalar, returns new array. */
    NdArray<T, Rank> operator*(const T& val) const noexcept {
        if (!m_Data || m_Size == 0) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err = launchKernel1D(
            (void (*)(T*, const T*, T, uint32_t))kernel::mul<T>, 0, m_Stream, m_Size,
            result.m_Data.get(), m_Data.get(), val, static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();
        return result;
    }

    /** Element-wise division with scalar, returns new array. */
    NdArray<T, Rank> operator/(const T& val) const noexcept {
        if (!m_Data || m_Size == 0) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err = launchKernel1D(
            (void (*)(T*, const T*, T, uint32_t))kernel::div<T>, 0, m_Stream, m_Size,
            result.m_Data.get(), m_Data.get(), val, static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();
        return result;
    }

    // -------------------------------------------------------------------------
    // Array-array elementwise arithmetic (return new array)
    // -------------------------------------------------------------------------
    /** Element-wise addition of two arrays, returns new array. */
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

    /** Element-wise subtraction of two arrays, returns new array. */
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
        return result;
    }

    /** Element-wise multiplication of two arrays (using operator% to avoid conflict with
     * matrix multiplication). */
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
        return result;
    }

    /** Element-wise multiplication of two arrays – for Rank == 1 (no matrix
     * multiplication conflict). */
    template <std::size_t diffRank>
        requires(Rank == diffRank && Rank == 1)
    NdArray<T, Rank> operator*(const NdArray<T, diffRank>& rhs) const noexcept {
        return (*this) % rhs;  // reuse elementwise operator%
    }

    /** Element-wise division of two arrays, returns new array. */
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
        return result;
    }

    // -------------------------------------------------------------------------
    // In‑place scalar arithmetic
    // -------------------------------------------------------------------------
    /** In-place addition with scalar. */
    void operator+=(const T& val) const noexcept {
        if (!m_Data || m_Size == 0) return;
        cudalernErr err = launchKernel1D(
            (void (*)(T*, const T*, T, uint32_t))kernel::add<T>, 0, m_Stream, m_Size,
            m_Data.get(), m_Data.get(), val, static_cast<uint32_t>(m_Size));
        if (err) CUDALERN_ERR("In-place scalar addition failed");
    }

    /** In-place subtraction with scalar. */
    void operator-=(const T& val) const noexcept {
        if (!m_Data || m_Size == 0) return;
        cudalernErr err = launchKernel1D(
            (void (*)(T*, const T*, T, uint32_t))kernel::sub<T>, 0, m_Stream, m_Size,
            m_Data.get(), m_Data.get(), val, static_cast<uint32_t>(m_Size));
        if (err) CUDALERN_ERR("In-place scalar subtraction failed");
    }

    /** In-place multiplication with scalar. */
    void operator*=(const T& val) const noexcept {
        if (!m_Data || m_Size == 0) return;
        cudalernErr err = launchKernel1D(
            (void (*)(T*, const T*, T, uint32_t))kernel::mul<T>, 0, m_Stream, m_Size,
            m_Data.get(), m_Data.get(), val, static_cast<uint32_t>(m_Size));
        if (err) CUDALERN_ERR("In-place scalar multiplication failed");
    }

    /** In-place division with scalar. */
    void operator/=(const T& val) const noexcept {
        if (!m_Data || m_Size == 0) return;
        cudalernErr err = launchKernel1D(
            (void (*)(T*, const T*, T, uint32_t))kernel::div<T>, 0, m_Stream, m_Size,
            m_Data.get(), m_Data.get(), val, static_cast<uint32_t>(m_Size));
        if (err) CUDALERN_ERR("In-place scalar division failed");
    }

    // -------------------------------------------------------------------------
    // In‑place array-array arithmetic
    // -------------------------------------------------------------------------
    /** In-place element-wise addition with another array. */
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
    }

    /** In-place element-wise subtraction with another array. */
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
    }

    /** In-place element-wise multiplication with another array (using operator%= for
     * explicit). */
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
    }

    /** In-place element-wise multiplication with another array (using operator*= for
     * convenience). */
    void operator*=(const NdArray<T, Rank>& rhs) const noexcept { (*this) %= rhs; }

    /** In-place element-wise division with another array. */
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

    // -------------------------------------------------------------------------
    // Comparison operators (return 0/1 arrays)
    // -------------------------------------------------------------------------
    /** Element-wise equality, returns array of 0/1. */
    NdArray<T, Rank> operator==(const NdArray<T, Rank>& rhs) const noexcept {
        if (!m_Data || !rhs.m_Data || m_Size != rhs.m_Size) return NdArray<T, Rank>();
        for (size_t d = 0; d < Rank; ++d)
            if (m_Dimensions[d] != rhs.m_Dimensions[d]) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err =
            launchKernel1D(kernel::eq_kernel<T>, 0, m_Stream, m_Size, result.m_Data.get(),
                           m_Data.get(), rhs.m_Data.get(), static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();
        return result;
    }

    /** Element-wise inequality, returns array of 0/1. */
    NdArray<T, Rank> operator!=(const NdArray<T, Rank>& rhs) const noexcept {
        if (!m_Data || !rhs.m_Data || m_Size != rhs.m_Size) return NdArray<T, Rank>();
        for (size_t d = 0; d < Rank; ++d)
            if (m_Dimensions[d] != rhs.m_Dimensions[d]) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err =
            launchKernel1D(kernel::ne_kernel<T>, 0, m_Stream, m_Size, result.m_Data.get(),
                           m_Data.get(), rhs.m_Data.get(), static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();
        return result;
    }

    /** Element-wise less-than, returns array of 0/1. */
    NdArray<T, Rank> operator<(const NdArray<T, Rank>& rhs) const noexcept {
        if (!m_Data || !rhs.m_Data || m_Size != rhs.m_Size) return NdArray<T, Rank>();
        for (size_t d = 0; d < Rank; ++d)
            if (m_Dimensions[d] != rhs.m_Dimensions[d]) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err =
            launchKernel1D(kernel::lt_kernel<T>, 0, m_Stream, m_Size, result.m_Data.get(),
                           m_Data.get(), rhs.m_Data.get(), static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();
        return result;
    }

    /** Element-wise greater-than, returns array of 0/1. */
    NdArray<T, Rank> operator>(const NdArray<T, Rank>& rhs) const noexcept {
        if (!m_Data || !rhs.m_Data || m_Size != rhs.m_Size) return NdArray<T, Rank>();
        for (size_t d = 0; d < Rank; ++d)
            if (m_Dimensions[d] != rhs.m_Dimensions[d]) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err =
            launchKernel1D(kernel::gt_kernel<T>, 0, m_Stream, m_Size, result.m_Data.get(),
                           m_Data.get(), rhs.m_Data.get(), static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();
        return result;
    }

    /** Element-wise less-or-equal, returns array of 0/1. */
    NdArray<T, Rank> operator<=(const NdArray<T, Rank>& rhs) const noexcept {
        if (!m_Data || !rhs.m_Data || m_Size != rhs.m_Size) return NdArray<T, Rank>();
        for (size_t d = 0; d < Rank; ++d)
            if (m_Dimensions[d] != rhs.m_Dimensions[d]) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err =
            launchKernel1D(kernel::le_kernel<T>, 0, m_Stream, m_Size, result.m_Data.get(),
                           m_Data.get(), rhs.m_Data.get(), static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();
        return result;
    }

    /** Element-wise greater-or-equal, returns array of 0/1. */
    NdArray<T, Rank> operator>=(const NdArray<T, Rank>& rhs) const noexcept {
        if (!m_Data || !rhs.m_Data || m_Size != rhs.m_Size) return NdArray<T, Rank>();
        for (size_t d = 0; d < Rank; ++d)
            if (m_Dimensions[d] != rhs.m_Dimensions[d]) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err =
            launchKernel1D(kernel::ge_kernel<T>, 0, m_Stream, m_Size, result.m_Data.get(),
                           m_Data.get(), rhs.m_Data.get(), static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();
        return result;
    }

    // -------------------------------------------------------------------------
    // Element-wise mathematical functions (return new array)
    // -------------------------------------------------------------------------
    /** Element-wise power: c[i] = pow(a[i], exponent). */
    NdArray<T, Rank> pow(T exponent) const noexcept {
        if (!m_Data || m_Size == 0) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err = launchKernel1D(kernel::pow_kernel<T>, 0, m_Stream, m_Size,
                                         result.m_Data.get(), m_Data.get(), exponent,
                                         static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();
        return result;
    }

    /** Element-wise exponential: c[i] = exp(a[i]). */
    NdArray<T, Rank> exp() const noexcept {
        if (!m_Data || m_Size == 0) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err = launchKernel1D(kernel::exp_kernel<T>, 0, m_Stream, m_Size,
                                         result.m_Data.get(), m_Data.get(),
                                         static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();
        return result;
    }

    /** Element-wise natural logarithm: c[i] = log(a[i]). */
    NdArray<T, Rank> log() const noexcept {
        if (!m_Data || m_Size == 0) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err = launchKernel1D(kernel::log_kernel<T>, 0, m_Stream, m_Size,
                                         result.m_Data.get(), m_Data.get(),
                                         static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();
        return result;
    }

    /** Element-wise square root: c[i] = sqrt(a[i]). */
    NdArray<T, Rank> sqrt() const noexcept {
        if (!m_Data || m_Size == 0) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err = launchKernel1D(kernel::sqrt_kernel<T>, 0, m_Stream, m_Size,
                                         result.m_Data.get(), m_Data.get(),
                                         static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();
        return result;
    }

    /** Element-wise absolute value: c[i] = abs(a[i]). */
    NdArray<T, Rank> abs() const noexcept {
        if (!m_Data || m_Size == 0) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err = launchKernel1D(kernel::abs_kernel<T>, 0, m_Stream, m_Size,
                                         result.m_Data.get(), m_Data.get(),
                                         static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();
        return result;
    }

    // -------------------------------------------------------------------------
    // Activation functions (return new array)
    // -------------------------------------------------------------------------
    /** Element-wise ReLU: c[i] = max(0, a[i]). */
    NdArray<T, Rank> relu() const noexcept {
        if (!m_Data || m_Size == 0) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err =
            launchKernel1D(kernel::relu<T>, 0, m_Stream, m_Size, result.m_Data.get(),
                           m_Data.get(), static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();
        return result;
    }

    /** Element-wise Leaky ReLU: c[i] = a[i] if a[i]>0 else alpha*a[i]. */
    NdArray<T, Rank> leaky_relu(T alpha = static_cast<T>(0.01)) const noexcept {
        if (!m_Data || m_Size == 0) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err = launchKernel1D(kernel::leaky_relu<T>, 0, m_Stream, m_Size,
                                         result.m_Data.get(), m_Data.get(), alpha,
                                         static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();
        return result;
    }

    /** Element-wise sigmoid: c[i] = 1/(1+exp(-a[i])). */
    NdArray<T, Rank> sigmoid() const noexcept {
        if (!m_Data || m_Size == 0) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err =
            launchKernel1D(kernel::sigmoid<T>, 0, m_Stream, m_Size, result.m_Data.get(),
                           m_Data.get(), static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();
        return result;
    }

    /** Element-wise tanh: c[i] = tanh(a[i]). */
    NdArray<T, Rank> tanh() const noexcept {
        if (!m_Data || m_Size == 0) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        cudalernErr err = launchKernel1D(kernel::tanh_kernel<T>, 0, m_Stream, m_Size,
                                         result.m_Data.get(), m_Data.get(),
                                         static_cast<uint32_t>(m_Size));
        if (err) return NdArray<T, Rank>();
        return result;
    }

    /**
     * @brief Softmax (1D only): c[i] = exp(a[i]) / sum(exp(a)).
     * @note This is a simplified implementation; for production, use a dedicated wrapper.
     */
    NdArray<T, Rank> softmax() const noexcept {
        static_assert(Rank == 1, "softmax only valid for 1D arrays");
        if (!m_Data || m_Size == 0) return NdArray<T, Rank>();
        NdArray<T, Rank> result(m_Dimensions);
        // Placeholder – the kernel requires custom block/grid and shared memory.
        // For a full implementation, adapt your launchKernel wrapper.
        // Returning empty for now.
        return NdArray<T, Rank>();
    }

  public:
    template <class... Args>
        requires(sizeof...(Args) == Rank)
    T operator()(Args... dims) const {
        return read(
            std::array<std::size_t, sizeof...(Args)>{static_cast<std::size_t>(dims)...});
    }

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

    template <class... Args>
        requires(sizeof...(Args) == Rank)
    [[nodiscard]] T read(Args... dims) const {
        return read(
            std::array<std::size_t, sizeof...(Args)>{static_cast<std::size_t>(dims)...});
    }

    [[nodiscard]] T read(std::array<std::size_t, Rank> dims) const {
        std::size_t offset{};
        for (std::size_t i = 0; i < Rank; ++i) {
            offset += m_Strides[i] * dims[i];
        }

        T temp;
        [[maybe_unused]] auto err =
            memcpy(&temp, m_Data.get() + offset, 1, memcpyKind::DeviceToHost, m_Stream);

        synchronize();

        return temp;
    }

    [[nodiscard]] std::vector<T> data() const {
        std::unique_ptr<T[]> temp{new T[m_Size]};
        [[maybe_unused]] auto err =
            memcpy(temp.get(), m_Data.get(), m_Size, memcpyKind::DeviceToHost, m_Stream);

        synchronize();

        return std::vector<T>{temp.get(), temp.get() + m_Size};
    }

    void to_host(std::vector<T>& out) const noexcept { out = data(); }

    [[nodiscard]] T* release() noexcept {
        auto temp{m_Data.get()};
        m_Data = std::shared_ptr<T>{nullptr, [](T* ptr) {}};
        m_Size = {};
        return temp;
    }

    // [[nodiscard]] std::shared_ptr<T> getUnderlying() const noexcept { return m_Data; }

    void synchronize() const noexcept {
        [[maybe_unused]]
        auto err = m_Stream.synchronize();
        if (err)
            CUDALERN_CRITICAL(
                CUDALERN_ERROR_MESSAGE("Error synchronizing the stream! ", err));
    }

    [[nodiscard]] bool empty() const noexcept { return m_Size == 0; }

    [[nodiscard]] size_t nbytes() const noexcept { return m_Size * sizeof(T); }

  public:
    static NdArray zeros(const std::array<size_t, Rank>& dims) noexcept {
        NdArray arr(dims);
        arr.fill(static_cast<T>(0));
        return arr;
    }

    static NdArray ones(const std::array<size_t, Rank>& dims) noexcept {
        NdArray arr(dims);
        arr.fill(static_cast<T>(1));
        return arr;
    }

    static NdArray full(const std::array<size_t, Rank>& dims, T value) noexcept {
        NdArray arr(dims);
        arr.fill(value);
        return arr;
    }

    static NdArray arange(T start, T stop, T step = 1) noexcept {
        static_assert(Rank == 1, "arange only valid for Rank=1");
        size_t n = static_cast<size_t>((stop - start) / step);
        std::array<size_t, 1> dims = {n};
        NdArray arr(dims);

        [[maybe_unused]] auto err =
            launchKernel1D(kernel::arange<T>, 0, arr.m_Stream, arr.m_Size,
                           arr.m_Data.get(), arr.m_Size, start, step);
        if (err)
            CUDALERN_ERR(
                CUDALERN_ERROR_MESSAGE("Failed to generate arange NdArray", err));

        arr.synchronize();
        return arr;
    }

    // eye – rank 2 identity matrix
    static NdArray eye(size_t n) noexcept {
        static_assert(Rank == 2, "eye only valid for Rank=2");
        std::array<size_t, 2> dims = {n, n};
        NdArray arr(dims);

        [[maybe_unused]] auto err = launchKernel1D(kernel::eye<T>, 0, arr.m_Stream,
                                                   arr.size(), arr.m_Data.get(), n, n);
        if (err)
            CUDALERN_ERR(CUDALERN_ERROR_MESSAGE(
                "Failed to construct the NdArray during the kernel stage", err));

        arr.synchronize();
        return arr;
    }

    static NdArray random_uniform(const std::array<size_t, Rank>& dims, T low = 0,
                                  T high = 1) noexcept {
        NdArray arr(dims);

        static unsigned int seed_counter = 0;
        unsigned int seed =
            static_cast<unsigned int>(std::time(nullptr)) + seed_counter++;

        [[maybe_unused]] auto err =
            launchKernel1D(kernel::random_uniform<T>, 0, arr.m_Stream, arr.m_Size,
                           arr.m_Data.get(), arr.m_Size, low, high, seed);
        if (err)
            CUDALERN_ERR(
                CUDALERN_ERROR_MESSAGE("Failed to generate random_uniform NdArray", err));

        arr.synchronize();
        return arr;
    }

    static NdArray random_normal(const std::array<size_t, Rank>& dims, T mean = 0,
                                 T std = 1) noexcept {
        NdArray arr(dims);

        static unsigned int seed_counter = 0;
        unsigned int seed =
            static_cast<unsigned int>(std::time(nullptr)) + seed_counter++;

        [[maybe_unused]] auto err =
            launchKernel1D(kernel::random_normal<T>, 0, arr.m_Stream, arr.m_Size,
                           arr.m_Data.get(), arr.m_Size, mean, std, seed);
        if (err)
            CUDALERN_ERR(
                CUDALERN_ERROR_MESSAGE("Failed to generate random_normal NdArray", err));

        arr.synchronize();
        return arr;
    }

    static NdArray fromHost(const std::vector<T>& data,
                            const std::array<size_t, Rank>& dims,
                            Stream stream) noexcept {
        NdArray arr(dims);
        if (data.size() != arr.size())
            CUDALERN_CRITICAL("Data size does not match array size");
        [[maybe_unused]] auto err = memcpy(arr.m_Data.get(), data.data(), data.size(),
                                           memcpyKind::HostToDevice, stream);
        arr.synchronize();
        return arr;
    };

    static NdArray pinned(const std::array<size_t, Rank>& dims) noexcept {
        std::size_t total = 1;
        for (auto d : dims)
            total *= d;
        std::shared_ptr<T> data(
            allocator<T>::template allocate<allocatorPolicy::Pinned>(total),
            PinnedDeleter<T>());
        return NdArray(std::move(data), dims);
    };

    static NdArray managed(const std::array<size_t, Rank>& dims) noexcept {
        std::size_t total = 1;
        for (auto d : dims)
            total *= d;

        std::shared_ptr<T> data(
            allocator<T>::template allocate<allocatorPolicy::Managed>(total),
            ManagedDeleter<T>());
        return NdArray(std::move(data), dims);
    };

  public:
    [[nodiscard]] constexpr auto rank() const noexcept { return Rank; }

    [[nodiscard]] auto size() const noexcept { return m_Size; }

    [[nodiscard]] auto stream() const noexcept { return *m_Stream; }

    [[nodiscard]] const auto& dims() const noexcept { return m_Dimensions; }

    [[nodiscard]] auto strides() const noexcept { return m_Strides; }

  private:
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

    void fill(const T& value) noexcept {
        if (m_Size == 0 || !m_Data) return;

        if constexpr (std::is_integral_v<T> && sizeof(T) == 1) {
            memset(m_Data.get(), static_cast<int>(value), m_Size * sizeof(T),
                   m_Stream.get());
        } else {
            [[maybe_unused]] auto err = launchKernel1D(
                kernel::fill<T>, 0, m_Stream, m_Size, m_Data.get(), m_Size, value);
            if (err)
                CUDALERN_ERR(CUDALERN_ERROR_MESSAGE(
                    "Failed to fill NdArray during kernel launch", err));
        }
    }

    void computeStrides() {
        m_Strides[Rank - 1] = 1;
        for (std::size_t i = Rank - 1; i > 0; --i)
            m_Strides[i - 1] = m_Strides[i] * m_Dimensions[i];
    }

    void cleanup() noexcept {
        m_Data.reset();
        m_Size = 0;
        m_Dimensions.fill(0);
    }
};

template <class T>
class NdArray<T, 0> {};

}  // namespace cudalern