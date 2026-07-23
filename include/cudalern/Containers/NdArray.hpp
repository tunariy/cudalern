#pragma once

#include <cudalern/ABI/allocator.hpp>
#include <cudalern/ABI/stream.hpp>
#include <cudalern/Core/concepts.hpp>
#include <cudalern/Core/err.hpp>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "benchtools/Loggers/Logger.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace cudalern {

template <class T, std::size_t Rank>
    requires(CudaCompatible<T>)
class NdArray {
    static_assert(Rank > 0, "Use NdArray<T, 0> specialization for scalars!");

    std::shared_ptr<T> m_Data{};
    std::size_t m_Size{0};
    Stream m_Stream{};
    std::array<std::size_t, Rank> m_Dimensions{};
    std::array<std::size_t, Rank> m_Strides{};

  public:
    NdArray() = default;

    ~NdArray() noexcept { cleanup(); }

    explicit NdArray(const std::array<std::size_t, Rank>& dims) noexcept
        : m_Dimensions(dims) {
        if (m_Size != 0) cleanup();

        m_Size = 1;
        for (const auto d : m_Dimensions)
            m_Size *= d;

        // Use allocator
        m_Data = std::shared_ptr<T>(
            allocator<T>::template allocate<allocatorPolicy::Device>(m_Size, m_Stream),
            DeviceDeleter<T>());
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
            deduceDimensions<0>(std::get<0>(std::tuple(args...)), m_Dimensions);
        } else {
            m_Dimensions[0] = sizeof...(Sequence_t);

            auto first_arg = std::get<0>(std::tuple(args...));
            deduceDimensions<1>(first_arg, m_Dimensions);
        }

        computeStrides();

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
                BENCHTOOLS_CRITICAL("Number of arguments does not match first dimension");
            ([&](const auto& arg) { decomposeRanges<1>(arg, hostData, offset); }(args),
             ...);
        }

        // if the offset does not match the expected size
        // not enough or more than enough elements were supplied
        if (offset != m_Size) BENCHTOOLS_CRITICAL("Element count mismatch!");

        [[maybe_unused]] auto err = memcpy(m_Data.get(), hostData.data(), m_Size,
                                           memcpyKind::HostToDevice, m_Stream);
        synchronize();
    }

    NdArray(const NdArray<T, Rank>& other) {
        if (this == &other) return;
        if (!other.m_Size) return;

        if (this->m_Size)  // if we have anything allocated
            cleanup();

        m_Size = other.m_Size;
        m_Stream = other.m_Stream;
        m_Dimensions = other.m_Dimensions;
        m_Strides = other.m_Strides;

        m_Data = std::shared_ptr<T>(
            allocator<T>::template allocate<allocatorPolicy::Device>(m_Size, m_Stream),
            DeviceDeleter<T>());

        [[maybe_unused]] auto err =
            memcpy(m_Data.get(), other.m_Data.get(), m_Size, memcpyKind::DeviceToDevice);
    };

    [[nodiscard]] NdArray& operator=(const NdArray<T, Rank>& other) {
        if (this == &other) return *this;
        if (!other.m_Size) return *this;

        if (this->m_Size)  // if we have anything allocated
            cleanup();

        m_Size = other.m_Size;
        m_Stream = other.m_Stream;
        m_Dimensions = other.m_Dimensions;
        m_Strides = other.m_Strides;

        m_Data = std::shared_ptr<T>(
            allocator<T>::template allocate<allocatorPolicy::Device>(m_Size, m_Stream),
            DeviceDeleter<T>());

        [[maybe_unused]] auto err =
            memcpy(m_Data.get(), other.m_Data.get(), m_Size, memcpyKind::DeviceToDevice);

        return *this;
    }

    NdArray(NdArray<T, Rank>&& other) noexcept {
        if (!other.m_Size) return;  // discard move if other nd is empty

        m_Size = std::move(other.m_Size);
        m_Stream = std::move(other.m_Stream);
        m_Dimensions = std::move(other.m_Dimensions);
        m_Strides = std::move(other.m_Strides);

        m_Data = std::move(other.m_Data);
        other.m_Data.reset();
    }

    [[nodiscard]] NdArray& operator=(NdArray<T, Rank>&& other) noexcept {
        if (!other.m_Size) return *this;  // discard move if other nd is empty

        m_Size = std::move(other.m_Size);
        m_Stream = std::move(other.m_Stream);
        m_Dimensions = std::move(other.m_Dimensions);
        m_Strides = std::move(other.m_Strides);

        m_Data = std::move(other.m_Data);
        other.m_Data.reset();

        return *this;
    }

  public:
    template <class... Args>
        requires(sizeof...(Args) == Rank)
    T operator()(Args... dims) const {
        return read(
            std::array<std::size_t, sizeof...(Args)>{static_cast<std::size_t>(dims)...});
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
            memcpy(&temp, m_Data.get() + offset, 1, memcpyKind::DeviceToHost);
        return temp;
    }

    [[nodiscard]] std::vector<T> data() {
        std::unique_ptr<T[]> temp{new T[m_Size]};
        [[maybe_unused]] auto err =
            memcpy(temp.get(), m_Data.get(), m_Size, memcpyKind::DeviceToHost);

        return std::vector<T>{temp.get(), temp.get() + m_Size};
    }

    [[nodiscard]] const std::vector<T> data() const{
        std::unique_ptr<T[]> temp{new T[m_Size]};
        [[maybe_unused]] auto err =
            memcpy(temp.get(), m_Data.get(), m_Size, memcpyKind::DeviceToHost);

        return std::vector<T>{temp.get(), temp.get() + m_Size};
    }

    [[nodiscard]] T* release() noexcept {
        T* temp{m_Data.get()};
        m_Data = nullptr;
        return temp;
    }

    void synchronize() noexcept {
        [[maybe_unused]]
        auto err = m_Stream.synchronize();
        if (err)
            BENCHTOOLS_CRITICAL("Error synchronizing the stream! "s +
                                CUDALERN_ERROR(err));
    }

    [[nodiscard]] bool empty() const noexcept { return m_Size == 0; }

    [[nodiscard]] size_t nbytes() const noexcept { return m_Size * sizeof(T); }

    static NdArray zeros(const std::array<size_t, Rank>& dims) noexcept;

    static NdArray ones(const std::array<size_t, Rank>& dims) noexcept;

    static NdArray full(const std::array<size_t, Rank>& dims, T value) noexcept;

    template <std::size_t R = Rank>
    static std::enable_if<R == 1, NdArray> arange() noexcept;

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
            BENCHTOOLS_CRITICAL("Dimension size mismatch at level " + std::to_string(I));

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
                BENCHTOOLS_CRITICAL("Dimension " + std::to_string(I) + " has zero size!");
            const auto& first_sub = *std::ranges::begin(range);
            deduceDimensions<I + 1>(first_sub, dims);
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