#include <cudalern/ABI/memory.hpp>
#include <cudalern/Core/concepts.hpp>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "benchtools/Loggers/Logger.hpp"

#include <array>
#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace cudalern {

template <class T, std::size_t Rank>
class NdArray {
    static_assert(Rank > 0, "Use NdArray<T, 0> specialization for scalars!");

    using DeviceAddr = T*;
    using HostAddr = T*;

    DeviceAddr m_Data{};
    std::size_t m_Size{0};
    std::array<std::size_t, Rank> m_Dimensions{};
    std::array<std::size_t, Rank> m_Strides{};

  public:
    NdArray() = default;

    explicit NdArray(const std::array<std::size_t, Rank>& dims) noexcept
        : m_Dimensions(dims) {
        if (m_Size != 0) {
            cleanup();
        }
        m_Size = 1;
        for (const auto d : m_Dimensions)
            m_Size *= d;
        m_Data = allocateDevice<T>(m_Size);
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

        if (m_Size != 0) {
            cleanup();
        }

        if constexpr (sizeof...(Sequence_t) == 1) {
            deduceDimensions<0>(std::get<0>(std::tuple(args...)), m_Dimensions);
        } else {
            m_Dimensions[0] = sizeof...(Sequence_t);

            auto first_arg = std::get<0>(std::tuple(args...));
            deduceDimensions<1>(first_arg, m_Dimensions);
        }

        m_Size = 1;
        for (std::size_t d : m_Dimensions)
            m_Size *= d;

        cudaStream_t stream{};
        cudaStreamCreate(&stream);
        cudaMallocAsync(&m_Data, m_Size * sizeof(T), stream);

        std::vector<T> hostData(m_Size);
        std::size_t offset = 0;

        if constexpr (sizeof...(Sequence_t) == 1) {
            auto tup = std::tuple(args...);
            decomposeRanges<0>(std::get<0>(tup), hostData, offset);
        } else {
            if (sizeof...(Sequence_t) != m_Dimensions[0])
                BENCHTOOLS_CRITICAL("Number of arguments does not match first dimension");
            ([&](const auto& arg) { decomposeRanges<1>(arg, hostData, offset); }(args),
             ...);
        }

        if (offset != m_Size) BENCHTOOLS_CRITICAL("Element count mismatch!");

        cudaMemcpyAsync(m_Data, hostData.data(), m_Size * sizeof(T),
                        cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }

    NdArray(const NdArray<T, Rank>& other) {
        if (!other.m_Size) return;  // discard copy if other nd is empty

        if (this->m_Size)  // if we have anything allocated
            deallocateDevice(m_Data);

        m_Size = other.m_Size;
        m_Data = allocateDevice<T>(m_Size);

        [[maybe_unused]] auto err =
            memcpy(m_Data, other.m_Data, m_Size, memcpyKind::DeviceToDevice);
    };

    [[nodiscard]] NdArray& operator=(const NdArray<T, Rank>& other) {
        if (!other.m_Size) return NdArray();  // discard copy if other nd is empty

        if (this->m_Size)  // if we have anything allocated
            deallocateDevice(m_Data);

        m_Size = other.m_Size;
        m_Data = allocateDevice<T>(m_Size);

        [[maybe_unused]] auto err =
            memcpy(m_Data, other.m_Data, m_Size, memcpyKind::DeviceToDevice);

        return *this;
    }

    NdArray(NdArray<T, Rank>&& other) noexcept {
        if (!other.m_Size) return;  // discard move if other nd is empty

        if (this->m_Size)  // if we have anything allocated
            deallocateDevice(m_Data);

        m_Size = std::move(other.m_Size);

        m_Data = std::move(other.m_Data);
        other.m_Data = nullptr;
    }

    [[nodiscard]] NdArray& operator=(NdArray<T, Rank>&& other) noexcept {
        if (!other.m_Size) *this;  // discard move if other nd is empty

        if (this->m_Size)  // if we have anything allocated
            deallocateDevice(m_Data);

        m_Size = std::move(other.m_Size);

        m_Data = std::move(other.m_Data);
        other.m_Data = nullptr;
        return *this;
    }

    ~NdArray() noexcept {
        deallocateDevice(m_Data);
        m_Size = 0;
        m_Dimensions = {};
    }

    template <class... Args>
        requires(sizeof...(Args) == Rank)
    [[nodiscard]] T read(Args... dims) {
        return read(
            std::array<std::size_t, sizeof...(Args)>{static_cast<std::size_t>(dims)...});
    }

    [[nodiscard]] T read(std::array<std::size_t, Rank> dims) {
        std::size_t offset{};
        for (std::size_t i{}; i < m_Dimensions.size(); i++) {
            if (i != m_Dimensions.size() - 1) {
                offset += m_Dimensions[i] * dims[i];
                continue;
            }
            offset += dims[i];
        }

        T temp;
        [[maybe_unused]] auto err =
            memcpy(&temp, m_Data + offset, 1, memcpyKind::DeviceToHost);
        return temp;
    }

    [[nodiscard]] std::vector<T> data() {
        std::unique_ptr<T> temp{new T[m_Size]};
        [[maybe_unused]] auto err =
            memcpy(temp.get(), m_Data, m_Size, memcpyKind::DeviceToHost);

        return std::vector<T>{temp.get(), temp.get() + m_Size};
    };

    [[nodiscard]] constexpr auto rank() const { return Rank; }

    [[nodiscard]] auto size() const { return m_Size; }

    [[nodiscard]] const auto& dims() const { return m_Dimensions; }

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
        /*
         * Row-major: stride[i] = stride[i+1] * dims[i+1]
         */
    }

    void cleanup() {
        if (m_Data) {
            deallocateDevice(m_Data);
            m_Data = nullptr;
        }
        m_Size = 0;
        m_Dimensions.fill(0);
    }
};

template <class T>
class NdArray<T, 0> {};

}  // namespace cudalern
