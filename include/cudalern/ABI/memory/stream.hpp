// Stream.hpp
#pragma once

#include "cudalern/Core/core.hpp"
#include "driver_types.h"

#include <memory>

namespace cudalern {

class Stream {
  public:
    Stream();

    explicit Stream(cudaStream_t stream);

    Stream(const Stream& other) = default;

    Stream& operator=(const Stream& other) = default;

    Stream(Stream&& other) noexcept;

    Stream& operator=(Stream&& other) noexcept;

    ~Stream();

    operator cudaStream_t() const noexcept;

    [[nodiscard]] auto operator*() const noexcept -> cudaStream_t;

    [[nodiscard]] auto get() const noexcept -> cudaStream_t;

    [[nodiscard]] auto synchronize() const -> cudalernErr;

    [[nodiscard]] auto valid() const noexcept -> bool;

    [[nodiscard]] auto reset() -> cudalernErr;

    auto take(cudaStream_t stream) noexcept -> void;

    [[nodiscard]] auto release() noexcept -> cudaStream_t = delete;

  private:
    std::shared_ptr<cudaStream_t> m_Stream{};
};

}  // namespace cudalern