#include "cudalern/ABI/memory/stream.hpp"
#include "cudalern/Core/core.hpp"
#include "cudalern/Core/err.hpp"

#include "cuda_runtime_api.h"

#include <memory>
#include <utility>

namespace cudalern {

struct StreamDeleter {
    void operator()(cudaStream_t* ptr) const noexcept {
        if (ptr && *ptr) {
            auto err = cudaStreamDestroy(*ptr);
            if (err)
                CUDALERN_CRITICAL(
                    CUDALERN_ERROR_MESSAGE("Failed to destroy stream", err));
            return;
        }
    }
};

Stream::Stream() {
    cudaStream_t raw = nullptr;
    cudaStreamCreate(&raw);
    m_Stream = std::shared_ptr<cudaStream_t>(new cudaStream_t(raw), StreamDeleter());
}

Stream::Stream(cudaStream_t stream)
    : m_Stream(std::shared_ptr<cudaStream_t>(new cudaStream_t(stream), StreamDeleter())) {
}

Stream::Stream(Stream&& other) noexcept : m_Stream(std::move(other.m_Stream)) {}

Stream& Stream::operator=(Stream&& other) noexcept {
    if (this != &other) {
        m_Stream = std::move(other.m_Stream);
    }
    return *this;
}

Stream::~Stream() = default;

Stream::operator cudaStream_t() const noexcept {
    return m_Stream ? *m_Stream : nullptr;
}

auto Stream::operator*() const noexcept -> cudaStream_t {
    return m_Stream ? *m_Stream : nullptr;
}

auto Stream::get() const noexcept -> cudaStream_t {
    return m_Stream ? *m_Stream : nullptr;
}

auto Stream::synchronize() const -> cudalernErr {
    if (!m_Stream || !*m_Stream) return cudaErrorInvalidValue;
    return cudaStreamSynchronize(*m_Stream);
}

auto Stream::valid() const noexcept -> bool {
    return m_Stream && *m_Stream != nullptr;
}

auto Stream::reset() -> cudalernErr {
    cudaStream_t raw = nullptr;
    auto err = cudaStreamCreate(&raw);
    if (err == cudaSuccess) {
        m_Stream = std::shared_ptr<cudaStream_t>(new cudaStream_t(raw), StreamDeleter());
    }
    return err;
}

auto Stream::take(cudaStream_t stream) noexcept -> void {
    m_Stream = std::shared_ptr<cudaStream_t>(new cudaStream_t(stream), StreamDeleter());
}

}  // namespace cudalern