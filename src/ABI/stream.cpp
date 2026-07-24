#include "cudalern/ABI/stream.hpp"

#include "cuda_runtime_api.h"

#include <utility>

namespace cudalern {

struct StreamDeleter {
    void operator()(cudaStream_t* ptr) const noexcept {
        if (ptr && *ptr) {
            cudaStreamDestroy(*ptr);
        }
        delete ptr;
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

auto Stream::synchronize() const -> error_t {
    if (!m_Stream || !*m_Stream) return cudaErrorInvalidValue;
    return cudaStreamSynchronize(*m_Stream);
}

auto Stream::valid() const noexcept -> bool {
    return m_Stream && *m_Stream != nullptr;
}

auto Stream::reset() -> error_t {
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

auto Stream::release() noexcept -> cudaStream_t {
    if (!m_Stream) return nullptr;
    cudaStream_t raw = *m_Stream;
    m_Stream.reset();
    return raw;
}

}  // namespace cudalern