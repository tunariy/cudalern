#pragma once

#include <cudalern/Core/concepts.hpp>
#include <cudalern/Core/constants.hpp>
#include <cudalern/Core/err.hpp>

#include "driver_types.h"

namespace cudalern {
using cudalernErr = cudaError_t;

namespace internal {
    constexpr auto DEFAULT_DEVICE{0};
}
}  // namespace cudalern
