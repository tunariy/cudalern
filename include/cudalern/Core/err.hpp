#pragma once

#include "driver_types.h"

#include "benchtools/Loggers/Logger.hpp"

#include <string>

namespace cudalern {

using namespace std::string_literals;

using cudalernErr = cudaError_t;

constexpr std::string errMessagePrefix = " Error Code:";

#define CUDALERN_CRITICAL(...) BENCHTOOLS_CRITICAL(__VA_ARGS__)
#define CUDALERN_ERR(...) BENCHTOOLS_ERR(__VA_ARGS__)
#define CUDALERN_INFO(...) BENCHTOOLS_INFO(__VA_ARGS__)
#define CUDALERN_TRACE(...) BENCHTOOLS_TRACE(__VA_ARGS__)
#define CUDALERN_WARN(...) BENCHTOOLS_WARN(__VA_ARGS__)

#define CUDALERN_ERROR_MESSAGE(x, err) (x + errMessagePrefix + std::to_string(err))

}  // namespace cudalern