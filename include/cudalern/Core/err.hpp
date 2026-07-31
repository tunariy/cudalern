#pragma once

#include <string>

using namespace std::string_literals;

#define CUDALERN_ERROR(err) ("Error Code:"s + std::to_string(err))

#define CUDALERN_ERROR_MESSAGE(x, err) (x + " Error Code:"s + std::to_string(err))
