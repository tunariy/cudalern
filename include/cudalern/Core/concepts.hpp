#pragma once

#include <cstddef>
#include <ranges>
#include <type_traits>

template <class T>
concept ElementIterable = std::ranges::range<T>;

template <class T>
concept SizeValue = std::is_same_v<T, std::size_t> || std::is_same_v<T, size_t> ||
                    std::is_same_v<T, unsigned int>;

template <typename T>
concept CudaCompatible = std::is_trivially_copyable_v<T> && !std::is_pointer_v<T>;

template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

template <typename R>
concept Range = std::ranges::range<R>;

template <typename R>
concept NestedRange = Range<R> && Range<std::ranges::range_value_t<R>>;
