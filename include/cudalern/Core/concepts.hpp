#include <cstddef>
#include <ranges>
#include <type_traits>

template <class T>
concept ElementIterable = std::ranges::range<T>;

template <class T>
concept SizeValue = std::is_same_v<T, std::size_t> || std::is_same_v<T, size_t> ||
                    std::is_same_v<T, unsigned int>;

template <typename R>
concept Range = std::ranges::range<R>;

template <typename R>
concept NestedRange = Range<R> && Range<std::ranges::range_value_t<R>>;
