#include "cudalern/Core/device.hpp"
#include <cudalern/Containers/NdArray.hpp>

#include <array>
#include <iostream>
#include <vector>

auto main() -> int {
    cudalern::InitializeContext(0);
    std::vector<std::vector<std::vector<int>>> data3d(
        3, std::vector<std::vector<int>>(3, std::vector<int>(3, 42)));

    const std::array<std::size_t, 3> dims = {3, 3, 3};

#if 1
    auto nd = cudalern::NdArray<int, 3>(data3d);
    std::clog << nd.read(1, 2, 3) << std::endl;
    auto a = nd.data();
    for (auto x : a) {
        std::clog << x << " ";
    }
    std::clog << std::endl;

    auto nd2 = nd;
    std::clog << nd2.read(1, 1, 1) << std::endl;

    auto nd3 = std::move(cudalern::NdArray<int, 3>(data3d));
    std::clog << nd3(1, 1, 1) << std::endl;

    auto t = nd.release();
    std::clog << t;

    auto ndddddd = cudalern::NdArray<int, 3>::pinned(dims);
#endif
    /*
     * SPACING
     * SPACING
     * SPACING
     * SPACING
     */
    std::clog
        << std::endl;  // to get rid of the shitty ass % terminal content wrapper thingy
    return 0;
}