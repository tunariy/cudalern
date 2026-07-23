/* NOTE TO SELF
 * shit is complicated we need a ND array with view, insert, replace etc. functionalities
 * nd array should be on the device not host right?
 * how should the device handle it...
 */

#include "benchtools/Loggers/Logger.hpp"
#include "cudalern/Core/core.cuh"

#include <array>
#include <cudalern/Containers/NdArray.hpp>

#include <iostream>
#include <vector>

auto main() -> int {
    cudalern::CUDAContextInit(0);
    std::vector<std::vector<std::vector<int>>> data3d(
        3, std::vector<std::vector<int>>(3, std::vector<int>(3, 42)));

    const std::array<std::size_t, 3> dims = {3, 3, 3};

#if 1
    auto nd = cudalern::NdArray<int, 3>(data3d);
    BENCHTOOLS_INFO(nd.read(1, 2, 3));
    auto a = nd.data();
    for (auto x : a) {
        std::clog << x << " ";
    }
    std::clog << std::endl;

    auto nd2 = nd;
    BENCHTOOLS_INFO(nd2.read(1, 1, 1));

    auto nd3 = std::move(cudalern::NdArray<int, 3>(data3d));
    BENCHTOOLS_INFO(nd3(1, 1, 1));

    auto t = nd.release();
    std::clog << t;
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