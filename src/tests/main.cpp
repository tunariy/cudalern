#include "cudalern/Core/Device/device.hpp"
#include <cudalern/Containers/NdArray.hpp>

#include <iostream>
#include <vector>

auto main() -> int {
    cudalern::internal::InitializeContext(0);
    std::vector<std::vector<std::vector<int>>> data3d(
        3, std::vector<std::vector<int>>(3, std::vector<int>(3, 0)));

    int value{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                data3d[i][j][k] = value++;
            }
        }
    }

    auto nd3 = std::move(cudalern::NdArray<int, 3>(data3d));

    auto a = nd3[1];
    std::clog << nd3[1][1][1].get() << std::endl;
    nd3[1][1][1] = 31;
    std::clog << nd3[1][1][1].get() << std::endl;
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