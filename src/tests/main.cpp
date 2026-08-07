#include <array>
#include <cudalern/Containers/NdArray.hpp>

#include <benchtools/Benchmark/Benchmark.hpp>

#include <iostream>

using namespace cudalern;

static const std::array<size_t, 2> dims = {1024, 1024};

void benchmark_matmul_1024() {
    static auto A = NdArray<float, 2>::random_uniform(dims);
    static auto B = NdArray<float, 2>::random_uniform(dims);
    {
        auto C = A * B;
    }
}

auto main() -> int {
    cudalern::internal::InitializeContext(0);

    benchtools::benchmark<benchtools::Policy::Wall>(
        1000, benchtools::time_unit::nanoseconds, benchmark_matmul_1024);

    static auto D = NdArray<float, 3>::random_uniform(std::array{3ul, 3ul, 3ul});

    std::clog << D.print();

    std::clog
        << std::endl;  // to get rid of the shitty ass % terminal content wrapper thingy
    return 0;
}