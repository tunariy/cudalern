#include "benchtools/Core/Time.hpp"
#include "cuda_runtime_api.h"
#include "cudalern/Core/Device/device.hpp"
#include <cudalern/Containers/NdArray.hpp>

#include "benchtools/Benchmark/Benchmark.hpp"
#include "benchtools/Timers/WallTimer.hpp"
#include "benchtools/Timers/Wrappers/ScopedTimer.hpp"

#include <iostream>

using namespace cudalern;

void benchmark_matmul_1024() {
    static const std::array<size_t, 2> dims = {1024, 1024};
    static auto A = NdArray<float, 2>::random_uniform(dims);
    static auto B = NdArray<float, 2>::random_uniform(dims);
    auto stream = A.stream();  // capture the stream

    {
        auto C = A * B;
        C.synchronize();  // wait for the kernel
    }  // C is destroyed, its cudaFreeAsync is enqueued on stream

    cudaStreamSynchronize(stream);
}

auto main() -> int {
    cudalern::internal::InitializeContext(0);

    benchtools::benchmark<benchtools::Policy::Wall>(
        1000, benchtools::time_unit::nanoseconds, benchmark_matmul_1024);

    std::clog
        << std::endl;  // to get rid of the shitty ass % terminal content wrapper thingy
    return 0;
}