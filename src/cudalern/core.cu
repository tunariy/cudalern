#include "benchtools/Loggers/Logger.hpp"

#include "cudalern/Core/core.cuh"
#include "cudalern/Core/kernel.cuh"

#include <iostream>
#include <ostream>
#include <sstream>
#include <string>

namespace cudalern {

DeviceMemoryStatus::DeviceMemoryStatus() {
    cudaError_t cudaStatus;

    cudaStatus = cudaMemGetInfo(&mFreeAmount, &mTotalAmount);
    if (cudaStatus) {
        BENCHTOOLS_ERR(
            "Failed to retrieve memory info from the current context device\n");
    }
    mUsedAmount = mTotalAmount - mFreeAmount;
}

__host__ std::ostream& operator<<(std::ostream& stream,
                                  const DeviceMemoryStatus& memStatus) {
    stream << "Free amount: " << (memStatus.mFreeAmount / MB) << "MB" << std::endl
           << "Used amount: " << (memStatus.mUsedAmount / MB) << "MB" << std::endl
           << "Total available amount: " << (memStatus.mTotalAmount / MB) << "MB"
           << std::flush;
    return stream;
}

__host__ std::ostream& operator<<(std::ostream& stream, const cudaDeviceProp& devProps) {
    stream << "Device Properties:\n"
           << "--------------------------------\n"
           << "Device name: " << devProps.name << "\n"
           << "totalGlobalMem: " << (devProps.totalGlobalMem / MB) << " megabytes"
           << "\n"
           << "sharedMemPerBlock: " << devProps.sharedMemPerBlock << " bytes"
           << "\n"
           // << "regsPerBlock: " << devProps.regsPerBlock << "\n"
           << "maxThreadsPerBlock: " << devProps.maxThreadsPerBlock << " threads"
           << "\n"
           << "maxThreadsDim(x): " << devProps.maxThreadsDim[0] << " threads"
           << "\n"
           // << "maxThreadsDim(y): " << devProps.maxThreadsDim[1] << " threads" <<
           // "\n"
           // << "maxThreadsDim(z): " << devProps.maxThreadsDim[2] << " threads" <<
           // "\n"
           << "maxGridSize(x): " << devProps.maxGridSize[0] << " grids"
           << "\n"
           // << "maxGridSize(y): " << devProps.maxGridSize[1] << " grids" << "\n"
           // << "maxGridSize(z): " << devProps.maxGridSize[2] << " grids" << "\n"
           << "major CUDA compute capability: " << devProps.major << "\n"
           << "minor CUDA compute capability: " << devProps.minor << "\n"
           << "multiProcessorCount: " << devProps.multiProcessorCount << " processors"
           << "\n"
           << "memoryBusWidth: " << devProps.memoryBusWidth << " bits"
           << "\n"
           // << "l2CacheSize: " << (devProps.l2CacheSize / 1000000) << "MB" << "\n"
           << "maxThreadsPerMultiProcessor: " << devProps.maxThreadsPerMultiProcessor
           << " threads" << "\n"
           << "--------------------------------";
    return stream;
}

[[nodiscard]] __host__ std::string format(const cudaDeviceProp& devProps) {
    std::stringstream ss;
    ss << "Device Properties:\n"
       << "--------------------------------\n"
       << "Device name: " << devProps.name << "\n"
       << "totalGlobalMem: " << (devProps.totalGlobalMem / MB) << " megabytes"
       << "\n"
       << "sharedMemPerBlock: " << devProps.sharedMemPerBlock << " bytes"
       << "\n"
       // << "regsPerBlock: " << devProps.regsPerBlock << "\n"
       << "maxThreadsPerBlock: " << devProps.maxThreadsPerBlock << " threads"
       << "\n"
       << "maxThreadsDim(x): " << devProps.maxThreadsDim[0] << " threads"
       << "\n"
       // << "maxThreadsDim(y): " << devProps.maxThreadsDim[1] << " threads" <<
       // "\n"
       // << "maxThreadsDim(z): " << devProps.maxThreadsDim[2] << " threads" <<
       // "\n"
       << "maxGridSize(x): " << devProps.maxGridSize[0] << " grids"
       << "\n"
       // << "maxGridSize(y): " << devProps.maxGridSize[1] << " grids" << "\n"
       // << "maxGridSize(z): " << devProps.maxGridSize[2] << " grids" << "\n"
       << "major CUDA compute capability: " << devProps.major << "\n"
       << "minor CUDA compute capability: " << devProps.minor << "\n"
       << "multiProcessorCount: " << devProps.multiProcessorCount << " processors" << "\n"
       << "memoryBusWidth: " << devProps.memoryBusWidth << " bits"
       << "\n"
       // << "l2CacheSize: " << (devProps.l2CacheSize / 1000000) << "MB" << "\n"
       << "maxThreadsPerMultiProcessor: " << devProps.maxThreadsPerMultiProcessor
       << " threads" << "\n"
       << "--------------------------------";
    return ss.str();
}

__host__ void CUDAContextInit(int device) {
    auto cudaStatus = cudaSetDevice(device);
    if (cudaStatus) {
        BENCHTOOLS_ERR("Failed to set device! (incompatible GPU?)");
        return;
    }

    cudaDeviceProp deviceProps;
    if (cudaStatus) {
        BENCHTOOLS_ERR("Failed to retrieve device properties! (incompatible GPU?)");
        return;
    }

    BENCHTOOLS_INFO("Initializing context for device: " + std::to_string(device));
    cudaLaunchKernel(kernelWarmup, 1, 1, nullptr);
    cudaDeviceSynchronize();

    cudaStatus = cudaGetDeviceProperties(&deviceProps, 0);
    if (cudaStatus) {
        BENCHTOOLS_ERR("Failed to retrieve device properties! (unknown)");
        return;
    }
    BENCHTOOLS_INFO(format(deviceProps));
}
}  // namespace cudalern