#include "cudalern/Core/device.hpp"
#include "cudalern/ABI/kernel.cuh"
#include "cudalern/Core/constants.hpp"
#include "cudalern/Core/err.hpp"

#include <cuda_runtime_api.h>

#include "benchtools/Loggers/Logger.hpp"

#include <cstddef>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>

namespace cudalern {

DeviceProps::DeviceProps(int device) noexcept {
    [[maybe_unused]] auto err = setDevice(device);

    err = cudaGetDeviceProperties(&m_DeviceProps, 0);
    if (err) {
        BENCHTOOLS_ERR("Failed to retrieve device properties! " + CUDALERN_ERROR(err));
        return;
    }

    name = m_DeviceProps.name;
    totalGlobalMem = m_DeviceProps.totalGlobalMem;
    sharedMemPerBlock = m_DeviceProps.sharedMemPerBlock;
    regsPerBlock = m_DeviceProps.regsPerBlock;
    maxThreadsPerBlock = m_DeviceProps.maxThreadsPerBlock;
    for (size_t i{0}; i < 3; i++) {
        maxThreadsDim[i] = m_DeviceProps.maxThreadsDim[i],
        maxGridSize[i] = m_DeviceProps.maxGridSize[i];
    }
    majorComp = m_DeviceProps.major;
    minorComp = m_DeviceProps.minor;
    multiProcessorCount = m_DeviceProps.multiProcessorCount;
    memoryBusWidth = m_DeviceProps.memoryBusWidth;
    l2CacheSize = m_DeviceProps.l2CacheSize;
    maxThreadsPerMultiProcessor = m_DeviceProps.maxThreadsPerMultiProcessor;
};

DeviceMemoryStatus::DeviceMemoryStatus(int device) noexcept {
    [[maybe_unused]] auto err = setDevice(device);

    err = cudaMemGetInfo(&mFreeAmount, &mTotalAmount);
    if (err)
        BENCHTOOLS_ERR(
            "Failed to retrieve memory info from the current context device\n" +
            CUDALERN_ERROR(err));

    mUsedAmount = mTotalAmount - mFreeAmount;
}

DeviceProps DeviceInfo::Properties(int device) noexcept {
    return DeviceProps(device);
}

DeviceMemoryStatus DeviceInfo::MemoryStatus(int device) noexcept {
    return DeviceMemoryStatus(device);
}

[[nodiscard]] __host__ error_t setDevice(int deviceID = 0) noexcept {
    auto err = cudaSetDevice(deviceID);
    if (err)
        BENCHTOOLS_ERR("Failed to set device! " + CUDALERN_ERROR(err));
    else
        BENCHTOOLS_INFO("Device is now set to: " + std::to_string(deviceID));
    return err;
}

__host__ void InitializeContext(int device) {
    cudaSetDevice(device);

    BENCHTOOLS_INFO("Initializing context for device: " + std::to_string(device));

    // Warmup the kernel
    cudaLaunchKernel((void*)kernel::emptyCall, 1, 1, nullptr, 0, nullptr);
    cudaDeviceSynchronize();

    BENCHTOOLS_INFO(format(DeviceInfo::Properties(0)));
}

[[nodiscard]] __host__ std::string format(std::ostream& stream,
                                          const DeviceMemoryStatus& memStatus) {
    std::stringstream ss;
    ss << "Free amount: " << (memStatus.mFreeAmount / MB) << "MB" << std::endl
       << "Used amount: " << (memStatus.mUsedAmount / MB) << "MB" << std::endl
       << "Total available amount: " << (memStatus.mTotalAmount / MB) << "MB"
       << "\n";
    return ss.str();
}

[[nodiscard]] __host__ std::string format(const DeviceProps& devProps) {
    std::stringstream ss;
    ss << "Device Properties:\n"
       << "--------------------------------\n"
       << "Device name: " << devProps.name << "\n"
       << "totalGlobalMem: " << (devProps.totalGlobalMem / MB) << " megabytes"
       << "\n"
       << "sharedMemPerBlock: " << devProps.sharedMemPerBlock << " bytes"
       << "\n"
       << "regsPerBlock: " << devProps.regsPerBlock << "\n"
       << "maxThreadsPerBlock: " << devProps.maxThreadsPerBlock << " threads"
       << "\n"
       << "maxThreadsDim(x): " << devProps.maxThreadsDim[0] << " threads"
       << "\n"
       << "maxThreadsDim(y): " << devProps.maxThreadsDim[1] << " threads" << "\n"
       << "maxThreadsDim(z): " << devProps.maxThreadsDim[2] << " threads" << "\n"
       << "maxGridSize(x): " << devProps.maxGridSize << " grids"
       << "\n"
       << "maxGridSize(x): " << devProps.maxGridSize[0] << "\n"
       << "maxGridSize(y): " << devProps.maxGridSize[1] << "\n"
       << "maxGridSize(z): " << devProps.maxGridSize[2] << "\n"
       << "CUDA compute capability: " << "sm_" << devProps.majorComp << devProps.minorComp
       << "\n"
       << "multiProcessorCount: " << devProps.multiProcessorCount << " processors" << "\n"
       << "memoryBusWidth: " << devProps.memoryBusWidth << " bits"
       << "\n"
       << "l2CacheSize: " << (devProps.l2CacheSize / MB) << "MB" << "\n"
       << "maxThreadsPerMultiProcessor: " << devProps.maxThreadsPerMultiProcessor
       << " threads" << "\n"
       << "--------------------------------";
    return ss.str();
}
}  // namespace cudalern