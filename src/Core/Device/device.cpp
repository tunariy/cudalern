#include "cudalern/Core/Device/device.hpp"
#include "cudalern/ABI/kernel/kernel.cuh"
#include "cudalern/ABI/memory/stream.hpp"

#include "cudalern/Core/core.hpp"

#include <cuda_runtime_api.h>

#include <driver_types.h>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>

namespace cudalern {

namespace internal {

    [[nodiscard]] __host__ cudalernErr setDevice(int deviceID) noexcept {
        auto err = cudaSetDevice(deviceID);
        if (err)
            CUDALERN_ERROR_MESSAGE("Failed to set device! ", err);
        else
            CUDALERN_INFO("Device is now set to: " + std::to_string(deviceID));
        return err;
    }

    __host__ void InitializeContext(int device) {
        [[maybe_unused]] auto err = setDevice(device);
        if (err) CUDALERN_ERROR_MESSAGE("Failed to set device! ", err);

        CUDALERN_INFO("Initializing context for device: " + std::to_string(device));

        // Warmup the kernel
        Stream stream{};
        err = cudaLaunchKernel((void*)kernel::emptyCall, 1, 1, nullptr, 0, stream);
        if (err)
            CUDALERN_ERROR_MESSAGE("Failed to warmup the kernel! ", err);

        cudaDeviceSynchronize();
        if (err)
            CUDALERN_ERROR_MESSAGE("Failed to synch the device! ", err);

        CUDALERN_INFO(format(DeviceInfo::Properties(cudalern::internal::DEFAULT_DEVICE)));
    }
}  // namespace internal

DeviceProperties::DeviceProperties(int device) noexcept {

    m_deviceID = device;
    auto err = cudaGetDeviceProperties(&m_DeviceProps, device);
    if (err) CUDALERN_ERROR_MESSAGE("Failed to get device props!", err);

    m_name = m_DeviceProps.name;
    m_totalGlobalMem = m_DeviceProps.totalGlobalMem;
    m_sharedMemPerBlock = m_DeviceProps.sharedMemPerBlock;
    m_regsPerBlock = m_DeviceProps.regsPerBlock;
    m_maxThreadsPerBlock = m_DeviceProps.maxThreadsPerBlock;

    for (auto i{0}; i < 3; ++i) {
        m_maxThreadsDim[i] = m_DeviceProps.maxThreadsDim[i];
        m_maxGridSize[i] = m_DeviceProps.maxGridSize[i];
    }

    m_majorComp = m_DeviceProps.major;
    m_minorComp = m_DeviceProps.minor;
    m_multiProcessorCount = m_DeviceProps.multiProcessorCount;
    m_memoryBusWidth = m_DeviceProps.memoryBusWidth;
    m_l2CacheSize = m_DeviceProps.l2CacheSize;
    m_maxThreadsPerMultiProcessor = m_DeviceProps.maxThreadsPerMultiProcessor;
}

DeviceMemoryState::DeviceMemoryState(int device) noexcept {
    auto err = cudaMemGetInfo(&m_FreeAmount, &m_TotalAmount);
    if (err)
        CUDALERN_ERROR_MESSAGE(
            "Failed to retrieve memory info from the current context device", err);

    m_UsedAmount = m_TotalAmount - m_FreeAmount;
}

DeviceProperties& DeviceInfo::Properties(int device) noexcept {
    return DeviceProperties::getDeviceProps(device);
}

DeviceMemoryState& DeviceInfo::MemoryStatus(int device) noexcept {
    return DeviceMemoryState::getMemoryStatus(device);
}

[[nodiscard]] __host__ std::string format(std::ostream& stream,
                                          const DeviceMemoryState& memStatus) {
    std::stringstream ss;
    ss << "Free amount: " << (memStatus.m_FreeAmount / MB) << "MB" << std::endl
       << "Used amount: " << (memStatus.m_UsedAmount / MB) << "MB" << std::endl
       << "Total available amount: " << (memStatus.m_TotalAmount / MB) << "MB"
       << "\n";
    return ss.str();
}

[[nodiscard]] __host__ std::string format(const DeviceProperties& devProps) {
    std::stringstream ss;
    ss << "Device Properties:\n"
       << "--------------------------------\n"
       << "Device id: " << devProps.getID() << "\n"
       << "Device name: " << devProps.getName() << "\n"
       << "totalGlobalMem: " << (devProps.getTotalGlobalMem() / MB) << " megabytes\n"
       << "sharedMemPerBlock: " << devProps.getSharedMemPerBlock() << " bytes\n"
       << "regsPerBlock: " << devProps.getRegsPerBlock() << "\n"
       << "maxThreadsDim(x): " << devProps.getMaxThreadsDim()[0] << " threads\n"
       << "maxThreadsDim(y): " << devProps.getMaxThreadsDim()[1] << " threads\n"
       << "maxThreadsDim(z): " << devProps.getMaxThreadsDim()[2] << " threads\n"
       << "maxGridSize(x): " << devProps.getMaxGridSize()[0] << "\n"
       << "maxGridSize(y): " << devProps.getMaxGridSize()[1] << "\n"
       << "maxGridSize(z): " << devProps.getMaxGridSize()[2] << "\n"
       << "CUDA compute capability: sm_" << devProps.getMajorComp()
       << devProps.getMinorComp() << "\n"
       << "multiProcessorCount: " << devProps.getMultiProcessorCount() << " processors\n"
       << "memoryBusWidth: " << devProps.getMemoryBusWidth() << " bits\n"
       << "l2CacheSize: " << (devProps.getL2CacheSize() / MB) << "MB\n"
       << "maxThreadsPerMultiProcessor: " << devProps.getMaxThreadsPerMultiProcessor()
       << " threads\n"
       << "--------------------------------";
    return ss.str();
}
}  // namespace cudalern