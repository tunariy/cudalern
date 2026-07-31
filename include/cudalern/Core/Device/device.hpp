#pragma once
#include "cudalern/Core/core.hpp"

#include <driver_types.h>

#include <string>

namespace cudalern {

namespace internal {

    /**
     * @brief Set the current device context
     *
     * @param deviceID id of the device
     * @return error_t
     */
    [[nodiscard]] cudalernErr
    setDevice(int deviceID = cudalern::internal::DEFAULT_DEVICE) noexcept;

    /**
     * \brief Host function for initializing CUDA context
     **/
    __host__ void InitializeContext(int device = cudalern::internal::DEFAULT_DEVICE);
}  // namespace internal

/**
 * @brief Wrapper object for cudaDeviceProps to retrieve relevant information
 */
class DeviceProperties {
  public:
    static DeviceProperties&
    getDeviceProps(int device = cudalern::internal::DEFAULT_DEVICE) {
        static auto singleton = DeviceProperties(device);
        return singleton;
    }

    [[nodiscard]] int getID() const noexcept { return m_deviceID; }
    [[nodiscard]] std::string getName() const noexcept { return m_name; }
    [[nodiscard]] size_t getTotalGlobalMem() const noexcept { return m_totalGlobalMem; }
    [[nodiscard]] size_t getSharedMemPerBlock() const noexcept {
        return m_sharedMemPerBlock;
    }
    [[nodiscard]] int getRegsPerBlock() const noexcept { return m_regsPerBlock; }
    [[nodiscard]] int getMaxThreadsPerBlock() const noexcept {
        return m_maxThreadsPerBlock;
    }
    [[nodiscard]] const std::array<int, 3>& getMaxThreadsDim() const noexcept {
        return m_maxThreadsDim;
    }
    [[nodiscard]] const std::array<int, 3>& getMaxGridSize() const noexcept {
        return m_maxGridSize;
    }
    [[nodiscard]] int getMajorComp() const noexcept { return m_majorComp; }
    [[nodiscard]] int getMinorComp() const noexcept { return m_minorComp; }
    [[nodiscard]] int getMultiProcessorCount() const noexcept {
        return m_multiProcessorCount;
    }
    [[nodiscard]] int getMemoryBusWidth() const noexcept { return m_memoryBusWidth; }
    [[nodiscard]] int getL2CacheSize() const noexcept { return m_l2CacheSize; }
    [[nodiscard]] int getMaxThreadsPerMultiProcessor() const noexcept {
        return m_maxThreadsPerMultiProcessor;
    }

  private:
    /**
     * @brief Get the device properties of a specific device
     *
     * @param device device id (default = 0)
     */
    DeviceProperties(int device = cudalern::internal::DEFAULT_DEVICE) noexcept;

  private:
    int m_deviceID;
    std::string m_name;
    size_t m_totalGlobalMem{0};
    size_t m_sharedMemPerBlock{0};
    int m_regsPerBlock{0};
    int m_maxThreadsPerBlock{0};
    std::array<int, 3> m_maxThreadsDim{};
    std::array<int, 3> m_maxGridSize{};
    int m_majorComp{0};
    int m_minorComp{0};
    int m_multiProcessorCount{0};
    int m_memoryBusWidth{0};
    int m_l2CacheSize{0};
    int m_maxThreadsPerMultiProcessor{0};
    cudaDeviceProp m_DeviceProps;
};

/**
 * \brief Struct for retrieving memory statistics of current device
 **/
class DeviceMemoryState {
  public:
    static DeviceMemoryState&
    getMemoryStatus(int device = internal::DEFAULT_DEVICE) noexcept {
        static auto singleton = DeviceMemoryState(device);
        return singleton;
    }

    [[nodiscard]] size_t getFreeAmount() const noexcept { return m_FreeAmount; }
    [[nodiscard]] size_t getTotalAmount() const noexcept { return m_TotalAmount; }
    [[nodiscard]] size_t getUsedAmount() const noexcept { return m_UsedAmount; }

    friend class DeviceInfo;
    friend __host__ std::string format(std::ostream&, const DeviceMemoryState&);

  private:
    DeviceMemoryState(int device = cudalern::internal::DEFAULT_DEVICE) noexcept;

  private:
    size_t m_TotalAmount;
    size_t m_UsedAmount;
    size_t m_FreeAmount;
};

struct DeviceInfo {
    static DeviceProperties& Properties(int device = internal::DEFAULT_DEVICE) noexcept;

    static DeviceMemoryState&
    MemoryStatus(int device = internal::DEFAULT_DEVICE) noexcept;
};

class DeviceAPI {
  public:
    DeviceAPI(int device = cudalern::internal::DEFAULT_DEVICE) noexcept
        : m_deviceID(device), m_Props(DeviceInfo::Properties(device)) {}

    [[nodiscard]] cudalernErr setDevice(int deviceID) noexcept {
        m_deviceID = deviceID;
        [[maybe_unused]] auto err = internal::setDevice(deviceID);
        return err;
    };

    [[nodiscard]] int getDeviceID() const noexcept { return m_deviceID; }

    [[nodiscard]] DeviceProperties& getDeviceProps() const noexcept { return m_Props; }

    [[nodiscard]] DeviceMemoryState& getMemoryStatus() const noexcept {
        return DeviceInfo::MemoryStatus(m_deviceID);
    }

  private:
    DeviceProperties& m_Props;
    int m_deviceID{internal::DEFAULT_DEVICE};
};

[[nodiscard]] __host__ std::string format(const DeviceMemoryState& memStatus);

[[nodiscard]] __host__ std::string format(const DeviceProperties& devProps);

inline DeviceAPI gDeviceHandler{};
}  // namespace cudalern