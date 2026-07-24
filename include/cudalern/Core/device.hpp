#include "cudalern/Core/core.hpp"

#include <driver_types.h>

#include <string>

#if defined(_WIN32)
    #include <Windows.h>
#elif defined(__unix__)
    #include <unistd.h>
#endif

namespace cudalern {

[[nodiscard]] error_t setDevice(int deviceID) noexcept;

/**
 * @brief Wrapper object for cudaDeviceProps to retrieve relevant information
 */
struct DeviceProps {
  public:
    /**
     * @brief Get the device properties of a specific device
     *
     * @param device device id (default = 0)
     */
    DeviceProps(int device = 0) noexcept;

    // Name of the device (e.g., "GeForce RTX 3090")
    std::string name;

    // Total amount of global memory (VRAM) in bytes
    size_t totalGlobalMem;

    // Maximum shared memory per block in bytes
    size_t sharedMemPerBlock;

    // Maximum number of 32‑bit registers per thread block
    int regsPerBlock;

    // Maximum number of threads per block
    int maxThreadsPerBlock;

    // Maximum dimensions of a block (x, y, z)
    int maxThreadsDim[3];

    // Maximum dimensions of a grid (x, y, z)
    int maxGridSize[3];

    // Major compute capability version (e.g., 8 for sm_80)
    int majorComp;

    // Minor compute capability version (e.g., 6 for sm_86)
    int minorComp;

    // Number of streaming multiprocessors (SMs) on the device
    int multiProcessorCount;

    // Width of the memory bus in bits
    int memoryBusWidth;

    // Size of L2 cache in bytes
    int l2CacheSize;

    // Maximum number of threads that can be resident per SM (including all warps)
    int maxThreadsPerMultiProcessor;

  private:
    cudaDeviceProp m_DeviceProps;
};

/**
 * \brief Struct for retrieving memory statistics of current device
 **/
struct DeviceMemoryStatus {
    DeviceMemoryStatus(int device = 0) noexcept;

    size_t mFreeAmount;
    size_t mTotalAmount;
    size_t mUsedAmount;
};

struct DeviceInfo {
    static DeviceProps Properties(int device = 0) noexcept;

    static DeviceMemoryStatus MemoryStatus(int device = 0) noexcept;
};

/**
 * \brief Host function for initializing CUDA context
 **/
__host__ void InitializeContext(int device);

[[nodiscard]] __host__ std::string format(const DeviceMemoryStatus& memStatus);

[[nodiscard]] __host__ std::string format(const DeviceProps& devProps);

}  // namespace cudalern