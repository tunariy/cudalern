#pragma once
// cudart
#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// cuBLAS
#include <cublasLt.h>
#include <cublas_v2.h>

#include <benchtools/Loggers/Logger.hpp>

#if defined(_WIN32)
    #include <Windows.h>
#elif defined(__unix__)
    #include <unistd.h>
#endif

namespace cudalern {

/**
 * Constants for size conversion
 **/
constexpr auto KB = 1024u;
constexpr auto MB = KB * 1024;
constexpr auto GB = MB * 1024;

/**
 * \brief Struct for storing memory statistics of current device
 **/
struct DeviceMemoryStatus {
    DeviceMemoryStatus();
    size_t mFreeAmount;
    size_t mTotalAmount;
    size_t mUsedAmount;
};

#if defined(_WIN32)
inline uint64_t getTotalSystemMemory() {
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return status.ullTotalPhys;
}
#elif defined(__unix__)
inline uint64_t getTotalSystemMemory() {
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return pages * page_size;
}
#endif

__host__ std::ostream& operator<<(std::ostream& stream,
                                  const DeviceMemoryStatus& memStatus);

__host__ std::ostream& operator<<(std::ostream& stream, const cudaDeviceProp& devProps);

[[nodiscard]] __host__ std::string format(const cudaDeviceProp& devProps);

/**
 * \brief Host function for initializing CUDA context
 **/
__host__ void CUDAContextInit(int device);

}  // namespace cudalern
