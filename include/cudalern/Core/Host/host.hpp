#pragma once

#include "cudalern/Core/core.hpp"

#include <cstdint>
#if defined(_WIN32)
    #include <Windows.h>
#elif defined(__unix__)
    #include <unistd.h>
#endif

namespace cudalern {
namespace internal {

#if defined(_WIN32)
    inline uint64_t getHostMemoryBytes() {
        MEMORYSTATUSEX status;
        status.dwLength = sizeof(status);
        GlobalMemoryStatusEx(&status);
        return status.ullTotalPhys;
    }

    inline uint64_t getHostFreeMemoryBytes() {
        MEMORYSTATUSEX status;
        status.dwLength = sizeof(status);
        GlobalMemoryStatusEx(&status);
        return status.ullAvailPhys;  // free physical memory
    }

#elif defined(__linux__)
    #include <sys/sysinfo.h>

    inline uint64_t getHostMemoryBytes() {
        struct sysinfo info;
        if (sysinfo(&info) != 0) return 0;
        return info.totalram;  // total RAM in bytes
    }

    inline uint64_t getHostFreeMemoryBytes() {
        struct sysinfo info;
        if (sysinfo(&info) != 0) return 0;
        return info.freeram;  // free RAM in bytes
    }

#else
    // Fallback for other Unix (macOS, BSD, etc.)
    inline uint64_t getHostMemoryBytes() {
        long pages = sysconf(_SC_PHYS_PAGES);
        long page_size = sysconf(_SC_PAGE_SIZE);
        return pages * page_size;
    }

    inline uint64_t getHostFreeMemoryBytes() {
        // Not implemented for this platform; return 0 or throw.
        return 0;
    }
#endif

    class HostProperties {
      public:
        static HostProperties& getHostProps() {
            static auto singleton = HostProperties();
            return singleton;
        }

      private:
        HostProperties() noexcept;

      private:
    };

    /**
     * \brief Struct for retrieving memory statistics of current Host
     **/
    class HostMemoryState {
      public:
        static HostMemoryState& getMemoryStatus() noexcept {
            static auto singleton = HostMemoryState();
            return singleton;
        }

        [[nodiscard]] size_t getFreeAmount() const noexcept { return m_FreeAmount; }
        [[nodiscard]] size_t getTotalAmount() const noexcept { return m_TotalAmount; }
        [[nodiscard]] size_t getUsedAmount() const noexcept { return m_UsedAmount; }

        friend class HostInfo;
        friend __host__ std::string format(std::ostream&, const HostMemoryState&);

      private:
        HostMemoryState() noexcept
            : m_TotalAmount(internal::getHostMemoryBytes()),
              m_FreeAmount(internal::getHostFreeMemoryBytes()),
              m_UsedAmount(m_TotalAmount - m_FreeAmount) {};

      private:
        size_t m_TotalAmount{};
        size_t m_FreeAmount{};
        size_t m_UsedAmount{};
    };

    struct HostInfo {
        static HostProperties& Properties() noexcept;

        static HostMemoryState& MemoryStatus() noexcept;
    };
}  // namespace internal

class HostAPI {
  public:
    HostAPI() noexcept : m_Props(internal::HostInfo::Properties()) {}

    [[nodiscard]] internal::HostProperties getHostProps() const noexcept {
        return m_Props;
    }

    [[nodiscard]] internal::HostMemoryState& getHostMemory() const noexcept {
        return internal::HostMemoryState::getMemoryStatus();
    }

  private:
    internal::HostProperties& m_Props;
};

[[nodiscard]] __host__ std::string format(const internal::HostMemoryState& memStatus);

[[nodiscard]] __host__ std::string format(const internal::HostProperties& devProps);

}  // namespace cudalern
