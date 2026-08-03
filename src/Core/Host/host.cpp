#include <cudalern/Core/Host/host.hpp>

namespace cudalern {
[[nodiscard]] __host__ std::string format(const internal::HostMemoryState& memStatus) {
    std::stringstream ss;
    ss << "Free amount: " << (memStatus.getFreeAmount() / MB) << "MB" << std::endl
       << "Used amount: " << (memStatus.getUsedAmount() / MB) << "MB" << std::endl
       << "Total available amount: " << (memStatus.getTotalAmount() / MB) << "MB"
       << "\n";
    return ss.str();
};
}  // namespace cudalern