# cudalern

- Machine Learning library for C++(20) with CUDA.

## Current Features

- [x] N-dimensional array on host (pinned) or device memory

```cpp
static const std::array<size_t, 3> dims = {3, 3, 3};
auto nd {cudalern::NdArray<int, 3>::random_uniform(dims)};
nd[1][1][1] = 31;
std::cout << nd[1][1][1];
```

```cpp
// or initialize with
auto nd {cudalern::NdArray<int, 2>(
    std::vector<int>{1, 2}, 
    std::vector<int>{2, 1})} 
    // produces [[1, 2], [2, 1]]
```

- [x] Support for many operations

```cpp
static const std::array<size_t, 2> dims = {1024, 1024};
auto A = NdArray<float, 2>::random_uniform(dims);
auto B = NdArray<float, 2>::random_uniform(dims);
auto C = A * B; // matrix multiplication
```

- [x] Visualize your N-dimensional arrays

```cpp
auto A = NdArray<float, 3>::random_uniform(dims);
```

```bash
[[[0.470848, 0.473661, 0.340334], [0.990252, 0.549664, 0.745849], ..]]
```

- [x] Runtime device/host API for retrieving memory information or properties

```txt
[2026-08-03 06:01:29.741] [GLOBAL] [info] Device is now set to: 
[2026-08-03 06:01:29.741] [GLOBAL] [info] Initializing context for device: 0
[2026-08-03 06:01:29.742] [GLOBAL] [info] Device Properties:
--------------------------------
Device id: 0
Device name: NVIDIA GeForce RTX 4060 Laptop GPU
totalGlobalMem: 7834 megabytes
sharedMemPerBlock: 49152 bytes
regsPerBlock: 65536
maxThreadsDim(x): 1024 threads
maxThreadsDim(y): 1024 threads
maxThreadsDim(z): 64 threads
maxGridSize(x): 2147483647
maxGridSize(y): 65535
maxGridSize(z): 65535
CUDA compute capability: sm_89
multiProcessorCount: 24 processors
memoryBusWidth: 128 bits
l2CacheSize: 32MB
maxThreadsPerMultiProcessor: 1536 threads
--------------------------------
```

- [x] Custom wrappers for launching kernels

```cpp
launchKernel1D(kernel::eye<T>, 0, stream, size, device_ptr, dim, dim);
```

- [x] Kernel launch configuration is done at runtime optimized for the currently set device

```cpp
auto& props = gDeviceHandler.getDeviceProps();
auto maxThreadsPerBlock = props.getMaxThreadsPerBlock();

auto threadsPerBlock = maxThreadsPerBlock;
auto numBlocks = (N + maxThreadsPerBlock - 1) / maxThreadsPerBlock;
```

- [x] Custom allocator for device, host memory management

```cpp
enum class allocatorPolicy : uint8_t { Pinned, Device, Managed };

template <class T>
    requires(std::is_integral_v<T> || std::is_floating_point_v<T>)
class allocator {...};
```

- [x] Custom CUDA stream management

## TODO

## Autograd and Training

- [ ] Core Autograd
- [ ] Neural Network Layers
- [ ] Loss Functions
- [ ] Optimizers
