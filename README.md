# cudalern

- Machine Learning library for C++(20) with CUDA.

## Current Features

- [x] N-dimensional array on pinned or device memory

```cpp
auto nd3 {cudalern::NdArray<int, 3>(data3d)};

nd3[1][1][1] = 31;
```

- [x] Runtime device/host API for retrieving memory information or properties

```bash
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

- [x] Custom wrappers for kernel launches

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

## NdArray

- [ ] Full element wise operation support

## Autograd and Training

- [ ] Core Autograd
- [ ] Neural Network Layers
- [ ] Loss Functions
- [ ] Optimizers
