# NdArray - Development TODO

## Core Infrastructure

### Stream Management

- [x] Add `cudaStream_t m_Stream` member to NdArray
- [x] Add public getter: `[[nodiscard]] cudaStream_t stream() const noexcept { return m_Stream; }`
- [x] Add public sync method: `void synchronize() { if (m_Stream) cudaStreamSynchronize(m_Stream); }`
- [x] Create Stream RAII wrapper class (`Stream.hpp`/`Stream.cpp`)
- [x] Replace all raw `cudaStream_t` usage in NdArray with `Stream` class
- [x] Update constructors to create/destroy streams properly
- [x] Update copy/move semantics for stream ownership

### Memory Management

- [ ] Design and implement C++-conforming CUDA allocator
  - [ ] deallocate_n???: `void deallocate(T* ptr, std::size_t n)`
    - isn't even possible unless a wrapper is created for cuda array type(s)????

### Type Traits and Concepts

- [x] Add concepts for valid NdArray element types
- [x] Add more compile-time checks: `static_assert(CudaCompatible<T>, "T must be trivially copyable");`

## Memory Operations

### Data Movement

- [x] Host to Device transfers
  - [x] Asynchronous/Synchronous: `static NdArray from_host_async(const std::vector<T>& data, const std::array<size_t, Rank>& dims, cudaStream_t stream)`
- [x] Device to Host transfers
  - [x] Asynchronous: `void to_host_async(std::vector<T>& out, cudaStream_t stream) const`
- [x] Device to Device transfers
  - [x] Copy constructor: `NdArray(const NdArray& other)`
  - [x] Copy assignment: `NdArray& operator=(const NdArray& other)`

### Memory Types

- [x] Device memory (default)
  - [x] `cudaMalloc` / `cudaFree`
  - [x] `cudaMallocAsync` / `cudaFreeAsync` with streams
- [x] Pinned (page-locked) memory
  - [x] `cudaMallocHost` / `cudaFreeHost`
  - [x] Factory method: `static NdArray pinned(const std::array<size_t, Rank>& dims)`
- [x] Unified memory (optional)
  - [x] `cudaMallocManaged` / `cudaFree`
  - [x] Factory method: `static NdArray managed(const std::array<size_t, Rank>& dims)`

### Initialization

- [x] Fill with value
  - [x] `cudaMemsetAsync` for byte-sized types
  - [x] Custom kernel for other types
- [x] Factory methods
  - [x] `static NdArray zeros(const std::array<size_t, Rank>& dims)`
  - [x] `static NdArray ones(const std::array<size_t, Rank>& dims)`
  - [x] `static NdArray full(const std::array<size_t, Rank>& dims, const T& value)`
  - [x] `static NdArray arange(T start, T stop, T step = 1)` (Rank=1 only)
  - [x] `static NdArray eye(size_t n)` (Rank=2 only)
  - [x] `static NdArray random_uniform(const std::array<size_t, Rank>& dims, T low = 0, T high = 1)`
  - [x] `static NdArray random_normal(const std::array<size_t, Rank>& dims, T mean = 0, T std = 1)`

## Array Operations

### Core Operations

- [ ] Element-wise operations
  - [ ] Unary: `negate()`, `abs()`, `sqrt()`, `exp()`, `log()`, `sin()`, `cos()`, `tanh()`
  - [ ] Binary: `add()`, `sub()`, `mul()`, `div()`, `pow()`, `mod()`
  - [ ] Comparison: `eq()`, `ne()`, `lt()`, `gt()`, `le()`, `ge()`
  - [ ] In-place versions: `operator+=`, `operator-=`, `operator*=`, `operator/=`
- [ ] Reduction operations
  - [ ] `sum(dim)`, `mean(dim)`, `max(dim)`, `min(dim)`, `prod(dim)`
  - [ ] `argmax(dim)`, `argmin(dim)`
  - [ ] `all(dim)`, `any(dim)`
  - [ ] Global reductions: `sum_all()`, `mean_all()`, `max_all()`, `min_all()`
- [ ] Matrix operations (Rank=2)
  - [ ] `matmul(const NdArray& other) const` - matrix multiplication
  - [ ] `transpose() const`
  - [ ] `inverse() const` (optional)
  - [ ] `determinant() const` (optional)
  - [ ] `trace() const`

### Shape Manipulation

- [ ] `reshape(const std::array<size_t, NewRank>& new_dims) const` - returns view
- [ ] `flatten() const` - returns 1D view
- [ ] `squeeze() const` - remove size-1 dimensions
- [ ] `unsqueeze(size_t dim) const` - add size-1 dimension
- [ ] `transpose(const std::array<size_t, Rank>& permutation) const` - reorder dimensions
- [ ] `broadcast_to(const std::array<size_t, NewRank>& new_shape) const` - returns view with broadcast semantics

### Views and Slicing

- [ ] `operator[](size_t index) const` - returns view dropping first dimension
- [ ] `slice(const std::array<size_t, Rank>& starts, const std::array<size_t, Rank>& ends, const std::array<size_t, Rank>& steps = {}) const`
- [ ] `contiguous() const` - returns contiguous copy if needed
- [ ] `const_view() const` - read-only view

### Indexing

- [x] `T read(indices...) const` - single element access
- [ ] `void write(const T& value, indices...)` - single element assignment
- [ ] `T operator()(indices...) const` - function call syntax for read
- [ ] `T& operator()(indices...)` - function call syntax for write (needs proxy)
- [ ] Advanced indexing: integer arrays, boolean masks

## Machine Learning Core

### Automatic Differentiation

- [ ] Computational graph infrastructure
  - [ ] `class Node` - represents operations in graph
  - [ ] `class Edge` - dependencies between nodes
  - [ ] `topological_sort()` - order nodes for evaluation
- [ ] Gradient tape / tracer
  - [ ] `GradientTape` class records operations
  - [ ] `tape.watch(tensor)` - track tensor for gradients
  - [ ] `tape.stop_recording()` / `tape.resume_recording()`
  - [ ] Gradient checkpointing for memory optimization
- [ ] Backward pass implementation
  - [ ] Chain rule for scalar functions
  - [ ] Vector-Jacobian product (VJP) for matrix operations
  - [ ] Gradient computation for each operation type
- [ ] Gradient utilities
  - [ ] `zero_grad()` - reset gradients
  - [ ] `backward()` - compute gradients
  - [ ] `detach()` - detach from computation graph
  - [ ] `requires_grad()` flag
  - [ ] `grad()` - access computed gradients

### Neural Network Layers

- [ ] Base Layer class
  - [ ] `virtual Tensor forward(const Tensor& input) = 0`
  - [ ] `virtual Tensor backward(const Tensor& grad_output) = 0`
  - [ ] `parameters()` - return trainable parameters
  - [ ] `to_device(cudaStream_t stream)` - move layer to device
  - [ ] `train()` / `eval()` - set mode
- [ ] Linear / Dense layer
  - [ ] `Linear` class with weights and bias
  - [ ] Weight initialization: Xavier, Kaiming, Uniform, Normal
  - [ ] `forward(input)` - computes `input @ weight.T + bias`
- [ ] Convolutional layers
  - [ ] `Conv1d`, `Conv2d`, `Conv3d` classes
  - [ ] Padding modes: zero, reflect, replicate
  - [ ] Stride and dilation support
  - [ ] Groups for depthwise convolution
- [ ] Pooling layers
  - [ ] `MaxPool1d`, `MaxPool2d`, `MaxPool3d`
  - [ ] `AvgPool1d`, `AvgPool2d`, `AvgPool3d`
  - [ ] `AdaptiveMaxPool`, `AdaptiveAvgPool`
- [ ] Normalization layers
  - [ ] `BatchNorm1d`, `BatchNorm2d`, `BatchNorm3d`
  - [ ] `LayerNorm`
  - [ ] `InstanceNorm`
  - [ ] `GroupNorm`
- [ ] Activation functions
  - [ ] `ReLU`, `LeakyReLU`, `PReLU`
  - [ ] `Sigmoid`, `Tanh`
  - [ ] `GELU`, `Swish`, `Mish`
  - [ ] `Softmax`, `LogSoftmax`
- [ ] Dropout layers
  - [ ] `Dropout` with training/eval mode
  - [ ] `Dropout2d`, `Dropout3d`
- [ ] Recurrent layers
  - [ ] `RNN`, `LSTM`, `GRU` classes
  - [ ] Bidirectional support
  - [ ] Stacked layers (num_layers > 1)

### Loss Functions

- [ ] Regression losses
  - [ ] `MSELoss` - Mean Squared Error
  - [ ] `MAELoss` - Mean Absolute Error
  - [ ] `HuberLoss`
  - [ ] `SmoothL1Loss`
- [ ] Classification losses
  - [ ] `CrossEntropyLoss`
  - [ ] `BCELoss` - Binary Cross Entropy
  - [ ] `NLLLoss` - Negative Log Likelihood
  - [ ] `KLDivLoss` - KL Divergence
- [ ] Custom loss support
  - [ ] `LambdaLoss` for custom functions
  - [ ] `CombinedLoss` for multiple losses

### Optimizers

- [ ] Base Optimizer class
  - [ ] `step()` - update parameters
  - [ ] `zero_grad()` - reset gradients
  - [ ] `add_param_group()` - add parameter groups
  - [ ] `state_dict()` / `load_state_dict()` - save/load optimizer state
- [ ] SGD
  - [ ] Momentum support
  - [ ] Nesterov acceleration
  - [ ] Weight decay
- [ ] Adam
  - [ ] Beta1, Beta2 parameters
  - [ ] Epsilon for numerical stability
  - [ ] Weight decay
- [ ] AdamW - decoupled weight decay
- [ ] RMSprop
- [ ] Adagrad
- [ ] Adadelta
- [ ] Learning rate schedulers
  - [ ] `StepLR`, `MultiStepLR`
  - [ ] `ExponentialLR`
  - [ ] `CosineAnnealingLR`
  - [ ] `ReduceLROnPlateau`

### Model Management

- [ ] Model container classes
  - [ ] `Sequential` - sequential container
  - [ ] `ModuleList`, `ModuleDict`
  - [ ] `ParameterList`, `ParameterDict`
- [ ] Model serialization
  - [ ] `state_dict()` - return all parameters
  - [ ] `load_state_dict()` - load parameters
  - [ ] `save(const std::string& path)` - save model
  - [ ] `load(const std::string& path)` - load model
- [ ] Model modes
  - [ ] `train()` - enable training mode
  - [ ] `eval()` - enable evaluation mode
- [ ] Parameter utilities
  - [ ] `freeze()` / `unfreeze()` - disable/enable gradient computation
  - [ ] `clip_grad_norm()` - gradient clipping
  - [ ] `clip_grad_value()` - gradient clipping by value

### Data Pipeline

- [ ] Dataset abstraction
  - [ ] `Dataset<Sample>` concept
  - [ ] `MapDataset` - map-style dataset
  - [ ] `IterableDataset` - iterable-style dataset
- [ ] DataLoader
  - [ ] `DataLoader` class with batch iteration
  - [ ] Shuffle support
  - [ ] Multi-threaded data loading
  - [ ] Pinned memory for faster transfers
  - [ ] `__iter__()` and `__next__()` support
- [ ] Data transforms
  - [ ] `ToTensor` - convert to NdArray
  - [ ] `ToDevice` - move to device
  - [ ] `Normalize` - mean/std normalization
  - [ ] `Standardize` - z-score standardization
  - [ ] `RandomHorizontalFlip`, `RandomVerticalFlip`
  - [ ] `RandomRotation`, `RandomCrop`, `RandomResizedCrop`
  - [ ] `Compose` - chain transforms together
- [ ] Common datasets
  - [ ] `MNIST`, `FashionMNIST`
  - [ ] `CIFAR10`, `CIFAR100`
  - [ ] `ImageFolder` for custom image datasets
  - [ ] `CSVDataset` for tabular data

### Training Loop

- [ ] Training utilities
  - [ ] `train_epoch()` - single epoch training
  - [ ] `validate()` - evaluation loop
  - [ ] `train_loop()` - full training with early stopping
- [ ] Metrics
  - [ ] `Accuracy`, `Precision`, `Recall`, `F1`
  - [ ] `ConfusionMatrix`
  - [ ] `AUC_ROC`
- [ ] Checkpointing
  - [ ] Save best model
  - [ ] Resume from checkpoint
- [ ] Progress tracking
  - [ ] Loss history
  - [ ] Learning rate tracking
  - [ ] Integration with benchtools logging
  - [ ] Progress bar support

### CUDA Kernel Optimizations

- [ ] Memory coalescing patterns for better performance
- [ ] Shared memory usage for reductions
- [ ] Tile-based matrix multiplication
- [ ] Kernel fusion for element-wise operations
- [ ] Warp-level primitives for reductions
- [ ] Async kernel launches with streams
- [ ] TensorCore utilization for matrix operations

## Utilities

### Debugging and Inspection

- [ ] `print(size_t limit = 10)` - log shape and first N elements
- [ ] `info()` - return string with metadata (shape, strides, size, memory)
- [ ] `is_contiguous()` - check if strides match row-major
- [ ] `memory_usage()` - return bytes used
- [ ] `device()` - return current device

### Serialization

- [ ] `save(const std::string& path)` - write dimensions + raw binary
- [ ] `static NdArray load(const std::string& path)` - read and allocate on device
- [ ] `.npy` format compatibility
- [ ] HDF5 support (optional)

### Iteration

- [ ] Host iterators (requires copying to host)
- [ ] Device iterators for use in kernels
- [ ] Range-based for support on host

## Type Specializations

### Scalar Support (Rank=0)

- [ ] `NdArray<T, 0>` specialization
  - [ ] Store single element on device
  - [ ] `operator T()` - convert to host value
  - [ ] `operator=(T)` - assign from host value
  - [ ] Arithmetic operators: `+=`, `-=`, `*=`, `/=`
  - [ ] Comparison operators: `==`, `!=`, `<`, `>`, `<=`, `>=`

### Vector Support (Rank=1)

- [ ] Additional methods for 1D arrays
  - [ ] `dot(const NdArray& other) const` - dot product
  - [ ] `norm(NormType type = L2) const` - L1, L2, infinity
  - [ ] `sort()` - sort in-place
  - [ ] `unique()` - return unique elements

### Matrix Support (Rank=2)

- [ ] Additional methods for 2D arrays
  - [ ] `transpose()` - matrix transpose
  - [ ] `matmul(const NdArray& other) const` - matrix multiplication
  - [ ] `solve(const NdArray& b) const` - solve linear systems
  - [ ] `det()` - determinant
  - [ ] `inv()` - inverse

## Testing

### Unit Tests

- [ ] Construction and destruction
- [ ] Copy and move semantics
- [ ] `fill()` and factory methods
- [ ] `read()` and `write()`
- [ ] Shape operations (`reshape`, `flatten`, etc.)
- [ ] Element-wise operations
- [ ] Reduction operations
- [ ] Matrix operations
- [ ] Autograd functionality
- [ ] Neural network layers
- [ ] Optimizers
- [ ] DataLoader
- [ ] Loss functions

### Integration Tests

- [ ] Stream and async operations
- [ ] Pinned memory transfers
- [ ] Multi-stream scenarios
- [ ] Memory leak detection
- [ ] End-to-end training loop
- [ ] Model serialization

### Benchmarks

- [ ] vs numpy for CPU
- [ ] vs cupy for GPU
- [ ] vs PyTorch for GPU
- [ ] Scaling with array size
- [ ] Scaling with number of streams
- [ ] Kernel performance profiling

## Documentation

### API Documentation

- [ ] Doxygen comments for all public methods
- [ ] Usage examples for common operations
- [ ] Performance notes and gotchas

### User Guide

- [ ] Getting started guide
- [ ] Basic operations tutorial
- [ ] Advanced usage (streams, pinned memory)
- [ ] Neural network training tutorial
- [ ] Custom layers and models
- [ ] Complete API reference

### Developer Guide

- [ ] Architecture overview
- [ ] Adding new operations
- [ ] Custom memory allocators
- [ ] Kernel development guidelines
- [ ] Adding new layers
- [ ] Autograd implementation details

## Future Considerations

### External Integration

- [ ] Python bindings (pybind11 or Cython)
- [ ] ONNX runtime support
- [ ] TensorRT integration
- [ ] PyTorch tensor conversion
- [ ] JAX-like functional transformations

### Extended Functionality

- [ ] Sparse array support
- [ ] GPU-accelerated FFT
- [ ] GPU-accelerated BLAS/LAPACK
- [ ] Multi-GPU support (DataParallel, ModelParallel)
- [ ] Distributed training (NCCL)
- [ ] Mixed precision training (FP16, BF16)
- [ ] Quantization support
- [ ] Pruning and sparsity

### Compiler Support

- [ ] NVCC compatibility??
- [ ] MSVC CUDA support

### Performance Optimizations

- [ ] cuBLAS integration for matrix operations
- [ ] cuDNN integration for neural network layers
- [ ] TensorCore utilization
- [ ] Automatic kernel tuning
- [ ] JIT compilation for custom kernels (smh)
