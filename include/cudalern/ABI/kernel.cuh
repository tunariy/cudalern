#pragma once

#include <device_types.h>

#ifndef CUDALERN_KERNEL_CUH
    #define CUDALERN_KERNEL_CUH

    #include "cudalern/Core/concepts.hpp"

    #ifdef __CUDACC__
        #include <cuda_runtime.h>
        #include <device_launch_parameters.h>
    #else
        #define __syncthreads()
        #define __shfl_down_sync(mask, var, delta, width) var
        #define atomicMin(addr, val) (*(addr) = (*(addr) < (val) ? *(addr) : (val)))
        #define atomicAdd(addr, val) (*(addr) += (val))
    #endif

    #include <climits>
    #include <cstdint>
    #include <functional>
    #include <math.h>
    #include <numbers>

namespace cudalern {
namespace kernel {
    #if 1
    /**
     * @brief Empty kernel call to speed up the initial kernel call
     */
    __device__ inline void emptyCall() noexcept {};

    /**
     * @brief Applies a function to each element of an array.
     * @tparam T Element type (must satisfy CudaCompatible).
     * @tparam Func Callable type (must be device‑compatible).
     * @tparam Args Additional arguments passed to the function.
     * @param data Pointer to device array.
     * @param n Number of elements.
     * @param func Callable object.
     * @param args Extra arguments.
     */
    template <class T, typename Func, typename... Args>
        requires(CudaCompatible<T>)
    __global__ void foreach (T* data, size_t n, Func func, Args... args);

    /**
     * @brief Fills an array with a constant value.
     * @tparam T Element type.
     * @param data Pointer to device array.
     * @param n Number of elements.
     * @param value Constant value.
     */
    template <typename T>
    __global__ void fill(T* data, size_t n, T value);

    /**
     * @brief Fills a 1D array with values start, start+step, start+2*step, ...
     * @tparam T Element type.
     * @param data Pointer to device array.
     * @param n Number of elements.
     * @param start Starting value.
     * @param step Increment step.
     */
    template <typename T>
    __global__ void arange(T* data, size_t n, T start, T step);

    /**
     * @brief Creates an identity matrix (eye) in row‑major order.
     * @tparam T Element type.
     * @param data Pointer to device array (size rows*cols).
     * @param rows Number of rows.
     * @param cols Number of columns (must equal rows for identity).
     */
    template <typename T>
    __global__ void eye(T* data, size_t rows, size_t cols);

    /**
     * @brief Fills array with random numbers from a uniform distribution.
     * @tparam T Element type.
     * @param data Pointer to device array.
     * @param n Number of elements.
     * @param low Lower bound (inclusive).
     * @param high Upper bound (exclusive).
     * @param seed Base seed for PRNG.
     */
    template <typename T>
    __global__ void random_uniform(T* data, size_t n, T low, T high, unsigned int seed);

    /**
     * @brief Fills array with random numbers from a normal distribution.
     * @tparam T Element type.
     * @param data Pointer to device array.
     * @param n Number of elements.
     * @param mean Mean of the distribution.
     * @param std Standard deviation.
     * @param seed Base seed for PRNG.
     */
    template <typename T>
    __global__ void random_normal(T* data, size_t n, T mean, T std, unsigned int seed);

    /**
     * @brief Element‑wise addition of two arrays: c[i] = a[i] + b[i].
     * @tparam T Element type.
     * @param c Output array.
     * @param a First input array.
     * @param b Second input array.
     * @param size Number of elements.
     */
    template <class T>
        requires(CudaCompatible<T>)
    __global__ void add(T* c, const T* a, const T* b, uint32_t size);

    /**
     * @brief Element‑wise subtraction: c[i] = a[i] - b[i].
     * @tparam T Element type.
     * @param c Output array.
     * @param a First input array.
     * @param b Second input array.
     * @param size Number of elements.
     */
    template <class T>
        requires(CudaCompatible<T>)
    __global__ void sub(T* c, const T* a, const T* b, uint32_t size);

    /**
     * @brief Element‑wise multiplication: c[i] = a[i] * b[i].
     * @tparam T Element type.
     * @param c Output array.
     * @param a First input array.
     * @param b Second input array.
     * @param size Number of elements.
     */
    template <class T>
        requires(CudaCompatible<T>)
    __global__ void mul(T* c, const T* a, const T* b, uint32_t size);

    /**
     * @brief Element‑wise division: c[i] = a[i] / b[i].
     * @tparam T Element type.
     * @param c Output array.
     * @param a First input array.
     * @param b Second input array.
     * @param size Number of elements.
     */
    template <class T>
        requires(CudaCompatible<T>)
    __global__ void div(T* c, const T* a, const T* b, uint32_t size);

    /**
     * @brief Element‑wise addition with scalar: c[i] = a[i] + b.
     * @tparam T Element type.
     * @param c Output array.
     * @param a Input array.
     * @param b Scalar value.
     * @param size Number of elements.
     */
    template <class T>
        requires(CudaCompatible<T>)
    __global__ void add(T* c, const T* a, T b, uint32_t size);

    /**
     * @brief Element‑wise subtraction with scalar: c[i] = a[i] - b.
     * @tparam T Element type.
     * @param c Output array.
     * @param a Input array.
     * @param b Scalar value.
     * @param size Number of elements.
     */
    template <class T>
        requires(CudaCompatible<T>)
    __global__ void sub(T* c, const T* a, T b, uint32_t size);

    /**
     * @brief Element‑wise multiplication with scalar: c[i] = a[i] * b.
     * @tparam T Element type.
     * @param c Output array.
     * @param a Input array.
     * @param b Scalar value.
     * @param size Number of elements.
     */
    template <class T>
        requires(CudaCompatible<T>)
    __global__ void mul(T* c, const T* a, T b, uint32_t size);

    /**
     * @brief Element‑wise division with scalar: c[i] = a[i] / b.
     * @tparam T Element type.
     * @param c Output array.
     * @param a Input array.
     * @param b Scalar value.
     * @param size Number of elements.
     */
    template <class T>
        requires(CudaCompatible<T>)
    __global__ void div(T* c, const T* a, T b, uint32_t size);

    /**
     * @brief Element‑wise power: c[i] = pow(a[i], exponent).
     * @tparam T Element type.
     * @param c Output array.
     * @param a Input array.
     * @param exponent Exponent.
     * @param size Number of elements.
     */
    template <typename T>
    __global__ void pow_kernel(T* c, const T* a, T exponent, uint32_t size);

    /**
     * @brief Element‑wise modulo (integer only): c[i] = a[i] % modulus.
     * @tparam T Element type (integral).
     * @param c Output array.
     * @param a Input array.
     * @param modulus Modulus value.
     * @param size Number of elements.
     */
    template <typename T>
    __global__ void mod_kernel(T* c, const T* a, T modulus, uint32_t size);

    /**
     * @brief Element‑wise equality: c[i] = (a[i] == b[i]) ? 1 : 0.
     * @tparam T Element type.
     * @param c Output array.
     * @param a First input array.
     * @param b Second input array.
     * @param size Number of elements.
     */
    template <typename T>
    __global__ void eq_kernel(T* c, const T* a, const T* b, uint32_t size);

    /**
     * @brief Element‑wise inequality: c[i] = (a[i] != b[i]) ? 1 : 0.
     * @tparam T Element type.
     * @param c Output array.
     * @param a First input array.
     * @param b Second input array.
     * @param size Number of elements.
     */
    template <typename T>
    __global__ void ne_kernel(T* c, const T* a, const T* b, uint32_t size);

    /**
     * @brief Element‑wise less‑than: c[i] = (a[i] < b[i]) ? 1 : 0.
     * @tparam T Element type.
     * @param c Output array.
     * @param a First input array.
     * @param b Second input array.
     * @param size Number of elements.
     */
    template <typename T>
    __global__ void lt_kernel(T* c, const T* a, const T* b, uint32_t size);

    /**
     * @brief Element‑wise greater‑than: c[i] = (a[i] > b[i]) ? 1 : 0.
     * @tparam T Element type.
     * @param c Output array.
     * @param a First input array.
     * @param b Second input array.
     * @param size Number of elements.
     */
    template <typename T>
    __global__ void gt_kernel(T* c, const T* a, const T* b, uint32_t size);

    /**
     * @brief Element‑wise less‑than‑or‑equal: c[i] = (a[i] <= b[i]) ? 1 : 0.
     * @tparam T Element type.
     * @param c Output array.
     * @param a First input array.
     * @param b Second input array.
     * @param size Number of elements.
     */
    template <typename T>
    __global__ void le_kernel(T* c, const T* a, const T* b, uint32_t size);

    /**
     * @brief Element‑wise greater‑than‑or‑equal: c[i] = (a[i] >= b[i]) ? 1 : 0.
     * @tparam T Element type.
     * @param c Output array.
     * @param a First input array.
     * @param b Second input array.
     * @param size Number of elements.
     */
    template <typename T>
    __global__ void ge_kernel(T* c, const T* a, const T* b, uint32_t size);

    /**
     * @brief Element‑wise negation: c[i] = -a[i].
     * @tparam T Element type.
     * @param c Output array.
     * @param a Input array.
     * @param size Number of elements.
     */
    template <typename T>
    __global__ void negate_kernel(T* c, const T* a, uint32_t size);

    /**
     * @brief Element‑wise absolute value: c[i] = abs(a[i]).
     * @tparam T Element type.
     * @param c Output array.
     * @param a Input array.
     * @param size Number of elements.
     */
    template <typename T>
    __global__ void abs_kernel(T* c, const T* a, uint32_t size);

    /**
     * @brief Element‑wise square root: c[i] = sqrt(a[i]).
     * @tparam T Element type.
     * @param c Output array.
     * @param a Input array.
     * @param size Number of elements.
     */
    template <typename T>
    __global__ void sqrt_kernel(T* c, const T* a, uint32_t size);

    /**
     * @brief Element‑wise exponential: c[i] = exp(a[i]).
     * @tparam T Element type.
     * @param c Output array.
     * @param a Input array.
     * @param size Number of elements.
     */
    template <typename T>
    __global__ void exp_kernel(T* c, const T* a, uint32_t size);

    /**
     * @brief Element‑wise natural logarithm: c[i] = log(a[i]).
     * @tparam T Element type.
     * @param c Output array.
     * @param a Input array.
     * @param size Number of elements.
     */
    template <typename T>
    __global__ void log_kernel(T* c, const T* a, uint32_t size);

    /**
     * @brief Matrix multiplication: C = A * B (row‑major, no optimizations).
     * @tparam T Element type.
     * @param A Matrix A (M x K).
     * @param B Matrix B (K x N).
     * @param C Output matrix (M x N).
     * @param M Rows of A and C.
     * @param N Columns of B and C.
     * @param K Inner dimension.
     */
    template <class T>
        requires(CudaCompatible<T>)
    __global__ void matrix_multiply(const T* A, const T* B, T* C, uint32_t M, uint32_t N,
                                    uint32_t K);

    /**
     * @brief Out‑of‑place matrix transpose.
     * @tparam T Element type.
     * @param dst Destination array (rows*cols).
     * @param src Source array (cols*rows).
     * @param rows Number of rows.
     * @param cols Number of columns.
     */
    template <typename T>
    __global__ void transpose(T* dst, const T* src, size_t rows, size_t cols);

    /**
     * @brief Sum reduction: computes sum of all elements.
     * @tparam T Element type.
     * @param data Input array.
     * @param output Output scalar (accumulated via atomicAdd).
     * @param n Number of elements.
     */
    template <typename T>
    __global__ void sum_reduce(const T* data, T* output, size_t n);

    /**
     * @brief Mean reduction: computes arithmetic mean.
     * @tparam T Element type.
     * @param data Input array.
     * @param output Output scalar (accumulated sum, then divided by n).
     * @param n Number of elements.
     */
    template <typename T>
    __global__ void mean_reduce(const T* data, T* output, size_t n);

    /**
     * @brief Maximum reduction: computes max element.
     * @tparam T Element type.
     * @param data Input array.
     * @param output Output scalar (max).
     * @param n Number of elements.
     */
    template <typename T>
    __global__ void max_reduce(const T* data, T* output, size_t n);

    /**
     * @brief Minimum reduction: computes min element.
     * @tparam T Element type.
     * @param data Input array.
     * @param output Output scalar (min).
     * @param n Number of elements.
     */
    template <typename T>
    __global__ void min_reduce(const T* data, T* output, size_t n);

    /**
     * @brief Argmax reduction: computes index of maximum element.
     * @tparam T Element type.
     * @param data Input array.
     * @param output Output index (atomicMin used to keep smallest index on ties).
     * @param n Number of elements.
     */
    template <typename T>
    __global__ void argmax_reduce(const T* data, int* output, size_t n);

    /**
     * @brief ReLU activation: c[i] = max(0, a[i]).
     * @tparam T Element type.
     * @param c Output array.
     * @param a Input array.
     * @param size Number of elements.
     */
    template <typename T>
    __global__ void relu(T* c, const T* a, uint32_t size);

    /**
     * @brief Leaky ReLU activation: c[i] = (a[i] > 0) ? a[i] : alpha * a[i].
     * @tparam T Element type.
     * @param c Output array.
     * @param a Input array.
     * @param alpha Slope for negative part.
     * @param size Number of elements.
     */
    template <typename T>
    __global__ void leaky_relu(T* c, const T* a, T alpha, uint32_t size);

    /**
     * @brief Sigmoid activation: c[i] = 1 / (1 + exp(-a[i])).
     * @tparam T Element type.
     * @param c Output array.
     * @param a Input array.
     * @param size Number of elements.
     */
    template <typename T>
    __global__ void sigmoid(T* c, const T* a, uint32_t size);

    /**
     * @brief Tanh activation: c[i] = tanh(a[i]).
     * @tparam T Element type.
     * @param c Output array.
     * @param a Input array.
     * @param size Number of elements.
     */
    template <typename T>
    __global__ void tanh_kernel(T* c, const T* a, uint32_t size);

    /**
     * @brief Softmax (1D): c[i] = exp(a[i]) / sum(exp(a)).
     * @tparam T Element type.
     * @param c Output array.
     * @param a Input array.
     * @param size Number of elements.
     * @note Uses shared memory for block‑level reduction, supports only one block.
     */
    template <typename T>
    __global__ void softmax(T* c, const T* a, uint32_t size);
    #endif

    // Device‑side implementations
    #ifdef __CUDACC__

    static __device__ unsigned int xorshift32(unsigned int& state) {
        unsigned int x = state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        state = x;
        return x;
    }

    static __device__ float uniform_rand(unsigned int& state) {
        return static_cast<float>(xorshift32(state)) / 4294967295.0f;
    }

    static __device__ float normal_rand(unsigned int& state) {
        float u1 = uniform_rand(state);
        float u2 = uniform_rand(state);
        float r = sqrtf(-2.0f * logf(u1 + 1e-20f));
        float theta = 2.0f * std::numbers::pi * u2;
        return r * cosf(theta);
    }

    template <class T, typename Func, typename... Args>
        requires(CudaCompatible<T>)
    __global__ void foreach (T* data, size_t n, Func func, Args... args) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            std::invoke(func, data[idx], args...);
        }
    }

    template <typename T>
    __global__ void fill(T* data, size_t n, T value) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) data[idx] = value;
    }

    template <typename T>
    __global__ void arange(T* data, size_t n, T start, T step) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) data[idx] = start + static_cast<T>(idx) * step;
    }

    template <typename T>
    __global__ void eye(T* data, size_t rows, size_t cols) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t total = rows * cols;
        if (idx < total) {
            size_t row = idx / cols;
            size_t col = idx % cols;
            data[idx] = (row == col) ? static_cast<T>(1) : static_cast<T>(0);
        }
    }

    template <typename T>
    __global__ void random_uniform(T* data, size_t n, T low, T high, unsigned int seed) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            unsigned int state = seed + idx * 2654435761u;
            float val = low + (high - low) * uniform_rand(state);
            data[idx] = static_cast<T>(val);
        }
    }

    template <typename T>
    __global__ void random_normal(T* data, size_t n, T mean, T std, unsigned int seed) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            unsigned int state = seed + idx * 2654435761u;
            float val = mean + std * normal_rand(state);
            data[idx] = static_cast<T>(val);
        }
    }

    template <class T>
        requires(CudaCompatible<T>)
    __global__ void add(T* c, const T* a, const T* b, uint32_t size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) c[i] = a[i] + b[i];
    }

    template <class T>
        requires(CudaCompatible<T>)
    __global__ void sub(T* c, const T* a, const T* b, uint32_t size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) c[i] = a[i] - b[i];
    }

    template <class T>
        requires(CudaCompatible<T>)
    __global__ void mul(T* c, const T* a, const T* b, uint32_t size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) c[i] = a[i] * b[i];
    }

    template <class T>
        requires(CudaCompatible<T>)
    __global__ void div(T* c, const T* a, const T* b, uint32_t size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) c[i] = a[i] / b[i];
    }

    template <class T>
        requires(CudaCompatible<T>)
    __global__ void add(T* c, const T* a, T b, uint32_t size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) c[i] = a[i] + b;
    }

    template <class T>
        requires(CudaCompatible<T>)
    __global__ void sub(T* c, const T* a, T b, uint32_t size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) c[i] = a[i] - b;
    }

    template <class T>
        requires(CudaCompatible<T>)
    __global__ void mul(T* c, const T* a, T b, uint32_t size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) c[i] = a[i] * b;
    }

    template <class T>
        requires(CudaCompatible<T>)
    __global__ void div(T* c, const T* a, T b, uint32_t size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) c[i] = a[i] / b;
    }

    template <typename T>
    __global__ void pow_kernel(T* c, const T* a, T exponent, uint32_t size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) c[i] = powf(a[i], exponent);
    }

    template <typename T>
    __global__ void mod_kernel(T* c, const T* a, T modulus, uint32_t size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) c[i] = a[i] % modulus;
    }

    template <typename T>
    __global__ void eq_kernel(T* c, const T* a, const T* b, uint32_t size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) c[i] = (a[i] == b[i]) ? static_cast<T>(1) : static_cast<T>(0);
    }

    template <typename T>
    __global__ void ne_kernel(T* c, const T* a, const T* b, uint32_t size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) c[i] = (a[i] != b[i]) ? static_cast<T>(1) : static_cast<T>(0);
    }

    template <typename T>
    __global__ void lt_kernel(T* c, const T* a, const T* b, uint32_t size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) c[i] = (a[i] < b[i]) ? static_cast<T>(1) : static_cast<T>(0);
    }

    template <typename T>
    __global__ void gt_kernel(T* c, const T* a, const T* b, uint32_t size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) c[i] = (a[i] > b[i]) ? static_cast<T>(1) : static_cast<T>(0);
    }

    template <typename T>
    __global__ void le_kernel(T* c, const T* a, const T* b, uint32_t size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) c[i] = (a[i] <= b[i]) ? static_cast<T>(1) : static_cast<T>(0);
    }

    template <typename T>
    __global__ void ge_kernel(T* c, const T* a, const T* b, uint32_t size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) c[i] = (a[i] >= b[i]) ? static_cast<T>(1) : static_cast<T>(0);
    }

    template <typename T>
    __global__ void negate_kernel(T* c, const T* a, uint32_t size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) c[i] = -a[i];
    }

    template <typename T>
    __global__ void abs_kernel(T* c, const T* a, uint32_t size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) c[i] = fabsf(a[i]);
    }

    template <typename T>
    __global__ void sqrt_kernel(T* c, const T* a, uint32_t size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) c[i] = sqrtf(a[i]);
    }

    template <typename T>
    __global__ void exp_kernel(T* c, const T* a, uint32_t size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) c[i] = expf(a[i]);
    }

    template <typename T>
    __global__ void log_kernel(T* c, const T* a, uint32_t size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) c[i] = logf(a[i]);
    }

    template <class T>
        requires(CudaCompatible<T>)
    __global__ void matrix_multiply(const T* A, const T* B, T* C, uint32_t M, uint32_t N,
                                    uint32_t K) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < M && col < N) {
            T sum = 0;
            for (int i = 0; i < K; ++i) {
                sum += A[row * K + i] * B[i * N + col];
            }
            C[row * N + col] = sum;
        }
    }

    template <typename T>
    __global__ void transpose(T* dst, const T* src, size_t rows, size_t cols) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < rows && col < cols) {
            dst[col * rows + row] = src[row * cols + col];
        }
    }

    template <typename T>
    __device__ T warpReduceSum(T val) {
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        return val;
    }

    template <typename T>
    __device__ T blockReduceSum(T val, T* shared) {
        int lane = threadIdx.x & 31;
        int wid = threadIdx.x >> 5;
        val = warpReduceSum(val);
        if (lane == 0) shared[wid] = val;
        __syncthreads();
        val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : (T)0;
        if (wid == 0) val = warpReduceSum(val);
        return val;
    }

    template <typename T>
    __global__ void sum_reduce(const T* data, T* output, size_t n) {
        extern __shared__ T shared[];
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        T val = (idx < n) ? data[idx] : (T)0;
        T result = blockReduceSum(val, shared);
        if (threadIdx.x == 0) atomicAdd(output, result);
    }

    template <typename T>
    __global__ void mean_reduce(const T* data, T* output, size_t n) {
        extern __shared__ T shared[];
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        T val = (idx < n) ? data[idx] : (T)0;
        T result = blockReduceSum(val, shared);
        if (threadIdx.x == 0) atomicAdd(output, result / (T)n);
    }

    template <typename T>
    __device__ T warpReduceMax(T val) {
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            val = max(val, __shfl_down_sync(0xffffffff, val, offset));
        return val;
    }

    template <typename T>
    __device__ T blockReduceMax(T val, T* shared) {
        int lane = threadIdx.x & 31;
        int wid = threadIdx.x >> 5;
        val = warpReduceMax(val);
        if (lane == 0) shared[wid] = val;
        __syncthreads();
        val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : (T)0;
        if (wid == 0) val = warpReduceMax(val);
        return val;
    }

    template <typename T>
    __global__ void max_reduce(const T* data, T* output, size_t n) {
        extern __shared__ T shared[];
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        T val = (idx < n) ? data[idx] : (T)-INFINITY;
        T result = blockReduceMax(val, shared);
        if (threadIdx.x == 0) atomicMax(output, result);
    }

    template <typename T>
    __device__ T warpReduceMin(T val) {
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            val = min(val, __shfl_down_sync(0xffffffff, val, offset));
        return val;
    }

    template <typename T>
    __device__ T blockReduceMin(T val, T* shared) {
        int lane = threadIdx.x & 31;
        int wid = threadIdx.x >> 5;
        val = warpReduceMin(val);
        if (lane == 0) shared[wid] = val;
        __syncthreads();
        val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : (T)0;
        if (wid == 0) val = warpReduceMin(val);
        return val;
    }

    template <typename T>
    __global__ void min_reduce(const T* data, T* output, size_t n) {
        extern __shared__ T shared[];
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        T val = (idx < n) ? data[idx] : (T)INFINITY;
        T result = blockReduceMin(val, shared);
        if (threadIdx.x == 0) atomicMin(output, result);
    }

    struct Pair {
        float val;
        int idx;
    };

    static __device__ Pair warpReduceMaxPair(Pair p) {
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            float other_val = __shfl_down_sync(0xffffffff, p.val, offset);
            int other_idx = __shfl_down_sync(0xffffffff, p.idx, offset);
            if (other_val > p.val || (other_val == p.val && other_idx < p.idx)) {
                p.val = other_val;
                p.idx = other_idx;
            }
        }
        return p;
    }

    static __device__ Pair blockReduceMaxPair(Pair p, Pair* shared) {
        int lane = threadIdx.x & 31;
        int wid = threadIdx.x >> 5;
        p = warpReduceMaxPair(p);
        if (lane == 0) shared[wid] = p;
        __syncthreads();
        p = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : Pair{-INFINITY, -1};
        if (wid == 0) p = warpReduceMaxPair(p);
        return p;
    }

    template <typename T>
    __global__ void argmax_reduce(const T* data, int* output, size_t n) {
        extern __shared__ Pair shared[];
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        Pair p{(idx < n) ? (float)data[idx] : -INFINITY, (int)idx};
        Pair result = blockReduceMaxPair(p, shared);
        if (threadIdx.x == 0) atomicMin(output, result.idx);
    }

    template <typename T>
    __global__ void relu(T* c, const T* a, uint32_t size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) c[i] = (a[i] > 0) ? a[i] : (T)0;
    }

    template <typename T>
    __global__ void leaky_relu(T* c, const T* a, T alpha, uint32_t size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) c[i] = (a[i] > 0) ? a[i] : alpha * a[i];
    }

    template <typename T>
    __global__ void sigmoid(T* c, const T* a, uint32_t size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) c[i] = (T)1 / ((T)1 + expf(-a[i]));
    }

    template <typename T>
    __global__ void tanh_kernel(T* c, const T* a, uint32_t size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) c[i] = tanhf(a[i]);
    }

    template <typename T>
    __global__ void softmax(T* c, const T* a, uint32_t size) {
        extern __shared__ T shared[];
        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + tid;

        T max_val = (idx < size) ? a[idx] : (T)-INFINITY;
        max_val = blockReduceMax(max_val, shared);

        if (threadIdx.x == 0) shared[0] = max_val;
        __syncthreads();
        max_val = shared[0];

        T exp_val = (idx < size) ? expf(a[idx] - max_val) : (T)0;
        T sum = blockReduceSum(exp_val, shared);

        if (threadIdx.x == 0) shared[0] = sum;
        __syncthreads();
        sum = shared[0];

        if (idx < size) c[idx] = exp_val / sum;
    }

    #endif  // __CUDACC__

}  // namespace kernel
}  // namespace cudalern

#endif  // CUDALERN_KERNEL_CUH