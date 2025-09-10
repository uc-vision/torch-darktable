#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <torch/extension.h>
#include <ATen/ATen.h>

// Common 2D launch configuration helpers
static inline int div_up(int x, int y) { return (x + y - 1) / y; }
static constexpr int image_block_size = 16;
static inline dim3 grid2d(int nx, int ny) {
    return dim3(div_up(nx, image_block_size), div_up(ny, image_block_size));
}
static constexpr dim3 block_size_2d = dim3(image_block_size, image_block_size);

// CUDA error checking macros
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
    } \
} while(0)

#define CUDA_CHECK_KERNEL() do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA kernel error: ") + cudaGetErrorString(err)); \
    } \
} while(0)

struct CudaTimer {
    std::vector<cudaEvent_t> events;
    std::vector<std::string> names;
    cudaStream_t stream;
    
    CudaTimer(cudaStream_t s) : stream(s) {}
    
    ~CudaTimer() {
        for(auto& event : events) cudaEventDestroy(event);
    }
    
    void record(const std::string& name) { 
        names.push_back(name);
        record_event();
    }
    
    void record_event() {
        cudaEvent_t event;
        cudaEventCreate(&event);
        events.push_back(event);
        cudaEventRecord(event, stream); 
    }

    
    void print_timings() {
        record_event();
        cudaStreamSynchronize(stream);
        float total = 0;
        
        for(int i = 0; i < events.size() - 1; i++) {
            float time;
            cudaEventElapsedTime(&time, events[i], events[i+1]);
            printf("%s=%.1f\n", names[i].c_str(), time);
            total += time;
        }
        printf("total=%.1f\n", total);
    }
};

// No-op timer (default)
struct NullTimer {
    NullTimer(cudaStream_t) {}
    void record(const std::string&) {}
    void end() {}
    void print_timings() {}
};


// Type definitions for fp16/fp32
using half_t = at::Half;
using float_t = float;

// Storage type trait - can be switched between fp32 and fp16
template<bool use_fp16>
struct storage_type {
    using type = float;
    static constexpr torch::ScalarType dtype = torch::kFloat32;
};

template<>
struct storage_type<true> {
    using type = half_t;
    static constexpr torch::ScalarType dtype = torch::kFloat16;
};

// Clean conversion wrappers
__device__ inline float_t to_float(half_t h) { return static_cast<float>(h); }
__device__ inline half_t to_half(float_t f) { return half_t(f); }

// Generic conversion wrappers
template<typename T>
__device__ inline float_t to_float_t(T val) { return static_cast<float_t>(val); }

template<>
__device__ inline float_t to_float_t<half_t>(half_t val) { return to_float(val); }

template<typename T>
__device__ inline T to_storage_type(float_t val) { return static_cast<T>(val); }

template<>
__device__ inline half_t to_storage_type<half_t>(float_t val) { return to_half(val); }



// Image access helpers
__device__ inline float read_imagef(float* data, int2 coord, int2 size) {
    return data[coord.y * size.x + coord.x];
}

__device__ inline void write_imagef(float* output, int2 coord, int2 size, float value) {
    output[coord.y * size.x + coord.x] = value;
}

__device__ inline float_t read_imagef_half(half_t* data, int2 coord, int2 size) {
    return to_float(data[coord.y * size.x + coord.x]);
}

__device__ inline void write_imagef_half(half_t* output, int2 coord, int2 size, float_t value) {
    output[coord.y * size.x + coord.x] = to_half(value);
}

// Generic image access helpers
template<typename T>
__device__ inline float_t read_imagef_generic(T* data, int2 coord, int2 size) {
    return to_float_t(data[coord.y * size.x + coord.x]);
}

template<>
__device__ inline float_t read_imagef_generic<float>(float* data, int2 coord, int2 size) {
    return data[coord.y * size.x + coord.x];
}

template<typename T>
__device__ inline void write_imagef_generic(T* output, int2 coord, int2 size, float_t value) {
    output[coord.y * size.x + coord.x] = to_storage_type<T>(value);
}

template<>
__device__ inline void write_imagef_generic<float>(float* output, int2 coord, int2 size, float_t value) {
    output[coord.y * size.x + coord.x] = value;
}
