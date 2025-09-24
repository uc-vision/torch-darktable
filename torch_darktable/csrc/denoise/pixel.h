#pragma once

#include <cuda_runtime.h>

// Get channel count from type
template<typename T> constexpr int channels() = delete;
template<> constexpr int channels<float>() { return 1; }
template<> constexpr int channels<float3>() { return 3; }

// Atomic add operations
__device__ __forceinline__ void atomic_add(float* addr, float value) {
    atomicAdd(addr, value);
}

__device__ __forceinline__ void atomic_add(float3* addr, float3 value) {
    atomicAdd(&addr->x, value.x);
    atomicAdd(&addr->y, value.y);
    atomicAdd(&addr->z, value.z);
}

// Channel access
__device__ __forceinline__ float get_channel(float pixel, int i) {
    return pixel;
}

__device__ __forceinline__ float get_channel(const float3& pixel, int i) {
    return (&pixel.x)[i];
}

__device__ __forceinline__ void set_channel(float& pixel, int i, float value) {
    pixel = value;
}

__device__ __forceinline__ void set_channel(float3& pixel, int i, float value) {
    (&pixel.x)[i] = value;
}
