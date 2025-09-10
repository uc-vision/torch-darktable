#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "device_math.h"

namespace cg = cooperative_groups;

// Atomic float operations
__device__ __forceinline__ float atomicMinFloat(float* addr, float value) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
            __float_as_int(fminf(value, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
            __float_as_int(fmaxf(value, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

// Cooperative groups reduce + atomic write patterns (float)
__device__ void reduce_add(float value, float* target) {
    auto group = cg::coalesced_threads();
    float result = cg::reduce(group, value, cg::plus<float>{});
    if (group.thread_rank() == 0) {
        atomicAdd(target, result);
    }
}

__device__ void reduce_min(float value, float* target) {
    auto group = cg::coalesced_threads();
    float result = cg::reduce(group, value, cg::less<float>{});
    if (group.thread_rank() == 0) {
        atomicMinFloat(target, result);
    }
}

__device__ void reduce_max(float value, float* target) {
    auto group = cg::coalesced_threads();
    float result = cg::reduce(group, value, cg::greater<float>{});
    if (group.thread_rank() == 0) {
        atomicMaxFloat(target, result);
    }
}

// Cooperative groups reduce + atomic write patterns (float3)
__device__ void reduce_add(float3 value, float3* target) {
    auto group = cg::coalesced_threads();
    float3 result = cg::reduce(group, value, cg::plus<float3>{});
    if (group.thread_rank() == 0) {
        atomicAdd(&target->x, result.x);
        atomicAdd(&target->y, result.y);
        atomicAdd(&target->z, result.z);
    }
}

__device__ void reduce_min(float3 value, float3* target) {
    reduce_min(value.x, &target->x);
    reduce_min(value.y, &target->y);
    reduce_min(value.z, &target->z);
}

__device__ void reduce_max(float3 value, float3* target) {
    reduce_max(value.x, &target->x);
    reduce_max(value.y, &target->y);
    reduce_max(value.z, &target->z);
}
