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
__device__ __forceinline__ void reduce_add(float value, float* target) {
    auto group = cg::coalesced_threads();
    float result = cg::reduce(group, value, cg::plus<float>{});
    if (group.thread_rank() == 0) {
        atomicAdd(target, result);
    }
}

__device__ __forceinline__ void reduce_min(float value, float* target) {
    auto group = cg::coalesced_threads();
    float result = cg::reduce(group, value, cg::less<float>{});
    if (group.thread_rank() == 0) {
        atomicMinFloat(target, result);
    }
}

__device__ __forceinline__ void reduce_max(float value, float* target) {
    auto group = cg::coalesced_threads();
    float result = cg::reduce(group, value, cg::greater<float>{});
    if (group.thread_rank() == 0) {
        atomicMaxFloat(target, result);
    }
}

// Cooperative groups reduce + atomic write patterns (float3)
__device__ __forceinline__  void reduce_add(float3 value, float3* target) {
    auto group = cg::coalesced_threads();
    float3 result = cg::reduce(group, value, cg::plus<float3>{});
    if (group.thread_rank() == 0) {
        atomicAdd(&target->x, result.x);
        atomicAdd(&target->y, result.y);
        atomicAdd(&target->z, result.z);
    }
}

__device__ __forceinline__ void reduce_min(float3 value, float3* target) {
    reduce_min(value.x, &target->x);
    reduce_min(value.y, &target->y);
    reduce_min(value.z, &target->z);
}

__device__ __forceinline__ void reduce_max(float3 value, float3* target) {
    reduce_max(value.x, &target->x);
    reduce_max(value.y, &target->y);
    reduce_max(value.z, &target->z);
}


// Compare and swap inline function for sorting network
__device__ __forceinline__ void cas(float& a, float& b) {
  float x = a;
  int c = a > b;
  a = c ? b : a;
  b = c ? x : b;
}

// 3x3 sorting network - returns median value (s4)
__device__ __forceinline__ float sort3x3_median(float s0, float s1, float s2, 
                                                 float s3, float s4, float s5, 
                                                 float s6, float s7, float s8) {
  cas(s1, s2);
  cas(s4, s5);
  cas(s7, s8);
  cas(s0, s1);
  cas(s3, s4);
  cas(s6, s7);
  cas(s1, s2);
  cas(s4, s5);
  cas(s7, s8);
  cas(s0, s3);
  cas(s5, s8);
  cas(s4, s7);
  cas(s3, s6);
  cas(s1, s4);
  cas(s2, s5);
  cas(s4, s7);
  cas(s4, s2);
  cas(s6, s4);
  cas(s4, s2);
  return s4;
}

// Template specializations for direct component access - no branches
template<int ch> __device__ __forceinline__ float get(const float3& v);
template<> __device__ __forceinline__ float get<0>(const float3& v) { return v.x; }
template<> __device__ __forceinline__ float get<1>(const float3& v) { return v.y; }
template<> __device__ __forceinline__ float get<2>(const float3& v) { return v.z; }

// Helper to compute channel difference for a single pixel
template<int ch1, int ch2>
__device__ __forceinline__ float diff(const float3& p) {
  return get<ch1>(p) - get<ch2>(p);
}

// Median of channel differences across 3x3 neighborhood (e.g., R-G, B-G)
template<int ch1, int ch2>
__device__ __forceinline__ float diff_median3x3(const float3* buf, int w) {
  return sort3x3_median(
    diff<ch1, ch2>(buf[-w-1]),
    diff<ch1, ch2>(buf[-w  ]),
    diff<ch1, ch2>(buf[-w+1]),
    diff<ch1, ch2>(buf[-1  ]),
    diff<ch1, ch2>(buf[ 0  ]),
    diff<ch1, ch2>(buf[ 1  ]),
    diff<ch1, ch2>(buf[ w-1]),
    diff<ch1, ch2>(buf[ w  ]),
    diff<ch1, ch2>(buf[ w+1])
  );
}

// Block-level reduction for computing means within a single block
__device__ __forceinline__ float block_reduce_mean(float value, int count) {
    __shared__ float result;
    auto group = cg::coalesced_threads();
    float sum = cg::reduce(group, value, cg::plus<float>());
    if (group.thread_rank() == 0) {
        result = sum / count;
    }
    __syncthreads();
    return result;
}


// Exact median computation within a warp using bitonic sort
__device__ __forceinline__ float warp_median(float my_val) {
    // Exact median using bitonic sort within warp (32 values)
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    int tid = warp.thread_rank();
    
    // Bitonic sort: 5 stages for 32 elements
    for (int stage = 0; stage < 5; stage++) {
        for (int step = stage; step >= 0; step--) {
            int partner = tid ^ (1 << step);
            float partner_val = warp.shfl(my_val, partner);
            
            bool ascending = ((tid >> (stage + 1)) & 1) == 0;
            bool should_swap = (my_val > partner_val) == ascending;
            
            if (should_swap && partner > tid) {
                my_val = partner_val;
            }
        }
    }
    
    // After sorting, thread 15 and 16 have the median values
    float median15 = warp.shfl(my_val, 15);
    float median16 = warp.shfl(my_val, 16);
    
    return (median15 + median16) / 2.0f;  // Exact median of 32 values
}
