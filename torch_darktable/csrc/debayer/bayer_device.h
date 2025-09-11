#pragma once

// Forward declaration of shared function from ppg_kernels.cu
__global__ void border_interpolate_kernel(float* input, float3* output, 
    int width, int height, BayerPattern pattern, int border);


// Inline function instead of macro for better type safety and debugging
__device__ __forceinline__ int fc(int row, int col, BayerPattern pattern) {
  return (static_cast<unsigned int>(pattern) >> ((((row) << 1 & 14) + ((col) & 1)) << 1)) & 3;
}


__device__ __forceinline__ int2 offset2x2(int C) {
  return make_int2(C % 2, C / 2);
}



__device__ constexpr int order_rggb[4] = {0, 1, 1, 2};
__device__ constexpr int order_bggr[4] = {2, 3, 1, 0};
__device__ constexpr int order_grbg[4] = {1, 0, 2, 3};
__device__ constexpr int order_gbrg[4] = {3, 2, 0, 1};

template<BayerPattern pattern>
__device__ __forceinline__ constexpr int get_pixel_type(int C) {
  switch (pattern) {
    case BayerPattern::RGGB: { return order_rggb[C]; }
    case BayerPattern::BGGR: { return order_bggr[C]; }
    case BayerPattern::GRBG: { return order_grbg[C]; }
    case BayerPattern::GBRG: { return order_gbrg[C]; }
  }
  return 0; // Should never reach here
}