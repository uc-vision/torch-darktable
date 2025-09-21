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


__device__ constexpr int order_rggb[4] = {0, 1, 2, 3};  // R, G1, G2, B
__device__ constexpr int order_bggr[4] = {3, 1, 2, 0};  // B, G1, G2, R  
__device__ constexpr int order_grbg[4] = {1, 0, 3, 2};  // G1, R, B, G2
__device__ constexpr int order_gbrg[4] = {1, 3, 0, 2};  // G1, B, R, G2

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

// Extract RGB from 2x2 Bayer patch - RGGB example: p00=R, p01=G, p10=G, p11=B
__device__ __forceinline__ float3 bayer_2x2_to_rgb(float p00, float p01, float p10, float p11, BayerPattern pattern) {
    switch (pattern) {
        case BayerPattern::RGGB: return make_float3(p00, (p01 + p10) * 0.5f, p11);
        case BayerPattern::BGGR: return make_float3(p11, (p01 + p10) * 0.5f, p00);
        case BayerPattern::GRBG: return make_float3(p01, (p00 + p11) * 0.5f, p10);
        case BayerPattern::GBRG: return make_float3(p10, (p00 + p11) * 0.5f, p01);
    }
    return make_float3(0.0f, 0.0f, 0.0f);
}