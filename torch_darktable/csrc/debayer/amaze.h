#pragma once

#include <torch/torch.h>
#include "../device_math.h"
#include "demosaic.h"
#include "bayer_device.h"

// AMaZE algorithm constants
__device__ constexpr float eps = 1e-5f;
__device__ constexpr float epssq = 1e-10f;
__device__ constexpr float arthresh = 0.75f;
__device__ constexpr float nyqthresh = 0.5f;

__device__ constexpr float gaussodd[4] = {
    0.14659727707323927f, 0.103592713382435f, 0.0732036125103057f, 0.0365543548389495f
};

__device__ constexpr float gaussgrad[6] = {
    nyqthresh * 0.07384411893421103f, nyqthresh * 0.06207511968171489f,
    nyqthresh * 0.0521818194747806f,  nyqthresh * 0.03687419286733595f,
    nyqthresh * 0.03099732204057846f, nyqthresh * 0.018413194161458882f
};

__device__ constexpr float gausseven[2] = { 0.13719494435797422f, 0.05640252782101291f };
__device__ constexpr float gquinc[4] = { 0.169917f, 0.108947f, 0.069855f, 0.0287182f };

// Clamping functions (replacing macros)
__device__ __forceinline__ float lim(float x, float min_val, float max_val) {
    return fmaxf(min_val, fminf(x, max_val));
}

__device__ __forceinline__ float ulim(float x, float y, float z) {
    return (y < z) ? lim(x, y, z) : lim(x, z, y);
}

// Direction-based indexing helpers
__device__ __forceinline__ int north(int idx, int width) { return idx - width; }
__device__ __forceinline__ int south(int idx, int width) { return idx + width; }
__device__ __forceinline__ int west(int idx) { return idx - 1; }
__device__ __forceinline__ int east(int idx) { return idx + 1; }
__device__ __forceinline__ int north2(int idx, int width) { return idx - 2 * width; }
__device__ __forceinline__ int south2(int idx, int width) { return idx + 2 * width; }
__device__ __forceinline__ int west2(int idx) { return idx - 2; }
__device__ __forceinline__ int east2(int idx) { return idx + 2; }

// Extract 4-directional values (using float4: x=north, y=south, z=west, w=east)
__device__ __forceinline__ float4 get_neighbors(const float* data, int idx, int width) {
    return make_float4(data[north(idx, width)], data[south(idx, width)], data[west(idx)], data[east(idx)]);
}

__device__ __forceinline__ float4 get_neighbors2(const float* data, int idx, int width) {
    return make_float4(data[north2(idx, width)], data[south2(idx, width)], data[west2(idx)], data[east2(idx)]);
}

// Helper for gaussian convolution on quincunx pattern
__device__ __forceinline__ float gaussian_quincunx_conv(
    const float* data, int idx, int width, const float* kernel
) {
    return kernel[0] * data[idx]
         + kernel[1] * (data[idx - width - 1] + data[idx - width + 1] + 
                        data[idx + width - 1] + data[idx + width + 1])
         + kernel[2] * (data[idx - 2*width] + data[idx - 2] + 
                        data[idx + 2] + data[idx + 2*width])
         + kernel[3] * (data[idx - 2*width - 2] + data[idx - 2*width + 2] + 
                        data[idx + 2*width - 2] + data[idx + 2*width + 2]);
}

// Helper for gaussian convolution on regular grid
__device__ __forceinline__ float gaussian_grid_conv(
    const float* data, int idx, int width, const float* kernel
) {
    return kernel[0] * data[idx]
         + kernel[1] * (data[idx - width] + data[idx + 1] + data[idx - 1] + data[idx + width])
         + kernel[2] * (data[idx - width - 1] + data[idx - width + 1] + 
                        data[idx + width - 1] + data[idx + width + 1])
         + kernel[3] * (data[idx - 2*width] + data[idx - 2] + data[idx + 2] + data[idx + 2*width])
         + kernel[4] * (data[idx - 2*width - 1] + data[idx - 2*width + 1] + 
                        data[idx - width - 2] + data[idx - width + 2] + 
                        data[idx + width - 2] + data[idx + width + 2] + 
                        data[idx + 2*width - 1] + data[idx + 2*width + 1])
         + kernel[5] * (data[idx - 2*width - 2] + data[idx - 2*width + 2] + 
                        data[idx + 2*width - 2] + data[idx + 2*width + 2]);
}

// Helper to compute variance over a 4-element window
__device__ __forceinline__ float compute_variance_4(const float* data, int idx, int offset1, int offset2, int offset3) {
    const float sum = data[idx] + data[idx + offset1] + data[idx + offset2] + data[idx + offset3];
    const float mean = sum * 0.25f;  // sum / 4
    return sqrf(data[idx] - mean) + sqrf(data[idx + offset1] - mean) + 
           sqrf(data[idx + offset2] - mean) + sqrf(data[idx + offset3] - mean);
}

// Compute RGB from input value and color difference based on pixel type
__device__ __forceinline__ float3 compute_rgb_from_pixel_type(int pixel_type, float input_val, float color_diff) {
    if (pixel_type == 1 || pixel_type == 3) { // Green pixels
        return make_float3(input_val - color_diff, input_val, input_val - color_diff);
    } else if (pixel_type == 0) { // Red pixel
        const float g = input_val + color_diff;
        return make_float3(input_val, g, g - color_diff);
    } else { // Blue pixel (type 2)
        const float g = input_val + color_diff;
        return make_float3(g - color_diff, g, input_val);
    }
}
