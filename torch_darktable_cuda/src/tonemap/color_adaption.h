 #pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include "../device_math.h"
#include "tonemap.h"

// Common helper functions for tonemap kernels

// Extract adaptation parameters from 7-element metrics tensor
struct AdaptationParams {
    float map_key;
    float3 global_mean;
    float exposure;
};

__device__ __forceinline__ float compute_map_key(float log_mean) {
    // Direct mapping from log_mean to map_key
    // log_mean range: [-9.21, 0] -> map_key range: [0.3, 1.0]
    
    const float log_range = 9.21034f;  // -log(1e-4) = 9.21034
    const float gamma = 1.4f;
    
    // Normalize log_mean to [0, 1], apply power curve, map to [0.3, 1.0]
    float normalized = fmaxf(0.0f, fminf(1.0f, (-log_mean) / log_range));
    float powered = powf(normalized, gamma);
    return 0.3f + 0.7f * powered;
}

__device__ __forceinline__ float3 extract_global_mean(const float* metrics) {
    return make_float3(metrics[2], metrics[3], metrics[4]); // rgb_mean_r, rgb_mean_g, rgb_mean_b
}

__device__ __forceinline__ AdaptationParams extract_adaptation_params(
    const float* metrics, 
    float intensity) {
    
    AdaptationParams params;
    params.map_key = compute_map_key(metrics[0]); // log_mean only
    params.global_mean = extract_global_mean(metrics);
    params.exposure = expf(intensity);
    return params;
}

__device__ __forceinline__ float3 compute_local_global_blend(
    float light_adapt,
    const float3& global_mean,
    const float3& local_color) {
    
    return lerp(light_adapt, global_mean, local_color);
}

__device__ __forceinline__ float3 apply_adaptation(
    const float3& adapt_mean,
    float exposure,
    float map_key) {
    
    return pow(adapt_mean / exposure, map_key);
}

// Single function that does all adaptation computation
__device__ __forceinline__ float3 compute_adaptation(
    const float* metrics,
    const float3& pixel_color,
    float light_adapt,
    float intensity) {
    
    // Extract adaptation parameters from metrics
    AdaptationParams adapt_params = extract_adaptation_params(metrics, intensity);
    
    // Per-pixel local vs global adaptation (like reference)
    float3 adapt_mean = compute_local_global_blend(light_adapt, adapt_params.global_mean, pixel_color);
    
    // Apply adaptation
    return apply_adaptation(adapt_mean, adapt_params.exposure, adapt_params.map_key);
}

// Image metrics
torch::Tensor compute_image_bounds(
    const std::vector<torch::Tensor>& images,
    int stride);

torch::Tensor compute_image_metrics(
    const std::vector<torch::Tensor>& images,
    int stride,
    float min_gray,
    bool rescale);