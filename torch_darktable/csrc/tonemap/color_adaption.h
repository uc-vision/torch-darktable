 #pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include "../device_math.h"
#include "tonemap.h"

struct ColorTransform {

  float exposure;

  // statistics
  float3 adapt_mean; // target mean
  float  map_key;

  // scaling of the input intensity to 0..1
  float  bounds_min;
  float  range;
};


// Host function to compute and upload transform from metrics
void metrics_to_transform(
    ColorTransform& transform_out,
    const torch::Tensor& metrics,
    const TonemapParams& params,
    cudaStream_t stream);

// Image metrics
torch::Tensor compute_image_bounds(
    const std::vector<torch::Tensor>& images,
    int stride);

torch::Tensor compute_image_metrics(
    const std::vector<torch::Tensor>& images,
    int stride,
    float min_gray,
    bool rescale);