#pragma once

#include <torch/extension.h>

// Image metrics computation
torch::Tensor compute_image_bounds(const torch::Tensor& image, int stride = 8);
torch::Tensor compute_image_metrics(const std::vector<torch::Tensor>& images, int stride = 8, float min_gray = 1e-4f);

// Tone mapping
torch::Tensor reinhard_tonemap(
    const torch::Tensor& image,
    const torch::Tensor& metrics,
    float gamma = 1.0f,
    float intensity = 1.0f,
    float light_adapt = 0.8f
);

torch::Tensor aces_tonemap(const torch::Tensor& image, float gamma = 2.2f);
