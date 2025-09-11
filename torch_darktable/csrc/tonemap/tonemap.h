#pragma once

#include <torch/extension.h>

// Image metrics computation
torch::Tensor compute_image_bounds(torch::Tensor const& image, int stride = 8);
torch::Tensor compute_image_metrics(std::vector<torch::Tensor> const& images, int stride = 8, float min_gray = 1e-4f);

// Tone mapping
torch::Tensor reinhard_tonemap(
    torch::Tensor image,
    torch::Tensor metrics,
    float gamma = 1.0f,
    float intensity = 1.0f,
    float light_adapt = 0.8f
);

torch::Tensor aces_tonemap(torch::Tensor const& image, float gamma = 2.2f);
