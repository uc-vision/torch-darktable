#pragma once

#include <torch/extension.h>

// Tonemap parameters struct
struct TonemapParams {
    float gamma = 1.0f;
    float intensity = 0.0f;
    float light_adapt = 0.8f;
     
    TonemapParams() = default;
    TonemapParams(float gamma, float intensity, float light_adapt) 
        : gamma(gamma), intensity(intensity), light_adapt(light_adapt){}
};

// Image metrics computation
torch::Tensor compute_image_bounds(const std::vector<torch::Tensor>& images, int stride = 8);
torch::Tensor compute_image_metrics(const std::vector<torch::Tensor>& images, int stride = 8, float min_gray = 1e-4f);

torch::Tensor reinhard_tonemap(
    const torch::Tensor& image, const torch::Tensor& metrics,
    const TonemapParams& params
);

torch::Tensor aces_tonemap(
    const torch::Tensor& image, const torch::Tensor& metrics,
    const TonemapParams& params
);

torch::Tensor linear_tonemap(
    const torch::Tensor& image, const torch::Tensor& metrics,
    const TonemapParams& params
);


