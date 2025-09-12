#pragma once

#include <torch/torch.h>
#include <vector>
#include "debayer/demosaic.h"

// Apply white balance gains to a Bayer image
torch::Tensor apply_white_balance(
    const torch::Tensor& bayer_image,
    const torch::Tensor& gains,
    BayerPattern pattern
);

// Estimate white balance from multiple Bayer images
torch::Tensor estimate_white_balance(
    const std::vector<torch::Tensor>& bayer_images,
    BayerPattern pattern,
    float quantile = 0.95f,
    int stride = 8
);
