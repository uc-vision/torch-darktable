#pragma once

#include <torch/torch.h>

// PyTorch tensor interface functions
torch::Tensor compute_luminance(const torch::Tensor& rgb);
torch::Tensor modify_luminance(const torch::Tensor& rgb, const torch::Tensor& new_luminance);

// Individual conversion functions
torch::Tensor rgb_to_xyz(const torch::Tensor& rgb);
torch::Tensor xyz_to_lab(const torch::Tensor& xyz);
torch::Tensor lab_to_xyz(const torch::Tensor& lab);
torch::Tensor xyz_to_rgb(const torch::Tensor& xyz);

// Direct conversions
torch::Tensor rgb_to_lab(const torch::Tensor& rgb);
torch::Tensor lab_to_rgb(const torch::Tensor& lab);
