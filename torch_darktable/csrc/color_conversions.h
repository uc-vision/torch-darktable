#pragma once

#include <torch/torch.h>

// PyTorch tensor interface functions
torch::Tensor compute_luminance(const torch::Tensor& rgb);
torch::Tensor modify_luminance(const torch::Tensor& rgb, const torch::Tensor& new_luminance);
torch::Tensor compute_log_luminance(const torch::Tensor& rgb, float eps);
torch::Tensor modify_log_luminance(const torch::Tensor& rgb, const torch::Tensor& log_luminance, float eps);
torch::Tensor modify_saturation(const torch::Tensor& rgb, float saturation);
torch::Tensor modify_saturation_mult_add(const torch::Tensor& rgb, float saturation_mult, float saturation_add);

// Individual conversion functions
torch::Tensor rgb_to_xyz(const torch::Tensor& rgb);
torch::Tensor xyz_to_lab(const torch::Tensor& xyz);
torch::Tensor lab_to_xyz(const torch::Tensor& lab);
torch::Tensor xyz_to_rgb(const torch::Tensor& xyz);

// Direct conversions
torch::Tensor rgb_to_lab(const torch::Tensor& rgb);
torch::Tensor lab_to_rgb(const torch::Tensor& lab);

// Generic 3x3 matrix transform
torch::Tensor color_transform_3x3(const torch::Tensor& input, const torch::Tensor& matrix_3x3);
