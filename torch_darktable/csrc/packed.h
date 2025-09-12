#pragma once

#include <torch/extension.h>

// 12-bit encoding functions
torch::Tensor encode12_u16(const torch::Tensor& input, bool ids_format = false);
torch::Tensor encode12_float(const torch::Tensor& input, bool ids_format = false, bool scaled = true);

// 12-bit decoding functions
torch::Tensor decode12_float(const torch::Tensor& input, bool ids_format = false, bool scaled = true);
torch::Tensor decode12_half(const torch::Tensor& input, bool ids_format = false, bool scaled = true);
torch::Tensor decode12_u16(const torch::Tensor& input, bool ids_format = false);
