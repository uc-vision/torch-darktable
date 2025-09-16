#pragma once

#include <torch/torch.h>
#include <torch/extension.h>

struct Wiener {
    virtual ~Wiener() = default;
    virtual torch::Tensor process(const torch::Tensor &input, const torch::Tensor &noise_sigmas) = 0;
    virtual int get_overlap_factor() const = 0;
};

std::shared_ptr<Wiener> create_wiener(
    torch::Device device,
    int width,
    int height,
    int overlap_factor,
    int tile_size = 32,
    int channels = 3);



