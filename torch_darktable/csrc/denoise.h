#pragma once

#include <torch/torch.h>
#include <torch/extension.h>

struct Wiener {
    virtual ~Wiener() = default;
    virtual torch::Tensor process(const torch::Tensor &input, const torch::Tensor &noise_sigmas) = 0;
    virtual void set_eps(float eps) = 0;
    virtual float get_eps() const = 0;
};

std::shared_ptr<Wiener> create_wiener(
    torch::Device device,
    int width,
    int height,
    float eps,
    int overlap_factor);



