#pragma once

#include <torch/torch.h>
#include <torch/extension.h>
#include <memory>

struct Bilateral {
    virtual ~Bilateral() = default;
    virtual torch::Tensor process(const torch::Tensor &luminance) = 0;
    virtual py::dict get_parameters() const = 0;

    virtual void set_sigma_s(float sigma_s) = 0;
    virtual void set_sigma_r(float sigma_r) = 0;
    virtual void set_detail(float detail) = 0;
};

std::shared_ptr<Bilateral> create_bilateral(
    torch::Device device,
    int width,
    int height,
    float sigma_s,
    float sigma_r,
    float detail);


