#pragma once

#include <torch/torch.h>
#include <torch/extension.h>
#include <memory>

struct Bilateral {
    virtual ~Bilateral() = default;
    virtual torch::Tensor process(const torch::Tensor &luminance, float detail) = 0;
    virtual void set_sigma_s(float sigma_s) = 0;
    virtual void set_sigma_r(float sigma_r) = 0;
    // Read accessors (for concise property bindings)
    virtual float get_sigma_s() const = 0;
    virtual float get_sigma_r() const = 0;
    virtual int get_width() const = 0;
    virtual int get_height() const = 0;
};

std::shared_ptr<Bilateral> create_bilateral(
    torch::Device device,
    int width,
    int height,
    float sigma_s,
    float sigma_r);


