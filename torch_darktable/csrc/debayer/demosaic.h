#pragma once

#include <torch/torch.h>
#include <torch/extension.h>
#include <memory>

// PPG demosaic interface
struct PPG {
    virtual ~PPG() = default;
    virtual torch::Tensor process(const torch::Tensor& input) = 0;
    virtual int get_width() const = 0;
    virtual int get_height() const = 0;
    // Parameters
    virtual void set_median_threshold(float threshold) = 0;
    virtual float get_median_threshold() const = 0;
};

// RCD demosaic interface
struct RCD {
    virtual ~RCD() = default;
    virtual torch::Tensor process(const torch::Tensor& input) = 0;
    virtual int get_width() const = 0;
    virtual int get_height() const = 0;
    // Parameters
    virtual void set_input_scale(float scale) = 0;
    virtual void set_output_scale(float scale) = 0;
    virtual float get_input_scale() const = 0;
    virtual float get_output_scale() const = 0;
};

// PostProcess interface
struct PostProcess {
    virtual ~PostProcess() = default;
    virtual torch::Tensor process(const torch::Tensor& input) = 0;
    // Parameters
    virtual void set_color_smoothing_passes(int passes) = 0;
    virtual void set_green_eq_local(bool enabled) = 0;
    virtual void set_green_eq_global(bool enabled) = 0;
    virtual void set_green_eq_threshold(float threshold) = 0;
    virtual int get_color_smoothing_passes() const = 0;
    virtual bool get_green_eq_local() const = 0;
    virtual bool get_green_eq_global() const = 0;
    virtual float get_green_eq_threshold() const = 0;
};

// Factory functions - implementations are in .cu files
std::shared_ptr<PPG> create_ppg(torch::Device device, int width, int height, 
  uint32_t filters, float median_threshold);

std::shared_ptr<RCD> create_rcd(torch::Device device, int width, int height, 
  uint32_t filters, float input_scale, float output_scale);

std::shared_ptr<PostProcess> create_postprocess(torch::Device device, int width, int height,
  uint32_t filters, int color_smoothing_passes, bool green_eq_local, bool green_eq_global,
  float green_eq_threshold);