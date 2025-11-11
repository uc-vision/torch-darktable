#pragma once

#include <torch/torch.h>
#include <torch/extension.h>
#include <vector>
#include <map>
#include <string>

// CUDA wrapper implementation constants
constexpr int max_levels = 30;

// Base class for type erasure - allows reusable workspace with dynamic G
struct Laplacian {
    virtual ~Laplacian() = default;
    virtual torch::Tensor process(const torch::Tensor& input) = 0;
    
    // Adjustable parameter setters
    virtual void set_sigma(float sigma) = 0;
    virtual void set_shadows(float shadows) = 0;
    virtual void set_highlights(float highlights) = 0;
    virtual void set_clarity(float clarity) = 0;

    // Read accessors (for concise property bindings)
    virtual float get_sigma() const = 0;
    virtual float get_shadows() const = 0;
    virtual float get_highlights() const = 0;
    virtual float get_clarity() const = 0;
    virtual int get_width() const = 0;
    virtual int get_height() const = 0;
};

// Factory function for workspace creation (for reuse across multiple images)
std::shared_ptr<Laplacian> create_laplacian(
    torch::Device device, 
    int width, int height,
    int num_gamma, 
    float sigma, float shadows, float highlights, float clarity);
