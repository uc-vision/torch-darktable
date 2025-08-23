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
    virtual py::dict get_parameters() const = 0;
    
    // Adjustable parameter setters
    virtual void set_sigma(float sigma) = 0;
    virtual void set_shadows(float shadows) = 0;
    virtual void set_highlights(float highlights) = 0;
    virtual void set_clarity(float clarity) = 0;
};

// Factory function for workspace creation (for reuse across multiple images)
std::shared_ptr<Laplacian> create_laplacian(
    torch::Device device, int num_gamma, 
    int width, int height, 
    float sigma, float shadows, float highlights, float clarity);
