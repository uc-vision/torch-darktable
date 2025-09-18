/*
 * AMaZE Demosaic CUDA Implementation
 * Based on the AMaZE algorithm from darktable/RawTherapee
 */

#include <cuda_runtime.h>
#include <torch/types.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/nn/functional.h>
#include "../cuda_utils.h"
#include <cstdint>
#include <algorithm>
#include <cmath>

#include "amaze.h"



// AMaZE Implementation
struct AmazeImpl : public AMaZE {
    BayerPattern pattern_;
    float clip_pt_;
    int width_;
    int height_;
    torch::Device device_;
    
    // Temporary buffers (reused across calls)
    torch::Tensor dirwts0_;
    torch::Tensor dirwts1_;
    torch::Tensor delhvsqsum_;
    torch::Tensor vcd_;
    torch::Tensor hcd_;
    torch::Tensor hvwt_;
    torch::Tensor cddiffsq_;
    torch::Tensor nyquist_;
    torch::Tensor nyqutest_;
    torch::Tensor padded_input_;
    
    AmazeImpl(torch::Device device, int width, int height, BayerPattern pattern, float clip_pt)
        : pattern_(pattern), clip_pt_(clip_pt), width_(width), height_(height), device_(device) {
        
        const int pad = 16;
        const int2 padded_size = make_int2(width, height) + 2 * pad;
        
        const auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        const auto uint8_opts = torch::TensorOptions().dtype(torch::kUInt8).device(device);
        
        // Pre-allocate all temporary buffers
        dirwts0_ = torch::zeros({padded_size.y, padded_size.x}, float_opts);
        dirwts1_ = torch::zeros({padded_size.y, padded_size.x}, float_opts);
        delhvsqsum_ = torch::zeros({padded_size.y, padded_size.x}, float_opts);
        vcd_ = torch::zeros({padded_size.y, padded_size.x}, float_opts);
        hcd_ = torch::zeros({padded_size.y, padded_size.x}, float_opts);
        hvwt_ = torch::zeros({padded_size.y, padded_size.x}, float_opts);
        cddiffsq_ = torch::zeros({padded_size.y, padded_size.x}, float_opts);
        nyquist_ = torch::zeros({padded_size.y, padded_size.x}, uint8_opts);
        nyqutest_ = torch::zeros({padded_size.y, padded_size.x}, float_opts);
        padded_input_ = torch::zeros({padded_size.y, padded_size.x}, float_opts);
    }
    
    ~AmazeImpl() override = default;
    
    int get_width() const override { return width_; }
    int get_height() const override { return height_; }
    void set_clip_pt(float clip_pt) override { clip_pt_ = clip_pt; }
    float get_clip_pt() const override { return clip_pt_; }
    
    torch::Tensor process(const torch::Tensor& input) override;
    
private:
    template<BayerPattern pattern>
    torch::Tensor process_with_pattern(const torch::Tensor& input);
};


// Reflect padding kernel
__global__ void reflect_pad_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int input_width,
    int input_height,
    int output_width,
    int output_height,
    int pad
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= output_width || y >= output_height) return;
    
    // Map output coordinates to input coordinates with reflection
    int src_x = x - pad;
    int src_y = y - pad;
    
    // Reflect padding
    if (src_x < 0) src_x = -src_x;
    else if (src_x >= input_width) src_x = 2 * input_width - src_x - 2;
    
    if (src_y < 0) src_y = -src_y;
    else if (src_y >= input_height) src_y = 2 * input_height - src_y - 2;
    
    // Clamp to valid range (safety)
    src_x = clamp_int(src_x, 0, input_width - 1);
    src_y = clamp_int(src_y, 0, input_height - 1);
    
    output[y * output_width + x] = input[src_y * input_width + src_x];
}

// Step 1: Compute horizontal and vertical gradients
__global__ void compute_gradients_kernel(
    const float* __restrict__ input,
    float* __restrict__ dirwts0,
    float* __restrict__ dirwts1, 
    float* __restrict__ delhvsqsum,
    int width,
    int height
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < 2 || x >= width - 2 || y < 2 || y >= height - 2) return;
    
    const int idx = y * width + x;
    
    const float4 neighbors = get_neighbors(input, idx, width);
    const float4 neighbors2 = get_neighbors2(input, idx, width);
    const float center = input[idx];
    
    // Use vectorized operations for differences
    const float4 neighbor_diffs = abs(make_float4(neighbors.x, neighbors.y, neighbors.z, neighbors.w) - 
                                     make_float4(neighbors.y, neighbors.x, neighbors.w, neighbors.z));
    const float delv = neighbor_diffs.x;  // |north - south|
    const float delh = neighbor_diffs.z;  // |west - east|
    
    const float4 center2_diffs = abs(neighbors2 - center);
    
    dirwts0[idx] = eps + center2_diffs.y + center2_diffs.x + delv;  // south + north + delv
    dirwts1[idx] = eps + center2_diffs.w + center2_diffs.z + delh;  // east + west + delh
    delhvsqsum[idx] = sqrf(delh) + sqrf(delv);
}

// Step 2: Interpolate color differences and compute color difference squares
__global__ void interpolate_color_differences_kernel(
    const float* __restrict__ input,
    const float* __restrict__ dirwts0,
    const float* __restrict__ dirwts1,
    float* __restrict__ vcd,
    float* __restrict__ hcd,
    float* __restrict__ cddiffsq,
    int width,
    int height,
    BayerPattern pattern,
    float clip_pt
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < 4 || x >= width - 4 || y < 4 || y >= height - 4) return;
    
    const int idx = y * width + x;
    const float clip_pt8 = 0.8f * clip_pt;
    
    const float center = input[idx];
    const float4 neighbors = get_neighbors(input, idx, width);
    const float4 neighbors2 = get_neighbors2(input, idx, width);
    const float4 dirwts0_neighbors = get_neighbors(dirwts0, idx, width);
    const float4 dirwts0_neighbors2 = get_neighbors2(dirwts0, idx, width);
    const float4 dirwts1_neighbors = get_neighbors(dirwts1, idx, width);
    const float4 dirwts1_neighbors2 = get_neighbors2(dirwts1, idx, width);
    
    // Color ratios in each cardinal direction (exactly matching original)
    const float cru = neighbors.x * (dirwts0_neighbors2.x + dirwts0[idx])
                    / (dirwts0_neighbors2.x * (eps + center) + dirwts0[idx] * (eps + neighbors2.x));
    const float crd = neighbors.y * (dirwts0_neighbors2.y + dirwts0[idx])
                    / (dirwts0_neighbors2.y * (eps + center) + dirwts0[idx] * (eps + neighbors2.y));
    const float crl = neighbors.z * (dirwts1_neighbors2.z + dirwts1[idx])
                    / (dirwts1_neighbors2.z * (eps + center) + dirwts1[idx] * (eps + neighbors2.z));
    const float crr = neighbors.w * (dirwts1_neighbors2.w + dirwts1[idx])
                    / (dirwts1_neighbors2.w * (eps + center) + dirwts1[idx] * (eps + neighbors2.w));
    
    // Hamilton-Adams interpolation (vectorized)
    const float4 hamilton_adams_interp = neighbors + 0.5f * (center - neighbors2);
    
    // Adaptive ratio interpolation for each direction
    const float guar = (fabsf(1.0f - cru) < arthresh) ? center * cru : hamilton_adams_interp.x;
    const float gdar = (fabsf(1.0f - crd) < arthresh) ? center * crd : hamilton_adams_interp.y;
    const float glar = (fabsf(1.0f - crl) < arthresh) ? center * crl : hamilton_adams_interp.z;
    const float grar = (fabsf(1.0f - crr) < arthresh) ? center * crr : hamilton_adams_interp.w;
    
    // Adaptive weights
    const float hwt = dirwts1_neighbors.z / (dirwts1_neighbors.z + dirwts1_neighbors.w);  // west/(west+east)
    const float vwt = dirwts0_neighbors.x / (dirwts0_neighbors.y + dirwts0_neighbors.x);  // north/(south+north)
    
    // Hamilton-Adams G interpolation (for comparison/fallback)
    const float Gintvha = vwt * hamilton_adams_interp.y + (1.0f - vwt) * hamilton_adams_interp.x;  // vertical HA
    const float Ginthha = hwt * hamilton_adams_interp.w + (1.0f - hwt) * hamilton_adams_interp.z;  // horizontal HA
    
    // Adaptive ratio G interpolation (primary method)
    const float Gintv_adaptive = vwt * gdar + (1.0f - vwt) * guar;  // vertical adaptive
    const float Ginth_adaptive = hwt * grar + (1.0f - hwt) * glar;  // horizontal adaptive
    
    // Color differences based on pixel type (use unpadded coordinates for Bayer pattern)
    const int c = fc(y, x, pattern);
    const bool fcswitch = (c & 1) != 0;
    
    // Use adaptive ratio interpolation by default
    if (fcswitch) {
        vcd[idx] = center - Gintv_adaptive;
        hcd[idx] = center - Ginth_adaptive;
    } else {
        vcd[idx] = Gintv_adaptive - center;
        hcd[idx] = Ginth_adaptive - center;
    }
    
    // Use Hamilton-Adams if highlights are clipped
    if (center > clip_pt8 || Gintv_adaptive > clip_pt8 || Ginth_adaptive > clip_pt8) {
        // Switch to Hamilton-Adams interpolation for clipped highlights
        if (fcswitch) {
            vcd[idx] = center - Gintvha;
            hcd[idx] = center - Ginthha;
        } else {
            vcd[idx] = Gintvha - center;
            hcd[idx] = Ginthha - center;
        }
    }
    
    // Compute color difference squares (only for R/B pixels, needed for Nyquist detection)
    if ((c & 1) == 0) {
        cddiffsq[idx] = sqrf(vcd[idx] - hcd[idx]);
    }
}


// Step 3: Compute color difference variances and adaptive weights (strided for R/B pixels only)
__global__ void compute_adaptive_weights_kernel(
    const float* __restrict__ vcd,
    const float* __restrict__ hcd,
    const float* __restrict__ dirwts0,
    const float* __restrict__ dirwts1,
    const float* __restrict__ delhvsqsum,
    float* __restrict__ hvwt,
    int width,
    int height,
    BayerPattern pattern
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < 3 || x >= (width - 6) / 2 || y < 6 || y >= height - 6) return;
    
    // Strided indexing: process every 2nd pixel starting from R/B positions
    const int col = 6 + (fc(y, 2, pattern) & 1) + 2 * x;  // Start at R/B pixel, stride by 2
    const int idx = y * width + col;
    
    if (col >= width - 6) return;
    
    // R/B pixels - compute adaptive weights
    const float Dgrbvvaru = compute_variance_4(vcd, idx, -width, -2*width, -3*width);
    const float Dgrbvvard = compute_variance_4(vcd, idx, width, 2*width, 3*width);
    const float Dgrbhvarl = compute_variance_4(hcd, idx, -1, -2, -3);
    const float Dgrbhvarr = compute_variance_4(hcd, idx, 1, 2, 3);

    const float hwt = dirwts1[idx - 1] / (dirwts1[idx - 1] + dirwts1[idx + 1]);
    const float vwt = dirwts0[idx - width] / (dirwts0[idx + width] + dirwts0[idx - width]);

    const float vcdvar = epssq + vwt * Dgrbvvard + (1.0f - vwt) * Dgrbvvaru;
    const float hcdvar = epssq + hwt * Dgrbhvarr + (1.0f - hwt) * Dgrbhvarl;

    // Determine adaptive weights for G interpolation
    hvwt[idx] = hcdvar / (vcdvar + hcdvar);
}

// Step 4: Nyquist texture detection
__global__ void nyquist_texture_kernel(
    const float* __restrict__ delhvsqsum,
    const float* __restrict__ cddiffsq,
    unsigned char* __restrict__ nyquist,
    float* __restrict__ nyqutest,
    int width,
    int height,
    BayerPattern pattern
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < 6 || x >= width - 6 || y < 6 || y >= height - 6) return;
    
    const int idx = y * width + x;
    const int c = fc(y, x, pattern);
    
    // Only process R/B pixels
    if ((c & 1) != 0) return;
    
    // Nyquist texture test using gaussian convolution helpers
    nyqutest[idx] = gaussian_quincunx_conv(cddiffsq, idx, width, gaussodd)
                  - gaussian_grid_conv(delhvsqsum, idx, width, gaussgrad);
    
    // Set nyquist flag based on test result
    nyquist[idx] = (nyqutest[idx] > 0.0f) ? 1 : 0;
}


template<BayerPattern pattern>
__global__ void amaze_demosaic_kernel(
    const float* __restrict__ input,
    const float* __restrict__ vcd,
    const float* __restrict__ hcd,
    const float* __restrict__ hvwt,
    float3* __restrict__ output,
    int width,
    int height,
    int pw,
    int ph,
    int pad
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    const int2 base = make_int2(x, y) * 2;
    if (base.x >= width - 1 || base.y >= height - 1) return;

     // Process all 4 pixels in the 2x2 block
     #pragma unroll
     for (int C = 0; C < 4; ++C) {
         const int pixel_type = get_pixel_type<pattern>(C);
         const int2 p = base + offset2x2(C);
         
         const int padded_idx = (p.y + pad) * pw + (p.x + pad);
         const float input_val = input[padded_idx];
         
         // Find nearest R/B pixel for hvwt lookup (since hvwt only computed for R/B pixels)
         const int hvwt_x = ((p.x + pad) & ~1) + (fc(p.y + pad, 2, pattern) & 1);  // Round to nearest R/B column
         const int hvwt_idx = (p.y + pad) * pw + hvwt_x;
         const float weight = hvwt[hvwt_idx];
         const float color_diff = (1.0f - weight) * vcd[padded_idx] + weight * hcd[padded_idx];
         const float3 rgb = compute_rgb_from_pixel_type(pixel_type, input_val, color_diff);
         
         const int out_idx = p.y * width + p.x;
         output[out_idx] = clamp01(rgb);
     }
}

// Implementation of process method
torch::Tensor AmazeImpl::process(const torch::Tensor& input) {
    TORCH_CHECK(input.device() == device_, "Input tensor must be on the same device as AMaZE instance");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(input.dim() == 3, "Input tensor must be 3D (H, W, 1)");
    TORCH_CHECK(input.size(2) == 1, "Input tensor must have 1 channel");
    TORCH_CHECK(input.size(0) == height_ && input.size(1) == width_, 
                "Input size must match AMaZE instance size");
    
    // Call templated processing function based on pattern
    switch(pattern_) {
        case BayerPattern::RGGB: return process_with_pattern<BayerPattern::RGGB>(input);
        case BayerPattern::BGGR: return process_with_pattern<BayerPattern::BGGR>(input);
        case BayerPattern::GRBG: return process_with_pattern<BayerPattern::GRBG>(input);
        case BayerPattern::GBRG: return process_with_pattern<BayerPattern::GBRG>(input);
    }
    
    // Should never reach here
    return torch::empty({0});
}

// Template implementation that avoids switch statement repetition
template<BayerPattern pattern>
torch::Tensor AmazeImpl::process_with_pattern(const torch::Tensor& input) {
    // Create output tensor (only thing allocated each time)
    const auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    auto output = torch::zeros({height_, width_, 3}, float_opts);
    
    // Squeeze input to 2D (H, W) and ensure contiguous for data_ptr() access
    const int pad = 16;
    auto input_2d = input.squeeze(-1);
    if (!input_2d.is_contiguous()) {
        input_2d = input_2d.contiguous();
    }
    
    // Use custom padding kernel instead of PyTorch's pad function
    dim3 block(16, 16);
    const int2 padded_size = make_int2(width_, height_) + 2 * pad;
    dim3 pad_grid((padded_size.x + block.x - 1) / block.x, 
                  (padded_size.y + block.y - 1) / block.y);
    
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    reflect_pad_kernel<<<pad_grid, block, 0, stream>>>(
        input_2d.data_ptr<float>(), padded_input_.data_ptr<float>(),
        width_, height_, padded_size.x, padded_size.y, pad
    );
    CUDA_CHECK(cudaGetLastError());
    
    dim3 padded_grid((padded_size.x + block.x - 1) / block.x, (padded_size.y + block.y - 1) / block.y);
    dim3 block_grid(((width_ + 1) / 2 + block.x - 1) / block.x, ((height_ + 1) / 2 + block.y - 1) / block.y);   
    
    // Step 1: Compute gradients
    compute_gradients_kernel<<<padded_grid, block, 0, stream>>>(
        padded_input_.data_ptr<float>(), dirwts0_.data_ptr<float>(),
        dirwts1_.data_ptr<float>(), delhvsqsum_.data_ptr<float>(),
        padded_size.x, padded_size.y
    );
    CUDA_CHECK(cudaGetLastError());
    
    // Step 2: Interpolate color differences and compute color difference squares
    interpolate_color_differences_kernel<<<padded_grid, block, 0, stream>>>(
        padded_input_.data_ptr<float>(), dirwts0_.data_ptr<float>(), dirwts1_.data_ptr<float>(),
        vcd_.data_ptr<float>(), hcd_.data_ptr<float>(), cddiffsq_.data_ptr<float>(),
        padded_size.x, padded_size.y, pattern, clip_pt_
    );
    CUDA_CHECK(cudaGetLastError());
    
    // Step 3: Compute adaptive weights
    compute_adaptive_weights_kernel<<<padded_grid, block, 0, stream>>>(
        vcd_.data_ptr<float>(), hcd_.data_ptr<float>(), dirwts0_.data_ptr<float>(),
        dirwts1_.data_ptr<float>(), delhvsqsum_.data_ptr<float>(), hvwt_.data_ptr<float>(),
        padded_size.x, padded_size.y, pattern
    );
    CUDA_CHECK(cudaGetLastError());
    
    // Step 4: Nyquist texture detection
    nyquist_texture_kernel<<<padded_grid, block, 0, stream>>>(
        delhvsqsum_.data_ptr<float>(), cddiffsq_.data_ptr<float>(),
        nyquist_.data_ptr<unsigned char>(), nyqutest_.data_ptr<float>(),
        padded_size.x, padded_size.y, pattern
    );
    CUDA_CHECK(cudaGetLastError());
    
    // Final step: Process 2x2 blocks (includes green interpolation)
    amaze_demosaic_kernel<pattern><<<block_grid, block, 0, stream>>>(
        padded_input_.data_ptr<float>(), vcd_.data_ptr<float>(), hcd_.data_ptr<float>(),
        hvwt_.data_ptr<float>(), reinterpret_cast<float3*>(output.data_ptr<float>()),
        width_, height_, padded_size.x, padded_size.y, pad
    );
    CUDA_CHECK(cudaGetLastError());
    
    return output;
}

// Factory function
std::shared_ptr<AMaZE> create_amaze(torch::Device device, int width, int height,
                                   BayerPattern pattern, float clip_pt) {
    return std::make_shared<AmazeImpl>(device, width, height, pattern, clip_pt);
}
