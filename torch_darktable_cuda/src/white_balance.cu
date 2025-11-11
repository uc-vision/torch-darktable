#include <cuda_runtime.h>
#include <torch/torch.h>
#include "debayer/demosaic.h"
#include "debayer/bayer_device.h"
#include "reduction.h"
#include "device_math.h"
#include "cuda_utils.h"

// Apply white balance to Bayer image
__global__ void apply_white_balance_kernel(
    float* bayer,           // Input/output Bayer image (H*W)
    const float3 wb_gains,  // White balance gains (R, G, B)
    const BayerPattern pattern,
    const int width,
    const int height
) {
    int2 pos = pixel_index();
    if (pos.x >= width || pos.y >= height) return;
    
    const int idx = pos.y * width + pos.x;
    float pixel_value = bayer[idx];
    
    // Determine which color channel this pixel represents
    int channel = fc(pos.y, pos.x, pattern);
    
    // Apply appropriate white balance gain
    float gain;
    switch (channel) {
        case 0: // R
            gain = wb_gains.x;
            break;
        case 2: // B
            gain = wb_gains.z;
            break;
        default:
            gain = wb_gains.y; // Default to green
            break;
    }
    
    // Apply gain and clamp to prevent overflow
    bayer[idx] = clamp(pixel_value * gain, 0.0f, 1.0f);
}


__device__ __forceinline__ float4 load_bayer_2x2(const float* bayer, 
    const int width, const int2 p) {
    return make_float4(
        bayer[p.y * width + p.  x], 
        bayer[p.y * width + (p.x + 1)], 
        bayer[(p.y + 1) * width + p.x],
        bayer[(p.y + 1) * width + (p.x + 1)]);
}



// Collect RGB patches with intensity values for quantile computation
__global__ void collect_color_samples_kernel(
    const float* bayer_image,
    const BayerPattern pattern,
    const int width,
    const int height,
    const int stride,
    float* chromacities,
    float* intensities,
    bool* mask
) {
    // Each thread processes one 2x2 patch with stride
    int2 pos = pixel_index();
    if (pos.x + 1 >= width / stride || pos.y + 1 >= height / stride) return;
    
    float4 bayer = load_bayer_2x2(bayer_image, width, pos * 2);
    float3 rgb = bayer_2x2_to_rgb(bayer.x, bayer.y, bayer.z, bayer.w, pattern);

    float intensity = rgb.x + rgb.y + rgb.z;
    int linear_idx = pos.y * (width / stride) + pos.x;
    chromacities[linear_idx * 2 + 0] = rgb.x / intensity;
    chromacities[linear_idx * 2 + 1] = rgb.y / intensity;

    intensities[linear_idx] = intensity;
    float max_bayer = fmaxf(fmaxf(bayer.x, bayer.y), fmaxf(bayer.z, bayer.w));
    mask[linear_idx] = max_bayer < 1.0f;
}

__device__ __forceinline__ float max_float4(const float4& v) {
    return fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w));
}

__device__ __forceinline__ void float2_store(const float2& val, float* output, int idx) {
    output[idx * 2 + 0] = val.x;
    output[idx * 2 + 1] = val.y;
}


std::tuple<torch::Tensor, torch::Tensor> collect_samples(
    const std::vector<torch::Tensor>& bayer_images,
    BayerPattern pattern,
    int stride
) {
    const auto& first_image = bayer_images[0];
    const int height = first_image.size(0);
    const int width = first_image.size(1);
    const int sh = height / stride;
    const int sw = width / stride;
    const int n_samples = bayer_images.size() * sh * sw;
    
    auto opts = torch::TensorOptions().device(first_image.device()).dtype(torch::kFloat32);
    auto chromacities = torch::empty({n_samples, 2}, opts);
    auto intensities = torch::empty({n_samples}, opts);
    auto masks = torch::empty({n_samples}, opts.dtype(torch::kBool));
    
    for (size_t i = 0; i < bayer_images.size(); ++i) {
        int offset = i * sh * sw;
        collect_color_samples_kernel<<<grid2d(sw, sh), block_size_2d>>>(
            bayer_images[i].data_ptr<float>(),
            pattern, width, height, stride,
            chromacities.data_ptr<float>() + offset * 2,
            intensities.data_ptr<float>() + offset,
            masks.data_ptr<bool>() + offset
        );
    }
    
    CUDA_CHECK_KERNEL();
    
    // Filter to valid samples (mask is true for valid)
    return std::make_tuple(
        chromacities.index({masks}), 
        intensities.index({masks})
    );
}

torch::Tensor estimate_white_balance(
    const std::vector<torch::Tensor>& bayer_images,
    BayerPattern pattern,
    float quantile,
    int stride
) {
    if (bayer_images.empty()) {
        throw std::runtime_error("No images provided");
    }
    
    auto [chromaticities, intensities] = collect_samples(bayer_images, pattern, stride);
    auto opts = torch::TensorOptions().device(bayer_images[0].device()).dtype(torch::kFloat32);
    
    if (chromaticities.size(0) == 0) {
        return torch::tensor({1.0f, 1.0f, 1.0f}, opts);
    }
    
    // Select bright samples
    auto bright_chromas = chromaticities.index({intensities >= torch::quantile(intensities, quantile)});
    
    if (bright_chromas.size(0) == 0) {
        return torch::tensor({1.0f, 1.0f, 1.0f}, opts);
    }
    
    // Convert to white balance gains (Green = 1.0)
    auto mean_chroma = bright_chromas.mean(0);
    return torch::stack({
        mean_chroma[0] / mean_chroma[1], 
        torch::tensor(1.0f, opts), 
        (1.0f - mean_chroma[0] - mean_chroma[1]) / mean_chroma[1]
    });
}

torch::Tensor apply_white_balance(
    const torch::Tensor& bayer_image,
    const torch::Tensor& gains,
    BayerPattern pattern
) {
    auto result = bayer_image.clone();
    
    const int height = bayer_image.size(0);
    const int width = bayer_image.size(1);
    
    const float3 wb_gains = make_float3(
      gains[0].item<float>(), gains[1].item<float>(), gains[2].item<float>());
    
    apply_white_balance_kernel<<<grid2d(width, height), block_size_2d>>>(
        result.data_ptr<float>(), wb_gains, pattern, width, height
    );
    
    CUDA_CHECK_KERNEL();
    return result;
}

