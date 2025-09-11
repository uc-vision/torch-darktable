#include "../cuda_utils.h"
#include "../device_math.h"
#include "../reduction.h"
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Reinhard tone mapping kernel
__global__ void reinhard_tonemap_kernel(
    const float* __restrict__ input,
    uint8_t* __restrict__ output,
    const float* __restrict__ metrics,
    float gamma,
    float intensity,
    float light_adapt,
    int height,
    int width
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Extract metrics
    float bounds_min = metrics[0];
    float bounds_max = metrics[1]; 
    float log_bounds_min = metrics[2];
    float log_bounds_max = metrics[3];
    float log_mean = metrics[4];
    float mean = metrics[5];
    float3 rgb_mean = make_float3(metrics[6], metrics[7], metrics[8]);
    
    // Load input pixel
    int idx = y * width + x;
    float3 rgb = float3_load(input, idx);
    
    // Scale to [0,1] range
    float range = bounds_max - bounds_min;
    float3 scaled = (rgb - bounds_min) / range;
    
    float gray = rgb_to_gray(scaled);
    
    // Compute key and map_key
    float key = (log_bounds_max - log_mean) / (log_bounds_max - log_bounds_min + 1e-6f);
    float map_key = 0.3f + 0.7f * powf(key, 1.4f);
    
    // Blend between mean and RGB mean for adaptation
    float3 adapt_mean = mean + light_adapt * (rgb_mean - mean);
    
    // Apply tone mapping
    float exp_neg_intensity = expf(-intensity);
    float3 adapt = pow(adapt_mean * exp_neg_intensity, map_key);
    
    float3 tonemapped = scaled / (adapt + scaled);
    
    // Apply gamma correction and convert to 8-bit
    float3 gamma_corrected = pow(fmax(tonemapped, 0.0f), 1.0f / gamma);
    float3_to_uint8_rgb(gamma_corrected, output, idx);
}

torch::Tensor reinhard_tonemap(
    const torch::Tensor& image,
    const torch::Tensor& metrics,
    float gamma,
    float intensity,
    float light_adapt
) {
    TORCH_CHECK(image.device().is_cuda(), "Input must be on CUDA device");
    TORCH_CHECK(metrics.device().is_cuda(), "Metrics must be on CUDA device");
    TORCH_CHECK(image.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(metrics.dtype() == torch::kFloat32, "Metrics must be float32");
    TORCH_CHECK(image.dim() == 3 && image.size(2) == 3, "Input must be (H, W, 3)");
    TORCH_CHECK(metrics.numel() == 9, "Metrics must have 9 elements");

    int height = image.size(0);
    int width = image.size(1);
    
    auto output = torch::empty({height, width, 3}, torch::dtype(torch::kUInt8).device(image.device()));

    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, 
                   (height + block_size.y - 1) / block_size.y);

    reinhard_tonemap_kernel<<<grid_size, block_size>>>(
        image.data_ptr<float>(),
        output.data_ptr<uint8_t>(),
        metrics.data_ptr<float>(),
        gamma,
        intensity,
        light_adapt,
        height,
        width
    );

    CUDA_CHECK_KERNEL();
    return output;
}

// Image metrics computation functions
// Compute image bounds kernel using cooperative groups
__global__ void compute_bounds_kernel(
    const float* __restrict__ image,
    float* __restrict__ bounds,
    int height,
    int width,
    int stride
) {
    
    int x = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    int y = (blockIdx.y * blockDim.y + threadIdx.y) * stride;
    
    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;
    
    // Each thread processes one pixel
    if (x < width && y < height) {
        // RGB image: (height, width, 3)
        int pixel_idx = y * width + x;
        float3 rgb = float3_load(image, pixel_idx);
        
        local_min = fminf(fminf(rgb.x, rgb.y), rgb.z);
        local_max = fmaxf(fmaxf(rgb.x, rgb.y), rgb.z);
    }
    
    // Reduce + atomic write in one step
    reduce_min(local_min, &bounds[0]);
    reduce_max(local_max, &bounds[1]);
}

// Compute image metrics kernel
__global__ void compute_metrics_kernel(
    const float* __restrict__ image,
    float* __restrict__ metrics,
    float bounds_min,
    float bounds_max,
    int height,
    int width,
    int stride,
    float min_gray
) {
    int x = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    int y = (blockIdx.y * blockDim.y + threadIdx.y) * stride;
    
    if (x >= width || y >= height) return;
    
    // Load RGB pixel
    int pixel_idx = y * width + x;
    float3 rgb = float3_load(image, pixel_idx);
    
    // Scale to [0,1] range
    float range = bounds_max - bounds_min + 1e-6f;
    float3 scaled = (rgb - bounds_min) / range;
    
    float gray = rgb_to_gray(scaled);
    float log_gray = logf(fmaxf(gray, min_gray));
    
    // Reduce + atomic writes in one step
    reduce_add(log_gray, &metrics[4]);     // log_mean
    reduce_add(gray, &metrics[5]);         // mean  
    reduce_add(scaled.x, &metrics[6]);     // rgb_mean.r
    reduce_add(scaled.y, &metrics[7]);     // rgb_mean.g
    reduce_add(scaled.z, &metrics[8]);     // rgb_mean.b
    reduce_min(log_gray, &metrics[2]);     // log_bounds_min
    reduce_max(log_gray, &metrics[3]);     // log_bounds_max
}

torch::Tensor compute_image_bounds(const torch::Tensor& image, int stride) {
    TORCH_CHECK(image.device().is_cuda(), "Input must be on CUDA device");
    TORCH_CHECK(image.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(image.dim() == 3 && image.size(2) == 3, "Input must be (H, W, 3)");

    int height = image.size(0);
    int width = image.size(1);
    
    auto bounds = torch::tensor({FLT_MAX, -FLT_MAX}, torch::dtype(torch::kFloat32).device(image.device()));

    int sample_width = (width + stride - 1) / stride;
    int sample_height = (height + stride - 1) / stride;
    
    const dim3 block_size(16, 16);
    const dim3 grid_size((sample_width + block_size.x - 1) / block_size.x,
                         (sample_height + block_size.y - 1) / block_size.y);

    compute_bounds_kernel<<<grid_size, block_size>>>(
        image.data_ptr<float>(),
        bounds.data_ptr<float>(),
        height,
        width,
        stride
    );

    CUDA_CHECK_KERNEL();
    return bounds;
}

torch::Tensor compute_image_metrics(std::vector<torch::Tensor>& images, int stride, float min_gray) {

    // Initialize metrics tensor: [bounds_min, bounds_max, log_bounds_min, log_bounds_max, log_mean, mean, rgb_mean_r, rgb_mean_g, rgb_mean_b]
    auto metrics = torch::tensor({FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, 
                                torch::dtype(torch::kFloat32).device(image.device()));

    for (const auto& image : images) {
      TORCH_CHECK(image.device().is_cuda(), "Input must be on CUDA device");
      TORCH_CHECK(image.dtype() == torch::kFloat32, "Input must be float32");
      TORCH_CHECK(image.dim() == 3 && image.size(2) == 3, "Input must be (H, W, 3)");

      int height = image.size(0);
      int width = image.size(1);
      
      // First compute bounds
      auto bounds = compute_image_bounds(image, stride);
      metrics[0] = torch::min(metrics[0], bounds[0]);
      metrics[1] = torch::max(metrics[1], bounds[1]);
      
      int sample_width = (width + stride - 1) / stride;
      int sample_height = (height + stride - 1) / stride;
      int total_pixels = sample_width * sample_height;
      
      const dim3 block_size(16, 16);
      const dim3 grid_size((sample_width + block_size.x - 1) / block_size.x,
                          (sample_height + block_size.y - 1) / block_size.y);

      compute_metrics_kernel<<<grid_size, block_size>>>(
          image.data_ptr<float>(),
          metrics.data_ptr<float>(),
          bounds[0].item<float>(),
          bounds[1].item<float>(),
          height,
          width,
          stride,
          min_gray
      );

      CUDA_CHECK_KERNEL();
    }
    // Normalize accumulated values
    float norm_factor = 1.0f / total_pixels;
    metrics.slice(0, 4, 9) *= norm_factor;  // Normalize log_mean, mean, rgb_mean
    
    return metrics;
}
torch::Tensor compute_image_metrics(const std::vector<torch::Tensor>& images, int stride, float min_gray) {
    TORCH_CHECK(!images.empty(), "Images vector must not be empty");
    auto device = images[0].device();
    for (const auto& img : images) {
        TORCH_CHECK(img.device() == device, "All images must be on the same device");
        TORCH_CHECK(img.device().is_cuda(), "Inputs must be on CUDA device");
        TORCH_CHECK(img.dtype() == torch::kFloat32, "Inputs must be float32");
        TORCH_CHECK(img.dim() == 3 && img.size(2) == 3, "Inputs must be (H, W, 3)");
    }

    std::vector<torch::Tensor> metrics_list;
    metrics_list.reserve(images.size());
    for (const auto& img : images) {
        metrics_list.push_back(compute_image_metrics(img, stride, min_gray));
    }
    return torch::stack(metrics_list, 0);
}

